# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on QuAC (Bert)."""


from ast import AsyncFunctionDef
from distutils.errors import DistutilsFileError
import os
from ossaudiodev import SNDCTL_DSP_SETFRAGMENT
import re
import argparse
import glob
import copy
import logging
import random
import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from quac_processors_step1_infer import (
    QuacProcessor,
    quac_convert_examples_to_features,
    QuacResult
)
from quac_metrics import (
    compute_predictions_logits,
    read_target_dict,
    quac_performance,
    _get_best_indexes,
    get_final_text
)

from modeling_auto_bert_ts import AutoModelForQuestionAnswering
from uce import eceloss, uceloss
from uce_plot import plot_save_conf, plot_save_entr
import matplotlib
import matplotlib.pyplot as plt
import collections
from collections import defaultdict, Counter
from transformers import BasicTokenizer
import json


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, prefix="", write_predictions=True):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    #TODO
    start_logits = []
    start_labels = []
    end_logits = []
    end_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "temp_scale" : args.temp_scale, #TODO TODO
                "bayesian" : args.bayesian, #TODO TODO
                "T" : args.T, #TODO TODO
                "label_smoothing" : args.label_smoothing, #TODO TODO,
                "mc_drop_mask_num": args.mc_drop_mask_num, #TODO TODO,
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]
            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)
        
        if args.bayesian: # for uncertainty
            start_output = torch.softmax(outputs[0], dim=2).mean(dim=0)
            end_output = torch.softmax(outputs[1], dim=2).mean(dim=0)
        else:
            start_output = torch.softmax(outputs[0], dim=1)
            end_output = torch.softmax(outputs[1], dim=1)

        start_logits.append(start_output.detach())
        end_logits.append(end_output.detach())
        start_labels.append(batch[6].detach())
        end_labels.append(batch[7].detach())

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            
            if args.bayesian:
                start_end_outputs = outputs[:2]
                start_end_output = [output[:, i, :] for output in start_end_outputs]
                mc_start_logits, mc_end_logits = start_end_output
                cls_logits = outputs[2][i].tolist()

                mean_start_logits = mc_start_logits.mean(dim=0).tolist()
                mean_end_logits = mc_end_logits.mean(dim=0).tolist()

                result = QuacResult(unique_id, mean_start_logits, mean_end_logits, cls_logits)
            else:
                output = [to_list(output[i]) for output in outputs]
                fq_start_logits, fq_end_logits, cls_logits = output
                result = QuacResult(unique_id, fq_start_logits, fq_end_logits, cls_logits)

            all_results.append(result)
    
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    
    #TODO: for plot
    tensor_start_logits = torch.cat(start_logits, dim=0)
    tensor_end_logits = torch.cat(end_logits, dim=0)
    tensor_start_labels = torch.cat(start_labels, dim=0)
    tensor_end_labels = torch.cat(end_labels, dim=0)

    torch.save(tensor_start_logits, args.output_dir+'/plot/tensor_start_logits.pt')
    torch.save(tensor_end_logits, args.output_dir+'/plot/tensor_end_logits.pt')
    torch.save(tensor_start_labels, args.output_dir+'/plot/tensor_start_labels.pt')
    torch.save(tensor_end_labels, args.output_dir+'/plot/tensor_end_labels.pt')

    ece_start, acc_start, conf_start, confidence_start = eceloss(tensor_start_logits, tensor_start_labels)
    ece_end, acc_end, conf_end, confidence_end = eceloss(tensor_end_logits, tensor_end_labels)

    mean_confidence = (confidence_start + confidence_end) / 2

    uce_start, err_start, entr_start, uncertainty_start = uceloss(tensor_start_logits, tensor_start_labels)
    uce_end, err_end, entr_end, uncertainty_end = uceloss(tensor_end_logits, tensor_end_labels)

    mean_uncertainty = (uncertainty_start + uncertainty_end) / 2

    # mean_ece = (ece_start + ece_end) / 2
    # mean_uce = (uce_start + uce_end) / 2

    # if args.conf_or_uncer == 'uncer':
    #     print(mean_uce.item()*100)
    # elif args.conf_or_uncer == 'conf':
    #     print(mean_ece.item()*100)

    if args.bayesian:

        if args.temp_scale:
            plot_save_conf(args, ece_start, acc_start, conf_start, 'MC Start Calib.', args.output_dir+'/plot/mc_calib_conf_start')
            plot_save_entr(args, uce_start, err_start, entr_start, 'MC Start Calib.', args.output_dir+'/plot/mc_calib_entr_start')

            plot_save_conf(args, ece_end, acc_end, conf_end, 'MC End Calib.', args.output_dir+'/plot/mc_calib_conf_end')
            plot_save_entr(args, uce_end, err_end, entr_end, 'MC End Calib.', args.output_dir+'/plot/mc_calib_entr_end')

        else: 
            plot_save_conf(args, ece_start, acc_start, conf_start, 'MC Start Uncalib.', args.output_dir+'/plot/mc_uncalib_conf_start')
            plot_save_entr(args, uce_start, err_start, entr_start, 'MC Start UnCalib.', args.output_dir+'/plot/mc_uncalib_entr_start')
            
            plot_save_conf(args, ece_end, acc_end, conf_end, 'MC End Uncalib.', args.output_dir+'/plot/mc_uncalib_conf_end')
            plot_save_entr(args, uce_end, err_end, entr_end, 'MC End UnCalib.', args.output_dir+'/plot/mc_uncalib_entr_end')
    else:
        if args.temp_scale:
            plot_save_conf(args, ece_start, acc_start, conf_start, 'Freq. Start Calib.', args.output_dir+'/plot/frequentist_calib_conf_start')
            plot_save_entr(args, uce_start, err_start, entr_start, 'Freq. Start Calib.', args.output_dir+'/plot/frequentist_calib_entr_start')

            plot_save_conf(args, ece_end, acc_end, conf_end, 'Freq. End Calib.', args.output_dir+'/plot/frequentist_calib_conf_end')
            plot_save_entr(args, uce_end, err_end, entr_end, 'Freq. End Calib.', args.output_dir+'/plot/frequentist_calib_entr_end')

        else: 
            plot_save_conf(args, ece_start, acc_start, conf_start, 'Freq. Start Uncalib.', args.output_dir+'/plot/frequentist_uncalib_conf_start')
            plot_save_entr(args, uce_start, err_start, entr_start, 'Freq. Start UnCalib.', args.output_dir+'/plot/frequentist_uncalib_entr_start')

            plot_save_conf(args, ece_end, acc_end, conf_end, 'Freq. End Uncalib.', args.output_dir+'/plot/frequentist_uncalib_conf_end')
            plot_save_entr(args, uce_end, err_end, entr_end, 'Freq. End UnCalib.', args.output_dir+'/plot/frequentist_uncalib_entr_end')

    example_index_to_features = collections.defaultdict(list) 
    for feature, conf, uncer in zip(features, mean_confidence.tolist(), mean_uncertainty.tolist()):
        example_index_to_features[feature.example_index].append([feature, conf, uncer])

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "class_logit"]
    )
    all_nbest_start_json = collections.OrderedDict()

    for (example_index, example) in enumerate(examples):
        _features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        null_class_logit = None

        for (feature_index, _feature) in enumerate(_features):
            feature = _feature[0]
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, 1)
            end_indexes = _get_best_indexes(result.end_logits, 1)
            
            # if we could have irrelevant answers, get the min score of irrelevant
            feature_null_score = result.start_logits[0] + result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = feature_index
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]
                null_class_logit = result.cls_logits

            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            class_logit=result.cls_logits
                        )
                    )
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=min_null_feature_index,
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit,
                class_logit=null_class_logit
            )
        )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPredictionStart = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPredictionStart", ["text", "start_logit", "end_logit", "answer_start", "confidence", "uncertainty"]
        )
        seen_predictions = {}

        #TODO
        nbest_start = []
        for pred in prelim_predictions:
            _feature = _features[pred.feature_index]
            feature = _feature[0]
            conf = _feature[1]
            uncer = _feature[2]

            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
                
                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                
                final_text = get_final_text(tok_text, orig_text, args.do_lower_case, args.verbose_logging)
                
                #TODO
                actual_doc_text = example.context_text
                answer_start = actual_doc_text.find(final_text)

                confidence = conf
                uncertainty = uncer

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = 'CANNOTANSWER'
                seen_predictions[final_text] = True
                #TODO
                actual_doc_text = example.context_text
                answer_start = actual_doc_text.find(final_text)

                confidence = conf
                uncertainty = uncer

            #TODO
            nbest_start.append(_NbestPredictionStart(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,
             answer_start=answer_start, confidence=confidence, uncertainty=uncertainty))

        # if we didn't include the empty option in the n-best, include it
        if "CANNOTANSWER" not in seen_predictions:
            #TODO
            nbest_start.append(_NbestPredictionStart(text="CANNOTANSWER", start_logit=null_start_logit, end_logit=null_end_logit,
             answer_start=answer_start, confidence=confidence, uncertainty=uncertainty))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest_start:
            #TODO
            nbest_start.append(_NbestPredictionStart(text="CANNOTANSWER", start_logit=0.0, end_logit=0.0,
             answer_start=answer_start, confidence=confidence, uncertainty=uncertainty))

        assert len(nbest_start) >= 1, "No valid predictions"

        nbest_start_json = []
        for (i, entry_start) in enumerate(nbest_start):
            output = collections.OrderedDict()
            output["text"] = entry_start.text
            output["start_logit"] = entry_start.start_logit
            output["end_logit"] = entry_start.end_logit
            output["answer_start"] = entry_start.answer_start
            output["confidence"] = entry_start.confidence
            output["uncertainty"] = entry_start.uncertainty

            nbest_start_json.append(output)
        #TODO
        all_nbest_start_json[example.qas_id] = nbest_start_json


        
    output_nbest_with_start_index_conf_uncer_file = os.path.join(args.output_dir, "nbest_predictions_with_start_idx_conf_uncer_{}.json".format(prefix))
    
    with open(output_nbest_with_start_index_conf_uncer_file, "w") as writer:
        writer.write(json.dumps(all_nbest_start_json, indent=4) + "\n")


    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    #TODO
    output_nbest_with_start_index_file = os.path.join(args.output_dir, "nbest_predictions_with_start_idx_{}.json".format(prefix))

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

    else:
        predictions, nbest_predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            output_nbest_with_start_index_file, #TODO
            args.verbose_logging,
            args.null_score_diff_threshold,
            tokenizer,
            write_predictions=write_predictions,
        )

    input_file = os.path.join(args.data_dir, args.predict_file)
    target_dict = read_target_dict(input_file)

    # Compute the F1 and exact scores.
    results = quac_performance(predictions, target_dict)

    return results
    

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_transformers_{}_{}_{}_{}".format(
            "eval" if evaluate else "train",
            args.predict_file if evaluate else args.train_file,
            list(filter(None, args.model_name_or_path.split("/"))).pop() if args.cache_prefix is None else args.cache_prefix,
            str(args.max_seq_length),
            str(args.max_query_length)
        ),
    )
    
    
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
      
        processor = QuacProcessor(tokenizer=tokenizer, max_history=args.max_history)
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = quac_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads
        )
        
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=128,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal QuAC evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument(
        "--cache_prefix",
        default=None,
        type=str,
        help="prefix for cached file of datasets, features, and examples",
    )
    
    parser.add_argument("--orig_loss_coeff", type=float, help="coeff for original loss")

    #TODO TODO
    parser.add_argument("--bayesian", action='store_true', help="to use bayesian")
    parser.add_argument("--temp_scale", action='store_true', help="to use temp_scale")
    parser.add_argument("--T", type=float, default="1.0", help="to use temp_scale")
    parser.add_argument("--label_smoothing", action='store_true', help="to use label_smoothing")
    parser.add_argument("--mc_drop_mask_num", type=int, default="10", help="mc_drop_mask_num")
    parser.add_argument("--max_history", type=int, help="max_history")
    parser.add_argument("--conf_or_uncer", type=str, default="", help="whether to use confidence or uncertainty")
    
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(        
        filename=args.output_dir+'/logs.log', # 
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        force=True
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        #output_hidden_states=True,
        #output_attentions=True,
        #return_dict = True 
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if re.search("checkpoint", checkpoint) else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            f1 = evaluate(args, model, tokenizer, prefix=global_step)
            result = {'F1' : f1}
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()