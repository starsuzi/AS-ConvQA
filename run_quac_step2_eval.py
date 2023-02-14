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
""" Finetuning the library models for question-answering on QuAC (BERT)."""


from ast import AsyncFunctionDef
from bdb import set_trace
from concurrent.futures import process
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
from quac_processors_step2_eval import (
    QuacProcessor,
    quac_convert_examples_to_features,
    QuacResult
)
from quac_metrics import (
    compute_predictions_logits,
    read_target_dict,
    quac_performance,
    _get_best_indexes,
    get_final_text,
    read_target_dict_exclude_goldCannotAnswer,
    quac_performance_exclude_goldCannotAnswer
)

from modeling_auto_bert_ts import AutoModelForQuestionAnswering
from uce import eceloss, uceloss
from uce_utils import nentr
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
    
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    #eval_sampler = SequentialSampler(dataset)
    #eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #import pdb; pdb.set_trace()

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    all_examples = []
    all_features = []
    all_start_logits = []
    all_start_labels = []
    all_end_logits = []
    all_end_labels = []
    #for args.exclude_cannotanswer
    all_cannot_answer_uid = []
    all_results_without_cannotanswer = []

    start_time = timeit.default_timer()

    processor = QuacProcessor(tokenizer=tokenizer, threshold=args.threshold, conf_or_uncer = args.conf_or_uncer)

    with open(os.path.join(args.data_dir, args.predict_file), "r", encoding="utf-8") as reader: 
        input_data = json.load(reader)["data"]
    len_example = len(input_data)

    dict_prediction = {}
    unique_id = 1000000000
    example_index = 0

    for ex_idx in tqdm(range(len_example)):
        len_qa_idx_per_example = processor._calcaulte_qas_in_examples_number([input_data[ex_idx]])
        
        for qa_idx in range(len_qa_idx_per_example):
            data, example, features= load_examples(args, tokenizer, evaluate=True, output_examples=True, ex_idx=ex_idx, qa_idx=qa_idx, processor=processor, input_data=input_data, predicted_previous_qas=dict_prediction)
            
            eval_sampler = SequentialSampler(data)
            eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.eval_batch_size) #
            
            for batch in eval_dataloader:
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "temp_scale" : args.temp_scale, #TODO TODO
                        "bayesian" : args.bayesian, #TODO TODO
                        "T" : args.T, #TODO TODO,
                        "mc_drop_mask_num": args.mc_drop_mask_num, #TODO TODO,
                        #"label_smoothing" : args.label_smoothing, #TODO TODO
                    }

                    if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                        del inputs["token_type_ids"]

                    feature_indices = batch[3]
                    #import pdb; pdb.set_trace() #tokenizer.decode(batch[0][0])
                    # XLNet and XLM use more arguments for their predictions
                    if args.model_type in ["xlnet", "xlm"]:
                        inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                        # for lang_id-sensitive xlm models
                        if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                            inputs.update(
                                {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                            )
                    outputs = model(**inputs)
                
                #import pdb; pdb.set_trace()
                for i, feature_index in enumerate(feature_indices):
                    features[feature_index.item()].example_index = example_index
                    features[feature_index.item()].unique_id = unique_id
                    unique_id += 1
                    eval_feature = features[feature_index.item()]

                    _unique_id = int(eval_feature.unique_id)

                    #import pdb; pdb.set_trace()
                    
                    if args.bayesian:
                        start_end_outputs = outputs[:2]
                        #output = [output[:, i, :] for output in outputs]
                        start_end_output = [output[:, i, :] for output in start_end_outputs]
                        mc_start_logits, mc_end_logits = start_end_output
                        cls_logits = outputs[2][i].tolist()

                        mean_start_logits = mc_start_logits.mean(dim=0).tolist()
                        mean_end_logits = mc_end_logits.mean(dim=0).tolist()

                        start_output = torch.softmax(outputs[0], dim=2).mean(dim=0)
                        end_output = torch.softmax(outputs[1], dim=2).mean(dim=0)
                        
                        result = QuacResult(_unique_id, mean_start_logits, mean_end_logits, cls_logits)
                    else:
                        output = [to_list(output[i]) for output in outputs]
                        fq_start_logits, fq_end_logits, cls_logits = output

                        start_output = torch.softmax(outputs[0], dim=1)
                        end_output = torch.softmax(outputs[1], dim=1)

                        result = QuacResult(_unique_id, fq_start_logits, fq_end_logits, cls_logits)
                    
                    all_results.append(result)
                    #import pdb; pdb.set_trace()

                example_index += 1

                start_output = start_output.detach()
                end_output = end_output.detach()

                all_start_logits.append(start_output)
                all_end_logits.append(end_output)
                all_start_labels.append(batch[6].detach())
                all_end_labels.append(batch[7].detach())

                #confidences = (torch.max(start_output, 1)[0] + torch.max(end_output, 1)[0]) / 2
                
                uncertainties = (nentr(start_output, base=start_output.size(1)) + nentr(end_output, base=end_output.size(1))) / 2
                
                #import pdb; pdb.set_trace()
                predicted_answer, start_index, end_index, f_index = convert_predicted_answer_to_text(args, tokenizer, example[0], all_results, features)
                confidence = (start_output[f_index][start_index] + end_output[f_index][end_index]) / 2
                
                #if len(uncertainties) > 1:
                #    if f_index != 0:
                #        import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                dict_prediction[example[0].qas_id] = {'qid': example[0].qas_id,
                                                      'predicted_answer_text': predicted_answer,
                                                      'confidence': float(confidence),
                                                      'uncertainty': float(uncertainties[f_index]),
                                                      'answers': example[0].answers}
                #import pdb; pdb.set_trace()
                if args.exclude_cannotanswer:               
                    if predicted_answer != 'CANNOTANSWER':
                        all_examples = all_examples + example
                        all_features = all_features + features
                    else:
                        all_cannot_answer_uid = all_cannot_answer_uid + [f.unique_id for f in features]
                
                else:
                    all_examples = all_examples + example
                    all_features = all_features + features


    for r in all_results:
        if r.unique_id not in all_cannot_answer_uid:
            all_results_without_cannotanswer.append(r)

    if args.exclude_cannotanswer:
        logger.info("exclude_cannotanswer")
        logger.info("len(all_results): %d ", len(all_results))
        logger.info("len(all_features): %d ", len(all_features))
        logger.info("len(all_results_without_cannotanswer): %d ", len(all_results_without_cannotanswer))
        logger.info("len(all_examples): %d ", len(all_examples))
        
        all_results = all_results_without_cannotanswer
    else:
        logger.info("not exclude_cannotanswer")
        logger.info("len(all_results): %d ", len(all_results))
        logger.info("len(all_features): %d ", len(all_features))
        logger.info("len(all_examples): %d ", len(all_examples))
    #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs", evalTime)

    tensor_start_logits = torch.cat(all_start_logits, dim=0)
    tensor_end_logits = torch.cat(all_end_logits, dim=0)
    tensor_start_labels = torch.cat(all_start_labels, dim=0)
    tensor_end_labels = torch.cat(all_end_labels, dim=0)

    torch.save(tensor_start_logits, args.output_dir+'/plot/tensor_start_logits.pt')
    torch.save(tensor_end_logits, args.output_dir+'/plot/tensor_end_logits.pt')
    torch.save(tensor_start_labels, args.output_dir+'/plot/tensor_start_labels.pt')
    torch.save(tensor_end_labels, args.output_dir+'/plot/tensor_end_labels.pt')

    #import pdb; pdb.set_trace()
    ece_start, acc_start, conf_start, confidence_start = eceloss(tensor_start_logits, tensor_start_labels)
    ece_end, acc_end, conf_end, confidence_end = eceloss(tensor_end_logits, tensor_end_labels)

    mean_confidence = (confidence_start + confidence_end) / 2

    uce_start, err_start, entr_start, uncertainty_start = uceloss(tensor_start_logits, tensor_start_labels)
    uce_end, err_end, entr_end, uncertainty_end = uceloss(tensor_end_logits, tensor_end_labels)

    mean_uncertainty = (uncertainty_start + uncertainty_end) / 2

    mean_ece = (ece_start + ece_end) / 2
    mean_uce = (uce_start + uce_end) / 2

    #for valid
    if args.conf_or_uncer == 'uncer':
        print(mean_uce.item()*100)
    elif args.conf_or_uncer == 'conf':
        print(mean_ece.item()*100)

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


    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    #TODO
    output_nbest_with_start_index_file = os.path.join(args.output_dir, "nbest_predictions_with_start_idx_{}.json".format(prefix))
    output_nbest_conf_uncer_index_file = os.path.join(args.output_dir, "nbest_predictions_conf_uncer_{}.json".format(prefix))
    
    with open(output_nbest_conf_uncer_index_file, "w") as writer:
        writer.write(json.dumps(dict_prediction, indent=4) + "\n")

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

    else:
        predictions, nbest_predictions = compute_predictions_logits(
            all_examples,
            all_features,
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
            exclude_cannotanswer=args.exclude_cannotanswer
        )

    input_file = os.path.join(args.data_dir, args.predict_file)
    #import pdb; pdb.set_trace()
    # Compute the F1 and exact scores.
    if args.exclude_goldCannotAnswer:
        target_dict = read_target_dict_exclude_goldCannotAnswer(input_file)
        results, num_goldCannotAnswer, num_total_qas = quac_performance_exclude_goldCannotAnswer(predictions, target_dict)
        
        logger.info("len(num_goldCannotAnswer): %d", num_goldCannotAnswer)
        logger.info("len(num_total_qas): %d", num_total_qas)

    else:
        target_dict = read_target_dict(input_file)
        results = quac_performance(predictions, target_dict)

    return results
    

def convert_predicted_answer_to_text(args, tokenizer, example, all_results, features):
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "class_logit"]
    )

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min null score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    null_class_logit = None

    unique_id_to_result = {}
    for _result in all_results:
        unique_id_to_result[_result.unique_id] = _result

    for (feature_index, feature) in enumerate(features):
        #import pdb; pdb.set_trace()
        
        result = unique_id_to_result[feature.unique_id]
        
        start_indexes = _get_best_indexes(result.start_logits, 20)
        end_indexes = _get_best_indexes(result.end_logits, 20)
        
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
                #confidences = (start_output[0][start_index] + end_output[0][end_index]) / 2
                
                
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                #if example.qas_id == "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#1":
                #    import pdb; pdb.set_trace()
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
    #import pdb; pdb.set_trace()
    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

    #import pdb; pdb.set_trace() #왜 len(prelim_predictions)이 2인지, 위에 다시 확인
    seen_predictions = {}
    nbest = []

    #TODO
    for pred in prelim_predictions:
        if len(nbest) >= 1:
            break
        feature = features[pred.feature_index]
        #import pdb; pdb.set_trace()
        #conf = lst_features_conf_uncer[pred.feature_index][1]
        #uncer = lst_features_conf_uncer[pred.feature_index][2]
        
        #_start_index = prelim_predictions[0][1]
        #_end_index = prelim_predictions[0][2]
        f_index = pred.feature_index
        #import pdb; pdb.set_trace()

        
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
            
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = 'CANNOTANSWER'
            seen_predictions[final_text] = True
        
        nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        return final_text, pred.start_index, pred.end_index, f_index

    

def load_examples(args, tokenizer, evaluate=False, output_examples=False, ex_idx=0, qa_idx=0, processor=None, input_data=None, predicted_previous_qas=None):
    if evaluate:
        examples_per_ex_idx = processor._create_examples([input_data[ex_idx]], "dev")
        #import pdb; pdb.set_trace()
        examples_per_ex_idx[qa_idx].question_text = processor._concat_history(input_data[ex_idx]['paragraphs'][0]['qas'], predicted_previous_qas, qa_idx, 
                                                         max_history=args.max_history)
        example = [examples_per_ex_idx[qa_idx]]
        #import pdb; pdb.set_trace()
    features, dataset = quac_convert_examples_to_features(
        examples=example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
        excord=args.excord,
    )

    if output_examples:
        return dataset, example, features
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
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
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
    
    parser.add_argument("--excord", action='store_true', help="to use excord")
    parser.add_argument("--orig_loss_coeff", type=float, help="coeff for original loss")

    #TODO TODO
    parser.add_argument("--bayesian", action='store_true', help="to use bayesian")
    parser.add_argument("--temp_scale", action='store_true', help="to use temp_scale")
    parser.add_argument("--T", type=float, default="1.0", help="to use temp_scale")
    parser.add_argument("--label_smoothing", action='store_true', help="to use label_smoothing")
    parser.add_argument("--mc_drop_mask_num", type=int, default="10", help="mc_drop_mask_num")
    parser.add_argument("--threshold", type=float, help="threshold")
    parser.add_argument("--conf_or_uncer", type=str, help="conf_or_uncer")
    parser.add_argument("--max_history", type=int, help="max_history")
    parser.add_argument("--exclude_cannotanswer", action='store_true', help="to exclude cannotanswer")
    parser.add_argument("--exclude_goldCannotAnswer", action='store_true', help="to exclude goldCannotAnswer")
    
    
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
            #print(f1)
            result = {'F1' : f1}
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()