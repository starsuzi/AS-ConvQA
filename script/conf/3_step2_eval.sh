#!/bin/bash

INPUT_DIR=./datasets/quac
GPU_DEVICE=0
SEED=42
MAX_HISTORY=1
MC=mc_drop_false
TS=temp_scale_true
DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)

STEP1_T=1.1
STEP2_T=1.1

###############STEP2 Eval with 0.15###################
MODEL=predicted_conf_bsz12_epoch1_seed${SEED}/${MC}/${TS}/${STEP1_T}/max_history/${MAX_HISTORY} ###
#MODEL_DIR=BERT/step2/${SEED}/${MODEL}/${DATE}
MODEL_DIR=BERT/step2/${SEED}/${MODEL}
STEP2_TRAIN_OUTPUT_DIR=./model/${MODEL_DIR}

STEP2_EVAL_OUTPUT_DIR=./outputs/${MODEL_DIR}/${MC}/${TS}/dev/${STEP2_T}/threshold/0.15 ###


if [ -d ${STEP2_EVAL_OUTPUT_DIR} ] 
then
    echo "Directory ${STEP2_EVAL_OUTPUT_DIR} exists." 
else
    echo "Directory ${STEP2_EVAL_OUTPUT_DIR} does not exists, so made it!"
	mkdir -p ${STEP2_EVAL_OUTPUT_DIR}
	mkdir -p ${STEP2_EVAL_OUTPUT_DIR}/plot
fi

 ###
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python run_quac_step2_eval.py \
	--model_type bert \
	--model_name_or_path ${STEP2_TRAIN_OUTPUT_DIR} \
	--cache_prefix bert-base-uncased \
	--data_dir ${INPUT_DIR} \
	--predict_file val.json \
	--output_dir ${STEP2_EVAL_OUTPUT_DIR} \
	--do_eval \
	--per_gpu_eval_batch_size 100 \
	--threads 20 \
	--do_lower_case \
	--seed ${SEED} \
	--overwrite_cache \
	--T ${STEP2_T} \
	--temp_scale \
	--threshold 0.15 \
	--conf_or_uncer conf \
	--max_history ${MAX_HISTORY}


###############STEP2 Eval with 0.20###################
MODEL=predicted_conf_bsz12_epoch1_seed${SEED}/${MC}/${TS}/${STEP1_T}/max_history/${MAX_HISTORY} ###
#MODEL_DIR=BERT/step2/${SEED}/${MODEL}/${DATE}
MODEL_DIR=BERT/step2/${SEED}/${MODEL}
STEP2_TRAIN_OUTPUT_DIR=./model/${MODEL_DIR}

STEP2_EVAL_OUTPUT_DIR=./outputs/${MODEL_DIR}/${MC}/${TS}/dev/${STEP2_T}/threshold/0.20 ###


if [ -d ${STEP2_EVAL_OUTPUT_DIR} ] 
then
    echo "Directory ${STEP2_EVAL_OUTPUT_DIR} exists." 
else
    echo "Directory ${STEP2_EVAL_OUTPUT_DIR} does not exists, so made it!"
	mkdir -p ${STEP2_EVAL_OUTPUT_DIR}
	mkdir -p ${STEP2_EVAL_OUTPUT_DIR}/plot
fi

 ###
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python run_quac_step2_eval.py \
	--model_type bert \
	--model_name_or_path ${STEP2_TRAIN_OUTPUT_DIR} \
	--cache_prefix bert-base-uncased \
	--data_dir ${INPUT_DIR} \
	--predict_file val.json \
	--output_dir ${STEP2_EVAL_OUTPUT_DIR} \
	--do_eval \
	--per_gpu_eval_batch_size 100 \
	--threads 20 \
	--do_lower_case \
	--seed ${SEED} \
	--overwrite_cache \
	--T ${STEP2_T} \
	--temp_scale \
	--threshold 0.20 \
	--conf_or_uncer conf \
	--max_history ${MAX_HISTORY}