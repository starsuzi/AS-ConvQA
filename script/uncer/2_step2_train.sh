#!/bin/bash

INPUT_DIR=./datasets/quac
GPU_DEVICE=0
SEED=42
MAX_HISTORY=1
MC=mc_drop_true
TS=temp_scale_true
DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)

STEP1_T=1.4
STEP2_T=1.4

###############STEP2 Train###################
MODEL=step1/${SEED}/bsz12_epoch1_seed${SEED}/max_history/${MAX_HISTORY}
MODEL_DIR=BERT/${MODEL}
STEP1_TRAIN_OUTPUT_DIR=./model/${MODEL_DIR}

STEP2_TRAIN_FILE=train_${MC}_${TS}_${STEP1_T}_${SEED}_${MAX_HISTORY}.json

MODEL=predicted_uncer_bsz12_epoch1_seed${SEED}/${MC}/${TS}/${STEP1_T}/max_history/${MAX_HISTORY} ###
#MODEL_DIR=BERT/step2/${SEED}/${MODEL}/${DATE}
MODEL_DIR=BERT/step2/${SEED}/${MODEL}
STEP2_TRAIN_OUTPUT_DIR=./model/${MODEL_DIR}

mkdir -p ${STEP2_TRAIN_OUTPUT_DIR}

 ###
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python run_quac_step2_train.py \
	--model_type bert \
	--model_name_or_path ${STEP1_TRAIN_OUTPUT_DIR} \
	--do_train \
	--data_dir ${INPUT_DIR} \
	--train_file ${STEP2_TRAIN_FILE} \
	--output_dir ${STEP2_TRAIN_OUTPUT_DIR} \
	--per_gpu_train_batch_size 12 \
	--num_train_epochs 1 \
	--learning_rate 3e-5 \
	--weight_decay 0.01 \
	--threads 20 \
	--orig_loss_coeff 1.0 \
	--do_lower_case \
	--seed ${SEED} \
	--conf_or_uncer uncer \
	--overwrite_cache \
	--max_history ${MAX_HISTORY} \
	#--overwrite_output_dir