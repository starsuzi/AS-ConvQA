INPUT_DIR=./datasets/quac
GPU_DEVICE=0
SEED=42
MAX_HISTORY=1

DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)

###############STEP1 Train###################
MODEL=step1/${SEED}/bsz12_epoch1_seed${SEED}/max_history/${MAX_HISTORY}
#MODEL_DIR=BERT/${MODEL}/${DATE}
MODEL_DIR=BERT/${MODEL}
STEP1_TRAIN_OUTPUT_DIR=./model/${MODEL_DIR}

mkdir -p ${STEP1_TRAIN_OUTPUT_DIR}
 ###
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python run_quac_step1_train.py \
	--model_type bert \
	--model_name_or_path bert-base-uncased \
	--do_train \
	--data_dir ${INPUT_DIR} \
	--train_file train.json \
	--output_dir ${STEP1_TRAIN_OUTPUT_DIR} \
	--per_gpu_train_batch_size 12 \
	--num_train_epochs 1 \
	--learning_rate 3e-5 \
	--weight_decay 0.01 \
	--threads 20 \
	--orig_loss_coeff 1.0 \
	--do_lower_case \
	--seed ${SEED} \
	--overwrite_cache \
	--max_history ${MAX_HISTORY}