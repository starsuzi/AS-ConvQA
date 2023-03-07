INPUT_DIR=./datasets/quac
GPU_DEVICE=0
SEED=42
MAX_HISTORY=1

MC=mc_drop_true
TS=temp_scale_true
DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)

STEP1_T=1.4

###############STEP1 Inference###################
MODEL=step1/${SEED}/bsz12_epoch1_seed${SEED}/max_history/${MAX_HISTORY}
MODEL_DIR=BERT/${MODEL}
STEP1_TRAIN_OUTPUT_DIR=./model/${MODEL_DIR}

#STEP1_EVAL_OUTPUT_DIR=./outputs/${MODEL_DIR}/${MC}/${TS}/${DATE}/${STEP1_T}
STEP1_EVAL_OUTPUT_DIR=./outputs/${MODEL_DIR}/${MC}/${TS}/${STEP1_T}

TRAIN_SPLIT2_PREDICTION_FILE=${STEP1_EVAL_OUTPUT_DIR}/nbest_predictions_with_start_idx_conf_uncer_.json

STEP2_TRAIN_FILE=train_${MC}_${TS}_${STEP1_T}_${SEED}_${MAX_HISTORY}.json
OUTPUT_STEP2_TRAIN_PATH=${INPUT_DIR}/${STEP2_TRAIN_FILE}

if [ -d ${STEP1_EVAL_OUTPUT_DIR} ] 
then
    echo "Directory ${STEP1_EVAL_OUTPUT_DIR} exists." 
else
    echo "Directory ${STEP1_EVAL_OUTPUT_DIR} does not exists, so made it!"
	mkdir -p ${STEP1_EVAL_OUTPUT_DIR}
	mkdir -p ${STEP1_EVAL_OUTPUT_DIR}/plot
fi

CUDA_VISIBLE_DEVICES=${GPU_DEVICE} python run_quac_step1_infer.py \
	--model_type bert \
	--model_name_or_path ${STEP1_TRAIN_OUTPUT_DIR} \
	--cache_prefix bert-base-uncased \
	--data_dir ${INPUT_DIR} \
	--predict_file train.json \
	--output_dir ${STEP1_EVAL_OUTPUT_DIR} \
	--do_eval \
	--per_gpu_eval_batch_size 100 \
	--threads 20 \
	--do_lower_case \
	--seed ${SEED} \
	--max_history ${MAX_HISTORY} \
	--overwrite_cache \
	--bayesian \
	--temp_scale \
	--T ${STEP1_T}
	
python make_train_include_prediction.py --nbest_pred_1_path ${TRAIN_SPLIT2_PREDICTION_FILE} --output_step2_train_path ${OUTPUT_STEP2_TRAIN_PATH} --original_train_path ./datasets/quac/train.json
