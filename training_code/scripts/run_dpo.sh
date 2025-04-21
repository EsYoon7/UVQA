#!/bin/bash
# export NCCL_P2P_DISABLE=1
export TZ="Asia/Seoul"

set -e

OUTPUT=$1

MODEL_NAME="VLM-RLAIF-Video-LLAVA-7b"
LOG_DIR=[YOUR_LOG_HOME_DIRECTORY_HERE]

VISION_TOWER=[YOUR_VISON_TOWER_DIRECTORY_HERE]
SFT_MODEL_PATH=[YOUR_SFT_MODEL_DIRECTORY_HERE]
MODEL_BASE_PATH=[YOUR_MODEL_BASE_DIRECTORY_HERE]

# use the dataset setting as you want
# DATASET_SETTING="uvqa_relation+uvqa_object+uvqa_attribute+video_chatgpt1+video_chatgpt2+video_chatgpt3"
DATASET_SETTING="uvqa_relation+uvqa_object+uvqa_attribute" 

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="${LOG_DIR}/log-all/dpo-${MODEL_NAME/'/'/_}-$TIME_STEP"
fi
mkdir -p $OUTPUT

# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=1e-6
BATCH_SIZE=4
EVAL_BATCH_SIZE=8
GRAD_ACCUMULATION=16

LOG_STEPS=5
EVAL_STEPS=100
SAVE_STEPS=5

# Set your number of GPUs here
NUM_GPUS=2

RUN_NAME="DPO_vlm_rlaif_lr${LR}_bs${BATCH_SIZE}_gradaccum_${GRAD_ACCUMULATION}"

ACCELERATE_CONFIG='./training_code/accelerate_configs/deepspeed_zero2.yaml'if [[ "${ACCELERATE_CONFIG}" == "" ]]; then
  EXTRA_ACCELERATE_ARGS=""
else
  EXTRA_ACCELERATE_ARGS="--config_file $ACCELERATE_CONFIG"
fi

CMD="""
accelerate launch $EXTRA_ACCELERATE_ARGS \
    --num_processes=$NUM_GPUS \
    ./training_code/code/dpo.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --run_name $RUN_NAME \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $SFT_MODEL_PATH \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --max_eval_samples 300 \
    --bits 16 \
    --output_dir $OUTPUT \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOG_STEPS \
    --save_total_limit 50 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --report_to "wandb" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --image_aspect_ratio 'pad' \
    --torch_dtype bfloat16 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --optim "adamw_torch" \
    --generate_during_eval True \
    --lora_enable False \
    --dataset_setting $DATASET_SETTING \
    $EXTRA_TRAINING_ARGS
"""

{ # try
    echo $CMD
    eval "$CMD"
} || { # catch
    # save log for exception 
    echo "Operation Failed!"
    exit 1
}
exit 0
