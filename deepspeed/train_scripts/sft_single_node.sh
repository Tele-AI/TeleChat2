#!/bin/bash
# 单机多卡微调(SFT)模型

# 配置可用GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# 模型文件路径、输出路径、DeepSpeed配置路径、训练数据路径
MODEL_PATH=$1
OUTPUT_DIR=./test_output
DEEPSPEED=./deepspeed_configs/ds_z2_config.json
DATA_CONFIG_FILE=./dummy_data/sft_data_config.json

# 训练配置
# TRAIN_MODE 可选 full \ lora \ qlora
TASK_TYPE=sft
TRAIN_MODE=full

# 训练参数配置
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=3e-5
MAX_SEQ_LENGTH=4096
WARMUP_RATIO=0.05
LOGGING_STEPS=10
SAVE_STEPS=100
LR_SCHEDULER_TYPE=cosine
GRADIENT_CHECKPOINTING=true
SEED=42
FP16=true
BF16=false
OPTIM=adamw_apex_fused

deepspeed --include localhost:$CUDA_VISIBLE_DEVICES ./train.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --data_config_file $DATA_CONFIG_FILE \
    --deepspeed $DEEPSPEED \
    --task_type $TASK_TYPE \
    --train_mode $TRAIN_MODE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --warmup_ratio $WARMUP_RATIO \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --seed $SEED \
    --fp16 $FP16 \
    --bf16 $BF16 \
    --save_total_limit 1 \
    --optim $OPTIM
