#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

GLOBAL_BATCH_SIZE=64
export NCCL_DEBUG=WARN

BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_sft.py \
    --use_liger False \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path ./data/train_file/split/overall_quality/train.jsonl \
    --val_path ./data/train_file/split/overall_quality/val.jsonl \
    --image_folder ./data/DIQA/train/res \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/deqa \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((4 * 28 * 28)) \
    --image_max_pixels $((16000 * 28 * 28)) \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --dataloader_num_workers 16 \
    --level_prefix "The quality of the image is" \
    --level_names five four three two one \
    --softkl_loss True \
    --train_deqa \
    --max_grad_norm 1

