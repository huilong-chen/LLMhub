#!/bin/bash
set -x
set -e

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file rlhf/ds_config/deepspeed_zero3.yaml rlhf/reward_model.py \
  --max_length None \
  --train_data_path "rlhf/data.jsonl" \
  --output_dir "rlhf/output" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 16 \
  --learning_rate 0.0002 \
  --logging_steps 100 \
  --save_steps 500 \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --lr_scheduler_type "constant_with_warmup" \
  --warmup_steps 10 \
  --optim "paged_adamw_32bit" \
  --seed 42 \
  --report_to "tensorboard" \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --remove_unused_columns False \
  --bf16 True \
  --model_name_or_path "model/Qwen2-7B" \
  --trust_remote_code True \
  --use_peft True \
  --torch_dtype "bfloat16" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_task_type "SEQ_CLS" \
  --load_in_4bit False \
  --bnb_4bit_quant_type "nf4" \
  --use_bnb_nested_quant True