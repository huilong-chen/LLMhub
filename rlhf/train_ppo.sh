#!/bin/bash
set -x
set -e

CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --config_file ./ds_config/deepspeed_zero3.yaml rlhf_train.py \
  --train_args_path "rlhf/args/ppo_config.py" \
  --train_mode "lora" \
  --use_dora False \
  --rlhf_type "PPO" \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05