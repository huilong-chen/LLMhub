#!/bin/bash
set -x
set -e
echo $PYTHONPATH
MODEL_PATH=/mnt/data/chenhuilong/model/Meta-llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=4,5,6,7
TASK_NAMES=gsm8k
OUTPUT_DIR=/predict

python eval/core/generate_with_api.py \
  --model_path $MODEL_PATH \
  --devices $CUDA_VISIBLE_DEVICES \
  --task_names $TASK_NAMES \
  --output_dir $OUTPUT_DIR

python eval/core/calculate_metrics.py \
  --model_path $MODEL_PATH \
  --task_names $TASK_NAMES \
  --output_dir $OUTPUT_DIR
