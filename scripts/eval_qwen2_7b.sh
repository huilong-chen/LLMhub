#!/bin/bash
set -x
set -e
echo $PYTHONPATH
MODEL_PATH=/mnt/data/chenhuilong/model/Qwen2-7B
CUDA_VISIBLE_DEVICES=4,5,6,7
TASK_NAMES=gsm8k
OUTPUT_DIR=/predict
SERVER_PORT=6006

python eval/core/generate.py \
  --model_path $MODEL_PATH \
  --devices $CUDA_VISIBLE_DEVICES \
  --task_names $TASK_NAMES \
  --output_dir $OUTPUT_DIR \
  --server_port $SERVER_PORT

python eval/core/calculate_metrics.py \
  --model_path $MODEL_PATH \
  --task_names $TASK_NAMES \
  --output_dir $OUTPUT_DIR
