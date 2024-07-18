#!/bin/bash
set -x
set -e
echo $PYTHONPATH
MODEL_PATH=$1
CUDA_VISIBLE_DEVICES=$2
TASK_NAMES=$3
OUTPUT_DIR=$4
SERVER_PORT=$5

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
