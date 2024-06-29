#!/bin/bash
set -x
set -e
echo $PYTHONPATH
MODEL_PATH=path_to_model
CUDA_VISIBLE_DEVICES=0,1,2,3
TASK_NAMES=gsm8k
OUTPUT_DIR=/predict

python eval/core/generate.py \
  --model_path $MODEL_PATH \
  --devices $CUDA_VISIBLE_DEVICES \
  --task_names $TASK_NAMES \
  --output_dir $OUTPUT_DIR

python eval/core/calculate_metrics.py \
  --model_path $MODEL_PATH \
  --task_names $TASK_NAMES \
  --output_dir $OUTPUT_DIR
