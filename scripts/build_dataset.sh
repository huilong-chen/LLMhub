#!/bin/bash
set -x
set -e

export PYTHONPATH="./":$PYTHONPATH

python finetune/build_dataset.py \
  --input_file_path data/Code-Feedback/Code-Feedback.jsonl \
  --output_dir data/build_dataset \
  --max_seq_length 8192 \
  --tokenizer_path model/Meta-Llama-3-8B
