#!/bin/bash
set -x
set -e
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd ${PROJECT_DIR}

export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
export PYTHONPATH="./Megatron-LM:./":$PYTHONPATH

TP=8
PP=4

LOAD_PATH=model/Meta-Llama-3-8B
SAVE_PATH=model/Meta-Llama-3-8B-Megatron

python megatron_utils/convert/convert_llama3_to_megatron.py \
    --load_path $LOAD_PATH \
    --save_path $SAVE_PATH \
    --tokenizer_path $LOAD_PATH \
    --print-checkpoint-structure \
    --target_tensor_model_parallel_size $TP \
    --target_pipeline_model_parallel_size $PP \
    --target_data_parallel_size 1 \
    --target_params_dtype bf16
