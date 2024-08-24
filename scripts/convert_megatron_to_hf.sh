#!/bin/bash
set -x
set -e
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd ${PROJECT_DIR}

export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
export PYTHONPATH="./":$PYTHONPATH

python utils/megatron_utils/convert/convert_llama3_to_megatron.py \
    --load_path model/Meta-Llama-3-8B-Megatron/release \
    --save_path model/Meta-Llama-3-8B-HF \
    --tokenizer_path model/Meta-Llama-3-8B \
    --target_tensor_model_parallel_size 2 \
    --target_pipeline_model_parallel_size 1 \
    --target_data_parallel_size 1 \
    --target_params_dtype bf16 \
    --print-checkpoint-structure \
    --convert_checkpoint_from_megatron_to_transformers \
