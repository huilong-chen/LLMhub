set -e
set -x

DIR=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd ${DIR}

MODEL_NAME=Meta-llama-3-8B-Instruct
MODEL_PATH=/mnt/data/chenhuilong/model/Meta-llama-3-8B-Instruct
TENSOR_PARALLEL=4
PORT=6006

if [ -z "$MODEL_GROUP" ]; then
  MODEL_GROUP=nlp-llm
fi

if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH=/home/shared/models/$MODEL_NAME-$MODEL_TAG
  wget https://infra-ai-studio-public.oss-cn-beijing-internal.aliyuncs.com/aic --no-verbose && chmod +x ./aic
  ./aic --token "$AIC_TOKEN" model prepare --name "$MODEL_NAME" --group "$MODEL_GROUP" --tag "$MODEL_TAG" -p "$MODEL_PATH"
  MODEL_NAME=$MODEL_NAME-$MODEL_TAG

  export PYTHONUSERBASE=./pkg
  export PATH=$PATH:./pkg/bin
  python -m cupyx.tools.install_library --library nccl --cuda 11.x
fi
if [[ $(pip list | grep -c vllm) -eq 0 ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
  export PATH=/opt/conda/envs/nlp-llm-eval/bin:$PATH
fi

python -m server.api.api_vllm --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --tensor_parallel "$TENSOR_PARALLEL" \
  --port $PORT
