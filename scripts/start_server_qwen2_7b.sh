set -e
set -x

DIR=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd ${DIR}

MODEL_NAME=Qwen2-7B
MODEL_PATH=/mnt/data/chenhuilong/model/Qwen2-7B
TENSOR_PARALLEL=1
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

  is_cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
  if [ $is_cuda_available = "False" ]; then
      echo "CUDA is not available. Unset LD_LIBRARY_PATH."
      unset LD_LIBRARY_PATH
  else
      echo "CUDA is available."
  fi
fi

if [[ $(pip list | grep -c vllm) -eq 0 ]]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
  export PATH=/opt/conda/envs/nlp-llm-eval/bin:$PATH
fi
export PATH=/opt/conda/envs/nlp-llm-eval/bin:$PATH

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((TENSOR_PARALLEL - 1)))
  export CUDA_VISIBLE_DEVICES

CUDA_VISIBLE_DEVICES=6

fi
export VLLM_WORKER_MULTIPROC_METHOD=fork
python -m server.api.api_vllm --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --tensor_parallel "$TENSOR_PARALLEL" \
  --port $PORT
