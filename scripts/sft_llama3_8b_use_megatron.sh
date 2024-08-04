#!/bin/bash
set -x
set -e
# NOTE: 以下这些环境变量仅适用于 A800 机器
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none
export NCCL_NVLS_ENABLE=0

export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
export PYTHONPATH="./":$PYTHONPATH
export MNT_HOME=$(cd ".." && pwd)

if [ -z $MASTER_ADDR ]; then
  MASTER_ADDR=localhost
  MASTER_PORT=$(shuf -n 1 -i 10000-65535)
  NNODES=1
  NODE_RANK=0
  GPUS_PER_NODE=gpu
else
  NNODES=${WORLD_SIZE}
  NODE_RANK=${RANK}
  GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

SEED=42

YOUR_TASK_NAME=sft-llama3-8b-code-feedback
YOUR_USER_NAME=chenhuilong
YOUR_SAVE_NAME=${YOUR_TASK_NAME}

PRETRAIN_CHECKPOINT_PATH=model/Meta-Llama-3-8B-Megatron
TOKENIZER_PATH=model/Meta-Llama-3-8B

# ----------- hyperparameters -----------
MAX_LENGTH=8192
GLOBAL_BATCH_SIZE=64
BATCH_SIZE=1
LR=5e-6
MIN_LR=1e-6
TP=2
PP=1
EPOCH=3
SAVE_INT=10000
SAVE_EPOCH=1
EVAL_INT=0
LR_WARMUP_ITERS=0

# load data
TRAIN_DATASET_PATH=data/my-code-feedback/train
# save
SAVE_CHECKPOINT_BASEPATH=model/checkpoint/${YOUR_TASK_NAME}/output
TENSORBOARD_BASEPATH=model/tensorboard/${YOUR_TASK_NAME}
# convert
CONVERT_BACK_TO_LLAMA=each
SAVE_HF_MODEL_PATH=model/$YOUR_SAVE_NAME

current_time=$(date "+%Y.%m.%d")
FT_NAME="${YOUR_SAVE_NAME}-lr-${LR}-bs-${BATCH_SIZE}-${current_time}"
TENSORBOARD_DIR="${TENSORBOARD_BASEPATH}/${FT_NAME}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_BASEPATH}/${FT_NAME}"
if [[ -d "${SAVE_CHECKPOINT_PATH}" && ! -z "$(ls -A ${SAVE_CHECKPOINT_PATH})" ]]; then
  echo "The path ${SAVE_CHECKPOINT_PATH} already exists and is not empty."
  exit 1
fi
mkdir -p "${TENSORBOARD_DIR}"
mkdir -p "${SAVE_CHECKPOINT_PATH}"

DISTRIBUTED_ARGS="
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT
"

MODEL_ARGS="
  --num-layers 32 \
  --hidden-size 4096 \
  --num-attention-heads 32 \
  --seq-length $MAX_LENGTH \
  --max-position-embeddings $MAX_LENGTH \
  --ffn-hidden-size 14336 \
  --init-method-std 0.01 \
  --position-embedding-type rope \
  --swiglu \
  --attention-dropout 0 \
  --hidden-dropout 0 \
  --norm-epsilon 1e-5 \
  --group-query-attention \
  --num-query-groups 8 \
  --rope-base 500000.0 \
  --no-rope-fusion \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --normalization RMSNorm \
  --transformer-impl transformer_engine \
"

TRAIN_ARGS="
  --load ${PRETRAIN_CHECKPOINT_PATH} \
  --save ${SAVE_CHECKPOINT_PATH} \
  --train-data ${TRAIN_DATASET_PATH} \
  --micro-batch-size ${BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --epochs ${EPOCH} \
  --eval-iters 2 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --eval-interval $EVAL_INT \
  --save-interval $SAVE_INT \
  --save-epoch-interval $SAVE_EPOCH \
  --tensorboard-queue-size 5 \
  --tensorboard-dir ${TENSORBOARD_DIR} \
  --log-timers-to-tensorboard \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --finetune \
  --no-load-optim \
  --no-load-rng \
  --seed ${SEED} \
  --tokenizer-type NullTokenizer \
  --bf16 \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --num-workers 3 \
  --use-distributed-optimizer \
  --use-flash-attn \
  --no-save-optim \
  --log-throughput \
  --empty-unused-memory-level 2 \
  --tokenizer_path ${TOKENIZER_PATH}
"

LR_ARGS="
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style linear \
  --lr-warmup-iters ${LR_WARMUP_ITERS} --sequence-parallel
"

torchrun $DISTRIBUTED_ARGS \
  finetune/sft/sft_llama3_8b.py \
  $MODEL_ARGS \
  $TRAIN_ARGS \
  $LR_ARGS