#!/bin/bash
set -e
###########################################################################################
# AI Hub 提交：Latent 域训练
###########################################################################################

# === 配置 ===
CONFIG=configs/robocasa365_pretrain.yaml
NUM_GPUS=32
TASK_ID="robocasa365_pretrain"

# === 模型权重路径 ===
MODEL_BASE_PATH=/mnt/workspace/lintong.lt/wam/pretrain_checkpoint

###########################################################################################

ENVS="WANDB_MODE=online,WANDB_API_KEY=,\
DIFFSYNTH_MODEL_BASE_PATH=${MODEL_BASE_PATH},\
DIFFSYNTH_SKIP_DOWNLOAD=true,\
CUDA_HOME=/usr/local/cuda-12,\
HF_HOME=/mnt/workspace/lintong.lt/cache/huggingface,\
HF_ENDPOINT=https://hf-mirror.com,\
TOKENIZERS_PARALLELISM=false,\
TORCH_NCCL_ENABLE_MONITORING=1,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,\
NCCL_ASYNC_ERROR_HANDLING=1,\
NCCL_DEBUG=WARN,\
PYTHONUNBUFFERED=1,\
PYTHONPATH=src"

USER_PARAMS="--config ${CONFIG}"

echo "=============================================="
echo "Submit: Latent 训练"
echo "  CONFIG    = ${CONFIG}"
echo "  NUM_GPUS  = ${NUM_GPUS}"
echo "  TASK_ID   = ${TASK_ID}"
echo "=============================================="

ai-hub-cli train mdl --queue= \
  --name= \
  --namespace= \
  --token= \
  --entry="scripts/train_latent.py" \
  --algo_name=pytorch280 \
  --worker_count=${NUM_GPUS} \
  --user_params="${USER_PARAMS}" \
  --file.cluster_file=./cluster.json \
  --job_name="fastwam_train_${TASK_ID}" \
  --nas_file_system_id= \
  --nas_file_system_mount_path= \
  --env="${ENVS}"
