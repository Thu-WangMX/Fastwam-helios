#!/bin/bash
set -e
###########################################################################################
# AI Hub 提交：Stage-2 Latent 编码
###########################################################################################

# === 配置 ===
CONFIG=configs/robocasa365_pretrain.yaml
MANIFEST_PATH=./data/robocasa365_pretrain.jsonl
NUM_WORKERS=128

# === 模型权重路径 ===
MODEL_BASE_PATH=/mnt/workspace/lintong.lt/wam/pretrain_checkpoint

###########################################################################################

ENVS="WANDB_MODE=disabled,WANDB_DISABLED=true,\
CUDA_HOME=/usr/local/cuda-12,\
HF_HOME=/mnt/workspace/lintong.lt/cache/huggingface,\
HF_ENDPOINT=https://hf-mirror.com,\
NCCL_ASYNC_ERROR_HANDLING=1,\
NCCL_DEBUG=WARN,\
OMP_NUM_THREADS=8,\
TOKENIZERS_PARALLELISM=false,\
DIFFSYNTH_SKIP_DOWNLOAD=true,\
DIFFSYNTH_MODEL_BASE_PATH=${MODEL_BASE_PATH},\
PYTHONPATH=src"

USER_PARAMS="--config ${CONFIG} --manifest_path ${MANIFEST_PATH}"

echo "=============================================="
echo "Submit: Latent 编码"
echo "  CONFIG        = ${CONFIG}"
echo "  MANIFEST_PATH = ${MANIFEST_PATH}"
echo "  NUM_WORKERS   = ${NUM_WORKERS}"
echo "=============================================="

ai-hub-cli train mdl --queue= \
  --name= \
  --namespace= \
  --token= \
  --entry="scripts/latent/generate_latents.py" \
  --algo_name=pytorch280 \
  --worker_count=${NUM_WORKERS} \
  --user_params="${USER_PARAMS}" \
  --file.cluster_file=./cluster.json \
  --job_name="fastwam_generate_latents" \
  --nas_file_system_id= \
  --nas_file_system_mount_path= \
  --env="${ENVS}"
