#!/bin/bash
set -e
###########################################################################################
# 本地预处理脚本（CPU 操作，无需 GPU / AI Hub）
#
# 包含：
#   1. Scan 元数据 → manifest + sample index
#   2. 归一化统计
#   3. Train/Val 划分
###########################################################################################

CONFIG=configs/robocasa365_pretrain.yaml
MANIFEST_PATH=./data/robocasa365_pretrain.jsonl
NUM_WORKERS=32

export PYTHONPATH=src
PYTHON=/mnt/workspace/lintong.lt/env/fastwam/bin/python

echo "=============================================="
echo " 本地预处理（scan + stats + split）"
echo " CONFIG: ${CONFIG}"
echo "=============================================="

# ─── Step 1: Scan 元数据 ───
echo ""
echo "[1/3] Scan 元数据 → manifest..."
mkdir -p ./data
$PYTHON scripts/latent/scan_dataset_meta.py \
    --config ${CONFIG} \
    --manifest_path ${MANIFEST_PATH} \
    --num_workers ${NUM_WORKERS}
echo "[1/3] 完成"

# ─── Step 2: 归一化统计 ───
echo ""
echo "[2/3] 归一化统计计算..."
$PYTHON scripts/compute_dataset_stats.py \
    --config ${CONFIG} \
    --num_workers ${NUM_WORKERS}
echo "[2/3] 完成"

# ─── Step 3: Train/Val 划分 ───
echo ""
echo "[3/3] Train/Val 划分..."
$PYTHON scripts/split_train_val.py --config ${CONFIG}
echo "[3/3] 完成"

echo ""
echo "=============================================="
echo " 全部完成"
echo "=============================================="
