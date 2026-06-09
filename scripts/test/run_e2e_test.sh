#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# FastWAM 端到端测试脚本
# 数据集: /mnt/nas-9/robot_manipulation/robocasa365_test (4个子数据集)
# ═══════════════════════════════════════════════════════════════════
set -e

cd /mnt/workspace/lintong.lt/wam/tmp/FastWAM

export PYTHONPATH=src
PYTHON=/mnt/workspace/lintong.lt/env/fastwam/bin/python
CONFIG=configs/robocasa365_test.yaml
MANIFEST=./data/robocasa365_test_manifest.jsonl
NUM_WORKERS=8

echo "═══════════════════════════════════════════════════"
echo " FastWAM 端到端测试"
echo " config: $CONFIG"
echo " manifest: $MANIFEST"
echo "═══════════════════════════════════════════════════"

# # ─── Step 1: Text Embedding 生成 ───
# echo ""
# echo "[Step 1/5] Text Embedding 生成..."
# $PYTHON scripts/precompute_text_embeds.py --config $CONFIG
# echo "[Step 1/5] 完成"

# # ─── Step 2a: Scan 元数据 → Manifest ───
# echo ""
# echo "[Step 2a/5] Scan 元数据..."
# mkdir -p ./data
# $PYTHON scripts/latent/scan_dataset_meta.py \
#     --config $CONFIG \
#     --manifest_path $MANIFEST \
#     --num_workers $NUM_WORKERS
# echo "[Step 2a/5] 完成"

# # ─── Step 2b: Latent 编码（4卡并行） ───
# echo ""
# echo "[Step 2b/5] Latent 编码（4卡并行）..."
# torchrun --standalone --nproc_per_node=4 scripts/latent/generate_latents.py \
#     --config $CONFIG \
#     --manifest_path $MANIFEST
# echo "[Step 2b/5] 完成"

# # ─── Step 3: 归一化统计 ───
# echo ""
# echo "[Step 3/5] 归一化统计计算..."
# $PYTHON scripts/compute_dataset_stats.py \
#     --config $CONFIG \
#     --num_workers $NUM_WORKERS
# echo "[Step 3/5] 完成"

# # ─── Step 4: Train/Val 划分 ───
# echo ""
# echo "[Step 4/5] Train/Val 划分..."
# $PYTHON scripts/split_train_val.py --config $CONFIG
# echo "[Step 4/5] 完成"

# ─── Step 5: 训练（100 步，4卡） ───
echo ""
echo "[Step 5/5] 训练（100 步，4卡）..."
torchrun --standalone --nproc_per_node=4 scripts/train_latent.py --config $CONFIG
echo "[Step 5/5] 完成"

echo ""
echo "═══════════════════════════════════════════════════"
echo " 全部测试通过"
echo "═══════════════════════════════════════════════════"
