#!/bin/bash
# ActionDiT 权重预处理：从 VideoDiT 线性插值初始化
export DIFFSYNTH_MODEL_BASE_PATH="/mnt/workspace/lintong.lt/wam/pretrain_checkpoint"
export PYTHONPATH=src

/mnt/workspace/lintong.lt/env/fastwam/bin/python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/robocasa365_pretrain.yaml \
  --device cuda \
  --dtype bfloat16 \
  --output /mnt/workspace/lintong.lt/wam/pretrain_checkpoint/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
