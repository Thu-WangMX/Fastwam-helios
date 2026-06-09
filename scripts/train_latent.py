"""Latent 域训练入口。

调用方式:
    torchrun --standalone --nproc_per_node=4 scripts/train_latent.py \
        --config configs/robocasa365_test.yaml

    python scripts/train_latent.py --config configs/robocasa365_test.yaml
"""
import argparse
import logging
import os
import threading

import torch
import yaml

from fastwam.datasets.lerobot.latents.latent_mixture_dataset import LatentMixtureDataset
from fastwam.models.wan22.fastwam_memory import FastWAMMemory
from fastwam.trainer_latent import Wan22LatentTrainer
from fastwam.utils.logging_config import setup_logging
from fastwam.utils import misc


def _resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def _mixed_precision_to_dtype(mp: str) -> torch.dtype:
    mp = str(mp).strip().lower()
    if mp == "bf16":
        return torch.bfloat16
    if mp == "fp16":
        return torch.float16
    return torch.float32


def _create_model(model_cfg: dict, device: str, model_dtype: torch.dtype) -> FastWAMMemory:
    """从 yaml model 配置构建 FastWAMMemory。"""
    video_scheduler = model_cfg.get("video_scheduler", {})
    action_scheduler = model_cfg.get("action_scheduler", {})
    loss_cfg = model_cfg.get("loss", {})
    mtm_history_sizes = tuple(int(s) for s in model_cfg.get("mtm_history_sizes", [16, 2, 1]))
    mtm_patch_kernel_long = tuple(int(s) for s in model_cfg.get("mtm_patch_kernel_long", [4, 8, 8]))
    mtm_patch_kernel_mid = tuple(int(s) for s in model_cfg.get("mtm_patch_kernel_mid", [2, 4, 4]))
    mtm_patch_kernel_current = tuple(int(s) for s in model_cfg.get("mtm_patch_kernel_current", [1, 2, 2]))

    return FastWAMMemory.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=str(model_cfg["model_id"]),
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", model_cfg["model_id"])),
        tokenizer_max_len=int(model_cfg.get("tokenizer_max_len", 128)),
        load_text_encoder=bool(model_cfg.get("load_text_encoder", False)),
        proprio_dim=int(model_cfg["proprio_dim"]) if "proprio_dim" in model_cfg else None,
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
        video_dit_config=dict(model_cfg.get("video_dit_config", {})),
        action_dit_config=dict(model_cfg.get("action_dit_config", {})),
        action_dit_pretrained_path=model_cfg.get("action_dit_pretrained_path"),
        skip_dit_load_from_pretrain=bool(model_cfg.get("skip_dit_load_from_pretrain", False)),
        mot_checkpoint_mixed_attn=bool(model_cfg.get("mot_checkpoint_mixed_attn", False)),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss_cfg.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss_cfg.get("lambda_action", 1.0)),
        multi_term_memory=bool(model_cfg.get("multi_term_memory", True)),
        mtm_history_sizes=mtm_history_sizes,
        mtm_patch_kernel_long=mtm_patch_kernel_long,
        mtm_patch_kernel_mid=mtm_patch_kernel_mid,
        mtm_patch_kernel_current=mtm_patch_kernel_current,
        mtm_pred_size=int(model_cfg.get("mtm_pred_size", 2)),
        mtm_amplify_history=bool(model_cfg.get("mtm_amplify_history", False)),
        mtm_zero_history_timestep=bool(model_cfg.get("mtm_zero_history_timestep", True)),
    )


def main():
    setup_logging(
        log_level=logging.INFO,
        is_main_process=not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0,
    )

    parser = argparse.ArgumentParser(description="Latent 域训练")
    parser.add_argument("--config", required=True, help="task yaml 路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trainer_cfg = cfg["trainer"]
    misc.register_work_dir(trainer_cfg["output_dir"])

    if not torch.distributed.is_initialized():
        from accelerate import PartialState
        PartialState()

    import tqdm as _tqdm_mod
    if not hasattr(_tqdm_mod.tqdm, '_lock'):
        _tqdm_mod.tqdm._lock = threading.Lock()

    device = _resolve_device()
    model_dtype = _mixed_precision_to_dtype(trainer_cfg.get("mixed_precision", "bf16"))
    model_cfg = dict(cfg["model"])
    if "proprio_dim" not in model_cfg and "proprio_dim" in cfg.get("data", {}):
        model_cfg["proprio_dim"] = cfg["data"]["proprio_dim"]
    model = _create_model(model_cfg, device=device, model_dtype=model_dtype)

    cond_corruption = model_cfg.get("cond_latent_corruption")
    if cond_corruption is not None and hasattr(model, "set_cond_latent_corruption"):
        model.set_cond_latent_corruption(cond_corruption)

    seed = int(trainer_cfg.get("seed", 42))
    train_ds = LatentMixtureDataset.from_config(cfg, is_training=True, seed=seed)
    val_ds = LatentMixtureDataset.from_config(cfg, is_training=False, seed=seed)
    if len(val_ds) == 0:
        val_ds = None

    trainer = Wan22LatentTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
