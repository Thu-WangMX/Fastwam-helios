"""Latent 域训练器。

基于 dict config + LatentMixtureDataset + MixtureBucketSampler。

用法:
    trainer = Wan22LatentTrainer(model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    trainer.train()
"""
import json
import os
import time
from math import ceil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .datasets.lerobot.latents.latent_mixture_dataset import LatentMixtureDataset
from .datasets.lerobot.latents.mixture_bucket_sampler import MixtureBucketSampler, MixtureSampler
from .models.wan22.fastwam_memory import FastWAMMemory
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger
from .utils.pytorch_utils import set_global_seed
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


def _to_jsonable(v):
    if isinstance(v, (int, float, str, bool)):
        return v
    try:
        return float(v)
    except Exception:
        return str(v)


logger = get_logger(__name__)


class Wan22LatentTrainer:

    def __init__(
        self,
        model: FastWAMMemory,
        train_dataset: LatentMixtureDataset,
        val_dataset: Optional[LatentMixtureDataset] = None,
        *,
        cfg: dict,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg

        trainer_cfg = cfg["trainer"]
        self.output_dir = str(trainer_cfg["output_dir"])
        self.learning_rate = float(trainer_cfg["learning_rate"])
        self.weight_decay = float(trainer_cfg.get("weight_decay", 1e-2))
        self.batch_size = int(trainer_cfg["batch_size"])
        self.num_workers = int(trainer_cfg.get("num_workers", 8))
        self.num_epochs = int(trainer_cfg.get("num_epochs", 1))
        max_steps = trainer_cfg.get("max_steps")
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(trainer_cfg.get("log_every", 10))
        self.save_every = int(trainer_cfg.get("save_every", 2500))
        self.eval_every = int(trainer_cfg.get("eval_every", 500))
        self.eval_num_inference_steps = int(trainer_cfg.get("eval_num_inference_steps", 10))
        self.gradient_accumulation_steps = int(trainer_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(trainer_cfg.get("max_grad_norm", 1.0))
        self.seed = int(trainer_cfg.get("seed", 42))
        self.resume = trainer_cfg.get("resume")

        self.mixed_precision = str(trainer_cfg.get("mixed_precision", "bf16")).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision: {self.mixed_precision}")

        wandb_cfg = cfg.get("wandb", {})
        self.wandb_enabled = bool(wandb_cfg.get("enabled", False))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        logger.info(
            "Accelerate: distributed_type=%s world_size=%d process_index=%d mixed_precision=%s",
            self.accelerator.distributed_type, self.accelerator.num_processes,
            self.accelerator.process_index, self.accelerator.mixed_precision,
        )

        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate,
            weight_decay=self.weight_decay, betas=(0.9, 0.95),
        )

        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            self.val_loader = self._build_eval_loader(self.val_dataset, worker_init_fn=worker_init_fn)
        else:
            self.val_loader = None
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=trainer_cfg.get("lr_scheduler_type", "cosine"),
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        for d in [self.output_dir, self.checkpoint_root, self.weights_dir, self.state_dir, self.eval_dir]:
            ensure_dir(d)

        if self.accelerator.is_main_process:
            with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
                import yaml
                yaml.dump(cfg, f, default_flow_style=False)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler,
        )
        self.optimizer.zero_grad(set_to_none=True)

        self.wandb_run = None
        self._init_wandb()
        self._local_metrics_fp = None
        if self.accelerator.is_main_process:
            metrics_path = os.path.join(self.output_dir, "train_metrics.jsonl")
            self._local_metrics_fp = open(metrics_path, "a", buffering=1)

        self._resume_or_load_checkpoint()
        logger.info("Train/val dataset size: %d / %d", len(self.train_dataset),
                     len(self.val_dataset) if self.val_dataset else 0)

    # ══════════════════════════════════════════════════════════════════════
    # Dataset length 一致性检查
    # ══════════════════════════════════════════════════════════════════════

    def _assert_dataset_length_consistent(self, dataset, name: str):
        local_len = len(dataset)
        gathered = self.accelerator.gather(
            torch.tensor([local_len], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if not torch.all(gathered == gathered[0]):
            if self.accelerator.is_main_process:
                for rank, rank_len in enumerate(gathered.cpu().tolist()):
                    logger.error("  rank %d: %s len=%d", rank, name, rank_len)
            self.accelerator.wait_for_everyone()
            raise RuntimeError(f"{name} length mismatch across ranks: {gathered.cpu().tolist()}")

    # ══════════════════════════════════════════════════════════════════════
    # DataLoader
    # ══════════════════════════════════════════════════════════════════════

    def _build_loader(self, dataset: LatentMixtureDataset, worker_init_fn=None) -> DataLoader:
        bucket_groups = dataset.get_all_bucket_groups()
        if len(bucket_groups) > 1:
            self.train_sampler = MixtureBucketSampler(
                dataset, batch_size=self.batch_size,
                global_rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes,
                seed=self.seed,
            )
            return DataLoader(
                dataset, batch_sampler=self.train_sampler,
                num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
            )
        else:
            self.train_sampler = MixtureSampler(
                dataset, global_rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes, seed=self.seed,
            )
            return DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False,
                sampler=self.train_sampler, num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
            )

    def _build_eval_loader(self, dataset: LatentMixtureDataset, worker_init_fn=None) -> DataLoader:
        sampler = MixtureSampler(
            dataset, global_rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes, shuffle=False, seed=self.seed,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            sampler=sampler, num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
        )
        return self.accelerator.prepare(loader)

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)
        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        dataset_len = len(self.train_dataset)
        micro_steps_per_epoch = max(ceil(dataset_len / global_batch_size), 1)
        opt_steps_per_epoch = max(ceil(micro_steps_per_epoch / self.gradient_accumulation_steps), 1)
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    # ══════════════════════════════════════════════════════════════════════
    # Scheduler
    # ══════════════════════════════════════════════════════════════════════

    def _build_scheduler(self, scheduler_type: str, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        remaining = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main = CosineAnnealingLR(self.optimizer, T_max=remaining, eta_min=self.learning_rate * 0.01)
        elif scheduler_type == "constant":
            main = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        if warmup_steps <= 0:
            return main
        warmup = LinearLR(self.optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps)
        return SequentialLR(self.optimizer, schedulers=[warmup, main], milestones=[warmup_steps])

    # ══════════════════════════════════════════════════════════════════════
    # Model mode
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    def _set_dit_only_train_mode(self):
        self._apply_dit_only_train_mode(self.accelerator.unwrap_model(self.model))

    # ══════════════════════════════════════════════════════════════════════
    # wandb + 本地日志
    # ══════════════════════════════════════════════════════════════════════

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        import wandb
        wcfg = self.cfg.get("wandb", {})
        self.wandb_run = wandb.init(
            entity=wcfg.get("workspace"), project=wcfg.get("project", "fast-wam-latent"),
            name=wcfg.get("name"), group=wcfg.get("group"),
            mode=wcfg.get("mode", "online"), dir=self.output_dir,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=self.global_step)

    def _local_log(self, kind: str, payload: dict):
        if self._local_metrics_fp is None:
            return
        record = {"kind": kind, "step": int(self.global_step), "wallclock": time.time()}
        for k, v in payload.items():
            if v is not None:
                record[k] = _to_jsonable(v)
        try:
            self._local_metrics_fp.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:
            pass

    def _close_local_metrics(self):
        if self._local_metrics_fp is not None:
            try:
                self._local_metrics_fp.flush()
                self._local_metrics_fp.close()
            except Exception:
                pass
            self._local_metrics_fp = None

    def _finish_wandb(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    # ══════════════════════════════════════════════════════════════════════
    # Checkpoint
    # ══════════════════════════════════════════════════════════════════════

    def _resume_or_load_checkpoint(self):
        if not self.resume:
            return
        resume_path = Path(str(self.resume))
        if resume_path.is_dir():
            self.accelerator.load_state(input_dir=str(resume_path))
            state_file = resume_path / "trainer_state.json"
            if state_file.exists():
                state = json.loads(state_file.read_text())
                self.global_step = int(state["global_step"])
                self.epoch = int(state.get("epoch", 0))
                self.batch_in_epoch = int(state.get("batch_in_epoch", 0))
            logger.info("Resumed from %s at step=%d", resume_path, self.global_step)
        elif resume_path.exists():
            self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
            logger.info("Loaded weights from %s", resume_path)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        self.accelerator.wait_for_everyone()

        ckpt_path = None
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
            model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)

        self.accelerator.wait_for_everyone()
        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            with open(os.path.join(state_path, "trainer_state.json"), "w") as f:
                json.dump({"global_step": self.global_step, "epoch": self.epoch,
                           "batch_in_epoch": self.batch_in_epoch}, f)
        self.accelerator.wait_for_everyone()
        return {"weights_path": ckpt_path, "state_path": state_path}

    # ══════════════════════════════════════════════════════════════════════
    # Eval
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return None
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.dit.training
        model.eval()

        # ── 阶段 1：多 batch 平均 val_loss ──
        total_loss = 0.0
        num_batches = 0
        max_eval_batches = max(1, self.eval_every // 10)

        for sample in self.val_loader:
            with self.accelerator.autocast():
                val_loss, _ = model.training_loss(sample)
                total_loss += val_loss.float().item()
                num_batches += 1
            if num_batches >= max_eval_batches:
                break

        avg_loss = total_loss / max(num_batches, 1)

        # ── 阶段 2：单样本推理（latent_mse + 解码视频）──
        eval_sample = next(iter(self.val_loader))
        single = {}
        for k, v in eval_sample.items():
            if isinstance(v, torch.Tensor):
                single[k] = v[:1]
            elif isinstance(v, list):
                single[k] = v[:1]
            else:
                single[k] = v

        action_horizon = int(single["action"].shape[1]) if single.get("action") is not None else 32

        infer_out = model.infer_from_latents(
            history_long=single["history_long"],
            history_mid=single["history_mid"],
            history_short=single["history_short"],
            action_horizon=action_horizon,
            prompt=None,
            context=single["context"],
            context_mask=single["context_mask"],
            proprio=single["proprio"][:, 0, :] if single.get("proprio") is not None else None,
            action=single["action"] if single.get("action") is not None else None,
            num_inference_steps=self.eval_num_inference_steps,
            seed=42,
            tiled=False,
            decode_video=True,
        )
        pred_latents = infer_out["pred_latents"]
        pred_video_pil = infer_out["video"]

        gt_latents = single["input_latents_precomputed"][:1].to(
            device=model.device, dtype=model.torch_dtype,
        )
        latent_mse = F.mse_loss(pred_latents.float(), gt_latents.float()).item()

        history_short_for_gt = single["history_short"][:1].to(
            device=model.device, dtype=model.torch_dtype,
        )
        gt_latents_for_decode = torch.cat([history_short_for_gt, gt_latents], dim=2)
        gt_video_pil = model._decode_latents(gt_latents_for_decode, tiled=False)[1:]
        pred_video_tensor = pil_frames_to_video_tensor(pred_video_pil)
        gt_video_tensor = pil_frames_to_video_tensor(gt_video_pil)

        min_t = min(pred_video_tensor.shape[1], gt_video_tensor.shape[1])
        pred_video_tensor = pred_video_tensor[:, :min_t]
        gt_video_tensor = gt_video_tensor[:, :min_t]

        psnr = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        stitched = torch.cat([pred_video_tensor, gt_video_tensor], dim=2).contiguous()
        frames = []
        for t in range(stitched.shape[1]):
            frame = (stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            frames.append(Image.fromarray(frame))
        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(frames, video_path, fps=8)

        # 跨 rank gather
        local_metrics = torch.tensor(
            [avg_loss, latent_mse, psnr, ssim],
            device=self.accelerator.device, dtype=torch.float32,
        ).unsqueeze(0)
        gathered = self.accelerator.gather_for_metrics(local_metrics)
        mean = gathered.mean(dim=0)

        if was_training:
            self._set_dit_only_train_mode()

        return {
            "val_loss": float(mean[0].item()),
            "latent_mse": float(mean[1].item()),
            "psnr_pred_decoded": float(mean[2].item()),
            "ssim_pred_decoded": float(mean[3].item()),
            "video_path": video_path,
        }

    # ══════════════════════════════════════════════════════════════════════
    # 训练循环
    # ══════════════════════════════════════════════════════════════════════

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done = max(self.global_step - self.run_start_step, 1)
        speed = done / elapsed
        remaining = max(self.max_steps - self.global_step, 0)
        eta_s = int(remaining / max(speed, 1e-9))
        h, rem = divmod(eta_s, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}", speed

    def train(self):
        self._set_dit_only_train_mode()
        logger.info("Starting training: max_steps=%d", self.max_steps)

        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        cuda_available = torch.cuda.is_available()

        def _cuda_sync():
            if cuda_available:
                torch.cuda.synchronize()

        t_fwd_accum = 0.0
        t_bwd_accum = 0.0

        while self.global_step < self.max_steps:
            _cuda_sync()
            t0 = time.perf_counter()

            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                if hasattr(self.train_sampler, "set_epoch"):
                    self.train_sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            _cuda_sync()
            t1 = time.perf_counter()

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                _cuda_sync()
                t2 = time.perf_counter()

                self.accelerator.backward(loss)
                _cuda_sync()
                t3 = time.perf_counter()

                t_fwd_accum += (t2 - t1)
                t_bwd_accum += (t3 - t2)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    _cuda_sync()
                    t4 = time.perf_counter()
                    self.global_step += 1

                    t_data = t1 - t0
                    t_fwd = t_fwd_accum
                    t_bwd = t_bwd_accum
                    t_step = t4 - t3
                    t_total = t_data + t_fwd + t_bwd + t_step
                    t_fwd_accum = 0.0
                    t_bwd_accum = 0.0

                    # gather loss + loss_dict + grad_norm
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        mt = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(self.accelerator.gather(mt).mean().item())
                    grad_norm_tensor = torch.tensor(float(grad_norm), device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        samples_per_sec = steps_per_sec * self.batch_size * self.accelerator.num_processes
                        total_safe = max(t_total, 1e-9)

                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch, self.global_step, self.max_steps, global_loss,
                        )
                        if global_loss_metrics:
                            detail = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr, steps_per_sec, samples_per_sec, eta_str,
                        )
                        logger.info(description)
                        logger.info(
                            "[timing] step=%d data=%.3fs(%.1f%%) fwd=%.3fs(%.1f%%) bwd=%.3fs(%.1f%%) step=%.3fs(%.1f%%) total=%.3fs",
                            self.global_step,
                            t_data, 100.0 * t_data / total_safe,
                            t_fwd, 100.0 * t_fwd / total_safe,
                            t_bwd, 100.0 * t_bwd / total_safe,
                            t_step, 100.0 * t_step / total_safe,
                            t_total,
                        )

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                            "timing/t_data": t_data,
                            "timing/t_fwd": t_fwd,
                            "timing/t_bwd": t_bwd,
                            "timing/t_step": t_step,
                            "timing/t_total": t_total,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)
                        self._local_log("train", {"epoch": self.epoch, **wandb_payload})

                    if self.eval_every > 0 and self.val_loader is not None and self.global_step % self.eval_every == 0:
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics and self.accelerator.is_main_process:
                            logger.info(
                                "[eval] step=%d val_loss=%.4f latent_mse=%.6f psnr=%.2f ssim=%.4f",
                                self.global_step, metrics["val_loss"],
                                metrics.get("latent_mse", 0), metrics.get("psnr_pred_decoded", 0),
                                metrics.get("ssim_pred_decoded", 0),
                            )
                            eval_payload = {f"eval/{k}": v for k, v in metrics.items() if k != "video_path"}
                            self._wandb_log(eval_payload)
                            self._local_log("eval", eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info("[ckpt] step=%d weights=%s state=%s",
                                        self.global_step, ckpt["weights_path"], ckpt["state_path"])
                            self._local_log("ckpt", {
                                "weights_path": ckpt["weights_path"],
                                "state_path": ckpt["state_path"],
                            })

                    if self.global_step >= self.max_steps:
                        ckpt = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info("[done] max_steps reached step=%d weights=%s state=%s",
                                        self.global_step, ckpt["weights_path"], ckpt["state_path"])
                            self._local_log("ckpt", {
                                "weights_path": ckpt["weights_path"],
                                "state_path": ckpt["state_path"],
                                "reason": "max_steps",
                            })
                        self._close_local_metrics()
                        self._finish_wandb()
                        return

        ckpt = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info("[done] training finished step=%d weights=%s state=%s",
                        self.global_step, ckpt["weights_path"], ckpt["state_path"])
            self._local_log("ckpt", {
                "weights_path": ckpt["weights_path"],
                "state_path": ckpt["state_path"],
                "reason": "epochs_complete",
            })
        self._close_local_metrics()
        self._finish_wandb()
