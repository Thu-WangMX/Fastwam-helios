"""多数据集混合的 latent 数据集。

从 data_mix 解析出所有子数据集，为每个创建 LatentSingleDataset，
通过加权采样提供统一的 __getitem__ 接口。

用法:
    mixture_ds = LatentMixtureDataset.from_config(
        config=yaml_config,
        is_training=True,
    )
    sample = mixture_ds[0]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from fastwam.datasets.lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from fastwam.datasets.lerobot.registry.data_config import BaseDataConfig
from fastwam.utils.logging_config import get_logger

from .latent_single_dataset import LatentSingleDataset

logger = get_logger(__name__)


class LatentMixtureDataset(Dataset):
    """多数据集混合。

    每个 __getitem__ 先按权重选数据集，再在该数据集内按 sample 索引取数据。
    """

    def __init__(
        self,
        datasets: list[LatentSingleDataset],
        weights: list[float],
        seed: int = 42,
    ):
        if len(datasets) == 0:
            raise ValueError("No valid datasets in mixture")
        if len(datasets) != len(weights):
            raise ValueError(f"datasets ({len(datasets)}) and weights ({len(weights)}) length mismatch")

        self.datasets = datasets
        self.seed = seed
        self._epoch = 0

        # 各子数据集的 sample 数
        self._dataset_lengths = np.array([len(ds) for ds in datasets])

        # 加权采样概率：weight_i * num_samples_i
        raw_weights = np.array(weights, dtype=np.float64) * self._dataset_lengths
        raw_weights = np.maximum(raw_weights, 1e-8)
        self._dataset_probs = raw_weights / raw_weights.sum()

        # 累积偏移（用于全局索引 → 子数据集索引映射）
        self._cumulative_sizes = np.cumsum(self._dataset_lengths)
        self._total_samples = int(self._cumulative_sizes[-1])

        # 每个子数据集的 video_size（用于 bucket 分组）
        self._dataset_video_sizes: list[tuple[int, int]] = []
        for ds in datasets:
            self._dataset_video_sizes.append(ds.data_config.video_size)

        logger.info(
            "LatentMixtureDataset: %d 个子数据集, total_samples=%d",
            len(datasets), self._total_samples,
        )
        for i, ds in enumerate(datasets):
            logger.info(
                "  [%d] %s: %d samples, weight=%.3f, prob=%.4f, video_size=%s",
                i, ds._dataset_name, len(ds), weights[i],
                self._dataset_probs[i], ds.data_config.video_size,
            )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        is_training: bool = True,
        seed: int = 42,
    ) -> "LatentMixtureDataset":
        """从 task yaml config 构建 MixtureDataset。"""
        data_cfg = config["data"]
        model_cfg = config.get("model", {})
        data_mix = data_cfg["data_mix"]

        num_frames = int(data_cfg.get("num_frames", 33))
        stride = int(data_cfg.get("stride", num_frames))
        action_video_freq_ratio = int(data_cfg.get("action_video_freq_ratio", 4))
        context_len = int(model_cfg.get("tokenizer_max_len", 128))
        zero_pad_len = int(data_cfg.get("zero_pad_len", 18))
        mtm_history_sizes = tuple(model_cfg.get("mtm_history_sizes", [16, 2, 1]))
        mtm_pred_size = int(model_cfg.get("mtm_pred_size", 2))
        padded_memory_ratio = float(data_cfg.get("padded_memory_ratio", 0.0))
        memory_truncate_ratio = float(data_cfg.get("memory_truncate_ratio", 0.0))
        latent_subdir = str(data_cfg.get("latent_subdir", "latents"))

        mixture = DATASET_NAMED_MIXTURES[data_mix]

        # 去重：同一个 dataset_dir 只创建一个 LatentSingleDataset
        seen: dict[str, int] = {}
        datasets: list[LatentSingleDataset] = []
        weights: list[float] = []

        for dataset_dir, weight, robot_type in mixture:
            if dataset_dir in seen:
                continue
            seen[dataset_dir] = len(datasets)

            data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]

            # 读 train/val split
            split_name = f"train_val_split_stride{stride}_nf{num_frames}.json"
            split_path = Path(dataset_dir) / "meta" / split_name
            train_episodes = None
            val_episodes = None
            if split_path.exists():
                split = json.loads(split_path.read_text(encoding="utf-8"))
                train_episodes = split.get("train_episodes")
                val_episodes = split.get("val_episodes")

            try:
                ds = LatentSingleDataset(
                    dataset_dir=dataset_dir,
                    data_config=data_config,
                    num_frames=num_frames,
                    stride=stride,
                    action_video_freq_ratio=action_video_freq_ratio,
                    context_len=context_len,
                    is_training=is_training,
                    mtm_history_sizes=mtm_history_sizes,
                    mtm_pred_size=mtm_pred_size,
                    zero_pad_len=zero_pad_len,
                    padded_memory_ratio=padded_memory_ratio,
                    memory_truncate_ratio=memory_truncate_ratio,
                    train_episodes=train_episodes,
                    val_episodes=val_episodes,
                    latent_subdir=latent_subdir,
                )
            except Exception as exc:
                logger.warning("跳过 %s: %s", dataset_dir, exc)
                continue

            if len(ds) == 0:
                logger.warning("跳过 %s: 0 samples", dataset_dir)
                continue

            datasets.append(ds)
            weights.append(weight)

        return cls(datasets=datasets, weights=weights, seed=seed)

    # ══════════════════════════════════════════════════════════════════════
    # 索引映射
    # ══════════════════════════════════════════════════════════════════════

    def _global_to_local(self, global_idx: int) -> tuple[int, int]:
        """全局 sample 索引 → (子数据集索引, 本地 sample 索引)。"""
        ds_idx = int(np.searchsorted(self._cumulative_sizes, global_idx, side="right"))
        local_idx = global_idx - (int(self._cumulative_sizes[ds_idx - 1]) if ds_idx > 0 else 0)
        return ds_idx, local_idx

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx, local_idx = self._global_to_local(idx)
        sample = self.datasets[ds_idx][local_idx]
        sample["_ds_idx"] = ds_idx
        return sample

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    # ══════════════════════════════════════════════════════════════════════
    # Bucket 信息（供 sampler 使用）
    # ══════════════════════════════════════════════════════════════════════

    def get_bucket_key(self, global_idx: int) -> tuple[int, int]:
        """返回该 sample 所属的 video_size bucket。"""
        ds_idx, _ = self._global_to_local(global_idx)
        return self._dataset_video_sizes[ds_idx]

    def get_all_bucket_groups(self) -> dict[tuple[int, int], list[int]]:
        """按 video_size 分组，返回 {(H, W): [global_indices]}。"""
        groups: dict[tuple[int, int], list[int]] = {}
        offset = 0
        for ds_idx, ds in enumerate(self.datasets):
            bucket_key = self._dataset_video_sizes[ds_idx]
            indices = list(range(offset, offset + len(ds)))
            groups.setdefault(bucket_key, []).extend(indices)
            offset += len(ds)
        return groups

    def get_dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_probs.copy()
