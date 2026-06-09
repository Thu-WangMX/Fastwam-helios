"""单子数据集 latent 加载器。

一个 LatentSingleDataset 对应一个 dataset_dir，负责：
- latent .pt 加载 + MTM 切片（history_long/mid/short + target）
- action/state 从 parquet 懒加载 + 归一化
- text embedding 从 dataset_dir/text_embeds/ 读取
- train/val 按 episode list 过滤

多数据集混合由上层 LatentMixtureDataset 负责。
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from einops import rearrange

from fastwam.datasets.lerobot.registry.data_config import BaseDataConfig
from fastwam.datasets.lerobot.registry.prompt_utils import build_prompt_from_config, get_text_embeds_dir
from fastwam.datasets.lerobot.registry.stats_utils import STATS_FILENAME, load_stats_cache
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

EPSILON = 5e-4


class LatentSingleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_dir: str,
        data_config: BaseDataConfig,
        # 训练参数
        num_frames: int = 33,
        stride: int = 1,
        action_video_freq_ratio: int = 4,
        context_len: int = 128,
        is_training: bool = True,
        # MTM 参数
        mtm_history_sizes: tuple[int, ...] = (16, 2, 1),
        mtm_pred_size: int = 2,
        zero_pad_len: int = 18,
        # 增强
        padded_memory_ratio: float = 0.0,
        memory_truncate_ratio: float = 0.0,
        # split
        train_episodes: list[int] | None = None,
        val_episodes: list[int] | None = None,
        # 其他
        latent_subdir: str = "latents",
        max_retry: int = 50,
    ):
        self.dataset_dir = str(dataset_dir)
        self.data_config = data_config
        self.num_frames = int(num_frames)
        self.stride = int(stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.context_len = int(context_len)
        self.is_training = is_training
        self.mtm_history_sizes = tuple(int(s) for s in mtm_history_sizes)
        self.mtm_pred_size = int(mtm_pred_size)
        self.zero_pad_len = int(zero_pad_len)
        self.padded_memory_ratio = float(padded_memory_ratio) if is_training else 0.0
        self.memory_truncate_ratio = float(memory_truncate_ratio) if is_training else 0.0
        self.latent_subdir = latent_subdir
        self.max_retry = int(max_retry)

        self._dataset_path = Path(dataset_dir)
        self._dataset_name = self._dataset_path.name

        # ── 预声明：一次性读取所有元数据 ──
        self._info = self._load_info()
        self._modality_meta = self._load_modality_meta()
        self._tasks = self._load_tasks()
        self._data_path_pattern = self._info["data_path"]
        self._chunks_size = int(self._info.get("chunks_size", 1000))
        self._total_episodes = int(self._info.get("total_episodes", 0))

        # 预计算 state/action 的列名和切片映射
        self._state_col_slices = self._build_col_slices("state", data_config.state_keys)
        self._action_col_slices = self._build_col_slices("action", data_config.action_keys)
        self._parquet_columns = self._collect_parquet_columns()

        # 归一化统计
        self._stats = self._load_stats()
        self._norm_params = self._build_norm_params()

        # Sample index
        allowed_episodes = set(train_episodes) if is_training and train_episodes else None
        if not is_training and val_episodes:
            allowed_episodes = set(val_episodes)
        self.samples = self._build_samples(allowed_episodes)

        # MTM
        t_lat = self.mtm_pred_size + 1
        self._padded_chunk_limit = (self.zero_pad_len + t_lat - 1) // t_lat
        memory_len = sum(self.mtm_history_sizes[:-1])
        self._memory_truncate_levels = list(range(0, memory_len, t_lat))

        # text embed 编码器 ID（从 model_id 推导，用于文件名）
        self._enc_id = "wan22ti2v5b"

        # parquet 缓存（当前 episode）
        self._cached_ep_idx: int | None = None
        self._cached_parquet: dict[str, np.ndarray] | None = None

        logger.info(
            "LatentSingleDataset[%s]: %d samples, %d episodes, is_training=%s",
            self._dataset_name, len(self.samples),
            len(allowed_episodes) if allowed_episodes else self._total_episodes,
            is_training,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 预声明：元数据加载
    # ══════════════════════════════════════════════════════════════════════

    def _load_info(self) -> dict:
        p = self._dataset_path / "meta" / "info.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def _load_modality_meta(self) -> dict:
        p = self._dataset_path / "meta" / "modality.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _load_tasks(self) -> dict[int, str]:
        p = self._dataset_path / "meta" / "tasks.jsonl"
        tasks: dict[int, str] = {}
        if p.exists():
            for line in p.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                rec = json.loads(line)
                tasks[int(rec["task_index"])] = str(rec["task"])
        return tasks

    def _load_stats(self) -> dict | None:
        stats_path = self._dataset_path / STATS_FILENAME
        cache_config = {"mode": self.data_config.action_mode}
        return load_stats_cache(stats_path, cache_config)

    # ══════════════════════════════════════════════════════════════════════
    # 预声明：列切片映射
    # ══════════════════════════════════════════════════════════════════════

    def _build_col_slices(
        self, modality: str, keys: list[str],
    ) -> list[tuple[str, str, int, int]]:
        """从 modality.json 构建 (full_key, original_col, start, end) 列表。"""
        meta = self._modality_meta.get(modality, {})
        slices = []
        for key in keys:
            subkey = key.removeprefix(f"{modality}.")
            if subkey in meta:
                cfg = meta[subkey]
                original_col = cfg.get("original_key", f"{modality}.{subkey}")
                slices.append((key, original_col, int(cfg["start"]), int(cfg["end"])))
        return slices

    def _collect_parquet_columns(self) -> list[str]:
        """需要从 parquet 读取的列名集合。"""
        cols = {"task_index", "frame_index"}
        for _, original_col, _, _ in self._state_col_slices:
            cols.add(original_col)
        for _, original_col, _, _ in self._action_col_slices:
            cols.add(original_col)
        return sorted(cols)

    # ══════════════════════════════════════════════════════════════════════
    # 预声明：归一化参数
    # ══════════════════════════════════════════════════════════════════════

    def _build_norm_params(self) -> dict[str, dict]:
        """从 stats + DataConfig.transform() 的 normalization_modes 构建每个子 key 的归一化参数。"""
        if self._stats is None:
            return {}

        transform = self.data_config.transform()
        norm_modes: dict[str, str] = {}
        if hasattr(transform, 'transforms'):
            for t in transform.transforms:
                if hasattr(t, 'normalization_modes'):
                    norm_modes.update(t.normalization_modes)

        params: dict[str, dict] = {}
        for slices_list in [self._action_col_slices, self._state_col_slices]:
            for full_key, original_col, start, end in slices_list:
                if original_col not in self._stats:
                    continue
                col_stats = self._stats[original_col]
                mode = norm_modes.get(full_key)
                if mode is None or mode == "binary":
                    continue
                params[full_key] = {
                    "mode": mode,
                    "min": np.array(col_stats["min"][start:end], dtype=np.float32),
                    "max": np.array(col_stats["max"][start:end], dtype=np.float32),
                    "mean": np.array(col_stats["mean"][start:end], dtype=np.float32),
                    "std": np.array(col_stats["std"][start:end], dtype=np.float32),
                    "q01": np.array(col_stats["q01"][start:end], dtype=np.float32),
                    "q99": np.array(col_stats["q99"][start:end], dtype=np.float32),
                }
        return params

    def _normalize(self, key: str, value: np.ndarray) -> np.ndarray:
        if key not in self._norm_params:
            return value
        p = self._norm_params[key]
        mode = p["mode"]
        if mode == "min_max":
            low, high = p["min"], p["max"]
            return (value - low) / (high - low + EPSILON) * 2.0 - 1.0
        if mode == "q99":
            low, high = p["q01"], p["q99"]
            return (value - low) / (high - low + EPSILON) * 2.0 - 1.0
        return value

    # ══════════════════════════════════════════════════════════════════════
    # Sample index 构建
    # ══════════════════════════════════════════════════════════════════════

    def _build_samples(self, allowed_episodes: set[int] | None) -> list[dict[str, Any]]:
        index_name = f"latent_sample_index_stride{self.stride}_nf{self.num_frames}.json"
        index_path = self._dataset_path / "meta" / index_name
        if not index_path.exists():
            logger.warning("Sample index 不存在: %s", index_path)
            return []

        payload = json.loads(index_path.read_text(encoding="utf-8"))
        episodes_dict = payload["episodes"]

        samples: list[dict[str, Any]] = []
        for ep_key, triples in episodes_dict.items():
            ep_idx = int(ep_key)
            if allowed_episodes is not None and ep_idx not in allowed_episodes:
                continue
            for tr in triples:
                samples.append({
                    "episode_idx": ep_idx,
                    "offset": int(tr["offset"]),
                    "chunk_idx": int(tr["chunk_idx"]),
                    "latent_path": str(tr["latent_path"]),
                    "action_start_frame": int(tr["action_start_frame"]),
                })
        return samples

    # ══════════════════════════════════════════════════════════════════════
    # Parquet 懒加载
    # ══════════════════════════════════════════════════════════════════════

    def _get_parquet_path(self, episode_idx: int) -> str:
        chunk_idx = episode_idx // self._chunks_size
        return str(self._dataset_path / self._data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_idx,
        ))

    def _load_parquet_episode(self, episode_idx: int) -> dict[str, np.ndarray]:
        """读取整个 episode 的 parquet，缓存当前 episode。"""
        if self._cached_ep_idx == episode_idx and self._cached_parquet is not None:
            return self._cached_parquet

        parquet_path = self._get_parquet_path(episode_idx)
        table = pq.read_table(parquet_path, columns=self._parquet_columns)
        result: dict[str, np.ndarray] = {}
        for col in self._parquet_columns:
            if col in ("task_index", "frame_index"):
                result[col] = np.array(table.column(col).to_pylist(), dtype=np.int64)
            else:
                result[col] = np.array(table.column(col).to_pylist(), dtype=np.float32)

        self._cached_ep_idx = episode_idx
        self._cached_parquet = result
        return result

    def _get_action_state(
        self, episode_idx: int, start_frame: int,
    ) -> dict[str, torch.Tensor]:
        """从 parquet 读取 action/state 并按子 key 拆分 + 归一化。"""
        parquet = self._load_parquet_episode(episode_idx)
        total_frames = len(parquet["frame_index"])
        end_frame = min(start_frame + self.num_frames, total_frames)

        result: dict[str, torch.Tensor] = {}

        for full_key, original_col, start, end in self._action_col_slices:
            raw = parquet[original_col][start_frame:end_frame, start:end]
            normed = self._normalize(full_key, raw)
            result[full_key] = torch.from_numpy(normed).float()

        for full_key, original_col, start, end in self._state_col_slices:
            raw = parquet[original_col][start_frame:end_frame, start:end]
            normed = self._normalize(full_key, raw)
            result[full_key] = torch.from_numpy(normed).float()

        # 拼接为统一的 action / proprio tensor
        action_parts = [result[k] for k, _, _, _ in self._action_col_slices if k in result]
        state_parts = [result[k] for k, _, _, _ in self._state_col_slices if k in result]
        action = torch.cat(action_parts, dim=-1) if action_parts else torch.zeros(end_frame - start_frame, 0)
        proprio = torch.cat(state_parts, dim=-1) if state_parts else torch.zeros(end_frame - start_frame, 0)

        actual_len = end_frame - start_frame
        if actual_len < self.num_frames:
            pad_len = self.num_frames - actual_len
            logger.warning(
                "Padding detected: dataset=%s ep=%d start_frame=%d, "
                "actual_len=%d < num_frames=%d, padding %d frames with last-frame repeat",
                self._dataset_name, episode_idx, start_frame, actual_len, self.num_frames, pad_len,
            )
            action = torch.cat([action, action[-1:].expand(pad_len, -1)], dim=0)
            proprio = torch.cat([proprio, proprio[-1:].expand(pad_len, -1)], dim=0)

        task_index = int(parquet["task_index"][start_frame])

        return {
            "action": action,
            "proprio": proprio,
            "task_index": task_index,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Text embedding
    # ══════════════════════════════════════════════════════════════════════

    def _get_text_context(self, task_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        task = self._tasks.get(task_index, "")
        prompt = build_prompt_from_config(task, self.data_config)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_dir = get_text_embeds_dir(self.dataset_dir)
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.{self._enc_id}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding: {cache_path}. Run precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        context = payload["context"]
        context_mask = payload["mask"].bool()
        context = context.clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return context, context_mask

    # ══════════════════════════════════════════════════════════════════════
    # MTM latent 切片
    # ══════════════════════════════════════════════════════════════════════

    def _slice_latent(
        self, latent_path: str, chunk_idx: int, force_padded: bool, truncate_memory: bool,
    ) -> dict[str, torch.Tensor]:
        data = torch.load(latent_path, map_location="cpu", weights_only=False)
        vae_latent = data["vae_latent"]  # (num_chunks, C, T_lat, H, W)
        total_sections = int(vae_latent.shape[0])

        if force_padded and self._padded_chunk_limit > 0:
            upper = min(total_sections, self._padded_chunk_limit)
            chunk_idx = int(np.random.randint(upper))

        C, T_lat, H_lat, W_lat = (
            int(vae_latent.shape[1]), int(vae_latent.shape[2]),
            int(vae_latent.shape[3]), int(vae_latent.shape[4]),
        )
        history_window = sum(self.mtm_history_sizes)

        latent_f = vae_latent.float()
        temp = rearrange(latent_f, "b c t h w -> c (b t) h w")
        zero_pad = torch.zeros(C, self.zero_pad_len, H_lat, W_lat, dtype=temp.dtype)
        cont = torch.cat([zero_pad, temp], dim=1)

        start = chunk_idx * T_lat
        history_latent = cont[:, start:start + history_window]
        target_latent = cont[:, start + history_window:start + history_window + self.mtm_pred_size]

        if truncate_memory and self._memory_truncate_levels:
            k = self._memory_truncate_levels[int(np.random.randint(len(self._memory_truncate_levels)))]
            memory_len = sum(self.mtm_history_sizes[:-1])
            zero_count = memory_len - k
            if zero_count > 0:
                history_latent = history_latent.clone()
                history_latent[:, :zero_count] = 0.0

        L, M, S = self.mtm_history_sizes
        return {
            "history_long": history_latent[:, :L].contiguous(),
            "history_mid": history_latent[:, L:L + M].contiguous(),
            "history_short": history_latent[:, L + M:L + M + S].contiguous(),
            "input_latents_precomputed": target_latent.contiguous(),
        }

    # ══════════════════════════════════════════════════════════════════════
    # 主接口
    # ══════════════════════════════════════════════════════════════════════

    def __len__(self) -> int:
        return len(self.samples)

    def _get(self, idx: int, force_padded: bool = False, truncate_memory: bool = False) -> dict[str, Any]:
        sample = self.samples[idx]
        ep_idx = sample["episode_idx"]
        chunk_idx = sample["chunk_idx"]
        action_start_frame = sample["action_start_frame"]
        latent_path = sample["latent_path"]

        if force_padded:
            action_start_frame = sample["offset"] + chunk_idx * self.num_frames

        # latent 切片
        latents = self._slice_latent(latent_path, chunk_idx, force_padded, truncate_memory)

        # action / state
        action_state = self._get_action_state(ep_idx, action_start_frame)

        # text embedding
        context, context_mask = self._get_text_context(action_state["task_index"])

        return {
            **latents,
            "action": action_state["action"],
            "proprio": action_state["proprio"],
            "context": context,
            "context_mask": context_mask,
            "episode_index": ep_idx,
            "chunk_idx": chunk_idx,
            "offset": sample["offset"],
            "action_start_frame": action_start_frame,
            "dataset_dir": self.dataset_dir,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retry + 1):
            try:
                force_padded = (
                    self.padded_memory_ratio > 0.0
                    and float(np.random.random()) < self.padded_memory_ratio
                )
                truncate_memory = (
                    self.memory_truncate_ratio > 0.0
                    and float(np.random.random()) < self.memory_truncate_ratio
                )
                return self._get(idx, force_padded=force_padded, truncate_memory=truncate_memory)
            except Exception as err:
                last_err = err
                sample = self.samples[idx]
                logger.warning(
                    "LatentSingleDataset[%s] __getitem__(idx=%d) attempt %d failed: ep=%d path=%s\n%s",
                    self._dataset_name, idx, attempt, sample["episode_idx"],
                    sample["latent_path"], traceback.format_exc(),
                )
                idx = int(np.random.randint(len(self)))
        raise RuntimeError(
            f"LatentSingleDataset[{self._dataset_name}] failed after {self.max_retry + 1} attempts"
        ) from last_err
