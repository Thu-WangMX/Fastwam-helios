"""数据集统计计算 + 缓存管理。

移植自 StarVLA datasets.py 的 stats 机制：
- calculate_dataset_statistics: 基础统计（mean/std/min/max/q01/q99）
- calculate_delta_action_statistics: delta 模式（a[t] - a[t-1]）
- calculate_rel_action_statistics: rel 模式（a[t] - s[0]）
- 缓存：meta/stats_gr00t.json，含 __format_version + __cache_config
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

STATS_FILENAME = "meta/stats_gr00t.json"
STATS_FORMAT_VERSION = 2


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """读全部 parquet，按列计算 mean/std/min/max/q01/q99。"""
    all_data_list = []
    for parquet_path in sorted(parquet_paths):
        all_data_list.append(pd.read_parquet(parquet_path))

    if not all_data_list:
        raise FileNotFoundError(f"未找到 parquet 文件: {[str(p) for p in parquet_paths[:3]]}")

    all_data = pd.concat(all_data_list, axis=0)
    statistics = {}
    for col in all_data.columns:
        if "task_info" in col:
            continue
        try:
            np_data = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in all_data[col]]
            )
        except Exception:
            continue
        statistics[col] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return statistics


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    modality_meta: dict,
    action_keys: list[str],
    state_keys: list[str],
    action_indices: list[int],
    state_indices: list[int],
    base_stats: dict | None = None,
) -> dict:
    """Delta 模式：a[t] - a[t-1]，t=0 时为 a[0] - s[0]。"""
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    col_slices = _get_action_col_slices(modality_meta, action_keys, state_keys)
    if not col_slices:
        return base_stats

    action_indices_arr = np.array(action_indices)
    state_indices_arr = np.array(state_indices)

    accum: dict[str, list[np.ndarray]] = {col: [] for col in col_slices}
    for parquet_path in sorted(parquet_paths):
        data = pd.read_parquet(parquet_path)
        traj_len = len(data)
        for action_col, slice_list in col_slices.items():
            if action_col not in data.columns:
                continue
            action_matrix = np.stack(data[action_col])
            n_dims = action_matrix.shape[1]
            n_steps = len(action_indices_arr)
            all_chunks = np.empty((traj_len, n_steps, n_dims), dtype=action_matrix.dtype)
            for base_idx in range(traj_len):
                chunk = _get_chunk(action_matrix, action_indices_arr + base_idx)
                for a_slice, state_col, s_slice in slice_list:
                    if state_col not in data.columns:
                        continue
                    state_matrix = np.stack(data[state_col])
                    state_part = state_matrix[:, s_slice[0]:s_slice[1]]
                    s_chunk = _get_chunk(state_part, state_indices_arr + base_idx)
                    a_part = chunk[:, a_slice[0]:a_slice[1]].copy()
                    if len(a_part) > 1:
                        a_part[1:] = a_part[1:] - a_part[:-1]
                    a_part[0] = a_part[0] - s_chunk[0]
                    chunk[:, a_slice[0]:a_slice[1]] = a_part
                all_chunks[base_idx] = chunk
            accum[action_col].append(all_chunks.reshape(-1, n_dims))

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series in accum.items():
        if not series:
            continue
        vals = np.concatenate(series, axis=0).astype(np.float32)
        delta_stats[action_col] = _compute_col_stats(vals)
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    modality_meta: dict,
    action_keys: list[str],
    state_keys: list[str],
    action_indices: list[int],
    state_indices: list[int],
    base_stats: dict | None = None,
) -> dict:
    """Rel 模式：a[t] - s[0]（所有 t 减去当前起始帧 state）。"""
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    col_slices = _get_action_col_slices(modality_meta, action_keys, state_keys)
    if not col_slices:
        return base_stats

    action_indices_arr = np.array(action_indices)
    state_indices_arr = np.array(state_indices)

    accum: dict[str, list[np.ndarray]] = {col: [] for col in col_slices}
    for parquet_path in sorted(parquet_paths):
        data = pd.read_parquet(parquet_path)
        traj_len = len(data)
        for action_col, slice_list in col_slices.items():
            if action_col not in data.columns:
                continue
            action_matrix = np.stack(data[action_col])
            n_dims = action_matrix.shape[1]
            n_steps = len(action_indices_arr)
            all_chunks = np.empty((traj_len, n_steps, n_dims), dtype=action_matrix.dtype)
            for base_idx in range(traj_len):
                chunk = _get_chunk(action_matrix, action_indices_arr + base_idx)
                for a_slice, state_col, s_slice in slice_list:
                    if state_col not in data.columns:
                        continue
                    state_matrix = np.stack(data[state_col])
                    state_part = state_matrix[:, s_slice[0]:s_slice[1]]
                    s_chunk = _get_chunk(state_part, state_indices_arr + base_idx)
                    chunk[:, a_slice[0]:a_slice[1]] = chunk[:, a_slice[0]:a_slice[1]] - s_chunk[0]
                all_chunks[base_idx] = chunk
            accum[action_col].append(all_chunks.reshape(-1, n_dims))

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series in accum.items():
        if not series:
            continue
        vals = np.concatenate(series, axis=0).astype(np.float32)
        rel_stats[action_col] = _compute_col_stats(vals)
    return rel_stats


def _compute_col_stats(vals: np.ndarray) -> dict:
    return {
        "mean": np.mean(vals, axis=0).tolist(),
        "std": np.std(vals, axis=0).tolist(),
        "min": np.min(vals, axis=0).tolist(),
        "max": np.max(vals, axis=0).tolist(),
        "q01": np.quantile(vals, 0.01, axis=0).tolist(),
        "q99": np.quantile(vals, 0.99, axis=0).tolist(),
    }


def _get_chunk(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    max_len = array.shape[0]
    clamped = np.clip(indices, 0, max_len - 1)
    return array[clamped]


def _get_action_col_slices(
    modality_meta: dict,
    action_keys: list[str],
    state_keys: list[str],
) -> dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]]:
    """从 modality.json 解析 action-state 切片映射。"""
    action_meta = modality_meta.get("action", {})
    state_meta = modality_meta.get("state", {})
    result: dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]] = {}

    for action_key in action_keys:
        subkey = action_key.removeprefix("action.")
        state_key = action_key.replace("action.", "state.", 1)
        if state_key not in state_keys:
            continue
        state_subkey = state_key.removeprefix("state.")
        if subkey not in action_meta or state_subkey not in state_meta:
            continue
        a_cfg = action_meta[subkey]
        s_cfg = state_meta[state_subkey]
        action_col = a_cfg.get("original_key", f"action.{subkey}")
        state_col = s_cfg.get("original_key", f"state.{state_subkey}")
        a_slice = (a_cfg["start"], a_cfg["end"])
        s_slice = (s_cfg["start"], s_cfg["end"])
        result.setdefault(action_col, []).append((a_slice, state_col, s_slice))

    return result


# ═══ 缓存管理 ═══

def load_stats_cache(stats_path: Path, expected_config: dict) -> dict | None:
    if not stats_path.exists():
        return None
    try:
        with open(stats_path, "r") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("__format_version") != STATS_FORMAT_VERSION:
        stats_path.unlink()
        return None
    if payload.get("__cache_config") != expected_config:
        stats_path.unlink()
        return None
    return payload.get("statistics")


def save_stats_cache(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    payload = {
        "__format_version": STATS_FORMAT_VERSION,
        "__cache_config": cache_config,
        "statistics": statistics,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stats_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp, stats_path)
