"""并行计算数据集归一化统计。

从 data_mix 解析所有子数据集，根据 DataConfig 的 action_mode 计算
abs / delta / rel 统计，存到各子数据集的 meta/stats_gr00t.json。

用法:
    python scripts/compute_dataset_stats.py \
        --config config_tmp/robocasa365_pretrain.yaml \
        --num_workers 32
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from fastwam.datasets.lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from fastwam.datasets.lerobot.registry.stats_utils import (
    STATS_FILENAME,
    calculate_dataset_statistics,
    calculate_delta_action_statistics,
    calculate_rel_action_statistics,
    load_stats_cache,
    save_stats_cache,
)
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _compute_one_dataset(args_tuple: tuple) -> dict[str, Any]:
    """Worker：计算一个子数据集的统计。"""
    dataset_dir, robot_type, overwrite = args_tuple
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    action_mode = data_config.action_mode
    stats_path = Path(dataset_dir) / STATS_FILENAME
    cache_config = {"mode": action_mode}
    ds_name = Path(dataset_dir).name

    if not overwrite:
        cached = load_stats_cache(stats_path, cache_config)
        if cached is not None:
            return {"status": "cached", "dataset_dir": dataset_dir, "name": ds_name}

    parquet_paths = sorted(Path(dataset_dir).glob("data/*/*.parquet"))
    if not parquet_paths:
        return {"status": "skip", "dataset_dir": dataset_dir, "name": ds_name,
                "reason": "no parquet files"}

    try:
        if action_mode == "abs":
            statistics = calculate_dataset_statistics(parquet_paths)
        elif action_mode == "delta":
            modality_path = Path(dataset_dir) / "meta" / "modality.json"
            if not modality_path.exists():
                return {"status": "skip", "dataset_dir": dataset_dir, "name": ds_name,
                        "reason": "no modality.json for delta mode"}
            modality_meta = json.loads(modality_path.read_text())
            statistics = calculate_delta_action_statistics(
                parquet_paths=parquet_paths,
                modality_meta=modality_meta,
                action_keys=data_config.action_keys,
                state_keys=data_config.state_keys,
                action_indices=data_config.action_indices,
                state_indices=data_config.observation_indices,
            )
        elif action_mode == "rel":
            modality_path = Path(dataset_dir) / "meta" / "modality.json"
            if not modality_path.exists():
                return {"status": "skip", "dataset_dir": dataset_dir, "name": ds_name,
                        "reason": "no modality.json for rel mode"}
            modality_meta = json.loads(modality_path.read_text())
            statistics = calculate_rel_action_statistics(
                parquet_paths=parquet_paths,
                modality_meta=modality_meta,
                action_keys=data_config.action_keys,
                state_keys=data_config.state_keys,
                action_indices=data_config.action_indices,
                state_indices=data_config.observation_indices,
            )
        else:
            return {"status": "skip", "dataset_dir": dataset_dir, "name": ds_name,
                    "reason": f"unknown action_mode: {action_mode}"}

        save_stats_cache(stats_path, cache_config, statistics)
        n_cols = len(statistics)
        return {"status": "ok", "dataset_dir": dataset_dir, "name": ds_name,
                "mode": action_mode, "n_cols": n_cols}
    except Exception as exc:
        return {"status": "error", "dataset_dir": dataset_dir, "name": ds_name,
                "reason": f"{type(exc).__name__}: {exc}"}


def main():
    setup_logging(log_level=logging.INFO)
    parser = argparse.ArgumentParser(description="并行计算数据集归一化统计")
    parser.add_argument("--config", required=True, help="task yaml 路径")
    parser.add_argument("--num_workers", type=int, default=None, help="并行 worker 数")
    parser.add_argument("--overwrite", action="store_true", help="强制重算")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_mix = cfg["data"]["data_mix"]
    num_workers = args.num_workers or max(1, (os.cpu_count() or 8) // 2)

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    seen = set()
    unique_datasets = []
    for dataset_dir, _weight, robot_type in mixture:
        if dataset_dir not in seen:
            seen.add(dataset_dir)
            unique_datasets.append((dataset_dir, robot_type, args.overwrite))

    logger.info("data_mix=%s, %d 个去重子数据集, num_workers=%d",
                data_mix, len(unique_datasets), num_workers)

    computed = 0
    cached = 0
    skipped = 0
    errors = 0

    if num_workers <= 1:
        results = [_compute_one_dataset(d) for d in tqdm(unique_datasets, desc="stats")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_compute_one_dataset, unique_datasets, chunksize=1),
                total=len(unique_datasets), desc="stats", dynamic_ncols=True,
            ))

    for res in results:
        if res["status"] == "ok":
            computed += 1
        elif res["status"] == "cached":
            cached += 1
        elif res["status"] == "skip":
            skipped += 1
            logger.warning("跳过 %s: %s", res["name"], res.get("reason"))
        elif res["status"] == "error":
            errors += 1
            logger.error("错误 %s: %s", res["name"], res.get("reason"))

    logger.info("完成: computed=%d, cached=%d, skipped=%d, errors=%d",
                computed, cached, skipped, errors)


if __name__ == "__main__":
    main()
