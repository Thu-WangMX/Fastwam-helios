"""按轨迹级别切分 train/val。

读取 scan_dataset_meta.py 生成的 latent_sample_index_*.json，
按轨迹集合切分 train/val，写入各子数据集的 meta/ 下。

用法:
    python scripts/split_train_val.py \
        --config config_tmp/robocasa365_pretrain.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from fastwam.datasets.lerobot.registry import DATASET_NAMED_MIXTURES
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _find_sample_index(ds_dir: str, num_frames: int, stride: int) -> Path | None:
    name = f"latent_sample_index_stride{stride}_nf{num_frames}.json"
    p = Path(ds_dir) / "meta" / name
    return p if p.exists() else None


def _split_episodes(
    episode_keys: list[str],
    val_proportion: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    total = len(episode_keys)
    n_val = round(total * val_proportion)
    if n_val < 1:
        n_val = 1
    if n_val >= total:
        n_val = max(1, total - 1)

    all_eps = sorted(int(k) for k in episode_keys)
    rng = random.Random(seed)
    rng.shuffle(all_eps)
    val_eps = sorted(all_eps[:n_val])
    train_eps = sorted(all_eps[n_val:])
    return train_eps, val_eps


def _write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main():
    setup_logging(log_level=logging.INFO)
    parser = argparse.ArgumentParser(description="按轨迹切分 train/val")
    parser.add_argument("--config", required=True, help="task yaml 路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有的 split 文件")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    data_mix = data_cfg["data_mix"]
    val_proportion = float(data_cfg.get("val_set_proportion", 0.01))
    seed = int(data_cfg.get("base_seed", 42))
    num_frames = int(data_cfg.get("num_frames", 33))
    stride = int(data_cfg.get("stride", num_frames))

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    logger.info("data_mix=%s, %d 个子数据集, val_proportion=%.3f, seed=%d",
                data_mix, len(mixture), val_proportion, seed)

    total_datasets = 0
    total_train = 0
    total_val = 0
    skipped = 0

    for dataset_dir, _weight, _robot_type in mixture:
        idx_path = _find_sample_index(dataset_dir, num_frames, stride)
        if idx_path is None:
            logger.warning("跳过 %s: 未找到 sample index (stride=%d, nf=%d)",
                           dataset_dir, stride, num_frames)
            skipped += 1
            continue

        out_path = idx_path.parent / f"train_val_split_stride{stride}_nf{num_frames}.json"
        if out_path.exists() and not args.overwrite:
            data = json.loads(out_path.read_text())
            total_train += len(data["train_episodes"])
            total_val += len(data["val_episodes"])
            total_datasets += 1
            continue

        sample_index = json.loads(idx_path.read_text())
        episode_keys = list(sample_index["episodes"].keys())

        if len(episode_keys) == 0:
            logger.warning("跳过 %s: 无 episode", dataset_dir)
            skipped += 1
            continue

        train_eps, val_eps = _split_episodes(episode_keys, val_proportion, seed)

        train_samples = sum(len(sample_index["episodes"][str(e)]) for e in train_eps)
        val_samples = sum(len(sample_index["episodes"][str(e)]) for e in val_eps)

        payload = {
            "val_proportion": val_proportion,
            "seed": seed,
            "total_episodes": len(episode_keys),
            "train_episodes": train_eps,
            "val_episodes": val_eps,
            "train_samples": train_samples,
            "val_samples": val_samples,
        }
        _write_json_atomic(payload, out_path)

        ds_name = Path(dataset_dir).name
        logger.info("  %s: %d train / %d val episodes (%d / %d samples)",
                     ds_name, len(train_eps), len(val_eps), train_samples, val_samples)

        total_train += len(train_eps)
        total_val += len(val_eps)
        total_datasets += 1

    logger.info("完成: %d 个数据集, train=%d / val=%d 轨迹, 跳过=%d",
                total_datasets, total_train, total_val, skipped)


if __name__ == "__main__":
    main()
