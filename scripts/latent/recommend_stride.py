"""推荐最优 stride：在训练预算内覆盖至少 target_epochs 轮的最小 stride。

原理：
  stride 越小 → 滑窗视角越多（数据多样性高），但 manifest 更大、一轮更久
  stride 越大 → chunk 数越少，训练更快，但多样性低

脚本不读 parquet/mp4，只从 episodes.jsonl 读 episode length，秒级完成。

用法:
    # NAS（从 yaml 读 data_mix）
    python scripts/latent/recommend_stride.py \
        --config configs/robocasa365_pretrain.yaml \
        --bs 64 --steps 60000

    # 直接指定 data_mix
    python scripts/latent/recommend_stride.py \
        --data_mix robocasa365_pretrain_all \
        --num_frames 33 --bs 64 --steps 60000

    # OSS（通过 ossutil 读 episodes.jsonl）
    python scripts/latent/recommend_stride.py \
        --config configs/robocasa365_pretrain.yaml \
        --bs 64 --steps 60000 \
        --backend oss --oss_mount /data/oss_bucket_0:oss://xlab-dev
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from fastwam.datasets.lerobot.registry import DATASET_NAMED_MIXTURES
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _divisors_ascending(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _read_episode_lengths_nas(ds_dir: str) -> list[int]:
    path = Path(ds_dir) / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"episodes.jsonl missing: {path}")
    lengths = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lengths.append(int(json.loads(line)["length"]))
    return lengths


def _read_episode_lengths_oss(ds_dir: str, translator) -> list[int]:
    oss_uri = translator(str(Path(ds_dir) / "meta" / "episodes.jsonl"))
    fd, tmp = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        subprocess.run(["ossutil", "cp", "-f", oss_uri, tmp],
                       capture_output=True, text=True, timeout=120, check=True)
        lengths = []
        with open(tmp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lengths.append(int(json.loads(line)["length"]))
        return lengths
    finally:
        os.unlink(tmp) if os.path.exists(tmp) else None


def _make_oss_translator(mount_spec: str):
    local_root, oss_root = mount_spec.split(":", 1)
    local_root = local_root.rstrip("/") + "/"
    oss_root = oss_root.rstrip("/") + "/"
    def _translate(path: str) -> str:
        if not path.startswith(local_root):
            raise ValueError(f"路径 {path} 不在挂载根 {local_root} 下")
        return oss_root + path[len(local_root):]
    return _translate


def _chunks_per_episode(T: int, F: int, s: int) -> tuple[int, int]:
    num_offsets = F // s
    records = chunks = 0
    for i in range(num_offsets):
        n = (T - i * s) // F
        if n > 0:
            records += 1
            chunks += n
    return records, chunks


def _aggregate(Ts: list[int], F: int, s: int) -> tuple[int, int]:
    tot_records = tot_chunks = 0
    for T in Ts:
        r, c = _chunks_per_episode(T, F, s)
        tot_records += r
        tot_chunks += c
    return tot_records, tot_chunks


def _fmt_int(n: float) -> str:
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def main():
    setup_logging(log_level=logging.INFO)
    parser = argparse.ArgumentParser(description="推荐最优 stride")
    parser.add_argument("--config", type=str, default=None, help="task yaml 路径")
    parser.add_argument("--data_mix", type=str, default=None, help="直接指定 data_mix")
    parser.add_argument("--num_frames", type=int, default=None, help="每 chunk 帧数（不指定则从 yaml 读）")
    parser.add_argument("--bs", type=int, required=True, help="global batch size")
    parser.add_argument("--steps", type=int, required=True, help="总训练步数")
    parser.add_argument("--target_epochs", type=float, default=1.0, help="目标覆盖轮数")
    parser.add_argument("--backend", choices=["nas", "oss"], default="nas")
    parser.add_argument("--oss_mount", default="/data/oss_bucket_0:oss://xlab-dev")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data_mix = cfg["data"]["data_mix"]
        F = args.num_frames or int(cfg["data"].get("num_frames", 33))
    elif args.data_mix:
        data_mix = args.data_mix
        F = args.num_frames or 33
    else:
        raise SystemExit("必须提供 --config 或 --data_mix")

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    translator = _make_oss_translator(args.oss_mount) if args.backend == "oss" else None

    Ts: list[int] = []
    for ds_dir, _weight, _robot_type in mixture:
        try:
            if translator:
                lens = _read_episode_lengths_oss(ds_dir, translator)
            else:
                lens = _read_episode_lengths_nas(ds_dir)
            Ts.extend(lens)
            logger.info("  %s: %d episodes (mean length %.1f)",
                        Path(ds_dir).name, len(lens), sum(lens) / max(len(lens), 1))
        except Exception as e:
            logger.warning("  跳过 %s: %s", Path(ds_dir).name, e)

    if not Ts:
        raise SystemExit("未找到任何 episode")

    budget = args.steps * args.bs
    target_budget = budget / args.target_epochs

    print()
    print("=== Stride 推荐 ===")
    print(f"  data_mix          = {data_mix}")
    print(f"  num_frames        = {F}")
    print(f"  episodes          = {len(Ts):,}  (total frames = {_fmt_int(sum(Ts))})")
    print(f"  batch_size        = {args.bs}")
    print(f"  steps             = {args.steps:,}")
    print(f"  budget (bs*steps) = {_fmt_int(budget)} samples")
    print(f"  target_epochs     = {args.target_epochs}")
    print()
    print(f"  {'stride':>6} {'#records':>11} {'#chunks':>11} {'budget%':>8} {'epochs':>8} {'fits':>5}")
    print(f"  {'-'*6:>6} {'-'*11:>11} {'-'*11:>11} {'-'*8:>8} {'-'*8:>8} {'-'*5:>5}")

    chosen = None
    rows = []
    for d in _divisors_ascending(F):
        records, chunks = _aggregate(Ts, F, d)
        used = chunks / target_budget if target_budget > 0 else float("inf")
        epochs = budget / chunks if chunks > 0 else float("inf")
        fits = chunks <= target_budget
        rows.append((d, records, chunks, used, epochs, fits))
        if fits and chosen is None:
            chosen = d

    for d, records, chunks, used, epochs, fits in rows:
        marker = "  <- recommended" if d == chosen else ""
        epoch_str = f"{epochs:>7.2f}x" if epochs != float("inf") else "    inf"
        print(f"  {d:>6} {_fmt_int(records):>11} {_fmt_int(chunks):>11} "
              f"{used * 100:>7.1f}% {epoch_str:>8} {('Y' if fits else 'N'):>5}{marker}")

    print()
    if chosen is None:
        d, _, chunks, _, epochs, _ = rows[-1]
        print(f"  没有 stride 能在 {args.target_epochs} epoch 内跑完。")
        print(f"  最大 stride={d} 仍有 {_fmt_int(chunks)} chunks ({epochs:.2f}x epochs)。")
        print(f"  建议：增加 --steps / --bs，或降低 --target_epochs。")
    else:
        d, records, chunks, used, epochs, _ = next(r for r in rows if r[0] == chosen)
        print(f"  推荐 stride = {chosen}  (= num_frames / {F // chosen}, 即每 episode {F // chosen} 个 offset)")
        print(f"    manifest 大小  = {_fmt_int(records)} records / {_fmt_int(chunks)} chunks")
        print(f"    预算消耗       = {used * 100:.1f}%   =>   {epochs:.2f}x epochs")


if __name__ == "__main__":
    main()
