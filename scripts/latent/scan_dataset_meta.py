"""Stage-1: 并行扫描 LeRobot 数据集元数据，生成 latent manifest。

支持两种 IO 后端：
  --backend nas（默认）：NAS 挂载路径直读
  --backend oss：通过 ossutil cp 到 /tmp 访问

调用方式：
    # NAS
    python scripts/latent/scan_dataset_meta.py \
        --config config_tmp/robocasa365_pretrain.yaml \
        --manifest_path ./data/robocasa365_pretrain.jsonl \
        --num_workers 32

    # OSS
    python scripts/latent/scan_dataset_meta.py \
        --config config_tmp/robocasa365_pretrain.yaml \
        --manifest_path oss://xlab-dev/.../latent_manifest.jsonl \
        --backend oss --oss_mount /data/oss_bucket_0:oss://xlab-dev
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from fastwam.datasets.lerobot.latents.latent_io import (
    _make_lerobot_video_key,
    compute_stride_chunk_plans,
    get_episode_chunk_index,
    get_episode_latent_save_path_strided,
    get_episode_parquet_path,
    get_episode_video_paths,
    probe_video_num_frames,
    read_dataset_info,
    read_episode_parquet,
    write_json_atomic,
    write_jsonl_atomic,
)
from fastwam.datasets.lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_VAE_SPATIAL_DOWNSAMPLE = 16
DEFAULT_VAE_TEMPORAL_DOWNSAMPLE = 4
DEFAULT_OSS_MOUNT = "/data/oss_bucket_0:oss://xlab-dev"


# ═══════════════════════════════════════════════════════════════════════════════
# OSS 工具函数
# ═══════════════════════════════════════════════════════════════════════════════


def _is_oss(p: str) -> bool:
    return p.startswith("oss://")


def _make_oss_translator(mount_spec: str) -> Callable[[str], str]:
    if ":" not in mount_spec or not mount_spec.split(":", 1)[1].startswith("oss://"):
        raise ValueError(f"--oss_mount 格式应为 LOCAL:oss://BUCKET[/PREFIX]，got {mount_spec!r}")
    local_root, oss_root = mount_spec.split(":", 1)
    local_root = local_root.rstrip("/") + "/"
    oss_root = oss_root.rstrip("/") + "/"

    def _translate(local_path: str) -> str:
        s = str(local_path)
        if not s.startswith(local_root):
            raise ValueError(f"路径 {s!r} 不在挂载根 {local_root!r} 下")
        return oss_root + s[len(local_root):]

    return _translate


def _ossutil_cp(src: str, dst: str, timeout_seconds: float = 600.0) -> None:
    cmd = ["ossutil", "cp", "-f", src, dst]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise RuntimeError("`ossutil` not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ossutil cp timed out ({timeout_seconds:.0f}s): {src} -> {dst}") from exc
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or proc.stdout or "").splitlines()[-3:])[:300]
        raise RuntimeError(f"ossutil cp failed (exit={proc.returncode}): {src} -> {dst}: {tail}")


def _write_jsonl_local(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, output_path)


def _write_json_local(payload: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)


def _write_jsonl_any(records: list[dict[str, Any]], out_str: str) -> None:
    if _is_oss(out_str):
        fd, tmp = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            _write_jsonl_local(records, Path(tmp))
            _ossutil_cp(tmp, out_str)
        finally:
            os.unlink(tmp) if os.path.exists(tmp) else None
    else:
        _write_jsonl_local(records, Path(out_str))


def _write_json_any(payload: Any, out_str: str) -> None:
    if _is_oss(out_str):
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            _write_json_local(payload, Path(tmp))
            _ossutil_cp(tmp, out_str)
        finally:
            os.unlink(tmp) if os.path.exists(tmp) else None
    else:
        _write_json_local(payload, Path(out_str))


def _derived_path(base: str, ext_old: str, suffix_new: str) -> str:
    if base.endswith(ext_old):
        return base[:-len(ext_old)] + suffix_new
    return base + suffix_new


# ═══════════════════════════════════════════════════════════════════════════════
# 核心计算（NAS / OSS 共用）
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_n_lat_per_chunk(num_frames: int, action_video_freq_ratio: int, temporal_downsample: int) -> int:
    sample_indices = list(range(0, num_frames, max(1, action_video_freq_ratio)))
    n_video = len(sample_indices)
    return 1 + (n_video - 1) // temporal_downsample


# ═══════════════════════════════════════════════════════════════════════════════
# NAS 模式：直接文件系统读
# ═══════════════════════════════════════════════════════════════════════════════


def _scan_one_episode_nas(job: dict[str, Any]) -> dict[str, Any]:
    ds_dir = job["dataset_dir"]
    ep_idx = job["episode_index"]
    cam_keys = job["camera_keys"]
    num_frames = int(job["num_frames"])
    stride = int(job["stride"])
    try:
        info = read_dataset_info(ds_dir)
        parquet = read_episode_parquet(ds_dir, ep_idx, info=info)
        n_rows = parquet["num_rows"]
        if n_rows == 0:
            return {"status": "skip", "reason": "empty_parquet", "dataset_dir": ds_dir, "episode_index": ep_idx}
        fmin = int(parquet["frame_index"][0])
        fmax = int(parquet["frame_index"][-1])
        if fmax - fmin + 1 != n_rows:
            return {"status": "skip", "reason": "non_contiguous_frame_index",
                    "dataset_dir": ds_dir, "episode_index": ep_idx,
                    "frame_index_range": [fmin, fmax], "num_rows": n_rows}

        video_paths = get_episode_video_paths(ds_dir, ep_idx, cam_keys, info=info)
        per_cam_frames = {cam: probe_video_num_frames(p) for cam, p in video_paths.items()}
        unique_lens = set(per_cam_frames.values())
        if len(unique_lens) != 1:
            return {"status": "skip", "reason": "camera_length_mismatch",
                    "dataset_dir": ds_dir, "episode_index": ep_idx,
                    "per_camera_frames": per_cam_frames, "num_rows": n_rows}
        video_frames = next(iter(unique_lens))
        if video_frames != n_rows:
            return {"status": "skip", "reason": "parquet_video_length_mismatch",
                    "dataset_dir": ds_dir, "episode_index": ep_idx,
                    "video_frames": video_frames, "num_rows": n_rows}

        return _build_manifest_records(job, info, parquet, n_rows, fmin, fmax)
    except Exception as exc:
        return {"status": "skip", "reason": f"exception:{type(exc).__name__}:{exc}",
                "dataset_dir": ds_dir, "episode_index": ep_idx}


# ═══════════════════════════════════════════════════════════════════════════════
# OSS 模式：ossutil 下载到 tempdir
# ═══════════════════════════════════════════════════════════════════════════════


def _download_episode_files_oss(
    job: dict[str, Any], workdir: Path,
) -> tuple[Path, dict[str, Path]]:
    ds_oss = job["_ds_oss_uri"]
    info = job["_info_json"]
    ep_idx = int(job["episode_index"])
    cam_keys = list(job["camera_keys"])
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_i = get_episode_chunk_index(ep_idx, chunks_size)

    parquet_rel = info["data_path"].format(episode_chunk=chunk_i, episode_index=ep_idx)
    parquet_oss = f"{ds_oss}/{parquet_rel}"
    parquet_local = workdir / Path(parquet_rel).name

    video_targets: list[tuple[str, str, Path]] = []
    for cam in cam_keys:
        rel = info["video_path"].format(
            episode_chunk=chunk_i, episode_index=ep_idx,
            video_key=_make_lerobot_video_key(cam),
        )
        video_targets.append((cam, f"{ds_oss}/{rel}", workdir / f"{cam}__{Path(rel).name}"))

    with ThreadPoolExecutor(max_workers=1 + len(video_targets)) as ex:
        futs = [ex.submit(_ossutil_cp, parquet_oss, str(parquet_local))]
        for cam, oss_uri, local_p in video_targets:
            futs.append(ex.submit(_ossutil_cp, oss_uri, str(local_p)))
        for f in futs:
            f.result()

    return parquet_local, {cam: local_p for cam, _, local_p in video_targets}


def _scan_one_episode_oss(job: dict[str, Any]) -> dict[str, Any]:
    ds_dir = job["dataset_dir"]
    ep_idx = int(job["episode_index"])
    cam_keys = list(job["camera_keys"])
    num_frames = int(job["num_frames"])
    stride = int(job["stride"])
    info = job["_info_json"]

    workdir = Path(tempfile.mkdtemp(prefix=f"ep_{ep_idx:06d}_"))
    try:
        try:
            parquet_local, video_locals = _download_episode_files_oss(job, workdir)
        except RuntimeError as exc:
            return {"status": "skip", "reason": f"ossutil_cp:{exc}",
                    "dataset_dir": ds_dir, "episode_index": ep_idx}
        try:
            table = pq.read_table(
                str(parquet_local),
                columns=["frame_index", "timestamp", "task_index", "episode_index"],
            )
            frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
            task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
            n_rows = int(len(frame_index))
            if n_rows == 0:
                return {"status": "skip", "reason": "empty_parquet",
                        "dataset_dir": ds_dir, "episode_index": ep_idx}
            fmin = int(frame_index[0])
            fmax = int(frame_index[-1])
            if fmax - fmin + 1 != n_rows:
                return {"status": "skip", "reason": "non_contiguous_frame_index",
                        "dataset_dir": ds_dir, "episode_index": ep_idx,
                        "frame_index_range": [fmin, fmax], "num_rows": n_rows}

            per_cam_frames = {cam: probe_video_num_frames(p) for cam, p in video_locals.items()}
            unique_lens = set(per_cam_frames.values())
            if len(unique_lens) != 1:
                return {"status": "skip", "reason": "camera_length_mismatch",
                        "dataset_dir": ds_dir, "episode_index": ep_idx,
                        "per_camera_frames": per_cam_frames, "num_rows": n_rows}
            video_frames = next(iter(unique_lens))
            if video_frames != n_rows:
                return {"status": "skip", "reason": "parquet_video_length_mismatch",
                        "dataset_dir": ds_dir, "episode_index": ep_idx,
                        "video_frames": video_frames, "num_rows": n_rows}

            parquet_data = {
                "num_rows": n_rows,
                "frame_index": frame_index,
                "task_index": task_index,
                "path": get_episode_parquet_path(ds_dir, ep_idx, info=info),
            }
            return _build_manifest_records(job, info, parquet_data, n_rows, fmin, fmax)
        except Exception as exc:
            return {"status": "skip", "reason": f"exception:{type(exc).__name__}:{exc}",
                    "dataset_dir": ds_dir, "episode_index": ep_idx}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 共享：manifest record 构建
# ═══════════════════════════════════════════════════════════════════════════════


def _build_manifest_records(
    job: dict[str, Any], info: dict, parquet: dict, n_rows: int, fmin: int, fmax: int,
) -> dict[str, Any]:
    ds_dir = job["dataset_dir"]
    ep_idx = job["episode_index"]
    cam_keys = job["camera_keys"]
    num_frames = int(job["num_frames"])
    stride = int(job["stride"])

    plans = compute_stride_chunk_plans(n_rows, num_frames, stride)
    H, W = job["video_size"]
    sd = int(job["vae_spatial_downsample"])
    H_lat, W_lat = H // sd, W // sd
    n_lat_per_chunk = _compute_n_lat_per_chunk(
        num_frames, int(job["action_video_freq_ratio"]), int(job["vae_temporal_downsample"])
    )

    parquet_path_str = parquet.get("path") or get_episode_parquet_path(ds_dir, ep_idx, info=info)
    video_paths_str = get_episode_video_paths(ds_dir, ep_idx, cam_keys, info=info)

    records: list[dict[str, Any]] = []
    for plan in plans:
        if plan.num_chunks <= 0:
            continue
        records.append({
            "dataset_dir": ds_dir,
            "episode_index": ep_idx,
            "offset": plan.offset,
            "task_index": int(parquet["task_index"][0]),
            "parquet_path": parquet_path_str,
            "video_paths": video_paths_str,
            "latent_save_path": get_episode_latent_save_path_strided(
                ds_dir, ep_idx, plan.offset, latent_subdir=job["latent_subdir"], info=info,
            ),
            "total_frames": n_rows,
            "frame_index_min": fmin,
            "frame_index_max": fmax,
            "num_chunks": int(plan.num_chunks),
            "bucket_key": [int(plan.num_chunks), H_lat, W_lat],
            "n_lat_per_chunk": n_lat_per_chunk,
            "num_frames": num_frames,
            "stride": stride,
            "action_video_freq_ratio": int(job["action_video_freq_ratio"]),
            "video_size": [H, W],
            "concat_multi_camera": job["concat_multi_camera"],
            "camera_keys": cam_keys,
        })
    if not records:
        return {"status": "skip", "reason": "no_chunks_at_any_offset",
                "dataset_dir": ds_dir, "episode_index": ep_idx,
                "total_frames": n_rows, "num_frames": num_frames, "stride": stride}
    return {"status": "ok", "records": records}


# ═══════════════════════════════════════════════════════════════════════════════
# 调度入口
# ═══════════════════════════════════════════════════════════════════════════════


def _scan_one_episode(job: dict[str, Any]) -> dict[str, Any]:
    if "_ds_oss_uri" in job:
        return _scan_one_episode_oss(job)
    return _scan_one_episode_nas(job)


def _build_jobs(
    data_mix: str,
    data_cfg: dict[str, Any],
    latent_subdir: str,
    translator: Optional[Callable[[str], str]] = None,
) -> list[dict[str, Any]]:
    mixture = DATASET_NAMED_MIXTURES[data_mix]
    num_frames = int(data_cfg.get("num_frames", 33))
    stride = int(data_cfg.get("stride", num_frames))
    action_video_freq_ratio = int(data_cfg.get("action_video_freq_ratio", 4))

    jobs: list[dict[str, Any]] = []
    for dataset_dir, _weight, robot_type in mixture:
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        camera_keys = data_config._camera_keys_stripped()
        video_size = list(data_config.video_size)
        concat_mode = data_config.video_concat_mode

        if translator is not None:
            try:
                ds_oss_uri = translator(dataset_dir)
            except ValueError as exc:
                logger.warning("跳过 (translate failed): %s", exc)
                continue
            fd, tmp = tempfile.mkstemp(suffix=".info.json")
            os.close(fd)
            try:
                _ossutil_cp(ds_oss_uri.rstrip("/") + "/meta/info.json", tmp)
                info = json.loads(Path(tmp).read_text(encoding="utf-8"))
            except RuntimeError as exc:
                logger.warning("跳过 %s (info.json 获取失败): %s", dataset_dir, exc)
                continue
            finally:
                os.unlink(tmp) if os.path.exists(tmp) else None
        else:
            try:
                info = read_dataset_info(dataset_dir)
            except Exception as exc:
                logger.warning("跳过 %s (info.json 读取失败): %s", dataset_dir, exc)
                continue

        total_episodes = int(info.get("total_episodes", 0))
        if total_episodes <= 0:
            logger.warning("数据集 %s 的 total_episodes=0，跳过", dataset_dir)
            continue

        for ep in range(total_episodes):
            job = {
                "dataset_dir": str(dataset_dir),
                "episode_index": int(ep),
                "camera_keys": camera_keys,
                "num_frames": num_frames,
                "stride": stride,
                "action_video_freq_ratio": action_video_freq_ratio,
                "video_size": video_size,
                "concat_multi_camera": concat_mode,
                "vae_spatial_downsample": DEFAULT_VAE_SPATIAL_DOWNSAMPLE,
                "vae_temporal_downsample": DEFAULT_VAE_TEMPORAL_DOWNSAMPLE,
                "latent_subdir": latent_subdir,
            }
            if translator is not None:
                job["_ds_oss_uri"] = ds_oss_uri.rstrip("/")
                job["_info_json"] = info
            jobs.append(job)
    return jobs


def _scan_all(jobs: list[dict[str, Any]], num_workers: int):
    ok_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if num_workers <= 1:
        for job in tqdm(jobs, desc="scan", dynamic_ncols=True):
            res = _scan_one_episode(job)
            if res["status"] == "ok":
                ok_records.extend(res["records"])
            else:
                skipped.append(res)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for res in tqdm(
                pool.imap_unordered(_scan_one_episode, jobs, chunksize=4),
                total=len(jobs), desc="scan", dynamic_ncols=True,
            ):
                if res["status"] == "ok":
                    ok_records.extend(res["records"])
                else:
                    skipped.append(res)

    ok_records.sort(key=lambda r: (r["dataset_dir"], r["episode_index"], r["offset"]))
    skipped.sort(key=lambda r: (r.get("dataset_dir", ""), r.get("episode_index", -1)))
    return ok_records, skipped


def _summarise_buckets(records: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counter: Counter[tuple[int, int, int]] = Counter()
    per_dataset_episode_offset_counter: Counter[str] = Counter()
    per_dataset_episode_counter: dict[str, set[int]] = {}
    total_chunks = 0
    for rec in records:
        bucket_counter[tuple(rec["bucket_key"])] += 1
        ds = rec["dataset_dir"]
        per_dataset_episode_offset_counter[ds] += 1
        per_dataset_episode_counter.setdefault(ds, set()).add(int(rec["episode_index"]))
        total_chunks += int(rec["num_chunks"])
    return {
        "total_records": len(records),
        "total_chunks": total_chunks,
        "buckets": [
            {"bucket_key": list(k), "num_records": v}
            for k, v in sorted(bucket_counter.items(), key=lambda kv: -kv[1])
        ],
        "per_dataset": [
            {"dataset_dir": k, "num_records": per_dataset_episode_offset_counter[k],
             "num_episodes": len(per_dataset_episode_counter[k])}
            for k in sorted(per_dataset_episode_offset_counter.keys())
        ],
    }


def _write_sample_index_files(
    ok_records: list[dict[str, Any]],
    num_frames: int,
    stride: int,
    translator: Optional[Callable[[str], str]] = None,
) -> list[str]:
    per_ds: dict[str, list[dict[str, Any]]] = {}
    for rec in ok_records:
        per_ds.setdefault(rec["dataset_dir"], []).append(rec)

    written: list[str] = []
    num_offsets = max(1, num_frames // stride)
    for ds_dir, recs in per_ds.items():
        recs_sorted = sorted(recs, key=lambda r: (int(r["episode_index"]), int(r["offset"])))
        episodes_dict: dict[str, list[dict[str, Any]]] = {}
        total_sample_count = 0
        for rec in recs_sorted:
            ep_idx = int(rec["episode_index"])
            offset = int(rec["offset"])
            n_ck = int(rec["num_chunks"])
            latent_path = str(rec["latent_save_path"])
            ep_key = str(ep_idx)
            episodes_dict.setdefault(ep_key, [])
            for chunk_idx in range(n_ck):
                episodes_dict[ep_key].append({
                    "episode_idx": ep_idx, "offset": offset, "chunk_idx": chunk_idx,
                    "latent_path": latent_path,
                    "action_start_frame": offset + chunk_idx * num_frames,
                })
                total_sample_count += 1
        payload = {
            "stride": int(stride), "num_frames": int(num_frames),
            "num_offsets": int(num_offsets), "total_samples": total_sample_count,
            "episodes": episodes_dict,
        }
        rel_name = f"latent_sample_index_stride{stride}_nf{num_frames}.json"
        local_target = str(Path(ds_dir) / "meta" / rel_name)
        if translator is not None:
            oss_target = translator(local_target)
            _write_json_any(payload, oss_target)
            written.append(oss_target)
        else:
            write_json_atomic(payload, local_target)
            written.append(local_target)
        logger.info("写入 sample index `%s` (%d episodes, %d samples)",
                     written[-1], len(episodes_dict), total_sample_count)
    return written


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    setup_logging(log_level=logging.INFO)
    parser = argparse.ArgumentParser(description="Stage-1: 扫描数据集元数据，生成 latent manifest")
    parser.add_argument("--config", required=True, help="task yaml 路径")
    parser.add_argument("--manifest_path", required=True, help="manifest 输出路径（本地或 oss://）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行 worker 数（默认 CPU/2）")
    parser.add_argument("--backend", choices=["nas", "oss"], default="nas")
    parser.add_argument("--oss_mount", default=DEFAULT_OSS_MOUNT,
                        help="OSS 挂载映射，格式 LOCAL:oss://BUCKET")
    parser.add_argument("--latent_subdir", default="latents")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    data_mix = data_cfg["data_mix"]

    manifest_path = args.manifest_path
    summary_path = _derived_path(manifest_path, ".jsonl", ".summary.json")
    skipped_path = _derived_path(manifest_path, ".jsonl", ".skipped.jsonl")
    num_workers = args.num_workers or max(1, (os.cpu_count() or 8) // 2)
    latent_subdir = args.latent_subdir

    translator = _make_oss_translator(args.oss_mount) if args.backend == "oss" else None
    if translator:
        logger.info("OSS 模式: %s", args.oss_mount)

    logger.info("data_mix=%s, backend=%s, num_workers=%d", data_mix, args.backend, num_workers)

    jobs = _build_jobs(data_mix, data_cfg, latent_subdir, translator)
    if not jobs:
        raise RuntimeError("没有找到有效的 episode")
    logger.info("共 %d 个 episode 待扫描", len(jobs))

    ok_records, skipped = _scan_all(jobs, num_workers)
    if not ok_records:
        raise RuntimeError("所有 episode 验证失败，无有效记录")

    num_frames = int(data_cfg.get("num_frames", 33))
    stride = int(data_cfg.get("stride", num_frames))

    start = time.time()
    if _is_oss(manifest_path) or translator:
        _write_jsonl_any(ok_records, manifest_path)
    else:
        write_jsonl_atomic(ok_records, manifest_path)

    summary = _summarise_buckets(ok_records)
    summary["num_skipped"] = len(skipped)
    summary["manifest_path"] = manifest_path
    summary["data_mix"] = data_mix
    summary["num_frames"] = num_frames
    summary["stride"] = stride
    if translator:
        summary["oss_mount"] = args.oss_mount

    index_files = _write_sample_index_files(ok_records, num_frames, stride, translator)
    summary["sample_index_files"] = index_files

    if _is_oss(summary_path) or translator:
        _write_json_any(summary, summary_path)
    else:
        write_json_atomic(summary, summary_path)

    if skipped:
        if _is_oss(skipped_path) or translator:
            _write_jsonl_any(skipped, skipped_path)
        else:
            write_jsonl_atomic(skipped, skipped_path)

    logger.info("写入 manifest 耗时 %.2fs", time.time() - start)
    logger.info("Manifest: %s (%d records)", manifest_path, len(ok_records))
    logger.info("Summary: %s", summary_path)
    if skipped:
        logger.warning("跳过 %d 个 episode，详见 %s", len(skipped), skipped_path)
        for rec in skipped[:10]:
            logger.warning("  skip[%s ep=%s] reason=%s",
                           rec.get("dataset_dir"), rec.get("episode_index"), rec.get("reason"))
    logger.info("Top buckets:")
    for bk in summary["buckets"][:10]:
        logger.info("  bucket=%s records=%d", bk["bucket_key"], bk["num_records"])


if __name__ == "__main__":
    main()
