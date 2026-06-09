"""Stage-2: 分布式 VAE latent 编码（基于 Stage-1 manifest）。

调用方式：
    torchrun --standalone --nproc_per_node=8 scripts/latent/generate_latents.py \
        --config config_tmp/robocasa365_pretrain.yaml \
        --manifest_path ./data/robocasa365_pretrain.jsonl
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as _futures
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from fastwam.datasets.lerobot.latents.latent_bucket_sampler import (
    LatentEpisodeBucketSampler,
    LatentManifestDataset,
)
from fastwam.datasets.lerobot.latents.latent_io import (
    atomic_torch_save,
    concat_multi_camera,
    preprocess_video_for_vae,
    read_dataset_info,
    read_episode_parquet,
    read_episode_videos,
    video_sample_indices,
)
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"

# Atomic counter key used by the dynamic dispatch pool. A fresh init_process_group
# creates a fresh TCPStore so the counter implicitly starts at 0 each run.
COUNTER_KEY = "fastwam:latent_task_counter"


def _get_shared_store():
    """Return the TCPStore that `init_process_group(env://)` created on rank-0.

    Used as a lock-free atomic counter for dynamic task dispatch. Returns ``None``
    if no store is available (e.g. single-process runs), in which case the caller
    falls back to static slicing.
    """
    try:
        from torch.distributed.distributed_c10d import _get_default_store
        return _get_default_store()
    except Exception as exc:
        logger.warning("Could not access shared store for dynamic dispatch: %s", exc)
        return None


def _shared_counter_iter(store, work_indices: list[int], counter_key: str):
    """Yield dataset indices by atomically claiming slots from a shared counter.

    Every rank runs this independently; the TCPStore's `add` is atomic so each
    slot is handed out to exactly one rank. When the counter exceeds the global
    work-list length the iterator terminates.
    """
    total = len(work_indices)
    while True:
        new_val = store.add(counter_key, 1)
        slot = int(new_val) - 1
        if slot >= total:
            return
        yield work_indices[slot]


def _keepalive_until_barrier(
    *,
    device: str,
    torch_dtype: torch.dtype,
    matrix_size: int,
    burst_seconds: float,
    sleep_seconds: float,
    rank: int,
) -> None:
    """Hold a non-blocking barrier and keep the GPU minimally busy until peers join.

    Cluster managers commonly kill jobs whose GPUs report sustained 0% utilisation.
    After the dynamic counter is exhausted, this rank still has to wait for the
    slowest peer; we burn a small matmul burst every few seconds so nvml shows
    non-zero util without meaningfully eating power or compute.
    """
    barrier_handle = dist.barrier(async_op=True)
    if not device.startswith("cuda"):
        barrier_handle.wait()
        return

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch_dtype)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch_dtype)

    # Calibrate: how many matmuls fit into the requested burst window?
    torch.cuda.synchronize()
    t_cal = time.time()
    for _ in range(4):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    per_op = max((time.time() - t_cal) / 4.0, 1e-4)
    ops_per_burst = max(1, int(burst_seconds / per_op))

    t_start = time.time()
    t_last_log = t_start
    n_bursts = 0
    while not barrier_handle.is_completed():
        for _ in range(ops_per_burst):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        n_bursts += 1
        time.sleep(sleep_seconds)
        now = time.time()
        if now - t_last_log >= 60.0:
            logger.info(
                "Rank %d keepalive: waited %.0fs across %d bursts (%d ops each)",
                rank, now - t_start, n_bursts, ops_per_burst,
            )
            t_last_log = now

    waited = time.time() - t_start
    if waited > 1.0:
        logger.info("Rank %d keepalive done after %.1fs.", rank, waited)
    del a, b


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            logger.warning("LOCAL_RANK=%d >= device_count=%d, clamping.", local_rank, device_count)
            local_rank = max(device_count - 1, 0)
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _load_vae(model_cfg: dict | None, device: str, torch_dtype: torch.dtype):
    model_id = DEFAULT_MODEL_ID
    tokenizer_model_id = DEFAULT_TOKENIZER_MODEL_ID
    redirect_common_files = True
    if model_cfg is not None:
        model_id = str(model_cfg.get("model_id", model_id))
        tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", tokenizer_model_id))
        redirect_common_files = bool(model_cfg.get("redirect_common_files", redirect_common_files))

    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device).eval()
    return vae, str(vae_config.path)


def _video_sample_indices(num_frames: int, action_video_freq_ratio: int) -> list[int]:
    """Thin wrapper kept for backward compatibility; delegates to the canonical
    ``video_sample_indices`` in ``latent_io``."""
    return video_sample_indices(num_frames, action_video_freq_ratio)


def _action_range_for_chunk(
    parquet_frame_index: list[int] | Any,
    parquet_timestamp: list[float] | Any,
    chunk_start: int,
    num_frames: int,
    total_real: int,
) -> tuple[list[int], list[float]]:
    """Return [first, last] frame_index and timestamp for the `num_frames - 1` actions in a chunk.

    The stride pipeline discards tail chunks rather than padding, so callers must guarantee
    ``chunk_start + num_frames - 1 < total_real``. The clamping below is retained as a final
    safety net only.
    """
    last_action_local = num_frames - 2  # action covers obs[t..t+1]; last action is at obs index num_frames-2
    first_global = chunk_start
    last_global = chunk_start + last_action_local
    first_global_clamped = min(first_global, total_real - 1)
    last_global_clamped = min(last_global, total_real - 1)
    return (
        [int(parquet_frame_index[first_global_clamped]), int(parquet_frame_index[last_global_clamped])],
        [float(parquet_timestamp[first_global_clamped]), float(parquet_timestamp[last_global_clamped])],
    )


def _stage_record(rec: dict[str, Any]) -> dict[str, Any]:
    """CPU/IO half: read parquet + decode videos + concat + preprocess to a CPU tensor.

    Returns a dict carrying the preprocessed ``(C, T_total, H, W)`` float32 CPU tensor in
    ``[-1, 1]`` plus parquet columns needed downstream. Pure-CPU so it runs safely in a
    background prefetch thread while the GPU is busy with the previous record.
    """
    ds_dir = rec["dataset_dir"]
    ep_idx = int(rec["episode_index"])
    cam_keys: list[str] = list(rec["camera_keys"])
    concat_mode = rec.get("concat_multi_camera")
    video_size = list(rec["video_size"])

    info = read_dataset_info(ds_dir)
    parquet = read_episode_parquet(ds_dir, ep_idx, info=info)
    n_rows = parquet["num_rows"]
    if n_rows != int(rec["total_frames"]):
        raise RuntimeError(
            f"manifest total_frames={rec['total_frames']} disagrees with parquet rows={n_rows} for {ds_dir} ep={ep_idx}"
        )

    videos = read_episode_videos(ds_dir, ep_idx, cam_keys, info=info)
    for cam, arr in videos.items():
        if arr.shape[0] != n_rows:
            raise RuntimeError(
                f"Camera `{cam}` length {arr.shape[0]} != parquet rows {n_rows} for {ds_dir} ep={ep_idx}"
            )

    concat = concat_multi_camera(videos, concat_mode, cam_keys)  # (T, C, H, W) uint8
    video_tensor = preprocess_video_for_vae(concat, video_size)  # (C, T, H, W) float32 CPU

    return {
        "video_tensor": video_tensor,
        "parquet_frame_index": parquet["frame_index"],
        "parquet_timestamp": parquet["timestamp"],
        "n_rows": n_rows,
    }


def _encode_staged(
    rec: dict[str, Any],
    staged: dict[str, Any],
    vae,
    device: str,
    torch_dtype: torch.dtype,
) -> dict[str, Any]:
    """GPU half: stack all chunks of one record into a single batched VAE forward."""
    ds_dir = rec["dataset_dir"]
    ep_idx = int(rec["episode_index"])
    offset = int(rec["offset"])
    cam_keys: list[str] = list(rec["camera_keys"])
    num_frames = int(rec["num_frames"])
    action_video_freq_ratio = int(rec["action_video_freq_ratio"])
    concat_mode = rec.get("concat_multi_camera")
    video_size = list(rec["video_size"])
    expected_num_chunks = int(rec["num_chunks"])
    if expected_num_chunks <= 0:
        raise RuntimeError(
            f"Manifest record has num_chunks={expected_num_chunks} for {ds_dir} ep={ep_idx} offset={offset}"
        )

    n_rows: int = staged["n_rows"]
    parquet_frame_index = staged["parquet_frame_index"]
    parquet_timestamp = staged["parquet_timestamp"]
    video_tensor = staged["video_tensor"].to(device=device, dtype=torch_dtype, non_blocking=True)

    sample_indices_t = torch.tensor(
        _video_sample_indices(num_frames, action_video_freq_ratio), device=device, dtype=torch.long,
    )

    chunk_frame_idx_ranges: list[list[int]] = []
    chunk_timestamp_ranges: list[list[float]] = []
    segs_sampled: list[torch.Tensor] = []
    for chunk_i in range(expected_num_chunks):
        s = offset + chunk_i * num_frames
        e = s + num_frames
        if e > n_rows:
            raise RuntimeError(
                f"Tail chunk out of range for {ds_dir} ep={ep_idx} offset={offset} chunk_i={chunk_i}: "
                f"need frames [{s}, {e}) but total={n_rows}"
            )
        seg = video_tensor[:, s:e]
        segs_sampled.append(seg.index_select(1, sample_indices_t))
        f_range, t_range = _action_range_for_chunk(
            parquet_frame_index, parquet_timestamp, s, num_frames, n_rows,
        )
        chunk_frame_idx_ranges.append(f_range)
        chunk_timestamp_ranges.append(t_range)

    batched = torch.stack(segs_sampled, dim=0)  # (B, C, T, H, W)
    with torch.no_grad():
        latent_batch = vae.single_encode(batched, device=device)  # (B, z, T_lat, H, W)
    latent_tensor = latent_batch.detach().to("cpu", dtype=torch.bfloat16).contiguous()

    payload = {
        "vae_latent": latent_tensor,
        "chunk_frame_idx": torch.tensor(chunk_frame_idx_ranges, dtype=torch.long),
        "chunk_timestamp": torch.tensor(chunk_timestamp_ranges, dtype=torch.float32),
        "episode_index": ep_idx,
        "offset": offset,
        "task_index": int(rec["task_index"]),
        "dataset_dir": ds_dir,
        "num_chunks": int(expected_num_chunks),
        "num_frames": num_frames,
        "stride": int(rec.get("stride", num_frames)),
        "action_video_freq_ratio": action_video_freq_ratio,
        "video_size": [int(video_size[0]), int(video_size[1])],
        "concat_multi_camera": concat_mode,
        "camera_keys": cam_keys,
    }
    return payload


def main():
    setup_logging(log_level=logging.INFO)
    parser = argparse.ArgumentParser(description="Stage-2: 分布式 VAE latent 编码")
    parser.add_argument("--config", required=True, help="task yaml 路径")
    parser.add_argument("--manifest_path", required=True, help="Stage-1 输出的 manifest JSONL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefetch_depth", type=int, default=2)
    parser.add_argument("--prefetch_workers", type=int, default=None)
    parser.add_argument("--static_dispatch", action="store_true",
                        help="禁用动态分发，回退到静态分片")
    parser.add_argument("--keepalive_matrix_size", type=int, default=2048)
    parser.add_argument("--keepalive_burst_seconds", type=float, default=0.5)
    parser.add_argument("--keepalive_sleep_seconds", type=float, default=3.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})

    is_distributed, rank, world_size, local_rank = _init_distributed()

    manifest_path = args.manifest_path
    seed = args.seed

    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}; run scan_dataset_meta.py first.")

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16

    logger.info(
        "Stage-2 latent encoding: rank=%d/%d device=%s dtype=%s manifest=%s",
        rank, world_size, device, torch_dtype, manifest_path,
    )

    vae, vae_path = _load_vae(model_cfg, device=device, torch_dtype=torch_dtype)
    logger.info("Loaded WAN2.2 VAE from %s", vae_path)

    dataset = LatentManifestDataset(manifest_path)

    use_dynamic = (not args.static_dispatch) and is_distributed
    store = _get_shared_store() if use_dynamic else None
    if use_dynamic and store is None:
        logger.warning("Dynamic dispatch requested but shared store unavailable; falling back to static slicing.")
        use_dynamic = False

    if use_dynamic:
        global_sampler = LatentEpisodeBucketSampler(
            dataset, global_rank=0, world_size=1, shuffle=True, seed=seed,
        )
        work_indices = list(global_sampler)
        total_work_global = len(work_indices)
        nominal_per_rank = (total_work_global + world_size - 1) // world_size
        # Make sure all ranks reach the loop before anyone starts claiming, so the
        # counter starts from a clean state for this run.
        dist.barrier()
        logger.info(
            "Rank %d: dynamic dispatch over %d global records (~%d expected per rank).",
            rank, total_work_global, nominal_per_rank,
        )
        it = _shared_counter_iter(store, work_indices, COUNTER_KEY)
        pbar_total = nominal_per_rank
    else:
        sampler = LatentEpisodeBucketSampler(
            dataset, global_rank=rank, world_size=world_size, shuffle=True, seed=seed,
        )
        my_indices = list(sampler)
        logger.info(
            "Rank %d: %d records assigned (out of %d total, each = one (episode, offset)).",
            rank, len(my_indices), len(dataset),
        )
        it = iter(my_indices)
        pbar_total = len(my_indices)

    skipped_existing = 0
    encoded = 0
    failed = 0

    prefetch_depth = max(1, args.prefetch_depth)
    prefetch_workers = max(1, args.prefetch_workers or prefetch_depth)

    pbar = tqdm(
        total=pbar_total,
        desc=f"encode (rank {rank}/{world_size})" if is_distributed else "encode",
        unit="rec",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    )
    t_start = time.time()

    pool = _futures.ThreadPoolExecutor(
        max_workers=prefetch_workers, thread_name_prefix="latent_prefetch"
    )

    def _submit(idx: int) -> tuple[dict[str, Any], Path, _futures.Future | None]:
        """Resolve a manifest index into (rec, save_path, staging-future-or-None).

        Returns ``future=None`` when the latent already exists and should be skipped — in that
        case the main loop bumps the skip counter without doing any work.
        """
        rec = dataset[idx]
        save_path = Path(rec["latent_save_path"])
        if save_path.exists():
            return rec, save_path, None
        return rec, save_path, pool.submit(_stage_record, rec)

    try:
        pending: collections.deque[tuple[dict[str, Any], Path, _futures.Future | None]] = (
            collections.deque()
        )
        for _ in range(prefetch_depth):
            try:
                pending.append(_submit(next(it)))
            except StopIteration:
                break

        while pending:
            rec, save_path, fut = pending.popleft()
            # Keep the IO thread warm by submitting the next record before we block on the GPU.
            try:
                pending.append(_submit(next(it)))
            except StopIteration:
                pass

            if fut is None:
                skipped_existing += 1
                pbar.update(1)
                continue

            try:
                staged = fut.result()
                payload = _encode_staged(rec, staged, vae, device=device, torch_dtype=torch_dtype)
                atomic_torch_save(payload, save_path)
                encoded += 1
            except Exception as exc:
                logger.error(
                    "Failed encoding %s ep=%s offset=%s: %s",
                    rec.get("dataset_dir"), rec.get("episode_index"), rec.get("offset"), exc,
                )
                failed += 1
            pbar.update(1)
    finally:
        pool.shutdown(wait=True)
        pbar.close()

    elapsed = time.time() - t_start
    logger.info(
        "Rank %d finished own work in %.1fs (encoded=%d skipped=%d failed=%d).",
        rank, elapsed, encoded, skipped_existing, failed,
    )

    if is_distributed:
        # Stragglers may still be running. Hold a non-blocking barrier and keep the GPU
        # warm with a tiny matmul so cluster watchdogs don't kill us for low utilization.
        _keepalive_until_barrier(
            device=device,
            torch_dtype=torch_dtype,
            matrix_size=args.keepalive_matrix_size,
            burst_seconds=args.keepalive_burst_seconds,
            sleep_seconds=args.keepalive_sleep_seconds,
            rank=rank,
        )

        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        counts = torch.tensor([encoded, skipped_existing, failed], device=reduce_device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if rank == 0:
            logger.info(
                "Done. encoded=%d skipped_existing=%d failed=%d (global)",
                int(counts[0]), int(counts[1]), int(counts[2]),
            )
        dist.barrier()
        dist.destroy_process_group()
    else:
        logger.info(
            "Done in %.1fs. encoded=%d skipped_existing=%d failed=%d",
            elapsed, encoded, skipped_existing, failed,
        )


if __name__ == "__main__":
    main()
