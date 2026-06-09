"""Shared utilities for the FastWAM latent generation pipeline.

This module provides lightweight, dependency-minimal helpers used by both the
Stage-1 metadata scanning script and the Stage-2 distributed latent encoding
script. It deliberately does NOT reuse `RobotVideoDataset` to keep the code
path simple and easy to reason about for offline preprocessing.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as transforms_F



def video_sample_indices(num_frames: int, action_video_freq_ratio: int) -> list[int]:
    """Return frame indices for sub-sampling a clip of ``num_frames`` at ``action_video_freq_ratio``.

    This is the **canonical** implementation shared by both the offline latent
    generation pipeline (``scripts/latent/generate_latents.py``) and the online
    MTM simulation rollout (``fastwam.inference.mtm_sim_state``).

    Example:
        >>> video_sample_indices(33, 4)
        [0, 4, 8, 12, 16, 20, 24, 28, 32]  # 9 frames
    """
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}")
    if action_video_freq_ratio <= 0:
        raise ValueError(f"`action_video_freq_ratio` must be positive, got {action_video_freq_ratio}")
    return list(range(0, num_frames, max(1, action_video_freq_ratio)))


def _make_lerobot_video_key(camera_key: str) -> str:
    return f"observation.images.{camera_key}" if camera_key != "default" else "observation.images"


def read_dataset_info(dataset_dir: str | Path) -> dict[str, Any]:
    """Read `<dataset_dir>/meta/info.json` and return the parsed dict."""
    info_path = Path(dataset_dir) / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_episode_chunk_index(episode_index: int, chunks_size: int = 1000) -> int:
    return int(episode_index) // int(chunks_size)


def get_episode_parquet_path(dataset_dir: str | Path, episode_index: int, info: dict[str, Any] | None = None) -> str:
    if info is None:
        info = read_dataset_info(dataset_dir)
    chunks_size = int(info.get("chunks_size", 1000))
    rel = info["data_path"].format(
        episode_chunk=get_episode_chunk_index(episode_index, chunks_size),
        episode_index=int(episode_index),
    )
    return str(Path(dataset_dir) / rel)


def get_episode_video_paths(
    dataset_dir: str | Path, episode_index: int, camera_keys: Iterable[str], info: dict[str, Any] | None = None
) -> dict[str, str]:
    """Build absolute mp4 paths per camera from `info.json.video_path` template."""
    if info is None:
        info = read_dataset_info(dataset_dir)
    chunks_size = int(info.get("chunks_size", 1000))
    out: dict[str, str] = {}
    for cam in camera_keys:
        rel = info["video_path"].format(
            episode_chunk=get_episode_chunk_index(episode_index, chunks_size),
            episode_index=int(episode_index),
            video_key=_make_lerobot_video_key(cam),
        )
        out[cam] = str(Path(dataset_dir) / rel)
    return out


def get_episode_latent_save_path(
    dataset_dir: str | Path, episode_index: int, latent_subdir: str = "latents", info: dict[str, Any] | None = None
) -> str:
    if info is None:
        info = read_dataset_info(dataset_dir)
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = get_episode_chunk_index(episode_index, chunks_size)
    return str(Path(dataset_dir) / latent_subdir / f"chunk-{chunk_idx:03d}" / f"episode_{int(episode_index):06d}.pt")


def get_episode_latent_save_path_strided(
    dataset_dir: str | Path,
    episode_index: int,
    offset: int,
    latent_subdir: str = "latents",
    info: dict[str, Any] | None = None,
) -> str:
    """Stride-aware variant of `get_episode_latent_save_path`.

    Filename layout: `<dataset_dir>/<latent_subdir>/chunk-XXX/episode_YYYYYY_offset_ZZ.pt`.
    `ZZ` is the starting-frame offset (0 / stride / 2*stride / ...).
    """
    if info is None:
        info = read_dataset_info(dataset_dir)
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = get_episode_chunk_index(episode_index, chunks_size)
    return str(
        Path(dataset_dir)
        / latent_subdir
        / f"chunk-{chunk_idx:03d}"
        / f"episode_{int(episode_index):06d}_offset_{int(offset):02d}.pt"
    )


def probe_video_num_frames(video_path: str | Path) -> int:
    """Return the number of decoded frames of an mp4 using pyav.

    We deliberately use `stream.frames` first and fall back to a full decode
    iteration if the metadata is missing or zero. This keeps Stage-1 fast for
    well-formed mp4s (most of the time).
    """
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        n = int(stream.frames)
        if n > 0:
            return n
        n = 0
        for _ in container.decode(stream):
            n += 1
        return n
    finally:
        container.close()


def read_episode_parquet(dataset_dir: str | Path, episode_index: int, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the per-episode parquet and return the relevant columns as numpy.

    Returns a dict with keys: `frame_index`, `timestamp`, `task_index`, `episode_index`, `num_rows`, `path`.
    """
    path = get_episode_parquet_path(dataset_dir, episode_index, info=info)
    table = pq.read_table(path, columns=["frame_index", "timestamp", "task_index", "episode_index"])
    frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    timestamp = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float32)
    task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
    episode_index_arr = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    return {
        "path": path,
        "num_rows": int(len(frame_index)),
        "frame_index": frame_index,
        "timestamp": timestamp,
        "task_index": task_index,
        "episode_index": episode_index_arr,
    }


def read_episode_videos(
    dataset_dir: str | Path,
    episode_index: int,
    camera_keys: list[str],
    info: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Decode all frames for every camera of an episode using pyav.

    Returns a dict mapping camera key -> uint8 numpy array of shape (T, H, W, 3).
    """
    paths = get_episode_video_paths(dataset_dir, episode_index, camera_keys, info=info)
    out: dict[str, np.ndarray] = {}
    for cam, p in paths.items():
        container = av.open(str(p))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            frames: list[np.ndarray] = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
        finally:
            container.close()
        if len(frames) == 0:
            raise RuntimeError(f"Decoded 0 frames from `{p}`")
        out[cam] = np.stack(frames, axis=0)
    return out


@dataclass
class ChunkPlan:
    num_chunks: int
    last_chunk_real_len: int
    last_chunk_pad_len: int
    boundaries: list[tuple[int, int]]


@dataclass
class StridedChunkPlan:
    """Per-offset chunk plan under the stride (sliding-window) regime.

    Attributes:
        offset: starting frame offset (0, stride, 2*stride, ...).
        num_chunks: ``(total_frames - offset) // num_frames`` (tail discarded, no padding).
        boundaries: list of ``(start, end_exclusive)`` over the original frame index space.
    """

    offset: int
    num_chunks: int
    boundaries: list[tuple[int, int]]


def compute_stride_chunk_plans(total_frames: int, num_frames: int, stride: int) -> list[StridedChunkPlan]:
    """Split `total_frames` into non-overlapping chunks under multiple sliding-window offsets.

    Each offset ``s`` (0, stride, 2*stride, ..., num_frames - stride) defines an independent
    chunking starting at frame ``s``; chunks are non-overlapping of size ``num_frames``; the
    tail (fewer than ``num_frames`` remaining frames) is **discarded** (no padding).

    Constraints:
    - ``stride`` must be a positive integer that divides ``num_frames``.
    - ``1 <= stride <= num_frames``.
    - When ``stride == num_frames`` only ``offset=0`` is returned (degenerates to non-stride mode).
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if stride <= 0 or stride > num_frames:
        raise ValueError(f"stride must be in [1, num_frames={num_frames}], got {stride}")
    if num_frames % stride != 0:
        raise ValueError(f"stride={stride} must divide num_frames={num_frames}")

    plans: list[StridedChunkPlan] = []
    for offset in range(0, num_frames, stride):
        num_chunks = max(0, (total_frames - offset) // num_frames)
        boundaries = [(offset + i * num_frames, offset + (i + 1) * num_frames) for i in range(num_chunks)]
        plans.append(StridedChunkPlan(offset=int(offset), num_chunks=int(num_chunks), boundaries=boundaries))
    return plans


def compute_chunk_plan(total_frames: int, num_frames: int) -> ChunkPlan:
    """Split `total_frames` into non-overlapping chunks of size `num_frames`.

    The last chunk is right-padded with the last frame (filled by the caller)
    to reach length `num_frames` whenever `total_frames % num_frames != 0`.
    Boundary tuples are `(start, end_exclusive)` over the *original* frame
    index space, so the last tuple's range may be shorter than `num_frames`.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    num_chunks = (total_frames + num_frames - 1) // num_frames
    last_chunk_real_len = total_frames - (num_chunks - 1) * num_frames
    last_chunk_pad_len = num_frames - last_chunk_real_len

    boundaries: list[tuple[int, int]] = []
    for i in range(num_chunks):
        start = i * num_frames
        end = min(start + num_frames, total_frames)
        boundaries.append((start, end))
    return ChunkPlan(
        num_chunks=num_chunks,
        last_chunk_real_len=last_chunk_real_len,
        last_chunk_pad_len=last_chunk_pad_len,
        boundaries=boundaries,
    )


def _stack_per_camera_to_thwc(frames_dict: dict[str, np.ndarray], camera_keys: list[str]) -> np.ndarray:
    """Stack per-camera (T,H,W,3) arrays into (num_cams, T, H, W, 3) ensuring identical T/H/W."""
    if len(camera_keys) == 0:
        raise ValueError("`camera_keys` must be non-empty")
    arrs = []
    for cam in camera_keys:
        if cam not in frames_dict:
            raise KeyError(f"Camera `{cam}` missing in frames_dict")
        arrs.append(frames_dict[cam])
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"Per-camera frame shapes mismatch across cameras: {shapes}")
    return np.stack(arrs, axis=0)


def concat_multi_camera(
    frames_dict: dict[str, np.ndarray], mode: str | None, camera_keys: list[str]
) -> torch.Tensor:
    """Concatenate per-camera frames along the spatial axis.

    Input: dict of `cam_key -> (T, H, W, 3) uint8` numpy arrays plus an ordered
    `camera_keys` list (must match `shape_meta.images` ordering).
    Output: `(T, C=3, H_out, W_out)` uint8 torch tensor.
    """
    stacked = _stack_per_camera_to_thwc(frames_dict, camera_keys)  # (N, T, H, W, 3)
    video = torch.from_numpy(stacked).permute(0, 1, 4, 2, 3).contiguous()  # (N, T, C, H, W)
    num_cameras = video.shape[0]

    if mode == "robotwin":
        if num_cameras != 3:
            raise ValueError(f"`concat_multi_camera='robotwin'` requires 3 cams, got {num_cameras}")
        cam_top = transforms_F.resize(video[0], size=[256, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
        cam_left = transforms_F.resize(video[1], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
        cam_right = transforms_F.resize(video[2], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
        bottom = torch.cat([cam_left, cam_right], dim=-1)
        return torch.cat([cam_top, bottom], dim=-2)
    if mode == "robocasa":
        if num_cameras != 4:
            raise ValueError(f"`concat_multi_camera='robocasa'` requires 4 cams, got {num_cameras}")
        top = torch.cat([video[2], video[3]], dim=-1)
        bottom = torch.cat([video[0], video[1]], dim=-1)
        return torch.cat([top, bottom], dim=-2)
    if num_cameras > 1:
        if mode == "horizontal":
            return torch.cat([video[i] for i in range(num_cameras)], dim=-1)
        if mode == "vertical":
            return torch.cat([video[i] for i in range(num_cameras)], dim=-2)
        raise ValueError(f"Invalid concat_multi_camera={mode!r}; expected one of: horizontal, vertical, robotwin, robocasa.")
    return video.squeeze(0)


def preprocess_video_for_vae(video: torch.Tensor, video_size: tuple[int, int] | list[int]) -> torch.Tensor:
    """Resize-smallest-side -> center-crop -> normalize to [-1, 1].

    Input  : `(T, C, H, W)` uint8 tensor.
    Output : `(C, T, H_out, W_out)` float tensor in `[-1, 1]` (caller chooses dtype).

    Note: resize/crop are performed in float32 [0, 1] domain (not uint8) to avoid
    BICUBIC overshoot being clamped by uint8, which previously caused a small but
    consistent color/brightness drift compared to the online encoding path used by
    `Wan22Trainer` (which feeds `[-1, 1]` floats directly into the VAE without any
    uint8-domain interpolation).
    """
    if video.ndim != 4:
        raise ValueError(f"Expected (T, C, H, W) tensor, got shape {tuple(video.shape)}")
    if video.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 input, got {video.dtype}")

    H_out, W_out = int(video_size[0]), int(video_size[1])
    video_f = video.to(dtype=torch.float32) / 255.0

    _, _, H_in, W_in = video_f.shape
    scale = max(H_out / H_in, W_out / W_in)
    new_h, new_w = int(round(H_in * scale)), int(round(W_in * scale))
    out = transforms_F.resize(video_f, size=[new_h, new_w],
                              interpolation=transforms_F.InterpolationMode.BICUBIC, antialias=True)

    _, _, H_cur, W_cur = out.shape
    top = (H_cur - H_out) // 2
    left = (W_cur - W_out) // 2
    out = out[:, :, top:top + H_out, left:left + W_out]

    out = out * 2.0 - 1.0
    return out.permute(1, 0, 2, 3).contiguous()


def write_jsonl_atomic(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write a list of dicts as JSON-lines atomically (tmp file + os.replace)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, output_path)


def write_json_atomic(payload: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def atomic_torch_save(payload: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp))
    os.replace(tmp, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m fastwam.datasets.lerobot.latents.latent_io <dataset_dir> <episode_index> [cam_key1,cam_key2,...]")
        sys.exit(1)

    ds_dir = sys.argv[1]
    ep_idx = int(sys.argv[2])
    cam_keys_arg = sys.argv[3] if len(sys.argv) > 3 else "robot0_eye_in_hand,robot0_agentview_left,robot0_agentview_right"
    cam_keys = [c for c in cam_keys_arg.split(",") if c]

    info = read_dataset_info(ds_dir)
    parquet = read_episode_parquet(ds_dir, ep_idx, info=info)
    paths = get_episode_video_paths(ds_dir, ep_idx, cam_keys, info=info)
    n_frames = {c: probe_video_num_frames(p) for c, p in paths.items()}
    plan = compute_chunk_plan(parquet["num_rows"], 33)
    print(f"parquet rows = {parquet['num_rows']}, video frames = {n_frames}")
    print(f"chunk plan: num_chunks={plan.num_chunks}, last_real={plan.last_chunk_real_len}, last_pad={plan.last_chunk_pad_len}")
    print(f"first task_index = {int(parquet['task_index'][0])}")
    print(f"latent_save_path = {get_episode_latent_save_path(ds_dir, ep_idx, info=info)}")
