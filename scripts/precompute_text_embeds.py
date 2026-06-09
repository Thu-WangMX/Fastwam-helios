"""基于注册表的 text embedding 预计算脚本。

从 task yaml 的 data.data_mix 解析出所有子数据集，
为每个子数据集读取 tasks.jsonl，用 DataConfig 的 prompt 模板构造 prompt，
编码后存到 <dataset_dir>/text_embeds/{hash}.pt。

用法:
    python scripts/precompute_text_embeds.py --config config_tmp/robocasa365_pretrain.yaml
    torchrun --nproc_per_node=8 scripts/precompute_text_embeds.py --config config_tmp/robocasa365_pretrain.yaml
"""

import argparse
import hashlib
import json
import logging
import os
import re
import uuid
import yaml
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm

from fastwam.datasets.lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from fastwam.datasets.lerobot.registry.prompt_utils import build_prompt_from_config, get_text_embeds_dir
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            logger.warning(f"LOCAL_RANK={local_rank} >= device_count={device_count}, clamping")
            local_rank = max(device_count - 1, 0)
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _read_unique_prompts_for_dataset(dataset_dir: str, data_config) -> list[str]:
    """读取 tasks.jsonl，用 DataConfig 的 prompt 模板构造去重后的 prompts。"""
    tasks_path = Path(dataset_dir) / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
    prompts: list[str] = []
    seen: set[str] = set()
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "task" not in record:
                continue
            task = str(record["task"])
            prompt = build_prompt_from_config(task, data_config)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def _collect_all_prompts(data_mix: str) -> list[tuple[str, str, list[str]]]:
    """从注册表解析 mixture，返回 [(dataset_dir, robot_type, [prompts]), ...]。"""
    mixture = DATASET_NAMED_MIXTURES[data_mix]
    results: list[tuple[str, str, list[str]]] = []
    for dataset_dir, _weight, robot_type in mixture:
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        prompts = _read_unique_prompts_for_dataset(dataset_dir, data_config)
        results.append((dataset_dir, robot_type, prompts))
    return results


def main():
    setup_logging(log_level=logging.INFO)

    parser = argparse.ArgumentParser(description="基于注册表预计算 text embeddings")
    parser.add_argument("--config", type=str, required=True, help="task yaml 路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有的 embedding 文件")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_mix = cfg["data"]["data_mix"]
    model_cfg = cfg.get("model", {})
    overwrite = args.overwrite

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logger.info("分布式模式: world_size=%d", world_size)

    # --- 模型参数 ---
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    context_len = int(model_cfg.get("tokenizer_max_len", DEFAULT_CONTEXT_LEN))
    enc_id = _model_id_to_enc_id(model_id)

    # --- 收集所有 prompts ---
    if (not is_distributed) or rank == 0:
        logger.info("从注册表解析 data_mix='%s'...", data_mix)
    all_dataset_prompts = _collect_all_prompts(data_mix)
    total_datasets = len(all_dataset_prompts)
    total_prompts = sum(len(p) for _, _, p in all_dataset_prompts)
    if (not is_distributed) or rank == 0:
        logger.info("共 %d 个子数据集, %d 个去重 prompt", total_datasets, total_prompts)

    # --- 构建全局 prompt 列表（去重 + 记录每个 prompt 要写入哪些 dataset_dir）---
    prompt_to_dirs: dict[str, list[str]] = {}
    for dataset_dir, _robot_type, prompts in all_dataset_prompts:
        for prompt in prompts:
            prompt_to_dirs.setdefault(prompt, []).append(dataset_dir)
    unique_prompts = list(prompt_to_dirs.keys())
    if (not is_distributed) or rank == 0:
        logger.info("全局去重后: %d 个唯一 prompt", len(unique_prompts))

    # --- 加载 T5 编码器 ---
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16

    logger.info("加载 text encoder: model_id=%s, context_len=%d, device=%s", model_id, context_len, device)
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder",
        torch_dtype=torch_dtype, device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path, seq_len=context_len, clean="whitespace",
    )

    # --- 分片 ---
    my_prompts = unique_prompts[rank::world_size] if is_distributed else unique_prompts

    # --- 跳过已有 ---
    if not overwrite:
        prompts_to_encode: list[str] = []
        skipped = 0
        for prompt in my_prompts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"
            all_exist = all(
                (Path(d) / "text_embeds" / filename).exists()
                for d in prompt_to_dirs[prompt]
            )
            if all_exist:
                skipped += 1
            else:
                prompts_to_encode.append(prompt)
        if (not is_distributed) or rank == 0:
            logger.info("跳过已有: %d, 待编码: %d", skipped, len(prompts_to_encode))
        my_prompts = prompts_to_encode

    # --- 编码 + 存储 ---
    new_count = 0
    with tqdm(
        total=len(my_prompts),
        desc=f"编码 (rank {rank}/{world_size})" if is_distributed else "编码",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(my_prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = my_prompts[start:start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {"context": context_i, "mask": mask_i}
                    filename = f"{hashed}.t5_len{context_len}.{enc_id}.pt"

                    for dataset_dir in prompt_to_dirs[prompt]:
                        output_path = Path(dataset_dir) / "text_embeds" / filename
                        if output_path.exists() and not overwrite:
                            continue
                        _atomic_torch_save(payload, output_path)
                        new_count += 1

                pbar.update(len(batch_prompts))

    logger.info("Rank %d 完成: 新写入 %d 个文件", rank, new_count)

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
