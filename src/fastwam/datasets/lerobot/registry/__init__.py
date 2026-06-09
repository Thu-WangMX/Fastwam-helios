"""注册表入口模块。

自动从 `examples/*/data_registry/data_config.py` 发现并合并外部注册。

使用方式::

    from fastwam.datasets.lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        DATASET_NAMED_MIXTURES,
    )
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from .data_config import ROBOT_TYPE_CONFIG_MAP as _BASE_CONFIG_MAP
from .embodiment_tags import EmbodimentTag
from .mixtures import DATASET_NAMED_MIXTURES as _BASE_MIXTURES

logger = logging.getLogger(__name__)

ROBOT_TYPE_CONFIG_MAP: dict = dict(_BASE_CONFIG_MAP)
DATASET_NAMED_MIXTURES: dict = dict(_BASE_MIXTURES)

_DISCOVERED = False


def _find_registry_dirs() -> list[Path]:
    """扫描 examples/*/data_registry/ 目录。"""
    repo_root = Path(__file__).resolve().parents[5]
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        return []
    dirs: list[Path] = []
    for bench_dir in sorted(examples_dir.iterdir()):
        registry_dir = bench_dir / "data_registry"
        if registry_dir.is_dir():
            dirs.append(registry_dir)
        train_registry = bench_dir / "train_files" / "data_registry"
        if train_registry.is_dir():
            dirs.append(train_registry)
    return dirs


def _load_module_from_path(module_name: str, file_path: Path):
    """从文件路径动态导入模块。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_and_merge() -> None:
    """扫描并合并外部注册表到全局注册表。"""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    for registry_dir in _find_registry_dirs():
        bench_name = registry_dir.parent.name
        prefix = f"_fastwam_data_registry_{bench_name}"

        cfg_file = registry_dir / "data_config.py"
        if cfg_file.is_file():
            mod = _load_module_from_path(f"{prefix}.data_config", cfg_file)
            if mod:
                if hasattr(mod, "ROBOT_TYPE_CONFIG_MAP"):
                    ROBOT_TYPE_CONFIG_MAP.update(mod.ROBOT_TYPE_CONFIG_MAP)
                    logger.debug(
                        "从 %s 加载了 DataConfig: %s",
                        bench_name, list(mod.ROBOT_TYPE_CONFIG_MAP.keys()),
                    )
                if hasattr(mod, "DATASET_NAMED_MIXTURES"):
                    DATASET_NAMED_MIXTURES.update(mod.DATASET_NAMED_MIXTURES)
                    logger.debug(
                        "从 %s 加载了 mixtures: %s",
                        bench_name, list(mod.DATASET_NAMED_MIXTURES.keys()),
                    )

        mixtures_file = registry_dir / "mixtures.py"
        if mixtures_file.is_file():
            mod = _load_module_from_path(f"{prefix}.mixtures", mixtures_file)
            if mod and hasattr(mod, "DATASET_NAMED_MIXTURES"):
                DATASET_NAMED_MIXTURES.update(mod.DATASET_NAMED_MIXTURES)
                logger.debug(
                    "从 %s/mixtures.py 加载了 mixtures: %s",
                    bench_name, list(mod.DATASET_NAMED_MIXTURES.keys()),
                )


discover_and_merge()
