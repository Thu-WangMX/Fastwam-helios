import functools
import random
from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ..schema import DatasetMetadata, RotationType, StateActionMetadata
from .base import InvertibleModalityTransform, ModalityTransform


class RotationTransform:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        import pytorch3d.transforms as pt

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary", "scale"]

    def __init__(self, mode: str, statistics: dict, binary_threshold: float = 0.5):
        self.mode = mode
        self.statistics = statistics
        self.binary_threshold = binary_threshold
        for key, value in self.statistics.items():
            if not key.startswith("_"):
                self.statistics[key] = torch.tensor(value)

        self._mask = None
        if mode == "q99":
            self._mask = self.statistics["q01"] != self.statistics["q99"]
        elif mode == "mean_std":
            self._mask = self.statistics["std"] != 0
        elif mode == "min_max":
            self._mask = self.statistics["min"] != self.statistics["max"]
        elif mode == "scale":
            abs_max = torch.max(torch.abs(self.statistics["min"]), torch.abs(self.statistics["max"]))
            self._mask = abs_max != 0
            self.statistics["_abs_max"] = abs_max

        self._masked_stats = {}
        if self._mask is not None:
            for k, v in self.statistics.items():
                if k.startswith("_"):
                    continue
                if v.shape == self._mask.shape:
                    self._masked_stats[k] = v[self._mask]
            if "_abs_max" in self.statistics:
                self._masked_stats["_abs_max"] = self.statistics["_abs_max"][self._mask]
            if mode == "q99":
                self._masked_stats["_range"] = self._masked_stats["q99"] - self._masked_stats["q01"]
            elif mode == "min_max":
                self._masked_stats["_range"] = self._masked_stats["max"] - self._masked_stats["min"]

        self._cached_dtype = None
        self._cached_stats = {}
        self._cached_masked_stats = {}

    def _get_stats(self, dtype: torch.dtype) -> tuple[dict, dict]:
        if self._cached_dtype == dtype:
            return self._cached_stats, self._cached_masked_stats
        self._cached_stats = {k: v.to(dtype) for k, v in self.statistics.items() if not k.startswith("_")}
        if "_abs_max" in self.statistics:
            self._cached_stats["_abs_max"] = self.statistics["_abs_max"].to(dtype)
        self._cached_masked_stats = {k: v.to(dtype) for k, v in self._masked_stats.items()}
        self._cached_dtype = dtype
        return self._cached_stats, self._cached_masked_stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats, mstats = self._get_stats(x.dtype)
        mask = self._mask

        if self.mode == "q99":
            normalized = x.clone()
            normalized[..., mask] = 2 * (x[..., mask] - mstats["q01"]) / mstats["_range"] - 1
            normalized = torch.clamp(normalized, -2.2, 2.2)
        elif self.mode == "mean_std":
            normalized = x.clone()
            normalized[..., mask] = (x[..., mask] - mstats["mean"]) / mstats["std"]
        elif self.mode == "min_max":
            normalized = torch.zeros_like(x)
            normalized[..., mask] = 2 * (x[..., mask] - mstats["min"]) / mstats["_range"] - 1
        elif self.mode == "scale":
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / mstats["_abs_max"]
        elif self.mode == "binary":
            normalized = (x > self.binary_threshold).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")
        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        stats, _ = self._get_stats(x.dtype)
        if self.mode == "q99":
            return (x + 1) / 2 * (stats["q99"] - stats["q01"]) + stats["q01"]
        elif self.mode == "mean_std":
            return x * stats["std"] + stats["mean"]
        elif self.mode == "min_max":
            return (x + 1) / 2 * (stats["max"] - stats["min"]) + stats["min"]
        elif self.mode == "binary":
            return (x > self.binary_threshold).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class StateActionToTensor(InvertibleModalityTransform):
    input_dtypes: dict[str, np.dtype] = Field(default_factory=dict)
    output_dtypes: dict[str, torch.dtype] = Field(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    v[key] = getattr(torch, dtype.split(".")[-1])
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    v[key] = np.dtype(dtype.split(".")[-1])
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(value, np.ndarray), f"Expected np.ndarray, got {type(value)}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(value, torch.Tensor), f"Expected torch.Tensor, got {type(value)}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    apply_to: list[str] = Field(...)
    normalization_modes: dict[str, str] = Field(default_factory=dict)
    target_rotations: dict[str, str] = Field(default_factory=dict)
    normalization_statistics: dict[str, dict] = Field(default_factory=dict)
    binary_threshold: float = Field(default=0.5)
    modality_metadata: dict[str, StateActionMetadata] = Field(default_factory=dict)

    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {"min": [-1, -1, -1, -1, -1, -1], "max": [1, 1, 1, 1, 1, 1]},
        "euler_angles": {"min": [-np.pi, -np.pi, -np.pi], "max": [np.pi, np.pi, np.pi]},
        "quaternion": {"min": [-1, -1, -1, -1], "max": [1, 1, 1, 1]},
        "axis_angle": {"min": [-np.pi, -np.pi, -np.pi], "max": [np.pi, np.pi, np.pi]},
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes", "target_rotations"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            v[modality_key] = config
        return v

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        for key in self.apply_to:
            split_key = key.split(".", 1)
            assert len(split_key) == 2
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(modality_metadata, modality), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        for key in self.normalization_modes:
            split_key = key.split(".", 1)
            assert len(split_key) == 2
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(dataset_statistics, modality), f"{state_key} statistics not found"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[state_key].model_dump()

        for key in self.target_rotations:
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"
            to_rep = RotationType(self.target_rotations[key])
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(
                    from_rep=from_rep.value, to_rep=to_rep.value
                )

        for key in self.normalization_modes:
            if key in self._rotation_transformers:
                if self.modality_metadata[key].absolute:
                    assert self.normalization_modes[key] == "min_max"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                else:
                    raise ValueError(f"Cannot normalize relative rotations: {key}")
            elif not self.modality_metadata[key].continuous and self.normalization_modes[key] != "binary":
                raise ValueError(f"{key} is not continuous, should use `binary` mode")
            else:
                statistics = self.normalization_statistics[key]
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key],
                statistics=statistics,
                binary_threshold=self.binary_threshold,
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            if key not in self._input_dtypes:
                self._input_dtypes[key] = data[key].dtype
            state = data[key]
            rot = self._rotation_transformers.get(key)
            if rot is not None:
                state = rot.forward(state)
            norm = self._normalizers.get(key)
            if norm is not None:
                state = norm.forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
            data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            data[key] = torch.cat([torch.sin(state), torch.cos(state)], dim=-1)
        return data
