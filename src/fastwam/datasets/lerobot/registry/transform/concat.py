from typing import Optional

import numpy as np
import torch
from pydantic import Field

from ..schema import DatasetMetadata, StateActionMetadata
from .base import InvertibleModalityTransform


class ConcatTransform(InvertibleModalityTransform):
    apply_to: list[str] = Field(default_factory=list)

    video_concat_order: list[str] = Field(...)
    state_concat_order: Optional[list[str]] = Field(default=None)
    action_concat_order: Optional[list[str]] = Field(default=None)
    action_dims: dict[str, int] = Field(default_factory=dict)
    state_dims: dict[str, int] = Field(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "video_concat_order", "state_concat_order", "action_concat_order"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    def apply(self, data: dict) -> dict:
        data_keys = frozenset(data.keys())
        if not hasattr(self, "_grouped_keys_cache") or self._grouped_keys_last != data_keys:
            grouped_keys = {}
            for key in data.keys():
                try:
                    modality, _ = key.split(".")
                except Exception:
                    modality = "language" if "annotation" in key else "others"
                grouped_keys.setdefault(modality, []).append(key)
            self._grouped_keys_cache = grouped_keys
            self._grouped_keys_last = data_keys
        grouped_keys = self._grouped_keys_cache

        if "video" in grouped_keys:
            unsqueezed_videos = []
            for video_key in self.video_concat_order:
                video_data = data.pop(video_key)
                unsqueezed_videos.append(np.expand_dims(video_data, axis=-4))
            data["video"] = np.concatenate(unsqueezed_videos, axis=-4)

        if "state" in grouped_keys and self.state_concat_order is not None:
            data["state"] = torch.cat([data.pop(key) for key in self.state_concat_order], dim=-1)

        if "action" in grouped_keys and self.action_concat_order is not None:
            data["action"] = torch.cat([data.pop(key) for key in self.action_concat_order], dim=-1)

        return data

    def unapply(self, data: dict) -> dict:
        if "action" in data and self.action_concat_order is not None:
            start_dim = 0
            action_tensor = data.pop("action")
            for key in self.action_concat_order:
                if key not in self.action_dims:
                    raise ValueError(f"Action dim {key} not found in action_dims.")
                end_dim = start_dim + self.action_dims[key]
                data[key] = action_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        if "state" in data and self.state_concat_order is not None:
            start_dim = 0
            state_tensor = data.pop("state")
            for key in self.state_concat_order:
                end_dim = start_dim + self.state_dims[key]
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

    def get_modality_metadata(self, key: str) -> StateActionMetadata:
        modality, subkey = key.split(".")
        assert self.dataset_metadata is not None
        modality_config = getattr(self.dataset_metadata.modalities, modality)
        assert subkey in modality_config
        return modality_config[subkey]

    def get_state_action_dims(self, key: str) -> int:
        modality_config = self.get_modality_metadata(key)
        shape = modality_config.shape
        assert len(shape) == 1
        return shape[0]

    def is_rotation_key(self, key: str) -> bool:
        return self.get_modality_metadata(key).rotation_type is not None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        if self.action_concat_order is not None:
            for key in self.action_concat_order:
                self.action_dims[key] = self.get_state_action_dims(key)
        if self.state_concat_order is not None:
            for key in self.state_concat_order:
                self.state_dims[key] = self.get_state_action_dims(key)
