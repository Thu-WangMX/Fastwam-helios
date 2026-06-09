from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_serializer

from .embodiment_tags import EmbodimentTag


class RotationType(Enum):
    AXIS_ANGLE = "axis_angle"
    QUATERNION = "quaternion"
    ROTATION_6D = "rotation_6d"
    MATRIX = "matrix"
    EULER_ANGLES_RPY = "euler_angles_rpy"
    EULER_ANGLES_RYP = "euler_angles_ryp"
    EULER_ANGLES_PRY = "euler_angles_pry"
    EULER_ANGLES_PYR = "euler_angles_pyr"
    EULER_ANGLES_YRP = "euler_angles_yrp"
    EULER_ANGLES_YPR = "euler_angles_ypr"


class LeRobotModalityField(BaseModel):
    original_key: Optional[str] = Field(default=None)


class LeRobotStateActionMetadata(LeRobotModalityField):
    start: int
    end: int
    rotation_type: Optional[RotationType] = None
    absolute: bool = True
    dtype: str = "float64"
    range: Optional[tuple[float, float]] = None
    original_key: Optional[str] = None


class LeRobotStateMetadata(LeRobotStateActionMetadata):
    original_key: Optional[str] = Field(default="observation.state")


class LeRobotActionMetadata(LeRobotStateActionMetadata):
    original_key: Optional[str] = Field(default="action")


class LeRobotModalityMetadata(BaseModel):
    state: dict[str, LeRobotStateMetadata]
    action: dict[str, LeRobotActionMetadata]
    video: dict[str, LeRobotModalityField]
    annotation: Optional[dict[str, LeRobotModalityField]] = None

    def get_key_meta(self, key: str) -> LeRobotModalityField:
        split_key = key.split(".")
        modality = split_key[0]
        subkey = ".".join(split_key[1:])
        registry = {
            "state": self.state,
            "action": self.action,
            "video": self.video,
            "annotation": self.annotation,
        }
        store = registry.get(modality)
        if store is None:
            raise ValueError(f"Key: {key}, unexpected modality: {modality}")
        if modality == "annotation":
            assert store is not None, "No annotations in this dataset"
        if subkey not in store:
            raise ValueError(f"Key: {key}, subkey {subkey} not found, available: {list(store.keys())}")
        return store[subkey]


class DatasetStatisticalValues(BaseModel):
    max: list[float] = Field(...)
    min: list[float] = Field(...)
    mean: list[float] = Field(...)
    std: list[float] = Field(...)
    q01: list[float] = Field(...)
    q99: list[float] = Field(...)


class DatasetStatistics(BaseModel):
    state: dict[str, DatasetStatisticalValues]
    action: dict[str, DatasetStatisticalValues]


class VideoMetadata(BaseModel):
    resolution: tuple[int, int]
    channels: int
    fps: float


class StateActionMetadata(BaseModel):
    absolute: bool
    rotation_type: Optional[RotationType] = None
    shape: tuple[int, ...]
    continuous: bool


class DatasetModalities(BaseModel):
    video: dict[str, VideoMetadata]
    state: dict[str, StateActionMetadata]
    action: dict[str, StateActionMetadata]


class DatasetMetadata(BaseModel):
    statistics: DatasetStatistics
    modalities: DatasetModalities
    embodiment_tag: EmbodimentTag
