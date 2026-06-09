from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .embodiment_tags import EmbodimentTag
from .transform.base import ComposedModalityTransform, ModalityTransform
from .transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from .transform.video import CenterCrop, ConcatCameras, Normalize, Resize


class ModalityConfig(BaseModel):
    delta_indices: list[int]
    modality_keys: list[str]


class BaseDataConfig(ABC):
    video_keys: list[str]
    state_keys: list[str]
    action_keys: list[str]
    language_keys: list[str]
    observation_indices: list[int]
    action_indices: list[int]

    video_concat_mode: str = "horizontal"
    lerobot_version: str = "v2.0"
    video_size: tuple[int, int] = (256, 256)
    action_mode: str = "abs"  # 对原始 action 做变换 "abs" / "delta" / "rel"

    prompt_embodiment: str = "A robot"
    prompt_views: str = "observed from a camera"
    prompt_control: str = "controlled by motor commands"

    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass

    def _camera_keys_stripped(self) -> list[str]:
        """video_keys 去掉 'video.' 前缀，与 lerobot 的 camera_key 对齐。"""
        return [k.removeprefix("video.") for k in self.video_keys]

    def video_transforms(self) -> list:
        """返回 latent 生成时的视频处理流水线。
        dict[str, ndarray] → ConcatCameras → Resize → CenterCrop → Normalize → (C,T,H,W) float32
        """
        return [
            ConcatCameras(
                camera_keys=self._camera_keys_stripped(),
                mode=self.video_concat_mode,
            ),
            Resize(height=self.video_size[0], width=self.video_size[1]),
            CenterCrop(height=self.video_size[0], width=self.video_size[1]),
            Normalize(),
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# RoboCasa365 (Franka, 3 cameras, robosuite)
# 来源: /mnt/nas-9/.../CloseBlenderLid/meta/modality.json
# ═══════════════════════════════════════════════════════════════════════════════


class Robocasa365DataConfig(BaseDataConfig):
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.robot0_eye_in_hand",
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
    ]
    # observation.state, 总维度 16
    state_keys = [
        "state.base_position",                    # [0:3]
        "state.base_rotation",                    # [3:7] 四元数
        "state.end_effector_position_relative",   # [7:10]
        "state.end_effector_rotation_relative",   # [10:14] 四元数
        "state.gripper_qpos",                     # [14:16]
    ]
    # action, 总维度 12
    action_keys = [
        "action.base_motion",            # [0:4] delta
        "action.control_mode",           # [4:5] absolute
        "action.end_effector_position",  # [5:8] delta
        "action.end_effector_rotation",  # [8:11] delta
        "action.gripper_close",          # [11:12] binary
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(32))

    video_concat_mode = "horizontal"
    video_size = (256, 768)  # 3 × 256 水平拼接

    prompt_embodiment = "A Franka robot arm"
    prompt_views = "observed from 3 horizontally-concatenated cameras (eye-in-hand, left agentview, right agentview)"
    prompt_control = "controlled by delta end-effector position and rotation"

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys + self.state_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.base_motion": "min_max",
                    "action.end_effector_position": "min_max",
                    "action.end_effector_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.base_position": "min_max",
                    "state.base_rotation": "min_max",
                    "state.end_effector_position_relative": "min_max",
                    "state.end_effector_rotation_relative": "min_max",
                    "state.gripper_qpos": "min_max",
                },
            ),
        ])

    def video_transforms(self) -> list:
        return [
            ConcatCameras(camera_keys=self._camera_keys_stripped(), mode="horizontal"),
            Normalize(),
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 注册表
# ═══════════════════════════════════════════════════════════════════════════════


ROBOT_TYPE_CONFIG_MAP: dict[str, BaseDataConfig] = {
    "robocasa365": Robocasa365DataConfig(),
}
