"""视频处理 transform 积木，三层结构。

第一层（单视角）: PerCameraResize, PerCameraCenterCrop
  输入/输出: dict[str, ndarray] (T, H, W, 3)

第二层（拼接）: ConcatCameras
  输入: dict[str, ndarray] → 输出: (T, C, H, W) uint8 Tensor

第三层（整体）: Resize, CenterCrop, Pad, Normalize
  输入/输出: Tensor
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F_nn
import torchvision.transforms.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 第一层：单视角操作  dict[str, ndarray] → dict[str, ndarray]
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PerCameraResize:
    """各相机独立 resize。

    sizes: {camera_key: (H, W)}，未列出的相机保持原尺寸。
    """
    sizes: dict[str, tuple[int, int]]

    def __call__(self, frames_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {}
        for cam, arr in frames_dict.items():
            if cam in self.sizes:
                h, w = self.sizes[cam]
                t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)
                t = F.resize(t, size=[h, w], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
                result[cam] = t.permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
            else:
                result[cam] = arr
        return result


@dataclass
class PerCameraCenterCrop:
    """各相机独立中心裁剪。

    sizes: {camera_key: (H, W)}，未列出的相机保持原尺寸。
    """
    sizes: dict[str, tuple[int, int]]

    def __call__(self, frames_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {}
        for cam, arr in frames_dict.items():
            if cam in self.sizes:
                target_h, target_w = self.sizes[cam]
                _, h, w, _ = arr.shape
                top = (h - target_h) // 2
                left = (w - target_w) // 2
                result[cam] = arr[:, top:top + target_h, left:left + target_w, :]
            else:
                result[cam] = arr
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 第二层：拼接  dict[str, ndarray] → Tensor (T, C, H, W) uint8
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConcatCameras:
    """多相机拼接。

    camera_keys: 按顺序拼接的相机列表。
    mode: "horizontal" / "vertical" / "top_bottom" / "grid_2x2"
    pad_to: 拼接前将各相机 pad 到此尺寸 (H, W)，右下方补 0。
            None 表示不 pad。
    """
    camera_keys: list[str]
    mode: str = "horizontal"
    pad_to: Optional[tuple[int, int]] = None

    def __call__(self, frames_dict: dict[str, np.ndarray]) -> torch.Tensor:
        videos = []
        for cam in self.camera_keys:
            if cam not in frames_dict:
                raise KeyError(f"Camera `{cam}` missing (available: {list(frames_dict.keys())})")
            arr = frames_dict[cam]  # (T, H, W, 3)
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
            videos.append(t)

        if self.pad_to is not None:
            pad_h, pad_w = self.pad_to
            padded = []
            for v in videos:
                _, _, h, w = v.shape
                dh = max(0, pad_h - h)
                dw = max(0, pad_w - w)
                if dh > 0 or dw > 0:
                    v = F_nn.pad(v, (0, dw, 0, dh), value=0)
                padded.append(v)
            videos = padded

        if len(videos) == 1:
            return videos[0]

        if self.mode == "horizontal":
            return torch.cat(videos, dim=-1)
        if self.mode == "vertical":
            return torch.cat(videos, dim=-2)
        if self.mode == "top_bottom":
            top = videos[0]
            bottom = torch.cat(videos[1:], dim=-1)
            return torch.cat([top, bottom], dim=-2)
        if self.mode == "grid_2x2":
            if len(videos) != 4:
                raise ValueError(f"grid_2x2 requires 4 cameras, got {len(videos)}")
            top = torch.cat([videos[0], videos[1]], dim=-1)
            bottom = torch.cat([videos[2], videos[3]], dim=-1)
            return torch.cat([top, bottom], dim=-2)

        raise ValueError(f"Unknown concat mode: {self.mode!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# 第三层：整体操作  Tensor → Tensor
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Resize:
    """resize 最短边保持比例。输入/输出: (T, C, H, W)。"""
    height: int
    width: int

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h_in, w_in = video.shape
        scale = max(self.height / h_in, self.width / w_in)
        new_h = int(round(h_in * scale))
        new_w = int(round(w_in * scale))
        if video.dtype == torch.uint8:
            video = video.to(torch.float32)
        return F.resize(video, size=[new_h, new_w],
                        interpolation=F.InterpolationMode.BICUBIC, antialias=True)


@dataclass
class CenterCrop:
    """中心裁剪。输入/输出: (T, C, H, W)。"""
    height: int
    width: int

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h, w = video.shape
        top = (h - self.height) // 2
        left = (w - self.width) // 2
        return video[:, :, top:top + self.height, left:left + self.width]


@dataclass
class Pad:
    """pad 到目标尺寸（右下方填充）。输入/输出: (T, C, H, W)。"""
    height: int
    width: int
    fill: int = 0

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h, w = video.shape
        dh = max(0, self.height - h)
        dw = max(0, self.width - w)
        if dh > 0 or dw > 0:
            video = F_nn.pad(video, (0, dw, 0, dh), value=self.fill)
        return video


@dataclass
class Normalize:
    """uint8 [0,255] → float32 [-1,1]，permute 为 (C, T, H, W)。始终作为最后一步。"""

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.dtype == torch.uint8:
            video = video.to(torch.float32) / 255.0
        elif video.max() > 1.0:
            video = video / 255.0
        video = video * 2.0 - 1.0
        return video.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
