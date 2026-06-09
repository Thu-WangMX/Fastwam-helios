"""Cond-branch latent corruption for IDM teacher-forcing training.

Used by `FastWAMMemoryIDM._training_loss_core` to corrupt the gt video latent
that feeds the action cond branch, while leaving the noisy_pred branch, the
video loss target, and history latents untouched.

Two modes are supported (Helios-style, but applied only to the action cond
input):

- ``"noise"``    : ``x' = (1 - sigma) * x + sigma * eps`` with sigma sampled
                   per sample from ``U(0, sigma_max)``.
- ``"downsample"``: latent-space spatial low-pass, ``x' = upsample(downsample(x, r))``
                   with ratio ``r`` sampled per batch from ``U(ratio_min, 1.0)``.

Both modes respect ``pass_through_prob``: with that probability the input is
returned unchanged so the model also sees the un-corrupted distribution.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


_SUPPORTED_MODES = ("noise", "downsample")


@torch.no_grad()
def corrupt_cond_latents(
    z: torch.Tensor,
    mode: Optional[str],
    sigma_max: float = 0.5,
    ratio_min: float = 0.3,
    pass_through_prob: float = 0.0,
) -> torch.Tensor:
    """Return a corrupted copy of ``z`` for the action cond branch.

    ``z`` is expected to have shape ``[B, C, T, H, W]``. The op is purely on
    the latent tensor; no autograd is recorded (the cond branch input is gt
    and never carries gradients).
    """
    if mode is None or mode == "none":
        return z
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unknown cond_latent_corruption mode={mode!r}. "
            f"Supported: {_SUPPORTED_MODES}."
        )
    if z.ndim != 5:
        raise ValueError(
            f"corrupt_cond_latents expects shape [B, C, T, H, W], got {tuple(z.shape)}."
        )

    if pass_through_prob > 0.0:
        u = torch.rand((), device=z.device).item()
        if u < float(pass_through_prob):
            return z

    if mode == "noise":
        if not (0.0 < float(sigma_max) <= 1.0):
            raise ValueError(f"sigma_max must be in (0, 1], got {sigma_max}.")
        batch_size = z.shape[0]
        sigma = torch.rand(
            (batch_size, 1, 1, 1, 1),
            device=z.device,
            dtype=z.dtype,
        ).mul_(float(sigma_max))
        eps = torch.randn_like(z)
        return (1.0 - sigma) * z + sigma * eps

    # mode == "downsample"
    if not (0.0 < float(ratio_min) <= 1.0):
        raise ValueError(f"ratio_min must be in (0, 1], got {ratio_min}.")
    rmin = float(ratio_min)
    if rmin >= 1.0:
        return z
    ratio = rmin + (1.0 - rmin) * torch.rand((), device=z.device).item()
    if ratio >= 1.0:
        return z

    batch_size, num_channels, num_frames, height, width = z.shape
    h_small = max(1, int(round(height * ratio)))
    w_small = max(1, int(round(width * ratio)))
    if h_small == height and w_small == width:
        return z

    z_flat = z.permute(0, 2, 1, 3, 4).reshape(
        batch_size * num_frames, num_channels, height, width
    )
    # antialias requires float32 on some backends; cast & restore.
    orig_dtype = z_flat.dtype
    z_flat_fp = z_flat.float()
    z_small = F.interpolate(
        z_flat_fp,
        size=(h_small, w_small),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    z_back = F.interpolate(
        z_small,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    z_back = z_back.to(orig_dtype)
    return z_back.reshape(
        batch_size, num_frames, num_channels, height, width
    ).permute(0, 2, 1, 3, 4).contiguous()
