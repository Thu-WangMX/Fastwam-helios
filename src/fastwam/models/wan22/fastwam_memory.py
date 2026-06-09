"""FastWAM Multi-Term Memory (MTM) variant.

Implements Helios-style multi-term memory on top of FastWAM. The key idea: replace
the single `current_latent` block in fastwam by `history_context = [long(16) | mid(2) | current(1) | pred(2)]`,
where `long / mid / current` are clean history latents (zero-timestep AdaLN, attention sees them as
single-direction guidance), and only `pred` is denoised.

Three new classes live here:
    - `MTMVideoDiT`: subclass of `WanVideoDiT` with three patch convs (short / mid / long),
      shared RoPE frame indices, MTM attention mask builder, and overridden `pre_dit / post_dit`.
    - `MTMMoT`: subclass of `MoT` that threads `layer_idx + mtm_history_seq_len` into
      `_build_expert_attention_io` to apply per-head learnable history-key amplification.
    - `FastWAMMemory`: subclass of `FastWAM` that wires history fields from the dataloader
      sample into `pre_dit`, builds the MTM joint attention mask, and routes inference paths.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .fastwam import FastWAM, _resolve_joint_steps
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .wan_video_dit import (
    WanVideoDiT,
    create_group_causal_attn_mask,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# 3D conv padding / center down-sample helpers (ported from helios).
# -----------------------------------------------------------------------------


def pad_for_3d_conv(x: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
    """Replicate-pad a 5D tensor `(B, C, T, H, W)` so that `T/H/W` are multiples of `kernel_size`.

    Mirrors helios' `pad_for_3d_conv` so that mid / long history latents can be cleanly fed into
    `Conv3d(stride=kernel_size)` without losing trailing voxels. Padding is applied to the right
    end of each spatial-temporal axis to keep the leading index aligned with global frame index 0.
    """
    if x.dim() != 5:
        raise ValueError(f"`pad_for_3d_conv` expects 5D `(B,C,T,H,W)`, got shape {tuple(x.shape)}")
    kt, kh, kw = int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2])
    if kt <= 0 or kh <= 0 or kw <= 0:
        raise ValueError(f"`kernel_size` entries must be positive, got {kernel_size}")
    _, _, t, h, w = x.shape
    pad_t = (-t) % kt
    pad_h = (-h) % kh
    pad_w = (-w) % kw
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x: torch.Tensor, ratio: Tuple[int, int, int]) -> torch.Tensor:
    """Average-pool a real-valued tensor `(T, H, W, D)` along T/H/W by `ratio`.

    Used to downsample dense RoPE frequencies (treated as real `head_dim` channels) onto the coarse
    mid / long token grid so that all segments share the same spatial/temporal coordinate basis.
    """
    if x.dim() != 4:
        raise ValueError(f"`center_down_sample_3d` expects 4D `(T,H,W,D)`, got shape {tuple(x.shape)}")
    rt, rh, rw = int(ratio[0]), int(ratio[1]), int(ratio[2])
    if rt <= 0 or rh <= 0 or rw <= 0:
        raise ValueError(f"`ratio` entries must be positive, got {ratio}")
    if rt == 1 and rh == 1 and rw == 1:
        return x
    # Treat the trailing dim as channel for avg_pool3d. Insert batch dim for the API.
    pooled = F.avg_pool3d(
        x.permute(3, 0, 1, 2).unsqueeze(0).contiguous(),
        kernel_size=(rt, rh, rw),
        stride=(rt, rh, rw),
    )
    return pooled.squeeze(0).permute(1, 2, 3, 0).contiguous()


def _downsample_freqs_cis(
    freqs_cis: torch.Tensor, ratio: Tuple[int, int, int]
) -> torch.Tensor:
    """Down-sample a complex RoPE frequency grid `(T, H, W, head_dim_part)` along T/H/W.

    Real and imaginary parts are pooled independently. The result is no longer unit-modulus, but
    matches helios behavior and applies as a (slightly) attenuated phase rotation in `rope_apply`.
    """
    if not torch.is_complex(freqs_cis):
        raise ValueError("`_downsample_freqs_cis` expects complex-valued freqs.")
    real = center_down_sample_3d(freqs_cis.real.to(torch.float32), ratio)
    imag = center_down_sample_3d(freqs_cis.imag.to(torch.float32), ratio)
    return torch.complex(real, imag).to(freqs_cis.dtype)


# -----------------------------------------------------------------------------
# MTMVideoDiT: WanVideoDiT + multi-term memory (three patches + shared RoPE).
# -----------------------------------------------------------------------------


class MTMVideoDiT(WanVideoDiT):
    """`WanVideoDiT` extended with helios-style multi-term memory.

    Adds:
      - `patch_mid` (kernel `(2,4,4)`) and `patch_long` (`(4,8,8)`) Conv3d for compressed history.
      - Optional per-layer-per-head learnable `history_key_scale_logit` for `is_amplify_history`.
      - `pre_dit` builds a `[long | mid | short | pred]` token sequence with helios-style shared RoPE
        frame indices `[0..15] | [16,17] | [18] | [19, 19+T_pred-1]`, and per-token zero-history-timestep
        AdaLN modulation when `mtm_zero_history_timestep=True`.
      - `post_dit` slices off history tokens and only un-patchifies the pred segment.
      - `build_mtm_self_attn_mask` returns the joint self-attention visibility matrix.

    The `patch_short` path is *not* a separate module: the user-confirmed contract puts the most
    recent clean latent (`current(1)`) and the noisy `pred(T_pred)` on the same `(1,2,2)` patch grid,
    so we reuse `self.patch_embedding` for both segments. Only mid / long need their own conv.
    """

    def __init__(
        self,
        *args: Any,
        multi_term_memory: bool = False,
        mtm_history_sizes: Tuple[int, int, int] = (16, 2, 1),
        mtm_patch_kernel_long: Tuple[int, int, int] = (4, 8, 8),
        mtm_patch_kernel_mid: Tuple[int, int, int] = (2, 4, 4),
        mtm_patch_kernel_current: Tuple[int, int, int] = (1, 2, 2),
        mtm_pred_size: int = 2,
        mtm_amplify_history: bool = False,
        mtm_zero_history_timestep: bool = True,
        **kwargs: Any,
    ) -> None:
        # The base class hard-asserts `fuse_vae_embedding_in_latents=True`; in MTM mode we want
        # to *disable* it at runtime (no fixed first-frame anchor). Force-pass True to satisfy
        # the assertion, then flip the attribute below if the caller actually requested False.
        requested_fuse = kwargs.get("fuse_vae_embedding_in_latents", True)
        if multi_term_memory and not requested_fuse:
            kwargs["fuse_vae_embedding_in_latents"] = True
        super().__init__(*args, **kwargs)

        self.multi_term_memory = bool(multi_term_memory)
        self.mtm_history_sizes = tuple(int(s) for s in mtm_history_sizes)
        self.mtm_patch_kernel_long = tuple(int(k) for k in mtm_patch_kernel_long)
        self.mtm_patch_kernel_mid = tuple(int(k) for k in mtm_patch_kernel_mid)
        self.mtm_patch_kernel_current = tuple(int(k) for k in mtm_patch_kernel_current)
        self.mtm_pred_size = int(mtm_pred_size)
        self.mtm_amplify_history = bool(mtm_amplify_history)
        self.mtm_zero_history_timestep = bool(mtm_zero_history_timestep)

        if not self.multi_term_memory:
            return

        if len(self.mtm_history_sizes) != 3:
            raise ValueError(
                f"`mtm_history_sizes` must be a length-3 tuple `(long, mid, current)`, "
                f"got {self.mtm_history_sizes}"
            )
        long_size, mid_size, curr_size = self.mtm_history_sizes
        if long_size < 0 or mid_size < 0:
            raise ValueError(
                f"`mtm_history_sizes` long/mid must be >= 0, got {self.mtm_history_sizes}"
            )
        if curr_size < 1:
            raise ValueError(
                f"`mtm_history_sizes` current (last element) must be >= 1, got {curr_size}"
            )
        if self.mtm_pred_size <= 0:
            raise ValueError(f"`mtm_pred_size` must be positive, got {self.mtm_pred_size}")
        if not self.seperated_timestep:
            raise ValueError(
                "MTM requires `seperated_timestep=True` so each token can have its own timestep."
            )

        # Validate patch kernels against base patch_size.
        base_ps = tuple(self.patch_size)
        if self.mtm_patch_kernel_current != base_ps:
            raise ValueError(
                f"`mtm_patch_kernel_current` must equal `patch_size` {base_ps}, "
                f"got {self.mtm_patch_kernel_current}. Current segment shares `patch_embedding` with pred."
            )
        for name, kernel in [("long", self.mtm_patch_kernel_long), ("mid", self.mtm_patch_kernel_mid)]:
            if len(kernel) != 3 or any(k <= 0 for k in kernel):
                raise ValueError(f"`mtm_patch_kernel_{name}` must be 3 positive ints, got {kernel}")
            kt, kh, kw = kernel
            if kh != kw:
                raise ValueError(
                    f"`mtm_patch_kernel_{name}` spatial dims must be equal (square), got ({kh}, {kw})"
                )
            if kh % base_ps[1] != 0 or kw % base_ps[2] != 0 or kt % base_ps[0] != 0:
                raise ValueError(
                    f"`mtm_patch_kernel_{name}` {kernel} must be integer multiples of "
                    f"`patch_size` {base_ps}"
                )

        # Precompute temporal/spatial ratios for RoPE downsampling.
        base_t, base_h, _ = base_ps
        self._long_temporal_ratio = self.mtm_patch_kernel_long[0] // base_t
        self._long_spatial_ratio = self.mtm_patch_kernel_long[1] // base_h
        self._mid_temporal_ratio = self.mtm_patch_kernel_mid[0] // base_t
        self._mid_spatial_ratio = self.mtm_patch_kernel_mid[1] // base_h

        # Honor the original request: in MTM mode we never run the I2V first-frame fusion path.
        if multi_term_memory and not requested_fuse:
            self.fuse_vae_embedding_in_latents = False

        # Three-patch embedding: short branch reuses `patch_embedding` (same kernel as pred).
        self.patch_mid = nn.Conv3d(
            self.in_dim, self.hidden_dim,
            kernel_size=self.mtm_patch_kernel_mid, stride=self.mtm_patch_kernel_mid,
        )
        self.patch_long = nn.Conv3d(
            self.in_dim, self.hidden_dim,
            kernel_size=self.mtm_patch_kernel_long, stride=self.mtm_patch_kernel_long,
        )
        self._init_mtm_patches_from_patch_embedding()

        if self.mtm_amplify_history:
            num_layers = len(self.blocks)
            # Match helios: init logit=1 -> scale = 1 + 9 * sigmoid(1) ≈ 7.58.
            self.history_key_scale_logit = nn.Parameter(
                torch.ones(num_layers, self.num_heads)
            )

    def _init_mtm_patches_from_patch_embedding(self) -> None:
        """Copy `patch_embedding` weights into `patch_mid / patch_long` with spatial replication.

        Mirrors helios' `initialize_weight_from_another_conv3d` so that on a constant input the mid /
        long convs initially produce the same per-voxel mean as `patch_embedding`. The bias is copied
        verbatim; weight is repeated along T/H/W and divided by the replication factor.
        """
        weight = self.patch_embedding.weight.detach().clone()  # (D, C, base_t, base_h, base_w)
        bias = self.patch_embedding.bias.detach().clone()
        base_ps = tuple(self.patch_size)
        if weight.shape[2:] != base_ps:
            raise ValueError(
                f"Expected `patch_embedding` kernel `{base_ps}`, got {tuple(weight.shape[2:])}"
            )
        rt_m, rs_m = self._mid_temporal_ratio, self._mid_spatial_ratio
        rt_l, rs_l = self._long_temporal_ratio, self._long_spatial_ratio
        mid_factor = float(rt_m * rs_m * rs_m)
        long_factor = float(rt_l * rs_l * rs_l)
        with torch.no_grad():
            self.patch_mid.weight.copy_(
                repeat(weight, f"d c t h w -> d c (t {rt_m}) (h {rs_m}) (w {rs_m})") / mid_factor
            )
            self.patch_mid.bias.copy_(bias)
            self.patch_long.weight.copy_(
                repeat(weight, f"d c t h w -> d c (t {rt_l}) (h {rs_l}) (w {rs_l})") / long_factor
            )
            self.patch_long.bias.copy_(bias)

    def build_mtm_self_attn_mask(
        self,
        L_long: int,
        L_mid: int,
        L_curr: int,
        L_pred: int,
        tokens_per_frame_pred: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the joint `(total, total)` boolean self-attention mask for `[history | pred]`.

        Following the user-confirmed `pred_only_mode`:
          - history (`long + mid + current`) attends bidirectionally within itself.
          - pred attends to the full history (`pred -> history = True`).
          - pred attends within itself per `video_attention_mask_mode` (`bidirectional` /
            `per_frame_causal` / `first_frame_causal`).
          - history *cannot* see pred (single-direction guidance).
        """
        if min(L_long, L_mid, L_curr, L_pred) < 0:
            raise ValueError(
                f"All segment lengths must be non-negative, got long={L_long}, mid={L_mid}, "
                f"curr={L_curr}, pred={L_pred}"
            )
        L_hist = L_long + L_mid + L_curr
        total = L_hist + L_pred
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        if L_hist > 0:
            mask[:L_hist, :L_hist] = True
            if L_pred > 0:
                mask[L_hist:, :L_hist] = True
        if L_pred > 0:
            mask[L_hist:, L_hist:] = self.build_video_to_video_mask(
                video_seq_len=L_pred,
                video_tokens_per_frame=tokens_per_frame_pred,
                device=device,
            )
        return mask

    def _build_segment_freqs(
        self,
        frame_indices: torch.Tensor,
        h: int,
        w: int,
        spatial_ratio: int,
        temporal_ratio: int,
    ) -> torch.Tensor:
        """Build RoPE freqs for one history/pred segment using a shared `(h, w)` spatial basis.

        Produces freqs at the dense grid `(len(frame_indices), h, w, head_dim/2)`, then average-pools
        by `(temporal_ratio, spatial_ratio, spatial_ratio)` so the result matches the segment's
        post-patch token count exactly.

        Returns a `(L_seg, 1, head_dim/2)` complex tensor ready for `torch.cat` along the seq axis.
        """
        f_freqs_cis, h_freqs_cis, w_freqs_cis = self.freqs
        device = frame_indices.device
        f_part = f_freqs_cis.to(device=device).index_select(0, frame_indices)  # (T_dense, dim_t)
        h_part = h_freqs_cis.to(device=device)[:h]  # (h, dim_h)
        w_part = w_freqs_cis.to(device=device)[:w]  # (w, dim_w)
        t_dense = int(frame_indices.shape[0])
        f_grid = f_part.view(t_dense, 1, 1, -1).expand(t_dense, h, w, -1)
        h_grid = h_part.view(1, h, 1, -1).expand(t_dense, h, w, -1)
        w_grid = w_part.view(1, 1, w, -1).expand(t_dense, h, w, -1)
        dense = torch.cat([f_grid, h_grid, w_grid], dim=-1)  # (T_dense, h, w, head_dim/2) complex
        if temporal_ratio > 1 or spatial_ratio > 1:
            # Pad before pool so trailing voxels survive (mirrors `pad_for_3d_conv`).
            dense_real = dense.real.to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
            dense_imag = dense.imag.to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
            pad_t = (-t_dense) % temporal_ratio
            pad_h = (-h) % spatial_ratio
            pad_w = (-w) % spatial_ratio
            if pad_t or pad_h or pad_w:
                dense_real = F.pad(dense_real, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
                dense_imag = F.pad(dense_imag, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
            kernel = (temporal_ratio, spatial_ratio, spatial_ratio)
            real_pool = F.avg_pool3d(dense_real, kernel_size=kernel, stride=kernel)
            imag_pool = F.avg_pool3d(dense_imag, kernel_size=kernel, stride=kernel)
            real_pool = real_pool.squeeze(0).permute(1, 2, 3, 0).contiguous()
            imag_pool = imag_pool.squeeze(0).permute(1, 2, 3, 0).contiguous()
            dense = torch.complex(real_pool, imag_pool).to(f_freqs_cis.dtype)
        seg_t, seg_h, seg_w, _ = dense.shape
        return dense.reshape(seg_t * seg_h * seg_w, 1, -1)

    def _validate_mtm_inputs(
        self,
        x: torch.Tensor,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
    ) -> None:
        if x.dim() != 5 or history_long.dim() != 5 or history_mid.dim() != 5 or history_short.dim() != 5:
            raise ValueError(
                "All MTM tensors must be 5D `(B,C,T,H,W)`; got "
                f"x={tuple(x.shape)}, long={tuple(history_long.shape)}, "
                f"mid={tuple(history_mid.shape)}, short={tuple(history_short.shape)}"
            )
        if x.shape[2] != self.mtm_pred_size:
            raise ValueError(
                f"`x` must have T={self.mtm_pred_size} (pred), got T={x.shape[2]}"
            )
        long_size, mid_size, curr_size = self.mtm_history_sizes
        if history_long.shape[2] != long_size:
            raise ValueError(f"`history_long` T must be {long_size}, got {history_long.shape[2]}")
        if history_mid.shape[2] != mid_size:
            raise ValueError(f"`history_mid` T must be {mid_size}, got {history_mid.shape[2]}")
        if history_short.shape[2] != curr_size:
            raise ValueError(
                f"`history_short` T must be {curr_size}, got {history_short.shape[2]}"
            )
        ref = (x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        for name, tensor in (
            ("history_long", history_long),
            ("history_mid", history_mid),
            ("history_short", history_short),
        ):
            if (tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4]) != ref:
                raise ValueError(
                    f"`{name}` must share `(B, C, H, W)` with `x` `{ref}`, "
                    f"got {(tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4])}"
                )

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Fall back to base pre_dit when MTM is disabled or no history tensors provided.
        no_history = history_long is None and history_mid is None and history_short is None
        if (not self.multi_term_memory) or no_history:
            return super().pre_dit(
                x=x,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                control_camera_latents_input=control_camera_latents_input,
            )

        if history_long is None or history_mid is None or history_short is None:
            raise ValueError(
                "MTM mode requires all three of `history_long`, `history_mid`, `history_short`."
            )
        if fuse_vae_embedding_in_latents:
            raise ValueError(
                "MTM mode is incompatible with `fuse_vae_embedding_in_latents=True`."
            )
        if control_camera_latents_input is not None:
            raise NotImplementedError("MTM mode does not support `control_camera_latents_input` yet.")

        # Reuse base validation for `(x, timestep, context, context_mask, action)` shape contracts.
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        self._validate_mtm_inputs(x, history_long, history_mid, history_short)

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame_pred = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        # ---- 1. Patchify each segment via its dedicated conv. -----------------
        x_pred = self.patchify(x)  # (B, D, T_pred, h, w)
        x_short = self.patch_embedding(history_short)  # (B, D, S, h, w)

        f_pred, h, w = x_pred.shape[2], x_pred.shape[3], x_pred.shape[4]
        f_short, h_short, w_short = x_short.shape[2], x_short.shape[3], x_short.shape[4]
        if (h_short, w_short) != (h, w):
            raise ValueError(
                f"`patch_short` spatial shape ({h_short},{w_short}) must match pred ({h},{w})."
            )

        long_size, mid_size, curr_size = self.mtm_history_sizes
        device = x.device
        segments_tokens: list[torch.Tensor] = []
        segments_freqs: list[torch.Tensor] = []

        # Long segment (skipped when long_size == 0).
        if long_size > 0:
            x_long_in = pad_for_3d_conv(history_long, self.mtm_patch_kernel_long)
            x_long = self.patch_long(x_long_in)
            f_long, h_long, w_long = x_long.shape[2], x_long.shape[3], x_long.shape[4]
            L_long = f_long * h_long * w_long
            segments_tokens.append(rearrange(x_long, "b d t h w -> b (t h w) d"))
            idx_long = torch.arange(0, long_size, device=device)
            freqs_long = self._build_segment_freqs(
                frame_indices=idx_long, h=h, w=w,
                spatial_ratio=self._long_spatial_ratio,
                temporal_ratio=self._long_temporal_ratio,
            )
            if freqs_long.shape[0] != L_long:
                raise RuntimeError(
                    f"long freqs len {freqs_long.shape[0]} != L_long {L_long} (downsample mismatch)"
                )
            segments_freqs.append(freqs_long)
        else:
            L_long = 0

        # Mid segment (skipped when mid_size == 0).
        if mid_size > 0:
            x_mid_in = pad_for_3d_conv(history_mid, self.mtm_patch_kernel_mid)
            x_mid = self.patch_mid(x_mid_in)
            f_mid, h_mid, w_mid = x_mid.shape[2], x_mid.shape[3], x_mid.shape[4]
            L_mid = f_mid * h_mid * w_mid
            segments_tokens.append(rearrange(x_mid, "b d t h w -> b (t h w) d"))
            idx_mid = torch.arange(long_size, long_size + mid_size, device=device)
            freqs_mid = self._build_segment_freqs(
                frame_indices=idx_mid, h=h, w=w,
                spatial_ratio=self._mid_spatial_ratio,
                temporal_ratio=self._mid_temporal_ratio,
            )
            if freqs_mid.shape[0] != L_mid:
                raise RuntimeError(
                    f"mid freqs len {freqs_mid.shape[0]} != L_mid {L_mid} (downsample mismatch)"
                )
            segments_freqs.append(freqs_mid)
        else:
            L_mid = 0

        # Short (current) segment — always present (curr_size >= 1).
        L_curr = f_short * h_short * w_short
        segments_tokens.append(rearrange(x_short, "b d t h w -> b (t h w) d"))
        idx_curr = torch.arange(
            long_size + mid_size, long_size + mid_size + curr_size, device=device
        )
        freqs_short = self._build_segment_freqs(
            frame_indices=idx_curr, h=h, w=w, spatial_ratio=1, temporal_ratio=1
        )
        segments_freqs.append(freqs_short)

        # Pred segment — always present.
        L_pred = f_pred * h * w
        segments_tokens.append(rearrange(x_pred, "b d t h w -> b (t h w) d"))
        pred_start = long_size + mid_size + curr_size
        idx_pred = torch.arange(pred_start, pred_start + f_pred, device=device)
        freqs_pred = self._build_segment_freqs(
            frame_indices=idx_pred, h=h, w=w, spatial_ratio=1, temporal_ratio=1
        )
        segments_freqs.append(freqs_pred)

        L_hist = L_long + L_mid + L_curr
        total_seq = L_hist + L_pred
        tokens = torch.cat(segments_tokens, dim=1).contiguous()
        freqs = torch.cat(segments_freqs, dim=0).to(device)

        # ---- 3. Per-token timestep with zero-history-timestep. ----------------
        token_timesteps = torch.empty(
            (batch_size, total_seq), dtype=timestep.dtype, device=timestep.device
        )
        if self.mtm_zero_history_timestep:
            if L_hist > 0:
                token_timesteps[:, :L_hist] = 0
        else:
            token_timesteps[:, :L_hist] = timestep.view(batch_size, 1)
        token_timesteps[:, L_hist:] = timestep.view(batch_size, 1)
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        ).reshape(batch_size, total_seq, self.hidden_dim)
        t_mod = self.time_projection(t_emb).unflatten(2, (6, self.hidden_dim))
        t_for_head = t_emb[:, L_hist:, :].contiguous()

        # ---- 4. Text / action context + per-token visibility mask. ------------
        context = self.text_embedding(context)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            num_temporal_groups = self.mtm_pred_size - 1
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned MTM requires `mtm_pred_size >= 2`, "
                    f"got {self.mtm_pred_size}."
                )
            if action.shape[1] % num_temporal_groups != 0:
                raise ValueError(
                    f"`action` length {action.shape[1]} must be divisible by `pred-1`={num_temporal_groups}."
                )
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)
            action_pos = sinusoidal_embedding_1d(
                self.hidden_dim,
                torch.arange(action_len, device=action_emb.device),
            )
            action_emb = action_emb + action_pos.unsqueeze(0)
            context = torch.cat([context, action_emb], dim=1)
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame_pred,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(device=context.device)
            final_context_mask = torch.zeros(
                (batch_size, total_seq, context.shape[1]), dtype=torch.bool, device=context.device
            )
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(
                -1, total_seq, -1
            )
            # Action key visible only to pred frames `[1:]` (skip the first pred frame, like base).
            pred_action_start = L_hist + tokens_per_frame_pred
            final_context_mask[:, pred_action_start:total_seq, context_len:] = action_group_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            final_context_mask = final_context_mask
        else:
            if self.action_conditioned and action is None:
                raise ValueError(
                    "Action-conditioned MTM requires non-null `action`. "
                    "Pure-text fallback (`f==1`) is not supported in MTM mode."
                )
            final_context_mask = context_mask.unsqueeze(1).expand(-1, total_seq, -1)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t_for_head,
            "t_mod": t_mod,
            "context": context,
            "context_mask": final_context_mask,
            "meta": {
                "grid_size": (f_pred, h, w),  # kept for back-compat (post_dit fallback)
                "tokens_per_frame": tokens_per_frame_pred,
                "batch_size": batch_size,
                "mtm": {
                    "L_long": L_long,
                    "L_mid": L_mid,
                    "L_curr": L_curr,
                    "L_pred": L_pred,
                    "L_hist": L_hist,
                    "tokens_per_frame_pred": tokens_per_frame_pred,
                    "grid_size_pred": (f_pred, h, w),
                },
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        mtm_meta = pre_state["meta"].get("mtm")
        if mtm_meta is None:
            return super().post_dit(x_tokens, pre_state)
        L_hist = int(mtm_meta["L_hist"])
        f, h, w = mtm_meta["grid_size_pred"]
        x_tokens = x_tokens[:, L_hist:, :].contiguous()
        x = self.head(x_tokens, pre_state["t"])
        return self.unpatchify(x, (f, h, w))


# -----------------------------------------------------------------------------
# MTMMoT: MoT + per-layer-per-head history-key amplification.
# -----------------------------------------------------------------------------


class MTMMoT(MoT):
    """MoT subclass that threads `layer_idx + mtm_history_seq_len` into Q/K/V construction.

    The only behavioral change vs `MoT` is in `_build_expert_attention_io`: when the corresponding
    expert is an `MTMVideoDiT` with `mtm_amplify_history=True`, the *history* segment of K is
    multiplied per-head by `1 + 9 * sigmoid(history_key_scale_logit[layer_idx])` (range `[1, 10]`).
    All forward passes accept an optional `mtm_history_seq_lens: Dict[str, int]` kwarg that maps
    each expert name to its `L_hist` token count.
    """

    def _build_expert_attention_io(
        self,
        expert: nn.Module,
        block: nn.Module,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        layer_idx: int = 0,
        mtm_history_seq_len: int = 0,
    ):
        q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gc = (
            super()._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=freqs,
                t_mod=t_mod,
            )
        )
        if (
            mtm_history_seq_len > 0
            and getattr(expert, "multi_term_memory", False)
            and getattr(expert, "mtm_amplify_history", False)
            and hasattr(expert, "history_key_scale_logit")
        ):
            logit = expert.history_key_scale_logit[layer_idx]  # (H,)
            scale = (1.0 + 9.0 * torch.sigmoid(logit)).to(k.dtype).to(k.device)
            B, S, D = k.shape
            H = block.num_heads
            Dh = D // H
            k = k.view(B, S, H, Dh).contiguous()
            scaled_hist = k[:, :mtm_history_seq_len] * scale.view(1, 1, H, 1)
            k = torch.cat(
                [scaled_hist, k[:, mtm_history_seq_len:]], dim=1
            ).reshape(B, S, D)
        return (
            q,
            k,
            v,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gc,
        )

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        mtm_history_seq_lens: Optional[Dict[str, int]] = None,
    ):
        if mtm_history_seq_lens is None:
            mtm_history_seq_lens = {}

        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks: list[torch.Tensor] = []
            k_chunks: list[torch.Tensor] = []
            v_chunks: list[torch.Tensor] = []
            cached: Dict[str, dict] = {}
            seq_lens: list[int] = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gc,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                    layer_idx=layer_idx,
                    mtm_history_seq_len=int(mtm_history_seq_lens.get(name, 0)),
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gc,
                }

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )
                tokens_all[name] = updated_tokens
                start = end

        return tokens_all

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        mtm_history_seq_lens: Optional[Dict[str, int]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        video_hist_len = int((mtm_history_seq_lens or {}).get("video", 0))
        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gc,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
                layer_idx=layer_idx,
                mtm_history_seq_len=video_hist_len,
            )
            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v, attention_mask=video_attention_mask
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gc,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        mtm_history_seq_lens: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        action_hist_len = int((mtm_history_seq_lens or {}).get("action", 0))

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gc,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
                layer_idx=layer_idx,
                mtm_history_seq_len=action_hist_len,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gc,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x


# -----------------------------------------------------------------------------
# FastWAMMemory: top-level orchestrator.
# -----------------------------------------------------------------------------


class FastWAMMemory(FastWAM):
    """FastWAM variant with multi-term memory threaded through training & inference."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        multi_term_memory: bool = False,
        mtm_history_sizes: Tuple[int, int, int] = (16, 2, 1),
        mtm_patch_kernel_long: Tuple[int, int, int] = (4, 8, 8),
        mtm_patch_kernel_mid: Tuple[int, int, int] = (2, 4, 4),
        mtm_patch_kernel_current: Tuple[int, int, int] = (1, 2, 2),
        mtm_pred_size: int = 2,
        mtm_amplify_history: bool = False,
        mtm_zero_history_timestep: bool = True,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        self.multi_term_memory = bool(multi_term_memory)
        self.mtm_history_sizes = tuple(int(s) for s in mtm_history_sizes)
        self.mtm_pred_size = int(mtm_pred_size)
        self.mtm_amplify_history = bool(mtm_amplify_history)
        self.mtm_zero_history_timestep = bool(mtm_zero_history_timestep)

        if self.multi_term_memory:
            if not isinstance(self.video_expert, MTMVideoDiT):
                raise ValueError(
                    "`multi_term_memory=True` requires `video_expert` to be an MTMVideoDiT instance."
                )
            if not isinstance(self.mot, MTMMoT):
                raise ValueError(
                    "`multi_term_memory=True` requires `mot` to be an MTMMoT instance."
                )
            if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
                raise ValueError(
                    "MTM mode is incompatible with `video_expert.fuse_vae_embedding_in_latents=True`."
                )

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: Optional[Dict[str, Any]] = None,
        action_dit_config: Optional[Dict[str, Any]] = None,
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        multi_term_memory: bool = False,
        mtm_history_sizes: Tuple[int, int, int] = (16, 2, 1),
        mtm_patch_kernel_long: Tuple[int, int, int] = (4, 8, 8),
        mtm_patch_kernel_mid: Tuple[int, int, int] = (2, 4, 4),
        mtm_patch_kernel_current: Tuple[int, int, int] = (1, 2, 2),
        mtm_pred_size: int = 2,
        mtm_amplify_history: bool = False,
        mtm_zero_history_timestep: bool = True,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAMMemory.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAMMemory.")

        # MTM mode flips the assertion contract: base WanVideoDiT requires fuse=True; MTMVideoDiT
        # requires fuse=False at runtime. We force fuse=True for the loader (so the registry
        # constructor passes), then construct MTMVideoDiT with fuse=False after weight loading.
        loader_dit_config = dict(video_dit_config)
        if multi_term_memory:
            user_fuse = bool(video_dit_config.get("fuse_vae_embedding_in_latents", True))
            if user_fuse:
                raise ValueError(
                    "MTM mode requires `video_dit_config['fuse_vae_embedding_in_latents']=False`."
                )
            loader_dit_config["fuse_vae_embedding_in_latents"] = True

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=loader_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        base_video_expert = components.dit
        if multi_term_memory:
            mtm_dit_config = dict(video_dit_config)  # keep user's `fuse=False`
            video_expert = MTMVideoDiT(
                multi_term_memory=True,
                mtm_history_sizes=mtm_history_sizes,
                mtm_patch_kernel_long=mtm_patch_kernel_long,
                mtm_patch_kernel_mid=mtm_patch_kernel_mid,
                mtm_patch_kernel_current=mtm_patch_kernel_current,
                mtm_pred_size=mtm_pred_size,
                mtm_amplify_history=mtm_amplify_history,
                mtm_zero_history_timestep=mtm_zero_history_timestep,
                **mtm_dit_config,
            ).to(device=device, dtype=torch_dtype)
            video_expert.load_state_dict(base_video_expert.state_dict(), strict=False)
            # Re-derive mid/long patch weights from the freshly-loaded `patch_embedding`.
            video_expert._init_mtm_patches_from_patch_embedding()
            del base_video_expert
        else:
            video_expert = base_video_expert

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot_cls = MTMMoT if multi_term_memory else MoT
        mot = mot_cls(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            multi_term_memory=multi_term_memory,
            mtm_history_sizes=mtm_history_sizes,
            mtm_patch_kernel_long=mtm_patch_kernel_long,
            mtm_patch_kernel_mid=mtm_patch_kernel_mid,
            mtm_patch_kernel_current=mtm_patch_kernel_current,
            mtm_pred_size=mtm_pred_size,
            mtm_amplify_history=mtm_amplify_history,
            mtm_zero_history_timestep=mtm_zero_history_timestep,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    # ------------------------------------------------------------------
    # Training data prep & joint mask.
    # ------------------------------------------------------------------

    def _build_inputs_from_precomputed_latents(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Build the same dict as `FastWAM.build_inputs` but with VAE encode skipped.

        Used when the dataloader already supplies `sample['input_latents_precomputed']`
        (a `(B, C, T_pred, H, W)` latent tensor produced by the offline latent pipeline).
        Mirrors `FastWAM.build_inputs` for everything else: context / proprio fusion,
        action / action_is_pad transfer, dtype/device casting. `image_is_pad` is dropped
        (set to None) because the MTM training path predicts only `mtm_pred_size` latents
        and the existing `_compute_video_loss_per_sample` cannot align a per-pixel-frame
        mask onto that two-step output without changes outside this scope.
        """
        if "input_latents_precomputed" not in sample:
            raise ValueError("`sample['input_latents_precomputed']` is required for the latent passthrough path.")
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "Latent-passthrough build_inputs requires `sample['context']` and `sample['context_mask']`."
            )
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        input_latents = sample["input_latents_precomputed"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if input_latents.ndim != 5:
            raise ValueError(
                f"`sample['input_latents_precomputed']` must be 5D `(B, C, T_pred, H, W)`, got shape {tuple(input_latents.shape)}"
            )
        if input_latents.shape[2] != self.mtm_pred_size:
            raise ValueError(
                f"`sample['input_latents_precomputed']` T must equal `mtm_pred_size={self.mtm_pred_size}`, "
                f"got T={input_latents.shape[2]}"
            )

        batch_size = input_latents.shape[0]
        context = sample["context"]
        context_mask = sample["context_mask"]
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be `[B,L,D]/[B,L]`, got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D `[B, T, a_dim]`, got shape {tuple(action.shape)}")
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2 or action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action.shape[1]:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action.shape[1]})"
                )
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if self.proprio_encoder is not None:
            proprio = sample.get("proprio", None)
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D `[B, T, d]`, got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": None,
            "fuse_vae_embedding_in_latents": False,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": None,
        }

    def build_inputs(self, sample, tiled: bool = False):
        if self.multi_term_memory and "input_latents_precomputed" in sample:
            inputs = self._build_inputs_from_precomputed_latents(sample)
        else:
            inputs = super().build_inputs(sample, tiled=tiled)
        if not self.multi_term_memory:
            return inputs

        for key in ("history_long", "history_mid", "history_short"):
            if key not in sample:
                raise ValueError(f"MTM mode requires `sample['{key}']`.")

        long_size, mid_size, curr_size = self.mtm_history_sizes
        history_long = sample["history_long"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        history_mid = sample["history_mid"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        history_short = sample["history_short"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if history_long.ndim != 5 or history_long.shape[2] != long_size:
            raise ValueError(
                f"`history_long` must be `(B, C, {long_size}, H, W)`, got {tuple(history_long.shape)}"
            )
        if history_mid.ndim != 5 or history_mid.shape[2] != mid_size:
            raise ValueError(
                f"`history_mid` must be `(B, C, {mid_size}, H, W)`, got {tuple(history_mid.shape)}"
            )
        if history_short.ndim != 5 or history_short.shape[2] != curr_size:
            raise ValueError(
                f"`history_short` must be `(B, C, {curr_size}, H, W)`, got {tuple(history_short.shape)}"
            )

        if inputs["input_latents"].shape[2] != self.mtm_pred_size:
            raise ValueError(
                f"MTM mode expects `input_latents` with T={self.mtm_pred_size} (pred), "
                f"got T={inputs['input_latents'].shape[2]}."
            )

        # No first-frame anchor in MTM mode: history_short already plays that role.
        inputs["first_frame_latents"] = None
        inputs["fuse_vae_embedding_in_latents"] = False
        inputs["history_long"] = history_long
        inputs["history_mid"] = history_mid
        inputs["history_short"] = history_short
        return inputs

    @torch.no_grad()
    def _build_mtm_joint_attention_mask(
        self,
        mtm_meta: Dict[str, Any],
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the `(Sv+Sa, Sv+Sa)` joint attention mask for MTM mode.

        Layout of the video sub-mask follows `MTMVideoDiT.build_mtm_self_attn_mask`:
            [long | mid | current | pred]; action attends to the full history
            (long + mid + current) to leverage historical motion context.
        """
        L_long = int(mtm_meta["L_long"])
        L_mid = int(mtm_meta["L_mid"])
        L_curr = int(mtm_meta["L_curr"])
        L_pred = int(mtm_meta["L_pred"])
        tokens_per_frame_pred = int(mtm_meta["tokens_per_frame_pred"])

        L_hist = L_long + L_mid + L_curr
        video_seq_len = L_hist + L_pred
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video <-> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_mtm_self_attn_mask(
            L_long=L_long,
            L_mid=L_mid,
            L_curr=L_curr,
            L_pred=L_pred,
            tokens_per_frame_pred=tokens_per_frame_pred,
            device=device,
        )

        # # action <-> action
        # if action_seq_len > 0:
        #     mask[video_seq_len:, video_seq_len:] = True
        #     curr_start = L_long + L_mid
        #     curr_end = curr_start + L_curr
        #     mask[video_seq_len:, curr_start:curr_end] = True

        # action <-> action + action -> full history (long + mid + current)
        if action_seq_len > 0:
            mask[video_seq_len:, video_seq_len:] = True
            mask[video_seq_len:, :L_hist] = True
        return mask

    # ------------------------------------------------------------------
    # Training loss (MTM-aware override).
    # ------------------------------------------------------------------

    def training_loss(self, sample, tiled: bool = False):
        if not self.multi_term_memory:
            return super().training_loss(sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=False,
            history_long=inputs["history_long"],
            history_mid=inputs["history_mid"],
            history_short=inputs["history_short"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        mtm_meta = video_pre["meta"]["mtm"]

        attention_mask = self._build_mtm_joint_attention_mask(
            mtm_meta=mtm_meta,
            action_seq_len=action_tokens.shape[1],
            device=video_tokens.device,
        )
        L_hist = int(mtm_meta["L_hist"])
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            mtm_history_seq_lens={"video": L_hist, "action": 0},
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # No first-frame slicing: pred_video shape `(B, C, T_pred, H, W)` already matches target.
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=True,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    # ------------------------------------------------------------------
    # IDM-Probe-aware training loss: same forward as `training_loss`, but
    # additionally exposes predicted clean latents (x0_hat) for V↔A alignment.
    #
    # IMPORTANT: this method MUST be kept in sync with `training_loss` above.
    # We intentionally duplicate the body (instead of refactoring `training_loss`)
    # to avoid touching the main-line training path.
    # ------------------------------------------------------------------
    def training_loss_with_predictions(self, sample, tiled: bool = False):
        """Same as `training_loss` but also returns intermediate tensors needed
        for IDM probe alignment. MTM mode only.

        Returns:
            loss_total: Tensor scalar (FM video loss + FM action loss, weighted).
            loss_dict: dict[str, float] of detached loss components.
            extras: dict containing the following keys (all on `self.device`,
                already in `self.torch_dtype`, with batch dim):
                - 'x0_hat_video':  (B, C, P, H, W) recovered clean video latent
                - 'x0_hat_action': (B, T_a, action_dim) recovered clean action
                - 'history_short': (B, C, S, H, W) GT current obs latent
                - 'action_gt':     (B, T_a, action_dim) GT action (normalized)
                - 'sigma_v':       (B,) σ ∈ [0, 1] for video FM step
                - 'sigma_a':       (B,) σ ∈ [0, 1] for action FM step
                - 'image_is_pad':  pad mask (or None)
                - 'action_is_pad': (B, T_a) pad mask (or None)
        """
        if not self.multi_term_memory:
            raise ValueError(
                "`training_loss_with_predictions` is MTM-only; "
                "use `training_loss` for non-MTM models."
            )

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=False,
            history_long=inputs["history_long"],
            history_mid=inputs["history_mid"],
            history_short=inputs["history_short"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        mtm_meta = video_pre["meta"]["mtm"]

        attention_mask = self._build_mtm_joint_attention_mask(
            mtm_meta=mtm_meta,
            action_seq_len=action_tokens.shape[1],
            device=video_tokens.device,
        )
        L_hist = int(mtm_meta["L_hist"])
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            mtm_history_seq_lens={"video": L_hist, "action": 0},
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=True,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

        # ----- Recover clean latents (x0_hat) for IDM probe alignment -----
        # Flow matching parameterization (scheduler_continuous.py):
        #   x_t = (1-σ) x0 + σ noise,    target = noise - x0
        #   => x0_hat = x_t - σ * v_pred
        sigma_v = (
            timestep_video.to(dtype=torch.float32)
            / float(self.train_video_scheduler.num_train_timesteps)
        ).clamp(0.0, 1.0)  # (B,)
        sigma_a = (
            timestep_action.to(dtype=torch.float32)
            / float(self.train_action_scheduler.num_train_timesteps)
        ).clamp(0.0, 1.0)  # (B,)
        sigma_v_b = sigma_v.to(dtype=latents.dtype).view(-1, 1, 1, 1, 1)
        sigma_a_b = sigma_a.to(dtype=noisy_action.dtype).view(-1, 1, 1)
        x0_hat_video = latents - sigma_v_b * pred_video
        x0_hat_action = noisy_action - sigma_a_b * pred_action

        extras = {
            "x0_hat_video": x0_hat_video,
            "x0_hat_action": x0_hat_action,
            "history_short": inputs["history_short"],
            "action_gt": action,
            "sigma_v": sigma_v,
            "sigma_a": sigma_a,
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
        }
        return loss_total, loss_dict, extras

    # ------------------------------------------------------------------
    # Inference path: thread `history_long/mid/short` through joint denoising.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.multi_term_memory:
            return super()._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                gt_action=gt_action,
            )
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError("MTM `_predict_joint_noise` requires `history_long/mid/short`.")

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=False,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        mtm_meta = video_pre["meta"]["mtm"]
        attention_mask = self._build_mtm_joint_attention_mask(
            mtm_meta=mtm_meta,
            action_seq_len=action_pre["tokens"].shape[1],
            device=video_pre["tokens"].device,
        )
        L_hist = int(mtm_meta["L_hist"])
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            mtm_history_seq_lens={"video": L_hist, "action": 0},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[Dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        mtm_history_seq_lens: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        if not self.multi_term_memory:
            return super()._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )

        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            mtm_history_seq_lens=mtm_history_seq_lens or {},
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        joint_steps = _resolve_joint_steps(
            num_inference_steps,
            num_video_inference_steps,
            num_action_inference_steps,
            method="FastWAMMemory.infer_joint",
        )
        if not self.multi_term_memory:
            return super().infer_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                action=action,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                num_video_inference_steps=num_video_inference_steps,
                num_action_inference_steps=num_action_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                test_action_with_infer_action=test_action_with_infer_action,
            )
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError(
                "MTM `infer_joint` requires `history_long/mid/short` (callers must provide a "
                "first-chunk placeholder; this implementation does not auto-roll history)."
            )

        # Build context if needed (no proprio expansion logic differs from base).
        self.eval()
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        # Single-shape inference: pred = mtm_pred_size, history latents shapes set by user.
        history_long = history_long.to(device=self.device, dtype=self.torch_dtype)
        history_mid = history_mid.to(device=self.device, dtype=self.torch_dtype)
        history_short = history_short.to(device=self.device, dtype=self.torch_dtype)
        latent_h = history_short.shape[3]
        latent_w = history_short.shape[4]
        latent_t = self.mtm_pred_size

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (history_short.shape[0], self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (history_short.shape[0], action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=joint_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=joint_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=False,
                gt_action=action,
                history_long=history_long,
                history_mid=history_mid,
                history_short=history_short,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)

        # For visualization: concatenate history_short (current obs, 1 latent frame)
        # with predicted latent frames (P frames) before decoding, so that the
        # decoded video contains the full chunk: (P+1) latent frames → V pixel frames.
        # Without this, only P latent frames are decoded → fewer pixel frames.
        decode_latents = torch.cat([history_short, latents_video], dim=2)
        return {
            "video": self._decode_latents(decode_latents, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # `infer_action` only runs the action denoise stage; `num_video_inference_steps` is
        # accepted for API symmetry but has no effect here.
        action_steps = int(
            num_action_inference_steps if num_action_inference_steps is not None else num_inference_steps
        )
        if not self.multi_term_memory:
            return super().infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                num_video_inference_steps=num_video_inference_steps,
                num_action_inference_steps=num_action_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError("MTM `infer_action` requires `history_long/mid/short`.")

        self.eval()
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        history_long = history_long.to(device=self.device, dtype=self.torch_dtype)
        history_mid = history_mid.to(device=self.device, dtype=self.torch_dtype)
        history_short = history_short.to(device=self.device, dtype=self.torch_dtype)

        # Build a zero-pred placeholder so MTM pre_dit produces a full `[hist | pred]` sequence
        # that the action attention mask can index. Pred values are unused (action only attends
        # to the `current` segment per the joint mask).
        batch_size = history_short.shape[0]
        latent_h = history_short.shape[3]
        latent_w = history_short.shape[4]
        zero_pred = torch.zeros(
            (batch_size, history_short.shape[1], self.mtm_pred_size, latent_h, latent_w),
            device=self.device,
            dtype=self.torch_dtype,
        )
        timestep_video = torch.zeros((batch_size,), dtype=self.torch_dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=zero_pred,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=False,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
        )
        mtm_meta = video_pre["meta"]["mtm"]
        L_hist = int(mtm_meta["L_hist"])
        video_seq_len = int(video_pre["tokens"].shape[1])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        attention_mask = self._build_mtm_joint_attention_mask(
            mtm_meta=mtm_meta,
            action_seq_len=int(latents_action.shape[1]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            mtm_history_seq_lens={"video": L_hist, "action": 0},
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=action_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                mtm_history_seq_lens={"video": L_hist, "action": 0},
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_from_latents(
        self,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
        action_horizon: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        decode_video: bool = True,
    ) -> Dict[str, Any]:
        """MTM rollout starting from precomputed history latents (skips VAE encode).

        Differences vs `infer_joint`:
          - No `input_image` and no `_encode_input_image_latents_tensor` call: latent space
            (`H_lat, W_lat, mtm_pred_size`) is determined from `history_short`.
          - Returns the raw `pred_latents` tensor in addition to the optional decoded video.
          - When `decode_video=False`, the VAE decode is skipped (faster eval / unit tests).

        Args:
            history_long/mid/short: ``(B, C, T_seg, H_lat, W_lat)`` history latents that match
                ``self.mtm_history_sizes``. ``B`` is taken from ``history_short.shape[0]``.
            action_horizon: number of action steps to predict (length of action latent T axis).
            prompt: T5 prompt string, mutually exclusive with ``context/context_mask``.
            context, context_mask: ``[L, D] / [L]`` (single sample) or ``[B, L, D] / [B, L]``
                pre-encoded text tokens, mutually exclusive with ``prompt``.
            proprio: optional ``[D]`` or ``[1, D]`` proprio vector for fusion into context.
            action: optional ``[T, a_dim]`` or ``[1, T, a_dim]`` GT action condition for the
                video expert. Pure rollout passes ``None``.
            decode_video: when ``False``, skip ``_decode_latents`` and return ``video=None``.

        Returns:
            ``{"pred_latents": (B, C, mtm_pred_size, H_lat, W_lat),
               "video": list[PIL.Image] | None,
               "action": (T, a_dim) cpu float tensor}``.
        """
        if not self.multi_term_memory:
            raise ValueError("`infer_from_latents` requires `multi_term_memory=True`.")
        joint_steps = _resolve_joint_steps(
            num_inference_steps,
            num_video_inference_steps,
            num_action_inference_steps,
            method="FastWAMMemory.infer_from_latents",
        )
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError("`infer_from_latents` requires `history_long/mid/short`.")
        if history_long.ndim != 5 or history_mid.ndim != 5 or history_short.ndim != 5:
            raise ValueError(
                "All history latents must be 5D `(B, C, T_seg, H_lat, W_lat)`. Got "
                f"long={tuple(history_long.shape)}, mid={tuple(history_mid.shape)}, "
                f"short={tuple(history_short.shape)}"
            )
        long_size, mid_size, curr_size = self.mtm_history_sizes
        if history_long.shape[2] != long_size:
            raise ValueError(f"`history_long` T must be {long_size}, got {history_long.shape[2]}")
        if history_mid.shape[2] != mid_size:
            raise ValueError(f"`history_mid` T must be {mid_size}, got {history_mid.shape[2]}")
        if history_short.shape[2] != curr_size:
            raise ValueError(f"`history_short` T must be {curr_size}, got {history_short.shape[2]}")

        self.eval()
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio,
            )

        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        history_long = history_long.to(device=self.device, dtype=self.torch_dtype)
        history_mid = history_mid.to(device=self.device, dtype=self.torch_dtype)
        history_short = history_short.to(device=self.device, dtype=self.torch_dtype)
        batch_size = history_short.shape[0]
        latent_h = history_short.shape[3]
        latent_w = history_short.shape[4]
        latent_t = self.mtm_pred_size

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=joint_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=joint_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video, infer_deltas_video, infer_timesteps_action, infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=False,
                gt_action=action,
                history_long=history_long,
                history_mid=history_mid,
                history_short=history_short,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        pred_latents = latents_video.detach()
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        # WAN VAE 时间因果卷积下采样 4×：N 帧 latent -> 1 + 4*(N-1) 帧像素，且第 1 帧 latent 必须是 I-frame，
        # 否则解码会出现颜色/亮度畸变。`pred_latents` 全部是 P-frame 类型 latent（mtm_pred_size 帧），
        # 因此 decode 前需要把 `history_short`（geometry 上正好是当前 chunk 的 I-frame）拼到最前，
        # decode 后再丢掉首 1 帧像素（属于 history，不属于 prediction）。对齐 `infer_joint` 的做法。
        if decode_video:
            decode_latents = torch.cat([history_short, pred_latents], dim=2)
            full_video = self._decode_latents(decode_latents, tiled=tiled)
            decoded_video = full_video[1:]
        else:
            decoded_video = None
        return {
            "pred_latents": pred_latents,
            "video": decoded_video,
            "action": action_out,
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            num_video_inference_steps=num_video_inference_steps,
            num_action_inference_steps=num_action_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
        )


# -----------------------------------------------------------------------------
# Smoke test: shape correctness + backward, runnable via `python -m fastwam.models.wan22.fastwam_memory`.
# -----------------------------------------------------------------------------


def _smoke_test() -> None:
    """End-to-end shape + backward smoke test on a tiny model (CPU, no pretrained weights).

    Runs five checks:
        1. `_init_mtm_patches_from_patch_embedding`: long/mid path on constant input matches
           `patch_embedding` mean (local-mean equivalence).
        2. Three-patch output shapes match the helios-style token budget.
        3. `build_mtm_self_attn_mask` visibility matrix obeys
           `mask[:L_hist, L_hist:].any() == False` and `mask[L_hist:, :].all() == True`.
        4. `pre_dit -> blocks -> post_dit` end-to-end produces `(B, C, T_pred, H, W)` and
           `loss.backward()` runs without errors.
        5. `is_amplify_history` flag toggling has no effect when history segment is empty.
    """
    print("[fastwam_memory smoke] starting...")
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    # Tiny model. patch_size=(1,2,2), in_dim=16 (matches Wan VAE z_dim).
    cfg = dict(
        hidden_dim=64,
        in_dim=16,
        ffn_dim=128,
        out_dim=16,
        text_dim=64,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=32,
        num_layers=2,
        has_image_input=False,
        has_image_pos_emb=False,
        has_ref_conv=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=False,  # MTM mode
        action_conditioned=False,
        video_attention_mask_mode="bidirectional",
    )
    expert = MTMVideoDiT(
        multi_term_memory=True,
        mtm_history_sizes=(16, 2, 1),
        mtm_pred_size=2,
        mtm_amplify_history=True,
        mtm_zero_history_timestep=True,
        **cfg,
    ).to(device=device, dtype=dtype)
    expert.eval()  # patch-init equivalence is best checked in eval (no dropout).

    # Spatial latent dims chosen so all kernel strides divide cleanly: H=16, W=16 => h_pred=8, w_pred=8.
    B, C, H, W = 1, 16, 16, 16
    long_size, mid_size, curr_size = expert.mtm_history_sizes
    pred_size = expert.mtm_pred_size

    # ---- Check 1: patch-init local mean equivalence on constant input. ---------
    const = torch.full((1, C, 4, H, W), 0.7, device=device, dtype=dtype)
    with torch.no_grad():
        emb_out = expert.patch_embedding(const)  # (1, D, 4, h, w)
        long_out = expert.patch_long(pad_for_3d_conv(const, expert.mtm_patch_kernel_long))
        emb_mean = emb_out.mean()
        long_mean = long_out.mean()
    diff_local_mean = float((emb_mean - long_mean).abs().item())
    assert diff_local_mean < 1e-4, f"patch_long local-mean drift {diff_local_mean:.2e}"
    print(f"  [check 1] patch_long local-mean ~= patch_embedding (|delta|={diff_local_mean:.2e})  OK")

    # ---- Check 2: three-patch output shapes. ----------------------------------
    history_long = torch.randn(B, C, long_size, H, W, device=device, dtype=dtype)
    history_mid = torch.randn(B, C, mid_size, H, W, device=device, dtype=dtype)
    history_short = torch.randn(B, C, curr_size, H, W, device=device, dtype=dtype)
    pred = torch.randn(B, C, pred_size, H, W, device=device, dtype=dtype)
    with torch.no_grad():
        x_long = expert.patch_long(pad_for_3d_conv(history_long, expert.mtm_patch_kernel_long))
        x_mid = expert.patch_mid(pad_for_3d_conv(history_mid, expert.mtm_patch_kernel_mid))
        x_short = expert.patch_embedding(history_short)
        x_pred = expert.patchify(pred)
    h_pred, w_pred = pred.shape[3] // 2, pred.shape[4] // 2  # (1,2,2) patch
    assert x_short.shape[2:] == (1, h_pred, w_pred), f"short shape {tuple(x_short.shape)}"
    assert x_mid.shape[2:] == (1, h_pred // 2, w_pred // 2), f"mid shape {tuple(x_mid.shape)}"
    assert x_pred.shape[2:] == (pred_size, h_pred, w_pred), f"pred shape {tuple(x_pred.shape)}"
    print(
        f"  [check 2] patch shapes long={tuple(x_long.shape)}, mid={tuple(x_mid.shape)}, "
        f"short={tuple(x_short.shape)}, pred={tuple(x_pred.shape)}  OK"
    )

    # ---- Check 3: MTM self-attn mask visibility. ------------------------------
    L_long = int(x_long.numel() / (x_long.shape[0] * x_long.shape[1]))
    L_mid = int(x_mid.numel() / (x_mid.shape[0] * x_mid.shape[1]))
    L_curr = int(x_short.numel() / (x_short.shape[0] * x_short.shape[1]))
    L_pred = int(x_pred.numel() / (x_pred.shape[0] * x_pred.shape[1]))
    L_hist = L_long + L_mid + L_curr
    mask = expert.build_mtm_self_attn_mask(
        L_long=L_long, L_mid=L_mid, L_curr=L_curr, L_pred=L_pred,
        tokens_per_frame_pred=h_pred * w_pred, device=device,
    )
    assert mask.shape == (L_hist + L_pred, L_hist + L_pred)
    assert not mask[:L_hist, L_hist:].any(), "history must NOT see pred"
    assert mask[L_hist:, :].all(), "pred must see all history + all pred"
    assert mask[:L_hist, :L_hist].all(), "history must be bidirectional internally"
    print(f"  [check 3] MTM self-attn mask `(L_hist={L_hist}, L_pred={L_pred})`  OK")

    # ---- Check 4: end-to-end forward + backward via blocks loop. --------------
    expert.train()
    context = torch.randn(B, 4, cfg["text_dim"], device=device, dtype=dtype)
    context_mask = torch.ones(B, 4, dtype=torch.bool, device=device)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)
    pre = expert.pre_dit(
        x=pred,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        history_long=history_long,
        history_mid=history_mid,
        history_short=history_short,
    )
    self_attn_mask = mask
    x_tokens = pre["tokens"]
    for block in expert.blocks:
        x_tokens = block(
            x_tokens,
            pre["context"],
            pre["t_mod"],
            pre["freqs"],
            context_mask=pre["context_mask"],
            self_attn_mask=self_attn_mask,
        )
    out = expert.post_dit(x_tokens, pre)
    expected_shape = (B, cfg["out_dim"], pred_size, H, W)
    assert tuple(out.shape) == expected_shape, f"post_dit shape {tuple(out.shape)} != {expected_shape}"
    loss = out.float().pow(2).mean()
    loss.backward()
    print(f"  [check 4] end-to-end forward shape={tuple(out.shape)}, loss={loss.item():.4f}, backward OK")

    # ---- Check 5: MTMMoT amplification produces non-zero grad on `history_key_scale_logit`. ----
    from fastwam.models.wan22.action_dit import ActionDiT
    expert.zero_grad(set_to_none=True)
    action_horizon, action_dim = 4, 7
    action_expert = ActionDiT(
        hidden_dim=cfg["hidden_dim"],
        action_dim=action_dim,
        ffn_dim=cfg["ffn_dim"],
        text_dim=cfg["text_dim"],
        freq_dim=cfg["freq_dim"],
        eps=cfg["eps"],
        num_heads=cfg["num_heads"],
        attn_head_dim=cfg["attn_head_dim"],
        num_layers=cfg["num_layers"],
        use_gradient_checkpointing=False,
    ).to(device=device, dtype=dtype)
    mot = MTMMoT(mixtures={"video": expert, "action": action_expert}, mot_checkpoint_mixed_attn=False)

    pre_v = expert.pre_dit(
        x=pred,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        history_long=history_long,
        history_mid=history_mid,
        history_short=history_short,
    )
    pre_a = action_expert.pre_dit(
        action_tokens=torch.randn(B, action_horizon, action_dim, device=device, dtype=dtype),
        timestep=timestep,
        context=context,
        context_mask=context_mask,
    )
    L_hist = int(pre_v["meta"]["mtm"]["L_hist"])
    Sv = int(pre_v["tokens"].shape[1])
    Sa = int(pre_a["tokens"].shape[1])
    joint_mask = torch.zeros((Sv + Sa, Sv + Sa), dtype=torch.bool, device=device)
    joint_mask[:Sv, :Sv] = mask
    joint_mask[Sv:, Sv:] = True
    joint_mask[Sv:, :L_hist] = True

    tokens_out = mot(
        embeds_all={"video": pre_v["tokens"], "action": pre_a["tokens"]},
        attention_mask=joint_mask,
        freqs_all={"video": pre_v["freqs"], "action": pre_a["freqs"]},
        context_all={
            "video": {"context": pre_v["context"], "mask": pre_v["context_mask"]},
            "action": {"context": pre_a["context"], "mask": pre_a["context_mask"]},
        },
        t_mod_all={"video": pre_v["t_mod"], "action": pre_a["t_mod"]},
        mtm_history_seq_lens={"video": L_hist, "action": 0},
    )
    out_v = expert.post_dit(tokens_out["video"], pre_v)
    out_a = action_expert.post_dit(tokens_out["action"], pre_a)
    loss2 = out_v.float().pow(2).mean() + out_a.float().pow(2).mean()
    loss2.backward()
    grad = expert.history_key_scale_logit.grad
    assert grad is not None, "history_key_scale_logit.grad is None -- amplify branch did not engage"
    grad_norm = float(grad.norm().item())
    assert grad_norm > 0, f"history_key_scale_logit grad norm is zero ({grad_norm})"
    print(
        f"  [check 5] MTMMoT path: history_key_scale_logit grad norm={grad_norm:.4e} (>0), "
        f"loss={loss2.item():.4f}  OK"
    )

    # ---- Check 6: zero-segment configs forward+backward. ------------------------
    for tag, sizes in [("no_memory", (0, 0, 1)), ("long_only", (18, 0, 1))]:
        expert_z = MTMVideoDiT(
            multi_term_memory=True,
            mtm_history_sizes=sizes,
            mtm_pred_size=2,
            mtm_amplify_history=False,
            mtm_zero_history_timestep=True,
            **cfg,
        ).to(device=device, dtype=dtype)
        expert_z.train()
        ls, ms, cs = sizes
        h_long_z = torch.randn(B, C, ls, H, W, device=device, dtype=dtype)
        h_mid_z = torch.randn(B, C, ms, H, W, device=device, dtype=dtype)
        h_short_z = torch.randn(B, C, cs, H, W, device=device, dtype=dtype)
        pred_z = torch.randn(B, C, pred_size, H, W, device=device, dtype=dtype)
        pre_z = expert_z.pre_dit(
            x=pred_z, timestep=timestep, context=context, context_mask=context_mask,
            history_long=h_long_z, history_mid=h_mid_z, history_short=h_short_z,
        )
        x_tok = pre_z["tokens"]
        mtm_z = pre_z["meta"]["mtm"]
        mask_z = expert_z.build_mtm_self_attn_mask(
            L_long=int(mtm_z["L_long"]), L_mid=int(mtm_z["L_mid"]),
            L_curr=int(mtm_z["L_curr"]), L_pred=int(mtm_z["L_pred"]),
            tokens_per_frame_pred=int(mtm_z["tokens_per_frame_pred"]), device=device,
        )
        for block in expert_z.blocks:
            x_tok = block(
                x_tok, pre_z["context"], pre_z["t_mod"], pre_z["freqs"],
                context_mask=pre_z["context_mask"], self_attn_mask=mask_z,
            )
        out_z = expert_z.post_dit(x_tok, pre_z)
        assert tuple(out_z.shape) == expected_shape, f"{tag}: shape {tuple(out_z.shape)} != {expected_shape}"
        loss_z = out_z.float().pow(2).mean()
        loss_z.backward()
        print(f"  [check 6/{tag}] sizes={sizes} forward+backward OK, loss={loss_z.item():.4f}")

    # ---- Check 7: checkpoint compat (16,2,1) -> (0,0,1). ----------------------
    sd_orig = expert.state_dict()
    expert_compat = MTMVideoDiT(
        multi_term_memory=True,
        mtm_history_sizes=(0, 0, 1),
        mtm_pred_size=2,
        mtm_amplify_history=True,
        mtm_zero_history_timestep=True,
        **cfg,
    ).to(device=device, dtype=dtype)
    missing, unexpected = expert_compat.load_state_dict(sd_orig, strict=False)
    assert len(unexpected) == 0, f"unexpected keys: {unexpected}"
    print(f"  [check 7] ckpt (16,2,1)->(0,0,1) loaded, missing={len(missing)} (expected 0)")

    print("[fastwam_memory smoke] all checks passed.")


if __name__ == "__main__":
    _smoke_test()

