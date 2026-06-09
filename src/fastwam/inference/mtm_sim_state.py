"""Stateful MTM latent-timeline controller for closed-loop simulation rollout.

This module is **environment-agnostic**: it only manages the latent timeline
(zero-pad -> chunk-append -> sliding-window cut) and delegates VAE encode / decode
to the injected ``model``. Simulation-specific code (image preprocessing, env
stepping, action denormalisation) stays in each ``eval_*_single.py`` / policy
module.

Usage sketch (see ``experiments/robocasa/eval_robocasa_single.py`` for a full
integration)::

    from fastwam.inference.mtm_sim_state import MTMRolloutConfig, MTMSimState

    mtm_cfg = MTMRolloutConfig.from_cfg_and_model(cfg, model)
    state   = MTMSimState(model, mtm_cfg, device, dtype)

    # -- episode loop --
    state.reset()
    obs_pixel = _obs_to_video_tensor(env_obs, ...)  # (1,3,H,W) in [-1,1]

    for chunk_idx in range(max_chunks):
        state.set_observation_rgb(obs_pixel)
        h_long, h_mid, h_short = state.build_histories()
        pred = model.infer_action(history_long=h_long, history_mid=h_mid,
                                  history_short=h_short, ...)
        # execute actions, collect future_rgbs at every r-th step
        future_pixels = ...  # (1, 3, N_future, H, W) in [-1,1]
        state.after_chunk_executed(future_pixels)
        obs_pixel = future_pixels[:, :, -1:, :, :]  # reuse last future as next obs
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MTMRolloutConfig:
    """All numeric constants for one MTM rollout session, derived from cfg + model."""

    L: int                      # mtm_history_sizes[0] (long segment latent frames)
    M: int                      # mtm_history_sizes[1] (mid segment latent frames)
    S: int                      # mtm_history_sizes[2] (current / short; must be 1)
    P: int                      # mtm_pred_size (predicted latent frames per chunk)
    r: int                      # action_video_freq_ratio
    num_actions: int            # num_frames - 1
    V: int                      # video frames per chunk = (num_actions // r) + 1
    N_future: int               # future obs to collect = num_actions // r
    Z: int                      # zero_pad_len (left-pad length for timeline init)
    temporal_factor: int        # vae.temporal_downsample_factor (typically 4)
    z_dim: int                  # vae latent channel dim
    latent_h: int               # latent spatial height
    latent_w: int               # latent spatial width

    @classmethod
    def from_cfg_and_model(cls, cfg: DictConfig, model) -> "MTMRolloutConfig":
        """Parse and validate all MTM rollout constants from Hydra cfg + model.

        Args:
            cfg: Full Hydra config (must contain ``cfg.model`` and ``cfg.data.train``).
            model: Loaded ``FastWAMMemory`` instance (provides VAE attributes).
        """
        # --- model-side ---
        mtm_sizes = cfg.model.get("mtm_history_sizes")
        if mtm_sizes is None:
            raise ValueError(
                "`cfg.model.mtm_history_sizes` is required for MTM rollout. "
                "Make sure you are using a `fastwam_memory` model config."
            )
        if isinstance(mtm_sizes, (DictConfig, ListConfig)):
            mtm_sizes = OmegaConf.to_container(mtm_sizes, resolve=True)
        mtm_sizes = list(mtm_sizes)
        if len(mtm_sizes) != 3:
            raise ValueError(f"`mtm_history_sizes` must be length 3 (L, M, S), got {mtm_sizes}")
        L, M, S = int(mtm_sizes[0]), int(mtm_sizes[1]), int(mtm_sizes[2])
        if S != 1:
            raise ValueError(
                f"MTM rollout currently requires S (current/short) == 1, got S={S}. "
                "The model and dataset only support S=1."
            )

        P = int(cfg.model.get("mtm_pred_size", 2))

        # --- data-side ---
        data_train = cfg.data.train
        num_frames = int(data_train.num_frames)
        r = int(data_train.action_video_freq_ratio)
        num_actions = num_frames - 1

        if num_actions <= 0:
            raise ValueError(f"`num_frames` must be >= 2, got {num_frames}")
        if r <= 0:
            raise ValueError(f"`action_video_freq_ratio` must be positive, got {r}")
        if num_actions % r != 0:
            raise ValueError(
                f"`num_actions` ({num_actions} = num_frames-1) must be divisible by "
                f"`action_video_freq_ratio` ({r}), but {num_actions} % {r} = {num_actions % r}."
            )

        N_future = num_actions // r
        V = N_future + 1  # 1 (current obs) + N_future

        # --- VAE geometry ---
        temporal_factor = int(model.vae.temporal_downsample_factor)
        z_dim = int(model.vae.model.z_dim)
        upsampling_factor = int(model.vae.upsampling_factor)

        # Validate V vs temporal_factor: (V - 1) must be divisible by temporal_factor
        if (V - 1) % temporal_factor != 0:
            raise ValueError(
                f"Video frames per chunk V={V} is incompatible with VAE "
                f"temporal_downsample_factor={temporal_factor}: "
                f"(V-1)={V - 1} is not divisible by {temporal_factor}."
            )
        chunk_latent_t = (V - 1) // temporal_factor + 1
        if chunk_latent_t != P + 1:
            raise ValueError(
                f"VAE encodes V={V} pixel frames into {chunk_latent_t} latent frames, "
                f"but expected P+1={P + 1} (mtm_pred_size={P}). Check num_frames / "
                f"action_video_freq_ratio / mtm_pred_size consistency."
            )

        # --- zero_pad_len ---
        Z_cfg = data_train.get("zero_pad_len", None)
        if Z_cfg is not None:
            Z = int(Z_cfg)
        else:
            # Fallback: zero_pad_len not in cfg (non-latent data config).
            Z = L + M
            logger.warning(
                "`zero_pad_len` not found in cfg.data.train; using L+M=%d as fallback.", Z
            )

        # Sanity: L + M + S should equal Z + 1 (training convention)
        if L + M + S != Z + 1:
            logger.warning(
                "L+M+S=%d != Z+1=%d (zero_pad_len=%d). This may cause "
                "train/eval distribution mismatch. Please double-check configs.",
                L + M + S, Z + 1, Z,
            )

        # --- latent spatial dims ---
        video_size = data_train.get("video_size", [256, 768])
        if isinstance(video_size, (DictConfig, ListConfig)):
            video_size = OmegaConf.to_container(video_size, resolve=True)
        pixel_h, pixel_w = int(video_size[0]), int(video_size[1])
        latent_h = pixel_h // upsampling_factor
        latent_w = pixel_w // upsampling_factor

        result = cls(
            L=L, M=M, S=S, P=P, r=r,
            num_actions=num_actions,
            V=V, N_future=N_future, Z=Z,
            temporal_factor=temporal_factor,
            z_dim=z_dim,
            latent_h=latent_h, latent_w=latent_w,
        )
        logger.info(
            "MTMRolloutConfig: L=%d M=%d S=%d P=%d r=%d num_actions=%d V=%d "
            "N_future=%d Z=%d temporal_factor=%d z_dim=%d latent=(%d,%d)",
            L, M, S, P, r, num_actions, V, N_future, Z,
            temporal_factor, z_dim, latent_h, latent_w,
        )
        return result


# ---------------------------------------------------------------------------
# Stateful controller
# ---------------------------------------------------------------------------

class MTMSimState:
    """Stateful latent-timeline controller for closed-loop MTM rollout.

    Maintains a growing **latent timeline** ``(1, C, T_tot, H_lat, W_lat)``
    initialised with ``Z`` zero-padded frames (matching training's
    ``zero_pad_len``). Each replan cycle appends ``P+1`` new latent frames,
    and ``build_histories()`` slices the tail ``L+M+S`` frames into the three
    MTM history segments.

    Lifecycle per replan cycle
    --------------------------
    1. ``set_observation_rgb(pixel)`` -- cache current obs pixel + single-frame
       VAE encode (for ``infer_action`` fast path).
    2. ``build_histories()`` -- slice tail ``L+M+S`` of timeline into
       ``(history_long, history_mid, history_short)``.
    3. *(caller runs ``model.infer_action`` or ``model.infer_from_latents``)*
    4. *(caller executes ``num_actions`` actions, collecting ``N_future`` future
       obs at every ``r``-th step)*
    5. ``after_chunk_executed(future_pixels)`` -- concatenate
       ``[last_obs_pixel | future_pixels]`` (``V`` frames total), VAE-encode
       to ``P+1`` latent frames, and append to the timeline.
    6. Next iteration: caller reuses the **last** future pixel as the new
       observation (``should_skip_env_fetch_obs() == True``).
    """

    def __init__(
        self,
        model,
        mtm_cfg: MTMRolloutConfig,
        device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.cfg = mtm_cfg
        self.device = device
        self.dtype = dtype

        # Timeline buffer: starts as (1, C, Z, H, W) zeros.
        self._timeline_lat: torch.Tensor = torch.zeros(
            (1, mtm_cfg.z_dim, mtm_cfg.Z, mtm_cfg.latent_h, mtm_cfg.latent_w),
            device=device, dtype=dtype,
        )

        # Cached current observation pixel and its single-frame latent.
        self._last_obs_pixel: Optional[torch.Tensor] = None   # (1,3,H,W) in [-1,1]
        self._last_short_latent: Optional[torch.Tensor] = None  # (1,C,1,H_lat,W_lat)

        # Chunk counter for first-chunk detection.
        self._chunk_count: int = 0

        logger.info(
            "MTMSimState initialized: Z=%d, P+1=%d, V=%d, timeline shape=%s",
            mtm_cfg.Z, mtm_cfg.P + 1, mtm_cfg.V, tuple(self._timeline_lat.shape),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset timeline to zero-pad and clear caches (call at episode start)."""
        self._timeline_lat = torch.zeros(
            (1, self.cfg.z_dim, self.cfg.Z, self.cfg.latent_h, self.cfg.latent_w),
            device=self.device, dtype=self.dtype,
        )
        self._last_obs_pixel = None
        self._last_short_latent = None
        self._chunk_count = 0

    @torch.no_grad()
    def set_observation_rgb(self, image_pixel: torch.Tensor) -> None:
        """Cache current observation pixel and produce its single-frame latent.

        Args:
            image_pixel: ``(1, 3, H, W)`` tensor in ``[-1, 1]``, already on the
                correct device and dtype for VAE input.
        """
        if image_pixel.ndim != 4 or image_pixel.shape[0] != 1 or image_pixel.shape[1] != 3:
            raise ValueError(
                f"`image_pixel` must be (1, 3, H, W), got {tuple(image_pixel.shape)}"
            )
        self._last_obs_pixel = image_pixel.to(device=self.device, dtype=self.dtype)
        self._last_short_latent = self.model._encode_input_image_latents_tensor(
            self._last_obs_pixel
        )  # (1, C, 1, H_lat, W_lat)

    def build_histories(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice tail ``L+M+S`` of the timeline into three history segments.

        When ``S == 1``, the last frame of the timeline slice is **overridden**
        with the cached single-frame encode of the current observation to avoid
        any drift between single-frame and multi-frame VAE encoding paths.

        Returns:
            ``(history_long, history_mid, history_short)`` each shaped
            ``(1, C, T_seg, H_lat, W_lat)``.
        """
        if self._last_short_latent is None:
            raise RuntimeError(
                "Call `set_observation_rgb()` before `build_histories()`."
            )

        L, M, S = self.cfg.L, self.cfg.M, self.cfg.S
        total_needed = L + M + S
        T_cur = self._timeline_lat.shape[2]

        if T_cur < total_needed:
            # Timeline shorter than L+M+S: left-pad with zeros.
            pad = torch.zeros(
                (1, self.cfg.z_dim, total_needed - T_cur,
                 self.cfg.latent_h, self.cfg.latent_w),
                device=self.device, dtype=self.dtype,
            )
            window = torch.cat([pad, self._timeline_lat], dim=2)
        else:
            window = self._timeline_lat[:, :, -total_needed:]

        history_long = window[:, :, :L].contiguous()
        history_mid = window[:, :, L:L + M].contiguous()
        # Override short segment with the single-frame encode of current obs.
        history_short = self._last_short_latent.contiguous()  # (1, C, 1, H, W)

        return history_long, history_mid, history_short

    def is_first_chunk(self) -> bool:
        """True if no chunk has been executed yet (timeline is still pure zero-pad)."""
        return self._chunk_count == 0

    def should_skip_env_fetch_obs(self) -> bool:
        """After the first chunk, the caller should reuse the last future pixel
        as the next observation instead of re-fetching from env."""
        return not self.is_first_chunk()

    @torch.no_grad()
    def after_chunk_executed(self, future_rgb_pixels: torch.Tensor) -> None:
        """Encode ``[last_obs | future_rgbs]`` and append to the timeline.

        Args:
            future_rgb_pixels: ``(1, 3, N_future, H, W)`` tensor in ``[-1, 1]``.
                These are the ``N_future`` observations collected at every ``r``-th
                action step during the just-completed replan cycle.

        The method:
          1. Concatenates ``_last_obs_pixel`` (``1`` frame) with ``future_rgb_pixels``
             (``N_future`` frames) along the time axis -> ``(1, 3, V, H, W)``.
          2. Calls ``model._encode_video_latents`` -> ``(1, C, P+1, H_lat, W_lat)``.
          3. Appends the ``P+1`` latent frames to the right of ``_timeline_lat``.
          4. Optionally trims the timeline to a reasonable max length.
        """
        if self._last_obs_pixel is None:
            raise RuntimeError(
                "Call `set_observation_rgb()` before `after_chunk_executed()`."
            )

        N_future = self.cfg.N_future
        V = self.cfg.V
        P_plus_1 = self.cfg.P + 1

        # Validate future_rgb_pixels shape.
        if future_rgb_pixels.ndim == 4:
            # (N_future, 3, H, W) -> (1, 3, N_future, H, W)
            future_rgb_pixels = future_rgb_pixels.unsqueeze(0)
        if future_rgb_pixels.ndim != 5:
            raise ValueError(
                f"`future_rgb_pixels` must be 5D (1, 3, N_future, H, W) or "
                f"4D (N_future, 3, H, W), got {tuple(future_rgb_pixels.shape)}"
            )
        if future_rgb_pixels.shape[2] != N_future:
            raise ValueError(
                f"Expected N_future={N_future} future frames, "
                f"got {future_rgb_pixels.shape[2]}."
            )

        future_rgb_pixels = future_rgb_pixels.to(device=self.device, dtype=self.dtype)

        # Construct (1, 3, V, H, W): [last_obs (1 frame) | futures (N_future frames)].
        obs_5d = self._last_obs_pixel.unsqueeze(2)  # (1, 3, 1, H, W)
        video_pixels = torch.cat([obs_5d, future_rgb_pixels], dim=2)  # (1, 3, V, H, W)

        if video_pixels.shape[2] != V:
            raise ValueError(
                f"Concatenated video has {video_pixels.shape[2]} frames, expected V={V}."
            )

        # VAE encode: vae.encode expects a list of (C, T, H, W) tensors.
        # _encode_video_latents passes its arg directly to vae.encode, so we
        # must wrap the single video in a list.
        video_for_vae = video_pixels.squeeze(0)  # (3, V, H, W)
        new_latents = self.model._encode_video_latents([video_for_vae])
        # vae.encode returns stacked (B, C, T_lat, H_lat, W_lat); keep batch dim.
        if new_latents.ndim == 4:
            new_latents = new_latents.unsqueeze(0)

        # Validate shape.
        actual_t = new_latents.shape[2]
        if actual_t != P_plus_1:
            raise ValueError(
                f"VAE encoded {V} pixel frames into {actual_t} latent frames, "
                f"expected P+1={P_plus_1}."
            )

        # Append to timeline.
        new_latents = new_latents.to(device=self.device, dtype=self.dtype)
        self._timeline_lat = torch.cat(
            [self._timeline_lat, new_latents], dim=2
        )

        # Trim timeline to bound GPU memory (keep at most Z + 100*(P+1) frames).
        max_len = max(self.cfg.Z + 100 * P_plus_1, self.cfg.L + self.cfg.M + self.cfg.S)
        if self._timeline_lat.shape[2] > max_len:
            self._timeline_lat = self._timeline_lat[:, :, -max_len:]

        self._chunk_count += 1
        logger.debug(
            "MTMSimState: chunk %d done, timeline T=%d",
            self._chunk_count, self._timeline_lat.shape[2],
        )

    # ------------------------------------------------------------------
    # Convenience: replan_steps validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_replan_steps(replan_steps: int, action_video_freq_ratio: int) -> None:
        """Assert that ``replan_steps`` is divisible by ``action_video_freq_ratio``.

        This must hold so that future obs collection covers exactly
        ``replan_steps // r`` frames, aligned with the VAE temporal grid.
        Raises ``ValueError`` with a clear message if violated.
        """
        if replan_steps % action_video_freq_ratio != 0:
            r = action_video_freq_ratio
            raise ValueError(
                f"`replan_steps` ({replan_steps}) must be divisible by "
                f"`action_video_freq_ratio` ({r}) for MTM rollout, "
                f"because we need to collect exactly replan_steps/r future "
                f"observations. Please set replan_steps to a multiple of {r} "
                f"(e.g. {r * (replan_steps // r)} or {r * (replan_steps // r + 1)})."
            )
