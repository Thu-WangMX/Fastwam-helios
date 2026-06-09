"""FastWAM Memory + IDM teacher-forcing variant.

Combines `FastWAMMemory`'s multi-term-memory video DiT with `FastWAMIDM`'s teacher-forcing
training and two-stage inference: action denoising attends to a *clean* copy of the
predicted future video alongside long/mid/current history, never to the noisy pred.

Training sequence layout (joint MoT input):
    [ history | noisy_pred | cond_pred | action ]

Attention visibility (training):
    history    -> history                                 : True (bidirectional)
    history    -> noisy_pred / cond_pred / action         : False
    noisy_pred -> history + noisy_pred(per-mode)          : True
    noisy_pred -> cond_pred / action                      : False
    cond_pred  -> history + cond_pred(per-mode)           : True
    cond_pred  -> noisy_pred / action                     : False
    action     -> history + cond_pred + action            : True
    action     -> noisy_pred                              : False

Video loss is computed only on the `noisy_pred` slice; `cond_pred` exists purely as
clean K/V for the noisy and action branches.

Inference is two-stage (mirrors `FastWAMIDM` with MTM history threaded into stage 1):
    Stage 1: video-only loop with `[history | noisy_pred]` and the existing
             `MTMVideoDiT.build_mtm_self_attn_mask` (no action expert).
    Stage 2: re-pre_dit the clean prediction at timestep 0 to obtain `[history |
             cond_pred]`, prefill K/V cache, run action denoising with action attending
             to the full clean video sequence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.latent_corruption import corrupt_cond_latents
from fastwam.utils.logging_config import get_logger

from .fastwam_memory import FastWAMMemory

logger = get_logger(__name__)


class FastWAMMemoryIDM(FastWAMMemory):
    """MTM model where action is conditioned on a clean teacher-forcing video copy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.multi_term_memory:
            raise ValueError(
                "`FastWAMMemoryIDM` requires `multi_term_memory=True`; "
                "use `FastWAMIDM` for the no-history variant."
            )
        # Optional latent corruption applied ONLY to the action cond branch input
        # during training. Configured post-construction via `set_cond_latent_corruption`.
        self.cond_latent_corruption_config: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Cond-branch latent corruption (training-time only).
    # ------------------------------------------------------------------

    def set_cond_latent_corruption(self, config: Optional[Dict[str, Any]]) -> None:
        """Enable/disable corruption of the gt video latent feeding the action cond branch.

        ``config`` schema (all keys optional except ``mode``)::

            mode:               "noise" | "downsample" | "none" | null
            sigma_max:          float in (0, 1]   (noise mode, default 0.5)
            ratio_min:          float in (0, 1]   (downsample mode, default 0.3)
            pass_through_prob:  float in [0, 1)   (default 0.0)

        Pass ``None`` or ``{"mode": "none"}`` to disable.
        """
        if config is None:
            self.cond_latent_corruption_config = None
            return
        if not isinstance(config, dict):
            raise ValueError(
                f"`cond_latent_corruption` must be a dict, got {type(config)}."
            )
        mode = config.get("mode", None)
        if mode in (None, "none"):
            self.cond_latent_corruption_config = None
            return
        if mode not in ("noise", "downsample"):
            raise ValueError(
                f"`cond_latent_corruption.mode` must be one of "
                f"{{'noise','downsample','none',None}}, got {mode!r}."
            )
        cleaned = {
            "mode": mode,
            "sigma_max": float(config.get("sigma_max", 0.5)),
            "ratio_min": float(config.get("ratio_min", 0.3)),
            "pass_through_prob": float(config.get("pass_through_prob", 0.0)),
        }
        if not (0.0 <= cleaned["pass_through_prob"] < 1.0):
            raise ValueError(
                f"`cond_latent_corruption.pass_through_prob` must be in [0, 1), "
                f"got {cleaned['pass_through_prob']}."
            )
        self.cond_latent_corruption_config = cleaned
        logger.info(
            "FastWAMMemoryIDM cond_latent_corruption enabled: %s", cleaned
        )

    def _maybe_corrupt_cond_latents(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cond_latent_corruption_config
        if cfg is None or not self.training:
            return z
        return corrupt_cond_latents(
            z,
            mode=cfg["mode"],
            sigma_max=cfg["sigma_max"],
            ratio_min=cfg["ratio_min"],
            pass_through_prob=cfg["pass_through_prob"],
        )

    # ------------------------------------------------------------------
    # Sequence assembly helpers (training).
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_pred_branches(video_pre_noisy: Dict[str, Any], video_pre_cond: Dict[str, Any]):
        """Concatenate `[hist | noisy_pred | cond_pred]` from two `MTMVideoDiT.pre_dit` calls.

        Both calls share the same history segment (history tokens are zero-timestep + the
        same patch convs + the same RoPE indices), so we keep the noisy call's history
        slice and append the cond call's pred slice.

        Returns (tokens, freqs, t_mod, context_mask, L_hist, L_pred).
        """
        L_hist = int(video_pre_noisy["meta"]["mtm"]["L_hist"])
        L_pred = int(video_pre_noisy["meta"]["mtm"]["L_pred"])
        if int(video_pre_cond["meta"]["mtm"]["L_hist"]) != L_hist:
            raise ValueError("noisy/cond pre_dit produced inconsistent L_hist")
        if int(video_pre_cond["meta"]["mtm"]["L_pred"]) != L_pred:
            raise ValueError("noisy/cond pre_dit produced inconsistent L_pred")

        hist_tokens = video_pre_noisy["tokens"][:, :L_hist]
        noisy_tokens = video_pre_noisy["tokens"][:, L_hist:]
        cond_tokens = video_pre_cond["tokens"][:, L_hist:]
        merged_tokens = torch.cat([hist_tokens, noisy_tokens, cond_tokens], dim=1)

        hist_freqs = video_pre_noisy["freqs"][:L_hist]
        noisy_freqs = video_pre_noisy["freqs"][L_hist:]
        cond_freqs = video_pre_cond["freqs"][L_hist:]
        merged_freqs = torch.cat([hist_freqs, noisy_freqs, cond_freqs], dim=0)

        hist_tmod = video_pre_noisy["t_mod"][:, :L_hist]
        noisy_tmod = video_pre_noisy["t_mod"][:, L_hist:]
        cond_tmod = video_pre_cond["t_mod"][:, L_hist:]
        merged_tmod = torch.cat([hist_tmod, noisy_tmod, cond_tmod], dim=1)

        hist_cmask = video_pre_noisy["context_mask"][:, :L_hist]
        noisy_cmask = video_pre_noisy["context_mask"][:, L_hist:]
        cond_cmask = video_pre_cond["context_mask"][:, L_hist:]
        merged_cmask = torch.cat([hist_cmask, noisy_cmask, cond_cmask], dim=1)

        return merged_tokens, merged_freqs, merged_tmod, merged_cmask, L_hist, L_pred

    @torch.no_grad()
    def _build_idm_mtm_joint_attention_mask(
        self,
        L_hist: int,
        L_pred: int,
        action_seq_len: int,
        tokens_per_frame_pred: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Joint attention mask for `[history | noisy_pred | cond_pred | action]`.

        See module docstring for the visibility table.
        """
        noisy_start = L_hist
        noisy_end = L_hist + L_pred
        cond_start = noisy_end
        cond_end = noisy_end + L_pred
        total_video = cond_end
        total = total_video + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        # history <-> history (bidirectional).
        if L_hist > 0:
            mask[:L_hist, :L_hist] = True

        # noisy_pred -> history + noisy_pred(per-mode).
        if L_pred > 0:
            if L_hist > 0:
                mask[noisy_start:noisy_end, :L_hist] = True
            mask[noisy_start:noisy_end, noisy_start:noisy_end] = (
                self.video_expert.build_video_to_video_mask(
                    video_seq_len=L_pred,
                    video_tokens_per_frame=tokens_per_frame_pred,
                    device=device,
                )
            )

            # cond_pred -> history + cond_pred(per-mode).
            if L_hist > 0:
                mask[cond_start:cond_end, :L_hist] = True
            mask[cond_start:cond_end, cond_start:cond_end] = (
                self.video_expert.build_video_to_video_mask(
                    video_seq_len=L_pred,
                    video_tokens_per_frame=tokens_per_frame_pred,
                    device=device,
                )
            )

        # action <-> action, action -> history, action -> cond_pred.
        if action_seq_len > 0:
            mask[total_video:, total_video:] = True
            if L_hist > 0:
                mask[total_video:, :L_hist] = True
            if L_pred > 0:
                mask[total_video:, cond_start:cond_end] = True

        return mask

    # ------------------------------------------------------------------
    # Training loss (teacher-forcing + MTM history).
    # ------------------------------------------------------------------

    def training_loss(self, sample, tiled: bool = False):
        loss_total, loss_dict, _ = self._training_loss_core(sample, tiled=tiled, return_extras=False)
        return loss_total, loss_dict

    def training_loss_with_predictions(self, sample, tiled: bool = False):
        return self._training_loss_core(sample, tiled=tiled, return_extras=True)

    def _training_loss_core(self, sample, tiled: bool, return_extras: bool):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # Branch A: noisy pred video.
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )

        # Branch C: clean cond pred video (timestep 0).
        # Optionally corrupt the gt latent feeding the action cond branch — leaves
        # `input_latents` / `target_video` / `latents_noisy` (loss target) untouched.
        cond_input_latents = self._maybe_corrupt_cond_latents(input_latents)
        timestep_video_cond = torch.zeros_like(timestep_video)

        # Action branch.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Two pre_dit calls: same history (zero-timestep), different pred segments.
        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=False,
            history_long=inputs["history_long"],
            history_mid=inputs["history_mid"],
            history_short=inputs["history_short"],
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=cond_input_latents,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=False,
            history_long=inputs["history_long"],
            history_mid=inputs["history_mid"],
            history_short=inputs["history_short"],
        )
        merged_tokens, merged_freqs, merged_tmod, merged_cmask, L_hist, L_pred = (
            self._merge_pred_branches(video_pre_noisy, video_pre_cond)
        )
        tokens_per_frame_pred = int(video_pre_noisy["meta"]["mtm"]["tokens_per_frame_pred"])

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_idm_mtm_joint_attention_mask(
            L_hist=L_hist,
            L_pred=L_pred,
            action_seq_len=int(action_pre["tokens"].shape[1]),
            tokens_per_frame_pred=tokens_per_frame_pred,
            device=merged_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={"video": merged_tokens, "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": merged_freqs, "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre_noisy["context"], "mask": merged_cmask},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": merged_tmod, "action": action_pre["t_mod"]},
            mtm_history_seq_lens={"video": L_hist, "action": 0},
        )

        # Slice noisy_pred output (with its history prefix so post_dit can use the same
        # `pre_state` shape contract: the head reads `t_for_head = t[:, L_hist:]`).
        video_out = tokens_out["video"]
        noisy_branch_tokens = torch.cat(
            [video_out[:, :L_hist], video_out[:, L_hist : L_hist + L_pred]], dim=1
        )
        pred_video = self.video_expert.post_dit(noisy_branch_tokens, video_pre_noisy)
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

        if not return_extras:
            return loss_total, loss_dict, None

        # Recover x0_hat for IDM-probe alignment (FM: x0_hat = x_t - sigma * v_pred).
        sigma_v = (
            timestep_video.to(dtype=torch.float32)
            / float(self.train_video_scheduler.num_train_timesteps)
        ).clamp(0.0, 1.0)
        sigma_a = (
            timestep_action.to(dtype=torch.float32)
            / float(self.train_action_scheduler.num_train_timesteps)
        ).clamp(0.0, 1.0)
        sigma_v_b = sigma_v.to(dtype=latents_noisy.dtype).view(-1, 1, 1, 1, 1)
        sigma_a_b = sigma_a.to(dtype=noisy_action.dtype).view(-1, 1, 1)
        x0_hat_video = latents_noisy - sigma_v_b * pred_video
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
    # Inference: two-stage (video alone, then action against clean cache).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _video_only_forward(
        self,
        video_pre: Dict[str, Any],
        attn_mask_v: torch.Tensor,
    ) -> torch.Tensor:
        """Run the video expert blocks alone over `[history | pred]` tokens.

        Mirrors `WanVideoDiT.forward`'s block loop but uses the MTM-aware mask supplied
        by the caller. Used by Stage 1 of the two-stage inference.
        """
        x_tokens = video_pre["tokens"]
        for block in self.video_expert.blocks:
            x_tokens = block(
                x_tokens,
                video_pre["context"],
                video_pre["t_mod"],
                video_pre["freqs"],
                context_mask=video_pre["context_mask"],
                self_attn_mask=attn_mask_v,
            )
        return self.video_expert.post_dit(x_tokens, video_pre)

    @torch.no_grad()
    def _build_stage2_attention_mask(
        self,
        mtm_meta: Dict[str, Any],
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Inference-time joint mask for `[history | cond_pred | action]` (no noisy branch).

        Like `_build_mtm_joint_attention_mask` (parent) but with the action row also
        seeing the cond pred segment, matching training-time visibility.
        """
        L_long = int(mtm_meta["L_long"])
        L_mid = int(mtm_meta["L_mid"])
        L_curr = int(mtm_meta["L_curr"])
        L_pred = int(mtm_meta["L_pred"])
        tokens_per_frame_pred = int(mtm_meta["tokens_per_frame_pred"])
        L_hist = L_long + L_mid + L_curr
        video_seq_len = L_hist + L_pred
        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_mtm_self_attn_mask(
            L_long=L_long,
            L_mid=L_mid,
            L_curr=L_curr,
            L_pred=L_pred,
            tokens_per_frame_pred=tokens_per_frame_pred,
            device=device,
        )
        if action_seq_len > 0:
            mask[video_seq_len:, video_seq_len:] = True
            if L_hist > 0:
                mask[video_seq_len:, :L_hist] = True
            mask[video_seq_len:, L_hist:video_seq_len] = True
        return mask

    @torch.no_grad()
    def _denoise_video_with_history(
        self,
        latents_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
        num_inference_steps: int,
        sigma_shift: Optional[float],
    ) -> torch.Tensor:
        """Stage 1: denoise the pred latents alone, conditioned on MTM history."""
        infer_timesteps_video, infer_deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
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
            attn_mask_v = self.video_expert.build_mtm_self_attn_mask(
                L_long=int(mtm_meta["L_long"]),
                L_mid=int(mtm_meta["L_mid"]),
                L_curr=int(mtm_meta["L_curr"]),
                L_pred=int(mtm_meta["L_pred"]),
                tokens_per_frame_pred=int(mtm_meta["tokens_per_frame_pred"]),
                device=video_pre["tokens"].device,
            )
            pred_video = self._video_only_forward(video_pre, attn_mask_v)
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
        return latents_video

    @torch.no_grad()
    def _denoise_action_with_clean_video(
        self,
        latents_action: torch.Tensor,
        latents_video_clean: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
        num_inference_steps: int,
        sigma_shift: Optional[float],
    ) -> torch.Tensor:
        """Stage 2: prefill `[history | clean_pred]` cache and denoise action against it."""
        batch_size = latents_video_clean.shape[0]
        timestep_video_zero = torch.zeros(
            (batch_size,), dtype=latents_video_clean.dtype, device=self.device
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_video_clean,
            timestep=timestep_video_zero,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=False,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
        )
        mtm_meta = video_pre_cond["meta"]["mtm"]
        L_hist = int(mtm_meta["L_hist"])
        L_pred = int(mtm_meta["L_pred"])
        video_seq_len = L_hist + L_pred
        action_seq_len = int(latents_action.shape[1])

        attention_mask = self._build_stage2_attention_mask(
            mtm_meta=mtm_meta,
            action_seq_len=action_seq_len,
            device=video_pre_cond["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
            video_context_payload={
                "context": video_pre_cond["context"],
                "mask": video_pre_cond["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            mtm_history_seq_lens={"video": L_hist, "action": 0},
        )

        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
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
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
        return latents_action

    @torch.no_grad()
    def _prepare_context(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ):
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
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} "
                    f"and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

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
        del text_cfg_scale
        video_steps = int(num_video_inference_steps if num_video_inference_steps is not None else num_inference_steps)
        action_steps = int(num_action_inference_steps if num_action_inference_steps is not None else num_inference_steps)
        if action is not None:
            logger.warning(
                "`FastWAMMemoryIDM.infer_from_latents` ignores `action`; the model conditions "
                "action on its own predicted clean video, not on a GT action input."
            )

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
        context, context_mask = self._prepare_context(prompt, context, context_mask, proprio)

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

        # Stage 1.
        latents_video = self._denoise_video_with_history(
            latents_video=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=video_steps,
            sigma_shift=sigma_shift,
        )
        # Stage 2.
        latents_action = self._denoise_action_with_clean_video(
            latents_action=latents_action,
            latents_video_clean=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=action_steps,
            sigma_shift=sigma_shift,
        )

        pred_latents = latents_video.detach()
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
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
        del negative_prompt, text_cfg_scale, test_action_with_infer_action
        video_steps = int(num_video_inference_steps if num_video_inference_steps is not None else num_inference_steps)
        action_steps = int(num_action_inference_steps if num_action_inference_steps is not None else num_inference_steps)
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError(
                "`FastWAMMemoryIDM.infer_joint` requires `history_long/mid/short` "
                "(callers must provide history latents; this implementation does not "
                "auto-roll history)."
            )
        if action is not None:
            logger.warning(
                "`FastWAMMemoryIDM.infer_joint` ignores `action`; action is conditioned "
                "on the predicted clean video, not on a GT input."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                "`input_image` must be resized before infer; expected multiples of 16 but got "
                f"HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        self.eval()
        context, context_mask = self._prepare_context(prompt, context, context_mask, proprio)

        history_long = history_long.to(device=self.device, dtype=self.torch_dtype)
        history_mid = history_mid.to(device=self.device, dtype=self.torch_dtype)
        history_short = history_short.to(device=self.device, dtype=self.torch_dtype)

        latent_h = history_short.shape[3]
        latent_w = history_short.shape[4]
        latent_t = self.mtm_pred_size
        batch_size = history_short.shape[0]

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

        latents_video = self._denoise_video_with_history(
            latents_video=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=video_steps,
            sigma_shift=sigma_shift,
        )
        latents_action = self._denoise_action_with_clean_video(
            latents_action=latents_action,
            latents_video_clean=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=action_steps,
            sigma_shift=sigma_shift,
        )

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
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
        del input_image, negative_prompt, text_cfg_scale
        video_steps = int(num_video_inference_steps if num_video_inference_steps is not None else num_inference_steps)
        action_steps = int(num_action_inference_steps if num_action_inference_steps is not None else num_inference_steps)
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError("`FastWAMMemoryIDM.infer_action` requires `history_long/mid/short`.")

        self.eval()
        context, context_mask = self._prepare_context(prompt, context, context_mask, proprio)

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

        latents_video = self._denoise_video_with_history(
            latents_video=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=video_steps,
            sigma_shift=sigma_shift,
        )
        latents_action = self._denoise_action_with_clean_video(
            latents_action=latents_action,
            latents_video_clean=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=action_steps,
            sigma_shift=sigma_shift,
        )

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}


# -----------------------------------------------------------------------------
# Smoke test: shape correctness + backward, runnable via
#   `python -m fastwam.models.wan22.fastwam_memory_idm`.
# -----------------------------------------------------------------------------


def _smoke_test() -> None:
    print("[fastwam_memory_idm smoke] starting...")
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    from .fastwam_memory import MTMVideoDiT
    from .action_dit import ActionDiT

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
        fuse_vae_embedding_in_latents=False,
        action_conditioned=False,
        video_attention_mask_mode="first_frame_causal",
    )
    expert = MTMVideoDiT(
        multi_term_memory=True,
        mtm_history_sizes=(16, 2, 1),
        mtm_pred_size=2,
        mtm_amplify_history=False,
        mtm_zero_history_timestep=True,
        **cfg,
    ).to(device=device, dtype=dtype)

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

    from .fastwam_memory import MTMMoT
    mot = MTMMoT(mixtures={"video": expert, "action": action_expert}, mot_checkpoint_mixed_attn=False)

    # Construct a thin model directly (skip VAE / text encoder to keep the smoke fast).
    class _DummyVAEModel:
        z_dim = 16

    class _DummyVAE:
        model = _DummyVAEModel()
        upsampling_factor = 16
        temporal_downsample_factor = 4

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    model = FastWAMMemoryIDM(
        video_expert=expert,
        action_expert=action_expert,
        mot=mot,
        vae=_DummyVAE(),
        text_encoder=None,
        tokenizer=None,
        text_dim=cfg["text_dim"],
        proprio_dim=None,
        device=device,
        torch_dtype=dtype,
        multi_term_memory=True,
        mtm_history_sizes=(16, 2, 1),
        mtm_pred_size=2,
        mtm_amplify_history=False,
        mtm_zero_history_timestep=True,
    )

    B, C, H, W = 1, 16, 16, 16
    long_size, mid_size, curr_size = model.mtm_history_sizes
    pred_size = model.mtm_pred_size

    # --- Check 1: training mask layout & visibility. --------------------------
    history_long = torch.randn(B, C, long_size, H, W, device=device, dtype=dtype)
    history_mid = torch.randn(B, C, mid_size, H, W, device=device, dtype=dtype)
    history_short = torch.randn(B, C, curr_size, H, W, device=device, dtype=dtype)
    pred = torch.randn(B, C, pred_size, H, W, device=device, dtype=dtype)
    context = torch.randn(B, 4, cfg["text_dim"], device=device, dtype=dtype)
    context_mask = torch.ones(B, 4, dtype=torch.bool, device=device)
    timestep = torch.tensor([500.0], device=device, dtype=dtype)
    pre_n = expert.pre_dit(
        x=pred, timestep=timestep, context=context, context_mask=context_mask,
        history_long=history_long, history_mid=history_mid, history_short=history_short,
    )
    L_hist = int(pre_n["meta"]["mtm"]["L_hist"])
    L_pred = int(pre_n["meta"]["mtm"]["L_pred"])
    tokens_per_frame_pred = int(pre_n["meta"]["mtm"]["tokens_per_frame_pred"])
    L_a = action_horizon
    mask = model._build_idm_mtm_joint_attention_mask(
        L_hist=L_hist, L_pred=L_pred, action_seq_len=L_a,
        tokens_per_frame_pred=tokens_per_frame_pred, device=device,
    )
    assert mask.shape == (L_hist + 2 * L_pred + L_a, L_hist + 2 * L_pred + L_a)
    noisy_lo, noisy_hi = L_hist, L_hist + L_pred
    cond_lo, cond_hi = noisy_hi, noisy_hi + L_pred
    a_lo = cond_hi
    assert mask[:L_hist, L_hist:].any() == False, "history must NOT see anything past hist"
    assert mask[noisy_lo:noisy_hi, cond_lo:cond_hi].any() == False, "noisy must NOT see cond"
    assert mask[cond_lo:cond_hi, noisy_lo:noisy_hi].any() == False, "cond must NOT see noisy"
    assert mask[a_lo:, cond_lo:cond_hi].all(), "action must see cond"
    assert mask[a_lo:, :L_hist].all(), "action must see history"
    assert mask[a_lo:, noisy_lo:noisy_hi].any() == False, "action must NOT see noisy"
    assert mask[a_lo:, a_lo:].all(), "action self-attention must be full"
    assert mask[noisy_lo:noisy_hi, :L_hist].all(), "noisy must see history"
    assert mask[cond_lo:cond_hi, :L_hist].all(), "cond must see history"
    print(f"  [check 1] training mask `(L_hist={L_hist}, L_pred={L_pred}, L_a={L_a})`  OK")

    # --- Check 2: training_loss forward + backward. ---------------------------
    sample = {
        "input_latents_precomputed": pred.clone(),
        "context": context.clone(),
        "context_mask": context_mask.clone(),
        "history_long": history_long.clone(),
        "history_mid": history_mid.clone(),
        "history_short": history_short.clone(),
        "action": torch.randn(B, action_horizon, action_dim, device=device, dtype=dtype),
    }
    model.train()
    loss, loss_dict = model.training_loss(sample)
    loss.backward()
    print(f"  [check 2] training_loss={loss.item():.4f}, dict={loss_dict}  OK")

    # --- Check 3: stage2 inference mask shape. --------------------------------
    s2_mask = model._build_stage2_attention_mask(
        mtm_meta=pre_n["meta"]["mtm"],
        action_seq_len=L_a,
        device=device,
    )
    assert s2_mask.shape == (L_hist + L_pred + L_a, L_hist + L_pred + L_a)
    Sv = L_hist + L_pred
    assert s2_mask[Sv:, :L_hist].all(), "stage2: action must see history"
    assert s2_mask[Sv:, L_hist:Sv].all(), "stage2: action must see cond_pred"
    assert s2_mask[:L_hist, L_hist:].any() == False, "stage2: history must not see pred or action"
    print(f"  [check 3] stage2 mask `(Sv={Sv}, L_a={L_a})`  OK")

    print("[fastwam_memory_idm smoke] all checks passed.")


if __name__ == "__main__":
    _smoke_test()
