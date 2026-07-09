# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only PiD model.

Merges PixelDiTModel + PidModel + PidDistillModel into a single flat class
with no imaginaire dependency. Only ``generate_samples_from_batch`` is exposed.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn

from .config import PID_SAMPLING_CONFIG
from .pid_net import PidNet
from .text_encoder import GemmaTextEncoder

logger = logging.getLogger(__name__)


class PidInferenceModel(nn.Module):
    """Inference-only PiD model (model-agnostic).

    Args:
        net_kwargs: Passed directly to PidNet.__init__.
        gemma_model_id: Local path or HF ID for gemma-2-2b-it.
        sampling_overrides: Optional overrides for PID_SAMPLING_CONFIG.
    """

    def __init__(
        self,
        net_kwargs: dict,
        gemma_model_id: str,
        sampling_overrides: dict | None = None,
    ):
        super().__init__()
        self.net = PidNet(**net_kwargs)
        self.text_encoder = GemmaTextEncoder(gemma_model_id)

        samp = dict(PID_SAMPLING_CONFIG)
        if sampling_overrides:
            samp.update(sampling_overrides)
        self._cfg = type("Cfg", (), samp)()

        self.autocast_dtype = torch.bfloat16
        logger.info(
            "PidInferenceModel: net params=%s",
            f"{sum(p.numel() for p in self.net.parameters()):,}",
        )

    # ------------------------------------------------------------------
    # Velocity <-> x0 conversion
    # ------------------------------------------------------------------

    def _velocity_to_x0(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """velocity -> x0: x0 = x_t - t * v"""
        s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
        t_shaped = t.double().view(*s)
        return (x_t.double() - t_shaped * v.double()).to(x_t.dtype)

    # ------------------------------------------------------------------
    # Timestep schedule
    # ------------------------------------------------------------------

    def _get_t_list(self, device, num_steps: int | None = None) -> torch.Tensor:
        target = num_steps or self._cfg.student_sample_steps
        full_t = torch.tensor(
            self._cfg.student_t_list, device=device, dtype=torch.float32
        )
        if target != len(full_t) - 1:
            indices = (
                torch.linspace(0, len(full_t) - 1, target + 1).round().long()
            )
            return full_t[indices]
        return full_t

    # ------------------------------------------------------------------
    # SDE sample loop
    # ------------------------------------------------------------------

    def _sample_loop(
        self,
        noise: torch.Tensor,
        t_list: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: torch.Tensor,
        degrade_sigma: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        B = noise.shape[0]
        timescale = self._cfg.fm_timescale
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype
            else nullcontext()
        )
        x = noise

        with autocast_ctx:
            for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
                t_cur_batch = t_cur.expand(B)
                t_scaled = t_cur_batch * timescale

                v_pred = self.net(
                    x, t_scaled, caption_embs,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )

                if t_next.item() > 0:
                    x0_pred = self._velocity_to_x0(x, v_pred, t_cur_batch)
                    eps_infer = torch.randn(
                        x0_pred.shape, device=x0_pred.device,
                        dtype=x0_pred.dtype, generator=generator,
                    )
                    x = (1.0 - t_next) * x0_pred + t_next * eps_infer
                else:
                    x = self._velocity_to_x0(x, v_pred, t_cur_batch)

        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        lq_latent: torch.Tensor,        # (B, C_lq, zH, zW)
        caption: str | list[str],
        output_size: tuple[int, int],    # (H, W) pixel output
        degrade_sigma: float = 0.0,
        num_steps: int = 4,
        seed: int = 0,
    ) -> torch.Tensor:
        """Run PiD decode.

        Returns:
            (B, 3, H, W) tensor in [-1, 1].
        """
        if isinstance(caption, str):
            caption = [caption]
        B = len(caption)

        caption_embs = self.text_encoder.encode(caption)
        caption_embs = caption_embs.to(device="cuda", dtype=torch.bfloat16)

        lq_latent = lq_latent.to(device="cuda", dtype=torch.bfloat16)
        degrade_sigma_tensor = torch.full(
            (B,), float(degrade_sigma), device="cuda", dtype=torch.float32
        )

        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        img_h, img_w = output_size
        noise = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)

        effective_steps = num_steps or self._cfg.student_sample_steps

        if effective_steps == 1:
            t_student = torch.full(
                (B,), self._cfg.student_t_list[0],
                device="cuda", dtype=torch.float32,
            )
            t_scaled = t_student * self._cfg.fm_timescale
            v = self.net(
                noise, t_scaled, caption_embs,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            x0 = self._velocity_to_x0(noise, v, t_student)
        else:
            t_list = self._get_t_list(torch.device("cuda"), effective_steps)
            x0 = self._sample_loop(
                noise, t_list, caption_embs, lq_latent,
                degrade_sigma_tensor, generator=gen,
            )

        return x0.clamp(-1, 1)
