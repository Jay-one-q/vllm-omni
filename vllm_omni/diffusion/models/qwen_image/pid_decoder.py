# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PiD (Pixel Diffusion) decoder for Qwen-Image pipeline.

Thin wrapper around ``vllm_omni.diffusion.pid`` -- only the Qwen-Image-specific
net config and default paths live here. Other backbone pipelines (Flux, SD3,
etc.) can add their own equally-thin wrappers.

All PiD network code is now self-contained in ``vllm_omni.diffusion.pid``.
No external PiD library dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.pid import (
    PID_SAMPLING_CONFIG,
    QWENIMAGE_PID_NET_CONFIG,
    PidInferenceModel,
    load_pid_checkpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PidDecodeConfig:
    """Configuration for the PiD super-resolution decode path.

    When ``OmniDiffusionConfig.pid_decode`` is ``None`` or ``enabled`` is
    ``False``, the pipeline falls back to the standard VAE decoder.
    """

    enabled: bool = False
    # Path to the PiD distilled checkpoint (.pth).
    checkpoint_path: str = ""
    # Local directory containing gemma-2-2b-it weights (required).
    gemma_model_path: str = ""
    # Super-resolution factor applied to the LDM output resolution.
    scale: int = 4
    # Number of distilled SDE sampling steps (4 for the distilled checkpoint).
    num_steps: int = 4
    # Base RNG seed. The pipeline may override this per request.
    seed: int = 0
    # Noise level injected into the LQ latent. 0.0 means the clean x_0 latent.
    degrade_sigma: float = 0.0
    # Compute precision preset: "bfloat16" (default, matches distilled
    # checkpoint training), "float16" (fp16 autocast), or "float32" (pure
    # fp32 forward, disables autocast). The tensor container is always
    # float32; non-float32 values enable autocast for matmuls only.
    precision: str = "bfloat16"


# ---------------------------------------------------------------------------
# PidDecoder
# ---------------------------------------------------------------------------


class PidDecoder:
    """Decode an LDM ``x_0`` latent into a high-resolution RGB image via PiD.

    The PiD model (PidNet + Gemma text encoder) is loaded lazily on the first
    :meth:`decode` call and stays resident in GPU memory for the lifetime of
    this object, so subsequent requests reuse it without re-loading weights.

    Uses the Qwen-Image backbone config from ``vllm_omni.diffusion.pid.config``.
    """

    def __init__(self, config: PidDecodeConfig):
        self._config = config
        self._model: PidInferenceModel | None = None

    # -- lazy loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        cfg = self._config
        logger.info("Loading PiD model from %s ...", cfg.checkpoint_path)

        self._model = PidInferenceModel(
            net_kwargs=dict(QWENIMAGE_PID_NET_CONFIG),
            gemma_model_id=cfg.gemma_model_path,
            sampling_overrides=dict(PID_SAMPLING_CONFIG),
        )
        load_pid_checkpoint(self._model, cfg.checkpoint_path)
        self._model.eval()
        self._model.to("cuda")
        logger.info("PiD model loaded (PidNet + Gemma text encoder) and resident.")

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        lq_latent: torch.Tensor,
        caption: str | list[str],
        output_size: tuple[int, int],
        degrade_sigma: float | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Run PiD decoding.

        Args:
            lq_latent: LDM ``x_0`` latent, shape ``(B, 16, zH, zW)``.
                Qwen-Image's per-channel normalized latent matches the PiD
                training frame format, so no conversion is needed.
            caption: Original text prompt. A single ``str`` is broadcast to
                the whole batch; a ``list[str]`` must match ``lq_latent``'s
                batch size.
            output_size: Target pixel resolution ``(H_pixel, W_pixel)``,
                e.g. ``(4096, 4096)`` for a 4x super-resolution of a 1024^2
                latent.
            degrade_sigma: Noise level injected into the LQ latent.
                ``None`` to use the config default (0.0 for clean x_0).
            num_steps: Number of distilled SDE sampling steps.
                ``None`` to use the config default.
            seed: RNG seed. ``None`` to use the config default.

        Returns:
            Tensor of shape ``(B, 3, H_pixel, W_pixel)`` with values in
            ``[-1, 1]``.
        """
        self._ensure_loaded()

        return self._model.generate_samples_from_batch(
            lq_latent=lq_latent,
            caption=caption,
            output_size=output_size,
            degrade_sigma=(
                degrade_sigma
                if degrade_sigma is not None
                else self._config.degrade_sigma
            ),
            num_steps=num_steps or self._config.num_steps,
            seed=seed if seed is not None else self._config.seed,
        )
