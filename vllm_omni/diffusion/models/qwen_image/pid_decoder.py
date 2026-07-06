# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PiD (Pixel Diffusion) decoder integration for vllm-omni.

Thin glue layer that wraps the external ``pid`` package and decodes a
latent-diffusion-model (LDM) ``x_0`` latent into a high-resolution RGB image.

Design notes
------------
* PiD is treated as an external library (``pip install -e /path/to/PiD``).
  No PiD source code is copied into vllm-omni.
* Two known pitfalls of calling PiD from a foreign codebase are handled here:

  1. ``load_model_from_checkpoint`` validates ``config_file`` with
     ``os.path.isfile`` against the current working directory, which breaks
     when PiD is pip-installed and the CWD is vllm-omni. We resolve the
     absolute path via ``__file__`` of the imported config module.
  2. PiD downloads the Gemma-2-2b-it text encoder from HuggingFace by default.
     To load it from a local directory instead (without modifying PiD source),
     we monkey-patch ``_TEXT_ENCODER_DICT`` before model construction.

* The model is lazily loaded on the first ``decode`` call and then kept
  resident in GPU memory for all subsequent requests (service-friendly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default PiD checkpoint/experiment for the Qwen-Image 2k→4k distilled path.
# Users override these via PidDecodeConfig at construction time.
_QWENIMAGE_PID_CHECKPOINT = (
    "checkpoints/"
    "PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth"
)
_QWENIMAGE_PID_EXPERIMENT = (
    "PiD_res2kto4k_sr4x_official_qwenimage_distill_4step"
)


@dataclass(frozen=True)
class PidDecodeConfig:
    """Configuration for the PiD super-resolution decode path.

    When ``OmniDiffusionConfig.pid_decode`` is ``None`` or ``enabled`` is
    ``False``, the pipeline falls back to the standard VAE decoder.
    """

    enabled: bool = False
    # Path to the PiD distilled checkpoint (.pth).
    checkpoint_path: str = _QWENIMAGE_PID_CHECKPOINT
    # PiD experiment name (selects network architecture + config).
    experiment: str = _QWENIMAGE_PID_EXPERIMENT
    # Local directory containing gemma-2-2b-it weights. When ``None``, PiD
    # downloads the encoder from HuggingFace (requires network access).
    local_gemma_path: str | None = None
    # Super-resolution factor applied to the LDM output resolution.
    scale: int = 4
    # Number of distilled SDE sampling steps (4 for the distilled checkpoint).
    num_steps: int = 4
    # Base RNG seed. The pipeline may override this per request.
    seed: int = 0
    # Noise level injected into the LQ latent. 0.0 means the clean x_0 latent.
    degrade_sigma: float = 0.0
    # Load EMA weights into the regular (non-EMA) model params.
    load_ema_to_reg: bool = True


# ---------------------------------------------------------------------------
# PiD internal-path patchers (run before model construction)
# ---------------------------------------------------------------------------

def _resolve_config_path() -> str:
    """Return the absolute path to PiD's config file, regardless of CWD.

    ``pid._ext.imaginaire.utils.config_helper.get_config_module`` checks
    ``os.path.isfile(config_file)`` against the process CWD. When PiD is
    pip-installed and vllm-omni is the CWD, the relative path
    ``"pid/_src/configs/pid/config.py"`` does not exist, so the assertion
    fails. We sidestep this by handing PiD the *absolute* path obtained from
    the imported module's ``__file__`` attribute.
    """
    import pid._src.configs.pid.config as _mod  # noqa: PLC0415 (lazy import)

    return _mod.__file__


def _patch_text_encoder_path(local_gemma_path: str) -> None:
    """Override PiD's Gemma model id to a local path.

    PiD maps a text-encoder name to a HuggingFace model id via
    ``_TEXT_ENCODER_DICT`` in ``pid._src.models.pixeldit_model``. The id is
    then passed to ``AutoTokenizer.from_pretrained`` /
    ``AutoModelForCausalLM.from_pretrained``, both of which accept a local
    directory path. Replacing the dict entry therefore forces PiD to load
    Gemma from disk without touching PiD source code.
    """
    import pid._src.models.pixeldit_model as _pm  # noqa: PLC0415 (lazy import)

    old = _pm._TEXT_ENCODER_DICT.get("gemma-2-2b-it")
    _pm._TEXT_ENCODER_DICT["gemma-2-2b-it"] = local_gemma_path
    logger.info("PiD text encoder path patched: %s -> %s", old, local_gemma_path)


# ---------------------------------------------------------------------------
# PidDecoder
# ---------------------------------------------------------------------------

class PidDecoder:
    """Decode an LDM ``x_0`` latent into a high-resolution RGB image via PiD.

    The underlying PiD model (PidNet + Gemma text encoder + VAE) is loaded
    lazily on the first :meth:`decode` call and stays resident in GPU memory
    for the lifetime of this object, so subsequent requests reuse it without
    re-loading weights.
    """

    def __init__(
        self,
        checkpoint_path: str,
        experiment: str = _QWENIMAGE_PID_EXPERIMENT,
        local_gemma_path: str | None = None,
        load_ema_to_reg: bool = True,
    ):
        self._checkpoint_path = checkpoint_path
        self._experiment = experiment
        self._local_gemma_path = local_gemma_path
        self._load_ema_to_reg = load_ema_to_reg
        self._model = None
        self._config = None

    # -- lazy loading -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # Gemma path must be patched BEFORE model construction, since
        # ``load_model_from_checkpoint`` triggers ``_load_text_encoder``.
        if self._local_gemma_path is not None:
            _patch_text_encoder_path(self._local_gemma_path)

        from pid._src.utils.model_loader import load_model_from_checkpoint  # noqa: PLC0415

        logger.info(
            "Loading PiD model (experiment=%s, checkpoint=%s) ...",
            self._experiment,
            self._checkpoint_path,
        )
        self._model, self._config = load_model_from_checkpoint(
            experiment_name=self._experiment,
            checkpoint_path=self._checkpoint_path,
            config_file=_resolve_config_path(),
            enable_fsdp=False,
            experiment_opts=[],
            strict=False,
            load_ema_to_reg=self._load_ema_to_reg,
        )
        self._model.eval()
        logger.info(
            "PiD model loaded (PidNet + Gemma text encoder + VAE) and resident."
        )

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        lq_latent: torch.Tensor,
        caption: str | list[str],
        output_size: tuple[int, int],
        degrade_sigma: float = 0.0,
        num_steps: int = 4,
        seed: int = 0,
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
                e.g. ``(4096, 4096)`` for a 4x super-resolution of a 1024²
                latent.
            degrade_sigma: Noise level injected into the LQ latent. ``0.0``
                for the clean ``x_0``.
            num_steps: Number of distilled SDE sampling steps.
            seed: RNG seed.

        Returns:
            Tensor of shape ``(B, 3, H_pixel, W_pixel)`` with values in
            ``[-1, 1]``.
        """
        self._ensure_loaded()

        # Normalize caption to a list whose length matches the batch.
        if isinstance(caption, str):
            captions = [caption] * lq_latent.shape[0]
        else:
            captions = list(caption)
            if len(captions) != lq_latent.shape[0]:
                raise ValueError(
                    f"caption batch size {len(captions)} does not match "
                    f"lq_latent batch size {lq_latent.shape[0]}"
                )

        data_batch = {
            self._model.config.input_caption_key: captions,
            "LQ_latent": lq_latent.to(dtype=torch.bfloat16, device="cuda"),
            "degrade_sigma": torch.tensor(
                [degrade_sigma] * lq_latent.shape[0],
                device="cuda",
                dtype=torch.float32,
            ),
        }

        samples = self._model.generate_samples_from_batch(
            data_batch,
            cfg_scale=1.0,
            num_steps=num_steps,
            seed=seed,
            image_size=output_size,
        )
        # samples: (B, 3, 1, H, W) -> squeeze time dim -> (B, 3, H, W)
        return samples.squeeze(2).float().clamp(-1, 1)
