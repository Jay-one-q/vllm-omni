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
import os
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

def _patch_config_module_resolver() -> None:
    
    import pid._ext.imaginaire.utils.config_helper as _ch  # noqa: PLC0415
    import pid._src.utils.model_loader as _ml  # noqa: PLC0415

    # 幂等：已 patch 过则跳过
    if getattr(_ch.get_config_module, "_pid_patched", False):
        return

    # 通过 import 拿到 PiD config 模块的规范名，不依赖文件路径。
    import pid._src.configs.pid.config as _cfg_mod  # noqa: PLC0415
    _expected_module_name = _cfg_mod.__name__  # "pid._src.configs.pid.config"
    # PiD 默认传入的相对路径，以及它的归一化形式（兼容 Windows 反斜杠）。
    _expected_rel_paths = {
        "pid/_src/configs/pid/config.py",
        "pid\\_src\\configs\\pid\\config.py",
    }

    _orig_get_config_module = _ch.get_config_module

    def _patched_get_config_module(config_file: str) -> str:
        # 归一化比较：支持相对路径、绝对路径、正反斜杠。
        normalized = config_file.replace("\\", "/").rstrip("/")
        if (
            normalized in _expected_rel_paths
            or normalized.endswith("pid/_src/configs/pid/config.py")
        ):
            return _expected_module_name
        # 其它路径走原逻辑（保持 PiD 对自定义 config 的支持）
        return _orig_get_config_module(config_file)

    _patched_get_config_module._pid_patched = True  # type: ignore[attr-defined]

    # patch 两处引用：config_helper 原始定义 + model_loader 的 from-import
    _ch.get_config_module = _patched_get_config_module
    _ml.get_config_module = _patched_get_config_module
    logger.info("PiD get_config_module patched to resolve config by module name.")


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

        # Config resolver 必须先 patch：load_model_from_checkpoint 内部
        # 调用 get_config_module，patch 后不依赖 CWD 即可解析 PiD config。
        _patch_config_module_resolver()

        # Gemma path 必须在模型构造前 patch，因为 load_model_from_checkpoint
        # 会触发 _load_text_encoder。
        if self._local_gemma_path is not None:
            _patch_text_encoder_path(self._local_gemma_path)

        from pid._src.utils.model_loader import load_model_from_checkpoint  # noqa: PLC0415

        # checkpoint 转绝对路径，避免任何 CWD 假设。
        abs_checkpoint = os.path.abspath(self._checkpoint_path)

        logger.info(
            "Loading PiD model (experiment=%s, checkpoint=%s) ...",
            self._experiment,
            abs_checkpoint,
        )
        # config_file 用 PiD 默认的相对路径即可，_patch_config_module_resolver
        # 会把它直接映射到已安装的 pid._src.configs.pid.config 模块名。
        self._model, self._config = load_model_from_checkpoint(
            experiment_name=self._experiment,
            checkpoint_path=abs_checkpoint,
            config_file="pid/_src/configs/pid/config.py",
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
