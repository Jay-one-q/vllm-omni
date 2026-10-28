# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""SenCache backend lifecycle.

Validation scope: Cosmos3 (``nvidia/Cosmos3-Nano``) request-mode diffusion with
eager execution and no parameter sharding. CFG-parallel is allowed because the
gate depends only on the noisy latent and timestep, which are replicated across
guidance ranks, so every branch evaluates the same score without a cross-branch
collective. Unsupported combinations fail at startup instead of silently
degrading.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.sencache.calibrate import ProfileAccumulator
from vllm_omni.diffusion.cache.sencache.config import SenCacheConfig
from vllm_omni.diffusion.cache.sencache.hook import (
    SenCacheHook,
    apply_sen_cache_hook,
)
from vllm_omni.diffusion.cache.sencache.profile import SensitivityProfile
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)

SEN_CACHE_BOUNDARY = "cosmos3.gen_stack"
SUPPORTED_PIPELINES = {
    "Cosmos3OmniDiffusersPipeline",
    "Cosmos3OmniPipeline",
}


def _cfg_parallel_world_size() -> int:
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_classifier_free_guidance_world_size,
        )

        return int(get_classifier_free_guidance_world_size())
    except Exception:
        return 1


class SenCacheBackend(CacheBackend):
    """Sensitivity-aware caching for the Cosmos3 GEN stack."""

    def __init__(self, config: DiffusionCacheConfig | None = None):
        super().__init__(config if config is not None else DiffusionCacheConfig())
        self.sen_config: SenCacheConfig | None = None
        self.hook: SenCacheHook | None = None

    def enable(self, pipeline: Any) -> None:
        pipeline_type = pipeline.__class__.__name__
        if pipeline_type not in SUPPORTED_PIPELINES:
            raise ValueError(
                f"sen_cache does not support pipeline {pipeline_type}. "
                f"Supported: {sorted(SUPPORTED_PIPELINES)}"
            )
        cfg_world_size = _cfg_parallel_world_size()
        if cfg_world_size > 1:
            # The score only reads the noisy latent and timestep, which are
            # replicated across guidance ranks, so every branch evaluates the
            # same decision without a cross-branch collective. Voting across
            # CFG ranks is avoided because branch dispatch may be uneven.
            logger.info(
                "SenCache enabling with CFG-parallel world size %d; "
                "skip decisions agree structurally (conditioning-independent gate)",
                cfg_world_size,
            )

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError(f"{pipeline_type} has no transformer for sen_cache")

        sen_config = SenCacheConfig.from_diffusion_cache_config(self.config)
        profile: SensitivityProfile | None = None
        calibrator: ProfileAccumulator | None = None
        if sen_config.calibrating:
            metadata = {
                "model_id": getattr(pipeline, "model_name_or_path", None) or pipeline_type,
                "boundary": SEN_CACHE_BOUNDARY,
                "pipeline": pipeline_type,
            }
            calibrator = ProfileAccumulator(
                time_max=sen_config.time_max,
                metadata=metadata,
                num_bins=sen_config.calibration_time_bins,
            )
            logger.info(
                "SenCache calibration mode: every step runs full; samples from all ranks "
                "are merged into a single shared .npz under %s",
                sen_config.calibration_output,
            )
        else:
            if sen_config.profile_path is None:
                raise ValueError(
                    "sen_cache requires cache_config.sen_profile_path, or set "
                    "sen_calibration_output to create a profile"
                )
            profile = SensitivityProfile.load(sen_config.profile_path)
            logger.info(
                "SenCache profile loaded: %s (bins=%d, model_id=%s)",
                sen_config.profile_path,
                profile.time_bins.shape[0],
                profile.metadata.get("model_id"),
            )

        hook = apply_sen_cache_hook(
            transformer,
            sen_config,
            profile,
            current_step_callback=lambda: getattr(pipeline, "current_step_index", None),
            num_steps_callback=lambda: getattr(pipeline, "num_timesteps", None),
            calibrator=calibrator,
            calibration_output=sen_config.calibration_output,
        )
        pipeline._cache_context_factory = hook.cache_context
        pipeline._sen_cache_hook = hook

        self.sen_config = sen_config
        self.hook = hook
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        hook = getattr(pipeline, "_sen_cache_hook", None)
        if not isinstance(hook, SenCacheHook) or not self.enabled:
            if verbose:
                logger.warning("SenCache is not enabled; nothing to refresh")
            return
        previous_full = hook.state.full_count
        previous_skip = hook.state.skip_count
        hook.refresh(pipeline.transformer, num_inference_steps)
        # The SeaCache-style context factory is installed once; keep it bound
        # to the live hook after a refresh/re-enable.
        pipeline._cache_context_factory = hook.cache_context
        if verbose:
            total = previous_full + previous_skip
            ratio = (previous_skip / total) if total else 0.0
            logger.info(
                "SenCache request summary: full=%d skip=%d skip_ratio=%.3f (next steps=%s)",
                previous_full,
                previous_skip,
                ratio,
                num_inference_steps,
            )


__all__ = [
    "SEN_CACHE_BOUNDARY",
    "SUPPORTED_PIPELINES",
    "SenCacheBackend",
]
