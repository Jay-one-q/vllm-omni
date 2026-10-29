# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""SenCache backend lifecycle.

Supported scopes:

- Cosmos3 (``nvidia/Cosmos3-Nano``): block-residual boundary over the GEN stack.
  The pipeline exposes ``current_step_index`` and installs
  ``_cache_context_factory``, so branch names come from the pipeline.
- Wan2.2 T2V/I2V: full-prediction boundary. Wan's forward is monolithic, so the
  complete transformer output is cached per branch. Branch names come from the
  ``CFGParallelMixin`` stamp (``transformer.cfg_branch``) and the denoise step
  from the forward context. Both MoE experts (``transformer`` / ``transformer_2``)
  are hooked; they are active on disjoint timestep ranges, so each keeps its own
  request state while sharing one calibration accumulator.

CFG-parallel is allowed because the gate depends only on the noisy latent and
timestep, which are replicated across guidance ranks, so every branch evaluates
the same score without a cross-branch collective. Unsupported combinations
(FSDP/HSDP, pipeline parallelism) fail at startup instead of silently degrading.
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

COSMOS3_BOUNDARY = "cosmos3.gen_stack"
WAN_BOUNDARY = "wan2_2.full_prediction"
SEN_CACHE_BOUNDARY = COSMOS3_BOUNDARY

COSMOS3_PIPELINES = {
    "Cosmos3OmniDiffusersPipeline",
    "Cosmos3OmniPipeline",
}
WAN_PIPELINES = {
    "Wan22Pipeline",
    "Wan22I2VPipeline",
}
SUPPORTED_PIPELINES = COSMOS3_PIPELINES | WAN_PIPELINES


def _cfg_parallel_world_size() -> int:
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_classifier_free_guidance_world_size,
        )

        return int(get_classifier_free_guidance_world_size())
    except Exception:
        return 1


def _pipeline_parallel_world_size() -> int:
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_pp_group

        return int(get_pp_group().world_size)
    except Exception:
        return 1


def _denoise_step(pipeline: Any) -> int | None:
    """Resolve the denoise step, falling back to the forward context.

    Wan2.2 does not expose ``current_step_index`` but records each step through
    ``DenoiseProgressMixin.record_denoise_step``.
    """
    value = getattr(pipeline, "current_step_index", None)
    if value is not None:
        return value
    try:
        from vllm_omni.diffusion.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            return get_forward_context().denoise_step_idx
    except Exception:
        return None
    return None


def _cfg_branch(pipeline: Any) -> str | None:
    """Read the CFG branch stamped by ``CFGParallelMixin`` for this call."""
    for name in ("transformer", "transformer_2"):
        module = getattr(pipeline, name, None)
        value = getattr(module, "cfg_branch", None)
        if value is not None:
            return value
    return None


class SenCacheBackend(CacheBackend):
    """Sensitivity-aware caching for Cosmos3 and Wan2.2."""

    def __init__(self, config: DiffusionCacheConfig | None = None):
        super().__init__(config if config is not None else DiffusionCacheConfig())
        self.sen_config: SenCacheConfig | None = None
        self.hook: SenCacheHook | None = None
        self.hooks: list[SenCacheHook] = []
        self._cache_context_hook: SenCacheHook | None = None

    def enable(self, pipeline: Any) -> None:
        pipeline_type = pipeline.__class__.__name__
        if pipeline_type not in SUPPORTED_PIPELINES:
            raise ValueError(
                f"sen_cache does not support pipeline {pipeline_type}. "
                f"Supported: {sorted(SUPPORTED_PIPELINES)}"
            )
        pp_world_size = _pipeline_parallel_world_size()
        if pp_world_size > 1:
            raise RuntimeError(
                "SenCache does not support pipeline parallelism yet; "
                f"got pp world size {pp_world_size}"
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

        is_wan = pipeline_type in WAN_PIPELINES
        boundary = WAN_BOUNDARY if is_wan else COSMOS3_BOUNDARY
        mode = "full_prediction" if is_wan else "block_residual"

        modules = [transformer]
        if is_wan:
            second = getattr(pipeline, "transformer_2", None)
            if second is not None:
                modules.append(second)

        sen_config = SenCacheConfig.from_diffusion_cache_config(self.config)
        profile: SensitivityProfile | None = None
        calibrator: ProfileAccumulator | None = None
        if sen_config.calibrating:
            metadata = {
                "model_id": getattr(pipeline, "model_name_or_path", None) or pipeline_type,
                "boundary": boundary,
                "pipeline": pipeline_type,
                "mode": mode,
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

        def step_callback() -> int | None:
            return _denoise_step(pipeline)

        def num_steps_callback() -> int | None:
            return getattr(pipeline, "num_timesteps", None)

        def branch_callback() -> str | None:
            return _cfg_branch(pipeline) if is_wan else None

        hooks = [
            apply_sen_cache_hook(
                module,
                sen_config,
                profile,
                current_step_callback=step_callback,
                num_steps_callback=num_steps_callback,
                branch_callback=branch_callback,
                calibrator=calibrator,
                calibration_output=sen_config.calibration_output,
            )
            for module in modules
        ]

        self.sen_config = sen_config
        self.hooks = hooks
        self.hook = hooks[0]
        self._cache_context_hook = None if is_wan else hooks[0]
        pipeline._sen_cache_hook = hooks[0]
        if is_wan:
            pipeline._sen_cache_hooks = hooks
        else:
            pipeline._cache_context_factory = hooks[0].cache_context
        logger.info(
            "SenCache enabled for %s (boundary=%s, modules=%d)",
            pipeline_type,
            boundary,
            len(hooks),
        )
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        hooks = getattr(pipeline, "_sen_cache_hooks", None)
        if not hooks:
            hook = getattr(pipeline, "_sen_cache_hook", None)
            hooks = [hook] if isinstance(hook, SenCacheHook) else []
        if not hooks or not self.enabled:
            if verbose:
                logger.warning("SenCache is not enabled; nothing to refresh")
            return

        previous_full = sum(hook.state.full_count for hook in hooks)
        previous_skip = sum(hook.state.skip_count for hook in hooks)
        for hook in hooks:
            hook.refresh(pipeline.transformer, num_inference_steps)
        if self._cache_context_hook is not None:
            # The SeaCache-style context factory is installed once; keep it bound
            # to the live hook after a refresh/re-enable.
            pipeline._cache_context_factory = self._cache_context_hook.cache_context
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
    "COSMOS3_BOUNDARY",
    "SEN_CACHE_BOUNDARY",
    "SUPPORTED_PIPELINES",
    "WAN_BOUNDARY",
    "SenCacheBackend",
]
