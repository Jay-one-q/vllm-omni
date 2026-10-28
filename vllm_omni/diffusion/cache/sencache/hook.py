# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Sensitivity-aware caching hook.

The gate follows SenCache's first-order model::

    S = alpha_x(t_ref) * rms(x_t - x_ref) + alpha_t(t_ref) * |t - t_ref|

and skips the cacheable transformer stack when ``S`` stays below the current
tolerance. ``alpha_x``/``alpha_t`` come from a calibrated sensitivity profile;
the online part only measures the actual displacement since the last executed
step, which is what makes the decision sample specific.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.sencache.calibrate import (
    ProfileAccumulator,
    merge_into_file,
    profile_issues,
)
from vllm_omni.diffusion.cache.sencache.config import SenCacheConfig
from vllm_omni.diffusion.cache.sencache.profile import SensitivityProfile
from vllm_omni.diffusion.cache.sencache.state import SenCacheState
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext, get_extractor
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sp_group,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook

logger = init_logger(__name__)

_LATENT_KEY = "sencache_latent"
_TIMESTEP_KEY = "sencache_timestep"


def _cfg_group_info() -> tuple[int, int]:
    """Return ``(world_size, rank)`` for the CFG group, or ``(1, 0)``."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_cfg_group,
            is_cfg_group_initialized,
        )

        if not is_cfg_group_initialized():
            return 1, 0
        group = get_cfg_group()
        return int(group.world_size), int(group.rank_in_group)
    except Exception:
        return 1, 0


def _is_parameter_sharded(module: torch.nn.Module) -> bool:
    """Detect FSDP/HSDP runtimes whose collective order must not be skipped."""
    for submodule in module.modules():
        module_type = type(submodule)
        if callable(getattr(submodule, "_get_fsdp_state", None)):
            return True
        if module_type.__name__ == "FullyShardedDataParallel" and module_type.__module__.startswith(
            "torch.distributed.fsdp"
        ):
            return True
        for parameter in submodule.parameters(recurse=False):
            parameter_type = type(parameter)
            if (
                parameter_type.__name__ == "FlatParameter"
                and parameter_type.__module__.startswith("torch.distributed.fsdp")
            ) or (
                parameter_type.__name__ == "DTensor"
                and parameter_type.__module__.startswith("torch.distributed.tensor")
            ):
                return True
    return False


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().pow(2).mean().sqrt())


def _per_sample_rms_max(tensor: torch.Tensor) -> float:
    flat = tensor.detach().float().flatten(1)
    return float(flat.pow(2).mean(dim=1).sqrt().max())


def _first_video(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("SenCache received an empty model output")
        return output[0]
    return output


class SenCacheHook(ModelHook):
    """Root-transformer hook implementing the SenCache gate."""

    _HOOK_NAME = "sen_cache"

    def __init__(
        self,
        config: SenCacheConfig,
        profile: SensitivityProfile | None,
        *,
        current_step_callback: Callable[[], int | None] | None = None,
        num_steps_callback: Callable[[], int | None] | None = None,
        extractor_fn: Callable[..., CacheContext] | None = None,
        calibrator: ProfileAccumulator | None = None,
        calibration_output: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.profile = profile
        self.current_step_callback = current_step_callback
        self.num_steps_callback = num_steps_callback
        self.extractor_fn = extractor_fn
        self.calibrator = calibrator
        self.calibration_output = calibration_output
        self.state = SenCacheState()
        self._active_branch: str | None = None
        self._num_steps: int | None = None
        self._warned: set[str] = set()
        self._calibration_prev: dict[str, Any] | None = None
        self._calibration_done = False
        self._calibration_path = self._resolve_calibration_path(calibration_output)

    # ------------------------------------------------------------------ setup
    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.extractor_fn is None:
            self.extractor_fn = get_extractor(type(module))
        if _is_parameter_sharded(module):
            raise RuntimeError(
                "SenCache does not support FSDP/HSDP parameter sharding yet; "
                "run without --enable-hsdp / FSDP for the current validation scope"
            )
        if self.profile is None and not self.config.calibrating:
            raise RuntimeError(
                "SenCache requires a sensitivity profile. Provide sen_profile_path "
                "or run with sen_calibration_output to generate one."
            )
        return module

    @contextmanager
    def cache_context(self, name: str) -> Iterator[None]:
        previous = self._active_branch
        self._active_branch = name or "default"
        try:
            yield
        finally:
            self._active_branch = previous

    def refresh(self, module: torch.nn.Module, num_inference_steps: int | None = None) -> None:
        self.state.reset()
        self._calibration_prev = None
        self._calibration_done = False
        if num_inference_steps is not None:
            self._num_steps = int(num_inference_steps)

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.refresh(module)
        return module

    # ------------------------------------------------------------------ forward
    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        if self.extractor_fn is None:
            raise RuntimeError("SenCache extractor was not initialized")
        if torch.is_grad_enabled():
            self._warn_once("SenCache is inference-only; autograd-enabled calls run in full.")
            return module._omni_original_forward(*args, **kwargs)  # type: ignore[attr-defined]
        if args:
            self._warn_once("SenCache requires keyword-only transformer calls; running in full.")
            return module._omni_original_forward(*args, **kwargs)  # type: ignore[attr-defined]

        ctx = self.extractor_fn(module, *args, **kwargs)
        extra = ctx.extra_states or {}
        latent = extra.get(_LATENT_KEY)
        timestep = extra.get(_TIMESTEP_KEY)
        if not isinstance(latent, torch.Tensor) or timestep is None:
            self._warn_once(
                "SenCache extractor did not expose the raw latent/timestep; running the full stack."
            )
            return ctx.postprocess(self._run_full(ctx))

        step = self._resolve_step()
        if step is None:
            self._warn_once("SenCache requires denoise step metadata; running the full stack.")
            return ctx.postprocess(self._run_full(ctx))

        num_steps = self._resolve_num_steps()
        t_value = float(timestep.detach().flatten()[0].item())
        if not math.isfinite(t_value):
            self._warn_once("SenCache received a non-finite timestep; running the full stack.")
            return ctx.postprocess(self._run_full(ctx))

        branch = self._active_branch or "cond"
        calibrating = self.calibrator is not None

        if calibrating:
            # Calibration needs exact reference outputs, so never skip.
            prediction = ctx.postprocess(self._run_full(ctx))
            self._record_calibration(module, kwargs, step, num_steps, latent, t_value, prediction)
            self._refresh_reference(step, latent, t_value)
            return prediction

        should_compute = self._resolve_gate(step, num_steps, latent, t_value)
        cached_residual = self.state.residual_for(branch, ctx.hidden_states)
        if not should_compute and cached_residual is not None:
            self.state.skip_count += 1
            return ctx.postprocess(ctx.hidden_states + cached_residual)

        output = self._run_full(ctx)
        self.state.branch_residuals[branch] = (output - ctx.hidden_states).detach()
        self.state.full_count += 1
        if should_compute:
            self._refresh_reference(step, latent, t_value)
        else:
            # A branch without a usable residual ran full while the other
            # branches reused the old reference. Drop the reference so the next
            # step re-synchronizes every branch with a full execution.
            self._warn_once(f"SenCache had no reusable residual for branch {branch!r}; running full.")
            self.state.reference_latent = None
            self.state.reference_timestep = None
        return ctx.postprocess(output)

    # ------------------------------------------------------------------ helpers
    def _run_full(self, ctx: CacheContext) -> torch.Tensor:
        outputs = ctx.run_transformer_blocks()
        if not outputs:
            raise RuntimeError("SenCache extractor returned no transformer outputs")
        return outputs[0]

    def _resolve_step(self) -> int | None:
        if self.current_step_callback is None:
            return None
        value = self.current_step_callback()
        if value is None:
            return None
        return int(value.item()) if isinstance(value, torch.Tensor) else int(value)

    def _resolve_num_steps(self) -> int:
        if self._num_steps is not None:
            return self._num_steps
        if self.num_steps_callback is not None:
            value = self.num_steps_callback()
            if value is not None:
                return int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        return 0

    def _resolve_gate(self, step: int, num_steps: int, latent: torch.Tensor, t_value: float) -> bool:
        state = self.state
        if state.last_gate_step != step:
            state.reset_step_gates()
            state.last_gate_step = step

        cached = state.step_gate.get(step)
        if cached is not None:
            return cached

        if state.reference_latent is None or state.reference_timestep is None:
            should_compute = True
        elif (
            state.reference_latent.shape != latent.shape
            or state.reference_latent.device != latent.device
            or state.reference_latent.dtype != latent.dtype
        ):
            self._warn_once("SenCache reference latent shape/dtype changed; refreshing the cache.")
            should_compute = True
        elif not bool(torch.isfinite(latent).all()):
            self._warn_once("SenCache received a non-finite latent; refreshing the cache.")
            should_compute = True
        else:
            rms_dx = _per_sample_rms_max(latent - state.reference_latent)
            dt = abs(t_value - state.reference_timestep)
            score = state.alpha_x * rms_dx + state.alpha_t * dt
            threshold = self.config.threshold_for_step(step, num_steps)
            should_compute = score > threshold or state.consecutive_reuses >= self.config.max_consecutive_reuses

        should_compute = self._synchronize_decision(latent.device, should_compute)
        state.step_gate[step] = should_compute
        if not should_compute:
            # K counts denoising steps, not per-branch forwards.
            state.consecutive_reuses += 1
        return should_compute

    def _refresh_reference(self, step: int, latent: torch.Tensor, t_value: float) -> None:
        if self.state.last_ref_step == step:
            return
        self.state.reference_latent = latent.detach().clone()
        self.state.reference_timestep = t_value
        if self.profile is not None:
            self.state.alpha_x, self.state.alpha_t = self.profile.lookup(t_value)
        self.state.consecutive_reuses = 0
        self.state.last_ref_step = step

    def _synchronize_decision(self, device: torch.device, should_compute: bool) -> bool:
        """Keep skip/compute identical across SP ranks that share collectives.

        Only sequence-parallel ranks are voted. CFG-parallel ranks are not
        voted on purpose: they may dispatch a different number of transformer
        calls per step, and a cross-branch vote would then deadlock. Branch
        agreement is instead structural: the score only reads the noisy latent
        and timestep, which are replicated across CFG ranks, so every branch
        evaluates the same ``S`` on identical inputs.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return should_compute
        if not model_parallel_is_initialized():
            return should_compute
        sp_group = get_sp_group()
        if sp_group.world_size <= 1:
            return should_compute
        flag = torch.tensor([1 if should_compute else 0], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=sp_group.device_group)
        return bool(flag.item())

    def _warn_once(self, message: str) -> None:
        if message not in self._warned:
            logger.warning(message)
            self._warned.add(message)

    # ------------------------------------------------------------- calibration
    def _record_calibration(
        self,
        module: torch.nn.Module,
        kwargs: dict[str, Any],
        step: int,
        num_steps: int,
        latent: torch.Tensor,
        t_value: float,
        output: torch.Tensor,
    ) -> None:
        if self.calibrator is None or self._calibration_done:
            return
        cfg_world_size, cfg_rank = _cfg_group_info()
        if cfg_world_size > 1 and cfg_rank != 0:
            # Calibrate the conditional branch only; other CFG ranks run full
            # but must not mix an uncond curve into the same profile.
            return
        previous = self._calibration_prev
        if previous is not None and previous["step"] == step:
            # Later branches of the same step must not overwrite the record.
            return
        video = _first_video(output).detach()
        if previous is not None and previous["step"] == step - 1:
            prev_latent = previous["latent"]
            prev_t = previous["timestep"]
            prev_video = previous["video"]
            if prev_latent.shape == latent.shape and prev_video.shape == video.shape:
                try:
                    dx_rms = _rms(latent - prev_latent)
                    dt = abs(t_value - prev_t)
                    alpha_x = None
                    alpha_t = None
                    if dx_rms > 1e-12:
                        probe_x = self._probe(module, kwargs, latent, prev_t)
                        alpha_x = _rms(probe_x - prev_video) / dx_rms
                    if dt > 1e-12:
                        probe_t = self._probe(module, kwargs, prev_latent, t_value)
                        alpha_t = _rms(probe_t - prev_video) / dt
                    self.calibrator.add(prev_t, alpha_x, alpha_t)
                except Exception as error:  # calibration must never break serving
                    logger.warning("SenCache calibration probe failed at step %d: %s", step, error)

        self._calibration_prev = {
            "step": step,
            "latent": latent.detach().clone(),
            "timestep": t_value,
            "video": video,
        }
        if num_steps > 0 and step >= num_steps - 1:
            self._write_calibration()

    def _probe(
        self,
        module: torch.nn.Module,
        kwargs: dict[str, Any],
        latent: torch.Tensor,
        timestep_value: float,
    ) -> torch.Tensor:
        probe_kwargs = dict(kwargs)
        probe_kwargs["hidden_states"] = latent
        current = probe_kwargs.get("timestep")
        if isinstance(current, torch.Tensor):
            probe_kwargs["timestep"] = torch.full_like(current, timestep_value)
        else:
            probe_kwargs["timestep"] = torch.tensor(
                [timestep_value], device=latent.device, dtype=torch.float32
            )
        with torch.inference_mode():
            output = module._omni_original_forward(**probe_kwargs)  # type: ignore[attr-defined]
        return _first_video(output).detach()

    @staticmethod
    def _resolve_calibration_path(output: str | None) -> str | None:
        if output is None:
            return None
        base = output[:-4] if output.endswith(".npz") else output
        return f"{base}.npz"

    def _reset_accumulator(self) -> None:
        """Start a fresh per-request accumulator after a successful merge."""
        if self.calibrator is None:
            return
        self.calibrator = ProfileAccumulator(
            time_max=self.calibrator.time_max,
            metadata=dict(self.calibrator.metadata),
            num_bins=self.calibrator.num_bins,
        )

    def _write_calibration(self) -> None:
        if self.calibrator is None or self._calibration_path is None or self._calibration_done:
            return
        try:
            merged = merge_into_file(self._calibration_path, self.calibrator)
            profile = merged.finalize()
            occupied = merged.count > 0 if merged.count is not None else None
            issues = profile_issues(profile, occupied=occupied)
            for issue in issues:
                logger.warning("SenCache calibration profile check: %s", issue)
            if not issues:
                sampled = int(occupied.sum()) if occupied is not None else -1
                logger.info(
                    "SenCache calibration profile check passed: %s (bins=%d, sampled=%d)",
                    self._calibration_path,
                    profile.time_bins.shape[0],
                    sampled,
                )
            self._reset_accumulator()
            logger.info("SenCache merged calibration artifact: %s", self._calibration_path)
        except Exception as error:
            logger.warning(
                "SenCache failed to merge calibration artifact %s: %s",
                self._calibration_path,
                error,
            )
        finally:
            self._calibration_done = True
            self._calibration_prev = None


def apply_sen_cache_hook(
    module: torch.nn.Module,
    config: SenCacheConfig,
    profile: SensitivityProfile | None,
    *,
    current_step_callback: Callable[[], int | None] | None = None,
    num_steps_callback: Callable[[], int | None] | None = None,
    extractor_fn: Callable[..., CacheContext] | None = None,
    calibrator: ProfileAccumulator | None = None,
    calibration_output: str | None = None,
) -> SenCacheHook:
    """Register the SenCache hook on a transformer module."""
    registry = HookRegistry.get_or_create(module)
    hook = SenCacheHook(
        config,
        profile,
        current_step_callback=current_step_callback,
        num_steps_callback=num_steps_callback,
        extractor_fn=extractor_fn,
        calibrator=calibrator,
        calibration_output=calibration_output,
    )
    registry.register_hook(SenCacheHook._HOOK_NAME, hook)
    return hook
