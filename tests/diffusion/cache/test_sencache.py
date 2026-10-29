# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.cache.sencache import (
    SenCacheBackend,
    SenCacheConfig,
    SenCacheHook,
    SensitivityProfile,
    apply_sen_cache_hook,
)
from vllm_omni.diffusion.cache.sencache.calibrate import (
    ProfileAccumulator,
    merge_files,
    profile_issues,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_METADATA = {"model_id": "tiny-cosmos3", "boundary": "cosmos3.gen_stack"}


class TinyCosmos3Transformer(torch.nn.Module):
    """Tiny model whose GEN stack doubles its input (so residuals are visible)."""

    def __init__(self) -> None:
        super().__init__()
        self.stack_calls = 0

    def _run_gen_layers(self, gen: torch.Tensor) -> torch.Tensor:
        self.stack_calls += 1
        return gen * 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del timestep
        if kwargs:
            raise TypeError(f"Unexpected tiny transformer kwargs: {sorted(kwargs)}")
        offset = 0.0 if text_ids is None else float(text_ids.flatten()[0])
        gen = hidden_states.movedim(1, -1).flatten(1, 3)
        return self._run_gen_layers(gen) + offset


def _offset(text_ids: torch.Tensor | None) -> float:
    return 0.0 if text_ids is None else float(text_ids.flatten()[0])


def _extract_tiny_context(
    module: TinyCosmos3Transformer,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    text_ids: torch.Tensor | None = None,
    **kwargs,
) -> CacheContext:
    if kwargs:
        raise TypeError(f"Unexpected tiny transformer kwargs: {sorted(kwargs)}")
    gen = hidden_states.movedim(1, -1).flatten(1, 3)
    offset = _offset(text_ids)
    return CacheContext(
        modulated_input=gen,
        hidden_states=gen,
        encoder_hidden_states=None,
        temb=torch.zeros_like(gen[:, 0]),
        run_transformer_blocks=lambda: (module._run_gen_layers(gen) + offset,),
        postprocess=lambda output: output,
        extra_states={
            "sencache_latent": hidden_states,
            "sencache_timestep": timestep,
        },
    )


def _latent(value: float) -> torch.Tensor:
    return torch.full((1, 2, 2, 2, 2), value)


def _profile(alpha_x: float, alpha_t: float, bins: int = 8, time_max: float = 1000.0) -> SensitivityProfile:
    return SensitivityProfile(
        time_bins=np.linspace(0.0, 1.0, bins),
        alpha_x=np.full(bins, alpha_x),
        alpha_t=np.full(bins, alpha_t),
        time_max=time_max,
        metadata={
            **_METADATA,
            "time_max": time_max,
            "schema_version": 1,
        },
    )


def _apply(
    transformer: TinyCosmos3Transformer,
    *,
    profile: SensitivityProfile | None,
    config: SenCacheConfig | None = None,
    metadata: SimpleNamespace | None = None,
    calibrator: ProfileAccumulator | None = None,
    calibration_output: str | None = None,
):
    metadata = metadata or SimpleNamespace(step=0, num_steps=5)
    hook = apply_sen_cache_hook(
        transformer,
        config or SenCacheConfig(threshold_start=1.0, threshold_main=1.0, max_consecutive_reuses=10),
        profile,
        current_step_callback=None if metadata is None else (lambda: metadata.step),
        num_steps_callback=lambda: metadata.num_steps,
        extractor_fn=_extract_tiny_context,
        calibrator=calibrator,
        calibration_output=calibration_output,
    )
    return hook, metadata


def _run_step(
    transformer: TinyCosmos3Transformer,
    hook,
    metadata: SimpleNamespace,
    *,
    step: int,
    t: float,
    value: float,
    branch: str = "cond",
) -> torch.Tensor:
    metadata.step = step
    text_ids = torch.tensor([0.0 if branch == "cond" else 10.0])
    with torch.inference_mode(), hook.cache_context(branch):
        return transformer(hidden_states=_latent(value), timestep=torch.tensor([t]), text_ids=text_ids)


def test_config_validation() -> None:
    config = SenCacheConfig()
    assert config.threshold_for_step(0, 10) == config.threshold_start
    assert config.threshold_for_step(2, 10) == config.threshold_main
    with pytest.raises(ValueError, match="threshold_main"):
        SenCacheConfig(threshold_main=-1.0)
    with pytest.raises(ValueError, match="start_fraction"):
        SenCacheConfig(start_fraction=1.5)
    with pytest.raises(ValueError, match="max_consecutive_reuses"):
        SenCacheConfig(max_consecutive_reuses=-1)
    with pytest.raises(ValueError, match="calibration_time_bins"):
        SenCacheConfig(calibration_time_bins=4)


def test_profile_roundtrip_and_nearest_fill(tmp_path) -> None:
    accumulator = ProfileAccumulator(time_max=1000.0, metadata=dict(_METADATA), num_bins=8)
    accumulator.add(1000.0, alpha_x=2.0, alpha_t=0.5)
    accumulator.add(0.0, alpha_x=1.0, alpha_t=0.25)
    profile = accumulator.finalize()
    assert profile.lookup(1000.0) == (2.0, 0.5)
    assert profile.lookup(0.0) == (1.0, 0.25)
    # Empty interior bins borrow the nearest occupied bin.
    assert profile.lookup(500.0)[0] in {1.0, 2.0}

    path = tmp_path / "profile.npz"
    profile.save(path)
    loaded = SensitivityProfile.load(path)
    assert np.allclose(loaded.alpha_x, profile.alpha_x)
    assert np.allclose(loaded.alpha_t, profile.alpha_t)
    assert loaded.time_max == 1000.0


def test_first_step_computes_then_reuses() -> None:
    transformer = TinyCosmos3Transformer()
    hook, metadata = _apply(transformer, profile=_profile(0.0, 0.0))
    first = _run_step(transformer, hook, metadata, step=0, t=1000.0, value=1.0)
    assert transformer.stack_calls == 1
    assert torch.allclose(first, torch.full_like(first, 2.0))

    second = _run_step(transformer, hook, metadata, step=1, t=900.0, value=1.5)
    assert transformer.stack_calls == 1
    assert hook.state.skip_count == 1
    # Cached residual is f(x_ref) - x_ref = x_ref, applied to the new latent.
    assert torch.allclose(second, torch.full_like(second, 2.5))


def test_max_consecutive_reuses_forces_refresh() -> None:
    transformer = TinyCosmos3Transformer()
    config = SenCacheConfig(threshold_start=1.0, threshold_main=1.0, max_consecutive_reuses=2)
    hook, metadata = _apply(transformer, profile=_profile(0.0, 0.0), config=config)
    for step in range(5):
        _run_step(transformer, hook, metadata, step=step, t=1000.0 - step * 100.0, value=1.0)
    # Fresh compute at step 0 and step 3 (after two consecutive reuses).
    assert transformer.stack_calls == 2
    assert hook.state.full_count == 2
    assert hook.state.skip_count == 3


def test_branches_share_gate_but_keep_their_own_residual() -> None:
    transformer = TinyCosmos3Transformer()
    hook, metadata = _apply(transformer, profile=_profile(0.0, 0.0))
    _run_step(transformer, hook, metadata, step=0, t=1000.0, value=1.0, branch="cond")
    _run_step(transformer, hook, metadata, step=0, t=1000.0, value=1.0, branch="uncond")
    assert transformer.stack_calls == 2

    cond = _run_step(transformer, hook, metadata, step=1, t=900.0, value=1.5, branch="cond")
    uncond = _run_step(transformer, hook, metadata, step=1, t=900.0, value=1.5, branch="uncond")
    # Both branches reuse the step gate: no new stack execution.
    assert transformer.stack_calls == 2
    assert torch.allclose(cond, torch.full_like(cond, 2.5))
    assert torch.allclose(uncond, torch.full_like(uncond, 12.5))


def test_refresh_resets_state_and_counts() -> None:
    transformer = TinyCosmos3Transformer()
    hook, metadata = _apply(transformer, profile=_profile(0.0, 0.0))
    _run_step(transformer, hook, metadata, step=0, t=1000.0, value=1.0)
    _run_step(transformer, hook, metadata, step=1, t=900.0, value=1.0)
    hook.refresh(transformer, num_inference_steps=4)
    assert hook.state.skip_count == 0
    _run_step(transformer, hook, metadata, step=0, t=1000.0, value=1.0)
    assert transformer.stack_calls == 2


def test_missing_step_metadata_runs_full() -> None:
    transformer = TinyCosmos3Transformer()
    hook = apply_sen_cache_hook(
        transformer,
        SenCacheConfig(threshold_start=1.0, threshold_main=1.0),
        _profile(0.0, 0.0),
        current_step_callback=None,
        num_steps_callback=lambda: 5,
        extractor_fn=_extract_tiny_context,
    )
    with torch.inference_mode():
        transformer(hidden_states=_latent(1.0), timestep=torch.tensor([1000.0]))
        transformer(hidden_states=_latent(1.0), timestep=torch.tensor([900.0]))
    assert transformer.stack_calls == 2
    assert hook.state.skip_count == 0


def test_calibration_writes_accumulator(tmp_path) -> None:
    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, num_steps=3)
    accumulator = ProfileAccumulator(time_max=1000.0, metadata=dict(_METADATA), num_bins=8)
    output = tmp_path / "calib.npz"
    hook, metadata = _apply(
        transformer,
        profile=None,
        config=SenCacheConfig(
            threshold_start=1.0,
            threshold_main=1.0,
            calibration_output=str(output),
        ),
        metadata=metadata,
        calibrator=accumulator,
        calibration_output=str(output),
    )
    for step, (t, value) in enumerate([(1000.0, 1.0), (900.0, 1.5), (800.0, 2.0)]):
        _run_step(transformer, hook, metadata, step=step, t=t, value=value)

    artifact = tmp_path / "calib.npz"
    assert artifact.is_file()
    loaded = ProfileAccumulator.load(artifact)
    assert loaded.count.sum() >= 1.0
    # The tiny model ignores the timestep, so alpha_t is 0 and alpha_x is 2.
    occupied = loaded.count > 0
    assert np.allclose(loaded.alpha_t_sum[occupied] / loaded.count[occupied], 0.0)
    assert np.allclose(loaded.alpha_x_sum[occupied] / loaded.count[occupied], 2.0)


def test_calibration_merges_into_one_artifact(tmp_path) -> None:
    """Two "workers" writing the same path must accumulate, not overwrite."""
    output = tmp_path / "calib.npz"
    per_run = 2.0  # steps 1 and 2 each contribute one sample pair
    for _ in range(2):
        transformer = TinyCosmos3Transformer()
        metadata = SimpleNamespace(step=0, num_steps=3)
        accumulator = ProfileAccumulator(time_max=1000.0, metadata=dict(_METADATA), num_bins=8)
        hook, metadata = _apply(
            transformer,
            profile=None,
            config=SenCacheConfig(
                threshold_start=1.0,
                threshold_main=1.0,
                calibration_output=str(output),
            ),
            metadata=metadata,
            calibrator=accumulator,
            calibration_output=str(output),
        )
        for step, (t, value) in enumerate([(1000.0, 1.0), (900.0, 1.5), (800.0, 2.0)]):
            _run_step(transformer, hook, metadata, step=step, t=t, value=value)

    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert not list(tmp_path.glob("*.tmp.*.npz"))
    loaded = ProfileAccumulator.load(output)
    assert loaded.count.sum() == 2.0 * per_run
    occupied = loaded.count > 0
    # Duplicated identical samples leave the mean unchanged.
    assert np.allclose(loaded.alpha_x_sum[occupied] / loaded.count[occupied], 2.0)


def test_calibration_accumulates_across_requests(tmp_path) -> None:
    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, num_steps=3)
    accumulator = ProfileAccumulator(time_max=1000.0, metadata=dict(_METADATA), num_bins=8)
    output = tmp_path / "calib.npz"
    hook, metadata = _apply(
        transformer,
        profile=None,
        config=SenCacheConfig(
            threshold_start=1.0,
            threshold_main=1.0,
            calibration_output=str(output),
        ),
        metadata=metadata,
        calibrator=accumulator,
        calibration_output=str(output),
    )
    for _ in range(2):
        for step, (t, value) in enumerate([(1000.0, 1.0), (900.0, 1.5), (800.0, 2.0)]):
            _run_step(transformer, hook, metadata, step=step, t=t, value=value)
        hook.refresh(transformer, num_inference_steps=3)

    assert ProfileAccumulator.load(output).count.sum() == 4.0


def test_profile_issues_flags_degenerate_curves() -> None:
    assert profile_issues(_profile(1.0, 0.5)) == []
    issues = profile_issues(_profile(1.0, 0.0))
    assert any("alpha_t" in issue for issue in issues)
    issues = profile_issues(_profile(0.0, 0.5))
    assert any("alpha_x" in issue for issue in issues)


def test_merge_calibration_artifacts(tmp_path) -> None:
    paths = []
    for index, alpha in enumerate((2.0, 4.0)):
        accumulator = ProfileAccumulator(time_max=1000.0, metadata=dict(_METADATA), num_bins=8)
        accumulator.add(1000.0, alpha_x=alpha, alpha_t=1.0)
        path = tmp_path / f"part-{index}.npz"
        accumulator.save(path)
        paths.append(path)

    merged_path = tmp_path / "merged.npz"
    merged = merge_files(paths, merged_path)
    profile = merged.finalize()
    assert profile.lookup(1000.0) == (3.0, 1.0)
    assert SensitivityProfile.load(merged_path).lookup(1000.0) == (3.0, 1.0)


class Cosmos3OmniDiffusersPipeline:
    def __init__(self) -> None:
        self.transformer = TinyCosmos3Transformer()
        self._current_step_index: int | None = 0
        self._num_timesteps: int | None = 5

    @property
    def current_step_index(self) -> int | None:
        return self._current_step_index

    @property
    def num_timesteps(self) -> int | None:
        return self._num_timesteps


def test_backend_enable_and_refresh(tmp_path) -> None:
    profile_path = tmp_path / "profile.npz"
    _profile(0.5, 0.5).save(profile_path)
    cache_config = DiffusionCacheConfig(sen_profile_path=str(profile_path))

    backend = get_cache_backend("sen_cache", cache_config)
    assert isinstance(backend, SenCacheBackend)

    pipeline = Cosmos3OmniDiffusersPipeline()
    backend.enable(pipeline)
    assert backend.is_enabled()
    assert callable(pipeline._cache_context_factory)
    assert pipeline._sen_cache_hook is backend.hook

    backend.refresh(pipeline, num_inference_steps=7)
    assert backend.hook is not None
    assert backend.hook._num_steps == 7


def test_backend_allows_cfg_parallel(tmp_path, monkeypatch) -> None:
    from vllm_omni.diffusion.cache.sencache import backend as backend_module

    profile_path = tmp_path / "profile.npz"
    _profile(0.5, 0.5).save(profile_path)
    monkeypatch.setattr(backend_module, "_cfg_parallel_world_size", lambda: 2)

    backend = SenCacheBackend(DiffusionCacheConfig(sen_profile_path=str(profile_path)))
    pipeline = Cosmos3OmniDiffusersPipeline()
    backend.enable(pipeline)
    assert backend.is_enabled()


def test_calibration_is_skipped_on_non_zero_cfg_rank(tmp_path, monkeypatch) -> None:
    from vllm_omni.diffusion.cache.sencache import hook as hook_module

    monkeypatch.setattr(hook_module, "_cfg_group_info", lambda: (2, 1))
    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, num_steps=3)
    accumulator = ProfileAccumulator(time_max=1000.0, metadata=dict(_METADATA), num_bins=8)
    output = tmp_path / "calib.npz"
    hook, metadata = _apply(
        transformer,
        profile=None,
        config=SenCacheConfig(threshold_start=1.0, threshold_main=1.0, calibration_output=str(output)),
        metadata=metadata,
        calibrator=accumulator,
        calibration_output=str(output),
    )
    for step, value in enumerate([1.0, 1.5, 2.0]):
        _run_step(transformer, hook, metadata, step=step, t=1000.0 - step * 100.0, value=value)

    assert accumulator.count.sum() == 0.0
    assert not (tmp_path / "calib.npz").exists()


def test_synchronize_decision_without_distributed() -> None:
    transformer = TinyCosmos3Transformer()
    hook, _ = _apply(transformer, profile=_profile(0.0, 0.0))
    assert hook._synchronize_decision(torch.device("cpu"), False) is False
    assert hook._synchronize_decision(torch.device("cpu"), True) is True


def test_backend_rejects_unknown_pipeline(tmp_path) -> None:
    profile_path = tmp_path / "profile.npz"
    _profile(0.5, 0.5).save(profile_path)
    backend = SenCacheBackend(DiffusionCacheConfig(sen_profile_path=str(profile_path)))
    with pytest.raises(ValueError, match="does not support pipeline"):
        backend.enable(SimpleNamespace(transformer=TinyCosmos3Transformer()))


def test_backend_requires_profile_or_calibration() -> None:
    backend = SenCacheBackend(DiffusionCacheConfig())
    with pytest.raises(ValueError, match="sen_profile_path"):
        backend.enable(Cosmos3OmniDiffusersPipeline())


class WanTransformer3DModel(torch.nn.Module):
    """Tiny Wan-like transformer: monolithic forward, full-output caching."""

    def __init__(self) -> None:
        super().__init__()
        self.stack_calls = 0
        self.cfg_branch = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        intermediate_tensors: object | None = None,
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
    ):
        del timestep, encoder_hidden_states_image, intermediate_tensors, return_dict, attention_kwargs
        self.stack_calls += 1
        offset = 0.0 if encoder_hidden_states is None else float(encoder_hidden_states.flatten()[0])
        return (hidden_states * 2 + offset,)


def _run_wan(
    transformer: WanTransformer3DModel,
    metadata: SimpleNamespace,
    *,
    step: int,
    t: float,
    value: float,
    encoder: float = 0.0,
):
    metadata.step = step
    with torch.inference_mode():
        return transformer(
            hidden_states=_latent(value),
            timestep=torch.tensor([t]),
            encoder_hidden_states=torch.tensor([encoder]),
        )


def _apply_wan(
    transformer: WanTransformer3DModel,
    *,
    metadata: SimpleNamespace,
    branch_state: dict[str, str] | None = None,
) -> SenCacheHook:
    return apply_sen_cache_hook(
        transformer,
        SenCacheConfig(threshold_start=1.0, threshold_main=1.0, max_consecutive_reuses=10),
        _profile(0.0, 0.0),
        current_step_callback=lambda: metadata.step,
        num_steps_callback=lambda: metadata.num_steps,
        branch_callback=(lambda: branch_state["branch"]) if branch_state is not None else None,
    )


def test_full_prediction_reuses_whole_output() -> None:
    transformer = WanTransformer3DModel()
    metadata = SimpleNamespace(step=0, num_steps=5)
    hook = _apply_wan(transformer, metadata=metadata)

    first = _run_wan(transformer, metadata, step=0, t=1000.0, value=1.0)
    second = _run_wan(transformer, metadata, step=1, t=900.0, value=5.0)

    assert transformer.stack_calls == 1
    assert hook.state.full_count == 1
    assert hook.state.skip_count == 1
    assert torch.allclose(second[0], first[0])
    assert torch.allclose(second[0], torch.full_like(second[0], 2.0))


def test_full_prediction_keeps_branches_separate() -> None:
    transformer = WanTransformer3DModel()
    metadata = SimpleNamespace(step=0, num_steps=5)
    branch_state = {"branch": "positive"}
    hook = _apply_wan(transformer, metadata=metadata, branch_state=branch_state)

    branch_state["branch"] = "positive"
    _run_wan(transformer, metadata, step=0, t=1000.0, value=1.0, encoder=0.0)
    branch_state["branch"] = "negative"
    _run_wan(transformer, metadata, step=0, t=1000.0, value=1.0, encoder=10.0)

    branch_state["branch"] = "positive"
    positive = _run_wan(transformer, metadata, step=1, t=900.0, value=5.0, encoder=0.0)
    branch_state["branch"] = "negative"
    negative = _run_wan(transformer, metadata, step=1, t=900.0, value=5.0, encoder=10.0)

    assert transformer.stack_calls == 2
    assert hook.state.skip_count == 2
    assert set(hook.state.branch_predictions) == {"cond", "uncond"}
    assert torch.allclose(positive[0], torch.full_like(positive[0], 2.0))
    assert torch.allclose(negative[0], torch.full_like(negative[0], 12.0))


def test_branch_falls_back_to_call_order() -> None:
    transformer = WanTransformer3DModel()
    metadata = SimpleNamespace(step=0, num_steps=5)
    hook = _apply_wan(transformer, metadata=metadata)

    _run_wan(transformer, metadata, step=0, t=1000.0, value=1.0, encoder=0.0)
    _run_wan(transformer, metadata, step=0, t=1000.0, value=1.0, encoder=10.0)
    positive = _run_wan(transformer, metadata, step=1, t=900.0, value=5.0, encoder=0.0)
    negative = _run_wan(transformer, metadata, step=1, t=900.0, value=5.0, encoder=10.0)

    assert transformer.stack_calls == 2
    assert torch.allclose(positive[0], torch.full_like(positive[0], 2.0))
    assert torch.allclose(negative[0], torch.full_like(negative[0], 12.0))
    assert set(hook.state.branch_predictions) == {"cond", "uncond"}


def test_scalar_timestep_uses_max_for_i2v() -> None:
    from vllm_omni.diffusion.cache.sencache.hook import _scalar_timestep

    assert _scalar_timestep(torch.tensor([[0.0, 0.0, 500.0, 500.0]])) == 500.0


class _ProbeRecorder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def record(self, hidden_states: torch.Tensor, timestep: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        del kwargs
        self.seen.append(timestep.clone())
        return (hidden_states,)


def test_probe_preserves_i2v_zero_timesteps() -> None:
    hook = SenCacheHook(SenCacheConfig(), _profile(0.5, 0.5))
    recorder = _ProbeRecorder()
    recorder._omni_original_forward = recorder.record  # type: ignore[attr-defined]
    current = torch.tensor([[700.0, 700.0, 0.0, 0.0]])
    template = torch.tensor([[900.0, 900.0, 0.0, 0.0]])

    hook._probe(
        recorder,
        {"timestep": current},
        torch.zeros(1, 2, 2, 2, 2),
        900.0,
        timestep_template=template,
    )

    assert torch.equal(recorder.seen[-1], torch.tensor([[900.0, 900.0, 0.0, 0.0]]))


def test_backend_denoise_step_falls_back_to_forward_context(monkeypatch) -> None:
    from vllm_omni.diffusion import forward_context as forward_context_module
    from vllm_omni.diffusion.cache.sencache import backend as backend_module

    monkeypatch.setattr(forward_context_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context_module,
        "get_forward_context",
        lambda: SimpleNamespace(denoise_step_idx=7),
    )

    assert backend_module._denoise_step(SimpleNamespace()) == 7


class Wan22Pipeline:
    def __init__(self) -> None:
        self.transformer = WanTransformer3DModel()
        self.transformer_2 = WanTransformer3DModel()
        self._num_timesteps = 5

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps


def test_backend_hooks_both_wan_experts(tmp_path) -> None:
    profile_path = tmp_path / "profile.npz"
    _profile(0.5, 0.5).save(profile_path)
    pipeline = Wan22Pipeline()
    backend = SenCacheBackend(DiffusionCacheConfig(sen_profile_path=str(profile_path)))

    backend.enable(pipeline)

    assert len(backend.hooks) == 2
    assert pipeline._sen_cache_hooks == backend.hooks
    assert backend.hooks[0].state is not backend.hooks[1].state
    assert not hasattr(pipeline, "_cache_context_factory")

    backend.refresh(pipeline, num_inference_steps=7)
    assert all(hook._num_steps == 7 for hook in backend.hooks)
