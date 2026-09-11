# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion compile backend unit tests with mocked optional dependencies."""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest
import torch

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _ParallelConfig:
    world_size: int = 1
    sequence_parallel_size: int = 1
    use_hsdp: bool = False


@dataclass
class _CompileConfig:
    diffusion_compile_granularity: str = "regional"
    diffusion_compile_dynamic: bool = False
    dtype: torch.dtype = torch.bfloat16
    quantization_config: object | None = None
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False
    cache_backend: str | None = "none"
    lora_path: str | None = None
    enable_sleep_mode: bool = False


@dataclass
class _FusionPatterns:
    enable_wan_residual_gate: bool = True
    enable_qwen_residual_gate: bool = True


@dataclass
class _BackendFactory:
    return_value: object = field(default_factory=object)
    calls: int = 0
    side_effect: Exception | None = None

    def __call__(self) -> object:
        self.calls += 1
        if self.side_effect is not None:
            raise self.side_effect
        return self.return_value


@pytest.fixture
def npu_platform(monkeypatch):
    """Load the Omni NPU class without importing the Ascend runtime."""
    ascend = ModuleType("vllm_ascend.platform")
    setattr(ascend, "NPUPlatform", type("NPUPlatform", (), {}))
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", ascend)

    path = Path(__file__).resolve().parents[2] / "vllm_omni/platforms/npu/platform.py"
    spec = importlib.util.spec_from_file_location("_test_npu_compile_platform", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NPUOmniPlatform


@pytest.fixture
def compile_config():
    return _CompileConfig()


@pytest.fixture
def mindiesd(monkeypatch):
    module = ModuleType("mindiesd.compilation")

    class CompilationConfig:
        fusion_patterns = _FusionPatterns()

    backend = object()
    backend_factory = _BackendFactory(return_value=backend)
    setattr(module, "CompilationConfig", CompilationConfig)
    setattr(module, "MindieSDBackend", backend_factory)
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", module)
    return module


@pytest.mark.parametrize(
    "platform",
    [OmniPlatformEnum.CUDA, OmniPlatformEnum.ROCM, OmniPlatformEnum.XPU, OmniPlatformEnum.MUSA],
)
def test_inductor_platforms_select_inductor_backend(monkeypatch, platform):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", platform, raising=False)
    monkeypatch.setattr(OmniPlatform, "supports_torch_inductor", lambda: True)

    assert OmniPlatform.get_diffusion_compile_backend(None) == "inductor"


def test_platform_without_inductor_support_runs_eager(monkeypatch):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", OmniPlatformEnum.UNSPECIFIED, raising=False)
    monkeypatch.setattr(OmniPlatform, "supports_torch_inductor", lambda: False)

    assert OmniPlatform.get_diffusion_compile_backend(None) is None


def test_npu_falls_back_when_mindiesd_is_unavailable(npu_platform, compile_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", None)

    assert npu_platform.get_diffusion_compile_backend(compile_config) is None


def test_npu_selects_available_mindiesd(npu_platform, compile_config, mindiesd):
    assert npu_platform.supports_torch_inductor() is False

    backend = npu_platform.get_diffusion_compile_backend(compile_config)

    assert backend is mindiesd.MindieSDBackend.return_value
    assert mindiesd.MindieSDBackend.calls == 1


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("diffusion_compile_dynamic", True),
        ("dtype", torch.float16),
        ("quantization_config", object()),
        ("enable_cpu_offload", True),
        ("enable_layerwise_offload", True),
        ("enable_distributed_layerwise_offload", True),
        ("cache_backend", "cache_dit"),
        ("lora_path", "adapter"),
        ("enable_sleep_mode", True),
    ],
)
def test_npu_falls_back_for_incompatible_config(
    npu_platform,
    compile_config,
    mindiesd,
    field_name,
    value,
):
    setattr(compile_config, field_name, value)

    assert npu_platform.get_diffusion_compile_backend(compile_config) is None
    assert mindiesd.MindieSDBackend.calls == 0


def test_npu_falls_back_for_multiple_devices(npu_platform, compile_config, mindiesd):
    compile_config.parallel_config.world_size = 2

    assert npu_platform.get_diffusion_compile_backend(compile_config) is None
    assert mindiesd.MindieSDBackend.calls == 0


@pytest.mark.parametrize("field_name", ["use_hsdp", "sequence_parallel_size"])
def test_npu_falls_back_for_full_compile_parallelism(npu_platform, compile_config, mindiesd, field_name):
    compile_config.diffusion_compile_granularity = "full"
    setattr(compile_config.parallel_config, field_name, True if field_name == "use_hsdp" else 2)

    assert npu_platform.get_diffusion_compile_backend(compile_config) is None
    assert mindiesd.MindieSDBackend.calls == 0


def test_npu_accepts_full_static_compile(npu_platform, compile_config, mindiesd):
    compile_config.diffusion_compile_granularity = "full"

    assert npu_platform.get_diffusion_compile_backend(compile_config) is mindiesd.MindieSDBackend.return_value


def test_npu_disables_residual_gate_patterns(npu_platform, compile_config, mindiesd):
    patterns = mindiesd.CompilationConfig.fusion_patterns
    patterns.enable_wan_residual_gate = True
    patterns.enable_qwen_residual_gate = True

    npu_platform.get_diffusion_compile_backend(compile_config)

    assert patterns.enable_wan_residual_gate is False
    assert patterns.enable_qwen_residual_gate is False


def test_npu_supports_mindiesd_without_fusion_pattern_flags(npu_platform, compile_config, mindiesd):
    patterns = mindiesd.CompilationConfig.fusion_patterns
    del patterns.enable_qwen_residual_gate
    del patterns.enable_wan_residual_gate

    assert npu_platform.get_diffusion_compile_backend(compile_config) is mindiesd.MindieSDBackend.return_value


def test_npu_supports_mindiesd_without_fusion_patterns(npu_platform, compile_config, mindiesd):
    del mindiesd.CompilationConfig.fusion_patterns

    assert npu_platform.get_diffusion_compile_backend(compile_config) is mindiesd.MindieSDBackend.return_value


def test_npu_backend_constructor_error_propagates(npu_platform, compile_config, mindiesd):
    mindiesd.MindieSDBackend.side_effect = TypeError("incompatible constructor")

    with pytest.raises(TypeError, match="incompatible constructor"):
        npu_platform.get_diffusion_compile_backend(compile_config)
