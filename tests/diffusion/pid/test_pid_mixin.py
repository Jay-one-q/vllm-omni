# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.pid import PidDecodeConfig
from vllm_omni.diffusion.pid.mixin import PidDecodeMixin

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _cfg(**kw):
    kw.setdefault("enabled", True)
    kw.setdefault("checkpoint_path", "/tmp/pid.pth")
    kw.setdefault("gemma_model", "/tmp/gemma")
    return PidDecodeConfig(**kw)


def _pipe(mocker, cfg, backbone="qwenimage"):
    """Build a mixin pipeline with PidDecoder mocked (no PidNet / Gemma)."""
    mocker.patch("vllm_omni.diffusion.pid.mixin.PidDecoder")
    cls = type("_Pipe", (PidDecodeMixin,), {"PID_BACKBONE": backbone})
    pipe = cls()
    pipe._init_pid_decoder(SimpleNamespace(pid_decode=cfg))
    return pipe


def _stub_decode(mocker, pipe, shape=(1, 3, 1024, 1024)):
    pipe._pid_decoder = mocker.Mock()
    pipe._pid_decoder.decode = mocker.Mock(return_value=torch.zeros(*shape))
    return pipe._pid_decoder.decode


# -- _resolve_pid_config (staticmethod, no instance needed) ------------------


def test_resolve_pid_config_none_returns_none():
    assert PidDecodeMixin._resolve_pid_config(SimpleNamespace(pid_decode=None)) is None


def test_resolve_pid_config_dict_normalized():
    raw = {"enabled": True, "scale": 2, "num_steps": 1}
    cfg = PidDecodeMixin._resolve_pid_config(SimpleNamespace(pid_decode=raw))
    assert isinstance(cfg, PidDecodeConfig)
    assert cfg.scale == 2 and cfg.num_steps == 1


def test_resolve_pid_config_bad_type_raises():
    with pytest.raises(TypeError, match="pid_decode"):
        PidDecodeMixin._resolve_pid_config(SimpleNamespace(pid_decode=123))


# -- _init_pid_decoder ------------------------------------------------------


def test_init_without_backbone_raises():
    class NoBackbone(PidDecodeMixin):
        PID_BACKBONE = ""

    with pytest.raises(RuntimeError, match="PID_BACKBONE"):
        NoBackbone()._init_pid_decoder(SimpleNamespace(pid_decode=_cfg()))


def test_disabled_config_creates_no_decoder(mocker):
    pipe = _pipe(mocker, _cfg(enabled=False))
    assert pipe._pid_decoder is None


def test_init_declares_pid_resident(mocker):
    pipe = _pipe(mocker, _cfg())
    assert "_pid_decoder" in pipe._resident_modules
