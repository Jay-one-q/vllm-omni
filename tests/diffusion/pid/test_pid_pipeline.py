# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.pid import PidDecodeConfig, PidDecodeMixin

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# (B, num_patches, channels) = 512x512 @ 8x VAE compression + 2x2 patch packing
_LATENTS = torch.zeros(1, 1024, 16)
_PROMPT = "a cat sitting on a windowsill"


def _make_pipeline(mocker, pid_config):
    """Real QwenImagePipeline instance, skipping __init__ (no weights / Gemma)."""
    decoder_cls = mocker.patch("vllm_omni.diffusion.pid.mixin.PidDecoder")
    pipe = object.__new__(QwenImagePipeline)
    torch.nn.Module.__init__(pipe)
    pipe._resident_modules = []
    pipe.vae_scale_factor = 8
    pipe._init_pid_decoder(SimpleNamespace(pid_decode=pid_config, enforce_eager=False))
    return pipe, decoder_cls


def _make_state(pid_override, *, height=512, width=512, prompt=_PROMPT):
    sampling = SimpleNamespace(height=height, width=width, output_type="pil", pid_decode=pid_override)
    return SimpleNamespace(sampling=sampling, prompt=prompt, latents=_LATENTS)


# -- PiD enabled loads and runs ----------------------------------------------


def test_qwen_image_pipeline_declares_pid_backbone():
    assert QwenImagePipeline.PID_BACKBONE == "qwenimage"
    assert issubclass(QwenImagePipeline, PidDecodeMixin)


def test_pipeline_pid_enabled_loads_decoder_and_stays_resident(mocker):
    pipe, decoder_cls = _make_pipeline(
        mocker,
        PidDecodeConfig(enabled=True, checkpoint_path="/tmp/pid.pth", gemma_model="/tmp/gemma"),
    )
    assert pipe._pid_config is not None and pipe._pid_config.enabled is True
    assert pipe._pid_decoder is not None
    # eager loading: load_weights() runs during __init__ (weights resident)
    decoder_cls.return_value.load_weights.assert_called_once()
    assert decoder_cls.call_args.kwargs["backbone"] == "qwenimage"
    # resident: the CPU offloader must not evict PiD
    assert "_pid_decoder" in pipe._resident_modules


def test_pipeline_pid_disabled_creates_no_decoder(mocker):
    pipe, decoder_cls = _make_pipeline(mocker, PidDecodeConfig(enabled=False))
    assert pipe._pid_decoder is None
    assert "_pid_decoder" not in pipe._resident_modules


# -- Request pid params reach the PiD module ---------------------------------


def test_post_decode_threads_request_pid_params_to_pid_decoder(mocker):
    """pid_decode in the request -> _pid_override/_pid_caption -> decode args."""
    pipe, _ = _make_pipeline(mocker, PidDecodeConfig(enabled=True, scale=4))
    pipe._pid_decoder = mocker.Mock()
    pipe._pid_decoder.decode = mocker.Mock(return_value=torch.zeros(1, 3, 1024, 1024))

    out = pipe.post_decode(
        _make_state({"enabled": True, "scale": 2, "seed": 7, "num_steps": 3, "degrade_sigma": 0.5})
    )

    # request params parsed into pipeline attributes
    assert pipe._pid_override == {"enabled": True, "scale": 2, "seed": 7, "num_steps": 3, "degrade_sigma": 0.5}
    assert pipe._pid_caption == [_PROMPT]

    # forwarded verbatim into the PiD module via maybe_pid_decode
    decode_kwargs = pipe._pid_decoder.decode.call_args.kwargs
    assert decode_kwargs["output_size"] == (1024, 1024)  # 512 * scale=2
    assert decode_kwargs["seed"] == 7
    assert decode_kwargs["num_steps"] == 3
    assert decode_kwargs["degrade_sigma"] == 0.5
    assert decode_kwargs["caption"] == [_PROMPT]
    assert decode_kwargs["lq_latent"].dim() == 4  # (B,C,1,zH,zW) squeezed to 4D

    # returns the PiD-decoded result
    assert isinstance(out, DiffusionOutput)
    assert out.output.shape == (1, 3, 1024, 1024)


def test_post_decode_request_disables_pid_falls_back_to_vae(mocker):
    """pid_decode.enabled=False skips PiD and falls back to VAE."""
    pipe, _ = _make_pipeline(mocker, PidDecodeConfig(enabled=True))
    pipe._pid_decoder = mocker.Mock()
    pipe._pid_decoder.decode = mocker.Mock(return_value=torch.zeros(1, 3, 2048, 2048))

    # minimal VAE mock to exercise the fallback path
    pipe.vae = mocker.Mock()
    pipe.vae.dtype = torch.float32
    pipe.vae.config = mocker.Mock()
    pipe.vae.config.latents_mean = [0.0] * 4
    pipe.vae.config.latents_std = [1.0] * 4
    pipe.vae.config.z_dim = 4
    pipe.vae.decode = mocker.Mock(return_value=[torch.zeros(1, 4, 1, 64, 64)])

    out = pipe.post_decode(_make_state({"enabled": False}))

    assert pipe._pid_decoder.decode.call_count == 0  # PiD not called
    pipe.vae.decode.assert_called_once()
    assert isinstance(out, DiffusionOutput)
    assert out.output.shape == (1, 4, 64, 64)  # VAE baseline (no upscale)


def test_pipeline_request_enables_pid_without_pipeline_config_raises(mocker):
    """Enabling PiD per request without --pid-enable raises (no lazy weights)."""
    pipe, _ = _make_pipeline(mocker, None)
    assert pipe._pid_decoder is None

    with pytest.raises(RuntimeError, match="--pid-enable"):
        pipe.post_decode(_make_state({"enabled": True}))
