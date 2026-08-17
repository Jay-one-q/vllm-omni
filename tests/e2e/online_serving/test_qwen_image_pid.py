# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2/L3 e2e for PiD super-resolution on Qwen-Image (chat route).

Server starts with ``--pid-enable``; verifies:
- L2: basic request outputs LDM size x scale.
- L3: per-request ``pid_decode`` overrides (scale/seed) and the error path
  when ``enabled=True`` is sent to a server started without ``--pid-enable``.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServerParams,
    dummy_messages_from_mix_data,
)

pytestmark = [pytest.mark.diffusion]

MODEL = "Qwen/Qwen-Image"
# Empty means the official checkpoint is auto-downloaded by backbone from nvidia/PiD.
PID_CKPT = os.environ.get("PID_CKPT", "")
PID_GEMMA = os.environ.get("PID_GEMMA", "Efficient-Large-Model/gemma-2-2b-it")
T2I_PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
H100 = hardware_marks(res={"cuda": "H100"})


def _pid_server_args() -> list[str]:
    args = ["--pid-enable", "--pid-gemma", PID_GEMMA]
    if PID_CKPT:
        args += ["--pid-checkpoint", PID_CKPT]
    return args


_PID_SERVER = pytest.param(
    OmniServerParams(model=MODEL, server_args=_pid_server_args()),
    id="pid_default_4x",
    marks=H100,
)

_NO_PID_SERVER = pytest.param(
    OmniServerParams(model=MODEL),
    id="pid_not_enabled",
    marks=H100,
)


def _t2i_request(model: str, **pid_override) -> dict:
    return {
        "model": model,
        "messages": dummy_messages_from_mix_data(content_text=T2I_PROMPT),
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "true_cfg_scale": 4.0,
            "seed": 42,
            "pid_decode": {"enabled": True, "scale": 4, **pid_override},
        },
    }


@pytest.mark.core_model
@pytest.mark.parametrize("omni_server", [_PID_SERVER], indirect=True)
def test_text_to_image_pid_scale_4x(omni_server, openai_client) -> None:
    """L2: 512x512 input -> 2048x2048 output (scale=4)."""
    responses = openai_client.send_diffusion_request(_t2i_request(omni_server.model))
    assert responses[0].images is not None
    assert responses[0].images[0].size == (2048, 2048)


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", [_PID_SERVER], indirect=True)
def test_text_to_image_pid_per_request_override(omni_server, openai_client) -> None:
    """L3: per-request pid_decode overrides scale/seed (not exposed via CLI)."""
    responses = openai_client.send_diffusion_request(_t2i_request(omni_server.model, scale=2, seed=7))
    assert responses[0].images is not None
    assert responses[0].images[0].size == (1024, 1024)


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", [_PID_SERVER], indirect=True)
def test_text_to_image_pid_override_disable_falls_back(omni_server, openai_client) -> None:
    """L3: per-request enabled=False -> VAE fallback (no upscaling)."""
    req = _t2i_request(omni_server.model)
    req["extra_body"]["pid_decode"] = {"enabled": False}
    responses = openai_client.send_diffusion_request(req)
    assert responses[0].images is not None
    assert responses[0].images[0].size == (512, 512)


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server_function", [_NO_PID_SERVER], indirect=True)
def test_text_to_image_pid_request_without_server_flag_errors(omni_server_function, openai_client_function) -> None:
    """L3 negative: enabled=True on a server without --pid-enable errors (mixin RuntimeError)."""
    body = {
        "model": omni_server_function.model,
        "prompt": T2I_PROMPT,
        "size": "512x512",
        "pid_decode": {"enabled": True, "scale": 4},
    }
    openai_client_function.send_images_generations_http_request(
        {
            "json": body,
            "timeout": 300,
            "err_code": (400, 500),
            "err_message": "--pid-enable",
        }
    )
