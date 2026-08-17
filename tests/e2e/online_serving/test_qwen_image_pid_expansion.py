# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L4 expansion for PiD on Qwen-Image (nightly).

Kept lean on purpose: only large-size 4x and offload-resident decode survive.
scale=2/num_steps=1 overrides overlap with L2/L3, and PiD resident status is
already asserted at L1 unit level.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL = "Qwen/Qwen-Image"
PID_CKPT = os.environ.get("PID_CKPT", "")
PID_GEMMA = os.environ.get("PID_GEMMA", "Efficient-Large-Model/gemma-2-2b-it")
T2I_PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
H100 = hardware_marks(res={"cuda": "H100"})


def _pid_server_args() -> list[str]:
    args = ["--pid-enable", "--pid-gemma", PID_GEMMA]
    if PID_CKPT:
        args += ["--pid-checkpoint", PID_CKPT]
    return args


_LARGE_SERVER = pytest.param(
    OmniServerParams(model=MODEL, server_args=_pid_server_args()),
    id="pid_large_4x",
    marks=H100,
)

_OFFLOAD_SERVER = pytest.param(
    OmniServerParams(model=MODEL, server_args=[*_pid_server_args(), "--enable-cpu-offload"]),
    id="pid_offload_resident",
    marks=H100,
)


def _t2i_request(model: str, *, height: int, width: int, scale: int) -> dict:
    return {
        "model": model,
        "messages": dummy_messages_from_mix_data(content_text=T2I_PROMPT),
        "extra_body": {
            "height": height,
            "width": width,
            "num_inference_steps": 2,
            "true_cfg_scale": 4.0,
            "seed": 42,
            "pid_decode": {"enabled": True, "scale": scale},
        },
    }


@pytest.mark.parametrize("omni_server", [_LARGE_SERVER], indirect=True)
def test_text_to_image_pid_large_4x(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """L4: 1024x1024 input -> 4096x4096 output (scale=4)."""
    responses = openai_client.send_diffusion_request(_t2i_request(omni_server.model, height=1024, width=1024, scale=4))
    assert responses[0].images is not None
    assert responses[0].images[0].size == (4096, 4096)


@pytest.mark.parametrize("omni_server", [_OFFLOAD_SERVER], indirect=True)
def test_text_to_image_pid_resident_under_offload(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """L4: PiD stays functional with CPU offload (resident module never offloaded)."""
    responses = openai_client.send_diffusion_request(_t2i_request(omni_server.model, height=512, width=512, scale=4))
    assert responses[0].images is not None
    assert responses[0].images[0].size == (2048, 2048)
