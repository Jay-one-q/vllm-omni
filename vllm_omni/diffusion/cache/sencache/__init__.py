# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""SenCache: sensitivity-aware full-forward caching.

Validation scope: Cosmos3-Nano (block-residual boundary) and Wan2.2 T2V/I2V
(full-prediction boundary, dual experts). It requires an offline sensitivity
profile; see ``calibrate.py`` for the artifact workflow.
"""

from vllm_omni.diffusion.cache.sencache.backend import SenCacheBackend
from vllm_omni.diffusion.cache.sencache.config import SenCacheConfig
from vllm_omni.diffusion.cache.sencache.hook import (
    SenCacheHook,
    apply_sen_cache_hook,
)
from vllm_omni.diffusion.cache.sencache.profile import SensitivityProfile

__all__ = [
    "SenCacheBackend",
    "SenCacheConfig",
    "SenCacheHook",
    "SensitivityProfile",
    "apply_sen_cache_hook",
]
