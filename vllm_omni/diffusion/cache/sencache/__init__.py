# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""SenCache: sensitivity-aware full-forward caching.

This is an experimental backend validated on Cosmos3-Nano. It requires an
offline sensitivity profile; see ``calibrate.py`` for the artifact workflow.
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
