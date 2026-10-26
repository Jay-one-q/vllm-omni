# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runtime configuration for sensitivity-aware caching (SenCache)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SenCacheConfig:
    """Knobs shared by the inference hook and the offline calibrator.

    Thresholds are expressed in RMS output-change units: the gate predicts
    ``rms(f(x_t, t) - f(x_ref, t_ref))`` and compares it against the threshold.
    Values are model specific; sweep them with the validation workflow instead
    of copying the paper's raw-L2 numbers.

    ``calibration_output`` switches the backend into profile generation mode:
    every cacheable step is executed and two probe forwards are issued so that
    the Jacobian-norm profile can be estimated online.
    """

    profile_path: str | None = None
    threshold_start: float = 0.02
    threshold_main: float = 0.08
    start_fraction: float = 0.2
    max_consecutive_reuses: int = 2
    time_max: float = 1000.0
    calibration_output: str | None = None
    calibration_time_bins: int = 128

    @property
    def calibrating(self) -> bool:
        return self.calibration_output is not None

    def threshold_for_step(self, step_index: int, num_steps: int) -> float:
        """Return the tolerance for ``step_index`` (strict early phase)."""
        if num_steps <= 0:
            return self.threshold_main
        switch = int(round(num_steps * self.start_fraction))
        return self.threshold_start if step_index < switch else self.threshold_main

    def __post_init__(self) -> None:
        if self.profile_path is not None and not isinstance(self.profile_path, str):
            raise ValueError(f"profile_path must be a string, got {self.profile_path!r}")
        if self.calibration_output is not None and not isinstance(self.calibration_output, str):
            raise ValueError(f"calibration_output must be a string, got {self.calibration_output!r}")
        for name in ("threshold_start", "threshold_main"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if not math.isfinite(self.start_fraction) or not 0.0 <= self.start_fraction <= 1.0:
            raise ValueError(f"start_fraction must be within [0, 1], got {self.start_fraction}")
        if (
            isinstance(self.max_consecutive_reuses, bool)
            or not isinstance(self.max_consecutive_reuses, int)
            or self.max_consecutive_reuses < 0
        ):
            raise ValueError(
                f"max_consecutive_reuses must be a non-negative integer, got {self.max_consecutive_reuses!r}"
            )
        if not math.isfinite(self.time_max) or self.time_max <= 0.0:
            raise ValueError(f"time_max must be finite and positive, got {self.time_max}")
        if (
            isinstance(self.calibration_time_bins, bool)
            or not isinstance(self.calibration_time_bins, int)
            or self.calibration_time_bins < 8
        ):
            raise ValueError(
                f"calibration_time_bins must be an integer >= 8, got {self.calibration_time_bins!r}"
            )

    @classmethod
    def from_diffusion_cache_config(cls, config: Any) -> SenCacheConfig:
        """Project the optional ``sen_*`` fields of ``DiffusionCacheConfig``."""

        def _get(name: str, default: Any) -> Any:
            value = getattr(config, name, None)
            return default if value is None else value

        return cls(
            profile_path=_get("sen_profile_path", None),
            threshold_start=_get("sen_threshold_start", 0.02),
            threshold_main=_get("sen_threshold_main", 0.08),
            start_fraction=_get("sen_start_fraction", 0.2),
            max_consecutive_reuses=_get("sen_max_consecutive_reuses", 2),
            time_max=_get("sen_time_max", 1000.0),
            calibration_output=_get("sen_calibration_output", None),
            calibration_time_bins=_get("sen_calibration_time_bins", 128),
        )
