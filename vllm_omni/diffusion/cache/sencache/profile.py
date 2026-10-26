# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Sensitivity profile artifacts for SenCache.

A profile stores the per-timestep Jacobian-norm estimates used by the cache
gate. Sensitivity is binned on a normalized timestep grid so that profiles
estimated with different step counts can be merged and reused.

Two quantities are stored per bin:

- ``alpha_x``: RMS output change divided by RMS latent change at fixed t.
- ``alpha_t``: RMS output change divided by the raw timestep delta at fixed x.

The gate then predicts ``alpha_x * rms(dx) + alpha_t * |dt|``, which has RMS
output-change units for both terms.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def _as_float_array(value: Any, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {array.shape[0]}")
    return array


def _fill_empty_bins(values: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Fill empty bins with the nearest occupied bin (edges propagate)."""
    if occupied.all():
        return values
    if not occupied.any():
        raise ValueError("sensitivity profile has no valid samples")
    indices = np.arange(values.shape[0])
    populated = indices[occupied]
    insertion = np.searchsorted(populated, indices)
    right = populated[np.clip(insertion, 0, populated.shape[0] - 1)]
    left = populated[np.clip(insertion - 1, 0, populated.shape[0] - 1)]
    use_right = np.abs(right - indices) <= np.abs(indices - left)
    nearest = np.where(use_right, right, left)
    return values[nearest]


@dataclass(frozen=True)
class SensitivityProfile:
    """Finalized per-timestep sensitivity curves."""

    time_bins: np.ndarray
    alpha_x: np.ndarray
    alpha_t: np.ndarray
    time_max: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        bins = _as_float_array(self.time_bins, "time_bins")
        alpha_x = _as_float_array(self.alpha_x, "alpha_x", bins.shape[0])
        alpha_t = _as_float_array(self.alpha_t, "alpha_t", bins.shape[0])
        if bins.shape[0] < 2:
            raise ValueError("time_bins must contain at least two entries")
        if not np.all(np.diff(bins) > 0):
            raise ValueError("time_bins must be strictly increasing")
        if bins[0] < 0.0 or bins[-1] > 1.0:
            raise ValueError("time_bins must lie within [0, 1]")
        for name, array in (("alpha_x", alpha_x), ("alpha_t", alpha_t)):
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.time_max) or self.time_max <= 0.0:
            raise ValueError(f"time_max must be finite and positive, got {self.time_max}")
        object.__setattr__(self, "time_bins", bins)
        object.__setattr__(self, "alpha_x", alpha_x)
        object.__setattr__(self, "alpha_t", alpha_t)

    def lookup(self, timestep: float) -> tuple[float, float]:
        """Return ``(alpha_x, alpha_t)`` for a raw model timestep."""
        if not math.isfinite(timestep):
            raise ValueError(f"timestep must be finite, got {timestep}")
        normalized = min(max(float(timestep) / self.time_max, 0.0), 1.0)
        index = int(round(normalized * (self.time_bins.shape[0] - 1)))
        return float(self.alpha_x[index]), float(self.alpha_t[index])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata["schema_version"] = SCHEMA_VERSION
        metadata["time_max"] = self.time_max
        np.savez(
            path,
            alpha_x=self.alpha_x,
            alpha_t=self.alpha_t,
            time_bins=self.time_bins,
            time_max=np.float64(self.time_max),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> SensitivityProfile:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"SenCache profile not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if "time_bins" not in keys or "time_max" not in keys:
                raise ValueError(f"{path} is not a SenCache profile (missing time grid)")
            bins = _as_float_array(data["time_bins"], "time_bins")
            time_max = float(data["time_max"])
            metadata = _read_metadata(data, path)
            if "alpha_x" in keys and "alpha_t" in keys:
                alpha_x = _as_float_array(data["alpha_x"], "alpha_x", bins.shape[0])
                alpha_t = _as_float_array(data["alpha_t"], "alpha_t", bins.shape[0])
            else:
                alpha_x, alpha_t = _finalize_sums(data, bins.shape[0], path)

            if abs(float(metadata.get("time_max", time_max)) - time_max) > 1e-9:
                raise ValueError(f"{path} metadata time_max does not match its time grid")

        return cls(
            time_bins=bins,
            alpha_x=alpha_x,
            alpha_t=alpha_t,
            time_max=time_max,
            metadata=metadata,
        )


def _read_metadata(data: Any, path: Path) -> dict[str, Any]:
    if "metadata" not in data.files:
        raise ValueError(f"{path} is missing profile metadata")
    raw = data["metadata"]
    try:
        text = str(raw.item()) if raw.ndim == 0 else str(raw)
        metadata = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} has invalid profile metadata: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} profile metadata must be a JSON object")
    return metadata


def _finalize_sums(data: Any, length: int, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if "alpha_x_sum" not in data.files or "alpha_t_sum" not in data.files or "count" not in data.files:
        raise ValueError(f"{path} must provide either finalized alphas or sum/count accumulators")
    alpha_x_sum = _as_float_array(data["alpha_x_sum"], "alpha_x_sum", length)
    alpha_t_sum = _as_float_array(data["alpha_t_sum"], "alpha_t_sum", length)
    count = _as_float_array(data["count"], "count", length)
    if np.any(count < 0):
        raise ValueError(f"{path} has negative sample counts")
    occupied = count > 0
    alpha_x = np.divide(alpha_x_sum, count, out=np.zeros_like(alpha_x_sum), where=occupied)
    alpha_t = np.divide(alpha_t_sum, count, out=np.zeros_like(alpha_t_sum), where=occupied)
    return _fill_empty_bins(alpha_x, occupied), _fill_empty_bins(alpha_t, occupied)
