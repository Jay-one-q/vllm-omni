# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline/online calibration helpers for SenCache sensitivity profiles.

The inference hook can run in calibration mode: every cacheable step is
executed and two probe forwards estimate ``alpha_x`` and ``alpha_t`` at the
current timestep. Each calibration request writes one accumulator artifact;
``merge`` averages artifacts from several prompts before the profile is used
for inference.

Usage::

    python -m vllm_omni.diffusion.cache.sencache.calibrate merge \
        --inputs "/tmp/sen/profile-*.npz" --out /tmp/sen/cosmos3_nano.npz
    python -m vllm_omni.diffusion.cache.sencache.calibrate inspect \
        --profile /tmp/sen/cosmos3_nano.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.diffusion.cache.sencache.profile import (
    SCHEMA_VERSION,
    SensitivityProfile,
    _as_float_array,
    _fill_empty_bins,
    _read_metadata,
)


def default_time_bins(num_bins: int) -> np.ndarray:
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    return np.linspace(0.0, 1.0, num_bins, dtype=np.float64)


@dataclass
class ProfileAccumulator:
    """Running mean of per-bin sensitivity samples."""

    time_max: float
    metadata: dict[str, Any]
    num_bins: int = 128
    alpha_x_sum: np.ndarray | None = None
    alpha_t_sum: np.ndarray | None = None
    count: np.ndarray | None = None

    def __post_init__(self) -> None:
        bins = default_time_bins(self.num_bins)
        if self.alpha_x_sum is None:
            self.alpha_x_sum = np.zeros_like(bins)
        if self.alpha_t_sum is None:
            self.alpha_t_sum = np.zeros_like(bins)
        if self.count is None:
            self.count = np.zeros_like(bins)
        self.alpha_x_sum = _as_float_array(self.alpha_x_sum, "alpha_x_sum", self.num_bins)
        self.alpha_t_sum = _as_float_array(self.alpha_t_sum, "alpha_t_sum", self.num_bins)
        self.count = _as_float_array(self.count, "count", self.num_bins)

    @property
    def time_bins(self) -> np.ndarray:
        return default_time_bins(self.num_bins)

    def bin_index(self, timestep: float) -> int:
        normalized = min(max(float(timestep) / self.time_max, 0.0), 1.0)
        return int(round(normalized * (self.num_bins - 1)))

    def add(self, timestep: float, alpha_x: float | None, alpha_t: float | None) -> None:
        """Record one sample pair; ``None`` leaves that term unchanged."""
        if alpha_x is None and alpha_t is None:
            return
        for value in (alpha_x, alpha_t):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                return
        index = self.bin_index(timestep)
        assert self.alpha_x_sum is not None and self.alpha_t_sum is not None and self.count is not None
        if alpha_x is not None:
            self.alpha_x_sum[index] += alpha_x
        if alpha_t is not None:
            self.alpha_t_sum[index] += alpha_t
        self.count[index] += 1.0

    def merge(self, other: ProfileAccumulator) -> None:
        if abs(self.time_max - other.time_max) > 1e-9 or self.num_bins != other.num_bins:
            raise ValueError("Cannot merge sensitivity profiles with different time grids")
        for key in ("model_id", "boundary"):
            if self.metadata.get(key) != other.metadata.get(key):
                raise ValueError(f"Cannot merge sensitivity profiles with different {key!r}")
        # Bins without an x sample may still carry a t sample (and vice versa),
        # so accumulate each numerator and keep one combined count.
        assert self.alpha_x_sum is not None and self.alpha_t_sum is not None and self.count is not None
        assert other.alpha_x_sum is not None and other.alpha_t_sum is not None and other.count is not None
        self.alpha_x_sum += other.alpha_x_sum
        self.alpha_t_sum += other.alpha_t_sum
        self.count += other.count

    def finalize(self) -> SensitivityProfile:
        assert self.alpha_x_sum is not None and self.alpha_t_sum is not None and self.count is not None
        occupied = self.count > 0
        alpha_x = np.divide(self.alpha_x_sum, self.count, out=np.zeros_like(self.alpha_x_sum), where=occupied)
        alpha_t = np.divide(self.alpha_t_sum, self.count, out=np.zeros_like(self.alpha_t_sum), where=occupied)
        alpha_x = _fill_empty_bins(alpha_x, occupied)
        alpha_t = _fill_empty_bins(alpha_t, occupied)
        metadata = dict(self.metadata)
        metadata["schema_version"] = SCHEMA_VERSION
        metadata["time_max"] = self.time_max
        metadata["calibration_samples"] = float(occupied.sum())
        return SensitivityProfile(
            time_bins=self.time_bins,
            alpha_x=alpha_x,
            alpha_t=alpha_t,
            time_max=self.time_max,
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        """Write both accumulators and the finalized curves."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = self.finalize()
        metadata = dict(profile.metadata)
        metadata["schema_version"] = SCHEMA_VERSION
        np.savez(
            path,
            alpha_x=profile.alpha_x,
            alpha_t=profile.alpha_t,
            alpha_x_sum=self.alpha_x_sum,
            alpha_t_sum=self.alpha_t_sum,
            count=self.count,
            time_bins=profile.time_bins,
            time_max=np.float64(self.time_max),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> ProfileAccumulator:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if not {"alpha_x_sum", "alpha_t_sum", "count", "time_bins", "time_max"} <= keys:
                raise ValueError(f"{path} has no calibration accumulators; merge only raw calibration artifacts")
            bins = _as_float_array(data["time_bins"], "time_bins")
            metadata = _read_metadata(data, path)
            return cls(
                time_max=float(data["time_max"]),
                metadata=metadata,
                num_bins=int(bins.shape[0]),
                alpha_x_sum=data["alpha_x_sum"],
                alpha_t_sum=data["alpha_t_sum"],
                count=data["count"],
            )


def merge_files(inputs: list[str | Path], output: str | Path) -> ProfileAccumulator:
    if not inputs:
        raise ValueError("No calibration artifacts were provided")
    accumulator: ProfileAccumulator | None = None
    for item in inputs:
        current = ProfileAccumulator.load(item)
        if accumulator is None:
            accumulator = current
        else:
            accumulator.merge(current)
    assert accumulator is not None
    accumulator.save(output)
    return accumulator


def _print_profile(profile: SensitivityProfile) -> None:
    print(f"time_max={profile.time_max} bins={profile.time_bins.shape[0]}")
    print("metadata=" + json.dumps(profile.metadata, sort_keys=True))
    print(f"{'t':>8}  {'alpha_x':>12}  {'alpha_t':>12}")
    for normalized, alpha_x, alpha_t in zip(profile.time_bins, profile.alpha_x, profile.alpha_t, strict=True):
        print(f"{normalized * profile.time_max:8.1f}  {alpha_x:12.6f}  {alpha_t:12.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge", help="Average several calibration artifacts")
    merge.add_argument("--inputs", nargs="+", required=True, help="Artifact paths or glob patterns")
    merge.add_argument("--out", required=True, help="Output profile path")

    inspect = sub.add_parser("inspect", help="Print a finalized profile")
    inspect.add_argument("--profile", required=True)

    args = parser.parse_args(argv)
    if args.command == "merge":
        paths: list[str] = []
        for pattern in args.inputs:
            matched = sorted(glob.glob(pattern))
            paths.extend(matched if matched else [pattern])
        accumulator = merge_files(paths, args.out)
        _print_profile(accumulator.finalize())
        print(f"wrote {args.out}")
        return 0

    _print_profile(SensitivityProfile.load(args.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
