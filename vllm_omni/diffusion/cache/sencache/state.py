# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-request SenCache state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SenCacheState:
    """Reference point, cached residuals and counters for one generation.

    The reference latent/timestep is shared by every branch (cond, uncond,
    cond_no_control) because they all consume the same noisy latent for a given
    denoising step. Cached residuals are kept per branch because the denoiser
    output differs with the conditioning.
    """

    reference_latent: torch.Tensor | None = None
    reference_timestep: float | None = None
    alpha_x: float = 0.0
    alpha_t: float = 0.0
    branch_residuals: dict[str, torch.Tensor] = field(default_factory=dict)
    consecutive_reuses: int = 0
    full_count: int = 0
    skip_count: int = 0
    step_gate: dict[int, bool] = field(default_factory=dict)
    last_gate_step: int | None = None
    last_ref_step: int | None = None

    def reset(self) -> None:
        self.reference_latent = None
        self.reference_timestep = None
        self.alpha_x = 0.0
        self.alpha_t = 0.0
        self.branch_residuals.clear()
        self.consecutive_reuses = 0
        self.full_count = 0
        self.skip_count = 0
        self.step_gate.clear()
        self.last_gate_step = None
        self.last_ref_step = None

    def reset_step_gates(self) -> None:
        """Drop stale gates when a new denoising step starts."""
        self.step_gate.clear()

    def residual_for(self, branch: str, latent: torch.Tensor) -> torch.Tensor | None:
        residual = self.branch_residuals.get(branch)
        if residual is None:
            return None
        if residual.shape != latent.shape or residual.device != latent.device or residual.dtype != latent.dtype:
            return None
        return residual
