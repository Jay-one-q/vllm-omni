# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only PiD model.

Merges PixelDiTModel + PidModel + PidDistillModel into a single flat class
with no imaginaire dependency. Only ``generate_samples_from_batch`` is exposed.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn

from .config import PID_SAMPLING_CONFIG
from .pid_net import PidNet
from .text_encoder import GemmaTextEncoder

logger = logging.getLogger(__name__)


class PidInferenceModel(nn.Module):
    """Inference-only PiD model (model-agnostic).

    Args:
        net_kwargs: Passed directly to PidNet.__init__.
        gemma_model_id: Local path or HF ID for gemma-2-2b-it.
        sampling_overrides: Optional overrides for PID_SAMPLING_CONFIG.
        precision: Compute precision preset, one of "float32" / "float16" /
            "bfloat16". Mirrors PixelDiTModelConfig.precision. For any
            non-float32 value, the tensor container stays float32 (matching
            the student's calibrated distribution) and matmuls run under
            ``torch.autocast(..., dtype=precision)``. ``"float32"`` disables
            autocast entirely (pure fp32 forward), used for precision
            baselines or checkpoints that overflow under bf16/fp16.
    """

    def __init__(
        self,
        net_kwargs: dict,
        gemma_model_id: str,
        sampling_overrides: dict | None = None,
        precision: str = "bfloat16",
    ):
        super().__init__()
        self.net = PidNet(**net_kwargs)
        self.text_encoder = GemmaTextEncoder(gemma_model_id)

        samp = dict(PID_SAMPLING_CONFIG)
        if sampling_overrides:
            samp.update(sampling_overrides)
        self._cfg = type("Cfg", (), samp)()

        # Replicate PixelDiTModel.__init__ precision resolution: the tensor
        # container is always float32; for non-float32 precision, matmuls run
        # under autocast(dtype=requested). precision="float32" disables
        # autocast entirely (pure fp32 forward).
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16, 
        }
        if precision not in _dtype_map:
            raise ValueError(
                f"precision must be one of {list(_dtype_map)}, got {precision!r}"
            )
        requested_dtype = _dtype_map[precision]
        if requested_dtype != torch.float32:
            self.autocast_dtype = requested_dtype
            self.precision = torch.float32
        else:
            self.autocast_dtype = None
            self.precision = torch.float32
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        # torch.compile support (opt-in via enable_compile()).
        # Compilation is lazy and cached per output resolution (H, W).
        self._compile_enabled = False
        self._compiled_nets: dict[tuple[int, int], torch.nn.Module] = {}

        logger.info(
            "PidInferenceModel: net params=%s, precision=%s (autocast=%s, container=%s)",
            f"{sum(p.numel() for p in self.net.parameters()):,}",
            precision,
            self.autocast_dtype,
            self.precision,
        )

    # ------------------------------------------------------------------
    # Velocity <-> x0 conversion
    # ------------------------------------------------------------------

    def _velocity_to_x0(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """velocity -> x0: x0 = x_t - t * v"""
        s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
        t_shaped = t.double().view(*s)
        return (x_t.double() - t_shaped * v.double()).to(x_t.dtype)

    # ------------------------------------------------------------------
    # torch.compile (opt-in)
    # ------------------------------------------------------------------

    def enable_compile(self, mode: str = "default") -> None:
        """Arm torch.compile for :attr:`net`.

        Compilation is lazy — the actual ``torch.compile`` call happens on the
        first ``generate_samples_from_batch`` for each output resolution and
        is cached thereafter.  ``mode`` is passed directly to
        ``torch.compile``; use ``"max-autotune"`` for maximum throughput at
        the cost of a much slower first compile.
        """
        self._compile_enabled = True
        self._compile_mode = mode
        logger.info("PidInferenceModel: torch.compile armed (lazy, per resolution).")

    def _maybe_compile_net(
        self, image_h: int, image_w: int, text_len: int
    ) -> torch.nn.Module:
        """Return compiled net for this shape, or eager net if compile is off."""
        if not self._compile_enabled:
            return self.net
        key = (int(image_h), int(image_w))
        compiled = self._compiled_nets.get(key)
        if compiled is None:
            logger.info(
                "PidInferenceModel: warming pos caches + compiling net for %dx%d",
                image_h, image_w,
            )
            self.net.precompute_positional_caches(
                image_height=image_h,
                image_width=image_w,
                text_length=text_len,
                device="cuda",
                pixel_dtype=self.precision,
            )
            compiled = torch.compile(
                self.net, mode=self._compile_mode, dynamic=False
            )
            self._compiled_nets[key] = compiled
        return compiled

    # ------------------------------------------------------------------
    # Timestep schedule
    # ------------------------------------------------------------------

    def _get_t_list(self, device, num_steps: int | None = None) -> torch.Tensor:
        target = num_steps or self._cfg.student_sample_steps
        full_t = torch.tensor(
            self._cfg.student_t_list, device=device, dtype=torch.float32
        )
        if target != len(full_t) - 1:
            indices = (
                torch.linspace(0, len(full_t) - 1, target + 1).round().long()
            )
            return full_t[indices]
        return full_t

    # ------------------------------------------------------------------
    # SDE sample loop
    # ------------------------------------------------------------------

    def _sample_loop(
        self,
        noise: torch.Tensor,
        t_list: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: torch.Tensor,
        degrade_sigma: torch.Tensor,
        net: torch.nn.Module | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        B = noise.shape[0]
        timescale = self._cfg.fm_timescale
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype
            else nullcontext()
        )
        if net is None:
            net = self.net
        x = noise

        # vllm-omni may have left allow_tf32=False after the LDM stage;
        # PiD internally runs under autocast(bf16) but the outer matmul
        # dispatch still depends on this flag for intermediate ops.
        if not torch.backends.cuda.matmul.allow_tf32:
            logger.warning(
                "PiD: torch.backends.cuda.matmul.allow_tf32 is False — "
                "forcing to True for this call (A100 matmul depends on tf32)."
            )
            torch.backends.cuda.matmul.allow_tf32 = True

        step_times = []
        with autocast_ctx:
            for step_idx, (t_cur, t_next) in enumerate(zip(t_list[:-1], t_list[1:])):
                t_cur_batch = t_cur.expand(B)
                t_scaled = t_cur_batch * timescale

                evt_start = torch.cuda.Event(enable_timing=True)
                evt_net = torch.cuda.Event(enable_timing=True)
                evt_v2x = torch.cuda.Event(enable_timing=True)
                evt_start.record()
                v_pred = net(
                    x, t_scaled, caption_embs,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )
                evt_net.record()

                if t_next.item() > 0:
                    x0_pred = self._velocity_to_x0(x, v_pred, t_cur_batch)
                    evt_v2x.record()
                    eps_infer = torch.randn(
                        x0_pred.shape, device=x0_pred.device,
                        dtype=x0_pred.dtype, generator=generator,
                    )
                    x = (1.0 - t_next) * x0_pred + t_next * eps_infer
                else:
                    x = self._velocity_to_x0(x, v_pred, t_cur_batch)
                    evt_v2x.record()

                torch.cuda.synchronize()
                step_ms = evt_start.elapsed_time(evt_v2x)
                net_ms = evt_start.elapsed_time(evt_net)
                v2x_ms = evt_net.elapsed_time(evt_v2x)
                step_times.append((step_ms, net_ms, v2x_ms))

        logger.info(
            "PiD _sample_loop timings (ms): steps=%s, net_avg=%.0f, v2x_avg=%.0f",
            [f"{s:.0f}" for s, _, _ in step_times],
            sum(n for _, n, _ in step_times) / len(step_times),
            sum(v for _, _, v in step_times) / len(step_times),
        )
        torch.cuda.empty_cache()
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        lq_latent: torch.Tensor,        # (B, C_lq, zH, zW)
        caption: str | list[str],
        output_size: tuple[int, int],    # (H, W) pixel output
        degrade_sigma: float = 0.0,
        num_steps: int = 4,
        seed: int = 0,
    ) -> torch.Tensor:
        """Run PiD decode.

        Returns:
            (B, 3, H, W) tensor in [-1, 1].
        """
        if isinstance(caption, str):
            caption = [caption]
        B = len(caption)

        # Use tensor_kwargs (always device="cuda", dtype=float32 container)
        # to match the original PixelDiTModel: the student was calibrated
        # against a float32 container; autocast(...) inside _sample_loop
        # handles mixed-precision matmuls. Feeding bf16 directly skips the
        # float32 container and shifts the input distribution.
        caption_embs = self.text_encoder.encode(caption)
        caption_embs = caption_embs.to(**self.tensor_kwargs)

        lq_latent = lq_latent.to(**self.tensor_kwargs)
        degrade_sigma_tensor = torch.full(
            (B,), float(degrade_sigma), **self.tensor_kwargs
        )

        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        img_h, img_w = output_size
        noise = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)

        # Resolve the net to use: compiled (per-resolution cache) or eager.
        text_len = min(caption_embs.shape[1], self.net.txt_max_length)
        net = self._maybe_compile_net(img_h, img_w, text_len)

        effective_steps = num_steps or self._cfg.student_sample_steps

        if effective_steps == 1:
            t_student = torch.full(
                (B,), self._cfg.student_t_list[0],
                **self.tensor_kwargs,
            )
            t_scaled = t_student * self._cfg.fm_timescale
            v = net(
                noise, t_scaled, caption_embs,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            x0 = self._velocity_to_x0(noise, v, t_student)
        else:
            t_list = self._get_t_list(torch.device("cuda"), effective_steps)
            x0 = self._sample_loop(
                noise, t_list, caption_embs, lq_latent,
                degrade_sigma_tensor, net=net, generator=gen,
            )

        return x0.clamp(-1, 1)
