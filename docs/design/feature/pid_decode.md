# PiD (Pixel Diffusion) Super-Resolution Decode

!!! info "Feature Status"
    Enabled per pipeline. Currently wired for **Qwen-Image**; the core
    decoder is backbone-agnostic (Flux / SD3 / SDXL / Flux2 net configs are
    already registered). PiD is an optional **post-denoise** decoder — when
    disabled, the standard VAE path is unchanged.

This document describes how PiD (Pixel Diffusion) is integrated as an
optional super-resolution decoder for LDM pipelines. It captures the
**parameter surface** added by the feature and the **adaptation recipe**
for wiring a new backbone pipeline to PiD.

---

## Table of Contents

- [References](#references)
- [Overview](#overview)
- [Parameter Surface](#parameter-surface)
- [Architecture](#architecture)
- [Adaptation Recipe](#adaptation-recipe)
- [Usage Examples](#usage-examples)
- [Related Files](#related-files)

---

## References

- **Paper**: [PiD: Pixel Diffusion for Fast Super-Resolution](https://arxiv.org/abs/2605.23902)
- **Project**: [PiD — NVIDIA Research (SIL)](https://research.nvidia.com/labs/sil/projects/pid/)
- **Official repository**: [nv-tlabs/PiD](https://github.com/nv-tlabs/PiD)
- **Weights / checkpoints (Hugging Face)**: [nvidia/PiD](https://huggingface.co/nvidia/PiD)
- **Qwen-Image backbone checkpoint** (used by this feature):
  `checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth`
  — the v1pt5 **4-step distilled** checkpoint in `.pth` format.

---

## Overview

PiD replaces the VAE decode step with a distilled pixel-diffusion model that
takes the LDM `x_0` latent as a low-quality (LQ) condition and synthesises a
higher-resolution RGB image. The distilled checkpoint runs **4 SDE steps** by
default and uses a Gemma-2-2b-it text encoder for caption conditioning.

**Core behaviour:**

- Drop-in hook in `_decode_latents`: returns an image tensor when PiD is
  active, otherwise `None` and the pipeline falls back to VAE.
- Model-agnostic core (`vllm_omni/diffusion/pid/`): a pipeline only declares
  its `PID_BACKBONE` name; per-VAE differences (latent channels, spatial
  compression) live in the config registry.
- Weights are **eager-loaded** at pipeline `__init__` and stay resident on
  GPU; PiD is declared as a *resident module* so the CPU offloader never
  swaps it out.
- Compilation follows the main model: `PidDecoder` inherits the pipeline's
  `enforce_eager` flag, so PiD is compiled when the main model compiles and
  stays eager when the main model is eager.
- Checkpoint resolution supports a **local `.pth` path**, an **HF reference**
  (`<repo>/<subfolder>/<file>`), or — when `--pid-checkpoint` is omitted —
  an **automatic download** of the matching official checkpoint from the
  `nvidia/PiD` repo, keyed by backbone.
- Per-request overrides (enable/disable, scale, num_steps, seed,
  `degrade_sigma`) flow through `OmniDiffusionSamplingParams.pid_decode`.
- Under tensor parallelism, PiD decode runs **only on rank 0** to avoid
  redundant computation and OOM on other ranks.

---

## Parameter Surface

The feature adds parameters at three layers. Each lower layer is the typed
form of the layer above; CLI flags are packed into a `dict` and re-injected
as `pid_decode` so they flow through the normal stage-config plumbing.

### 1. CLI Flags (`vllm_omni/entrypoints/cli/serve.py`)

All flags live in the `omni_config_group` and are prefixed with `--pid-`.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--pid-enable` | flag (bool) | `False` | Master switch. When set, the matching `--pid-*` keys are packed into a `pid_decode` dict. |
| `--pid-checkpoint` | str | `None` | PiD decoder checkpoint: a local `.pth` path, an HF reference `<repo>/<subfolder>/<file>`, or `None` for auto-download. |
| `--pid-gemma` | str | `Efficient-Large-Model/gemma-2-2b-it` | Gemma text encoder used by PiD (HF id or a local directory). |

> `scale` / `num_steps` / `seed` / `degrade_sigma` are **not** exposed as CLI
> flags. They default in `PidDecodeConfig` and can be overridden **per
> request** through `pid_decode` (see §4).

**Packing logic** (`AsyncOmniEngine.__init__`): when `--pid-enable` is set,
the engine pops the two keys above from `kwargs` and re-injects a single
`pid_decode` dict using the **config field names**:

| CLI key | Config key |
|---|---|
| `pid_checkpoint` | `checkpoint_path` |
| `pid_gemma` | `gemma_model` |

The dict is then forwarded into `OmniDiffusionConfig.pid_decode` via the
stage-config plumbing (`async_omni_engine.py::_init_diffusion_engine` adds
`"pid_decode": kwargs.get("pid_decode")` to the engine kwargs).

### 2. Stage Config (`vllm_omni/diffusion/data.py`)

`OmniDiffusionConfig` gains one field:

```python
pid_decode: "PidDecodeConfig | dict[str, Any] | None" = None
```

`None` keeps the standard VAE path. A `dict` is accepted for ergonomic CLI
plumbing; it is normalised to `PidDecodeConfig` by `PidDecodeMixin`.

### 3. Typed Config (`vllm_omni/diffusion/pid/decoder.py`)

```python
@dataclass(frozen=True)
class PidDecodeConfig:
    enabled: bool = False
    checkpoint_path: str = ""      # empty -> auto-download from nvidia/PiD by backbone
    gemma_model: str = ""          # Gemma text encoder (HF id or local dir)
    scale: int = 4
    num_steps: int = 4
    seed: int = 0
    degrade_sigma: float = 0.0     # noise injected into the LQ latent
    precision: str = "bfloat16"    # "bfloat16" | "float16" | "float32"
```

### 4. Per-Request Override (`vllm_omni/inputs/data.py`)

```python
@dataclass
class OmniDiffusionSamplingParams:
    ...
    pid_decode: dict[str, Any] | None = None
```

Accepted keys (all optional): `enabled`, `scale`, `num_steps`, `seed`,
`degrade_sigma`. The override is applied with `dataclasses.replace` on the
frozen `PidDecodeConfig`, so the pipeline-level config is never mutated.

| Override `enabled` | Pipeline `--pid-enable` | Behaviour |
|---|---|---|
| `False` | any | Skip PiD, use VAE. |
| `True` | `False` | **Error** — PiD weights are not lazily loaded; restart with `--pid-enable`. |
| `True` | `True` | Run PiD with per-request overrides. |
| `None` / absent | `True` | Run PiD with pipeline-level config. |

### 5. HTTP API (`vllm_omni/entrypoints/openai/protocol/images.py`)

`ImageGenerationRequest` gains one optional field:

```python
pid_decode: dict[str, Any] | None = Field(
    default=None,
    description="Per-request PiD decode configuration. "
                "Keys: enabled, scale, num_steps, seed, degrade_sigma.",
)
```

The field is wired through two paths in `api_server.py::generate_images`:

- `extra_body["pid_decode"]` — forwarded to the chat handler for the
  chat-completion-style image path.
- `_update_if_not_none(gen_params, "pid_decode", ...)` — forwarded to the
  standalone image-generation path.

`serving_chat.py::OmniOpenAIServingChat.generate_diffusion_images` reads
`extra_body.get("pid_decode")` and passes it into
`OmniDiffusionSamplingParams`.

---

## Architecture

### Data Flow

```
                    ┌──────────────────────────────────────────────┐
                    │           LDM Pipeline (e.g. QwenImage)      │
                    │                                              │
  denoise loop ───> │  x_0 latent (B, num_patches, C)              │
                    │       │                                      │
                    │       ▼                                      │
                    │  _unpack_latents  →  (B, C, 1, zH, zW)       │
                    │       │                                      │
                    │  _pid_override ← sampling.pid_decode         │
                    │  _pid_caption  ← prompt                      │
                    │       │                                      │
                    │       ▼                                      │
                    │  PidDecodeMixin.maybe_pid_decode(            │
                    │        latents_4d, height, width)            │
                    │       │                                      │
                    │       ├── override enabled=False?            │
                    │       │     return None  ──> VAE.decode(...) │
                    │       │                                      │
                    │       ├── rank != 0 (TP)?                    │
                    │       │     return None  (rank 0 only)       │
                    │       │                                      │
                    │       └── PidDecoder.decode(lq_latent,       │
                    │             caption, output_size=(H·s, W·s)) │
                    │             │                                │
                    │             ▼                                │
                    │       PidInferenceModel.generate_samples...  │
                    │       (PidNet + GemmaTextEncoder)            │
                    │             │                                │
                    │             ▼                                │
                    │       (B, 3, H·s, W·s) in [-1, 1]            │
                    └──────────────────────────────────────────────┘
```

### Key Components

1. **`PidDecodeMixin`** (`pid/mixin.py`): the only surface a pipeline
   touches. Provides `_resolve_pid_config`, `_init_pid_decoder`,
   `_declare_pid_resident`, and the `maybe_pid_decode` hook
   (`(latents_4d, height, width)`). Reads `_pid_caption` / `_pid_override`
   from the pipeline, resolves per-request overrides, and gates rank 0.

2. **`PidDecoder`** (`pid/decoder.py`): `nn.Module` wrapping
   `PidInferenceModel`. Loads weights eagerly in `load_weights()` (called
   from `_init_pid_decoder`) and exposes a `decode(...)` entry point. Its
   `enforce_eager` flag mirrors the pipeline's so PiD compiles iff the main
   model compiles.

3. **`PidInferenceModel`** (`pid/pid_model.py`): holds `PidNet` +
   `GemmaTextEncoder`, runs the 4-step distilled SDE sampler. Precision is
   resolved once at construction (autocast dtype vs. pure fp32).

4. **`PidDecodeConfig`** (`pid/decoder.py`): frozen dataclass normalising
   both CLI dict and programmatic construction.

5. **Config registry** (`pid/config.py`): per-backbone `*_PID_NET_CONFIG`
   (Qwen-Image / Flux / SD3 / SDXL / Flux2) + shared `PID_SAMPLING_CONFIG`
   + `PID_CHECKPOINT_REGISTRY` mapping each backbone to its official
   `nvidia/PiD` checkpoint.

6. **Checkpoint loader** (`pid/checkpoint.py`): `resolve_pid_checkpoint_path`
   handles local / HF-ref / auto-download; `load_pid_checkpoint` strips the
   `net.` prefix, drops `net_ema.*` and training-only aux heads
   (`lq_proj.lq_aux_rgb_head`), and tolerates zero-init LQ-projection keys.

### Tensor Parallelism

Under TP, `maybe_pid_decode` returns `None` on every rank except rank 0:

```python
if dist.is_initialized() and dist.get_rank() != 0:
    return None
```

This avoids redundant PiD forward passes and prevents OOM on non-zero ranks.
The LDM denoise loop still runs on all ranks as usual; only the post-denoise
decode is gated.

---

## Adaptation Recipe

Adapting a new LDM backbone pipeline to PiD requires **four** edits. No
per-model wrapper file is needed — the core is backbone-agnostic.

### Step 1: Inherit `PidDecodeMixin` and declare `PID_BACKBONE`

```python
from vllm_omni.diffusion.pid import PidDecodeMixin

class YourPipeline(
    nn.Module,
    YourCFGParallelMixin,
    PidDecodeMixin,                       # <-- add mixin
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    PID_BACKBONE: ClassVar[str] = "qwenimage"   # <-- backbone name
    ...
```

`PID_BACKBONE` must be one of the keys registered in
`vllm_omni/diffusion/pid/config.py::get_pid_net_config`
(`qwenimage`, `flux`, `sd3`, `sdxl`, `flux2`).

### Step 2: Initialise the decoder in `__init__`

```python
def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
    super().__init__()
    ...
    self._init_pid_decoder(od_config)     # <-- add this line
```

`_init_pid_decoder` is a no-op when `od_config.pid_decode` is `None` or
`enabled=False`, so pipelines that don't use PiD pay zero cost. When active,
it eagerly loads the weights and aligns `enforce_eager` with the pipeline.

### Step 3: Add the decode hook in `_decode_latents`

Insert the hook **right after latent unpack**, before the VAE normalisation.
The caption and per-request override are read from the pipeline attributes
`_pid_caption` / `_pid_override` (set by the call site, see Step 4):

```python
def _decode_latents(self, latents, height, width, output_type: str = "pil"):
    if output_type == "latent":
        return DiffusionOutput(output=latents, ...)

    latents_4d = self._unpack_latents(latents, height, width, self.vae_scale_factor)

    # PiD hook: returns image tensor or None (fall back to VAE).
    pid_out = self.maybe_pid_decode(latents_4d, height, width)
    if pid_out is not None:
        return DiffusionOutput(output=pid_out, ...)

    # ... existing VAE path unchanged ...
```

### Step 4: Set `_pid_caption` / `_pid_override` at the call sites

`_decode_latents` is called from two places in a typical pipeline — both
must set the two attributes before the call:

```python
# In the per-batch path:
self._pid_override = getattr(state.sampling, "pid_decode", None)
self._pid_caption, _ = self._extract_prompts([state.prompt] if state.prompt is not None else [])
return self._decode_latents(state.latents, height, width, output_type)

# In the single-request path:
self._pid_override = getattr(common_sampling_params, "pid_decode", None)
self._pid_caption = prompt
result = self._decode_latents(latents, height, width, output_type)
```

`_pid_caption` is the original text prompt; `_pid_override` is the per-request
override from `OmniDiffusionSamplingParams.pid_decode`.

### (Optional) Add a new backbone net config

If the backbone's VAE characteristics are not yet in the registry, add a
new entry in `vllm_omni/diffusion/pid/config.py`:

```python
YOUR_BACKBONE_PID_NET_CONFIG = _make_net_config(
    lq_latent_channels=<VAE latent channels>,
    latent_spatial_down_factor=<VAE spatial compression>,
)
```

register it in `get_pid_net_config`, and (optionally) add a matching entry
to `PID_CHECKPOINT_REGISTRY` for auto-download. All other `*_SHARED_*` args
are identical across backbones.

---

## Usage Examples

### Enable PiD at startup (CLI)

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --pid-enable \
  --pid-checkpoint /path/to/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth \
  --pid-gemma /path/to/gemma-2-2b-it
```

Every request on this server will use PiD decode with these defaults.
`--pid-checkpoint` and `--pid-gemma` may be omitted: the former then
auto-downloads the official checkpoint from `nvidia/PiD` (by backbone), and
the latter falls back to the `Efficient-Large-Model/gemma-2-2b-it` HF id.
There is **no** `--pid-scale` / `--pid-num-steps` / `--pid-seed` CLI flag;
tune those per request via `pid_decode`.

### Per-request override via HTTP API

Once the server is started with `--pid-enable`, individual requests can
tune or disable PiD. Send `pid_decode` inside the request body:

```json
{
  "model": "Qwen/Qwen-Image",
  "prompt": "a cat sleeping on a windowsill",
  "size": "1024x1024",
  "pid_decode": {
    "enabled": true,
    "scale": 4,
    "num_steps": 4,
    "seed": 42
  }
}
```

To disable PiD for one request on a PiD-enabled server:

```json
{
  "model": "Qwen/Qwen-Image",
  "prompt": "...",
  "pid_decode": {"enabled": false}
}
```

### Programmatic construction

When building `OmniDiffusionConfig` directly (e.g. in tests or benchmarks):

```python
from vllm_omni.diffusion.pid import PidDecodeConfig

od_config = OmniDiffusionConfig(
    ...,
    pid_decode=PidDecodeConfig(
        enabled=True,
        checkpoint_path="/path/to/model_ema_bf16.pth",   # or "" for auto-download
        gemma_model="/path/to/gemma-2-2b-it",
        scale=4,
        num_steps=4,
        seed=0,
        precision="bfloat16",
    ),
)
```

A `dict` with the same keys is also accepted (`pid_decode={...}`) for CLI
ergonomics.

---

## Related Files

- `vllm_omni/diffusion/pid/__init__.py` — public API exports
- `vllm_omni/diffusion/pid/config.py` — per-backbone `*_PID_NET_CONFIG`,
  `PID_SAMPLING_CONFIG`, `PID_CHECKPOINT_REGISTRY`, typed wrappers
- `vllm_omni/diffusion/pid/decoder.py` — `PidDecodeConfig`, `PidDecoder`
- `vllm_omni/diffusion/pid/mixin.py` — `PidDecodeMixin` (pipeline-facing
  surface)
- `vllm_omni/diffusion/pid/pid_model.py` — `PidInferenceModel`
  (PidNet + Gemma, precision handling, sampler)
- `vllm_omni/diffusion/pid/pid_net.py` / `pixeldit.py` / `lq_projection_2d.py`
  — PidNet backbone / PixDiT-T2I modules / LQ projection
- `vllm_omni/diffusion/pid/text_encoder.py` — `GemmaTextEncoder`
  (chi-prompt prefix, 300-token layout, `from_pretrained_with_prefetch`)
- `vllm_omni/diffusion/pid/checkpoint.py` — checkpoint resolution
  (local / HF-ref / auto-download) and loader (`net.` prefix stripping)
- `vllm_omni/diffusion/pid/context_parallel.py` — context-parallel helpers
  adapted to standard `torch.distributed`
- `vllm_omni/diffusion/data.py` — `OmniDiffusionConfig.pid_decode` field
- `vllm_omni/inputs/data.py` — `OmniDiffusionSamplingParams.pid_decode`
- `vllm_omni/engine/async_omni_engine.py` — CLI flag packing
  (`--pid-*` → `pid_decode` dict)
- `vllm_omni/entrypoints/cli/serve.py` — `--pid-*` CLI flags
- `vllm_omni/entrypoints/openai/protocol/images.py` —
  `ImageGenerationRequest.pid_decode`
- `vllm_omni/entrypoints/openai/api_server.py` — HTTP wiring
- `vllm_omni/entrypoints/openai/serving_chat.py` — chat-path wiring
- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py` —
  reference adaptation (Qwen-Image)
