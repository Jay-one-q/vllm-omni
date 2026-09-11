# Regional Compilation

Regional compilation applies `torch.compile` to the repeated transformer blocks
declared by a diffusion model. It is the default compilation scope when
diffusion inference runs without `--enforce-eager`.

## Configuration

Dynamic compilation is enabled by default so the compiled regions can handle
mixed resolutions. For a fixed-shape workload, disable it explicitly:

```bash
vllm serve <model> --omni --no-diffusion-compile-dynamic
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    diffusion_compile_granularity: regional
    diffusion_compile_dynamic: false
```

For an experimental whole-transformer compile scope, set
`--diffusion-compile-granularity full` or use
`diffusion_compile_granularity: full` in the deploy configuration. Full scope may
still contain graph breaks. It is rejected when HSDP, sequence parallelism, CPU
offload, or layerwise offload is enabled. Use regional scope with those features.

These settings control the generic model-runner compilation path. Pipelines
that provide their own `setup_compile()` implementation manage their compilation
policy independently. Compilation is lazy, so backend or graph errors can first
surface on the initial request.

## Pinning one packed shape (MiniMax-H3)

Compiled regions are keyed by input shape. MiniMax-H3 packs a request into a
row count that follows the request -- a longer prompt, more frames, a larger
frame -- and rounds it up to the next 64-row boundary, so two requests can land
on two different packed lengths.

Under the default `diffusion_compile_dynamic: true` a new packed length is
absorbed by the dynamic shape, so it is not by itself a recompilation. It still
hands the backend a shape it has not autotuned, which is the risk this knob
exists to remove: a fixed-shape deployment (`--no-diffusion-compile-dynamic`)
recompiles the transformer blocks per shape, and a first-seen large shape can
push autotuning into its worst-case memory use.

A request pins the length with `extra_args["pad_seq_len"]`:

```python
sampling_params = SamplingParams(extra_args={"pad_seq_len": 54080})
```

The value must be a positive multiple of 64 and must cover the rows the request
actually uses; the packed sequence is then padded to it instead of to the next
64-row boundary. Pick a bucket that covers every request dimension the
deployment accepts -- prompt length, frame count, frame size and reference
blocks all change the used row count. The server logs the effective length as
`MiniMax H3 packed sequence: ... pad_seq_len=... used=... seq_len=...` whenever
a request pins it. The padding rows are masked out, so their cost is the
attention and feed-forward work on those rows.

Use `--enforce-eager` to disable the model runner's generic compile setup.
Pipelines that compile internally define their own eager-mode behavior.

## Ascend NPU backend

On Ascend, when the MindIE-SD package is installed and the diffusion
configuration is compatible, the platform selects MindIE-SD pattern fusion
automatically; otherwise the pipeline runs eagerly. No extra flag is needed:

```bash
vllm serve Qwen/Qwen-Image --omni --dtype bfloat16 \
    --diffusion-compile-granularity full \
    --no-diffusion-compile-dynamic --cache-backend none
```

The same fields are accepted by the Python API and per-stage deploy config:

```yaml
stages:
  - stage_id: 0
    enforce_eager: false
    diffusion_compile_granularity: full
    diffusion_compile_dynamic: false
    cache_backend: none
```

The MindIE-SD backend currently requires unquantized BF16 weights, one NPU,
static shapes (`--no-diffusion-compile-dynamic`), no CPU/layerwise offload, no
diffusion cache backend, and no LoRA or sleep mode. Full compilation
additionally rejects HSDP and sequence parallelism. Only generic MindIE-SD
fusion patterns run; model-specific residual-gate patterns stay disabled. When
the configuration is incompatible, the server logs the reason and falls back to
eager execution; `--enforce-eager` always disables compilation.

MindIE-SD is an optional NPU dependency, not installed for other platforms.
Use a build compatible with the target PyTorch, torch_npu and CANN versions.
The build must export `MindieSDBackend` and `CompilationConfig` from
`mindiesd.compilation`. See the
[MindIE-SD compilation documentation](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docs/zh/features/compilation.md).
An NPU-tested dependency version matrix is not yet established.

Backend construction happens inside the worker, before loading weights so
MindIE-SD can register its CANN custom operators early. Synchronous compile
setup failures are logged and the pipeline runs uncompiled; lazy compilation
errors surface on the first request. Backend selection itself only falls back
to eager on a missing dependency or an incompatible configuration, with the
reason logged. Pipelines with their own `setup_compile()` continue to
own their Inductor policy. Callable platform backends use the generic
model-runner compilation path.

### NPU validation checklist

Unit tests cover backend selection, argument forwarding and lazy Dynamo backend
invocation, not the correctness or performance of MindIE-SD on Qwen-Image.
Before treating this experimental path as validated:

1. Record the model revision, Omni/vLLM/vLLM-Ascend revisions, MindIE-SD build,
   PyTorch/torch_npu/CANN versions, device model and available memory.
2. Run eager and compiled servers separately, using the same model, prompt,
   seed, inference steps, resolution and guidance. For eager, add
   `--enforce-eager` to the command above. Do not enable offload or cache.
3. Send identical requests sequentially, separating first-request latency from
   multiple warm requests. The
   [text-to-image example](../examples/online_serving/text_to_image.md)
   describes requests. Capture `TORCH_LOGS=graph_breaks,recompiles` server logs
   and MindIE-SD backend debug logs to confirm actual graph processing.
   A "selected backend" or "configured for lazy compile" message is not proof.
4. Compare intermediate denoising outputs/latents and final images against eager
   with identical initial noise; record numerical error as well as visual quality.
5. Record warm end-to-end latency, synchronized DiT step latency and peak NPU
   memory. Inspect graph breaks and recompilations before claiming a speedup.
   Server startup warmup may already consume some compilation cost.
6. Test changes in prompt length and resolution separately. Runtime backend
   failures propagate; the runner does not retry a partially executed request
   eagerly.

Other models and offload/cache combinations require separate NPU validation.
