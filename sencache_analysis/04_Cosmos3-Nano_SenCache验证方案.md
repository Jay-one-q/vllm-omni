# Cosmos3-Nano SenCache 验证方案

分析日期：2026-10-23。范围：`nvidia/Cosmos3-Nano` 的 T2V/T2I 请求级路径；首轮基准是单卡、eager、顺序 CFG、batch=1，CFG-parallel 与 SP 作为单独的多卡项验证。
本文是**验证步骤与判据**，不是性能结论：代码已落地，但本文没有在 H200/H100 上实际运行。

---

## 1. 已落地的代码

| 文件 | 作用 |
| --- | --- |
| `vllm_omni/diffusion/cache/sencache/config.py` | `SenCacheConfig`：阈值、K、时间网格、标定开关 |
| `vllm_omni/diffusion/cache/sencache/profile.py` | `SensitivityProfile`：npz 读写、时间分箱查表、空箱最近邻填充 |
| `vllm_omni/diffusion/cache/sencache/calibrate.py` | `ProfileAccumulator` + `merge/inspect` CLI |
| `vllm_omni/diffusion/cache/sencache/state.py` | 参考点、每分支残差、连续复用计数 |
| `vllm_omni/diffusion/cache/sencache/hook.py` | 判定与复用主逻辑，含在线标定探针 |
| `vllm_omni/diffusion/cache/sencache/backend.py` | `enable/refresh`，Cosmos3 pipeline 白名单 |
| `vllm_omni/diffusion/cache/selector.py` | 新增 `sen_cache` 路由 |
| `vllm_omni/diffusion/data.py` | 新增 `sen_*` 配置字段 |
| `vllm_omni/diffusion/cache/teacache/extractors.py` | Cosmos3 extractor 暴露原始 latent 与 timestep |
| `vllm_omni/entrypoints/cli/serve.py` | `--cache-backend` 帮助文本 |
| `tests/diffusion/cache/test_sencache.py` | 判定、K 上限、分支共享、刷新、标定、合并、backend |

实现要点：

- 缓存边界是 `extract_cosmos3_context()` 给出的 `run_transformer_blocks()`，即 `_gen_preprocess` 之后、`_gen_postprocess` 之前的 GEN stack。
- 判定使用原始 latent（`hidden_states` 参数）和原始 timestep，不经过 patchify/pack，符合论文的输入空间定义。
- 命中时执行 `postprocess(hidden + cached_residual)`，norm/proj/unpatchify 仍每步执行，与 TeaCache/SeaCache 的边界一致。
- 判定在每个 denoising step 只做一次（cond 首个调用），同一步的 `cond/uncond/cond_no_control` 共享决策，残差按分支独立保存。
- 阈值是 **RMS 输出变化** 口径，不是论文里 Wan 的绝对 L2 数值；默认值只是占位，必须扫参。

---

## 2. 算法与 Cosmos3 的映射

| SenCache 量 | Cosmos3 中的来源 |
| --- | --- |
| `x_t` | extractor 的 `hidden_states`（`[B, C, T, H, W]` 噪声 latent） |
| `t` | extractor 的 `timestep`（FlowMatch 的 `[0, 1000]`） |
| `f(x,t)` | 模型返回的 video 预测（`postprocess` 之后的 `noise_pred`；标定只取 video，不含 action/sound） |
| `alpha_x(t)` | 标定均值的 `rms(f(x+dx,t)-f(x,t)) / rms(dx)` |
| `alpha_t(t)` | 标定均值的 `rms(f(x,t+dt)-f(x,t)) / |dt|` |
| 连续上限 `K` | `sen_max_consecutive_reuses` |
| 早/晚阈值 | `sen_threshold_start` / `sen_threshold_main`，切换点 `sen_start_fraction` |

在线判定：

```text
S = alpha_x(t_ref) * max_b rms(x_t - x_ref)_b + alpha_t(t_ref) * |t - t_ref|
skip  <=>  S <= threshold(step)  and  consecutive_reuses < K
```

`max_b` 表示 batch 内取最保守样本；Cosmos3 验证固定 batch=1。

---

## 3. 前置条件

1. 单卡 H200/H100，`vllm-omni` 环境可跑 `vllm serve nvidia/Cosmos3-Nano --omni`（见 `recipes/models/nvidia/Cosmos3-Nano.yaml`）。
2. 关闭会影响缓存边界、判定一致性或性能归因的开关，分三类（原因见 3.1）：
   - **代码直接拒绝**：`--enable-hsdp`/FSDP、`--step-execution`。
   - **已放开、建议单独实测**：`--cfg-parallel-size > 1`（判定与条件无关，跨 CFG rank 做 `MAX` 投票）、`--ulysses-degree/--usp > 1`（SP/CP）。
   - **编译**：推荐 `--enforce-eager`；若要开编译只用 `--diffusion-compile-granularity regional`，不要用 `full`。
3. 准备 4~8 条覆盖静态/快速运动/人物/镜头变化的 prompt，写入 `assets/sen_prompts.txt`（一行一条）。
4. 固定一组采样参数并贯穿标定与推理：`size=1280x720`、`num_frames=189`、`fps=24`、`num_inference_steps=35`、`flow_shift=10.0`、`guidance_scale=6.0`、`seed` 集合固定。

> 标定产物与 shape / 步数 / scheduler / guidance 绑定。PoC 想更快可以降到 `640x360 / 49 帧 / 21 步`，但标定与推理必须使用同一组参数。

### 3.1 为什么是这些限制

- **eager / 不用 full compile**：`SenCacheHook.new_forward` 包含数据相关的 Python 分支（阈值比较、`.item()`、状态字典），并标注了 `@torch.compiler.disable`；同时 extractor 的 `run_transformer_blocks()` 直接调用 `module._run_gen_stack()`，会绕过 `full` 模式编译后的根 `forward`。因此 `full` 既拿不到编译收益，又可能引入 graph break 和反复重编译。`regional` 只编译 block，根 hook 仍是 eager，理论上可以共存，但需要和 SenCache 分开做 A/B，不能混在首轮验证里。
- **不开 SP/CP**：skip 决策必须在同一个 collective 域内所有 rank 完全一致。只要一部分 rank 跳过 GEN stack、另一部分进入 attention 的 all-to-all/all-gather，就会挂死或产生错误结果。当前实现用“复制的原始 latent + SP 组内 `MAX` all-reduce”来保证一致（`hook._synchronize_decision`），设计上可用，但 v1 没有多卡实测；而且标定探针会在每个 rank 重复执行，成本按 world size 放大。
- **CFG-parallel（已放开）**：CFG-parallel 下每个 rank 只跑 cond 或 uncond 一支，但 SenCache 的分数只读取 `x_t` 和 `t`，两者在 CFG rank 间是复制的，因此各 rank 对同样输入算出同样的 `S` 和同样的参考点更新序列，**不需要跨分支广播**。这里刻意**不**在 CFG 组内投票：CFG 各 rank 的 transformer 调用次数可能不均衡（例如 guidance interval 只跳过 uncond），跨分支 collective 反而会死锁；这与 SeaCache 的做法一致（它只在 SP/FS/DLO 组内投票）。标定只在 `cfg_rank == 0` 上执行，避免把 uncond 曲线混进同一份 profile。
- **不开 HSDP/FSDP**：跳过整段 GEN stack 会改变参数 all-gather / reduce-scatter 的调用顺序，可能死锁或读到未聚合的分片；hook 安装时直接拒绝。
- **不开 step execution**：runner 的 step 模式还没有 request 级缓存状态、取消与重排处理，当前明确抛出 `Step mode does not support cache_backend yet`。

---

## 4. 验证流程

### 4.0 基线：关闭所有近似缓存

```bash
vllm serve nvidia/Cosmos3-Nano --omni --port 8000 --init-timeout 1800
```

每条 prompt 跑 2 个固定 seed，保存 `baseline_<prompt>_<seed>.mp4`，并记录端到端耗时 P50（先各跑 1 次预热）。

### 4.1 标定：生成 sensitivity artifact

**这一步做什么**：SenCache 的判定需要两个模型属性——`alpha_x(t)`（latent 方向导数）和 `alpha_t(t)`（timestep 方向导数），它们无法从权重直接读出，只能用小规模真实轨迹做有限差分估计。标定就是把这个“每 timestep 的灵敏度曲线”一次性算出来，推理时只查表，不再做任何额外前向。

**怎么做**：任意相邻两步之间，我们已经有真实轨迹 `(x_k, t_k, y_k)` 和 `(x_{k+1}, t_{k+1}, y_{k+1})`，再补两次探针：

```text
J_x(t_k) ≈ rms(f(x_{k+1}, t_k) - y_k) / rms(x_{k+1} - x_k)      # 固定 t，只动 latent
J_t(t_k) ≈ rms(f(x_k, t_{k+1}) - y_k) / |t_{k+1} - t_k|        # 固定 latent，只动 timestep
```

两条曲线按归一化时间箱累加，多个 prompt 先分别累加再 `merge` 取均值。

**为什么每一步都必须全算**：探针要用真实的 `x_k / t_k / y_k` 作为参考点。一旦在某一步复用缓存，参考点就与真实轨迹脱节，算出来的 `J` 不是当前轨迹上的导数，曲线会系统性地偏小。所以标定模式强制 compute，且跳过所有近似复用。

**为什么约为正常推理的 3 倍开销**：每步 = 1 次真实前向 + 2 次探针前向。这也是标定只跑 4~8 条 prompt、并且建议用小分辨率 PoC 的原因。

**为什么要多个 prompt**：单条轨迹的 `J` 方差大，且只覆盖一种运动模式。论文显示 8 条多样视频已足够稳定；所有请求的样本按时间箱累加求均值，空箱用最近邻填充。

**产物**：`/tmp/sen/cosmos3_nano_calib.npz`（单一共享文件）。它绑定模型、缓存边界、shape、步数、scheduler/flow_shift 和 CFG 配置；换任一条件都应重新标定。

```bash
mkdir -p /tmp/sen
vllm serve nvidia/Cosmos3-Nano --omni --port 8000 --init-timeout 1800 \
  --cache-backend sen_cache \
  --cache-config '{"sen_calibration_output": "/tmp/sen/cosmos3_nano_calib"}'
```

PowerShell 下把最外层单引号换成单引号包裹双引号 JSON（PowerShell 单引号内保留双引号即可）。

对每条标定 prompt 发一次请求（参数与 4.0 完全一致）：

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=$(sed -n '1p' assets/sen_prompts.txt)" \
  -F "negative_prompt=$(cat assets/negative_prompt.json)" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" \
  -F "num_inference_steps=35" -F "guidance_scale=6.0" \
  -F "flow_shift=10.0" -F "seed=123" \
  --output /tmp/sen/calib_1.mp4
```

预期日志：

```text
SenCache calibration mode: every step runs full; samples from all ranks are merged into a single shared .npz under /tmp/sen/cosmos3_nano_calib
SenCache calibration profile check passed: /tmp/sen/cosmos3_nano_calib.npz (bins=128, sampled=..)
SenCache merged calibration artifact: /tmp/sen/cosmos3_nano_calib.npz
```

所有 worker 进程对同一个路径做"加锁 → 读取已有产物 → 合并 → 原子替换"，所以：

- **一次部署只产生一个 `.npz`**，后续每条请求的数据自动累加进去，不需要手工 merge；
- 多节点部署要求该路径位于共享文件系统；若各节点写本地 `/tmp`，仍会各有一份；
- 同目录会出现一个 `<name>.npz.lock` 文件，用于跨进程互斥，可保留。

### 4.2 检查 profile（merge 仅用于历史产物）

标定结束时已经自动合并并检查，通常只需要 `inspect`：

```bash
python -m vllm_omni.diffusion.cache.sencache.calibrate inspect \
  --profile /tmp/sen/cosmos3_nano_calib.npz
```

`inspect` 做两件事：打印时间网格与逐箱 `alpha_x/alpha_t`，并跑与在线写入相同的语义检查（`profile check: OK` 或列出 WARNING）。

只有当产物来自旧版本（每进程一个 `<pid>.npz`）时才需要手工合并：

```bash
python -m vllm_omni.diffusion.cache.sencache.calibrate merge \
  --inputs "/tmp/sen/cosmos3_nano_calib.*.npz" \
  --out /tmp/sen/cosmos3_nano_profile.npz
```

检查项：

- 曲线非负、有限，两端（t≈1000 与 t≈0）不出现异常尖峰；
- `alpha_t` 不应恒为 0；若恒为 0，说明探针没有真正改变 timestep（检查模型是否忽略 `timestep` 输入）；
- 占用箱数量与步数匹配即可（35 步 / 128 箱时约 27% 是正常的，不应据此判定失败）。

### 4.3 用 profile 跑 SenCache

```bash
vllm serve nvidia/Cosmos3-Nano --omni --port 8000 --init-timeout 1800 \
  --cache-backend sen_cache \
  --cache-config '{"sen_profile_path": "/tmp/sen/cosmos3_nano_calib.npz",
                   "sen_threshold_start": 0.02,
                   "sen_threshold_main": 0.08,
                   "sen_start_fraction": 0.2,
                   "sen_max_consecutive_reuses": 2,
                   "sen_time_max": 1000.0}'
```

标定产物本身同时包含原始累加器与 finalize 后的曲线，可以直接作为 `sen_profile_path`；如果你做了 §4.2 的手工 merge，则改用 merge 的输出路径。

用 4.0 相同的 prompt / seed / 参数请求，保存 `sencache_<prompt>_<seed>.mp4`。

每次请求结束（下一次请求准入前）会打印上一条请求的统计：

```text
SenCache request summary: full=.. skip=.. skip_ratio=.. (next steps=35)
```

`skip_ratio == 0` 表示判定从未命中，优先调大 `sen_threshold_main`。

### 4.4 参数扫描

固定 profile，按“先主阈值、再早阈值、最后 K”的顺序扫：

| 参数 | 建议网格 |
| --- | --- |
| `sen_threshold_main` | 0.01, 0.02, 0.05, 0.10, 0.20 |
| `sen_threshold_start` | `main/4`, `main/2`（保证早段更严格） |
| `sen_max_consecutive_reuses` | 1, 2, 3 |
| `sen_start_fraction` | 0.1, 0.2, 0.3 |

每组配置至少 2 条 prompt × 2 seed 初筛，候选配置再扩到 10~20 条。每次只改一个维度，避免把 skip 率变化误归因。

### 4.5 质量与时延

质量（与同 seed baseline 逐帧对比）：

```bash
ffmpeg -y -i baseline_1_123.mp4 -vsync 0 /tmp/base/%04d.png
ffmpeg -y -i sencache_1_123.mp4  -vsync 0 /tmp/sen/%04d.png
python - <<'PY'
import glob, numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

base = sorted(glob.glob("/tmp/base/*.png"))
cand = sorted(glob.glob("/tmp/sen/*.png"))
assert len(base) == len(cand) and base, (len(base), len(cand))
psnr, ssims = [], []
for a, b in zip(base, cand):
    x, y = load(a), load(b)
    mse = float(np.mean((x - y) ** 2))
    psnr.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    ssims.append(ssim(x, y, channel_axis=2, data_range=1.0))
print(f"PSNR={np.mean(psnr):.2f} dB  SSIM={np.mean(ssims):.4f}")
PY
```

需要 LPIPS 时安装 `lpips` 后对同一批帧计算；音频轨可用 `ffmpeg -map 0:a` 单独导出后比较。

时延：同一配置、同一 prompt，预热后重复 5 次取 P50；同时记录 `skip_ratio` 与端到端耗时，不要只用单次数字。

---

## 5. 判定门槛（建议）

| 检查项 | 通过条件 |
| --- | --- |
| 功能 | 三个配置（baseline / 标定 / SenCache）都能稳定出片；重复请求结果一致（固定 seed） |
| 正确性 | `skip_ratio > 0`；`skip_ratio` 随阈值单调变化；无 OOM、无 hang |
| 质量 | 与 baseline 相比 SSIM 下降在预先写定的预算内（例如 ≥0.95），PSNR 下降 ≤1 dB |
| 收益 | 在质量预算内，端到端 P50 相对 baseline 有可测收益；与同模型可用的 cache-dit/sea_cache 比较后再决定是否保留 |
| 状态 | A→B→A 连续请求无状态串扰；异常/取消后下一次请求仍从首步全算 |

若 `skip_ratio` 很高但质量明显下降，先降 `sen_threshold_main`，再降 `K`；若质量几乎无变化但时延没有收益，说明固定开销（文本编码、VAE、preprocess）占比过高，应停止扩大范围。

---

## 6. 排错

| 现象 | 原因 / 处理 |
| --- | --- |
| `sen_cache does not support pipeline ...` | 只对 `Cosmos3OmniDiffusersPipeline` 开放 |
| `requires a sensitivity profile` | 没给 `sen_profile_path`，且未开 `sen_calibration_output` |
| CFG/SP 多卡下 hang | 检查各 rank 的 latent/timestep 是否一致、SP 组是否初始化；SP 组内做 `MAX` 投票，CFG 组刻意不投票（避免不均衡 dispatch 死锁） |
| `does not support FSDP/HSDP parameter sharding` | 关闭 `--enable-hsdp` / FSDP |
| 标定不写文件 | 请求未带 `num_inference_steps`、请求被取消、输出目录不可写，或与已有产物的 `model_id/boundary` 不一致导致合并失败（看 WARNING） |
| 日志 `requires denoise step metadata` | pipeline 没暴露 `current_step_index`；Cosmos3 正常路径应该可用，检查是否被其它 wrapper 替换 |
| `alpha_t` 全 0 | 探针的 timestep 未生效；确认 transformer 真正使用 `timestep` 参数 |
| `skip_ratio` 始终 0 | profile 的 `alpha_*` 偏大或阈值偏小；先 `inspect`，再放大阈值 |
| 开启 `--step-execution` 报错 | runner 明确不支持 cache backend + step mode |

---

## 7. 已知限制

- 仅验证 Cosmos3；其它模型需要各自的 extractor 暴露 `sencache_latent` / `sencache_timestep`，并重新标定。
- 标定只取 video 输出；joint audio/action 的质量与同步未纳入判定。
- I2V/transfer 的探针只扰动 `hidden_states`，固定 controls；可用但未验证。
- batch>1 使用“最保守样本”决策，残差仍按分支共享一个参考点；连续批处理未支持。
- `SenCache` 根 hook 是 `torch.compiler.disable` 的，且判定含 host 同步；不要与 `full` 编译或 CUDA/NPU graph 叠加验证。
- layerwise offload 未验证：跳过 GEN stack 时被跳过层的 offload hook 不会触发，收益与状态需要单独测量。
- CFG-parallel / SP 已放开但未做多卡实测；首次运行时建议把进程组超时调大，并观察是否在某个 step 挂住。
- 标定在 CFG-parallel 下只由 `cfg_rank == 0` 记录；同一部署内所有记录进程在文件锁保护下合并到同一个 `.npz`。多节点部署需要共享文件系统，否则每个节点会各写一份。

---

## 8. 回滚

去掉 `--cache-backend sen_cache --cache-config ...` 即恢复原生路径；代码改动中只有 `selector.py`、`data.py`、`teacache/extractors.py`（新增 `extra_states` 键）和 `serve.py` 帮助文本触及既有文件，其余为新增目录与测试，删除 `vllm_omni/diffusion/cache/sencache/` 并还原上述四处即可完全移除。
