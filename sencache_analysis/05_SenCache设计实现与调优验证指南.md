# SenCache 设计、实现与调优验证完整指南

编写日期：2026-10-27。对象：当前工作区 `vllm_omni/diffusion/cache/sencache/` 与 Cosmos3（`nvidia/Cosmos3-Nano`）接入路径。
本文是**代码与工程说明**，不是性能实测报告：所有收益数字都是模型化的推算，仓库内没有 H200/H100 实跑数据。

相关文档：[01 论文讲解](01_论文中文讲解.md)、[02 落地方案](02_vllm_omni缓存现状与落地方案.md)、[03 代码证据索引](03_代码证据索引.md)、[04 Cosmos3-Nano 验证方案](04_Cosmos3-Nano_SenCache验证方案.md)。本文补充的是 04 未展开的**实现细节、调参机理、验证矩阵与风险清单**。

---

## 0. 结论摘要

1. SenCache 在本仓库是一个**实验性、单模型、请求级**的全前向缓存 backend：缓存边界是 Cosmos3 GEN stack（`_gen_preprocess` 之后、`_gen_postprocess` 之前）。
2. 判定不是在输出空间做差分，而是用**标定得到的敏感度曲线** `alpha_x(t)`、`alpha_t(t)` 预测"如果现在不跑，输出会变化多少"，再与阈值比较。因此它必须先生成 profile 才能推理。
3. 决策量 `S = alpha_x(t_ref)·rms(x_t − x_ref) + alpha_t(t_ref)·|t − t_ref|`，物理量纲是**输出 RMS 变化**；阈值默认 `0.02/0.08` 只是占位，必须扫。
4. 同一 denoising step 内的多个分支（cond / uncond / cond_no_control）**共享一次判定**，残差按分支独立保存；这是它相对 TeaCache 的关键差异，也是"CFG 下不被判两次"的原因。
5. 未显式传 `num_inference_steps` 的请求**不会触发 refresh**（runner 的白名单不含 `sen_cache`），会造成跨请求状态残留与标定只写一次文件。这是当前最容易被忽略的运维约束，详见 §9.1。
6. 收益只来自被跳过的 GEN stack；`preprocess/postprocess`（patchify、投影、norm、unpatchify）、文本编码、VAE 每步/每请求照常执行。端到端加速比的上限是 `1 / (1 − skip_ratio · T_stack/T_step)`。

---

## 1. 设计原理

### 1.1 缓存什么、不缓存什么

Cosmos3 的一次 denoising step 可以粗分为四段：

```text
noisy latent x_t, t
    │
    ├─ _gen_preprocess      patchify / pack / 位置编码 / control 拼接     [每步执行]
    ├─ _run_gen_stack       GEN transformer blocks（主体算力）            [☑ 可跳过]
    └─ _gen_postprocess     norm / proj / unpatchify / 还原多模态         [每步执行]
    │
video / action / sound prediction
```

`extract_cosmos3_context()` 把中间一段包装成 `run_transformer_blocks()`，前后两段包装成 `postprocess()`，见 [extractors.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/teacache/extractors.py#L1748-L1798)。命中时执行的是：

```text
output = postprocess(hidden_states + cached_residual)
```

其中 `hidden_states` 是**当前步**的 packed 输入，`cached_residual` 是**上次真正执行**时 `stack(x_ref) − x_ref` 的差值。也就是说，跳过的只是"block 堆叠"这段映射，而该映射被近似为"输入平移 + 常数残差"。

这带来两个直接后果：

- 省下的时间 = GEN stack 的时间，不含前后处理；
- 残差是**分支相关**的（cond 与 uncond 的输出不同），但**step 相关**的参考点（`x_ref`, `t_ref`）是分支共享的。

### 1.2 一阶敏感度门控

论文的核心假设是：在一小段轨迹上，模型输出对 (latent, timestep) 的联合变化可用一阶展开近似：

```text
f(x_t, t) − f(x_ref, t_ref) ≈  ∂f/∂x · (x_t − x_ref) + ∂f/∂t · (t − t_ref)
```

两个偏导的尺度无法从权重直接读出，于是预先标定成两条与时间相关的曲线：

| 符号 | 含义 | 量纲 |
| --- | --- | --- |
| `alpha_x(t)` | 固定 t，输出 RMS 变化 / latent RMS 变化 | 无量纲 |
| `alpha_t(t)` | 固定 x，输出 RMS 变化 / timestep 差 | 1 / timestep |

于是门控分数：

```text
S = alpha_x(t_ref) * rms(x_t − x_ref) + alpha_t(t_ref) * |t − t_ref|
skip  ⇔  S <= threshold(step)  and  consecutive_reuses < K
```

`S` 与 threshold 都是**输出 RMS 变化**，所以阈值可以直接理解为"这一步允许输出偏多少"。这也是为什么不能照抄论文里 Wan 的绝对 L2 数值。

### 1.3 与 TeaCache / SeaCache / MagCache 的差异

| 维度 | TeaCache | SeaCache | MagCache | SenCache（本实现） |
| --- | --- | --- | --- | --- |
| 判据来源 | 输入变化量的经验多项式缩放 | 频域滤波指标的累积距离 | 历史幅值比外推 | 标定得到的 `alpha_x/alpha_t` 查表 |
| 是否需要离线产物 | 否（内置系数） | 否 | 可选标定 | **是（profile 必需）** |
| 是否看 timestep | 否 | 间接 | 否 | **是（显式 `alpha_t·dt`）** |
| 多分支决策 | 通常每分支独立 | 每分支 | 每分支 | **每 step 一次，分支共享** |
| 残差粒度 | 输出残差 | 频谱残差 | 输出残差 | **输出残差，按分支保存** |
| 决策开销 | 小 | 中（FFT） | 小 | 中（一次全 latent RMS + 若干 host sync） |

SenCache 的独特性在于"**先验曲线 + 在线位移**"：曲线来自标定（模型属性），位移来自当前请求（样本属性）。因此它对不同 prompt 的自适应性比固定多项式更好，但引入了对 profile 正确性的强依赖。

### 1.4 为什么必须标定

`alpha_x/alpha_t` 是模型的局部雅可比范数，随 timestep 剧烈变化（扩散早期输出对输入极敏感，后期平缓）。如果不标定：

- 用常数近似会让早期过度跳过（画质崩坏）或后期完全不跳过（没有收益）；
- 用 TeaCache 式的输入缩放系数，缺少 `dt` 项，无法解释"latent 几乎没动但 timestep 走了很远"的情形。

标定产物是**单一 `.npz`**，绑定：模型 id、缓存边界、timestep 网格、采样器/`flow_shift`、步数。换任一条件应重新标定。

---

## 2. 代码架构

### 2.1 文件职责

| 文件 | 关键符号 | 职责 |
| --- | --- | --- |
| [config.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/config.py#L13-L45) | `SenCacheConfig`、`threshold_for_step` | 阈值、K、时间网格、标定开关；参数合法性校验 |
| [profile.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/profile.py#L57-L136) | `SensitivityProfile` | npz 读写、按 timestep 查表、空箱最近邻填充、schema 校验 |
| [calibrate.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/calibrate.py#L47-L257) | `ProfileAccumulator`、`merge_into_file`、`profile_issues`、CLI | 在线累加、单文件加锁合并、语义检查、`inspect` 打印 |
| [state.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/state.py#L13-L63) | `SenCacheState` | 参考点、分支残差、连续复用计数、step 级门控缓存 |
| [hook.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/hook.py#L100-L229) | `SenCacheHook`、`apply_sen_cache_hook` | 判定与复用主逻辑、在线标定探针、SP 一致性投票 |
| [backend.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/backend.py#L50-L150) | `SenCacheBackend` | pipeline 白名单、profile 加载/标定器构建、hook 安装、请求边界 refresh 与统计 |
| [selector.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/selector.py#L52-L53) | `sen_cache` / `sencache` 路由 | 按名字构造 backend |
| [data.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/data.py#L608-L622) | `sen_*` 字段 | `--cache-config` 的用户面配置 |
| [extractors.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/teacache/extractors.py#L1791-L1798) | `sencache_latent` / `sencache_timestep` | 暴露**原始**噪声 latent 与 timestep |
| [serve.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/entrypoints/cli/serve.py#L833-L841) | `--cache-backend` help | 把 `sen_cache` 暴露给用户 |

### 2.2 调用链

```text
启动
  DiffusionModelRunner.__init__            (worker/diffusion_model_runner.py:423-448)
    └─ get_cache_backend("sen_cache", cfg)  → SenCacheBackend
    └─ backend.enable(pipeline)
         ├─ 校验 pipeline 白名单（Cosmos3OmniDiffusersPipeline / Cosmos3OmniPipeline）
         ├─ 标定模式：构建 ProfileAccumulator
         │  否则：SensitivityProfile.load(sen_profile_path)
         ├─ apply_sen_cache_hook(transformer, ...)
         │    └─ HookRegistry.register_hook("sen_cache", hook)  ← 包装 transformer.forward
         └─ pipeline._cache_context_factory = hook.cache_context

请求准入
  _refresh_cache_for_requests               (diffusion_model_runner.py:706-739)
    └─ backend.refresh(pipeline, num_inference_steps)
         ├─ 打印上一条请求的 full/skip/skip_ratio
         └─ hook.refresh() → state.reset()、_calibration_done=False

每条请求的每个 denoising step、每个分支
  pipeline_denoise_loop
    └─ predict_noise(_cache_context="cond"/"uncond"/"cond_no_control", **kwargs)
         └─ with pipeline._cache_context_factory(name):
              └─ transformer(**kwargs) → SenCacheHook.new_forward
```

Cosmos3 侧的关键接入点是 [pipeline_cosmos3.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1402-L1431)：它从 kwargs 里弹出 `_cache_context` 作为分支名，并用 `nullcontext` 兜底；而步号来自 `_set_denoise_step_metadata` 维护的 `current_step_index` / `num_timesteps`，见 [pipeline_cosmos3.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1986-L1999)。

### 2.3 状态机

`SenCacheState` 的各字段语义：

| 字段 | 写入时机 | 语义 |
| --- | --- | --- |
| `reference_latent` / `reference_timestep` | 真正执行 full 的那一步 | 下一次判定比较的基准；**分支共享** |
| `alpha_x` / `alpha_t` | 刷新参考点时按 `t_ref` 查 profile | 当前基准处的敏感度 |
| `branch_residuals[branch]` | 每次 full | `stack(x) − hidden_states`，按分支隔离 |
| `consecutive_reuses` | 每次判定为 skip | 连续复用步数，达到 K 强制重算 |
| `full_count` / `skip_count` | 每次执行/跳过 | 请求级统计，用于日志与调参反馈 |
| `step_gate[step]` | 每个 step 首次判定 | 同一步多分支共享的决策缓存 |
| `last_gate_step` / `last_ref_step` | 判定 / 刷新参考点 | 防止同一 step 重复判定或重复刷新 |

注意 `consecutive_reuses` 的粒度是**denoising step**，不是 transformer 调用次数；同一步的 uncond 复用不会把计数加两次（见 [hook.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/hook.py#L286-L289)）。

### 2.4 参考点与残差的组合语义

一次完整往返（以 CFG 为例）：

| step | 事件 | `x_ref` | cond 残差 | uncond 残差 | 输出 |
| --- | --- | --- | --- | --- | --- |
| 0 | 无参考点 → 强制 full | `x0` | `r_c0` | `r_u0` | 真实 |
| 1 | `S` 足够小 → split | `x0` | `r_c0` | `r_u0` | 近似 |
| 2 | `S` 仍小但 `reuses==K` → 强制 full | `x2` | `r_c2` | `r_u2` | 真实 |
| 3 | `S` 大 → full | `x3` | `r_c3` | `r_u3` | 真实 |

若某分支在该 step 没有可用残差（例如首次出现、或 shape/dtype 变化），该分支单独执行 full，并**清空参考点**，让下一步所有分支重新同步（见 [hook.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/hook.py#L217-L229)）。这是"宁可多算一步"的保守策略。

---

## 3. 实现逐条解读

### 3.1 入口与降级路径

`new_forward` 标注了 `@torch.compiler.disable`，并且有一串"降级为全算"的保护分支（见 [hook.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/hook.py#L169-L199)）：

| 条件 | 行为 | 原因 |
| --- | --- | --- |
| 开启 autograd | 全算 | 训练/可微场景不做近似 |
| 存在位置参数 | 全算 | 该 hook 依赖 keyword-only 调用才能拿到 latent/timestep |
| extractor 未暴露 latent/timestep | 全算 | 其它模型未适配 |
| 无 step 元数据 | 全算 | 阈值分段与 K 都依赖步号 |
| timestep 非有限 | 全算 | 防止 NaN 污染状态 |
| latent 非有限 | 刷新缓存并全算 | 同上 |
| 参考点 shape/device/dtype 变化 | 刷新缓存并全算 | 分辨率或批次变化 |
| FSDP/HSDP 参数分片 | **启动即报错** | 跳过堆叠会打乱 all-gather 顺序 |
| profile 缺失且非标定模式 | **启动即报错** | 没有先验无法判定 |

这些降级是"服务不挂"的兜底，不是性能路径；线上应保证一个都不触发（可通过 `_warn_once` 的 WARNING 观察）。

### 3.2 判定流程（伪代码）

```python
step = pipeline.current_step_index
num_steps = pipeline.num_timesteps
t = float(timestep[0])
branch = _active_branch or "cond"

if calibrating:
    out = postprocess(run_gen_stack())
    record_probes(...)            # 1 次真实 + 2 次探针
    refresh_reference()
    return out

compute = gate(step, num_steps, latent, t)      # 每 step 只算一次
residual = state.residual_for(branch, packed_input)
if not compute and residual is not None:
    skip_count += 1
    return postprocess(packed_input + residual)

out = run_gen_stack()
branch_residuals[branch] = out - packed_input
full_count += 1
if compute:
    refresh_reference()
else:
    clear_reference()             # 该分支没残差，强制下一步全员重算
return postprocess(out)
```

### 3.3 门控细节

`_resolve_gate`（[hook.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/hook.py#L255-L289)）的顺序很重要：

1. 若 `last_gate_step != step`，清空 `step_gate`，保证新 step 重新判定；
2. 如果该 step 已判定过，直接复用（分支共享）；
3. 无参考点 → full；
4. 参考点 shape/device/dtype 不一致 → full；
5. latent 含非有限值 → full；
6. 计算位移：`rms_dx = max_b rms((x_t − x_ref)_b)`（batch 内最保守）；
7. `score = alpha_x * rms_dx + alpha_t * dt`；
8. `compute = score > threshold(step) or reuses >= K`；
9. 做 SP 组内 `MAX` 同步；
10. 写回 `step_gate[step]`；若 skip，`reuses += 1`。

`threshold(step)` 按 `switch = round(num_steps * start_fraction)` 切成两段：`step < switch` 用 `threshold_start`，否则用 `threshold_main`。默认 `start_fraction=0.2`，即前 20% 的步更严格（扩散早期改动代价最大）。

### 3.4 阈值、K 与"参考点年龄"

- 参考点只在 full 时刷新，所以 `rms(x_t − x_ref)` 衡量的是**自上次真正计算以来累积的位移**，而不是相邻两步的位移。这与论文"累计误差预算"的设定一致。
- 因此 K 与阈值是**两个独立的刹车**：阈值控制"允许多大累积偏差"，K 控制"最多连续偷懒几步"。K 过大时，即使每步位移很小，残差近似也会随步数漂移。
- 首步必然 full（无参考点）；最后一步没有特殊保护，若判定为 skip，末步也是近似输出。若质量对末步敏感，可把 `start_fraction` 之外的末段阈值调小，或依赖 K=1/2 限制。

### 3.5 分布式一致性

| 并行 | 行为 | 理由 |
| --- | --- | --- |
| CFG-parallel | **不做跨分支投票**，各 rank 独立判定 | 分数只读 latent/timestep，两者在 CFG rank 间复制，结果天然一致；而各分支调用次数可能不均衡，跨分支 collective 会死锁 |
| SP/CP | 在 SP 组内做 `MAX` all-reduce（[hook.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/hook.py#L301-L320)） | skip 必须全组一致，否则一部分 rank 跳过 attention collective、另一部分进入，会挂死或算错 |
| FSDP/HSDP | 启动拒绝 | 参数 all-gather 顺序不能被打断 |
| TP/PP | 未声明支持，未验证 | 属于未覆盖组合 |

SP 下残差是 rank-local 的：`_shard_gen_prep(prep, defer_gather=True)` 先切分，`postprocess` 在命中/未命中时都会 gather。因此每个 rank 只保存自己分片的残差，显存开销随 SP 下降。

### 3.6 标定探针：公式与实现

代码在相邻两步 `(x_k, t_k, y_k)`、`(x_{k+1}, t_{k+1}, y_{k+1})` 之间补两次前向（[hook.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/hook.py#L328-L397)）：

```text
dx_rms  = rms(x_{k+1} − x_k)
probe_x = f(x_{k+1}, t_k)        → alpha_x = rms(probe_x − y_k) / dx_rms
dt      = |t_{k+1} − t_k|
probe_t = f(x_k, t_{k+1})        → alpha_t = rms(probe_t − y_k) / dt
```

两个样本都记到 `t_k` 所在的时间箱。要点：

- **单侧差分**：`alpha_x` 固定的是 `t_k`、动的是 latent；`alpha_t` 固定的是 `x_k`、动的是 timestep。二者都不是中心差分，步长也由采样器决定，所以曲线是"该采样轨迹上的局部斜率"，不是全局雅可比。
- **探针必须整段执行**：探针调用的是 `module._omni_original_forward`，绕过 hook，因此不会递归，也不走缓存。
- **标定强制全算**：一旦复用，`y_k/x_k/t_k` 就与真实轨迹脱节，斜率会系统性偏小。所以标定模式下 `calibrating=True` 分支永远 full。
- **只取 video 输出**：`_first_video(output)` 取第一个元素，action/sound 不参与曲线。
- **CFG 只在 rank 0 采样**：避免把 uncond 曲线混进同一份 profile；其余 CFG rank 全算但不写产物。
- **写入时机**：`step >= num_steps − 1` 时，把本请求的累加器与共享文件做"加锁 → 读取 → 合并 → 原子替换"，每条请求只写一次（`_calibration_done`）。写成功后清空内存累加器，避免下一条请求重复计入。

### 3.7 编译与 graph 约束

`new_forward` 是 `torch.compiler.disable` 的，且每步包含多个 host 同步点：

- `timestep.detach().flatten()[0].item()`（取 t）
- `torch.isfinite(latent).all()` 转 bool
- `_per_sample_rms_max()` 返回 float
- SP 投票的 `flag.item()`

因此：

- 与 `--diffusion-compile-granularity full` 叠加**没有收益**：根 `forward` 被 hook 接管，`run_transformer_blocks()` 直接调用 `_run_gen_stack()`，编译后的根 `forward` 被绕过；同时数据相关分支会造成 graph break/重编译。
- `regional` 只编译 block，根 hook 保持 eager，理论上可以共存，但必须与 SenCache 分开做 A/B，不能混在首轮验收里。
- 推荐首轮 `--enforce-eager`。

---

## 4. 配置参考

用户面字段来自 [data.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/data.py#L608-L622)，经 [config.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/config.py#L77-L94) 投影为 `SenCacheConfig`。

| CLI / config 字段 | 属性 | 默认 | 含义 | 调参影响 |
| --- | --- | --- | --- | --- |
| `sen_profile_path` | `profile_path` | `None` | 标定产物路径 | 必需（标定模式除外） |
| `sen_threshold_start` | `threshold_start` | `0.02` | 早段输出 RMS 变化预算 | 越小越保守 |
| `sen_threshold_main` | `threshold_main` | `0.08` | 主段预算 | 主要收益旋钮 |
| `sen_start_fraction` | `start_fraction` | `0.2` | 早段占比，切换点在 `round(N·f)` | 越大越保守 |
| `sen_max_consecutive_reuses` | `max_consecutive_reuses` | `2` | K，连续复用上限 | 越大收益高、漂移风险高 |
| `sen_time_max` | `time_max` | `1000.0` | timestep 归一化尺度 | 必须与模型 timestep 量程一致 |
| `sen_calibration_output` | `calibration_output` | `None` | 标定输出前缀（实际文件为 `<base>.npz`，全部署共享） | 设置即为标定模式 |
| `sen_calibration_time_bins` | `calibration_time_bins` | `128` | 时间箱数量（≥8） | 与步数匹配更稳 |

约束（`SenCacheConfig.__post_init__`）：阈值必须有限且非负；`start_fraction ∈ [0,1]`；K 为非负整数；`time_max > 0`；箱数 ≥ 8。

Cosmos3 的 timestep 由 FlowUniPC 生成：`timesteps = sigmas * num_train_timesteps`，`num_train_timesteps=1000`（[scheduling_flow_unipc_multistep.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/models/schedulers/scheduling_flow_unipc_multistep.py#L104-L115)），所以 `sen_time_max=1000` 正确。换模型必须核对这一点，否则所有查表都会落到错误的箱。

---

## 5. 使用指南

### 5.1 前置条件

1. 单卡 H100/H200，环境能跑 `vllm serve nvidia/Cosmos3-Nano --omni`。
2. 关闭会破坏缓存边界的开关：`--enable-hsdp`/FSDP、`--step-execution`；编译推荐 `--enforce-eager`。
3. `--cfg-parallel-size > 1` 已放开但未实测；SP/CP 已放开但未实测，首次运行建议放大进程组超时。
4. 固定一组采样参数并贯穿标定与推理：`size`、`num_frames`、`fps`、`num_inference_steps`、`flow_shift`、`guidance_scale`、seed 集合。
5. 准备 4~8 条覆盖静态/快速运动/人物/镜头变化的 prompt。
6. **请求必须显式指定步数**（`num_inference_steps`，或等价的 `timesteps`/`sigmas`）（原因见 §9.1）。

### 5.2 基线

```bash
vllm serve nvidia/Cosmos3-Nano --omni --port 8000 --init-timeout 1800
```

每条 prompt 跑 2 个固定 seed，保存 `baseline_<prompt>_<seed>.mp4`，记录预热后的 P50。

### 5.3 标定

```bash
mkdir -p /tmp/sen
vllm serve nvidia/Cosmos3-Nano --omni --port 8000 --init-timeout 1800 \
  --cache-backend sen_cache \
  --cache-config '{"sen_calibration_output": "/tmp/sen/cosmos3_nano_calib"}'
```

对每条标定 prompt 发一次请求，参数与基线完全一致，且**带 `num_inference_steps`**（否则不会 refresh、只会写一次文件）：

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=..." -F "negative_prompt=..." \
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

所有 worker 进程都指向同一个路径，写入时执行"文件锁 → 读取已有产物 → 合并 → 临时文件 + 原子替换"：

- 一次部署只产生一个 `.npz`，后续请求自动累加，无需手工 merge；
- 多节点需要共享文件系统，否则每个节点各写一份；
- 同目录会留下 `<name>.npz.lock`（跨进程互斥用），可保留；
- 某次合并失败时，内存累加器不会被清空，下一条请求会带着这些样本重试合并。

### 5.4 检查（merge 仅为兼容旧产物）

标定结束已自动合并并检查，通常只需 `inspect`：

```bash
python -m vllm_omni.diffusion.cache.sencache.calibrate inspect \
  --profile /tmp/sen/cosmos3_nano_calib.npz
```

`inspect` 做两件事：打印 `time_max`、箱数、metadata 与逐箱 `alpha_x/alpha_t`；然后跑与在线写入相同的检查，输出 `profile check: OK` 或列出 WARNING（`alpha_x` 全 0、`alpha_t` 全 0、无样本、只有极少数箱被采样、metadata `time_max` 不一致）。

只有把旧版本的多进程产物合到一起时才需要：

```bash
python -m vllm_omni.diffusion.cache.sencache.calibrate merge \
  --inputs "/tmp/sen/cosmos3_nano_calib.*.npz" \
  --out /tmp/sen/cosmos3_nano_profile.npz
```

merge 的硬约束（[calibrate.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/calibrate.py#L93-L105)）：`time_max`、箱数、`metadata.model_id`、`metadata.boundary` 必须一致；输入必须是含 `alpha_*_sum/count` 的原始产物，不能拿已 finalize 的 profile 再 merge。merge 结束同样会自动打印检查结果。

检查项：

- 曲线有限、非负；
- `alpha_t` 不恒为 0（否则探针的 timestep 没生效）；
- 两级时间（t≈1000 与 t≈0）不出现异常尖峰；
- **不要期望 `count` 覆盖大部分箱**：35 步 + 128 箱时占用率天然只有约 27%，且这些箱正好落在推理用的同一组 t 上，查表仍然精确。只有换步数时才会落到邻近箱（见 §9.4）。

### 5.5 推理

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

标定产物同时保存原始累加器和 finalize 后的曲线，可直接作为 `sen_profile_path`；若走了 §5.4 的历史产物 merge，则用 merge 输出路径。

用与基线相同的 prompt/seed/参数请求，保存 `sencache_<prompt>_<seed>.mp4`。

每次请求结束、下一次请求准入前，会打印上一条请求的统计（[backend.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/backend.py#L129-L150)）：

```text
SenCache request summary: full=.. skip=.. skip_ratio=.. (next steps=35)
```

`full`/`skip` 的粒度是**分支调用次数**，不是步数；CFG 下 `full + skip` 约为 `2 × 步数`，`skip_ratio` 才是跨分支的平均跳过比例。

### 5.6 从零跑一遍的最小闭环

1. 基线跑 1 条 prompt × 1 seed，确认出片与耗时。
2. 标定同一条 prompt，得到 `.npz`。
3. `inspect` 确认 `alpha_t != 0`。
4. 用该 profile 推理，确认 `skip_ratio > 0`。
5. 再补 3~7 条标定 prompt（数据自动累加到同一文件），重新 `inspect`。
6. 进入 §6 的扫参。

---

## 6. 调参调优

### 6.1 参数语义

- **`threshold_main`**：主旋钮。它直接决定"每步允许的输出 RMS 偏差"。由于 `S` 与输出量纲一致，最优值取决于模型输出的典型 RMS（可用基线视频的预测范数量级估计），不能跨模型复制。
- **`threshold_start`**：早段预算。扩散早期 `alpha_x` 通常更大，用更小阈值避免结构性错误（构图/运动方向）被冻结。建议始终 ≤ `main/2`。
- **`start_fraction`**：早段长度。与 `num_inference_steps` 绑定：`switch = round(N·f)`，N=35、f=0.2 → 前 7 步严格。
- **`K`**：防漂移。`K=1` 相当于"最多连续跳 1 步"，质量最稳；`K=3` 收益更高但残差误差会累积。实际扫参时先固定 K=2。
- **`time_max`**：不是调优量，是**正确性量**。填错会让 `alpha` 查错箱、`dt` 项尺度错，表现为"阈值完全不敏感"或"首步就乱跳"。

### 6.2 扫参顺序与网格

严格按"主阈值 → 早阈值 → K → 早段比例"的顺序，每次只动一个维度：

| 轮次 | 维度 | 网格 | 固定 |
| --- | --- | --- | --- |
| 1 | `threshold_main` | 0.01, 0.02, 0.05, 0.10, 0.20 | start=0.02, f=0.2, K=2 |
| 2 | `threshold_start` | `main/4`, `main/2` | 取第 1 轮最优 main |
| 3 | `K` | 1, 2, 3 | 前两轮最优 |
| 4 | `start_fraction` | 0.1, 0.2, 0.3 | 前三轮最优 |

初筛用 2 条 prompt × 2 seed；候选配置扩到 10~20 条。记录 `(skip_ratio, SSIM, PSNR, P50)` 四元组。

### 6.3 由粗到细的实操流程

```text
1) 用 profile 跑一次，读 skip_ratio
   ├─ 0.00 → 检查 alpha_t 是否为 0 / time_max / 阈值是否过小（见 §6.4）
   └─ >0   → 进入 2)
2) 把 threshold_main 乘 2，观察 skip_ratio 是否近似单调上升
   ├─ 不敏感 → profile 尺度或时间网格有问题
   └─ 敏感   → 进入 3)
3) 二分搜索：以"质量预算刚好满足"为边界，找到最大可用阈值
4) 在该阈值附近 ±25% 做细网格，配合 K 的 1/2/3
5) 用最终 2~3 组配置跑全量 prompt，选 P50 最优且质量达标者
```

### 6.4 诊断树

| 症状 | 首要怀疑 | 处理 |
| --- | --- | --- |
| `skip_ratio == 0` | 阈值太小 / `alpha` 偏大 / profile 不匹配 | 先 `inspect`；确认 `time_max`、步数、`flow_shift` 一致；再把 main 放大 2~5 倍 |
| `skip_ratio` 随阈值不单调 | 时间箱错位或 `alpha_t` 异常大 | 检查 `alpha_t` 量级；用同一步数重新标定 |
| 首步就大量跳过 | `start_fraction` 过小或 `threshold_start` 过大 | 提高 start 阈值严格度、增大 f |
| 画面结构性错误（构图/朝向） | 早段被跳过 | 降低 `threshold_start`、增大 `start_fraction` |
| 画面"糊"、细节漂移 | K 过大或 main 过大 | 降 K、降 main |
| 质量几乎无损但时延没变 | 固定开销占比高（文本编码/VAE/preprocess） | 停止扩大范围；换更小分辨率做 PoC 验证上限 |
| `alpha_t` 全 0 | 模型忽略 timestep 或探针未生效 | 检查 transformer 是否真正使用 `timestep` |
| 日志出现 `had no reusable residual` | 分支首次出现或 shape 变化 | 正常兜底；若频繁出现说明分支序列异常 |
| `does not support FSDP/HSDP` | 开了参数分片 | 关闭 `--enable-hsdp`/FSDP |
| `requires a sensitivity profile` | 没给 profile 也没开标定 | 二选一 |

### 6.5 PoC 快配置

标定约为正常推理的 3 倍开销（每步 1 真实 + 2 探针），建议先用小配置打通：

```text
640x360 / 49 帧 / 21 步 / flow_shift 固定 / guidance 6.0
sen_calibration_time_bins = 32
```

注意：**标定与推理必须使用同一组 shape/步数/scheduler/guidance**；PoC 产出的 profile 不能用于正式分辨率。

### 6.6 时间箱数量的选择（本仓库特有）

`SensitivityProfile.lookup` 用 `index = round(t/time_max * (bins−1))` 做最近箱查找（[profile.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/profile.py#L86-L92)）。

- 当 `bins` 远大于步数时：占用箱稀疏、空箱靠最近邻填充，但**同一套步数的查表仍然命中标定箱**，精度不受影响；只是 `inspect` 输出很长、直观上"覆盖率低"。
- 当 `bins` 与步数同量级（如 32 箱 / 35 步）时：占用率高、曲线更平滑，换步数时退化也更温和。
- 建议：首次验证用 32 或 64；跨步数复用 profile 时接受平滑误差，或直接重标定。

---

## 7. 验证

### 7.1 单测覆盖（已有）

`tests/diffusion/cache/test_sencache.py` 用一个小 transformer（GEN stack 把输入乘 2）覆盖了核心语义：

| 用例 | 覆盖点 |
| --- | --- |
| `test_config_validation` | 阈值/比例/K/箱数校验 |
| `test_profile_roundtrip_and_nearest_fill` | npz 往返、空箱最近邻 |
| `test_first_step_computes_then_reuses` | 首步必算、残差复用正确性 |
| `test_max_consecutive_reuses_forces_refresh` | K 强制刷新 |
| `test_branches_share_gate_but_keep_their_own_residual` | 分支共享决策、残差隔离 |
| `test_refresh_resets_state_and_counts` | 请求边界复位 |
| `test_missing_step_metadata_runs_full` | 无步号降级 |
| `test_calibration_writes_accumulator` | 探针数值与单文件产物写入 |
| `test_calibration_merges_into_one_artifact` | 两个"worker"写同一路径时累加而非覆盖 |
| `test_calibration_accumulates_across_requests` | 同一进程多请求自动累加 |
| `test_profile_issues_flags_degenerate_curves` | `alpha_x`/`alpha_t` 全 0 的语义检查 |
| `test_merge_calibration_artifacts` | 跨产物均值 |
| `test_backend_enable_and_refresh` | backend 生命周期与 `_cache_context_factory` |
| `test_backend_allows_cfg_parallel` | CFG-parallel 放行 |
| `test_calibration_is_skipped_on_non_zero_cfg_rank` | 仅 rank 0 采样 |
| `test_synchronize_decision_without_distributed` | 单机直通 |
| `test_backend_rejects_unknown_pipeline` / `test_backend_requires_profile_or_calibration` | 启动期拒绝 |

**未覆盖**的真实行为需要在运行时验证：SP 投票、CFG-parallel 端到端、取消/异常、batch>1、`num_inference_steps=None`、多请求状态隔离。

### 7.2 运行时验证矩阵

| # | 场景 | 配置 | 期望 |
| --- | --- | --- | --- |
| V1 | 基线 | 无 cache | 稳定出片，记录 P50 |
| V2 | 标定 | `sen_calibration_output` | 所有进程合并到同一个 `<base>.npz`，日志含 check 结果 |
| V3 | 标定跨请求累加 | 同进程两条请求 | 仍是同一文件，`count` 增加 |
| V4 | 推理 | profile + 默认阈值 | `skip_ratio > 0`，无报错 |
| V5 | 阈值单调性 | main ∈ {0.02,0.05,0.1,0.2} | `skip_ratio` 单调不减 |
| V6 | K 上限 | K ∈ {1,2,3} | 连续 skip 不超过 K |
| V7 | 状态隔离 | 请求 A → B → A，固定 seed | A 两次输出一致；无跨请求串扰 |
| V8 | 取消 | 推理中取消，再发新请求 | 新请求首步全算，无异常 |
| V9 | 分辨率切换 | 720p 请求后 360p 请求 | shape 不匹配时全算，不崩 |
| V10 | 无步数 | 请求不传 `num_inference_steps`/`timesteps`/`sigmas` | 出现 warning、不 refresh；作为**已知风险**记录 |
| V11 | batch>1 | 同参数两条 prompt 合批 | 不崩；整批保守同跳；质量单独评估 |
| V12 | CFG-parallel | `--cfg-parallel-size 2` | 不 hang；skip 决策与单卡行为一致 |
| V13 | SP/CP | `--ulysses-degree 2` | 不 hang；各 rank skip 一致 |
| V14 | 编译 A/B | eager vs `regional` | eager 为基准；regional 单独结论，不与 full 混 |
| V15 | step execution | `--step-execution` | 启动即报错，不静默降级 |

### 7.3 质量与时延

质量：与同 seed 基线逐帧比 PSNR/SSIM（脚本见 04 文档 §4.5）；需要感知指标时再算 LPIPS。建议同时比较音频轨（Cosmos3 joint 输出时）。

时延：同配置同 prompt，预热后重复 5 次取 P50；同时记录 `skip_ratio`，避免把"阈值改了但没生效"误读成"没有收益"。

### 7.4 验收门槛（建议）

| 检查项 | 通过条件 |
| --- | --- |
| 功能 | baseline / 标定 / 推理三条路径都稳定出片；固定 seed 可复现 |
| 正确性 | `skip_ratio > 0` 且随阈值单调；无 OOM、无 hang、无 NaN |
| 质量 | SSIM 下降在预算内（如 ≥0.95）、PSNR 下降 ≤1 dB |
| 收益 | 质量预算内 P50 有可测收益；与 cache_dit/sea_cache 同模型对比后再决定保留 |
| 状态 | V7/V8/V9 通过；无跨请求串扰 |
| 可观测 | 每次请求有 summary；降级 WARNING 为零（正常路径） |

---

## 8. 性能模型

设一次 step 的墙钟时间为 `T_step`，其中 GEN stack 占 `r = T_stack / T_step`，被跳过的步数比例为 `p`（≈ `skip_ratio` 按步计，不是按分支计），则：

```text
加速比 ≈ 1 / (1 − p · r)
```

示例（仅示意，不是实测）：

| `r` | `p` | 加速比 |
| --- | --- | --- |
| 0.8 | 0.20 | 1.19× |
| 0.8 | 0.40 | 1.47× |
| 0.8 | 0.50 | 1.67× |
| 0.6 | 0.40 | 1.32× |

两个重要修正项：

1. **判定开销**：每 step 一次全 latent 的 `rms`、一次 `isfinite().all()`、至少一次 `item()`、SP 下还有 all-reduce + `item()`。对 1280×720×189 帧的 latent 这是几十 MB 级的 elementwise，通常远小于 stack，但会打断 CPU 的 launch 流水。若发现"skip_ratio 很高但没收益"，先怀疑这一项。
2. **固定开销**：文本编码、VAE 编解码、`preprocess/postprocess` 不随跳过比例下降。若 `r` 本身就低，端到端收益会被稀释。

因此调参目标不是"最大化 skip_ratio"，而是"在质量预算内最大化 `p · r`"。

---

## 9. 风险、限制与已发现的实现问题

### 9.1 【高】请求无法解析步数时不 refresh

runner 的 refresh 流程在无法从请求解析步数（`num_inference_steps`、`timesteps`、`sigmas` 都没有）时，只对 `tea_cache` / `sea_cache` / `step_cache` 回退到 pipeline 默认值（[diffusion_model_runner.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/worker/diffusion_model_runner.py#L713-L739)），`sen_cache` 不在其中。而 Cosmos3 在请求未指定步数时会内部回退到 35/50 步，两者不一致。后果：

- `hook.refresh()` 不被调用 → `state` 不复位、`full/skip` 统计跨请求累加；
- 标定模式下 `_calibration_done` 不复位 → 同进程后续请求**不再合并新样本**到产物；
- 参考点可能沿用上一条请求的末步 latent（通常位移很大，会触发首步全算，但不保证）。

**规避**：所有请求显式传 `num_inference_steps`；或在 runner 的回退白名单中加入 `sen_cache`（需单独评审，属于行为变更）。

### 9.2 【中】标定累加器用共享 `count` 做两个分母

`ProfileAccumulator.add` 在 `alpha_x` 或 `alpha_t` 任一为 `None` 时仍只加一次 `count`（[calibrate.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/cache/sencache/calibrate.py#L78-L91)），finalize 时两个和都除以同一个 `count`。若某箱里出现 `dx_rms≈0`（只算了 t 项）或 `dt≈0`（只算了 x 项），另一项会被除以偏大的分母而**系统性偏小**。

实际影响：Cosmos3 的相邻步 `dx/dt` 几乎不可能同时为 0，命中概率低；但若做小步数/high-shift 实验，可能出现。若要修，应分别维护 `count_x` / `count_t`。

### 9.3 【中】SP 下标定成本与口径

标定探针调用 `module._omni_original_forward`，传入的是**未分片的**原始 latent/kwargs，且在每个 rank 都执行。于是：

- 成本按 world size 放大（3 倍 × 卡数）；
- profile 是在"未分片路径"上估的，而推理 gate 读的也是未分片的原始 latent，口径一致；
- 但标定期间各 rank 仍会跑 `_gen_preprocess` 与真实 stack，显存/时间都要预留。

建议：**优先单卡标定**。多卡标定时各 rank 的样本会合并进同一文件，重复样本不会改变均值（均值对同值重复不敏感），但探针开销按 rank 数放大，除浪费没有额外收益。

### 9.4 【中】profile 与步数强绑定

查表是最近箱。步数改变时，`t_k` 会落到相邻箱，等价于对曲线做轻微平滑；步数差别大时早/晚段可能错位。加上 `flow_shift` 改变会改变 timestep 分布，因此 profile 的适用范围是"同模型 + 同采样器配置"。

### 9.5 【低】判定假设 cond 先于其它分支

标定记录用 `previous["step"] == step` 丢弃同 step 的后续分支，隐含假设"每步第一个分支是 cond"。Cosmos3 三种 loop 都是 cond 先调用（[pipeline_cosmos3.py](file:///c:/Users/jayQ/PycharmProjects/vllm-omni/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L3008-L3044)），所以当前成立；换 pipeline 需复核。

### 9.6 【低】batch>1 是"整批保守"

`_per_sample_rms_max` 取 batch 内最大位移，因此整批要么全算、要么全跳。残差按整批张量保存，未做逐样本残差选择。批内样本运动差异大时，跳过的收益会被最保守样本吃掉。连续批处理（chunked prefill 式）未支持。

### 9.7 其它已知边界

- 仅 Cosmos3；其它模型需 expose `sencache_latent`/`sencache_timestep`，并加入 backend 白名单、重新标定。
- 标定只取 video；joint audio/action 的质量与同步未纳入判据。
- I2V/transfer 的探针只扰动 `hidden_states`，固定 controls；可用但未验证。
- layerwise offload 未验证：跳过 GEN stack 时被跳过层的 offload hook 不触发，收益与状态需单独测量。
- 与 CUDA/NPU graph、`full` 编译不应叠加；`regional` 需单独 A/B。
- `torch.compiler.disable` 的根 hook + 数据相关分支决定了它不可能进入编译图。

---

## 10. 排错手册

| 现象 | 原因 / 处理 |
| --- | --- |
| `sen_cache does not support pipeline ...` | 只对 `Cosmos3OmniDiffusersPipeline` / `Cosmos3OmniPipeline` 开放 |
| `requires a sensitivity profile` | 未给 `sen_profile_path` 且未开 `sen_calibration_output` |
| `does not support FSDP/HSDP parameter sharding` | 关闭 `--enable-hsdp` / FSDP |
| 标定不写文件 | 请求未带 `num_inference_steps`；请求被取消；输出目录不可写；未走到最后一步 |
| `requires denoise step metadata` | pipeline 没暴露 `current_step_index`，或 hook 被其它 wrapper 替换 |
| `alpha_t` 全 0 | 模型忽略 `timestep`，或探针 timestep 未真正传入 |
| `skip_ratio` 始终 0 | profile 的 `alpha_*` 偏大或阈值偏小；先 `inspect` 再放大阈值 |
| CFG/SP 多卡 hang | 检查各 rank latent/timestep 是否一致、SP 组是否初始化；SP 组内 `MAX` 投票，CFG 组刻意不投票 |
| 多请求结果互相影响 | 请求未显式传 `num_inference_steps` → 未 refresh |
| 开启 `--step-execution` 报错 | runner 明确不支持 cache backend + step mode |
| 与编译叠加后变慢 | 根 hook 被 disable 且绕过编译路径；改回 eager 或只做 regional A/B |

---

## 11. 回滚

删除 `--cache-backend sen_cache --cache-config ...` 即恢复原生路径。代码层面，触及既有文件的只有四处：

1. `vllm_omni/diffusion/cache/selector.py`（新增路由）
2. `vllm_omni/diffusion/data.py`（`sen_*` 字段）
3. `vllm_omni/diffusion/cache/teacache/extractors.py`（新增 `extra_states` 键）
4. `vllm_omni/entrypoints/cli/serve.py`（帮助文本）

其余为新增目录 `vllm_omni/diffusion/cache/sencache/` 与 `tests/diffusion/cache/test_sencache.py`。还原上述四处并删除新增目录即可完全移除。

---

## 12. 一页速查

```text
默认配置：
  sen_threshold_start = 0.02
  sen_threshold_main  = 0.08
  sen_start_fraction  = 0.2
  sen_max_consecutive_reuses = 2
  sen_time_max = 1000.0
  sen_calibration_time_bins = 128

启动（标定）：
  --cache-backend sen_cache --cache-config '{"sen_calibration_output": "/tmp/sen/calib"}'
启动（推理）：
  --cache-backend sen_cache --cache-config '{"sen_profile_path": "/tmp/sen/profile.npz"}'

合并：
  # 标定期已自动合并为单一文件，通常无需此步；仅用于旧版多文件产物
  python -m vllm_omni.diffusion.cache.sencache.calibrate merge --inputs "<glob>" --out <npz>
查看（并自动跑规范检查）：
  python -m vllm_omni.diffusion.cache.sencache.calibrate inspect --profile <npz>

必须：
  1) 请求显式 num_inference_steps
  2) 标定与推理同 shape/步数/scheduler/guidance
  3) 单卡标定，多卡推理；多节点标定需共享文件系统
  4) eager 或 regional，不要 full compile
  5) 不看 skip_ratio 单项，看 (skip_ratio, SSIM, PSNR, P50) 四元组
```
