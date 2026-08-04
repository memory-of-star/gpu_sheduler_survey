# CTA 级 PDL：B200 受控实验报告（GPU 预算 < 1 小时）

日期：2026-08-03（UTC）
设备：单卡 NVIDIA B200，148 SM，CC 10.0，183,359 MiB
代码基线：Git `c2056b68f0aaab247054833e1b39e84fb99436ec` + 本报告列出的工作树修正
结论等级：**机制可行性证据；不是生产负载端到端收益结论**

## 1. 执行摘要与决策

本轮先审计了原实验，再在用户限定的 1 GPU·小时内选择最高信息量的实验。原 `cta_dep_bench` 的 Tier 1/2 数据因 trigger 时机和协议问题不能用于判断 CTA 级收益；因此保留原始结果作为失败证据，另建了语义修正的受控 pilot。

最终判断：**GO，通过一波、低资源合成负载的机制可行性 gate；但只应进入一个小规模 applicability gate，不能直接宣称真实 LLM/DSA 有 35% 收益。**

最重要的数字如下：

- 更保守的 `tail=0` 点：软件 CTA interval-backoff 相对 grid Floor 从 `1.001568 ms` 降到 `0.901792 ms`，即 **1.1105× / 9.95% latency gain**；只兑现 no-wait 空间的 **26.15%**。这个数字最接近“没有大段 producer 独立尾部可隐藏”的场景，适合作为 headline。
- tail-rich 默认点：interval degree 1–64 的 Floor 约 `1.409 ms`，软件实现为 `0.901–0.916 ms`，相对 Floor 降低 **34.99–36.08%**，兑现 **96.33–99.57%** 的 no-wait gap。这是构造性上限行为，不是产品收益预测。
- degree 从 1 增到 64 没有让理论空间在 32 附近消失，但软件等待/解码成本随 degree 增长；Impl 与 Ceiling 的差距从约 `2.3 µs` 增到 `18.0 µs`。
- strided degree 32 的 interval cover 紧度仅 `0.2264`，interval Impl 比 exact Impl 慢约 `20.2 µs`；编码假边将兑现比例从约 `96.50%` 降到 `92.47%`。结构形状与依赖度必须分开建模。
- B200 上 eager 跨 stream programmatic event 与普通 event 都是约 `1.00×`；same-stream、直接 CUDA Graph programmatic edge、以及把同样双流代码 capture 成 graph 均约 `2.00×`。该设备/driver 上可用路径是 same-stream 或 graph，不是 eager 双流。
- 离线 oracle 显示真实候选依赖常是“高 degree、低结构复杂度”：Qwen FFN degree 128、DSA 32K degree 256、DSA 1M degree 8192 都是精确连续区间，假边率 0。不能把 `degree > 32` 单独作为淘汰规则。
- 24 个 pilot 配置、5 种模式、每模式 31 次，共 **3,720 个 timed samples**；所有 96 个应校验的软件/Floor 模式均 PASS，24 个 no-wait 模式按设计不做正确性背书。
- 修正后 GPU 程序的保守进程墙钟合计约 **0.0195 GPU·小时**。即便从 preflight 开始，把废弃轮次、CPU 阶段、报告期间空档一直算到最终 smoke，连续会话仍 **< 0.590 GPU·小时**，满足 `<1 GPU·小时` 约束。

## 2. 范围、预算与未执行项

原方案是约 8 GPU·小时的完整 campaign。本轮按预算缩减为：

| 项目 | 本轮状态 | 目的 |
|---|---|---|
| 环境/preflight 与代码审计 | 完成 | 先排除无效数据和环境阻塞 |
| Tier 0 基础事实 | 修正后完整重跑 | PDL 链、容量、fence、CLC |
| Tier 0.2 跨 stream/graph | 完整重跑 | 验证 B200 路径差异 |
| 原 Tier 1/2 FAST | 跑过但判定无效 | 只作为 harness 失败证据，不纳入结论 |
| 校正 CTA pilot | 24 配置 × 5 模式 × 31 次 | 高价值机制筛选 |
| Qwen/DeepSeek/GLM 依赖 oracle | 7 组，CPU-only | 检查真实依赖形状 |
| Tier 4 Qwen3.6-27B | 未跑 | 模型不在本机；runner 与 PDL 档位验证也不可靠 |
| Tier 5 DSA GPU 算子链 | 未跑 | 当前实现的长序列 `S×32×S` 张量在 131K/1M 必然 OOM，Floor/Ceiling 也未真正移除 GPU 依赖 |
| nsys/ncu profiling | 未跑 | 工具未安装 |

GPU 预算核算：

| GPU 工作 | 进程墙钟包络 |
|---|---:|
| corrected Tier 0 | 3.620 s |
| 跨 stream/graph 全套 | 38.611 s |
| 24 点 corrected pilot | 27.575 s |
| 最终 64 KiB launch smoke | 0.464 s |
| **合计** | **70.270 s = 0.0195 GPU·h** |

Pilot 内 CUDA event 计时总和只有 `3.917 s`；上表还包含进程启动、CPU 控制和日志开销。极端保守地从 preflight 编译开始算到最终 GPU smoke，把失败轮次、CPU 分析和全部空档都当成整卡占用，为 `<0.590 GPU·h`。实验前已按用户授权停止原有 vLLM GPU 服务；报告收尾时显存使用为 `0 MiB`、无 compute process。

## 3. 可复现实验环境

| 组件 | 版本/状态 |
|---|---|
| GPU | NVIDIA B200，148 SM，CC 10.0，183,359 MiB |
| Driver | 580.126.09 |
| CUDA / nvcc | CUDA 13.0 / nvcc 13.0.88 |
| 编译目标 | `sm_100` |
| CUDA flags | `-O3 -std=c++17 -arch=sm_100 -lineinfo -I. --expt-relaxed-constexpr` |
| OS / kernel | Ubuntu 22.04.5 / Linux 6.8.0-1046-aws |
| GCC | 11.4.0 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Triton / vLLM | 3.6.0 / 0.23.0 |
| Transformers / NumPy | 5.12.0 / 2.2.6 |
| 缺失工具 | `nsys`、`ncu`、Compute Sanitizer、matplotlib |
| 模型 | Qwen3.6-27B 不在本机；只有 Qwen3.5-9B 的部分/本地内容 |

本轮工作树不是纯 Git HEAD；复现必须同时保存新增/修改的源码和结果快照，不能只记录 commit。

## 4. 为什么原 Tier 1/2 结果被拒绝

原始 FAST 数据保存在 [`bench/results_budget1h/`](../bench/results_budget1h/)，但不应进入性能结论。它显示 Tier 1 大多在 `-1%..+0.3%` 附近，这不是“CTA PDL 没收益”，而是实验退化造成的协议噪声。

阻塞性问题包括：

1. 原 producer 先写数据、发布全部 `done[]`，之后才发 PDL trigger。dependent grid 只有在所有 producer CTA 都 trigger 后才有资格启动，因此 consumer 开始时所有 CTA flag 已 ready；CTA wait 退化为立即返回。
2. `WAIT_COUNTER` 用“全局完成数量 ≥ hi+1”推断前缀 `[0,hi]` 已完成，但 CTA 可乱序退休；高编号 CTA 能错误地满足计数。
3. correctness 失败只打印 `FAIL`，进程仍返回 0；driver 会写 `.done`，分析器甚至可能把错误模式选成“最快实现”。
4. timed consumer epilogue 串行遍历所有 parent，普通计算量是 O(degree)，把 degree 与同步成本混在一起。
5. random parent 可重复采样；grid wait 与软件 wait 的线程参与方式不对称。
6. 首轮还暴露两个独立运行错误：zero-smem kernel 仍访问 `g_smem[0]` 导致 illegal access；64 KiB dynamic smem 未显式 opt-in，导致 invalid launch。

此外，现有 LLM runner 使用了不匹配 vLLM 0.23 的 `prompt_ids=` API，FULL CUDA Graph 开关和 ceiling patch 都没有可靠验证；DSA runner 的“去 host synchronize”并未解除同一 CUDA stream 上的 GPU 顺序依赖。因此本轮没有为了凑齐表格而运行这些无效档位。

本轮修正：

- [`bench/tier0_facts.cu`](../bench/tier0_facts.cu)：zero-smem 不再访问动态 shared memory；64 KiB 显式 opt-in。
- [`bench/cta_dep_bench.cu`](../bench/cta_dep_bench.cu)：为 64 KiB producer/consumer occupancy query 与 launch 增加 opt-in。原 benchmark 的 trigger 语义未被当作已修复；其性能数据仍弃用。
- [`bench/common/dep_wait.cuh`](../bench/common/dep_wait.cuh)：修正帮助文本已公开、但 parser 原先不接受的 `--wait grid` / `--wait none` 短别名。
- 新增 [`bench/cta_dep_pilot.cu`](../bench/cta_dep_pilot.cu)：独立、受限、可验证的 corrected pilot。
- 新增 [`tools/analyze_pilot.py`](../tools/analyze_pilot.py)：逐样本中位数、seed 汇总和 deterministic bootstrap。
- 修正 [`tools/cta_timeline.py`](../tools/cta_timeline.py)：并发度改为半开区间的精确端点扫描；`t_dep-t_launch` 正确标为“prologue + 可能的 wait”，不再冒充纯 dependency stall。

## 5. 校正 pilot 的实验语义

### 5.1 四点包夹与公式

原文定义 Floor / Impl / Ceiling / Ideal。对于“时间越小越好”的指标，本报告使用正向百分比：

```text
理论收益空间 H = (Floor - Ceiling) / Floor
软件实际收益 G = (Floor - Impl)    / Floor
空间兑现比例 C = (Floor - Impl)    / (Floor - Ceiling)
speedup          = Floor / Impl
```

`Ceiling` 是不等待的 unsafe timing reference，不做正确性承诺，也不是可部署实现；`Ideal` 需要可信的原语开销注入，本轮未给出，避免伪精确。

### 5.2 三条关键执行路径

| 点 | Producer | Consumer | 含义 |
|---|---|---|---|
| Floor (`grid`) | readiness work → 写数据 → PDL trigger → 独立 tail | 可启动后做 prologue → `cudaGridDependencySynchronize()` → epilogue | 标准 grid 级 PDL |
| Impl (`interval-backoff`) | **入口 trigger** → readiness work → 写数据 → release flag → 独立 tail | prologue → 按 interval acquire/backoff wait → epilogue | 软件 CTA readiness |
| Ceiling (`none`) | **入口 trigger** → readiness work → 写数据 → tail | prologue → 不等待 → epilogue | unsafe no-wait 时间参照 |

预先声明 `interval-backoff` 为 Impl，未在结果出来后挑最快模式。另测 `interval-spin` 和 `exact-backoff` 只用于协议/编码诊断；不安全 counter 被移除。

### 5.3 控制变量和正确性

- P/C 均不超过 148 CTA，保证一波、一 CTA/SM 的 launch-gate 条件；producer occupancy 为 16 CTA/SM、consumer 为 10 CTA/SM，允许 co-residency。
- 默认 `ready=400K cycles`、`tail=1M`、`prologue=200K`、`epilogue=1M`，readiness 有 8 个 skew bins。
- timed dependent payload 是 O(1)，不会随 degree 增长；所有真实 parent 的检查在独立 untimed validation invocation 中完成。
- 每次 invocation 都 poison data/out、清零 flags 和 error，避免上一轮正确值掩盖错误。
- 软件 flag 使用 device-scope release/acquire；leader-only wait 后统一 `__syncthreads()`。
- 每配置每模式 3 次 warmup + 31 次 timed repeat；seed 为 101/202/303。
- 每配置取 31 次中位数，再对 3 个 seed 的配置中位数取中位数，并报告 seed 范围。10,000 次 nonparametric bootstrap 95% CI 只描述本 session 的 timer repeat noise，不代表跨设备、跨 driver 或跨 workload 的外推显著性。

## 6. Tier 0 结果

### 6.1 同 stream PDL 链

| stages | PDL off (ms) | PDL on (ms) | speedup | 模型反解 `implied_depth` |
|---:|---:|---:|---:|---:|
| 1 | 1.02502 | 1.02499 | 1.0000× | 1.000 |
| 2 | 2.04701 | 1.53702 | 1.3318× | 1.993 |
| 3 | 3.06893 | 2.04883 | 1.4979× | 2.983 |
| 4 | 4.09168 | 2.55904 | 1.5989× | 3.986 |
| 5 | 5.11261 | 3.06995 | 1.6654× | 4.977 |
| 6 | 6.13386 | 3.58077 | 1.7130× | 5.969 |

正确解释是：六级链上的**相邻级** tail/prologue overlap 能持续传递，总时延符合流水模型；`implied_depth=5.969` 不是“六个 grid 同时并发”。修正后的精确 endpoint sweep 显示峰值为 296 CTA，即最多 2 个 148-CTA grid 同时驻留；chain6 中相邻首对 overlap 为 `0.509 ms`，约占 producer span 的 `49.9%`。同 ID producer/consumer 在该一波构造中 148/148 落在同一 SM，但不能外推到多波 grid。

### 6.2 跨 stream、CUDA Graph 与 diamond

平衡点 `tail=prologue=20M cycles`：

| 模式 | median (ms) | vs BASE | correctness |
|---|---:|---:|---|
| ordinary event, cross-stream BASE | 20.383 | 1.00× | PASS |
| eager programmatic event, cross-stream | 20.384 | 1.00× | PASS |
| 同样双流代码 capture → graph | 10.193 | 2.00× | PASS |
| 直接构造 programmatic graph edge | 10.193 | 2.00× | PASS |
| same-stream PDL | 10.196 | 2.00× | PASS |
| no-dependency concurrent reference | 10.197 | 2.00× | n/a |

从 1M 到 80M cycles 的 7 点 sweep，graph speedup 均为约 `2.00×`，eager cross-stream 始终约 `1.00×`。平衡 diamond 中 ordinary edges 为 `40.754 ms`，programmatic edges 为 `20.386 ms`，也是 `2.00×`。这说明该 B200/driver/toolkit 上 graph/same-stream 路径能实现相邻阶段流水；PDL 是 opportunistic，不能写成所有版本上的永久保证。Diamond 也是平衡 synthetic DAG，不能代表任意非对称或资源重型 DAG。

### 6.3 Occupancy 容量曲线

| threads | smem/CTA | CTA/SM | 全卡一波容量 | measured one-wave time (ms) |
|---:|---:|---:|---:|---:|
| 128 | 0 KiB | 16 | 2368 | 0.54886 |
| 128 | 8 KiB | 16 | 2368 | 0.54928 |
| 128 | 16 KiB | 13 | 1924 | 0.55002 |
| 128 | 32 KiB | 6 | 888 | 0.51754 |
| 128 | 64 KiB | 3 | 444 | 0.51722 |
| 256 | 0 KiB | 8 | 1184 | 0.54906 |
| 256 | 8 KiB | 8 | 1184 | 0.54874 |
| 256 | 16 KiB | 8 | 1184 | 0.54880 |
| 256 | 32 KiB | 6 | 888 | 0.54893 |
| 256 | 64 KiB | 3 | 444 | 0.51699 |

可靠产出是容量曲线。这里始终只发恰好一波等待 CTA、没有 productive background kernel，`0.52–0.55 ms` 不能被解释为“等待 CTA 对系统吞吐的代价”；真正的 B2 定价仍需低 occupancy + background 的对照实验。

### 6.4 Fence 与 CLC

| fence scope | median kernel time (ms) | reported ns/fence |
|---|---:|---:|
| none | 0.01869 | 0.000 |
| CTA | 0.04957 | 15.440 |
| GPU | 1.42877 | 705.040 |
| SYS | 8.90672 | 4444.016 |

这些数来自 148 blocks × 128 threads 的饱和循环，是摊销后的 critical-path/throughput 指标，不是可直接注入 Ideal 模型的“孤立单 fence latency”。

CLC：4096 次 attempt 中 1728 个 pending CTA 被唯一取消，duplicate claim 为 0；剩余 2368 恰好等于 `148 SM × 16 resident CTA/SM`，是已驻留并正常执行的 CTA，不是漏任务。`success_rate=42.19%` 反映 pending/resident 比例，不是独立的仲裁成功概率；聚合报告为 `1650.83 cycles/attempt`。

## 7. Corrected pilot 结果

下表是“每配置 31 次中位数，再跨 3 seed 取中位数”。Impl 固定为 interval-backoff。

| family | Floor ms | Ceiling ms | Impl ms | Floor/Impl | Space % | Impl gain % | gap captured % |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval d1 | 1.409088 | 0.898368 | 0.900768 | 1.5645× | 36.243 | 36.082 | 99.555 |
| interval d8 | 1.409216 | 0.898368 | 0.902464 | 1.5616× | 36.250 | 35.961 | 99.204 |
| interval d32 | 1.409152 | 0.898368 | 0.908128 | 1.5517× | 36.251 | 35.555 | 98.040 |
| interval d64 | 1.409120 | 0.898016 | 0.916064 | 1.5382× | 36.277 | 34.989 | 96.469 |
| strided d32 | 1.409056 | 0.898176 | 0.936640 | 1.5043× | 36.253 | 33.523 | 92.470 |
| tail=0, d8 | 1.001568 | 0.619872 | 0.901792 | 1.1105× | 38.086 | 9.952 | 26.150 |
| tail=2M, d8 | 1.918144 | 1.407936 | 1.408192 | 1.3620× | 26.586 | 26.578 | 99.994 |
| grid=64, d8 | 1.408544 | 0.897472 | 0.901952 | 1.5617× | 36.284 | 35.966 | 99.123 |

三个 seed 的 `gap captured` 范围依次为：d1 `[99.442,99.574]`、d8 `[98.992,99.267]`、d32 `[97.964,98.244]`、d64 `[96.326,96.501]`、strided `[92.433,92.713]`、tail0 `[26.066,26.205]`、tail2M `[99.850,100.000]`、grid64 `[99.092,99.199]`。方向性结论在三个 seed 上一致。

代表点 `interval d8, seed=202` 的 10,000 次 repeat-level bootstrap 95% CI：Floor `[1.408576,1.409248] ms`、Ceiling `[0.898016,0.898560] ms`、Impl `[0.901888,0.902496] ms`、Space `[36.212,36.268]%`、Impl gain `[35.937,35.993]%`、gap captured `[99.167,99.342]%`。这个极窄区间只说明单 session repeat noise 很小。

### 7.1 Degree

从 d1 到 d64，Floor/Ceiling 与约 36.25% 理论空间基本不变，不支持“degree > 32 后收益空间突然归零”。按 seed 配对后，interval-backoff 相对 Ceiling 的额外时间中位数约为 d1 `2.27 µs`、d8 `4.06 µs`、d32 `10.02 µs`、d64 `18.05 µs`，呈渐增趋势。当前只测到 64，不能外推到 256/8192。

### 7.2 结构/编码

interval d32 与 strided d32 的理论空间都约 36.25%，但 interval 编码在 strided 上把真实 degree 32 扩为平均 141.76 个 interval entries，tightness=`0.2264`。三个 seed 的 primary interval-backoff 中位数约 `0.936640 ms`，exact-backoff 约 `0.916192 ms`，精确编码恢复约 `20.2 µs`，将 gap capture 从 `92.47%` 提高到约 `96.50%`。这个单点同时包含 flag scan、地址计算和等待晚 parent 的成本，只能证明“假边成本可测”，不能直接拟合硬件开销。

### 7.3 Tail/prologue 几何

tail=0 的绝对 Floor-Ceiling gap 约 `381.7 µs`，但软件只恢复约 `99.8 µs`；默认 tail=1M 的绝对 gap 约 `510.8 µs`，软件恢复约 `506.8 µs`；tail=2M 的绝对 gap 仍约 `510.2 µs`，相对百分比变成 26.6% 只是总时长分母变大。结论是：**理论空间和可兑现空间必须同时报告**；足够长的 producer 独立 tail 让 CTA 协议接近 Ceiling，无 tail 时占位/轮询效应吃掉大部分空间。

### 7.4 Grid 与重复性

grid 64 与 148 的各延迟差异约 1 µs 以内，只能说 one-wave/underfilled 范围内未观察到 grid-size effect，不能外推到 2/4 wave 或数千 CTA。

日志中 19/3,720 个 sample 比所属配置/mode 中位数高超过 0.5%，没有对应的低尾；最大偏差约 1.54%。31 次中位数对这些孤立正向抖动不敏感，没有删除样本或改用均值。

## 8. 真实候选负载的离线依赖 oracle

Oracle 从 tile decomposition 解析依赖，不使用 GPU。大 grid 最多抽样 256 个 consumer，因此以下是结构性检查，不是运行时 profile。

| 模型/链 | shape | mean degree | interval tightness | false edges | interval/exact storage |
|---|---|---:|---:|---:|---:|
| Qwen DeltaNet intra→inter | seq 4K | 1.984 | 1.0 | 0% | 1.008× |
| Qwen DeltaNet intra→inter | seq 32K/128K | 1.996 | 1.0 | 0% | 1.002× |
| Qwen RMSNorm→gate/up GEMM | 4K/32K/128K | 128 | 1.0 | 0% | 0.015625× |
| DeepSeek/GLM DSA indexer→topk | 32K | 256 | 1.0 | 0% | 0.0078125× |
| DeepSeek/GLM DSA indexer→topk | 1M | 8192 | 1.0 | 0% | 0.000244× |
| DSA topk→sparse attention | 32K/1M | 1 | 1.0 | 0% | 2× |
| GLM IndexShare topk→attn L0–L3 | span 1–4 | 1 | 1.0 | 0% | 2× |

最强结论不是这些链一定有性能收益，而是 **degree 与结构复杂度正交**：degree 8192 仍可由 O(1) 精确区间表达。下一步选择机制时，应以 tightness/假边、tail 几何、wave 数和 occupancy 联合判断，而非 degree 单阈值。

## 9. 能说明什么，不能说明什么

本轮能说明：

- 在 B200 上，一波、低资源、规则区间依赖、存在足够独立 tail 时，device-scope release/acquire 的软件 CTA wait 能正确运行并接近 unsafe no-wait timing gap。
- d64 未出现 d32 附近的 cliff；结构松散会带来独立于 degree 的编码成本。
- same-stream/CUDA Graph programmatic edge 在本环境有效，eager cross-stream event 没有观测到 early overlap。
- 真实候选依赖中确实存在“高 degree 但精确区间”的模式。
- B300/sm_103 上的定量边界；本轮设备是 B200/sm_100，只能回答本机机制与路径问题。

本轮不能说明：

- 真实 GEMM/attention/DSA 的端到端收益，或 Qwen3.6-27B 的 TPS/latency 收益。
- multi-wave grid、occupancy 1–2、真实 register/smem、带宽/L2 竞争、degree >64 下的效果。
- 软件 polling 对其他 kernel 的 L2 干扰；本机没有 ncu。
- Floor→Impl 的全部差异都来自“CTA 粒度”。该对比同时改变 trigger 时机和 bitmap/wait 协议。
- Ceiling 是可实现硬件上界或永远最快。它是不正确的 no-wait reference。
- pilot 的 spin epilogue 等价于真实 dependent compute；它只是固定时长 placeholder。
- pilot 已直接 trace 到 `consumer launch < producer ready`。当前 timing 形状强烈支持 overlap，但尚未记录逐 CTA ready/return 时间戳。
- 跨流程序的 correctness 证据与 pilot 同等级。跨流程序只用固定输入末尾验证、buffer 未 epoch 化，性能路径结论强，正确性防陈旧值能力较弱。

## 10. Gate 判定与下一步

RUNBOOK 的门槛是：多数配置收益空间 ≥8% 时继续 Tier 2/3 + LLM + DSA；2–8% 只做 LLM；<2% 停止。当前 synthetic pilot 的 Space 都 >26%，tail=0 的正确 Impl gain 仍约 9.95%，所以**机制可行性 gate 通过**。但当前矩阵不是原计划的 multi-wave/resource-realistic Tier 1 全空间，不能据此直接“跑满预算”。

建议只做三个高信息量实验：

1. 给 pilot 加一次 CTA trace，直接证明 `consumer launch < producer ready`，分别记录 wait-return 和 ready 对齐。
2. 测 2/4-wave grid，并用 shared memory/register 将 co-residency 压到 2 和 1 CTA/SM；至少覆盖 tail=0 和一组 realistic tail/prologue ratio。
3. 选一个真实 tile kernel chain，只走已验证的 same-stream 或 CUDA Graph 路径；若目标框架只能 eager cross-stream，本环境下应先判 NO-GO。

建议 applicability gate：resource-realistic、至少 2-wave 的正确 Impl 对 grid Floor 中位 gain 仍 ≥5%（若严格沿项目原门槛则 ≥8%），且 trace 直接确认 overlap，才进入 LLM/DSA 集成；若 multi-wave/occupancy≤2 后多数点 <2%，或任何 correctness 失败，则停止软件 CTA 协议方向。

## 11. 数据与复现入口

- corrected 原始结果：[`bench/results_budget1h_corrected/`](../bench/results_budget1h_corrected/)
- Pilot 原始 3,720 samples：[`pilot_matrix.log`](../bench/results_budget1h_corrected/pilot_matrix.log)
- 逐配置统计和 CI：[`pilot_summary.csv`](../bench/results_budget1h_corrected/pilot_summary.csv)、[`pilot_analysis.json`](../bench/results_budget1h_corrected/pilot_analysis.json)
- Tier 0：[`tier0_facts.log`](../bench/results_budget1h_corrected/tier0_facts.log)、[`tier0_clc.log`](../bench/results_budget1h_corrected/tier0_clc.log)
- 跨流/graph：[`tier0_xstream.log`](../bench/results_budget1h_corrected/tier0_xstream.log)
- CTA trace：[`tier0_chain_trace.csv`](../bench/results_budget1h_corrected/tier0_chain_trace.csv)、[`tier0_chain_timeline.json`](../bench/results_budget1h_corrected/tier0_chain_timeline.json)
- 最终 64 KiB launch smoke（occupancy=3 CTA/SM、correctness PASS）：[`smem64_launch_smoke.log`](../bench/results_budget1h_corrected/smem64_launch_smoke.log)
- Oracle：同目录下 `oracle_*.json` / `oracle_*.log`
- 被拒绝的原始 smoke：[`bench/results_budget1h/`](../bench/results_budget1h/)

- SHA-256 数据/源码清单：[`EXPERIMENT_MANIFEST_SHA256.txt`](../EXPERIMENT_MANIFEST_SHA256.txt)
- 可搬运归档：[`cta_pdl_b200_budget1h_20260803.tar.gz`](../cta_pdl_b200_budget1h_20260803.tar.gz)（旁有 `.sha256`）
核心复现命令：

```bash
cd /workspace/gpu_sheduler_survey/cta_level_PDL_design/bench
ARCH=sm_100 ./build.sh
RESULTS=results_budget1h_corrected ./run_all.sh tier0

# 代表性 corrected pilot
./cta_dep_pilot --producers 148 --consumers 148 \
  --structure interval --degree 8 --repeats 31 \
  --ready 400000 --tail 1000000 --prologue 200000 --epilogue 1000000 \
  --skew-bins 8 --seed 202 --tag interval_d8_s202

cd ..
python3 tools/analyze_pilot.py \
  bench/results_budget1h_corrected/pilot_matrix.log \
  --json bench/results_budget1h_corrected/pilot_analysis.json \
  --csv bench/results_budget1h_corrected/pilot_summary.csv \
  --iterations 10000
```

完整报告应与当前源码、原始日志和 SHA-256 manifest 一起归档；只保留汇总表不足以复核 trigger 语义、correctness 或统计处理。
