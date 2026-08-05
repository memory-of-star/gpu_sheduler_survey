# CTA 级 PDL：B200 多波正式 campaign 总报告

## 1. 报告头与证据等级

报告日期：2026-08-05（UTC）\
实验日期：2026-08-05（UTC）\
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0，183359 MiB\
软件环境：Driver 580.126.09，CUDA 13.0，nvcc 13.0.88，正式编译目标 `sm_100`\
正式结果目录：Tier 1 / Tier 0.3 使用 [`bench/results_20260805_b200_multiwave_v2/`](../bench/results_20260805_b200_multiwave_v2/)；Tier 0.1 使用 [`bench/results_20260805_b200_multiwave_v4/`](../bench/results_20260805_b200_multiwave_v4/)；Tier 2/3 使用 [`bench/results_20260805_b200_tier23_native_v2/`](../bench/results_20260805_b200_tier23_native_v2/)；Tier 4 使用 [`results/tier4_schema_v3_formal_v1_20260805/`](../results/tier4_schema_v3_formal_v1_20260805/)；Tier 5 native 使用 [`bench/dsa/results_20260805_b200_native_formal_strict_v9/`](../bench/dsa/results_20260805_b200_native_formal_strict_v9/)；Tier 5 production compact-14 scoped formal 使用 [`bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/`](../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/)
证据等级：**B200 合成微基准正式测量 + Tier 2/3 native 语义准入 + Qwen3.6-27B target-specific 端到端包夹 + Tier 5 native-v9 四上下文正式准入 + production compact-14 scoped formal 准入**。本报告可支持 Tier 1 合成 headroom 地图、Tier 0.1/0.3 基础事实、Tier 2/3 所列机制横评、指定模型和 decode 配置的 Tier 4 零等待时间上界、native-v9 synthetic work-complete proxy，以及 compact-14 明示范围内的 workload-component timing。production exact-26 的 v1/v2 尝试均已 `REJECTED`，canonical exact-26 保持 `INCOMPLETE`；compact PASS 不支持 32K/1M timing、CTA/LLM production headroom、B300 定量外推或最终硬件设计坐标。

v2 正式 session 从 `05:01:52` 到 `05:07:04`，墙钟 `312 s = 5 min 12 s`。该 session 当时的 preflight 为 **20 ok / 2 warn / 0 fail**，两个 warning 是 `nsys`、`ncu` 尚未安装；这只描述 v2 历史快照。工具随后已补齐：Tier 2/3 和 Tier 4 均取得 Nsight Systems 证据，Nsight Compute 则因 `ERR_NVGPUCTRPERM` 无法读取物理 counter。为用更严格的 artifact-binding schema 固定 Tier 0.1 证据，v4 又从 `06:09:50` 运行到 `06:14:51`，墙钟 `301 s = 5 min 1 s`；`semantics=3` strict chain validator 返回 `PASS`。

## 2. 执行摘要与 campaign 状态

机器 gate 的权威判决为 **`GO`**。Tier 1.1 的 88 个配置全部满足 manifest、31 次重复、唯一 parent、语义证明和计划覆盖要求；`2×/8×/32×SM` 均由真实时间戳证明为多波。gate 使用的典型实测操作性 `Ceiling − Floor` gap 为 **36.1062%**，范围 **1.5764%–39.1731%**；软件主实现相对 Floor 的全矩阵中位数为 **34.9042%**。正式 Tier 1 共 93 个配置、14415 条 `SAMPLE`，`0` 次 trace retry、`0` 个无效配置。

`GO` 只回答“合成矩阵中典型 `Ceiling − Floor` operational gap 是否超过 8%”。它不等于软件实现获益，更不等于真实负载获益。这个区别在大 grid 上是决定性的：`32×SM` 的按配置中位 gap 仍为 **4.5320%**，但 interval-backoff 相对 Floor 的中位变化为 **−5.6928%**；`8×SM` 对应为 **10.3165%** 与 **−23.4101%**。也就是说，当前完整 software operational path 在两个大多波档都没有兑现收益；本轮未将 priority-stream topology、polling/cover 与 64 KiB resident envelope 的各自成本分离。

Tier 0.3 正式完成 15 个资源点，每点 31 组成对重复，即 465 对、930 条路径样本。resident-wait 相对同样工作量但延后派发的 waiter，使 productive-background 吞吐损失的15 点中位数取值范围为 **16.7280%–53.0463%**，配对端到端增量范围为 **1.817696–10.222560 ms**。这些是跨资源点的 point-estimate envelope，不是 bootstrap CI；它们只定价当前合成构造的 B2 代价，详见 [`reports/tier0_base_facts/0_3_resident_wait_throughput.md`](tier0_base_facts/0_3_resident_wait_throughput.md)。

Tier 0.1 在 v4 中固定为 **DONE**：6 个 chain-length 配置各保留 31 组相邻交替顺序的 off/on pair，合计 186 pairs、372 samples、12 次独立全边 validation 和 1,776 条 trace row。在 6-stage 点，PDL off/on 中位数为 **6.134208 / 3.576160 ms**，配对 speedup 为 **1.715359×**；trace 实测峰值是 **2 个 grid / 296 CTA** 同时活跃，不是 6 个 grid。

Tier 2/3 后续以全新的 native harness 正式完成，而不是复用被拒绝的 `cta_dep_bench`：strict validator 对 **35 个配置、5,084 个 timed sample、182,460 条 trace row** 报 `PASS`，安全档全边校验和 unsafe Ceiling 错误性证明均闭环。它支持结构敏感的编码选择、diamond 伪顺序边代价、C1 数据传递与 CLC 软件调度横评；不能提供因权限缺失而未取得的物理 L2/DRAM counter。

Tier 4 schema-v3 也已正式完成。Qwen3.6-27B 的 `headline_full_decode` 四点在 grid PDL 打开后，`Ceiling − PDL_grid` 零等待时间上界依 batch 1/4/16/64 分别为 **1.333416% / 1.309376% / 1.830129% / 8.060855%**；对应 grid PDL 相对 off 为 **1.956148% / 1.982322% / 1.619815% / 1.068397%**。Ceiling 在每个点 31/31 次均输出错误并产生 non-finite logprob，所以这些数字是时间上界，不是可实现 CTA-level 收益。

后续阶段状态如下；`GO` 只提供执行许可，后续 `DONE` 均来自各自独立 admission：

| 计划项 | 本轮状态 | 可发布结论 |
|---|---|---|
| §5.1 Tier 1.1 degree × structure | **DONE** | 完整正式矩阵；gate=`GO` |
| §4.1 Tier 0.1 overlap depth | **DONE** | v4 `semantics=3`：6 配置、186 pairs、372 samples、12 validations、1,776 trace rows；validator=`PASS` |
| §4.3 Tier 0.3 occupancy / background | **DONE** | 15 点配对曲线；只解释合成 resident-wait 代价 |
| §7 Tier 2/3 | **DONE** | native-v2 35/35 配置、5,084 samples、182,460 trace rows、0 errors；§7.2 仍只是 Tier 0 条件化 scenario envelope |
| §8.2 Tier 4 `Ceiling − PDL_grid` | **DONE** | schema-v3 decode/prefill admission 均通过；headline 只取 4 个 full-decode 点 |
| §9 Tier 5 DSA | **PARTIAL** | v6/v7/v8 永久拒收；native-v9 四点 formal `PASS`；production compact-14 scoped formal `PASS/DONE`、`accepted_compact_workload_timing=1`、14/1,302/62；exact-26 v1/v2 均 `REJECTED`、canonical 范围 `INCOMPLETE`，CTA/LLM production headroom 未定义 |

因此，计划的前四项最小交付、Tier 5 native-v9，以及 production compact-14 scoped formal 都已有各自独立的合格证据，但 §9 在 canonical production exact-26 完成前仍为 **PARTIAL**。compact-14 的 PASS 只准入它自己的窄 workload-timing 范围，不替代 exact-26 或 CTA bracket。Tier 4 只完成了“时间上界”这一规格，并没有实现可正确执行的 CTA-level 档。建议方向仍只能是 **provisional**：规整连续集合优先保留 interval，稀疏集合必须考虑 exact representation，且 resident polling 不能直接当作候选终点；在 production exact-26 准入和可正确执行的 CTA/LLM Impl 证据闭合之前，不选择最终 `[H-]`/`[H+]` 硬件坐标。

## 3. 程序实际执行的语义

### 3.1 Tier 1 多波三档包夹 + 两个协议诊断

[`bench/cta_dep_pilot.cu`](../bench/cta_dep_pilot.cu) 在同一进程中运行五档，headline 三档相邻：

| 档位 | 实际 launch / wait 路径 | 语义 |
|---|---|---|
| Floor (`grid`) | programmatic CUDA Graph edge；producer 在真实 ready 后 trigger，consumer 执行 `cudaGridDependencySynchronize()` | 当前 grid-PDL 基线 |
| Impl (`interval-backoff`) | 高优先级 producer stream + 先排队的低优先级 consumer stream；producer 在入口 trigger，ready 后 release-store flag；consumer acquire/backoff 轮询 | 软件 CTA readiness 主实现 |
| 协议控制 | `interval-spin`、`exact-backoff` | 隔离退避与编码影响，不进入 gate 的 Impl 选择 |
| Ceiling (`none`) | 与 Impl 相同的独立 priority streams，但不等待 | 故意读取 poison/stale 输出，只提供实测 no-wait 操作参考 |

五档使用相同 producer/consumer CTA 数、128 threads/CTA、计算工作量以及 64 KiB consumer dynamic shared memory；launch 前的资源算术保证每 SM 还留有 producer 槽位。timed post-wait payload 为 O(1)；每个 timed invocation 都重新 poison，所有真实 parent 的完整检查放在独立、不计时的 validation invocation 中。random parent 由无重复置换生成，因而 requested degree 等于真实唯一 parent 数；`effective_degree` 是 interval cover 宽度，在稀疏不规则结构上可远大于 requested degree。

跨 SM makespan 与逐 CTA producer `start/ready/end`、consumer `start/dependency/end` 都使用 `%globaltimer`。多波配置只有同时证明 consumer 入场时仍有 producer 未启动、全部 producer 最终完成、Floor 在 producer tail 中开始，以及 Ceiling 确实观察到 stale 输出，才会得到 `launch_gate=trace_verified`。event/stream 正常完成但必需 timestamp 不完整时，只允许记录 `REJECTED_ATTEMPT` 并做最多 3 次 infrastructure-only retry；它不产生 `SAMPLE`，且 correctness、timeout、缺 overlap 或性能离群均不得重试。正式 v2 没有触发该 retry。

### 3.2 Tier 0.1 same-stream chain 重叠

[`bench/tier0_facts.cu`](../bench/tier0_facts.cu) 对 1→6 stage chain 成对测量 PDL off/on。off 路径使用普通 same-stream 串行派发；on 路径为每个 stage 启用 PSS，先执行独立 prologue，再在 dependency point 等前驱 grid，写出递推 checkpoint 并经 CTA barrier 后 trigger，最后执行可重叠 tail。

每个 timed invocation 用独立 epoch seed 毒化初始状态；对每个 stage/block 的 checkpoint 和最终状态的完整校验放在独立 validation invocation。跨 stage/SM makespan 与 launch/dependency/value-ready/trigger/end trace 均来自 `%globaltimer`。v4 的 `semantics=3` 在 validation、timed sample 和保留 trace 中显式记录单调 epoch，同时输出 checkpoint digest 与 final-state digest。strict validator 独立重算这两类 digest，并校验 epoch 调度、validation/sample 记录顺序、固定 workload、配对统计和 trace 绑定；6 个配置的 372 个 timed sample、12 次 validation 和完整 trace 全部通过。

### 3.3 Tier 0.3 resident-wait 定价

[`bench/tier0_background.cu`](../bench/tier0_background.cu) 的两档使用同一个 waiter kernel、grid、资源参数、poison、全输出验证和 `griddepcontrol` wait。唯一差别是：

- `deferred_gate` 用普通同 stream 顺序，让 waiter 在单 CTA producer 退休后才进入；
- `resident_wait` 对同一 waiter 启用 PSS，使 waiter 能在 ready 前占据 SM 资源并等待。

两档都与同一个执行 LCG 更新的 productive-background kernel 并发，并在同一进程中成对、交替顺序测量。这测的是合成工作下“等待 CTA 占槽”带来的吞吐与端到端代价，不是生产 kernel 的调度曲线，也不是可直接执行的 `[H+]` 派发前门控。

### 3.4 Tier 2/3 native 机制矩阵

[`bench/tier23_protocol_encoding.cu`](../bench/tier23_protocol_encoding.cu) 及三个 Tier 3 native
程序重新实现了旧 `cta_dep_bench` 未能提供的准入语义。安全软件档均在 producer kernel entry
触发 dependent launch、数据 ready 后 release-store identity-safe epoch；grid Floor 在数据
ready 后触发并执行 `cudaGridDependencySynchronize()`。每个 epoch 重新 poison，timed payload
保持 O(1)，完整 parent/data 检查位于独立 validation。unsafe `none` 还用 sentinel/latch 或
trace 强制证明真实 RAW 边读取 poison，避免调度偶然串行把 Ceiling 伪装成正确路径。

四个子实验分别执行协议/编码、CTA diamond、C1 数据传递和 CLC persistent scheduler。计时与
跨 SM trace 使用 `%globaltimer`；只有 CLC 单 CTA 原语片段使用 `clock64`。同正式 binary SHA 的
Compute Sanitizer 覆盖按各报告声明的安全路径并报 0 errors；Nsight Systems 时间线成功，
Nsight Compute 的物理 counter 因宿主权限被明确拒绝。旧 `cta_dep_bench` 仍然是拒绝路径，
native-v2 的完成不会使其历史 timing 重新有效。

### 3.5 Tier 4 Qwen3.6-27B target-specific 包夹

[`bench/llm/tier4_driver.py`](../bench/llm/tier4_driver.py) 对每个 cohort 只构造一次 vLLM engine，
在同一 worker cohort 内生成隔离的 `pdl_off`、`pdl_grid`、`ceiling` compiled variant 和 FULL
CUDA Graph。off 的目标 PTX wait/launch 均为 0；grid 同时具有
`griddepcontrol.wait`/`launch_dependents`；Ceiling 保留 launch、把目标 Triton `gdc_wait`
lowering 改成 no-op。每次调用前核对 active variant identity，Nsight Systems 在独立 proof
window 中把 PTX entry 绑定到实际执行的 graph node；正式 timing 不带 profiler 开销。

每个相邻 triplet 共享新的确定性 prompt/epoch，off 与 grid 的 token 及 cumulative-logprob hex
必须完全一致。Ceiling 必须观测到错误，只取主机端 `LLM.generate()` 墙钟。历史
[`reports/rejected/tier4_llm_semantic_audit.md`](rejected/tier4_llm_semantic_audit.md) 仍是旧路径的
拒绝记录；后续 schema-v3 formal 使用新模型落盘、同进程三档、PTX/cubin 和 graph-node 证据，
没有复用旧 timing。

### 3.6 Tier 5 native 历史链与 production 边界

旧 Python 同 stream 路径无法构造彼此独立的 grid-PDL Floor、CTA Impl 和无序 Ceiling，旧
128K/1M 显式张量方案也不可承载；该路径永久保持拒绝。后续 native formal
按 fail-closed 边界形成以下历史链，三个拒收目录的 timing 都不得被后续修复追认：

- **v6 forward-progress 拒绝**：4K 完成后，32K exact 的 Impl 首次 validation 运行
  `871.366079 s` 仍没有有界前进；整轮 `status=REJECTED, accepted_timing=0`。
- **v7 overlap-validator 拒绝**：4K GPU 路径完成，但旧准入错把
  `consumer_start < upstream_ready` 当成 grid-PDL overlap 必要条件，拒绝了“trigger 后入场、
  upstream tail 结束前开始”的合法重叠；整轮 `accepted_timing=0`。
- **v8 `O(Q²D)` validator 拒绝**：1M GPU raw/trace 已完整产生，但冻结 Python validator
  在 Full Floor 上对每个 query 重扫 dependency wave，被 runner 终止；后续 `O(QD)`
  只读 replay `PASS` 只证明 validator 修复，不恢复 v8 timing。
- **v9 formal `PASS`**：全新目录独立重跑 4K/32K/128K/1M 四点，统一 admission
  为 `PASS, accepted_timing=1`；四点各有 3 warmups、31 repeats 和 4 rungs，即每点
  124 个 timed samples。

v9 中 4K/32K 是 exact CTA mapping；128K/1M 将 physical degree 固定为 64，每个
producer CTA 顺序覆盖 16/128 个 key tile，虽保留完整 logical pair work 与 history loads，
但只能称为 `work_complete_packed_proxy`。Full Floor→Impl 同时包含 full-grid 与
bounded-wave 调度/控制、submission protocol 差异，只是 mechanism envelope；只有
matched Wave-Floor→Impl 才对齐 wave protocol。Ceiling 在四点各 31/31 次输出错误，
是 unsafe no-wait/no-publish 对照，不是可实现上界。production exact-26 是另一条独立准入
链；v1/v2 两次尝试均已原子拒收，canonical 26-row 范围保持 `INCOMPLETE`。用户授权的一小时
compact-14 使用全新结果根，现已获得 scoped formal `PASS/DONE`：14 correctness rows、
1,302 samples、62 summaries，`accepted_compact_workload_timing=1`。它排除 32K/1M timing，
不完成 exact-26，也不提供 CTA bracket；legacy acceptance 字段保持 0，CTA/LLM production
headroom 未定义。因此 §9 仍为 `PARTIAL`。完整边界见
[`production_compact14_scoped_formal_20260805.md`](tier5_dsa/production_compact14_scoped_formal_20260805.md)。

## 4. 配置、计时与统计

| 项目 | Tier 1 formal v2 | Tier 0.3 formal v2 |
|---|---:|---:|
| 配置 | 88 个 Tier 1.1 + 5 个 tail 点 = 93 | 5 个 smem 档 × 3 个 register 档 = 15 |
| Grid | 32/64/128/148/296/1184/4736 CTA | waiter grid = occupancy API 资源上限 × 148 SM |
| 多波覆盖 | `2×/8×/32×SM`，均 trace-proven | 不用于 Tier 1 gate |
| Degree / structure | 1→1024；self/interval/grouped/strided/random | smem 0/8/16/32/64 KiB；实际 registers 26/54/92 |
| Warmup | 每配置每模式 3 次 | 每点每档 3 次 |
| Timed repeats | 每配置每模式 31 次 | 每点 31 组成对重复 |
| Timing | `%globaltimer` 跨流 makespan | `%globaltimer` |
| 聚合 | 每配置各档中位数；按 repeat 配对 bootstrap 5000 次、95% CI | 逐 pair 指标中位数；deterministic bootstrap 2000 次、95% CI |
| 正式样本 | `93 × 31 × 5 = 14415` | `15 × 31 × 2 = 930` 条路径样本（465 对） |
| 完整性 | manifest 93/93；31 repeats；0 retry；0 invalid | 15/15 valid；31 repeats |

Tier 1.1 的 structure 轴固定 degree 32，但 `self` 按定义是 degree 1 的语义控制点；`interval,d32` 与 degree 轴复用同一物理配置，没有重复加权。Tier 1.2 的 5 个 tail 点不进入 gate 中位数。正式结果的 `plan_sweep_complete`、`semantic_proof_complete`、`tier11_manifest_complete`、`statistics_complete` 均为 `true`。

Tier 0.1 v4 的统计账本为 6 个 chain-length 配置、每点 3 组 warmup 与 31 组相邻 off/on pair；每个 repeat 交替两档顺序。每配置分别对 off/on 执行独立全边 validation，并用 2,000 次 deterministic paired bootstrap 报告中位数 95% CI。严格验证账本是 `6 × 31 = 186` pairs、372 samples、12 validations 与 1,776 trace rows；`semantics=3` 还将每条记录绑定到可重算的 epoch/digest 链。

本报告按 session 隔离使用数据：**Tier 1 与 Tier 0.3 的 headline 只来自 v2，Tier 0.1 只来自 v4**。v4 目录虽然也含 Tier 1/Tier 0.3 timing，本报告不使用它们；两个 headline session 不拼接样本、不扩充重复数，也不做跨 session 性能差值归因。v3 timing 没有被判为无效；它是被具有更严格 epoch/digest 与记录顺序绑定的 v4 schema 取代为 Tier 0.1 唯一正式源，不与 v4 混样本。

FAST v5 只用 5 repeats 验证正式入口、语义拒收链和结果 plumbing；它的 `statistics_complete=false`、`plan_sweep_complete=false`，本报告没有从 FAST 提取任何性能数字。第一轮 formal 则因 13 个配置共 30 条 trace-incomplete 记录而整体 `INVALID`；它的 timing 全部禁用，v2 是独立重跑，不与首轮拼样本。

Tier 2/3 native-v2 的统计账本为：

| 子实验 | 配置 | timed samples | trace rows | 统计与验证 |
|---|---:|---:|---:|---|
| §7.1 protocol + §7.3 encoding | 17 | 2,635 | 82,880 | 每配置 3 warmups、31 repeats、2,000 次 bootstrap；全边 validation |
| §7.4 diamond | 10 | 1,240 | 23,680 | 每配置四档、31 repeats；全 stage/block validation |
| §7.5 C1 locality | 7 | 1,085 | 10,360 | 每配置五档、31 repeats；逐 word validation |
| §7.6 CLC scheduler | 1 | 124 | 65,540 | 四策略、31 repeats；token conservation + 全任务 validation |
| **总计** | **35** | **5,084** | **182,460** | strict validator `PASS`，0 errors |

Tier 4 schema-v3 分为两个互不混样本的 cohort：decode 为 batch 1/4/16/64、seq=64、gen=16，
prefill 诊断为 batch 1、seq=4K/32K/128K、gen=2。每点每档 3 warmups、31 timed repeats；
decode 有 `4×3×31=372` 个 timed sample 与 `4×31=124` 次 correctness validation，prefill 有
`3×3×31=279` 个 timed sample 与 `3×31=93` 次 validation。三档使用 Latin-3 相邻顺序；
中位数和配对百分比 CI 使用 2,000 次 bootstrap。只有 decode 四点分类为
`headline_full_decode`；prefill 三点保持 `production_mixed_mode_non_headline`。

Tier 5 native-v9 的统计账本是 4 个上下文、每点 4 个 rung、3 warmups 和 31 timed
repeats，合计 `4 × 4 × 31 = 496` 个 timed samples。每点三个正确路径的全元素
validation 都是 0 mismatch，Ceiling 各 31/31 次观测到错误；每点独立 validator
与统一 campaign admission 均为 `PASS`。v6/v7/v8 分别因 forward progress、overlap
predicate 和 validator 复杂度被整轮拒收，不进入这个账本；production exact-26 v1/v2
也均被拒收，不能与 native-v9 或 compact-14 拼样本。compact-14 的独立统计账本精确为
14 correctness rows、1,302 samples 与 62 summaries；本 umbrella 不在分层报告之外转录或
推测其 timing 数值。

## 5. 头条数字复算

### 5.1 Tier 1 gate 与多波退化

对每个 Tier 1.1 配置，分析器定义：

~~~text
space_pct    = (Floor_ms - Ceiling_ms) / Floor_ms × 100%
captured_pct = (Floor_ms - Impl_ms)    / Floor_ms × 100%
~~~

88 个 Tier 1.1 配置为偶数个，因此全矩阵中位数是排序后第 44、45 项的平均：

~~~text
median space
  = (36.1051025752% + 36.1073152956%) / 2
  = 36.1062089354%  → 36.1062%

median captured
  = (34.8773655804% + 34.9310541311%) / 2
  = 34.9042098557%  → 34.9042%
~~~

space 的两个端点直接来自正式 CSV：

~~~text
min = (9.148864 - 9.004640) / 9.148864 × 100%
    = 1.5764142958%  → 1.5764%       [g4736, random]

max = (1.479136 - 0.899712) / 1.479136 × 100%
    = 39.1731389135% → 39.1731%      [g296, random]
~~~

`32×SM = 4736` 有 15 个 Tier 1.1 配置，中位数是排序后第 8 项：

~~~text
32× space median
  = (9.133280 - 8.719360) / 9.133280 × 100%
  = 4.5319972671%  → 4.5320%

32× captured median
  = (9.135392 - 9.655456) / 9.135392 × 100%
  = -5.6928482106% → -5.6928%
~~~

`8×SM = 1184` 同样有 15 个配置：

~~~text
8× space median
  = (2.735200 - 2.453024) / 2.735200 × 100%
  = 10.3164668032% → 10.3165%

8× captured median
  = (2.735232 - 3.375552) / 2.735232 × 100%
  = -23.4100800225% → -23.4101%
~~~

space 与 captured 的中位项不要求来自同一配置；它们分别对各自的 15 个值排序。`GO` 的算术只读取 88 个 `space_pct` 的中位数：`36.1062% ≥ 8%`。因此它证明 operational-gap gate 通过，却不批准 interval-backoff，后者在 8×/32× 的中位数均慢于 Floor。

样本完整性复算为：

~~~text
formal configurations = 88 Tier 1.1 + 5 Tier 1.2 = 93
timed SAMPLE count     = 93 × 31 repeats × 5 modes = 14415
trace retries          = 0
invalid configurations = 0
~~~

### 5.2 Tier 0.1 same-stream chain

v4 `semantics=3` 完整性账本可复算为：

~~~text
chain configurations = 6
paired repetitions   = 6 × 31 = 186
timed samples        = 186 × 2 = 372
validations          = 6 × 2 = 12
saved trace rows     = 6 stages × 148 CTA × 2 modes = 1776
~~~

六个 chain-length 档的正式统计如下；方括号均为 2,000 次 deterministic bootstrap 的 95% CI：

| stages | PDL off median ms [95% CI] | PDL on median ms [95% CI] | paired speedup [95% CI] | model-implied depth [95% CI] | PDL-on peak median grid / CTA |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.020448 [1.020384, 1.020608] | 1.020576 [1.020448, 1.020640] | 0.999969× [0.999875, 1.000094] | 0.999937 [0.999749, 1.000188] | 1 / 148 |
| 2 | 2.043200 [2.043072, 2.043584] | 1.531840 [1.531584, 1.532032] | 1.333779× [1.333626, 1.334120] | 2.002006 [2.001285, 2.002855] | 2 / 296 |
| 3 | 3.065376 [3.065120, 3.065632] | 2.042272 [2.041952, 2.042784] | 1.500854× [1.500525, 1.500979] | 3.006844 [3.004202, 3.007850] | 2 / 296 |
| 4 | 4.088256 [4.088032, 4.088576] | 2.553152 [2.552832, 2.553248] | 1.601160× [1.600995, 1.601645] | 4.014548 [4.012469, 4.020642] | 2 / 296 |
| 5 | 5.111520 [5.110496, 5.112384] | 3.064512 [3.064192, 3.064896] | 1.667972× [1.667648, 1.668351] | 5.023587 [5.016839, 5.030472] | 2 / 296 |
| 6 | 6.134208 [6.133472, 6.134752] | 3.576160 [3.575424, 3.576928] | 1.715359× [1.715058, 1.715768] | 6.026400 [6.015630, 6.036515] | 2 / 296 |

配对 speedup 是先在同一 repeat 内求 `off_i/on_i`，再取 31 个 ratio 的中位数，不是两个独立中位数相除。保留的 6-stage 最终 pair trace 实测 PDL on 峰值为 2 个同时活跃 grid / 296 CTA，PDL off 为 1 grid / 148 CTA。`model_implied_chain_depth=6.026400` 是把配对 speedup 代入 `speedup/(2-speedup)` 的流水线模型反解，**不是实测的同时活跃 grid 深度**。这两个量不能混写。

### 5.3 Tier 0.3 测得范围与条件化 scenario envelope

15 个点各有 31 个 `deferred_gate` / `resident_wait` pair：

~~~text
pair count        = 15 × 31 = 465
rung sample count = 465 × 2 = 930
~~~

吞吐损失和端到端增量都先在同一个 rep 内配对，再取 31 个 pair 的中位数。全矩阵最小/最大点的中位项可从原始日志复算：

~~~text
minimum throughput loss [low regs, 32 KiB, rep 2]
  = (16748.890421 - 13947.132828) / 16748.890421 × 100%
  = 16.7280191259% → 16.7280%

maximum throughput loss [low regs, 64 KiB, rep 3]
  = (16747.468961 - 7863.556665) / 16747.468961 × 100%
  = 53.0462980208% → 53.0463%

minimum median e2e delta [low regs, 32 KiB, rep 2]
  = 10.866176 - 9.048480
  = 1.817696 ms

maximum median e2e delta [low regs, 64 KiB, rep 2]
  = 19.271744 - 9.049184
  = 10.222560 ms
~~~

这里列出的 rep 是各自排序后的中位项，不是挑选单次最佳/最差样本。对不可直接实现的 B2 派发前门控，§7.2 只能在 `resident ≤ hypothetical [H+] ≤ deferred` 的吞吐单调性及对应 e2e 单调性假设下，给每个资源点 **`[0, measured paired delta]`** 的 scenario envelope。上界不是派发前门控的直接测量值，该 envelope 也不是无条件硬件保证。

### 5.4 Tier 2/3 native-v2 代表性结果

总账先由四个子实验相加：

~~~text
configurations = 17 + 10 + 7 + 1 = 35
timed samples  = 2635 + 1240 + 1085 + 124 = 5084
trace rows     = 82880 + 23680 + 10360 + 65540 = 182460
~~~

协议和 encoding 的两个主要放大量为：

~~~text
prefix / per-identity polls, P=1184 = 134039 / 1184 = 113.209×
strided interval / exact polls, d=2 = 44104 / 592   = 74.500×
strided interval / exact polls, d=64 = 84968 / 18944 = 4.485×
~~~

diamond、C1 与 CLC 的代表点分别复算为：

~~~text
diamond 1:1 ordered penalty = 0.785536 - 0.630496 = 0.155040 ms
diamond 1:10 relative penalty = (2.162144 - 2.006400) / 2.162144
                              = 7.2032%

C1 64 KiB fused-cluster upper-bound saving
  = (0.437856 - 0.432928) / 0.437856 = 1.1255%

CLC consumer / producer = 2.995552 / 0.372992 = 8.0311×
CLC locality / producer = 3.417824 / 0.372992 = 9.1633×
~~~

这些是不同子实验的独立机制结果，不能相加成一个“CTA PDL 总收益”。软件
`poll_loads` 也不是 NCU 物理 L2 request。

### 5.5 Tier 4 full-decode 四点包夹

定义与独立分析器一致：

~~~text
grid_vs_off_pct = 100 × (off_ms - grid_ms) / off_ms
headroom_pct    = 100 × (grid_ms - ceiling_ms) / grid_ms
~~~

四个 headline 从各档 31 次 raw `elapsed_s` 的中位数复算：

~~~text
batch 1: off=225.521418732, grid=221.109885722, ceiling=218.161571771 ms
  grid vs off = 1.956148%; headroom = 1.333416%

batch 4: off=229.201121256, grid=224.657617044, ceiling=221.716003027 ms
  grid vs off = 1.982322%; headroom = 1.309376%

batch 16: off=247.436778154, grid=243.428759743, ceiling=238.973699976 ms
  grid vs off = 1.619815%; headroom = 1.830129%

batch 64: off=448.370289989, grid=443.579917308, ceiling=407.823583111 ms
  grid vs off = 1.068397%; headroom = 8.060855%
~~~

例如 batch 64：

~~~text
100 × (443.579917308 - 407.823583111) / 443.579917308
  = 8.060855%
~~~

四点各自独立，没有本报告认可的跨 batch 平均值。Ceiling 的 31/31 错误输出使这些差值只能
叫零等待时间上界，不能叫可正确执行的 CTA-level speedup。

### 5.6 Tier 5 native-v9 四点统计与 production 边界

每格是 31 次 timed repeat 的中位数；括号内是预注册的配对 delta，定义为
`100 × (median_A - median_B) / median_A`，因此负数表示 Impl 更慢。

| 上下文 | mapping | Full Floor ms | Wave-Floor ms | Impl ms | Ceiling ms（unsafe） | Full→Impl | Wave→Impl |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4K | exact | 0.826592 | 1.271840 | 1.497568 | 1.161536 | -81.173783% | -17.748144% |
| 32K | exact | 24.903584 | 28.050816 | 34.333600 | 27.755552 | -37.866100% | -22.397865% |
| 128K | packed proxy | 29.755584 | 44.118048 | 53.168864 | 42.455680 | -78.685332% | -20.514996% |
| 1M | packed proxy | 335.172192 | 465.388736 | 590.106560 | 441.309760 | -76.060716% | -26.798634% |

4K/32K 是 exact CTA mapping；128K/1M 是保留完整 logical pair work 的 packed proxy，不是
production kernel 或真实模型长上下文曲线。Full→Impl 包含 full-grid 与 bounded-wave 的
调度/控制和 submission 差异；Wave→Impl 才是 matched protocol 比较。四点的 Impl
均比 matched Wave-Floor 慢 17.748144%–26.798634%。Ceiling 各点 31/31 次错误，
它的数值不是安全性能上界。详细 CI、trace、PTX/SASS 和 profiler 边界见
[`reports/tier5_dsa/native_v9_four_context_formal_20260805.md`](tier5_dsa/native_v9_four_context_formal_20260805.md)。

~~~text
native-v9 formal accepted timing = 1
production exact-26 status       = INCOMPLETE (v1/v2 REJECTED)
production compact-14 status     = PASS / DONE (scoped formal)
accepted compact workload timing = 1
legacy timing / CTA acceptance   = 0 / 0
CTA/LLM production headroom      = undefined
~~~

“未定义”不能改写成 0%，也不能用 native-v9 synthetic proxy、Tier 1 synthetic
headroom 或 Tier 4 unsafe Ceiling 填补。v6/v7/v8 的任何 timing 仍位于各自整轮拒收边界内。

## 6. 能成立的结论

1. 在这张 B200 合成矩阵上，Tier 1.1 典型实测 `Ceiling − Floor` operational gap 为 36.1062%，超过 8% gate；manifest、31 次重复、语义证明以及 `2×/8×/32×SM` 多波覆盖均完整，所以机器判决 `GO` 有发布资格。
2. `GO` 不等于当前 Tier 1 software Impl 可部署。8×/32× 的 median captured 均为负，说明该组合 operational path 在大多波合成点慢于 Graph Floor；v2 本身没有 profiler 证据将根因归结为单一调度或等待开销。
3. Tier 0.3 证明 resident waiter 在当前合成 productive background 上有可测且很大的资源代价；B2 派发前门控值得继续评估，但今天只能在明示工程单调性假设下，从 1.817696–10.222560 ms 的配对增量构造条件化 scenario envelope。
4. Tier 0.1 v4 证明 6-stage same-stream PDL chain 可以传递相邻 stage 的 prologue/tail 重叠；它的配对 speedup 为 1.715359×，但 trace 实测峰值只有 2 grids / 296 CTA。12 次独立校验、schema 3 epoch/digest 链和 strict validator 均通过。
5. Tier 2/3 native-v2 已正式完成 35 个配置、5,084 个 timed sample 与 182,460 条 trace row。依赖度和结构必须作为独立轴：规整区间上 interval 最省，而 strided 从 degree=2 起 exact representation 已避免显著假边轮询；identity-safe prefix 的逻辑 acquire traffic 可放大到逐 identity 的 113.209×。
6. Tier 3 进一步证明：本 CTA diamond 删除唯一伪顺序边可恢复约 0.155 ms；C1 的 64 KiB fused-cluster 可实现上界只比 separate Floor 快 1.1255% 且 occupancy 上限下降；CLC producer-priority 在该 1-to-1 合成任务上明显快于另外两策略。这些是三个独立实验，不是可相加的总收益。
7. Tier 4 schema-v3 对 decode 与 prefill cohort 均通过完整 admission。可作为 headline 的四个 full-decode 点在 grid PDL 后剩余 1.333416%、1.309376%、1.830129%、8.060855% 的零等待时间上界；off/grid 正确性与 PTX/cubin、active worker variant、Nsight graph-node 证据闭环。
8. fail-closed 链区分了可接收 formal 与拒绝轮：Tier 1 首轮 trace 不完整是 `INVALID`；Tier 2/3 native-v2 和 Tier 4 schema-v3 独立通过；Tier 5 native-v6/v7/v8 分别因 forward progress、overlap validator 和 `O(Q²D)` validator 失败而整轮 `accepted_timing=0`；native-v9 在全新目录独立重跑后 `PASS, accepted_timing=1`，没有拼接前三轮数据。
9. §9 当前为 **PARTIAL**：native-v9 四上下文 synthetic formal 与 production compact-14 scoped formal 均已完成，后者终态为 `PASS`、`accepted_compact_workload_timing=1`、14/1,302/62；production exact-26 v1/v2 均已拒收，canonical 范围仍未完成。compact 不替代 exact-26 或 CTA bracket，CTA/LLM production headroom 仍未定义。

## 7. 不能成立的结论

1. 不能把 36.1062% 写成真实 LLM、DSA、GEMM、attention 或生产服务 speedup；它来自 synthetic spin workload。
2. 不能说 interval-backoff 捕获了所有规模的 headroom。全矩阵中位数为正主要由较小 grid 主导，8×/32× 已经是负收益。
3. 不能从 `GO`、Tier 4 时间上界、native-v9 synthetic proxy 或 compact-14 scoped PASS 推导 resident-wait/polling 是最终 B1/B2 坐标。Tier 4 没有实现 CTA Impl，Tier 0 还显示占槽代价，production exact-26 仍未准入；compact-14 也没有 CTA Impl/Ceiling。
4. 不能把 Tier 0 的上界当成 `[H+]` pre-dispatch gate 的实测收益。当前可发布的只有在明示工程单调性假设下的每资源点 `[0, measured delta]` scenario envelope。
5. 不能让 native-v2 的完成追认旧 `cta_dep_bench` timing。旧 trigger 语义仍被拒绝；§7.2 仍只有 Tier 0 在工程单调性假设下构造的条件化 scenario envelope。
6. 不能从 Tier 2/3 的逻辑 load 计数声称物理 L2/DRAM traffic，也不能把 synthetic diamond、fused cluster 或 CLC 策略排名外推为生产 kernel 的普遍排序。
7. 不能把 Tier 4 的 1.333%–8.061% 称为 CTA-level 实际收益。Ceiling 删除 wait 且输出错误；四点不存在认可的平均值，batch 64 也不能外推到其他 batch、prompt、模型或服务并发。
8. 不能把三个长上下文点升级成 prefill PDL headline。它们混合 prefill 与 2-token generation，而 target-specific graph/PTX 改动只声明 FULL-decode path。
9. 不能宣称 §9 或 production Tier 5 已完成，也不能引用 native-v6/v7/v8 的 timing、median、CI 或性能排序。可引用的 native synthetic timing 只来自 v9，且不能外推为 production 或 LLM CTA headroom。
10. 不能复用首轮 [`bench/results_20260805_b200_multiwave/`](../bench/results_20260805_b200_multiwave/) 的任何 timing。该轮 13 个配置、30/13640（0.220%）Tier 1.1 mode records 缺少完整 trace，整轮判 `INVALID`；可复用的只有拒绝计数、根因和回归方法。
11. 不能外推到 B300/sm_103、其他 driver/toolkit、不同资源包络或生产调度。各实验的 launch topology 也是解释边界。
12. 不能给出 profiler 级 L2/DRAM/polling traffic 结论。`nsys` 已安装并提供执行/graph-node 证据，但 `ncu` 的物理 counter 访问返回 `ERR_NVGPUCTRPERM`；软件计数不能替代它。
13. 不能把 Tier 0.1 的 `model_implied_chain_depth=6.026400` 写成 6 个 grid 同时驻留或同时活跃；它是模型反解，实测 trace 峰值是 2 grids / 296 CTA。
14. 不能跨 admission 边界拼接样本：Tier 1/Tier 0.3 只用 v2，Tier 0.1 只用 v4，Tier 2/3 只用 native-v2，Tier 4 只用 schema-v3，Tier 5 native synthetic 只用 v9；任何拒绝轮均不得补入，v9、exact-26 v1/v2 与 compact-14 也不得相互拼样本。

综上，本轮的设计建议保持 **provisional**：编码应区分规整 interval 与稀疏 exact set，且暂停把 software resident polling 当成候选终点；Tier 4 已给出真实 LLM 时间上界，Tier 5 native-v9 已给出 synthetic work-complete proxy 的四点结果，但要等 production exact-26 和可正确执行的 CTA/LLM Impl 证据后，才能决定 B1/B2/A3 的最终硬件组合。

## 8. 证据入口

本报告对正式数据源做严格分区：Tier 1 / Tier 0.3 只使用 v2，Tier 0.1 只使用 v4，Tier 2/3
只使用 native-v2，Tier 4 只使用 schema-v3，Tier 5 native synthetic 只使用 v9，production
compact-14 只使用其独立 scoped-formal 根；各分区不跨 session 拼样本。exact-26 v1/v2 的
timing 均处于拒收边界内，compact PASS 不追认它们。

Tier 1 / Tier 0.3 v2：

- 机器 gate：[`bench/results_20260805_b200_multiwave_v2/gate.json`](../bench/results_20260805_b200_multiwave_v2/gate.json)
- Tier 1 每配置 CSV：[`bench/results_20260805_b200_multiwave_v2/pilot_summary.csv`](../bench/results_20260805_b200_multiwave_v2/pilot_summary.csv)
- Tier 1 分析 JSON：[`bench/results_20260805_b200_multiwave_v2/pilot_analysis.json`](../bench/results_20260805_b200_multiwave_v2/pilot_analysis.json)
- Tier 1 原始逐样本记录：[`bench/results_20260805_b200_multiwave_v2/pilot_matrix.log`](../bench/results_20260805_b200_multiwave_v2/pilot_matrix.log)
- Tier 1 expected manifest：[`bench/results_20260805_b200_multiwave_v2/pilot_expected_tags.txt`](../bench/results_20260805_b200_multiwave_v2/pilot_expected_tags.txt)
- Tier 0.3 派生 JSON：[`bench/results_20260805_b200_multiwave_v2/analysis_tier0.json`](../bench/results_20260805_b200_multiwave_v2/analysis_tier0.json)（文件同时含 v2 Tier 0.1 rows，本报告不使用它们）
- Tier 0.3 汇总 CSV：[`bench/results_20260805_b200_multiwave_v2/summary_parsed.csv`](../bench/results_20260805_b200_multiwave_v2/summary_parsed.csv)（同上，只取 Tier 0.3）
- Tier 0.3 原始汇总行：[`bench/results_20260805_b200_multiwave_v2/summary.txt`](../bench/results_20260805_b200_multiwave_v2/summary.txt)（同上，只取 Tier 0.3）
- 完整 session 日志：[`bench/results_20260805_b200_multiwave_v2/session.log`](../bench/results_20260805_b200_multiwave_v2/session.log)
- 设备记录：[`bench/results_20260805_b200_multiwave_v2/device.txt`](../bench/results_20260805_b200_multiwave_v2/device.txt)
- v2 不可变收集归档：[`cta_pdl_results_20260805_050659.tar.gz`](../cta_pdl_results_20260805_050659.tar.gz)

Tier 0.1 v4：

- 正式目录：[`bench/results_20260805_b200_multiwave_v4/`](../bench/results_20260805_b200_multiwave_v4/)
- Tier 0.1 逐样本、schema 3 epoch/digest 与汇总：[`bench/results_20260805_b200_multiwave_v4/tier0_facts.log`](../bench/results_20260805_b200_multiwave_v4/tier0_facts.log)
- Tier 0.1 汇总镜像：[`bench/results_20260805_b200_multiwave_v4/summary.txt`](../bench/results_20260805_b200_multiwave_v4/summary.txt)（本报告只取 `tier0=chain` rows）
- strict validator 结果：[`bench/results_20260805_b200_multiwave_v4/tier0_chain_validation.json`](../bench/results_20260805_b200_multiwave_v4/tier0_chain_validation.json)
- 保留的带 epoch off/on trace：[`bench/results_20260805_b200_multiwave_v4/tier0_chain_trace.csv`](../bench/results_20260805_b200_multiwave_v4/tier0_chain_trace.csv)
- 正式 session 日志：[`bench/results_20260805_b200_multiwave_v4/session.log`](../bench/results_20260805_b200_multiwave_v4/session.log)
- 设备记录：[`bench/results_20260805_b200_multiwave_v4/device.txt`](../bench/results_20260805_b200_multiwave_v4/device.txt)
- v4 不可变收集归档：[`cta_pdl_results_20260805_061446.tar.gz`](../cta_pdl_results_20260805_061446.tar.gz)

Tier 2/3 native-v2：

- 正式目录：[`bench/results_20260805_b200_tier23_native_v2/`](../bench/results_20260805_b200_tier23_native_v2/)
- strict verdict：[`bench/results_20260805_b200_tier23_native_v2/tier23_validation.json`](../bench/results_20260805_b200_tier23_native_v2/tier23_validation.json)
- 汇总与 raw：[`bench/results_20260805_b200_tier23_native_v2/tier23_summary.csv`](../bench/results_20260805_b200_tier23_native_v2/tier23_summary.csv)、[`bench/results_20260805_b200_tier23_native_v2/tier23_matrix.log`](../bench/results_20260805_b200_tier23_native_v2/tier23_matrix.log)
- profiler 状态：[`bench/results_20260805_b200_tier23_native_v2/nsys_status.txt`](../bench/results_20260805_b200_tier23_native_v2/nsys_status.txt)、[`bench/results_20260805_b200_tier23_native_v2/ncu_status.txt`](../bench/results_20260805_b200_tier23_native_v2/ncu_status.txt)
- sanitizer 边界：[`bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_coverage.json`](../bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_coverage.json)
- 分层报告：[`reports/tier2_mechanisms/7_1_protocol_7_3_encoding.md`](tier2_mechanisms/7_1_protocol_7_3_encoding.md)、[`reports/tier3_dimensions/7_4_diamond_ordering.md`](tier3_dimensions/7_4_diamond_ordering.md)、[`reports/tier3_dimensions/7_5_c1_locality.md`](tier3_dimensions/7_5_c1_locality.md)、[`reports/tier3_dimensions/7_6_clc_scheduler.md`](tier3_dimensions/7_6_clc_scheduler.md)

Tier 4 schema-v3：

- 分层主报告：[`reports/tier4_llm/qwen36_27b_full_decode_bracket_20260805.md`](tier4_llm/qwen36_27b_full_decode_bracket_20260805.md)
- 不可变契约：[`results/tier4_schema_v3_formal_v1_20260805/manifest.json`](../results/tier4_schema_v3_formal_v1_20260805/manifest.json)
- 模型身份：[`results/tier4_schema_v3_formal_v1_20260805/model_identity.json`](../results/tier4_schema_v3_formal_v1_20260805/model_identity.json)
- Decode admission / raw / analysis：[`results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/admission.json`](../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/admission.json)、[`results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/raw_triplet.json`](../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/raw_triplet.json)、[`results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/analysis.json`](../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/analysis.json)
- Prefill admission / raw / analysis：[`results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/admission.json`](../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/admission.json)、[`results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/raw_triplet.json`](../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/raw_triplet.json)、[`results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/analysis.json`](../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/analysis.json)
- Nsight 报告：[`results/tier4_schema_v3_formal_v1_20260805/profiles/decode.nsys-rep`](../results/tier4_schema_v3_formal_v1_20260805/profiles/decode.nsys-rep)、[`results/tier4_schema_v3_formal_v1_20260805/profiles/prefill.nsys-rep`](../results/tier4_schema_v3_formal_v1_20260805/profiles/prefill.nsys-rep)

Tier 5 production compact-14 scoped formal：

- 分层主报告：[`reports/tier5_dsa/production_compact14_scoped_formal_20260805.md`](tier5_dsa/production_compact14_scoped_formal_20260805.md)
- 独立结果根：[`bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/`](../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/)
- 终态 admission：[`compact_campaign_admission.json`](../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/compact_campaign_admission.json)，SHA-256=`e76daf0b27bc8d1082126f135a2d57eb78a9caed18fa59f62c011cad4465069a`
- 闭包：source manifest SHA-256=`5c5f80900529728d5f57a8eb78a6d55c10ffb3772bf0a46be64541fa0778c89c`；runtime build SHA-256=`1a6741f693e40777ad2810fd2e0331b6f6854757e5823fc09ff03bc4954d3444`
- profiler sidecar：[`nsys_sidecar.json`](../bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/nsys_sidecar.json) 为独立 `PASS`，不进入 timing matrix
- 准入边界：14/1,302/62；32K/1M timing 排除；legacy timing/CTA acceptance=0；CTA/LLM production headroom 未定义

分层主报告与源码：

- Tier 1 主报告：[`reports/tier1_benefit_map/multiwave_degree_structure_map.md`](tier1_benefit_map/multiwave_degree_structure_map.md)
- Tier 0.3 主报告：[`reports/tier0_base_facts/0_3_resident_wait_throughput.md`](tier0_base_facts/0_3_resident_wait_throughput.md)
- Tier 1 benchmark：[`bench/cta_dep_pilot.cu`](../bench/cta_dep_pilot.cu)
- Tier 0.1 benchmark：[`bench/tier0_facts.cu`](../bench/tier0_facts.cu)
- Tier 0.3 benchmark：[`bench/tier0_background.cu`](../bench/tier0_background.cu)
- 正式 session 入口：[`run_session.sh`](../run_session.sh)
- 实验规格与 gate：[`EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md)
- 有效性规则：[`AGENTS.md`](../AGENTS.md)

拒绝审计、Tier 5 failure-mechanism 与修复入口：

- 首轮 formal 拒绝报告：[`reports/rejected/tier1_multiwave_trace_incomplete_20260805.md`](rejected/tier1_multiwave_trace_incomplete_20260805.md)
- Tier 4 旧路径拒绝报告（未被删除、不得与 schema-v3 混样本）：[`reports/rejected/tier4_llm_semantic_audit.md`](rejected/tier4_llm_semantic_audit.md)
- Tier 5 旧 Python 路径拒绝报告：[`reports/rejected/tier5_dsa_semantic_audit.md`](rejected/tier5_dsa_semantic_audit.md)
- Tier 5 native-v6 formal 拒绝报告：[`reports/rejected/tier5_native_v6_forward_progress_20260805.md`](rejected/tier5_native_v6_forward_progress_20260805.md)
- Tier 5 native-v7 overlap-validator 拒绝报告：[`reports/rejected/tier5_native_v7_overlap_validator_20260805.md`](rejected/tier5_native_v7_overlap_validator_20260805.md)
- Tier 5 native-v8 `O(Q²D)` validator 拒绝报告：[`reports/rejected/tier5_native_v8_validator_complexity_20260805.md`](rejected/tier5_native_v8_validator_complexity_20260805.md)
- Tier 5 native-v9 四上下文 formal 报告：[`reports/tier5_dsa/native_v9_four_context_formal_20260805.md`](tier5_dsa/native_v9_four_context_formal_20260805.md)
- native-v9 unified admission：[`bench/dsa/results_20260805_b200_native_formal_strict_v9/campaign_admission.json`](../bench/dsa/results_20260805_b200_native_formal_strict_v9/campaign_admission.json)
- native-v6 formal 原子拒绝记录：[`bench/dsa/results_20260805_b200_native_formal_strict_v6/formal_rejection.json`](../bench/dsa/results_20260805_b200_native_formal_strict_v6/formal_rejection.json)
- native-v6 32K 最小诊断：[`bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/diagnostic.log`](../bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/diagnostic.log)
- 独立 native-v6 strict smoke（只能作 harness 证据）：[`bench/dsa/results_20260805_b200_native_smoke_strict_v6/`](../bench/dsa/results_20260805_b200_native_smoke_strict_v6/)
- Tier 5 B200 审计目录：[`bench/dsa/results_20260805_b200_audit/`](../bench/dsa/results_20260805_b200_audit/)
- FAST v5 plumbing gate（非性能）：[`bench/results_20260805_smoke_v5/gate.json`](../bench/results_20260805_smoke_v5/gate.json)
- 首轮 `INVALID` 原始目录（禁止 timing）：[`bench/results_20260805_b200_multiwave/`](../bench/results_20260805_b200_multiwave/)
