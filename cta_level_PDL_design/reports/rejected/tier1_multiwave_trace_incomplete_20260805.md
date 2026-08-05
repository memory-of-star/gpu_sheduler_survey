# Tier 1 多波正式首轮：时间戳不完整拒绝审计报告

报告日期：2026-08-05（UTC）  
实验日期：2026-08-05（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0，183359 MiB  
证据等级：**REJECTED-TIMING / TRACE-FAILURE-DIAGNOSTIC**——可支持首轮正式
session 为何被判 `INVALID`、失败的规模分布、reset ordering race 的修复与非性能回归验证；
不能支持任何 latency、throughput、speedup、headroom、median 或置信区间结论

## 1. 执行摘要

首轮正式多波结果目录
[`bench/results_20260805_b200_multiwave/`](../../bench/results_20260805_b200_multiwave/)
必须整体拒绝。机器 gate 给出的判决是 `INVALID`：Tier 1.1 的 88 个配置中有 13 个配置未能
提供完整的 `%globaltimer` 语义轨迹，因而不能证明这些配置满足多波前进性与 launch-gate
准入条件。

该轮 Tier 1.1 共计划 `88 × 31 × 5 = 13640` 个逐模式样本，其中 30 条记录出现
`trace_complete=0`，占 `30 / 13640 = 0.220%`。这 30 条全部集中在 1184 CTA（8×SM）与
4736 CTA（32×SM）两个规模：前者 6 条、后者 24 条；其余正式 grid 没有出现同类记录。
失败跨越 Floor、三种软件模式和 Ceiling，不是某一种等待协议独有的现象。

失败比例很小不改变判决。`AGENTS.md` 的 fail-closed 规则要求每个正式配置都有完整语义证明；
任一配置缺失即使其余记录完整，也不能通过删除坏样本或只汇总幸存配置来恢复有效性。因此：

- 首轮 session 的**全部 timing 均不可复用**，包括同一 session 内看似独立完成的 Tier 0；
- 首轮目录与归档必须原样保留，供审计 reset/trace failure；
- 修复后新的 `bench/results_20260805_b200_multiwave_v2/` 若成功，其结果只能在新的 Tier 1
  主报告和 umbrella report 中独立陈述，不能追溯性地修补本轮。

本报告不引用首轮的任何性能时间、中位数、置信区间或由它们计算的百分比。

## 2. 程序实际执行的语义与失效点

首轮 `cta_dep_pilot` 的每个 repeat 相邻运行五个模式，并为每个 producer CTA 写
`start/ready/end`，为每个 consumer CTA 写 `start/dependency-satisfied/end`。首轮准入代码实际只把
producer/consumer 的 `start/end` 四组数组纳入 `trace_complete`；`ready`、
`dependency-satisfied` 尚未被 host 端逐槽检查，也没有验证每个 CTA 内的时间单调性。即使按这份
较窄的旧检查，13 个配置仍出现零槽并 fail-closed。修复版随后把六组数组的非零与
`start ≤ middle ≤ end` 全部纳入准入，不能把这项更强检查追溯性地写成首轮已经具备。

首轮多波配置还要求 producer 最终全部完成、Floor 在 producer tail 中启动，且软件 consumer
在仍有 producer 未启动时入场，才能写 `launch_gate=trace_verified`。零槽使这些必要聚合证明
无法成立。

首轮的 reset 会清 poison、flag、error、stale counter 与六组 timestamp 数组，随后从
`cudaStreamNonBlocking` 的 Graph、producer 和 consumer stream 发起工作。问题是 reset 与这些
非阻塞 launch stream 之间没有一个明确的 device-completion barrier。一次迟到的 timestamp
reset 因而可能与新 kernel 写 timestamp 竞争，并把已经写出的有效槽位重新清零。对应记录中，
launch 路径已经走到完成事件，但 host 回读的必需 timestamp 仍有零槽；旧实现于是只能把
headline trace 标为不完整。

这不是可以忽略的 profiler 丢包。`%globaltimer` 数组本身就是本实验证明真实跨波 overlap 和
producer progress 的准入证据。某个槽位为零时，分析器无法区分“CTA 没有执行”“写入被 reset
覆盖”或“时间线不可见”，所以相关配置必须写
`launch_gate=trace_failed producer_progress_complete=0 multiwave_overlap_proven=0 valid=0`，随后由
gate 拒绝整轮。

原始 `failures.log` 使用了通用的 “validation failure” 标签；本次代码级审计把具体原因定位为
**必需 trace/semantic proof 不完整**。这不等于已经证明某条真实依赖边计算错误，也不能据此
把首轮改写成 correctness PASS。

## 3. 配置、统计与失败分布

Tier 1.1 正式矩阵使用 148 SM 的实际设备信息，grid 为
`32/64/128/148/296/1184/4736`；依赖度轴覆盖 1→1024，结构轴覆盖
`self/interval/grouped/strided/random`，其中 `self` 按定义为 degree 1。每个配置 31 个 repeat，
每个 repeat 五个模式。下面只统计 trace 完整性，不统计任何性能量。

| Grid | 相对 148 SM | 受影响配置数 | `trace_complete=0` 记录数 | 受影响配置 |
|---:|---:|---:|---:|---|
| 1184 | 8× | 5 | 6 | `d4`、`d8`、`d128`、`d1024`、`strided,d32` |
| 4736 | 32× | 8 | 24 | `d1`、`d2`、`d16`、`d128`、`d256`、`d512`、`grouped,d32`、`strided,d32` |
| **合计** | — | **13** | **30** | — |

按执行模式拆分，30 条不完整轨迹覆盖全部五档：

| 模式 | 不完整记录数 |
|---|---:|
| `grid`（Floor） | 11 |
| `interval-spin` | 6 |
| `interval-backoff` | 4 |
| `exact-backoff` | 6 |
| `none`（Ceiling） | 3 |
| **合计** | **30** |

受影响的 13 个机器 tag 为：

~~~text
t11p_g1184_d4        t11p_g1184_d8        t11p_g1184_d128
t11p_g1184_d1024     t11ps_g1184_strided
t11p_g4736_d1        t11p_g4736_d2        t11p_g4736_d16
t11p_g4736_d128      t11p_g4736_d256      t11p_g4736_d512
t11ps_g4736_grouped  t11ps_g4736_strided
~~~

首轮 manifest、31-repeat 数量与 parent uniqueness 本身均存在，但 13 个配置的语义轨迹失败，
所以 `semantic_proof_complete=false`、`plan_sweep_complete=false`。完整 nominal 参数覆盖不能替代
逐配置准入证明。

## 4. 诊断数字复算

Tier 1.1 的分母不包含五个 Tier 1.2 tail 点：

~~~text
Tier 1.1 samples = 88 configurations × 31 repeats × 5 modes
                   = 13640
trace-incomplete   = 30
failure fraction   = 30 / 13640 × 100%
                   = 0.219941...%
                   = 0.220%  (rounded to three decimals)
~~~

规模分布可独立复算为：

~~~text
affected configurations = 5 (1184 CTA) + 8 (4736 CTA) = 13
trace-incomplete records = 6 (1184 CTA) + 24 (4736 CTA) = 30
mode count               = 11 + 6 + 4 + 6 + 3 = 30
~~~

这些是完整性诊断计数，不是性能 headline。本轮没有可合法复算的 Floor、Impl、Ceiling
性能差值；任何使用首轮 timing 进行的算术都违反 `INVALID` 的整轮拒绝规则。

## 5. 修复与非性能验证

修复后的 [`bench/cta_dep_pilot.cu`](../../bench/cta_dep_pilot.cu) 做了三层收口：

1. `resetPilot()` 在 reset 前后执行显式 device synchronize，使全部 poison/清零在任何
   non-blocking launch stream 开始之前完成。
2. 完成事件之后显式 synchronize 对应 Graph/producer/consumer stream，再回读六组 timestamp，
   建立清晰的 stream-completion 与 host-visibility 边界。
3. 回读时分别统计六种 missing slot 和 producer/consumer 时间顺序错误。只有 event/stream
   已完成但 trace 不完整的 attempt 才可输出 `REJECTED_ATTEMPT` 并内部重试；每个 rep/mode
   最多重试 3 次。timeout、CUDA error、正确性失败、缺失 overlap、Ceiling 未观察到 stale
   输出或性能离群均不得借此重试。被拒 attempt 不写 `SAMPLE`、不进入统计；重试耗尽非零退出。

最终修复后的定向压力验证保存在
[`bench/results_20260805_trace_retry_debug_v2/`](../../bench/results_20260805_trace_retry_debug_v2/)：
它覆盖 8×SM 与 32×SM 的五个配置，包括 interval 的低/高 degree 和 32×SM strided 点；每点
31 repeats、五个模式，共 `5 × 31 × 5 = 775` 个完整 `SAMPLE`。五个配置均完成，分析器报告
`all_valid=true`、`minimum_repeats=31`、`statistics_complete=true`，且该压力轮没有
`REJECTED_ATTEMPT`。

随后正式入口的 FAST v5 在
[`bench/results_20260805_smoke_v5/`](../../bench/results_20260805_smoke_v5/) 完成 26 个 plumbing
配置、650 个短样本，无 missing/unexpected configuration、无 failed validation，gate 的
`semantic_proof_complete=true`。FAST 只有 5 repeats，按设计
`statistics_complete=false`、`plan_sweep_complete=false`，因此这里仅把它作为修复后入口与
fail-closed 判决链能运行的证据，绝不作为性能结论。

压力验证与 FAST 没有再次触发 race，支持上述 ordering 修复有效；它们不证明 race 在所有未来
driver/toolkit 或无限运行中绝不会再出现。新的完整 formal v2 必须独立通过自己的 manifest、
31-repeat、semantic proof 和 gate，成功后另写主报告。

## 6. 能成立的结论

本审计支持：

1. 首轮正式 gate 的权威判决为 `INVALID`，直接原因是 13 个 Tier 1.1 配置缺少完整必需轨迹。
2. 30/13640（0.220%）的不完整记录全部出现在 8×SM 与 32×SM，且横跨五种模式，符合
   reset/launch ordering race 而非单一等待协议失效的特征。
3. 缺失 timestamp 会同时破坏 producer progress 和 multiwave overlap 的证明；即使 CUDA
   完成事件已出现，也不能用事件存在替代逐 CTA `%globaltimer` 证据。
4. 显式 reset barrier、完成后的 stream synchronization、逐字段 trace audit 与受限的
   infrastructure-only retry 关闭了已识别的失效路径。
5. 修复后的 8×/32×定向压力验证与 FAST v5 均通过各自适用的非性能完整性检查。
6. 首轮结果目录和归档保留了失败现场，没有被修复代码覆盖。

### 6.1 可保留和复用的材料

可复用的只有审计与工程材料：实际 B200 设备记录、正式 manifest、13 个失败 tag、30 条
trace-incomplete 记录及其 grid/mode 分布、`gate.json` 的 `INVALID` 判决、原始日志、reset race
根因、修复补丁和定向回归方法。这些材料可以用于未来 harness 的回归测试与 fail-closed 设计。

首轮原始目录、归档和失败日志必须继续原样保存；“可复用”不表示可以从其中恢复任何 timing，
也不表示可以把首轮中未受影响的配置拼接到 v2。

## 7. 不能成立的结论

本审计不支持：

1. 首轮任何 Floor、Impl 或 Ceiling 的 latency、throughput、speedup、space、captured ratio、
   median、range 或置信区间。
2. 首轮中未出现 `trace_complete=0` 的配置仍可单独作为性能数据；gate 的拒绝范围是整轮。
3. 同一首轮 session 的 Tier 0 timing 可以保留。它们同样处于本轮全局 `INVALID` 边界内，必须
   由成功的新 session 重测。
4. 13 个配置已经被证明存在依赖计算错误。已证明的是 trace/semantic admission 不完整，不能
   把通用 “validation failure” 日志标签扩写成未观察到的 correctness 结论。
5. 0.220% 很小所以可以静默丢弃；缺失与大 grid 相关，静默筛选还可能引入选择偏差。
6. 修复后的 stress/FAST 证明任何 CTA-level PDL 性能收益，或证明未来永远不会出现 trace retry。
7. v2 的成功能够追溯性地使首轮有效，或允许把两轮样本混合统计。v2 若成功，只能作为独立
   session 在新的 Tier 1 主报告中解释。
8. 首轮 `INVALID` 可以被重判为 `GO`、`LLM_ONLY` 或 `STOP`。数值 gate 在准入失败时没有发布资格。

## 8. 证据入口

- 首轮完整原始目录：[`bench/results_20260805_b200_multiwave/`](../../bench/results_20260805_b200_multiwave/)
- 首轮机器 gate：[`bench/results_20260805_b200_multiwave/gate.json`](../../bench/results_20260805_b200_multiwave/gate.json)
- 首轮逐样本矩阵：[`bench/results_20260805_b200_multiwave/pilot_matrix.log`](../../bench/results_20260805_b200_multiwave/pilot_matrix.log)
- 首轮 expected manifest：[`bench/results_20260805_b200_multiwave/pilot_expected_tags.txt`](../../bench/results_20260805_b200_multiwave/pilot_expected_tags.txt)
- 首轮失败清单：[`bench/results_20260805_b200_multiwave/failures.log`](../../bench/results_20260805_b200_multiwave/failures.log)
- 首轮设备记录：[`bench/results_20260805_b200_multiwave/device.txt`](../../bench/results_20260805_b200_multiwave/device.txt)
- 首轮完整 session 日志：[`bench/results_20260805_b200_multiwave/session.log`](../../bench/results_20260805_b200_multiwave/session.log)
- 首轮不可变收集归档：[`cta_pdl_results_20260805_044753.tar.gz`](../../cta_pdl_results_20260805_044753.tar.gz)
- 修复后 pilot source：[`bench/cta_dep_pilot.cu`](../../bench/cta_dep_pilot.cu)
- 最终定向压力矩阵：[`bench/results_20260805_trace_retry_debug_v2/stress_matrix.log`](../../bench/results_20260805_trace_retry_debug_v2/stress_matrix.log)
- 最终定向压力分析：[`bench/results_20260805_trace_retry_debug_v2/stress_analysis.json`](../../bench/results_20260805_trace_retry_debug_v2/stress_analysis.json)
- FAST v5 gate：[`bench/results_20260805_smoke_v5/gate.json`](../../bench/results_20260805_smoke_v5/gate.json)
- Benchmark 有效性规则：[`AGENTS.md`](../../AGENTS.md)
- Tier 1 规格与 gate：[`EXPERIMENT_PLAN.md`](../../EXPERIMENT_PLAN.md)

新正式目录 `bench/results_20260805_b200_multiwave_v2/` 不属于本拒绝报告的性能证据。其成功或
失败必须由该目录自己的 gate 决定；成功结果在 Tier 1 主报告中另述，失败则另建拒绝审计，二者
都不得改写本报告记录的首轮事实。
