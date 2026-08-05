# Tier 0.1：Same-Stream PDL 多级链实验报告

## 1. 报告头

- 报告日期：2026-08-05（UTC）
- 实验日期：2026-08-05（UTC）
- 设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0
- 正式结果目录：`bench/results_20260805_b200_multiwave_v4`
- 证据等级：**通过 semantics=3 严格 admissibility validator 的 B200 合成微基准证据**；可以支持本实现、本配置下的 device makespan、正确性与 CTA 时间线结论，不能支持真实应用收益或跨设备架构保证。

本报告的所有性能数字、区间、raw pair、digest 和 trace 只来自上述 v4 正式目录。v3 是已完成的前序回归；它的 timing 不因 v4 出现而被判为无效，但 v4 以更严格的 semantics=3 证据契约取代它作为本报告的唯一正式来源。v3/v4 不拼接样本、不合并重复数，也不用两轮差值做性能归因。2026-08-03 的 5-repeat 旧结果仍只作被拒绝 harness 的诊断材料。

## 2. 执行摘要

在同一 non-blocking CUDA stream 中，实验分别运行 1–6 stage 的 PDL off/on chain。每个配置先做 3 对 warmup，再做 31 对相邻计时；偶数 repetition 为 off→on，奇数 repetition 为 on→off。每次 invocation 的头条时间都是 CTA `%globaltimer` 的完整 chain makespan，即 `max(t_end) - min(t_launch)`，不是 CUDA event 时间。

主要结果如下：

- 1 stage 没有可重叠边，paired speedup 中位数为 `0.999969×`，95% bootstrap CI 为 `[0.999875, 1.000094]`。
- 2–6 stage 的 paired speedup 中位数从 `1.333779×` 增至 `1.715359×`；6 stage 的 PDL off/on makespan 中位数分别为 `6.134208 ms` 与 `3.576160 ms`。
- 每个 2–6 stage 配置的 31 次 PDL-on invocation 都观测到恰好 2 个 grid、296 个 CTA 的峰值活跃量；最终 6-stage trace 也精确重算为 2 grid/296 CTA。**实际 trace 峰值是 2，不是 6。**
- 6-stage 的 `model_implied_chain_depth=6.026400` 只是把 paired speedup 代入理想半阶段流水公式后的代数变换；它不是同时活跃 grid 数，也不是硬件并发深度。
- 12 条独立、非计时 validation（6 个链长 × 2 个模式）全部 `mismatches=0`。semantics=3 validator 用每条 validation 的 epoch 独立重算 checkpoint digest 和 final-state digest，并从 372 条 raw SAMPLE 与 1,776 条 trace record 重算统计与并发计数，最终给出 `PASS` / `errors=0`。

因此，v4 支持的窄结论是：**在这张 B200 上，这个 148-CTA/stage、平衡 prologue/tail 的 same-stream synthetic chain 能把相邻 stage overlap 传播至六级，达到 `1.715359×` 的 6-stage paired speedup；任一时刻实测最多只有两个 stage grid 同时活跃。**

## 3. 程序实际行为

源码中的 `chainK` 对每个 stage 执行以下路径：

~~~text
记录 t_launch（%globaltimer）
    ↓
spin 1,000,000 cycles                    独立 prologue
    ↓
PDL on: cudaGridDependencySynchronize()
    ↓
记录 t_dep_satisfied
    ↓
以非交换递推更新本 block 的 state；validation run 保存该 stage checkpoint
    ↓
记录 t_value_ready；CTA barrier
    ↓
PDL on: 记录 t_trigger，再调用 cudaTriggerProgrammaticLaunchCompletion()
    ↓
spin 1,000,000 cycles                    可重叠 tail
    ↓
记录 t_end（%globaltimer）
~~~

PDL off 使用普通 same-stream kernel launch，不调用 dependency wait/trigger；PDL on 使用 `cudaLaunchKernelEx` 和 `cudaLaunchAttributeProgrammaticStreamSerialization`。写值和 CTA barrier 位于 trigger 之前，后继 stage 的独立 prologue 可以提早开始，但其状态读写位于 dependency wait 之后。`cudaGridDependencySynchronize()` 等到 direct predecessor grid 完成后才返回，因此数据可见性不依赖 trigger 本身。

每次 validation、warmup 和 timed invocation 都用新的 epoch seed 初始化 148 个 block 的 state，避免上次执行留下的正确值造成伪 PASS。semantics=3 在每条 `VALIDATION_TIER0_CHAIN` 和 `SAMPLE_TIER0_CHAIN` 中落盘 epoch，最终 trace 的每行也携带 epoch。validator 根据 chain 长度、warmup/repeat 数和交替顺序核对单调 epoch schedule，不只相信 `poison=` 字符串。

独立 validation 对每个 block 运行非交换递推，检查所有 stage checkpoint 和最终 state；例如 6-stage 的每种模式各检查 740 条相邻 block-edge、888 个 stage output 和 148 个 final output。validator 不仅要求 observed/expected 相等，还用 epoch、stage 和 block 数在 Python 中独立重算 checkpoint 与 final digest；任一 mismatch、digest 不一致或 trace 不完整都会使准入失败。

每次 timed invocation 同步完成 poison 初始化后才启动 chain；初始化、host launch overhead 与末尾 D2H copy 不计入头条时间。程序从该 invocation 的全部 CTA trace 计算 makespan、半开区间 `[t_launch,t_end)` 的峰值活跃 CTA/grid、相邻 stage 是否 early launch，以及所有后继 CTA 的 dependency point 是否晚于前驱 grid 的最晚 `t_end`。

semantics=3 validator 将 trace 重算的 `peak_active_ctas`、`peak_active_grids`、`early_links`、`dependency_safe_links` 和 `serial_links` 与最终 SAMPLE 做**整数精确相等**比较，只对六位小数的 makespan 使用打印精度容差。它还绑定正式 workload 常量、record 顺序、trace 声明路径和 trace epoch；缺字段、非数值、越界 stage、重复行或内部异常都记为 FAIL。写 `--json` 前先清除旧结果，新结果经临时文件原子替换，避免畸形产物导致旧 PASS JSON 被继续引用。

## 4. 配置与统计

| 项目 | 正式 v4 配置 |
|---|---:|
| GPU | NVIDIA B200，148 SM，CC 10.0 |
| 证据语义 | `semantics=3` |
| Chain stages | 1、2、3、4、5、6 |
| Grid / block | 148 CTA/stage，128 threads/CTA |
| Stage work | 2,000,000 cycles |
| Prologue / tail | 1,000,000 / 1,000,000 cycles |
| 模式 | PDL off、PDL on |
| Warmup | 3 个相邻 pair/配置 |
| Timed repeats | 31 个相邻 pair/配置 |
| 顺序 | adjacent alternating：偶数 off→on，奇数 on→off |
| Epoch | validation/SAMPLE 逐 invocation 单调；trace 行与最终 SAMPLE epoch 精确一致 |
| 原始计时记录 | 6 × 31 × 2 = 372 条 `SAMPLE_TIER0_CHAIN` |
| 时间源 | 每个 CTA 的 `%globaltimer`；invocation makespan = `max(t_end)-min(t_launch)` |
| 汇总 | 单模式中位数；逐 pair speedup 的中位数 |
| 不确定性 | 2,000 次 deterministic bootstrap，中位数的 95% percentile CI |
| 独立正确性 | 12 条 validation；checkpoint/final digest 均经 validator 独立重算 |
| 最终 trace | rep 30；off epoch 419、on epoch 420；各 6 × 148 = 888 条，共 1,776 条 |

下表的每个中位数与 CI 都由 semantics=3 validator 从 v4 的 31 个相邻 raw pair 重算；方括号内均为 95% bootstrap CI。最后一列是 trace 区间扫描的直接计数，格式为 `中位数 [CI] / 最大值`，不是模型反解值。

| Stages | PDL off ms [CI] | PDL on ms [CI] | Paired speedup [CI] | Model-implied depth [CI] | PDL-on 峰值 grid；CTA |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.020448 [1.020384, 1.020608] | 1.020576 [1.020448, 1.020640] | 0.999969× [0.999875, 1.000094] | 0.999937 [0.999749, 1.000188] | 1 [1, 1] / 1；148 [148, 148] / 148 |
| 2 | 2.043200 [2.043072, 2.043584] | 1.531840 [1.531584, 1.532032] | 1.333779× [1.333626, 1.334120] | 2.002006 [2.001285, 2.002855] | 2 [2, 2] / 2；296 [296, 296] / 296 |
| 3 | 3.065376 [3.065120, 3.065632] | 2.042272 [2.041952, 2.042784] | 1.500854× [1.500525, 1.500979] | 3.006844 [3.004202, 3.007850] | 2 [2, 2] / 2；296 [296, 296] / 296 |
| 4 | 4.088256 [4.088032, 4.088576] | 2.553152 [2.552832, 2.553248] | 1.601160× [1.600995, 1.601645] | 4.014548 [4.012469, 4.020642] | 2 [2, 2] / 2；296 [296, 296] / 296 |
| 5 | 5.111520 [5.110496, 5.112384] | 3.064512 [3.064192, 3.064896] | 1.667972× [1.667648, 1.668351] | 5.023587 [5.016839, 5.030472] | 2 [2, 2] / 2；296 [296, 296] / 296 |
| 6 | 6.134208 [6.133472, 6.134752] | 3.576160 [3.575424, 3.576928] | 1.715359× [1.715058, 1.715768] | 6.026400 [6.015630, 6.036515] | 2 [2, 2] / 2；296 [296, 296] / 296 |

## 5. 头条数字复算

`paired_speedup` 是 31 个 `off_i/on_i` 的中位数，不是两个单模式中位数之比。6-stage raw pair 的 rep 11 是排序后的中位样本；它是奇数 repetition，日志顺序为 PDL on（order 0）后 PDL off（order 1）：

~~~text
rep 11 PDL on  (epoch 381) = 3.577056 ms
rep 11 PDL off (epoch 382) = 6.135936 ms

paired_speedup_11 = 6.135936 / 3.577056
                    = 1.715359222780
                    → 1.715359×
~~~

理想的等长半阶段流水模型把深度 `d` 的 speedup 写成 `2d/(d+1)`；反解只是：

~~~text
model_implied_chain_depth
  = paired_speedup / (2 - paired_speedup)
  = 1.715359222780 / (2 - 1.715359222780)
  = 6.026400150858
  → 6.026400
~~~

这个结果略高于实际 6-stage 链，正说明它只是带测量波动和固定开销的模型坐标，不能当作并发计数。真实并发由 CTA 区间扫描得到：6-stage PDL-on 的 31 个 raw sample 都是 `peak_active_grids=2`、`peak_active_ctas=296`；最终 rep 30、epoch 420 的 888 条 PDL-on trace 重算为：

~~~text
makespan = (max(t_end) - min(t_launch)) / 1e6 = 3.577248 ms
peak active grids = 2
peak active CTAs  = 296 = 2 × 148
early links = 5 / 5
dependency-safe links = 5 / 5
serial links = 0 / 5
~~~

同一最终 pair 的 PDL-off（epoch 419）888 条 trace 重算为 `6.133024 ms`、1 grid/148 CTA、0/5 early、5/5 dependency-safe、5/5 serial。两种模式合计 `888 + 888 = 1,776` 条；validator 将五个整数指标与最终 SAMPLE 逐字段精确比较，makespan 也在六位打印精度内一致。

correctness 另有独立复算链。以 6-stage 为例，PDL off validation 的 epoch 351 对应 checkpoint/final digest `16920009572652058963` / `11570757668265459218`；PDL on 的 epoch 352 对应 `14811194027036825937` / `13896373960284195133`。这四个值都不是仅检查 producer 自报相等；validator 从 epoch 和递推定义独立生成同一 expected digest 后才允许 PASS。

## 6. 能成立的结论

1. 在该 B200、CUDA 13.0 与当前 driver 环境中，这个 same-stream PDL 路径确实让后继 stage 提早进入：2–6 stage 的每次 PDL-on timed invocation 都覆盖全部相邻 early link，同时全部 dependency point 保持安全。
2. 对这个 148-CTA/stage、1M-cycle prologue + 1M-cycle tail 的平衡 synthetic chain，overlap 能沿六个 stage 传播；6-stage paired speedup 中位数为 `1.715359×`，95% CI `[1.715058, 1.715768]`。
3. 本配置的 trace 实测同时活跃峰值为两个 grid、296 CTA。六级流水传播与六个 grid 同时活跃是两件不同的事。
4. v4 的 372 条性能样本具有相邻配对、交替顺序、逐 invocation epoch、raw record 和 bootstrap CI；12 条独立 validation 检查所有 stage checkpoint/final output，两类 digest 均经独立重算并通过。
5. 6-stage 最终 trace 同时证明：后继 grid 在前驱结束前 launch，但其 dependency point 不早于前驱 grid 的最晚结束时间；本次性能收益没有以错误消费前驱 state 为代价。
6. semantics=3 validator 对本 v4 产物的结论是 `PASS`：6 个配置、12 条 validation、186 个 pair、372 条 SAMPLE、1,776 条 trace、0 个 error。畸形或不完整产物不会沿用旧 PASS JSON。

## 7. 不能成立的结论

1. `model_implied_chain_depth=6.026400` **不证明**六个或 6.0264 个 grid 同时活跃，也不是 B200 的架构并发上限；trace 的直接答案是 2。
2. 本实验不证明任意 kernel chain、GEMM、Attention、LLM 或 DSA workload 都能获得 `1.715359×`。它只覆盖平衡的 spin-cycle synthetic stage。
3. 本实验不覆盖 multi-wave grid、不同 CTA 数、register/shared-memory 压力、stage 不平衡或复杂 DAG，不能外推这些条件下的 overlap、occupancy 或 forward progress。
4. PDL 是 opportunistic；这里 31 次重复都出现相同的两-grid 峰值，不构成 CUDA 对未来设备、driver 或 toolkit 的派发保证，也不保证 CTA id 到 SM 的固定映射。
5. `%globaltimer` makespan 是 device 上从最早 CTA launch 到最晚 CTA end 的区间；它不包含 poison 初始化、host launch overhead、同步和 D2H copy，因而不是应用端 wall-clock latency。
6. v3 的数据不因 semantics=3 出现而自动变成无效 timing；但它没有 v4 的逐记录 epoch、checkpoint/final digest 独立重算、整数 trace 精确回绑和畸形产物 fail-closed 契约，因而仅作被 v4 取代的前序回归，不参与本报告任何数字或 CI 计算。
7. 2026-08-03 `bench/results_budget1h_corrected` 中的旧 Tier 0.1 chain 只有 5 repeats，使用旧计时/trace schema，且没有独立 end-to-end correctness validation。其数字已拒绝用于当前性能或正确性结论，只能用于诊断旧 harness 为什么需要后续重测。
8. 本报告不从 Tier 0.1 单项结果推导 Tier 1 gate、CTA-level PDL 的最终设计坐标或真实工作负载投资回报；这些决定需要各自的正式证据链。

## 8. 证据入口

- Benchmark 源码：[bench/tier0_facts.cu](../../bench/tier0_facts.cu)
- 严格 validator：[tools/validate_tier0_chain.py](../../tools/validate_tier0_chain.py)
- v4 设备记录：[bench/results_20260805_b200_multiwave_v4/device.txt](../../bench/results_20260805_b200_multiwave_v4/device.txt)
- v4 原始 CONFIG、12 条 validation、372 条 SAMPLE 与 6 条 SUMMARY：[bench/results_20260805_b200_multiwave_v4/tier0_facts.log](../../bench/results_20260805_b200_multiwave_v4/tier0_facts.log)
- v4 汇总副本：[bench/results_20260805_b200_multiwave_v4/summary.txt](../../bench/results_20260805_b200_multiwave_v4/summary.txt)
- v4 最终 off/on CTA trace：[bench/results_20260805_b200_multiwave_v4/tier0_chain_trace.csv](../../bench/results_20260805_b200_multiwave_v4/tier0_chain_trace.csv)
- v4 严格验证重算结果：[bench/results_20260805_b200_multiwave_v4/tier0_chain_validation.json](../../bench/results_20260805_b200_multiwave_v4/tier0_chain_validation.json)
- v4 完整 session 日志：[bench/results_20260805_b200_multiwave_v4/session.log](../../bench/results_20260805_b200_multiwave_v4/session.log)
- v4 不可变收集归档：[cta_pdl_results_20260805_061446.tar.gz](../../cta_pdl_results_20260805_061446.tar.gz)
- 本次 campaign 总报告：[reports/campaign_b200_multiwave_20260805.md](../campaign_b200_multiwave_20260805.md)
- 仅作拒绝诊断、不得混入本报告数字的旧日志：[bench/results_budget1h_corrected/tier0_facts.log](../../bench/results_budget1h_corrected/tier0_facts.log)
