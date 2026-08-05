# Tier 1：B200 多波依赖度 × 结构收益地图

## 1. 报告头

报告日期：2026-08-05（UTC）

实验日期：2026-08-05（UTC）

设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0，183359 MiB

证据等级：**VALID / SYNTHETIC-MICROBENCH / B200-ONLY**——可支持当前
128-thread、64 KiB consumer、spin-work synthetic 构造在欠填充、1×、2×、8×、32× SM
规模上的 Floor/软件 Impl/Ceiling 包夹、真实多波 trace 准入和 §6 gate 判决；不能支持真实
GEMM、Attention、LLM、DSA 的端到端收益，也不能支持一个尚未实现的硬件 Ideal 点

本报告只使用成功正式目录
[`bench/results_20260805_b200_multiwave_v2/`](../../bench/results_20260805_b200_multiwave_v2/)
的计时。首轮正式 session 已被整体判 `INVALID`；它只通过
[`reports/rejected/tier1_multiwave_trace_incomplete_20260805.md`](../rejected/tier1_multiwave_trace_incomplete_20260805.md)
保留为拒绝审计，本报告没有引用或混入其中的任何 timing。

本实验直接观察 [`docs/cta_pdl_design_space.md`](../../docs/cta_pdl_design_space.md) E1 的
degree/grid 收益边界，并以 A3 的 tight interval 与
conservative cover、B1 的 spin/backoff、B2 的 resident wait 作软件诊断坐标。
它产出 Floor / 三个 Impl 诊断 / Ceiling，**没有 Ideal**；它改变的决策是
[`EXPERIMENT_PLAN.md`](../../EXPERIMENT_PLAN.md) §6 的 Tier 1 Gate，而不是直接选定一个硬件实现。

## 2. 执行摘要

权威机器判决是 **`GO`**。Gate 只对 88 个 Tier 1.1 配置的逐配置
`space_pct = (Floor - Ceiling) / Floor` 取中位数，得到：

| Gate headline | 结果 |
|---|---:|
| Tier 1.1 配置数 | 88 |
| Median headroom (`space_pct`) | **36.1062%** |
| Headroom range | **1.5764% .. 39.1731%** |
| Median software captured (`captured_pct`) | **34.9042%** |
| 判决 | **GO**（阈值为 `space_pct >= 8%`） |

`captured_pct` 只描述预先声明的软件 Impl `interval-backoff` 相对 Floor 的差值，**不参与
Gate 判决**。Gate 同时确认 manifest、31 repeats、parent uniqueness、2×/8×/32× 真实多波
trace、语义证明和完整 plan sweep 均通过。因此这个 `GO` 不是先前只覆盖 `P,C ≤ SM` ratio 的 provisional verdict，
而是 [`EXPERIMENT_PLAN.md`](../../EXPERIMENT_PLAN.md) §6 定义下的完整 Tier 1 synthetic gate
结果。

但跨规模的结构比总中位数更重要：

| Grid | Tier 1.1 配置数 | Median headroom | Median software captured |
|---:|---:|---:|---:|
| 148（1×SM） | 12 | 36.0865% | 35.4905% |
| 296（2×SM） | 13 | 39.1231% | 38.2968% |
| 1184（8×SM） | 15 | 10.3165% | -23.4101% |
| 4736（32×SM） | 15 | **4.5320%** | **-5.6928%** |

同一正式 session 的各 grid 子矩阵中，收益空间从 1× 的 36.0865% 与 2× 的
39.1231%，降到 8× 的 10.3165% 和 32× 的 4.5320%。同时，当前软件
`interval-backoff` 在 8× 与 32× 的典型配置中比 Floor 更慢。尤其不能把总体 36.1062%
解释成“32× 大 grid 仍有 36%”：Gate 是 88 个配置的无权重中位数，较小 grid 的配置数量足以
决定其中点。

结构轴也不是一句“irregular 都差”就能概括。在 32× 上，`random` / `strided` 的保守
interval cover 分别膨胀到平均 4377.90 / 4589.00 个 parent，`captured_pct` 为
-308.9478% / -370.9188%；但这是当前 interval-cover 软件编码、独立 stream 调度和 synthetic
spin 共同作用的结果，不是任何未来硬件不规则依赖机制的普遍结论。

## 3. 程序实际执行了什么

### 3.1 三档包夹与两个不同 launch topology

每个 timed repeat 在同一进程内相邻运行五个 mode；用于 headline 的三档以及两个协议诊断为：

| 角色 | Mode | 实际 launch / wait 语义 | 正确性 |
|---|---|---|---|
| Floor | `grid` | producer→consumer 的 programmatic CUDA Graph edge；producer 数据 ready 后 trigger，consumer 执行 `cudaGridDependencySynchronize()` | 单独全边验证 |
| Impl | `interval-backoff` | consumer 先入低优先级独立 stream，producer 后入高优先级独立 stream；producer 数据写完后 release-store per-CTA flag，consumer 对 interval cover 做 acquire + backoff | 单独全边验证 |
| Ceiling | `none` | 与 Impl 相同的独立优先级 streams，但完全不等 readiness | **故意错误，只取 timing** |
| 协议诊断 | `interval-spin` | 与 Impl 相同，interval cover 紧轮询 | 单独全边验证 |
| 协议诊断 | `exact-backoff` | 与 Impl 相同，只枚举真实 parent 并 backoff | 单独全边验证 |

这里必须保留一个解释边界：Floor 是一条 programmatic Graph 路径，而 Impl/Ceiling 是两条
independent priority streams。后两者虽在 producer 入口执行 programmatic trigger 指令，但
consumer 的 launch eligibility 不依赖该 trigger；它已经由独立 stream 提交。因而
Floor→Impl/Ceiling 的差值是本实验定义的**操作性包夹**，同时包含 launch topology、等待粒度、
flag 协议和调度交互，不能被改写成“只替换一条 wait 指令”的纯协议成本。

### 3.2 Producer、consumer 与计时负载

每个 producer CTA 的 thread 0 在 readiness spin 后只写一个 float，软件模式随后以
device-scope release store 发布 `done[cta]`，再运行独立 tail。每个 consumer CTA 先运行
prologue，在相应 dependency point 后立刻读取一个代表性 parent，再运行固定 epilogue。
计时内的 post-wait datum work 因而是 O(1)，不会随 degree 增长；对全部真实边的 O(degree)
检查放在另一轮不计时 validation invocation。

Tier 1.1 的固定 synthetic 参数为：

| 参数 | 数值 |
|---|---:|
| Threads/CTA | 128 |
| Producer ready base | 400K cycles |
| Readiness skew | 8 bins，即 400K..750K cycles |
| Producer tail | 1M cycles |
| Consumer prologue | 200K cycles |
| Consumer epilogue | 1M cycles |
| Consumer dynamic shared memory | 64 KiB/CTA，所有五个 mode 相同 |

64 KiB 不是实际 workload 的资源画像，而是为 resident-wait 多波构造选择的控制包络。每条正式
日志都报告相同的资源审计：`512/2048` mixed threads、`14592/65536` registers、
`196608/233472` shared-memory bytes、`4/32` blocks；查询 occupancy 为 producer 16 CTA/SM、
consumer 3 CTA/SM，并显式保留一个 producer slot。程序也在 64 KiB launch 前调用
`cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` 并实际 touch 该 shared memory。

### 3.3 依赖轴与结构轴

Tier 1.1 把两条轴分开：

- degree 轴固定 `structure=interval`，在各 grid 上扫可行的 1→1024 对数点；
- structure 轴以 `degree=32` 比较 `interval/grouped/strided/random`；`self` 按定义是
  degree 1 的语义控制点，**不是** degree 32；
- 同一 grid 的 `interval,d32` 物理点只运行一次，同时供两条轴使用，避免 Gate 双重加权；
- `random` 通过 affine permutation 产生互异 parent；分析结果的
  `all_unique_parents=true` 覆盖全部 93 个正式配置。

### 3.4 Correctness、poison 与多波准入

每次 timed invocation 都先 poison producer data 与 consumer output，并清 flag、error、stale
counter 和六组 timestamp。除 Ceiling 外，93 个配置 × 4 个应正确 mode 共 **372 次**独立
full-edge validation 都输出 PASS。Ceiling 不被验证为正确；相反，每条 summary 都要求
`ceiling_wrong=1`，即真实观察到 stale/poison 结果，防止“无等待”被调度偶然串行化成正确路径。

每个 CTA 用 `%globaltimer` 记录 producer `start/ready/end` 与 consumer
`start/dependency-satisfied/end`。对三个计划多波比率，原始 `SAMPLE` 的关键范围如下：

| Grid | 比率 | Tier 1.1 配置数 | 独立-stream：首个 consumer 启动时尚未启动 producer | Floor：首个 consumer 启动时未完成 producer | Ceiling stale outputs |
|---:|---:|---:|---:|---:|---:|
| 296 | 2× | 13 | 296 | 264..290 | 296 |
| 1184 | 8× | 15 | 1184 | 481..505 | 659..745 |
| 4736 | 32× | 15 | 4736 | 339..761 | 444..470 |

每个多波 summary 还同时满足 `producer_progress_complete=1`、
`multiwave_overlap_proven=1`、`floor_early_launch_proven=1` 和
`launch_gate=trace_verified`。也就是说，资源算术不是准入替代品：trace 实际看到 consumer
在 producer 尚未全部启动/完成时入场，并看到 producer 最终全部前进完成。CUDA 并不保证两个
独立 kernel 的公平调度；本程序用资源包络、10 秒 watchdog 和逐样本 trace 把本次观察
fail-closed，但这不是未来任意 kernel 的前进性保证。

### 3.5 `AGENTS.md` 有效性规则逐项核对

| 规则 | v2 证据 |
|---|---|
| Trigger 不得击穿 wait | Floor 在 datum ready 后 trigger；软件路径在入口 trigger，但 release flag 位于 data 写入和 CTA barrier 之后；93 条 summary 均自报三个 trigger 点 |
| 不用 global completion counter 代替 identity | 等待逐 parent flag；源文件中无 completion-count readiness 协议 |
| Timed post-wait payload 为 O(1) | 只读一个代表性 parent；全边检查另行执行 |
| Poison 与真实 correctness | 每次 timed invocation 重新 poison；372 次非 Ceiling 全边 validation PASS；Ceiling 必须 stale |
| 跨 SM 计时源 | 93 条 summary 均为 `timer=globaltimer`；headline `ms` 是两 kernel 完整时间线的 makespan |
| Degree / structure 正交 | 60 个 interval degree 点 + 28 个额外结构点；`interval,d32` 复用一次 |
| Floor 是 grid PDL | `floor_path=programmatic_graph`，consumer 执行 grid dependency wait |
| Ceiling 故意错误 | `ceiling_wrong=1` 为逐配置准入字段；只报告其时间 |
| Wait 路径对称 | 五档共用 128-thread producer/consumer kernel、64 KiB smem touch 和 dependency point 后 barrier；wait body 是受控差异 |
| Random parent 唯一 | `all_unique_parents=true`，Gate 也报告 unique parent sets 通过 |
| Launch hygiene | 64 KiB 显式 opt-in；资源包络审计并保留 producer slot |

## 4. 配置与统计方法

### 4.1 覆盖和样本账本

成功正式矩阵的账本为：

~~~text
Tier 1.1 = 60 interval degree configurations
           + 7 grids × 4 additional structure configurations
           = 88 configurations

Tier 1.2 = 5 tail/prologue configurations
All       = 88 + 5 = 93 configurations

Timed samples = 93 configurations × 5 modes × 31 repeats
              = 14,415 SAMPLE records
Per mode      = 93 × 31 = 2,883 SAMPLE records
~~~

expected manifest、实际 summary 均为 93，missing/unexpected 都为空。原始日志包含 93 条
`SUMMARY_PILOT`、14,415 条 `SAMPLE`，五个 mode 各 2,883 条；结果目录未生成
`failures.log`，campaign/session log 中也没有 `FAIL`，且 `failed_validation=[]`。本轮没有使用修复后预留的 timestamp retry：
`REJECTED_ATTEMPT=0`、`total_trace_retries=0`、所有配置
`trace_max_attempts_observed=1`。因此 v2 的统计没有删除或替换任何 timed attempt。

### 4.2 Warmup、配对顺序和中位数

每个配置、每个 mode 先运行 3 次 warmup，再运行 31 次 timed repeat。偶数 repeat 的
顺序是 Floor / headline Impl / Ceiling / 两个协议诊断；奇数 repeat 将五档整体反转。
因此三个 headline 在两个方向上始终相邻，同时平衡固定的热/顺序偏差。分析器要求
每个 mode 都恰好包含 rep 0..30，不能只保留幸存 repeat。

每个配置的 `floor_ms`、`ceiling_ms`、`impl_ms` 都是各自 31 次的中位数，然后计算：

~~~text
space_pct    = 100 × (median(Floor) - median(Ceiling)) / median(Floor)
captured_pct = 100 × (median(Floor) - median(Impl))    / median(Floor)
Impl         = interval-backoff
~~~

负 `captured_pct` 表示软件 Impl 比 Floor 慢，不表示 headroom 为负。

### 4.3 Paired bootstrap 95% CI

[`tools/analyze_pilot.py`](../../tools/analyze_pilot.py) 对每个配置做 5,000 次 deterministic
nonparametric bootstrap。一次 bootstrap 只抽一组 repeat id，并用相同 id 同时索引
Floor/Impl/Ceiling，保留同进程相邻测量的配对协方差；随后重新取三档中位数并重算派生百分比，
最后取 2.5% 与 97.5% 分位点。

这些 CI 是**单配置、单 session 的 repeat-level timer uncertainty**。下节按 grid 的 median/range
是配置之间的描述统计，不是 bootstrap CI；Gate 的 88 点 headline 也没有宣称跨配置或跨机器
置信区间。本轮只有一个确定性 seed，不能用这些窄 CI 替代跨 seed、跨日或跨设备复验。

## 5. 头条复算与完整结果地图

### 5.1 Gate headline 的逐项复算

Gate 只排序 88 个 `t11p_*` / `t11ps_*` 的逐配置 point estimate，排除五个 `t12p_*`。
88 为偶数，所以中位数取排序后第 44、45 个值的平均：

~~~text
space 中间两项（Floor / Ceiling / Impl 三档中位数）：
  t11p_g128_d16       = 1.405408 / 0.897984 / 0.903648 ms
  space_pct           = 100 × (1.405408 - 0.897984) / 1.405408
                      = 36.10510257519525%

  t11ps_g128_grouped  = 1.405056 / 0.897728 / 0.908192 ms
  space_pct           = 100 × (1.405056 - 0.897728) / 1.405056
                      = 36.107315295618115%

median space_pct
  = (36.10510257519525 + 36.107315295618115) / 2
  = 36.106208935406684%
  = 36.1062%  (4 decimals)

captured 中间两项（Floor / Ceiling / Impl 三档中位数）：
  t11p_g128_d64       = 1.405152 / 0.897952 / 0.915072 ms
  captured_pct        = 100 × (1.405152 - 0.915072) / 1.405152
                      = 34.87736558037849%

  t11p_g64_d64        = 1.404000 / 0.896224 / 0.913568 ms
  captured_pct        = 100 × (1.404000 - 0.913568) / 1.404000
                      = 34.931054131054125%

median captured_pct
  = (34.87736558037849 + 34.931054131054125) / 2
  = 34.9042098557163%
  = 34.9042%  (4 decimals)
~~~

headroom 最小值来自 32× `random,d32` 的 1.5764142958%，最大值来自 2×
`random,d32` 的 39.1731389135%，故报告 range 为 1.5764%..39.1731%。
[`gate.json`](../../bench/results_20260805_b200_multiwave_v2/gate.json) 以 36.1062% 对照
`GO >= 8%` 得出 `GO`；34.9042% 不进入这个比较。

### 5.2 按 grid 的配置中位数和范围

下表每一行都在该 grid 的全部 Tier 1.1 配置上计算；括号为最小值..最大值：

| Grid | P,C / SM | 配置数 | Space median（range） | Captured median（range） |
|---:|---:|---:|---:|---:|
| 32 | 0.216× | 10 | 36.1983%（36.1929..36.2428） | 35.8045%（35.3852..36.0178） |
| 64 | 0.432× | 11 | 36.1700%（36.1487..36.1928） | 35.7867%（34.8625..36.0449） |
| 128 | 0.865× | 12 | 36.1096%（36.0885..36.1523） | 35.5683%（33.5768..35.9596） |
| 148 | 1× | 12 | 36.0865%（36.0560..36.1569） | 35.4905%（33.4956..35.8889） |
| 296 | 2× | 13 | 39.1231%（39.0946..39.1731） | 38.2968%（33.9506..38.8239） |
| 1184 | 8× | 15 | 10.3165%（10.1279..10.3374） | -23.4101%（-211.2776..-4.8037） |
| 4736 | 32× | 15 | **4.5320%**（1.5764..4.6658） | **-5.6928%**（-370.9188..-0.6945） |

由于 degree 不能超过 producer 数，各行最大 degree 不同，所以这些行中位数不是严格的
matched-pair 统计。下面 degree 表的同一行、structure 表的同一结构才是跨 grid 的同配置对照。

1×→2× 没有观察到 headroom 收缩；首次测得的明显收缩在 8×，因为未测 4×，转折只能定位于
2×–8×，32× 又进一步下降。32× 的所有
software captured 都为负，最接近 Floor 的 `self,d1` 仍为 -0.6945%。这说明当前软件 resident
wait 构造在大 grid 上没有兑现剩余空间；它不改变 Gate 的权威 `GO`，但会改变下一阶段应优先
解释的区域。

### 5.3 Interval degree × grid 地图

每格为 `space_pct / captured_pct`（%）。该表固定 `structure=interval`、tightness=1.0；破折号
表示 degree 大于该 grid、未运行，不是失败。

| Degree | 32 | 64 | 128 | 148（1×） | 296（2×） | 1184（8×） | 4736（32×） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.2049 / 36.0178 | 36.1680 / 36.0449 | 36.1098 / 35.9596 | 36.0970 / 35.8807 | 39.1211 / 38.8224 | 10.3175 / -4.8314 | 4.5271 / -0.9102 |
| 2 | 36.2026 / 35.9950 | 36.1722 / 36.0195 | 36.1217 / 35.9304 | 36.0560 / 35.8351 | 39.1198 / 38.8059 | 10.3278 / -6.5206 | 4.4776 / -1.3640 |
| 4 | 36.1983 / 35.9815 | 36.1755 / 35.9841 | 36.1094 / 35.9090 | 36.0720 / 35.8101 | 39.1032 / 38.7503 | 10.3374 / -11.0717 | 4.5412 / -1.8525 |
| 8 | 36.1983 / 35.8582 | 36.1681 / 35.9174 | 36.0885 / 35.8561 | 36.0834 / 35.7374 | 39.1306 / 38.6717 | 10.3173 / -18.5486 | 4.5632 / -2.2551 |
| 16 | 36.1981 / 35.7508 | 36.1764 / 35.7867 | 36.1051 / 35.7021 | 36.0850 / 35.6250 | 39.1314 / 38.5576 | 10.3189 / -20.9163 | 4.5856 / -3.0542 |
| 32 | 36.1929 / 35.4124 | 36.1726 / 35.4660 | 36.1109 / 35.4345 | 36.1005 / 35.3560 | 39.1066 / 38.2968 | 10.3128 / -23.4101 | 4.5473 / -4.8363 |
| 64 | — | 36.1664 / 34.9311 | 36.0957 / 34.8774 | 36.0993 / 34.8359 | 39.1231 / 37.8045 | 10.3060 / -24.9588 | 4.5439 / -7.6208 |
| 128 | — | — | 36.1100 / 33.5891 | 36.0839 / 33.7949 | 39.1297 / 36.7699 | 10.2954 / -28.3791 | 4.4996 / -11.1285 |
| 256 | — | — | — | — | 39.1123 / 34.3770 | 10.3165 / -59.3097 | 4.4868 / -22.6019 |
| 512 | — | — | — | — | — | 10.3091 / -106.7370 | 4.5283 / -42.4253 |
| 1024 | — | — | — | — | — | 10.3165 / -194.6340 | 4.5320 / -83.3930 |

在固定 grid 内，Floor↔Ceiling 的 space 对 degree 几乎不敏感；这是合理的，因为这两档不执行
headline interval flag 扫描。软件 captured 却随 degree 增大而下降，8×/32× 上尤其明显。
因此“degree 大导致总收益空间消失”和“degree 大使当前软件协议成本上升”是两个不同命题；
本表只支持后者。

### 5.4 Structure 地图：1×、2×、8×、32×

`interval/grouped/strided/random` 的 requested degree 均为 32；`self` 是明确标出的 degree-1
例外。`eff` 是 interval cover 平均实际等待宽度，tightness 是逐 consumer 的
`true_degree / cover_width` 再取平均。

| Grid | Structure（degree） | Eff / tightness | Space | Captured |
|---:|---|---:|---:|---:|
| 148（1×） | self（1） | 1.00 / 1.0000 | 36.0915% | 35.8889% |
| 148（1×） | interval（32） | 32.00 / 1.0000 | 36.1005% | 35.3560% |
| 148（1×） | grouped（32） | 47.68 / 0.8941 | 36.0780% | 33.5732% |
| 148（1×） | strided（32） | 141.76 / 0.2264 | 36.0880% | 33.5027% |
| 148（1×） | random（32） | 137.41 / 0.2463 | 36.1569% | 33.4956% |
| 296（2×） | self（1） | 1.00 / 1.0000 | 39.0946% | 38.8239% |
| 296（2×） | interval（32） | 32.00 / 1.0000 | 39.1066% | 38.2968% |
| 296（2×） | grouped（32） | 39.14 / 0.9759 | 39.1402% | 34.3759% |
| 296（2×） | strided（32） | 287.54 / 0.1113 | 39.1261% | 34.0334% |
| 296（2×） | random（32） | 275.34 / 0.1278 | 39.1731% | 33.9506% |
| 1184（8×） | self（1） | 1.00 / 1.0000 | 10.3242% | -4.8037% |
| 1184（8×） | interval（32） | 32.00 / 1.0000 | 10.3128% | -23.4101% |
| 1184（8×） | grouped（32） | 32.00 / 1.0000 | 10.3159% | -21.8287% |
| 1184（8×） | strided（32） | 1148.00 / 0.0279 | 10.2241% | -211.2776% |
| 1184（8×） | random（32） | 1096.83 / 0.0331 | 10.1279% | -172.9978% |
| 4736（32×） | self（1） | 1.00 / 1.0000 | 4.6658% | -0.6945% |
| 4736（32×） | interval（32） | 32.00 / 1.0000 | 4.5473% | -4.8363% |
| 4736（32×） | grouped（32） | 32.00 / 1.0000 | 4.5891% | -5.6928% |
| 4736（32×） | strided（32） | 4589.00 / 0.0070 | 2.2151% | -370.9188% |
| 4736（32×） | random（32） | 4377.90 / 0.0093 | 1.5764% | -308.9478% |

1×/2× 时结构复杂度主要降低软件兑现比例，尚未消灭正收益；8×/32× 时，巨大的 conservative
cover 会把 resident software wait 成本放大数倍。`strided` 与 `random` 的相对次序并不稳定：
8× random 比 strided 少负，32× 则 strided 更负。可稳定表述的是两者都远差于 tight
interval/grouped，而不是宣称某一种 irregular shape 永远最差。

### 5.5 Tail / prologue 扫描（Tier 1.2，不进入 Gate）

固定 P=C=148、interval degree 8、prologue=200K cycles。`F/I/C` 是三档各自 31-repeat
中位数，方括号是 paired-bootstrap 95% CI：

| Tail/prologue | Tail cycles | F / I / C median（ms） | Space % [95% CI] | Captured % [95% CI] |
|---:|---:|---:|---:|---:|
| 1 | 200K | 0.996928 / 0.901152 / 0.614944 | 38.3161 [38.3042, 38.3349] | 9.6071 [9.5747, 9.6334] |
| 2 | 400K | 1.099392 / 0.902496 / 0.615712 | 43.9952 [43.9773, 44.0684] | 17.9095 [17.8687, 18.0161] |
| 4 | 800K | 1.303488 / 0.903296 / 0.796608 | 38.8864 [38.8703, 38.9374] | 30.7016 [30.6725, 30.7781] |
| 8 | 1.6M | 1.711808 / 1.206272 / 1.205120 | 29.5996 [29.5608, 29.6210] | 29.5323 [29.4993, 29.5648] |
| 16 | 3.2M | 2.526976 / 2.020352 / 2.019200 | 20.0942 [20.0841, 20.1137] | 20.0486 [20.0395, 20.0671] |

随着 tail/prologue 从 1 增到 16，软件兑现的空间比例从 25.07% 升至约 99.77%，但
`space_pct` 自身并不单调：分子是可重叠的绝对 makespan 差，分母 Floor 也随 tail 增长。
因此 tail 扫描支持“几何比例会改变软件兑现”，不支持“tail 越长相对 headroom 必然越大”。

## 6. 能成立的结论

1. 成功 v2 正式矩阵通过全部声明的准入条件；Gate 的权威判决是 `GO`，Tier 1.1 典型
   headroom 为 36.1062%，范围为 1.5764%..39.1731%。
2. §5.3 的 2×/8×/32× 不是 nominal grid 标签：逐 CTA `%globaltimer` 证明 consumer 入场时仍有
   producer 未启动，且所有 producer 最终完成，所以 `plan_multi_complete=true`。
3. 同一 B200 synthetic 构造的 headroom 有显著规模边界：按 grid 的配置中位数在
   1×/2× 约为 36%..39%，8× 为 10.3165%，32× 为 4.5320%。不能用一个总中位数替代这条曲线。
4. 当前 `interval-backoff` 在 1×/2× 大体兑现 headroom，在 8×/32× 的典型配置中反而慢于
   Graph Floor；32× median captured 为 -5.6928%。这足以否决“现有软件协议在任意 grid 都近似
   Ceiling”的说法。
5. 在 tight interval 内，总 headroom 基本不随 degree 变化，但软件成本随 degree 增长。这支持
   继续把边数与结构复杂度作为独立设计轴，而不是沿用一个“degree > 32 即无收益”的混合阈值。
6. `random`/`strided` 的 interval cover 在大 grid 上膨胀到数千 parent，并与极负
   captured 同时出现；当前软件表示需要 tightness-aware 的适用性判断。
7. 93 个配置没有 validation failure、manifest 缺口、重复数缺口、parent duplicate 或 trace
   retry；本报告不依赖筛选后的幸存样本。
8. Gate 按计划打开后续机制/真实负载验证预算，但大 grid 结果要求后续工作优先解释
   8×/32× 的调度与表示成本，而不是只复述总体 `GO`。

## 7. 不能成立的结论

1. **不能宣称真实 workload 获得 36.1062% speedup。** 这是 synthetic microbenchmark 中
   Floor→错误 Ceiling 的 latency headroom，中位数也不是 Floor/Impl 的 speedup。
2. **不能宣称 32× 仍有 36% headroom。** 32× 的逐 grid 中位数是 4.5320%，且 random 点只有
   1.5764%；总体中位数受 88 点配置构成影响。
3. **不能把负 captured 推广成“CTA-level hardware 无价值”。** 本轮没有 Ideal 点；软件
   interval polling、cover 扫描、64 KiB resident resource 和 stream 调度成本中，哪些可被硬件
   消除尚未隔离。
4. **不能把 Floor→Impl 差值归因于单一 wait protocol。** Floor 使用 programmatic Graph edge，
   Impl/Ceiling 使用 consumer-first 的独立优先级 streams，launch topology 并不相同。
5. **不能宣称 CUDA 对 resident waiter 与 producer 有公平性保证。** 本轮 trace 证明这 14,415
   个样本最终前进；资源 slot 与 watchdog 只保护本构造，不建立运行时契约。
6. **不能把 random/strided 的极负数外推到任意 irregular hardware encoding。** 当前 Impl 用
   conservative interval cover；exact adjacency、压缩 bitmap、硬件 scoreboard 或 pre-dispatch
   gating 可能有完全不同的成本。
7. **不能把 64 KiB/128-thread 资源包络当作生产 kernel。** 更高寄存器、不同 shared memory、
   tensor-core 与 memory traffic 都可能改变驻留和调度边界。
8. **不能从 fixed-cycle spin 推导真实 GEMM/Attention 的 overlap。** spin 没有生产算子的 cache、
   DRAM、tensor-core、barrier 或 warp-specialization 行为；本轮也只有一个 seed、一张 B200、一个
   driver/toolkit 环境。
9. **不能给出 profiler 级根因。** 正式机缺少 `nsys` 与 `ncu`；本报告有完整 `%globaltimer`
   准入轨迹，但没有 kernel dispatch timeline、L2/DRAM traffic、stall reason 或逐指令指标。
10. **不能报告 Ceiling correctness。** `none` 故意读取 stale/poison 数据；它只是一条 measured
    no-dependency timing reference，不是可部署配置，也不是数学意义的绝对硬件上界。
11. **不能给出硬件 Ideal−Impl 的价值。** 本实验没有执行 Ideal rung，也没有把 polling、wake-up、
    fence、表示与 scheduler 成本逐项注入一个硬件模型。
12. **不能复用首轮 `INVALID` session 的任何时间。** v2 是独立完整 session；两轮不能拼接、
    增样本或追溯性修补。

## 8. 证据入口

- 本报告唯一正式原始目录：[`bench/results_20260805_b200_multiwave_v2/`](../../bench/results_20260805_b200_multiwave_v2/)
- 权威 Gate：[`bench/results_20260805_b200_multiwave_v2/gate.json`](../../bench/results_20260805_b200_multiwave_v2/gate.json)
- 逐配置 JSON 与 paired bootstrap：[`bench/results_20260805_b200_multiwave_v2/pilot_analysis.json`](../../bench/results_20260805_b200_multiwave_v2/pilot_analysis.json)
- 逐配置 CSV：[`bench/results_20260805_b200_multiwave_v2/pilot_summary.csv`](../../bench/results_20260805_b200_multiwave_v2/pilot_summary.csv)
- 14,415 条逐 repeat 原始记录：[`bench/results_20260805_b200_multiwave_v2/pilot_matrix.log`](../../bench/results_20260805_b200_multiwave_v2/pilot_matrix.log)
- 93 点 expected manifest：[`bench/results_20260805_b200_multiwave_v2/pilot_expected_tags.txt`](../../bench/results_20260805_b200_multiwave_v2/pilot_expected_tags.txt)
- 实际设备记录：[`bench/results_20260805_b200_multiwave_v2/device.txt`](../../bench/results_20260805_b200_multiwave_v2/device.txt)
- 完整 session / preflight：[`bench/results_20260805_b200_multiwave_v2/session.log`](../../bench/results_20260805_b200_multiwave_v2/session.log)
- Benchmark 源码：[`bench/cta_dep_pilot.cu`](../../bench/cta_dep_pilot.cu)
- 唯一 parent 生成：[`bench/common/dep_pattern.cuh`](../../bench/common/dep_pattern.cuh)
- 分析器：[`tools/analyze_pilot.py`](../../tools/analyze_pilot.py)
- Gate 实现：[`tools/gate.py`](../../tools/gate.py)
- 实验规格与阈值：[`EXPERIMENT_PLAN.md`](../../EXPERIMENT_PLAN.md)
- 设计空间坐标：[`docs/cta_pdl_design_space.md`](../../docs/cta_pdl_design_space.md)
- Benchmark 有效性规则：[`AGENTS.md`](../../AGENTS.md)
- 首轮拒绝审计（禁止复用 timing）：[`reports/rejected/tier1_multiwave_trace_incomplete_20260805.md`](../rejected/tier1_multiwave_trace_incomplete_20260805.md)
- 本轮总报告：[`reports/campaign_b200_multiwave_20260805.md`](../campaign_b200_multiwave_20260805.md)
