# Tier 0.3：Resident Waiting CTA 对 Productive Background 的配对定价

报告日期：2026-08-05（UTC）  
实验日期：2026-08-05（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0，183,359 MiB  
证据等级：**B200 上有效的 synthetic productive-background 配对机制定价；支持当前 harness 坐标的 B2 成本结论，不支持真实 GEMM/LLM 或 `[H+]` 硬件收益声称**

## 1. 执行摘要

正式矩阵覆盖 `low/mid/high` 三档寄存器模板 × `0/8/16/32/64 KiB` dynamic shared memory，共15 个资源点。每点丢弃 3 组 warmup，保留31 组相邻配对重复；每组各含 `deferred_gate` 与 `resident_wait` 一个样本，因而正式计时样本数为 `15 × 31 × 2 = 930`。

在这个单 CTA producer、`8×SM` nominal grid（1,184 CTA）LCG background 与满容量 waiter grid 构造下，相对于 `deferred_gate`，`resident_wait` 造成的配对 background 吞吐损失在15 个资源点的中位数取值范围为 **16.7280%–53.0463%**，配对端到端增量的对应范围为 **1.817696–10.222560 ms**。这是跨资源点的 point-estimate envelope，不是联合置信区间。两个最小值均在 `low / 32 KiB`，两个最大值均在 `low / 64 KiB`。

15 份每配置保留 trace 的语义复核结果是：

- `deferred_gate` 的全部 waiter 都满足 `wait_enter >= producer_end`，提前进入数为 0；
- `resident_wait` 的15 个点均有 `early_waiters > 0`，合计16,639 / 16,724 个 waiter 在保留 trace 中提前进入；
- 所有提前进入的 waiter 都满足 `wait_exit >= producer_end`，违规退出数为 0。

这里的 `deferred_gate` 是用普通 same-stream 顺序派发构造的软件对照，**不是生产 Floor，也不是已实现的硬件 pre-dispatch gate**。

## 2. 程序实际执行了什么

源码是 [bench/tier0_background.cu](../../bench/tier0_background.cu)。每次运行有三条执行链：

1. dependency stream 先启动单 CTA producer。producer 在 kernel entry 调用 `cudaTriggerProgrammaticLaunchCompletion()`，然后执行 4,000,000 cycles 的 readiness work，最后写入 producer 数据。
2. 同一 dependency stream 上紧接着启动 `dependencyWaiter`。每个 waiter CTA 的 leader 在 wait 前把编译期寄存器模板读入 live state，可选触碰 dynamic smem，然后调用 `cudaGridDependencySynchronize()`；wait 后通过 CTA barrier 传播可见性并计算、写回每个 waiter 输出。
3. 独立 background stream 启动1,184 CTA × 128 threads，grid ratio 为 nominal `8×SM`。每个线程串行执行1,000,000 次 LCG update，总工作量为151,552,000,000 updates；host 用 jump-ahead oracle 校验每个线程的最终输出。`bg_peak_ctas=1184` 表示所有 CTA 的活动区间曾同时重叠，不能把该 ratio 改称为 8 个实际调度 wave。

两种模式使用相同的 producer、waiter kernel、waiter grid、寄存器模板、dynamic smem、参数形状、控制流、background work、输入 poison 和全输出校验。为防止陈旧值，pair 中两运行的 epoch 刻意不同，但不改变工作量或控制路径。两边 waiter 也都走 `cudaLaunchKernelEx`。唯一 launch 语义差异是：

| 模式 | waiter launch |
|---|---|
| `deferred_gate` | 不加 PSS attribute；普通 same-stream ordering 使 waiter 在 producer 退休后才能入场 |
| `resident_wait` | 只给同一 waiter launch 加 `cudaLaunchAttributeProgrammaticStreamSerialization`，允许 CTA 在 ready 前驻留并执行同一 wait |

每次重复之前，producer、waiter 和 background 输出均以 `0xa5` poison；计时后逐元素校验所有 producer、所有 waiter 和所有 background 输出。任一输出、timestamp 或 wait 语义校验失败都会使进程非零退出；本v2矩阵的930 个计时样本全部 `valid=1`。

## 3. 配置与统计

### 3.1 资源模板

| 档位 | requested register words | `__launch_bounds__` min blocks/SM | 实际 registers/thread | local bytes/thread |
|---|---:|---:|---:|---:|
| low | 8 | 8 | 26 | 0 |
| mid | 40 | 4 | 54 | 0 |
| high | 80 | 2 | 92 | 0 |

实际寄存器数与 local bytes 来自 `cudaFuncGetAttributes`，档位名不作为证据。所有点的 static smem 均为0；dynamic smem 是表中扫描的0/8/16/32/64 KiB。64 KiB 在 occupancy query 和 launch 前显式 opt in，0 KiB 路径不触碰 dynamic smem。

### 3.2 采样与计时

| 参数 | 正式值 |
|---|---:|
| 资源点 | 15 |
| Warmup | 3 组 pair/点 |
| Timed repeats | 31 组 pair/点 |
| Timed samples | 930 |
| Threads/CTA | 128 |
| Background | 1,184 CTA，151,552 threads，nominal `8×SM` grid |
| LCG iterations/thread | 1,000,000 |
| Producer | 1 CTA，entry trigger 后4,000,000 cycles |
| 计时源 | 逐 CTA `%globaltimer` |
| 中心统计量 | 31 个配对值的中位数 |
| 95% CI | 固定 seed、2,000 次 bootstrap median，2.5%/97.5% 分位 |

每个 repeat 内两档相邻运行，奇数 repeat 先跑 `resident_wait`，偶数 repeat 先跑 `deferred_gate`，用交替顺序降低时钟与温度漂移的定向影响。background 有效时间从 producer/background 两者最早活动时刻起，到 background 完成时刻止，因此包含 waiter 引起的 dispatch delay；吞吐量为：

~~~text
background_gupdates_s
    = 151,552,000,000 updates / (bg_end - min(producer_start, bg_start))
~~~

端到端时间的终点还包括 waiter end。吞吐损失与端到端增量都是先在每个 repeat 内配对求差，再对31 个派生值取中位数并 bootstrap；它们不是两个独立中位数相减。

## 4. 完整15 点结果

方括号内为95% bootstrap CI。`early/total` 是 `resident_wait` 提前进入 wait 的 CTA 数中位数/当点 waiter grid；15 个 `deferred_gate` 点的 early 数均为0。

| 档位（regs） | Smem | Occupancy API CTA/SM | Waiter CTA | Resident early/total | Control Gupdates/s [95% CI] | Resident Gupdates/s [95% CI] | 配对吞吐损失 % [95% CI] | 配对 e2e 增量 ms [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low (26) | 0 KiB | 16 | 2,368 | 2,367/2,368 | 16750.667586 [16743.442800, 16769.231859] | 9255.730599 [9254.844333, 9256.309477] | 44.7482 [44.7339, 44.8048] | 7.326688 [7.324928, 7.336352] |
| low (26) | 8 KiB | 16 | 2,368 | 2,352/2,368 | 16747.765079 [16745.633265, 16772.082416] | 9254.627314 [9253.036152, 9255.187967] | 44.7558 [44.7301, 44.8326] | 7.330528 [7.324128, 7.340704] |
| low (26) | 16 KiB | 13 | 1,924 | 1,911/1,924 | 16745.574056 [16739.477740, 16759.736997] | 8840.109008 [8829.412982, 8843.426903] | 47.2413 [47.1800, 47.3025] | 8.093248 [8.089248, 8.116928] |
| low (26) | 32 KiB | 6 | 888 | 882/888 | 16751.674814 [16749.127355, 16755.408537] | 13957.655491 [13951.364647, 13960.617852] | **16.7280 [16.6469, 16.7908]** | **1.817696 [1.807936, 1.823424]** |
| low (26) | 64 KiB | 3 | 444 | 441/444 | 16753.748899 [16749.601242, 16769.350613] | 7867.253388 [7866.443208, 7867.867668] | **53.0463 [53.0382, 53.0871]** | **10.222560 [10.219616, 10.226880]** |
| mid (54) | 0 KiB | 9 | 1,332 | 1,331/1,332 | 16742.259003 [16737.584642, 16750.252881] | 9252.132327 [9249.512233, 9252.891529] | 44.7463 [44.7322, 44.7792] | 7.330176 [7.327584, 7.336928] |
| mid (54) | 8 KiB | 9 | 1,332 | 1,323/1,332 | 16749.008887 [16744.449159, 16765.729376] | 8828.096311 [8824.082890, 8832.459288] | 47.3321 [47.2829, 47.3782] | 8.122624 [8.116352, 8.130784] |
| mid (54) | 16 KiB | 9 | 1,332 | 1,323/1,332 | 16747.291295 [16742.554937, 16771.547862] | 8242.327204 [8235.920159, 8824.378837] | 50.7600 [47.5762, 50.8496] | 9.330848 [8.169952, 9.360800] |
| mid (54) | 32 KiB | 6 | 888 | 882/888 | 16749.482768 [16747.765079, 16755.230703] | 13951.035870 [13946.968537, 13954.571035] | 16.7312 [16.7153, 16.8098] | 1.817728 [1.816256, 1.825408] |
| mid (54) | 64 KiB | 3 | 444 | 441/444 | 16748.949654 [16745.633265, 16775.528132] | 7867.776174 [7866.586938, 7869.188041] | 53.0432 [53.0241, 53.0945] | 10.218848 [10.213280, 10.224096] |
| high (92) | 0 KiB | 5 | 740 | 740/740 | 16750.726832 [16743.501994, 16776.300642] | 9859.087493 [9856.481636, 9859.703251] | 41.1777 [41.1368, 41.2390] | 6.331328 [6.323456, 6.340256] |
| high (92) | 8 KiB | 5 | 740 | 735/740 | 16753.689632 [16747.824303, 16776.954359] | 8809.376331 [8805.838330, 8817.265158] | 47.4168 [47.3898, 47.4986] | 8.161216 [8.144384, 8.166912] |
| high (92) | 16 KiB | 5 | 740 | 735/740 | 16761.101221 [16748.120434, 16780.580447] | 8812.605599 [8805.085234, 8818.283042] | 47.4435 [47.4163, 47.4671] | 8.161792 [8.146720, 8.172736] |
| high (92) | 32 KiB | 5 | 740 | 735/740 | 16754.519404 [16747.350517, 16778.440271] | 8812.736786 [8807.148370, 8817.199497] | 47.4365 [47.3760, 47.4868] | 8.155328 [8.146976, 8.169824] |
| high (92) | 64 KiB | 3 | 444 | 441/444 | 16749.719718 [16743.857168, 16763.949014] | 7868.377463 [7867.292595, 7870.757378] | 53.0288 [53.0081, 53.0960] | 10.215104 [10.208480, 10.223040] |

`mid / 16 KiB` 的 resident 吞吐 CI（8235.920159–8824.378837 Gupdates/s）、配对损失 CI（47.5762%–50.8496%）与 e2e 增量 CI（8.169952–9.360800 ms）明显宽于其它多数点。原始31 点呈现离散的双峰调度状态：约20 点低于8.3k、11 点高于8.8k Gupdates/s，使 bootstrap median CI 跨越两簇。因此本报告保留该离散性，不把资源扫描硬解释为单调曲线；这不影响全矩阵最小/最大中位数的定位。

## 5. 头条数字复算

### 5.1 配对量的定义

对每个资源点的 repeat `i`：

~~~text
loss_i = 100 × (control_gupdates_i - resident_gupdates_i)
                   / control_gupdates_i

delta_i = resident_e2e_i - control_e2e_i

headline_loss  = median(loss_0 ... loss_30)
headline_delta = median(delta_0 ... delta_30)
~~~

对 `loss_i` 与 `delta_i` 各自保持整31 个 pair，然后分别做2,000 次 bootstrap median。

### 5.2 最小缺口：`low / 32 KiB`

吞吐损失中位数落在 repeat 2 的配对值：

~~~text
100 × (16748.890421 - 13947.132828) / 16748.890421
= 16.7280191259%
→ 16.7280%  [bootstrap 95% CI: 16.6469%, 16.7908%]
~~~

同一 repeat 2 也是e2e 增量中位值：

~~~text
10.866176 ms - 9.048480 ms
= 1.817696 ms
→ 1.817696 ms  [bootstrap 95% CI: 1.807936, 1.823424 ms]
~~~

这不是独立中位数相减。如果错用表4的独立吞吐中位数，会得到：

~~~text
100 × (16751.674814 - 13957.655491) / 16751.674814
= 16.6790447%  != 16.7280%

10.857984 ms - 9.046976 ms
= 1.811008 ms  != 1.817696 ms
~~~

### 5.3 最大缺口：`low / 64 KiB`

吞吐损失中位数落在 repeat 3：

~~~text
100 × (16747.468961 - 7863.556665) / 16747.468961
= 53.0462980208%
→ 53.0463%  [bootstrap 95% CI: 53.0382%, 53.0871%]
~~~

e2e 增量中位数落在 repeat 2：

~~~text
19.271744 ms - 9.049184 ms
= 10.222560 ms
→ 10.222560 ms  [bootstrap 95% CI: 10.219616, 10.226880 ms]
~~~

因此，15 个配置中位数的取值范围为：

~~~text
paired throughput loss: 16.7280% .. 53.0463%
paired e2e delta:         1.817696 .. 10.222560 ms
formal timed samples:     15 × 31 pairs × 2 modes = 930
~~~

## 6. 能成立的结论

1. 在本 B200 synthetic 构造中，允许 dependent CTA 在 ready 前驻留并等待，会将 productive LCG background 吞吐压低16.7280%–53.0463%，并将包含同一 waiter 工作的复合e2e 时间增加1.817696–10.222560 ms；范围随表4的具体资源点变化。
2. 差异不是由删掉 control 的 waiter 工作得到的；两边都执行同一 waiter grid、同一 wait 和全输出校验，唯一 launch 语义差异是 PSS attribute。
3. 保留的15 份 `%globaltimer` trace 证明对照与实验两档确实分别是 deferred entry 和 real resident wait：control early 为0，resident 每点 early 大于0，early waiter 提前退出违规为0。
4. 当前编译产物的实际资源档为26/54/92 registers/thread，local bytes 均为0；occupancy API 上限为3–16 CTA/SM，对应全卡444–2,368 个 waiter CTA。
5. B2 的“能容纳多少 waiting CTA”与“这些 CTA 对有用工作的系统代价”必须同时报告；只看 occupancy 容量不能推出吞吐代价。

## 7. 不能成立的结论

### 7.1 外推与归因边界

本实验不支持：

1. 把 `deferred_gate` 称为生产 Floor。它是普通 same-stream 顺序派发构造的软件对照，不是真实负载中已开启的 grid-level PDL；本实验是 Tier 0 B2 定价，不是 Tier 1 `Ceiling - Floor` 收益地图。
2. 把16.7280%–53.0463% 解释为 GEMM、Attention、LLM 或生产 inference 的收益/损失。background 是1,184 CTA 的 nominal `8×SM` LCG grid，producer 只有1 个 CTA，而 waiter grid 按 occupancy API 上限铺满；这些都不是真实工作负载分布。
3. 把 occupancy API 返回的3–16 CTA/SM 称为运行时实际达成的每 SM 常驻量。它是资源上限；trace 证明了全卡有 early resident waiter，但没有证明每个 SM 同时达到 occupancy 上限。
4. 把档位差异归因为“纯寄存器数”效应。编译期 live-across-wait 模板还在 wait 后执行 checksum，因此寄存器压力与 checksum 指令工作量混杂。
5. 声称 smem 或 register 扫描具有严格单调关系。尤其 `mid / 16 KiB` 的置信区间较宽，不允许把单个中位数当作稳定单调趋势。
6. 将观测差值全部归因为“占 CTA slot”一项微观原因。这是整机级差值，可同时包含 dispatch、warp scheduler、cache 和带宽互作用。
7. 外推到 B300、Rubin 或其它 GPU。本报告只测了 B200 CC 10.0；它没有测量 Rubin，也不从公开的 Rubin 描述推断其实现。

### 7.2 `[H+]` pre-dispatch 的条件化 scenario envelope

当前 B200 上没有实现本项目所枚举的 `[H+]` pre-dispatch gating，因此本实验不能给硬件点估计，只能构造明示假设的 scenario envelope。对表4每个资源点 `r`，令 `L_r` 为已测的配对吞吐损失，`D_r` 为已测的配对 e2e 增量。只在下列工程单调性假设成立时，envelope 才是：

~~~text
resident throughput ≤ hypothetical [H+] throughput ≤ deferred throughput
deferred e2e       ≤ hypothetical [H+] e2e       ≤ resident e2e
~~~

在该条件下：

~~~text
[H+] 相对 resident 可回收的吞吐缺口_r（deferred=100） ∈ [0, L_r] percentage points
[H+] 相对 resident 可节省的 e2e 时间_r                    ∈ [0, D_r] ms
~~~

在当前15 个坐标上：

- 吞吐缺口逐点为 **`[0,L_r]` 个百分点**；15 点的上界取值范围为 **16.73–53.05 个百分点**；
- e2e 节省逐点为 **`[0,D_r]` ms**；15 点的上界取值范围为 **1.8177–10.2226 ms**。

上界取当点全部 resident-vs-deferred 差值，相当于假设 `[H+]` 可完全避免 ready 前常驻的系统代价。`deferred_gate` 只是软件 proxy，它依靠普通 same-stream 顺序把waiter 整体延到 producer 退休后，不是真实 `[H+]` 硬件，也没有实现 tile 级的精确 admission。如果上述单调性不成立，硬件 bookkeeping 可使回收为负，而更精确的 admission 也可能超过 deferred proxy；因此 `[0,L_r]` / `[0,D_r]` 不是无条件硬件保证。

## 8. 证据入口

- 源码：[bench/tier0_background.cu](../../bench/tier0_background.cu)
- Harness 契约：[bench/README.md](../../bench/README.md)
- 驱动：[bench/run_all.sh](../../bench/run_all.sh)
- 严格验证器：[tools/validate_tier0_background.py](../../tools/validate_tier0_background.py)；对本正式目录返回 `PASS`，15 configs / 465 paired repetitions / 930 samples / 15 trace files / 68,998 trace data rows / 0 errors
- 正式设备记录：[bench/results_20260805_b200_multiwave_v2/device.txt](../../bench/results_20260805_b200_multiwave_v2/device.txt)
- 派生分析：[bench/results_20260805_b200_multiwave_v2/analysis_tier0.json](../../bench/results_20260805_b200_multiwave_v2/analysis_tier0.json)
- 严格验证结果：[bench/results_20260805_b200_multiwave_v2/tier0_background_validation.json](../../bench/results_20260805_b200_multiwave_v2/tier0_background_validation.json)
- 15 条汇总行：[bench/results_20260805_b200_multiwave_v2/summary.txt](../../bench/results_20260805_b200_multiwave_v2/summary.txt)
- 15 份原始样本日志与15 份配对 trace：[bench/results_20260805_b200_multiwave_v2/](../../bench/results_20260805_b200_multiwave_v2/)（`tier0_bg_{low,mid,high}_smem{0,8,16,32,64}.log` 与同 stem 的 `_trace.csv`）
- 正式 session 日志：[bench/results_20260805_b200_multiwave_v2/session.log](../../bench/results_20260805_b200_multiwave_v2/session.log)
- 本次 campaign 总报告：[reports/campaign_b200_multiwave_20260805.md](../campaign_b200_multiwave_20260805.md)

本报告中的所有 timing 数字只来自上述唯一成功正式目录 `bench/results_20260805_b200_multiwave_v2/`。首轮 `INVALID` campaign 的 timing 没有进入本报告的任何表格、复算或结论。
