# Tier 0.1：Same-Stream PDL 多级链实验报告

日期：2026-08-03（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0  
实验状态：**有效的合成时序/trace 实验；不是实际应用性能或通用并发深度结论**

## 1. 执行摘要

这个实验测试同一 CUDA stream 中长度为 1–6 的 kernel chain。每个 stage 都被拆成等长的独立 prologue 和可重叠 tail；PDL on 时，后继 stage 获得 programmatic launch eligibility，并在 dependency point 等待前驱 grid 完成。

主要结果：

- 1 个 stage 时 PDL on/off 都约 1.025 ms，没有可重叠的相邻 stage。
- 2–6 个 stage 的 speedup 从 1.3318× 增长到 1.7130×。
- 6-stage trace 的总 span 为 3.574048 ms，精确半开区间扫描得到峰值 296 个同时活跃 CTA，即最多两个 148-CTA grid 同时活跃。
- 第一对相邻 stage 的执行区间重叠 509.408 µs，为 producer span 的 49.928%，与“半个 stage prologue 与半个 stage tail 重叠”的构造一致。
- 日志中的 `implied_depth=5.969` 是由理想流水延迟公式反解出的模型参数，不表示 6 个 grid 同时驻留。

结论：**B200 当前环境中的 same-stream PDL 能把相邻 stage 的 prologue/tail overlap 沿多级链持续传递，但实测峰值只是两个 grid 同时活跃。**

## 2. 实验程序与问题

源码：[bench/tier0_facts.cu](../../bench/tier0_facts.cu)。  
trace 分析脚本：[tools/cta_timeline.py](../../tools/cta_timeline.py)。

实验回答两个不同的问题：

1. 链长从 1 增加到 6 时，端到端 latency 是否符合相邻 stage 流水模型？
2. CTA trace 中实际上同时存在多少个 grid/CTA？

这两个问题不能混为一谈。端到端时间可体现流水传播深度，但不能独立证明有同样数量的 grid 同时驻留。

## 3. Kernel 语义

每个 `chainK` stage 的逻辑为：

~~~text
记录 CTA launch 时间
    ↓
spin_cycles(work / 2)                 独立 prologue
    ↓
PDL on: cudaGridDependencySynchronize()
    ↓
记录 dependency point 时间
    ↓
线程 0 更新 buf[blockIdx.x]
    ↓
PDL on: cudaTriggerProgrammaticLaunchCompletion()
    ↓
spin_cycles(work / 2)                 可重叠 tail
    ↓
记录 CTA end 时间
~~~

PDL off 时使用普通 same-stream launch，不执行 programmatic wait/trigger，因此所有 stage 按 stream 顺序串行执行。

PDL on 时，每个 stage 使用 `cudaLaunchKernelEx` 和 `cudaLaunchAttributeProgrammaticStreamSerialization`。trigger 只提供后继 grid 的 launch eligibility；是否立即派发仍由 GPU 决定。

## 4. 实验配置

| 参数 | 数值 |
|---|---:|
| GPU | NVIDIA B200 |
| SM 数 | 148 |
| Grid | 148 CTA/stage |
| Threads/CTA | 128 |
| Stage work | 2,000,000 cycles |
| Prologue | 1,000,000 cycles |
| Tail | 1,000,000 cycles |
| Chain stages | 1、2、3、4、5、6 |
| 模式 | PDL off、PDL on |
| Warmup | 3 次/点 |
| Timed repeats | 5 次/点 |
| 汇总 | 5 次 CUDA event 时间的中位数 |

`148 CTA` 只表示 grid size 等于 SM 数，不保证固定的一 CTA 对一 SM 映射，也不等于按 occupancy 定义的完整驻留容量。

## 5. 计时与 trace

每次 timed invocation 在同一 non-blocking stream 上：

1. 清零数据与 trace buffer；
2. 记录 begin CUDA event；
3. 连续启动 n 个 stage；
4. 记录并同步 end event；
5. 计算两个 event 之间的 device elapsed time。

最后一个 `n=6、PDL on` timed run 的 CTA 时间戳被写入 CSV。每个 stage 有 148 条记录，共 888 条。

`t_dep_satisfied - t_launch` 包含固定 prologue 和可能的 dependency wait，不能解释成纯 dependency stall。

## 6. 完整 latency 结果

| Stages | PDL off ms | PDL on ms | Speedup | Implied depth |
|---:|---:|---:|---:|---:|
| 1 | 1.02502 | 1.02499 | 1.0000× | 1.000 |
| 2 | 2.04701 | 1.53702 | 1.3318× | 1.993 |
| 3 | 3.06893 | 2.04883 | 1.4979× | 2.983 |
| 4 | 4.09168 | 2.55904 | 1.5989× | 3.986 |
| 5 | 5.11261 | 3.06995 | 1.6654× | 4.977 |
| 6 | 6.13386 | 3.58077 | 1.7130× | 5.969 |

理想的相邻半阶段流水模型为：

~~~text
T_serial(n) ≈ n × T_stage
T_pipeline(n) ≈ (n + 1) / 2 × T_stage
ideal speedup ≈ 2n / (n + 1)
implied_depth = speedup / (2 - speedup)
~~~

实测数值紧贴这个构造模型。`implied_depth` 只是把 speedup 代回公式得到的等效链深度。

## 7. Chain-6 trace 结果

| Trace 指标 | 数值 |
|---|---:|
| Records | 888 |
| 总 span | 3.574048 ms |
| 峰值活跃 CTA | 296 |
| 单 stage 峰值 CTA | 148 |
| 第一对 stage 的 producer span | 1.020288 ms |
| Consumer span | 1.020256 ms |
| 两者重叠 | 0.509408 ms |
| 重叠 / producer span | 49.928% |
| 第一对相邻 stage 中，相同 block id 落在相同 SM | 148/148 |
| Consumer 使用的 SM 数 | 148 |
| 单 SM 最大 consumer CTA | 1 |

第一对相邻 stage 的相同 block id 在本次 148-CTA trace 中全部落到相同 SM。这是一个实测现象，不是 CUDA 的固定映射契约，不能外推到更大 grid、其它驱动或资源重型 kernel。

## 8. 能成立的结论

本实验支持：

1. 当前 B200 环境的 same-stream PDL 路径确实产生了相邻 stage overlap。
2. overlap 可以沿 6-stage chain 传递，使端到端 latency 接近半阶段流水模型。
3. 真实峰值活跃量为 296 CTA，而不是 6×148 CTA；“流水传递六级”不等于“六个 grid 同时驻留”。
4. 在这个固定 148-CTA、低资源 spin kernel 中，相邻 stage 重叠约半个 producer span。

## 9. 不能成立的结论

本实验不支持：

1. 任意 CUDA kernel chain 都能达到 1.71×。
2. 六个 grid 可以同时驻留。
3. CTA id 到 SM 的映射具有稳定 API 保证。
4. multi-wave、occupancy 1–2、真实 register/shared-memory 压力下仍有相同流水行为。
5. `t_dep-t_launch` 是纯 wait latency。
6. 真实 GEMM、Attention 或 LLM 链会获得相同收益。

程序没有对最终 `buf` 做独立的端到端 correctness validation；写操作主要用于保持代码路径和 trace。该实验的证据重点是 timing 与调度时间线。

## 10. 证据入口

- 源码：[bench/tier0_facts.cu](../../bench/tier0_facts.cu)
- 原始运行日志：[bench/results_budget1h_corrected/tier0_facts.log](../../bench/results_budget1h_corrected/tier0_facts.log)
- 原始 trace：[bench/results_budget1h_corrected/tier0_chain_trace.csv](../../bench/results_budget1h_corrected/tier0_chain_trace.csv)
- Trace 分析结果：[bench/results_budget1h_corrected/tier0_chain_timeline.json](../../bench/results_budget1h_corrected/tier0_chain_timeline.json)
- 分析脚本：[tools/cta_timeline.py](../../tools/cta_timeline.py)
- 总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)
