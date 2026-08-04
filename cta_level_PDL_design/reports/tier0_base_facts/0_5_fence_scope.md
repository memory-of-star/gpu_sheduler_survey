# Tier 0.5：CUDA Fence Scope 成本校准实验报告

日期：2026-08-03（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0  
实验状态：**有效的饱和循环 critical-path/throughput 校准；不是孤立单条 fence latency**

## 1. 执行摘要

实验在 148 CTA × 128 threads 的 kernel 中循环执行 2,000 次 none、CTA、GPU 或 system-scope fence，并用无 fence 路径作为 baseline。

结果：

| Scope | Median kernel ms | Baseline-adjusted ns/iteration |
|---|---:|---:|
| none | 0.01869 | 0.000 |
| CTA | 0.04957 | 15.440 |
| GPU | 1.42877 | 705.040 |
| SYS | 8.90672 | 4444.016 |

scope 增大时成本显著上升。但这些值来自大量线程并发执行的饱和循环，表示当前 kernel 的 baseline-adjusted critical-path/throughput 指标，不能直接当作一条孤立 fence 的硬件 latency。

## 2. 实验程序与指令

源码：[bench/tier0_facts.cu](../../bench/tier0_facts.cu)。

四条路径：

| Scope | CUDA 调用 |
|---|---|
| none | 不执行 fence |
| CTA | `__threadfence_block()` |
| GPU | `__threadfence()` |
| SYS | `__threadfence_system()` |

Kernel 内每个线程循环 2,000 次。每次循环由线程 0 写当前 CTA 的 buffer，所有线程随后执行所选 fence，再读取 buffer 累加，以防循环被完全删除。该 scaffolding 没有 CTA barrier，程序也不检查读取值；它不是跨线程通信正确性测试，只用于保持并计时 fence-heavy 代码路径。

## 3. 配置与统计

| 参数 | 数值 |
|---|---:|
| Grid | 148 CTA |
| Threads/CTA | 128 |
| Iterations/thread | 2,000 |
| Warmup | 3 次/scope |
| Timed repeats | 5 次/scope |
| 计时 | CUDA event |
| 汇总 | 5 次中位数 |

报告中的 `ns_per_fence` 计算为：

~~~text
ns_per_iteration(scope)
    = (median_ms(scope) - median_ms(none)) × 1,000,000 / 2,000
~~~

分母没有乘 CTA 数或线程数，因为这些线程在 GPU 上并发执行；该指标是 device makespan 的每迭代增量，不是把所有动态 fence 总数摊到单条指令的平均服务时间。

## 4. 结果复算

### 4.1 CTA scope

~~~text
(0.04957 - 0.01869) ms × 1e6 / 2000
= 15.440 ns/iteration
~~~

### 4.2 GPU scope

~~~text
(1.42877 - 0.01869) ms × 1e6 / 2000
= 705.040 ns/iteration
~~~

### 4.3 System scope

~~~text
(8.90672 - 0.01869) ms × 1e6 / 2000
= 4444.015 ns/iteration
~~~

日志按三位小数报告 4444.016 ns，差异来自未显示的原始浮点时间精度。

## 5. 如何解释

在当前饱和构造中：

- CTA-scope fence 的增量最小；
- device/GPU-scope 比 CTA-scope 高约两个数量级以内；
- system-scope 又显著高于 GPU-scope。

这说明为软件 CTA readiness 选择内存 scope 时，扩大 scope 可能非常昂贵。但 producer–consumer release/acquire atomic 与这个显式 fence 循环不是同一指令序列，不能把这里的数值直接加到 corrected pilot 或硬件 Ideal 模型。

## 6. 能成立的结论

本实验支持：

1. 当前 B200 上 fence scope 对饱和循环完成时间有巨大影响。
2. 在该构造中，CTA/GPU/SYS 的 baseline-adjusted 顺序为 15.44 ns、705.04 ns、4444.016 ns/iteration。
3. 系统级可见性不应在设计中被默认视为低成本。

## 7. 不能成立的结论

本实验不支持：

1. 一条孤立 `__threadfence()` 的 latency 就是 705 ns。
2. release/acquire atomic 的成本等于同 scope fence 成本。
3. 将这些值直接注入 CTA PDL Ideal 模型即可得到准确预测。
4. 不同 cache 状态、写集合、并发度和 GPU 上仍有相同数值。
5. 该 kernel 验证了某个跨 CTA 数据结构的功能正确性；它只是成本校准。

## 8. 证据入口

- 源码：[bench/tier0_facts.cu](../../bench/tier0_facts.cu)
- 原始日志：[bench/results_budget1h_corrected/tier0_facts.log](../../bench/results_budget1h_corrected/tier0_facts.log)
- 汇总行：[bench/results_budget1h_corrected/summary.txt](../../bench/results_budget1h_corrected/summary.txt)
- 总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)
