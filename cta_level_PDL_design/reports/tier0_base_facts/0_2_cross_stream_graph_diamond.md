# Tier 0.2：跨 Stream、CUDA Graph 与 Diamond PDL 实验报告

日期：2026-08-03（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0  
实验状态：**有效的路径对比实验；结论限定于当前 device/driver/toolkit 和合成负载**

## 1. 执行摘要

这个实验比较六条 producer→consumer 执行路径，并增加一个 diamond DAG：

- eager 跨 stream 普通 event；
- eager 跨 stream programmatic event；
- 将同样的双 stream programmatic-event 代码 capture 成 CUDA Graph；
- 直接构造 programmatic CUDA Graph edge；
- same-stream PDL；
- 无依赖并发 reference。

在 `tail=prologue=20M cycles` 的平衡点：

- 普通跨 stream event：20.383 ms；
- eager programmatic event：20.384 ms，约 1.00×，没有观察到提前 overlap；
- captured graph：10.193 ms；
- built programmatic graph edge：10.193 ms；
- same-stream PDL：10.196 ms；
- no-dependency reference：10.197 ms。

1M–80M cycles 的 7 点 sweep 中，built graph 与 same-stream 始终约为普通 event 的 2.00×，eager programmatic event 始终约为 1.00×。平衡 diamond 中 programmatic edges 也把 40.754 ms 降到 20.386 ms。

结论：**当前 B200 环境中可靠观察到 overlap 的是 same-stream 和 CUDA Graph 路径；eager 跨 stream programmatic event 没有观察到收益。**

## 2. 负载与执行结构

该程序是平衡 synthetic producer–consumer 负载。producer tail 和 consumer prologue 都由固定 cycles 的计算模拟，不是 GEMM、Attention 或真实数据处理。

双节点结构：

~~~text
Producer: [前置工作 / trigger] [独立 tail]
Consumer:          [独立 prologue] [dependency point / dependent work]
~~~

Diamond 结构：

~~~text
             ┌→ Mid A ┐
Producer ────┤        ├──→ Final
             └→ Mid B ┘
~~~

所有路径使用相同规模的合成阶段，变化的是 host-side dependency 表达方式。

## 3. 固定配置

| 参数 | 数值 |
|---|---:|
| GPU | NVIDIA B200 |
| SM 数 | 148 |
| Blocks/kernel | 148 |
| Threads/block | 128 |
| 元素数 | 18,944 |
| 详细点 tail | 20,000,000 cycles |
| 详细点 prologue | 20,000,000 cycles |
| 详细点 repeats | 50 |
| Sweep cycles | 1M、2M、5M、10M、20M、40M、80M |
| Diamond tail/prologue | 20M/20M cycles |
| Diamond repeats | 50 |

## 4. 六种路径

| 模式 | Dependency 表达 | 角色 |
|---|---|---|
| BASE | 两条 stream，ordinary event | 串行 reference |
| PDL_XS | 两条 stream，eager programmatic event | 测 eager cross-stream PDL |
| PDL_CAPTURE | 同样双 stream 代码 capture 为 graph | 测 capture 是否改变路径 |
| PDL_GRAPH | 显式 graph nodes + programmatic edge | 直接 Graph PDL |
| PDL_SS | same stream + programmatic serialization | same-stream PDL |
| CONC | 无 dependency 的并发执行 | unsafe timing reference |

`CONC` 只用于时间参照，不提供依赖正确性。

## 5. 详细点结果

| 模式 | Median ms | Min ms | vs BASE | 日志 correctness |
|---|---:|---:|---:|---|
| BASE ordinary event | 20.383 | 20.380 | 1.00× | PASS |
| PDL_XS eager programmatic event | 20.384 | 20.379 | 1.00× | PASS |
| PDL_CAPTURE captured graph | 10.193 | 10.192 | 2.00× | PASS |
| PDL_GRAPH built programmatic edge | 10.193 | 10.191 | 2.00× | PASS |
| PDL_SS same-stream | 10.196 | 10.193 | 2.00× | PASS |
| CONC no dependency | 10.197 | 10.194 | 2.00× | n/a |

PDL_CAPTURE 与 PDL_GRAPH 几乎相同，说明当前环境下决定行为的关键是进入 CUDA Graph programmatic dependency 路径，而不是必须手工构造 graph。

## 6. 1M–80M cycles sweep

| Cycles | BASE ms | PDL_XS ms | PDL_GRAPH ms | PDL_SS ms | CONC ms | Graph speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1M | 1.033 | 1.033 | 0.516 | 0.518 | 0.520 | 2.00× |
| 2M | 2.049 | 2.050 | 1.026 | 1.029 | 1.029 | 2.00× |
| 5M | 5.107 | 5.107 | 2.553 | 2.557 | 2.558 | 2.00× |
| 10M | 10.199 | 10.199 | 5.101 | 5.104 | 5.104 | 2.00× |
| 20M | 20.383 | 20.384 | 10.195 | 10.197 | 10.198 | 2.00× |
| 40M | 40.752 | 40.753 | 20.382 | 20.384 | 20.385 | 2.00× |
| 80M | 81.497 | 81.496 | 40.756 | 40.760 | 40.762 | 2.00× |

eager `PDL_XS` 与 BASE 在所有点都相同；`PDL_GRAPH`、`PDL_SS` 与无依赖 reference 在所有点都接近一半时间。

## 7. Diamond 结果

| Variant | Median ms | Min ms | Speedup | Correctness |
|---|---:|---:|---:|---|
| DIAMOND_PLAIN ordinary edges | 40.754 | 40.748 | 1.00× | PASS |
| DIAMOND_PDL programmatic edges | 20.386 | 20.382 | 2.00× | PASS |

该结果显示 programmatic graph edge 不只在单 producer→consumer 边上观察到收益；在这个完全平衡、低资源的四节点 diamond 中，timing 也与预期流水一致。

## 8. Correctness 证据边界

日志为五条有依赖路径报告 PASS，但该跨 stream 程序的验证弱于 corrected producer–consumer pilot：

- 使用固定输入/固定期望值检查末尾输出；
- buffer 没有每轮 epoch 化或 poison；
- 未保存逐 CTA ready/wait-return trace；
- `CONC` 按设计不验证。

因此性能路径差异证据较强，但防止“陈旧正确值”掩盖竞态的能力较弱，不能把其 correctness 等级等同于 corrected pilot。

## 9. 能成立的结论

本实验支持：

1. 当前 B200/driver/toolkit 下 eager cross-stream programmatic event 没有比 ordinary event 更早完成。
2. same-stream PDL、captured graph 和显式 programmatic graph edge 都在平衡 synthetic 负载上接近 2.00×。
3. 将同一双 stream 代码 capture 为 graph 足以获得与手工 graph edge 相同的 timing。
4. programmatic graph edge 在平衡 diamond 中也观察到 2.00×。

## 10. 不能成立的结论

本实验不支持：

1. eager cross-stream PDL 在所有驱动/设备上都无效。
2. 任意不平衡 DAG 都能获得 2×。
3. Graph PDL 一定等同于无依赖并发；这里恰好是对称、低资源构造。
4. 真实应用、资源竞争或 multi-wave 下仍有相同结果。
5. 所有 PASS 都具有 corrected pilot 同等级的防陈旧值保证。

## 11. 复现与证据限制

本目录保存了完整运行日志，但执行该实验的原始 `pdl_bench` 源码位于实验时的外部 sibling 工程，没有随本目录一起保存。因此本报告可以精确复核结果和配置，但源码级独立复现材料不如其它 Tier 0 实验完整。

## 12. 证据入口

- 原始运行日志：[bench/results_budget1h_corrected/tier0_xstream.log](../../bench/results_budget1h_corrected/tier0_xstream.log)
- B200 总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)
- PDL/Graph 接口笔记：[docs/cuda_13.4_pdl_clc_interfaces.md](../../docs/cuda_13.4_pdl_clc_interfaces.md)
