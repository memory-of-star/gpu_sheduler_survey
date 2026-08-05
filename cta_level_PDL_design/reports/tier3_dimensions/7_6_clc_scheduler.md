# Tier 3：CLC 持久化 kernel 调度策略（§7.6）

| 项 | 值 |
|---|---|
| 报告 / 实验日期 | 2026-08-05（UTC） |
| 设备 | NVIDIA B200，148 SM，CC 10.0；Driver 580.126.09，CUDA 13.0 |
| 正式结果 | `bench/results_20260805_b200_tier23_native_v2/` |
| 证据等级 | **A-：真实 CLC PTX 原语、pending clusters、token conservation、全任务校验与 31-repeat trace** |

## 1. 执行摘要

在 4,096 producer + 4,096 consumer task 的单 persistent-kernel 复现中，
producer-priority 为 0.372992 ms；readiness-aware consumer-priority 为 2.995552 ms
（8.03×），locality-priority 为 3.417824 ms（9.16×）。三种安全策略的 CLC
success rate 都是 5,824/8,192=0.7109375，因而差距来自拿到 token 后的软件任务选择与
等待行为，不是 cancel 成功率变化。locality 策略命中 3,342 次，却仍比 consumer-priority
慢 14.10%。

| policy | median ms | CLC success | wait acquire loads | locality hits | cycles/attempt | attempts/ms |
|---|---:|---:|---:|---:|---:|---:|
| producer-priority | 0.372992 | 0.7109375 | 4,115 | 0 | 1,960.116 | 21,962.94 |
| consumer-priority | 2.995552 | 0.7109375 | 4,096 | 0 | 2,009.614 | 2,734.72 |
| locality | 3.417824 | 0.7109375 | 4,096 | 3,342 | 2,009.074 | 2,396.85 |
| unsafe none | 0.366688 | 0.7109375 | 0 | 0 | 1,957.650 | 22,340.52 |

## 2. 程序实际做了什么

[`bench/tier23_clc_scheduler.cu`](../../bench/tier23_clc_scheduler.cu) 发射 8,192 个一-CTA
cluster，而 occupancy capacity 只有 `16×148=2,368`，确保存在 pending clusters。每个已执行
CTA 循环调用 `clusterlaunchcontrol.try_cancel.async`，直到第一次失败；单 CTA 可成功取消多个
pending cluster。5,824 次成功加 2,368 个执行 CTA 各自的终止失败，构成 8,192 次 attempt。
原 launch token 加取消得到的 token 都交给同一 persistent kernel 内的软件 scheduler。validator 对每个
原始 cluster 强制 `executed + canceled = 1`，并要求恰好处理 8,192 个 task token。

三策略只改变 task 领取顺序：producer-priority 先生产；consumer-priority 优先领取已经 ready
的 consumer，否则补 producer；locality 优先领取当前 CTA 刚生产 tile 的 consumer。
producer 在数据写完后 release-store epoch，安全 consumer acquire 后读取；`none` 不等 ready，
4,096 个 consumer 全部读错，用作 unsafe Ceiling。

## 3. 配置与统计

* 4,096 tiles、8,192 launch clusters、128 threads、occupancy=16 blocks/SM。
* producer/consumer 各 100,000 spin cycles；3 warmups、31 timed repeats、四档相邻反序。
* 124 samples、65,540 trace rows；每档 2,000 次 bootstrap median CI。
* 每次 validation 检查全部 producer/consumer、digest、epoch RAW 时序、task 数和 token
  conservation；0 coverage errors。
* CLC cycles 用 `clock64` 计单 CTA 的原语片段；kernel makespan 与跨 SM trace 用
  `%globaltimer`。
* 同正式 binary SHA 的 non-timing Compute Sanitizer 覆盖四档并报
  `ERROR SUMMARY: 0 errors`。

## 4. 头条数字复算

```text
CLC success rate = 5,824 / 8,192 = 0.7109375

consumer / producer = 2.995552 / 0.372992 = 8.0311×
locality / producer = 3.417824 / 0.372992 = 9.1633×
locality 相对 consumer 增加 = (3.417824 - 2.995552) / 2.995552 = 14.0966%

producer 相对 unsafe none 开销 =
  (0.372992 - 0.366688) / 0.366688 = 1.7192%
```

## 5. 可以成立的结论

1. B200 的 CLC 原语足以承载一个 token-conserving persistent scheduler；这不再只是
   `try_cancel` 可编译的可行性探针。
2. 对本轮 1-to-1 ready 关系，producer-priority 明显更快。consumer-priority 只 claim 已 ready
   的 consumer；没有 ready consumer 时会先补 producer，并可能在 scheduler 选择循环中等待，
   因而不能把差异描述成“未 ready consumer 过早领 token”。
3. locality hit 不是充分目标：locality 策略更慢 14.10%，但本实验没有把 readiness probes、
   O(tiles) 扫描、CAS/claim 开销与 stall 单独隔离，不能只归因于 readiness stall。
4. producer-priority 距 unsafe none 仅 1.72%，说明正确的调度顺序在这个合成点捕获了大部分
   可见 headroom。

## 6. 不能成立的结论

1. 本结果不是可编程硬件 TB scheduler；它是 CLC 支撑的单 persistent-kernel 软件复现。
2. 不能把该排名外推到多 producer、多 consumer、非均匀 task 或 cache-sensitive workload。
3. `wait acquire loads` 只计 consumer task 内的 acquire，不含策略选择阶段的 `clc_ready()`
   probes；它既不代表总 readiness traffic，也不是物理 L2 request。
4. unsafe none 的输出全部错误，不能作为可部署策略或可实现 speedup。
5. locality 命中定义为“同一软件 CTA 接续本 tile consumer”，不等同于测得的 L2 locality。

## 7. 证据入口

* 源码：[`bench/tier23_clc_scheduler.cu`](../../bench/tier23_clc_scheduler.cu)
* raw：[`t23_clc.log`](../../bench/results_20260805_b200_tier23_native_v2/t23_clc.log)
* trace：[`t23_clc_trace.csv`](../../bench/results_20260805_b200_tier23_native_v2/t23_clc_trace.csv)
* strict verdict：[`tier23_validation.json`](../../bench/results_20260805_b200_tier23_native_v2/tier23_validation.json)
* 汇总：[`tier23_summary.csv`](../../bench/results_20260805_b200_tier23_native_v2/tier23_summary.csv)
* sanitizer coverage：[`sanitizer_v2_coverage.json`](../../bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_coverage.json)
