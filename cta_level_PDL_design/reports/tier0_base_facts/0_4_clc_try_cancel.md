# Tier 0.4：Blackwell CLC `try_cancel` 实验报告

日期：2026-08-03（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0  
实验状态：**有效的单点 CLC 竞争/仲裁探针；不是完整 persistent scheduler 性能报告**

## 1. 执行摘要

实验启动 4,096 个低资源 launch units，每个 resident CTA 的线程 0 反复调用 `clusterlaunchcontrol.try_cancel`，尝试取消并领取尚未启动的 cluster/CTA 工作。

单点结果：

| 指标 | 数值 |
|---|---:|
| Launch units | 4096 |
| Median kernel time | 0.00899 ms |
| Aggregated attempts | 4096 |
| Successful cancellations | 1728 |
| Final failed attempts | 2368 |
| Reported success rate | 42.19% |
| Cycles/attempt | 1650.83 |
| Duplicate claims | 0 |
| Unclaimed entries | 2368 |

`2368 = 148 SM × 16 CTA/SM`，与低资源 128-thread kernel 的一次 resident capacity 相符；`1728 = 4096 - 2368` 是仍在 pending queue 中、可被成功取消和领取的工作。该日志没有单独打印 `clcProbeK` 的 occupancy API 查询，因此 16 CTA/SM 是由计数关系和相邻容量实验支持的解释，不是本探针额外输出的一项独立 occupancy 测量。

最重要的结论是单一赢家检查：1,728 个成功领取的 id 没有重复。`42.19%` 不是一个固定的仲裁成功概率，而是当前 pending/resident 构成的结果。

## 2. CLC 在实验中的含义

CLC（Cluster Launch Control）允许正在运行的 cluster/CTA 尝试取消一个尚未派发的 launch unit，并获得其第一个 CTA id，从而在 persistent kernel 内接管这份工作。

本实验使用 inline PTX：

- `clusterlaunchcontrol.try_cancel.async...b128` 发起取消；
- mbarrier 跟踪 16-byte response 完成；
- `clusterlaunchcontrol.query_cancel.is_canceled` 判断成功；
- `clusterlaunchcontrol.query_cancel.get_first_ctaid::x` 取得被取消工作 id。

源码：[bench/clc_probe.cu](../../bench/clc_probe.cu)。

## 3. Kernel 执行流程

每个 resident CTA：

~~~text
线程 0 初始化 shared-memory mbarrier
    ↓
记录 clock64
    ↓
mbarrier expect_tx(16 bytes)
    ↓
try_cancel
    ↓
等待 mbarrier 完成
    ↓
累计 cycles 和 attempts
    ↓
query_cancel
    ├── 成功：取得 ctaid，claimed[ctaid] += 1，继续尝试
    └── 失败：退出循环
~~~

整个 CTA 用 `__syncthreads_or` 广播 leader 的成功/失败结果。

## 4. 实验配置与统计

| 参数 | 数值 |
|---|---:|
| Grid / clusters 参数 | 4096 |
| Threads/CTA | 128 |
| Warmup | 3 次 |
| Timed repeats | 10 次 |
| 计时 | CUDA events |
| Median | 10 次 kernel makespan 中位数 |
| Counter/claim 检查 | 最后一次 invocation 的 host copy |

`cycles/attempt` 的 clock64 区间包含 `mbarrier.arrive.expect_tx`、`try_cancel` 和等待 response 完成，不包含随后 query response、`claimed[]` atomic 或完整 kernel 调度时间。因此它也不是一条裸 `try_cancel` 指令的孤立 latency。

## 5. 计数关系

最终一次 invocation 满足：

~~~text
attempts = successful cancellations + final failed attempts
4096     = 1728                     + 2368
~~~

每个 resident CTA 在 queue 耗尽后各观察一次失败并退出；成功的 CTA 在此之前可多次领取 pending work。

`unclaimed=2368` 表示这些 id 没有被 CLC 取消领取。它与 resident capacity 相同，应解释为已经被正常派发/执行的 launch units，而不是遗漏任务。

## 6. Single-winner 验证

host 将整个 `claimed[4096]` 数组拷回：

- `claimed[id] > 1` 的条目数为 0；
- `claimed[id] == 0` 的条目数为 2368；
- 其余 1728 个条目各被领取一次。

这在当前 invocation 中验证了成功取消的 id 没有重复分配。它不是对所有 grid/cluster 形状和所有竞争强度的形式化证明。

## 7. 能成立的结论

本实验支持：

1. B200/sm_100 上 inline-PTX CLC `try_cancel` 路径能够运行。
2. 当前 4096-unit 竞争构造中，1728 个 pending units 被唯一领取，没有 duplicate claim。
3. 2368 个未领取 units 与一次 resident capacity 一致。
4. 当前争用与 mbarrier 路径下聚合成本为约 1650.83 cycles/attempt。

## 8. 不能成立的结论

本实验不支持：

1. CLC 的独立硬件 latency 恒定为 1650.83 cycles。
2. 任意时刻 `try_cancel` 成功概率都是 42.19%。
3. CLC persistent scheduler 一定提升 producer–consumer 性能。
4. 多 CTA cluster、不同 cluster dimension 或 multicast response 有相同行为。
5. 该单点已经比较了 producer-priority、consumer-priority 或 locality policy。

## 9. 后续需要的实验

要把 CLC 用作 CTA scheduler，还需要：

1. 改变 total launch units 与 resident capacity 的比例；
2. 测不同 threads/register/smem 和多 CTA cluster；
3. 加入真实 work stealing payload；
4. 对比 persistent CLC scheduler 与普通 grid scheduling；
5. 测量被接管工作对缓存局部性和 background throughput 的影响。

## 10. 证据入口

- 源码：[bench/clc_probe.cu](../../bench/clc_probe.cu)
- 原始日志：[bench/results_budget1h_corrected/tier0_clc.log](../../bench/results_budget1h_corrected/tier0_clc.log)
- 汇总：[bench/results_budget1h_corrected/summary.txt](../../bench/results_budget1h_corrected/summary.txt)
- PDL/CLC 接口说明：[docs/cuda_13.4_pdl_clc_interfaces.md](../../docs/cuda_13.4_pdl_clc_interfaces.md)
- 总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)
