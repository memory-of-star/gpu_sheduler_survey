# Tier 0.3：Resident Waiting CTA Occupancy 容量实验报告

日期：2026-08-03（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0  
实验状态：**有效的 occupancy 容量曲线；不是等待 CTA 对系统吞吐影响的定价实验**

## 1. 执行摘要

实验改变 threads/CTA 和 dynamic shared memory，调用 CUDA occupancy API 得到每 SM 最大 active CTA 数，然后启动恰好 `CTA/SM × 148 SM` 个 waiting CTA。

主要结果：

- 128 threads、0/8/16/32/64 KiB 时，容量分别为 16/16/13/6/3 CTA/SM。
- 256 threads、0/8/16/32/64 KiB 时，容量分别为 8/8/8/6/3 CTA/SM。
- 全卡一次容量从 2,368 CTA 降到 444 CTA。
- 各点 kernel 中位时间约 0.517–0.550 ms，没有随容量单调变化。
- 64 KiB dynamic shared memory 需要显式 opt-in；修正后另做的 64 KiB producer–consumer launch smoke 成功并 PASS。

正确结论是容量曲线本身。由于没有 productive background kernel，本实验不能回答“resident waiting CTA 抢占了多少真实工作吞吐”。

## 2. 实验程序

源码：[bench/tier0_facts.cu](../../bench/tier0_facts.cu)。

`waiterK` 的逻辑为：

~~~text
可选：线程 0 触碰 dynamic shared memory
    ↓
全 CTA spin_cycles(1,000,000)
    ↓
线程 0 写 sink[blockIdx.x]
~~~

这个 kernel 模拟一个已驻留并占用资源的 CTA，但它没有等待真实 producer flag，也没有与有用 background kernel 并发。

## 3. 实验配置

| 参数 | 数值 |
|---|---:|
| GPU | NVIDIA B200 |
| SM 数 | 148 |
| Threads/CTA | 128、256 |
| Dynamic smem/CTA | 0、8、16、32、64 KiB |
| Wait | 1,000,000 cycles |
| Grid size | occupancy API 结果 × 148 SM |
| Warmup | 3 次/点 |
| Timed repeats | 5 次/点 |
| 汇总 | CUDA event elapsed time 中位数 |

64 KiB 点在 occupancy query 和 launch 前调用：

~~~text
cudaFuncSetAttribute(
    waiterK,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    64 * 1024)
~~~

## 4. 完整结果

| Threads | Smem/CTA | CTA/SM | 全卡容量 CTA | Median ms |
|---:|---:|---:|---:|---:|
| 128 | 0 KiB | 16 | 2368 | 0.54886 |
| 128 | 8 KiB | 16 | 2368 | 0.54928 |
| 128 | 16 KiB | 13 | 1924 | 0.55002 |
| 128 | 32 KiB | 6 | 888 | 0.51754 |
| 128 | 64 KiB | 3 | 444 | 0.51722 |
| 256 | 0 KiB | 8 | 1184 | 0.54906 |
| 256 | 8 KiB | 8 | 1184 | 0.54874 |
| 256 | 16 KiB | 8 | 1184 | 0.54880 |
| 256 | 32 KiB | 6 | 888 | 0.54893 |
| 256 | 64 KiB | 3 | 444 | 0.51699 |

## 5. 容量变化的含义

128-thread kernel 在低 smem 时可达到 16 CTA/SM；增加到 16 KiB 后容量降到 13，32 KiB 后降到 6，64 KiB 后降到 3。

256-thread kernel 在低 smem 时先受线程数约束为 8 CTA/SM；32 KiB 和 64 KiB 才进一步被 shared memory 压到 6 和 3。

这些是 `waiterK` 在当前编译结果和 B200 上的 occupancy API 输出，不是所有使用相同 threads/smem 的 kernel 的通用容量。寄存器数等其它资源也会改变真实 occupancy。

## 6. 为什么时间不能解释成“等待代价”

每个点都只启动恰好填满 occupancy 容量的一批 CTA，并让它们执行相同长度的 spin：

- 没有 productive background kernel；
- 没有测量 background throughput；
- 没有 producer/consumer readiness；
- 不同点的 CTA 总数不同；
- 0.517–0.550 ms 的小差异可能包含调度、时钟和测量噪声。

所以 `median_ms` 只能描述该 full-capacity spin grid 的完成时间，不能直接定价“一个等待 CTA 造成多少损失”。

## 7. 64 KiB launch smoke

原始 FAST 轮次的 64 KiB producer–consumer 点因未 opt-in dynamic shared memory 而报 `invalid argument`。修正后运行了一个独立 smoke：

| 参数 | 数值 |
|---|---:|
| Structure | self |
| Degree | 1 |
| P/C | 64/64 |
| Threads | 128 |
| Smem | 64 KiB |
| Occupancy | 3 CTA/SM |
| Tail/prologue | 10K/10K cycles |
| Repeats | 3 |
| Wait mode | grid |
| Median | 0.01494 ms |
| Correctness | PASS |

这个 smoke 只证明 64 KiB 配置可以正确 query/launch，不提供性能收益结论。

## 8. 能成立的结论

本实验支持：

1. 当前 `waiterK` 的 occupancy 容量随 threads 和 dynamic smem 呈现表中的阶梯变化。
2. 64 KiB dynamic smem 必须显式 opt-in；修正后在 B200 上可启动。
3. 后续测试真实 resident-wait 成本时，应至少覆盖 16、8、6、3 CTA/SM 等容量区间。

## 9. 不能成立的结论

本实验不支持：

1. waiting CTA 的系统代价约为 0.52–0.55 ms。
2. shared memory 越多，waiting CTA 反而越快。
3. 当前容量就是真实 GEMM/Attention kernel 的 occupancy。
4. pre-dispatch gating 能恢复多少真实应用吞吐。
5. 与 background kernel 共存时没有 L2、warp scheduler 或带宽干扰。

## 10. 证据入口

- 源码：[bench/tier0_facts.cu](../../bench/tier0_facts.cu)
- Corrected Tier 0 日志：[bench/results_budget1h_corrected/tier0_facts.log](../../bench/results_budget1h_corrected/tier0_facts.log)
- 64 KiB launch smoke：[bench/results_budget1h_corrected/smem64_launch_smoke.log](../../bench/results_budget1h_corrected/smem64_launch_smoke.log)
- 原 64 KiB 失败日志：[bench/results_budget1h/t03_smem64.log](../../bench/results_budget1h/t03_smem64.log)
- 总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)

