# Tier 3：C1 数据传递四版本对比（§7.5）

| 项 | 值 |
|---|---|
| 报告 / 实验日期 | 2026-08-05（UTC） |
| 设备 | NVIDIA B200，148 SM，CC 10.0；Driver 580.126.09，CUDA 13.0 |
| 正式结果 | `bench/results_20260805_b200_tier23_native_v2/` |
| 证据等级 | **B：四实现真机时间、occupancy 与全量数据校验；物理 L2/DRAM counter 因宿主权限不可用** |

## 1. 执行摘要

扫描 1–64 KiB/CTA 后，L2 access-policy persistence 与默认分离路径没有可分辨的时间收益：
64 KiB 时两者的 median 都是 0.437856 ms。cluster/DSMEM 融合上界从 1 KiB 的 0.101%
优势增长到 64 KiB 的 1.125%，但该点的 occupancy API 上限从 separate consumer 的 16 blocks/SM 降到
3 blocks/SM。unsafe no-wait 在 64 KiB 给出的总 headroom 为 5.174%。

| bytes/CTA | separate default | separate persist | fused cluster | forced-refetch `.cv` | unsafe none |
|---:|---:|---:|---:|---:|---:|
| 1 KiB | 0.411008 ms | 0.410880 | 0.410592 | 0.411040 | 0.409888 |
| 4 KiB | 0.412928 ms | 0.412864 | 0.412192 | 0.412896 | 0.410720 |
| 16 KiB | 0.417920 ms | 0.417856 | 0.416672 | 0.417856 | 0.411648 |
| 32 KiB | 0.424416 ms | 0.424448 | 0.422080 | 0.424384 | 0.412704 |
| 64 KiB | 0.437856 ms | 0.437856 | 0.432928 | 0.437888 | 0.415200 |

## 2. 程序实际做了什么

[`bench/tier23_c1.cu`](../../bench/tier23_c1.cu) 的分离路径由 producer grid 写每 CTA 的
X bytes，ready 后 programmatic trigger，consumer grid 用 `cudaGridDependencySynchronize()`
后读取全部 words。`separate-persist` 只额外在 stream 上设置 L2 access-policy window；
`separate-default` 使用普通 load；`separate-cv` 使用 `ld.global.cv` 作为强制 refetch 的
悲观 control，不冒充硬件 L2 bypass。

`fused-cluster` 用两 CTA cluster：rank 0 将数据写入 dynamic shared memory，cluster sync 后
rank 1 通过 DSMEM 读取。64 KiB/CTA 时 cluster 总 dynamic shared memory 为 131,072 bytes。
`none` 保留分离 producer/consumer 工作，但把 trigger 从 data-ready 移到 entry 并省略 wait；
validator 要求至少观察到 stale，正式矩阵实际在每一点都观察到全量 stale。

每个 X 点、每个 epoch 都 poison intermediate/output，并逐 word 校验。报告中的
`software_transfer_gbps` 只按逻辑 bytes/elapsed 计算，源码和 schema 均显式标注
`software_bytes_not_dram_counters=1`。

## 3. 配置与统计

* X=`1/2/4/8/16/32/64 KiB`，148 tiles、128 threads；3 warmups、31 timed repeats。
* 7 配置、1,085 samples、10,360 trace rows、0 errors；每档 2,000 次 bootstrap median CI。
* Floor=`separate-default`，Impl=`separate-persist`，Ceiling=`none`，可实现上界控制=
  `fused-cluster`，悲观控制=`separate-cv`。
* 64 KiB 点 occupancy：fused kernel 3 blocks/SM，separate consumer 16 blocks/SM；最大 persistence window
  82,903,040 bytes，device access-window limit 134,217,728 bytes。
* NCU 返回 `ERR_NVGPUCTRPERM`；因此计划中的物理 DRAM traffic 与 L2 hit rate 没有被伪造。
* 同正式 binary SHA 的 Compute Sanitizer 覆盖 separate-default/persist 并报 0 errors。
  工具会把中间的 unsafe Ceiling 串行成正确结果，冻结 binary 因 fail-closed 在此退出，故
  后置 fused/CV 没有冒充 memchecked；它们由正式逐 word 校验、trace 与静态 binary 证据覆盖。

## 4. 头条数字复算

```text
64 KiB cluster 节省 = 0.437856 - 0.432928 = 0.004928 ms
相对 Floor          = 0.004928 / 0.437856 = 1.1255%

64 KiB no-wait 空间 = 0.437856 - 0.415200 = 0.022656 ms
相对 Floor          = 0.022656 / 0.437856 = 5.1743%

1 KiB cluster 节省  = 0.411008 - 0.410592 = 0.000416 ms = 0.1012%
```

occupancy 代价按 API 上限为 `16 / 3 = 5.33×`；它是潜在并发容量差，不等于某个生产
workload 已经损失 5.33× 吞吐。

## 5. 可以成立的结论

1. 在本 resource envelope 下，仅设置 persistence window 没有产生可报告的 end-to-end 优势。
2. DSMEM 融合的端点优势总体从 1 KiB 的 0.416 µs 增到 64 KiB 的 4.928 µs（2 KiB 点
   曾降至 0.384 µs，并非严格单调）；64 KiB 相对收益只有 1.13%，同时付出显著 occupancy
   API 上限下降，“融合永远更好”不成立。
3. 64 KiB 的 5.17% 是删除正确 RAW 顺序后得到的 **unsafe 总上界**；它不能拆解成
   synchronization、launch、cache 或 fused-cluster 各自可实现的优化余量。

## 6. 不能成立的结论

1. 没有 NCU counter，不能声称 persistence 改变了多少 L2 hit、DRAM bytes 或 cache residency。
2. `.cv` 是 forced-refetch control，不是“强制 L2 bypass”的硬件等价实现。
3. `software_transfer_gbps` 不是总线吞吐，不能拿来发布带宽结论。
4. fused cluster 是同一 kernel 内的可实现上界控制，不是跨 kernel CTA-PDL 实现。
5. 这些数字不覆盖不同 occupancy、不同 working-set reuse 或多租户 cache 压力。
6. `16/3=5.33×` 比较的是 separate consumer 与 fused kernel 各自的 active-block API 上限，
   不是完整 separate pipeline、cluster 数或 tile 吞吐比。

## 7. 证据入口

* 源码：[`bench/tier23_c1.cu`](../../bench/tier23_c1.cu)
* strict verdict：[`tier23_validation.json`](../../bench/results_20260805_b200_tier23_native_v2/tier23_validation.json)
* 汇总：[`tier23_summary.csv`](../../bench/results_20260805_b200_tier23_native_v2/tier23_summary.csv)
* 64 KiB raw/trace：[`t23_c1_kb64.log`](../../bench/results_20260805_b200_tier23_native_v2/t23_c1_kb64.log)、
  [`t23_c1_kb64_trace.csv`](../../bench/results_20260805_b200_tier23_native_v2/t23_c1_kb64_trace.csv)
* profiler 边界：[`ncu_status.txt`](../../bench/results_20260805_b200_tier23_native_v2/ncu_status.txt)
* sanitizer 边界：[`sanitizer_v2_coverage.json`](../../bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_coverage.json)
