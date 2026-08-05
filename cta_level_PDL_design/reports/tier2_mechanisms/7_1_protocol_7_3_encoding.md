# Tier 2：同步协议与依赖编码横评（§7.1 / §7.3）

| 项 | 值 |
|---|---|
| 报告 / 实验日期 | 2026-08-05（UTC） |
| 设备 | NVIDIA B200，148 SM，CC 10.0；Driver 580.126.09，CUDA 13.0 |
| 正式结果 | `bench/results_20260805_b200_tier23_native_v2/` |
| 证据等级 | **B：合格 native harness 的真机时序、全边正确性与软件逻辑访存计数；无权限读取物理 NCU L2/DRAM counter** |

## 1. 执行摘要

正式 validator 对 17 个配置（协议 3、编码 14）全部给出 `PASS`：2,635 个 timed
sample、82,880 条 `%globaltimer` trace，所有安全档均通过独立全边校验，所有 `none`
档均由 device sentinel 证明至少一条真实 RAW 边读取了本 epoch 的 poison。

§7.1 的核心结果是：在 148、296、1,184 个 1-to-1 依赖上，fixed-spin 与 backoff
每轮中位逻辑 acquire-load 数分别恰好等于 CTA 数；identity-safe contiguous-prefix
则从 2,148 增到 134,039。三种安全软件协议的总 makespan 与 grid-PDL Floor 的
bootstrap CI 大量重叠，配对 background 吞吐在同一点各档之间的最大跨度仅
0.0035%、0.0081%、0.0185%。因此本矩阵没有证明软件协议的端到端速度差，但清楚证明
prefix 维护的逻辑轮询放大。

| P=C | mode | median ms | median wait | dependency-ready→snapshot | 逻辑 acquire loads | background GB/s |
|---:|---|---:|---:|---:|---:|---:|
| 148 | grid | 21.319744 | 704 ns | — | 0 | 686.267 |
| 148 | fixed-spin | 21.326752 | 800 ns | 20.463 ms | 148 | 686.291 |
| 148 | backoff | 21.325888 | 800 ns | 20.465 ms | 148 | 686.286 |
| 148 | monotonic-prefix | 21.326400 | 896 ns | 20.334 ms | 2,148 | 686.277 |
| 296 | grid | 21.522912 | 704 ns | — | 0 | 686.043 |
| 296 | fixed-spin | 21.521856 | 800 ns | 20.605 ms | 296 | 686.072 |
| 296 | backoff | 21.457568 | 800 ns | 20.577 ms | 296 | 686.072 |
| 296 | monotonic-prefix | 21.527456 | 864 ns | 20.499 ms | 8,595 | 686.016 |
| 1,184 | grid | 21.945024 | 672 ns | — | 0 | 685.275 |
| 1,184 | fixed-spin | 21.930464 | 800 ns | 20.988 ms | 1,184 | 685.310 |
| 1,184 | backoff | 21.934528 | 800 ns | 20.991 ms | 1,184 | 685.270 |
| 1,184 | monotonic-prefix | 21.929312 | 864 ns | 20.831 ms | 134,039 | 685.183 |

grid 的 `—` 表示该软件 wake 指标不适用；raw 中的 `0` 是空 `wakes` 数组的 sentinel，
不是实测的零延迟。软件档的约 20–21 ms 不是自旋指令的固有 wake latency：consumer CTA 在 producer
ready 后很久才被实际放置，而它一旦进入 wait，观测的 wait 中位数只有 0.8–0.9 µs。
该列是“某条依赖 ready 到该 consumer 完成 RAW snapshot”的组合调度指标，也包含 decode、
wait 与 snapshot，不是纯轮询指令延迟。

§7.3 的交叉点取决于结构，而不是只取决于 degree：

* `interval` 结构在 degree 1–64 全程由 interval 二元组同时取得最低 decode 中位数
  （224 ns）和最少 metadata loads；扫描区间恰好等于真实 parent 集。
* `strided` 在 degree=1 时三种表示的真实轮询量相同；从 **degree=2** 起，interval
  bounding range 已产生 44,104 次轮询，而 bitmask/CSR 只有 592 次，交叉点就是 1→2。
  degree=64 时两者分别为 84,968 与 18,944，即 interval 多 4.485×。
* bitmask decode 在本矩阵为 832–864 ns、metadata loads 固定 5,920；CSR decode
  从 544–576 ns 增至 4,640 ns，metadata loads 从 1,776 增至 39,072。因此对精确的
  稀疏集合，CSR 在低 degree 比 bitmask 解码便宜，约在 degree 4–8 之间反转；对规整
  区间，二者都输给 O(1) interval。

| structure / degree | interval decode / metadata / polls | bitmask decode / metadata / polls | CSR decode / metadata / polls |
|---|---:|---:|---:|
| interval / 1 | 224 ns / 1,184 / 296 | 864 ns / 5,920 / 296 | 576 ns / 1,776 / 296 |
| interval / 64 | 224 ns / 1,184 / 18,944 | 832 ns / 5,920 / 18,944 | 4,640 ns / 39,072 / 18,944 |
| strided / 2 | 224 ns / 1,184 / 44,104 | 864 ns / 5,920 / 592 | 672 ns / 2,368 / 592 |
| strided / 64 | 224 ns / 1,184 / 84,968 | 864 ns / 5,920 / 18,944 | 4,640 ns / 39,072 / 18,944 |

## 2. 程序实际做了什么

[`bench/tier23_protocol_encoding.cu`](../../bench/tier23_protocol_encoding.cu) 的 producer
在软件档和 Ceiling 中于 kernel entry 发 programmatic trigger；完成 per-CTA ready work 并写数据后，
只有安全软件档 release-store epoch flag。grid Floor 在数据 ready 后才 trigger，consumer
调用 `cudaGridDependencySynchronize()`；Ceiling 不发布 epoch，只使用下述独立 proof latch。
fixed 与 backoff 只差固定 64 ns 的 nominal nanosleep 和 64→2,048 ns 的 nominal 指数退避；
prefix 不是已否决的“完成数量计数器”，而是逐 identity acquire
后才能 CAS 前移的 contiguous prefix。

编码实验把 representation decode 与实际 wait 分成两次扫描。`decode_ns` 是每次 invocation
内 296 个 consumer CTA 的中位 `%globaltimer` 差；`metadata_loads` 和 `poll_loads` 是全 grid
软件计数。timed post-wait payload 固定 O(1)，O(degree) 全边检查在单独、不计时的 invocation
完成。每个 epoch 重新 poison 数据和输出。

Ceiling 没有 wait。为防 CUDA 调度偶然把它串行成正确结果，child 0 在依赖点立即 snapshot，
其真实最后 parent 在写数据前等待这个 snapshot 的 release latch；trace 必须满足
`child0.t_dep <= sentinel_parent.t_ready`。该 proof latch 的 loads 单列为
`ceiling_schedule_latch_loads`，没有混进协议 `poll_loads`，Ceiling 时间也不用于安全实现结论。

## 3. 配置与统计

* 协议：`P=C=148/296/1184`，`self`、degree=1；模式为 grid、fixed、backoff、prefix、none。
* 编码：`P=C=296`，`interval/strided × degree {1,2,4,8,16,32,64}`；模式为 grid、
  interval、bitmask、CSR、none。
* 每配置 3 warmups、31 timed repeats，奇偶 repeat 反转档位顺序；每个指标使用 2,000 次
  bootstrap median 95% CI。
* consumer 使用 128 threads、32 KiB dynamic shared memory；正式 trace 使用 `%globaltimer`。
* strict validator 总体为 35 配置、5,084 samples、182,460 trace rows、0 errors；本报告对应
  其中 17 配置、2,635 samples、82,880 trace rows。
* Nsight Systems 已成功采集执行时间线。Nsight Compute 返回
  `ERR_NVGPUCTRPERM`；这是非阻断 profiler sidecar，物理 counter 不由软件数值替代。
* 同正式 binary SHA 的 non-timing Compute Sanitizer v2 对 grid/fixed/backoff/prefix
  四安全协议均报 `ERROR SUMMARY: 0 errors`。unsafe `none` 在工具会改变 PSS 调度时由外部
  wrapper 明确排除，其错误性仍由正式 sentinel trace 证明；coverage 不是全路径的伪称。

## 4. 头条数字复算

prefix 相对逐 flag 的轮询放大：

```text
P=148:  2,148 / 148   = 14.514×
P=296:  8,595 / 296   = 29.037×
P=1184: 134,039 / 1184 = 113.209×
```

strided interval 的假边放大：

```text
degree=2:  interval 44,104 / exact 592    = 74.500×
degree=64: interval 84,968 / exact 18,944 = 4.485×
```

CSR 与 bitmask 的 decode 反转由原始中位数直接定位：degree=8 时
`CSR=1,088 ns > bitmask=864 ns`，degree=4 时 `CSR=800 ns < bitmask=864 ns`；考虑
bootstrap 离散粒度，报告只声称交叉位于 4–8，而不声称连续 degree 的精确阈值。

## 5. 可以成立的结论

1. identity-safe prefix 可以正确工作，但其逻辑 acquire traffic 随 grid 增长显著快于逐 flag
   协议；在当前点上没有观察到抵消这项放大的 makespan 收益。
2. degree 与结构必须正交扫描：连续区间中 interval 是明确赢家，而 strided 从 degree=2
   起 exact set 表示已避免大量假边轮询。
3. bitmask 提供固定 metadata footprint；CSR 的成本随真实 degree 增长，低 degree 更便宜、
   高 degree 更贵。
4. 本实现满足 entry trigger、安全软件档 ready 后 release publication、O(1) timed payload、
   独立全边验证、epoch poison、真实错误 Ceiling 和 `%globaltimer` trace 的准入约束；grid 与
   Ceiling 分别使用 PDL wait 和独立 proof latch，而不冒充软件 epoch publication。

## 6. 不能成立的结论

1. `poll_loads` **不是** L2 read requests。没有 NCU 权限，不能发布物理 L2 请求、hit rate、
   DRAM traffic 或 cache saturation 数字。
2. 约 20 ms 的 ready→snapshot 不能叫“自旋唤醒延迟”；它包含 CTA 放置和跨 kernel 调度。
3. 本矩阵的 makespan CI 重叠，不能据此宣称 fixed-spin、backoff 或 prefix 有稳定端到端胜负。
4. 结果只覆盖 B200、给定 resource envelope 和 degree≤64；不是 Rubin 实现或未来硬件的测量。
5. strided 的 interval 仍是正确的保守 over-wait；本报告证明的是假边成本，不是错误性。

## 7. 证据入口

* 源码：[`bench/tier23_protocol_encoding.cu`](../../bench/tier23_protocol_encoding.cu)、
  [`bench/common/tier23_native.cuh`](../../bench/common/tier23_native.cuh)
* 正式目录：[`bench/results_20260805_b200_tier23_native_v2/`](../../bench/results_20260805_b200_tier23_native_v2/)
* strict verdict：[`tier23_validation.json`](../../bench/results_20260805_b200_tier23_native_v2/tier23_validation.json)
* 汇总与 raw：[`tier23_summary.csv`](../../bench/results_20260805_b200_tier23_native_v2/tier23_summary.csv)、
  [`tier23_matrix.log`](../../bench/results_20260805_b200_tier23_native_v2/tier23_matrix.log)
* profiler 状态：[`ncu_status.txt`](../../bench/results_20260805_b200_tier23_native_v2/ncu_status.txt)、
  [`nsys_status.txt`](../../bench/results_20260805_b200_tier23_native_v2/nsys_status.txt)
* sanitizer 边界：[`sanitizer_v2_status.tsv`](../../bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_status.tsv)、
  [`sanitizer_v2_coverage.json`](../../bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_coverage.json)
