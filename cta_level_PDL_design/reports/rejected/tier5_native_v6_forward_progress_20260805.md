# Tier 5 native v6 正式轮 forward-progress 拒绝审计

| 元数据 | 值 |
|---|---|
| 报告日期 | 2026-08-05（UTC） |
| 实验日期 | 2026-08-05（UTC） |
| 设备 | 单卡 NVIDIA B200，148 SM，Compute Capability 10.0 |
| 证据等级 | **REJECTED / FAILURE-MECHANISM EVIDENCE ONLY**——可支持 v6 在 32K exact Impl validation 上未前进以及整轮 timing 必须清零；不能支持任何性能比较，也不能把本轮已完成的 4K 点单独捞出 |

## 1. 执行摘要

native v6 正式轮在 4K 点完成后进入 32K exact 点，运行
`871.366079 s` 仍没有完成记录。冻结的运行时间预测上界是
`163.00766928 s`，实耗已达该上界的

~~~text
871.366079 / 163.00766928 = 5.345553×
~~~

操作者因此终止 stalled child process group；runner 返回 143。正式拒绝记录固定为
`status=REJECTED`、`accepted_timing=0`。这个决定覆盖目录中的**全部** timing，包括先完成的
4K 点；该目录禁止 resume，也禁止与其他轮次拼接。

随后以相同 v6 source/binary 做了不计时的 32K 最小诊断。Floor 完成全元素校验并输出
`valid=1`，Impl 的第一轮 validation 在 60 秒内没有输出完成记录，外层 timeout 返回 124。
这把失效位置收窄到 Impl forward progress，而不是 Floor、工作量生成或最终 validator。

## 2. 程序实际执行的语义

32K exact 配置有 512 个 query block、256 个 key tile、131,072 个 producer CTA、512 个
top-k CTA 和 512 个 attention CTA。Floor 是三节点 programmatic CUDA Graph；Impl 在三个
不同优先级 stream 上先提交 attention、再 top-k、最后 indexer producer，并由前两类
consumer resident-spin 等待 per-producer epoch flag。

资源记录显示 attention kernel 的 occupancy 上限为 1 block/SM。因而多波配置可以让等待中
的 consumer CTA 占满所有 148 个 SM，同时仍有大量 consumer/producer CTA 尚未派发。CUDA
stream priority 不能充当跨 kernel 的 forward-progress 保证。正式轮和最小诊断的共同现象
与这种 resident-wait starvation 一致；这是根据已执行结构和失效点作出的根因推断，不是对
未公开 GPU block scheduler 策略的测量。

## 3. 配置、停止规则与证据

正式 32K 尝试使用 exact 全工作量，没有缩小 query 数。GPU exclusivity monitor 在
`871.366079 s` 内记录 10,498 次 observation，状态为 `PASS`，说明没有发现外来 compute
进程；它只能证明独占性，不能把持续 GPU 活跃解释为有界前进。

该轮当时还没有 per-point 强制 watchdog，因此在超过冻结上界 5.345553 倍后由操作者执行
SIGTERM。这个缺陷本身也阻止该 formal campaign 进入准入：修复版必须在 runner 内对每个点
设置正的强制 timeout，超时即原子写入永久拒绝标记并停止收集 timing。

不计时诊断固定如下：

| 项 | 值 |
|---|---:|
| seq | 32,768 exact |
| warmup / repeats | 0 / 1 |
| 外层 timeout | 60 s |
| Floor | 全 score/index/output/flag 与 131,072 次 history load 校验通过 |
| Impl | 首次 validation 未完成 |
| 返回码 | 124 |
| 诊断前/后外来 GPU process | 0 / 0 |

## 4. 拒绝算术与不可变边界

正式拒绝 JSON 自身的 SHA-256 为
`a2ca6017232f64b37639c9e3398db9ab574e0035e48881b129428c5334a41246`。它继续绑定：

| artefact | SHA-256 |
|---|---|
| build manifest | `04b84e39b33915cc410c33a5a09cdc9c54ec82a076f63701f5adc6fbbe5c0e40` |
| binary proof | `d9f1a9383be6f2bf91a742d5972b53abe00b26e46bce86e5468cdeb773f780bc` |
| 32K monitor | `d863b1c5a38d56c24f3733cbe317547aa3e47f65a76f875e6dc1e725e9434cf5` |
| 32K observations | `e6a733c66de346b945840058eef4c0fa2f71f90203259a998d5216a23399c85a` |
| 32K precheck | `3f204e05f9eacce7edfaf159dc6fb9315a73818bd5acedc09c35240af594068f` |
| completed 4K validation | `1d1069427a6f40671a1038505399650b1fd9e0f4880d16eae0f7614609a767ac` |
| 最小诊断 log | `a7835ae308d4cb5ff0484c72852f4b632f141287b4e079c8451b184868d9b878` |

因此，本报告不会列出 4K latency、CI 或 Floor/Impl/Ceiling 差值。`accepted_timing=0` 是整轮
原子边界，而不是只清除卡住的 32K 行。

## 5. 修复契约

后继版本必须同时满足：

1. 保留一次完整三节点 Graph launch 的 full-grid grid-PDL Floor，不能把它偷偷切成 wave 后
   仍称 production baseline；另设 `wave_floor` 与 Impl/Ceiling 使用完全相同 query wave。
   四路必须完成完全相同数量的 score、index、history 和输出工作，不能通过抽样或删边获得
   前进性。full-grid Floor→Impl 只能解释为包含 wave 调度成本的 mechanism envelope；
   wave-Floor→Impl 才是波界匹配的协议差。
2. 每个 wave 的等待 consumer 数量由实际 occupancy/resource 查询推导，并保留可复核的
   free-SM 下界；不得再依赖“大 grid 总会让 producer 插进来”的放置运气。
3. 4K/32K exact 与 128K/1M packed boundary 分别完整执行；任何不可承载点必须在运行前作
   机器可读资源拒绝，不能换成 sampled proxy。
4. 每个正式点有正的 runner-enforced watchdog；timeout 永久拒绝整轮并令 accepted timing
   为 0。
5. 修复后使用全新目录，重新生成 source/binary/PTX/SASS/resource/Profiler/sanitizer 证据；
   不从 v6 formal 复用样本。

## 6. 能成立的结论

1. v6 32K exact formal 没有在冻结时间上界内完成，且在 5.345553 倍上界后仍无完成记录。
2. 相同 binary 的最小诊断证明 Floor 全元素校验完成，而 Impl 首次 validation 超时；问题
   不是由 31 次 timing 重复或 formal 聚合器造成。
3. v6 formal 整轮必须拒收，4K 已完成样本也不例外。
4. 独立的 v6 strict smoke 仍可作为该独立 campaign 的 harness 证据，但不能修复或补齐这次
   formal。

## 7. 不能成立的结论

本次失败不支持：

1. v6 Impl 比 Floor 慢多少，或 32K latency 的任何有限估计。
2. 持续 GPU utilization 等价于 forward progress。
3. 4K 通过即可证明 32K/128K/1M 多波结构安全。
4. B200 scheduler 的未公开调度规则，或 B300/Rubin 的行为。
5. 简单提高 timeout 能修好协议；已观察到的是没有有界前进证据，而不是统计样本不足。

## 8. 可以保留和复用什么；不能复用什么

可以复用的是 failure-mechanism 与工程审计：v6 的 frozen source/binary/build 身份、32K
monitor 的独占性证据、正式超时位置、最小诊断中 Floor 完成而 Impl 未完成的定位，以及由此
导出的分波、entry gate 和强制 watchdog 回归要求。独立的 strict-v6 smoke 仍按它自己的
admission 边界保留。

不能复用的是本 formal 目录的任何 timing、median、CI、headroom 或性能排序；已完成的 4K
也不能例外，不能与 v7 或 production campaign 拼接。诊断的 60 秒 timeout 只用于定位，不是
32K latency 下界。本文的 starvation 根因也不能作为 B200 未公开调度策略或 Rubin 实现事实
复用。

## 9. 证据入口

- 正式拒绝记录：[bench/dsa/results_20260805_b200_native_formal_strict_v6/formal_rejection.json](../../bench/dsa/results_20260805_b200_native_formal_strict_v6/formal_rejection.json)
- 目录级拒绝说明：[bench/dsa/results_20260805_b200_native_formal_strict_v6/REJECTED.md](../../bench/dsa/results_20260805_b200_native_formal_strict_v6/REJECTED.md)
- 32K 正式 log：[bench/dsa/results_20260805_b200_native_formal_strict_v6/dsa_exact_seq32768.log](../../bench/dsa/results_20260805_b200_native_formal_strict_v6/dsa_exact_seq32768.log)
- 32K exclusivity monitor：[bench/dsa/results_20260805_b200_native_formal_strict_v6/dsa_exact_seq32768_gpu_monitor.json](../../bench/dsa/results_20260805_b200_native_formal_strict_v6/dsa_exact_seq32768_gpu_monitor.json)
- 4K strict validation（只证明该行曾完成，不准复用 timing）：[bench/dsa/results_20260805_b200_native_formal_strict_v6/dsa_exact_seq4096_validation.json](../../bench/dsa/results_20260805_b200_native_formal_strict_v6/dsa_exact_seq4096_validation.json)
- 最小诊断 log / 返回码 / binary 身份：[bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/diagnostic.log](../../bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/diagnostic.log)、[bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/returncode.txt](../../bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/returncode.txt)、[bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/source_binary_sha256.txt](../../bench/dsa/results_20260805_b200_native_forward_progress_diag_v1/source_binary_sha256.txt)
- 原始 v6 source：[bench/dsa/dsa_native.cu](../../bench/dsa/dsa_native.cu)；该路径后续会演进，正式身份以拒绝 JSON 绑定的 build/binary artefact 为准
- 旧 Python harness 的独立语义审计：[reports/rejected/tier5_dsa_semantic_audit.md](tier5_dsa_semantic_audit.md)
