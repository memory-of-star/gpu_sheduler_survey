# Tier 2/3 native v1 拒绝审计：race-based Ceiling

| 项 | 值 |
|---|---|
| 审计 / 实验日期 | 2026-08-05（UTC） |
| 设备 | NVIDIA B200，148 SM，CC 10.0 |
| 被拒绝目录 | `bench/results_20260805_b200_tier23_native_v1/` |
| 证据等级 | **REJECTED：accepted timing=0；只可复用根因与修复审计** |

## 1. 执行摘要

v1 整轮被拒绝，不能与 replacement v2 拼接。3 个 protocol 与 14 个 encoding 配置的安全档
虽然都通过全边验证，但 `none` 均得到 `stale=0`：正式 grid 上，PSS 可能把 consumer 放在
producer 数据已经 ready 之后。原 Ceiling 因而依赖一次有利 scheduling race，并不是稳定的
无 wait 错结果控制。17 点在进入 timed samples 前 fail-closed；其余 18 点即使跑完也因
campaign-level invalidation 不可复用。可采信 timing 数为 **0**。

## 2. 程序实际发生了什么

原 consumer 确实省略 wait，但没有设备侧机制强制它在某条真实 parent store 前 snapshot。
“没有依赖边”并不等于“调度器一定先运行 consumer”；因此 `none` 偶尔得到正确数据。
这会让 Ceiling 与安全档收敛，正是不能靠报告文字补救的 harness 语义失败。

首轮 CPU validator 尾部被外层终止，rc=143；随后优化后的纯 CPU strict 重跑完成，并在
`tier23_validation.json` 写出 `status=FAIL`、17 errors、18 个已完成配置与 2,449 个不可复用
samples。结构化拒绝账本另记 `accepted_timing_samples=0`。

## 3. 修复与重新准入

replacement 在 child 0 的真实 O(1) parent 上加入 device-only adversarial sentinel：unsafe
consumer 在依赖点立即 snapshot poison，再 release-store proof epoch；该 parent 保留完全相同
的 store/work，但必须等 proof epoch 后才写。strict trace 强制
`child0.t_dep <= sentinel_parent.t_ready`。proof latch loads 与协议 poll loads 分账。

先用 g148 protocol 和 g1184 strided/d64 的 targeted stress 证明这套 schedule，再运行完整
v2。v2 的 35 配置、5,084 samples、182,460 trace rows 全部 strict PASS，才成为正式来源。

## 4. 可以复用什么

1. v1 的失败模式、strict validator 输出和 root-cause 分析可用于 regression。
2. “省略 wait 不保证先读到 poison”这一 CUDA scheduling 事实可指导后续 Ceiling 设计。
3. v1 目录必须原样保留，作为为什么 v2 引入 sentinel 的审计链。

## 5. 不能复用什么

1. v1 的任何 log、trace、summary 或 timing 都不得进入性能表、CI、headroom 或跨轮统计。
2. 安全档局部 PASS 不能把整轮升级成 PARTIAL timing；同轮 Ceiling 语义无效使 bracket 失效。
3. targeted stress 只负责准入 sentinel，不替代完整 formal 矩阵。

## 6. 证据入口

* 机器拒绝账本：[`tier23_rejection.json`](../../bench/results_20260805_b200_tier23_native_v1/tier23_rejection.json)
* v1 strict FAIL：[`tier23_validation.json`](../../bench/results_20260805_b200_tier23_native_v1/tier23_validation.json)
* targeted sentinel：[`bench/targeted_tier23_sentinel_v3_20260805/`](../../bench/targeted_tier23_sentinel_v3_20260805/)
* replacement formal：[`bench/results_20260805_b200_tier23_native_v2/`](../../bench/results_20260805_b200_tier23_native_v2/)
* replacement 源码：[`bench/tier23_protocol_encoding.cu`](../../bench/tier23_protocol_encoding.cu)

