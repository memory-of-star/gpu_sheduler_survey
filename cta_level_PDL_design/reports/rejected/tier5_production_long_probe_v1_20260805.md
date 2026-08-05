# Tier 5 production 1M long probe v1 永久拒收报告

| 元数据 | 值 |
|---|---|
| 报告日期 | 2026-08-05（UTC） |
| 实验目录 | `bench/dsa/results_20260805_b200_production_long_probe_v1` |
| 目标行 | `glm5.indexshare_fsss.seq1048576` |
| 设备 | 单卡 NVIDIA B200，UUID `GPU-eb98326c-c1dd-1104-c4d7-24decdb06aef`；本拒收目录未封存 SM 数/Compute Capability，故不补写 |
| 证据等级 | **REJECTED / FEASIBILITY-DIAGNOSTIC EVIDENCE ONLY** |

## 1. 结论

这是一次 `FAST=1` 、nonformal、定向的 GLM-5 `indexshare_fsss` 1,048,576-token
可行性探测，不是 production formal campaign。操作者在可行性诊断中中断运行；
runner 以返回码 130 将唯一已启动 segment 原子移入 `failed_segments/`，其
`segment_rejection.json` 固定为 `status=REJECTED`、
`reason=runner_interrupted_or_unhandled_failure_rc_130`、`accepted_timing=0`、
`accepted_workload_timing=0`、`accepted_CTA_bracket=0`。

该 segment **永久拒收**，不得 resume、改名或从中捞取样本。目录中没有发布后的
measurement；被拒收 manifest 仍是 `status=RUNNING`、`measurement_emitted=false`。
因此本轮的 timing、latency、throughput、中途速率、完成时间外推以及 off/on
性能排序都不可引用；CTA bracket 与 LLM headroom 也不成立。

## 2. 冻结的运行语义

campaign contract 明确记录 `formal=false`、`campaign_mode=nonformal_short`、
`allow_short=true`、`warmup=0`、`repeats=1`。contract 的 ordered matrix 含目标行和
`glm5.moe32` 两行；fragment-only 选择器只执行了
`glm5.indexshare_fsss.seq1048576`，`moe32` 未启动，不能被视为已执行或已通过。

被拒收 manifest 记录的几何学是全量、无抽样的因果下三角：

| 字段 | 冻结值 |
|---|---:|
| query rows | 1,048,576 |
| query sampling | `NONE` |
| causal pairs | 549,756,338,176 |
| indexer causal FMA FLOPs | 4,503,603,922,337,792 |
| query chunk tokens | 128 |
| chunks | 8,192 |
| official max position | 202,752 |
| within official range | `false` |

这些值只是 manifest 中的输入几何与复杂度证明，不是已完成工作量。它们说明
v1 路径将 1M 全量校验分成了大量小 chunk，可以用于解释为什么需要后续长
上下文实现优化；它们不能证明任何实际进度百分比。

## 3. orphan child 事故与修复边界

操作者中断本轮时暴露了旧 fragment runner 没有可证明地回收完整 child process
group 的问题：子 Python 进程可能脱离 runner，继续占用 GPU。本次保存目录没有
orphan PID/start-ticks 快照或事后清理记录，所以本报告不将任何精确 PID、显存值或
存活时间升格为 artefact-proven 结论。可复用的是故障类型和必须增加的回归要求。

历史 manifest 绑定的旧 `run_production_tier5_fragments.sh` SHA-256 是
`2a20fc020d6c1f3068bef15e51d05ef6866ba027ec938ff77d127bfbb1bc1624`。当前修复版
的源码 SHA-256 是
`52a2ee7919c9a744fba4dfd15f7cc97cc86bbb4644d892ab37e6ab86bec03d4a`：它记录 child PID、
`/proc` start ticks 和独立 PGID，在 `EXIT/INT/TERM/HUP` 路径先 TERM/CONT，有界等待后
KILL，再 reap child 与 monitor，然后才拒收 stage；正式首次 finalize 后还立即执行
fresh `check-final`。当前回归源码
`test_production_tier5.py` SHA-256 是
`15a6d1c92a9d48aec5f0f21fdebef736d4e93e4bcc60cff18048c1c9f524d4e9`，其包含 SIGINT/SIGTERM、
monitor-failure 的 child/grandchild/monitor 回收断言和 finalize 后即时深验断言。这些是对事故类型的修复，不会
把 v1 segment 复活。

## 4. 证据 SHA-256

| artefact | SHA-256 |
|---|---|
| `campaign_binding.json` | `f1501bc8bd737e7eb45735c1ef63c3916eda91a98e3697af4f2b48a665d00ca1` |
| `campaign_contract.json` | `4b61b77d9fedc33d3b85d89c7e966a25420eba5689d84fba811468f8549bfaf9` |
| `campaign_runner.log` | `034ede525b751281f89fad9d9927f06adda6e7e4205ea9bfd532bafb2a0e4c03` |
| rejected `segment_rejection.json` | `e30af77d7e0995dcbe495d9f62ac66f3f3669a9a092724fd0239bc8fea1ffa4a` |
| rejected `manifest.json` | `75dd4f0ef8375196813a9dc658b9618ea33856a24884ceaaef6de98fbcf54692` |
| rejected `runner.log` | `70bb3dd1a4268c3d8954926cbc68ed0bd89d01761f2e8830907be32976171045` |
| rejected `gpu_observations.ndjson` | `3faa82fcc1bd85eaa081a34245a8d0cd74141d7be4a40c2282627041479bc35c` |

`segment_rejection.json` 还在内部封存了拒收前各文件的大小与 SHA；它是本轮的
原子拒收权威。

## 5. 能成立的结论

1. 这是 FAST/nonformal 的定向 1M GLM-5 indexshare-FSSS 可行性探测，不是 formal。
2. 被启动的 ordinal-0 segment 因操作者中断以 rc130 永久拒收，且没有发布
   measurement。
3. 冻结几何表明 v1 对 1M 全量下三角工作量使用了 8,192 个 128-token
   chunk；这是后续优化的复杂度根因证据。
4. 中断流程暴露了完整 process-group 回收要求；当前 runner 和回归源码已将该
   要求明文化。

## 6. 不能成立的结论

本次拒收不支持：

1. 任何 latency、throughput、timing、CI、headroom 或 off/on 快慢关系。
2. 任何中途 chunk 进度数、完成百分比或完成时间外推；保存 artefact 没有封存
   这些证明。
3. production CTA bracket：本 harness 的冻结 contract 本就记录
   `tier5_bracket_admitted=false`、`headroom_defined=false`。
4. production formal 已完成、另一个 `moe32` row 已执行，或 1M 点已通过。
5. 精确 orphan PID、显存占用或存活时间；这些没有被本目录的事后 artefact
   封存。

## 7. 可以保留和复用什么；不能复用什么

可以保留和复用的只有：冻结的全量工作量几何、v1 小 chunk 路径的复杂度根因、
rc130 与原子拒收边界，以及 orphan child 事故导出的完整 process-group 清理和回归
要求。这些只用于后续实现和 harness 审计。

不能复用为结论的是本 segment 的任何 timing/吞吐/性能排序、中途进度外推、CTA
bracket 或 headroom；也不能把它与后续 probe/formal 样本拼接。

v1 在实验线上已被 v2 长上下文
实现优化和全新 probe 取代；后继必须使用新目录、新 contract/source hash 和新
segment UUID 独立准入。后续实现或 runner 修复不能回填 v1，也不能把它的
`accepted_* = 0` 改写为 1。本报告不声称后续 production formal 已完成。

## 8. 复核命令

以下命令全部是只读复核，不启动 GPU：

~~~bash
cd /workspace/gpu_sheduler_survey/cta_level_PDL_design
ROOT=bench/dsa/results_20260805_b200_production_long_probe_v1
SEGMENT="$(find "$ROOT/failed_segments" -mindepth 1 -maxdepth 1 -type d -name '*.rejected.*' -print -quit)"

jq '{formal,campaign_mode,controls,ordered_matrix,accepted_timing,accepted_workload_timing,accepted_CTA_bracket}' \
  "$ROOT/campaign_contract.json"
jq '{status,measurement_emitted,formal_statistics_requested,warmup,repeats,shape_records,fragment}' \
  "$SEGMENT/manifest.json"
jq . "$SEGMENT/segment_rejection.json"
sha256sum "$ROOT/campaign_binding.json" "$ROOT/campaign_contract.json" \
  "$ROOT/campaign_runner.log" "$SEGMENT/segment_rejection.json" \
  "$SEGMENT/manifest.json" "$SEGMENT/runner.log" "$SEGMENT/gpu_observations.ndjson"

sha256sum bench/dsa/run_production_tier5_fragments.sh bench/dsa/test_production_tier5.py
rg -n 'terminate_active_processes|ACTIVE_CHILD_START_TICKS|ACTIVE_CHILD_PGID|trap.*INT|signal_active_processes' \
  bench/dsa/run_production_tier5_fragments.sh
rg -n 'test_fragment_runner_interrupt_reaps_child_group|test_fragment_runner_monitor_failure_reaps_live_child_group' \
  bench/dsa/test_production_tier5.py
~~~

## 9. 证据入口

- campaign contract：[bench/dsa/results_20260805_b200_production_long_probe_v1/campaign_contract.json](../../bench/dsa/results_20260805_b200_production_long_probe_v1/campaign_contract.json)
- campaign binding：[bench/dsa/results_20260805_b200_production_long_probe_v1/campaign_binding.json](../../bench/dsa/results_20260805_b200_production_long_probe_v1/campaign_binding.json)
- campaign runner log：[bench/dsa/results_20260805_b200_production_long_probe_v1/campaign_runner.log](../../bench/dsa/results_20260805_b200_production_long_probe_v1/campaign_runner.log)
- 永久拒收记录：[segment_rejection.json](../../bench/dsa/results_20260805_b200_production_long_probe_v1/failed_segments/000_glm5.indexshare_fsss.seq1048576.inprogress.o0OZCr.rejected.c6c3a6c9-8b08-4d5e-8c67-44f65fc62eac/segment_rejection.json)
- 被拒收 manifest：[manifest.json](../../bench/dsa/results_20260805_b200_production_long_probe_v1/failed_segments/000_glm5.indexshare_fsss.seq1048576.inprogress.o0OZCr.rejected.c6c3a6c9-8b08-4d5e-8c67-44f65fc62eac/manifest.json)
- 当前 cleanup runner：[bench/dsa/run_production_tier5_fragments.sh](../../bench/dsa/run_production_tier5_fragments.sh)
- cleanup 回归源码：[bench/dsa/test_production_tier5.py](../../bench/dsa/test_production_tier5.py)
