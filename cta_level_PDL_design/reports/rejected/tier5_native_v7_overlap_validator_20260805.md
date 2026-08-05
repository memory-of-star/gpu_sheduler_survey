# Tier 5 native v7 overlap-validator 拒绝审计

- **报告日期 / 实验日期**：2026-08-05
- **设备**：NVIDIA B200，148 SM，compute capability 10.0，driver 580.126.09
- **证据等级**：正式轮拒收与语义根因审计；只支持 harness/validator 结论，**不支持任何性能结论**
- **最终状态**：`REJECTED`，`accepted_timing=0`

## 执行摘要

`bench/dsa/results_20260805_b200_native_formal_strict_v7` 是永久拒收目录。4K 点完成了
4 个 admission invocation、12 个 warmup invocation 和 124 个 timed invocation，但 native
进程最终返回 2；`formal_rejection.json` 固定记录
`reason=native_step_failed`、`phase=dsa_exact_seq4096` 和 `accepted_timing=0`。该目录没有
可复用的 `.done` marker，也没有通过 aggregation、Profiler sidecar 或 unified campaign
admission。**目录内所有 latency、median、CI、rung delta 与性能排序一律不得引用。**

严格 validator 共报告 257 个错误：248 个逐样本 overlap-field 错误、8 个 final-trace
overlap-field 错误，以及 1 个 `SUMMARY_DSA valid=0` 错误。根因不是 GPU 无前进、数据错误或
外部进程污染，而是 v7 把 grid-PDL Floor/Wave-Floor 的合法时序错误定义成
`consumer_start < max(upstream_ready)`。programmatic dependent grid 可以在所有 upstream
trigger 之后才开始执行，同时仍在 upstream tail 结束前重叠；这种合法情况被旧定义全部判成
“no overlap”。

修复后的语义边界是：

```text
tail overlap       := consumer_start < max(upstream_end)
dependency safe    := consumer_dep   >= max(upstream_programmatic_trigger)
```

upstream `end` 只定义可重叠 tail 的窗口，**不是**依赖安全边界；安全边界是数据 ready 后发出的
programmatic trigger。消费者在 trigger 前或 trigger 后进入都可能合法，只要其 dependency point
不早于所有相关 trigger。Impl 的逐身份 `start < row_ready` / `dep >= row_ready` 证明没有放宽。

## 程序实际执行了什么

v7 在同一进程中按 cyclic Latin-4 顺序执行 `floor`、`wave_floor`、`impl`、`ceiling`：

- `floor`：单次三节点 programmatic CUDA Graph，完整 grid-PDL 基线；producer 在数据 ready 后
  trigger。
- `wave_floor`：有界 query-wave 的三节点 programmatic CUDA Graph，对齐 Impl 的分波控制成本；
  producer 同样在 ready 后 trigger。
- `impl`：消费者先进入，随后 producer 启动；消费者按生产者身份 acquire epoch flag。
- `ceiling`：同样的分波与 launch order，但不等待、不发布 readiness；只要求持续证明结果错误。

每个 invocation 都保留 `%globaltimer` CTA trace、完整 history-load 计数、poison、正确路径全元素
校验或 Ceiling wrongness 证明。4K native 进程确实走完 140 个 invocation；失败发生在旧 overlap
admission 汇总，而不是 watchdog timeout。GPU monitor 为 `PASS`，观测到 9 个采样点、无 foreign
process；PTX/resource/SASS proof 和 source-to-binary build provenance 也均为 `PASS`。这些事实只能
定位拒收原因，不能恢复 timing admission。

## 257 个错误如何产生

正式轮有 31 个 rep。每个 rep 的 `floor` 与 `wave_floor` 都被旧 validator 检查四个字段：
`topk_early`、`attention_early`、`topk_waited`、`attention_waited`。

```text
31 reps × 2 PDL modes × 4 incorrect overlap fields = 248
final trace × 2 PDL modes × 4 incorrect fields      =   8
SUMMARY_DSA valid=0                                  =   1
                                                        ---
                                                        257
```

旧 native 代码也使用同一错误定义决定 `overlap_ok`，因此正式模式把 summary 写成 `valid=0` 并
返回 2。FAST 使用 `--allow-short`，旧 native 与 validator 都绕过了这条 overlap admission；所以
v7 smoke 能通过并不能证明正式 overlap predicate 正确。这是 smoke 覆盖缺口，不是把正式失败
改写成可接受结果的理由。

## 从冻结 v7 trace 独立重算

不读取任何 timing 字段，只按最终 rep 的 CTA timestamp 重算依赖边界，4K trace 得到：

| mode | stage edge | `start >= max(trigger)` | `start < max(end)` | `dep >= max(trigger)` |
|---|---|---:|---:|---:|
| Floor | indexer → topk | 64 / 64 | 64 / 64 | 64 / 64 |
| Floor | topk → attention | 64 / 64 | 64 / 64 | 64 / 64 |
| Wave-Floor | indexer → topk | 64 / 64 | 64 / 64 | 64 / 64 |
| Wave-Floor | topk → attention | 64 / 64 | 64 / 64 | 64 / 64 |

因此冻结 trace 同时证明：消费者在所有相关 trigger 之后获得执行、在 upstream tail 退休前开始，
且 dependency point 没有越过安全边界。旧 `start < ready` 得到 0，不表示没有 overlap；它只表示
dependent grid 没有在获得合法 launch eligibility 之前执行。

这项重算用于证明 validator 根因，**不改变**该 formal 目录的永久拒收状态，也不准许从同一日志
提取任何 timing。

## v7b diagnostic 的边界

`bench/dsa/results_20260805_b200_native_forward_progress_diag_v7b` 是一次 32K、1 repeat、1 warmup
的非正式诊断。其严格语义 JSON 为 `PASS`，可证明 v7 bounded-wave entry gate 消除了 v6 的 32K
Impl forward-progress deadlock，并且 correct paths、Ceiling wrongness 与 trace completeness 在该
诊断调用中成立。

它不能用于性能报告，原因包括：repeat 数不足、不是 exact formal matrix、没有 unified campaign
admission，且其 `--allow-short` 路径没有执行正式 overlap admission。v7b 只能作为 forward-progress
定位证据；它既不能补齐 v7 formal，也不能与其他目录拼接。首个不规范
`results_20260805_b200_native_forward_progress_diag_v7` 仍是 validator `FAIL`，不属于本报告的正向
证据。

## v7 修复快照与后继 v8/v9 边界

下列三个哈希是启动 v8 时冻结的修复快照，不是当前工作树文件的哈希；后继 v8/v9 已继续
演进这些文件。该快照同步修改 native trace 分析、独立 Python validator 和 CPU tamper test：

- `bench/dsa/dsa_native.cu` SHA-256
  `62d14f9ab9b5342e12e77531460e0ce129d7b6955efd95bf986cf1c5b3405a28`；
- `bench/dsa/validate_dsa_native.py` SHA-256
  `b39a21ba0bbff16e6f1bf9a0c78ecb692609f5dc7a809e4839ef5412e1b27e9c`；
- `bench/dsa/test_dsa_pipeline.py` SHA-256
  `05d20eb5d34757c6d97369ce121cc458672ad8a320e8b394730fa197cd56dd8d`。

source 与 validator 都分别计算每个 Floor/Wave-Floor dependency wave 的
`max(t_trigger)` 和 `max(t_end)`，并冻结如下自描述 contract：

```text
floor_overlap_metric=consumer_start_before_upstream_kernel_end
floor_dependency_metric=consumer_dep_after_upstream_programmatic_trigger
```

当时的 CPU test `test_floor_trace_uses_trigger_for_safety_and_end_for_overlap` 覆盖“trigger 后开始但 tail 内
重叠”的合法样本，并把 `dep` 篡改到 trigger 前，要求 `safety_failures=1`。Impl 分支仍独立使用
per-row ready/acquire 规则。

后继结果没有恢复 v7。v8 完成 GPU raw/trace 后，旧 Python validator 的 `O(Q²D)` 重复扫描又使
该轮原子拒收，`accepted_timing=0`；优化后的 `O(QD)` replay `PASS` 只证明 validator 修复，不能
恢复 v8 timing。v9 随后在全新目录独立完成四点矩阵、PTX/SASS、Nsight、NCU 权限分类和 unified
admission，最终 `PASS`、`accepted_timing=1`。详见
[`tier5_native_v8_validator_complexity_20260805.md`](tier5_native_v8_validator_complexity_20260805.md)
与 [`native_v9_four_context_formal_20260805.md`](../tier5_dsa/native_v9_four_context_formal_20260805.md)。
v9 的成功只接纳 v9 本身；v7/v8 仍永久拒收。

## 能成立的结论

1. v7 formal 的最终状态是永久 `REJECTED`，正式 timing admission 为 0。
2. 4K native 没有因 watchdog、GPU contamination、binary proof 或 correctness 失败而退出；直接
   失败原因是旧 overlap predicate 令 summary `valid=0`。
3. 对 grid-PDL Floor/Wave-Floor，trigger 是依赖安全边界，kernel end 是 tail-overlap 边界；
   `start < ready` 不是合法的 launch-overlap 要求。
4. 冻结 v7 final trace 在正确边界下具有完整的 trigger safety 与 tail overlap；这足以证明
   validator 语义错误，但不足以恢复该轮 timing。
5. v7b 支持“32K bounded-wave forward progress 已观察到”这一诊断性陈述。

## 不能成立的结论

1. 不能引用 v7 formal 日志中的任何 latency、median、CI、百分比或模式排序。
2. 不能把 v7 的已完成 4K 点移植到 v8、与 v7b 拼接，或用当前 validator 重放后重新接纳。
3. 不能声称 v7b 是正式 32K 结果，或用其单次数据估计正式运行时间/性能。
4. 不能用后继 source、CPU replay 或 v9 成功反向恢复 v7/v8 formal。
5. 不能把 `dep >= upstream_end` 当作 PDL 安全要求；这会把 ready 后的 producer tail 错误变成
   必须等待的依赖工作，并抹掉实验要测的重叠窗口。

## 可保留、不可复用与 Superseded 边界

后继修复 supersede 的只是 v7 的 overlap **定义与实现**，不是 v7 formal 的 admission 状态。
`formal_rejection.json` 是永久 sentinel；runner 会在任何 resume、rebuild 或 GPU 动作前拒绝复用
该目录。v9 已独立通过，本报告仍作为 v7 拒收审计保留。

v7 smoke 与 v7b diagnostic 可继续作为历史的 smoke/forward-progress 证据，但 Tier 5 headline
中的 native synthetic 部分只取 v9 formal；不得与 v7/v8 拼接。production exact-26 是另一条
独立准入链，不由 native v9 自动补齐。

## 复现与审计命令

以下命令均为 CPU/文件审计，不启动 GPU：

```bash
cd /workspace/gpu_sheduler_survey/cta_level_PDL_design

# 永久拒收 sentinel 与关键证据哈希
python3 - <<'PY'
import json
from pathlib import Path
p = Path("bench/dsa/results_20260805_b200_native_formal_strict_v7")
r = json.loads((p / "formal_rejection.json").read_text())
assert r["status"] == "REJECTED"
assert r["accepted_timing"] == 0
assert r["native_returncode"] == 2
assert not any(p.glob("*.done"))
print(r["status"], r["accepted_timing"], r["reason"], r["phase"])
PY

# 257-error 分解；不打印或解析任何 timing
python3 - <<'PY'
import json
from pathlib import Path
p = Path("bench/dsa/results_20260805_b200_native_formal_strict_v7/")
e = json.loads((p / "dsa_exact_seq4096_validation.json").read_text())["errors"]
sample = sum(x.startswith("sample ") and "has no" in x for x in e)
final = sum(x.startswith("final trace ") and "has no" in x for x in e)
summary = sum("summary validity mismatch" in x for x in e)
assert (len(e), sample, final, summary) == (257, 248, 8, 1)
print(len(e), sample, final, summary)
PY

# 修复定义的正例与 dep-before-trigger 篡改负例
cd bench/dsa
python3 -m unittest -v \
  test_dsa_pipeline.WorkParityContractTests.test_floor_trace_uses_trigger_for_safety_and_end_for_overlap

# 文档完整性
cd ../..
python3 codex/check_docs.py
```

## 证据入口与 SHA-256

| 证据 | SHA-256 |
|---|---|
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/formal_rejection.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/formal_rejection.json) | `3acffd6554ba557c8043bb146130518ce59815b5ce26696a5f2f87b94a7e0635` |
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096.log`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096.log) | `9eb490e5c05687c994dc2c1ab9d7446219fe05a80cb5abc9bd673c78dced3b0f` |
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096_trace.csv`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096_trace.csv) | `6a06189f49b9f8f934627d28e2b87a03f9a491b00b41b43d75905191941c8ea6` |
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096_validation.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096_validation.json) | `a04177eec6b3c72d2a7673854bcee9b01521c6dc0d55b157984263bbc9fcc4f5` |
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096.validator.log`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096.validator.log) | `4f780d24fab1baefa27d31934fde74d4dac4d367b6dec6cbba8429ed6ea6be75` |
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_build_manifest.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_build_manifest.json) | `3a34f110451b6c197c397ebde7afcb1850805c803e4b5cb65c170098b23fe8e6` |
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_binary_proof.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_binary_proof.json) | `6314eaa04fa75a10efe7fad98cb0c4472c778c8e14d2b4e3fad3de58b5842973` |
| [`bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096_gpu_monitor.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v7/dsa_exact_seq4096_gpu_monitor.json) | `ca1896347b0f5bb4a8d8895d56a0648435bb1e25ca4ff5d952dac3a4e6c9ae8f` |
| [`bench/dsa/results_20260805_b200_native_forward_progress_diag_v7b/diagnostic.log`](../../bench/dsa/results_20260805_b200_native_forward_progress_diag_v7b/diagnostic.log) | `3fe465d225482d7ab1f425c756f78ebbc7db85b583c83e3307c5dcbf3b105e95` |
| [`bench/dsa/results_20260805_b200_native_forward_progress_diag_v7b/trace.csv`](../../bench/dsa/results_20260805_b200_native_forward_progress_diag_v7b/trace.csv) | `379b92131154ba41d449e988bfb34f3d02ee2df1102d1e46da034f0209edc104` |
| [`bench/dsa/results_20260805_b200_native_forward_progress_diag_v7b/validation.json`](../../bench/dsa/results_20260805_b200_native_forward_progress_diag_v7b/validation.json) | `5d74ac23fd116dc0d00c50ba882e459f7b001427b05ac830f2acf6e0aad17c08` |
| `bench/dsa/dsa_native.cu`（v8 启动快照；非当前工作树） | `62d14f9ab9b5342e12e77531460e0ce129d7b6955efd95bf986cf1c5b3405a28` |
| `bench/dsa/validate_dsa_native.py`（v8 启动快照；非当前工作树） | `b39a21ba0bbff16e6f1bf9a0c78ecb692609f5dc7a809e4839ef5412e1b27e9c` |
| `bench/dsa/test_dsa_pipeline.py`（v8 启动快照；非当前工作树） | `05d20eb5d34757c6d97369ce121cc458672ad8a320e8b394730fa197cd56dd8d` |
