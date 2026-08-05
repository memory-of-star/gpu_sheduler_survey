# Tier 5 native v8 validator-complexity 拒绝审计

| 元数据 | 值 |
|---|---|
| 报告日期 / 实验日期 | 2026-08-05（UTC） |
| 设备 | NVIDIA B200，148 SM，Compute Capability 10.0，driver 580.126.09 |
| 拒绝目录 | `bench/dsa/results_20260805_b200_native_formal_strict_v8/` |
| 最终状态 | `REJECTED`，`accepted_timing=0` |
| 证据等级 | validator 复杂度根因与修复 replay；**不支持任何 v8 性能结论** |

## 1. 执行摘要

v8 的 1M `work_complete_packed_proxy` native GPU 进程已经写出完整 raw log 和
CTA trace：raw 包含 3 条正确路径 validation、1 条 Ceiling wrongness proof、
12 条 warmup、124 条 timed sample、140 条 progress audit 和最终 summary；trace 共
4,325,376 行。GPU monitor 从 `12:11:12.087725` 到 `12:31:15.821431` 运行
1,203.733706 秒，最终为 `PASS`、`errors=[]`、`foreign_processes_detected=false`。

这些时间只是定位“GPU 阶段已结束、CPU validator 随后占用时间”的墙钟证据，
不是 workload latency、throughput 或 rung timing；本报告不提取 v8 raw 中的任何
性能字段。

GPU 进程结束后，冻结的 Python validator 打开 798,726,610-byte trace。
validator log 在 `12:31:15.831143` 创建，仍为空文件；运行约
179.770564 秒后，runner 在 `12:34:15.601707` 进入 post-check，validator 返回
143。`formal_rejection.json` 因此固定：

```text
status=REJECTED
reason=strict_validator_failed
failure_kind=validator
phase=dsa_work_complete_packed_proxy_seq1048576_validator
accepted_timing=0
```

该 sentinel 中的数字字段名为 `native_returncode=143`，但这是通用 step-rejection
writer 的历史字段名。`failure_kind=validator`、`phase=..._validator`、runner 日志和控制流
一致证明 143 属于 validator 阶段，不表示 GPU binary 在产生 raw 时失败。

## 2. 根因：旧 validator 的 `O(Q²D)` 扫描

v8 已完成点的 marker 冻结旧 `validate_dsa_native.py` SHA-256 为
`b39a21ba0bbff16e6f1bf9a0c78ecb692609f5dc7a809e4839ef5412e1b27e9c`。旧实现在
Floor/Wave-Floor 的每个 query 上重新扫描所属 dependency wave，分别求 upstream
trigger/end 极值。Full Floor 的 dependency wave 是整个 query grid，因此复杂度是
`O(Q²D)`，而不是 trace 行数线性。

1M proxy 的 `Q=16,384`、physical degree `D=64`；仅“每 query 重扫全部
producer”就对应 `16,384 × 16,384 × 64 = 17,179,869,184` 次 producer-row
访问，且 trigger 与 end 极值还是分开计算。这是 Python validator 的算法复杂度，
不是 GPU 实验执行的 pair-work 数，两者不得混写。

修复版先按 dependency wave 一次预计算 indexer/topk 的 trigger/end 极值，然后每个
query 只做常数次索引查找。扫描复杂度因而降为 `O(QD)`，并保留 partial-tail
wave 的独立边界，不能用全局极值代替。CPU regression
`test_wave_floor_extrema_are_precomputed_per_wave_with_partial_tail` 同时用 slow reference
对齐 Floor/Wave-Floor，并专门覆盖末尾不满 wave。

## 3. 优化 validator 对冻结 v8 raw/trace 的 replay

本报告使用当前优化 validator（SHA-256
`20aee5c09deb58b4e29686ab16d91d885b22baad4f6468bbdcab5d9fd74c440c`）只读 replay
v8 1M raw/trace，把 JSON 写到 `mktemp` 目录，没有写回 v8 拒绝目录。该次 replay
墙钟 30.853008 秒，返回 0，并重算得到：

```text
status=PASS
errors=0
seq=1048576
repeats=31
samples=124
validations=3
trace_rows=4325376
summary_valid=1
```

临时 replay JSON 的 SHA-256 为
`83a8c7058479d184df86e18c0cd16cec43f8b00986fdce1c4b3c5ae4a57cad8b`；它是可重生的
CPU 修复证据，不是 v8 的 admission artefact。partial-tail regression 也独立返回
`OK`；对应 `test_dsa_pipeline.py` SHA-256 为
`c6277ddc12ac1cea0cdfc97ef00cb94b24bb170aa55b151b59efe26a156a43b6`。

replay `PASS` 只证明：优化后 validator 可以在线性扫描复杂度下对这份冻结
raw/trace 完成同一语义重算。它不会生成 v8 `.done` marker、Profiler sidecar、
unified campaign admission 或新的 GPU 独占证据，也不能删除或覆盖
`formal_rejection.json`。

## 4. 原子拒绝和 v9 独立成功边界

v8 目录在 1M validator 返回非零后已被整轮原子拒绝。先前存在的 4K、32K、
128K `.done` marker 只能说明这些 step 在拒绝前通过；它们不构成可部分接收的
formal campaign。v8 没有完成 1M marker、aggregation、Profiler sidecar 和 unified finalizer，
所有上下文的 `accepted_timing` 一律为 0。

[`native_v9_four_context_formal_20260805.md`](../tier5_dsa/native_v9_four_context_formal_20260805.md)
引用的 v9 目录是新建的独立 campaign；它有自己的 source/binary/validator hash、
GPU 重跑、四点 validator、PTX/resource/SASS 证明、Nsight/NCU 边界和 unified
admission。v9 `campaign_admission.json` 为 `PASS`、`accepted_timing=1`，SHA-256 为
`135db12fc524e629525dc5b404fe9603ac67a6704e54bc27fe5ecc22a6bb7491`。v9 的正式成功不是
对 v8 的 resume、replay 接纳或样本拼接。

## 5. 能成立的结论

1. v8 的 1M GPU 进程已产生完整 raw/trace，直接失败阶段是冻结的 CPU validator。
2. 旧 validator 在 Full Floor 上为 `O(Q²D)`；1M point 的 query/degree 使重复扫描成为主导成本。
3. 优化后的 `O(QD)` validator 可在同一冻结 raw/trace 上 replay `PASS`，且 partial-tail wave 回归通过。
4. v8 正式目录仍永久 `REJECTED`，`accepted_timing=0`。
5. 可引用的 native 正式结果来自独立 v9，而不是重放后的 v8。

## 6. 不能成立的结论

1. 不能引用 v8 raw、summary 或已完成 marker 中的 latency、median、CI、delta 或模式排序。
2. 不能用优化 validator 的 replay `PASS` 把 v8 `accepted_timing` 从 0 改为 1。
3. 不能把 v8 先完成的三点与 v9 1M，或把 v8 1M raw 与 v9 的 Profiler/finalizer 拼接。
4. 不能把 validator 的 `O(Q²D)` CPU 访问数写成 GPU pair work、workload FLOPs 或实测硬件计数。
5. 不能由本拒绝审计推导 production DSA、LLM CTA headroom 或任何最终硬件坐标。

## 7. 可保留、不可复用与 superseded 边界

可保留的只有：v8 失败位置、raw/trace 完整性证据、旧 validator 复杂度根因、
`O(QD)` 极值预计算修复和 partial-tail regression。它们只用于 harness/validator 审计。

不可复用为结论的是 v8 任何 timing、性能排序或部分 formal 样本。优化 replay
supersede 的是 validator 算法，不是 v8 的 admission 状态。后续 v9 也只能作为
新目录的独立证据，不会回填 v8。

## 8. CPU-only 复核命令

以下命令不启动 GPU，且不改写 v8 目录：

```bash
cd /workspace/gpu_sheduler_survey/cta_level_PDL_design
root=bench/dsa/results_20260805_b200_native_formal_strict_v8

jq '{status,reason,failure_kind,phase,accepted_timing,native_returncode}' \
  "$root/formal_rejection.json"

python3 - <<'PY'
from collections import Counter
from pathlib import Path
p = Path("bench/dsa/results_20260805_b200_native_formal_strict_v8/" \
         "dsa_work_complete_packed_proxy_seq1048576.log")
c = Counter(line.split(maxsplit=1)[0] for line in p.open() if line.strip())
expected = {
    "VALIDATION_DSA": 3, "CEILING_PROOF_DSA": 1, "WARMUP_DSA": 12,
    "SAMPLE_DSA": 124, "PROGRESS_DSA": 140, "SUMMARY_DSA": 1,
}
assert all(c[key] == value for key, value in expected.items())
print(expected)
PY

cd bench/dsa
tmp=$(mktemp -d)
python3 validate_dsa_native.py \
  results_20260805_b200_native_formal_strict_v8/dsa_work_complete_packed_proxy_seq1048576.log \
  --trace results_20260805_b200_native_formal_strict_v8/dsa_work_complete_packed_proxy_seq1048576_trace.csv \
  --expected-gpu-uuid GPU-eb98326c-c1dd-1104-c4d7-24decdb06aef \
  --json "$tmp/replay.json"
jq -e '.status == "PASS" and (.errors | length) == 0 and
       .samples == 124 and .validations == 3 and .trace_rows == 4325376' \
  "$tmp/replay.json"
python3 -m unittest -v \
  test_dsa_pipeline.WorkParityContractTests.test_wave_floor_extrema_are_precomputed_per_wave_with_partial_tail
```

## 9. 证据入口与 SHA-256

| 证据 | SHA-256 |
|---|---|
| [`formal_rejection.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/formal_rejection.json) | `2e599c713e45191d3f2a28919f75e1f52e11b6f3731f1934c7b52b44413374cc` |
| [`dsa.log`](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/dsa.log) | `c34ee354aacdb094a2434aebc1e792c5df182f6621f590814d129de61eb16cb7` |
| [1M raw log](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/dsa_work_complete_packed_proxy_seq1048576.log) | `6aa9a21a4dc261f9f3ca879584535447ee98146cc5c428b3550eea5a6cf6a6e2` |
| [1M final trace](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/dsa_work_complete_packed_proxy_seq1048576_trace.csv) | `e8831c34b396da455825325733f49d0895a34c72d27df7cdc43a5edd8c0174db` |
| [1M validator log](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/dsa_work_complete_packed_proxy_seq1048576.validator.log)（空） | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| [1M GPU monitor](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/dsa_work_complete_packed_proxy_seq1048576_gpu_monitor.json) | `878b504c66362688f5d1c1e38af1d223d5d3841c0a2089d26022c03cef76c3cd` |
| [`dsa_build_manifest.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/dsa_build_manifest.json) | `0402e75064fdfd61061635cfbc4dad3d2a0e36e0e436d35f6bfcb4ec072be622` |
| [`dsa_binary_proof.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v8/dsa_binary_proof.json) | `0177092262661db84185ee6225d5a24739169246d673c385a02893c216830eec` |
| [优化 validator](../../bench/dsa/validate_dsa_native.py) | `20aee5c09deb58b4e29686ab16d91d885b22baad4f6468bbdcab5d9fd74c440c` |
| [partial-tail regression](../../bench/dsa/test_dsa_pipeline.py) | `c6277ddc12ac1cea0cdfc97ef00cb94b24bb170aa55b151b59efe26a156a43b6` |
| [v9 unified admission](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/campaign_admission.json) | `135db12fc524e629525dc5b404fe9603ac67a6704e54bc27fe5ecc22a6bb7491` |
