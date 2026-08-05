# Tier 5 production exact-26 v1/v2 永久拒收报告

| 元数据 | 值 |
|---|---|
| 报告日期 | 2026-08-05（UTC） |
| 实验日期 | 2026-08-05（UTC） |
| v1 根目录 | `bench/dsa/results_20260805_b200_production_formal_exact26_v1` |
| v2 根目录 | `bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g` |
| 设备 | 单卡 NVIDIA B200，UUID `GPU-eb98326c-c1dd-1104-c4d7-24decdb06aef`；保存的 identity 没有给出 SM 数或 Compute Capability，故不补写 |
| 证据等级 | **REJECTED / OPERATIONAL-DIAGNOSTIC EVIDENCE ONLY**；不能支持任何 timing、CTA bracket 或 headroom 结论 |

## 1. 执行摘要

两个 exact-26 根目录均已**永久拒收**。两份 `formal_rejection.json` 都固定为
`status=REJECTED`、`accepted_timing=0`、`accepted_workload_timing=0`、
`accepted_CTA_bracket=0` 和 `sealed_fragment_timing_reusable=false`。这个边界覆盖整个
根目录，包括中断前已经 sealed 的 9 行；不能从其中挑选“看起来完整”的行重新组成
另一轮 campaign。

- v1 使用 2,048 MiB 的 full-N logits 上限。1M 点只能取 512 个 query rows/chunk，
  即 `1,048,576 / 512 = 2,048` chunks。执行到
  `deepseek_v32.operator_chain.seq1048576` 时被中断并以 rc130 原子移入
  `failed_segments/`；该低效分区随后被 16 GiB 路径取代。
- v2 将上限提高到 16,384 MiB，使同一 1M 点成为 4,096 rows/chunk，即
  `1,048,576 / 4,096 = 256` chunks。它仍只 sealed 了前 9 行；进入同一个 1M
  operator-chain 行后，因用户授权把剩余实验压缩到一小时 compact scope 而中断，
  失败段同样以 rc130 永久拒收。

因此 exact-26 没有完成，也没有 campaign-level final admission。两轮的正式
timing 都是 0；production harness 本身又明确记录 `headroom_defined=false`、
`headroom_pct=null`，故不存在可复用的 LLM/CTA headroom。

后继 compact campaign 使用全新根目录
`bench/dsa/results_20260805_b200_production_compact_formal_v1_16g`、全新 contract/source
manifest 和全新 fragment invocation UUID；它不是从 v1 或 v2 的 sealed rows
拼接而成。compact 基础 contract 的 14 行范围是两个模型、4K/128K、三个 workload
以及每模型一行 MoE，即 `2 * (2 * 3 + 1) = 14`。它不是 exact-26；其基础 contract
仍诚实标为 `formal=false`、`is_exact_formal_matrix=false`，最终是否通过只能由独立的
compact-specific admission 决定，本拒收报告不预先宣告其结果。

## 2. 原 exact-26 contract 与实际执行

两个 frozen contract 的 ordered matrix 相同，矩阵 SHA-256 都是
`da399ea22839275c02ae98febebe98678ef99c18fbdc37248a63f445f4a7192c`。它包含两个
模型（DeepSeek-V3.2、GLM-5）、四个序列长度（4K、32K、128K、1M）、三个
workload（`operator_chain`、`single_layer`、`indexshare_fsss`），外加每模型一行
`moe32`：

```text
2 models * (4 sequence lengths * 3 workloads + 1 MoE row/model) = 26 rows
```

两轮都请求 warmup 5 次、timed repeats 31 次、seed `20260805`；off/on 在同一进程
相邻配对并按 repeat 交替顺序，预定统计量是 median、bootstrap 95% CI 和 paired
median-delta CI。每个持续时间由 CUDA events 计算。以上只是 frozen contract 的
预定方法；由于两轮均未完成并被 campaign 级拒收，本报告不重算、转录或引用任何
结果文件中的 latency 数字。

两份 `campaign_runner.log` 都证明实际只 sealed 了相同的前 9 行：DeepSeek-V3.2
在 4K、32K、128K 的三个 workload，即 `3 * 3 = 9`。ordinal 9
`deepseek_v32.operator_chain.seq1048576` 是两轮唯一启动但未 sealed 的下一行；其后
16 行没有执行。被中断 stage 已分别移至：

- v1：`failed_segments/009_deepseek_v32.operator_chain.seq1048576.inprogress.926sFi.rejected.a941e7c2-2f00-4bfa-b8b4-3aafb1513443`
- v2：`failed_segments/009_deepseek_v32.operator_chain.seq1048576.inprogress.woxrwB.rejected.5dac09b1-e341-4068-b630-71bc4fbb86e7`

两份 `segment_rejection.json` 都记录
`reason=runner_interrupted_or_unhandled_failure_rc_130`，并把 segment 自身的三个
`accepted_*` 固定为 0。根级 rejection 再把前 9 个 sealed fragments 一并排除，
因此“segment 被拒收但前 9 行仍可计时”不是允许的解释。

## 3. v1：2 GiB 分区被取代

v1 contract SHA-256 是
`d4c2d9be4ee93eb8fb3cfca962900d0c2fbd6204799be729ceaa6783db480f83`，source-manifest
SHA-256 是 `1f242e1f8123de7c3d75c8e0894d49ea17ef3bbe21d7e3841e0b4715c0ed3319`。
其 frozen `production_tier5.py` SHA-256 是
`c33801fbd9d3878ef6beb42e55d91166fa88bf69ffd6b4afa91d514ba85f4442`。

full-N logits 为 FP32；在 `seq=1,048,576` 时，512 个 query rows 的缓冲为：

```text
512 * 1,048,576 * 4 bytes = 2,147,483,648 bytes = 2,048 MiB
```

所以 1M query 必须分成 2,048 chunks。根级拒收原因明确写为
`operational_partition_superseded_after_2048_MiB_full_N_logits_budget_forced_2048_query_chunks_at_1M`。
这是实现尺寸与运行编排的诊断证据，不是“1M 性能慢多少”的 timing 证据；中断段
也不能用于估算完成百分比或外推总耗时。

## 4. v2：16 GiB 路径因一小时 compact scope 停止

v2 contract SHA-256 是
`3a56f9e03ecd00f380f52f7f556f95956b1cf9d52058df84b558f763c701da2b`，source-manifest
SHA-256 是 `de582cd4422e88c1b4f530d63aa96ec6ac1abe97a82e2ea81d7096427b0d92b2`。
其 frozen `production_tier5.py` SHA-256 是
`facff3e00a61201982d723f03937100fe6c7864d86b471ae56938a43d7878e4a`。

v2 把同一 FP32 full-N logits 缓冲扩至：

```text
4,096 * 1,048,576 * 4 bytes = 17,179,869,184 bytes = 16,384 MiB
```

因此 1M 点的 partition 从 2,048 chunks 降为 256 chunks。v2 的拒收不是把 v1
的数值“修正后继续”；它具有独立 contract、source manifest、campaign fingerprint
和 fragment UUID。根级 rejection 的原因是
`user_requested_compact_formal_within_one_hour`，并明确把 4K/128K 的两模型、三
workload 加 MoE 定义为后继 scope，同时把 32K 与 1M timing 排除在 compact
结论之外。范围变更不能将已中断的 exact-26 降格为一个可接受的 partial campaign，
所以 v2 也必须整体拒收。

## 5. compact 与两轮 exact-26 的隔离

compact 根的 frozen contract SHA-256 是
`3f3c604d9892f523bde2ccb51bad8c9808fc0d25ad4cb1a31c060964486f1029`，source-manifest
SHA-256 是 `5c5f80900529728d5f57a8eb78a6d55c10ffb3772bf0a46be64541fa0778c89c`，
`production_tier5.py` SHA-256 是
`078d6d95d45fed6e2fc2d4c02cd6912ef8cd72d3349d6b098e686194eefa0525`。这些值均不同于
v1/v2。compact 的 ordinal-0 completion marker 使用 invocation UUID
`170f7432-2747-4bde-b15b-392f1f071cb6`；v1/v2 的同一逻辑行分别使用
`14376ad0-b3d3-4652-95a9-843dae3525a9` 和
`46a5cd04-aedf-411b-b66d-da4f4dedbad3`。这证明后继行在新根中重新执行，而不是复制
旧 completion marker。

这个隔离规则是单向且永久的：未来 compact admission 即使 PASS，也只准入其自身
contract/source/root 下的新样本；它不会反向把 v1/v2 的 `accepted_*` 从 0 改为 1，
也不会使 exact-26 变成已完成。

## 6. 哪些证据可复用

可复用范围仅限**审计与实现诊断**：

1. 两份 frozen contract 中的 ordered matrix、控制参数、shape formulas、源码与包
   版本哈希，可用于还原当时计划执行的语义。
2. `campaign_binding.json` 中的 B200 UUID，以及各 row 的 identity/lease/monitor
   文件，可用于核对运行归属和排查环境问题；它们本身不准入 timing。
3. `campaign_runner.log`、根级 `formal_rejection.json` 与 failed segment 的
   `segment_rejection.json`，可用于证明 sealed 行数、失败行、rc130 和永久拒收
   边界。
4. v1 的 2 GiB → 512 rows/chunk → 2,048 chunks，以及 v2 的 16 GiB →
   4,096 rows/chunk → 256 chunks，可作为后续实现的内存/分区设计依据。
5. 前 9 个 row 目录可以原样保留为历史审计材料，用来证明“哪些 fragment 曾经
   sealed”；这种物理保留不等于允许复用其中的测量或 correctness 结论。

## 7. 哪些不可复用

以下内容一律不得进入任何正式或 compact 结论：

1. v1/v2 前 9 个 sealed rows 中的 `samples.jsonl`、`result.json`、summaries、medians、
   bootstrap CI、paired deltas、latency、throughput 或 off/on 排序。
2. 前 9 行的 correctness PASS 或 fragment PASS，不能作为 compact 行的替代证明，
   不能复制 completion marker，不能跳过 compact 的重新执行和重新校验。
3. ordinal-9 rejected segment 中的任何中途输出、GPU observation 数量、已完成 chunk
   数、运行时间或完成时间外推。
4. 跨 v1/v2/compact 拼接样本、拼接 rows、合并 repeat，或用 v1/v2 填补 compact
   未执行点；三者的 contract/source/fingerprint/UUID 边界不可跨越。
5. 任何 CTA Floor/Impl/Ceiling bracket、CTA headroom 或 LLM headroom。production
   contract 本就没有 CTA Impl/真正 unordered Ceiling，并明确
   `tier5_bracket_admitted=false`、`headroom_defined=false`。
6. “exact-26 已完成”“32K 或 1M timing 已被 compact 准入”或“GLM-5 rows 已在
   exact-26 中执行”的说法。

## 8. 能成立与不能成立的结论

能成立的结论是：两份 contract 都精确描述 26 行正式矩阵；两轮各 sealed 9 行并在
同一 ordinal-9 行以 rc130 中断；v1 揭示了 2 GiB 分区导致 2,048 chunks 的操作性
问题；v2 验证 contract 层面的 16 GiB/256-chunk 重分区，但随后按用户授权的一小时
范围变更整体拒收；后继 compact 使用独立的 14 行 contract、source manifest、根目录
和 invocation UUID。

不能成立的结论是：任一 exact-26 run 有可发布 timing、完成了 formal matrix、证明了
32K/1M 性能趋势、给出了 CTA bracket 或定义了 headroom。也不能因为前 9 行的局部
artefacts 看似完整，就绕过根级 `sealed_fragment_timing_reusable=false`。

## 9. 关键证据 SHA-256

| artefact | SHA-256 |
|---|---|
| v1 `campaign_contract.json` 文件 | `13126551d1cae34e90aeab8ec54b8cc5d8022c12e56f4179216e17122180a2ed` |
| v1 `campaign_binding.json` | `be7c86817f3a1e6438bb06f4f3b982810e830bf13ed91af07af4dfcdbb2ad767` |
| v1 `campaign_runner.log` | `18606821e42ae9ac5199c37811587261c98b5e174e5ab4e2f2569a97ad939845` |
| v1 `formal_rejection.json` | `2e704302000aca1e0e8acbb87c236e4187de529fed58c1e24f9be37585c8de37` |
| v1 rejected `segment_rejection.json` | `1a8d7fb5d462e07e067af80ee973c7e7de6f8995388d89d23c1863dfb4c3aa2d` |
| v2 `campaign_contract.json` 文件 | `1dda28f6dde3e6503d7c99030cb8d2819b9da072098616fa7bb1e3949ba54a1b` |
| v2 `campaign_binding.json` | `ce115e64c1a37707435dc615926a7b836b418f26b6f3e9238ad11c4dc07a9e3d` |
| v2 `campaign_runner.log` | `a9655403752e118e414c404068de885fe7f0d57653424ed74317756fa114af6d` |
| v2 `formal_rejection.json` | `091f979054ca21f1e124e977d6d2e571b3718ce536875761cc35f0b9cb8c6c50` |
| v2 rejected `segment_rejection.json` | `28c92834e78db24300c200153aed0b2519b4d00b098917cb26515b20154d742b` |

`campaign_contract.json` 的“文件 SHA-256”和其内部 canonical `contract_sha256` 是不同
概念；本表列前者，§3/§4 同时给出了内部 canonical hash。

## 10. CPU-only 复核命令

以下命令只读 JSON/文本并计算哈希，不启动 benchmark，也不需要 GPU：

~~~bash
cd /workspace/gpu_sheduler_survey/cta_level_PDL_design
V1=bench/dsa/results_20260805_b200_production_formal_exact26_v1
V2=bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g
COMPACT=bench/dsa/results_20260805_b200_production_compact_formal_v1_16g

for ROOT in "$V1" "$V2"; do
  jq '{status,reason,completed_fragment_count_before_rejection,failed_row_id,
       accepted_timing,accepted_workload_timing,accepted_CTA_bracket,
       sealed_fragment_timing_reusable,superseded_controls,superseded_by}' \
    "$ROOT/formal_rejection.json"
  find "$ROOT/rows" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  jq '{status,reason,row_id,ordinal,invocation_uuid,
       accepted_timing,accepted_workload_timing,accepted_CTA_bracket}' \
    "$ROOT"/failed_segments/*/segment_rejection.json
done

# 应输出 9、9；这只是 sealed-directory 数量，不是 accepted row 数量。
find "$V1/rows" -mindepth 1 -maxdepth 1 -type d | wc -l
find "$V2/rows" -mindepth 1 -maxdepth 1 -type d | wc -l

# 复核 1M 分区：v1 为 512/2048，v2 为 4096/256。
for ROOT in "$V1" "$V2"; do
  jq '.shape_records[] |
      select(.model=="deepseek_v32" and .seq==1048576) |
      {model,seq,query_chunk_tokens,num_query_chunks,
       fp32_logits_chunk_bytes:.tensor_bytes.fp32_logits_chunk}' \
    "$ROOT/campaign_contract.json"
done

# compact 与 v1/v2 的 contract/source manifest 必须不同。
for ROOT in "$V1" "$V2" "$COMPACT"; do
  jq '{contract_sha256,source_manifest_sha256,row_count,formal,
       is_exact_formal_matrix,controls}' "$ROOT/campaign_contract.json"
done

# 已 sealed fragment 的 invocation UUID 集合之间必须无交集；comm 无输出才通过。
comm -12 \
  <(jq -r '.invocation_uuid' "$V1"/rows/*/fragment.done.json | sort) \
  <(jq -r '.invocation_uuid' "$COMPACT"/rows/*/fragment.done.json | sort)
comm -12 \
  <(jq -r '.invocation_uuid' "$V2"/rows/*/fragment.done.json | sort) \
  <(jq -r '.invocation_uuid' "$COMPACT"/rows/*/fragment.done.json | sort)

sha256sum \
  "$V1/campaign_contract.json" "$V1/campaign_binding.json" \
  "$V1/campaign_runner.log" "$V1/formal_rejection.json" \
  "$V1"/failed_segments/*/segment_rejection.json \
  "$V2/campaign_contract.json" "$V2/campaign_binding.json" \
  "$V2/campaign_runner.log" "$V2/formal_rejection.json" \
  "$V2"/failed_segments/*/segment_rejection.json
~~~

## 11. 证据入口

- v1 contract：[bench/dsa/results_20260805_b200_production_formal_exact26_v1/campaign_contract.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v1/campaign_contract.json)
- v1 binding：[bench/dsa/results_20260805_b200_production_formal_exact26_v1/campaign_binding.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v1/campaign_binding.json)
- v1 runner log：[bench/dsa/results_20260805_b200_production_formal_exact26_v1/campaign_runner.log](../../bench/dsa/results_20260805_b200_production_formal_exact26_v1/campaign_runner.log)
- v1 根级永久拒收：[bench/dsa/results_20260805_b200_production_formal_exact26_v1/formal_rejection.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v1/formal_rejection.json)
- v1 failed segment 拒收：[segment_rejection.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v1/failed_segments/009_deepseek_v32.operator_chain.seq1048576.inprogress.926sFi.rejected.a941e7c2-2f00-4bfa-b8b4-3aafb1513443/segment_rejection.json)
- v2 contract：[bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/campaign_contract.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/campaign_contract.json)
- v2 binding：[bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/campaign_binding.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/campaign_binding.json)
- v2 runner log：[bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/campaign_runner.log](../../bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/campaign_runner.log)
- v2 根级永久拒收：[bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/formal_rejection.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/formal_rejection.json)
- v2 failed segment 拒收：[segment_rejection.json](../../bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/failed_segments/009_deepseek_v32.operator_chain.seq1048576.inprogress.woxrwB.rejected.5dac09b1-e341-4068-b630-71bc4fbb86e7/segment_rejection.json)
- compact 独立 contract：[bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/campaign_contract.json](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/campaign_contract.json)
- compact 独立 binding：[bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/campaign_binding.json](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/campaign_binding.json)
