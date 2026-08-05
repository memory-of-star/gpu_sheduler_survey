# Tier 5 native v9：四上下文正式实验报告

- **实验日期 / 报告日期**：2026-08-05（UTC）
- **设备**：NVIDIA B200，148 SM，Compute Capability 10.0，driver 580.126.09
- **正式证据目录**：[`bench/dsa/results_20260805_b200_native_formal_strict_v9/`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/)
- **最终状态**：`PASS`，`accepted_timing=1`
- **证据边界**：native synthetic work-complete dependency proxy；4K/32K 为 exact CTA mapping，128K/1M 为 work-complete packed proxy

## 执行摘要

v9 是本轮第一个通过四点 timing matrix、逐点独立 validator、build provenance、PTX/resource/SASS
证明、Nsight Systems sidecar、NCU 权限分类、GPU 独占监控和统一 campaign finalizer 的 Tier 5
native formal 目录。统一 admission 的 `status=PASS`、`errors=[]`、`accepted_timing=1`；四个 workload
均为 3 次 warmup、31 次 timed repeat、每次 4 个 rung，合计每点 124 个正式样本。rung 顺序按
cyclic Latin-4 轮换，同一 workload 进程内相邻测量。

正式结果没有显示 Impl 更快：在四个上下文上，Wave-Floor→Impl 的配对 delta 均为负，即 Impl
比匹配 wave protocol 的 PDL Floor 慢 17.75%–26.80%。Full Floor→Impl 也全部为负，但这项差值
包含 full-grid 与 bounded-wave 调度/控制差异，是 **mechanism envelope**，不是纯 CTA wait
headroom。Ceiling 在每点 31/31 个正式样本中都观察到错误；它是故意移除等待与发布的 unsafe
rung，不是可实现档位，也不能用来发布 production 或 LLM CTA headroom。

## 四个 rung 实际执行什么

| rung | 路径 | 依赖与正确性角色 |
|---|---|---|
| Full Floor | 单次三节点 programmatic CUDA Graph，完整 producer/topk/attention grid | topk/attention 走 `griddepcontrol.wait`；producer 在数据 ready 后 trigger；正确路径 |
| Wave-Floor | 同样的三节点 programmatic graph，按 49 个 query 的有界 wave 执行 | 与 Impl 对齐 wave 边界和 stage priority；仍走 grid-PDL wait；正确路径 |
| Impl | attention→topk→indexer 的有界 wave、逐 producer identity epoch flag | consumer 先进入，按身份 acquire readiness；正确路径 |
| Ceiling | 与 Impl 相同 wave、launch order 和 stage priority，但不等待、不发布 readiness | `unsafe_not_validated`；必须持续观察错误，只保留其时间，不形成安全实现结论 |

计时器是 device `%globaltimer`，范围为 `first CTA start → last CTA end`。第一次 CTA 之前的 host
launch 不在计时内；Full Floor 的单 graph submission 与其他 rung 的 wave submission 不同，runner
明确记录 `host_submission_path_differs=1` 和
`floor_graph_vs_impl_stream_submission=outside_timer_not_normalized`。因此不能把 Full Floor 与 Impl
之差窄化成某一条 CTA wait 指令的成本。

### Trigger-ordered tail overlap 的精确定义

Floor/Wave-Floor 的 trace admission 使用：

```text
tail overlap    := consumer_start < max(upstream_kernel_end in dependency wave)
dependency safe := consumer_dep   >= max(upstream_programmatic_trigger in dependency wave)
```

这是 **trigger-ordered overlap**：消费者的 dependency point 不早于所有相关 programmatic trigger，
同时消费者在 upstream tail 结束前开始。`topk_waited` / `attention_waited` 是这个布尔语义的计数，
不是消费者阻塞了多少纳秒，也不是 wait 指令的 duration。Impl 仍使用逐 query/producer identity 的
`start < row_ready` 与 `dep >= row_ready`，没有把安全边界放宽成 wave/global 聚合。

## 正式 timing

每格为 `median ms [95% bootstrap CI]`。每个 rung 各 31 个样本；CI 使用冻结实现中的 2,000 次
确定性 bootstrap。128K/1M 的数值只代表 packed proxy，不能与 4K/32K exact 点合并拟合成真实
长上下文模型曲线。

| 上下文 | mapping | Full Floor | Wave-Floor | Impl | Ceiling（unsafe） |
|---:|---|---:|---:|---:|---:|
| 4K | exact | 0.826592 [0.826336, 0.826912] | 1.271840 [1.269984, 1.272992] | 1.497568 [1.493728, 1.501728] | 1.161536 [1.161024, 1.163616] |
| 32K | exact | 24.903584 [24.897376, 24.910592] | 28.050816 [28.045280, 28.055360] | 34.333600 [34.315648, 34.394432] | 27.755552 [27.738912, 27.769792] |
| 128K | work-complete packed proxy | 29.755584 [29.722688, 29.809344] | 44.118048 [44.109472, 44.127104] | 53.168864 [53.130464, 53.212832] | 42.455680 [42.304032, 42.667136] |
| 1M | work-complete packed proxy | 335.172192 [334.956480, 335.374176] | 465.388736 [465.378368, 465.408480] | 590.106560 [590.098272, 590.123904] | 441.309760 [441.272352, 441.348064] |

### 四项预注册配对 delta

定义为：

```text
delta(A → B) = 100 × (median_A - median_B) / median_A
```

因此负数必须读作 **target B 更慢**，不能读成负开销或收益。下表列出 schema 中冻结的四项配对
delta 及 95% paired-bootstrap CI；其中 Full→Wave 和 Wave→Impl 是 protocol chain 的相邻比较，
Full→Impl 与 Full→Ceiling 是同一基准发出的 envelope/unsafe 对照。

| 上下文 | Full→Wave-Floor | Wave-Floor→Impl（matched protocol） | Full Floor→Impl（mechanism envelope） | Full Floor→Ceiling（unsafe） |
|---:|---:|---:|---:|---:|
| 4K | -53.865510% [-53.998374, -53.619261] | -17.748144% [-18.061785, -17.393054] | -81.173783% [-81.557345, -80.702230] | -40.521079% [-40.794517, -40.442656] |
| 32K | -12.637667% [-12.675488, -12.588105] | -22.397865% [-22.624273, -22.323538] | -37.866100% [-38.114860, -37.777389] | -11.452038% [-11.520826, -11.378208] |
| 128K proxy | -48.268130% [-48.444659, -47.975400] | -20.514996% [-20.627966, -20.425774] | -78.685332% [-79.014068, -78.326723] | -42.681387% [-43.544241, -42.171742] |
| 1M proxy | -38.850641% [-38.945955, -38.764115] | -26.798634% [-26.802894, -26.792643] | -76.060716% [-76.176334, -75.952791] | -31.666579% [-31.751373, -31.579247] |

准确解释是：例如 1M proxy 的 `Wave-Floor→Impl=-26.798634%` 表示 Impl 相对 Wave-Floor 慢
26.798634%；`Full Floor→Impl=-76.060716%` 表示 Impl 相对 Full Floor 慢 76.060716%，但后者
同时包含 bounded-wave scheduling/control 和不同 submission protocol，只能称为 native synthetic
mechanism envelope。所有 Full Floor→Ceiling 也为负，即本轮 unsafe Ceiling 反而比 Full Floor
更慢；这进一步说明 Ceiling 不是可直接解释的“零等待性能上界”。

## Exact 与长上下文 proxy 边界

| 上下文 | query blocks | logical / physical degree | producer CTAs | tiles/producer CTA（最大） | final trace rows |
|---:|---:|---:|---:|---:|---:|
| 4K exact | 64 | 32 / 32 | 2,048 | 1 | 8,704 |
| 32K exact | 512 | 256 / 256 | 131,072 | 1 | 528,384 |
| 128K proxy | 2,048 | 1,024 / 64 | 131,072 | 16 | 540,672 |
| 1M proxy | 16,384 | 8,192 / 64 | 1,048,576 | 128 | 4,325,376 |

4K/32K 为一 logical key tile 对应一个 producer CTA。128K/1M 把 physical degree 固定为 64，
每个 producer CTA 顺序覆盖 16/128 个 key tile；这保持所有 logical query×key tile 的 pair work、
索引与 history loads，但改变了 CTA 数量和 producer 粒度，所以只能称 `work_complete_packed_proxy`。
它不是生产 DSA kernel、真实模型层或 LLM 请求。

## 全元素、完整 pair work 与 raw evidence

正式配置固定 `pair_query=64`、`pair_key=128`，每个 score 显式执行 8,192 个 pair add，
`pair_closed_form=0`。四点的完整 pair-work items 分别为 16,777,216、1,073,741,824、
17,179,869,184 和 1,099,511,627,776。uint32 modulo accumulator 的 low16 与旧 uint64 语义由
边界样本证明等价；PTX proof 同时绑定显式 pair-add 与 global→shared query/key cache，SASS/resource
证明 worker 没有 local-memory spill。

每点有三次正确路径全元素 validation 和一次 Ceiling wrongness proof：

| 上下文 | score elements | index elements | output elements | 正确路径 mismatch | history loads / invocation | Ceiling 正式错误样本 |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 2,048 | 2,048 | 4,096 | 全部 0 | 2,048 | 31 / 31 |
| 32K | 131,072 | 131,072 | 32,768 | 全部 0 | 131,072 | 31 / 31 |
| 128K proxy | 2,097,152 | 2,097,152 | 131,072 | 全部 0 | 2,097,152 | 31 / 31 |
| 1M proxy | 134,217,728 | 33,554,432 | 1,048,576 | 全部 0 | 33,554,432 | 31 / 31 |

每点 124/124 个正式样本均有 `trace_complete=1`、`history_load_complete=1`；三个正确 rung 的
93/93 个样本均为 `stale_rows=0`，Ceiling 的 31/31 个样本均为 `ceiling_wrong=1`。最终 rep 保留
四个 mode 的完整 CTA trace；Python validator 不信任日志里的 PASS 标签，而是从 raw log、形状、
epoch/order、bootstrap 和 retained trace 独立重算，四个 validation JSON 均为 `PASS`、`errors=[]`。

## PTX、SASS 与 Nsight 证明

[`dsa_binary_proof.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_binary_proof.json)
为 `PASS`。冻结 PTX 对三个 worker 的精确语义计数包括：

- `dsaIndexer`：2 个 `launch_dependents`、1 个 global membar、1 个 release、0 个 wait；显式 pair add 与 cache 路径存在；
- `dsaTopk`：2 个 `launch_dependents`、1 个 wait、1 个 acquire、1 个 release；
- `dsaAttention`：2 个 `launch_dependents`、1 个 wait、1 个 acquire；Impl 的 acquire CFG 支配所有 history load 与语义贡献；
- 三个 worker 的指令顺序 proof 均为 PASS，PTX 是 PDL ordering 的语义权威。

真实 `sm_100` cubin 已提取并由 `nvdisasm` 生成 SASS；三个 worker 的 local-memory instruction
计数均为 0。SASS 用于证明真实目标机器码、worker presence 与无 local spill，不从 SASS mnemonic
反推 PDL ordering。

独立 4K Nsight Systems sidecar 不进入 timing matrix。其 validator 把 9 个必需 NVTX range
映射到实际 GPU kernel，包括四个 rung、poison、三个正确性 validation 和 Ceiling wrongness；
`dsaIndexer`、`dsaTopk`、`dsaAttention` 各观察到 21 个全局实例。原始 `.nsys-rep`、导出 SQLite、
CUDA kernel summary、NVTX projection 和 range→kernel summary 均被 finalizer 哈希绑定。

## NCU 权限与独占边界

Nsight Compute 2025.3.1 probe 返回 1，并明确记录 `ERR_NVGPUCTRPERM`、
`RMProfilingAdminOnly=1`、`hardware_counters_available=false`。这是权限分类，不影响本轮
`%globaltimer` timing admission；但 Section 9 只能标为 `PARTIAL`，不能声称物理 L2 hit rate、
DRAM bytes、stall breakdown 或硬件 counter 结论。替代证据只有 globaltimer trace 与 Nsight CUDA
timeline。

同一 GPU UUID `GPU-eb98326c-c1dd-1104-c4d7-24decdb06aef` 贯穿 runtime、lease、pre/post check、
四个 timing monitor、Nsight 和 NCU probe。监控采用 50 ms poll、2,000 ms query timeout；四个正式
点分别得到 9、62、87、883 次观测，全部 `PASS`、无 foreign process。Nsight/NCU monitor 也分别
以 31/13 次观测通过。独占模型的固有限制仍是：完全发生在两个已完成 sampling interval 之间的
短命 foreign GPU process 可能不可见；报告不把 interval sampling 写成连续硬件隔离。

## 能成立的结论

1. v9 四点 native timing matrix 和统一 campaign admission 均通过；本目录的正式 timing 可引用。
2. 4K/32K exact 与 128K/1M packed proxy 均执行完整 logical pair work、完整 history loads，并通过全元素正确性与最终 CTA trace 重算。
3. Full Floor 与 Wave-Floor 的合法 overlap 是 trigger-ordered tail overlap；本报告没有把计数字段误写成 wait duration。
4. 四个上下文中 Impl 均慢于 matched Wave-Floor；本 native synthetic harness 没有观察到 Impl 性能收益。
5. Full Floor→Impl 是包含 wave scheduling/control 与 submission protocol 差异的 mechanism envelope；Wave-Floor→Impl 才是 matched protocol 比较。
6. Ceiling 每个正式样本都错误，只能作为 unsafe no-wait/no-publish 路径的时间与 wrongness 证据。
7. PTX、真实 sm_100 cubin/SASS 与 Nsight range→kernel 证明闭合；NCU hardware counter 因权限不可用。

## 不能成立的结论

1. 不能把 128K/1M proxy 宣称为 exact CTA mapping、生产 DSA kernel 或模型端到端结果。
2. 不能从 Full Floor→Impl、Full Floor→Ceiling 或任何负 delta 推导“可用 CTA headroom”；负值的 target 实际更慢。
3. 不能把 Ceiling 当作可实现优化档，或用其错误输出对应的时间做设计推荐。
4. 不能把 `topk_waited` / `attention_waited` 当成阻塞时间、scheduler stall cycle 或 NCU wait counter。
5. 不能声称 production Tier 5 或 LLM 的 CTA headroom；本报告只覆盖 native synthetic work-complete proxy。
6. 不能发布 L2/DRAM/硬件 stall counter 结论，也不能把 software history-load count 等同于物理 cache traffic。
7. 不能把 50 ms 进程采样描述成能够观测任意短命外部 GPU 使用者的连续隔离。

## CPU-only 复核命令

以下复核不启动 GPU；validator replay 把临时 JSON 写到 `mktemp` 目录，不覆盖正式 artefact：

```bash
cd /workspace/gpu_sheduler_survey/cta_level_PDL_design
d=bench/dsa/results_20260805_b200_native_formal_strict_v9

# 统一 admission、terminal status 与四点矩阵
jq -e '.status == "PASS" and .accepted_timing == 1 and
       (.errors | length) == 0 and .timing_matrix.status == "PASS" and
       .binary_proof.status == "PASS" and .build_provenance.status == "PASS" and
       .profile_sidecar.status == "PASS"' "$d/campaign_admission.json"
jq -e '.status == "PASS" and (.errors | length) == 0' \
  "$d/terminal_status.json" "$d/validation_matrix.json"

# 四点 strict validator replay；不复用或改写正式 JSON
tmp=$(mktemp -d)
for tag in \
  dsa_exact_seq4096 dsa_exact_seq32768 \
  dsa_work_complete_packed_proxy_seq131072 \
  dsa_work_complete_packed_proxy_seq1048576; do
  python3 bench/dsa/validate_dsa_native.py "$d/${tag}.log" \
    --trace "$d/${tag}_trace.csv" \
    --expected-gpu-uuid GPU-eb98326c-c1dd-1104-c4d7-24decdb06aef \
    --json "$tmp/${tag}.json"
  jq -e '.status == "PASS" and (.errors | length) == 0' "$tmp/${tag}.json"
done
rm -r "$tmp"

# 冻结 patch 的 trigger/end 与 partial-tail slow-reference CPU tests
cd bench/dsa
python3 -m unittest -v \
  test_dsa_pipeline.WorkParityContractTests.test_floor_trace_uses_trigger_for_safety_and_end_for_overlap \
  test_dsa_pipeline.WorkParityContractTests.test_wave_floor_extrema_are_precomputed_per_wave_with_partial_tail
```

## 证据入口与 SHA-256

### Finalizer、build 与 profiler

| artefact | SHA-256 |
|---|---|
| [`campaign_admission.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/campaign_admission.json) | `135db12fc524e629525dc5b404fe9603ac67a6704e54bc27fe5ecc22a6bb7491` |
| [`terminal_status.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/terminal_status.json) | `3255070a8a7f892e666612cb64589a508b65d4cdfdcf5392195793f39587c19c` |
| [`validation_matrix.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/validation_matrix.json) | `41f84d905bf68bce63b96f2ff77e1dcab222c752c4d6fb6a28622d5225f03148` |
| [`dsa_build_manifest.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_build_manifest.json) | `694c36397cf4fea2a1ca99510c4574f7722109d0b2debf140b115d2edad56ff9` |
| [`dsa_binary_proof.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_binary_proof.json) | `5db9452983fc538394f9777a7a667836b60fb0e0989e554731a7e56d2e714eb0` |
| [`dsa_profile_seq4096_profile_validation.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_profile_seq4096_profile_validation.json) | `2fdd903809677d0070774b62b21c79a93cfd6e2db8b3b9301d9bd3a9c7dba4d8` |
| [`dsa_profile_seq4096.nsys-rep`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_profile_seq4096.nsys-rep) | `69932a88f87976a3e2b3f1a55c97d6ba8682a247c43e15308d2fff4b976151a3` |
| [`ncu_permission.json`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/ncu_permission.json) | `e3e1b7faf53e95b1e5866d4d5b27df7e396dc0ef8e821f64e78513eed22a9093` |
| [`dsa_native.ptx`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_native.ptx) | `10dbaae0d78d13b63d985dce5ce13af73dcf2c9601e625969fedb5b383b57f58` |
| [`dsa_native_resources.txt`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_native_resources.txt) | `82678e38b270d81ba0a79c80b65e400e69eaafa31d83fc1eabcc64611106b9cf` |
| [`dsa_native_sm_100.sass`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_native_sm_100.sass) | `4c044c70dc9c85ddb2d3d5c8898acd6e69346a18023494a2cf02e7b1126ce192` |

冻结 source/binary/validator/runner SHA-256 分别为：

```text
dsa_native.cu          84a6c18815557c4c2f7cde927b3fe5f9617754d32a1783af9b223c915a4ec850
dsa_native binary      a2110fdd7e5db4bf5c843972973ce23aa4dc64cf22c46b25d8dfcb245dbf295a
validate_dsa_native.py 20aee5c09deb58b4e29686ab16d91d885b22baad4f6468bbdcab5d9fd74c440c
run_dsa_chain.sh       f070b00463b40f0fe384e30c3639fbead50345db7aab0318bdda6ae2fd545a33
test_dsa_pipeline.py   c6277ddc12ac1cea0cdfc97ef00cb94b24bb170aa55b151b59efe26a156a43b6
```

### 四点 raw log、trace 与独立 validation

| 点 | raw log SHA-256 | final trace SHA-256 | validation SHA-256 |
|---|---|---|---|
| 4K exact | `65ccf755784635a81bae651b08ff3e33bb440c3f48eab0bbba297efcfe594fdc` | `5c0471cf9a82e27bb6627a4ba02f39cbdfa0ef5500f53e99b4621da8d0b5ec82` | `bcedea7448efacd6561cf4bfe4b350d62ce33fec1e805034e54cf90eb3002bc3` |
| 32K exact | `d75c628f6e3612ea80ee0256920b903074039f24117bc41796a6d74180031d02` | `7d7c823e186857514489b7b006ff0a8f83fa3213873d9561a57e1f48706fc6bb` | `0149ece859bed1b57fd8b53191e632aae32a0de0868bd903863abae293a47e01` |
| 128K proxy | `d9ddbea0e63047d44152341a8eb349b0ba638685c48268eb2bd0a6334d472b6e` | `63a1cc567e318a7efc131aba2e34e440e54407c08143c1c33bbe2108753882e0` | `a5d2d89e93f888c6732d4b52164f87338478d5d5f42848fc06c3d1b7ea0478ea` |
| 1M proxy | `325978ca85720380fe8f57ca12e3f0416e7ce2c0110afc438bc5c7c9ab46126b` | `521964954e5a31f46c826da8a22536bd04037b0b1d9977d49944b2eaa5e3740b` | `7bff5f9ba5a864ad60ff9b13a9ed29401985b445af98ffad0628299c8ec7627f` |

对应 artefact 入口：

- 4K：[`raw log`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_exact_seq4096.log)、[`trace`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_exact_seq4096_trace.csv)、[`validation`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_exact_seq4096_validation.json)
- 32K：[`raw log`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_exact_seq32768.log)、[`trace`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_exact_seq32768_trace.csv)、[`validation`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_exact_seq32768_validation.json)
- 128K：[`raw log`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_work_complete_packed_proxy_seq131072.log)、[`trace`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_work_complete_packed_proxy_seq131072_trace.csv)、[`validation`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_work_complete_packed_proxy_seq131072_validation.json)
- 1M：[`raw log`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_work_complete_packed_proxy_seq1048576.log)、[`trace`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_work_complete_packed_proxy_seq1048576_trace.csv)、[`validation`](../../bench/dsa/results_20260805_b200_native_formal_strict_v9/dsa_work_complete_packed_proxy_seq1048576_validation.json)

本报告没有引用 v7/v8 拒收目录中的任何 timing，也没有把 sidecar profiler 时间混入正式 31-repeat
matrix。
