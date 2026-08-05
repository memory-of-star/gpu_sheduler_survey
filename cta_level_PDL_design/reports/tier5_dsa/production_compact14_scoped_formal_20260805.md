# Tier 5 production compact-14：双模型双上下文 scoped formal 报告

- **实验日期 / 报告日期**：2026-08-05（UTC）
- **设备**：NVIDIA B200，148 SM，Compute Capability 10.0，UUID
  `GPU-eb98326c-c1dd-1104-c4d7-24decdb06aef`
- **正式证据目录**：[`bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/)
- **最终状态**：compact admission `PASS`，`accepted_compact_workload_timing=1`
- **证据等级**：B200 production API component-level、4K/128K scoped workload timing；不是
  exact-26、整模型端到端或 CTA Floor/Impl/Ceiling bracket

## 执行摘要

本轮在全新结果根执行预先冻结的 compact-14 矩阵：DeepSeek-V3.2 与 GLM-5，4K 与
128K 两个上下文，`operator_chain`、`single_layer`、`indexshare_fsss` 三类 workload，
再加每模型一个 MoE-32 点。每个 off/on 点有 5 次 warmup、31 次相邻 timed repeat；MoE
保持 framework-default 单档。最终账本为 14 个 correctness row、1,302 个 timed sample、
62 个 summary，14 个 fragment validator、统一 campaign `check-final` 与独立 compact
validator 全部 `PASS`、`errors=[]`。

12 个主 off/on comparison 的 paired median `on - off` 都为负；其中 6 个 95% bootstrap
CI 完全低于 0，另 6 个跨 0，没有一个完全高于 0。具体幅度从 `-0.003200 ms` 到
`-0.784452 ms`。这只是 production component 的 PDL-on/off **诊断差值**：top-k 没有
显式 PDL control，worker binary 也没有 production PTX/cubin proof，所以负值不能改名为
CTA headroom、CTA Impl 收益或整模型收益。

compact 准入刻意没有篡改 exact-26 字段。base campaign 保持 `formal=false`、
`campaign_mode=nonformal_short`、`accepted_workload_timing=0`；只有 source-bound
[`validate_production_tier5_compact.py`](../../bench/dsa/validate_production_tier5_compact.py)
在重新执行 whole-campaign `check-final`、核对精确 14-row 矩阵和 14/1302/62 基数后，单独将
`accepted_compact_workload_timing` 置 1。32K 与 1M production timing 明确排除；原 exact-26
仍未完成，也没有从任何被拒绝或部分结果根复用 sample。

| admission 字段 | 值 | 精确含义 |
|---|---:|---|
| `accepted_compact_workload_timing` | 1 | 只接受本报告的双模型、4K/128K component timing |
| `accepted_exact26_workload_timing` | 0 | 未完成 canonical exact-26 |
| `accepted_workload_timing` | 0 | base harness 为 exact-26 保留的字段未打开 |
| `accepted_timing` | 0 | legacy CTA-bracket timing 字段未打开 |
| `accepted_CTA_bracket` | 0 | 无 production CTA Impl / unordered Ceiling |
| `headroom_defined` / `headroom_pct` | `false` / `null` | CTA/LLM production headroom 未定义，不是 0% |

## 程序实际执行什么

[`production_tier5.py`](../../bench/dsa/production_tier5.py) 使用已安装 production API 与
模型 shape，但不是加载两个完整 pretrained checkpoint：

| workload | 实际执行语义 | timing component |
|---|---|---|
| `operator_chain` | preprojected full-work vLLM `SparseAttnIndexer`，DeepGEMM logits、vLLM top-k、FlashInfer sparse MLA | `indexer_topk`、`sparse_mla`、`chain_total` |
| `single_layer` | 随机 exact-shape projection、RMSNorm、RoPE、production MLA cache insertion、coherent causal indexer/KV history、`W_UV` 与 output projection；无 decoder residual/MLP | `attention_layer_total` |
| `indexshare_fsss` | 四个因果一致的 attention-only layer；每次调用严格一个 production indexer，逻辑 top-k 在四层 FSSS 复用 | `four_layer_fsss_total` |
| `moe32` | production `fused_topk` + `fused_experts`，32 个 resident expert、top-8、4,096 tokens | `fused_topk_plus_fused_experts` |

off/on 只控制两个可见入口：DeepGEMM 使用 `set_pdl(bool)` 并强制 readback 一致，FlashInfer
MLA 接收 `enable_pdl=bool`。vLLM top-k 没有对应 control，因此两档都执行同一个 top-k 路径。
安装的 API 也不暴露 identity-preserving CTA readiness 或真正不等待的 unordered Ceiling。
因此 PDL-on 只能称 `production_pdl_on_diagnostic`，PDL-off 也只是诊断控制，二者没有构成
四点 bracket。

每个 context 执行完整 causal lower triangle，`query_sampling=NONE`、
`causal_pair_sampling=NONE`。FP32 logits 驻留上限为 16 GiB，单个 query chunk 最多
4,096 tokens；4K 是 1 个 chunk，128K 是 32 个 chunk。每个 chunk 只构造一次只读 FP32
manual logits oracle，off/on 的 native replay、top-k、MLA 和 layer output 仍独立执行和验证。

## 配置、计时与统计

| 项 | 冻结值 |
|---|---|
| 模型 | `deepseek_v32`, `glm5` |
| 上下文 | 4,096；131,072；两点均在各自 official position range 内 |
| workload | `operator_chain`, `single_layer`, `indexshare_fsss`；另含每模型一个 `moe32` |
| Warmup / timed repeats | 5 / 31 |
| pairing | 同一 row 进程内、同 component 相邻 off/on；偶数 repeat `off→on`，奇数 repeat `on→off` |
| timing | CUDA Event device duration；full reference validation 不在 event 内 |
| 长上下文聚合 | 同一 repeat/component/mode 的 32 个 chunk event duration 求和 |
| 统计 | 中位数；4,000 次 deterministic bootstrap 95% CI；paired delta 先逐 repeat 求 `on-off` 再取中位数 |
| seed | `20260805` |
| runtime | PyTorch 2.11.0+cu130、vLLM 0.23.0、FlashInfer 0.6.12、CUDA runtime 13.0 |
| GPU monitor | 50 ms sampling、2,000 ms query timeout、UUID-scoped lease |

128K 数值是 32 段 GPU component duration 之和，不包含 manual oracle、chunk allocation、
host launch 间隙或每 chunk 后的 `empty_cache`，所以不是请求 wall time。operator 的
`indexer_topk`、`sparse_mla` 与 `chain_total` 又是独立 timed invocation；不能把两个 component
的中位数相加来“复算”chain 中位数。

## 十二个主 paired timing 点

每格为 `median ms [95% bootstrap CI]`；最后一列严格来自 31 个相邻 pair 的
`on_i - off_i`，不是两个独立中位数相减。负数表示本轮 PDL-on diagnostic 更快。

| 模型 | seq | workload / 主 component | PDL off | PDL on | paired on−off |
|---|---:|---|---:|---:|---:|
| DeepSeek-V3.2 | 4096 | `operator_chain / chain_total` | 1.632288 [1.628128, 1.639424] | 1.629184 [1.620992, 1.637440] | -0.004096 [-0.010400, 0.003136] |
| DeepSeek-V3.2 | 4096 | `single_layer / attention_layer_total` | 5.924832 [5.922848, 5.926848] | 5.920768 [5.918912, 5.921792] | -0.004448 [-0.008128, -0.002592] |
| DeepSeek-V3.2 | 4096 | `indexshare_fsss / four_layer_fsss_total` | 23.084160 [23.076063, 23.088320] | 23.076256 [23.072033, 23.082209] | -0.009121 [-0.011330, -0.002111] |
| DeepSeek-V3.2 | 131072 | `operator_chain / chain_total` | 153.581088 [153.322721, 153.808289] | 153.379968 [153.171711, 153.715391] | -0.129249 [-0.280672, 0.122144] |
| DeepSeek-V3.2 | 131072 | `single_layer / attention_layer_total` | 287.679199 [287.134594, 287.980865] | 287.528162 [286.795040, 287.930848] | -0.029537 [-0.214527, 0.117634] |
| DeepSeek-V3.2 | 131072 | `indexshare_fsss / four_layer_fsss_total` | 888.341982 [887.903872, 888.920221] | 886.965799 [886.563070, 888.127840] | -0.784452 [-1.387518, -0.203743] |
| GLM-5 | 4096 | `operator_chain / chain_total` | 1.438720 [1.431520, 1.453088] | 1.439680 [1.434624, 1.444864] | -0.003200 [-0.009216, 0.005152] |
| GLM-5 | 4096 | `single_layer / attention_layer_total` | 4.186112 [4.180960, 4.191264] | 4.179968 [4.173792, 4.188160] | -0.005152 [-0.017472, -0.001824] |
| GLM-5 | 4096 | `indexshare_fsss / four_layer_fsss_total` | 16.188513 [16.174208, 16.192896] | 16.171904 [16.157728, 16.186081] | -0.014688 [-0.043425, 0.008129] |
| GLM-5 | 131072 | `operator_chain / chain_total` | 109.998081 [109.477120, 110.293279] | 109.913408 [109.630240, 110.124192] | -0.012640 [-0.233280, 0.091136] |
| GLM-5 | 131072 | `single_layer / attention_layer_total` | 194.446848 [194.110465, 195.238081] | 194.281568 [193.886176, 194.984255] | -0.227071 [-0.283712, -0.105887] |
| GLM-5 | 131072 | `indexshare_fsss / four_layer_fsss_total` | 613.623779 [613.011452, 613.842237] | 612.976482 [612.570402, 613.243809] | -0.495132 [-0.790941, -0.183493] |

CI 完全低于 0 的六点是 DeepSeek 4K single/FSSS、DeepSeek 128K FSSS、GLM 4K single、
GLM 128K single/FSSS；其余六点跨 0。本报告只逐点描述，没有做多重比较校正，也不将
“CI 不跨 0”改写成 CTA 机制因果证明。

## Operator-chain component 分解

| 模型 | seq | component | PDL off | PDL on | paired on−off |
|---|---:|---|---:|---:|---:|
| DeepSeek-V3.2 | 4096 | `indexer_topk` | 0.240960 [0.226080, 0.242720] | 0.237504 [0.230688, 0.242912] | -0.009024 [-0.016512, 0.017760] |
| DeepSeek-V3.2 | 4096 | `sparse_mla` | 1.512416 [1.506336, 1.526784] | 1.508352 [1.504256, 1.528768] | 0.008192 [-0.020448, 0.016352] |
| DeepSeek-V3.2 | 131072 | `indexer_topk` | 90.643072 [90.171872, 90.997537] | 90.455424 [90.254624, 90.691584] | 0.530463 [-0.670944, 0.770847] |
| DeepSeek-V3.2 | 131072 | `sparse_mla` | 67.674208 [67.510016, 67.764448] | 67.635967 [67.589024, 67.718048] | 0.002657 [-0.057088, 0.061921] |
| GLM-5 | 4096 | `indexer_topk` | 0.210208 [0.201664, 0.214784] | 0.208000 [0.198464, 0.214720] | -0.000736 [-0.014016, 0.012384] |
| GLM-5 | 4096 | `sparse_mla` | 1.278912 [1.264672, 1.289248] | 1.287200 [1.277984, 1.292320] | 0.012160 [-0.008224, 0.030752] |
| GLM-5 | 131072 | `indexer_topk` | 58.793280 [58.641024, 59.279808] | 58.883776 [58.720193, 59.203072] | 0.454048 [-0.528960, 0.551232] |
| GLM-5 | 131072 | `sparse_mla` | 52.948512 [52.775616, 53.185152] | 53.059968 [52.753984, 53.157472] | -0.282528 [-0.384160, 0.349184] |

八个 component paired CI 全部跨 0。特别是 `indexer_topk` 同时包含没有 PDL control 的
top-k，不能把该行归因成 DeepGEMM PDL 的孤立效果。

## MoE-32 framework-default 单档

MoE API 在本 harness 中没有构造 off/on pair，只能发布单档 workload characterization：

| 模型 | median ms [95% CI] | min ms | max ms |
|---|---:|---:|---:|
| DeepSeek-V3.2 | 3.390496 [3.369856, 3.500288] | 3.343136 | 3.531840 |
| GLM-5 | 2.932512 [2.919712, 3.029408] | 2.896288 | 3.142912 |

两点各 31 个 sample。每模型完整验证 4,096 tokens、32,768 个 routing assignment；
DeepSeek 与 GLM 分别检查 29,360,128 和 25,165,824 个 output element，最大绝对误差为
`0.010372758` 与 `0.011366606`。这些数字不能产生 MoE PDL delta。

## 完整性与正确性复算

矩阵和 timing 账本为：

```text
rows = 2 models × 2 contexts × 3 paired workloads + 2 MoE = 14

operator samples = 4 rows × 3 components × 2 modes × 31 = 744
single samples   = 4 rows × 1 component  × 2 modes × 31 = 248
FSSS samples     = 4 rows × 1 component  × 2 modes × 31 = 248
MoE samples      = 2 rows × 1 mode × 31                  = 62
total samples                                              = 1302

summaries = 4×9 + 4×3 + 4×3 + 2×1 = 62
```

`samples.jsonl` 的 1,302/1,302 条记录均为有限正 duration、`poison_verified=true`、
`timed_validation=false`；paired 部分严格为 620 off + 620 on，MoE 为 62 个
`framework_default_uncontrolled` sample。

完整 causal work 可从聚合 correctness 独立复算：

```text
4K pairs per row   = 4096 × 4097 / 2       =       8,390,656
128K pairs per row = 131072 × 131073 / 2   =   8,590,000,128

matrix causal pairs
  = 6 × 8,390,656 + 6 × 8,590,000,128
  = 51,590,344,704

off/on valid logits cells = 2 × 51,590,344,704
                           = 103,180,689,408
```

12 个 DSA row 共执行 811,008 个 logical query row、198 个 chunk；两模式产生 396 个
mode-correctness record，全部 `PASS`。聚合最大 `calc_diff=9.9920072e-15`、最大逐行
`row_calc_diff=1.1779466e-12`、最大 logits 绝对差 `1.9073486e-6`；kernel/manual
non-finite、row-quality failure、acceptance mismatch、重复/越界 top-k、score violation 与
top-k mismatch 全为 0。共检查 159,450,660,864 个 attention element；attention/layer-output
最大绝对误差均为 `0.03125`。

14 个 row 各自获得 UUID-scoped GPU lease。14/14 monitor `PASS`，合计 6,355 次 50 ms
观测，foreign process 与 query failure 都为 0，进程显存采样峰值为 18,806 MiB；所有
identity 都绑定同一 B200 UUID。
interval sampling 的固有限制仍在：完全落在两个已完成 poll 之间的短命 foreign process
可能不可见，不能写成连续硬件隔离。

## 独立 1M capacity proof 与 profiler sidecar

### 1M 只证明可承载性，不进入 timing

独立目录
[`results_20260805_b200_production_long_probe_v3_1m_deepseek_fsss_16g`](../../bench/dsa/results_20260805_b200_production_long_probe_v3_1m_deepseek_fsss_16g/)
对 DeepSeek-V3.2 1M FSSS 执行 256 个 4,096-query chunk，覆盖全部 1,048,576 query row、
549,756,338,176 causal pair，两模式共 1,099,512,676,352 个 valid logits cell；没有抽样。
最大 `calc_diff=9.6589403e-15`、最大逐行 `calc_diff=5.0959237e-14`、最大绝对差
`2.3841858e-6`，non-finite 与 row failure 均为 0。GPU monitor 9,952 次观测、foreign=0，
进程显存采样峰值为 98,506 MiB。

该 probe 的 fragment validator 虽为 `PASS`，但所有 acceptance 字段都是 0；其冻结
`production_tier5.py` SHA 还是 compact-validator 加入 source manifest 之前的
`facff3e0...`。因此这里只把它当独立的 16 GiB logits / 1M full-causal capacity proof，
不把它纳入 compact source closure，也不引用其中的单次 timing。1M 还超出两个模型声明的
official position range。

### Fresh production Nsight sidecar

[`results_20260805_b200_production_nsys_sidecar_v4_compactsource`](../../bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/)
使用与 compact 完全相同的 production/validator source hashes。其
[`nsys_sidecar.json`](../../bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/nsys_sidecar.json)
为 `PASS`，将一个 DeepSeek 4K operator row 的唯一 NVTX invocation range 绑定到解析成功的
Nsight report，并在 CUDA GPU summary 中观察到 `mqa`、`topk`、`gemm` kernel token 与
`cudaLaunchKernel` API。sidecar 的 `accepted_timing`、`accepted_workload_timing` 和
`accepted_CTA_bracket` 全为 0；profiler duration 没有混入 31-repeat matrix。

native-v9 的 PTX/SASS、四 rung trace 与 Nsight 证据仍属于独立 synthetic admission，见
[`native_v9_four_context_formal_20260805.md`](native_v9_four_context_formal_20260805.md)。它不能
替 production API 补 worker binary、CTA Impl 或 Ceiling 证明；同理，本 production sidecar
也不改变 native-v9 的结论。

## 能成立的结论

1. 在精确冻结的 compact-14 scope 内，双模型 4K/128K production component timing 可引用；
   唯一授权字段是 `accepted_compact_workload_timing=1`。
2. 14 个 row、1,302 个 sample、62 个 summary、全 causal pair/valid logits cell 和所有声明
   correctness element 已由 whole-campaign final checker 与 compact validator 重算闭合。
3. 12 个主 paired median 都为负；六个 CI 完全低于 0，六个跨 0。这个描述只适用于表中
   component、模型 shape、上下文、版本和 B200。
4. 4K/128K 两点都位于两模型 official position range 内；128K 没有通过 query/pair 抽样来
   缩短实验。
5. 独立 1M probe 证明旧 source-bound 16 GiB chunking 方案在本 B200 上可完成 1M FSSS
   全因果工作与全正确性，但不提供正式 timing。
6. fresh production Nsight sidecar 证明同 compact source 的 4K operator invocation 确实执行
   mqa/topk/gemm 类 GPU kernel；它只是非 timing profiler evidence。

## 不能成立的结论

1. 不能说 exact-26 或计划 §9 的完整 production matrix 已完成；32K/1M production timing
   均不在本报告内，任何被拒绝 exact-26 根中的局部 row 也没有被复用。
2. 不能把 `accepted_compact_workload_timing=1` 简写成 `accepted_timing=1` 或
   `accepted_workload_timing=1`；后二者在正式 artefact 中都为 0。
3. 不能从 PDL-on/off diagnostic 推导 CTA headroom、CTA Impl 收益或硬件设计坐标。production
   API 没有 identity-safe CTA readiness，且没有 unordered Ceiling。
4. 不能把 top-k 归因给 PDL 开关；它没有显式 control。也不能从 operator component 中位数
   相加得到 chain 中位数。
5. 不能把随机 exact-shape component workload 写成 DeepSeek-V3.2/GLM-5 pretrained
   整模型、端到端请求延迟、TPS/user 或服务负载结果。
6. 不能从 MoE 单档比较 PDL on/off；本轮只有 32 experts，而不是完整模型的 256 routed
   experts，也没有 EP/TP 跨卡路径。
7. 不能把 128K 的 chunk-event 求和写成 host wall time；manual oracle、allocation、launch
   gap 和 cache release 均不在 event duration 内。
8. 不能将 1M capacity probe 的单次 duration、旧 source SHA 或超 official-range 形状升级为
   compact timing；capacity proof 与 timing admission 相互独立。
9. 不能用 native-v9 PTX/SASS 给 production worker binary 背书，也不能用 production
   Nsight 的 kernel-name token 反推 grid-PDL/CTA ordering。
10. 不能发布 production L2/DRAM/stall counter、B300/Rubin 定量外推或“CI 不跨 0 即机制因果”
    的结论；本轮没有 production NCU counter admission，也没有做多重比较校正。

## CPU-only 复核

以下命令不启动 GPU；compact validator 会重新调用 final checker 并把复核 JSON 写入临时目录：

```bash
cd /workspace/gpu_sheduler_survey/cta_level_PDL_design
d=bench/dsa/results_20260805_b200_production_compact_formal_v1_16g

python3 bench/dsa/production_tier5_campaign.py check-final \
  --root "$d" \
  --contract "$d/campaign_contract.json" \
  --binding "$d/campaign_binding.json"

tmp=$(mktemp -d)
python3 bench/dsa/validate_production_tier5_compact.py "$d" \
  --json "$tmp/compact_campaign_admission.json"
jq -e '.status == "PASS" and .errors == [] and
       .accepted_compact_workload_timing == 1 and
       .accepted_exact26_workload_timing == 0 and
       .accepted_timing == 0 and .accepted_workload_timing == 0 and
       .accepted_CTA_bracket == 0 and
       .observed_cardinalities == {"correctness_rows":14,"samples":1302,"summaries":62}' \
  "$tmp/compact_campaign_admission.json"
rm -r "$tmp"
```

## 证据入口与 SHA-256

### Compact admission 闭包

| artefact | SHA-256 |
|---|---|
| [`compact_campaign_admission.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/compact_campaign_admission.json) | `e76daf0b27bc8d1082126f135a2d57eb78a9caed18fa59f62c011cad4465069a` |
| [`campaign_contract.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/campaign_contract.json) | `f7c6f31481e25398de286f80793d256b0659eb443bbbb034dafed7dda12b6cef` |
| [`campaign_binding.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/campaign_binding.json) | `e0fe89d9ebfd744221037fcadb6a443a5d8aaa9287e903d1590d1f1f573c92b0` |
| [`campaign_validation.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/campaign_validation.json) | `67ece95170a29011fc8d102cbfb780e3b70acd7b4241f52c5b413bbf45f300c8` |
| [`manifest.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/manifest.json) | `dc4234e0e8beaf39d7844404f78472d3a493c685e8b39d527d22a530283c0799` |
| [`samples.jsonl`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/samples.jsonl) | `937eaab4e0ee00d193ef3399812ec08f6fc9c36244074a94825a7e7ba5b756e9` |
| [`correctness.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/correctness.json) | `e7adbf21b4877ce606c4f706abcba085e115f63c8980d7a1dc7326cab6fa5f62` |
| [`result.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/result.json) | `6842d85a42ca52757526f09f50fe17e94f1fd10ce18779b35b34fab9bab6da27` |
| [`production_candidate.done.json`](../../bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/production_candidate.done.json) | `cececf5ce193c66eac0e6ed23bb90024d28e6c2b8f0c2df78ecfffd735fab41f` |

contract 内部 canonical SHA 为
`3f3c604d9892f523bde2ccb51bad8c9808fc0d25ad4cb1a31c060964486f1029`；campaign
fingerprint 为 `9ebe458761d3b2aad60b679662fa1d3ba2a620b04ccc0ff4dc987feee3eebd52`；
uniform runtime build SHA 为
`1a6741f693e40777ad2810fd2e0331b6f6854757e5823fc09ff03bc4954d3444`。compact matrix
SHA 为 `80fdc6a359c99532def65cd424528809e1eff07bed90197c2a98ac46046a4514`，controls SHA
为 `2417125ce91a31696ec3d649ad1125b838159a76698b9e627b497ef3c67a76f6`，admission body
SHA 为 `64f2e1b127129334a15a2b2a4aa488c9a55a64f14622aa70061f1b0e57050a76`。

### 冻结本地源码

| source | SHA-256 |
|---|---|
| [`production_tier5.py`](../../bench/dsa/production_tier5.py) | `078d6d95d45fed6e2fc2d4c02cd6912ef8cd72d3349d6b098e686194eefa0525` |
| [`validate_production_tier5.py`](../../bench/dsa/validate_production_tier5.py) | `b5ec90b014305d69b0c2dee528c5e9d87abf6d2fb27778851d0dfb676ef097be` |
| [`validate_production_tier5_compact.py`](../../bench/dsa/validate_production_tier5_compact.py) | `aed51232e26f8675c2043852ebb82768778e2ccf9616f120f2d9ac1b9a287a0a` |
| [`production_tier5_campaign.py`](../../bench/dsa/production_tier5_campaign.py) | `0a112411fc591920cd143d1ea2e896600799e5f7350345ef4fc7392d8a07fbbc` |
| [`run_production_tier5_fragments.sh`](../../bench/dsa/run_production_tier5_fragments.sh) | `f1e51b13b8693270e62c2b6c5f849fcdb20c371a66921bf76ca73a9200adc8c4` |
| [`run_production_tier5.sh`](../../bench/dsa/run_production_tier5.sh) | `7ac9ad4b709d4d023e022822ac1a1b9d07e691f8e5361339ef328f1feccee952` |
| [`gpu_exclusivity.py`](../../bench/dsa/gpu_exclusivity.py) | `0b7c9509703cbddd8d2cb6868a659200c9bfd17656235b7139943ac7dc1622b9` |

source-manifest SHA 为
`5c5f80900529728d5f57a8eb78a6d55c10ffb3772bf0a46be64541fa0778c89c`，package-manifest
SHA 为 `0c42d55af10aa1eee471565ecc1798390b97992e5b5b2d25ef2e7902bdfd56e6`。

### 独立 capacity / profiler artefact

| artefact | SHA-256 |
|---|---|
| [1M `nonformal_fragment_only.done.json`](../../bench/dsa/results_20260805_b200_production_long_probe_v3_1m_deepseek_fsss_16g/nonformal_fragment_only.done.json) | `bf82ee1594ad3d1037f3212774c2cb820dc1b3fae7682155b09702a66333d5cf` |
| [1M `fragment_validation.json`](../../bench/dsa/results_20260805_b200_production_long_probe_v3_1m_deepseek_fsss_16g/rows/000_deepseek_v32.indexshare_fsss.seq1048576/fragment_validation.json) | `071fee0bfadb9fb29ebe2f2a8e011cf71ea48307452c7b5e40d6e183d7fefb2c` |
| [1M `correctness.json`](../../bench/dsa/results_20260805_b200_production_long_probe_v3_1m_deepseek_fsss_16g/rows/000_deepseek_v32.indexshare_fsss.seq1048576/correctness.json) | `c16503c65d9f070d99daf905ea4fa73524988f7a61ea974a6f01f602284eb914` |
| [production v4 `nsys_sidecar.json`](../../bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/nsys_sidecar.json) | `1149f3c783e84f339a271cad10ca64e8b1ee33e96e0a490c89c8968ab42d7704` |
| [production v4 `profile.nsys-rep`](../../bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/profile.nsys-rep) | `ae7ccebfa08c6bccac79dc4b70dd785e20e4669a84455973b319f099cac994bc` |
| [production v4 `profile.sqlite`](../../bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/profile.sqlite) | `6b60e58e8b7cfb3d82030f77f171b0cc96a9999c31ae913442215167918445c8` |
| [production v4 `nsys_stats.txt`](../../bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/nsys_stats.txt) | `b0da695a6edb803a253b45b9860102d46e3fee9150b0dda4bea57f121228b446` |

本报告的进度入口是
[`EXPERIMENT_REPORT_INDEX.md`](../../EXPERIMENT_REPORT_INDEX.md)，全 campaign 边界见
[`reports/campaign_b200_multiwave_20260805.md`](../campaign_b200_multiwave_20260805.md)。
