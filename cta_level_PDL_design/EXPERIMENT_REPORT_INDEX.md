# CTA-Level PDL 实验执行进度

更新时间：2026-08-05  
权威规格：[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)  
代码实现契约：[`bench/README.md`](bench/README.md)

> 本文件只记录执行进度、可信证据和续跑入口。测什么、如何判读与阈值由
> `EXPERIMENT_PLAN.md` 定义；数字和结论只在 `reports/` 中发布。

状态图例：

| 状态 | 含义 |
|---|---|
| `DONE` | 对应计划节已有满足声明边界的可信结果和报告 |
| `PARTIAL` | 只覆盖了计划的一部分，不能冒充完整实验 |
| `BLOCKED` | 当前 harness、模型、工具或可承载性不满足准入；不得发布 timing |
| `READY` | 实现和前置条件齐，可直接执行 |
| `GATED` | 等上游机器判决后再决定是否执行 |
| `REJECTED` | 跑过但准入失败；timing 不得进入任何结论 |

新增或修改报告时，必须同步更新本文件的 §0、§2 与 §4。

---

## 0. 当前状态（一眼）

| 项 | 当前值 |
|---|---|
| 有效正式 session | Tier 1 / Tier 0.3 headline：[`bench/results_20260805_b200_multiwave_v2/`](bench/results_20260805_b200_multiwave_v2/)；Tier 0.1 唯一正式源：[`bench/results_20260805_b200_multiwave_v4/`](bench/results_20260805_b200_multiwave_v4/)；均为 B200，148 SM，CC 10.0 |
| 总报告 | [`reports/campaign_b200_multiwave_20260805.md`](reports/campaign_b200_multiwave_20260805.md) |
| 完整证据归档 | [`cta_pdl_results_20260805_compact14_final.tar.gz`](cta_pdl_results_20260805_compact14_final.tar.gz)，201,249,641 bytes（Git LFS），SHA-256=`c3f70e0892667b5b4ddf387382f095a41f7e57715299f687dbbe41be08bbd9da`；内含 authoritative sessions、拒收根、报告与冻结验证源码 |
| §6 Gate | **`GO`**；Tier 1.1=88 配置，median space=36.1062%，`plan_sweep_complete=true` |
| 多波覆盖 | 2×/8×/32×SM 均由 `%globaltimer` trace 证明；31 repeats、manifest、unique parents、semantic proof 全通过 |
| 大 grid 边界 | 32×SM median space=4.5320%，当前 software captured median=-5.6928%；见 Tier 1 主报告 |
| §4.1 | `DONE`；v4 共 6 个 chain 长度、186 组相邻 pair / 372 samples，12 条独立 correctness validation；`semantics=3` strict validator 重算 epoch、checkpoint/final digest、CI 与最终 trace 后 PASS，实测 peak=2 grids / 296 CTA |
| §4.3 | `DONE`；v2 容量曲线 + 15 点 productive-background 配对曲线及正式 validator PASS；headline 不改用 v4 回归数值 |
| Tier 2/3 native formal | `DONE`；[`bench/results_20260805_b200_tier23_native_v2/`](bench/results_20260805_b200_tier23_native_v2/) 的 35 配置、5,084 samples、182,460 trace rows、0 errors，strict validator `PASS`；覆盖 §7.1/§7.3–§7.6 |
| Tier 2/3 证据边界 | 正式 `%globaltimer`、全边/全词校验、Ceiling sentinel、Nsight Systems 与 sanitizer 证据完整；NCU 因 `ERR_NVGPUCTRPERM` 不可用，软件逻辑 load/byte 计数不代替物理 L2/DRAM counter |
| 官方 FAST | [`bench/results_20260805_smoke_v5/`](bench/results_20260805_smoke_v5/)；仅 plumbing，5 repeats，不能作性能证据 |
| 首轮 formal | **`REJECTED / INVALID`**；[`bench/results_20260805_b200_multiwave/`](bench/results_20260805_b200_multiwave/) 的全部 timing（含 Tier 0）禁用 |
| Tier 4 full-decode formal | `DONE`；Qwen3.6-27B 的 4 个 headline 点均为 31 repeats，decode/prefill admission 独立复算 `PASS`；grid→unsafe Ceiling 上界为 1.3334% / 1.3094% / 1.8301% / 8.0609% |
| Tier 5 native formal | `DONE`；[`bench/dsa/results_20260805_b200_native_formal_strict_v9/`](bench/dsa/results_20260805_b200_native_formal_strict_v9/) 的 4K/32K exact CTA mapping 与 128K/1M `work_complete_packed_proxy` 四点统一 admission `PASS`、`accepted_timing=1`；只支持 native synthetic mechanism evidence |
| Production exact-26 | **`INCOMPLETE`**；[`v1`](bench/dsa/results_20260805_b200_production_formal_exact26_v1/) 与 [`v2`](bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/) 均已 `REJECTED`，partial rows/timing 禁止复用或拼接；canonical exact-26 未完成 |
| Production compact-14 scoped formal | **`PASS / DONE`**；[`compact_campaign_admission.json`](bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/compact_campaign_admission.json) 为 `status=PASS`、`accepted_compact_workload_timing=1`，精确完成 14 correctness rows / 1,302 samples / 62 summaries；[`正式报告`](reports/tier5_dsa/production_compact14_scoped_formal_20260805.md)。它不完成或替代 exact-26，legacy timing/CTA 字段保持 0、headroom 保持未定义 |
| 最大剩余缺口 | 原计划 §9 exact-26 仍未完成；compact 明确排除 32K/1M timing，且 production CTA-readiness/Impl 仍不存在 |
| 当前设计结论 | provisional：Tier 4 的 grid-PDL 后零等待上界不是 CTA 实现；native v9 在 synthetic mechanism proxy 上没有观察到 Impl 收益，且长上下文是 packed proxy。production/LLM CTA headroom 仍未定义，最终坐标继续等待 production exact-26 和真实 CTA-readiness 证据 |

### 下一步（按序）

1. **原计划 exact-26**：compact scoped formal 已完成，但不替代 32K/1M timing；exact-26 保持未完成且不得复用两个已拒绝 campaign 的 timing。
2. **历史 Tier 0 统计债**：§4.2/§4.4/§4.5 的旧报告缺少当前 §3.7 要求的 raw repeats + CI；需重写对应 harness 后新 session，不用 v4 的单个汇总行补洞。

Tier 1、Tier 0.1、Tier 0.3 与 Tier 2/3 无需为了补当前覆盖而重跑；换
GPU、driver、资源包络或 workload 时才需要新 session。旧 `cta_dep_bench tier23`
仍是 rejected harness，不得与 native v2 拼接。

---

## 1. 如何续跑或复核

### 1.1 机器角色

```bash
command -v nvidia-smi >/dev/null && echo "GPU BOX" || echo "DEV BOX"
```

- **DEV BOX**：只做夹具、分析、文档与代码审查；不得生成或声称新的 GPU 数字。
- **GPU BOX**：正式 phase 统一从 `run_session.sh` 进入，新结果必须使用新目录，历史结果不得覆盖。

### 1.2 入口

| 目标 | 命令 / 说明 |
|---|---|
| 复现实验到 Gate | `RESULTS=results_<new> SMOKE_RESULTS=smoke_<new> PILOT_SMS=148 STEP_TIMEOUT=300 ./run_session.sh --fresh` |
| 复核正式 Tier 1 | `python3 tools/analyze_pilot.py bench/results_20260805_b200_multiwave_v2/pilot_matrix.log --expected bench/results_20260805_b200_multiwave_v2/pilot_expected_tags.txt --json /tmp/pilot.json --csv /tmp/pilot.csv`，随后 `python3 tools/gate.py /tmp/pilot.json` |
| 复核正式 Tier 0.1 | `python3 tools/validate_tier0_chain.py bench/results_20260805_b200_multiwave_v4 --json /tmp/tier0_chain_validation.json` |
| 复核正式 Tier 0.3 | `python3 tools/validate_tier0_background.py bench/results_20260805_b200_multiwave_v2` |
| 复核正式 Tier 2/3 | `cd bench && python3 ../tools/validate_tier23_native.py results_20260805_b200_tier23_native_v2 --manifest results_20260805_b200_tier23_native_v2/tier23_manifest.tsv --json /tmp/tier23_validation.json --csv /tmp/tier23_summary.csv`；manifest 内路径以 `bench/` 为工作目录，结果应为 35 configs / 5,084 samples / 182,460 trace rows / 0 errors / `PASS` |
| 新跑 Tier 2/3 native | `cd bench && RESULTS=results_<new> GATE_JSON=results_20260805_b200_multiwave_v2/gate.json ./run_tier23_native.sh --fresh`；只使用 `tier23n` / native 入口，旧 `run_all.sh tier23` 仍是 rejected harness |
| 离线夹具链 | 见 `EXPERIMENT_PLAN.md` §10.2；preflight 已覆盖 analyzer→gate |
| 复核 Tier 4 formal | `python3 bench/llm/tier4_finalize.py --results results/tier4_schema_v3_formal_v1_20260805/cohorts/decode --verify-admission`，再对 `prefill` cohort 执行同一命令；两者都应输出 `VERIFY_ADMISSION status=ok` |
| 复核 Tier 5 native v9 | 按 [`native_v9_four_context_formal_20260805.md`](reports/tier5_dsa/native_v9_four_context_formal_20260805.md) 的 CPU-only 命令复核 `campaign_admission.json`、四点 validator 与冻结 SHA；不得从 v6/v7/v8 拒绝目录补样本 |
| 复核 Tier 5 production compact-14 | 按 [`production_compact14_scoped_formal_20260805.md`](reports/tier5_dsa/production_compact14_scoped_formal_20260805.md) 复核 14/1,302/62、fresh `check-final` 与 `compact_campaign_admission.json`；当前 scoped formal 为 `PASS/DONE`，不得把 smoke、long probe、拒绝的 exact-26 或单个 fragment 拼入 timing |

### 1.3 新 session 检查单

- [ ] 阅读计划 §3 准入条件与对应实验的“不能下的结论”
- [ ] `preflight.sh` 无 fail，正式目标架构与实际 GPU 一致
- [ ] 使用全新 `RESULTS` / `SMOKE_RESULTS`；不覆盖本文件 §6 的任何目录
- [ ] Tier 1 检查 expected=93、SUMMARY=93、SAMPLE=14,415、0 invalid，并读取机器 `gate.json`
- [ ] Tier 0.1 运行 strict validator，核对 `semantics=3` epoch 序列、checkpoint/final digest 独立重算、raw pair / CI 与 trace path/row-epoch→最终 SAMPLE 绑定；不把 model-implied depth 当实测层数
- [ ] Tier 0.3 运行严格 validator，而不是只看 `tools/analyze.py` 的展示表
- [ ] Tier 2/3 仅从 native manifest 运行 strict validator，核对 35 配置、31 repeats、Ceiling sentinel、全边/全词校验与 trace；不拼接 rejected v1 或旧 `cta_dep_bench tier23`
- [ ] 检查 `run_all.sh` marker schema 绑定 FAST、完整 argv、binary SHA-256 与 GPU/driver；`--fresh` 在正式目录只清一次，Tier 0 strict 任一失败最终必须使 session 返回 2
- [ ] 新报告同步更新本文件 §0、§2、§4

---

## 2. 计划章节 → 执行进度

### §4 Tier 0 — 基础事实

| 计划节 | 实验 | 状态 | 报告 / 证据边界 |
|---|---|---|---|
| §4.1 | 同 stream 重叠层数（B3） | `DONE` | [`0_1_same_stream_pdl_chain.md`](reports/tier0_base_facts/0_1_same_stream_pdl_chain.md)；当前证据为 v4 `semantics=3` strict validator PASS；实测 peak active grids=2，model-implied depth 不当实测层数 |
| §4.2 | 跨 stream / Graph / diamond（D3） | `PARTIAL` | [`0_2_cross_stream_graph_diamond.md`](reports/tier0_base_facts/0_2_cross_stream_graph_diamond.md)；路径关系可作机制证据，但旧报告无 CI 且 correctness 防陈旧值较弱 |
| §4.3 | waiting CTA 容量（B2） | `DONE` | [`0_3_resident_wait_throughput.md`](reports/tier0_base_facts/0_3_resident_wait_throughput.md) 同时保留实际 registers/local bytes、occupancy API 上限和全输出校验；旧 [`0_3_occupancy_capacity.md`](reports/tier0_base_facts/0_3_occupancy_capacity.md) 只作容量诊断 |
| §4.3 | productive-background 配对定价（B2） | `DONE` | [`0_3_resident_wait_throughput.md`](reports/tier0_base_facts/0_3_resident_wait_throughput.md)；15 点、930 samples、trace validator PASS |
| §4.4 | CLC `try_cancel`（B4） | `PARTIAL` | [`0_4_clc_try_cancel.md`](reports/tier0_base_facts/0_4_clc_try_cancel.md)；单点机制探针，旧报告仅10 repeats/无 CI |
| §4.5 | fence scope（C2） | `PARTIAL` | [`0_5_fence_scope.md`](reports/tier0_base_facts/0_5_fence_scope.md)；饱和循环标定，旧报告仅5 repeats/无 raw CI |

### §5–§6 Tier 1 + Gate

| 计划节 | 实验 | 状态 | 报告 / 证据边界 |
|---|---|---|---|
| §5.1 | degree × structure × grid 收益地图 | `DONE` | [`multiwave_degree_structure_map.md`](reports/tier1_benefit_map/multiwave_degree_structure_map.md)；88 个 Gate 配置 |
| §5.2 | tail / prologue 比 1→16 | `DONE` | 同一报告；5 个点，不进入 Gate |
| §5.3 | 2×/8×/32×SM 真多波 | `DONE` | 逐 CTA `%globaltimer` 证明 unstarted producer、最终前进、early Floor 与 wrong Ceiling |
| §6 | Tier 1 Gate | `DONE` | [`gate.json`](bench/results_20260805_b200_multiwave_v2/gate.json)：`GO`，完整 sweep |

### §7 Tier 2 / 3 — 机制与特定维度

| 计划节 | 实验 | 状态 | 报告 / 证据边界 |
|---|---|---|---|
| §7.1 | 同步协议横评（B1） | `DONE` | [`7_1_protocol_7_3_encoding.md`](reports/tier2_mechanisms/7_1_protocol_7_3_encoding.md)；3 个 grid 规模上的 grid / fixed-spin / backoff / identity-safe prefix / wrong Ceiling，配对 background、全边校验与 trace 全部通过；逻辑 acquire loads 不是 L2 requests，ready→snapshot 不是纯软询唤醒延迟 |
| §7.2 | 等待位置定价（B2） | `DONE`（条件化 scenario envelope） | [`0_3_resident_wait_throughput.md`](reports/tier0_base_facts/0_3_resident_wait_throughput.md) 仅在 `resident ≤ hypothetical [H+] ≤ deferred` 的吞吐/e2e 工程单调性假设下给出每资源点 `[0, measured delta]`；不是 `[H+]` 真机点或无条件保证 |
| §7.3 | 编码成本交叉点（A3） | `DONE` | [`7_1_protocol_7_3_encoding.md`](reports/tier2_mechanisms/7_1_protocol_7_3_encoding.md)；`interval/strided × degree {1…64}` 的 interval / bitmask / CSR 独立 decode、精确 poll 与全边校验；只发布软件 metadata/poll 计数 |
| §7.4 | diamond in-order（A1/B3） | `DONE` | [`7_4_diamond_ordering.md`](reports/tier3_dimensions/7_4_diamond_ordering.md)；10 个 K2:K3 ratio、grid ordered / CTA ordered / CTA unordered / wrong Ceiling，全 stage/block validation 和 branch-envelope trace `PASS` |
| §7.5 | C1 四版本 | `DONE` | [`7_5_c1_locality.md`](reports/tier3_dimensions/7_5_c1_locality.md)；1–64 KiB/CTA 的 separate default/persist、fused cluster、forced-refetch `.cv` 与 wrong Ceiling；全词校验通过，无物理 L2/DRAM counter |
| §7.6 | CLC 持久调度策略（B4） | `DONE` | [`7_6_clc_scheduler.md`](reports/tier3_dimensions/7_6_clc_scheduler.md)；真实 `try_cancel` PTX、pending clusters、token conservation 与 producer/consumer/locality/none 四策略正式横评 |

### §8–§9 Tier 4 / 5 — 真实负载

| 计划节 | 实验 | 状态 | 审计结果 |
|---|---|---|---|
| §8.2 | Qwen `Ceiling − PDL_grid` | `DONE` | [`qwen36_27b_full_decode_bracket_20260805.md`](reports/tier4_llm/qwen36_27b_full_decode_bracket_20260805.md)；4 个 full-decode headline 点、每档 31 repeats；grid→unsafe Ceiling 上界 1.3334% / 1.3094% / 1.8301% / 8.0609%，Ceiling 31/31 错误且 non-finite，只是时间上界 |
| §9 | DSA 真机链 | `PARTIAL`（native formal `DONE`；compact-14 scoped formal `DONE`；exact-26 `INCOMPLETE`） | [`native v9`](reports/tier5_dsa/native_v9_four_context_formal_20260805.md) 固定四点 unified admission `PASS`；[`production compact-14`](reports/tier5_dsa/production_compact14_scoped_formal_20260805.md) 为 `PASS/DONE`、`accepted_compact_workload_timing=1`、14/1,302/62。production exact-26 v1/v2 均 `REJECTED`；compact 不替代 exact-26/CTA bracket，32K/1M timing 排除，production/LLM CTA headroom 未定义，不得声称 §9 全部完成 |

### §10 离线与分析链

| 计划节 | 实验 | 状态 | 报告 / 备注 |
|---|---|---|---|
| §10.1 | 依赖 oracle | `PARTIAL` | [`dependency_oracle.md`](reports/offline/dependency_oracle.md)；符号模型，不是具体 kernel ground truth |
| §10.2 | analyzer / gate 夹具 | `DONE` | 93 配置、31 repeats synthetic fixture 与重试审计链通过；不是 GPU 结果 |
| 前提 | CUDA 13.3 vs 13.4 PTX | `DONE` | [`ptx_13_3_vs_13_4.md`](reports/offline/ptx_13_3_vs_13_4.md) |

### 已拒绝并保留

| 项 | 状态 | 审计报告 |
|---|---|---|
| 原 FAST campaign / `cta_dep_bench` timing | `REJECTED` | [`fast_campaign.md`](reports/rejected/fast_campaign.md) |
| 2026-08-05 首轮多波 formal | `REJECTED / INVALID` | [`tier1_multiwave_trace_incomplete_20260805.md`](reports/rejected/tier1_multiwave_trace_incomplete_20260805.md) |
| Tier 2/3 native v1 race-based Ceiling | `REJECTED` | [`tier23_native_v1_ceiling_race_20260805.md`](reports/rejected/tier23_native_v1_ceiling_race_20260805.md)；accepted timing=0，不得与 v2 拼接 |
| Tier 4 旧不合格 timing 路径 | `REJECTED` | [`tier4_llm_semantic_audit.md`](reports/rejected/tier4_llm_semantic_audit.md)；后续 schema-v3 formal 是独立新路径，不复用旧 timing |
| Tier 5 旧 timing 路径 | `REJECTED` | [`tier5_dsa_semantic_audit.md`](reports/rejected/tier5_dsa_semantic_audit.md) |
| Tier 5 native v6 formal | `REJECTED` | [`tier5_native_v6_forward_progress_20260805.md`](reports/rejected/tier5_native_v6_forward_progress_20260805.md)；32K exact 在冻结上界 5.345553 倍后仍未完成，相同 binary 的最小诊断定位到 Impl 首次 validation；整轮含 4K accepted timing=0 |
| Tier 5 native v7 formal | `REJECTED` | [`tier5_native_v7_overlap_validator_20260805.md`](reports/rejected/tier5_native_v7_overlap_validator_20260805.md)；旧 overlap predicate 错把 trigger 后、upstream tail 内的合法重叠拒收，修复定义不恢复 v7 timing |
| Tier 5 native v8 formal | `REJECTED` | [`tier5_native_v8_validator_complexity_20260805.md`](reports/rejected/tier5_native_v8_validator_complexity_20260805.md)；GPU raw/trace 完成但旧 `O(Q²D)` validator 使整轮原子拒绝，后续 `O(QD)` replay `PASS` 只证明修复 |
| Tier 5 production long probe v1 | `REJECTED` | [`tier5_production_long_probe_v1_20260805.md`](reports/rejected/tier5_production_long_probe_v1_20260805.md)；FAST/nonformal 1M 定向 segment 以 rc130 中断并原子拒收，只复用几何与 orphan-cleanup 需求，timing/headroom 禁用 |
| Tier 5 production exact-26 v1/v2 | `REJECTED`（canonical scope `INCOMPLETE`） | [`tier5_production_exact26_timebox_rejection_20260805.md`](reports/rejected/tier5_production_exact26_timebox_rejection_20260805.md)；两轮各 sealed 9 rows 后在首个 1M operator row 中断，根级 rejection 禁止复用全部 partial timing；后继 compact-14 是独立新 contract/root，其 scoped PASS 不追认 exact-26 timing |

---

## 3. 最小可交付对照（计划 §12）

| 优先级 | 项 | 进度 |
|---:|---|---|
| 1 | §5.1 degree × structure 收益地图 | **`DONE`**；完整多波正式矩阵 |
| 2 | §8.2 `Ceiling − PDL_grid` | **`DONE`**；4 个 Qwen3.6-27B full-decode headline 点通过正式 admission，grid→unsafe Ceiling 上界为 1.3334%–8.0609% |
| 3 | §4.1 重叠层数 | **`DONE`**；v4 相邻配对 + epoch/digest 正确性 + trace strict PASS |
| 4 | §4.3 occupancy / productive-background 曲线 | **`DONE`**；容量 + 15 点配对定价 |

---

## 4. 已采信报告清单

| 报告 | 计划节 | 主要维度 | 证据边界 |
|---|---|---|---|
| [`campaign_b200_multiwave_20260805.md`](reports/campaign_b200_multiwave_20260805.md) | 总览 | A3 B1 B2 E1 | 本轮 umbrella；Tier 5 的当前权威边界为 native v9 与 production compact-14 scoped formal 均已完成，production exact-26 v1/v2 `REJECTED` 且 canonical 范围 `INCOMPLETE`；各 admission 不得拼样本 |
| [`qwen36_27b_full_decode_bracket_20260805.md`](reports/tier4_llm/qwen36_27b_full_decode_bracket_20260805.md) | §8.2 | E1 | Qwen3.6-27B target-specific full-decode 正式包夹；Ceiling 全部错误，只支持零等待时间上界，不支持可实现 CTA 收益 |
| [`native_v9_four_context_formal_20260805.md`](reports/tier5_dsa/native_v9_four_context_formal_20260805.md) | §9 native formal | B1 B2 A3 | B200 native synthetic 四点正式证据；4K/32K exact、128K/1M packed proxy；Impl 未观察到收益，unsafe Ceiling 不得冒充 headroom |
| [`production_compact14_scoped_formal_20260805.md`](reports/tier5_dsa/production_compact14_scoped_formal_20260805.md) | §9 production scoped formal | E1 | B200 两模型 × 4K/128K × 三 workload + MoE 的独立 compact admission；14/1,302/62，32K/1M timing 排除；只准入 workload-component timing，不提供 CTA bracket/headroom |
| [`multiwave_degree_structure_map.md`](reports/tier1_benefit_map/multiwave_degree_structure_map.md) | §5–§6 | A3 B1 E1 | B200 synthetic，Graph Floor vs priority-stream Impl/Ceiling |
| [`0_3_resident_wait_throughput.md`](reports/tier0_base_facts/0_3_resident_wait_throughput.md) | §4.3 / §7.2 | B2 | 单 CTA producer、LCG background、满容量 waiter grid |
| [`7_1_protocol_7_3_encoding.md`](reports/tier2_mechanisms/7_1_protocol_7_3_encoding.md) | §7.1 / §7.3 | B1 A3 | native v2 协议/编码正式横评；软件逻辑 loads 不冒充物理 cache counter |
| [`7_4_diamond_ordering.md`](reports/tier3_dimensions/7_4_diamond_ordering.md) | §7.4 | A1 B3 | CTA diamond 十个 ratio；全 stage/block 校验与 `%globaltimer` branch envelope |
| [`7_5_c1_locality.md`](reports/tier3_dimensions/7_5_c1_locality.md) | §7.5 | C1 | 分离/persistence/cluster/`.cv`/wrong Ceiling；无权限发布物理 L2/DRAM 数字 |
| [`7_6_clc_scheduler.md`](reports/tier3_dimensions/7_6_clc_scheduler.md) | §7.6 | B4 | CLC pending-cluster 持久调度器；token conservation 和全任务校验 |
| [`corrected_producer_consumer_pilot.md`](reports/tier1_benefit_map/corrected_producer_consumer_pilot.md) | 历史 §5 | B1 A3 E1 | 旧 `P,C≤SM` ratio /不同 launch 路径，只作历史背景，不与 v2 做差归因 |
| [`0_1_same_stream_pdl_chain.md`](reports/tier0_base_facts/0_1_same_stream_pdl_chain.md) | §4.1 | B3 A1 | 报告保留同一 B200 synthetic 结论边界；当前 raw/strict 复核必须使用 v4，不能从已 superseded 的 v3 拼接 sample 或 CI |
| [`0_2_cross_stream_graph_diamond.md`](reports/tier0_base_facts/0_2_cross_stream_graph_diamond.md) | §4.2 | D3 | `PARTIAL`；路径机制证据，缺当前统计/correctness 准入 |
| [`0_3_occupancy_capacity.md`](reports/tier0_base_facts/0_3_occupancy_capacity.md) | §4.3 历史诊断 | B2 | 只给容量；正式容量/代价以 resident-wait 报告为准 |
| [`0_4_clc_try_cancel.md`](reports/tier0_base_facts/0_4_clc_try_cancel.md) | §4.4 | B4 A4 | `PARTIAL`；单点探针，10 repeats/无 CI |
| [`0_5_fence_scope.md`](reports/tier0_base_facts/0_5_fence_scope.md) | §4.5 | C2 | `PARTIAL`；饱和循环标定，5 repeats/无 raw CI |
| [`dependency_oracle.md`](reports/offline/dependency_oracle.md) | §10.1 | A1 A2 A3 | CPU 符号依赖模型 |
| [`ptx_13_3_vs_13_4.md`](reports/offline/ptx_13_3_vs_13_4.md) | 前提 | — | 离线编译对比 |
| [`tier1_multiwave_trace_incomplete_20260805.md`](reports/rejected/tier1_multiwave_trace_incomplete_20260805.md) | 拒绝审计 | — | 只采信失败计数、根因与回归方法；timing 禁用 |
| [`tier23_native_v1_ceiling_race_20260805.md`](reports/rejected/tier23_native_v1_ceiling_race_20260805.md) | §7 拒绝审计 | — | v1 Ceiling 依赖调度 race；accepted timing=0，只复用根因与 sentinel 修复审计 |
| [`tier4_llm_semantic_audit.md`](reports/rejected/tier4_llm_semantic_audit.md) | §8 | — | 准入 blocker；性能样本为 0 |
| [`tier5_dsa_semantic_audit.md`](reports/rejected/tier5_dsa_semantic_audit.md) | §9 | A2 A3 | 准入/显存审计；性能样本为 0 |
| [`tier5_native_v6_forward_progress_20260805.md`](reports/rejected/tier5_native_v6_forward_progress_20260805.md) | §9 拒绝审计 | B1 B2 | v6 32K resident-wait forward-progress 失败；只采信失效位置、整轮清零与 v7 修复契约，禁用全部 timing |
| [`tier5_native_v7_overlap_validator_20260805.md`](reports/rejected/tier5_native_v7_overlap_validator_20260805.md) | §9 拒绝审计 | B1 B2 | v7 overlap predicate 语义错误；冻结 trace 只用于根因复算，timing 禁用 |
| [`tier5_native_v8_validator_complexity_20260805.md`](reports/rejected/tier5_native_v8_validator_complexity_20260805.md) | §9 拒绝审计 | B1 B2 | v8 旧 `O(Q²D)` CPU validator 阻断统一准入；线性 replay 只证明修复，不恢复 timing |
| [`tier5_production_long_probe_v1_20260805.md`](reports/rejected/tier5_production_long_probe_v1_20260805.md) | §9 production 拒绝审计 | — | 1M FAST/nonformal 中断 segment；只保留工作量几何、原子拒收和 process-group cleanup 契约 |
| [`tier5_production_exact26_timebox_rejection_20260805.md`](reports/rejected/tier5_production_exact26_timebox_rejection_20260805.md) | §9 production exact-26 拒绝审计 | — | v1/v2 均整轮 `REJECTED`；只保留 frozen contract、分区诊断和拒收边界，partial rows/timing 禁止复用，canonical exact-26 仍 `INCOMPLETE` |
| [`fast_campaign.md`](reports/rejected/fast_campaign.md) | 历史拒绝 | — | 原 `cta_dep_bench` timing 禁用 |

---

## 5. 设计空间维度 → 当前证据

| 维度 | 当前证据 | 仍缺什么 |
|---|---|---|
| A1 / B3 | v4 Tier 0.1 trace peak=2 grids / 296 CTA；native v2 进一步覆盖 CTA ordered/unordered diamond 的 10 个 K2:K3 ratio | 换设备/资源包络需重测；diamond 仍是固定 K2 的 synthetic spin-cycle 模型 |
| D3 / C2 | 历史 Tier 0 机制探针 | 补 raw repeats、CI 与对应 correctness 后才能从 `PARTIAL` 升级 |
| A2 | 离线 oracle | 具体生产 kernel 的 CTA 映射 ground truth |
| A3 | Tier 1 degree/structure 多波地图 + native v2 interval/bitmask/CSR；strided degree=2 起 exact set 避免大量假边，CSR/bitmask decode 翻转位于 degree 4–8 | 没有硬件解码器、物理 cache traffic 或生产 kernel 结构的实测交叉点 |
| B1 | native v2 grid / fixed-spin / backoff / identity-safe prefix，配对 background 和逻辑 acquire-load 计数；prefix 轮询放大 14.514×→113.209× | 无 NCU L2 requests；ready→snapshot 含 CTA 放置/decode/snapshot，不是纯软询唤醒延迟 |
| B2 | resident capacity + productive-background 配对的条件化 scenario envelope | 真实 `[H+]` pre-dispatch 实现与生产 workload |
| B4 | native v2 真实 CLC PTX、pending clusters、token conservation 与三种安全策略；本点 producer-priority 最快 | 不能外推到多对多、非均匀任务或硬件 TB scheduler |
| C1 | 1–64 KiB/CTA 的 separate/persist/fused cluster/`.cv` 横评；64 KiB fused 为 1.125% 且 active-block API 上限 16→3 | 无物理 L2/DRAM counter；cluster 是同 kernel 可实现上界控制，不是跨 kernel CTA-PDL |
| E1 | 32→4736 CTA、2×/8×/32× trace-proven 收益边界 | 真实 LLM/DSA headroom |
| D1 / D2 | 无对口完整实验 | 集中式提交/分布式领取的直接横评与工程实现 |
| Tier 4 | Qwen3.6-27B full-decode target-specific formal：4 个 batch 点，grid→unsafe Ceiling 1.3334%–8.0609%；PTX/cubin、active variant 与 Nsight graph-node admission 闭环 | 可实现 CTA Impl、其他模型/服务负载与 B300/Rubin 外推仍未知 |
| Tier 5 | native v9 四点 unified admission `PASS`；production compact-14 scoped formal `PASS/DONE`、14/1,302/62、`accepted_compact_workload_timing=1` | production exact-26 v1/v2 已拒收、canonical 范围仍 `INCOMPLETE`；compact 明确排除 32K/1M timing，且不提供 CTA bracket；production/LLM CTA headroom 未定义，仍缺真实 production CTA-readiness/Impl 证据 |

---

## 6. Provenance 与不可变快照

### 本轮成功正式证据

- 原始目录：[`bench/results_20260805_b200_multiwave_v2/`](bench/results_20260805_b200_multiwave_v2/)
- 配套 smoke：[`bench/smoke_20260805_formal_v2/`](bench/smoke_20260805_formal_v2/)
- 收集归档：[`cta_pdl_results_20260805_050659.tar.gz`](cta_pdl_results_20260805_050659.tar.gz)
- 官方 FAST plumbing：[`bench/results_20260805_smoke_v5/`](bench/results_20260805_smoke_v5/) 与 [`bench/smoke_20260805_v5/`](bench/smoke_20260805_v5/)
- FAST 收集归档：[`cta_pdl_results_20260805_050122.tar.gz`](cta_pdl_results_20260805_050122.tar.gz)
- Tier 0.1 当前正式源/全链回归：[`bench/results_20260805_b200_multiwave_v4/`](bench/results_20260805_b200_multiwave_v4/) 与 [`bench/smoke_20260805_formal_v4/`](bench/smoke_20260805_formal_v4/)
- v4 两套 strict 结果：[`tier0_chain_validation.json`](bench/results_20260805_b200_multiwave_v4/tier0_chain_validation.json)（6 配置、12 validation、372 samples、1,776 trace rows）与 [`tier0_background_validation.json`](bench/results_20260805_b200_multiwave_v4/tier0_background_validation.json)（15 配置、930 samples、68,998 trace rows），均为 `PASS`
- v4 收集归档：[`cta_pdl_results_20260805_061446.tar.gz`](cta_pdl_results_20260805_061446.tar.gz)
- Tier 2/3 native v2 正式源：[`bench/results_20260805_b200_tier23_native_v2/`](bench/results_20260805_b200_tier23_native_v2/)；[`tier23_validation.json`](bench/results_20260805_b200_tier23_native_v2/tier23_validation.json) 为 35 配置、5,084 samples、182,460 trace rows、0 errors / `PASS`
- Tier 2/3 不可变 SHA-256：`tier23_validation.json`=`b177f84e5f3d3e6a3ba4dc4f9c6d7a32a34d03e6487ae536cafa1e71fcb5919b`；`tier23_summary.csv`=`a9d19052c2a2cf75b548e27a776b25c2c615fe1c3dae67b99062ddc22eae0ea2`；`tier23_matrix.log`=`0ff9e05efc9fa303923ac0dcbe8287f9966ae6fa1c808455a981aeb16dab9352`；`tier23_manifest.tsv`=`69e136f7452659748dd9cac5bee3b9d8c5f7b1c6baa882953dad3c229e3b0a2d`
- Tier 2/3 profiler / sanitizer 边界：[`nsys_status.txt`](bench/results_20260805_b200_tier23_native_v2/nsys_status.txt) 保留成功 Systems 采集；[`ncu_status.txt`](bench/results_20260805_b200_tier23_native_v2/ncu_status.txt) 保留 `ERR_NVGPUCTRPERM`；[`sanitizer_v2_coverage.json`](bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_coverage.json) 明示同 binary SHA 的 non-timing 覆盖边界
- Tier 4 schema-v3 正式源：[`results/tier4_schema_v3_formal_v1_20260805/`](results/tier4_schema_v3_formal_v1_20260805/)；decode 与 prefill cohort 的 `tier4_finalize.py --verify-admission` 均为 `status=ok`
- Tier 4 不可变 SHA-256：`manifest.json`=`d3b80d2839ce7c897cad8bf8fae225358ec51d0a352bef0cb665712befad3508`；decode `admission.json`=`a9bbe54df301b141250cc1a65dcc2fb6050cc16b545212f558cb992ae84ba128`、`raw_triplet.json`=`6abeab84d683d94e54445e8cb4614cc01f725f961116183a512e484587de0083`；prefill `admission.json`=`744b03b92bc2d71c3e98bb0451092e5bfeeb6bbb87e1bc7ab74f6efb3434122c`、`raw_triplet.json`=`ddd26be702cc4346084da28a30fb59af5ce31f78fad48fa4fcdc5d3b4a1fb4bd`
- Tier 5 native v9 正式源：[`bench/dsa/results_20260805_b200_native_formal_strict_v9/`](bench/dsa/results_20260805_b200_native_formal_strict_v9/)；[`campaign_admission.json`](bench/dsa/results_20260805_b200_native_formal_strict_v9/campaign_admission.json) 为 `PASS`、`accepted_timing=1`、`errors=[]`，SHA-256=`135db12fc524e629525dc5b404fe9603ac67a6704e54bc27fe5ecc22a6bb7491`
- Tier 5 native v9 证据边界：四点 strict validation、PTX/resource/SASS、Nsight sidecar、NCU permission classification 与 GPU 独占监控均纳入 unified finalizer；Nsight 不进 timing matrix，NCU 因 `ERR_NVGPUCTRPERM` 不支持物理 counter 结论
- Tier 5 production compact-14 scoped formal：[`结果根`](bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/) 与 [`compact_campaign_admission.json`](bench/dsa/results_20260805_b200_production_compact_formal_v1_16g/compact_campaign_admission.json) 为 `PASS`、`accepted_compact_workload_timing=1`、14/1,302/62；admission SHA-256=`e76daf0b27bc8d1082126f135a2d57eb78a9caed18fa59f62c011cad4465069a`
- compact closure：source manifest SHA-256=`5c5f80900529728d5f57a8eb78a6d55c10ffb3772bf0a46be64541fa0778c89c`，runtime build SHA-256=`1a6741f693e40777ad2810fd2e0331b6f6854757e5823fc09ff03bc4954d3444`；[`Nsight sidecar v4`](bench/dsa/results_20260805_b200_production_nsys_sidecar_v4_compactsource/nsys_sidecar.json) 独立 `PASS`，不进入 timing matrix
- compact 边界：`accepted_exact26_workload_timing=0`、legacy `accepted_timing=0` / `accepted_workload_timing=0` / `accepted_CTA_bracket=0`、`headroom_defined=false` / `headroom_pct=null`；32K/1M timing 排除，§9 仍 `PARTIAL`

v2 是 Tier 1 / Tier 0.3 已发布 headline 的唯一数据源；v4 是 Tier 0.1 的唯一正式数据源，并将全链 Gate 再次回归为 `GO`。v4 中同时生成的 Tier 1 / Tier 0.3 只作 session/validator 回归，不替换 v2 headline。此前 v3 Tier 0.1 已被 `semantics=3` v4 **superseded**：它只保留为历史快照，不与 v4 拼接 sample、CI、digest 或 trace，也不再作为当前复核入口。

### 本轮拒绝快照

- 首轮 formal：[`bench/results_20260805_b200_multiwave/`](bench/results_20260805_b200_multiwave/)
- 首轮配套 smoke：[`bench/smoke_20260805_formal/`](bench/smoke_20260805_formal/)
- 首轮归档：[`cta_pdl_results_20260805_044753.tar.gz`](cta_pdl_results_20260805_044753.tar.gz)
- 根因压力回归：[`bench/results_20260805_trace_retry_debug_v2/`](bench/results_20260805_trace_retry_debug_v2/)；只证明修复与分析链，不是正式性能矩阵
- Tier 2/3 native v1：[`bench/results_20260805_b200_tier23_native_v1/`](bench/results_20260805_b200_tier23_native_v1/)；[`tier23_rejection.json`](bench/results_20260805_b200_tier23_native_v1/tier23_rejection.json) 固定 `accepted_timing_samples=0`，只保留 race-based Ceiling 根因与 v2 sentinel 替代链
- Tier 5 native v6/v7/v8 分别为 forward-progress、overlap-validator 和 validator-complexity 原子拒绝；拒绝目录与报告只保留根因/修复证据，不得续跑、拼接或重放接纳 timing
- Tier 5 production exact-26 v1：[`bench/dsa/results_20260805_b200_production_formal_exact26_v1/`](bench/dsa/results_20260805_b200_production_formal_exact26_v1/)；`formal_rejection.json` 固定 `REJECTED`，partial rows/timing 不得复用
- Tier 5 production exact-26 v2：[`bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/`](bench/dsa/results_20260805_b200_production_formal_exact26_v2_16g/)；按用户切换一小时 compact 范围后原子拒收，partial rows/timing 不得复用；canonical exact-26 仍 `INCOMPLETE`
- Tier 5 production long probe v1：[`bench/dsa/results_20260805_b200_production_long_probe_v1/`](bench/dsa/results_20260805_b200_production_long_probe_v1/)；FAST/nonformal 定向 segment 永久拒收，不得与 exact-26 或 compact-14 campaign 拼接

这些目录、raw logs 与 tarball 是历史快照；不得编辑它们来配合后续重命名，也不得把首轮
`INVALID` 的“未受影响点”拼进 v2。Tier 2/3 v1 的安全档局部 PASS 同样不能例外；
该轮任何 log、trace、summary 或 timing 都不得与 native v2 合并。

### 更早的历史 campaign

[`reports/campaign_b200_1gpuh.md`](reports/campaign_b200_1gpuh.md) 及
`bench/results_budget1h_corrected/` 保留为先前 `P,C≤SM` ratio 的机制证据（该 ratio 不自动证明一个实际调度 wave，旧 Tier 0.1 timing 也不在可复用范围内）。2026-08-03 session 的
`EXPERIMENT_MANIFEST_SHA256.txt`、patch/status 与 tarball 未随仓库保存；这是历史缺件，不能
用本轮 artefact 代替。

---

## 7. 建议阅读顺序

1. 本文件 §0–§2：当前进度与 blocker
2. [`campaign_b200_multiwave_20260805.md`](reports/campaign_b200_multiwave_20260805.md)：本轮总览
3. [`multiwave_degree_structure_map.md`](reports/tier1_benefit_map/multiwave_degree_structure_map.md)：完整 Tier 1 地图与 Gate
4. [`0_1_same_stream_pdl_chain.md`](reports/tier0_base_facts/0_1_same_stream_pdl_chain.md)：B3 trace peak=2 grids / 296 CTA 与 model-implied depth 分界
5. [`0_3_resident_wait_throughput.md`](reports/tier0_base_facts/0_3_resident_wait_throughput.md)：B2 代价与 §7.2 条件化 scenario envelope
6. [`7_1_protocol_7_3_encoding.md`](reports/tier2_mechanisms/7_1_protocol_7_3_encoding.md)：Tier 2 协议、编码交叉点与物理 counter 边界
7. Tier 3 三报告：[`diamond`](reports/tier3_dimensions/7_4_diamond_ordering.md)、[`C1`](reports/tier3_dimensions/7_5_c1_locality.md)、[`CLC scheduler`](reports/tier3_dimensions/7_6_clc_scheduler.md)
8. [`qwen36_27b_full_decode_bracket_20260805.md`](reports/tier4_llm/qwen36_27b_full_decode_bracket_20260805.md)：Tier 4 target-specific full-decode 正式上界
9. [`native_v9_four_context_formal_20260805.md`](reports/tier5_dsa/native_v9_four_context_formal_20260805.md)：Tier 5 native synthetic 四点正式结果与 packed-proxy/headroom 边界
10. 九份历史拒绝审计：原 FAST、首轮多波 formal、Tier 2/3 native v1、Tier 4 旧路径、Tier 5 旧路径、native v6/v7/v8 与 production long probe v1
11. 回到 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) 与 [`docs/cta_pdl_design_space.md`](docs/cta_pdl_design_space.md) 判读下一坐标
