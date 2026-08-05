# CTA-Level PDL 实验执行进度

更新时间：2026-08-05  
权威规格：[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)  
代码实现进度：[`bench/README.md`](bench/README.md)

> **本文件只回答三件事**：计划跑到哪了、证据在哪、下一步怎么续跑。  
> 测什么 / 怎么判读 / 阈值是多少 → 只改计划；结论与数字 → 只写在 `reports/`。

状态图例：

| 状态 | 含义 |
|---|---|
| `DONE` | 有可信结果与报告，满足对应计划节的可采信边界 |
| `PARTIAL` | 已跑一部分，但未覆盖计划要求的全部范围或语义 |
| `BLOCKED` | 缺 harness / 语义改动，上机前必须先改代码（见计划 §13） |
| `READY` | 代码与前置条件齐，可直接开跑 |
| `GATED` | 等 §6 判决（及覆盖边界）后再决定是否跑 |
| `REJECTED` | 跑过，但语义无效，不得当结论 |

新增或修改报告时，必须同步更新：§0 当前状态、§2 计划进度表、§4 报告清单。

---

## 0. 当前状态（一眼）

| 项 | 值 |
|---|---|
| 最近有效 session | B200，148 SM，CC 10.0，`<1 GPU·h` corrected campaign |
| 原始日志 | [`bench/results_budget1h_corrected/`](bench/results_budget1h_corrected/) |
| 总报告 | [`reports/campaign_b200_1gpuh.md`](reports/campaign_b200_1gpuh.md) |
| §6 Gate（已有数据） | **机制可行**（typical Space ≫ 8%），矩阵仅 `P,C ≤ SM` → `single_wave_only` |
| DEV BOX（2026-08-05） | 离线夹具链 + gate 自检通过；`cta_dep_pilot` / `tier1p` **已放开多波**（未真机编译） |
| 项目最大缺口 | **§5.3 多波真机数字**（harness/`run_all.sh` 已就绪 → 等 GPU） |
| 第二缺口 | **§8.2 `Ceiling − PDL_grid`**（真实负载上界，尚未测量） |

### 下一步（按序）

1. **租 GPU**：`./run_session.sh` 或 `cd bench && ./run_all.sh --fresh tier1p`（默认 grids 含 `2×/8×/32× SM`）。先 `FAST=1` 冒烟再全量。
2. **回写 gate**：确认 `plan_multi_complete=true` 后，按新 caveats 读判决。
3. **开 Tier 4 session**：测 `Ceiling − PDL_grid`。
4. **仅当多波 gate = `GO` 且 `plan_multi_complete`**：再订机器跑 §7 Tier 2/3。

---

## 1. 怎么基于当前进度续跑

### 1.1 机器角色

```bash
command -v nvidia-smi >/dev/null && echo "GPU BOX" || echo "DEV BOX"
```

- **DEV BOX**：改 harness、写报告、跑 §10 离线链与夹具；**不要编造 GPU 数字**。CUDA `.cu` 改动须标注未编译验证。
- **GPU BOX**：只跑下面命令；断点续跑靠 `bench/results/<step>.done`。

### 1.2 续跑命令

| 目标 | 前置 | 命令 |
|---|---|---|
| **整场（含写报告）** | GPU + `codex` CLI | `./codex/run_campaign.sh`（见 [`codex/README.md`](codex/README.md)：脚本管测量，agent 管判断与行文） |
| 离线自检 | 无 | 见计划 §10.2；或夹具链（下方） |
| 多波 Tier 1p（下一步主路径） | GPU + 已合入的 pilot | `cd bench && FAST=1 ./run_all.sh --fresh tier1p` 冒烟 → 去掉 `FAST` 全量 |
| 整场决策点 session | 同上 | `./run_session.sh`（可续跑） |
| 只重跑 Tier 0 | 无 | `cd bench && ./run_all.sh tier0` |
| 单独写 gate | 已有 `pilot_analysis.json` | `python3 tools/gate.py bench/results/.../pilot_analysis.json` |
| Tier 4 LLM | gate ∈ {`GO`,`LLM_ONLY`} 等 | `cd bench/llm && FAST=1 ./run_llm_sweep.sh`（**勿在 8GB 开发机下模型**） |
| Tier 2/3 | 多波 gate=`GO` + `plan_multi_complete` + §13 harness | 勿用已否决的 `tier23` |

离线夹具链（DEV BOX，轻量）：

```bash
python3 tools/make_test_fixtures.py --out /tmp/ctafix
python3 tools/analyze_pilot.py /tmp/ctafix/pilot_matrix.log \
        --json /tmp/ctafix/pilot_analysis.json --csv /tmp/ctafix/pilot_summary.csv
python3 tools/gate.py /tmp/ctafix/pilot_analysis.json
```

### 1.3 续跑检查清单（开跑前勾）

- [ ] 读计划对应节的「不能下的结论」与 §3 准入条件
- [ ] `bench/README.md` 确认该 phase 的 harness **不是** `REJECTED`
- [ ] 多波 / Tier 2/3：确认 §13 代码缺口已关（多波 harness：已关；Tier 2/3：未关）
- [ ] 结果目录：默认 `bench/results/`；历史 corrected 数据勿覆盖，新跑用 `RESULTS=results_<tag>`
- [ ] 跑完：写/更新 `reports/`，并回写本文件 §0 / §2 / §4

---

## 2. 计划章节 → 执行进度

对齐 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) 的节号。

### §4 Tier 0 — 基础事实

| 计划节 | 实验 | 状态 | 报告 / 日志 | 备注 |
|---|---|---|---|---|
| §4.1 | 同 stream 重叠层数（B3） | `DONE` | [0_1_same_stream_pdl_chain.md](reports/tier0_base_facts/0_1_same_stream_pdl_chain.md) | chain6 1.7130×；trace 峰值仅两层 148-CTA grid |
| §4.2 | 跨 stream / Graph / diamond（D3） | `DONE` | [0_2_cross_stream_graph_diamond.md](reports/tier0_base_facts/0_2_cross_stream_graph_diamond.md) | eager ≈1.00×；Graph/same-stream ≈2.00× |
| §4.3 | waiting CTA occupancy（B2） | `PARTIAL` | [0_3_occupancy_capacity.md](reports/tier0_base_facts/0_3_occupancy_capacity.md) | **仅容量曲线**；无 productive background → 不能定价吞吐代价 |
| §4.4 | CLC `try_cancel`（B4） | `DONE` | [0_4_clc_try_cancel.md](reports/tier0_base_facts/0_4_clc_try_cancel.md) | 1728 唯一领取，duplicate=0 |
| §4.5 | fence scope（C2） | `DONE` | [0_5_fence_scope.md](reports/tier0_base_facts/0_5_fence_scope.md) | 饱和循环增量；不能直接当 Ideal 注入 |

### §5–§6 Tier 1 + Gate — 决策点

| 计划节 | 实验 | 状态 | 报告 / 日志 | 备注 |
|---|---|---|---|---|
| §5.1 | degree × structure 收益地图 | `PARTIAL` | [corrected_producer_consumer_pilot.md](reports/tier1_benefit_map/corrected_producer_consumer_pilot.md) | 单波有效；多波扫参已进驱动，**缺真机** |
| §5.2 | tail / prologue 比 | `PARTIAL` | 同上 | pilot 覆盖部分比 |
| §5.3 | 多波 `P,C > SM` | `READY` | harness 已改；无真机报告 | `cta_dep_pilot` 去掉 `P,C≤SM` 闸；`wave=` 入 SUMMARY；`tier1p` 默认含 2/8/32×SM。**.cu 未在本机编译** |
| §6 | Gate | `PARTIAL` | 既有单波 analysis + 更新后的 `tools/gate.py` | 既有数据仍 `single_wave_only`；新 gate 会检查 `plan_multi_complete` |

### §7 Tier 2/3 — 机制对比（`GATED`）

| 计划节 | 实验 | 状态 | 阻塞 |
|---|---|---|---|
| §7.1 | 同步协议横评（B1） | `BLOCKED` | 需满足 §3 的协议 harness；旧 `tier23` = `REJECTED` |
| §7.2 | 等待位置定价（B2） | `BLOCKED` | 依赖完整 §4.3 代价曲线（含 background 吞吐） |
| §7.3 | 编码成本交叉点（A3） | `BLOCKED` | 需合法 harness |
| §7.4 | diamond in-order（A1/B3） | `BLOCKED` | 计划 §13.5 |
| §7.5 | C1 四版本 | `BLOCKED` | 计划 §13.5 |
| §7.6 | CLC 持久调度策略（B4） | `BLOCKED` | §4.4 可行性前提已有；策略对比未做 |

### §8–§9 Tier 4 / 5 — 真实负载

| 计划节 | 实验 | 状态 | 备注 |
|---|---|---|---|
| §8.2 | `Ceiling − PDL_grid` | `READY`（逻辑）/ 未执行 | **最关键单个数字**；需 GPU + ~54GB 模型，**不要在 8GB 开发机做** |
| §9 | DSA 真机链 | 未执行 | 有离线 oracle |

### §10 离线

| 计划节 | 实验 | 状态 | 报告 / 备注 |
|---|---|---|---|
| §10.1 | 依赖 oracle | `PARTIAL` | [dependency_oracle.md](reports/offline/dependency_oracle.md)；DEV BOX 轻量复跑 qwen/glm 小配置 OK（~17MB RSS） |
| §10.2 | 分析链夹具 | `DONE` | 2026-08-05 复跑通过；夹具含多波点时 gate 报 `plan_multi_complete` |
| 前提 | CUDA 13.3 vs 13.4 PTX | `DONE` | [ptx_13_3_vs_13_4.md](reports/offline/ptx_13_3_vs_13_4.md) |

### 已否决（保留审计）

| 项 | 状态 | 报告 |
|---|---|---|
| 原 FAST campaign / `cta_dep_bench` 计时 | `REJECTED` | [fast_campaign.md](reports/rejected/fast_campaign.md) |

---

## 3. 最小可交付对照（计划 §12）

| 优先级 | 项 | 进度 |
|---|---|---|
| 1 | §5.1 degree × structure 地图 | `PARTIAL`（单波有数；多波 harness `READY`，等 GPU） |
| 2 | §8.2 `Ceiling − PDL_grid` | 未执行 |
| 3 | §4.1 重叠层数 | `DONE` |
| 4 | §4.3 occupancy 曲线 | `PARTIAL`（容量 `DONE`，代价定价未做） |

---

## 4. 已采信报告清单

| 实验 | 计划节 | 主要维度 | 报告 | 证据边界 |
|---|---|---|---|---|
| Corrected producer–consumer pilot | §5 / §6 | B1 A3 E1 | [corrected_producer_consumer_pilot.md](reports/tier1_benefit_map/corrected_producer_consumer_pilot.md) | 机制 GO；仅 `P,C ≤ SM` |
| Same-stream 1–6 stage PDL chain | §4.1 | B3 A1 | [0_1_same_stream_pdl_chain.md](reports/tier0_base_facts/0_1_same_stream_pdl_chain.md) | synthetic timing/trace |
| Cross-stream / Graph / diamond | §4.2 | D3 | [0_2_cross_stream_graph_diamond.md](reports/tier0_base_facts/0_2_cross_stream_graph_diamond.md) | 路径限定 |
| Occupancy capacity | §4.3 | B2 | [0_3_occupancy_capacity.md](reports/tier0_base_facts/0_3_occupancy_capacity.md) | 仅容量，非 waiting 吞吐代价 |
| Fence scope calibration | §4.5 | C2 | [0_5_fence_scope.md](reports/tier0_base_facts/0_5_fence_scope.md) | 饱和循环 |
| CLC try_cancel | §4.4 | B4 A4 | [0_4_clc_try_cancel.md](reports/tier0_base_facts/0_4_clc_try_cancel.md) | 单点探针 |
| Dependency oracle | §10.1 | A2 A3 A1 | [dependency_oracle.md](reports/offline/dependency_oracle.md) | CPU 解析模型 |
| CUDA 13.3 vs 13.4 PTX | 前提 | — | [ptx_13_3_vs_13_4.md](reports/offline/ptx_13_3_vs_13_4.md) | 离线编译 |
| 原 FAST campaign | — | — | [fast_campaign.md](reports/rejected/fast_campaign.md) | **弃用** |

---

## 5. 设计空间维度 → 证据（摘要）

| 维度 | 现有证据等级 | 相对计划的缺口 |
|---|---|---|
| A1–A4 / B1 / B3–B4 / C2 / D3 | 见既有报告 | 机制对比与集中式等仍缺 |
| B2 | 容量曲线 | 吞吐代价与派发前门控反推仍缺 |
| C1 / D1 / D2 | 无 / 无对口 | §7.5 等 |
| E1 | 单波 pilot | **多波 degree×grid 真机地图未测** |
| 真实负载 Tier 4/5 | 无 | `Ceiling − PDL_grid` 未测 |

---

## 6. Provenance（历史快照，勿改）

[reports/campaign_b200_1gpuh.md](reports/campaign_b200_1gpuh.md) 配套材料——**内容是当时快照，不要为后来的改名去改它们**：

- [bench/results_budget1h/](bench/results_budget1h/)（旧、已弃用）与 [bench/results_budget1h_corrected/](bench/results_budget1h_corrected/)（corrected）

以下四件是 2026-08-03 那场 session 在租用机上生成的，**未随仓库保留**，因此本文件与两份报告
只记名字、不给链接（2026-08-05 核实：子树内不存在）。若原件还在别处，放回本目录即可恢复引用：

- `EXPERIMENT_MANIFEST_SHA256.txt`（数据与源码 SHA-256 清单）
- `EXPERIMENT_TRACKED_CHANGES.patch`、`EXPERIMENT_GIT_STATUS.txt`
- `cta_pdl_b200_budget1h_20260803.tar.gz`（+ `.sha256`）

---

## 7. 建议阅读顺序（判读，非执行）

1. 本文件 §0–§2 —— 现在该干什么  
2. [corrected pilot](reports/tier1_benefit_map/corrected_producer_consumer_pilot.md) —— 机制可行性与单波边界  
3. Tier 0 基础事实报告  
4. [dependency oracle](reports/offline/dependency_oracle.md)  
5. [rejected FAST audit](reports/rejected/fast_campaign.md)  
6. 回 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) 对应节
