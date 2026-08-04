# CTA-Level PDL 实验索引

更新时间：2026-08-04  
目录：`cta_level_PDL_design`

这份文件是**设计空间与实验证据之间的连接表**。[`docs/cta_pdl_design_space.md`](docs/cta_pdl_design_space.md)
只枚举维度和选项、不给结论；结论只存在于 `reports/` 下的各份报告里；本文件负责回答
「**某个维度目前有没有证据、证据强到什么程度、缺口是什么**」。

新增或修改报告时必须同步更新本文件的 §1 覆盖矩阵、§2 清单和 §4 未执行清单。

---

## 1. 设计空间维度 → 证据覆盖矩阵

证据等级说明：**实测**＝真机产出并有原始日志；**离线**＝CPU 解析模型或编译产物；**无**＝没有可信的已执行结果。

### A 组：依赖的描述

| 维度 | 该维度待答的关键问题 | 现有证据 | 等级 | 缺口 |
|---|---|---|---|---|
| A1 跨 kernel 跨度 | CTA 依赖要跨几代 kernel 表达；span>1 是否必要 | oracle：GLM IndexShare 的 span 为 1–4 个 kernel，但 CTA 依赖仍是一对一 interval | 离线 | **Tier 3.1 未做**：in-order completion 在 diamond 上的代价、K2/K3 时长比扫描 |
| A2 依赖信息来源 | 依赖能否静态推出、精度与成本如何 | oracle：7 个模型/长度配置的解析式 tile 推导 | 离线 | 与真实 kernel trace 对照的假边率/漏边率未测；launch 期分析耗时是否随 gridDim 线性增长未测 |
| A3 表示与编码 | 区间／掩码／CSR 的成本交叉点在哪 | oracle：degree=8192 仍是精确连续 interval，tightness 1.0、false edge 0%。pilot：strided degree=32 的 interval tightness 仅 0.2264，exact 枚举相对 interval cover 可恢复约 20.2 µs | 离线＋实测 | **Tier 2.3 无有效结果**：编码交叉点只在被否决的 campaign 里跑过 |
| A4 持有者与方向 | 集中式 vs 分布式、pull vs push | CLC probe：1728 次成功领取、duplicate=0，说明硬件仲裁的分布式取工可行 | 实测 | 集中式是 `[H+]`，只能包夹估值、未做；per-CTA 分布式的完整实现未测；push 方向无实验 |

### B 组：依赖的执行

| 维度 | 该维度待答的关键问题 | 现有证据 | 等级 | 缺口 |
|---|---|---|---|---|
| B1 同步协议 | 就绪信号怎么产生、怎么等 | pilot：grid / none / interval-spin / interval-backoff / exact-backoff 五种模式，3720 timed samples | 实测 | 轮询的 L2 读请求量、对并发 background kernel 的干扰未测。注：单调完成计数器协议已被证伪（cardinality 不含 identity）并删除 |
| B2 等待位置与 occupancy | 等待占槽位要付多少代价 | 0.3：128 threads 下 0/8/16/32/64 KiB 的容量为 16/16/13/6/3 CTA/SM，全卡 2368→444 CTA | 实测（仅容量） | **B2 的定价依据仍然缺失**：0.3 没有 productive background kernel，因此「resident waiting CTA 抢走了多少真实吞吐」未知，「派发前门控值多少」无法反推 |
| B3 乱序窗口与完成顺序 | B200/B300 上实际能重叠几层 | 0.1：chain 1→6 的 speedup 从 1.3318× 增至 1.7130×；但 trace 峰值仅 296 个活跃 CTA，即最多两个 148-CTA grid 同时活跃。`implied_depth=5.969` 是模型反解参数，**不是驻留深度** | 实测 | 完成顺序约束（强制 in-order vs 允许乱序）未做对比 |
| B4 调度策略与资源分区 | producer/consumer/局部性优先谁更好 | 0.4：4096 launch units、1728 成功取消、0 重复、1650.83 cycles/attempt，说明用 CLC 持久化 kernel 在软件里复现调度策略是可行路径 | 实测（单点） | **Tier 3.3 未做**：三种优先级策略的实际对比 |

### C 组：数据与局部性

| 维度 | 该维度待答的关键问题 | 现有证据 | 等级 | 缺口 |
|---|---|---|---|---|
| C1 shared memory 与数据复用 | 跨 kernel 的 on-chip 复用能省多少 | **无** | 无 | **Tier 3.2 完全未做**：融合+DSMEM（上界）／L2 persistence／默认／强制 bypass（下界）四版本对比，以及融合导致的 occupancy 损失平衡点 |
| C2 内存可见性与一致性 | 可见性粒度要下推到多细、成本多少 | 0.5：baseline-adjusted 增量 CTA 15.440、GPU 705.040、SYS 4444.016 ns/iteration | 实测（饱和循环） | 该数值是饱和循环的 critical-path 指标，**不能直接注入 Ideal 点**；per-CTA release/acquire atomic 的真实成本需单独标定 |

### D 组：工程属性

| 维度 | 该维度待答的关键问题 | 现有证据 | 等级 | 缺口 |
|---|---|---|---|---|
| D1 降级策略 | 依赖推不出来时退到哪 | **无** | 无 | 逐级降级（紧区间→宽区间→grid 级）与全有全无的下界对比未做 |
| D2 正确性与调试 | 依赖描述本身是否 sound | **无**（评估方案 §8 已判定「无对口实验」） | 无 | pilot 的 96 次 per-mode validation PASS 只是 harness 自校验，不是依赖描述 soundness 的验证手段 |
| D3 与 CUDA 抽象的集成 | stream / event / Graph 怎么接 | 0.2：六条路径 + diamond。eager 跨流 programmatic event 约 1.00×；captured graph、built graph edge、same-stream 均约 2.00×；diamond 从 40.754 ms 降到 20.386 ms。**H100 上的结论在 B200 复验成立** | 实测 | CTA 级依赖信息如何下发设备端（A4 选集中式时需要新的 host-device 契约）未探索 |

### E 组 + 前提

| 项 | 该项待答的关键问题 | 现有证据 | 等级 | 缺口 |
|---|---|---|---|---|
| E1 收益边界判据 | 什么参数区间下才有收益 | pilot：tail=0、interval degree=8 时 Floor 1.001568 ms → Impl 0.901792 ms，1.1105×／latency 降 9.952%，只兑现 no-wait gap 的 26.150%；默认 tail=1M 时降 34.989–36.082%；degree 1→64 未出现 32 附近突降 | 实测（受限） | **本项目当前最高优先级缺口**：pilot 只覆盖 `P,C ≤ 148` 的欠填充区间，因此 BlockMaestro 的「degree > 32」与「grid > 2048 TB」两条边界**在 B200/B300 上都还没有被测到**。Tier 1.1 的 degree × grid 收益地图必须在 `P,C > SM` 多波区间重做 |
| 前提：ISA 现状 | CTA 级机制是否已进公开 ISA | PTX 13.3 vs 13.4：两版仍 lowering 为同一组 grid 级 PDL 指令 | 离线 | — |
| 真实负载（Tier 4/5） | grid 级 PDL 之后还剩多少空间 | **无** | 无 | **`Ceiling − PDL_grid` 是整个项目最关键的单个数字，尚未测量。** LLM 端到端与 DSA 真机算子链均未执行 |

---

## 2. 已执行的实验报告

| 实验 | Tier | 主要维度 | 报告 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| Corrected producer–consumer pilot | 1／2 | B1 A3 E1 | [reports/tier1_benefit_map/corrected_producer_consumer_pilot.md](reports/tier1_benefit_map/corrected_producer_consumer_pilot.md) | 有效，synthetic gate | 24 配置／3720 samples；机制 GO，应用收益未知，且仅限 `P,C ≤ SM` |
| Same-stream 1–6 stage PDL chain | 0.1 | B3 A1 | [reports/tier0_base_facts/0_1_same_stream_pdl_chain.md](reports/tier0_base_facts/0_1_same_stream_pdl_chain.md) | 有效，synthetic timing/trace | chain6 1.7130×；trace 峰值仅两个 148-CTA grid |
| Cross-stream / Graph / diamond | 0.2 | D3 | [reports/tier0_base_facts/0_2_cross_stream_graph_diamond.md](reports/tier0_base_facts/0_2_cross_stream_graph_diamond.md) | 有效，路径限定 | eager 跨流 1.00×；same-stream 与 Graph 约 2.00× |
| Occupancy capacity | 0.3 | B2 | [reports/tier0_base_facts/0_3_occupancy_capacity.md](reports/tier0_base_facts/0_3_occupancy_capacity.md) | 有效，仅容量曲线 | 16→3 CTA/SM；**不能**解释成 waiting throughput cost |
| Fence scope calibration | 0.5 | C2 | [reports/tier0_base_facts/0_5_fence_scope.md](reports/tier0_base_facts/0_5_fence_scope.md) | 有效，饱和循环 | CTA/GPU/SYS 增量约 15/705/4444 ns/iteration |
| CLC try_cancel | 0.4 | B4 A4 | [reports/tier0_base_facts/0_4_clc_try_cancel.md](reports/tier0_base_facts/0_4_clc_try_cancel.md) | 有效，单点探针 | 1728 units 唯一领取，duplicate=0 |
| Dependency oracle | 3.4 | A2 A3 A1 | [reports/offline/dependency_oracle.md](reports/offline/dependency_oracle.md) | 有效，CPU 解析模型 | degree 8192 仍可为 tightness=1 的精确 interval |
| CUDA 13.3 vs 13.4 PTX | 前提 | — | [reports/offline/ptx_13_3_vs_13_4.md](reports/offline/ptx_13_3_vs_13_4.md) | 有效，离线编译 | 两版仍 lowering 为同一 grid 级 PDL 指令 |
| 原 FAST campaign | — | — | [reports/rejected/fast_campaign.md](reports/rejected/fast_campaign.md) | **弃用** | 跑过，但 trigger／protocol／correctness 语义无效 |

被弃用的实验**不删除**。`reports/rejected/fast_campaign.md` 作为审计记录保留，它列出了哪些信息仍可复用、
哪些绝对不能当结论，是新 benchmark 的准入清单来源（见 [`AGENTS.md`](AGENTS.md) §4）。

---

## 3. 总报告与 provenance

[reports/campaign_b200_1gpuh.md](reports/campaign_b200_1gpuh.md) 是 `<1 GPU·hour` B200 campaign 的综合报告，
包含预算、环境、gate 与未执行项。§2 的各份独立报告把其中每个实验拆开，补充配置、执行语义、结果与结论边界。

该次 session 的 provenance 材料在目录根部，与上面这份总报告配套，**内容是历史快照，不要为了配合后来的
改名或移动去修改它们**：

- [EXPERIMENT_MANIFEST_SHA256.txt](EXPERIMENT_MANIFEST_SHA256.txt)：当时全树的 SHA256 清单
- [EXPERIMENT_TRACKED_CHANGES.patch](EXPERIMENT_TRACKED_CHANGES.patch)、[EXPERIMENT_GIT_STATUS.txt](EXPERIMENT_GIT_STATUS.txt)：当时的代码状态
- `cta_pdl_b200_budget1h_20260803.tar.gz`（+ `.sha256`）：结果打包
- [bench/results_budget1h/](bench/results_budget1h/)（旧、已弃用）与 [bench/results_budget1h_corrected/](bench/results_budget1h_corrected/)（corrected）：原始日志

---

## 4. 明确没有执行的实验

目录中有脚本或设计，但**没有**可信的已执行结果，因此没有结果报告：

按优先级（依据 [`docs/cta_pdl_eval_plan.md`](docs/cta_pdl_eval_plan.md) §10.2；[`RUNBOOK.md`](RUNBOOK.md) §7 是其镜像）：

1. **Tier 1.1 在 `P,C > SM` 多波区间的 degree × grid 收益地图** —— 决策点，决定整个方向是否成立；
2. **Tier 4 的 `Ceiling − PDL_grid`** —— Qwen3.6-27B 真实 vLLM／GPU 端到端；
3. Tier 3.1 diamond 上 in-order completion 的代价（A1／B3）；
4. Tier 3.2 C1 四版本对比；
5. Tier 3.3 完整 CLC persistent scheduling policy 对比（B4）；
6. Tier 5 DSA 真机 indexer→topk→attention 算子链（目前只有离线 oracle）；
7. resource-realistic occupancy 1–2、`P/C > SM` 的 corrected multi-wave pilot；
8. corrected producer–consumer 的逐 CTA ready／wait-return trace；
9. nsys／ncu profiling；
10. D1 降级策略与 D2 soundness 验证（后者评估方案已判定无对口实验）。

---

## 5. 阅读顺序

若目的是判断「CTA readiness 是否可行」：

1. [corrected producer–consumer pilot](reports/tier1_benefit_map/corrected_producer_consumer_pilot.md) —— 机制可行性
2. [same-stream chain](reports/tier0_base_facts/0_1_same_stream_pdl_chain.md) —— 可达重叠深度
3. [cross-stream / Graph](reports/tier0_base_facts/0_2_cross_stream_graph_diamond.md) —— 哪条接法有效
4. [occupancy](reports/tier0_base_facts/0_3_occupancy_capacity.md)、[fence](reports/tier0_base_facts/0_5_fence_scope.md)、[CLC](reports/tier0_base_facts/0_4_clc_try_cancel.md) —— 三个基础事实
5. [dependency oracle](reports/offline/dependency_oracle.md) —— 真实负载的依赖形态
6. [rejected FAST audit](reports/rejected/fast_campaign.md) —— 旧结果为什么不能用

---

## 6. 目录布局

```
cta_level_PDL_design/
├── AGENTS.md                       工作契约：方法论约束、有效性铁律、报告规范
├── EXPERIMENT_REPORT_INDEX.md      本文（唯一的实验入口）
├── RUNBOOK.md                      租用 GPU 的执行手册
├── rubin_design.md                 Rubin tile-level triggering 的公开描述（原文摘录）
├── docs/                           设计空间、评估方案、接口现状
├── archive/                        侧支与已归档材料，非权威、不作为结论依据
├── reports/
│   ├── campaign_b200_1gpuh.md      B200 campaign 综合报告
│   ├── tier0_base_facts/           0.1–0.5 基础事实
│   ├── tier1_benefit_map/          收益空间
│   ├── offline/                    不需要 GPU 的实验
│   └── rejected/                   被弃用实验的审计记录
├── bench/                          CUDA 微基准与无人值守驱动
├── tools/                          离线分析脚本
├── papers/                         参考文献
└── ptx_study/                      PTX 观察与版本对比
```
