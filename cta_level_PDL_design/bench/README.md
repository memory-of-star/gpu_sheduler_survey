# CTA 级 PDL 评估代码

配套文档：[`../docs/cta_pdl_eval_plan.md`](../docs/cta_pdl_eval_plan.md)

**开发机与实验机分离**：这里的代码在本地写完、语法自检通过，租用 GPU 后只负责产出原始数据，分析回本地做（[`../tools/`](../tools/)）。所有驱动脚本都可**无人值守**运行、**断点续跑**、**失败不中断**。

---

## 目录

```
bench/
├── common/
│   ├── cta_trace.cuh      原语 1：CTA 级 %globaltimer + %smid 打点
│   ├── dep_pattern.cuh    原语 2a：参数化依赖模式（结构与依赖度独立可调）
│   ├── dep_wait.cuh       原语 2b：可互换的同步协议（B1 维度的全部选项）
│   └── bench_util.cuh     公共宿主端工具
├── cta_dep_pilot.cu       Tier 1 收益地图（**修正后的 harness，gate 数据来源**；P,C ≤ SM）
├── cta_dep_bench.cu       旧微基准，**trigger 语义已被否决**，仅保留供复审（见「harness 状态」）
├── tier0_facts.cu         Tier 0 基础事实（重叠层数 / occupancy 代价 / fence 成本）
├── clc_probe.cu           Tier 0.4 CLC try_cancel 特性（需 sm_100+）
├── build.sh               编译（默认 sm_103）
├── run_all.sh             无人值守驱动
├── llm/                   Tier 4 LLM 端到端
│   ├── bench_llm.py       单点测量（三档之一）
│   └── run_llm_sweep.sh   batch × 序列长度扫描
└── dsa/                   Tier 5 DSA 算子链
    ├── dsa_chain.py       indexer → topk → sparse MLA，真实 shape
    └── run_dsa_chain.sh   上下文扫描 + MoE（缩减 expert 数）
```

---

## 快速开始

```bash
# 1. 编译（B300 默认 sm_103；H100 用 ARCH=sm_90）
./build.sh

# 2. 冒烟测试（分钟级，确认一切正常）
FAST=1 ./run_all.sh

# 3. 基础事实 + 修正后的 Tier 1 收益地图
./run_all.sh              # = tier0 + tier1p

# 4. 把 results/ 拷回本地分析
python3 ../tools/analyze_pilot.py results/pilot_matrix.log \
        --json pilot_analysis.json --csv pilot_summary.csv
python3 ../tools/analyze.py results/summary.txt
python3 ../tools/cta_timeline.py results/trace_*.csv
```

---

## harness 状态（先读这段再选 phase）

| phase | 驱动 | 状态 |
|---|---|---|
| `tier0` | `tier0_facts` / `clc_probe` | 有效 |
| `tier1p` | `cta_dep_pilot` | **有效，Tier 1 gate 的唯一合法数据源**。受 `P,C ≤ SM` 限制，只覆盖单波 |
| `tier1` `tier23` | `cta_dep_bench` | **已否决**：它在 PDL trigger 之前就发布全部 `done[]`，等待预先满足，计时不能作为 CTA 收益证据（[`../reports/rejected/fast_campaign.md`](../reports/rejected/fast_campaign.md)）。保留仅供复审，`run_all.sh` 会先打印 WARNING |

`./run_all.sh`（不带参数）**不会**跑 `tier1` / `tier23`——把决策点预算花在已知无效的数字上，正是这个驱动要避免的事。

两个分析脚本不可互换：`analyze.py` 读 `SUMMARY`，`analyze_pilot.py` 读 `SAMPLE` / `SUMMARY_PILOT`。

**当前最大缺口**：multi-wave（`P,C > SM`）收益地图两个二进制都产不出——`cta_dep_bench` 够得着但语义无效，放宽 pilot 的上限属于对 `.cu` 的语义改动。

---

## 两个关键实现细节

### 计时必须用 `%globaltimer`，不能用 `clock64()`

CTA 时间线重建本质是**跨 SM 的事件排序**：

| 计时源 | 性质 | 跨 SM 可比 |
|---|---|---|
| `clock64()` | SM 本地时钟计数器 | **否** |
| `%globaltimer` | 全局纳秒计时器 | 是 |

用错会得到看似合理但实际错乱的重叠关系，且**不会有任何报错**。`cta_trace.cuh` 用前者，`bench_util.cuh` 里的 `spin_cycles()` 用后者——因为那里测的是单 SM 上的持续时长，不涉及跨 SM 比较。

### 依赖度与结构复杂度必须独立扫描

`dep_pattern.cuh` 把两者做成正交的轴：

```bash
# 结构钉死为连续区间，只扫依赖度 → 收益变化归因于"边更多"
./cta_dep_pilot --producers 148 --consumers 148 --structure interval --degree 8

# 依赖度钉死为 32，只扫结构 → 收益变化归因于"形状更难"
./cta_dep_pilot --producers 148 --consumers 148 --structure strided --degree 32
```

`run_all.sh tier1p` 会把这两条轴各自扫完。pilot 只接受 `interval` / `grouped` / `strided` / `self`
四种结构（`random` / `all` / `none` 被显式拒绝），且要求 `P,C ≤ SM`。

**为什么必须这样做**：BlockMaestro Fig.12 注入的是 n-组全连接，两个变量同步增长，所以它的"依赖度 > 32 收益归零"无法区分成因。而真实 LLM 负载恰恰是**高依赖度 + 规整结构**——`tools/dep_oracle.py` 的推导显示，DSA 的 indexer→topk 在 1M 上下文下依赖度 8192（阈值的 256 倍），但区间编码紧度仍是 1.0、假边率 0%。照搬那条阈值会把最常见的模式错误排除。

---

## 四点包夹

`cta_dep_pilot` 每次调用都把下面这组点一次跑完，无需 `--wait` 开关：

| 点 | pilot mode | 含义 |
|---|---|---|
| **Floor** | `grid` | 现状：`griddepcontrol.wait`，整 grid all-or-nothing |
| **Impl** | `interval-spin` / `interval-backoff` / `exact-backoff` | 软件实现的 CTA 级协议 |
| **Ceiling** | `none` | 依赖完全去掉（**结果是错的，只测时间**） |

读数：`Ceiling − Floor` 是总收益空间（太小就放弃该负载）；`Impl − Floor` 是今天软件能拿到的；差额即硬件改动的价值。

`none` 档会跳过校验并在输出中标 `n/a`，这是故意的——它靠读未写入的数据来测"依赖零成本"的时间下界。
单调完成计数器协议**已被删除**：cardinality 不蕴含 identity，`completed_count >= hi+1` 并不表示
`[0,hi]` 这些 parent 都完成了，那不是保守的过度等待而是错误。

旧的 `cta_dep_bench` 用 `--all-waits` / `--wait` 表达同一组点，其中还包含已被证伪的 `cta-counter`。

---

## LLM 部分（Tier 4）

```bash
cd llm && FAST=1 ./run_llm_sweep.sh
python3 ../../tools/llm_bracket.py results_llm/summary_llm.txt
```

**Floor 是 `PDL_grid` 而不是 `PDL_off`**。TRT-LLM / vLLM / SGLang 都已经在生产中启用 grid 级 PDL，拿 PDL-off 当基线会把已经到手的收益重复计算。已公开的 PDL_off→PDL_grid 量级是 2–33%，`llm_bracket.py` 会在结果落在这个带宽之外时告警（通常意味着 CUDA graph 不是 FULL 模式，PDL 被静默关掉了）。

Ceiling 档通过把 Triton 的 `gdc_wait` 打成 no-op 实现，需显式设 `CTA_PDL_CEILING=1`，因为它**故意产生错误结果**。

---

## DSA 部分（Tier 5）

```bash
cd dsa && FAST=1 ./run_dsa_chain.sh
```

整模型单卡放不下，但**注意力路径放得下**。脚本先跑纯离线的依赖推导（不需要 GPU），再用真实 shape 实测算子链。MoE 用**缩减 expert 数**（32 而非 256）——依赖形态由 top-k 路由决定、与 expert 总数无关，所以 32 个就能复现结构。

分析见 [`../archive/dsa_dependency_analysis.md`](../archive/dsa_dependency_analysis.md)。

---

## 已知边界

- `clc_probe` 需要 sm_100+ 与 CUDA ≥ 12.8；在 H100 上 `build.sh` 会跳过它并继续
- `--wait none` 的结果**不可信**（故意的），只有时间有意义
- Tier 5 的 1M 上下文可能 OOM；驱动脚本记录失败后继续，不中断整个 campaign
- `[H+]` 选项（派发前门控、集中式依赖表）无法在真机实现，只能用 occupancy 曲线反推估值
