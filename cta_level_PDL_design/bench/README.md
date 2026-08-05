# CTA 级 PDL 评估代码

配套文档：[`../EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md)（实验计划与判读规则）、[`../AGENTS.md`](../AGENTS.md) §4（有效性铁律）

整场实验由 [`../run_session.sh`](../run_session.sh) 无人值守执行，分析工具在 [`../tools/`](../tools/)，可在租用机上直接跑。所有驱动脚本都可**无人值守**运行、**断点续跑**、**失败不中断**。

---

## 目录

```
bench/
├── common/
│   ├── cta_trace.cuh      原语 1：CTA 级 %globaltimer + %smid 打点
│   ├── dep_pattern.cuh    原语 2a：参数化依赖模式（结构与依赖度独立可调）
│   ├── dep_wait.cuh       原语 2b：可互换的同步协议（B1 维度的全部选项）
│   └── bench_util.cuh     公共宿主端工具
├── cta_dep_pilot.cu       Tier 1 收益地图（Graph Floor + 资源保留的多波独立-stream bracket）
├── cta_dep_bench.cu       旧微基准，**trigger 语义已被否决**，仅保留供复审（见「harness 状态」）
├── tier23_protocol_encoding.cu  §7.1 协议 / §7.3 interval-bitmask-CSR 原生 harness
├── tier23_diamond.cu       §7.4 ordered/unordered CTA diamond
├── tier23_c1.cu            §7.5 cluster DSMEM / L2 locality 四版本 + wrong Ceiling
├── tier23_clc_scheduler.cu §7.6 CLC 持久化三策略 + wrong Ceiling
├── run_tier23_native.sh    Tier 2/3 独立正式矩阵与 profiler sidecar 驱动
├── tier0_facts.cu         Tier 0 基础事实（重叠层数 / resident-wait 容量 / fence 成本）
├── tier0_background.cu    Tier 0.3 productive background 吞吐定价（真实 resident wait）
├── clc_probe.cu           Tier 0.4 CLC try_cancel 特性（需 sm_100+）
├── build.sh               编译（默认 sm_103）
├── run_all.sh             无人值守驱动
├── llm/                   Tier 4 LLM 端到端
│   ├── bench_llm.py       单档诊断工具（不可作为正式三档 timing）
│   ├── tier4_driver.py    同一 worker cohort 的隔离三 variant lowering / 相邻 triplet 驱动
│   ├── pdl_evidence.py    worker PTX/cubin/active callable/Nsight graph-node 证据闭环
│   ├── tier4_finalize.py  raw、CI、correctness 与 artefact 的 fail-closed admission/finalizer
│   └── run_llm_sweep.sh   decode/prefill cohort strict runner
└── dsa/                   Tier 5 DSA 算子链
    ├── dsa_native.cu      native strict 四模式 CTA 依赖链
    ├── run_dsa_chain.sh   native smoke/formal、profiler 证据与统一准入
    ├── production_tier5.py production component 完整工作量 harness
    ├── run_production_tier5_fragments.sh  逐 row 封存的 exact-26 驱动
    ├── production_tier5_campaign.py      契约、row seal/finalize/check-final
    ├── dsa_chain.py       已拒绝的旧 PyTorch 路径，只保留审计
    └── SEMANTIC_AUDIT.md   旧路径的历史语义审计
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
python3 ../tools/validate_tier0_chain.py results --json results/tier0_chain_validation.json
python3 ../tools/validate_tier0_background.py results \
        --json results/tier0_background_validation.json
```

---

## harness 状态（先读这段再选 phase）

| phase | 驱动 | 状态 |
|---|---|---|
| `tier0` | `tier0_facts` / `tier0_background` / `clc_probe` | 分探针准入：§4.1 chain 当前唯一正式源是 v4 `semantics=3` raw pair、CI、epoch/digest correctness 与严格 trace；§4.3 headline 仍只取 v2 正式配对定价；旧 CLC/fence 点仍为 `PARTIAL`，不能因 phase 返回 0 就整体称有效 |
| Tier 0.2（sibling） | `cross_stream_PDL_survey/bench/pdl_bench` | 路径机制证据可用，但旧报告缺当前要求的 raw CI 与防陈旧值 correctness 契约，状态仍为 `PARTIAL` |
| `tier1p` | `cta_dep_pilot` | 有效但逐点 fail-closed：Floor 是 programmatic Graph edge；Impl/Ceiling 是资源保留的独立优先级 streams；多波必须由 `%globaltimer` 实测证明仍有 producer CTA 未启动且全部 producer 最终完成 |
| `tier1` `tier23` | `cta_dep_bench` | **已否决**：它在 PDL trigger 之前就发布全部 `done[]`，等待预先满足，计时不能作为 CTA 收益证据（[`../reports/rejected/fast_campaign.md`](../reports/rejected/fast_campaign.md)）。保留仅供复审，`run_all.sh` 会先打印 WARNING |
| `tier23n` | `run_tier23_native.sh` | 新的合格入口：只在完整 Tier 1 `gate=GO` 后运行 §7.1/§7.3–§7.6；每点独立 poison/full-edge validation、相邻交替 31 repeats、bootstrap CI、`%globaltimer` CSV 与 strict validator。Ceiling 用 device-only sentinel schedule 保证一条真实 RAW 边在无 wait 时读 poison，并把 proof-latch loads 与协议 `poll_loads` 分账。旧 `tier23` 名称仍只指 rejected harness |
| Tier 4 | `llm/run_llm_sweep.sh` | 合格 schema-v3 路径：本地模型、单 worker cohort、三份隔离 cache/lowering/callable/FULL graph、worker-side PTX+cubin 与 Nsight graph-node 证明齐全后才接纳 timing；当前正式快照见 `../results/tier4_schema_v3_formal_v1_20260805/` |
| Tier 5 | `dsa/run_dsa_chain.sh` + `dsa/run_production_tier5_fragments.sh` | 两条独立的 fail-closed 合同：native strict 四模式给出 synthetic work-complete dependency proxy；production 固定 exact-26 component 矩阵，只可接纳 workload timing，不得冒充 CTA bracket。旧 `dsa_chain.py` PyTorch 路径和所有已拒绝目录均不可复活 |

`./run_all.sh`（不带参数）**不会**跑 `tier1` / `tier23`——把决策点预算花在已知无效的数字上，正是这个驱动要避免的事。
Tier 2/3 正式运行使用独立结果目录，避免与旧 `SUMMARY` schema 混合：

```bash
RESULTS=results_20260805_b200_tier23_native \
GATE_JSON=results_20260805_b200_multiwave_v2/gate.json \
./run_tier23_native.sh --fresh

python3 ../tools/validate_tier23_native.py results_20260805_b200_tier23_native \
  --manifest results_20260805_b200_tier23_native/tier23_manifest.tsv \
  --json results_20260805_b200_tier23_native/tier23_validation.json \
  --csv results_20260805_b200_tier23_native/tier23_summary.csv

# 只可在上述 formal strict PASS 后运行；该 memcheck 不产生正式 timing。
RESULTS=results_20260805_b200_tier23_native ./run_tier23_sanitizer.sh
```

`run_all.sh tier23n` 只是上述独立入口的便捷转发；它不会复活或重命名旧 `tier23`。
NCU sidecar 若因宿主权限返回 `ERR_NVGPUCTRPERM`，原文与状态会被保留但不阻断 raw
`%globaltimer` 矩阵；软件 `poll_loads` 只表示 harness 执行的逻辑 acquire load 次数，不能写成
实测 L2 request。Nsight Systems sidecar 则用于证明 CUDA kernel/stream 执行存在，不替代计时。
`run_tier23_sanitizer.sh` 对四个 native binary 各跑一个最小 admissible
Compute Sanitizer memcheck，并把 tool/target failure 原样写入独立 status；这些记录不在 formal
manifest 中，不能追加为 repeat 或性能样本。

两个分析脚本不可互换：`analyze.py` 读 `SUMMARY`，`analyze_pilot.py` 读 `SAMPLE` / `SUMMARY_PILOT`。
Tier 0.1 还必须通过 `validate_tier0_chain.py`；它从 raw pair 重算 bootstrap CI，并将模型反解 depth 与 trace 实测同时 active grids 分开。此前 v3 Tier 0.1 已被 v4
`semantics=3` **superseded**，只能保留为历史快照；当前复核不得把 v3 与 v4 的 sample、CI、
digest 或 trace 拼在一起。Tier 1 / Tier 0.3 已发布 headline 仍只来自 v2，不能用 v4 回归
中的轻微 timing 漂移改写。

多波路径不会把 `P,C > SM` 本身当成证据。每次 sample 都记录 producer/consumer 的
`%globaltimer` entry/end；只有首个 consumer 启动时仍有 producer CTA 未启动、全部 producer
最终完成、Floor 确实在 producer tail 内提前启动、Ceiling 捕获到 poisoned/stale 输出，汇总
才会写 `launch_gate=trace_verified multiwave_overlap_proven=1`。任一条件失败都非零退出，旧
`cta_dep_bench` 仍不可用。

---

## 关键实现细节

### Tier 0.1 `semantics=3` 与 artifact 绑定

`tier0_facts` 的 CONFIG/SUMMARY 固定声明 `work=2,000,000`、`prologue=1,000,000`、
`tail=1,000,000` cycles，并以 `epoch_schedule=monotonic_all_invocations` 覆盖每次独立
validation、warmup 与 timed invocation。每条 validation 记录 epoch、所有 stage checkpoint
的 observed/expected digest，以及 final-output observed/expected digest；Python validator 从
epoch、stage 和 block 数独立重做递推与两类 digest，不能只相信 producer 自报的
`mismatches=0`。

每条 timed SAMPLE 也携带预期 epoch。最终 off/on CSV 每行携带相同 epoch，日志中的
`TRACE_TIER0_CHAIN path=` 必须精确解析到当前结果目录实际被打开的
`tier0_chain_trace.csv`；validator 再把 makespan 和 CTA/grid/edge 整数指标逐项回绑到最终
SAMPLE。模型反解的 chain depth 只从 paired speedup 计算，物理 peak 只从 CTA 半开区间
`[t_launch,t_end)` 扫描，两者没有互换入口。

### 断点 marker、`--fresh` 与 strict session 结果

`run_all.sh` 写出的 step/pilot `.done` 与 `.invalid` 不是空 marker。schema 2 同时绑定
FAST/formal、完整 argv、可执行文件 SHA-256，以及 GPU UUID/name/compute capability/driver；
任何一项变化、或遇到旧空 marker，都会重跑对应 step，并从当前 admitted logs 重建
`summary.txt`，避免 FAST 行残留到 formal 汇总。

`run_session.sh --fresh` 会分别清理 smoke 目录，并在正式目录进入 Tier 0 时清理一次；正式
Tier 1p 随后不再二次传 `--fresh`，所以刚生成的 Tier 0 marker、raw log 和 summary 不会被
再次清掉。session 会先移除陈旧 strict 输出，再生成
`tier0_chain_validation.json` 与 `tier0_background_validation.json`。任一 validator 失败时，
测量流程仍 fail-soft 地完成 Tier 1 gate 与 collect，但 session 最终固定返回 2，即使 Tier 1
机器 gate 本身是 `GO`；旧 PASS JSON 不得代替本轮失败。

### 计时必须用 `%globaltimer`，不能用 `clock64()`

CTA 时间线重建本质是**跨 SM 的事件排序**：

| 计时源 | 性质 | 跨 SM 可比 |
|---|---|---|
| `clock64()` | SM 本地时钟计数器 | **否** |
| `%globaltimer` | 全局纳秒计时器 | 是 |

用错会得到看似合理但实际错乱的重叠关系，且**不会有任何报错**。`cta_trace.cuh` 的打点用 `%globaltimer`；`bench_util.cuh` 里的 `spin_cycles()` 用 `clock64()`——因为那里测的是单 SM 上的持续时长，不涉及跨 SM 比较。

### Tier 0.3 productive-background 定价

`tier0_facts` 的容量点只回答“这种 CTA 最多能驻留多少个”；它没有并发的
有用工作，因此不能回答 resident waiting CTA 挤掉多少吞吐。新增的
`tier0_background` 在同一进程内相邻比较：

| 模式 | 执行内容 |
|---|---|
| `deferred_gate` | entry-trigger producer 后，在同一 dependency stream **普通顺序**启动 waiter；无 PSS 属性，因此 waiter 只能在 producer 结束后入场 |
| `resident_wait` | 完全相同的 producer/waiter/background；仅 waiter launch 增加 PSS 属性，使其可提前驻留并在 `cudaGridDependencySynchronize()` 中真实等待 |

producer 是单 CTA 的 dependency holder：它在 kernel 入口 trigger，之后才执行 readiness
work。这里要定价的是 dependent grid 占用的资源而不是 producer 饱和度；若 producer
先铺满全部 SM，64 KiB/high 档只能在 producer CTA 退休时入场，真实等待窗口会退化成
不可复现的 grid-tail 抖动。两个模式使用同一个 waiter kernel、相同 grid、寄存器档、dynamic
smem、输出计算与全量校验，host 端也都走 `cudaLaunchKernelEx`；唯一的 launch 语义差异是
PSS 属性。逐 CTA `%globaltimer` 记录必须证明 control 的每个
`wait_enter >= max(producer_end)`，而 resident 至少有一个
`wait_enter < max(producer_end)` 且所有 early waiter 都有
`wait_exit >= max(producer_end)`。因此 `deferred_gate` 保留了依赖满足后的 consumer 工作，
差值不再把整个 consumer grid 删除。汇总报告 resident 的 `early_waiters` / `peak_waiters`
及其 bootstrap CI。

background 有效吞吐的 anchor 是 producer/background 两者最早开始活动的时刻，算到
background 完成，包含 waiting CTA 引起的 dispatch delay；另报两种模式各自的 active
kernel window、background peak CTA 及 CI。复合端到端时间在两边都包含 waiter end，且
`e2e_delta_ms` 是逐 pair 的 `resident_wait - deferred_gate` 后再 bootstrap，而不是两个独立
中位数相减。

正式 `run_all.sh tier0` 扫 `0/8/16/32/64 KiB × low/mid/high`，每点 3 次 warmup +
31 组配对重复和 deterministic bootstrap 95% CI。FAST 只跑 `0/64 KiB × low/high`，
并显式传 `--allow-short`，短跑输出不能冒充正式结果。寄存器档由 live-across-wait
编译期数组与 `__launch_bounds__` 构造，但档名不是证据；每条 `RESOURCE` / `SUMMARY`
都自报 `actual_num_regs`、`local_bytes`、dynamic smem 与 occupancy。64 KiB 在 query/launch
前显式 opt-in，`smem=0` 路径不访问 dynamic smem。

每次重复的两个模式都会 poison 并逐元素校验 producer、waiter 和全部 background 输出；
任何错误、control waiter 提前入场、resident 没有真实 early waiter、或 wait 在 producer
结束前返回，都会使进程非零退出，driver 不会写 `.done`。`--trace PATH` 把最后一组 pair
的两种 mode 写进同一个逐 CTA 原始时间戳文件，可直接复核两侧 launch 语义。

```bash
# 单点正式测量；少于 31 repeats 会被拒绝
./tier0_background --smem-kb 32 --reg-tier mid --repeats 31 \
  --trace results/tier0_bg_mid_smem32_trace.csv

# 极小 plumbing smoke，不能用于结论
./tier0_background --smem-kb 0 --reg-tier low --repeats 1 --warmup 0 \
  --bg-waves 1 --bg-iters 1024 --producer-cycles 500000 --allow-short
```

### 依赖度与结构复杂度必须独立扫描

`dep_pattern.cuh` 把两者做成正交的轴：

```bash
# 结构钉死为连续区间，只扫依赖度 → 收益变化归因于"边更多"
./cta_dep_pilot --producers 148 --consumers 148 --structure interval --degree 8

# 依赖度钉死为 32，只扫结构 → 收益变化归因于"形状更难"
./cta_dep_pilot --producers 148 --consumers 148 --structure strided --degree 32
```

`run_all.sh tier1p` 会把这两条轴各自扫完，并按设备 SM 数带上多波 grids
（`2×/8×/32× SM`）。Floor 保留 production grid-PDL 语义；软件/Ceiling 点绕开 grid trigger
门槛并用 trace proof 证明跨逻辑波重叠，不能把 grid 比例本身当证明。pilot
接受 `interval` / `grouped` / `strided` / `random` / `self`
五种结构（`all` / `none` 被显式拒绝）。`random` 用每个 child 独立的模置换生成父节点，
保证实际依赖度等于请求值；正式 degree 轴为 `1→1024` 对数步进。`self` 按定义恒为
degree 1，是结构轴的语义端点而非伪装成 degree 32。结构轴的 `interval,d=32` 与 degree 轴
同一物理配置，只测一次并由分析器复用，避免 gate 对它双重加权。可用 `PILOT_GRIDS` /
`PILOT_SMS` 覆盖默认扫参。

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

五档使用同一 producer/consumer kernel、相同 CTA 数、线程数、spin 工作量和 consumer
资源；consumer 都显式 opt-in 并 touch 64 KiB dynamic shared memory。这个资源包络在启动前
核对 threads/registers/shared-memory/block slots，确保按资源算术每 SM 仍容得下一个 producer
CTA；producer 走高优先级 stream，consumer 走低优先级 stream。CUDA 不承诺两个独立 kernel
的公平性，因此这项算术只是必要条件，真正的前进性证据仍是每次 sample 的完整时间线和
10 秒 watchdog。纯 CUDA Graph 的两个 independent roots 在本机不能作为替代：producer-first
会串行，consumer-first 会触发 watchdog。

Floor 单独使用同一进程中的 programmatic CUDA Graph edge，producer 在数据 ready 后 trigger，
consumer 用 `cudaGridDependencySynchronize()`；三种 Impl 和 Ceiling 使用无依赖的独立 streams。
headline `ms` 由两条流全部结束后的 `%globaltimer` makespan 得到，不把单条流的 event 当总时间。
每个 rep 先相邻执行 Floor/主 Impl/Ceiling，再执行两个协议控制；奇偶 rep 正反轮换以平衡顺序
偏差。正式运行 31 次，全部 timed invocation 都重新 poison；O(degree) 全边校验仅在独立、
不计时的 invocation 中执行。

reset 的全部 poison/清零在任何 timing event 之前完成，并以 device barrier 与三个
non-blocking launch stream 建立顺序。若 event/stream 已正常完成但六组必需时间戳仍有零槽，
该次只输出 `REJECTED_ATTEMPT`（含 producer `start/ready/end`、consumer `start/dep/end`
的缺失数，以及两侧 `start≤middle≤end` 乱序数），不输出 `SAMPLE`
也不进入统计；同一 rep/mode 最多内部重试 3 次。timeout、CUDA error、校验失败、缺少 overlap、
Ceiling 未读到 stale 值、性能离群点都不允许借此重试。最终每个 mode 仍须有恰好 31 个完整
`SAMPLE`；重试耗尽直接非零退出。`SUMMARY_PILOT` 记录实际 `trace_retries`、
`trace_retry_limit=3`、`trace_max_attempts=4` 和本次观测到的最大 attempt。

`none` 档会跳过“结果必须正确”的校验并在表格中标 `n/a`，这是故意的——它靠读未写入的
数据提供实测的 unsafe no-wait 操作参考，不是数学或架构时间下界；但 harness 反而要求至少捕获一个 stale/poison 输出，防止
所谓 Ceiling 实际仍被串行化。
单调完成计数器协议**已被删除**：cardinality 不蕴含 identity，`completed_count >= hi+1` 并不表示
`[0,hi]` 这些 parent 都完成了，那不是保守的过度等待而是错误。

旧的 `cta_dep_bench` 用 `--all-waits` / `--wait` 表达同一组点，其中还包含已被证伪的 `cta-counter`。

---

## LLM 部分（Tier 4）

```bash
cd llm
MODEL=/absolute/path/to/a/fully-staged-model \
RESULTS=/absolute/path/to/a/new-results-directory \
./run_llm_sweep.sh
```

旧驱动每一档启动一个新进程，不能满足实验计划 §3.7 的“同一进程、相邻时段完成三档”；
父进程里把 Triton `gdc_wait` monkeypatch 成 no-op，也不能证明 vLLM worker 实际编译、执行的
kernel 已经删掉 wait。该路径继续只作 rejected 历史，不能与 schema-v3 拼接。

当前 schema-v3 driver 已实现 target-specific 同 cohort 三档，并为每档保留独立 cache、
lowering、compiled callable、FULL CUDA graph 与 worker active-variant 记录：`pdl_off` 的目标
PTX 中 wait/launch 都为 0；`pdl_grid` 中二者都大于 0；`ceiling` 中 launch 大于 0 而 wait
为 0；三档都有配对 cubin，并由 Nsight Systems `--cuda-graph-trace=node` 把目标 entry 映射
到实际 graph-node 执行。`llm/pdl_evidence.py` 与 `tier4_finalize.py` 对这份证据 fail-closed，
缺一项仍返回 3，且不得接纳 timing。

正式快照 `../results/tier4_schema_v3_formal_v1_20260805/` 使用本地
Qwen3.6-27B，decode 与 prefill cohort 的 `--verify-admission` 均为 `status=ok`。正式 headline
只包括 `classification=headline_full_decode` 的四个 decode 点；长上下文 prefill+2-token 行
固定为 `production_mixed_mode_non_headline`，不能升级为 prefill PDL 结论。

`bench_llm.py` 只保留为显式 `--diagnostic-only` 的单档原始数据工具：已改用 vLLM 0.23 的
`prompts=` API，要求至少 31 次独立重复，输出 bootstrap 95% CI，并通过 RPC 检查 worker
配置；它只写 `status=diagnostic`，不能被正式 finalizer 接受。正式路径要求完整三档、同一
driver/worker/triplet、CI、off/grid 完整输出一致、Ceiling 错误 sentinel 及上述 PTX/graph
证明，否则返回 3。

**Floor 仍是 `PDL_grid` 而不是 `PDL_off`**。2–33% 只作为 PDL_off→PDL_grid 的诊断带宽；超出会告警，但落入该区间本身绝不构成 PDL 已生效的证明。

---

## DSA 部分（Tier 5）

Tier 5 有两条不可混合的当前入口。native strict 表达 CTA 依赖机制包络；
production fragment 表达真实 API 上的 component workload timing。旧 `dsa_chain.py`
PyTorch 三档路径只供历史审计，它的 oracle、rc 或 timing 不是这两条入口的准入条件。

### Native strict 合同

```bash
cd dsa

# 4K、3 repeats 语义冒烟；不是 formal timing。
RESULTS=results_native_smoke_strict_v8 FAST=1 PROFILE=0 ./run_dsa_chain.sh

# 固定 4K/32K/128K/1M、31 repeats、profiler 证据闭环。
RESULTS=results_native_formal_strict_v8 FAST=0 PROFILE=1 ./run_dsa_chain.sh
```

native 在同一进程内相邻、循环 Latin order 执行 `floor,wave_floor,impl,ceiling`
四模式：`floor` 是单次 full-grid programmatic CUDA Graph，`wave_floor` 是工作量等价的
有界分波 grid-PDL，`impl` 使用 per-producer epoch acquire，`ceiling` 不等待也不发布，
必须由 poison/stale 证明其结果确实错误。分波路径用资源有界的 query wave、system-scope
entry gate、进度轨迹和 watchdog 防止 resident consumer 堵死 producer。

4K/32K 保留 exact tile-to-CTA mapping；128K/1M 执行完整 pair/history 工作，但必须标为
`work_complete_packed_proxy`。两类证据不得混写。正式准入还要求：

- 编译产物与源码 hash 闭环，PTX/SASS 静态证据与实际 target arch 一致；
- Nsight Systems 4K sidecar 把所有必需 NVTX range 映射到实际 kernel，sidecar 不是 timing sample；
- NCU 只做单次硬件计数器权限探测。计数器可用或明确记录
  `ERR_NVGPUCTRPERM` 都是可表达的环境边界；模糊状态、缺少 NCU 或缺少 Nsight 证据会 fail-closed。

### Production fragment 合同

```bash
cd dsa

# 单短 row 管线冒烟，accepted_workload_timing 保持 0。
EXECUTE_GPU=1 TIER5_PRODUCTION_GPU_ALLOWED=1 FAST=1 \
  RESULTS=results_production_smoke ./run_production_tier5_fragments.sh

# 一小时预算的独立 compact-14 scoped formal；不是 exact-26。
EXECUTE_GPU=1 TIER5_PRODUCTION_GPU_ALLOWED=1 FAST=1 \
  MODELS=deepseek_v32,glm5 SEQS=4096,131072 \
  WORKLOADS=operator_chain,single_layer,indexshare_fsss \
  WARMUP=5 REPEATS=31 MOE_TOKENS=4096 \
  RESULTS=results_production_compact ./run_production_tier5_fragments.sh
python3 validate_production_tier5_compact.py results_production_compact

# 原计划完整范围：固定有序 exact-26 canonical rows。
EXECUTE_GPU=1 TIER5_PRODUCTION_GPU_ALLOWED=1 FAST=0 \
  RESULTS=results_production_formal ./run_production_tier5_fragments.sh
```

exact-26 contract 固定两个 model、4K/32K/128K/1M context 与三类 workload 展开得到的
26 个 canonical row，并固定 31 repeats、seed、MoE shape、`max_logits_mb=16384` 和
`max_query_chunk=4096`。正式启动前必须在 B200 上用 1M 最坏形状单独验证 16 GiB
logits 上限的峰值可承载性；streaming 只限制驻留 logits，每个 query row、
`S*(S+1)/2` 个 causal pair 与全部需验证元素仍必须执行，不得抽样。每个 operator-chain
chunk 都通过 mode-local correctness seal 记录完整 validation scope。

每个 row 在独立 GPU lease 下写入新的 `.inprogress` stage，只有通过语义验证才会
no-clobber publish 并 seal；失败或中断的 stage 原子转入 `failed_segments/`。中断处理会校验
PID/PGID 身份，对子进程组做有界 TERM/CONT/KILL 并 reap monitor，不允许遗留 GPU orphan。
已 seal row 只能在 hash 和契约重验证通过后 resume-skip。最后 `finalize` 重组合同矩阵，
`check-final` 再从磁盘独立复验每个 row、aggregate 和 completion marker。

只有 exact-26 全量通过时 `accepted_workload_timing=1`。用户为一小时上限授权的
compact-14 仍使用 5 warmups、31 repeats、相邻配对和全量 correctness；只有精确
14/1302/62 row/sample/summary 且 fresh `check-final` 通过时，独立 validator 才可置
`accepted_compact_workload_timing=1`。它必须保持旧 `accepted_workload_timing=0` 与
`accepted_exact26_workload_timing=0`，并把 32K/1M timing 标为排除。production API 不暴露 CTA readiness
或 unordered Ceiling，所以无论 component off/on timing 如何，`accepted_CTA_bracket=0`、
`headroom_defined=false`、`headroom_pct=null`；不得从 production component 差值推导 CTA headroom。

### 新目录与拒绝不可变规则

每次语义、schema、validator 或资源合同变更都必须用全新的版本化结果目录（例如
`*_strict_v8`），smoke 与 formal 也必须分目录。native 目录一旦出现
`formal_rejection.json`/`REJECTED.md`，或 production fragment 进入 `failed_segments/`，
该证据就永久只读；`--fresh`、resume、重签 seal 或复制 smoke 产物都不能把它恢复为正式样本。
修复后要换新目录从契约冻结、GPU 身份绑定和完整矩阵重新开始。

[`dsa/SEMANTIC_AUDIT.md`](dsa/SEMANTIC_AUDIT.md) 只解释为什么旧 PyTorch 路径永久无效；
它不是 native strict 或 production fragment 的当前运行说明。

---

## 已知边界

- Tier 1 多波用 CUDA Graph programmatic edge 表达 Floor，用独立高/低优先级流表达
  Impl/Ceiling；64 KiB consumer 动态共享内存为 producer 留出混合驻留资源。每个样本仍须由
  `%globaltimer` 证明 consumer 启动时存在尚未启动的 producer，且最终 producer 全部完成。
  CUDA 不承诺跨流公平调度，因此资源包络和 `STEP_TIMEOUT` 只能辅助止损；轨迹证据缺失时
  harness 会 fail-closed，而不会把该样本计作有效多波结果
- `clc_probe` 需要 sm_100+ 与 CUDA ≥ 12.8；在 H100 上 `build.sh` 会跳过它并继续
- `--wait none` 的结果**不可信**（故意的），只有时间有意义
- Tier 5 旧 `dsa_chain.py` 显式中间量路径仍永久拒绝；当前 native 长上下文只能按
  `work_complete_packed_proxy` 解释，production 只能按 component workload timing 解释，二者都不得扩张为整模型或纯 CTA headroom
- `[H+]` 选项（派发前门控、集中式依赖表）无法在真机实现，只能用 resident-wait 容量曲线反推估值
