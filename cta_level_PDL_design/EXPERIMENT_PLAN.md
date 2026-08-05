# CTA 级 PDL 实验计划

> **这份文件是实验的规格，不是脚本的说明书。**
>
> 它规定**要测什么、参数空间多大、一次测量在什么条件下才算数、结果怎么判读、以及哪些结论不能下**。驱动脚本（[`run_session.sh`](run_session.sh)、[`bench/run_all.sh`](bench/run_all.sh)）是本计划的**实现**。
>
> **两者不一致时，是脚本要改。** 脚本今天做不到的事情，不构成把它从计划里删掉的理由——那是 §13 的待办，不是本计划的边界。
>
> **有效性铁律见 [`AGENTS.md`](AGENTS.md) §4，与本文冲突时以 AGENTS.md 为准。** 能跑出一个数字不等于这个数字可采信。
>
> 姊妹文档：设计空间（有哪些选项可测）见 [`docs/cta_pdl_design_space.md`](docs/cta_pdl_design_space.md)；**代码当前实现到什么程度**见 [`bench/README.md`](bench/README.md)；**相对本计划的执行进度与续跑入口**见 [`EXPERIMENT_REPORT_INDEX.md`](EXPERIMENT_REPORT_INDEX.md)。本计划不记录实现进度与执行进度，那是那两份文件的职责。

---

## 0. 这份计划怎么用

**执行时**：先读 [`EXPERIMENT_REPORT_INDEX.md`](EXPERIMENT_REPORT_INDEX.md) §0–§1，确认哪些已 `DONE` / `BLOCKED`，再开跑。整场实验应当由单一入口无人值守跑完，人不介入。当前入口是 `./run_session.sh`（对驱动的要求见 §13）。

```bash
git clone <repo> && cd <repo>/cta_level_PDL_design
./run_session.sh
```

顺序为 preflight → 冒烟 → §4 Tier 0 → §5 Tier 1 → §6 gate → 按判决分支 → 收集。**在 §6 判决之前不要动 §8 / §9**：它们需要 54GB 模型下载与 vLLM，是整场里最容易卡住的一段，在 gate 说方向还活着之前不值得为它订机器。

**判读时**：每个实验都写明了**产出什么、怎么判读、以及不能由它推出什么**。第三项是强制的——AGENTS.md §7 要求每份报告都有「不成立的主张」一节，本计划各节的「不能下的结论」就是那一节的素材。

**写脚本时**：§3 是所有测量的准入条件，§13 是对驱动脚本的要求。新增或修改 harness 前先读这两节；违反 §3 的 harness 产出的数据一律作废，不论它跑得多顺。

必须落盘的产出，任何实现都要提供：

| 产出 | 内容 |
|---|---|
| `gate.json` | §6 判决：`GO` / `LLM_ONLY` / `STOP` / `INVALID`，及其背后的统计量 |
| `pilot_summary.csv` | 逐配置统计量，含置信区间 |
| `pilot_matrix.log` | Tier 1 原始逐次采样记录 |
| `tier0_facts.log` | Tier 0 原始记录 |
| `session.log` | 全程日志，时间戳从 session 起点计 |
| `failures.log` | 失败的步骤（失败不中断整场，见 §13） |
| `device.txt` | **实际跑在哪张卡上**——报告里必须写这个，不是写计划里假定的那张 |

---

## 1. 测量模型：四点包夹

**核心难题**：设计空间里绝大多数 CTA 级机制**在今天的硬件上并不存在**，无法直接实现测量。

**解法**：把「测机制」换成「**包夹机制**」——不去实现它，而是用真机可构造的执行模式把它的收益夹在一个区间里。

任何一个设计选项，都测同一组四个点：

| 点 | 含义 | 怎么构造 |
|---|---|---|
| **Floor** | 现状基线 | grid 级 `griddepcontrol.wait`；真实负载上是 **PDL 已开启**的生产配置（见 §8.3） |
| **Impl** | 该选项的软件实现 | 仅 `[S]` 选项有此点 |
| **Ceiling** | 依赖零成本 | 把依赖完全去掉、两 kernel 无约束并发（**结果是错的，只测时间**） |
| **Ideal** | 有硬件支持时的上限 | 在 Ceiling 基础上注入该机制的固有开销（依赖检查延迟、唤醒延迟等，由 §2 原语 3 标定） |

### 判读规则

```
Ceiling − Floor  =  该负载的总收益空间     ← 间隙小就直接放弃，不必往下做
Impl    − Floor  =  今天纯软件能拿到的
Ideal   − Impl   =  硬件改动的价值          ← 软件开销中可被硬件消除的部分
```

无法软件实现的 `[H-]` 选项只有 Floor / Ceiling / Ideal 三点，给出**区间估计**而非精确值——筛选阶段够用。

### 为什么每个负载都先看 `Ceiling − Floor`

这个差值几乎零成本就能测（两次 kernel launch 配置的差别），却能一票否决一个负载。**任何负载的实验都从这一步开始**，不要在没有收益空间的负载上花租用时间。

---

## 2. 三个测量原语

### 原语 1：CTA 级时间戳打点

每个 CTA 由 thread 0 写一条记录到 global memory：`(kernel_id, blockIdx, smid, t_launch, t_dep_satisfied, t_end)`。

**必须用 `%globaltimer`，不能用 `clock64()`：**

| 计时源 | 性质 | 能否跨 SM 比较 |
|---|---|---|
| `clock64()` | **SM 本地**时钟计数器 | **不能** |
| `%globaltimer` | **全局纳秒**计时器 | 能 |

CTA 时间线重建本质是**跨 SM 的事件排序**。用 `clock64()` 会得到看似合理但实际错乱的重叠关系——**这是静默错误，不会有任何报错**。

```cuda
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}
__device__ __forceinline__ unsigned int smid() {
    unsigned int s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}
```

开销：每 CTA 32 字节，10K CTA 共 320KB，可忽略；只由 thread 0 写，非原子。

支撑的指标：依赖停顿分布（`t_dep_satisfied − t_launch`，归一化到 CTA 执行时长）、CTA 并发度随时间曲线、生产者-消费者的 SM 亲和性、两 grid 的实际重叠比例。分析工具是 [`tools/cta_timeline.py`](tools/cta_timeline.py)。

### 原语 2：参数化微基准骨架

#### 依赖模式（`--structure`）——消费者 CTA *j* 依赖生产者的**哪些** CTA

这是**二分图的形状**，与依赖度（边的**条数**）正交，两者必须独立扫描（理由见下）。实现是 [`bench/common/dep_pattern.cuh`](bench/common/dep_pattern.cuh) 的 `dep_parent(pattern, child, k)`，下表的值就是命令行实参：

| `--structure` | child *j* 的 parent 集合 | 区间 `[lo,hi]` 编码 | 真实负载对应物 |
|---|---|---|---|
| `self` | `{j}`，依赖度恒为 1 | 精确 | DeltaNet chunk 间递归这类 1-to-1 长链 |
| `interval` | 一个**连续窗口**，随 *j* 滑动 | **精确，O(1)** | LLM FFN GEMM 链、DSA indexer→topk |
| `grouped` | 消费者分桶，每桶依赖对应的一桶生产者 | 桶内精确 | **BlockMaestro 的注入形状** |
| `strided` | 同样 d 个 parent，但**均匀散布全 grid** | **严重失真** | 对照组：区间编码失效的情形 |
| `random` | 伪随机 parent，需完整邻接表 | 失真最严重 | 最坏情况上界 |
| `all` / `none` | 全连接 / 无依赖 | 退化 | 等价于 grid 级 barrier / 并发上界 |

`cta_dep_pilot` 接受 `interval|grouped|strided|random|self`；`random` 必须保证每个
child 的 parent 唯一，使请求 degree 等于实际 degree。`all`/`none` 会被拒绝。
`cta_dep_bench` 七种都收，但它是被否决的 harness（§3）。

#### 为什么必须与依赖度独立扫描

固定 P=C=148、依赖度=32，看 consumer CTA #74：

| `--structure` | parent 集合 | 区间覆盖 | **区间编码下实际等待的 CTA 数** | tightness |
|---|---|---|---|---|
| `interval` | 58..89 | `[58,89]` | 32 | 1.00 |
| `grouped` | 64..95 | `[64,95]` | 32 | 1.00 |
| `strided` | {2, 6, 10, …, 146} | `[2,146]` | **145** | 0.22 |

**边数完全相同，等待范围差 4.5 倍。** `strided` 的 113 条是假边，收益退化到接近 grid 级 barrier。所以「依赖度 32」这个数字本身预测不了收益——BlockMaestro 的「依赖度 > 32 收益归零」之所以不可照搬，正是因为它用 `grouped` 注入，加大依赖度的同时结构也在变复杂，两个变量锁死在一起。

基准会直接输出 `tightness`（= 依赖度 / 区间宽度）与 `eff_degree`（区间编码下实际等待的父节点数），**判读时先看这两个量，再看时间**。

#### 其余可调维度

**记号**：全文用 **P** 表示生产者 kernel 的 CTA 数（`--producers`）、**C** 表示消费者 kernel 的 CTA 数（`--consumers`）、**SM** 表示设备的 SM 数（B200/B300 = 148）。

- **依赖度** 1→1024（`--degree`）
- **grid 规模** 64→8192 CTA（相对 SM：欠填充 / `=SM` / `2×·8×·32×SM` 多波候选，见 §5.3）。`cta_dep_pilot` 另有：
  - **每 SM 可容纳 CTA 数 ≥ 2** —— 由 shared memory 与寄存器用量决定。生产者与消费者必须能在同一 SM 上共存，否则消费者只能等生产者退休，重叠恒为零
  - `P,C` 与 SM 的比值只划分计划 grid-ratio 区间；真多波要由 `%globaltimer` trace 证明后续生产者 CTA 确实延后进入，不能从 ratio 推导（§5.3）
  - 接受 parent 唯一的 `random`；拒绝 `all` / `none` 结构
- **每 CTA 计算量**，单位 spin cycles，四段各自独立：
  - `--ready` 生产者写出数据**之前**的工作（`--skew-bins` 可让各 CTA 的就绪时刻错开）
  - `--tail` 生产者发布数据**之后**、与该数据无关的工作。**收益全部来自这一段**：CTA 级消费者可以在生产者还在跑 tail 时就开始做依赖工作，`griddepcontrol.wait` 必须等整个 grid 退休。tail 为零则无论依赖结构多理想都没有可重叠的窗口——这正是 §5.2 扫描 tail/prologue 比的原因
  - `--prologue` 消费者**等待之前**的工作 / `--epilogue` 消费者等待之后的主体工作
- **每 CTA shared memory** 0/8/16/32/64 KB；**每 CTA 寄存器数** 经 `__launch_bounds__` 控制
- **生产者写入量 / 消费者读取量** 1KB→64KB/CTA

### 原语 3：开销隔离微基准

单独标定各机制自身成本，供 **Ideal 点注入**使用：各 fence scope（`.cta`/`.gpu`/`.sys`）延迟、原子写完成位图成本、自旋轮询的 L2 访存量及其对并发 background kernel 的干扰、`griddepcontrol.wait` 自身延迟、CLC `try_cancel` 延迟与仲裁吞吐。

---

## 3. 准入条件：一次测量在什么条件下才算数

**这是对 harness 的硬性要求，不是建议。** 违反其中任何一条，产出的计时一律作废——不论它跑得多顺、输出格式多正常。写新 harness 或改旧 harness 之前先过一遍这些条件。

### 3.1 必须区分 launch eligibility trigger 与真实 readiness publication

launch-eligibility trigger 只决定消费者 grid 何时可以被派发；readiness publication 才表示数据已经真正可读。两者在不同包夹点的关系是：

| 点 | launch eligibility | readiness publication / wait |
|---|---|---|
| **Floor** | 生产者数据 ready 后才 trigger，即 `trigger_floor=ready` | 消费者走 grid 级 `griddepcontrol.wait` |
| **Impl** | 生产者入口就 trigger，即 `trigger_impl=entry` | 数据写出并完成所需内存顺序后才 release-store CTA ready flag；消费者等这些 flag |
| **Ceiling** | 生产者入口就 trigger，即 `trigger_ceiling=entry` | **不发布、也不等待 readiness**；故意产生错误结果，只取时间 |

因此，Impl 的 entry trigger 可以早于数据 ready，但它的 release flag **不得**早于数据 ready。否则消费者的等待在实际执行时已经预满足，测到的是「无依赖的并发」，而不是 CTA 级依赖收益。

**这条要单列，是因为它已经真实发生过一次。** 被否决的 `cta_dep_bench` 在 PDL trigger 之前已经发布全部 `done[]`，使消费者获得 launch eligibility 时等待已经预满足；它跑得动、不报错、输出格式完全正常，一整轮 campaign 的数据全部作废。审计记录见 [`reports/rejected/fast_campaign.md`](reports/rejected/fast_campaign.md)。

**这类错误无法靠跑一遍发现，只能靠设计审查与语义 trace 核对。** 因此任何 harness 都必须在输出里自报三个触发点（`trigger_floor=` / `trigger_impl=` / `trigger_ceiling=`），并证明 Impl 的 release flag 在对应数据 ready 之后、Ceiling 没有 readiness 等待，让审阅者不只依赖 `.cu` 中的意图注释。

### 3.2 Ceiling 必须真的去掉依赖，并承认结果是错的

Ceiling 点的语义是「依赖零成本」，实现方式是把依赖整个拿掉。**它算出来的结果必然是错的，只取时间。** 任何声称 Ceiling 结果正确的实现，说明它没真正去掉依赖，该点无效。

### 3.3 Floor 必须是现状基线，不是「什么都不做」

微基准上 Floor 是 grid 级 `griddepcontrol.wait`；真实负载上 Floor 是**生产框架已开启 PDL 的配置**（§8.3）。拿 PDL-off 当 Floor 会把 grid 级 PDL 已经拿走的收益重复计入 CTA 级的账上。

### 3.4 每个配置必须带正确性校验，且校验不进计时样本

校验要覆盖该配置的**全部真实依赖边**（不是抽样），确认每个消费者读到的都是对应生产者写出的值。校验本身是独立一轮，不能给计时样本增加 O(degree) 的额外工作。

任一配置校验失败 → 整轮判 `INVALID`，**该轮所有计时都不可用**（§6）。

校验记录还必须与本次 invocation 绑定，不能只自报 `mismatches=0`：至少保留每次唯一的
poison/epoch 身份与可独立重算的输出摘要，并让 validator 核对 validation 与 timed sample 的
执行顺序、固定 workload 参数和最终 trace。畸形字段、越界 trace 或旧 validation 行拼接都必须
得到结构化 `FAIL`，不得因 traceback 留下一份旧 `PASS` 文件。

### 3.5 必须自报覆盖边界

每条汇总记录都要带上：P、C、SM 数、每 SM 常驻 CTA 数、依赖度、结构、`tightness`（依赖度 / 区间宽度）、`eff_degree`（区间编码下实际等待的父节点数）、重复次数。

**缺这些就无法判断结论能外推到哪。** 尤其是 P、C 与 SM 的关系决定计划 grid-ratio 区间，但是否真多波还必须看 timestamp proof（§5.3）；这两项是判读覆盖边界时最先要看的东西。

### 3.6 依赖度与结构必须能独立设定

两者是正交的输入维度（§2），任何把它们绑在一起的注入方式都无法回答 §5.1 要回答的问题。

### 3.7 统计要求

每个配置至少 31 次重复，报告中位数与置信区间，不报单次值。丢弃前若干次预热。同一配置的 Floor / Impl / Ceiling 必须在**同一次进程内、相邻时间**测出，避免跨进程的时钟与频率漂移混进差值。

唯一可重试的计时尝试是**必要的 trace timestamp 未完整落盘**：在配置、repeat 与 mode 均不变的前提下，最多显式重试 3 次。每个被拒绝的尝试必须保留 `REJECTED_ATTEMPT`，写明 tag、repeat、mode、attempt 和缺失的 timestamp 计数；它**不得**输出 `SAMPLE`，也不得进入中位数或置信区间。重试耗尽后仍缺必要 timestamp，属于必要语义 trace proof 缺失，按 §6 判 `INVALID`。

`timeout`、CUDA 错误、correctness 失败、overlap 语义失败、`Ceiling-stale` 语义失败与性能离群点**都不得重试**。完整 trace 显示语义条件没成立，与 timestamp 没写全是两类事；前者不能藉重试挑到一次有利调度。性能离群样本也必须照常保留在 `SAMPLE` 与统计里。

---

## 4. Tier 0 — 基础事实

**第一优先。不测清楚这些，后面所有实验都可能被误读。** 预算约 1 小时。当前入口 `cd bench && ./run_all.sh tier0`。

### 4.1 · 同 stream 内实际能重叠几层 kernel — B3 可达性

产出 `bench/results/tier0_facts.log` 中 `tier0=chain` 行。构造 1→6 stage 的
K1→K2→…→K6 链，测 overlap 沿链传播的收益，并从逐 CTA trace 直接计算同时活跃 grid/CTA
峰值。扩到六级是对原四级扫描的覆盖扩展，用于把「流水传播层数」与「瞬时并发层数」分开；
模型反解 depth 不得代替 trace 计数。

**判读**：最大有效窗口深度。这个数字直接决定 B3 维度上哪些选项在本设备上**可达**——如果硬件实际只能重叠 2 层，设计空间里假设深流水的选项就没有意义。

**不能下的结论**：这是本设备的实测值，不是架构保证。换代号必须重测。

### 4.2 · 跨 stream 与 CUDA Graph 行为复验

本仓库既有结论来自 H100 实测：**eager 跨流拿不到收益，必须走 CUDA Graph 捕获**。在 Blackwell 上必须复验。

复用兄弟目录的 `cross_stream_PDL_survey/bench/pdl_bench`，产出 `bench/results/tier0_xstream.log`。**这一步必须由驱动自动执行**，不允许要求人手动切目录。

**判读**：预期 eager 跨 stream ≈ 1.00x、captured graph ≈ 2.00x。**若这个关系变了，本身就是一条关于 Blackwell programmatic events 的发现**，应单独成报告，而不是当成噪声略过。

### 4.3 · 等待中 CTA 的 occupancy 代价曲线 — B2 唯一定价依据

扫描等待者 kernel 的 shared memory（0/8/16/32/64KB）与寄存器数。每个资源点首先保留容量曲线：自报实际寄存器数、spill / local memory、dynamic shared memory、每 SM 可常驻等待 CTA 数与总等待 CTA 数。

在同一资源点上再做 productive-background 配对对照：

- `deferred_gate` 对照与 `resident_wait` 实验必须使用**完全相同的等待者 kernel、grid、资源参数、输入毒化与全输出校验**；对照不得省掉等待者。唯一差别是对照使用普通同 stream 派发、使等待者在生产者退休后才进入；实验对同一等待者启用 PSS launch path，允许它在 ready 前常驻并执行同一个 wait。
- 两档都与同一个有用计算的 background kernel 并发；不能用空 spin 代替 productive work。配对样本在同一进程内相邻测量，且交替两档顺序。
- 产出 `tier0=background` 逐次样本与汇总：延后 / 提前进入的等待者数、峰值常驻等待者、background 活跃时间、峰值并发 CTA、有用工作吞吐、等待时间、端到端时间及两档配对差值；中位数与置信区间均必须保留。

**判读**：资源扫描给出「等待 CTA 能占多少槽位」的容量曲线；productive-background 配对差值给出「这些常驻等待者实际挤掉多少有用工作」。两者合在一起才是 B2 维度的定价依据。§7.2 只能在明示的工程单调性假设下，用该差值构造「若等待不占用槽位能省多少」的条件化 scenario envelope；那个量在今天的硬件上无法直接测。

### 4.4 · CLC `try_cancel` 实测特性

产出 `bench/results/tier0_clc.log`。测延迟、并发仲裁吞吐、失败率。

**前置**：需 sm_100+ 且 CUDA ≥ 12.8。不满足时 `clc_probe` 不会被构建，该步骤自动跳过并在日志中说明——**这是预期行为，不是失败**。

**判读**：决定 B4 的软件复现路径与持久化消费者方案是否可行。

### 4.5 · 各 fence scope 成本标定

产出 `tier0=fence` 行。`.cta` / `.gpu` / `.sys` 三种 scope 的延迟表，供 §1 的 Ideal 点注入使用。

---

## 5. Tier 1 — 收益空间地图（决策点）

**可能推翻整个方向，尽早做。** 预算约 2 小时。当前入口 `cd bench && ./run_all.sh tier1p`。

**喂给 §6 gate 的数据必须来自满足 §3 全部准入条件的 harness。** 这是硬门槛：不满足的 harness 即使跑得出完整矩阵，其输出也不得进入 gate。

### 5.1 · 依赖度 × 结构复杂度的收益地图 — 单个信息量最大的实验

在真机上复刻 BlockMaestro Figure 12，**但修正其实验设计缺陷**。

**为什么关键**：BlockMaestro 在 **28 SM 的 Titan X**（896 并发 TB 槽位）上得出两条边界——依赖度 > 32 或 grid > 2048 TB 时收益归零。B200/B300 约 148 SM，并发 CTA 槽位多出约 5 倍，**这两条边界往哪边移是未知且决定性的**：

- 若随槽位数等比放大 → CTA 级方向的适用面显著扩大
- 若反而更紧（现代负载 grid 也更大，单 kernel 更容易填满 GPU）→ 整个方向的适用面需重新评估

**必须修正的设计缺陷**：BlockMaestro 注入依赖的方式是 **n-组全连接**，依赖度与结构复杂度**同步增长**，两个变量被混在一起，因此无法区分收益归零是「边太多」还是「结构太复杂」造成的。

真实负载里这两者常常分离——LLM 的 FFN GEMM 链依赖度 ≈ BM（128/256）但是连续区间；DSA 的 indexer→topk 依赖度可达数千同样是连续区间。**照搬其阈值会把 LLM 中最常见的两类模式错误排除。**

因此本实验必须把两者作为**独立轴**分别扫描，两组扫描各自固定另一个变量：

| 扫描 | 固定 | 变量 | 要求范围 |
|---|---|---|---|
| 依赖度轴 | 结构 = `interval` | 依赖度 | 1 → 1024（对数步进） |
| 结构轴 | 依赖度 = 32（`self` 例外，按定义恒为 1） | 结构 | `self` / `interval` / `grouped` / `strided` / `random` |

`self` 是 1-to-1 的语义控制点，若强行把它设成 degree 32 就不再是 `self`。因此它作为
degree=1 的端点与其余固定 degree=32 的结构并列；报告必须显式标出这个例外，不能把五点写成
严格等 degree。该澄清修正了结构名称与“固定 32”之间原本不可同时满足的规格矛盾。

两组都要在多个 grid 规模上重复，grid 范围要求见 §5.3。

**产出**：逐次采样记录 → 逐配置统计量（含 `space_pct`、`captured_pct`、`tightness`、`eff_degree` 与置信区间）。分析脚本必须能识别自己的 schema 并拒绝喂错的输入（§13）。

### 5.2 · tail / prologue 长度比扫描

依赖度固定为 8，扫 tail/prologue 比 1→16。

**判读**：收益随 tail 占比如何变化。**tail 是收益的唯一来源**（§2）——tail 极短时即使依赖结构再理想也没有可重叠的窗口，此时收益接近零属于预期，不是实现有问题。

### 5.3 · grid 规模：必须覆盖多波区间 `P,C > SM`

**这是判决能否外推到真实 grid 规模的关键边界。**

P、C 与 SM 的比值只是**计划的 grid-ratio 分类**，不是常驻或波次的语义证明。特别是，`P,C ≤ SM` **不能**推出「每个 SM 至多一个 CTA」：调度器可以在一个 SM 上放多个 CTA 而让另一个 SM 暂时不放，每 SM 真实常驻上限还取决于 threads、寄存器、shared memory 与 block 上限。

真实负载的 grid 通常远大于 SM 数，后续 CTA 会因资源与调度而延后进入。这种真多波情形与较小 grid 下 CTA 级依赖的收益结构**可能完全不同**——调度器本身就在做一部分「等待」，CTA 级机制能额外拿到的空间未知。

**没有 timestamp proof 的 grid 收益不能外推到真多波。** 只有 grid-ratio 覆盖而没有真多波语义证据时，§6 的判决只能读作「机制可行」，不能读作「方向值得投入」。

要求的 grid 范围：

| 计划 grid-ratio 类别 | P, C 相对 SM | 覆盖要求 |
|---|---|---|
| 欠 SM 数 | `< SM` | 至少 2 个点 |
| 等于 SM 数 | `= SM` | 必测 |
| 多波候选 | `2×SM`、`8×SM`、`32×SM` | **必测** |

**对 harness 的要求**：`2×/8×/32×SM` 只有在 `%globaltimer` trace 完整证明真多波后才可记为多波覆盖：至少一个消费者 CTA 已经开始时，仍有生产者 CTA 尚未开始；同时全部生产者的 start / end timestamp 必须完整，并证明它们最终前进并完成。在这个情形下仍要满足 §3.1：Floor 的 trigger 在 ready，Impl 的 launch trigger 可在 entry，但 ready flag 不能因为「后续生产者还没上」就提前发布，也不能退化成 grid 级等待。这是 `.cu` 的语义要求与 trace 准入条件，不是由 grid ratio 或一个命令行开关推导的标签（§13）。

---

## 6. Tier 1 Gate（权威定义）

**本节是 gate 阈值的唯一权威来源，并由 [`tools/gate.py`](tools/gate.py) 直接实现。**

判据是 Tier 1.1 在多数配置下的典型 `Ceiling − Floor`（`gate.py` 取各配置 space% 的**中位数**）：

| Tier 1 结果 | 判决 | 下一步 |
|---|---|---|
| **≥ 8%** | `GO` | 继续 Tier 2/3 + Tier 4 + Tier 5，跑满预算 |
| **2 – 8%** | `LLM_ONLY` | 跳过 Tier 2/3，直接做 Tier 4 端到端，确认真实负载上还剩多少 |
| **< 2%** | `STOP` | **停**。只跑 Tier 4 三档确认，然后收工 |
| 任一配置的 correctness 或必要语义 trace proof 失败 | `INVALID` | **本轮所有计时都不能用**，先修正准入失败 |

```bash
python3 tools/gate.py bench/results/pilot_analysis.json --json bench/results/gate.json
```

`INVALID` 的退出码是 2，其余是 0——退出码表示「gate 是否算得出来」，不表示「是否通过」，分支要看 `verdict` 字段。

manifest、计划参数覆盖或 31-repeat 统计不完整时，工具仍可给出仅供诊断的数值判决，但必须同时
写 `plan_sweep_complete=false`；这种 provisional 判决不能开启 Tier 2/3，也不能冒充完整正式
campaign。它与 `INVALID` 的区别是：前者缺少完整 campaign 的外推资格，后者已有配置的
correctness / 必要语义证据失败，因而整轮 timing 都不可用。FAST smoke 正是前一种情况。

**要改阈值改本节**，并在同一次改动里同步 `gate.py` 顶部的 `GO_THRESHOLD` / `STOP_THRESHOLD`，否则方法学与实际执行的代码会不一致。

### 判读时必须带上的证据边界

**不要用眼睛重新判一遍 gate。** 如果你不同意判决，分歧在于本节的阈值，应该改本节，而不是在报告里写一个不同的结论。

synthetic 微基准通过 gate 只说明「**机制在所述限制下可行**」，不等于真实负载上的收益。

**判决必须连同覆盖边界一起读。** 若产出该判决的矩阵没有覆盖 §5.3 要求的多波区间，`GO` 支撑的是「可以去测真实负载」，**不足以支撑「跑满预算做 Tier 2/3」**。gate 的实现必须自动检测这一点并把 caveat 印在判决旁边，不能指望判读的人记得（§13）。

---

## 7. Tier 2 / 3 — 机制对比与特定维度

预算约 3 小时。**仅在 §6 判为 `GO` 时执行**——判为 `LLM_ONLY` 时跳过本节直接做 §8。

本节各实验同样受 §3 约束。它们比较的是**机制之间的差异**，因此对触发语义的正确性比 Tier 1 更敏感：若各档共用一个被提前满足的等待，所有档位会收敛到同一个值，看起来像「协议选择无关紧要」这个错误结论。

### 7.1 · 同步协议横评 — B1

固定 1-to-1 依赖（`--structure self`），只换同步实现：软件自旋（固定间隔）/ 自旋 + `__nanosleep` 指数退避 / 单调完成计数器 / `griddepcontrol.wait`（对照）。

**指标**：依赖满足 → 开始执行的延迟、轮询产生的 L2 读请求数、对并发 background kernel 的干扰幅度。后两项是选型的关键——一个延迟更低但把 L2 打满的协议，在真实负载里是负收益。

### 7.2 · 等待位置的定价 — B2

派发前门控是 `[H+]`，真机无法实现。通过 §4.3 的 occupancy 与 productive-background 配对曲线，只在假设 `resident ≤ hypothetical [H+] ≤ deferred` 的吞吐关系以及对应 e2e 单调性成立时，构造「若等待不占槽位能省多少」的条件化 scenario envelope。若硬件 bookkeeping 或更精确 admission 使单调性不成立，观测差值不是硬件保证边界。不需要额外 GPU 时间。

### 7.3 · 依赖表示编码的成本交叉点 — A3

区间二元组 / 位掩码 / CSR 邻接表，扫描依赖度 1→64，测解码延迟、额外访存量、交叉点位置。

结构要同时取 `interval`（区间编码精确）与 `strided`（区间编码严重失真），**交叉点位置在这两种结构下必然不同**，这正是要测的东西。

### 7.4 · in-order completion 在 diamond 上的代价 — A1 B3

把 diamond 拓扑扩展到 CTA 级，对比「强制 K3 等 K2」vs「允许乱序完成」。**关键是扫描 K2/K3 的时长比（1:1 → 1:10）**：时长差越大，压平非线性拓扑的损失越大。

### 7.5 · C1 四版本对比 — C1

producer 写 X 字节/CTA、consumer 读同样数据，扫描 X（1KB→64KB）：融合 + cluster/DSMEM（上界）/ 分离 + L2 persistence / 分离 + 默认 / 分离 + 强制 L2 bypass（下界）。

**指标**：DRAM 流量、L2 命中率、端到端时间。**同时测融合版 shared memory 占用翻倍导致的 occupancy 损失**，找 trade-off 平衡点——只看访存量会得出「融合永远更好」的错误结论。

### 7.6 · CLC 持久化 kernel 的调度策略对比 — B4

TB 调度器不可编程，但**用 CLC + 持久化 kernel 可在软件里完整复现调度策略**：持久 kernel 自己决定优先领取生产者 tile 还是消费者 tile，从而真实对比 producer-priority / consumer-priority / 局部性优先。

**这是本设备上评估 B4 的唯一可行途径，且有硬件原语支撑**（可行性前提由 §4.4 提供：若 `try_cancel` 的仲裁吞吐不足，这条路走不通）。

---

## 8. Tier 4 — 真实 LLM 端到端（Qwen3.6-27B，单卡）

预算约 2 小时。需要约 54GB 模型下载与 vLLM，**因此排在 §6 判决之后**：无论判为 `GO` 还是 `LLM_ONLY` 都要做，判为 `STOP` 时只做 §8.2 的三档确认。

```bash
pip install vllm
huggingface-cli download Qwen/Qwen3.6-27B
cd bench/llm && ./run_llm_sweep.sh          # FAST=1 先冒烟
python3 tools/llm_bracket.py bench/llm/results_llm/summary_llm.txt
```

### 8.1 · 为什么是 Qwen3.6-27B

| 项 | 值 |
|---|---|
| 参数量 | 27B **dense**（非 MoE） |
| 层数 | 64 |
| 布局 | `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` |
| 即 | **48 层 Gated DeltaNet**（线性注意力，递归，**无 KV cache**）+ **16 层标准 GQA**（24Q/4KV，head dim 256） |
| hidden / FFN intermediate | 5120 / 17408 |
| 其他 | MTP 支持推测解码；BF16 约 54GB，单卡放得下且 KV cache 压力很小 |

选它的最大理由：分块线性注意力是 chunk 内并行、chunk 间顺序递归传状态，依赖形态是**低依赖度的 1-to-1 长链**（链长 = seq_len / chunk_size）——**这是 CTA 级依赖最有利的场景，而它占了 48/64 层**。本项目研究的依赖形态是这个模型的主体计算模式，不是边角情况。

另两类值得测的模式：FFN 的 GEMM 链（RMSNorm → gate/up GEMM → SwiGLU → down GEMM，输出 tile (m,n) 依赖上游 token 行区间 `[m·BM, (m+1)·BM)`，**依赖度高但结构极规整**，正是 §5.1 那个「必须分开测」要求的真实样本）；MTP 推测解码的 draft → verify 依赖。

### 8.2 · 核心测量：三档

| 档位 | 配置 |
|---|---|
| `PDL_off` | 全部关闭 |
| `PDL_grid`（**Floor**） | 现状生产配置 |
| `Ceiling` | 去掉 `gdc_wait`（结果错误，只测时间），代表依赖零成本 |

**`Ceiling − PDL_grid` = CTA 级还能拿到的上限。这是整个项目最关键的单个数字。**

### 8.3 · Floor 必须是「PDL 已开启」，不是「PDL 关闭」

生产框架**已经在用** grid 级 PDL：

| 框架 | 开关 | 备注 |
|---|---|---|
| TensorRT-LLM | `TRTLLM_ENABLE_PDL=1` | 用于 top-k、GEMM、MoE routing |
| vLLM | `TRTLLM_ENABLE_PDL` + `TORCHINDUCTOR_ENABLE_PDL` | **仅在 FULL CUDA graph 下启用**——PDL 的 host 开销在 prefill/piecewise 下反而是负收益 |
| SGLang / Triton | `tl.extra.cuda.gdc_wait()` / `gdc_launch_dependents()` | JIT kernel 与 MoE backend |

已公开的收益量级：vLLM BS1 约 2–3%（个别 10%）、TRT-LLM DeepSeek-R1 on B200 约 3%（168→173 TPS/user）、Triton 简单 kernel 15%、LLM 连续层最高 33%。

**拿 PDL-off 当基线会严重高估 CTA 级的收益空间。** `llm_bracket.py` 会在 grid 级收益落在 2–33% 区间外时给出告警，通常意味着 FULL CUDA graph 没开。

### 8.4 · 一个已部署且有明确改进空间的靶点

Triton 的 PDL 实现是**在任何 `tl.load` 之前无条件 `gdc_wait()`**，保守假设前驱 kernel 可能写了任意位置——这正是 BlockMaestro 所说的「全连接回退」，只不过现在是部署在生产框架里的现状。有 CTA 级依赖信息就能大幅缩小等待范围，**改进空间可以直接量化**。

### 8.5 · 扫描维度与 ground truth

扫描 batch size（1/4/16/64 decode + 各长度 prefill）与序列长度（4K/32K/128K，改变 DeltaNet chunk 链长）。**BS=1 decode 时 grid 最小、GPU 填不满、重叠空间最大**——vLLM 的 "never hurts in the low-batch scenario" 印证了这一点。这一扫描把 §5 的微基准结论与真实负载对齐。

CTA 级依赖 ground truth：对 **GEMM / norm / SwiGLU** 这类 tile 映射公开可知的 kernel，**直接从 CUTLASS/Triton 的 tile 划分推导，不需插桩**；仅对不透明 kernel 用 NVBit 记录每 CTA 读写地址范围再离线求交（开销 100x+，限少量 kernel）。产出真实依赖二分图，用于计算**区间紧度**与**假边率**。

工具链：**nsys** 全时间线（kernel 序列、grid 尺寸、时长）、**ncu** 单 kernel 指标（L2/DRAM 流量、achieved occupancy）。设备侧 `%globaltimer` 打点需要改 kernel，对框架 kernel 不现实，**只在微基准和自己复刻的 kernel 上使用**。

---

## 9. Tier 5 — DSA 算子链（DeepSeek-V3.2 / GLM-5.x）

```bash
cd bench/dsa

# 自研 native CTA 依赖链：先做短入场，再在全新目录跑四点正式矩阵。
RESULTS=results_native_smoke FAST=1 PROFILE=0 ./run_dsa_chain.sh
RESULTS=results_native_formal FAST=0 PROFILE=1 ./run_dsa_chain.sh

# production component：短跑、compact 与 exact-formal 都按 row fragment 独立封存。
EXECUTE_GPU=1 TIER5_PRODUCTION_GPU_ALLOWED=1 FAST=1 \
  RESULTS=results_production_smoke ./run_production_tier5_fragments.sh
# 一小时执行上限下的独立 compact-14：两模型、4K/128K、三 workload + MoE。
EXECUTE_GPU=1 TIER5_PRODUCTION_GPU_ALLOWED=1 FAST=1 \
  MODELS=deepseek_v32,glm5 SEQS=4096,131072 \
  WORKLOADS=operator_chain,single_layer,indexshare_fsss \
  WARMUP=5 REPEATS=31 MOE_TOKENS=4096 \
  RESULTS=results_production_compact ./run_production_tier5_fragments.sh
python3 validate_production_tier5_compact.py results_production_compact
# 原计划的完整 exact-26 仍是另一条、更宽的 admission。
EXECUTE_GPU=1 TIER5_PRODUCTION_GPU_ALLOWED=1 FAST=0 \
  RESULTS=results_production_formal ./run_production_tier5_fragments.sh
```

native 正式矩阵固定覆盖 4K/32K exact mapping 与 128K/1M
work-complete packed proxy；报告必须分开解释这两类证据。production 有两种不得混写的
准入范围：原计划 exact-26 只有全量重新验证后才能置
`accepted_workload_timing=1`；用户为一小时执行上限明确授权的 compact-14 固定为
两模型 × {4K,128K} × 三 workload，再加每模型一条 MoE，只能由独立 validator 置
`accepted_compact_workload_timing=1`。compact 必须保持
`accepted_workload_timing=0`、`accepted_exact26_workload_timing=0`，并明确排除
32K/1M timing，不能冒充 exact-26。production API 不暴露 CTA-readiness 或 unordered Ceiling，因此其
`accepted_CTA_bracket` 必须保持 0，不能从 component off/on timing 推导 CTA headroom。

### 9.1 · 单卡跑不了整模型，但注意力路径跑得了

| 模型 | 参数量 | FP8 | FP4 | 单卡 B300 (288GB) |
|---|---|---|---|---|
| DeepSeek-V3.2 | 671B | 671GB | 335GB | 装不下 |
| GLM-5.2 | 744B / 40B-active | 744GB | 372GB | 装不下 |

且 MoE 要求全部 expert 常驻，最少 2 卡（FP4）到 8 卡（FP8）。**但 DSA 的注意力层参数量很小，单层与算子链可以在单卡上用真实 shape 实测。**

GLM-5.2 关键配置：hidden 6144、`kv_lora_rank` 512、`q_lora_rank` 2048、`index_head_dim` 128、`index_n_heads` 32、`index_topk` 2048、78 层。开源实现可用：DeepSeek 官方 inference 代码、TileLang 的 DSA kernel、FlashMLA。

**三层处理**：架构级依赖图推导（纯离线，§10.1）→ 单层/算子链单卡实测 → 整模型端到端**仅作纸面外推**。

### 9.2 · 依赖形态分析（本节核心产出）

kernel 链为 `lightning indexer → top-k selection → sparse MLA attention`。

**(1) indexer → topk：依赖度极高但结构极规整。** topk 处理 query 块 *j* 时需要该 *j* 在 key 方向的**全部**得分，因此依赖 indexer 中该 query 块的**整行 CTA**。依赖度 = L / key_block，1M 上下文下可达**数千**，但结构是二维网格中的**一段连续区间，区间表示仅 O(1)**。

这是「高依赖度 ≠ 复杂结构」的典型样本，可直接用来检验 §5.1 的实验设计要求。**BlockMaestro「依赖度 > 32 收益归零」的结论会错误地排除它。**

**(2) topk → sparse attention：看似间接访存，实则 RAW 依赖链是规整的。**

| 数据 | 来源 | 是否构成 kernel 间 RAW 依赖 |
|---|---|---|
| 索引数组 `idx` | 本步 topk 产生 | **是**，且按 query 块是 **1-to-1** |
| 被选中的 KV 条目 | **更早的 decode step 或 prefill 写入** | **否** |

**间接寻址决定的是「读历史数据的哪个位置」，并不构成对紧邻前驱的不可预测依赖。** 两个后果：**DSA 对 CTA 级依赖其实是友好的**，不是想象中的困难场景；**BlockMaestro 的 Algorithm 1 会在这里错误地保守退化**——它一见到地址来源于 global load 就 bail out，无法区分「间接读的是本步产出」与「间接读的是历史数据」。**按数据的产生时间而非仅按地址来源判定**，是一个具体且可实现的算法改进点（已写入设计空间报告 A2）。

**(3) GLM-5.2 的 IndexShare 把依赖跨度拉长。** 每 4 层共享一个 indexer、top-k 索引在 4 层内复用：`indexer(L1) → attn(L1) → attn(L2) → attn(L3) → attn(L4)`。索引数组由**数个 kernel 之前**的算子产生，恰好满足 [`archive/prologue_inspector_cta_pdl.md`](archive/prologue_inspector_cta_pdl.md) §9 的硬约束（「结构数组不能由紧邻的生产者 kernel 写」），同时也是 A1 维度「跨度 > 1」的真实样本。

**(4) MoE 的 dispatch/combine 才是真正的困难场景。** `router → top-8 → permute/gather → grouped GEMM → unpermute/scatter`：permute 的索引由**紧邻前驱 router** 产生，属于「结构动态」，inspector 救不了；grouped GEMM 每个 CTA 的依赖度取决于该 expert 分到多少 token，运行时才确定。TRT-LLM 已在 MoE routing 上用 PDL，可作对照基线。

### 9.3 · 单卡可测 vs 只能外推

**可实测**：`indexer → topk → sparse MLA` 三算子链的三档对比（真实 shape）；扫描上下文 4K/32K/128K/1M（indexer 的 O(L²) 特性使长上下文下该链占比急剧上升，收益结构随之移动，**单点测量会误导**）；单层 MLA + DSA 完整 forward；MoE 层用**缩减 expert 数**（32 而非 256）复现 dispatch/combine 的依赖形态——依赖形态由 top-k 路由决定，与 expert 总数无关。

**只能纸面外推**：整模型端到端 TPS/user；EP/TP 并行下的跨卡依赖（超出本项目单卡范围）。

---

## 10. 离线实验（不需要 GPU）

可以在开发机上跑，也可以在等 GPU 任务时并行做。

### 10.1 · 依赖 oracle 与依赖分析精度 — A2

```bash
python3 tools/dep_oracle.py --model qwen3.6-27b --tokens 256 --seq 2048
```

用 oracle 依赖图对比各来源推出的图：**假边率**（决定性能损失）、**漏边率（必须为 0）**、可分析 kernel 覆盖率。launch 期分析的耗时需实测是否随 gridDim 线性增长。

### 10.2 · 分析链自检

任何分析脚本改动后都先在合成夹具上验证，不要拿真实结果当测试用例：

```bash
python3 tools/make_test_fixtures.py --out /tmp/ctafix
python3 tools/analyze_pilot.py /tmp/ctafix/pilot_matrix.log \
        --expected /tmp/ctafix/pilot_expected_tags.txt \
        --json /tmp/ctafix/pilot_analysis.json --csv /tmp/ctafix/pilot_summary.csv
python3 tools/gate.py          /tmp/ctafix/pilot_analysis.json
python3 tools/analyze.py       /tmp/ctafix/summary.txt
python3 tools/cta_timeline.py  /tmp/ctafix/trace.csv
python3 tools/llm_bracket.py   /tmp/ctafix/summary_llm.txt
```

`preflight.sh` 会自动跑其中的 gate 路径，因此**判决链坏掉这件事在租机器之前就能发现**。

---

## 11. 各维度可测性判定

| 类别 | 维度 | 说明 |
|---|---|---|
| **真机可直接测** | A1、A3、B1、B3、C2、D1、E1 | 软件即可构造 |
| **可用 CLC 持久化 kernel 在软件中复现后测** | B4、A4 的分布式变体 | 持久 kernel 自己做调度决策（§7.6） |
| **只能条件化包夹估值** | B2（派发前门控是 `[H+]`）、A4 的集中式变体 | 用 occupancy / productive-background 配对曲线构造明示单调性假设的 scenario envelope（§7.2） |
| **只能测上界** | C1 | 跨 kernel shared memory 所有权转移无法实现，用 fused+DSMEM 作上界 |
| **主要靠离线分析** | A2 | oracle 依赖图对比，不需 GPU（§10.1） |
| **无对口实验** | D2 | 依赖描述的 soundness 验证需要先有实现 |

---

## 12. 预算与最小可交付

| 阶段 | 预算 | 执行条件 |
|---|---|---|
| §4 Tier 0 基础事实 | ~1h | 无条件，第一优先 |
| §5 Tier 1 收益地图 | ~2h | 紧接 Tier 0，产出 §6 判决 |
| §8 Tier 4 LLM profiling | ~2h | 判决为 `GO` 或 `LLM_ONLY` |
| §7 Tier 2/3 机制对比 | ~3h | 仅判决为 `GO` |
| **合计** | **~8 GPU-hours** | 到判决为止约 3h |

### 最小可交付

若时间或预算被砍，按此优先级保底：

1. **§5.1 依赖度 × 结构收益地图** —— 单个信息量最大的实验，决定整个方向是否成立
2. **§8.2 的 `Ceiling − PDL_grid`** —— 真实负载上还剩多少空间
3. **§4.1 的重叠层数** —— 决定 B3 维度哪些选项可达
4. **§4.3 的 occupancy 曲线** —— B2 维度定价

前两项合计约 4 GPU-hours，足以支撑「这个方向值不值得做」的判断。

---

## 13. 对驱动脚本与工具的要求

本节是写脚本时的验收清单。**代码当前实现到哪一步不在本计划记录**——那是 [`bench/README.md`](bench/README.md) 的职责；执行到哪一步、下一步怎么续跑见 [`EXPERIMENT_REPORT_INDEX.md`](EXPERIMENT_REPORT_INDEX.md)。

### 13.1 驱动契约

- **无人值守**：整场从单一入口跑完，中途不需要人做任何决定，包括 §6 的分支
- **可断点续跑**：每步落一个完成标记，重跑自动跳过已完成的步骤。标记必须绑定 FAST/formal、
  完整参数、可执行文件内容、GPU 身份/CC 与 driver；任一项变化都要保留旧 attempt 并重跑，
  不能让同名 tag 复用异机或短跑结果。掉线重连后重跑必须安全
- **fail-soft**：单步失败记入 `failures.log` 后继续，不中断整场。**唯一立即退出的例外是自检失败**——机器或工具链坏掉时不要在坏掉的 harness 上烧 GPU 时间；正式 strict validator 失败可继续收集下游诊断，但最终 session 必须非零并标记 `INVALID`
- **原始数字全部落盘**：任何进入报告的数字都要能追溯到一个原始记录文件
- **自报设备**：`device.txt` 记录实际硬件，报告引用它而不是计划里假定的型号

### 13.2 判决必须机器可读

§6 的三态判决由工具计算并输出结构化结果（判决字符串 + 统计量 + 退出码），**不能要求人读表**。判决工具同时要检测覆盖边界：`P,C ≤ SM` 只是 grid ratio，不能自动命名为单波；`P,C > SM` 的真多波必须用 §5.3 时间戳证明。这些 caveat 随判决一起输出。

### 13.3 schema 隔离

不同 harness 的输出格式不得互相污染。每个分析脚本必须**识别自己的 schema 并拒绝喂错的输入**，且拒绝时要指明该用哪个脚本——静默地把别人的记录解析成一个看起来合理的结果，是比崩溃严重得多的失效模式。

### 13.4 上机前必须可离线自检

全部分析与判决链路要能在**没有 GPU 的机器上**用合成夹具跑通，并纳入 preflight。夹具要覆盖到判决工具，否则「判决链坏了」只能在花钱之后才发现。严格 validator 还必须有畸形字段、
整数 trace 不一致与旧 schema 的负面测试；失败路径要原子替换旧 JSON 为当前 `FAIL`。

### 13.5 参数空间要覆盖计划声明的范围

脚本的扫描范围以本计划各节为准。实现进度见 [`bench/README.md`](bench/README.md)；执行进度见 [`EXPERIMENT_REPORT_INDEX.md`](EXPERIMENT_REPORT_INDEX.md)。验收覆盖至少包括：

| 计划范围 | harness 必须覆盖的验收面 |
|---|---|
| §7.1 / §7.3 | 满足 §3 触发语义的协议横评；interval / bitmask / CSR 的独立 decode、精确边集合与错误 Ceiling sentinel |
| §7.4 / §7.5 / §7.6 | CTA diamond 的分支顺序、C1 四版本的完整数据校验、CLC 持久调度的 token conservation 与策略对照 |
| §8 | PDL-off / grid Floor / unsafe Ceiling 三档、真实模型身份、active variant、完整 decode/prefill admission 与 profiler 绑定 |
| §9 native | 四种 mode 工作量同构、programmatic edge 的 PTX/SASS 与 trace 证明、全元素正确性、四个 context、正式 profiler sidecar |
| §9 production | exact-26：26 个 canonical row；compact-14：仅两模型的 4K/128K 三 workload + MoE。两者都要求全 causal pair/元素覆盖、每片独占与失败原子封存及 fresh revalidation；compact 只置独立 scoped acceptance，CTA bracket 字段保持不可冒充 |

§5.3 多波：`cta_dep_pilot` / `run_all.sh tier1p` 已按本节要求放开 `P,C > SM`（仍须满足 §3.1）。**真机测量**是否完成不在本表跟踪。

改完先在 §10.2 的夹具上验证，再上机。
