# 前导探测式 CTA 级 PDL（Prologue-Inspector PDL）

> 设计草案 / brainstorm。目标：在**间接访存（gather）**场景下实现 CTA 粒度的依赖等待，
> 且**今天的 sm_90+ 硬件上就能用纯软件原型跑通并测量**，硬件扩展是可选的加速项而非前提。
>
> 关联文档：
> - `../docs/cuda_13.4_pdl_clc_interfaces.md` —— PDL/CLC 现有接口与硬件反推（本文 §8 直接对接其 §3.4）
> - `../papers/BlockMaestro_论文报告.md` —— ISCA'21，全静态分析路线
> - `../papers/Kim_PACT16_...pdf` —— PACT'16，CRCS + PPCS，粗粒度引用计数路线

---

## 1. 一句话

**让消费者 CTA 在 PDL prologue 里自己算出"我依赖哪几个生产者 CTA"，然后只等那几个**——
生产者的写集合由编译期静态分析给出（几乎总是仿射的），消费者的读集合由运行时自我探测给出（间接的那一半），
两者拼合即可覆盖 gather 类不规则负载，而这正是 BlockMaestro 纯静态路线放弃的场景。

---

## 2. 问题定位

现有三条路线各自的位置：

| 路线 | 代表 | 依赖粒度 | 依赖来源 | 对间接访存 |
|---|---|---|---|---|
| 全静态分析 | BlockMaestro (ISCA'21) | TB→TB 点对点 | JIT 期 PTX 值域分析 | **放弃**，退化为全连接 barrier |
| 粗粒度引用计数 | Kim et al. (PACT'16) | 引用计数记分牌 | 编译器 + 硬件 | 粗粒度，精度有限 |
| 显式标注 | NVIDIA PDL | **grid→grid** all-or-nothing | 程序员放置 trigger/wait | 无关（不表达数据依赖） |
| 任务图 | Wireframe / ATMI / CUDA Graph | 任务级 | 程序员构造 | 需人工给出 |

空白处很明确：**没有一条路线能在"程序员负担可接受"的前提下，为 gather 类访存提供 CTA 粒度的依赖**。

BlockMaestro 的失败点是它要求消费者的读集合在 kernel 执行前就静态可知。
但 `A[B[i]]` 里的 `B[]` 通常是**结构数组**——稀疏矩阵的 `colIdx[]`、图的邻接表、FEM 的单元-节点映射——
它在运行时是**确定的**，只是编译期不知道。

---

## 3. 核心洞察：读写不对称性

这是整个设计的支点。

**在真实 GPU kernel 中，间接寻址几乎只出现在读侧，不出现在写侧。**

```cuda
// 典型的 SpMV consumer
for (e = rowPtr[r]; e < rowPtr[r+1]; e++)
    sum += val[e] * x[colIdx[e]];   // ← 读：间接（gather）
y[r] = sum;                          // ← 写：仿射
```

原因是结构性的：scatter 写需要处理写冲突，代价高（原子操作或预排序），
所以绝大多数 kernel 被设计成 **gather-compute-affine-store** 的形态。
每个 CTA 负责一段连续的输出，输出映射是 `blockIdx` 的仿射函数。

由此得到一个关键的可行性判断：

| | 生产者 | 消费者 |
|---|---|---|
| 关心的集合 | **写**集合 | **读**集合 |
| 典型形态 | 仿射（`out[blockIdx.x * T + t]`） | 间接（`in[B[i]]`） |
| 可否静态分析 | **可以** | 不可以 |
| 本设计的解法 | 编译期分析（BlockMaestro Algorithm 1 的前半段即可） | **运行时自我探测** |

**结论**：不需要让静态分析包打天下。只要生产者一侧能给出"地址 → 生产者 CTA ID"的映射公式（静态可得），
消费者一侧就可以在运行时把自己的读地址代进这个公式，算出依赖的 CTA 区间。

这是 BlockMaestro（全静态，两侧都要求可分析）和 inspector-executor（全动态，两侧都靠运行时）
之间被遗漏的一个中间点，而它恰好命中了实际负载的形态。

---

## 4. 设计概览

```
                 producer grid                          consumer grid
                 ─────────────                          ─────────────
  t0   trigger（授权 consumer 启动）  ─────────────────►  launch
       │                                                 │
       │  CTA 0 ── compute ── fence ── set done[0]        │  prologue:
       │  CTA 1 ── compute ── fence ── set done[1]        │    ① inspector: 扫自己要读的 B[]
       │  CTA 2 ── compute ── fence ── set done[2]        │       → [min,max] 读地址
       │  ...                                            │    ② 代入静态映射公式
       │                                                 │       → 依赖的 producer CTA 区间 [p_lo,p_hi]
       │                                                 │    ③ wait_range(done, p_lo, p_hi)
       │                                                 │       ← 只等这几个，不等整个 grid
       ▼                                                 ▼
      完成                                            executor: 真正干活
```

三个组成部分：

1. **生产者侧完成位图**（`done[]`）：每个生产者 CTA 完成时，release-fence 后置位自己的 bit。
2. **消费者侧 prologue inspector**：每个消费者 CTA 扫描自己负责范围的结构数组，得出保守的读地址区间，
   代入静态映射公式换算成生产者 CTA 区间。
3. **区间等待原语**：`wait_range(done, p_lo, p_hi)`，替代 `cudaGridDependencySynchronize()`。

**PDL 在这里退化为纯粹的"启动使能"**——它只负责让消费者 grid 能提前上机器，
数据依赖的正确性完全由软件协议（fence + 位图）保证。这一点后面 §7 会展开，
因为它正是"零新硬件即可原型"的关键。

---

## 5. 详细机制

### 5.1 生产者侧：写集合的静态分析

需要从生产者 kernel 的 PTX 中提取一个**映射函数** `M: 地址 → CTA ID`。

对绝大多数 kernel，写地址形如 `base + stride * (blockIdx.x * blockDim.x + threadIdx.x)`，
于是 `M(addr) = (addr - base) / (stride * blockDim.x)`。这正是 BlockMaestro Algorithm 1
第 2–18 行（反向数据流追踪）的产出，**不需要第 19–21 行的逐线程值域枚举**——
只需要符号形式的表达式，代价极低，且完全可以在编译期完成、作为元数据随 fatbinary 分发。

失败时（生产者也是 scatter 写）直接降级为等待整个 grid，即今天 PDL 的行为。

### 5.2 生产者侧：完成位图

```cuda
__global__ void producer(float* out, int n) {
    // trigger 提到最前面：PDL 在此仅作"启动使能"，不承担数据可见性
    cudaTriggerProgrammaticLaunchCompletion();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = compute(i);

    __syncthreads();                          // 本 CTA 全部写完
    if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(done[blockIdx.x]);
        flag.store(1, cuda::memory_order_release);   // release：本 CTA 的写先于置位可见
    }
}
```

存储开销：每个生产者 CTA 1 个 int（可压缩到 1 bit，但 int 免去了位操作的原子开销）。
10K CTA 的 grid = 40KB 全局内存，可忽略。

> **trigger 位置的变更是本设计与标准 PDL 用法的一个实质差异。**
> 标准用法把 `cudaTriggerProgrammaticLaunchCompletion()` 放在"必须先完成的写之后、可重叠尾部之前"
> （见 `../docs/cuda_13.4_pdl_clc_interfaces.md` 附录 B.2 的实测确认）。
> 本设计把它提到 kernel 最前面，因为数据可见性不再由 `griddepcontrol.wait` 承担。
> 这是合法的——文档 §1.2 明确指出该指令**本身不提供任何内存可见性保证**，
> 它只影响 launch eligibility。

### 5.3 消费者侧：prologue inspector

```cuda
__global__ void consumer(const float* x, const int* rowPtr, const int* colIdx,
                         const int* done, float* y, int rows_per_cta) {
    int r0 = blockIdx.x * rows_per_cta;
    int r1 = min(r0 + rows_per_cta, num_rows);

    // ---------- prologue（与 producer 重叠执行）----------
    // ① inspector：扫自己要读的结构数组，求保守读区间
    int lo = INT_MAX, hi = INT_MIN;
    int e0 = rowPtr[r0], e1 = rowPtr[r1];
    for (int e = e0 + threadIdx.x; e < e1; e += blockDim.x) {
        int c = colIdx[e];                    // ← 这份数据 executor 阶段还要用，等于顺带预热
        lo = min(lo, c);
        hi = max(hi, c);
    }
    lo = blockReduceMin(lo);
    hi = blockReduceMax(hi);

    // ② 代入生产者写集合的静态映射公式
    int p_lo = lo / ELEMS_PER_PRODUCER_CTA;
    int p_hi = hi / ELEMS_PER_PRODUCER_CTA;

    // ③ 只等这个区间（替代 cudaGridDependencySynchronize()）
    wait_producer_range(done, p_lo, p_hi);

    // ---------- executor ----------
    for (int r = r0; r < r1; r++) { /* 正常的 SpMV */ }
}
```

**inspector 的成本几乎为零，甚至是负的**：它读的 `colIdx[]` 正是 executor 阶段无论如何都要读的数据，
所以这一遍扫描同时起到了 **cache 预热 / 隐式 prefetch** 的作用。
如果 CTA 的 nnz 规模允许，可以直接把 `colIdx` 缓存在 shared memory 里供 executor 复用，
那么 inspector 的净开销就只剩一次 block 级 min/max reduction。

**精度**：单个 `[min, max]` 区间对局部性好的数据（RCM / METIS 重排后的稀疏矩阵、带状矩阵、
分区后的图）很紧；对随机稀疏则退化为全区间——即今天 PDL 的行为，不会更差。

**精度可调**：如果生产者 CTA 数不多（≤64 / ≤128），可以用一个 64-bit 或 128-bit mask
代替 `[min,max]`，精确表达任意子集，代价是几个寄存器。
生产者 CTA 数很多时可以按桶粗化（每 bit 代表 k 个生产者 CTA）。
这是一个纯软件的精度/开销旋钮，不需要硬件参与。

### 5.4 等待原语（软件版）

```cuda
__device__ void wait_producer_range(const int* done, int lo, int hi) {
    if (threadIdx.x == 0) {
        for (int p = lo; p <= hi; p++) {
            cuda::atomic_ref<const int, cuda::thread_scope_device> flag(done[p]);
            int backoff = 32;
            while (flag.load(cuda::memory_order_acquire) == 0) {
                __nanosleep(backoff);
                backoff = min(backoff * 2, 1024);   // 指数退避，别烧 L2 带宽
            }
        }
    }
    __syncthreads();     // acquire 语义随 leader 传播给全 CTA
}
```

优化：不必逐个轮询。先线性扫一遍找出第一个未就绪的，只在那个上面等；
或者维护一个 `producer_completed_count` 前缀计数器，若生产者 CTA 完成顺序大致单调（GTO 调度下常见），
一次比较即可覆盖整个区间。

### 5.5 内存模型与正确性

依赖链是标准的 release-acquire：

```
producer CTA k:   写 out[...]  →  release store done[k] = 1
                                        ↓ synchronizes-with
consumer CTA j:   acquire load done[k] == 1  →  读 out[...]
```

`thread_scope_device` 足够（同一 GPU 内）。多 GPU / host 可见性需要 `thread_scope_system`。

**这一点很重要**：内存可见性由**软件 fence** 保证，不依赖任何新硬件。
`../docs/cuda_13.4_pdl_clc_interfaces.md` §3.4 第 4 条列出的"可见性下推到 tile 粒度"
是**硬件隐式做这件事**才需要的；显式协议下 CUDA 现有内存模型已经够用。

---

## 6. 无死锁论证

这是必须论证清楚的一点，因为等待中的消费者 CTA 会**占着 SM 槽位**
（不同于 BlockMaestro——它让未就绪的 TB 根本不上机器）。

朴素的担心是：消费者 CTA 占满所有 SM，等待某个尚未被派发的生产者 CTA → 死锁。

**这个场景不会发生，而且恰恰是被 PDL 那条"看似严苛"的规则排除的。**

`griddepcontrol.launch_dependents` 的语义是：**grid 内所有 CTA 都发过该指令（或已退出）后，
dependent 才有资格启动**（见 `../docs/cuda_13.4_pdl_clc_interfaces.md` §1.1）。因此在任何一个消费者 CTA
被派发的时刻，每个生产者 CTA 必然处于以下两种状态之一：

- **已发过 trigger** ⇒ 它已被派发且驻留在某个 SM 上，持有自己的槽位，不会被消费者抢占；
- **已退出** ⇒ 它的 `done[k]` 已置位。

**不存在"尚未派发的生产者 CTA"**。所以驻留中的生产者 CTA 总能继续推进直至完成并置位，
消费者的等待总会被满足。**死锁在构造上被排除。**

> 这条约束在别处是限制（见 §10），在这里却成了安全性保证。值得注意的是：
> 如果改用 §11 变体 B 的持久化消费者 kernel，这个保证就失效了，需要另行处理。

---

## 7. 今天就能做的软件原型（零新硬件）

**这是本设计相对 BlockMaestro 最大的工程优势：它不需要模拟器。**

上面 §5 的全部代码在 **sm_90+（Hopper 起）** 上都是合法 CUDA：

| 组件 | 依赖的现有能力 | 最低架构 |
|---|---|---|
| PDL 启动使能 | `cudaTriggerProgrammaticLaunchCompletion` + `cudaLaunchAttributeProgrammaticStreamSerialization` | sm_90 |
| 完成位图 | 普通全局内存 + `cuda::atomic_ref` | 任意 |
| release/acquire | CUDA 内存模型（`libcu++`） | 任意 |
| 自旋等待 + 退避 | `__nanosleep` | sm_70 |
| inspector | 普通计算 | 任意 |
| 生产者写集合分析 | 编译期，离线做，先手工推导即可 | — |

**不需要**新 PTX 指令、不需要改调度器、不需要 GPGPU-Sim。
你们的工具链（`../ptx_study/`，Docker nvcc 13.3/13.4，`-arch=sm_103`）可以直接编译。
剩下的唯一前提是一台能跑的 sm_90+ 机器。

对照组设计得也很干净：

| 组 | consumer 的等待方式 | 说明 |
|---|---|---|
| **Baseline** | 无 PDL，两个 kernel 串行 | 现状下界 |
| **PDL-grid** | `cudaGridDependencySynchronize()` | 今天 PDL 的标准用法 |
| **PIP-range** | `wait_producer_range(done, p_lo, p_hi)` | 本设计 |
| **PIP-mask** | 位掩码精确版 | 精度上限 |
| **Oracle** | 离线算出精确依赖，硬编码 | 收益天花板 |

一份实现即可覆盖全部五组，只需切换消费者的等待函数。

---

## 8. 需要的硬件扩展（对接 docs §3.4）

软件原型能证明**收益是否存在**；下面这些硬件扩展降低的是**开销常数**。
逐条对照 `../docs/cuda_13.4_pdl_clc_interfaces.md` §3.4 列出的四项缺失：

| docs §3.4 的缺失项 | 本设计是否需要 | 说明 |
|---|---|---|
| ① per-tile 就绪记分板 | **可选** | 软件位图已能实现；硬件版省掉原子写和轮询访存。成本极低——每 CTA 1 bit，10K CTA 仅 1.25KB |
| ② 细粒度依赖声明接口 | **不需要** | 依赖由消费者 CTA 在 prologue 中**自己算出**，不需要声明接口。这是本设计与 tile-level triggering 的根本差异 |
| ③ 数据驱动精确唤醒 | **想要** | 消除自旋轮询对 L2 带宽的消耗，并让等待中的 warp 真正挂起而非占发射槽 |
| ④ tile 粒度可见性 | **不需要** | 显式 release/acquire 已覆盖（见 §5.5） |

②不需要是本设计最重要的性质。tile-level triggering 之所以迟迟没有落地
（9.4 里 `griddepcontrol` 一字未改），很大原因是它需要一整套**新的软硬件契约**来表达
"consumer tile j 依赖 producer tile 集合 S(j)"。而本设计把这个表达问题
**从接口层移到了消费者 kernel 内部的计算**——依赖不需要被声明，因为它可以被算出来。

如果要提最小的 PTX 扩展，只需一条：

```
griddepcontrol.wait.range  lo, hi;     // 等待前驱 grid 中 CTA ID ∈ [lo,hi] 全部完成且可见
```

语义上是现有 `griddepcontrol.wait` 的严格泛化（`.wait` ≡ `.wait.range 0, gridDim-1`），
向后兼容，硬件侧只需 ①的位图 + ③的唤醒路由。相比 tile-level triggering 所需的
"依赖声明接口 + 依赖映射存储 + 精确唤醒"，这个扩展面小得多。

---

## 9. 适用范围与降级行为

### 硬约束

**结构数组 `B[]` 不能由生产者 kernel 写。** 否则 inspector 在 prologue 里读到的是未完成的数据。

这把负载分成两类（对应 BlockMaestro 报告附录 Q2 里讨论的"情况 A / 情况 B"）：

| | 结构 | 数值 | inspector 可用？ | 例子 |
|---|---|---|---|---|
| **A. 结构静态** | 固定 | 变化 | **是** | 稀疏迭代求解（CG/GMRES/BiCGStab）、固定拓扑图算法（PageRank、固定图 SSSP）、固定网格 FEM/FVM、固定 im2col 索引 |
| **B. 结构动态** | 变化 | 变化 | 否 | BFS frontier、动态 worklist、AMR |

A 类的覆盖面比直觉更大——**稀疏矩阵结构在一次求解中固定、迭代几百次**是 HPC 的标准形态。

### 优雅降级

这是本设计相比 BlockMaestro 全有全无结构的关键优势。**每一层失败都退到上一层，最坏等于现状**：

```
inspector 算出紧区间          → CTA 级等待（最优）
  ↓ 区间宽（随机稀疏）
inspector 算出宽区间          → 接近 grid 级等待（≈ 现状）
  ↓ 结构数组由 producer 写（B 类）
跳过 inspector                → griddepcontrol.wait（= 今天的 PDL）
  ↓ 生产者是 scatter 写，映射公式推不出
跳过整个机制                  → griddepcontrol.wait（= 今天的 PDL）
```

**下界 = 现状**。对一个要落地的设计而言，这个性质比"上界很高"值钱得多——
它意味着可以无条件启用，不需要先做负载分类。

---

## 10. 代价与已知问题

诚实列出，其中前两条是真问题。

### (1) 消费者 grid 的启动仍被"所有生产者 CTA 必须先 trigger"卡住

`launch_dependents` 要求生产者 grid 内**每个** CTA 都发过 trigger。
若生产者 grid 远大于 GPU 容量（比如 10K CTA、GPU 一次只放 2K），
那么消费者要等到生产者基本派发完才能启动，重叠窗口被压缩。

- 这**不是本设计引入的**，今天的 PDL 就有这个限制；
- 有意思的是它和 BlockMaestro 的实验发现同构：论文 Figure 12 显示
  **当单 kernel 的 TB 数达到约 2048（填满 GPU）时收益归零**。
  两者从不同角度指向同一个结论：**细粒度重叠只在"grid 规模适中、装得下 GPU"的区间里有意义**。
- 要突破需要走 §11 变体 B（持久化消费者）。

### (2) 等待中的消费者 CTA 占用 SM 槽位

BlockMaestro 让未就绪 TB 不上机器，本设计让它上机器后等待，损失一部分 occupancy。
同样地，这是今天 PDL 已经承担的代价，不是新增的。
硬件唤醒（§8 第③项）能让等待的 warp 挂起、不占发射槽，但寄存器和 shared memory 仍被占用。

### (3) 自旋轮询消耗 L2 带宽

指数退避 + 前缀计数器可以缓解，但在消费者 CTA 很多时仍是开销。这是需要实测量化的项。

### (4) 静态映射公式的正确性风险

若 `M: 地址 → CTA ID` 推错，后果是**静默数据损坏**（与 BlockMaestro 同类风险）。
缓解：提供 debug 模式，让生产者 CTA 在写时记录实际地址范围，运行时校验映射公式。
开发期开启，发布时关闭——类似 sanitizer。

### (5) inspector 与 executor 的数据一致性

inspector 读 `colIdx[]` 时若该数组同时被别的 kernel 修改，行为未定义。
A 类负载天然满足（结构不变），但需要在文档层面明确这个契约。

---

## 11. 变体

### 变体 A：位掩码精确版

生产者 CTA 数 ≤ 128 时，用 128-bit mask 精确表达依赖集合，而非 `[min,max]` 区间。
适合"依赖散乱但数量少"的场景，代价是几个寄存器和更复杂的等待逻辑。

### 变体 B：持久化消费者 + CLC

用持久化 consumer kernel 配合 `clusterlaunchcontrol.try_cancel`（sm_100+）动态领取 tile，
绕开 §10(1) 的启动限制。
代价：失去 §6 的无死锁保证，需要重新论证（持久 kernel 长期占用 SM，生产者可能饿死）。
这条路和你们 CLC 调研的主题直接衔接，值得单独展开。

### 变体 C：反向——生产者侧推送

不让消费者等，而让生产者 CTA 完成时**主动唤醒**已知的消费者 CTA。
需要"地址 → 消费者 CTA"的反向映射，而消费者是 gather 侧、反向映射静态不可得，
所以需要 inspector 先把依赖**登记**到一张表里，生产者查表推送。
多一次写表开销，但换来精确唤醒、零轮询。硬件成本高于变体 A/B。

---

## 12. 评估方案

### 负载

| 负载 | 类型 | 结构数组 | 期望效果 |
|---|---|---|---|
| SpMV 迭代（CG on 带状/RCM 重排矩阵） | A | `colIdx[]` | **最佳**，区间紧 |
| SpMV 迭代（随机稀疏） | A | `colIdx[]` | 降级到 ≈ 现状，验证不劣化 |
| PageRank（固定图） | A | 邻接表 | 中等，取决于图的局部性 |
| 分层 stencil / FDTD | A（仿射，无间接） | — | 对照：纯静态也能做，验证与 BlockMaestro 同级 |
| BFS（动态 frontier） | B | 动态 | 验证降级路径正确 |

### 指标

1. **端到端加速**（对 §7 的五个对照组）；
2. **重叠度**：消费者 CTA 从启动到通过 wait 的时间 / 总执行时间；
3. **区间紧度**：`(p_hi - p_lo + 1) / gridDim_producer` 的分布——这是 inspector 精度的直接度量，
   也是预测收益的最好指标；
4. **inspector 开销**：prologue 耗时占比，以及它对 executor 阶段 cache 命中率的影响（预期为正）；
5. **轮询访存量**：`done[]` 的读请求数，量化 §10(3)；
6. **occupancy 损失**：等待中 CTA 占用的 SM-cycle。

指标 3 值得单独强调：**它可以离线计算**（给定矩阵结构和分块参数），
所以可以在写任何 kernel 之前就预测某个负载值不值得上这套机制。

---

## 13. 实施路线图

| 阶段 | 内容 | 产出 | 前置条件 |
|---|---|---|---|
| **P0** | 手工写一对 SpMV producer/consumer，硬编码映射公式，跑通五组对照 | 收益是否存在的判据 | sm_90+ 机器 |
| **P1** | 指标 3（区间紧度）的离线分析工具，扫一批矩阵 | 适用范围地图 | 无（纯离线） |
| **P2** | 等待原语库化（`wait_producer_range` / mask 版 / 降级路径），封装成 header | 可复用组件 | P0 |
| **P3** | 生产者写集合的自动分析：PTX 反向数据流追踪（BlockMaestro Alg.1 第 2–18 行），输出映射公式 | 去掉手工推导 | P2 |
| **P4** | 变体 B（持久化 + CLC）探索 | 突破 §10(1) | P2、sm_100+ 机器 |
| **P5** | `griddepcontrol.wait.range` 的硬件成本估算，对接 docs §3.4 | 硬件提案 | P0 的实测数据 |

**P0 和 P1 是可以立刻开始的**，而且 P1 完全不需要 GPU。
建议先做 P1——用离线的区间紧度分析筛出最有希望的负载，再投入 P0 的实现。

---

## 14. 与已有工作的关系

| 维度 | BlockMaestro | Kim PACT'16 | 今天的 PDL | **本设计** |
|---|---|---|---|---|
| 依赖粒度 | TB→TB | 引用计数（粗） | grid→grid | **CTA 区间** |
| 依赖来源 | JIT 全静态 | 编译器+硬件 | 程序员放 trigger | **静态（写侧）+ 运行时（读侧）** |
| gather 支持 | 无（退化 barrier） | 有限 | 无关 | **有** |
| 硬件改动 | 22KB buffer + 调度器 | 记分牌 + CTA 调度器 | 已有 | **零（原型）/ 小（优化）** |
| 依赖图物化 | 需要，O(M·deg) | 需要 | 不需要 | **不需要**（在 CTA 寄存器里） |
| 失败行为 | 全连接 barrier | — | — | **逐级降级，下界=现状** |
| 可实测 | 仅模拟器 | 仅模拟器 | 真机 | **真机** |

最本质的差异是**依赖信息的所有权**：
BlockMaestro 由调度器集中持有一张全局依赖图；本设计让每个消费者 CTA **分布式地持有自己那一份**，
存在寄存器里，用完即弃。这消除了图的存储、传输和硬件缓冲，
代价是每个 CTA 要花 prologue 的时间自己算——而那段时间本来就是空等的。

---

## 15. 待解决的开放问题

1. 生产者 CTA 完成顺序在实际调度器（GTO）下的单调性有多强？若接近单调，
   `[min,max]` 区间等待可以简化为"等待完成计数器 ≥ p_hi"，省掉逐位轮询。**需实测。**
2. inspector 的 shared memory 缓存策略：`colIdx` 全缓存 vs 只缓存 min/max，
   在不同 nnz/CTA 下的最优点在哪。
3. 多级依赖链（K1→K2→K3）下，K3 的 inspector 需要 K2 的写集合映射——
   K2 若是 gather-affine 形态则可得，需要验证链式传播是否稳定。
4. 与 `cudaLaunchAttributeProgrammaticEvent`（跨 stream PDL）的组合行为，
   参见 `../../跨stream_PDL调研/`。
5. 变体 C 的反向登记表在什么规模下比轮询更划算。
