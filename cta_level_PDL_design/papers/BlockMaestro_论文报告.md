# BlockMaestro 论文阅读报告

**标题**：BlockMaestro: Enabling Programmer-Transparent Task-based Execution in GPU Systems
**作者**：AmirAli Abdolrashidi, Hodjat Asghari Esfeden, Ali Jahanshahi, Kaustubh Singh, Nael Abu-Ghazaleh, Daniel Wong（University of California, Riverside）
**会议**：ISCA 2021（第 48 届 ACM/IEEE 国际计算机体系结构研讨会），DOI: 10.1109/ISCA52012.2021.00034

---

## 一、一句话概括

BlockMaestro 是一套软硬件协同方案：通过**命令队列重排 + kernel 启动时（JIT）静态分析 + 运行时硬件依赖解析**，让现有的 CUDA/HIP 程序在**不改一行代码**的前提下，获得任务化（task-based）执行的收益——既隐藏 kernel launch 开销，又让下游 kernel 的 thread block 在其数据依赖满足后立即开始执行。在数据依赖型 benchmark 上平均加速 **51.76%**（最高 2.92 倍），硬件开销约 22KB。

---

## 二、问题背景与动机

现代 GPU 负载（CNN 推理/训练、stencil 计算、稀疏求解等）有两个共同特征：

1. **kernel 数量巨大**：一个应用动辄几百上千次 kernel launch，每次 launch 开销 5–30 μs，累积起来非常可观。
2. **kernel 之间存在数据依赖**：上一层的输出是下一层的输入。这种依赖在 SIMT 模型里只能用 kernel 边界这种**粗粒度隐式 barrier** 来表达。

由此产生两类浪费：

- **launch 开销**直接串在关键路径上；
- **依赖停顿（dependency stall）**：即使 `K2` 的某些 TB 所依赖的 `K1` 的 TB 已经完成，这些 TB 也必须等 `K1` **整个** kernel 结束才能被调度，导致 GPU 利用率不足。

现有的任务化执行模型（AMD ATMI、CUDA Graphs、OpenMP Tasks、Wireframe、Juggler 等）能解决这些问题，但**都要求程序员把应用重写成私有的任务编程模型**，需要领域知识把算法拆解成任务图，移植成本高。

---

## 三、核心思想："Thread Blocks as Tasks"

论文先把已有工作分成两类范式，再提出第三类：

| 范式 | 代表 | 优点 | 缺点 |
|---|---|---|---|
| Tasks as Kernels | CUDA Graphs、AMD ATMI | 合并 launch，降低启动开销 | 无法利用 kernel 间的细粒度依赖，TB 仍被 kernel 边界卡住 |
| Tasks as Thread Blocks | Wireframe、Juggler | 细粒度依赖解析 + persistent kernel，几乎无 launch 开销 | 需要程序员显式构造任务图，运行时管理开销 |
| **Thread Blocks as Tasks**（本文） | BlockMaestro | 程序员完全无感知 | 只能处理静态可分析的访存模式 |

关键洞察：**SIMT 模型里的 thread block 本身就是任务**——它的输入输出通过 kernel launch 参数中的 global memory 指针显式定义。因此不需要程序员定义任务图，而是可以**从已有的 kernel 代码里自动把任务图反推出来**。

进一步地，相邻两个 kernel（父 kernel `Kp` 和子 kernel `Kc`）之间的 TB 级依赖可以表示成一个**二分图（bipartite graph）**；整个应用就是一串二分图的级联，等价于一个被分解的任务图。

---

## 四、三大技术组件

### 1. 依赖识别（kernel 启动时的 JIT 静态分析）

**kernel 间依赖**：global memory 区域都通过 `cudaMalloc` 分配、基指针必须作为参数传给 kernel，所以从命令队列里比对各 kernel/memcpy 的指针参数即可判定 kernel 级依赖。

**强制按序完成（in-order completion）**：对于非线性依赖（如 K1→{K2,K3}→K4），若允许乱序完成，每个子 kernel 就要追踪任意多个父 kernel，不可扩展。BlockMaestro 允许 kernel **乱序执行但强制按序完成**——K3 即使先跑完也要等 K2 完成才标记完成。这样任何跨越多代的依赖都变成隐式满足，**依赖追踪只需限制在相邻 kernel 对之间**。代价是牺牲了一部分重叠机会。

**TB 级依赖（RAW）**：核心是识别每个 TB 实际触碰的数组下标范围。CUDA 程序员本来就用 `threadIdx / blockIdx / blockDim` 显式写出了线程到数据的映射，例如 `A + 4*(threadIdx.x + blockDim.x*blockIdx.x)`。算法（基于 GPGPU-Sim 的 PTX parser 实现）如下：

1. 找出 kernel 中所有 global load/store 指令；
2. 对每条指令的源操作数在 CFG 上做**反向数据流追踪**，回溯地址计算链；
3. 若地址来源于另一次 load（即间接访存 `A[B[i]]`、指针追逐），**保守地终止分析**，认为整个 kernel 依赖于前一个 kernel；
4. 否则所有源操作数都来自 kernel 启动时已知的变量，做**值域分析（value range analysis）**，得到每个 TB 的读集合 `L_K` 和写集合 `S_K`；
5. **`L_K ∩ S_{K-1}`** 即为 TB 级 RAW 依赖，构成二分图。

**为什么必须在 JIT 阶段而不是编译期做？** 因为 grid size 取决于输入数据规模，`blockDim` 和 `blockIdx` 的取值范围只有在 kernel launch 时才确定，编译期（CUDA→PTX）无法做值域分析。而 PTX→SASS 的 JIT 阶段恰好具备这些信息。这部分分析开销被 kernel 预启动技术掩盖，不在关键路径上。

**局限**：只处理静态可分析的访存；间接访存、指针追逐无能为力（退化为保守的全连接依赖）。Unified Memory 可以支持，因为 `cudaMallocManaged` 同样给出了地址范围。

### 2. Kernel 预启动（pre-launching）

目标是在前一个 kernel 还在跑的时候就把后续 kernel 发射出去，把 launch 开销从关键路径上挪走。需要解决三个障碍：

- **阻塞式 API**：`cudaMalloc`/`cudaMemcpy` 默认同步，会卡住 host，导致后续 kernel 迟迟进不了命令队列。由于依赖已由硬件保证，BlockMaestro 把这些调用当作非阻塞处理；只有**与 host 存在 RAW 冒险**的调用（如 device-to-host 的 `cudaMemcpy`）才必须真正同步。`cudaDeviceSynchronize` 在无 host RAW 的前提下同样可以跳过。
- **API 命令重排序**：分析命令队列中 API 之间的真实数据依赖，在保序前提下把 kernel launch 尽量往前挪、彼此靠拢，最大化预启动窗口（论文 Figure 5）。
- **同一命令队列并发多 kernel**：基线中一个 stream 同时只能跑一个 kernel。BlockMaestro 借助类似 NVIDIA Hyper-Q 的机制允许同一 stream 内多 kernel 并发。经验上**并发 2–3 个 kernel 就足以完全掩盖 launch 开销**。

### 3. 运行时依赖解析（硬件支持）

TB 调度器中新增两个结构：

- **Dependency List Buffer**：以父 kernel 的 TB ID 为索引，记录它的子 TB 列表（即二分图的邻接表）；
- **Parent Counter Buffer**：记录每个子 TB 尚未满足的父依赖计数。

工作流程：TB 被调度时，其依赖表项被预取到 buffer（此时还用不到，因此不在关键路径上）；TB 完成时，遍历其子 TB 并递减对应的 parent counter；**counter 归零即表示该子 TB 就绪**，可以被调度。完整的依赖表和计数器存在 global memory 中，buffer 只是缓存活跃部分。

**多 kernel 扩展**：在 TB ID 上追加 2 bit 作为 kernel 标识（即 kernel ID 的低 2 位），即可同时追踪 4 个 kernel。

**调度策略**：
- *Producer priority*（默认）：优先调度生产者 kernel 的 TB，消费者 TB 等生产者全部调度完才排队；
- *Consumer priority*：优先消费者 TB，允许更多"run-ahead"，并发度更高。

论文论证了**不会死锁**：即使消费者 TB 抢占资源，它们最终会因依赖未满足而让出，生产者总能被调度。

### 4. 依赖图的编码压缩

实际应用中依赖模式很少是任意的，论文归纳出六种常见模式并给出各自的存储开销（N 个父 TB、M 个子 TB）：

| 模式 | 存储开销 |
|---|---|
| 全连接（fully connected） | O(1)（不编码为 O(MN)） |
| n-组全连接 | O(M+N) |
| 1-to-1 | O(N) |
| 1-to-n | O(M+N) |
| n-to-1 | O(N) |
| 重叠（overlapped） | O(N + M·deg_max) |
| 独立 | O(1) |

全连接是最坏情况，功能上等价于 kernel 之间的同步 barrier，此时只能靠 launch 隐藏获益，无法靠重叠获益。若依赖度过大导致收益微薄，硬件可以直接退化为全连接处理。

---

## 五、评估

**平台**：改造版 GPGPU-Sim v3.2.2，Titan X Pascal 配置（28 SM，每 SM 最多 32 TB），GTO warp 调度，kernel launch 开销取 5 μs。

**Benchmark**：来自 Rodinia、PolyBench、SHOC、Tango 的 12 个多 kernel 应用——3MM、AlexNet(22 kernels)、BICG、FDTD-2D(24)、FFT(60)、GAUSSIAN(510)、GRAMSCHM(192)、HS、LUD(46)、MVT、NW(255)、PATH。

**主要结果**：

- **加速比**：平均 51.76%，最高 2.92×；把预启动 kernel 数增加到 3 个（consumer priority）时几何平均加速可达 **80.28%**，再增加则收益递减。
- **收益来源分化**：
  - GAUSSIAN、GRAMSCHM 这类 kernel 数量极多、每个 kernel 极短的应用，**光靠 kernel 预启动**就有显著收益（launch 开销是主要瓶颈）；
  - 3MM、BICG、FDTD 这类依赖易满足甚至 kernel 相互独立的应用，主要收益来自**细粒度依赖解析**；
  - AlexNet 这类计算密集、单 kernel 内 TB 数量巨大的应用，launch 开销占比小，收益有限（6.9%），但 TB 并发度仍有提升。
- **TB 并发度**：相对基线有明显提升（Figure 10）。
- **依赖停顿**：大部分应用的 TB 停顿时间显著下降；BICG 和 MVT 因为两个 kernel 本身独立可完全并行，停顿降幅最大。

**开销分析**：

- **互连度敏感性**：用 VectorAdd 微基准人为增加 TB 依赖度，发现**依赖度超过 32 之后收益迅速消失**，退化到全连接的水平；同时工作规模越大（TB 数越多）收益越小，到 2048 TB 时收益归零——因为一个 kernel 就吃满了资源，没有 run-ahead 空间。
- **面积开销**：dependency list buffer 和 parent counter buffer 各 28×32 = 896 项，每项存 4 个子 TB ID（32 bit），索引 32+2 bit，parent counter 6 bit，**合计约 22 KB** 存储外加控制逻辑。
- **访存开销**：平均仅 **1.36%** 的额外内存请求。
- **依赖图存储**：经模式编码后平均减少 **34.7%**（AlexNet 压到 0.012，GAUSSIAN 压到 1.77e-4；FDTD-2D、FFT、HS、NW、PATH 无法压缩）。

**对比实验**（6 个 4K 任务的 wavefront 依赖应用，归一化到 CDP）：

| 方案 | 加速比 | 说明 |
|---|---|---|
| CUDA Dynamic Parallelism | 1.0（基准） | Tasks as Kernels |
| Wireframe | 1.368 | Tasks as TBs，需程序员改写 |
| BlockMaestro（producer priority） | 1.058 | |
| **BlockMaestro（consumer priority）** | **2.0** | 依赖状态存 global memory，不受硬件缓冲容量限制 |

论文指出 Wireframe 受限于固定大小的硬件任务管理缓冲（pending update buffer），能同时 run-ahead 的任务量有上限；BlockMaestro 把状态放在 global memory，不受此限制，代价是略高的访存流量。**结论：无需程序员干预即可超过需要改写代码的 Wireframe。**

---

## 六、局限与未尽之处

论文自己承认的：

1. **只支持静态可分析的访存模式**。`A[B[i]]`、指针追逐、图遍历等运行时才确定的地址无法处理，会保守退化为 kernel 级全连接依赖。
2. **只能提取静态任务图**，定位与 CUDA Graph 相当；输入相关的动态任务图（如稀疏求解器中依赖矩阵结构的任务图）留作未来工作。
3. **强制 in-order kernel completion** 牺牲了部分 kernel 重叠机会，换取依赖追踪的可扩展性。
4. 全连接依赖模式下退化为普通 barrier，只剩 launch 隐藏收益。
5. 评估基于模拟器（GPGPU-Sim + Pascal 配置），launch 开销用固定 5 μs 建模，未在真实硬件上验证。
6. 主要针对**单默认 stream** 的应用（这类应用 launch 开销和利用率问题最严重）；对 CUDA Streams 应用的支持只做了原理性讨论。

---

## 七、与 PDL / CLC 方向的关联

这篇工作与本仓库中 `cta_level_PDL_design`、`cross_stream_PDL_survey` 的主题高度相关，可以作为学术侧的对照参考：

- **相同的问题定义**：都在攻击"kernel 边界作为粗粒度 barrier 导致 launch 开销 + 依赖停顿"这个问题。NVIDIA 的 PDL（Programmatic Dependent Launch）本质上就是论文中的 *kernel pre-launching*——让后继 kernel 提前启动、在前驱 kernel 尾部之前完成 prologue，然后在 `cudaGridDependencySynchronize()` 处等待。
- **BlockMaestro 更激进的地方**：PDL 的同步粒度仍然是 grid 级（`cudaGridDependencySynchronize` 是整 grid 等前驱整 grid），而 BlockMaestro 做的是 **TB 级（CTA 级）的点对点依赖解析**——正好对应 `cta_level_PDL_design` 想要达到的目标。论文给出的二分图表示、dependency list + parent counter 的硬件结构、以及六种依赖模式的编码压缩方案，是 CTA 级 PDL 设计可以直接借鉴的机制。
- **可迁移的关键结论**：
  - 并发 2–3 个 kernel 即可完全掩盖 launch 开销，再多收益递减；
  - **依赖度超过 32 后细粒度解析基本无收益**，此时直接退化为 grid 级同步即可——这对判断"什么场景值得做 CTA 级 PDL"是很有价值的量化边界；
  - 当单个 kernel 的 TB 数足以填满 GPU 时（约 2048 TB），run-ahead 空间消失，收益归零。
- **依赖提取路径的差异**：BlockMaestro 依赖 JIT 期的 PTX 值域分析自动提取依赖，属于"程序员透明"路线；而 PDL/CLC 走的是程序员显式标注路线（`cudaTriggerProgrammaticLaunchCompletion` / `cudaGridDependencySynchronize`）。如果要做 CTA 级 PDL 的接口设计，这两条路线的取舍（自动提取的适用范围 vs. 显式标注的表达能力）是核心决策点。

---

## 八、总体评价

**贡献扎实的一点**：它把"任务化执行"的收益和"任务化编程模型"的成本解耦开了。此前的结论似乎是"想要细粒度任务收益就必须重写代码"，这篇论文证明了在相当一部分规整负载上，编译器 + 硬件可以自动完成这件事，而且效果能超过需要手工改写的 Wireframe。

**最需要注意的边界**：全部收益建立在"访存下标能被静态值域分析算出来"这个前提上。规整的 stencil、GEMM、CNN 层间依赖满足这个条件，但图计算、稀疏求解、动态数据结构这些最需要细粒度任务调度的负载恰恰不满足。论文对此是诚实的（明确列为 future work），但这确实限制了方案的适用面。

---

## 九、开源情况

**没有开源。** BlockMaestro 和作者前作 Wireframe 都没有公开代码仓库，也没有参加 ISCA 2021 的 artifact evaluation（该届 AE 为自愿参加），论文正文和致谢中均无代码发布链接。

第一作者 AmirAli Abdolrashidi 的个人主页上，这个博士课题（"Improving Execution of GPU Applications with Inter-Kernel Data Dependencies"，2018.12–2021.8）只有文字描述，说明是通过修改 GPGPU-Sim（C++）评估的。该页面还透露了一条论文中没有出现的信息：他们**尝试过用机器学习模型（Python）预测 kernel 间的数据依赖**，推测是应对间接访存局限的后续尝试。

论文全文的免费获取渠道：
- 作者主页 PDF：<https://www.cs.ucr.edu/~ajaha004/files/BlockMaestro.pdf>
- NSF 公共访问库：<https://par.nsf.gov/servlets/purl/10298555>（受 NSF CCF-1815643、CNS-1955650、CNS-2047521 资助，有开放获取要求，但只覆盖论文不覆盖代码）

**复现所需工作量**：底座 [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) 是公开的（论文用 v3.2.2，当前最新 v4.2.1），benchmark（Rodinia / PolyBench-GPU / SHOC / Tango）也全部公开。需要自己实现三块：

1. PTX 静态分析（Algorithm 1）——GPGPU-Sim 自带 PTX parser，论文算法描述较详细，工作量可控；
2. TB 调度器改造——dependency list buffer + parent counter buffer + 两种调度策略，论文 Figure 6/7 状态机描述较清楚；
3. 命令队列层——kernel 预启动、API 重排序、同 stream 多 kernel 并发。这块最麻烦，涉及 CUDA runtime 模拟层改动，且论文对重排序算法描述最含糊（只有 Figure 5 示意，无具体算法）。

**最大的复现障碍是 5 μs launch 开销的建模方式**。GPGPU-Sim 默认不建模 host 侧 kernel launch 开销，论文是引用文献 [27] 的实测值人为注入的。BlockMaestro 的收益有相当一部分（尤其 GAUSSIAN、GRAMSCHM 这类 kernel 数量极多的负载）直接来自消除这个开销，因此该参数的取值基本决定了结果量级，做对比实验时需要特别谨慎。

---

## 附录：延伸讨论 Q&A

以下是围绕论文若干技术细节的展开讨论，主要澄清了论文中表述较简略、但对 CTA 级 PDL 设计有直接参考价值的部分。

### Q1. "只能处理静态可分析的访存模式"该怎么理解？

这里的"静态"不等于"编译期已知"，准确含义是**在 kernel 真正执行之前、仅凭 kernel launch 参数就能算出来的地址**。论文明确列举的可用信息只有三类：device 变量地址、立即数、kernel 参数（含 `gridDim`/`blockDim`）。判据是：一条 global load/store 的地址计算链，能否一路回溯到这三类源头而不经过任何一次访存。

可分析的形态是 `blockIdx`/`threadIdx` 的仿射函数，如 `C[i] = A[i] + B[i]`、`D[i*N + j]`。不可分析的是 `A[B[i]]`、`node = node->next`、CSR 稀疏遍历 `x[colIdx[e]]` 这类**地址值来源于另一次 global load** 的情况。

Algorithm 1 第 7–9 行是判定点：反向追踪时一旦撞上 global load 立即 END。失败后不报错，而是**保守回退为 fully-connected**（Table I 模式 1），功能上等价于 kernel 间插了个 barrier，正确性不受影响但只剩 launch 隐藏的收益。

注意这个 END 是 **kernel 级别**的，不是单指令级别——论文原文"we terminate and conservatively assume the entire kernel is dependent on the previous kernel"。

### Q2. 为什么论文做得这么保守？

**根本原因：这是一套没有回滚机制的硬件依赖执行机制。** TB 调度器完全信任二分图，父 TB 完成即递减 counter、归零即放行，整条路径上没有运行时校验、没有地址比对、没有冲突检测。漏判一条 RAW 边的后果是**静默数据损坏**。不像 TLS/事务内存有回滚兜底——GPU 上回滚一个 TB 的代价极高（寄存器、shared memory、已发出的写全要撤销），论文根本没有这个选项。在"必须 sound + 无法恢复"约束下，未知信息只能假设指向任意地址，读集合 = 全集，交集必然非空，结果必然是全连接。

**"整个 kernel 作废"没有看起来那么粗暴**：SIMT 下所有 TB 执行同一份代码，只要 kernel 里存在这条间接 load，每个 TB 的读集合都是未知的，逐 TB 分析最终仍是全连接图。所以 END 是一个**等价的提前退出**，不损失精度。而且全连接在 Table I 里存储开销是 O(1)，恰好是最省的回退路径。

**更激进的做法各自要付的代价**：inspector-executor（探测 kernel）会引入额外 launch，与消除 launch 开销的动机自相矛盾；运行时地址监控（Bloom filter 签名）面积功耗远超 22KB；推测+回滚在 GPU 上不可行；程序员标注违背 "programmer-transparent" 的立身之本。

**一层现实考量**：论文自己的 Figure 12 显示依赖度超过 32 收益就基本消失，而间接访存主导的负载往往依赖散乱、度数高。这是部分辩护，但不完全——带状矩阵 LU 这类稀疏且低度数的场景本来是有收益的。

**可改进之处（论文的空白）**：它是"要么全静态、要么全 barrier"的二值选择，中间地带空着。两个成本不高的中间态：(a) **按数组/基指针分别判定**——`out[i] = f(A[i], B[idx[i]])` 中 `A` 完全可分析，只需对涉及 `B` 的依赖保守处理；(b) 间接下标的**取值范围受数组长度约束**，保守边界应是"依赖 `A` 全部"而非"依赖前一 kernel 全部"，若前一 kernel 不写 `A` 则依赖为空。

### Q3. 关于三类可用信息源

**device 变量地址**指文件作用域的 `__device__` / `__constant__` / `__managed__` 变量——不经 `cudaMalloc` 动态分配，而是随 module 静态分配。地址是编译期符号，模块加载时解析为固定地址。在 PTX 中表现为 `.global` 符号定义，访问时地址操作数直接是符号名（`cvta.global.u64 %rd1, g_buf`）。按 Algorithm 1 第 11–13 行，符号和立即数一样不是本地寄存器，追踪链在此干净终止。

**论文单独列出它的原因**：它是三类源头中唯一**不出现在 kernel 参数列表里**的。论文识别 kernel 级依赖靠比对命令队列中各 kernel 的指针参数，而 `__device__` 变量按符号访问、launch 参数里看不见。只看参数列表会把共享同一 `__device__` 数组的两个 kernel 误判为无依赖——这是**正确性问题**而非性能问题。

三类源头在 PTX 中的形态与确定时机：

| 源头 | PTX 形态 | 何时确定 |
|---|---|---|
| device 变量地址 | `.global` 符号 / `cvta.global` | 模块加载时 |
| 立即数 | 指令内编码的常量偏移、步长 | 编译期 |
| kernel 参数 | `ld.param.u64 %rd1, [k_param_0]` | kernel launch 时 |

注意 Algorithm 1 第 7 行的终止条件特意限定为 "**global** load"——kernel 参数也是通过 `ld.param` 读的，但 param space 的值在 launch 时已固定写入，不触发保守回退。这个限定词是算法能工作的前提之一。

### Q4. `__device__` / `__constant__` / `__managed__` / `__shared__` 的区别

前三个**都在 global memory（显存）**，没有一个在 shared memory；`__shared__` 才是 shared memory。

| 限定符 | 物理位置 | PTX 空间 | 作用域 | 生命周期 | device 端 | host 访问 |
|---|---|---|---|---|---|---|
| `__device__` | 显存 | `.global` | module 内全局 | 应用全程 | 读写 | `cudaMemcpyToSymbol` |
| `__constant__` | 显存（走 constant cache） | `.const` | 同上 | 同上 | **只读** | `cudaMemcpyToSymbol` |
| `__managed__` | 统一内存，页面可迁移 | `.global` | 同上 | 同上 | 读写 | **直接按名访问** |
| `__shared__` | 片上 SRAM | `.shared` | 单个 thread block | block 存活期 | 读写 | 不可访问 |

`__constant__` 物理上仍在显存，但走独立只读路径 + per-SM constant cache，warp 内所有线程读同一地址时可广播（读不同地址则串行化，反而更慢），整个 module 上限 64KB。kernel 参数在 NVIDIA 硬件上也经由 constant bank 传递。`__managed__` 通常写作 `__device__ __managed__`，对应动态版本 `cudaMallocManaged`。

**对 BlockMaestro 的意义**：kernel 间依赖只可能通过 global memory 传递。`__shared__` 随 block 消亡，跨不了 kernel 边界，分析时可忽略；`__constant__` 在 device 端只读，两 kernel 间不可能形成 RAW，同样不用管；`__device__` 和 `__managed__` 是可读写的持久 global 数据，**必须纳入分析**。论文对统一内存有专门说明：分配时就知道要监控哪段地址范围，kernel 内访存形式与普通 global 完全一样，页面迁移是驱动层的事，与地址计算无关。

### Q5. CUDA 的 JIT 阶段是什么？论文的说法与标准 CUDA 有何差异？

**两段式编译**：`.cu` → (NVVM) → **PTX**（虚拟 ISA `compute_XX`，稳定、向前兼容）→ (ptxas) → **SASS**（真实机器码 `sm_XX`，每代架构都变，无正式公开文档，只能用 `cuobjdump -sass` / `nvdisasm` 查看）。产物是 fatbinary，可同时容纳多架构 SASS 和 PTX，由 `-gencode arch=compute_80,code=sm_80` / `code=compute_80` 控制装什么。

**JIT 触发时机**：当 fatbinary 里没有当前 GPU 可用的 SASS 时，driver 调用内置 PTX 编译器现场编译。准确时机是 **module 加载时**（Driver API 的 `cuModuleLoad`；Runtime API 下 CUDA 11.7+ 默认 lazy module loading，推迟到首次使用）。关键推论：**一个 module 只 JIT 一次，结果复用**——同一 kernel 用不同 grid/block 配置 launch 一万次，不会重编一万次。

**存在意义是向前兼容**：SASS 每代都变，只发布 SASS 的程序在新卡上跑不了；PTX 作为稳定虚拟 ISA 可被新 driver 编译到新架构。代价是首次运行有编译延迟，为此有磁盘缓存 `~/.nv/ComputeCache`（`CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE` / `CUDA_CACHE_DISABLE`，调试用 `CUDA_FORCE_PTX_JIT=1`）。注意与 **NVRTC** 区分：NVRTC 是运行时把**源码**编译成 PTX，处在流水线的不同位置。

**论文说法与标准 CUDA 的差异**：论文强调分析"只能在 kernel-launch-time 的 JIT 阶段做"，理由（值域分析需要启动配置）是对的，但**标准 CUDA 的 PTX→SASS JIT 发生在 module 加载时，那时启动配置还不存在，且只做一次**。所以论文提出的并非"复用现有 JIT 顺便做分析"，而是**在 launch 路径上新增一个分析步骤**。他们的实现方式印证了这点——用 GPGPU-Sim 内置的 PTX parser，模拟器里所有信息触手可及，不受真实 driver 流水线约束。

真实系统落地要么改 driver 在每次 launch 插入分析，要么按配置缓存复用。论文用"分析开销被预启动掩盖"一句带过，未量化——考虑到 Algorithm 1 第 19–21 行是**逐线程枚举地址**（`for all t ∈ Threads`），几十万线程的 kernel 代价恐怕不小，这是评估里较薄弱的一环。

### Q6. 同一份 PTX 会被 JIT 成多份 SASS 吗？

**会，有多条分叉轴**：目标架构不同（多 GPU 机器上同一进程内即可发生）；JIT 选项不同（`CU_JIT_MAX_REGISTERS`、`CU_JIT_OPTIMIZATION_LEVEL`、`CU_JIT_THREADS_PER_BLOCK` 都实打实改变代码生成）；context / 模块实例不同；链接单元不同（`cuLink*` 跨单元内联结果不同）；driver 版本不同（ptxas 本身在演进，这也是 ComputeCache 缓存键含 driver 版本、升级驱动后全部失效的原因）。

**但唯独不会随 launch 配置分叉**。`__launch_bounds__` / `-maxrregcount` 看似"按配置定制"，实际在 CUDA→PTX 阶段就编码进 PTX（`.maxntid` / `.minnctapersm`），属于 PTX 的一部分。真正的 `<<<grid, block>>>` 参数 driver 完全不会拿来重编译——一个 module 加载时 JIT 一次，之后无论怎么 launch 都执行同一份 SASS。

**推论**：分析应挂在 **PTX** 上而非 SASS 上。PTX 稳定（换架构、换驱动、换寄存器上限都不变），且有充分的类型和空间信息（`.global` / `.const` / `.param` 一目了然）；SASS 层每个变体都要重做，还要面对无公开文档的 ISA。论文选 PTX 层是对的。

相应地两套缓存键完全不同、无交集，不能复用 driver 的 JIT 缓存：
- JIT 缓存键：目标架构 + 编译选项 + driver 版本
- 依赖图缓存键：**context / module 实例 + 函数 + gridDim + blockDim + 各指针实参值**

### Q7. launch 配置在 PTX 层能知道吗？module 实例是什么？

**launch 配置既不在 PTX 里也不在 SASS 里**——它不是"编译到哪一层才知道"的问题，而是压根不来自代码。两层都用**特殊寄存器**表达：PTX 的 `%tid.x` / `%ctaid.x` / `%ntid.x` / `%nctaid.x`，SASS 的 `S2R R0, SR_TID.X` 等（blockDim/gridDim 通常从 constant bank `c[0x0][...]` 读，看起来像常量，实为 driver 在 dispatch 时写入，每次 launch 都可能不同）。唯一例外 `__launch_bounds__` 是**上界承诺**而非实际值。

**值域分析需要三份信息拼合**：
1. **PTX 提供符号表达式**——`addr = param0 + 4*(%tid.x + %ntid.x * %ctaid.x)`，module 加载时即有；
2. **launch API 提供代入值**——`<<<128,256>>>` 给出 `%ntid.x = 256`、`%ctaid.x ∈ [0,128)`；
3. **kernel 实参提供基地址**——`param0` 的具体指针值。

所以"必须在 kernel-launch-time 做"的**准确理由不是"那时才 JIT"，而是"那时 launch 配置才存在"**。SASS 不提供任何 PTX 没有的信息，反而因经过寄存器分配和优化使数据流更难追踪。

**module 实例**大致相当于 GPU 上的动态链接库，内容是 kernel 代码 + `__device__`/`__constant__` 变量的存储 + texture/surface 引用。CUDA **context** 相当于 GPU 上的进程地址空间。同一份 PTX 在不同 context 加载意味着：各自独立 JIT 一份 SASS；`__device__` 变量各有独立存储；**`cuModuleGetGlobal` 拿到的符号地址在不同实例里不同**。

**module 实例不包含 launch 配置**。module 装的是代码和静态数据，launch 配置是 launch 操作的参数，两者生命周期正交——同一个 `CUfunction` 可被 launch 一万次、每次配置不同而 module 纹丝不动。

这也是依赖图缓存键必须含 context 维度的原因：`__device__` 变量的符号地址是 per-instance 的，跨 context 复用依赖图会拿到错误地址范围，而这类错误在本机制中表现为**静默数据损坏**。

### Q8. 既然写完代码就知道 launch 配置，为什么非要等到 launch 前才分析？

要分成两个问题看。

**(a) 写完代码后能否得知 n —— 不能。** n 来自**输入**而非代码：`./gaussian -s 1024` 与 `-s 8192` 是同一个二进制、同一份代码，却对应完全不同的 n、循环轮数、grid 尺寸序列和依赖图。编译器面对的是代码，n 属于数据，两者在时间和信息上都是分离的。

**(b) n 确定后能否提前批量算完 510 次 —— 很大程度上可以。** "运行时"是很宽的区间，包含"程序启动、读完输入、进入循环之前"这个时刻，那时 n 已知。对 GAUSSIAN，第 t 轮 grid size 是 t 和 n 的闭式函数、循环轮数是 n-1，理论上可以进循环前一次性算完全部依赖图。

**真正的障碍不在"能不能算"，而在"BlockMaestro 的架构看不见 host 代码"。** 这套机制的信息入口是**命令队列**——逐个观察到达的 API 调用，对 host 程序结构一无所知（不知道有循环、不知道跑多少轮、不知道 grid 怎么算出来的）。要提前批量计算就必须**预测命令队列未来的内容**，即静态分析 host 二进制的循环结构，那是 host 侧程序分析，不是这套 PTX 分析框架的能力范围。

即使有该能力，仍有推不动的情况：`while (!converged)` 的迭代求解器（轮数取决于计算结果）、`if (residual > eps)` 的数据相关分支。GAUSSIAN 是"规整循环"的好例子但不可推广。

**更好的解法不是批量，是把每次的成本降下来**——见下一节。

### Q9. 论文最值得改进的一点：分析应当两阶段切分

Algorithm 1 的两个阶段对 launch 配置的依赖程度**完全不同**：

- **第 2–18 行（反向数据流追踪）完全不依赖 launch 配置**。它做的是找出所有 global load/store、沿 CFG 回溯地址计算链、判断源头是否落在三类可用信息里。这纯粹是 PTX 代码结构的性质。**包括"遇到间接访存就 bail out"的判定也是编译期就能确定的**——`A[B[i]]` 这个结构在编译期看得见。
- **第 19–21 行（值域分析）才真正需要 launch 配置**，要代入 `%ntid`、`%nctaid` 的具体值枚举地址集合。

因此合理的实现应当切成两半：

| 阶段 | 时机 | 产出 | 成本 |
|---|---|---|---|
| 结构分析 | 编译期（CUDA→PTX） | "本 kernel 可否静态分析" + 每条访存的**符号地址表达式** + 依赖**模式模板**，可写入 fatbinary 元数据 | 一次性 |
| 实例化 | kernel launch 时 | 代入 gridDim/blockDim/标量参数，得到具体依赖图参数 | 几次算术，接近 O(1) |

**论文自己的 Table I 已经暴露了关键事实：实际依赖图几乎总是参数化的规则模式**（1-to-1、1-to-n、n-to-1、n-组全连接），而不是任意图。既然如此，就不该用"逐线程枚举地址 + 集合求交"的暴力做法，而应当"编译期识别模式模板、运行时实例化参数"。

在这个方案下每次 launch 的分析成本可以忽略，Q8 中"要不要提前批量算"的问题自然消失；同时它保留了对 `while (!converged)` 这类不可预测控制流的支持，因为不需要预测未来，只需在每个 kernel 到达时便宜地实例化。

这是论文最可惜的地方：它在 Table I 里发现了"依赖模式是少数几种参数化模板"，却没有把这个洞察反过来用到分析算法本身上。

### Q10. NVIDIA Hyper-Q 是什么？与本文的关系

**解决的问题**：Kepler 之前（Fermi），host 到 GPU 工作分发器之间只有**一条硬件工作队列**，多个 CUDA stream 被复用到这一条队列上按 issue 顺序排列。结果是逻辑上独立的 kernel 因为在队列里前后相邻而被串行化，即**伪依赖**。当时程序员必须靠 launch 顺序规避：深度优先 issue（A1 A2 A3 B1 B2 B3）两 stream 基本串行，广度优先（A1 B1 A2 B2 A3 B3）才有像样的并发——一个硬件实现细节泄漏到了编程模型里。

**Hyper-Q 的做法**（Kepler GK110，CC 3.5，2012）：把队列扩展为 **32 条硬件管理的工作队列（connection）**，配合 Grid Management Unit 调度待分发 grid。不同 stream 映射到不同硬件队列，launch 顺序不再影响并发度。两个实用细节：实际连接数由 **`CUDA_DEVICE_MAX_CONNECTIONS`** 控制，**默认只有 8**、最大 32，超出会重新出现伪依赖（隐蔽的性能坑）；Hyper-Q 也是 **MPS** 的硬件基础。

**关键：Hyper-Q 只管 stream 之间，不管 stream 内部。** CUDA 规定同 stream 内操作按 issue 顺序串行，这是**语义保证**而非硬件限制。所以论文原话是 "...enables multiple kernel commands from different streams (**with our modification, from the same stream**)"——让多 kernel 同时在飞的**硬件能力**已经现成，BlockMaestro 只需**放宽 stream 内的顺序语义**，正确性由自己的 TB 级依赖解析兜底。这解释了为什么这部分修改的工程量比听上去小，也支撑了 22KB 硬件开销的说法。

另需注意 Hyper-Q 解决的是"多 kernel 能否**被同时分发**"，分发后**如何共享 SM 资源**是另一层问题——论文致谢中提到用 Warped-Slicer 的实现作为并发 kernel 执行的基线，指的就是这一层。

**与 PDL 的关系**：BlockMaestro 在 2021 年提出的"同 stream 内多 kernel 并发 + 硬件保证依赖"，正是 NVIDIA 后来在 Hopper 上产品化的 **PDL** 所做的事——放宽 stream 内顺序语义，用同步原语保证正确性。区别在粒度：PDL 是 grid 级（整 grid 等整 grid），BlockMaestro 是 TB 级点对点。可以理解为 **NVIDIA 采纳了这个方向，但选了硬件代价低得多的粗粒度实现**，而 CTA 级 PDL 想补的正是中间那一段。
