# 参考文献索引

按**设计维度**检索。维度编号见 [`../docs/cta_pdl_design_space.md`](../docs/cta_pdl_design_space.md) §0.3。

**B300 状态**列的含义（详见设计空间报告 §1）：

- `有` — 提案已产品化，参考价值只剩动机与量化数据
- `部分` — 方向被采纳但粒度/形式不同，缝隙处仍有空间
- `无` — 未实现，机制本身仍值得借鉴
- `反` — "没走的路"，NVIDIA 选了相反方向

---

## 一、核心：kernel 间依赖执行

这一类直接对应本项目的问题定义。

| 文件 | 会议/年份 | 维度 | 核心机制 | B300 |
|---|---|---|---|---|
| `BlockMaestro_...pdf` | ISCA'21 | A1 A2 A3 A4 B1 B2 B3 B4 D1 E1 | JIT 期 PTX 值域分析提取 TB 级依赖二分图 + kernel 预启动 + 调度器 dependency list/parent counter 门控 | 贡献#1 `有`（=PDL）<br/>贡献#2#3 `无` |
| `Kim_PACT16_...pdf` | PACT'16 | A2 A3 B1 B3 B4 | CRCS 粗粒度引用计数记分牌保证正确性 + PPCS 流水感知 CTA 调度器 | `无` |
| `Wireframe_MICRO17_...pdf` | MICRO'17 | A1 A2 A3 A4 B1 B2 | DepLinks 程序员标注 + CSR 依赖图下发 + DATS 依赖感知 TB 调度器（2KB 硬件开销） | `无` |
| `Juggler_PPoPP18_...pdf` | PPoPP'18 | A1 A2 A4 B1 B3 B4 | OpenMP 4.5 task directives → 源到源变换建 DAG + 持久 worker TB 从分布式无锁队列取任务 | 取工部分 `部分`（CLC） |
| `ATA_PACT19_...pdf` | PACT'19 | A2 B1 B3 E1 | 自适应任务聚合，把细粒度任务合并成层次单元 + SET 调度，面向稀疏求解器 | `无` |
| `Whippletree_TOG14_...pdf` | TOG'14 | A1 A4 B1 | persistent thread + 工作队列的动态任务调度 | `部分`（CLC） |
| `Softshell_TOG12_...pdf` | TOG'12 | A4 B1 B4 | 持久线程动态调度 runtime，TB 边界或显式抢占点做调度 | `部分`（CLC） |
| `Tzeng_Computer12_...pdf` | Computer'12 | A1 A4 | GPU 端任务图 + 依赖解析，就绪任务入队 | `部分`（CLC） |

## 二、TB 调度与局部性（C1 / B4 的主要来源）

| 文件 | 会议/年份 | 维度 | 核心机制 | B300 |
|---|---|---|---|---|
| `IKRA_TACO20_...pdf` | TACO'20 | B4 C1 | 把消费 TB 调到生产 TB 所在核，利用 producer-consumer TB 关系提升 **L1/L2** 复用 + work stealing 平衡负载。**观察到 self-dependency（同 ID TB）占多数** | `无` |
| `PAVER_TACO21_...pdf` | TACO'21 | A2 B4 C1 | JIT 从 PTX 提取 TB 的 load 地址范围 → 建 locality 图 → METIS 分区 → 同分区调到同 SM。**技术手段与 BlockMaestro 高度重合（同为 UCR 组），但用于 locality 而非 dependency** | `无` |
| `OWL_ASPLOS13_...pdf` | ASPLOS'13 | B4 C1 | CTA 感知的 warp 调度，提升 on-chip 局部性 | `无` |
| `LocalityDescriptor_ISCA18_...pdf` | ISCA'18 | C1 | 跨层 locality 表达抽象——**"程序员如何把局部性告诉硬件"的接口设计参考** | `无` |
| `CTAClustering_ASPLOS17_...pdf` | ASPLOS'17 | B4 C1 | CTA 聚类调度强化 SM 内 cache 共享。**与 Hopper cluster 概念撞车，值得核对 NVIDIA 是否借鉴** | `部分`（cluster） |
| `Improving_GPGPU_Performance_via_Cache_Locality_Aware_TB_Scheduling.pdf` | CAL'17 | B4 C1 | 局部性感知的 TB 调度提升 L1 复用 | `无` |
| `Lee_HPCA14_...pdf` | HPCA'14 | B4 | 替代 TB 调度策略，把相邻 TB 绑到同 SM 以利用 inter-TB 局部性。**本批被引最多的 TB 调度工作** | `无` |
| `NeitherMoreNorLess_PACT13_...pdf` | PACT'13 | B3 B4 | 动态调整并发 TB 数以匹配资源上限，平衡 occupancy 与访存争用 | `无` |

## 三、并发 kernel 执行与资源分区（B3 / B4）

| 文件 | 会议/年份 | 维度 | 核心机制 | B300 |
|---|---|---|---|---|
| `Warped-Slicer.pdf` | ISCA'16 | B3 B4 | intra-SM slicing：解析模型（性能-TB 数曲线的 water-filling）决定多 kernel 在单 SM 内的资源划分。**BlockMaestro 用其实现作并发 kernel 基线** | `无` |
| `Gregg_HotPar12_...pdf` | HotPar'12 | B3 B4 | 并发 GPGPU kernel 的细粒度资源共享 | `无` |
| `ElasticKernels_ASPLOS13_...pdf` | ASPLOS'13 | B3 B4 | kernel 运行时调整资源占用，便于多 kernel 共享 SM | `无` |
| `Kernelet_TPDS14_...pdf` | TPDS'14 | B3 B4 | 大 kernel 动态切片 + 交错调度提高并发吞吐 | `无` |
| `Equalizer_MICRO14_...pdf` | MICRO'14 | B4 | 运行时监控并调节 CTA 数、频率等执行参数 | `无` |
| `Zorua_MICRO16_...pdf` | MICRO'16 | B3 B4 | SM 资源虚拟化 / over-subscription | `无` |
| `CKE_HPCA18_...pdf` | HPCA'18 | B3 B4 | 通过缓解 memory pipeline stall 加速并发 kernel 执行 | `无` |
| `The Case for GPGPU Spatial Multitasking.pdf` | HPCA'12 | B4 | SM 粒度的空间分区，每 kernel 分一组 SM | `有`（MIG/MPS） |
| `Chimera.pdf` | ASPLOS'15 | B2 B4 | 协作式抢占：**SM flushing（利用幂等性直接丢弃 TB）** + context switch + draining 三选一 | 抢占 `有`<br/>SM-flushing `反` |
| `Enabling_preemptive_multiprogramming_on_GPUs.pdf` | ISCA'14 | B2 B4 | GPU 抢占式多道程序：context switch 与 draining | `有`（Pascal 起） |

## 四、launch 开销与设备侧启动（D3 / E1）

| 文件 | 会议/年份 | 维度 | 核心机制 | B300 |
|---|---|---|---|---|
| `Wang_IISWC14_...pdf` | IISWC'14 | E1 | **CDP launch overhead 的首要量化来源**（被本批 4/6 篇引用）：报告平均 36.1% 性能损失，分解为参数分配 / SMX→KMU / KMU→KDU 三阶段 | — |
| `EDGE_PACT19_...pdf` | PACT'19 | B2 D3 E1 | 事件驱动 GPU 执行：非 CPU 设备直接启动预配置 kernel + **warp 级抢占**。**BlockMaestro 那个 5 μs launch 开销数值的原始出处** | `无` |
| `DTBL_ISCA15_...pdf` | ISCA'15 | A1 B3 D3 | Dynamic Thread Block Launch：**TB 粒度**的设备侧 spawn 并合并到已有 kernel | `无`（CDP 仍是 kernel 粒度） |
| `LaPerm.pdf` | ISCA'16 | B4 D3 | 面向 dynamic parallelism 的局部性感知调度器 | `无` |
| `KLAP_MICRO16_...pdf` | MICRO'16 | A2 D3 E1 | 编译器做 kernel launch 聚合（融合同 warp/block/kernel 的 child launch）与提升（提前启动 child） | `无` |
| `Free Launch.pdf` | MICRO'15 | A2 D3 E1 | 编译器变换消除 subkernel launch，复用父线程执行子任务，**无需硬件扩展** | `无` |
| `SPAWN_HPCA17_...pdf` | HPCA'17 | A1 D3 | 控制 CDP 父子 kernel 的 launch 与调度，减少依赖链上的开销 | `无` |

## 五、同步原语与内存模型（B1 / C2）

| 文件 | 会议/年份 | 维度 | 核心机制 | B300 |
|---|---|---|---|---|
| `SSB_ISCA07_...pdf` | ISCA'07 | B1 | Synchronization State Buffer：众核上的**硬件细粒度同步状态**（full/empty 类） | `无` |
| `Lustig_HPCA13_...pdf` | HPCA'13 | B1 C2 | **full/empty bit 做 CPU-GPU 细粒度同步**，使数据传输与 kernel 流水重叠。**full/empty 方案的直接先例** | `无` |
| `Xiao_IPDPS10_...pdf` | IPDPS'10 | B1 | 跨 block 的快速 barrier 同步（软件自旋实现的 global barrier） | `有`（`grid.sync()`） |
| `Fine-Grained Synchronizations and DataﬂowProgramming on GPUs.pdf` | ICS'15 | B1 | shared memory 上的细粒度线程间同步 + dataflow 编程，应用于 NW wavefront | `有`（mbarrier） |
| `hLRC_MICRO16_...pdf` | MICRO'16 | C2 | 异构惰性释放一致性：只在同步变量迁移时做一致性动作，**主张取消 scope** | `反` |

## 六、数据复用与融合（C1）

| 文件 | 会议/年份 | 维度 | 核心机制 | B300 |
|---|---|---|---|---|
| `Stash_ISCA15_...pdf` | ISCA'15 | C1 | 统一 scratchpad + cache 的**全局可寻址且保持一致性**的 on-chip 结构。**IKRA 在 Discussion 里明确建议用它实现 inter-kernel reuse——"跨 kernel on-chip 复用"这个空白最接近的已有工作** | `部分`（DSMEM 是受限版本） |
| `KernelFusion_arXiv1305.1183_...pdf` | arXiv'13 | C1 | BLAS kernel 自动融合，中间数据留在寄存器/shared memory。**明确指出 kernel launch 前 on-chip 数据全部丢失**——正是本项目要填的空白 | `有`（手工/编译器融合） |
| `VersaPipe_MICRO17_...pdf` | MICRO'17 | B3 C1 | GPU 流水编程框架：persistent threads + SM-centric 映射 + 自动调优选执行模型 | `部分` |

## 七、波前依赖与不规则负载（A1 / B1）

| 文件 | 会议/年份 | 维度 | 核心机制 | B300 |
|---|---|---|---|---|
| `Highly_Efficient_Compensation-Based_Parallelism_for_Wavefront_Loops_on_GPUs.pdf` | IPDPS'18 | A1 B1 | 补偿法并行化 wavefront 循环：先忽略行内依赖再补偿，避免严格反对角线序 | `无` |

## 八、本项目自己的产出

| 文件 | 内容 |
|---|---|
| `BlockMaestro_论文报告.md` | BlockMaestro 精读报告 + 10 条延伸讨论 Q&A（静态分析判据、JIT 阶段、PTX/SASS 关系、Hyper-Q、两阶段切分改进等） |

---

## 按维度反查

| 维度 | 主要参考文献 |
|---|---|
| **A1** 跨 kernel 跨度 | BlockMaestro、Wireframe、Juggler、DTBL、SPAWN、Compensation-wavefront |
| **A2** 依赖来源 | BlockMaestro、PAVER、Kim PACT'16、Wireframe、Juggler、ATA、KLAP、Free Launch |
| **A3** 表示与编码 | Wireframe（CSR）、BlockMaestro（模式模板）、Kim PACT'16（引用计数） |
| **A4** 持有者与方向 | BlockMaestro、Wireframe（集中）；Juggler、Whippletree、Softshell、Tzeng（分布） |
| **B1** 同步协议 | SSB、Lustig、Xiao、Fine-Grained Sync、hLRC、Kim PACT'16、Wireframe |
| **B2** 等待位置 | BlockMaestro（派发前）、PDL（驻留后）、EDGE（warp 抢占）、Chimera、Tanasic |
| **B3** 窗口与完成顺序 | BlockMaestro、Warped-Slicer、Gregg、Elastic Kernels、Kernelet、Zorua、CKE、VersaPipe |
| **B4** 调度与资源分区 | BlockMaestro、PAVER、IKRA、OWL、Lee HPCA'14、Equalizer、Neither More Nor Less、Warped-Slicer、Spatial Multitasking |
| **C1** 数据复用 | **Stash**、IKRA、PAVER、Locality Descriptor、CTA Clustering、Kernel Fusion、VersaPipe |
| **C2** 内存可见性 | hLRC、Lustig |
| **D1** 降级策略 | BlockMaestro（唯一） |
| **D2** 正确性与调试 | 无对口文献（race detector 线未收录，目标不同） |
| **D3** CUDA 抽象集成 | DTBL、LaPerm、KLAP、Free Launch、EDGE、SPAWN |
| **E1** 收益边界 | BlockMaestro Fig.12、**Wang IISWC'14**、EDGE |

## 未收录但已定位的文献

以下在引用追踪中被识别为相关，尚未下载，需要时按标题检索：

- **D2 方向**：ScoRD (ISCA'20)、BARRACUDA (PLDI'17)、CURD (PLDI'18)、HAccRG (ICPP'13)、GMRace (TPDS'14)、iGUARD (SOSP'21)
- **C2 方向**：HRF / Heterogeneous-Race-Free (Hower ASPLOS'14)、QuickRelease (HPCA'14)、DeNovo for GPU (Sinclair MICRO'15)
- **B3/B4 方向**：SMK / Simultaneous Multikernel (HPCA'16)、GPU Maestro (ASPLOS'17)
- **A4/B1 方向**：Persistent Threads study (Gupta InPar'12)、SM-centric transformation (Wu ICS'15)、PeerWave (Belviranli ICS'15)
