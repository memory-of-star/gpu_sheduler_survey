# CUDA 13.4 — PDL 与 CLC 全部软硬件接口汇总 + 硬件设计分析

> 范围:CUDA Toolkit 13.4 Developer Preview（2026 年 7 月发布）。
> 涉及两个特性族：
> - **PDL**(Programmatic Dependent Launch,程序化依赖启动)
> - **CLC**(Cluster Launch Control,集群启动控制 / `clusterlaunchcontrol`)
>
> PTX 接口以 **PTX ISA 9.3(latest / GA,对应 CUDA 13.3 线)** 为准锚定。
> 主要一手来源:
> - **PTX ISA 9.3(latest/GA)**:`docs.nvidia.com/cuda/parallel-thread-execution/`
>   - §9.7.14.14 `griddepcontrol`
>   - §9.7.14.18 `clusterlaunchcontrol.try_cancel`
>   - §9.7.14.19 `clusterlaunchcontrol.query_cancel`
> - CUDA Runtime API §6.17 Execution Control
> - CUDA C++ Programming Guide — Programmatic Dependent Launch and Synchronization
>
> 注:PTX ISA 9.4(CUDA 13.4 预览版)里这三条指令的**正文与 9.3 逐字相同**,仅章节号整体后移为 §9.7.15.14 / .18 / .19(详见文末"PTX 9.3 vs 9.4 校对")。

---

## 第一部分:PDL(Programmatic Dependent Launch)

PDL 的目标:让**同一 stream 里的后继(secondary/consumer)kernel 在前驱(primary/producer)kernel 尚未全部完成时,就提前启动并跑完那些"与前驱无关"的工作**(如 preamble / 清零 / 加载常量),从而隐藏启动延迟、重叠执行。

### 1.1 PTX 接口 — `griddepcontrol`(PTX ISA 9.3 §9.7.14.14)

```
griddepcontrol.action;
.action = { .launch_dependents, .wait }
```

| 修饰符 | 语义 | 颗粒度 |
|---|---|---|
| `.launch_dependents` | producer 侧信号:告诉运行时"我指定的 dependents 可以被调度了"。当 grid 内**所有 CTA** 都发了该指令(或已退出)后,dependent 才**有资格**提前启动(不保证一定提前)。同一 CTA 内重复调用无额外副作用。前置 release fence(scope=`gpu` 且同内存同步域,或 scope=`sys`)可与 dependent grid 的启动建立 synchronizes-with 关系。 | **整个 grid** |
| `.wait` | consumer 侧栅栏:阻塞当前线程,直到**所有 in-flight 前驱 grid 全部完成**,且它们的内存操作对当前 grid 可见。 | **整个 grid**(all-or-nothing) |

- **PTX ISA 版本**:Introduced in **PTX ISA 7.8**
- **Target ISA**:Requires **`sm_90` or higher**(Hopper 起)
- **配对约束**:若前驱用了 `.launch_dependents`,dependent grid **必须**用 `.wait` 才能保证功能正确。

> 关键点:该指令自 PTX ISA 7.8 起语义稳定;**PTX ISA 9.3 与 9.4 正文逐字一致**,没有新增 tile 级/sub-grid 级修饰符。

### 1.2 CUDA 设备侧接口(device functions,§6.17)

| 函数 | 作用 | 对应 PTX |
|---|---|---|
| `__device__ void cudaTriggerProgrammaticLaunchCompletion(void)` | producer 侧触发。仅当 grid 内**每个 CTA** 都退出或至少调用过一次,才真正 kick off;否则在所有 warp 结束后、grid 完成前自动发生。**只启用 secondary kernel 的调度,本身不提供任何内存可见性保证**——需用户自行插入正确 scope 的 memory fence。 | `griddepcontrol.launch_dependents` |
| `__device__ void cudaGridDependencySynchronize(void)` | consumer 侧阻塞,直到**所有直接 grid 依赖完成**。须与 programmatic / launch-event / dependency 配合使用。 | `griddepcontrol.wait` |

### 1.3 CUDA 主机侧接口(launch attributes)

通过扩展启动 API 传入属性:
- `cudaLaunchKernelEx` / `cudaLaunchKernelExC(const cudaLaunchConfig_t*, func, args)`
  - `cudaLaunchConfig_t.attrs`(`cudaLaunchAttribute` 数组)+ `numAttrs`

关键属性(`cudaLaunchAttributeID`):

| 属性 | 含义 |
|---|---|
| `cudaLaunchAttributeProgrammaticStreamSerialization` | `programmaticStreamSerializationAllowed = 1` 时,授权 driver 在不等待 primary 完成/内存 flush 的情况下**提前启动 secondary**。这是"允许提前启动"的主机侧许可位。 |
| `cudaLaunchAttributeProgrammaticEvent` | 基于 event 的 programmatic 依赖;字段 `programmaticEvent.triggerAtBlockStart`(0/1)控制触发时机(block 启动时 vs 触发完成时)。 |

> 驱动 API 等价物:`cuLaunchKernelEx` + `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` / `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`。

### 1.4 CUDA Graphs 中的 PDL(edge data)

在图里用 **edge data** 表达 programmatic 依赖:

| 字段 | 取值 |
|---|---|
| `cudaGraphEdgeData.type` | `cudaGraphDependencyTypeProgrammatic` |
| `cudaGraphEdgeData.from_port` | `cudaGraphKernelNodePortProgrammatic`(对应 `triggerAtBlockStart=0` / stream serialization)<br>或 `cudaGraphKernelNodePortLaunchCompletion`(对应 `triggerAtBlockStart=1`) |

该 edge 使上游 kernel 对下游 kernel 里的 `cudaGridDependencySynchronize()` 可见。stream capture 与 explicit edge data 两种方式等价。

### 1.5 PDL 语义小结(用于后面硬件分析)

- 依赖颗粒度 = **grid → grid**;
- `.launch_dependents` 只影响 **launch eligibility(能否提前启动)**,不含数据就绪;
- `.wait` 保证 **整个**前驱 grid 完成 + 内存可见;
- 内存可见性与"启动"解耦,必须显式 fence。

---

## 第二部分:CLC(Cluster Launch Control / `clusterlaunchcontrol`)

CLC 的目标:支持**持久化 kernel(persistent kernel)的动态取工/工作窃取**。一个正在运行的 cluster **原子地"抢占/取消"一个尚未开始运行的 cluster 的启动**,并接管它本应处理的 tile(拿到那个 cluster 首个 CTA 的 `ctaid`),从而实现类似 **Stream-K** 的动态负载均衡,填满调度气泡。

### 2.1 PTX 接口 — `clusterlaunchcontrol.try_cancel`(PTX ISA 9.3 §9.7.14.18)

```
clusterlaunchcontrol.try_cancel.async{.space}.completion_mechanism{.multicast::cluster::all}.b128 [addr], [mbar];

.completion_mechanism = { .mbarrier::complete_tx::bytes };
.space                = { .shared::cta };
```

语义:
- **原子地请求取消一个尚未开始运行的 cluster** 的启动;**异步**写一个 16 字节 opaque response 到 shared memory,表示成功/失败。
- `.async`(强制):异步发起,控制权立即返回。
- 完成通过 **mbarrier 的 complete-tx 机制**(`.cluster` scope)追踪;`mbar` 用 generic-proxy 访问。
- **成功时**:response 含被取消 cluster 的**首个 CTA 的 `ctaid`**;同一 grid 内其它成功的 `try_cancel` 不会再返回同一个 id(保证单一赢家、无重复领取)。
- `.space=.shared::cta`:`addr` 与 `mbar` 都在 `.shared::cta`;否则按 generic 寻址。
- `.multicast::cluster::all`:用 weak async-proxy write 把 response 写到请求 cluster 内**每个 CTA** 的本地 shared memory,并各自用 complete-tx 通知本地 mbarrier。若 cluster 中有 CTA 已退出则行为未定义。
- `addr`:16 字节、自然对齐的 shared memory 地址,存放 response。
- 若已观察到某次 `try_cancel` **失败**,再发后续 `try_cancel` 行为未定义。

- **PTX ISA 版本**:Introduced in **PTX ISA 8.6**
- **Target ISA**:Requires **`sm_100` or higher**(Blackwell 起)
- `.multicast::cluster::all` 支持架构:`sm_100a`、`sm_101a`(9.0 起改名 `sm_110a`)、`sm_120a`;8.8 起支持 family-specific `sm_100f`/`sm_101f`(→`sm_110f`)/`sm_120f`/`sm_110f`。

### 2.2 PTX 接口 — `clusterlaunchcontrol.query_cancel`(PTX ISA 9.3 §9.7.14.19)

```
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred, try_cancel_response;
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {xdim,ydim,zdim,_}, try_cancel_response;
clusterlaunchcontrol.query_cancel.get_first_ctaid{::dimension}.b32.b128 reg, try_cancel_response;
::dimension = { ::x, ::y, ::z }
```

语义:
- 解码 `try_cancel` 写出的 16 字节 opaque response(需先 `ld.shared.b128` 到 128-bit 寄存器)。
- `.is_canceled`:成功取消则 `pred=true`,否则 `false`。
- `.get_first_ctaid`:成功时取出被取消 cluster 首个 CTA 的坐标;`.v4` 返回 x/y/z(+1 个未定义元素),或用 `::x`/`::y`/`::z` 取单坐标。请求失败时该指令行为未定义。

- **PTX ISA 版本**:Introduced in **PTX ISA 8.6**;**Target ISA**:`sm_100` or higher。

### 2.3 CLC 依赖的配套 PTX 原语

CLC 的正确使用高度依赖以下现有原语(见 §2.1 官方示例):
- `mbarrier.arrive.expect_tx` / `mbarrier.try_wait`(异步完成追踪)
- `barrier.cluster.arrive` / `barrier.cluster.wait`(cluster 内 CTA 同步)
- `fence.proxy.async::generic.{acquire,release}`(generic-proxy 弱读 与 async-proxy 弱写之间的 proxy 内存序)

### 2.4 CLC 的 CUDA 软件接口现状

- **Cluster 本身**的 CUDA 侧接口(启动配置)是齐全的:
  - `cudaFuncAttributeRequiredClusterWidth/Height/Depth`(必需 cluster 维度)
  - `cudaFuncAttributeNonPortableClusterSizeAllowed`
  - `cudaFuncAttributeClusterSchedulingPolicyPreference`(值类型 `cudaClusterSchedulingPolicy`)
  - 启动属性 `cudaLaunchAttributeClusterDimension` / `cudaLaunchAttributeClusterSchedulingPolicyPreference`
- **`try_cancel` / `query_cancel` 本身没有一等公民的 CUDA Runtime C API 包装**。它们目前主要通过:
  - PTX / inline asm,或
  - 上层库(如 **CUTLASS** 的 Stream-K / persistent-kernel 调度、CuTe DSL)封装使用。
- 结论:CLC 是**偏底层的 PTX 能力**,面向库作者与高性能 kernel 作者,而非直接的 runtime C API。

### 2.5 CLC 语义小结(用于后面硬件分析)

- 作用对象 = **尚未启动的 cluster 的 launch slot**;
- 核心是**原子取消 + 单一赢家仲裁 + 返回被取消 cluster 的 ctaid**;
- 完成是**异步**的,靠 mbarrier + proxy fence 收口;
- 典型用法:persistent kernel 里"处理当前 cluster 的同时,异步请求领取下一个 cluster"的流水循环。

---

## 第三部分:要支撑这些软件接口,硬件应该怎么设计?

下面从"软件契约 → 硬件必须提供的机制"逐条反推。分为**共性调度基础**、**PDL 专项**、**CLC 专项**、**内存模型**四块,最后给出**Rubin tile-level triggering 所需的额外硬件**。

### 3.0 共性基础:可编程的网格/CTA 调度器(GigaThread Engine)

两个特性都要求 GPU 前端的 **grid/cluster 调度器不是"黑盒 FIFO",而是一个可被 device 侧指令影响状态的状态机**:

- **待启动队列(pending launch queue)**:调度器维护每个 grid 未派发 CTA/cluster 的队列与 slot 状态(pending / launched / completed)。
- **doorbell / 事件通道**:SM 上运行的 warp 能通过特定指令向调度器发出信号(launch_dependents、try_cancel),即需要一条 **SM→调度器的低延迟事件上行通路**。
- **每-grid 计数器/记分板**:跟踪"已启动 CTA 数 / 已完成 CTA 数 / 已发 trigger 的 CTA 数"。
- **命令描述符里的许可位**:host 在 launch packet 里写入的许可位(如 stream serialization allowed)必须被命令处理器解析,决定调度器是否遵守 device 侧指令。**device 指令不能越过 host 策略**——这是安全/正确性边界。

### 3.1 支撑 PDL 的硬件

`.launch_dependents` / `cudaTriggerProgrammaticLaunchCompletion`:
1. **跨 grid 依赖表**:调度器需知道 stream 内 primary→secondary 的依赖边(来自 launch attribute 或 graph edge)。
2. **"提前启动资格"聚合逻辑**:必须等 primary grid 的**所有 CTA**都发过 trigger(或退出)才置位"secondary 可启动"。这要求一个 **grid 级 barrier/计数器**:每个 CTA 发 trigger 时对计数器 +1,达到 grid CTA 总数即触发。需处理 CTA 提前退出(隐式 trigger)。
3. **投机/机会式派发**:secondary 有资格后,调度器在有空闲 SM slot 时**机会式**派发(规范明确"不保证"),因此硬件只需"允许"而非"强制",实现成本低。
4. **启动 ≠ 数据可见的解耦**:硬件**不需要**为 trigger 附带 cache flush;这就是为什么 `.wait`/fence 必须由软件显式给出。

`.wait` / `cudaGridDependencySynchronize`:
5. **grid 完成检测 + 内存可见性栅栏**:当 secondary warp 执行 `.wait`,SM 需向调度器/内存子系统查询"该 secondary 依赖的所有 primary grid 是否全部 retire,且其写入已 flush 到对 secondary 可见的层级(通常是 L2/全局)"。
   - 需要 **grid retire 广播**(调度器→SM)+ **内存 fence 到 GPU/sys scope**。
   - 这是 **all-or-nothing、grid 颗粒度**的等待——硬件只需一个"grid 全退且内存可见"的单一就绪信号,**无需追踪 tile 级别的部分完成**,故 Hopper/Blackwell 上实现相对简单。
6. **warp 阻塞/唤醒**:`.wait` 需让 warp 挂起(不空转占用发射槽),由就绪信号唤醒——需要硬件的 warp 挂起/重调度支持(与 `bar`/`mbarrier` 等待类似)。

### 3.2 支撑 CLC 的硬件(更复杂)

`clusterlaunchcontrol.try_cancel` 要求硬件在**待启动 cluster 队列**上做原子操作:

1. **可原子取消的 launch slot**:调度器的 pending cluster 队列中,每个尚未派发的 cluster 必须是一个可被"认领/取消"的原子对象。运行中的 CTA 发 `try_cancel` = 向调度器申请**原子 CAS/dequeue** 一个 pending cluster。
2. **单一赢家仲裁(race-free)**:规范保证"同一 grid 内其它成功 try_cancel 不会返回同一 id"。硬件必须保证**同一个 pending cluster 只能被恰好一个请求者领取**——即调度器侧的原子队列/仲裁器(硬件锁或原子指针),多请求并发时只放行一个成功、其余返回失败。
3. **返回被取消 cluster 的身份**:取消成功后,硬件要把该 cluster 首个 CTA 的 `ctaid`(x/y/z)打包进 16B response。意味着调度器要能**枚举/反查 pending cluster 的坐标**。
4. **异步响应写回 + 完成信号**:response 通过 **async-proxy weak write** 写到 shared memory(可 `.multicast::cluster::all` 写到 cluster 内每个 CTA),并用 **mbarrier complete-tx** 通知。这复用了 **TMA/异步拷贝引擎 + mbarrier 事务计数**那套硬件:
   - 一个能把小块数据异步写入 SMEM 并对 mbarrier 做 complete-tx 的引擎;
   - multicast 版还需**cluster 内跨 CTA 的 SMEM 写入网络**(DSMEM/分布式共享内存互连)。
5. **失败即终止的状态约束**:"观察到失败后再 try_cancel 行为未定义"——硬件可假设请求者失败后不再纠缠,简化仲裁器状态机(无需支持失败重试语义)。
6. **cluster 存活性前提**:`.multicast::cluster::all` 要求 cluster 内无 CTA 退出——依赖 cluster 生命周期的硬件跟踪(cluster 作为一等调度对象,GPC 内 SM 组的协同调度)。

`clusterlaunchcontrol.query_cancel`:
7. 纯**寄存器内 opaque 解码**,无需调度器参与。硬件只需定义 128-bit response 的位布局(是否成功位 + 3×坐标),并提供解码指令。**这部分几乎零调度器成本**,把复杂度都留在 `try_cancel` 侧。

### 3.3 贯穿两者的内存模型硬件

- **多 proxy 一致性**:CLC 涉及 generic-proxy(普通 `ld.shared`)与 async-proxy(异步写 response)两个代理,硬件必须实现 `fence.proxy.async::generic.{acquire,release}` 才能让"弱读 handle"和"异步写 handle"之间有正确顺序。
- **mbarrier 硬件**:事务计数(expect_tx / complete_tx)、`.cluster` scope 到达、`try_wait` 的 acquire 语义——CLC 与现代异步拷贝共用同一套 mbarrier 硬件。
- **scoped fence**:PDL 的 `.wait` 与 trigger 前的 release fence 需要 `gpu`/`sys` scope 的内存栅栏,依赖 L2 一致性点与(多 GPU 时)NVLink 层的可见性传播。

### 3.4 若要支撑博客里的 Rubin "tile-level triggering",硬件还缺什么

现有 PDL(§3.1)只做到 **grid 级 all-or-nothing**。要做到"某 producer tile 数据一就绪,对应 consumer tile 立即开工",硬件需在现有基础上**新增**:

1. **sub-grid / per-tile 的完成记分板**:调度器/内存子系统要能追踪"producer 的第 i 块 tile 已完成且写入可见",而不是只有一个 grid 级 bit。成本:一张按 tile 粒度的就绪位图 + 其更新/查询通路。
2. **细粒度依赖映射**:需要一种接口让程序声明"consumer tile j 依赖 producer tile 集合 S(j)"。这正是现有 `griddepcontrol` **无法表达**的——所以要么扩展 `griddepcontrol` 修饰符,要么新增 PTX 指令/新的 launch 依赖描述。
3. **数据驱动唤醒**:consumer tile 在其依赖就绪前挂起,由硬件在对应 producer tile 就绪时**精确唤醒**(避免软件自旋轮询 flag 造成的 SM/L2 带宽浪费)。需要调度器侧的"就绪→唤醒"事件路由,颗粒度到 CTA/tile。
4. **可见性下推到 tile 粒度**:内存子系统要能保证"tile i 的写入已对 consumer 可见"这一**局部**可见性判定,而非只有 grid 级 flush。

> 这解释了为何 13.4 预览里 `griddepcontrol` 一字未改:上述 (1)(2)(3)(4) 属于**新的硬件机制 + 新的软件契约**,尚未在公开 PTX/Runtime 接口中落地。现有接口只能表达 grid 级 bulk triggering。

### 3.5 一句话总结

| 特性 | 软件契约核心 | 硬件必须提供 |
|---|---|---|
| **PDL** | grid→grid 的"可提前启动"+"等整个前驱完成" | 跨 grid 依赖表、grid 级 trigger 计数器、机会式派发、grid retire 广播 + scoped fence、warp 挂起/唤醒 |
| **CLC** | 原子取消未启动 cluster、单一赢家、返回 ctaid、异步收口 | pending cluster 队列的原子取消/仲裁器、坐标反查、async-proxy 写 + mbarrier complete-tx、(multicast)DSMEM 互连、多 proxy fence |
| **Rubin tile-triggering(前瞻)** | tile→tile 细粒度依赖 | per-tile 就绪记分板、细粒度依赖声明接口、数据驱动精确唤醒、tile 粒度可见性——**目前接口尚缺** |

---

## 附录:PTX 9.3 vs 9.4 校对(针对本文这三条指令)

对比对象:PTX ISA **9.3(latest/GA)** 与 **9.4(CUDA 13.4 Developer Preview)**。结论:**三条指令的语法、语义、ISA 引入版本、Target 要求全部逐字一致,唯一差异是章节号整体后移一位**。

| 指令 | 9.3 章节号 | 9.4 章节号 | 语法/语义 | 引入版本 | Target ISA |
|---|---|---|---|---|---|
| `griddepcontrol` | §9.7.14.14 | §9.7.15.14 | 完全一致 | PTX ISA 7.8 | `sm_90`+ |
| `clusterlaunchcontrol.try_cancel` | §9.7.14.18 | §9.7.15.18 | 完全一致 | PTX ISA 8.6 | `sm_100`+ |
| `clusterlaunchcontrol.query_cancel` | §9.7.14.19 | §9.7.15.19 | 完全一致 | PTX ISA 8.6 | `sm_100`+ |

**为什么章节号会变?**
9.4 在 §9.7 里**插入了一组新的子节**,使原来的 "Parallel Synchronization and Communication Instructions" 组从 **§9.7.14** 顶到 **§9.7.15**,其后的 "Warp Level Matrix Multiply-Accumulate Instructions" 也从 §9.7.15 顶到 §9.7.16。因此这是**结构性重新编号**,并非对这三条指令本身的任何改动。

**9.4 新增里与本主题相关的?**
9.4 的 "新增特性清单"(§1.3)里**没有**任何涉及 `griddepcontrol` / `clusterlaunchcontrol` / dependent-launch / triggering 的条目;新增项集中在 `sm_107` 目标、`tcgen05.*`(第五代 Tensor Core)、FP8/FP6/FP4 packed 类型、`spcompress`/`spdecompress`、mbarrier 扩展等——都与 kernel 触发机制无关。

> 校对结论:**把本文 PTX 引用锚定到 9.3 是安全的**;9.4 对 PDL/CLC 没有实质改动,读 9.3 与读 9.4 对这三条指令等价(只需注意章节号 9.7.14.x ↔ 9.7.15.x 的映射)。

---

## 附录 B:实测 PTX(B300 / sm_103,griddepcontrol 的实际下降)

### B.1 工具链获取:pip 走不通,改用 Docker 拿到真实 nvcc 13.3

**pip 路线为何失败(已用证据坐实,非 pip/Python 版本问题):**

| 检查项 | 结果 | 结论 |
|---|---|---|
| 本机 pip | `25.0.1`(**已是最新**) | "换更高 pip"无空间 |
| NVIDIA wheel 标签 | `py3-none-manylinux` | 与 Python 小版本无关,3.8 也能装 |
| `nvidia-cuda-nvrtc-cu13` / `-nvcc-cu13` / `-runtime-cu13` / `-cccl-cu13` / `-crt-cu13`(pypi.nvidia.com) | **全部 HTTP 404** | CUDA 13 **编译器/运行时核心未发 pip 包** |
| `nvidia-cuda-nvrtc-cu12`(对照) | 200,真实 whl(≤12.9.86) | 方法可靠 |
| PyPI `nvidia-cuda-nvrtc-cu13` | 仅占位 `0.0.1` | 无真实发行 |
| NVIDIA index 上 cu13 家族 | **92 个**(cudf/cuml/cutensor/cudnn/nccl/cusparselt…) | CUDA 13 **库**已发 pip,唯独缺 nvcc/nvrtc |

> 根因:**NVIDIA 未把 CUDA 13.x 的编译器(nvcc/NVRTC)发布为 pip wheel**(官方 index 404、PyPI 仅占位)。任何 Python/pip 版本都下载不到不存在的文件——所以升级 Python/pip 无效。

**改用 Docker 成功拿到真实工具链**:`docker pull nvidia/cuda:13.3.0-devel-ubuntu24.04`(7.12 GB),内含 **nvcc release 13.3, V13.3.33**(Built Apr 24 2026),对应 **PTX ISA 9.3**。至此 `.version 9.3` 为**真实编译产物**,非示意。

### B.2 源码与真实产物(B300 / sm_103,PTX 9.3)

编译目标为 **B300 / GB300 = 计算能力 10.3 = `sm_103`**(需 CUDA ≥ 13.0;`griddepcontrol` 自 `sm_90` 起支持,`sm_103` 自然支持)。容器内 `nvcc -arch=sm_103 -ptx` 编译两份源码:

- `pdl_demo.cu` —— 自包含版(inline asm,逐字取自 `cuda_device_runtime_api.h`)→ `pdl_demo_cuda133_sm103.ptx`。
- `pdl_demo_real.cu` —— **真 intrinsic 版**,`#include <cuda_runtime.h>` 后直接调用 `cudaTriggerProgrammaticLaunchCompletion()` / `cudaGridDependencySynchronize()` → `pdl_demo_real_cuda133_sm103.ptx`。

两者头部与关键指令**逐字一致**:

```
// Cuda compilation tools, release 13.3, V13.3.33
.version 9.3
.target sm_103
.address_size 64
...
// producer:
	griddepcontrol.launch_dependents;      // <- cudaTriggerProgrammaticLaunchCompletion()
...
// consumer:
	griddepcontrol.wait;                   // <- cudaGridDependencySynchronize()
```

**两点被实测证实**:
1. **真实的 CUDA 高层 intrinsic** 在真实 nvcc 13.3 下确实下降为 `griddepcontrol.launch_dependents` / `griddepcontrol.wait`(与自包含 inline-asm 版产物逐字相同)。
2. `launch_dependents` 位于 producer"必须先完成的写"之后、"可与 consumer 重叠的尾部"之前;`griddepcontrol.wait` 位于 consumer 依赖读取之前、独立 prologue 之后——与 §1 语义一致。

### B.3 真实 PTX 9.4(经 packages.nvidia.com preview 通道)

Docker Hub **没有** `nvidia/cuda:13.4.x-devel-*` 镜像(已 `manifest inspect` 探测,均 not found),13.4 也不在 PyPI。但 CUDA 13.4 Developer Preview 通过 **`packages.nvidia.com` 的 Early-Access(preview)apt 通道**发布,可在容器里装到真实 nvcc:

```bash
# 在 noble(ubuntu24.04)容器内,--network host 走宿主 127.0.0.1:7890 代理
wget https://packages.nvidia.com/noble/nvidia-preview-keyring.deb   # 启用 preview 通道
dpkg -i nvidia-preview-keyring.deb && apt-get update
apt-get install -y cuda-nvcc-13-4 cuda-crt-13-4 cuda-cudart-dev-13-4  # → nvcc release 13.4, V13.4.46
```

产物 `pdl_demo_real_cuda134_sm103.ptx` 头部:

```
// Cuda compilation tools, release 13.4, V13.4.46
.version 9.4
.target sm_103
```

关键指令与 9.3 **逐字相同**:`producer` 里 `griddepcontrol.launch_dependents;`、`consumer` 里 `griddepcontrol.wait;`。

**真实 9.3 vs 真实 9.4 的完整 diff**(`pdl_demo_real_cuda133_sm103.ptx` vs `pdl_demo_real_cuda134_sm103.ptx`):

```diff
< // Compiler Build ID: CL-37862127
< // Cuda compilation tools, release 13.3, V13.3.33
---
> // Compiler Build ID: CL-38501229
> // Cuda compilation tools, release 13.4, V13.4.46
9c9
< .version 9.3
---
> .version 9.4
21c21
< 	.reg .pred 	%p<3>;
---
> 	.reg .pred 	%p<2>;
43d42
< 	setp.ge.s32 	%p2, %r1, %r2;
47c46
< 	@%p2 bra 	$L__BB0_4;
---
> 	@%p1 bra 	$L__BB0_4;
```

除版本元信息外,**唯一的指令级差异是一处 codegen 优化**:producer 里第二个 `if (i < n)` 边界判断,13.3 会重算一个谓词(`setp.ge.s32 %p2` + 多用一个谓词寄存器),13.4 直接复用第一次判断的 `%p1`——少一个 `%p`、少一条 `setp`。这是 nvcc 13.4 对 `sm_103` 的指令选择改进,**与 PDL 无关**:两版的 `griddepcontrol.launch_dependents` / `griddepcontrol.wait` 逐字相同。

> 对比:同样源码在 `sm_90` 上,9.3→9.4 除版本目录外**逐字相同**(无上述谓词差异)。sm_103 上多出的这处差异纯属新目标的 codegen 调优,反而印证了附录 A——PDL/CLC 指令本身在 9.3/9.4 无任何变化。

> 工具链版本对照(均为真实编译,非示意):PTX 9.3 = nvcc **13.3, V13.3.33**(Docker `nvidia/cuda:13.3.0-devel`);PTX 9.4 = nvcc **13.4, V13.4.46**(preview apt 通道 `cuda-nvcc-13-4`)。

### B.4 同 stream / 跨 stream / CUDA Graph:三种接法,设备端 PTX 全相同(B300 实测)

把**同一对 producer/consumer kernel**用三种 host 方式建立 PDL 依赖:

| 接法 | host 侧机制 | 源文件 |
|---|---|---|
| 同 stream | `cudaLaunchAttributeProgrammaticStreamSerialization`(consumer 进同一 stream,靠流内序) | `pdl_streams.cu` |
| 跨 stream | `cudaLaunchAttributeProgrammaticEvent`(producer 记 event)+ `cudaStreamWaitEvent`(另一 stream 等待) | `pdl_streams.cu` |
| CUDA Graph | `cudaGraphAddKernelNode` ×2 + `cudaGraphAddDependencies` 加一条 `cudaGraphDependencyTypeProgrammatic` 边(`from_port = cudaGraphKernelNodePortProgrammatic`) | `pdl_graph.cu` |

三者 `nvcc -arch=sm_103 -ptx` 的产物与 `pdl_demo_real_cuda134_sm103.ptx` **全部逐字节相同**(`diff` 均无输出):

```
$ diff pdl_streams_cuda134_sm103.ptx pdl_demo_real_cuda134_sm103.ptx   # 无输出
$ diff pdl_graph_cuda134_sm103.ptx   pdl_demo_real_cuda134_sm103.ptx   # 无输出
$ grep -ci graph pdl_graph.cu                # host: 21
$ grep -ci graph pdl_graph_cuda134_sm103.ptx # PTX: 0
```

**结论**:同 stream / 跨 stream / CUDA Graph 的区别**完全在 host 端**——launch attribute、`cudaStreamWaitEvent`、graph 的节点与边(`cudaGraphDependencyTypeProgrammatic`)。这些是 CUDA runtime 调用/数据结构,不进入 kernel 的 PTX。设备端三条路径都只是同样的 `griddepcontrol.launch_dependents`(producer)与 `griddepcontrol.wait`(consumer)——**既没有"跨 stream 专用"、也没有"graph 专用"的 PTX 指令**。9.3 上同样成立(已实测 `diff` 一致)。

> 唯一例外(且仍不是"图结构进 PTX"):**device graph launch**——kernel 内用设备侧 `cudaGraphLaunch(execGraph, …)` 触发一张*已实例化*的图,此时 PTX 里出现的是对设备运行时的一次调用 + 一个不透明的 `cudaGraphExec_t` 句柄,**图的拓扑(节点/边)依旧不在 PTX 里**。

---

## 附录 C:PTX 9.4 相比 9.3 新增的相关内容

> 依据:PTX ISA 9.4 §1.3 "PTX ISA version 9.4 introduces the following new features"(即 9.3→9.4 的官方 delta)。

**先说结论(针对本文主题)**:
- **PDL(`griddepcontrol`)**:9.4 **零新增**。
- **CLC(`clusterlaunchcontrol.try_cancel` / `query_cancel`)**:9.4 **零新增**。

9.4 的新增没有直接触碰"kernel 触发/依赖"机制,但**大量增强了这两个特性所依赖的周边机制**(mbarrier / TMA 异步拷贝),以及 Rubin 博客里描述的其它加速路径。按相关度分组如下:

### C.1 新目标架构
- **`sm_107` / `sm_107f` / `sm_107a`**(family-specific / architecture-specific)——时间线上很可能对应 Rubin(文档未明文,属推断)。

### C.2 CLC 依赖的 mbarrier 机制增强(相关度高)
CLC 的完成靠 mbarrier complete-tx,9.4 对 mbarrier 的增强会直接惠及这类异步收口:
- `.multicast::cluster::32b`:用于 `mbarrier.expect_tx` / `complete_tx` / `arrive` / `arrive_drop`(以及 `cp.async.bulk[.tensor]`、`tcgen05.commit`)。
- `.phase_type::*`:用于 `mbarrier.test_wait` / `try_wait`。
- `reportPredicate` / `reportValue` 操作数:用于 `mbarrier.test_wait` / `try_wait`。
- `.layout` 限定符 + 新指令 `mbarrier.check_layout`。

### C.3 TMA / 异步 bulk 拷贝增强(对应博客 MoE 描述符共享)
Rubin 博客讲的 "inline descriptor update for TMA / MoE 描述符共享",在 PTX 9.4 里对应这些新增:
- **`.override::global_address` 和 `.override_attribute`**:用于 `cp.async.bulk.tensor` / `cp.reduce.async.bulk.tensor` / `cp.async.bulk.prefetch.tensor`——即"复用同一 descriptor、在指令里就地覆盖地址/属性",正是博客所述的 TMA 内联描述符更新。
- `.im2col_no_offs::w`:用于 `cp.async.bulk.tensor` / `cp.reduce.async.bulk.tensor`。
- `.level::eviction_priority`:用于 `cp.async.bulk.prefetch[.tensor]`。
- `applypriority.async.bulk` 和 `applypriority.async.bulk.tensor` 新指令。
- `.report_mechanism`:用于 `cp.async.bulk[.tensor]`。

### C.4 稀疏(对应博客 attention 2:4 稀疏)
- 新指令 **`spcompress` / `spdecompress`**。
- `.spcompress` 限定符:用于 `tcgen05.ld` / `tcgen05.ld.red`。
- `tcgen05.mma` 的 `.decompress::lut::b`、`.collector::b::*`。

### C.5 低精度 / Tensor Core(对应博客扩展精度、NVFP4、K 维翻倍)
- `add`/`sub`/`mul`/`fma` 支持 **FP8/FP6/FP4 x4 packed** 类型,以及 `.f16x2`/`.bf16x2`/`.f32x2`。
- `cvt` 新增 `.ue5m3x2`、`.scaled::n1::ue8m0`、`.rz`(对 e4m3x2/e5m2x2/e2m3x2/e3m2x2/e2m1x2)、`.pzo`。
- `tcgen05.mma[.sp/.ws]` 新增 `UE5M3` scale、`.kind::ti16`;`tcgen05.alloc/dealloc` 新增 `.exclusive`;`tcgen05.commit` 新增 `.sync_restrict::shared::read::mma::a`。
- `set` 指令支持 `.u8x4`/`.s8x4`/`.u16x2`/`.s16x2`;`ldmatrix` 支持 `.s8.s4`(`.m8n16`)。

### C.6 其它
- **per-CTA global memory**:`.minperctamemory` 指令 + `%perctamemoryoffset` / `%perctamemorysize` 特殊寄存器。
- `prefetch` 新增 `.valid_addr`、`.L1::32B`。
- `atom`/`red`/`cp.reduce.async.bulk`/`multimem.cp.reduce.async.bulk` 新增 `.noftz`(配 `.f32`)。
- `ld` 新增 `.proxy::readonly`。
- 调试/IR:`.loc_intermediate` 指令 + `.nv_intermediate_source_section`。

> 总结:9.4 相对 9.3 的新增**没有一条**改动 PDL/CLC 本身;真正与本文相关的是**mbarrier 与 TMA 异步拷贝的增强**(CLC/异步收口的地基),以及 Rubin 博客其余加速路径(稀疏、低精度、TC)。博客宣传的 **tile-level triggering 仍未出现在 9.4 的任何新指令/新修饰符里**。
