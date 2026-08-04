# Cross-stream PDL 的生产应用与开源案例调研

> 调研日期：2026-08-04
> 主题：NVIDIA Programmatic Dependent Launch（PDL）在跨 stream、CUDA Graph 跨节点依赖中的生产价值，以及公开开源实现。

## 1. 核心结论

当前实验已经证明 NVIDIA 的 PDL 机制可以把 producer kernel 的 tail 与 consumer kernel 的前置阶段重叠起来，并且这种能力可以跨 CUDA Graph node 生效。

不过，生产代码里更准确的抽象不是“跨 stream PDL”，而是：

> **CUDA Graph programmatic dependency / graph-node PDL**

跨 stream 通常只是 graph capture 之前的构图方式。一旦 capture 和 instantiate 完成，真正被 CUDA 调度器消费的是 graph node 之间的 programmatic edge；原来的 user stream 身份不再是核心语义。一个 graph 即使最终只提交到一个 stream，内部 kernel node 仍然可以通过 `griddepcontrol` 重叠执行。

截至 2026-08-04，公开代码中可以严格确认的两套独立 graph-node PDL 实现是：

1. **OpenXLA**：在 CUDA command buffer 中生成 Driver API programmatic graph edge，并在 Hopper+ 默认启用 PDL 编译路径。
2. **NVIDIA CUTLASS**：在 Distributed GEMM 中用 Runtime API programmatic graph edge 连接跨 GPU barrier 与 local GEMM。

没有找到主流仓库在生产路径中使用下面这套 eager 两流模式：

```text
ProgrammaticEvent(stream A)
    -> cudaStreamWaitEvent(stream B)
    -> dependent kernel
```

这与当前本地实验的结果一致：eager 两流依赖在测试平台上没有获得 overlap，而 capture 或直接建图后能够获得 overlap。

## 2. 对当前实验的解释

本目录已经覆盖了三条不同的软件路径：

| 路径                | Host 接口                                                          | 当前 H100 实测 |
| ------------------- | ------------------------------------------------------------------ | -------------: |
| 同 stream PDL       | `cudaLaunchAttributeProgrammaticStreamSerialization`             |     能 overlap |
| eager 跨 stream PDL | `cudaLaunchAttributeProgrammaticEvent` + `cudaStreamWaitEvent` |   没有 overlap |
| CUDA Graph PDL      | capture 后生成 programmatic edge，或直接设置`cudaGraphEdgeData`  |     能 overlap |

相关实验与实现见：

- [PDL_跨stream_总结.md](./PDL_跨stream_总结.md)
- [pdl_bench.cu](./bench/pdl_bench/pdl_bench.cu)
- [pdl_diamond.cu](./bench/pdl_bench/pdl_diamond.cu)

其中最关键的对照是：

- `PDL_XS` 与 `PDL_CAPTURE` 使用相同的 producer、consumer、programmatic event 和两个 stream；
- 唯一区别是后一种被 capture 成 CUDA Graph；
- eager 版本约为普通 event 的 `1.00x`，capture 后达到约 `2.00x`；
- 直接建立 `cudaGraphDependencyTypeProgrammatic` edge 也达到约 `2.00x`。

diamond 实验进一步证明 programmatic edge 能用于 DAG：

```text
producer
   |\
   | \
 midA midB
   |  /
   | /
 final
```

普通 edge 会让相邻层按完成顺序执行；programmatic edge 则允许下一层提前 admission，并由 consumer 内部的 `cudaGridDependencySynchronize()` 在真正读取依赖数据前完成同步。

### 2.1 对“只有 Graph 才支持”的修正

`cudaLaunchAttributeProgrammaticEvent` 并不是 capture-only API。官方 Runtime API 将它定义为可用于普通 kernel launch 的属性，Driver API 也说明它可以随 kernel launch 记录 programmatic event。

因此更准确的结论是：

> eager 跨 stream PDL 的依赖组合在正确性上受支持，但 event release 被观察的时间、consumer 的提前发射时机以及实际 overlap 都不属于性能保证；在当前测试平台和软件栈上，只有 Graph 路径实际获得了 overlap。

官方依据：

- [CUDA Runtime launch attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)
- [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [`cudaStreamWaitEvent`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
- [`cudaGridDependencySynchronize`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html)

需要区分两件事：

1. **依赖正确性**：event、stream wait 和 downstream grid wait 可以组成合法的依赖关系。
2. **性能时序**：CUDA 不保证 dependent kernel 精确在 trigger 时被调度，也不保证一定发生并发。

## 3. 官方 Graph PDL 语义

CUDA Graph 从 CUDA 12.3 开始支持带 edge data 的依赖。当前唯一的非默认 edge type 是 programmatic dependency。

典型设置为：

```cpp
cudaGraphEdgeData edge{};
edge.type = cudaGraphDependencyTypeProgrammatic;
edge.from_port = cudaGraphKernelNodePortProgrammatic;
```

也可以使用：

```cpp
edge.from_port = cudaGraphKernelNodePortLaunchCompletion;
```

两种端口的区别是：

| Outgoing port                               | 激活条件                                         |
| ------------------------------------------- | ------------------------------------------------ |
| `cudaGraphKernelNodePortProgrammatic`     | producer 的所有 CTA 都调用了 trigger，或已经终止 |
| `cudaGraphKernelNodePortLaunchCompletion` | producer 的所有 CTA 都已经开始执行               |
| default port                                | producer kernel 完整结束                         |

programmatic edge 的正确模型是：

```text
producer early port 激活
        |
        v
consumer 获得调度资格，执行独立 preamble
        |
        v
cudaGridDependencySynchronize()
        |
        v
等待所有 direct producer 完成并取得内存可见性
        |
        v
consumer 执行依赖数据的部分
```

重要限制：

- programmatic edge 只能用于 **kernel node -> kernel node**；
- early trigger 本身不是 memory fence；
- consumer 必须在首次依赖读取前执行 grid dependency wait；
- PDL overlap 需要 compute capability 9.0+；
- concurrency 是 opportunistic，程序正确性不能依赖它必然发生；
- 如果 producer 等待 consumer 释放资源，可能形成死锁。

参考：

- [CUDA Graph edge data](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [PTX `griddepcontrol`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)

## 4. 直接开源案例一：OpenXLA

OpenXLA 是目前公开代码中最强的通用框架案例。它不是测试或文档示例，而是 CUDA command buffer 的正式 compiler/runtime 路径。

### 4.1 CUDA Graph edge

当 consumer kernel 的 `use_pdl()` 为真，并且 predecessor 是 kernel node 时，OpenXLA 为每条 incoming edge 设置：

```cpp
edge_data_item.from_port = CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC;
edge_data_item.type = CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
```

随后通过 `cuGraphAddNode_v2` 将 kernel node 加入 graph。

源码：

- [OpenXLA `cuda_command_buffer.cc`](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/stream_executor/cuda/cuda_command_buffer.cc#L562-L602)
- [对应 Hopper 单元测试](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/stream_executor/cuda/cuda_command_buffer_test.cc#L184-L244)

Graph 端要求 CUDA driver 12.3 或更新版本，否则回退为普通 edge。

### 4.2 设备端 trigger/wait 插入

OpenXLA 的 PDL pass 会：

1. 在相关 kernel 入口插入 PDL wait；
2. 找到带 successor fusion 的 Triton GEMM；
3. 在最后一个 `tt.dot` 之后、lower-level epilogue 之前插入 `launch_dependents`；
4. 将 successor 中来自该 GEMM 的 operand 标记为 non-invariant，防止依赖数据在 grid wait 前通过 non-coherent/invariant load 被读取。

源码：

- [PDL launch annotation](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/backends/gpu/transforms/pdl_launch_annotation.cc#L82-L128)
- [插入 wait/launch 指令](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/backends/gpu/codegen/emitters/transforms/insert_pdl_ops.cc#L62-L84)
- [降低为 NVVM griddepcontrol](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/codegen/emitters/transforms/lower_pdl_ops.cc#L36-L57)

它表达的生产模式是：

```text
Triton GEMM producer
   ├─ tensor-core / dot mainloop
   ├─ launch_dependents
   └─ transpose、shared-memory move、global-store epilogue

successor fusion
   ├─ 提前获得调度资格
   ├─ 执行常量、索引和地址等准备工作
   ├─ griddepcontrol.wait
   └─ 读取 GEMM 输出并继续计算
```

### 4.3 默认可达性与性能证据

当前 OpenXLA main 中：

- `xla_gpu_enable_pdl` 默认开启；
- `xla_gpu_enable_pdl_launch` 默认开启；
- 仅 CUDA Hopper 或更新架构实际生效；
- fusion 默认允许进入 command buffer；
- 默认最小 graph size 为 5 个 thunk，因此较长的 fusion 序列通常可以进入 graph，短序列仍可能走 eager 路径。

源码：

- [PDL 默认配置](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/debug_options_flags.cc#L588-L590)
- [Hopper+ gating](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/service/gpu/ir_emission_utils.h#L79-L88)
- [Command buffer 默认配置](https://github.com/openxla/xla/blob/0b48cf0806d58d04cb117decf6d097e5165eaf5b/xla/debug_options_flags.cc#L304-L313)

[OpenXLA PR #38544](https://github.com/openxla/xla/pull/38544#issuecomment-3987402459) 展示了一串位于 CUDA Graph 中的小 dependent kernels。作者报告 H100 上 CUDA-event 端到端计时约有 [1.25x speedup](https://github.com/openxla/xla/pull/38544#issuecomment-3998819007)，reviewer 也报告成功[复现](https://github.com/openxla/xla/pull/38544#issuecomment-4029638236)。

[PR #43894](https://github.com/openxla/xla/pull/43894) 进一步加入了 Triton GEMM mainloop 与 epilogue 之间的显式 early trigger。其模型 benchmark 多数是无回退或约 `1%` 的提升。

需要谨慎解释：同一个 `use_pdl` 在非 Graph 路径会被转换成 same-stream programmatic serialization，因此普通模型 benchmark 没有单独隔离 graph edge 和 eager PDL 的贡献。可以确认的是：

- graph-edge 代码路径真实存在；
- 默认配置能够到达它；
- graph chain microbenchmark 有直接收益证据；
- 尚不能从现有公开 benchmark 精确量化 cross-node Graph PDL 对完整模型的独立贡献。

TensorFlow 的 `third_party/xla` 中也包含同源代码：

- [TensorFlow 中的 OpenXLA 镜像](https://github.com/tensorflow/tensorflow/blob/5effdcb69493794711cfa77fdcd9cf97171c5d33/third_party/xla/xla/stream_executor/cuda/cuda_command_buffer.cc#L581-L602)

它应视为同一实现的 vendored mirror，而不是第三套独立设计。

## 5. 直接开源案例二：NVIDIA CUTLASS Distributed GEMM

CUTLASS 的 experimental Distributed GEMM 用 programmatic graph edge 连接跨 GPU full barrier 与 local GEMM。

核心代码：

```cpp
cudaGraphEdgeData barrier_to_gemm_edge = {};
barrier_to_gemm_edge.from_port =
    HasMemcpy ? cudaGraphKernelNodePortLaunchCompletion
              : cudaGraphKernelNodePortProgrammatic;
barrier_to_gemm_edge.type = cudaGraphDependencyTypeProgrammatic;
```

源码：

- [构建 barrier-to-GEMM edge](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/include/cutlass/experimental/distributed/device/dist_gemm_universal_wrapper.hpp#L606-L618)
- [Graph instantiate](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/include/cutlass/experimental/distributed/device/dist_gemm_universal_wrapper.hpp#L647-L655)
- [Graph replay](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/include/cutlass/experimental/distributed/device/dist_gemm_universal_wrapper.hpp#L671-L699)

设备端也存在完整配对：

- full barrier kernel 调用 `launch_dependent_grids()`，然后继续执行 flag reset、peer atomic signal 和 spin wait；
- distributed GEMM 调用 `wait_on_dependent_grids()`，再通过 device flag 等待对应 buffer 的精确 readiness；
- GEMM 还会提前 release 后续 grid，形成 PDL chain。

源码：

- [full barrier kernel](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/include/cutlass/experimental/distributed/kernel/full_barrier.hpp#L47-L79)
- [distributed GEMM kernel wrapper](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/include/cutlass/experimental/distributed/kernel/dist_gemm_kernel_wrapper.hpp#L208-L227)

对应的 graph 形状近似为：

```text
                     ┌─ D2D memcpy / flag update
full barrier node ───┤
                     └─ local GEMM 0 -> GEMM 1 -> ...
```

其中 graph edge 负责 **early admission**，device flag/barrier 负责更细粒度的 **data readiness**。这是很有代表性的生产设计：

- 默认 Graph dependency 过于保守，会等待 full barrier kernel 完整结束；
- 完全删除 dependency 又不安全；
- programmatic edge 允许 GEMM 提前入场；
- GEMM 在真正消费通信数据前通过 PDL wait 和 device flag 保证正确性。

CUTLASS 提供的工作负载包括：

- AllGather + GEMM；
- GEMM + ReduceScatter；
- Hopper [Example 65](https://github.com/NVIDIA/cutlass/tree/f94ec46f4f63f96003d6cfdf2014731e7672c281/examples/65_distributed_gemm)；
- Blackwell Example 82。

需要注意，这仍位于 CUTLASS experimental API 和 example 中，属于生产形态明确的参考实现，而不是某个线上服务的公开部署证明。

## 6. 最有价值的生产场景

### 6.1 小 dependent-kernel 链

典型形态：

```text
kernel A -> kernel B -> kernel C -> ...
```

即使 consumer 没有很长的显式 preamble，PDL 也可能隐藏：

- graph node 到 kernel 首指令之间的 admission latency；
- kernel 参数、常量和索引计算；
- 地址生成等 boilerplate；
- producer 末尾的少量 epilogue。

它适用于 CUDA Graph 中重复 replay 的推理或训练 step，尤其是大量短小 elementwise、quantization、routing 或 bookkeeping kernels 的链。

OpenXLA 的 graph-chain microbenchmark 就属于这种情况。

### 6.2 GEMM mainloop 与 successor fusion 重叠

producer 在 tensor-core mainloop 完成后发出 trigger，但仍需完成：

- accumulator transform；
- shared-memory transpose；
- scale、activation 或 quantization epilogue；
- global-store。

下游 fusion 可以提前 admission，并完成不读取 GEMM 输出的准备阶段。OpenXLA 当前实现直接覆盖这一场景。

### 6.3 跨 GPU barrier / AllGather 与 local GEMM 重叠

在 tensor parallel 或 expert parallel 中，GPU kernel 可能需要：

- 向 peers 写 arrival flag；
- 等待远端 GPU；
- 搬运 remote buffer；
- 为多个 local GEMM 准备分块数据。

programmatic graph edge 可以让 local GEMM grid 提前入场，再由 device flag 决定每个 buffer 何时可读。CUTLASS Distributed GEMM 是直接案例。

### 6.4 RMSNorm/reduction 到 Q/K/V 或 gate/up GEMM 的 fan-out

典型 transformer 拓扑：

```text
                 ┌─ Q GEMM
RMSNorm output ──┼─ K GEMM
                 └─ V GEMM
```

或者：

```text
                 ┌─ gate GEMM
RMSNorm output ──┤
                 └─ up GEMM
```

在 activation 完全就绪前，各 consumer 可以预取：

- 静态 weights；
- TMA descriptor；
- scale / zero point；
- shared-memory pipeline metadata。

随后在首次读取 activation 前执行 PDL wait。

CUTLASS Example 63 已经在 same-stream PDL 中实现“等待 activation 时预取 weights”的模式：

- [GEMM with L2 weight prefetch](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/examples/63_hopper_gemm_with_weight_prefetch/README.md#L1-L20)

Graph PDL 的独特价值是可以把它从一条线性 chain 扩展到多个并行 consumer。这与当前 `pdl_diamond.cu` 的拓扑最接近，可能是最值得继续验证的单 GPU 生产场景。

### 6.5 Q/RoPE producer 到 GQA/attention

attention kernel 的输入并非都依赖紧邻的 producer：

- Q 可能由上一层 projection/RoPE kernel 刚刚生成；
- K/V cache、page table、sequence metadata 和 scale 可能已经就绪。

因此 attention 可以提前启动 K/V TMA load 或 metadata 准备，只在加载 Q 前等待 producer。

CUTLASS Blackwell GQA 的代码明确采用了这种拆分：

- Q load 前调用 `griddepcontrol.wait`；
- K/V load 不受该 wait 约束。

源码：

- [CUTLASS GQA：Q guarded，K/V unguarded](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/examples/93_blackwell_low_latency_gqa/tgv_gqa.cuh#L506-L606)

当前代码使用 same-stream PDL；如果 Q producer 与 attention 属于 graph 的不同 branch，这就是 graph-node PDL 的自然应用。

### 6.6 MoE routing/all-to-all 到 expert GEMM

可能的 pipeline 是：

```text
top-k / routing / all-to-all
          |
          v
多个 expert GEMM
```

expert GEMM 可以在 token mapping 或远端 token 数据完成前：

- admission 到 GPU；
- 预取 expert weights；
- 建立 tile scheduler 和 descriptor；
- 检查 device-side arrival flag。

随后在读取 token buffer 前等待 routing/communication producer。

TensorRT-LLM、FlashInfer 和 SGLang 已在 routing、MoE、all-reduce fusion、grouped GEMM 等 kernel 中广泛使用 same-stream PDL，证明这类 workload 有真实需求。Graph programmatic edge 特别适合固定 topology 或 max-shape replay 的 MoE graph。

风险是 MoE 的 expert 数量和 token 分布高度动态；如果每次都需要重建 graph，graph 管理成本可能抵消收益。实践中通常需要固定最大 grid，并通过 device metadata 决定实际工作量。

### 6.7 fan-in 场景

programmatic graph edge 还支持多个 direct producer 指向一个 consumer：

```text
producer A ─┐
            ├─> final consumer
producer B ─┘
```

consumer 可以提前执行共同的独立 preamble，而一次 `cudaGridDependencySynchronize()` 会等待其所有 direct grid dependencies。

可能的生产例子包括：

- 两个 projection/reduction branch 汇合到 residual add；
- 多个 partial reduction 汇合到 normalization/finalization；
- communication branch 与 compute branch 汇合到 fused epilogue。

当前 `pdl_diamond.cu` 已经验证了这种语义。

## 7. 适用条件和收益模型

可以用下面的近似式判断一个场景是否值得：

```text
潜在收益
  ≈ min(producer 独立 tail, consumer 独立 prefix)
    + 被隐藏的 kernel/node launch gap
    - 并发产生的资源干扰
```

### 7.1 适合 PDL 的条件

- graph topology 稳定，并且会多次 replay；
- producer trigger 后仍有明显 tail；
- consumer 在首次依赖读取前有可执行的独立工作；
- 或者 kernel 很短，node-to-node launch gap 占比较高；
- producer 与 consumer 没有同时饱和相同资源；
- 设备至少是 Hopper，即 compute capability 9.0+；
- 所有相关依赖都能表达为 kernel-to-kernel edge；
- 即使 CUDA 最终不发生 overlap，程序仍然完全正确。

### 7.2 需要检查的 GPU 资源

提前 admission 不一定产生收益。需要观察：

- producer 是否已经占满所有 SM；
- register 和 shared-memory occupancy 是否允许 consumer block 驻留；
- 两个 kernel 是否争用相同的 HBM/L2/TMA 带宽；
- consumer 的 wait 是否让大量 resident blocks 空转；
- 过早 prefetch 是否驱逐 producer 或其他 layer 的有效 cache line；
- fan-out 的多个 consumer 是否彼此造成更严重的 contention。

CUTLASS Example 63 明确建议按 GEMM 和端到端 transformer layer autotune overlap/prefetch ratio，而不是使用固定参数。

### 7.3 正确性要求

- trigger 只允许 dependent grid 被调度，不提供完整内存可见性；
- consumer 必须在首次读取依赖数据前执行 `cudaGridDependencySynchronize()`；
- 对 dependent pointer 不能在 wait 前进行不安全的 invariant/non-coherent load；
- producer 不能等待 consumer 完成某项工作，否则可能死锁；
- 不能把“当前 profiler 上有 overlap”当作正确性前提。

## 8. 不适合的场景

以下场景通常不值得使用 graph-node PDL：

### 8.1 完全独立的 kernels

如果两个 kernel 没有数据依赖，直接使用普通多 stream 或 CUDA Graph 并发即可，不需要 PDL dependency。

### 8.2 简单线性 chain

如果只有一个 producer 和一个 consumer，而且已有同 stream launch，`cudaLaunchAttributeProgrammaticStreamSerialization` 通常更简单。Graph-node PDL 的额外价值主要体现在复杂 DAG 或已经使用 CUDA Graph 的系统中。

### 8.3 consumer 一启动就必须读取 producer 输出

如果 consumer 没有独立 prefix，收益主要只剩 launch/admission gap。短 kernel chain 仍可能受益，但大型 kernel 通常收益有限。

### 8.4 两个 kernel 都饱和同一资源

例如两个 GEMM 都占满 SM、寄存器、shared memory 和 tensor cores。提前启动可能不会产生重叠，反而因资源竞争或 cache 干扰变慢。

### 8.5 dependency 包含非 kernel node

programmatic graph edge 只能连接 kernel node。不能直接把 memcpy、host node 或普通 NCCL API node 作为 programmatic producer/consumer。

可选方案包括：

- 把通信或同步实现为 GPU kernel；
- 使用 device-side flag/semaphore；
- 采用 CUTLASS Distributed GEMM 那样的“graph admission + device readiness”混合模式。

### 8.6 需要逐 tile 数据流

Hopper 和 Blackwell/B300 上的 PDL 是 whole-grid/bulk trigger。programmatic port 只有在所有 CTA 都 trigger 或终止后才激活，不能表达“producer 的第一个 tile 准备好后，consumer 立即处理该 tile”。

逐 tile pipeline 更适合：

- persistent kernel fusion；
- cooperative kernels；
- device semaphore/barrier；
- 通信与计算 fused kernel；
- 支持更细粒度 triggering 的后续架构能力。

## 9. 为什么公开 eager 跨 stream 案例很少

截至 2026-08-04，对以下主流仓库及全局公开代码索引进行了精确符号检查：

- OpenXLA / TensorFlow；
- NVIDIA CUTLASS；
- TensorRT-LLM；
- PyTorch；
- vLLM；
- SGLang；
- FlashInfer；
- Triton；
- NCCL；
- NVIDIA cuda-samples。

结论是：

- OpenXLA 和 CUTLASS 有直接 programmatic graph edge；
- TensorFlow 是 OpenXLA 的同源镜像；
- TensorRT-LLM 中唯一的 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT` 位于 [`#ifdef TLLM_TEST`](https://github.com/NVIDIA/TensorRT-LLM/blob/048ae4acde916d2cb40ecacdd971ed6e2c952800/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/CudaKernelLauncher.h#L95-L150)，没有找到传入非空 event 的生产 caller；
- CUTLASS 有一个 [CuTeDSL+CUDAGraph programmatic-event 教程](https://github.com/NVIDIA/cutlass/blob/f94ec46f4f63f96003d6cfdf2014731e7672c281/examples/python/CuTeDSL/dsl_tutorials/launch_completion_and_programmatic_events.py#L268-L330)，但不是 eager 两流 dependency；
- 其余仓库即使大量使用 PDL，也主要通过 same-stream serialization、`use_pdl` 或直接 `griddepcontrol`，没有发现公开的 eager cross-stream event chain。

这只能说明公开索引中没有更多可以验证的 exact-token 实现，不能证明闭源库、未索引代码或使用更高层封装的项目不存在这种用法。

这种稀缺性可能来自：

1. eager programmatic event 不提供稳定的提前观察/overlap 时序；
2. latency-sensitive inference 本身通常已经使用 CUDA Graph；
3. graph edge data 是 CUDA 12.3 才加入的年轻接口；
4. PDL overlap 要求 Hopper+；
5. 大多数 framework graph builder 尚未公开 edge-data 控制；
6. graph instantiate 后 stream identity 不再重要，因此代码和文档通常按 graph dependency 而不是 cross-stream 搜索和描述。

## 10. 建议补充的生产化实验

当前 synthetic benchmark 用相等且很长的 producer tail、consumer prologue 构造了接近 `2x` 的机制上限。下一步可以增加更接近生产负载的实验。

### 10.1 OpenXLA 风格的小 kernel chain

构造 5～20 个依赖的小 kernel：

- consumer 在入口立即执行 grid wait；
- 仅保留少量索引、常量和地址计算；
- 比较普通 graph edge、programmatic graph edge 和 same-stream PDL；
- 统计 node-to-node gap、首 CTA admission latency 和 end-to-end latency。

这可以验证即使没有显式长 prologue，PDL 是否仍能隐藏 graph launch gap。

### 10.2 Norm 到三个 weight-prefetch GEMM 的 fan-out

构造：

```text
RMSNorm/reduction
   ├─ Q-like GEMM consumer
   ├─ K-like GEMM consumer
   └─ V-like GEMM consumer
```

consumer 在 wait 前只预取静态 weights，在 wait 后读取 activation 并计算。需要同时测量：

- end-to-end latency；
- L2 hit rate；
- HBM/TMA 带宽；
- 三个 consumer 的资源竞争；
- 不同 prefetch ratio。

### 10.3 Barrier/communication 到 GEMM 的 DAG

单 GPU 可先用长尾 barrier kernel 和 device flag 模拟：

```text
barrier producer
   ├─ copy/flag branch
   └─ GEMM consumer chain
```

多 GPU 环境再替换为 peer flag、NVLink buffer 和实际 AllGather/ReduceScatter schedule，复现 CUTLASS Distributed GEMM 的结构。

### 10.4 Attention 输入拆分

构造 Q producer 与 GQA consumer：

- K/V cache 预先准备好；
- consumer 在 wait 前执行 K/V TMA load；
- Q load 前执行 grid wait；
- 比较普通 graph dependency、programmatic edge 和 kernel fusion。

### 10.5 更现实的性能指标

除总时间外，建议记录：

- producer trigger 到 consumer 首 CTA 开始的延迟；
- producer tail 与 consumer preamble 的真实重叠区间；
- active warps、eligible warps 和 SM occupancy；
- HBM、L2、TMA、tensor-core 利用率；
- graph replay 稳态与首次 instantiate 成本；
- 不同 grid size、cluster size 和 SM resource footprint；
- PDL 开启后 producer 自身是否变慢。

## 11. 最终判断

这种 pattern 最适合被定义为：

> **稳定、重复执行的 CUDA Graph DAG 中，用 programmatic kernel edge 把 producer 的非关键 tail、consumer 的独立 prefix 和 node launch gap 重叠起来。**

它不是普通 multi-stream concurrency 的替代品，也不是逐 tile producer-consumer pipeline。它的生产价值主要集中在：

- CUDA Graph 中的小 dependent-kernel chain；
- GEMM mainloop/epilogue 到 successor fusion；
- norm/reduction 到多个 projection GEMM 的 fan-out；
- attention 中 dependent Q 与 independent K/V load 的拆分；
- multi-GPU barrier/communication 与 local GEMM；
- MoE routing/all-to-all 到 expert GEMM。

现有公开证据表明，生产实现已经开始采用 graph-node PDL，但接口仍较新、性能高度依赖具体 kernel 结构和资源竞争。当前实验的 `2x` 结果证明了机制上限；OpenXLA 和 CUTLASS 则证明这种机制已经进入真实编译器和分布式 GEMM 代码路径。
