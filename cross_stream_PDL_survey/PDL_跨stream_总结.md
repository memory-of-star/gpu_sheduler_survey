# PDL 对跨 Stream 的支持 —— 阶段总结

> 主题：确认 CUDA Programmatic Dependent Launch (PDL) 到底能不能把上游 kernel 的
> **tail** 和下游 kernel 的 **prologue** 重叠起来，特别是**跨 stream / 跨 node** 的场景。
> 测试平台：NVIDIA H100 NVL（SM=132，SM clock≈1.78 GHz，CC 9.0）。目标平台：B300 (`sm_103`)。

---

## 0. 一句话结论

- **同流 PDL（stream serialization）**：eager 模式下就能 overlap，稳定 **2×**。
- **跨流 PDL（programmatic event + `cudaStreamWaitEvent`）**：在 **eager 模式下拿不到收益**（实测 1.00×，退化成普通 event）。
- **同一段跨流代码，一旦被 capture 成 CUDA Graph（或直接用图的 programmatic 边），立刻 overlap，达到 2×。**
- 结论：**PDL 的"提前触发"语义要靠 CUDA Graph（instantiation）才能真正保留并跨 stream/跨 node 生效**；纯 eager 的跨流 `cudaStreamWaitEvent` 只能等到上游整体完成。

---

## 1. PDL 的三条软件路径（务必区分）

底层 PTX 都是同一对指令：
- `griddepcontrol.launch_dependents`：**producer** 在 tail 之前"放行"下游（`cudaTriggerProgrammaticLaunchCompletion()`）。
- `griddepcontrol.wait`：**consumer** 在 prologue 之后、真正用到上游数据之前等待（`cudaGridDependencySynchronize()`）。

| 路径 | 主机侧接口 | 生效条件（实测） |
|---|---|---|
| **同流串行化** | `cudaLaunchAttributeProgrammaticStreamSerialization` | **eager 即生效** |
| **跨流 programmatic event** | `cudaLaunchAttributeProgrammaticEvent` + `cudaStreamWaitEvent` | eager **不 overlap**；被 capture 成图后才 overlap |
| **图内 programmatic 边** | `cudaGraphDependencyTypeProgrammatic` + `cudaGraphKernelNodePortProgrammatic` | 图内直接生效 |

> 接口细节（含 PTX 9.3 vs 9.4 差异、CLC 部分、Rubin tile-level triggering 硬件设计）见
> 姊妹主题 `../cta_level_PDL_design/docs/cuda_13.4_pdl_clc_interfaces.md`。

---

## 2. 实验设计

### 2.1 `pdl_bench.cu` —— 单 producer → 单 consumer，6 种模式
producer / consumer 都有可调的 `tail` / `prologue` 自旋周期，人为放大重叠收益。

| 模式 | 含义 |
|---|---|
| `BASE` | 跨流 + **普通 event**（基线，无重叠） |
| `PDL_XS` | 跨流 + **programmatic event**（eager） |
| `PDL_CAPTURE` | 把 `PDL_XS` 那套 `sA`/`sB` 两流代码 **stream-capture 成图**再回放 |
| `PDL_GRAPH` | 直接建图，producer→consumer 用 **programmatic 边** |
| `PDL_SS` | **同流** + stream serialization |
| `CONC` | 无依赖（并发天花板，结果会算错，仅作上界） |

### 2.2 `pdl_diamond.cu` —— 菱形图 `producer → {midA, midB} → final`
对比"**普通边**"与"**programmatic 边**"两种建图方式，验证 fan-out 场景下 PDL 的收益。

---

## 3. 实测结果（H100，tail=prologue≈11.2 ms，blocks=132 threads=128）

### 3.1 pdl_bench（wall-clock 中位数）

| 模式 | median (ms) | vs BASE | 是否 overlap |
|---|---|---|---|
| BASE(xstream, ordinary-event) | 22.5 | 1.00× | 否 |
| **PDL_XS(xstream, prog-event, eager)** | **22.5** | **1.00×** | **否** |
| PDL_CAPTURE(sA/sB captured→graph) | 11.2 | 2.00× | 是 |
| PDL_GRAPH(prog-edge, built) | 11.2 | 2.00× | 是 |
| PDL_SS(same-stream, serialize) | 11.2 | 2.00× | 是 |
| CONC(no-dep, ceiling) | 11.2 | 2.00× | 是（但算错） |

**结论**：只有 `PDL_XS`（eager 跨流）停在 1×，其余全部到 2× 天花板。
`PDL_CAPTURE` 与 `PDL_XS` 是**同一段代码**，唯一区别是"有没有被 capture 成图"——这是"图才解锁 overlap"的对照实验铁证。

### 3.2 diamond

| variant | median (ms) | 说明 |
|---|---|---|
| DIAMOND_PLAIN(ordinary edges) | 44.9 | 每层串行 |
| DIAMOND_PDL(programmatic edges) | 22.5 | 层间重叠，**2.00×** |

---

## 4. nsys node 级验证（关键证据）

采集时用了 `--cuda-graph-trace=node`，能看到图内每个 node 的起止。

### 4.1 pdl_bench 六模式逐 kernel 时间条（见 `pdl_bench_modes.png`）

| 模式 | producer 硬件流 | consumer 硬件流 | producer 与 consumer 起点差 | 总时长 |
|---|---|---|---|---|
| BASE | 13 | 14 | 11.23 ms（首尾相接） | 22.4 ms |
| **PDL_XS** | **13** | **14** | **11.21 ms（零重叠）** | **22.4 ms** |
| PDL_SS | 13 | 13 | 0.00 ms | 11.3 ms |
| CONC | 13 | 14 | 0.00 ms | 11.2 ms |
| PDL_GRAPH | 13 | 13 | 0.00 ms | 11.3 ms |
| PDL_CAPTURE | 13 | 13 | 0.00 ms | 11.2 ms |

**要点**：`PDL_XS` 的 profiler 记录明确显示 producer 在硬件流 **13**、consumer 在流 **14**——
`sA`/`sB` 两条流确实分到了不同队列，但两者仍然**首尾相接零重叠**。
所以"eager 跨流没收益"**不是流没分开**，而是 programmatic event 的提前触发语义在 eager 下被丢掉，
`cudaStreamWaitEvent(sB, progEvent)` 退化成"等 producer 整体完成"的普通 barrier。

### 4.2 diamond node 级（见 `diamond_node_kernels.png`）
`midA` / `midB` 被分到**不同的内部 lane**（stream 13 vs 15/17）横向并行；PDL 边让相邻层纵向重叠。

---

## 5. nsys 的 stream id 到底是什么（重要澄清）

1. **nsys/CUPTI 的 `streamId` 是软件 `CUstream` 的逻辑编号（per context）**，不是硬件队列号。
   - 普通 launch：`sA→13`、`sB→14` 一一对应，不同 id ⟺ 不同软件流。
2. **同 id ≠ 串行**：`PDL_GRAPH` / `PDL_CAPTURE` 里 producer 和 consumer **都是 stream 13**，却完全重叠——
   因为它们是图节点，重叠靠 `griddepcontrol` 握手，与"在不在同一条流"无关。
3. 图实例化时的 lane 分派规律：
   - **独立、可并行**的节点 → 不同 lane（不同 stream id），如 diamond 的 `midA`/`midB`。
   - **链式依赖**的节点 → 同一 lane（同一 stream id），如 producer→consumer。
4. **streamId 与硬件 command queue 是"软件→有限硬件通道"的多对少映射**：
   ```
   kernel → CUstream(nsys streamId, 逻辑) → 驱动映射到 M 个硬件通道(HyperQ/connections, 默认8, 可到32)
          → GPU 前端(Host Scheduler / CWD) → SM 实际并发
   ```
   - 流比通道多时会 round-robin 复用同一通道（"假依赖"）。→ **不同 stream id ≠ 一定不同硬件队列**。
   - 同一通道也能并发（由前端 + SM 资源决定）。→ **nsys streamId ≠ 硬件队列**。
   - 可用 `CUDA_DEVICE_MAX_CONNECTIONS` 调硬件通道数。
5. **两种并发机制的区分**：
   - **fan-out 并行**（diamond `midA∥midB`）：不同 lane，"横向"铺开。
   - **PDL 流水**（producer→consumer 同 lane）：同 lane，"纵向"重叠（consumer 的 prologue 压进 producer 的 tail）。

---

## 6. 为什么 eager 跨流拿不到收益（机理推断）

`cudaLaunchAttributeProgrammaticEvent` 的"提前触发"语义需要在**图的构建/instantiation**阶段被解析成
node 间的 programmatic 依赖（GPU 端用 semaphore/依赖解析在前端调度）。
eager 模式下 `cudaStreamWaitEvent` 只能把它当作一个普通 event：等到 producer **整体完成**才 record，
于是 consumer 的 prologue 无法与 producer 的 tail 重叠，行为等同 `BASE`。
—— 这也解释了为何官方把 programmatic event 定位成"给 stream capture 用"。

---

## 7. 目录结构与材料索引

本主题（`cross_stream_PDL_survey/`）聚焦**跨 stream / 跨 node 的 PDL 收益实测**。
PDL/CLC 接口、PTX 观察、以及 **Rubin CTA/tile-level triggering 的硬件设计**已拆到姊妹主题
`../cta_level_PDL_design/`。

```
gpu_sheduler_survey/
├── cross_stream_PDL_survey/        # 【本主题】跨 stream PDL 收益实测
│   ├── PDL_跨stream_总结.md        #   本文（阶段总结，入口）
│   ├── bench/                      #   基准测试
│   │   ├── pdl_bench/              #     源码：pdl_bench.cu(6模式) + pdl_diamond.cu + build.sh/run.sh/README.md
│   │   ├── pdl_bench_h100/         #     H100(sm_90) 离线包：源码 + 预编译静态二进制 + 脚本
│   │   ├── pdl_bench.tar.gz        #     迁移用打包（B300/sm_103）
│   │   └── pdl_bench_h100.tar.gz   #     迁移用打包（H100/sm_90）
│   ├── profiles/                   #   nsys 采集
│   │   ├── pdl_bench.nsys-rep / .sqlite      # 6 模式 node 级
│   │   ├── diamond_pdl.nsys-rep / .sqlite    # 菱形图 graph 级
│   │   └── diamond_node.nsys-rep / .sqlite   # 菱形图 node 级
│   ├── figures/                    #   图
│   │   ├── pdl_bench_modes.png     #     6 模式 producer/consumer 时间条（§4.1）
│   │   ├── diamond_node_kernels.png#     菱形图逐 kernel 时间线（§4.2）
│   │   ├── diamond_pdl_timeline.png#     菱形图 graph 级时间线
│   │   └── diamond_pdl_model.png   #     菱形图理论模型
│   └── tools/                      #   可视化脚本
│       ├── nsys_plot.py            #     从 .sqlite 生成 PNG（自动识别 node/graph 级）
│       └── nsys_viz.py             #     Web 可视化（5010 端口）
└── cta_level_PDL_design/           # 【姊妹主题】Rubin 新特性：CTA/tile-level triggering 设计
    ├── docs/
    │   └── cuda_13.4_pdl_clc_interfaces.md   # PDL+CLC 全部 PTX/CUDA 接口，9.3 vs 9.4 差异，硬件设计，§3.4 Rubin tile-level triggering
    └── ptx_study/                  # griddepcontrol PTX 观察 + 9.3/9.4 对比（新特性未进公开 PTX 的证据）
        ├── pdl_demo.cu / pdl_demo_real.cu / pdl_graph.cu / pdl_streams.cu   # 最小样例
        ├── *_cuda133_sm103.ptx     #   PTX 9.3 产物
        ├── *_cuda134_sm103.ptx     #   PTX 9.4 产物
        └── build_cuda134.sh / build_cuda134.log / nvrtc_compile.py         # Docker 内 13.4 工具链编译
```

---

## 8. 复现要点

```bash
# 目标机（B300 / sm_103 或 H100 / sm_90）
cd bench/pdl_bench && ./build.sh && ./run.sh      # 源码编译 + 跑 6 模式 + diamond
# 或用 H100 离线包内预编译二进制
cd bench/pdl_bench_h100 && ./run.sh

# profile（node 级，能看图内每个 node）
nsys profile --cuda-graph-trace=node -o pdl_bench ./pdl_bench ...

# 分析 / 出图（脚本用 argv 传 sqlite 路径，PNG 输出到当前目录）
nsys export --type sqlite -o profiles/pdl_bench.sqlite profiles/pdl_bench.nsys-rep
python3 tools/nsys_plot.py profiles/pdl_bench.sqlite       # 生成 PNG
python3 tools/nsys_viz.py  profiles/diamond_node.sqlite    # 或起 Web 可视化(:5010)
```
