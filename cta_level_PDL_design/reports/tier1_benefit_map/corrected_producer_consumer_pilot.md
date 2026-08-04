# Corrected CTA Producer–Consumer Pilot：完整实验报告

日期：2026-08-04  
实验日期：2026-08-03（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0  
证据等级：**P、C 均不超过 SM 数、低资源、合成 producer–consumer 负载的机制可行性证据；不是实际 LLM、GEMM、Attention 或 DSA 的端到端收益结论**

## 1. 执行摘要

本报告完整描述 corrected CTA producer–consumer pilot 已经执行的全部实验，而不是只描述 tail=0 单点。

一次 timed sample 只包含两个 kernel launch：

1. 一个 producer kernel，grid 中有 P 个 producer CTA；
2. 一个 consumer kernel，grid 中有 C 个 consumer CTA。

默认配置为 P=C=148，每个 CTA 有 128 个线程。这里的 148 表示一个 kernel grid 中的 148 个 CTA（CUDA thread block），不是 148 个 kernel。grid=64 实验则是一个 64-CTA producer kernel 加一个 64-CTA consumer kernel。

完整矩阵包括：

- 8 个实验 family；
- 每个 family 使用 seed 101、202、303，共 24 个配置；
- 每个配置测试 5 种同步模式；
- 每种模式 3 次 warmup、31 次 timed repeat；
- 总计 8 × 3 × 5 × 31 = **3,720 个 timed samples**；
- 24 个配置中，每个配置有 4 个进入 correctness validation 的模式，共 **96 次独立的 per-mode validation invocation 显示 PASS**；
- 24 个 none 模式按设计不提供正确性保证。

最重要的结果是：

- tail=0、interval degree=8：标准 grid Floor 为 1.001568 ms，软件 CTA Impl 为 0.901792 ms，speedup 为 1.1105×，latency reduction 为 9.952%，只兑现 no-wait timing gap 的 26.150%。
- 默认 tail=1M cycles：interval degree 1–64 的 latency reduction 为 34.989%–36.082%，但这是有充分独立 producer tail 的合成场景。
- degree 从 1 增长到 64 没有出现 degree=32 附近的突降，但 Impl 相对 Ceiling 的额外时间从约 2.3 µs 增长到约 18.0 µs。
- strided degree=32 的 interval cover tightness 只有 0.2264；按 seed 配对后，exact-parent 枚举/等待相对 interval cover 可恢复约 20.2 µs。这支持把依赖结构和依赖度作为两个独立实验轴分别研究。
- grid=64 与 grid=148 的差异约 1 µs，只能说明当前 P、C 均不超过 SM 数的 underfilled 范围内没有明显 grid-size effect。

最终判断是：**B200 上的软件 CTA readiness 机制通过了受限 synthetic gate；真实应用收益仍然未知。**

## 2. 这不是一个真实应用

实验程序是一个专门构造的 CUDA 微基准：[bench/cta_dep_pilot.cu](../../bench/cta_dep_pilot.cu)。

它没有执行 GEMM、Attention、DSA 或 LLM inference。producer 和 consumer 的主要“计算”由固定周期的 spin_cycles 模拟：

- producer readiness work：模拟输出数据准备之前的计算；
- producer tail：模拟数据发布以后、与 consumer 所需数据无关的独立工作；
- consumer prologue：模拟 consumer 在真正需要依赖数据之前可先执行的独立工作；
- consumer epilogue：模拟 wait 之后的固定时长工作。

producer 真正写入的数据非常小。每个 producer CTA 的线程 0 写一个 float：

~~~text
data[producer_cta_id] = 2 × producer_cta_id + 1
~~~

timed consumer 在 wait 和固定 epilogue 之后，只读取一个代表性 parent。完整的 parent 集合检查放在独立、非 timed 的 validation invocation 中。

因此，本实验主要聚焦 launch eligibility、CTA readiness、flag polling 和依赖表示相关成本，但并未把这些因素彼此完全解耦。微基准本身仍有寄存器占用，flag polling 和 data 访问也会产生 cache/memory 流量；只是这些资源规模、访问特征和数据相关控制流不代表真实 GEMM、Attention、DSA 或 LLM 算子。

## 3. CUDA 执行层级

CTA 在本报告中等同于 CUDA thread block。一次 invocation 的层级如下：

~~~text
一次 timed invocation
├── Producer kernel launch
│   ├── Producer CTA 0      128 threads
│   ├── Producer CTA 1      128 threads
│   ├── ...
│   └── Producer CTA P-1    128 threads
└── Consumer kernel launch
    ├── Consumer CTA 0      128 threads
    ├── Consumer CTA 1      128 threads
    ├── ...
    └── Consumer CTA C-1    128 threads
~~~

默认 P=C=148，正好等于 B200 的 148 个 SM。producer kernel 的查询 occupancy 为 16 CTA/SM，consumer kernel 为 10 CTA/SM，因此这两个 grid 都远小于按 occupancy 计算的全卡理论驻留容量。实验 harness 强制 P、C 均不超过 148，所以本轮只覆盖一个 `P,C≤SM` 的受限 underfilled 构造。这个约束本身不表示 occupancy 意义上的“恰好一波”，不保证一个 CTA 固定映射到一个 SM，也不能推出 P 或 C 一旦超过 148 就必然形成额外 CTA 批次。

两个 kernel 使用同一条 non-blocking CUDA stream。consumer 通过 cudaLaunchKernelEx 启动，并带有 ProgrammaticStreamSerializationAllowed 属性，从而允许 programmatic dependent launch。

## 4. Producer kernel 的完整语义

producer CTA 的逻辑顺序为：

~~~text
可选的入口 trigger
    ↓
readiness work + per-CTA skew
    ↓
线程 0 写 data[cta]
    ↓
CTA 内 __syncthreads()
    ↓
软件模式以 device-scope release store 发布 done[cta]=1
    ↓
grid 模式在此处 trigger
    ↓
独立 tail work
~~~

### 4.1 Readiness work 与 seed

基础 ready 参数固定为 400,000 cycles。每个 producer CTA 根据 CTA id 和 seed 进入 8 个 skew bin 之一：

~~~text
delay = 400K + 400K × bucket / 8
bucket ∈ {0,1,2,3,4,5,6,7}
~~~

所以实际 readiness delay 为 400K、450K、500K、550K、600K、650K、700K 或 750K cycles。

seed 取 101、202、303。对于本轮 interval 和 strided pattern，seed 主要改变 producer CTA 的 readiness bin 分布；它不改变 interval/strided parent 映射公式。三个 seed 用于检查不同 readiness 排列下结果方向是否稳定。

### 4.2 Trigger 时机

- grid 模式：producer CTA 在数据写入并完成 CTA barrier 后 trigger。
- none、interval-spin、interval-backoff、exact-backoff：producer CTA 在 kernel 入口 trigger。

PDL 的 dependent grid 只有在所有 producer CTA 都 trigger 或退出后才获得 launch eligibility。因此：

- grid 模式要等所有 producer CTA 各自完成 readiness work 和数据写入后，consumer grid 才有资格启动；
- 软件 CTA 模式在 producer CTA 一进入 kernel 就 trigger，使 consumer grid 更早获得启动资格；
- consumer 启动以后，再用 per-CTA flag 等待自己真正依赖的数据。

这里的 trigger 只改变 dependent grid 的 launch eligibility，不发布 producer 数据，也不提供跨 grid 的 memory visibility。grid 模式的数据可见性来自 consumer 随后的 `cudaGridDependencySynchronize()`：它在 dependency point 等 producer grid 完成。软件模式的数据可见性则来自 `data` 写入之后的 release store 与 consumer 的 acquire load。获得 launch eligibility 也不保证 GPU 会立即派发 consumer CTA；本轮没有逐 CTA trace，实际提前派发和 overlap 只能由 timing 间接推断。

这意味着 Floor→Impl 的差异同时包含 trigger 时机变化和 wait/flag 协议变化，不能把全部差异都归因于“等待粒度从 grid 变成 CTA”。

## 5. Consumer kernel 的完整语义

consumer CTA 的逻辑顺序为：

~~~text
固定 prologue work
    ↓
none / grid / interval / exact dependency wait
    ↓
CTA 内 __syncthreads()
    ↓
仅 validation invocation：检查全部真实 parent
    ↓
固定 epilogue work
    ↓
线程 0 读取一个代表性 parent 并写 out[child]
~~~

prologue 固定为 200K cycles，epilogue 固定为 1M cycles。

epilogue 是 wait 之后的固定时长 placeholder，不是真实的、使用 parent 数据进行计算的 GEMM 或 Attention payload。它的价值是保证 timed post-wait 工作量为 O(1)，不随 degree 增长；这修复了原始 benchmark 把 O(degree) 普通计算混入同步成本的问题。

## 6. Release/acquire flag 为什么能表达 CTA readiness

软件模式为每个 producer CTA 分配一个全局内存 flag：done[p]。

producer CTA 发布数据：

~~~text
完成 data[p] 写入
    ↓
CTA 内 barrier
    ↓
device-scope release store：done[p] = 1
~~~

consumer CTA 等待 parent：

~~~text
consumer CTA 的线程 0
    ↓
device-scope acquire load done[p]
    ↓
若为 0，继续 spin 或 nanosleep backoff
    ↓
观察到 1 后等待 CTA 内 __syncthreads()
    ↓
CTA 继续执行 wait 后工作
~~~

release 保证 producer 在 flag 之前完成的写入，在 consumer 通过 acquire 观察到 flag=1 后可见。device scope 表示该排序关系跨 SM 生效。

interval-backoff 传给 `__nanosleep` 的 requested backoff 从 32 ns 开始，每次翻倍，上限为 1024 ns；这不是对实际精确休眠时长的保证。interval-spin 不 sleep，持续 acquire polling。exact-backoff 使用相同 requested backoff，但只等待真实 parent。

本轮的“正确运行”仅表示：在这 24 个 `P,C≤SM` synthetic 配置中，每个配置的四种应验证同步模式各执行一次额外 validation，共 96 次 invocation，均未观察到 flag 或 datum 错误。它不是对 3,720 个 timed samples 的逐次验证，也不是对任意 kernel、任意 grid size 或所有 CUDA 调度情形的形式化证明。

## 7. 五种实际测试模式

| 模式 | Producer trigger | Consumer 同步 | 在四点包夹中的角色 | 正确性 |
|---|---|---|---|---|
| none | kernel 入口 | 完全不等待 | unsafe Ceiling timing reference | 不背书 |
| grid | 数据 ready 后 | cudaGridDependencySynchronize() | 正确的标准 PDL Floor | 验证 |
| interval-spin | kernel 入口 | acquire 扫描 interval cover，紧轮询 | 协议诊断 | 验证 |
| interval-backoff | kernel 入口 | acquire 扫描 interval cover，指数退避 | 预先声明的 Impl | 验证 |
| exact-backoff | kernel 入口 | 枚举真实 parent 并 acquire 等待，指数退避 | 依赖表示诊断 | 验证 |

### 7.1 Floor

Floor 是标准 grid-level PDL：

~~~text
Producer: [ready + skew] [write + trigger] [independent tail]
Consumer:                                  [prologue] [grid wait] [epilogue]
~~~

所有 producer CTA trigger 后，consumer grid 可在 producer tail 完成前获得 launch eligibility；若 GPU 实际提前派发 consumer CTA，它们可以先执行 prologue，但 grid wait 要等整个 producer grid 完成后才能返回。PDL 不保证一定提前派发。

### 7.2 Impl

Impl 固定为 interval-backoff：

~~~text
Producer: [entry trigger] [ready + skew] [write + release flag] [tail]
Consumer:                 [prologue] [wait parent flags] [epilogue]
~~~

consumer CTA 不在软件 wait 中等待整个 producer grid，而是等待 interval cover 覆盖到的所有 producer flags。对于 interval structure，这些正好是真实 parent；对于 strided structure，cover 还包含许多 false dependencies。

### 7.3 Ceiling

Ceiling 为 none：

~~~text
Producer: [entry trigger] [ready + skew] [write] [tail]
Consumer:                 [prologue] [no wait] [epilogue]
~~~

它没有任何同步保证，可能在 producer 数据 ready 前越过 dependency point。即使固定 epilogue 让最终代表性读取在某些 invocation 中碰巧读到新数据，也不能赋予该路径正确性。它只提供当前 none 路径实际测得的经验性 timing reference。

此外，none 不仅省掉 consumer wait，也不会执行软件模式中的 producer release store，并把 producer trigger 从 ready 后移到入口。因此它是当前 benchmark 的 unsafe、经验性 no-wait timing reference，不是“只把同步延迟设为零”得到的理论上界。Ceiling 不是实现方案、不是硬件保证，也不应被称为绝对硬件上界。

## 8. 依赖图构造

依赖图由 [bench/common/dep_pattern.cuh](../../bench/common/dep_pattern.cuh) 在 device 上按需计算，没有物化 adjacency graph。单个 parent 查询使用闭式公式；连续 interval 的 `[lo,hi]` 也是 O(1) 计算。strided 的保守 interval cover 会枚举 d 个 parent 求最小/最大值，exact-backoff 同样枚举 d 个真实 parent，因此这两条路径整体不是 O(1)。

### 8.1 Interval

对于 P 个 producer、C 个 consumer、degree=d，consumer child j 的连续 parent window 起点为：

~~~text
lo(j) = floor((P - d) × j / (C - 1))
parents(j) = {lo(j), lo(j)+1, ..., lo(j)+d-1}
~~~

interval 表示只需保存 lo 和 hi，编码为 O(1)。本轮 interval d1、d8、d32、d64 的 effective degree 分别等于真实 degree，tightness 均为 1.0。

### 8.2 Strided

strided d32 保持真实 degree=32，但把 parent 分散到 producer grid：

~~~text
stride = floor(P / d)
parent(j,k) = (j + k × stride) mod P
~~~

在 P=148、d=32 时 stride=4。interval 模式不能直接表示离散集合，只能等待最小 parent 到最大 parent 的保守 cover。

本轮 strided d32 的：

- 真实 degree：32；
- interval effective degree：平均 141.76；
- interval tightness：先对每个 consumer 计算 `32 / interval_width`，再跨 consumer 取平均，结果为 0.2264。它不是 `32 / 141.76`；后者是“真实 degree / 平均 effective degree”，约为 0.2257。

因此 interval-backoff 会扫描许多不是真实 parent 的 flag。exact-backoff 则只等待 32 个真实 parent。

## 9. 固定参数与变化参数

### 9.1 固定环境与控制变量

| 参数 | 数值 | 含义 |
|---|---:|---|
| GPU | NVIDIA B200 | 单卡 |
| SM 数 | 148 | 默认 P=C=148 |
| Compute Capability | 10.0 | sm_100 |
| Threads/CTA | 128 | producer、consumer 相同 |
| Ready base | 400K cycles | producer 数据 ready 前工作 |
| Skew bins | 8 | readiness delay 为 400K–750K |
| Consumer prologue | 200K cycles | wait 前独立工作 |
| Consumer epilogue | 1M cycles | wait 后固定 placeholder |
| Default tail | 1M cycles | producer 发布后独立工作 |
| Default grid | P=C=148 | `P,C≤SM` 的 underfilled 受限构造 |
| Seeds | 101、202、303 | readiness skew 排列 |
| Warmup | 3 次/模式/配置 | 不纳入统计 |
| Timed repeats | 31 次/模式/配置 | 取中位数 |
| 测试模式 | 5 | none/grid/interval-spin/interval-backoff/exact-backoff |
| Impl | interval-backoff | 结果前预先声明 |

### 9.2 八个实验 family

| Family | P/C | Structure | Degree | Tail | 隔离的实验轴 |
|---|---:|---|---:|---:|---|
| interval d1 | 148/148 | interval | 1 | 1M | degree |
| interval d8 | 148/148 | interval | 8 | 1M | degree；默认点 |
| interval d32 | 148/148 | interval | 32 | 1M | degree |
| interval d64 | 148/148 | interval | 64 | 1M | degree |
| strided d32 | 148/148 | strided | 32 | 1M | structure/false cover |
| tail=0 d8 | 148/148 | interval | 8 | 0 | tail geometry |
| tail=2M d8 | 148/148 | interval | 8 | 2M | tail geometry |
| grid=64 d8 | 64/64 | interval | 8 | 1M | underfilled grid size |

每个 family 运行 3 个 seed，所以总配置数为 8 × 3 = 24。

## 10. 计时、验证与统计口径

### 10.1 Timed sample

每次 pilotOnce 先在 stream 上：

1. poison data 和 out；
2. 将 done flags 和 error 清零；
3. 记录 begin CUDA event；
4. 启动 producer kernel；
5. 启动 consumer kernel；
6. 记录并同步 end CUDA event。

因此 reported milliseconds 包含两个 kernel 的 device makespan，不包含 begin event 之前的 buffer reset。

每个模式先运行 3 次 warmup，再运行 31 个 timed samples。每个配置、每个模式的基本统计量是 31 次的中位数。

### 10.2 Correctness invocation

31 次 timing 完成后，除 none 外的四种应验证同步模式各额外运行一次 validate=true 的 invocation：

- 检查每个 consumer 的每一个真实 parent；
- 软件模式检查 acquire flag 已 ready；
- 所有应验证同步模式检查 data[parent] 等于预期值；
- 任一不一致将 error 置为 1，并使程序返回非零。

none 直接跳过验证。因此 JSON 中 all_valid=true 的准确含义是：每个配置的所有应验证模式都通过；它不表示 none 正确。

### 10.3 跨 seed 汇总

先对每个配置、每个模式取 31 次中位数；再对同一 family 的三个 seed 分别对 Floor、Ceiling、Impl 和派生指标取中位数。

跨 seed 汇总中的不同列可能由不同 seed 的中位值贡献，它代表 family 的逐指标稳健中心，不一定等同于某一个实际 seed 行。

### 10.4 Bootstrap

[tools/analyze_pilot.py](../../tools/analyze_pilot.py) 对每个 seed 配置的 Floor、Ceiling 和 Impl repeat samples 独立进行 10,000 次 deterministic nonparametric bootstrap，并报告 2.5%–97.5% 区间。

例如 interval d8、seed=202：

- Floor 95% CI：[1.408576, 1.409248] ms；
- Ceiling 95% CI：[0.898016, 0.898560] ms；
- Impl 95% CI：[0.901888, 0.902496] ms；
- Impl gain 95% CI：[35.937, 35.993]%；
- gap captured 95% CI：[99.167, 99.342]%。

这些很窄的区间只说明当前 session 的 repeat-level timer noise 很小，不代表跨 GPU、driver、运行日期或 workload 的外推置信区间。

### 10.5 指标公式

对于 latency 越低越好的指标：

~~~text
观测参考差距 Ref gap  = (Floor - Ceiling) / Floor
软件收益 Impl gain   = (Floor - Impl) / Floor
差距兑现 Gap captured = (Floor - Impl) / (Floor - Ceiling)
Speedup              = Floor / Impl
~~~

CSV/JSON 中该参考差距沿用字段名 `space_pct`，但它应解释为当前 benchmark 的经验性 Floor→unsafe Ceiling 差距，而不是理论空间。报告中的 gain 百分比表示 latency reduction；speedup 用倍数表示。

## 11. 八个 family 的主要结果

下表对每个 family 的三个 seed 逐指标取中位数。Impl 固定为 interval-backoff。

| Family | Floor ms | Ceiling ms | Impl ms | Speedup | Ref gap % | Impl gain % | Gap captured % |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval d1 | 1.409088 | 0.898368 | 0.900768 | 1.5645× | 36.243 | 36.082 | 99.555 |
| interval d8 | 1.409216 | 0.898368 | 0.902464 | 1.5616× | 36.250 | 35.961 | 99.204 |
| interval d32 | 1.409152 | 0.898368 | 0.908128 | 1.5517× | 36.251 | 35.555 | 98.040 |
| interval d64 | 1.409120 | 0.898016 | 0.916064 | 1.5382× | 36.277 | 34.989 | 96.469 |
| strided d32 | 1.409056 | 0.898176 | 0.936640 | 1.5043× | 36.253 | 33.523 | 92.470 |
| tail=0 d8 | 1.001568 | 0.619872 | 0.901792 | 1.1105× | 38.086 | 9.952 | 26.150 |
| tail=2M d8 | 1.918144 | 1.407936 | 1.408192 | 1.3620× | 26.586 | 26.578 | 99.994 |
| grid=64 d8 | 1.408544 | 0.897472 | 0.901952 | 1.5617× | 36.284 | 35.966 | 99.123 |

## 12. Degree 实验：d1、d8、d32、d64

degree sweep 只改变 interval parent 数，其余默认参数保持不变。

### 12.1 观测 no-wait reference gap 没有在 d32 附近消失

Floor 与 Ceiling 在 d1–d64 基本不变，Ref gap 都约为 36.24%–36.28%。因此在当前 O(1) timed post-wait payload 和连续 interval 结构下，没有观察到“degree 超过 32 后，经验性的 no-wait reference gap 归零”。

这不表示任意高 degree 都免费。它只说明 degree 本身没有在 1–64 范围内造成突发 cliff。

### 12.2 Impl 相对 no-wait reference 的额外时间随 degree 增长

按 seed 配对计算 Impl−Ceiling：

| Degree | Impl−Ceiling 的跨 seed 中位数 |
|---:|---:|
| 1 | 2.272 µs |
| 8 | 4.064 µs |
| 32 | 10.016 µs |
| 64 | 18.048 µs |

Impl−Ceiling 不是纯软件协议成本：Impl 必须等待真实 dependency readiness，而 Ceiling 完全不等待；该差值还可能包含 polling/backoff、地址与循环、调度等成本。interval-backoff 确实由 CTA 线程 0 顺序检查 cover 内 flag，因此上述增长趋势与 degree 增大带来的扫描工作增加一致，但本实验没有 trace 或硬件计数器，不能进一步拆分各项贡献。结果支持“额外时间渐增”，不支持“d32 突然失效”。

本轮没有实测 d128、d256 或 d8192，不能把该趋势外推到这些范围。

## 13. Structure 实验：interval d32 与 strided d32

两个 family 的真实 degree 都是 32，Floor 和 Ceiling 也都约为 1.409 ms 和 0.898 ms，所以它们拥有近似相同的观测 Floor→unsafe Ceiling timing gap。

差异来自 parent 结构：

- interval d32：32 个连续 parent，effective degree=32，tightness=1.0；
- strided d32：32 个离散 parent，interval cover 平均包含 141.76 个 entry，tightness=0.2264。

strided d32 的逐指标跨 seed 中位时间为：

- interval-backoff：0.936640 ms；
- exact-backoff：0.916192 ms；
- interval gap captured：92.470%。

对每个 seed 先计算派生指标、再取跨 seed 中位数：

- interval-backoff−exact-backoff 为 20.192 µs，即约 20.2 µs；
- exact gap captured 为 96.505%，即约 96.50%。

这里不能直接用两个“逐指标跨 seed 中位时间”相减代替配对统计；0.936640−0.916192=20.448 µs，与 20.192 µs 口径不同。

因此，**degree 和结构复杂度必须分开建模**。相同 degree 下，interval false cover 会带来可测的额外扫描和等待成本。

该单点同时包含 flag scan、逐 parent 的 closed-form 解码、地址计算和等待较晚 parent 的成本，不能把 20.2 µs 直接当成某个未来硬件 primitive 的纯假边 latency。

## 14. Tail 实验：0、1M、2M cycles

tail 是 producer 发布数据之后、与 consumer 所需数据无关的独立工作。

### 14.1 tail=0

~~~text
Producer: [ready + skew] [publish] [结束]
~~~

此时没有 post-ready producer work 可供 consumer 的 dependent phase 隐藏。

结果：

- Floor：1.001568 ms；
- Ceiling：0.619872 ms；
- Impl：0.901792 ms；
- speedup：1.1105×；
- latency reduction：9.952%；
- gap captured：26.150%。

tail=0 仍有约 9.95% gain，不等于“没有任何潜在重叠空间”。软件模式在 producer 入口 trigger，允许 200K-cycle consumer prologue 与 producer 的 400K–750K readiness work 潜在重叠；标准 Floor 要等所有 producer CTA 在数据写入和 CTA barrier 后 trigger，consumer grid 才有资格启动。timing 与该机制解释一致，但没有逐 CTA trace 证明实际时间线。

因此，一个与当前代码和 timing 相符的机制解释是：tail=0 的收益来自 early launch eligibility 与 prologue/readiness overlap，并同时包含软件 wait 协议变化；它不是纯粹的 post-ready CTA overlap。由于本轮没有逐 CTA trace，这是一项机制推断，不是对 overlap 时间线的直接观测。

### 14.2 tail=1M

默认点的 producer 在 publish 之后继续 1M cycles 独立工作。软件 CTA consumer 在自己的 parent ready 后即可越过 wait 并执行 epilogue，而 grid Floor 要等整个 producer grid tail 完成。

interval d8：

- Floor：1.409216 ms；
- Ceiling：0.898368 ms；
- Impl：0.902464 ms；
- latency reduction：35.961%；
- gap captured：99.204%。

该点接近 no-wait timing reference，是 tail-rich 合成上限行为，不能直接当作应用预测。

### 14.3 tail=2M

结果：

- Floor：1.918144 ms；
- Ceiling：1.407936 ms；
- Impl：1.408192 ms；
- latency reduction：26.578%；
- gap captured：99.994%。

tail=1M 和 2M 的绝对 Floor−Ceiling 都约为 510 µs。tail=2M 的相对 Ref gap 降到 26.586%，主要因为总时长分母变大，而不是绝对参考差距减少。

因此 tail 分析必须同时报告：

- 绝对参考差距（Floor−Ceiling）和实际收益（Floor−Impl）；
- 相对 latency reduction；
- gap captured。

## 15. Grid size 实验：64 与 148

grid64 d8 只把 P=C 从 148 改成 64，其余默认参数保持不变。

| Grid | Floor ms | Ceiling ms | Impl ms | Impl gain % | Gap captured % |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.408544 | 0.897472 | 0.901952 | 35.966 | 99.123 |
| 148 | 1.409216 | 0.898368 | 0.902464 | 35.961 | 99.204 |

差异约为 1 µs 或更小。可支持的结论仅是：在 64–148 CTA、`P,C≤SM` 的 underfilled 低资源范围内，没有观察到明显 grid-size effect。

不能据此推断 P、C 大于 SM 数的更大 grid，也不能推断资源受限时实际出现多轮驻留/调度的行为。

## 16. 重复性与异常样本

三个 seed 的方向性结果一致。各 family 的 gap captured 范围为：

| Family | 三个 seed 的 Gap captured 范围 |
|---|---:|
| interval d1 | 99.442%–99.574% |
| interval d8 | 98.992%–99.267% |
| interval d32 | 97.964%–98.244% |
| interval d64 | 96.326%–96.501% |
| strided d32 | 92.433%–92.713% |
| tail=0 d8 | 26.066%–26.205% |
| tail=2M d8 | 99.850%–100.000% |
| grid=64 d8 | 99.092%–99.199% |

3,720 个 timed samples 中，有 19 个 sample 比所属配置/模式的中位数高超过 0.5%，最大正向偏差约 1.54%；没有对应的低尾。分析没有删除这些样本，也没有把中位数改成均值。31 次中位数对少量单边高尾不敏感。

## 17. Correctness 证据的准确边界

本轮验证比原始 FAST benchmark 更严格：

- 每次 invocation poison data，并清零 done flags 和 error；out 也会重置，但 validation 不读取或检查 out，因此不把它作为正确性证据；
- 软件模式使用 device-scope release/acquire；
- separate validation 检查所有真实 parent；
- timed post-wait 普通 payload 保持 O(1)，不会把随 degree 增长的 parent 数据遍历混入其中；
- 任一应验证同步模式失败会返回非零。

96 个应验证的模式全部 PASS，说明当前配置下没有观察到同步或数据可见性错误。

但仍有以下边界：

1. none 没有验证，也不应验证为正确；
2. validation 是同一次程序运行中的额外 invocation，不是每个 timed repeat 都逐 parent 验证；
3. validation 由每个 consumer CTA 的线程 0 检查 flags 和全部真实 parent data，没有验证其他线程的读取或完整 out 数组；
4. 当前没有 corrected pilot 的逐 CTA launch/ready/wait-return trace；
5. 只覆盖 `P,C≤SM`、producer occupancy 16、consumer occupancy 10 的 underfilled 低资源 kernel；
6. 没有覆盖 `P,C>SM` 的更大 grid、资源受限时实际出现的多轮驻留/调度、occupancy 1–2、真实 register/smem 或 L2/DRAM 竞争；
7. fixed epilogue 只是 spin placeholder，不是实际依赖数据的算子主体。

## 18. 为什么原始 FAST producer–consumer 结果被弃用

旧实验位于 [bench/results_budget1h/](../../bench/results_budget1h/)，使用 [bench/cta_dep_bench.cu](../../bench/cta_dep_bench.cu)。它确实执行过 degree、structure、tail、protocol、smem 和 trace sweep，但不能用于 CTA benefit 结论。

### 18.1 Trigger 时机使 CTA wait 退化

旧 producer 先：

1. 写数据；
2. 发布 done[cta]；
3. 再 trigger；
4. 最后执行 tail。

dependent grid 要等所有 producer CTA trigger 后才有资格启动，因此 consumer 启动时全部 done flag 已发布。软件 CTA wait 只是在扫描已经 ready 的 flag，没有测到实际 readiness 等待。

### 18.2 Global counter 协议不正确

旧 WAIT_COUNTER 用“全局完成数量 ≥ hi+1”推断 parent 前缀 [0,hi] 已完成。CTA 可以乱序完成，高编号 CTA 能把全局数量推过阈值，而某个低编号真实 parent 仍未发布。

### 18.3 Timed payload 混入 O(degree) 工作

旧 consumer 在 timed 路径中遍历所有 parent 并读取数据，使普通计算量随 degree 增长，无法分离同步成本。

### 18.4 Correctness 与 driver 状态不可靠

旧 harness 没有逐轮 poison 数据；竞态可能读到上一轮相同值。即使输出 FAIL，程序仍无条件返回 0，driver 仍会写 .done，分析器还可能选择不正确的最快模式。

因此，本报告的所有有效数值只来自 [bench/results_budget1h_corrected/](../../bench/results_budget1h_corrected/) 中的 corrected pilot，不使用旧 FAST 的 benefit、boundary 或 protocol winner。

## 19. 能成立的结论

本轮证据支持：

1. 在 B200 上，`P,C≤SM`、低资源、规则 interval 依赖且有足够 producer 独立 tail 时，device-scope release/acquire 软件 CTA readiness 的 96 次独立 validation invocation 均未观察到 flag/data 错误，timing 接近 unsafe no-wait reference。
2. interval degree 从 1 增长到 64 没有出现 d32 附近的突发 cliff，但 Impl 相对 unsafe no-wait reference 的额外时间持续增长；该增长不能被本实验拆解为纯扫描成本。
3. strided d32 单点支持把依赖结构与 degree 分开建模；interval false cover 会产生可测的额外时间。
4. tail geometry 是影响观测收益的关键变量；tail=0 的 Impl 只覆盖约四分之一的 empirical no-wait reference gap。
5. grid 64 与 148 在当前 `P,C≤SM` 的 underfilled 范围内结果接近。
6. 当前数据足以支持进入一个更现实的小规模 applicability gate。

## 20. 不能成立的结论

本轮证据不支持：

1. 真实 LLM、GEMM、Attention 或 DSA 会获得约 35% 收益；
2. Ceiling 是可部署实现或可实现硬件上界；
3. Floor→Impl 的全部差异只来自 CTA 粒度；
4. degree 256、8192 或更高仍有相同开销；
5. `P,C>SM` 的更大 grid、资源受限时实际出现的多轮驻留/调度、occupancy 1–2 或真实资源竞争下仍有效；
6. 软件 polling 不会影响其他 kernel 的 L2/调度行为；
7. 当前已经直接 trace 到 consumer launch 早于 producer ready；
8. 当前 confidence interval 可以外推到其他 GPU、driver 或 workload。

因此本轮 gate 应写成：

> **Mechanism GO；Application Benefit UNKNOWN。**

## 21. 建议的下一步 applicability gate

只建议继续三个高信息量实验：

1. 给 corrected pilot 加逐 CTA 时间戳，直接记录 producer ready、consumer launch、wait return 和 dependent phase；
2. 测 P、C 分别为 2×SM 和 4×SM 的 grid，并把 producer/consumer occupancy 压到 2 和 1 CTA/SM；
3. 选择一个真实 tile kernel chain，只使用已经验证可 overlap 的 same-stream 或 CUDA Graph programmatic 路径。

只有当 resource-realistic、grid size 至少达到 2×SM 的正确 Impl 对 grid Floor 仍稳定获得约 5%–8% 以上收益，且 trace 直接确认 overlap，才应进入 LLM/DSA 集成。

## 22. 附录 A：全部 24 个配置

字段说明：

- Eff. degree：interval 实现实际等待的平均 entry 数；
- Tightness：每个 consumer 的“真实 degree / interval width”再跨 consumer 取平均；
- Ref gap：Floor→unsafe Ceiling 的经验性 timing 差距；原始 CSV/JSON 字段名为 `space_pct`；
- Gain：Floor→Impl 的 latency reduction；
- Captured：Impl 恢复的 Floor→Ceiling gap 比例。

| Tag | Seed | Structure | Degree | Eff. degree | Tightness | P | C | Tail cycles | Floor ms | Ceiling ms | Impl ms | Speedup | Ref gap % | Gain % | Captured % |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid64_d8_s101 | 101 | interval | 8 | 8.00 | 1.0000 | 64 | 64 | 1000000 | 1.408544 | 0.897472 | 0.901952 | 1.5617× | 36.284 | 35.966 | 99.123 |
| grid64_d8_s202 | 202 | interval | 8 | 8.00 | 1.0000 | 64 | 64 | 1000000 | 1.408800 | 0.897664 | 0.901760 | 1.5623× | 36.282 | 35.991 | 99.199 |
| grid64_d8_s303 | 303 | interval | 8 | 8.00 | 1.0000 | 64 | 64 | 1000000 | 1.408544 | 0.897344 | 0.901984 | 1.5616× | 36.293 | 35.963 | 99.092 |
| interval_d1_s101 | 101 | interval | 1 | 1.00 | 1.0000 | 148 | 148 | 1000000 | 1.408864 | 0.898304 | 0.901152 | 1.5634× | 36.239 | 36.037 | 99.442 |
| interval_d1_s202 | 202 | interval | 1 | 1.00 | 1.0000 | 148 | 148 | 1000000 | 1.409088 | 0.898368 | 0.900544 | 1.5647× | 36.245 | 36.090 | 99.574 |
| interval_d1_s303 | 303 | interval | 1 | 1.00 | 1.0000 | 148 | 148 | 1000000 | 1.409248 | 0.898496 | 0.900768 | 1.5645× | 36.243 | 36.082 | 99.555 |
| interval_d32_s101 | 101 | interval | 32 | 32.00 | 1.0000 | 148 | 148 | 1000000 | 1.409152 | 0.898112 | 0.908128 | 1.5517× | 36.266 | 35.555 | 98.040 |
| interval_d32_s202 | 202 | interval | 32 | 32.00 | 1.0000 | 148 | 148 | 1000000 | 1.409088 | 0.898944 | 0.907904 | 1.5520× | 36.204 | 35.568 | 98.244 |
| interval_d32_s303 | 303 | interval | 32 | 32.00 | 1.0000 | 148 | 148 | 1000000 | 1.409216 | 0.898368 | 0.908768 | 1.5507× | 36.251 | 35.513 | 97.964 |
| interval_d64_s101 | 101 | interval | 64 | 64.00 | 1.0000 | 148 | 148 | 1000000 | 1.409088 | 0.898016 | 0.916064 | 1.5382× | 36.270 | 34.989 | 96.469 |
| interval_d64_s202 | 202 | interval | 64 | 64.00 | 1.0000 | 148 | 148 | 1000000 | 1.409408 | 0.898112 | 0.916000 | 1.5387× | 36.277 | 35.008 | 96.501 |
| interval_d64_s303 | 303 | interval | 64 | 64.00 | 1.0000 | 148 | 148 | 1000000 | 1.409120 | 0.897792 | 0.916576 | 1.5374× | 36.287 | 34.954 | 96.326 |
| interval_d8_s101 | 101 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 1000000 | 1.409216 | 0.897888 | 0.903040 | 1.5605× | 36.285 | 35.919 | 98.992 |
| interval_d8_s202 | 202 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 1000000 | 1.408960 | 0.898368 | 0.902112 | 1.5618× | 36.239 | 35.973 | 99.267 |
| interval_d8_s303 | 303 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 1000000 | 1.409248 | 0.898400 | 0.902464 | 1.5616× | 36.250 | 35.961 | 99.204 |
| strided_d32_s101 | 101 | strided | 32 | 141.76 | 0.2264 | 148 | 148 | 1000000 | 1.409280 | 0.898528 | 0.935744 | 1.5061× | 36.242 | 33.601 | 92.713 |
| strided_d32_s202 | 202 | strided | 32 | 141.76 | 0.2264 | 148 | 148 | 1000000 | 1.409056 | 0.898176 | 0.936832 | 1.5041× | 36.257 | 33.514 | 92.433 |
| strided_d32_s303 | 303 | strided | 32 | 141.76 | 0.2264 | 148 | 148 | 1000000 | 1.408960 | 0.898176 | 0.936640 | 1.5043× | 36.253 | 33.523 | 92.470 |
| tail0_d8_s101 | 101 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 0 | 1.001600 | 0.620416 | 0.901920 | 1.1105× | 38.058 | 9.952 | 26.150 |
| tail0_d8_s202 | 202 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 0 | 1.001184 | 0.619872 | 0.901792 | 1.1102× | 38.086 | 9.927 | 26.066 |
| tail0_d8_s303 | 303 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 0 | 1.001568 | 0.619712 | 0.901504 | 1.1110× | 38.126 | 9.991 | 26.205 |
| tail2m_d8_s101 | 101 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 2000000 | 1.917632 | 1.407936 | 1.407968 | 1.3620× | 26.579 | 26.578 | 99.994 |
| tail2m_d8_s202 | 202 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 2000000 | 1.918144 | 1.408192 | 1.408192 | 1.3621× | 26.586 | 26.586 | 100.000 |
| tail2m_d8_s303 | 303 | interval | 8 | 8.00 | 1.0000 | 148 | 148 | 2000000 | 1.918304 | 1.407936 | 1.408704 | 1.3618× | 26.605 | 26.565 | 99.850 |

## 23. 附录 B：证据与复核入口

- Corrected benchmark 源码：[bench/cta_dep_pilot.cu](../../bench/cta_dep_pilot.cu)
- 依赖图公式：[bench/common/dep_pattern.cuh](../../bench/common/dep_pattern.cuh)
- 3,720 个原始 timed samples：[bench/results_budget1h_corrected/pilot_matrix.log](../../bench/results_budget1h_corrected/pilot_matrix.log)
- 24 个配置及 bootstrap CI：[bench/results_budget1h_corrected/pilot_summary.csv](../../bench/results_budget1h_corrected/pilot_summary.csv)
- Family 聚合和逐模式中位数：[bench/results_budget1h_corrected/pilot_analysis.json](../../bench/results_budget1h_corrected/pilot_analysis.json)
- 统计分析脚本：[tools/analyze_pilot.py](../../tools/analyze_pilot.py)
- B200 完整实验总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)
- 被弃用的原 FAST 汇总：[bench/results_budget1h/summary.txt](../../bench/results_budget1h/summary.txt)
- 数据与源码 SHA-256 清单：[EXPERIMENT_MANIFEST_SHA256.txt](../../EXPERIMENT_MANIFEST_SHA256.txt)

本报告只重新组织和解释已有实验，不重新运行 GPU、不修改原始数据，也不改变任何 CUDA 源码。
