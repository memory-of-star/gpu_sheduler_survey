# DSA 架构模型的 CTA 级依赖分析（DeepSeek-V3.2 / GLM-5.x）

> Tier 5 的分析产出。配套脚本：[`../bench/dsa/`](../bench/dsa/)、[`../tools/dep_oracle.py`](../tools/dep_oracle.py)
>
> **一句话结论**：DSA 看起来是间接访存的困难场景，但它的 **kernel 间 RAW 依赖链其实是规整的**，
> 对 CTA 级依赖友好。真正的困难场景是 MoE 的 dispatch/combine。

---

## 1. 为什么单卡跑不了整模型

| 模型 | 参数量 | FP8 | FP4 | 单卡 B300 (288GB) | 单卡 B200 (192GB) |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | 671B | 671 GB | 335 GB | 装不下 | 装不下 |
| GLM-5.2 | 744B / 40B-active | 744 GB | 372 GB | 装不下 | 装不下 |

且 MoE 要求**全部 expert 常驻**（路由是逐 token 的，无法按需换入），最少需 2 卡（FP4）到 8 卡（FP8）。

**但不必退化为纯纸面分析**：DSA 的注意力层参数量很小，**单层与算子链可以在单卡上用真实 shape 实测**。分三层处理：

| 层次 | 手段 | 可得结论 |
|---|---|---|
| 架构级依赖图 | 纯离线推导（`dep_oracle.py`） | 依赖度、区间紧度、假边率、编码成本 |
| 单层 / 算子链 | 单卡真实 shape 实测（`bench/dsa/`） | 三档包夹、收益随上下文长度的变化 |
| 整模型端到端 | 纸面外推 | TPS/user、EP/TP 跨卡依赖 |

---

## 2. 架构参数（已核实）

### GLM-5.2（`GlmMoeDsaForCausalLM`）

| 项 | 值 |
|---|---|
| 参数量 | 744B 总 / 40B 激活 |
| 层数 | 78（前 3 层 dense FFN，其余 MoE） |
| hidden | 6144 |
| MoE | 256 expert，选 8 routed + 1 shared |
| `kv_lora_rank` | 512 |
| `q_lora_rank` | 2048 |
| `index_head_dim` | 128 |
| `index_n_heads` | 32 |
| `index_topk` | 2048 |
| 上下文 | 200K（GLM-5）/ 1M（GLM-5.2） |
| 特有 | **IndexShare**：每 4 层共享一个 indexer |

### DeepSeek-V3.2

| 项 | 值 |
|---|---|
| 参数量 | 671B |
| DSA | lightning indexer（64 heads，FP8，ReLU）+ top-k 选择 |
| `index_topk` | 2048 |
| 基座 | MLA 的 MQA 模式（每个 latent 被所有 query head 共享） |
| 特点 | **每层独立 indexer**，无跨层共享 |

---

## 3. DSA 的 kernel 链与依赖形态

```
lightning indexer  →  top-k selection  →  sparse MLA attention
```

### 3.1 indexer → topk：依赖度极高但结构极规整

topk 处理 query 块 *j* 时需要该 *j* 在 key 方向的**全部**得分，因此依赖 indexer 中该 query 块的**整行 CTA**。

`dep_oracle.py` 的推导结果：

| 上下文 | 生产者 CTA 数 | 依赖度 | 区间紧度 | 假边率 | 区间 vs 精确邻接表 |
|---|---|---|---|---|---|
| 32K | 131,072 | 256 | **1.0000** | 0.00% | 0.008× |
| 1M | 134,217,728 | **8192** | **1.0000** | 0.00% | 0.0002× |

**这是"高依赖度 ≠ 复杂结构"最极端的样本**：1M 上下文下依赖度是 8192，是 BlockMaestro 那条 "degree > 32 收益归零" 阈值的 **256 倍**，但区间编码依然**完全精确**（紧度 1.0、假边率 0），存储比精确邻接表省 4000 倍以上。

若照搬 BlockMaestro 的阈值，这个模式会被直接排除。**这是 E1 维度要求"依赖度与结构复杂度必须分开测"的最强证据。**

### 3.2 topk → sparse attention：看似间接访存，实则 RAW 依赖链规整

sparse attention 读两类数据：

| 数据 | 来源 | 是否构成 kernel 间 RAW 依赖 |
|---|---|---|
| 索引数组 `idx` | 本步 topk 产生 | **是**，按 query 块 **1-to-1** |
| 被选中的 KV 条目 `c_s` | **更早的 decode step 或 prefill 写入** | **否** |

**间接寻址决定的是"读历史数据的哪个位置"，并不构成对紧邻前驱的不可预测依赖。**

`dep_oracle.py` 验证：依赖度 1，紧度 1.0，假边率 0%。

#### 两个后果

**其一，DSA 对 CTA 级依赖是友好的**，不是想象中的困难场景。

**其二，BlockMaestro 的 Algorithm 1 会在这里错误地保守退化。** 它的判据是"地址是否来源于 global load"（第 7–9 行，撞上就 END），无法区分：

- 间接读的是**本步产出** → 真依赖，必须保守
- 间接读的是**历史数据** → 无 kernel 间 RAW 依赖，可以细粒度化

**改进方向**：判据应从"地址来源"改为"**数据的产生时间**"。具体地，对一条间接 load `A[B[i]]`，除了追踪 `B` 的来源，还要判断 `A` 是否被紧邻前驱 kernel 写过——若前驱不写 `A`，这条间接访存不产生 kernel 间依赖。这在命令队列层面是可判定的（比对各 kernel 的指针参数），成本极低。

已写入设计空间报告 A2 维度。

### 3.3 GLM-5.2 的 IndexShare 把依赖跨度拉长

每 4 层共享一个 indexer，top-k 索引在 4 层内复用：

```
indexer(L1) → attn(L1) → attn(L2) → attn(L3) → attn(L4)
                span=1     span=2     span=3     span=4
```

**这是 A1 维度"跨度 > 1"的真实样本**，也是 `prologue_inspector` 设计的理想场景——该设计的硬约束是"结构数组不能由紧邻的生产者 kernel 写"（见 [`prologue_inspector_cta_pdl.md`](../design_brainstorm/prologue_inspector_cta_pdl.md) §9），而 IndexShare 让索引数组由**数个 kernel 之前**的算子产生，天然满足。

对 span = 2/3/4 的层，`idx` 早已就绪，消费者 CTA 可以在 prologue 里安全地读它、算出自己依赖哪些 KV 区段。

IndexShare 的另一个效果：1M 上下文下 per-token FLOPs 降低 2.9×，意味着 indexer 在总时间中的占比下降，attention 链的占比上升——**收益结构随上下文长度变化**，这正是 §5 要扫描的。

### 3.4 MoE 的 dispatch/combine 才是真正的困难场景

```
router → top-8 → permute/gather → grouped GEMM → unpermute/scatter
```

| 环节 | 依赖形态 | 能否静态/inspector 处理 |
|---|---|---|
| router → permute | permute 的索引由**紧邻前驱 router** 产生 | **否**（结构动态） |
| permute → grouped GEMM | 每个 CTA 的依赖度 = 该 expert 分到的 token 数 / tile | 运行时才确定 |
| grouped GEMM → unpermute | 反向 scatter | 同上 |

这是"结构动态"（评估方案 §7.3 的情况 B），prologue inspector 救不了。TRT-LLM 已在 MoE routing 上用 PDL，可作对照基线。

---

## 4. 三类模式的对照总表

| 模式 | 依赖度 | 区间紧度 | 跨度 | CTA 级友好度 | 备注 |
|---|---|---|---|---|---|
| DSA indexer → topk | 极高（256–8192） | **1.0** | 1 | **高** | 高度但规整，区间编码精确 |
| DSA topk → attn | 1 | **1.0** | 1 | **高** | 间接访存但 RAW 链规整 |
| IndexShare topk → attn(L2-4) | 1 | **1.0** | **2–4** | **高** | 跨度 >1 的真实样本 |
| MoE router → grouped GEMM | 运行时定 | 低 | 1 | **低** | 结构动态，唯一的困难点 |

---

## 5. 单卡可实测的部分

配套脚本 [`../bench/dsa/run_dsa_chain.sh`](../bench/dsa/run_dsa_chain.sh)。

| 实验 | 配置 | 目的 |
|---|---|---|
| `indexer → topk → sparse MLA` 三算子链 | 真实 shape（hidden 6144、`index_topk` 2048、`index_n_heads` 32） | 三档包夹 |
| 上下文扫描 | 4K / 32K / 128K / 1M | indexer 的 O(L²) 使长上下文下该链占比急剧上升，收益空间随之变化 |
| 单层 MLA + DSA 完整 forward | 单层权重仅数百 MB | 层级收益 |
| MoE dispatch/combine | **缩减 expert 数（32 而非 256）** | 复现依赖形态而不需全量权重 |

**缩减 expert 数是关键技巧**：MoE 的依赖**形态**由 top-k 路由决定，与 expert 总数无关；用 32 个 expert 就能复现 dispatch/combine 的依赖结构，权重量降到 1/8，单卡轻松放下。

---

## 6. 只能纸面外推的部分

- 整模型端到端 TPS/user
- EP / TP 并行下的跨卡依赖（超出本项目单卡范围）
- IndexShare 在真实 1M 上下文下的端到端收益（需多卡）

---

## 7. 对设计空间各维度的输入

| 维度 | DSA 提供的证据 |
|---|---|
| **A1 跨度** | IndexShare 提供了 span = 2/3/4 的真实样本，证明"跨度 > 1"不是假想需求 |
| **A2 来源** | 暴露了 BlockMaestro 判据的缺陷，给出"按产生时间判定"的改进方向 |
| **A3 编码** | 区间编码在 DSA 上精确且省 4000× 存储，是"参数化模式模板"最有力的支持证据 |
| **D1 降级** | MoE 是必须降级的场景，验证"逐级降级"相对"全有全无"的价值 |
| **E1 边界** | 依赖度 8192 但紧度 1.0，直接证伪"依赖度 > 32 即无收益"的普适性 |
