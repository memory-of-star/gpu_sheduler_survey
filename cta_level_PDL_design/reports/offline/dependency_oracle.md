# 真实候选负载 CTA 依赖 Oracle 实验报告

日期：2026-08-03（UTC）  
执行设备：CPU-only，不使用 GPU  
实验状态：**有效的解析式 tile-decomposition oracle；不是实际 kernel trace 或性能实验**

## 1. 执行摘要

该实验根据预设的 kernel tile decomposition，在 CPU 上解析 producer CTA 写区间与 consumer CTA 读区间的交集，构造 CTA-level dependency graph，并计算 degree、interval tightness、false-edge rate 和编码条目数。

完成了 7 个模型/长度配置：

- Qwen3.6-27B：seq 4K、32K、128K；
- DeepSeek-v3.2 DSA：seq 32K、1M；
- GLM-5.2 DSA/IndexShare：seq 32K、1M。

最重要结果：

- Qwen RMSNorm→gate/up GEMM：degree=128，tightness=1.0，false edges=0%。
- DeepSeek/GLM DSA indexer→topk：32K 时 degree=256，1M 时 degree=8192，但都为精确连续 interval，false edges=0%。
- DSA topk→sparse attention：真实 inter-kernel RAW edge 只是每个 query block 的 index，degree=1。
- GLM IndexShare 到后续 L0–L3 attention：span 为 1–4 个 kernel，但 CTA dependency 仍为一对一 interval。

结论：**在这些解析模型中，高 dependency degree 不等于高结构复杂度；degree=8192 仍可用两个 interval endpoints 精确表达。**

## 2. 这是什么、又不是什么

分析脚本：[tools/dep_oracle.py](../../tools/dep_oracle.py)。

它是：

- 根据源码中显式定义的 tile/element ranges 做解析计算；
- 用 producer write range 与 consumer read range 相交定义 parent；
- 对大 grid 均匀抽样最多 256 个 consumer；
- 输出结构统计和编码条目数。

它不是：

- GPU kernel 执行；
- 对真实 CUTLASS/Triton binary 的反汇编或 instrumentation；
- runtime CTA scheduling trace；
- latency/speedup 测量；
- 对某个部署模型真实 tile shape 的自动发现。

结果的成立前提是脚本中的 tile decomposition 与目标实现一致。

## 3. 指标定义

对 consumer j 的真实 parent 集合 `P(j)`：

~~~text
degree(j) = |P(j)|
width(j)  = max(P(j)) - min(P(j)) + 1
tightness(j) = degree(j) / width(j)
~~~

汇总指标：

~~~text
interval_tightness_mean = mean_j(tightness(j))
false_edge_rate = 1 - sum_j(degree(j)) / sum_j(width(j))
exact adjacency entries = sum_j(degree(j))
interval entries = 2 × sampled consumers
~~~

`tightness=1.0` 表示 parent id 构成连续区间，`[lo,hi]` 不引入 false dependency。

JSON 中的 exact/interval entry 数都只针对本次实际分析的 consumer 集合；大 grid 时即 256 个 sampled consumers，不是整个真实 grid 的总 metadata 容量。storage ratio 在规则映射下仍可用于比较两种表示的相对规模。

## 4. 七个运行配置

| Run | Model | Tokens | Sequence | Kernel pairs |
|---|---|---:|---:|---:|
| qwen_4k | Qwen3.6-27B | 4096 | 4096 | 2 |
| qwen_32k | Qwen3.6-27B | 4096 | 32768 | 2 |
| qwen_128k | Qwen3.6-27B | 4096 | 131072 | 2 |
| deepseek_32k | DeepSeek-v3.2 DSA | 4096 | 32768 | 2 |
| deepseek_1m | DeepSeek-v3.2 DSA | 4096 | 1048576 | 2 |
| glm_32k | GLM-5.2 DSA | 4096 | 32768 | 6 |
| glm_1m | GLM-5.2 DSA | 4096 | 1048576 | 6 |

## 5. Qwen 结果

### 5.1 DeltaNet intra-chunk→inter-chunk scan

chunk size 固定为 64。consumer chunk c 读取 `[max(0,c-1), c]` 两个或一个 partial state。

| Seq | Producer/consumer CTA | Sampled consumers | Mean degree | Mean width | Tightness | False edges |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 64/64 | 64，完整枚举 | 1.984375 | 1.984375 | 1.0 | 0% |
| 32K | 512/512 | 256 | 1.996094 | 1.996094 | 1.0 | 0% |
| 128K | 2048/2048 | 256 | 1.996094 | 1.996094 | 1.0 | 0% |

这是低 degree、连续 interval 的 chain pattern。

### 5.2 RMSNorm→gate/up GEMM

脚本使用：

- tokens=4096；
- hidden=5120；
- intermediate=17408；
- GEMM BM=128、BN=128；
- RMSNorm 一 token 一 CTA；
- GEMM grid 为 32×136=4352 CTA。

每个 GEMM output CTA 需要 128 个 token row 对应的 RMSNorm producer CTA：

| Degree | Mean width | Tightness | False edges | Exact entries | Interval entries | Interval/exact |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1.0 | 0% | 32768 | 512 | 0.015625× |

三个 seq 运行中的 FFN tokens 参数都为 4096，因此这一 pair 的结果相同。

## 6. DeepSeek/GLM DSA 结果

脚本使用：

- key block=128；
- query block=64；
- top-k=2048；
- indexer grid 为 query blocks × key blocks。

### 6.1 Indexer→topk

topk consumer 需要同一 query block 的整行 score tiles。

| Seq | Producer CTA | Consumer CTA | Sampled | Degree | Width | Tightness | False edges | Interval/exact storage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 131072 | 512 | 256 | 256 | 256 | 1.0 | 0% | 0.0078125× |
| 1M | 134217728 | 16384 | 256 | 8192 | 8192 | 1.0 | 0% | 0.000244141× |

这里 degree 很高，但 parent id 是一个连续 score-row interval。

### 6.2 Topk→sparse attention

脚本把历史 KV 与当前 topk 输出区分开。topk 只在本轮产生 index；KV 来自更早的 decode step。因此当前 kernel pair 的直接 RAW dependency 是本 query block 的单个 index entry：

| Seq | Degree | Width | Tightness | False edges | Interval/exact storage |
|---:|---:|---:|---:|---:|---:|
| 32K | 1 | 1 | 1.0 | 0% | 2× |
| 1M | 1 | 1 | 1.0 | 0% | 2× |

degree=1 时，用两个 interval endpoints 反而比单个 adjacency entry 多一个条目；这只是条目计数，不包含对齐和 metadata。

## 7. GLM IndexShare 结果

GLM 模型中，一个 shared topk index 被 L0–L3 四个 attention kernel 使用，逻辑 span 为 1–4 个 kernel。

| Pair | Span | Degree | Width | Tightness | False edges |
|---|---:|---:|---:|---:|---:|
| topk→attn L0 | 1 | 1 | 1 | 1.0 | 0% |
| topk→attn L1 | 2 | 1 | 1 | 1.0 | 0% |
| topk→attn L2 | 3 | 1 | 1 | 1.0 | 0% |
| topk→attn L3 | 4 | 1 | 1 | 1.0 | 0% |

32K 与 1M 的结构统计相同，只是 producer/consumer grid 分别为 512 和 16384 CTA。

## 8. Sampling 边界

除 Qwen DeltaNet 4K 的 64 个 consumer 完整枚举外，大 grid 最多均匀抽样 256 个 consumer。

对脚本中这些规则、平移不变的 interval mapping，抽样结果很稳定；但这仍不是对所有 consumer 的 runtime instrumentation。若真实 kernel 有边界 tile、load balancing、split-K、ragged batch 或 data-dependent mapping，必须用真实 launch metadata 或 instrumentation 复核。

## 9. 能成立的结论

本实验支持：

1. 这组解析 tile model 中存在 degree 128、256、8192 但 tightness=1.0 的依赖。
2. degree 和 interval structure 应作为不同实验轴。
3. DSA topk→attention 的直接当前轮次依赖可比“间接 KV gather”表面上看到的结构简单。
4. GLM IndexShare 提供 span>1、degree=1 的规则依赖样例。

## 10. 不能成立的结论

本实验不支持：

1. 这些真实模型已经在 GPU 上获得 CTA-level PDL 性能收益。
2. 实际部署 binary 必然采用脚本中的 tile shape。
3. degree=8192 的软件 flag polling 成本可接受。
4. interval metadata、调度、同步和 cache 成本为零。
5. 所有 data-dependent 或间接访问都能解析成 interval。

## 11. 证据入口

- 分析脚本：[tools/dep_oracle.py](../../tools/dep_oracle.py)
- Qwen 4K：[bench/results_budget1h_corrected/oracle_qwen_4k.json](../../bench/results_budget1h_corrected/oracle_qwen_4k.json)
- Qwen 32K：[bench/results_budget1h_corrected/oracle_qwen_32k.json](../../bench/results_budget1h_corrected/oracle_qwen_32k.json)
- Qwen 128K：[bench/results_budget1h_corrected/oracle_qwen_128k.json](../../bench/results_budget1h_corrected/oracle_qwen_128k.json)
- DeepSeek 32K：[bench/results_budget1h_corrected/oracle_deepseek_32k.json](../../bench/results_budget1h_corrected/oracle_deepseek_32k.json)
- DeepSeek 1M：[bench/results_budget1h_corrected/oracle_deepseek_1m.json](../../bench/results_budget1h_corrected/oracle_deepseek_1m.json)
- GLM 32K：[bench/results_budget1h_corrected/oracle_glm_32k.json](../../bench/results_budget1h_corrected/oracle_glm_32k.json)
- GLM 1M：[bench/results_budget1h_corrected/oracle_glm_1m.json](../../bench/results_budget1h_corrected/oracle_glm_1m.json)
- 总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)
