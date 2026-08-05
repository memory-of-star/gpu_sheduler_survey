# Tier 5 DSA 算子链：语义与可承载性拒绝审计报告

报告日期：2026-08-05（UTC）  
审计日期：2026-08-05（UTC）  
审计环境：单卡 NVIDIA B200，148 SM，Compute Capability 10.0；准入审计已执行，**本轮未运行正式 Tier 5 timing**  
证据等级：**REJECTED-TIMING / STATIC-AUDIT**——可支持 harness 不准入、显式张量尺寸和离线依赖图结论；不能支持任何 Floor/Impl/Ceiling 性能数字

## 1. 执行摘要

原 `bench/dsa` 路径不能实现 [`EXPERIMENT_PLAN.md`](../../EXPERIMENT_PLAN.md) §1、§3、§9
要求的三档包夹：旧 Floor 只在普通同-stream PyTorch 算子之间执行 host
`torch.cuda.synchronize()`，旧 Ceiling 只删除该 host wait，GPU 上的 stream 顺序并未删除；
代码也没有 CTA/query-row Impl、全量正确性校验、同进程相邻测量或置信区间。

因此本轮没有尝试“修饰”旧时间数，而是把入口改成 fail-closed：

- `--audit-only` 只计算语义和显存证据，写机器可读 JSON，成功退出；
- 任何 timing 请求返回 2；
- driver 完成离线 oracle 和逐 shape 审计后仍返回 2，并写失败记录；
- 不输出任何可被汇总工具误收的 timing `SUMMARY`。

正式 4K/32K/128K/1M timing 均**未运行**。本轮保留的是
[`bench/dsa/results_20260805_b200_audit/`](../../bench/dsa/results_20260805_b200_audit/)
下的准入 JSON、离线 oracle 与 fail-closed 日志，而不是原始 timing 日志。
旧实现可能产生或曾意图产生的 DSA/MoE 时间，全部不可接受，不得进入报告、gate 或设计推荐。

## 2. 程序实际执行的语义

### 2.1 被拒绝的旧 timing 路径

旧 DSA 链为 `indexer → topk → sparse MLA`，但两个 rung 的 GPU 顺序相同：

| 档位 | 旧程序实际操作 | 为什么不满足计划 |
|---|---|---|
| Floor | 同一 CUDA stream 中按序提交算子，并在算子间 host synchronize | 没有 programmatic dependent launch，也没有 device `griddepcontrol.wait`，不是真实 grid-PDL |
| Impl | 不存在 | 没有按 query/CTA identity 发布和等待 readiness 的路径 |
| Ceiling | 删除 host synchronize，仍在同一 CUDA stream 中按序提交 | GPU 依赖顺序没有删除，不是无依赖并发 Ceiling |

旧 runner 还为每个 rung 启动独立进程，默认只有 10 次 timing；`verified=1` 是按 rung 写出的
标签，不是 poison 后的全量参考校验。MoE 中的 `int(mask.sum())` 会在每个 expert 上引入
device→host 同步。逐项代码级审计见
[`bench/dsa/SEMANTIC_AUDIT.md`](../../bench/dsa/SEMANTIC_AUDIT.md)。

### 2.2 当前 fail-closed 路径

当前 [`bench/dsa/dsa_chain.py`](../../bench/dsa/dsa_chain.py) 不分配旧大张量、不启动 DSA
timing kernel，只从 shape 公式计算每个显式张量的字节数，查询可见设备容量，并输出：

- `status=blocked`、`runnable=false`、`measurement_emitted=false`；
- 六项 DSA 语义 blocker，MoE 另加 host-scalar-sync blocker；
- 单个张量超过可见设备容量时的 allocation blocker；
- 解锁 Floor/Impl/Ceiling、validation 和 statistics 的最低契约。

当前 [`bench/dsa/run_dsa_chain.sh`](../../bench/dsa/run_dsa_chain.sh) 先运行纯离线 dependency
oracle，再对 FAST 或 formal shape 调用上述审计，最后以退出码 2 明确拒绝 timing。

### 2.3 可保留的离线 oracle

[`tools/dep_oracle.py`](../../tools/dep_oracle.py) 构造符号化 CTA 图，不启动 GPU timing：

- `indexer → topk`：每个 query block 依赖同一行的全部 key tiles；degree 高，但 parent 是连续区间；
- `topk → sparse attention`：紧邻 RAW 依赖只在本步生成的 `idx` 上，按 query block 为 1-to-1；
- GLM IndexShare：同一份索引跨多个 attention 层复用，形成 span > 1 的离线样本。

这些结论只描述 oracle 假设下的依赖形态。它们与旧 timing 相互独立，可以作为 A2/结构分析
输入，但不能证明真实 kernel 的 CTA 映射、PDL 语义或端到端收益。

## 3. 配置、统计与本轮执行范围

审计采用计划中的 GLM-5.2 形状参数：

| 参数 | 值 |
|---|---:|
| hidden | 6144 |
| `kv_lora_rank` | 512 |
| `q_lora_rank` | 2048 |
| `index_head_dim` | 128 |
| `index_n_heads` | 32 |
| `index_topk` | 2048 |
| FAST context | 4K |
| formal contexts | 4K / 32K / 128K / 1M |

轻量自检覆盖了 Python 语法、shell 语法、help、单点 audit、FAST driver 和四个 formal shape
的审计路径。随后在 B200 上以
`RESULTS=results_20260805_b200_audit ./run_dsa_chain.sh` 执行完整准入审计：四个 DSA
shape、两个 MoE shape 和四个 oracle 点均写入
证据，runner 按设计返回 2。预期且实得的 fail-closed 性质为：审计 JSON 标记 blocked，且
整个目录没有 timing `SUMMARY`。

统计项必须明确写为：

| 项 | 本轮值 |
|---|---:|
| Warmup timing | 0 |
| Timed repeats | 0 |
| Floor/Impl/Ceiling samples | 0 / 0 / 0 |
| 中位数 / CI | 不存在 |
| 正式 GPU timing | **未运行** |

临时 smoke 输出只用于确认 fail-closed 控制流，没有作为实验原始结果保留，也没有从中提取
任何性能数值。

## 4. 显存公式与 shape 结论复算

旧 indexer 显式产生 `(S, H_idx, S)` BF16，其中 `H_idx=32`；最终 scores 为 `(S,S)`
BF16；旧 sparse gather 显式产生 `(S,K,R)` BF16，其中 `K=2048`、`R=512`。因此：

~~~text
indexer_ths_bytes = S × 32 × S × 2
scores_bytes      = S × S × 2
sparse_gather     = S × 2048 × 512 × 2
~~~

以二进制 K（4K=`2^12`、32K=`2^15`、128K=`2^17`、1M=`2^20`）复算：

| S | indexer `(S,32,S)` | scores `(S,S)` | gather `(S,2048,512)` |
|---:|---:|---:|---:|
| 4K | `2^(12+5+12+1)=2^30` B = 1 GiB | `2^25` B = 0.03125 GiB | `2^(12+11+9+1)=2^33` B = 8 GiB |
| 32K | `2^36` B = 64 GiB | `2^31` B = 2 GiB | `2^36` B = 64 GiB |
| 128K | `2^40` B = 1 TiB | `2^35` B = 32 GiB | `2^38` B = 256 GiB |
| 1M | `2^46` B = 64 TiB | `2^41` B = 2 TiB | `2^41` B = 2 TiB |

本轮 B200 可见显存约 179 GiB。因此 128K 的单个 1 TiB indexer 张量和 256 GiB gather、
1M 的三个 TiB 级张量，任一个都足以证明旧完整实现不可承载。32K 表中三个张量合计已为
130 GiB，但这不是严格 peak 估计；还未包含输入、投影结果、输出、算子临时区和 allocator
碎片，所以不能据此声称 32K 一定可运行。4K 的显式张量小于设备容量，也不能修复包夹
语义缺失。

Streaming/tiled 实现可以降低物化显存，但不能删掉计划规定的完整 `O(S^2)` indexer 工作量。
若只处理少量 query 行，结果必须标为 sampled proxy，不能填充 §9.3 的完整-forward 矩阵。

## 5. 解除 BLOCKED 的最低契约

未来 native harness 只有同时满足以下条件，才能产生可报告 Tier 5 timing：

1. **Floor**：后继 launch 使用 programmatic-stream-serialization 属性，在首次依赖读取前执行
   device `griddepcontrol.wait`；生产者在真实数据就绪点 trigger。
2. **Impl（若声称）**：按 query/CTA identity 发布 readiness，trigger 发生在 readiness 之前；
   不能用无身份的全局完成数量替代真实 parent。
3. **Ceiling**：前后 kernel 之间没有 stream order、event 或 wait，确实允许无依赖并发；结果
   按设计错误，只计时并自报 `verified=0`。
4. 每个 timed repeat 前 poison 全部中间量和输出；Floor/Impl 另做不计时的完整参考校验，覆盖
   全部真实依赖，任一错误使整个配置非零退出。
5. 三档在同一进程、每次重复相邻执行；每档至少 31 个样本，报告中位数与置信区间。
6. 4K/32K/128K/1M 的完整工作量必须可承载，或在运行前机器可读地拒绝；不得用缩小 query
   数量的 proxy 冒充正式点。

这些要求落实了 [`AGENTS.md`](../../AGENTS.md) §4 的真实 Floor、错误 Ceiling、校验、统计和
触发时机规则；超时止损或“能够启动”都不能代替准入证明。

## 6. 能成立的结论

本审计支持：

1. 旧 Floor 不是真实 grid-PDL，旧 Ceiling 不是真正无依赖并发，两者不能形成性能包夹。
2. 旧路径缺少 Impl、全量校验、同进程相邻 ≥31 次重复和置信区间。
3. 128K/1M 的旧完整张量实现确定超出本轮单卡容量；“1M 可能 OOM”的表述不够严格。
4. 当前入口会在任何 timing 前 fail-closed，不会生成可误用的 Tier 5 时间汇总。
5. 离线 oracle 的符号依赖形态分析不受旧 timing 语义错误影响，可独立保留。

## 7. 不能成立的结论

本审计不支持：

1. 任何 DSA 或 MoE 的 Floor、Impl、Ceiling latency、吞吐、speedup 或 headroom 数值。
2. Grid-level PDL 在 DSA 上有益、无益或与 CTA-level 机制等价。
3. 4K 或 32K 因显存公式较小就一定能够完成有效测量。
4. Streaming/tiled 实现必然能在预算内完成 128K/1M 完整 forward。
5. 离线 oracle 的 CTA 图就是某个具体 TileLang、FlashMLA 或框架 kernel 的实测 ground truth。
6. 缩减 expert 数的 MoE timing 可以在当前旧 harness 中代表生产 MoE；该 timing 路径同样被拒绝。

## 8. 证据入口

- 详细准入审计：[bench/dsa/SEMANTIC_AUDIT.md](../../bench/dsa/SEMANTIC_AUDIT.md)
- 机器可读审计入口：[bench/dsa/dsa_chain.py](../../bench/dsa/dsa_chain.py)
- Fail-closed driver：[bench/dsa/run_dsa_chain.sh](../../bench/dsa/run_dsa_chain.sh)
- B200 准入审计总日志：[bench/dsa/results_20260805_b200_audit/dsa.log](../../bench/dsa/results_20260805_b200_audit/dsa.log)
- B200 设备记录：[bench/dsa/results_20260805_b200_audit/device.txt](../../bench/dsa/results_20260805_b200_audit/device.txt)
- 128K 机器可读拒绝证据：[bench/dsa/results_20260805_b200_audit/audit_dsa_seq131072.json](../../bench/dsa/results_20260805_b200_audit/audit_dsa_seq131072.json)
- 1M 机器可读拒绝证据：[bench/dsa/results_20260805_b200_audit/audit_dsa_seq1048576.json](../../bench/dsa/results_20260805_b200_audit/audit_dsa_seq1048576.json)
- 离线依赖 oracle：[tools/dep_oracle.py](../../tools/dep_oracle.py)
- Tier 5 实验规格：[EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md)
- Benchmark 有效性规则：[AGENTS.md](../../AGENTS.md)
- 本轮机器与实验坐标声明：[codex/state/coordinates.md](../../codex/state/coordinates.md)

本轮没有正式 Tier 5 timing 样本或原始 timing 日志；`results_20260805_b200_audit/` 只含
准入/离线证据。缺少 timing 入口是有意的 fail-closed 结果，不是证据遗失。
