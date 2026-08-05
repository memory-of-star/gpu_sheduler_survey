# Tier 5 DSA harness 准入审计

> **历史范围 / superseded：**本文只审计旧 `dsa_chain.py` PyTorch 三档路径；该路径继续
> 永久 `BLOCKED`，其 timing 仍为 0。当前实验入口已由 `dsa_native.cu` +
> `run_dsa_chain.sh` 的 native strict 路径，以及 `production_tier5.py` +
> `run_production_tier5_fragments.sh` 的 production-component 路径取代。新路径是否正式
> 准入必须读取各自结果目录的最终 admission/validation 和
> [`EXPERIMENT_REPORT_INDEX.md`](../../EXPERIMENT_REPORT_INDEX.md)，不能用本文的历史
> `BLOCKED` 结论覆盖，也不能反过来复活旧路径 timing。

审计日期：2026-08-05（UTC）  
结论：**BLOCKED / DO NOT TIME**。`dsa_chain.py` 现在只输出机器可读的阻塞证据；普通
测量请求返回 2，且绝不输出可被汇总器误收的 `SUMMARY` 时间记录。

## 1. 旧三档为什么不是包夹

旧 `Floor` 在普通 PyTorch 算子之间调用 `torch.cuda.synchronize()`。这些算子本来就在
同一个 CUDA stream 中按序执行，host synchronize 只让 CPU 等待，不会把后继 kernel
变成 `cudaLaunchAttributeProgrammaticStreamSerialization` launch，也没有在 device 侧执行
`griddepcontrol.wait`，因此不是计划规定的 grid-PDL Floor。

旧 `Ceiling` 只是删除上述 host synchronize。后继 PyTorch kernel 仍在同一 stream 中排在
前驱之后，GPU 依赖顺序完全没有删除，因此不是“真正无依赖并发”。它还把 Floor 和
Ceiling 放在不同进程中测量，默认仅 10 次，没有置信区间。

旧代码没有 CTA/query-row readiness 的 Impl，也没有 poison 后对全部依赖输出做独立参考
校验；`verified=1` 只是按 rung 写入的标签，不是校验结果。MoE 路径的
`int(mask.sum())` 还在每个 expert 上强制 device→host 同步。

## 2. 完整 shape 的显式张量下界

旧 indexer 的第一个 einsum 显式产生 `(S, 32, S)` BF16，sparse gather 显式产生
`(S, 2048, 512)` BF16。下表只是**单个显式张量**大小，不是包含输入、临时量和 allocator
碎片的峰值估计：

| S | `(S,32,S)` | scores `(S,S)` | gather `(S,2048,512)` |
|---:|---:|---:|---:|
| 4K | 1 GiB | 0.03125 GiB | 8 GiB |
| 32K | 64 GiB | 2 GiB | 64 GiB |
| 128K | 1 TiB | 32 GiB | 256 GiB |
| 1M | 64 TiB | 2 TiB | 2 TiB |

所以“1M 可能 OOM”过于宽松：在约 179 GiB 的本轮 B200 上，128K/1M 的旧完整实现由
单个张量就能确定不可承载。改成 streaming/tiled 实现虽可降低内存，但仍须执行完整
`O(S^2)` indexer；若只抽取少量 query 行，必须自报为 sampled proxy，不能冒充计划 §9.3
要求的完整 forward。

`dsa_chain.py --audit-only` 按输入 shape 重算这些字节数，并在 JSON 中记录可见 GPU、所有
语义阻塞项、超过设备容量的单个张量和未来 replacement contract。

## 3. 解除 BLOCKED 的最低契约

未来 native harness 必须在一个进程内、每次重复相邻执行：

1. **Floor**：后继 launch 带 programmatic-stream-serialization 属性，kernel 在首次依赖读
   之前执行真正的 device `griddepcontrol.wait`；生产者在正确的数据就绪点 trigger。
2. **Impl（若声称）**：按 query/CTA identity 发布 readiness，不能用全局完成数量代替身份；
   trigger 必须在 readiness 之前，使等待实际发生。
3. **Ceiling**：前后 kernel 之间没有 stream order、event 或 wait，允许真正并发；输出按
   设计错误，只计时且必须自报 `verified=0`。
4. 每个 timed repeat 前 poison 所有中间量和输出；Floor/Impl 另跑不进计时的完整参考
   校验，覆盖全部真实依赖，任一错误使整个配置非零退出。
5. 每档至少 31 个样本，报告中位数和置信区间；所有档使用相同 shape、工作量与进程。
6. 4K/32K/128K/1M 的完整工作量都要么可承载，要么在运行前给出机器可读的资源拒绝；
   不允许用缩小 query 数的 proxy 填充正式矩阵。

## 4. 当前仍可使用的部分

`run_dsa_chain.sh` 仍执行 `tools/dep_oracle.py` 的纯离线依赖图推导。该部分不依赖旧 timing
语义，可以继续作为 A2/依赖形态证据。除此之外，旧 DSA/MoE 时间数均不得进入报告或 gate。

轻量检查：

```bash
python3 dsa_chain.py --audit-only --seq 4096 --json /tmp/dsa-audit.json
FAST=1 RESULTS=/tmp/dsa-audit-run ./run_dsa_chain.sh  # 预期最终 exit 2
```
