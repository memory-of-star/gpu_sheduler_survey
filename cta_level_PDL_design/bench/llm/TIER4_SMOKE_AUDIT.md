# Tier 4 persistent-driver smoke 审计

所有 smoke timing 都是诊断数据，不进入正式统计。前四次失败被保留而没有覆盖；第五次首次完成真实 Qwen/vLLM 机制证明。schema-v2 smoke 将使用新的结果根并固定输出 `admissible=false`。

| smoke | 结果 | 主要发现 |
|---|---|---|
| smoke1 | REJECTED | vLLM 环境 cache 已冻结，grid 复用了 off AOT callable；off 112/112 PTX/cubin、grid 0/0、ceiling 仅 helper。 |
| smoke2 | REJECTED | reset 发生在 MM shell，真正的 Qwen compile wrapper 位于嵌套 `language_model.model`。 |
| smoke3 | REJECTED | 已独立编译 grid，但 GDN 第二次 trace 需要 terminal endpoint `max_num_batched_tokens + 1`。 |
| smoke4 | REJECTED | FULL capture 触发首次 trace，得到 decode-specialized callable，真实请求的 prefill 阶段失败。 |
| smoke5 | MECHANISM PASS | 每档先做 NONE/non-uniform general trace 再 capture FULL decode；三档 PTX/cubin、NVTX graph-node 和 worker cohort 全部通过。 |

smoke5 保留位置：

- 结果：`/tmp/cta_tier4_persistent_smoke5_20260805`
- Nsight report：`/tmp/cta_tier4_persistent_smoke5_20260805_profile.nsys-rep`
- SQLite：`/tmp/cta_tier4_persistent_smoke5_20260805_profile.sqlite`
- triplet：`8d4dd5c36a474d9993f550c4e20e6e0a`
- driver/worker PID：`130575`

smoke5 的独立 cache 扫描结果：

| rung | PTX | cubin | wait | launch |
|---|---:|---:|---:|---:|
| `pdl_off` | 113 | 113 | 0 | 0 |
| `pdl_grid` | 106 | 106 | 132 | 126 |
| `ceiling` | 106 | 106 | 0 | 126 |

三档均为 `sm_100a`，每个 PTX 均有同 stem cubin。精确 NVTX 窗口依次为 `[32726169,2660165669]`、`[2662604566,2741759816]`、`[2743652722,2824874878]`；每档窗口包含 2700 个 kernel，其中 1105 个带 graph node id，且全部能在 `CUDA_GRAPH_NODE_EVENTS` 找到。PTX entry 到同档 graph node 的匹配数为 off 21、grid wait 21 / launch 20、ceiling launch 20。

smoke5 使用 schema-v2 之前的 raw 格式，因此只证明机制，不会被 formal finalizer 接纳。下一次真模型 smoke 由 `run_tier4_schema_v2_smoke.sh` 执行：全新 root、同一 persistent process、1 repeat / 1 warmup、decode 与 4K prefill plumbing、独立 FULL-decode proof、完整 schema-v2 identity/evidence/finalize 路径。只有该 smoke 通过后才运行正式矩阵。
