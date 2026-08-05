# Tier 4 初始拒收审计（已解除，历史记录）

本文件保留最初 `BLOCKED / REJECTED` 决策的审计轨迹，但该决策已被后续环境补齐和真实模型 smoke 证明取代，**不能再作为当前环境状态**。初始拒收没有贡献任何正式 timing 样本。

初始缺口包括：Qwen3.6-27B 未落盘、Nsight Systems 不可用、三档分别运行在不同进程、没有 fresh-cache PTX/cubin 语义证明、没有把 PTX entry 连接到实际 FULL-decode CUDA graph node、Ceiling patch 未证明进入 worker。当前对应处置如下：

- 模型已完整落盘到 `/workspace/models/Qwen3.6-27B`；15 个 safetensors shard 由 index 完整枚举。
- `nsys` 已安装；每档使用精确 `TIER4_PROOF|rung=...|triplet=...` NVTX 窗口，并与 `CUDA_GRAPH_NODE_EVENTS` 联结。
- `tier4_driver.py` 在一个 persistent vLLM process/worker 内保留三套独立编译 callable 和 FULL-decode graph，再按 Latin-3 顺序相邻切换。
- 三档使用隔离 fresh cache，独立验证 `off=0/0`、`grid=>0/>0`、`ceiling=0/>0` 的 wait/launch PTX 语义、同 stem cubin、`sm_100*` target 和实际 graph-node entry。
- Qwen 的真实 compile wrapper 位于嵌套的 `language_model.model`；driver 递归发现、重编译、保存并恢复所有 wrapper，不依赖硬编码 shell 层。
- 正式范围明确限定为 `inductor_generated_full_decode_graph_kernels`；prefill 是单独的 `production_mixed_mode_non_headline`，并使用独立 decode 请求证明 FULL graph。
- `TRTLLM_ENABLE_PDL=0` 在三档保持一致，避免把非目标后端混入 PDL 归因。
- 顶层 `model_identity.json` 哈希 config、safetensors index、tokenizer 等非权重元文件内容，记录本地 revision 与完整 shard 名称/大小清单；raw/runtime/admission 均绑定其 SHA-256，且不重复读取约 58GB 权重正文。

当前准入实现是 fail-closed：正式 cohort 必须精确匹配声明矩阵、31 次 Latin-3 triplet、3 次 warmup、2000 次 bootstrap、同一 worker cohort、off/grid 完整 token 与 hex-logprob 相等，并通过 PTX/cubin/Nsight 证明。`--allow-short` 只生成 `diagnostic_validation.json` 且固定 `admissible=false`，不会形成正式 admission。

逐次 smoke 的保留原因和证据见 `TIER4_SMOKE_AUDIT.md`；smoke1 的原始专项说明仍见 `TIER4_SMOKE1_REJECTED.md`。
