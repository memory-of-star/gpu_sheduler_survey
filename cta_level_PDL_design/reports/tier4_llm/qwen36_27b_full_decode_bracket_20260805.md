# Tier 4：Qwen3.6-27B full-decode PDL 三档包夹

| 元数据 | 值 |
|---|---|
| 报告日期 | 2026-08-05（UTC） |
| 实验日期 | 2026-08-05（UTC） |
| 设备 | 单卡 NVIDIA B200，148 SM，Compute Capability 10.0 |
| 证据等级 | **ADMITTED / TARGET-SPECIFIC END-TO-END BRACKET**——可支持这台 B200、这份本地模型身份和所列 decode 配置上的端到端 `PDL_off` / `PDL_grid` / unsafe Ceiling 包夹；不能支持 Ceiling 正确性、可实现 CTA 机制的实际收益、prefill headline、B300/Rubin 或其他模型的外推 |

## 1. 执行摘要

`decode` 与 `prefill` 两个 cohort 的封存结果均通过独立的完整 admission 重算：

~~~text
VERIFY_ADMISSION status=ok .../cohorts/decode
VERIFY_ADMISSION status=ok .../cohorts/prefill
~~~

Headline 只取 `classification=headline_full_decode` 的 4 个 decode 点。表中延迟是每档 31 次 `LLM.generate()` 主机端到端时间的中位数；正的百分比表示右侧路径更快。

| 配置（seq=64，gen=16） | PDL_off ms | PDL_grid ms | Ceiling ms | PDL_grid vs off | 95% CI | `Ceiling − PDL_grid` headroom | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| batch 1 | 225.521419 | 221.109886 | 218.161572 | 1.956148% | [1.867084%, 2.011747%] | 1.333416% | [1.263817%, 1.397806%] |
| batch 4 | 229.201121 | 224.657617 | 221.716003 | 1.982322% | [1.865173%, 2.036825%] | 1.309376% | [1.237224%, 1.416650%] |
| batch 16 | 247.436778 | 243.428760 | 238.973700 | 1.619815% | [1.423633%, 1.798448%] | 1.830129% | [1.713939%, 1.952675%] |
| batch 64 | 448.370290 | 443.579917 | 407.823583 | 1.068397% | [0.873525%, 1.261456%] | 8.060855% | [7.903751%, 8.253172%] |

因此，已部署式 grid PDL 打开之后，当前 4 个 decode 点剩余的**零等待上界**分别为 1.333%、1.309%、1.830% 和 8.061%。这不是 CTA-level 实现的预期收益：Ceiling 删除 wait 后在每个点的 31/31 次重复中都产生了与正确参考不同的完整输出，并且 31/31 次均出现非有限 cumulative logprob；它只按实验计划提供时间上界。

另有 3 个长上下文点通过 admission，但它们必须保持 `production_mixed_mode_non_headline`：计时覆盖 prefill 加 2-token generation，而被改变并被 PTX/Nsight 证明的是 Inductor 生成的 FULL-decode graph kernel。它们不进入上表，不与 decode headline 合并。

## 2. 程序实际执行的语义

模型来自本地 `/workspace/models/Qwen3.6-27B`。身份清单记录 15 个 safetensors shard、总计 55,563,006,400 bytes，模型 fingerprint 为 `edf13415f2470aa5f40b0c271fb1a7d48ecad4a8cf8e1a8367b43cfee6c7f4b6`。该 fingerprint 绑定本地 metadata 与权重文件 inventory；本轮没有逐字节散列全部 55 GB 权重，不能把它解释成完整权重内容哈希。

每个 cohort 各自只构造一次 vLLM engine，并在同一 driver/worker cohort 内完成三次独立 lowering。三档使用隔离的新 cache，随后保留各自的 compiled callable 与 FULL CUDA graph，并在每次调用前切换和核对 active variant identity：

| 档位 | 实际 lowering / PTX 语义 | 正确性角色 |
|---|---|---|
| `pdl_off` | `TORCHINDUCTOR_ENABLE_PDL=0`；目标 cache 中 wait=0、launch=0 | 正确参考 |
| `pdl_grid` | Inductor PDL 打开；目标 PTX 同时含 `griddepcontrol.wait` 与 `griddepcontrol.launch_dependents` | 必须逐次与 off 的 token 和 hex logprob 完全一致 |
| `ceiling` | 保留 PDL launch，但在 lowering 前把 Triton `gdc_wait` 设为 no-op；目标 PTX wait=0、launch>0 | `unsafe_not_validated`；必须观察到错误，只取时间 |

`TRTLLM_ENABLE_PDL` 在三档均为 false。干预范围明确为 `inductor_generated_full_decode_graph_kernels`，不是 vLLM 内所有 kernel。engine 的 graph mode 是 `FULL_DECODE_ONLY`；Nsight Systems 用 `--cuda-graph-trace=node` 在每档各一个独立 proof window 中把 PTX entry 对应到实际执行的 CUDA graph node，正式 timed samples 本身不带 profiler 开销。

decode cohort 的单一 worker PID 为 202777，prefill cohort 为 208153。两者的 grid cache 均有 104 个 PTX 与 104 个配对 cubin，统计到 130 条 wait、124 条 launch，分布在 21 个 wait-bearing 和 19 个 launch-bearing entry；Ceiling 保持同样 124 条 launch 而 wait 为 0。Nsight 对这些 entry 的 graph-node name coverage 为 100%。off cache 在 decode/prefill 分别有 115/112 个 PTX-cubin 对，wait 与 launch 均为 0。

每个 repeat 使用新的确定性 poison token 序列；同一个相邻 triplet 的三档使用相同 epoch 与 prompt，避免旧输出伪装成正确结果。`pdl_off` 与 `pdl_grid` 的所有 request、生成 token 和 cumulative logprob hex 均纳入完整输出 digest。Ceiling 只检查其确实表现出错误，不检查也不声称其数值正确。

## 3. 配置与统计

| 项 | decode cohort | prefill cohort |
|---|---:|---:|
| 点 | batch 1/4/16/64；seq 64；gen 16 | batch 1；seq 4K/32K/128K；gen 2 |
| `max_model_len` | 80 | 131,074 |
| `max_num_seqs` | 64 | 1 |
| `gpu_memory_utilization` | 0.82 | 0.90 |
| warmup | 每点 3 个完整三档 triplet | 每点 3 个完整三档 triplet |
| timed repeats | 每点每档 31 | 每点每档 31 |
| timed samples | 4 × 3 × 31 = 372 | 3 × 3 × 31 = 279 |
| correctness validations | 4 × 31 = 124 | 3 × 31 = 93 |

三档在同一进程中相邻执行，repeat 按以下 Latin-3 顺序循环，以控制固定位置偏差：

~~~text
pdl_off  > pdl_grid > ceiling
pdl_grid > ceiling  > pdl_off
ceiling  > pdl_off   > pdl_grid
~~~

计时源是围绕同步返回的 `LLM.generate(prompts=..., use_tqdm=False)` 的 `time.perf_counter()` 主机墙钟，包含该请求的 vLLM 调度、prefill/decode 和输出返回，不是单 kernel CUDA event。每档报告 31 次的中位数；每档 latency CI 以及两个百分比 CI 都使用 2,000 次 bootstrap。百分比 CI 按 repeat index 配对重采样后计算两个中位数的比值，置信水平为 95%。没有删除 outlier，也没有因性能值重试。

软件环境为 vLLM 0.23.0、PyTorch 2.11.0+cu130、Triton 3.6.0、Transformers 5.12.0；KV offloading backend 记录为 `native`，offloading size 为 null。

## 4. Headline 原始值与算术复算

定义严格沿用分析器：

~~~text
PDL_grid vs off (%)       = 100 × (median_off - median_grid) / median_off
Ceiling − PDL_grid (%)    = 100 × (median_grid - median_ceiling) / median_grid
~~~

以下数值直接从 `raw_triplet.json` 的各 31 个 `elapsed_s` 重新取中位数，而不是抄写汇总；单位换算为 ms 后复算。

### batch 1

~~~text
off=225.521418732 ms, grid=221.109885722 ms, ceiling=218.161571771 ms
grid vs off = 100 × (225.521418732 - 221.109885722) / 225.521418732
            = 1.956148%
headroom    = 100 × (221.109885722 - 218.161571771) / 221.109885722
            = 1.333416%
~~~

### batch 4

~~~text
off=229.201121256 ms, grid=224.657617044 ms, ceiling=221.716003027 ms
grid vs off = 100 × (229.201121256 - 224.657617044) / 229.201121256
            = 1.982322%
headroom    = 100 × (224.657617044 - 221.716003027) / 224.657617044
            = 1.309376%
~~~

### batch 16

~~~text
off=247.436778154 ms, grid=243.428759743 ms, ceiling=238.973699976 ms
grid vs off = 100 × (247.436778154 - 243.428759743) / 247.436778154
            = 1.619815%
headroom    = 100 × (243.428759743 - 238.973699976) / 243.428759743
            = 1.830129%
~~~

### batch 64

~~~text
off=448.370289989 ms, grid=443.579917308 ms, ceiling=407.823583111 ms
grid vs off = 100 × (448.370289989 - 443.579917308) / 448.370289989
            = 1.068397%
headroom    = 100 × (443.579917308 - 407.823583111) / 443.579917308
            = 8.060855%
~~~

独立复算得到的 12 个中位数与 `analysis.json` 逐浮点值相等，8 个百分比与 analyzer 的差均小于 `1e-12` percentage point。

## 5. Prefill 扫描：`production_mixed_mode_non_headline`

这 3 点只作为混合路径诊断列出。其 raw 中位数和分析器结果如下，但**不能作为 prefill PDL headline，也不能与上面的 4 个 decode 点求平均**：

| 配置（batch=1，gen=2） | off ms | grid ms | Ceiling ms | grid vs off（95% CI） | wait-removal bracket（95% CI） |
|---|---:|---:|---:|---:|---:|
| 4K | 183.994799 | 184.753053 | 153.208367 | -0.412106%（[-0.952624%, 0.247017%]） | 17.073973%（[16.660395%, 17.586980%]） |
| 32K | 1464.404714 | 1471.997468 | 1470.378429 | -0.518487%（[-0.682466%, -0.353916%]） | 0.109989%（[0.009317%, 0.256656%]） |
| 128K | 7774.798142 | 7805.843033 | 7791.999409 | -0.399302%（[-0.507779%, -0.303770%]） | 0.177350%（[0.096143%, 0.252920%]） |

4K 的 grid-vs-off CI 跨过 0；32K 与 128K 在本次重复内显示 grid 较慢约 0.52% 与 0.40%。但这些端到端请求主要随 prefill 长度增长，而 proof point 是 batch 1、seq 64 的 full-decode request，PTX 改动范围也只声明 full-decode graph kernel。因此 4K Ceiling 的 17.07% 等数值只描述这个混合 harness 中 unsafe wait 删除后的现象，不能归因成“prefill 有 17.07% CTA headroom”。

## 6. 能成立的结论

1. 当前封存 artefact 上，decode 与 prefill 两个 cohort 都满足 formal admission；manifest、模型身份、raw、analysis、PTX/cubin、worker active identity 和 Nsight graph-node 证明能闭环重算。
2. 对 seq=64、gen=16 的 4 个 decode 点，grid PDL 相对 PDL-off 的中位端到端改善分别是 1.956%、1.982%、1.620% 和 1.068%，4 个配对 bootstrap CI 均为正。
3. 同四点从 grid PDL 到零 wait 的时间上界分别是 1.333%、1.309%、1.830% 和 8.061%。这些是各 batch 独立的上界，不存在本报告认可的跨 batch 单一平均值。
4. 所有 decode 与 prefill 配置中，`pdl_grid` 的 31/31 次完整输出都与 `pdl_off` 精确一致；prompt 每次改变，不能由上一 repeat 的残留输出满足检查。
5. Ceiling 在所有 7 个点均为 31/31 次完整输出不同且 31/31 次 non-finite logprob，符合“错误结果、只计时间”的实验边界；没有把它包装成正确推理。
6. prefill 三点只支持 `production_mixed_mode_non_headline` 的本机诊断：grid-vs-off 为约 -0.4% 至 -0.52%，不能升级为 prefill PDL headline。

## 7. 不能成立的结论

本实验不支持：

1. CTA-level 机制实际能拿到 1.3%、1.8% 或 8.1%。本轮没有实现 CTA 依赖；Ceiling 把 in-scope wait 直接删除，是不可正确执行的零等待上界。
2. Ceiling 的 token、logprob 或任意中间状态正确。恰恰相反，所有重复都观察到错误与非有限 logprob；其 latency 不可用于生产服务 SLA。
3. batch 64 的 8.061% 可外推到其他 batch、prompt、generation length、并发服务或模型。四个点也不显示可泛化的单调 batch 趋势。
4. prefill 4K 的 17.074% 是 prefill kernel 的 CTA headroom。长上下文点混合了 prefill 与仅 2 个 decode token，且 target-specific proof 只覆盖 FULL-decode graph path。
5. 本轮覆盖 vLLM 的全部算子、TensorRT-LLM PDL、FlashInfer/CUTLASS 内部依赖或 Qwen 的所有非 Inductor kernel；`TRTLLM_ENABLE_PDL=false`，声明的 PDL scope 更窄。
6. 端到端 `LLM.generate()` 墙钟差可直接分解为某条 GDC 指令的成本，或等价于 kernel latency、TPS/user、多租户吞吐和 tail latency。
7. 2,000 次 bootstrap CI 表示跨机器、跨版本或跨模型置信区间；它只量化本 cohort 的 31 个相邻 repeat 的重采样不确定性。
8. B200 结果能证明 B300 或 Rubin 上的收益，也不能证明任何未公开硬件实现细节。
9. 模型 fingerprint 是 55 GB 权重内容的逐字节哈希；身份清单本轮固定的是 metadata 与 shard inventory。

## 8. 证据入口

只读复核命令：

~~~bash
python3 bench/llm/tier4_finalize.py \
  --results results/tier4_schema_v3_formal_v1_20260805/cohorts/decode \
  --verify-admission
python3 bench/llm/tier4_finalize.py \
  --results results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill \
  --verify-admission
~~~

- 不可变实验契约（SHA-256 `d3b80d2839ce7c897cad8bf8fae225358ec51d0a352bef0cb665712befad3508`）：[results/tier4_schema_v3_formal_v1_20260805/manifest.json](../../results/tier4_schema_v3_formal_v1_20260805/manifest.json)
- 模型身份清单：[results/tier4_schema_v3_formal_v1_20260805/model_identity.json](../../results/tier4_schema_v3_formal_v1_20260805/model_identity.json)
- Decode admission / raw / analysis / evidence：[results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/admission.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/admission.json)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/raw_triplet.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/raw_triplet.json)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/analysis.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/analysis.json)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/evidence_validation.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/evidence_validation.json)
- Prefill admission / raw / analysis / evidence：[results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/admission.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/admission.json)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/raw_triplet.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/raw_triplet.json)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/analysis.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/analysis.json)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/evidence_validation.json](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/evidence_validation.json)
- 原始运行日志：[results/tier4_schema_v3_formal_v1_20260805/logs/decode.log](../../results/tier4_schema_v3_formal_v1_20260805/logs/decode.log)、[results/tier4_schema_v3_formal_v1_20260805/logs/prefill.log](../../results/tier4_schema_v3_formal_v1_20260805/logs/prefill.log)
- Nsight 原始报告与导出 SQLite：[results/tier4_schema_v3_formal_v1_20260805/profiles/decode.nsys-rep](../../results/tier4_schema_v3_formal_v1_20260805/profiles/decode.nsys-rep)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/profile.sqlite](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/decode/profile.sqlite)、[results/tier4_schema_v3_formal_v1_20260805/profiles/prefill.nsys-rep](../../results/tier4_schema_v3_formal_v1_20260805/profiles/prefill.nsys-rep)、[results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/profile.sqlite](../../results/tier4_schema_v3_formal_v1_20260805/cohorts/prefill/profile.sqlite)
- 执行与准入代码：[bench/llm/tier4_driver.py](../../bench/llm/tier4_driver.py)、[bench/llm/pdl_evidence.py](../../bench/llm/pdl_evidence.py)、[bench/llm/tier4_finalize.py](../../bench/llm/tier4_finalize.py)、[bench/llm/run_llm_sweep.sh](../../bench/llm/run_llm_sweep.sh)
- 实验规格：[EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md)；报告有效性规则：[AGENTS.md](../../AGENTS.md)
- 本轮 umbrella 入口：[reports/campaign_b200_multiwave_20260805.md](../campaign_b200_multiwave_20260805.md)

历史的 blocked 审计仍保留在 [reports/rejected/tier4_llm_semantic_audit.md](../rejected/tier4_llm_semantic_audit.md)，只说明旧路径为何不可用；本报告只使用后来生成并通过 admission 的 `tier4_schema_v3_formal_v1_20260805`，没有复用旧 timing。
