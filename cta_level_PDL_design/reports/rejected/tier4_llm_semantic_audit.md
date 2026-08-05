# Tier 4 Qwen/vLLM：PDL 三档语义拒绝审计报告

报告日期：2026-08-05（UTC）  
实验日期：**NOT EXECUTED**；准入审计日期为 2026-08-05（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0，可见显存 183359 MiB  
证据等级：**REJECTED-TIMING / EXECUTABLE-PREFLIGHT-AUDIT**——可支持当前 Tier 4 不准入、旧路径不可复用和解锁条件；不能支持任何 LLM latency、throughput、speedup 或 headroom 数值

## 1. 执行摘要

本轮 Tier 4 正式 timing 为 **0 个**。目标 `Qwen/Qwen3.6-27B` 没有完整本地权重，当前
runner 在任何模型加载、下载或 GPU workload 之前执行 CPU-only preflight，并以退出码 3
报告 `PREFLIGHT status=BLOCKED`。阻塞时不创建结果目录，也不产生可被 analyzer 接受的
`status=ok kind=measurement` 汇总行。

本轮实际数量如下：

| 项 | 数量/结果 |
|---|---:|
| 模型下载 | 0 |
| 正式模型加载 | 0 |
| 正式 GPU timing 调用 | 0 |
| `pdl_off` / `pdl_grid` / `ceiling` 正式样本 | 0 / 0 / 0 |
| 正式 warmup / timed repeats | 0 / 0 |
| 可报告的 `Ceiling − PDL_grid` | 不存在 |
| 当前 runner 退出码 | 3（BLOCKED） |

当前有四项独立 blocker：目标模型未本地完整落盘；缺少同进程相邻三档且 worker cohort 稳定
的 driver；没有目标 Qwen/vLLM kernel 的 worker/PTX/Ceiling wait-presence 证明；缺少实际
FULL CUDA graph node 执行证明。任一项都足以拒绝 timing。本轮没有用性能数字代替这些
语义证据，也没有把“未运行”写成“收益为 0”。

旧 runner/driver 的输出即使存在也不可作为 Tier 4 数据复用：它的 prompt API 与本机 vLLM
0.23 不兼容，三档分属不同进程，Ceiling 只在父进程 monkeypatch，且没有 target-specific
PTX、实际 graph-node、正确性和置信区间证据。因此旧数据不得进入报告、gate 或设计推荐。

## 2. 程序实际执行的语义

### 2.1 被拒绝的旧路径

旧 [`bench/llm/bench_llm.py`](../../bench/llm/bench_llm.py) 与旧 sweep 的实际行为存在以下
问题：

1. 调用 `LLM.generate(prompt_ids=...)`；本机 vLLM 0.23.0 的公开接口使用 `prompts=`，旧调用
   会在昂贵的模型加载之后才失败。
2. `pdl_off`、`pdl_grid`、`ceiling` 各启动独立 Python 进程。模型、worker、compile cache
   和 graph 状态均可能不同，不满足实验计划要求的同进程相邻三点。
3. `pdl_grid` 主要依赖环境变量宣称 PDL 已开启。环境变量为 1 不证明 Qwen 的实际 kernel
   由 Inductor 生成，也不证明 PTX 中存在 `griddepcontrol.wait` 与
   `griddepcontrol.launch_dependents`。
4. Ceiling 在父进程把 Triton `gdc_wait` 替换成 no-op。spawn worker 不一定继承该 Python
   对象；复用 compile cache 也可能绕过 patch。旧路径没有证明 worker 编译出的目标 kernel
   保留 launch 而删除 wait。
5. `enforce_eager=False` 只是允许 CUDA graph，不是测量时实际使用 FULL graph 的执行证据；
   旧路径也没有把含 GDC 的 PTX entry 与实际 replay 的 graph kernel node 对应起来。
6. 不同 batch 的有效重复数会随 `requests // batch` 下降，最大 batch 可能只有一个样本；旧
   汇总没有 95% CI，也没有比较非 Ceiling 两档的完整输出摘要。

因此旧路径可能输出的时间不是“低等级但可参考”的数据，而是**语义未证明的不可用数据**。
本报告不引用其中任何 latency 或 throughput。

### 2.2 当前 fail-closed 路径

当前 [`bench/llm/run_llm_sweep.sh`](../../bench/llm/run_llm_sweep.sh) 不再执行旧 sweep。它要求
显式指定完整本地 `MODEL` 和全新 `RESULTS`，设置 Hugging Face/Transformers offline 环境，
随后调用 [`bench/llm/preflight_llm.py`](../../bench/llm/preflight_llm.py)。当前 runner 明确
自报 `single_rung_processes`，所以即使其他条件未来满足，也不会误落回旧 measurement；
preflight 或最终 guard 都会返回 3。

单档 driver 已修正为 vLLM 0.23 的 `TokensPrompt` + `prompts=` 调用，并增加 worker RPC、
至少 31 次重复和 bootstrap 95% CI，但它只允许显式 `--diagnostic-only`，输出固定为
`status=diagnostic kind=diagnostic verified=0`。这条路径用于未来 proof driver 开发，不能
产生正式 Tier 4 结果。

## 3. 配置、统计与本轮执行范围

准入审计发生在实际可见的 B200 环境，而不是设计文档中的目标 B300：

| 项 | 本轮环境 |
|---|---|
| GPU | NVIDIA B200，148 SM，CC 10.0 |
| 可见 GPU 数 | 1 |
| 显存 | 183359 MiB（约 179.062 GiB） |
| vLLM | 0.23.0 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| Transformers | 5.12.0 |
| `nsys` / `ncu` / `compute-sanitizer` | 均缺失 |
| 目标模型权重 | 未完整本地落盘 |

统计项必须明确写为：

| 统计项 | 本轮值 |
|---|---:|
| 正式 warmup | 0 |
| 每档正式 timed repeats | 0 |
| 计时方法 | 无；未进入 timing |
| 聚合方法 | 无；不存在正式样本 |
| 中位数 / bootstrap 95% CI | 不存在 |
| 正式 SUMMARY 行 | 0 |

轻量自检只覆盖控制流和证据 validator，不是 LLM 实验：Python/shell/JSON 静态检查返回 0；
remote model ID 的 runner 返回 3 且不创建结果目录；合成的完整 proof fixture 可被 validator
接受，而给 Ceiling PTX 注入一条 wait、删除 triplet 行或使用伪 SQLite 都会返回 3。合成
fixture 的通过只证明拒收逻辑可执行，不是 Qwen kernel 的证据。

## 4. 四项当前 blocker

| # | Blocker | 当前事实 | 为什么禁止 timing |
|---:|---|---|---|
| 1 | `MODEL_NOT_STAGED` | `Qwen/Qwen3.6-27B` 不是带 `config.json` 和完整权重的本地目录 | 允许 vLLM 接收 remote ID 会在 proof 前隐式下载；模型 fingerprint 也无法固定 |
| 2 | `NO_SAME_PROCESS_TRIPLET_DRIVER` | 现有 sweep 是一档一进程；没有同一 driver、相邻 off/grid/ceiling 且稳定 worker PID cohort 的实现 | 跨进程模型、cache、worker 和 graph 状态不能形成计划规定的配对 bracket |
| 3 | `TARGET_PDL_SEMANTICS_UNPROVEN` | 没有目标 Qwen/vLLM 隔离 cache 的 PTX/cubin；没有 worker 侧 Floor wait-presence，也没有 Ceiling 保留 launch、删除 wait 的证明 | 环境变量或父进程 monkeypatch 不能证明实际目标 kernel 的 GDC 语义；也不能证明主要执行路径受 PDL 影响 |
| 4 | `FULL_GRAPH_EXECUTION_UNPROVEN` | 配置值不等于 runtime replay；本机缺 `nsys`，没有 CUDA graph node trace | 无法把含 wait/launch 的 PTX entry 精确连接到实际执行的 FULL graph kernel node |

机器可读审计把第 3、4 项进一步拆成 worker wait-presence、target-kernel coverage、Ceiling
传播和 FULL graph execution 等诊断代码；这些子代码在准入决策中归入上表四个 blocker，
不表示已经有部分正式测量可用。

## 5. 头条数复算与解锁契约

### 5.1 本轮数值复算

preflight 在第一次模型调用之前退出，因此：

~~~text
formal_timing_calls = calls_after_successful_preflight = 0
formal_samples      = pdl_off(0) + pdl_grid(0) + ceiling(0) = 0
model_downloads     = 0  (remote ID 在 vLLM import/构造前被拒绝)
~~~

剩余 CTA-level headroom 的定义要求同时存在有效的 `PDL_grid` 与 `Ceiling` 样本：

~~~text
headroom = improvement(PDL_grid, Ceiling)
         = improvement(undefined, undefined)
         = undefined
~~~

因此本轮 headroom 不是 `0%`，也没有可复算的吞吐或延迟。唯一设备换算为：

~~~text
183359 MiB / 1024 = 179.0615234375 GiB
~~~

该换算只描述可见设备容量，不是模型可运行性或性能结果。

### 5.2 解除 BLOCKED 的最低契约

未来实现只有同时满足以下条件，才允许从 proof 阶段进入正式 timing：

1. 目标模型完整本地落盘；以 `config.json` 与全部权重文件名/字节数固定 fingerprint。runner
   继续保持 offline，不负责隐式下载。
2. 在同一 driver 进程、同一稳定 worker PID cohort 内按 `pdl_off → pdl_grid → ceiling`
   相邻执行三档；每档使用新建、互不共享的 compile cache，并记录一致的软件版本。
3. 直接检查每档 cache 中的 PTX 和同 stem cubin：off 的 wait/launch 均为 0；grid 的
   wait/launch 均大于 0；Ceiling 的 wait 为 0 而 launch 大于 0。worker RPC 必须同时确认
   Inductor PDL 与 Ceiling hook 的实际状态。
4. 用 Nsight Systems `--cuda-graph-trace=node` 导出 SQLite；把 wait-bearing 与
   launch-bearing PTX 的 `.entry` 名精确连接到 `CUPTI_ACTIVITY_KIND_KERNEL` 中带
   `graphNodeId`/`graphId` 的实际节点，并确认最终 graph/dispatcher mode 都是 `FULL`。
5. proof 与正式测量必须属于相同模型 fingerprint、软件版本和证据契约；非 Ceiling 的
   `pdl_off`/`pdl_grid` 输出摘要必须相同，任何校验失败都使整点非零退出。Ceiling 按设计
   错误，只允许计时并标 `verified=0`。
6. 每档至少 31 个独立样本，报告中位数和 bootstrap 95% CI；三档 SUMMARY 必须相邻、完整，
   且只能在证据 validator 通过后标记 `status=ok kind=measurement`。

[`bench/llm/pdl_evidence.py`](../../bench/llm/pdl_evidence.py) 实现了第 2–4 项的机器可读拒收
边界；[`tools/llm_bracket.py`](../../tools/llm_bracket.py) 实现第 5–6 项的最终完整性检查。
当前缺少能够生成该 proof 的同进程 driver，所以 validator 的存在本身不会解锁测量。

## 6. 能成立的结论

本审计支持：

1. 本轮正式 Tier 4 timing、三档样本和性能 SUMMARY 均为 0；没有模型下载。
2. 当前 runner 会在证据不足时返回 3，且不会创建可与未来结果混合的默认结果目录。
3. 旧一档一进程、父进程 Ceiling patch、环境变量自报和无 CI 数据不能证明计划所需三档，
   旧数据不可复用。
4. 本机 vLLM 0.23 应使用 `prompts=`；当前诊断 driver 已修正 API，但仍不具备正式测量资格。
5. 当前实际环境为单卡 B200（148 SM、CC 10.0），且缺少 `nsys` 和目标模型完整本地权重。
6. 现有 validator 能拒绝不完整 triplet、错误 Ceiling wait、伪 Nsight SQLite、缺 CI 或
   PTX entry 未出现在 graph-node 表中的输入。

### 6.1 可保留和复用的材料

可复用的是准入与工程证据，而不是性能结论：机器可读 blocker、B200 软件环境记录、vLLM
0.23 prompt API 修正、offline/result-isolation preflight、PTX/Nsight validator 及其合成负例
自检，都可作为未来同进程 proof driver 的输入。旧日志若出现 API 异常，只能用于定位旧
driver 失败阶段；不得从其中保留或恢复任何 timing 数值。

## 7. 不能成立的结论

本审计不支持：

1. Qwen/vLLM 的 `pdl_off`、`pdl_grid` 或 Ceiling latency、throughput、TPS/user、speedup。
2. `Ceiling − PDL_grid` 为 0%、正值或负值；该 headroom 当前未定义。
3. `TORCHINDUCTOR_ENABLE_PDL=1`、`enforce_eager=False` 或配置 `FULL` 就证明目标 kernel
   实际执行了 grid-level PDL。
4. 父进程 no-op `gdc_wait` 就证明所有 worker 的 Ceiling 已删除 wait。
5. Qwen 的主要 Gated DeltaNet/attention/GEMM 路径由 Inductor PDL 覆盖，或 target-specific
   PTX 中已经存在 GDC 指令。
6. 183359 MiB 可见显存就证明目标模型及 4K/32K/128K、batch 1/4/16/64 全矩阵可承载。
7. 合成 proof fixture 通过就等于真实 Qwen proof 通过。
8. PDL_off→PDL_grid 落入公开 2–33% 诊断带就构成语义证明；该带宽只能触发异常检查。
9. Tier 4 已完成或可以据此更新设计推荐；当前状态只能是 `BLOCKED / REJECTED`。

## 8. 证据入口

- 机器可读拒收审计：[bench/llm/tier4_rejection.json](../../bench/llm/tier4_rejection.json)
- 原始中文准入说明：[bench/llm/TIER4_REJECTED.md](../../bench/llm/TIER4_REJECTED.md)
- CPU-only preflight：[bench/llm/preflight_llm.py](../../bench/llm/preflight_llm.py)
- PTX/cubin/worker/Nsight 证据 validator：[bench/llm/pdl_evidence.py](../../bench/llm/pdl_evidence.py)
- Fail-closed runner：[bench/llm/run_llm_sweep.sh](../../bench/llm/run_llm_sweep.sh)
- 单档 diagnostic driver：[bench/llm/bench_llm.py](../../bench/llm/bench_llm.py)
- 严格 bracket analyzer：[tools/llm_bracket.py](../../tools/llm_bracket.py)
- Tier 4 实验规格：[EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md)
- Benchmark 与报告有效性规则：[AGENTS.md](../../AGENTS.md)

本轮没有 raw timing log 或正式 SUMMARY；这些入口缺失是 preflight 在 GPU timing 前
fail-closed 的预期结果，不是证据遗失。该 `BLOCKED` 状态已纳入
[本轮 umbrella report](../campaign_b200_multiwave_20260805.md) 与
[`EXPERIMENT_REPORT_INDEX.md`](../../EXPERIMENT_REPORT_INDEX.md)，其中 timing=0 只表示未进入测量，
headroom 仍为未定义。
