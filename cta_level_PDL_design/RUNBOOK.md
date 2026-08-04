# 租用机器执行手册（B300 / B200 单卡）

> 按分钟计费，所以这份手册的目标是：**上机后不做任何调试，只产出数据**。
>
> 代码说明见 [`bench/README.md`](bench/README.md)，方法论见 [`docs/cta_pdl_eval_plan.md`](docs/cta_pdl_eval_plan.md)。
>
> **总预算约 8 GPU-hours。** 其中 Tier 1 结束时有一个决策点，可能提前终止。

---

## 0. 一页速查

```bash
# ── 上机后立刻做这三件事（并行）────────────────────────────
tmux new -s cta                              # 防断线
huggingface-cli download Qwen/Qwen3.6-27B &  # 54GB，后台下，Tier 0/1 不需要它
./preflight.sh                               # 环境自检，1 分钟

# ── 然后按序执行 ─────────────────────────────────────────
cd bench && FAST=1 ./run_all.sh              # 冒烟，~5 min
./run_all.sh tier0                           # ~1h
./run_all.sh tier1p                          # ~2h   ← 决策点在这里（修正后的 pilot）
python3 ../tools/analyze_pilot.py results/pilot_matrix.log \
    --json results/pilot_analysis.json --csv results/pilot_summary.csv | tee /tmp/gate.txt

# ── 决策：Tier 1 的收益空间够不够 ─────────────────────────
# 够  → cd llm && ./run_llm_sweep.sh ; cd ../dsa && ./run_dsa_chain.sh
# 不够 → 只补 LLM 端到端确认，然后收工
#
# 注意：phase 名是 tier1p 不是 tier1。tier1 / tier23 驱动被否决的 cta_dep_bench，
# 只在复审旧结果时才跑，其数字不能进 gate。详见 §3.3。

# ── 收工 ────────────────────────────────────────────────
cd .. && ./collect.sh                         # 打包所有结果
```

---

## 1. 租用前（本地做完，不花 GPU 钱）

### 1.1 打包

```bash
cd "$(git rev-parse --show-toplevel)"    # 仓库根，即 cta_level_PDL_design 的上一层
tar czf cta_pdl_bench.tar.gz \
    cta_level_PDL_design/bench \
    cta_level_PDL_design/tools \
    cta_level_PDL_design/preflight.sh \
    cta_level_PDL_design/collect.sh \
    cta_level_PDL_design/RUNBOOK.md \
    cross_stream_PDL_survey/bench/pdl_bench
```

只传代码，不传 PDF 和文档（几十 MB 的论文没必要上机器）。

**`cross_stream_PDL_survey/bench/pdl_bench` 必须一起打包**：§3.2 的 Tier 0.2 复验要 `cd` 到它，它是同级目录不是本子树的一部分，漏掉就会在机器上撞到「路径不存在」。解包后两个目录保持同级，§3.2 的相对路径才成立。

### 1.2 本地验证分析链

**这一步很重要**：分析脚本必须在上机前就确认能跑，否则会在机器上浪费时间调 Python。

```bash
cd cta_level_PDL_design
python3 tools/make_test_fixtures.py --out /tmp/ctafix
python3 tools/analyze.py      /tmp/ctafix/summary.txt
python3 tools/cta_timeline.py /tmp/ctafix/trace.csv
python3 tools/llm_bracket.py  /tmp/ctafix/summary_llm.txt
```

四个都应无报错。夹具的数字是编的，只有格式是真的。

### 1.3 确认租用规格

| 项 | 要求 | 为什么 |
|---|---|---|
| GPU | B300 (sm_103) 或 B200 (sm_100) 单卡 | CLC 需要 sm_100+；H100 (sm_90) 也能跑，但 Tier 0.4 会跳过 |
| 显存 | ≥ 180 GB | Qwen3.6-27B BF16 约 54GB + KV cache + 激活 |
| CUDA | ≥ 12.8 | CLC 的 PTX 需要；≥ 12.0 才有 libcu++ `atomic_ref` |
| 磁盘 | ≥ 150 GB | 模型 54GB + nsys trace 可能几 GB |

---

## 2. 上机第一步：环境自检

```bash
tar xzf cta_pdl_bench.tar.gz && cd cta_level_PDL_design
./preflight.sh
```

它会检查 GPU、CUDA、编译、Python 依赖，并**立即编译一次**。如果编译失败，在这里发现比在第三小时发现好。

**同时**把模型下载挂到后台——Tier 0/1 完全不需要模型，让它下着：

```bash
huggingface-cli download Qwen/Qwen3.6-27B > /tmp/dl.log 2>&1 &
```

> 若下载慢，试 `HF_ENDPOINT=https://hf-mirror.com`。这一步经常是整个 session 里最慢的环节，越早启动越好。

---

## 3. 执行序列

### 3.1 冒烟（约 5 分钟）

```bash
cd bench && FAST=1 ./run_all.sh
```

用极小的参数跑通全流程。**看到 `campaign finished` 且 `failures.log` 为空**再继续。

若有失败，看 `results/<step>.log`。常见原因见 §6。

### 3.2 Tier 0：基础事实（约 1 小时）

```bash
./run_all.sh tier0
```

产出四件事，每一件都会影响后面所有实验的解读：

| 探测 | 回答什么 |
|---|---|
| 0.1 chain overlap | B300 上同 stream 实际能重叠**几层** kernel |
| 0.3 occupancy | 等待中的 CTA 占槽位的代价曲线（B2 维度的唯一定价依据） |
| 0.4 CLC | `try_cancel` 的延迟与仲裁吞吐（决定 B4 能否用软件复现） |
| 0.5 fence | 各 scope 成本（Ideal 点注入用） |

**还要手动补一项（0.2）**：把之前 H100 上的跨 stream 结论在 B300 上复验。

```bash
cd ../../cross_stream_PDL_survey/bench/pdl_bench && ./build.sh && ./run.sh | tee ~/b300_xstream.log
```

重点看 `PDL_XS`（eager 跨流）是否仍是 1.00×、`PDL_CAPTURE`（同代码 capture 成图）是否仍到 2.00×。若结论变了，说明 Blackwell 改了 programmatic event 的行为，那是个独立发现。

### 3.3 Tier 1：收益地图（约 2 小时）← **决策点**

> **跑 `tier1p`，不要跑 `tier1`。** `tier1` 与 `tier23` 驱动的是 `cta_dep_bench`，它在 PDL trigger
> 之前就把所有 `done[]` 标志发布出去，等待全部预先满足，其计时**不能**作为 CTA 收益证据
> （见 [`reports/rejected/fast_campaign.md`](reports/rejected/fast_campaign.md) 与 AGENTS.md §4）。
> 拿它喂 gate 等于用 2 GPU-hours 买一组已知无效的数字。`tier1p` 驱动的是修正后的 `cta_dep_pilot`。

```bash
cd ../../../cta_level_PDL_design/bench
./run_all.sh tier1p
python3 ../tools/analyze_pilot.py results/pilot_matrix.log \
    --json results/pilot_analysis.json \
    --csv  results/pilot_summary.csv | tee /tmp/gate.txt
```

注意用 `analyze_pilot.py` 而不是 `analyze.py`：pilot 输出的是 `SAMPLE` / `SUMMARY_PILOT`，schema 与
`analyze.py` 读的 `SUMMARY` 不同，两者不可互换。

`/tmp/gate.txt` 每行一个配置：

```
configurations=N all_valid=1
base_tag floor_ms impl_ms space_pct captured_pct of_space_pct speedup
```

先确认 `all_valid=1`；任何一个配置 `valid=0` 都意味着同步协议漏了正确性，此时所有时间数字作废。
然后看三件事：

**(a) 依赖度边界** —— 比较 `t11p_g*_d*` 各 degree 的 `space_pct`。BlockMaestro 在 28 SM 的
Titan X 上得出的是 32，B300 约 148 SM。**这个数往哪边移是决定性的**：

- 显著大于 32 → 阈值随机器容量放大，LLM 的高依赖度规整模式在适用范围内
- 仍在 32 附近或更小 → 适用面比预期窄得多，需要重新评估整个方向

**(b) 结构对照** —— 比较同 grid 下 `t11ps_g*_interval` 与 `t11ps_g*_strided` 的 `captured_pct`。
依赖度相同（都钉在 32）、只有形状不同；差距大说明瓶颈是**编码**（A3 维度）而非依赖度。这是
BlockMaestro 的 n-组全连接注入无法做出的区分。

**(c) grid 规模边界** —— 比较各 `g` 的 `space_pct`。BlockMaestro 是 2048 TB。

#### 决策规则

阈值的权威定义在 [`docs/cta_pdl_eval_plan.md`](docs/cta_pdl_eval_plan.md) §10.1，下表是供上机离线查阅的镜像；**要改阈值改那边**。

| Tier 1 结果 | 下一步 |
|---|---|
| 收益空间在多数配置下 **≥ 8%** | 继续 Tier 2/3 + LLM + DSA，跑满预算 |
| **2–8%** | 跳过 Tier 2/3，直接做 LLM 端到端确认真实负载上还剩多少 |
| **< 2%** | **停**。只跑 LLM 三档确认，然后收工。省下的钱比数据值钱 |

**判读时必须带上这条限制**：`cta_dep_pilot` 拒绝 `P,C > SM`（它要求每个 producer CTA 都驻留、
能在不被 launch gate 串行化的前提下 trigger），所以 `tier1p` 全部是**单波、欠填充**的。
multi-wave 区间是本项目的头号缺口，**两个二进制今天都产不出它**——`cta_dep_bench` 够得着但语义
被否决，而放宽 pilot 的上限是对 `.cu` 的语义改动、不是加个 flag。因此 gate 通过只能说明
「机制在单波下可行」，不足以据此跑满预算。

### 3.4 Tier 2/3：机制对比（约 3 小时，视决策而定）

```bash
./run_all.sh tier23
```

> **这个 phase 目前不产出可用结论。** 它驱动的同样是被否决的 `cta_dep_bench`，脚本会先打印
> 一段 `WARNING` 再继续。在 `cta_dep_pilot` 承接 Tier 2/3 的协议横评与编码对比之前，跑它只有
> 复审价值——**不要用这 3 小时的输出写结论**。若预算紧张，优先把时间给 §3.5 的 Tier 4。

### 3.5 Tier 4：LLM 端到端（约 2 小时）

确认模型下载完成后：

```bash
cd llm
FAST=1 ./run_llm_sweep.sh          # 先小跑一遍确认 vllm 能起来
./run_llm_sweep.sh                 # 完整扫描
python3 ../../tools/llm_bracket.py results_llm/summary_llm.txt | tee /tmp/llm_gate.txt
```

**这里产出整个项目最关键的单个数字**：`Ceiling − PDL_grid`，即 grid 级 PDL 之后 CTA 级还剩多少空间。

脚本会自检 `PDL_off → PDL_grid` 的增益是否落在已公开的 2–33% 带宽内。若明显偏离，多半是 CUDA graph 没走 FULL 模式导致 PDL 被静默关掉——检查 `VLLM_USE_FULL_CUDA_GRAPH=1` 是否生效。

### 3.6 Tier 5：DSA 算子链（约 30 分钟）

```bash
cd ../dsa && ./run_dsa_chain.sh
```

整模型单卡放不下，这里只跑注意力路径。脚本会先做纯离线的依赖推导（不占 GPU），再实测算子链。1M 上下文可能 OOM，脚本会记录并继续。

---

## 4. 收工：打包数据

```bash
cd /path/to/cta_level_PDL_design && ./collect.sh
```

生成 `cta_pdl_results_<时间戳>.tar.gz`，包含所有 `results*/` 目录、`SUMMARY` 汇总、设备信息、以及 nsys trace（若有）。

拷回本地后：

```bash
tar xzf cta_pdl_results_*.tar.gz
python3 tools/analyze_pilot.py results/pilot_matrix.log \
        --json pilot_analysis.json --csv pilot_summary.csv   # ← Tier 1 的 gate 数据在这里
python3 tools/analyze.py      results/summary.txt --csv all.csv --json findings.json
python3 tools/cta_timeline.py results/trace_*.csv --plot concurrency.png
python3 tools/llm_bracket.py  results_llm/summary_llm.txt
```

`analyze.py` 读 `SUMMARY`（Tier 0 与被否决的 `cta_dep_bench`），`analyze_pilot.py` 读
`SAMPLE` / `SUMMARY_PILOT`（修正后的 pilot）。两者 schema 不同，别把日志喂错脚本。

---

## 5. 时间预算表

| 阶段 | 预算 | 可跳过 |
|---|---|---|
| 环境自检 + 冒烟 | 15 min | 否 |
| Tier 0 基础事实 | 1.0 h | 否 |
| Tier 0.2 跨 stream 复验 | 15 min | 可 |
| **Tier 1 收益地图（`tier1p`）** | **2.0 h** | **否（决策点）** |
| Tier 2/3 机制对比（被否决的 harness） | 3.0 h | 视决策，当前建议跳过 |
| Tier 4 LLM 端到端 | 2.0 h | 否 |
| Tier 5 DSA 算子链 | 0.5 h | 可 |
| 收工打包 | 10 min | 否 |

模型下载与 Tier 0/1 并行，不单独计时。

---

## 6. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `nvcc: command not found` | CUDA 未在 PATH | `export PATH=/usr/local/cuda/bin:$PATH` |
| `clc_probe` 编译失败 | arch < sm_100 或 CUDA < 12.8 | 预期行为，`build.sh` 会跳过并继续 |
| `cuda::atomic_ref` 找不到 | CUDA < 12.0 | 必须升级；`dep_wait.cuh` 依赖 libcu++ |
| 微基准 `verify FAIL` | 同步协议有 bug，或显存被别的进程占用 | 先 `nvidia-smi` 确认独占；`--wait none` 报 FAIL 是**正常的**（它故意不保证正确性） |
| Tier 1 全部 `space≈0` | tail/prologue 太短，被 launch 开销淹没 | 加大 `--tail`，或看 Tier 1.2 的比例扫描结果 |
| `pilot requires a one-CTA-per-SM grid (P,C <= N)` | 默认 `PILOT_GRIDS` 按 148 SM 设定，本机 SM 更少 | `PILOT_GRIDS="32 64 N" ./run_all.sh tier1p`（N 取本机 SM 数） |
| `pilot requires >=2 resident CTAs/SM` | `--threads` 太大，occupancy 掉到 1 | 减小 `--threads`；pilot 靠 ≥2 驻留 CTA 才能区分等待与排队 |
| `SAMPLE and SUMMARY_PILOT tag sets differ` | `pilot_matrix.log` 混进了未跑完步骤的日志 | 只拼接有 `.done` 的步骤（驱动脚本已这样做）；手工拼接时同理 |
| `analyze.py` 输出里出现陌生列 | 把 pilot 日志喂给了 `analyze.py` | 两个脚本 schema 不同，pilot 日志用 `analyze_pilot.py` |
| vllm OOM | `--gpu-mem-util` 太高或序列太长 | 降到 0.85，或减 `--seq` |
| vllm 起不来 | 模型没下完 | `du -sh ~/.cache/huggingface` 对照 54GB |
| LLM 三档差异极小 | PDL 没真正启用 | 确认 `VLLM_USE_FULL_CUDA_GRAPH=1`；piecewise 模式下 vLLM 会关掉 PDL |
| DSA 1M 上下文 OOM | 预期 | 脚本已记录并继续，不影响其他点 |
| 断线丢进程 | 没用 tmux | 重连后 `tmux attach -t cta`；所有脚本可断点续跑，直接重跑即可 |

**所有驱动脚本都支持断点续跑**：已完成的步骤有 `.done` 标记，重跑会跳过。要强制重来用 `--fresh`。

---

## 7. 最小可交付

权威定义在 [`docs/cta_pdl_eval_plan.md`](docs/cta_pdl_eval_plan.md) §10.2，下面是上机离线查阅的镜像。
如果时间或预算被砍，按这个优先级保底：

1. **Tier 1.1 的依赖度 × grid 收益地图** —— 单个信息量最大的实验，决定整个方向是否成立
2. **Tier 4 的 `Ceiling − PDL_grid`** —— 真实负载上还剩多少空间
3. **Tier 0.1 的重叠层数** —— 决定 B3 维度哪些选项可达
4. Tier 0.3 的 occupancy 曲线 —— B2 维度定价

前两项加起来约 4 GPU-hours，足以支撑"这个方向值不值得做"的判断。
