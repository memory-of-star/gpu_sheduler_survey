# Codex campaign workflow

用 Codex 在租用 GPU 机器上无人值守执行 [`../EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md) 的一整场
实验。规则以 [`../AGENTS.md`](../AGENTS.md) 为准，本目录只负责**怎么把它跑起来**。

---

## 1. 职责划分：脚本管测量，agent 管判断

这是本工作流唯一的设计决定，其余都是它的推论。

| 谁 | 负责什么 | 为什么 |
|---|---|---|
| **脚本** | preflight、冒烟、`run_session.sh`、`collect.sh`、§6 分支、文档自检 | 这些已经被计划规定死了，每次必须一模一样。把两小时的扫参放进 agent 的一个 turn 里，是掉线之后整场作废的标准方式 |
| **Codex** | 声明实验坐标、读判决及其 caveat、按 §7 写报告、同步 `EXPERIMENT_REPORT_INDEX.md` | 只有这几件事需要判断力和中文行文 |

所以 agent 只在长任务**之前和之后**被调用，不在中间。

## 2. 一条命令

```bash
cd cta_level_PDL_design
./codex/run_campaign.sh
```

它按顺序跑七个 stage，每个 stage 落一个 `codex/state/<stage>.done`，重跑自动跳过已完成的。
掉线重连后直接再跑一遍即可。

| stage | 谁执行 | 做什么 |
|---|---|---|
| `audit` | codex | 机器角色、设备事实、离线判决链自检、harness 是否被否决、**声明坐标 (a)–(d)**、危险项检查 |
| `smoke` | 脚本 | `FAST=1 ./run_session.sh` 打到 `bench/results_smoke`，证明管路通 |
| `measure` | 脚本 | `./run_session.sh`：Tier 0 + Tier 1p（含多波）+ gate |
| `tier1` | codex | 写 `reports/tier1_benefit_map/multiwave_degree_grid_map.md` 并回写索引 |
| `branch` | 脚本 | 读 `gate.json`，写 `codex/state/branch.json`，决定后面哪些 stage 允许跑 |
| `tier4` | 脚本 + codex | LLM 三档，产出 `Ceiling − PDL_grid`；仅在 branch 允许时执行 |
| `wrapup` | codex | 总报告、`collect.sh`、最终文档自检 |

单独跑某几个 stage：

```bash
./codex/run_campaign.sh tier1 branch          # 只补报告和分支
./codex/run_campaign.sh --fresh               # 忽略所有 .done，从头来
DRY_RUN=1 ./codex/run_campaign.sh             # 只打印每个 stage 会做什么
SKIP_CODEX=1 ./codex/run_campaign.sh          # 只跑脚本部分，验证管路
```

## 3. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RESULTS` | `results` | `bench/` 下的结果目录。**历史 corrected 数据不要覆盖**，新跑用 `results_<tag>` |
| `STEP_TIMEOUT` | `1800` | 单个 benchmark 调用的秒级上限，见下面第 5 节 |
| `CODEX_MODEL` | 用 `~/.codex/config.toml` 的 | 覆盖模型 |
| `CODEX_SANDBOX` | `workspace-write` | Tier 4 要下模型，脚本已顺带打开 network access；整机可弃用时可设 `danger-full-access` |
| `CAMPAIGN_COMMIT` | `0` | 设 `1` 则 wrapup 之后提交**仅 markdown** 的改动，防止租用机消失带走结论 |

## 4. 文档自检是这套流程的闭环

[`check_docs.py`](check_docs.py) 在**每个 codex stage 之后**自动跑一次，不通过就不落
`.done`。它机器化地检查 `AGENTS.md` 里三条本来只写在散文里的约束：

- §10 子树内每个相对 markdown 链接都能解析
- §7 `reports/` 下每份报告都被 `EXPERIMENT_REPORT_INDEX.md` 引用
- §7 每份报告都有「不能成立的结论」一节（`reports/rejected/` 另按「哪些可复用 / 哪些不可当结论」检查）

「已经叮嘱过 agent 要同步索引」不是控制手段，这个脚本才是。

已审阅的豁免项记在 [`known_debt.txt`](known_debt.txt) 里，每条都要写清楚原因和什么条件下可以销账
（当前为空——整棵树满足全部被检查的不变量）。**不要为了让检查变绿而往里加行**，那正是这个文件
存在时要防的事。

```bash
python3 codex/check_docs.py              # 有 error 则退出 1
python3 codex/check_docs.py --strict     # warning 也算失败
python3 codex/check_docs.py --list-debt  # 看当前豁免了什么
```

## 5. 为什么默认设了 `STEP_TIMEOUT`

fail-soft 的前提是每一步会结束。放开 `P,C ≤ SM` 之后这个前提不再成立：多波时一个已常驻的
consumer CTA 可能在自旋等待一个**还没被调度上去的** producer CTA，而所有等待循环都没有超时。
`strided` 是最坏情况——child 0 的 parent 撒遍整个 producer grid。

真挂住的话，租用机会一直卡在那里，而这恰恰是 fail-soft 契约要避免的结果。所以
[`../bench/run_all.sh`](../bench/run_all.sh) 增加了可选的 `STEP_TIMEOUT`（默认 `0`，即保持
原行为），本工作流把它设成 1800 秒：挂住会被记成一次 step 失败，整场继续。

这只是止损，**不是修复**。真正的修复是在等待循环里加前向进展保证或设备侧超时，属于 `.cu`
的语义改动。

## 6. 跑之前

```bash
# 1. 确认在 GPU 机器上，并且整个仓库都克隆了（Tier 0.2 要用兄弟目录）
command -v nvidia-smi >/dev/null && echo GPU_BOX || echo DEV_BOX
ls ../cross_stream_PDL_survey/bench/pdl_bench

# 2. 确认 codex 能用
codex --version

# 3. 离线判决链先过（不需要 GPU，preflight 也会再跑一遍）
python3 tools/make_test_fixtures.py --out /tmp/ctafix
python3 tools/analyze_pilot.py /tmp/ctafix/pilot_matrix.log \
        --json /tmp/ctafix/pilot_analysis.json --csv /tmp/ctafix/pilot_summary.csv
python3 tools/gate.py /tmp/ctafix/pilot_analysis.json
```

`bench/cta_dep_pilot.cu` 的多波改动是在没有 `nvcc` 的机器上写的，**上机前从未编译验证过**。
`audit` stage 的第一件事就是确认它在这台机器上编得过。

## 7. 目录

```
codex/
├── README.md            本文件
├── run_campaign.sh      编排器：stage、续跑、fail-soft、§6 分支
├── check_docs.py        文档不变量的机器检查
├── known_debt.txt       已审阅的豁免项
├── prompts/             每个 codex stage 的静态提示词（英文，按 AGENTS.md §9）
└── state/               运行时产物：.done、日志、coordinates.md、branch.json
```

`state/` 是本场 session 的执行痕迹，不是报告。结论只写在 `../reports/`，进度只写在
`../EXPERIMENT_REPORT_INDEX.md`。
