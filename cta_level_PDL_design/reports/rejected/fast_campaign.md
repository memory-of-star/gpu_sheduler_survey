# 原 FAST Producer–Consumer Campaign：执行记录与弃用审计报告

日期：2026-08-03（UTC）  
设备：单卡 NVIDIA B200，148 SM，Compute Capability 10.0  
实验状态：**REJECTED：确实执行过，但不得用于 CTA-level benefit、boundary 或 protocol winner 结论**

## 1. 执行摘要

原 FAST campaign 使用 `cta_dep_bench` 执行了 degree、structure、tail/prologue、protocol、smem 和 trace sweep。

执行规模：

- 34 个 `cta_dep_bench` 配置被尝试；
- 33 个写出 `SUMMARY`，1 个 64 KiB smem 配置 launch 失败；
- 31 个完整配置各运行 6 个 wait modes × 5 timed repeats；
- 2 个 trace 配置各运行 1 个 mode × 5 repeats；
- 合计约 940 个 timed invocations，不含 warmup；
- 另有旧 Tier 0 facts 运行失败、CLC 单点成功。

但是 benchmark 有四个阻塞性语义问题：

1. producer 在 trigger 前已经发布所有 `done[]`，CTA wait 不再测 readiness；
2. global counter 协议不能从完成数量推导特定 parent 前缀已完成；
3. timed consumer 普通工作量随 degree 线性增长；
4. correctness 可能被上一轮相同数据掩盖，且失败不会使进程返回非零。

因此这些数字只能证明“旧 campaign 跑过、harness 有问题”，不能解释为 CTA-level PDL 无收益或某协议胜出。有效 benefit 结果只来自后续 corrected pilot。旧 campaign 中独立的 CLC probe 不依赖 `cta_dep_bench`，其 corrected rerun 单独记录在 [reports/tier0_base_facts/0_4_clc_try_cancel.md](../tier0_base_facts/0_4_clc_try_cancel.md)。

## 2. 已执行矩阵

| Family | 配置 | 数量 | 状态 |
|---|---|---:|---|
| Tier 1.1a degree | grid 256/1024 × interval d1/d8/d64 | 6 | 完成但无效 |
| Tier 1.1b structure | grid 256/1024 × interval/grouped d32 | 4 | 完成但无效 |
| Tier 1.2 tail ratio | P=C=1024，d8，tail/prologue ratio 1/2/4/8/16 | 5 | 完成但无效 |
| Tier 2.1 protocol | self d1，grid 256/1024/4096 | 3 | 完成但无效 |
| Tier 2.3 encoding | P=C=2048，interval/strided/random × d4/d16/d64 | 9 | 完成但无效 |
| Tier 0.3 dep+smem | P=C=2048，d8，smem 0/8/16/32/64 KiB | 5 | 4 完成；64 KiB failed |
| CTA trace | P=C=1024，interval/random d16 | 2 | trace 生成但 readiness 语义无效 |

完整模式：

- none；
- grid；
- cta-spin；
- cta-backoff；
- cta-counter；
- cta-exact。

每个完整配置先做 3 次 warmup，再做 5 次 timed repeats，报告中位数和最小值。

## 3. 原始观测范围

下表只记录旧日志出现了什么，不赋予 benefit 含义：

| Family | 旧 `space_pct=(grid-none)/grid` 范围 |
|---|---:|
| Tier 1.1a degree | -0.747% 到 -0.051% |
| Tier 1.1b structure | -0.993% 到 +0.239% |
| Tier 1.2 tail ratio | -1.136% 到 +0.779% |
| Tier 2.1 protocol | -0.545% 到 +0.379% |
| Tier 2.3 encoding | -1.157% 到 +0.010% |
| Tier 0.3 smem（成功点） | -0.121% 到 +0.388% |

这些接近零甚至为负的 gap 不是“CTA PDL 没有潜力”的证据；它们主要反映旧执行语义退化、普通工作量和 timer noise。

## 4. 阻塞问题一：Trigger 时机使 CTA wait 退化

旧 producer 顺序为：

~~~text
写 data[cta]
    ↓
release store done[cta] = 1
    ↓
counter++
    ↓
cudaTriggerProgrammaticLaunchCompletion()
    ↓
tail
~~~

dependent grid 只有在所有 producer CTA 都 trigger 或退出后才有 launch eligibility。由于每个 CTA 都在 trigger 前发布自己的 flag，consumer 获得 eligibility 时全部 `done[]` 已经发布。

因此 cta-spin/backoff/exact 在启动后扫描的是已经 ready 的 flags，测到的是扫描/解码开销，不是真实 parent readiness wait。

## 5. 阻塞问题二：Global counter 不正确

旧 `WAIT_COUNTER` 等待：

~~~text
global_completed_count >= hi + 1
~~~

这不能推出 `[0,hi]` 的所有 parent 都完成。CTA 可以乱序完成；若高编号 CTA 先完成，它们也会增加同一个 counter，使数量达到阈值，而某个低编号真实 parent 仍可能未完成。

源码注释声称该协议只是“保守地多等”，这一注释不成立。全局 cardinality 不包含完成 CTA 的 identity。

## 6. 阻塞问题三：Timed payload 为 O(degree)

旧 consumer 在 wait 之后、CUDA event 计时范围内遍历每个真实 parent 并累加 `pin[parent]`。

所以 degree 增长同时增加：

- wait/flag 工作；
- parent id 解码；
- global load；
- 浮点累加和循环。

例如 d64 比 d1 慢不能被解释成同步成本增长。corrected pilot 将 timed post-wait 普通 payload 固定为 O(1)，把完整 parent 检查移到额外 validation invocation。

## 7. 阻塞问题四：Correctness 证据不可靠

旧 harness 每轮只清零 flags/counter，不 poison producer data 和 consumer output。producer 每轮又写相同的确定值。

如果某个 consumer 因竞态过早读取，它可能读到上一轮遗留的同一个“正确值”，导致验证 PASS。

此外：

- 验证只检查每个 mode 最后一次输出；
- none 按设计不验证；
- 即使某个应正确 mode 打印 FAIL，程序最后仍无条件 `return 0`；
- driver 因此仍可能写 `.done`；
- 下游分析可能继续把失败模式当候选。

所以日志中的 PASS 不能修复 benchmark 的同步语义问题。

另外还有两个次要混杂：random parent 生成允许重复 id，requested degree 不一定等于 unique parent 数；grid wait 只有 leader 调用，而其它软件 wait 路径在 leader wait 后执行 CTA barrier，线程参与和同步路径并不完全对称。

## 8. 两个独立 launch 错误

### 8.1 Tier 0 zero-smem illegal access

旧 `waiterK` 在 smem=0 时仍访问 `g_smem[0]`，导致：

~~~text
CUDA error tier0_facts.cu:157: an illegal memory access was encountered
~~~

后续 corrected source 只在 smem>0 时触碰动态 shared memory。

### 8.2 64 KiB dynamic smem invalid launch

原 `t03_smem64` 报：

~~~text
CUDA error cta_dep_bench.cu:128: invalid argument
~~~

原因是 64 KiB dynamic smem 超过默认 launch ceiling，未调用 `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` opt-in。修正后独立 smoke 成功。

## 9. Trace 为什么也不能用于 benefit

`trace_interval` 和 `trace_random` 确实生成了 per-CTA CSV，但使用同一个“flag publication before trigger” producer。它们可以显示旧程序的 CTA 时间线，却不能证明 consumer 在 parent readiness 之前启动并发生了真实 wait。

此外，旧 timeline 工具曾把 `t_dep-t_launch` 标成 dependency stall；修正后明确该区间包含 consumer prologue 和可能的 wait。

## 10. 哪些信息仍可保留

可以保留：

1. campaign 实际执行的配置清单与日志；
2. 两个 launch 错误及修复依据；
3. interval cover 在 strided/random 上会造成巨大扫描范围这一代码路径事实；
4. 构建 corrected pilot 时需要避免的问题清单。

不得保留为结论：

1. Floor→Ceiling benefit 或 lack of benefit；
2. 最快 wait protocol；
3. degree/structure/tail 边界；
4. smem 对正确 CTA readiness 的影响；
5. 旧 trace 对真实 readiness overlap 的证明。

## 11. 与 corrected pilot 的关系

后续 [reports/tier1_benefit_map/corrected_producer_consumer_pilot.md](../tier1_benefit_map/corrected_producer_consumer_pilot.md) 通过以下改动重新测试：

- software/no-wait 路径在 producer 入口 trigger；
- readiness work 在 trigger 后、release flag 前；
- 删除 counter；
- timed post-wait 普通 payload 固定 O(1)；
- 每次 poison data/out 并清 flags/error；
- 额外 validation 检查全部真实 parent；
- validation 失败使进程返回非零。

只有 corrected pilot 的数值可用于本目录的 producer–consumer benefit 结论。

## 12. 证据入口

- 旧 benchmark 源码：[bench/cta_dep_bench.cu](../../bench/cta_dep_bench.cu)
- 旧 wait 实现：[bench/common/dep_wait.cuh](../../bench/common/dep_wait.cuh)
- Campaign log：[bench/results_budget1h/campaign.log](../../bench/results_budget1h/campaign.log)
- 汇总：[bench/results_budget1h/summary.txt](../../bench/results_budget1h/summary.txt)
- Failures：[bench/results_budget1h/failures.log](../../bench/results_budget1h/failures.log)
- 旧结果目录：[bench/results_budget1h/](../../bench/results_budget1h/)
- Corrected 报告：[reports/tier1_benefit_map/corrected_producer_consumer_pilot.md](../tier1_benefit_map/corrected_producer_consumer_pilot.md)
- B200 总报告：[reports/campaign_b200_1gpuh.md](../campaign_b200_1gpuh.md)
