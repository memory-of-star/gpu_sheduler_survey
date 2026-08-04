# Corrected Producer–Consumer Pilot 实验报告设计

日期：2026-08-04  
状态：用户已确认报告范围与组织方案

## 1. 目标

在 `cta_level_PDL_design/` 下新增一份独立、可复核的 Markdown 实验报告，完整描述已经执行的 corrected producer–consumer pilot。报告必须让读者不查看 CUDA 源码，也能准确理解：一次 invocation 中有几个 kernel、每个 kernel 有多少 CTA、每个 CTA 有多少线程、producer 与 consumer 如何建立依赖、各时间段代表什么，以及参数变化如何影响结果。

正式交付文件：

`cta_level_PDL_design/EXPERIMENT_REPORT_CORRECTED_PRODUCER_CONSUMER_PILOT.md`

## 2. 报告范围

报告主体覆盖 `bench/cta_dep_pilot.cu` 实际完成的全部 corrected pilot：

- 8 个实验 family；
- 每个 family 使用 seed 101、202、303，共 24 个配置；
- 每个配置运行 5 种模式；
- 每种模式 3 次 warmup、31 次 timed repeat；
- 共 3,720 个 timed samples；
- 96 个需要正确性验证的模式通过，24 个 `none` 模式按设计不提供正确性保证。

8 个 family 为：

1. interval degree 1；
2. interval degree 8；
3. interval degree 32；
4. interval degree 64；
5. strided degree 32；
6. interval degree 8、tail=0；
7. interval degree 8、tail=2M cycles；
8. interval degree 8、grid=64。

默认点为 grid=148、ready=400K cycles、tail=1M cycles、consumer prologue=200K cycles、consumer epilogue=1M cycles、128 threads/CTA、8 个 readiness skew bins。

原 `cta_dep_bench` FAST sweep 不纳入有效结果表。报告设置独立的“弃用实验”章节，说明其实际执行范围、trigger 时机、counter、验证与返回码问题，防止读者误用 `results_budget1h/summary.txt`。

## 3. 事实来源与优先级

事实来源按以下优先级引用：

1. `bench/results_budget1h_corrected/pilot_matrix.log`：逐次 GPU timing、模式验证和每配置 SUMMARY；
2. `bench/results_budget1h_corrected/pilot_summary.csv`：24 个配置的逐配置中位数与 bootstrap CI；
3. `bench/results_budget1h_corrected/pilot_analysis.json`：按 family 聚合的统计结果；
4. `bench/cta_dep_pilot.cu`：kernel、trigger、flag、验证与计时语义；
5. `tools/analyze_pilot.py`：统计公式和聚合口径；
6. `EXPERIMENT_REPORT_B200_1GPUH.md`：已有审计结论与限制条件。

若概述文档与源码或原始日志存在差异，以源码和原始日志为准，并在新报告中明确指出。

## 4. 报告组织

正式报告采用“公共实验设计 + 分变量分析 + 完整配置附录”的结构：

1. 执行摘要：一句话结论、证据等级和禁止外推的范围；
2. 实验问题：标准 grid PDL 与软件 CTA readiness 分别解决什么问题；
3. CUDA 执行层级：两个 kernel launch、每个 grid 的 CTA 数、每 CTA 线程数，明确 CTA 等同于 thread block；
4. Producer–consumer 时间线：ready、publish、tail、prologue、wait、epilogue 的先后与可重叠区间；
5. 同步与内存语义：device-scope release/acquire flag、leader wait、CTA barrier、标准 grid wait；
6. 五种测试模式：`none`、`grid`、`interval-spin`、`interval-backoff`、`exact-backoff`，并标注 Floor、Ceiling、预声明 Impl；
7. 固定参数和变化参数：用表格区分控制变量、实验轴和 seed；
8. 执行与统计方法：warmup、repeat、中位数、跨 seed 汇总、bootstrap、correctness invocation；
9. 完整结果总表：8 个 family 的跨 seed 汇总；
10. 分变量分析：degree、结构/编码、tail 几何、grid size、重复性；
11. 正确性证据：哪些模式验证、如何 poison/reset、`none` 为什么不验证；
12. 弃用的原始 FAST 实验；
13. 成立的结论、不能成立的结论和下一步 applicability gate；
14. 附录：24 个配置的完整统计表、公式、源码与原始产物索引。

报告使用 Markdown 表格和等宽文本时间线，不生成新的图片或重新运行 GPU 实验。

## 5. 术语和结果口径

报告统一使用以下定义：

- Producer/consumer 各是一个 kernel launch；148 或 64 表示 grid 中的 CTA 数，不表示 kernel 数。
- CTA 与 CUDA thread block 在本报告中等价。
- Floor：`grid`，标准 grid-level PDL，producer 在数据 ready 后 trigger，consumer 使用 `cudaGridDependencySynchronize()`。
- Impl：`interval-backoff`，producer 入口 trigger，写数据后以 device-scope release 发布 CTA flag，consumer 以 acquire/backoff 等待 interval parent。
- Ceiling：`none`，producer 入口 trigger、consumer 不等待；它是不安全的 timing reference，不是正确实现或硬件上界。
- 理论空间：`(Floor - Ceiling) / Floor`。
- 软件收益：`(Floor - Impl) / Floor`。
- 空间兑现比例：`(Floor - Impl) / (Floor - Ceiling)`。
- speedup：`Floor / Impl`。

所有“收益”默认表示 latency reduction；同时给出 speedup，避免百分比口径混淆。

## 6. 必须突出呈现的结果

- tail=0：Floor 1.001568 ms、Ceiling 0.619872 ms、Impl 0.901792 ms，speedup 1.1105×、latency reduction 9.952%、gap captured 26.150%。
- 默认 tail=1M 的 interval degree 1–64：latency reduction 34.989%–36.082%，gap captured 96.469%–99.555%；Impl 相对 Ceiling 的额外时间随 degree 增长。
- strided degree 32：interval tightness 0.2264；interval-backoff 相对 exact-backoff 慢约 20.2 µs，说明假边/编码结构是独立变量。
- tail=2M：绝对 Floor–Ceiling gap 与默认点接近，但相对百分比因总时长分母变化而下降；必须同时报告绝对量与相对量。
- grid 64 与 148：只在 underfilled/one-wave 范围内差约 1 µs，不能外推 multi-wave。
- 3,720 个 timed samples 中有少量正向高尾；报告中保留全部样本并解释中位数的稳健性。

## 7. 限制与措辞约束

报告不得使用下列不受证据支持的表述：

- “真实 LLM/DSA 可获得约 35% 收益”；
- “Ceiling 是可实现硬件上界”；
- “Floor→Impl 的全部差异来自 CTA 粒度”；
- “degree 256 或 8192 仍能保持同样开销”；
- “已证明 multi-wave、occupancy 1–2 或真实资源竞争下有效”；
- “当前 pilot trace 已直接证明 consumer launch 早于 producer ready”。

必须明确：Floor 与 Impl 同时改变 trigger 时机和 wait/flag 协议；实验只有 one-wave、低资源、synthetic spin payload；consumer epilogue 是固定时长 placeholder；bootstrap CI 只反映当前 session 的 repeat noise。

## 8. 验收标准

正式报告完成时必须满足：

- 24 个配置和 3,720 samples 的计数与原始日志一致；
- 8 个 family 的汇总数字与 `pilot_analysis.json`/`pilot_summary.csv` 一致；
- 全部参数能够追溯到源码、日志或统计文件；
- 清楚区分 GPU 实测、CPU 派生统计、unsafe reference 和未运行项目；
- 没有 `TBD`、`TODO`、占位符或互相矛盾的结论；
- Markdown 内所有本地证据链接指向存在的文件；
- 不修改任何实验数据、CUDA 源码或原始结果。
