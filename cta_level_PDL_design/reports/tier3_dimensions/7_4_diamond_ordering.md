# Tier 3：CTA diamond 的 in-order completion 代价（§7.4）

| 项 | 值 |
|---|---|
| 报告 / 实验日期 | 2026-08-05（UTC） |
| 设备 | NVIDIA B200，148 SM，CC 10.0；Driver 580.126.09，CUDA 13.0 |
| 正式结果 | `bench/results_20260805_b200_tier23_native_v2/` |
| 证据等级 | **A-：native CUDA 真机四路径、全 stage/block 校验、31-repeat bootstrap CI 与 `%globaltimer` trace** |

## 1. 执行摘要

把 `K1→K2→K3→K4` 的强制顺序改成真实 diamond
`K1→{K2,K3}→K4`，在 K2:K3 从 1:1 扫到 1:10 时，每一点都恢复约 0.155 ms 的
K2/K3 branch-envelope overlap。绝对 ordered penalty 基本不随 K3 变长：1:1 为 0.155040 ms，
1:10 为 0.155744 ms；以 CTA-ordered 为分母的可恢复比例从 19.74% 降到 7.20%。这否定了“只要两支时长
差变大，flattening 的相对损失就必然增大”这一简单推断。

| K2:K3 | grid ordered ms | CTA ordered ms | CTA unordered ms | unordered branch-envelope overlap | Ceiling ms |
|---:|---:|---:|---:|---:|---:|
| 1:1 | 1.228992 | 0.785536 | 0.630496 | 0.154720 | 0.319584 |
| 1:2 | 1.382496 | 0.939616 | 0.785184 | 0.154976 | 0.465056 |
| 1:4 | 1.688352 | 1.244224 | 1.088288 | 0.155008 | 0.770976 |
| 1:6 | 1.994240 | 1.551968 | 1.395968 | 0.155008 | 1.076416 |
| 1:8 | 2.300544 | 1.860896 | 1.706656 | 0.154944 | 1.383744 |
| 1:10 | 2.606048 | 2.162144 | 2.006400 | 0.154880 | 1.688064 |

## 2. 程序实际做了什么

[`bench/tier23_diamond.cu`](../../bench/tier23_diamond.cu) 启动四个各 148 CTA 的 stage。
grid Floor 在同一 stream 上用 `cudaLaunchAttributeProgrammaticStreamSerialization`
（programmatic dependent launch / PSS），把四个 grid 全部串成
`K1>K2>K3>K4`；`cta-ordered` 在四条独立 stream 上按 per-CTA epoch flag 保留相同强制边；
`cta-unordered` 只删除 `K2>K3`，改为 K2/K3 都逐 CTA 等 K1，K4 仍逐 CTA 等两支。
`none` 删除全部 readiness wait，必须读到 poison，只提供 unsafe 操作参考时间。

K2 work 固定 300,000 cycles，K3 为其 1–10 倍；每 stage 另有相同 300,000-cycle tail。
每次 invocation 都重新 poison 四个 stage 的全部数据。安全档在独立 validation invocation
逐 stage、逐 block 比较完整输出；timed invocation 的依赖点立即 snapshot 输入，防止把
“先空转、后碰巧读对”误当无 wait 路径。

## 3. 配置与统计

* 10 个 ratio 配置；每配置 3 warmups、31 timed repeats、四档奇偶反序。
* 1,240 samples、23,680 条 trace、0 validation errors；每点 2,000 次 bootstrap median CI。
* 128 threads/CTA、16 KiB dynamic shared memory，occupancy API 报 13 blocks/SM，满足四 stage
  混合驻留的资源下界。
* `branch_overlap_ms` 由 K2/K3 各自 `[min(t_dep), max(t_ready)]` 的 grid envelope 交集直接重算，
  不是 host event 推断，也不是逐 CTA 重叠或利用率积分。
* 同正式 binary SHA 的 non-timing Compute Sanitizer 覆盖四档并报
  `ERROR SUMMARY: 0 errors`。

## 4. 头条数字复算

```text
ratio 1:1:
  ordered penalty = 0.785536 - 0.630496 = 0.155040 ms
  relative        = 0.155040 / 0.785536 = 19.7368%

ratio 1:10:
  ordered penalty = 2.162144 - 2.006400 = 0.155744 ms
  relative        = 0.155744 / 2.162144 = 7.2032%
```

十个点的 `cta-unordered` overlap 中位数范围为 0.154720–0.155072 ms；`cta-ordered`
全部为 0。测得 penalty 与被恢复 overlap 数值一致，支持归因于唯一删除的 `K2>K3` 边。

## 5. 可以成立的结论

1. 在该 CTA diamond 和固定 `K2=300,000 cycles` 的扫描上，强制 in-order completion
   损失约 0.155 ms；保留非线性拓扑可以稳定恢复它。
2. 在这个固定 K2 点，penalty 对 K3/K2 比例近似不敏感；证据支持 A1/B3 设计保留拓扑方向，
   但不证明 penalty 普遍由短支工作量决定或随短支线性缩放。
3. grid ordered 与 CTA ordered 的额外差距包含 grid 粒度 barrier；CTA unordered 进一步
   隔离出单条伪顺序边的代价。

## 6. 不能成立的结论

1. 不能声称“ratio 越大，flattening 相对损失越大”；本方向的扫描实测相反。
2. Ceiling 计算错误，不能拿它证明可实现 speedup 或 correctness。
3. 本实验是合成 spin-cycle diamond，不代表某个具体生产模型的 kernel duration 分布。
4. 结果没有测 pre-dispatch gating、Rubin 或集中式硬件 scheduler。

## 7. 证据入口

* 源码：[`bench/tier23_diamond.cu`](../../bench/tier23_diamond.cu)
* strict verdict：[`tier23_validation.json`](../../bench/results_20260805_b200_tier23_native_v2/tier23_validation.json)
* 汇总：[`tier23_summary.csv`](../../bench/results_20260805_b200_tier23_native_v2/tier23_summary.csv)
* 每点 raw/trace：[`bench/results_20260805_b200_tier23_native_v2/`](../../bench/results_20260805_b200_tier23_native_v2/)
* sanitizer coverage：[`sanitizer_v2_coverage.json`](../../bench/results_20260805_b200_tier23_native_v2/sanitizer_v2_coverage.json)
