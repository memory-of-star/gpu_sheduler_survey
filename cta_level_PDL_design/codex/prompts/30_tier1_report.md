# Stage: tier1 — write the Tier 1 multi-wave report

`./run_session.sh` has finished. Tier 0 and Tier 1p ran and the gate was evaluated. Your job
is to turn what is on disk into a report, and to leave the index so the next session can
resume from it alone.

This is the project's top gap being closed: `EXPERIMENT_PLAN.md` §5.3, the degree × grid
benefit map in the multi-wave regime `P,C > SM`. Read `AGENTS.md` §4 and §7, and
`EXPERIMENT_PLAN.md` §1, §3, §5, §6 before writing.

## Read, in this order

- `bench/<results>/gate.json` — the verdict and the statistics behind it
- `bench/<results>/pilot_summary.csv` — per-configuration statistics with bootstrap CIs
- `bench/<results>/pilot_matrix.log` — raw `SAMPLE` and `SUMMARY_PILOT` records
- `bench/<results>/device.txt` — the device you actually ran on
- `bench/<results>/failures.log` — steps that failed; the session continued past them
- `bench/<results>/session.log` — the whole run
- the previous single-wave report, `reports/tier1_benefit_map/corrected_producer_consumer_pilot.md`,
  so the new one states what changed rather than repeating it

## Rules that decide whether the report is admissible

- **Do not re-derive the verdict by eye.** `tools/gate.py` owns `EXPERIMENT_PLAN.md` §6. If
  you disagree with the verdict, the disagreement is with §6 and belongs there, not here.
- **Read the verdict together with its caveats.** `plan_multi_complete` tells you whether
  the `2×/8×/32× SM` set is actually present. A `GO` without it supports "go measure a real
  workload", not "spend the full Tier 2/3 budget".
- **`INVALID` means no timing from this run is usable.** If the verdict is `INVALID`, the
  report says which configurations failed validation and quotes no timings at all.
- **State how the harness satisfies each of `AGENTS.md` §4's rules, with evidence.** The
  `SUMMARY_PILOT` records self-report `trigger_floor=`, `trigger_impl=`, `trigger_ceiling=`,
  `wave=`, `sms=`, `tightness=`, `eff_degree=`, `valid=`; use those. Where a rule cannot be
  checked from the artefacts, write that it cannot be checked. Do not assert compliance you
  did not verify.
- **`--wait none` is the Ceiling and computes wrong results on purpose.** Report its time,
  never its correctness.
- **Any step in `failures.log` is part of the coverage story**, not an omission. Say which
  configurations are missing and what that costs the conclusion.

## The report

Path: `reports/tier1_benefit_map/multiwave_degree_grid_map.md`. Chinese prose, following
`reports/tier0_base_facts/0_5_fence_scope.md` as the structural model, with all eight
sections `AGENTS.md` §7 requires:

1. Header — report date, experiment date, device (name + SM count + compute capability), and
   an evidence grade saying what class of claim this data supports
2. Executive summary carrying the actual numbers
3. What the program actually does — executed semantics, not intent
4. Configuration and statistics — warmup count, timed repeats, timing method, aggregation
5. Recomputation of every headline number from raw values, arithmetic shown
6. 能成立的结论
7. **不能成立的结论** — mandatory; single-wave-vs-multi-wave extrapolation limits, synthetic
   microbenchmark limits, and anything `failures.log` cost you belong here
8. Evidence entry points — source, raw logs, summary lines, umbrella report

Link convention: visible text is the path relative to the subtree root, href is the real
relative path. From `reports/tier1_benefit_map/`, the benchmark source is text
`bench/cta_dep_pilot.cu`, href `../../bench/cta_dep_pilot.cu`.

The single most valuable comparison is **single-wave versus multi-wave at the same degree and
structure**: the old report has the `P,C <= SM` numbers, this run has `2×/8×/32× SM`. Say
whether the benefit structure changed, and by how much. That is the question §5.3 exists to
answer, and it decides whether the direction is worth investment rather than merely feasible.

## Also update, in this same change

`EXPERIMENT_REPORT_INDEX.md`: §0 current status, §2 plan-section checklist (§5.1, §5.2, §5.3,
§6 rows), §3 minimum-deliverable table, §4 credible-report list. The next session must be
able to resume from that file alone.

## Finally

Run `python3 codex/check_docs.py` and fix whatever it reports. Do not add entries to
`codex/known_debt.txt` to silence it.

## Prohibitions

- No number in the report that is not traceable to a file under `bench/results*/`.
- Do not soften or inflate. A synthetic microbenchmark passing a gate is "mechanism feasible
  under stated limits", never "N% speedup".
- Do not edit anything under `bench/results*/` — those are raw session evidence.
- Do not delete or rewrite the existing single-wave report; the new report supersedes nothing.
