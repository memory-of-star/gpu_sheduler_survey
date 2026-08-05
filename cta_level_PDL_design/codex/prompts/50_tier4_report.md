# Stage: tier4 — the real-workload bracket, `Ceiling − PDL_grid`

`codex/state/branch.json` allowed this stage, and `bench/llm/run_llm_sweep.sh` has run. Read
`EXPERIMENT_PLAN.md` §8 in full before writing anything.

**This is the second-highest-priority number in the whole project** (`EXPERIMENT_PLAN.md`
§12): after grid-level PDL is already on in production, how much headroom is left.

## The one thing that can invalidate this stage

**Floor is `PDL_grid`, not `PDL_off`** (`EXPERIMENT_PLAN.md` §8.3, `AGENTS.md` §4 rule 7).
TensorRT-LLM, vLLM and SGLang already ship grid-level PDL. Using PDL-off as the baseline
counts the benefit grid-level PDL has already taken and charges it to CTA-level.

`tools/llm_bracket.py` warns when the `PDL_off → PDL_grid` gain falls outside the published
2–33% band. That warning usually means FULL CUDA graph mode is off and PDL was silently
disabled, i.e. the Floor is not the production configuration. **If that warning fires, the
`Ceiling − PDL_grid` number is not usable — fix the configuration and re-run, or report the
run as inconclusive. Do not publish the number with a caveat attached.**

## Read

- `bench/llm/results_llm/summary_llm.txt` — raw three-rung records
- the output of `python3 tools/llm_bracket.py bench/llm/results_llm/summary_llm.txt`
- `bench/<results>/device.txt`, and the environment provenance from `collect.sh`
- `codex/state/branch.json` — which verdict allowed this stage and why

## The report

Path: `reports/tier4_llm/ceiling_minus_pdl_grid.md`. Chinese prose, all eight sections from
`AGENTS.md` §7. Headline number: `Ceiling − PDL_grid`, per configuration and as a range.

Cover:

- All three rungs (`PDL_off`, `PDL_grid`, `Ceiling`) with the sweep axes actually run —
  batch size and sequence length, per `EXPERIMENT_PLAN.md` §8.5. BS=1 decode is where the
  grid is smallest and the overlap space largest; say whether that showed up.
- The relationship to the Tier 1 microbenchmark result. Tier 1 measures a synthetic bracket;
  this measures a deployed stack. If they disagree, the real workload wins and the report
  says so.
- Ceiling here removes `gdc_wait` and therefore **computes wrong results by construction**.
  Report its time only.

## 不能成立的结论 — at minimum

- One model, one GPU, one framework version. Nothing here generalises to other stacks.
- Kernel-level attribution needs `nsys`/`ncu`; without them, the report cannot say which
  kernel pair the headroom sits in, only that it exists.
- `Ceiling − PDL_grid` is an upper bound on what any CTA-level mechanism could recover, not
  a prediction of what one would recover.

## Also update

`EXPERIMENT_REPORT_INDEX.md` §0, §2 (§8 rows), §3 (minimum deliverable item 2), §4. Then run
`python3 codex/check_docs.py` and fix what it reports.

## Prohibitions

- Do not download a model on a machine that cannot hold it; check free disk and GPU memory
  first and record the check.
- Do not quote a number the sweep did not produce. A configuration that OOMed is a coverage
  gap to state, not a gap to interpolate across.
