# Stage: audit — pre-run audit and declared coordinates

You are the agent for one rented-GPU campaign on the CTA-level PDL project. This stage runs
**before any GPU time is spent**. Nothing here may take more than a few minutes.

`AGENTS.md` in this directory governs everything you do. Read it first. Read
`EXPERIMENT_PLAN.md` §3 (admissibility), §5.3 (multi-wave), §6 (gate), §12 (budget), §13
(driver requirements), and `EXPERIMENT_REPORT_INDEX.md` §0–§2 (where the campaign stands).

## What this stage must establish

1. **Machine role.** Run `command -v nvidia-smi >/dev/null && echo GPU_BOX || echo DEV_BOX`.
   If DEV_BOX, say so plainly and treat every measurement claim below as `NOT EXECUTED`.
   If GPU_BOX, record the actual device: name, SM count, compute capability, driver, CUDA
   version. Reports name the real device, never the aspirational one (`AGENTS.md` §3).

2. **The analysis chain works before the GPU does.** Run the fixture chain from
   `EXPERIMENT_PLAN.md` §10.2 and confirm `tools/gate.py` produces a verdict. A session that
   measures perfectly but cannot reach a decision has wasted the machine.

3. **Harness admissibility.** For each phase this campaign will run, confirm from
   `bench/README.md` that the harness is not `REJECTED`. `tier1p` (`cta_dep_pilot`) is the
   only admissible source of Tier 1 gate data. `tier1` and `tier23` drive `cta_dep_bench`
   and must not run.

4. **Sweep coverage against plan §5.3.** Determine which grids `run_all.sh tier1p` will use
   on this device — it derives them from the SM count — and state whether the required
   `2×`, `8×`, `32× SM` points are all present. `tools/gate.py` reports this as
   `plan_multi_complete`; if it will come out false, say so now, not after the run.

5. **Declared coordinates, per `AGENTS.md` §5.** For every experiment this session will run,
   state (a) dimension and option rows, (b) which bracket points it produces, (c) the
   decision it changes, (d) GPU budget in minutes against the ~8 GPU-hour total.
   **If (c) is empty for an experiment, mark it DO NOT RUN and say why.**

6. **Known hazards for this specific campaign.** Check each and report what you find:
   - `bench/cta_dep_pilot.cu` was edited on a machine with no `nvcc`. Until `preflight.sh`
     builds it here it is unverified. Confirm it compiles before anything else.
   - The multi-wave change removed the `P,C <= SM` cap. In that regime a consumer CTA can
     be resident and spinning on a producer CTA from a wave that has not been scheduled
     yet; there is no forward-progress guarantee and no timeout in the wait loops. The
     `strided` structure is the worst case, because child 0's parents span the whole
     producer grid. Confirm `STEP_TIMEOUT` is set so a hang is recorded as a step failure
     instead of stalling the rented machine.

## Output

Write `codex/state/coordinates.md`, in Chinese, containing: machine and device facts, the
fixture-chain result, the harness admissibility table, the grid coverage statement, the
declared coordinates table (a)–(d), and the hazard checklist with what you actually observed.

## Prohibitions

- Do not run any full sweep, any Tier 4 download, or anything that takes GPU minutes.
- Do not write anything into `reports/` at this stage — there are no results yet.
- Do not state a number you did not read out of a file or a command's output.
