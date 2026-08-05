# Stage: wrapup — umbrella report for this rented session

Every measurement stage that was going to run has run. Write the one report that says what
this session bought, and leave the tree in a state the next session can pick up cold.

Model to follow: `reports/campaign_b200_1gpuh.md`, the umbrella report for the previous
session. Yours goes to `reports/campaign_<device>_<budget>.md` using the **real** device from
`bench/<results>/device.txt` and the real elapsed budget from `codex/state/campaign.log`.

## Assemble from what is on disk

- `codex/state/coordinates.md` — what this session declared it would measure, and why
- `codex/state/branch.json` — the §6 verdict and what it allowed
- `codex/state/failures.log` and `bench/<results>/failures.log` — what did not run
- every report written during this session
- `bench/<results>/` — the raw evidence for every number you quote

## The umbrella report must contain

1. Header with real device, real dates, and an evidence grade for the session as a whole.
2. **Scope and non-execution.** What was planned, what ran, what did not, and why. This
   section is as important as the results: a session that reports only its successes cannot
   be resumed correctly.
3. Budget actually spent, per tier, against the ~8 GPU-hour total in `EXPERIMENT_PLAN.md`
   §12. If a tier came in far under budget, say so — it changes what the next session can
   afford to widen.
4. The §6 verdict, quoted from `gate.json` with its caveats, and what it permits next.
5. Minimum-deliverable status against `EXPERIMENT_PLAN.md` §12's four priorities.
6. 能成立的结论 / **不能成立的结论**, at session scope.
7. Evidence entry points: raw directories, per-tier reports, and the environment provenance.

## Leave the tree resumable

- `EXPERIMENT_REPORT_INDEX.md`: §0 current status, §2 plan-section checklist, §3 minimum
  deliverable, §4 report list, and the "下一步（按序）" list — rewrite that list so it
  describes what the *next* session should do, not what this one did.
- `bench/README.md` only if the implementation status of a harness actually changed. Progress
  belongs there; execution progress belongs in the index; neither belongs in the plan.
- If an experiment ran and its output turned out to be inadmissible, it gets a report under
  `reports/rejected/` as an audit record — listing what may still be reused and what must
  not be reused as a conclusion. Never delete it.
- Run `python3 codex/check_docs.py` last and fix what it reports.

## Prohibitions

- Do not edit `bench/results*/`, `EXPERIMENT_MANIFEST_SHA256.txt`, result tarballs, or any
  other provenance artefact of a past session. They are historical snapshots.
- Do not record execution progress in `EXPERIMENT_PLAN.md`. The plan is the specification.
- Do not commit anything. `codex/run_campaign.sh` handles the commit, and only for markdown.
- Do not describe a synthetic result as a speedup. Calibrated claims are the product here.
