# AGENTS.md — CTA-level PDL design project

Scope: this file governs the whole `cta_level_PDL_design/` subtree.

## 1. What this project is

**Goal.** Produce a defensible design for **CTA-level (thread-block-level) inter-kernel
dependency management** — the granularity analogous to the thread-block-level PDL feature
announced for Rubin, whose public description is kept verbatim in
[`rubin_design.md`](rubin_design.md) at the root of this subtree. Today's `griddepcontrol` is
grid-level all-or-nothing; the whole project exists to answer what a CTA-granular replacement
should look like and whether it is worth building.

**The one governing question.** Grid-level PDL is *already deployed* in production inference
stacks. So the number that decides this project is not "is CTA-level faster than no PDL", it is:

> After grid-level PDL is already on, how much headroom is left, and which dimension of the
> design space captures it?

**Deliverable.** A chosen coordinate in the design space of
[`docs/cta_pdl_design_space.md`](docs/cta_pdl_design_space.md), with per-dimension evidence
and explicitly stated limits of that evidence.

**Explicit non-goals.** Shipping a kernel library. Implementing `[H+]` options (they are only
ever bracketed, never built).

**On Rubin — the design description is at `rubin_design.md`, subtree root.** That file is the
verbatim public source (NVIDIA developer blog, Figure 6: "Blackwell bulk triggering" vs "Rubin
tile-level triggering") and is the only Rubin material in this project. **Every claim about Rubin
must trace to it or another citation** — it is motivation, not measured evidence, and nothing
about the implementation may be inferred beyond what it states.

Two things it does state are load-bearing for the design space: Rubin's mechanism is described as
**data-driven polling**, and the consumer begins work **as required input data becomes available**
rather than after a broader dependency resolves. That places the announced feature at a specific
coordinate — resident-wait (B2) plus polling (B1.3), not pre-dispatch gating — which is the
coordinate the software pilot in `reports/tier1_benefit_map/` approximates. Anything further about
how Rubin implements it is unknown and must be written as unknown.

## 2. Document authority

Each file is the single source of truth for exactly one thing. Do not migrate content between
them, and do not duplicate conclusions.

| File | Authoritative for | Rule |
|---|---|---|
| `docs/cta_pdl_design_space.md` | The dimensions (A1–E1) and the options under each | **Deliberately neutral — enumerates trade-offs, gives no recommendation.** Do not write conclusions or benchmark results into it. Every option carries both an implementation tag (`[S]`/`[H-]`/`[H+]`) and a hardware-status tag (`[B300:有/部分/无]`); new options must carry both. |
| `EXPERIMENT_PLAN.md` (subtree root) | **The specification of the experiments**: four-point bracket (§1), admissibility conditions every harness must satisfy (§3), what each experiment measures and how to read it, gate thresholds (§6), budget and minimum deliverable (§12), harness requirements (§13) | **The plan leads, the scripts follow — when they disagree, the script is what changes.** Never narrow the plan to match what the code can currently do; an unimplemented experiment belongs in §13, not deleted. Change only with a stated reason; downstream reports cite it. §6 is implemented by `tools/gate.py` — change the thresholds in both in the same edit. Implementation progress belongs in `bench/README.md`, execution progress in `EXPERIMENT_REPORT_INDEX.md`; **do not record either here**. |
| `docs/cuda_13.4_pdl_clc_interfaces.md` | What today's ISA/runtime actually provides | Facts about existing interfaces only. |
| `rubin_design.md` (subtree root) | **Rubin's design description** — the public statement about its thread-block/tile-level triggering | Verbatim quoted source, and the project's only Rubin material. Do not paraphrase into it or extend it with inference; cite it for every Rubin claim. |
| `papers/README.md` | The reference corpus, indexed by dimension | Add new refs with the dimension they inform. |
| `reports/**/*.md` | Results and conclusions | The only place conclusions live. Grouped by tier — see §7. |
| `EXPERIMENT_REPORT_INDEX.md` | **Execution progress against `EXPERIMENT_PLAN.md`** | Where the campaign stands, what is blocked, how to resume, and links to credible reports. Authoritative for progress / next action — not for experiment specs (plan) or conclusions (reports). Must be updated in the same change as any new/changed report: refresh §0 status, §2 plan checklist, and §4 report list. |
| `run_session.sh` (subtree root) | The rented-GPU session, as executable code | The session procedure is a script, not a document, so it cannot drift from what actually runs. Change the procedure by changing it. |
| `bench/README.md` | Benchmark code contract | Implementation detail of the primitives. |
| `codex/README.md` | **How an agent drives a session** — which stages are script and which are agent | The procedure itself is `codex/run_campaign.sh`; this file only explains the split. It records no results and no progress. |

`archive/` holds side-branch and superseded material, kept only for reference. It is **not
authoritative**: do not cite it as a current source, as a design recommendation, or as evidence.
Existing links into it may stay, but new work should not build on it without first promoting the
content back out of `archive/` deliberately.

## 3. Machine model — establish which machine you are on before anything else

This repo is worked on from two places, and the rules differ. **Detect, do not assume:**

```bash
command -v nvidia-smi >/dev/null && echo "GPU BOX" || echo "DEV BOX"
```

### On the DEV BOX (macOS, no GPU, no `nvcc`)

- **Never present a GPU number that is not traceable to a file under `bench/results*/`.**
  Anything else is `NOT EXECUTED` and must be labelled as such.
- CUDA sources can be written and reviewed but **cannot be compile-checked**. Treat any `.cu`
  edit as unverified until it builds on the GPU box; say so.
- Fully doable here: docs, report writing from existing logs, offline dependency analysis,
  and the whole `tools/` chain (verified under Python 3.14).

### On the GPU BOX (the rented single-GPU machine)

The whole repo is cloned here and an agent runs the session unattended. Here you *may*
compile, run and measure — that is the point — but:

- **Run `./run_session.sh` rather than hand-assembling phases.** It is resumable and
  fail-soft, so re-running after a crash or a dropped connection is always safe and never
  repeats completed work.
- **Record the device you actually ran on** (`bench/<results>/device.txt`). The design-space
  doc reasons about B300 (sm_103); executed experiments so far are **B200 (sm_100)**. Reports
  must name the real device, never the aspirational one.
- Everything in §4 still binds. Being able to run a benchmark does not make its output
  admissible; the rejected harness still produces inadmissible numbers here.

### The autonomous session, end to end

```bash
git clone <repo> && cd <repo>/cta_level_PDL_design
./run_session.sh          # ~3 GPU-hours, unattended: preflight, smoke, Tier 0, Tier 1p, gate
```

It stops at the **decision point** on purpose. Tier 4 is a separate, fail-closed admission:
the model must be fully staged locally, and a target-specific same-process driver must prove the
worker-side PTX/cubin and active executed graph for every declared variant before any timing is
admitted. Tier 5 is not an extension of that driver. Its old Python same-stream path is
permanently rejected; the native strict campaign and the production-fragment campaign are two
independent entry points with independent final admission. Native strict is the CTA bracket path.
Production fragments may admit workload-component timing, but they provide neither a CTA Impl nor
an unordered Ceiling, so their timing must never be presented as a CTA bracket or CTA headroom.
The canonical wider production-formal scope is the exact 26-row matrix. A user-authorized,
one-hour compact campaign is a separate bounded admission, not completion of or a replacement for
that exact-26 scope. Its only admissible matrix is both production models at 4K and 128K across all
three workloads, plus one MoE-32 row per model: 14 correctness rows, 1,302 samples, and 62
summaries with the formal 5 warmups / 31 timed repeats unchanged. It excludes 32K and 1M timing.
Even after its own terminal validator passes, it may set only
`accepted_compact_workload_timing=1`; `accepted_timing`, `accepted_workload_timing`, and
`accepted_CTA_bracket` stay zero, while CTA headroom stays undefined (`headroom_defined=false`,
`headroom_pct=null`). Before that terminal admission exists, the compact campaign is not a PASS.
After it exists, the precise label is **compact-14 scoped formal PASS/DONE**, never production
exact-26 PASS or Tier 5 complete. The current B200 compact-14 campaign has reached that narrow
terminal state; its evidence boundary is fixed by
[`production_compact14_scoped_formal_20260805.md`](reports/tier5_dsa/production_compact14_scoped_formal_20260805.md),
while the exact-26 scope and §9 remain incomplete/partial.
None of these separate admissions is a reason to weaken the Tier 1 gate. What the session leaves
for the agent to read:

| Artefact | What it is |
|---|---|
| `bench/<results>/gate.json` | the verdict: `GO` / `LLM_ONLY` / `STOP` / `INVALID`, with the medians behind it |
| `bench/<results>/pilot_summary.csv` | per-configuration statistics with bootstrap CIs |
| `bench/<results>/pilot_matrix.log` | raw `SAMPLE` / `SUMMARY_PILOT` records |
| `bench/<results>/tier0_chain_validation.json` | Tier 0.1 raw-pair/bootstrap plus semantics-3 epoch/checkpoint/final-digest/trace-artifact recomputation |
| `bench/<results>/tier0_background_validation.json` | Tier 0.3 formal matrix, paired statistics, resource metadata, and all retained trace checks |
| `bench/<results>/session.log` | the whole run, timestamped from session start |
| `bench/<results>/failures.log` | steps that failed; the session continues past them |

The gate thresholds live in `EXPERIMENT_PLAN.md` §6 and are implemented by `tools/gate.py`.
**Do not re-derive the verdict by eye** — if you disagree with it, the disagreement is with §6
and belongs there, not in a report.

`INVALID` (exit 2) means at least one configuration failed correctness or its required per-config
semantic trace proof. When that happens no timing from the run is usable, and the report says so
instead of quoting numbers. A missing manifest row, too few repeats, or incomplete plan-wide
coverage is a different state: the numeric verdict is provisional and cannot open Tier 2/3, but it
does not by itself turn otherwise valid configurations into an `INVALID` run.

Tier 0 has its own fail-closed boundary before that gate. The Tier 0.1 semantics-3 record stream
must carry the exact monotonic epoch schedule, independently recomputable checkpoint and final
digests, and a declared trace path whose row epochs bind the CSV to the final SAMPLE. Both Tier 0
strict JSON files above must be freshly generated. If either validator fails, `run_session.sh`
may finish the fail-soft Tier 1/collect path for diagnostics, but the session itself returns 2 even
when the Tier 1 `gate.json` says `GO`.

```bash
# Offline analysis chain (no GPU needed) — always validate on fixtures first
python3 tools/make_test_fixtures.py --out /tmp/ctafix
python3 tools/analyze_pilot.py /tmp/ctafix/pilot_matrix.log \
        --expected /tmp/ctafix/pilot_expected_tags.txt \
        --json /tmp/ctafix/pilot_analysis.json --csv /tmp/ctafix/pilot_summary.csv
python3 tools/gate.py          /tmp/ctafix/pilot_analysis.json
python3 tools/analyze.py       /tmp/ctafix/summary.txt
python3 tools/cta_timeline.py  /tmp/ctafix/trace.csv
python3 tools/llm_bracket.py   /tmp/ctafix/summary_llm.txt
python3 tools/dep_oracle.py --help      # pure-CPU dependency oracle
```

The two summary schemas are not interchangeable: `analyze.py` reads `SUMMARY` lines
(`tier0_facts`, and the rejected `cta_dep_bench`), `analyze_pilot.py` reads
`SAMPLE`/`SUMMARY_PILOT` (`cta_dep_pilot`, the Tier 1 gate). Each now refuses the other's
input with a message naming the right script rather than emitting a plausible-looking result.

## 4. Benchmark validity rules — non-negotiable

An entire earlier campaign was invalidated for semantic reasons, audited in
[`reports/rejected/fast_campaign.md`](reports/rejected/fast_campaign.md).
Read it before touching `bench/`. Every new or modified benchmark must satisfy all of the
following, and the report must state that it does:

1. **Trigger timing must not defeat the wait.** For software CTA modes, the producer triggers at
   *kernel entry*, then does readiness work, then release-stores its flag. If flags are published
   before the trigger, the dependent grid only becomes eligible after every flag is already set
   and the consumer's "wait" degenerates into a scan of ready flags.
2. **No global completion counter as a readiness proxy.** Cardinality does not imply identity:
   `completed_count >= hi+1` does not mean parents `[0,hi]` are done. This is not a conservative
   over-wait, it is incorrect.
3. **Timed post-wait payload must be O(1) in degree.** Full parent-set checking belongs in a
   separate untimed validation invocation, otherwise a degree sweep measures payload growth.
4. **Correctness evidence must be real.** Poison producer data and consumer output every repeat,
   check every real parent in validation, and make validation failure exit non-zero. A stale
   correct-looking value from the previous repeat must not be able to produce a PASS.
5. **Timing source.** `%globaltimer` for anything compared across SMs; `clock64()` only for
   single-SM durations. Getting this wrong yields plausible-looking but wrong overlap, silently.
6. **Degree and dependency structure are independent axes.** Never grow them together — that is
   precisely the flaw that makes BlockMaestro's "degree > 32 ⇒ no benefit" threshold
   uninterpretable, and it would wrongly exclude LLM FFN GEMM chains and DSA indexer→topk, which
   are high-degree but contiguous.
7. **Floor for real workloads is PDL ON**, i.e. the production configuration, not PDL off.
8. **`--wait none` (Ceiling) intentionally computes wrong results.** Report only its time, never
   its correctness.
9. **Keep wait paths symmetric** in thread participation and barrier structure, so a protocol
   comparison isolates the protocol.
10. **Random parent generation must yield unique ids**, so requested degree equals actual degree.
11. **Launch hygiene**: opt in via `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` above
    the default dynamic-smem ceiling, and never touch dynamic smem on the `smem=0` path.

## 5. Every experiment must declare its coordinates before any code is written

State these four things first; if (c) is empty, do not run it.

- **(a) Dimension and rows.** Which of A1–E1, and which specific option rows it discriminates.
- **(b) Bracket points produced.** Which of Floor / Impl / Ceiling / Ideal, per
  `EXPERIMENT_PLAN.md` §1. `[H-]`/`[H+]` options get Floor/Ceiling/Ideal only — an
  interval estimate is the intended output, not a precise value.
- **(c) The decision it changes.** Which design choice or gate flips depending on the outcome.
- **(d) GPU budget** in minutes, against the ~8 GPU-hour total.

**Always pre-screen with `Ceiling − Floor` first.** It costs almost nothing and can veto a
workload outright. Never spend rented time on a workload whose headroom is already known to be
negligible.

## 6. Priority and gates

Execution order, gate thresholds and minimum deliverable are all owned by
`EXPERIMENT_PLAN.md` (§6 thresholds, §12 budget and minimum deliverable), and §6 is executed
by `tools/gate.py`:

- Tier 0 (base facts) → Tier 1 (benefit map, **decision point**) → Tier 2/3 (mechanism
  comparison) and Tier 4 (LLM end-to-end) → Tier 5 (DSA chain).
- Tier 1 gate on typical `Ceiling − Floor`: **≥ 8%** run everything; **2–8%** skip Tier 2/3 and
  go straight to LLM confirmation; **< 2%** stop and re-evaluate the direction.
- Minimum deliverable, in priority order: Tier 1.1 degree × grid map; Tier 4
  `Ceiling − PDL_grid`; Tier 0.1 achievable overlap depth; Tier 0.3 occupancy curve.

**Admission snapshot (2026-08-05).** Live execution progress and resume instructions belong in
`EXPERIMENT_REPORT_INDEX.md` §0–§2, not here. The `tier1p` contract uses a programmatic CUDA
Graph edge for Floor and resource-reserved high/low-priority streams for Impl/Ceiling. A nominal
`P,C > SM` point counts as multi-wave only when `%globaltimer` proves that a consumer started
while producer CTAs were still unstarted, all producers ultimately completed, Floor entered
during producer tails, and Ceiling observed poisoned/stale output. Missing that per-config proof
is an `INVALID` admission failure, not a caveat that may be repaired in prose. Missing manifest,
repeat, or plan-wide coverage instead keeps the numeric verdict provisional. Tier 2/3 remains
downstream of the complete Tier 1.1 gate; Tier 4 remains a separate real-workload admission.

## 7. Report conventions

**Location.** Reports live under `reports/`, grouped by the tier that produced them.
`EXPERIMENT_REPORT_INDEX.md` at the subtree root is the execution-progress entry point
(plan section → status → resume commands → report links).

```
reports/campaign_<device>_<budget>.md   umbrella report for one rented session
reports/tier0_base_facts/<n>_<m>_<scope>.md
reports/tier1_benefit_map/<scope>.md
reports/offline/<scope>.md              experiments needing no GPU
reports/rejected/<scope>.md             audit records for invalidated campaigns
```

**Structure.** Required sections, following
[`reports/tier0_base_facts/0_5_fence_scope.md`](reports/tier0_base_facts/0_5_fence_scope.md) as
the model:

1. Header: report date, experiment date, device (name + SM count + compute capability), and an
   **evidence grade** stating what class of claim the data can support.
2. Executive summary carrying the actual numbers, not adjectives.
3. What the program *actually does* — the executed semantics, not the intent.
4. Configuration and statistics: warmup count, timed repeats, timing method, aggregation.
5. Recomputation of every headline number from raw values, with the arithmetic shown.
6. **Claims that hold.**
7. **Claims that do NOT hold.** Mandatory. This section is the main defence of the project's
   credibility; a report without it is incomplete.
8. Evidence entry points: links to source, raw logs, summary lines, and the umbrella report.

Link convention inside reports: the visible text is the path relative to the subtree root, while
the href is the actual relative path — so a Tier 0 report links to the benchmark source with the
text `bench/tier0_facts.cu` and the href `../../bench/tier0_facts.cu`.

Additional rules:

- Update `EXPERIMENT_REPORT_INDEX.md` in the same change — its §0 current status, §2 plan-section
  checklist, and §4 report list (so the next session can resume from the file alone).
- **Never delete a rejected experiment.** It gets a report under `reports/rejected/` as an audit
  record, listing what may still be reused and what must not be reused as a conclusion.
- Provenance artefacts of a past session (`EXPERIMENT_MANIFEST_SHA256.txt`,
  `EXPERIMENT_TRACKED_CHANGES.patch`, `EXPERIMENT_GIT_STATUS.txt`, result tarballs, and the raw
  logs under `bench/results*/`) are **historical snapshots — never edit them** to match later
  renames or moves.
- Do not soften or inflate wording. Calibrated claims are this project's product; a synthetic
  microbenchmark passing a gate is "mechanism feasible under stated limits", never "N% speedup".

## 8. Build and run

On the GPU box, prefer the whole session; drop to individual phases only when debugging one.

```bash
./run_session.sh                # the whole session, unattended (see §3)
FAST=1 ./run_session.sh         # same path end to end, minutes instead of hours
STEP_TIMEOUT=1800 ./run_session.sh   # bound each benchmark step; see below
```

To run the session *and* the reporting around it, use the Codex workflow, which wraps
`run_session.sh` with the agent stages that declare coordinates beforehand and write the
reports afterwards (`codex/README.md`):

```bash
./codex/run_campaign.sh         # audit -> smoke -> measure -> report -> branch -> tier4 -> wrapup
```

`run_session.sh` still ends at the Tier 1 decision point. Post-decision work uses its own
fail-closed runners and fresh or explicitly resumable result roots: `bench/llm/run_llm_sweep.sh`
for Tier 4, `bench/dsa/run_dsa_chain.sh` for the Tier 5 native strict bracket, and
`bench/dsa/run_production_tier5.sh` for the independent production-fragment characterization.
The agent reads each runner's terminal admission before writing a report; success in one entry
does not admit another. The user-authorized one-hour compact-14 scope, when selected, uses its own
fresh result root and `bench/dsa/validate_production_tier5_compact.py`; it must not reuse partial
exact-26 rows or relabel the canonical exact-26 campaign complete.

`STEP_TIMEOUT` (seconds, default `0` = off) bounds one benchmark invocation. Fail-soft
assumes a step terminates, and multi-wave software waits have no forward-progress guarantee:
a resident consumer CTA can spin on a producer CTA from a wave the scheduler has not placed
yet. Bounding the step turns that from a stalled rented machine into a recorded failure.

```bash
cd bench                        # individual phases, for debugging
./build.sh                      # default ARCH=sm_103 (B300); ARCH=sm_90 for H100
FAST=1 ./run_all.sh             # smoke, minutes — must print "campaign finished" with empty failures.log
./run_all.sh tier0              # phases: tier0 | tier1p | tier1 | tier23 | all
./run_all.sh --fresh tier1p     # ignore .done markers
```

**Harness status — a phase name here is a correctness claim.** `tier1p` drives `cta_dep_pilot`,
the corrected multi-wave harness, and is the only admissible source of Tier 1 gate data. Floor
uses a programmatic CUDA Graph edge; Impl/Ceiling use resource-reserved priority streams. Every
point must carry complete `%globaltimer` producer/consumer traces, and nominal `P,C > SM` points
satisfy §5.3 only with `multiwave_overlap_proven=1` plus complete producer progress, early Floor,
and intentionally wrong Ceiling evidence. Trace-incomplete attempts may use only the bounded,
explicitly logged retry contract in `bench/README.md`; they never become samples. CUDA sources
edited on the DEV BOX are unverified until they build on the GPU box. `tier1` and `tier23` drive
`cta_dep_bench`, whose trigger semantics are rejected (§4) — they are retained for re-audit,
print a warning, and are excluded from `all`. Their timings must never reach a conclusion.

Driver contract, required for the rented-machine model: unattended, resumable via
`results/<step>.done`, fail-soft (a failing step is recorded and the campaign continues), and
every raw number teed to `results/`. A `run_all.sh` step/pilot marker is valid only when its
schema binds the FAST flag, complete argv, executable SHA-256, and GPU UUID/name/compute
capability/driver fingerprint; a legacy or mismatched marker must rerun. `run_session.sh --fresh`
clears the formal result state at the Tier 0 entry and must not pass `--fresh` a second time to
Tier 1p, which would erase the Tier 0 state just produced. Preserve these properties in any
change.
`clc_probe` needs sm_100+ / CUDA ≥ 12.8 and is built best-effort so H100 still gets everything else.

## 9. Conventions

- **Language.** Prose docs and experiment reports are in **Chinese**, matching the existing
  corpus. Code, comments, script output, CLI help, and commit messages are in **English**. This
  file is in English by request.
- **Commits.** Conventional commits, English, e.g. `docs: ...`, `bench: ...`, `tools: ...`.
- **Do not commit** compiled binaries, tarballs, `bench/results*/`, or profiler output without
  asking — several are currently untracked on purpose.
- `docs/superpowers/{plans,specs}/` was deleted deliberately. Do not recreate it.

## 10. Known housekeeping debt

Fix opportunistically when touching a nearby file; do not do a sweeping reorganisation without
asking first.

- **Link hygiene.** Every relative markdown link under this subtree currently resolves. After any
  move or rename, re-verify — a silently broken cross-reference is how the report set loses its
  traceability. The two `bench/results*/campaign.log` files still contain pre-rename sibling
  paths (`跨stream_PDL调研/...`) **by design**: they are raw session evidence, not
  documentation. `codex/check_docs.py` enforces this check; run it after any move or rename.
- **The 2026-08-03 session's provenance artefacts were never kept in the tree.**
  `EXPERIMENT_MANIFEST_SHA256.txt`, `EXPERIMENT_TRACKED_CHANGES.patch`,
  `EXPERIMENT_GIT_STATUS.txt` and `cta_pdl_b200_budget1h_20260803.tar.gz` are named in the
  index and in two reports but are not present; as of 2026-08-05 those references are plain
  text rather than links, and say so. If the originals turn up, restore them next to
  `EXPERIMENT_REPORT_INDEX.md` and the links can come back. Later sessions should keep
  theirs — `collect.sh` produces the tarball.
- `cross_stream_PDL_survey/PDL_跨stream_总结.md` §7 contains a repo tree diagram whose
  `cta_level_PDL_design/` half is outdated (it predates `bench/`, `tools/`, and `reports/`), and
  it does not list the `CLC_feature_survey/` sibling. That file belongs to the sibling topic, so
  update it there rather than from here.
- `CLC_feature_survey/` currently holds only a `ref` directory — no summary document yet.
