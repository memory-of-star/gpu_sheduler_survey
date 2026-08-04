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
| `docs/cta_pdl_eval_plan.md` | Methodology: the four-point bracket, tiers, GPU budget, gate thresholds (§10.1), minimum deliverable (§10.2) | Change only with a stated reason; downstream reports cite it. `RUNBOOK.md` §3.3/§7 mirror §10.1/§10.2 so they are readable on the rented box, where `docs/` is not shipped — change the numbers here and update the mirror in the same edit. |
| `docs/cuda_13.4_pdl_clc_interfaces.md` | What today's ISA/runtime actually provides | Facts about existing interfaces only. |
| `rubin_design.md` (subtree root) | **Rubin's design description** — the public statement about its thread-block/tile-level triggering | Verbatim quoted source, and the project's only Rubin material. Do not paraphrase into it or extend it with inference; cite it for every Rubin claim. |
| `papers/README.md` | The reference corpus, indexed by dimension | Add new refs with the dimension they inform. |
| `reports/**/*.md` | Results and conclusions | The only place conclusions live. Grouped by tier — see §7. |
| `EXPERIMENT_REPORT_INDEX.md` | The experiment ledger, at the subtree root | The join table between dimensions and evidence: per dimension, what evidence exists, its grade, and the remaining gap. Must be updated in the same change as any new/changed report. |
| `RUNBOOK.md` | Rented-GPU session procedure | Optimised for "no debugging on the clock". |
| `bench/README.md` | Benchmark code contract | Implementation detail of the primitives. |

`archive/` holds side-branch and superseded material, kept only for reference. It is **not
authoritative**: do not cite it as a current source, as a design recommendation, or as evidence.
Existing links into it may stay, but new work should not build on it without first promoting the
content back out of `archive/` deliberately.

## 3. Machine model — this is a hard constraint

**The dev box is macOS with no GPU and no CUDA toolchain** (`nvcc` and `nvidia-smi` are absent).
All GPU data is produced in a *separate* rented single-GPU session (so far: one B200, 148 SM,
CC 10.0) and copied back.

Consequences that agents must respect:

- **Never present a GPU number that is not traceable to a file under `bench/results*/`.**
  Anything else is `NOT EXECUTED` and must be labelled as such.
- CUDA sources can be written and reviewed locally but **cannot be compile-checked**. Treat any
  `.cu` edit as unverified until it builds on the rented machine; say so.
- Do not conflate devices. The design-space doc reasons about B300 (sm_103); the executed
  experiments so far are **B200 (sm_100)**. Reports must name the device they actually ran on.
- What *is* fully doable locally: docs, report writing from existing logs, offline dependency
  analysis, and the whole `tools/` chain (verified working under Python 3.14).

```bash
# Offline analysis chain (no GPU needed) — always validate on fixtures first
python3 tools/make_test_fixtures.py --out /tmp/ctafix
python3 tools/analyze.py       /tmp/ctafix/summary.txt
python3 tools/cta_timeline.py  /tmp/ctafix/trace.csv
python3 tools/llm_bracket.py   /tmp/ctafix/summary_llm.txt
python3 tools/dep_oracle.py --help      # pure-CPU dependency oracle
```

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
  `docs/cta_pdl_eval_plan.md` §1. `[H-]`/`[H+]` options get Floor/Ceiling/Ideal only — an
  interval estimate is the intended output, not a precise value.
- **(c) The decision it changes.** Which design choice or gate flips depending on the outcome.
- **(d) GPU budget** in minutes, against the ~8 GPU-hour total.

**Always pre-screen with `Ceiling − Floor` first.** It costs almost nothing and can veto a
workload outright. Never spend rented time on a workload whose headroom is already known to be
negligible.

## 6. Priority and gates

Execution order, gate thresholds and minimum deliverable are all owned by
`docs/cta_pdl_eval_plan.md` §10 (§10.1 thresholds, §10.2 minimum deliverable); `RUNBOOK.md`
§3.3/§7 carry a mirror for use on the rented box:

- Tier 0 (base facts) → Tier 1 (benefit map, **decision point**) → Tier 2/3 (mechanism
  comparison) and Tier 4 (LLM end-to-end) → Tier 5 (DSA chain).
- Tier 1 gate on typical `Ceiling − Floor`: **≥ 8%** run everything; **2–8%** skip Tier 2/3 and
  go straight to LLM confirmation; **< 2%** stop and re-evaluate the direction.
- Minimum deliverable, in priority order: Tier 1.1 degree × grid map; Tier 4
  `Ceiling − PDL_grid`; Tier 0.1 achievable overlap depth; Tier 0.3 occupancy curve.

**Current top gaps (snapshot 2026-08-04 — update when this changes).** Valid evidence today is
Tier 0.1/0.2/0.3/0.4/0.5, the corrected producer–consumer pilot, the CPU dependency oracle, and
the 13.3-vs-13.4 PTX diff. The two decisive things still missing are (1) **Tier 1.1 in the
`P,C > SM` multi-wave regime with the corrected harness** — the pilot only covers the
underfilled `P,C ≤ 148` case, so no grid-size or degree boundary has actually been measured; and
(2) **Tier 4's `Ceiling − PDL_grid`** on a real model. Do not let mechanism-level work (Tier 2/3)
jump ahead of these two.

## 7. Report conventions

**Location.** Reports live under `reports/`, grouped by the tier that produced them. The index
stays at the subtree root as the single entry point.

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

- Update `EXPERIMENT_REPORT_INDEX.md` in the same change — all three of its §1 coverage matrix,
  §2 report list, and §4 "explicitly not executed" list.
- **Never delete a rejected experiment.** It gets a report under `reports/rejected/` as an audit
  record, listing what may still be reused and what must not be reused as a conclusion.
- Provenance artefacts of a past session (`EXPERIMENT_MANIFEST_SHA256.txt`,
  `EXPERIMENT_TRACKED_CHANGES.patch`, `EXPERIMENT_GIT_STATUS.txt`, result tarballs, and the raw
  logs under `bench/results*/`) are **historical snapshots — never edit them** to match later
  renames or moves.
- Do not soften or inflate wording. Calibrated claims are this project's product; a synthetic
  microbenchmark passing a gate is "mechanism feasible under stated limits", never "N% speedup".

## 8. Build and run

```bash
cd bench
./build.sh                      # default ARCH=sm_103 (B300); ARCH=sm_90 for H100
FAST=1 ./run_all.sh             # smoke, minutes — must print "campaign finished" with empty failures.log
./run_all.sh tier0              # phases: tier0 | tier1p | tier1 | tier23 | all
./run_all.sh --fresh tier1p     # ignore .done markers
```

**Harness status — a phase name here is a correctness claim.** `tier1p` drives `cta_dep_pilot`,
the corrected harness, and is the only admissible source of Tier 1 gate data; it is capped at
`P,C <= SM` by the pilot itself, so it covers the single-wave regime only. `tier1` and `tier23`
drive `cta_dep_bench`, whose trigger semantics are rejected (§4) — they are retained for
re-audit, print a warning, and are excluded from `all`. Their timings must never reach a
conclusion. Consequently the project's top gap, the multi-wave map, is **producible by neither
binary today**: lifting the pilot's cap is a semantic change to the `.cu`, not a flag.

Driver contract, required for the rented-machine model: unattended, resumable via
`results/<step>.done`, fail-soft (a failing step is recorded and the campaign continues), and
every raw number teed to `results/`. Preserve these properties in any change.
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
  traceability. The two `bench/results*/campaign.log` files and `EXPERIMENT_MANIFEST_SHA256.txt`
  still contain pre-rename sibling paths (`跨stream_PDL调研/...`) **by design**: they are raw
  session evidence, not documentation.
- `cross_stream_PDL_survey/PDL_跨stream_总结.md` §7 contains a repo tree diagram whose
  `cta_level_PDL_design/` half is outdated (it predates `bench/`, `tools/`, and `reports/`), and
  it does not list the `CLC_feature_survey/` sibling. That file belongs to the sibling topic, so
  update it there rather than from here.
- `CLC_feature_survey/` currently holds only a `ref` directory — no summary document yet.
