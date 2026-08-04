# Corrected Producer–Consumer Pilot Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete, evidence-backed Markdown report for all executed corrected producer–consumer pilot configurations in `cta_level_PDL_design/`.

**Architecture:** Build one self-contained report from immutable CUDA source, raw timing logs, CSV statistics, and JSON family aggregates. Explain the shared harness once, analyze degree/structure/tail/grid axes separately, and include all 24 seed-level configurations while isolating the invalid FAST campaign.

**Tech Stack:** Markdown, CUDA C++ source as semantic evidence, shell/awk/rg/jq for read-only verification, Git for an isolated documentation commit.

---

### Task 1: Verify the evidence inventory

**Files:**
- Read: `cta_level_PDL_design/bench/cta_dep_pilot.cu`
- Read: `cta_level_PDL_design/bench/results_budget1h_corrected/pilot_matrix.log`
- Read: `cta_level_PDL_design/bench/results_budget1h_corrected/pilot_summary.csv`
- Read: `cta_level_PDL_design/bench/results_budget1h_corrected/pilot_analysis.json`
- Read: `cta_level_PDL_design/tools/analyze_pilot.py`

- [ ] **Step 1: Verify execution counts**

Run:

```bash
awk '/^SAMPLE /{samples++} /PASS$/{passes++} /n\/a$/{na++} /^Pilot semantics=2/{configs++} /^SUMMARY_PILOT/{summaries++} END{printf "samples=%d configs=%d summaries=%d safe_passes=%d unsafe_na=%d\n",samples,configs,summaries,passes,na; if(samples!=3720 || configs!=24 || summaries!=24 || passes!=96 || na!=24) exit 1}' cta_level_PDL_design/bench/results_budget1h_corrected/pilot_matrix.log
```

Expected: `samples=3720 configs=24 summaries=24 safe_passes=96 unsafe_na=24`.

- [ ] **Step 2: Verify the 24 CSV rows are valid**

Run:

```bash
awk -F, 'NR==1 {if($12!="valid") exit 2; next} {rows++; if($12!=1) invalid++} END{printf "rows=%d invalid=%d\n",rows,invalid; if(rows!=24 || invalid!=0) exit 1}' cta_level_PDL_design/bench/results_budget1h_corrected/pilot_summary.csv
```

Expected: `rows=24 invalid=0`.

- [ ] **Step 3: Confirm the eight family names and statistics**

Inspect `pilot_analysis.json` and confirm exactly these families: `interval_d1`, `interval_d8`, `interval_d32`, `interval_d64`, `strided_d32`, `tail0_d8`, `tail2m_d8`, and `grid64_d8`.

### Task 2: Write the formal report

**Files:**
- Create: `cta_level_PDL_design/EXPERIMENT_REPORT_CORRECTED_PRODUCER_CONSUMER_PILOT.md`

- [ ] **Step 1: Write scope, hierarchy, and evidence grade**

State: one B200 with 148 SM; one producer kernel followed by one consumer kernel; default grids contain 148 CTAs each, with 128 threads per CTA; the evidence is a one-wave low-resource synthetic mechanism result, not an LLM/DSA result.

Include this hierarchy:

```text
Timed invocation
├── Producer kernel launch: P CTAs × 128 threads
└── Consumer kernel launch: C CTAs × 128 threads
```

- [ ] **Step 2: Explain the Floor and Impl timelines**

Include:

```text
Floor / grid:
Producer: [ready + skew] [write] [trigger at ready] [tail]
Consumer:                         [prologue] [grid wait] [epilogue]

Impl / interval-backoff:
Producer: [entry trigger] [ready + skew] [write + release flag] [tail]
Consumer:                 [prologue] [acquire/backoff flags] [epilogue]
```

Explain that all producer CTAs must trigger or exit before the consumer grid becomes launch-eligible; the entry trigger creates early eligibility, and flags enforce parent readiness after launch.

- [ ] **Step 3: Document all five modes**

Use exactly these roles:

| Mode | Trigger | Consumer synchronization | Role | Correctness |
|---|---|---|---|---|
| `none` | entry | none | unsafe Ceiling | not asserted |
| `grid` | data ready | grid dependency wait | Floor | validated |
| `interval-spin` | entry | interval acquire polling | diagnostic | validated |
| `interval-backoff` | entry | interval acquire/backoff | predeclared Impl | validated |
| `exact-backoff` | entry | exact-parent acquire/backoff | encoding diagnostic | validated |

- [ ] **Step 4: Document fixed parameters, variable families, and statistics**

Fixed parameters: 128 threads/CTA, ready 400K cycles, prologue 200K, epilogue 1M, 8 skew bins, seeds 101/202/303, 3 warmups, 31 repeats. Variable families: interval d1/d8/d32/d64, strided d32, tail0 d8, tail2M d8, and grid64 d8. State the full count `8 × 3 × 5 × 31 = 3,720`.

- [ ] **Step 5: Explain release/acquire correctness**

State:

```text
producer: datum write → device-scope release store done[p]=1
consumer leader: device-scope acquire load until done[p]==1
consumer CTA: __syncthreads() before dependent work
```

Document buffer poisoning/reset and the separate validation invocation. Clarify that 96 safe modes pass and 24 `none` modes intentionally have no correctness assertion.

- [ ] **Step 6: Add the eight-family result table**

Use these values:

| Family | Floor ms | Ceiling ms | Impl ms | Speedup | Space % | Gain % | Captured % |
|---|---:|---:|---:|---:|---:|---:|---:|
| interval d1 | 1.409088 | 0.898368 | 0.900768 | 1.5645× | 36.243 | 36.082 | 99.555 |
| interval d8 | 1.409216 | 0.898368 | 0.902464 | 1.5616× | 36.250 | 35.961 | 99.204 |
| interval d32 | 1.409152 | 0.898368 | 0.908128 | 1.5517× | 36.251 | 35.555 | 98.040 |
| interval d64 | 1.409120 | 0.898016 | 0.916064 | 1.5382× | 36.277 | 34.989 | 96.469 |
| strided d32 | 1.409056 | 0.898176 | 0.936640 | 1.5043× | 36.253 | 33.523 | 92.470 |
| tail=0 d8 | 1.001568 | 0.619872 | 0.901792 | 1.1105× | 38.086 | 9.952 | 26.150 |
| tail=2M d8 | 1.918144 | 1.407936 | 1.408192 | 1.3620× | 26.586 | 26.578 | 99.994 |
| grid=64 d8 | 1.408544 | 0.897472 | 0.901952 | 1.5617× | 36.284 | 35.966 | 99.123 |

- [ ] **Step 7: Analyze controlled axes and limitations**

Degree: report Impl–Ceiling overhead growth from about 2.3 µs to 18.0 µs. Structure: report tightness 0.2264 and about 20.2 µs recovered by exact encoding. Tail: compare absolute and relative gaps at 0/1M/2M. Grid: restrict the conclusion to 64 versus 148 in underfilled/one-wave conditions.

Separate the invalid FAST campaign and explain: publication before trigger, invalid global counter inference, O(degree) timed payload, and zero exit status after correctness failure. Exclude claims about real LLM/DSA, multi-wave, occupancy 1–2, real resource competition, degree above 64, or a direct corrected-pilot CTA trace.

- [ ] **Step 8: Add the 24-row configuration appendix and evidence index**

Transcribe every CSV row with tag, seed, structure, degree, effective degree, tightness, P/C, tail, Floor, Ceiling, Impl, speedup, space, gain, and captured percentage. Add relative links to source, raw log, CSV, JSON, analyzer, and the B200 summary report.

### Task 3: Verify and commit the report

**Files:**
- Verify: `cta_level_PDL_design/EXPERIMENT_REPORT_CORRECTED_PRODUCER_CONSUMER_PILOT.md`

- [ ] **Step 1: Check sections and placeholders**

Run `rg -n '^#|^##'` on the report and verify all designed sections exist. Run `rg -n '^(TBD|TODO|FIXME|\?\?\?)($|:)'` and expect no matches.

- [ ] **Step 2: Check configuration completeness**

Compare appendix tags to CSV column 1. Expected: 24 report tags, 24 CSV tags, no missing or extra tag.

- [ ] **Step 3: Check links and archived data integrity**

Resolve every relative Markdown link and require every target to exist. Run `shasum -a 256 -c cta_level_PDL_design/EXPERIMENT_MANIFEST_SHA256.txt`; expected exit code 0 with every archived artifact reporting `OK`.

- [ ] **Step 4: Commit only the new report**

Run:

```bash
git add -- cta_level_PDL_design/EXPERIMENT_REPORT_CORRECTED_PRODUCER_CONSUMER_PILOT.md
git commit --only cta_level_PDL_design/EXPERIMENT_REPORT_CORRECTED_PRODUCER_CONSUMER_PILOT.md -m "docs: add corrected producer consumer experiment report"
```

Expected: one documentation commit containing only the formal report.

