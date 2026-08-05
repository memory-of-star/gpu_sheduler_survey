#!/usr/bin/env bash
# run_session.sh — the entire rented-GPU session, unattended, one command.
#
# Intended use: clone this repo onto the rented GPU box, start an agent there, have it run
#   ./run_session.sh
# and then write reports from what lands in bench/results/. No human steps, no tmux, no
# decisions taken by a person mid-run.
#
# Scope is deliberately "up to the decision point": preflight, smoke, Tier 0, Tier 1p, and
# the Tier 1 gate. That is roughly 3 GPU-hours and needs no model download and no vLLM, which
# is what makes it able to finish unattended. Tier 4/5 are a separate, later session -- the
# gate verdict is what decides whether they are worth booking at all.
#
# Properties inherited from bench/run_all.sh and required here: resumable via
# results/<step>.done, fail-soft (a failed step is recorded and the session continues), and
# every raw number on disk. Re-running after a crash or a dropped connection is always safe.
#
# Usage:
#   ./run_session.sh                 # full session
#   ./run_session.sh --fresh         # ignore previous .done markers, start over
#   FAST=1 ./run_session.sh          # minutes, not hours: proves the plumbing, not the science
#   RESULTS=results_run2 ./run_session.sh
#   STEP_TIMEOUT=1800 ./run_session.sh   # passed through to run_all.sh; bounds each step so a
#                                        # multi-wave hang is recorded, not left stalling
#
# Exit codes:
#   0  session completed and the gate was evaluated (read gate.json for the verdict)
#   1  preflight failed, or the smoke test failed -- nothing was measured
#   2  the gate could not be evaluated (no valid Tier 1 data)

set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

FAST="${FAST:-0}"
RESULTS="${RESULTS:-results}"
FRESH=""
[ "${1:-}" = "--fresh" ] && FRESH="--fresh"

BENCH_RESULTS="${ROOT}/bench/${RESULTS}"
mkdir -p "${BENCH_RESULTS}"
SESSION_LOG="${BENCH_RESULTS}/session.log"

START_EPOCH=$(date +%s)
say() {
    local now elapsed
    now=$(date +%s); elapsed=$((now - START_EPOCH))
    printf '[%02d:%02d:%02d] %s\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)) "$*" \
        | tee -a "${SESSION_LOG}"
}
rule() { say "────────────────────────────────────────────────────────────────"; }

say "session start $(date -Iseconds)  FAST=${FAST}  RESULTS=${RESULTS}"

# ---------------------------------------------------------------- 1. preflight
rule
say "1/6 preflight"
if ! ./preflight.sh >> "${SESSION_LOG}" 2>&1; then
    say "PREFLIGHT FAILED — nothing measured. See ${SESSION_LOG}."
    say "Most common cause: nvcc not on PATH (export PATH=/usr/local/cuda/bin:\$PATH)."
    exit 1
fi
say "preflight ok"

# ---------------------------------------------------------------- 2. smoke
rule
say "2/6 smoke test (tiny sweeps, proves the plumbing before spending hours)"
# 'all' rather than 'tier0': tier1p is the phase the campaign actually depends on, and its
# FAST grid set includes a multi-wave point, so this is where a broken pilot shows up. A
# smoke test that skips the decisive harness proves the wrong thing.
if ! ( cd bench && FAST=1 RESULTS="smoke" ./run_all.sh ${FRESH} all ) >> "${SESSION_LOG}" 2>&1; then
    say "SMOKE FAILED — the harness does not run on this machine. See ${SESSION_LOG}."
    exit 1
fi
if [ -s "${ROOT}/bench/smoke/failures.log" ]; then
    say "SMOKE recorded failures:"
    sed 's/^/    /' "${ROOT}/bench/smoke/failures.log" | tee -a "${SESSION_LOG}"
    say "Refusing to spend GPU hours on a harness that already fails."
    exit 1
fi
say "smoke ok"

# ---------------------------------------------------------------- 3. Tier 0
rule
say "3/6 Tier 0 base facts (~1h) — overlap depth, occupancy, CLC, fence scopes"
( cd bench && RESULTS="${RESULTS}" ./run_all.sh ${FRESH} tier0 ) >> "${SESSION_LOG}" 2>&1
say "Tier 0 done"

# ---------------------------------------------------------------- 4. Tier 0.2 cross-stream
# This used to be a manual step. The whole repo is cloned here, so the sibling benchmark is
# present and there is no reason for a person to run it by hand.
rule
say "4/6 Tier 0.2 cross-stream / CUDA Graph re-verification"
XS_DIR="${ROOT}/../cross_stream_PDL_survey/bench/pdl_bench"
XS_LOG="${BENCH_RESULTS}/tier0_xstream.log"
if [ -d "${XS_DIR}" ]; then
    if [ -f "${BENCH_RESULTS}/tier0_xstream.done" ]; then
        say "skip (already done)"
    elif ( cd "${XS_DIR}" && ./build.sh && ./run.sh ) > "${XS_LOG}" 2>&1; then
        touch "${BENCH_RESULTS}/tier0_xstream.done"
        say "cross-stream ok -> ${XS_LOG}"
        say "  (expect eager cross-stream ~1.00x and captured graph ~2.00x; a change there"
        say "   would itself be a finding about Blackwell programmatic events)"
    else
        say "cross-stream FAILED (recorded, session continues) -> ${XS_LOG}"
    fi
else
    say "skip: ${XS_DIR} not present — clone the whole repo, not just this subtree"
fi

# ---------------------------------------------------------------- 5. Tier 1p + gate
rule
say "5/6 Tier 1p benefit map (~2h) — the decision point, corrected cta_dep_pilot"
( cd bench && RESULTS="${RESULTS}" ./run_all.sh ${FRESH} tier1p ) >> "${SESSION_LOG}" 2>&1

MATRIX="${BENCH_RESULTS}/pilot_matrix.log"
ANALYSIS="${BENCH_RESULTS}/pilot_analysis.json"
GATE_JSON="${BENCH_RESULTS}/gate.json"

if [ ! -s "${MATRIX}" ]; then
    say "NO PILOT DATA — ${MATRIX} is empty. Check ${BENCH_RESULTS}/failures.log."
    exit 2
fi

if ! python3 tools/analyze_pilot.py "${MATRIX}" \
        --json "${ANALYSIS}" --csv "${BENCH_RESULTS}/pilot_summary.csv" \
        >> "${SESSION_LOG}" 2>&1; then
    say "analyze_pilot.py failed — see ${SESSION_LOG}"
    exit 2
fi

rule
say "gate evaluation"
python3 tools/gate.py "${ANALYSIS}" --json "${GATE_JSON}" 2>&1 | tee -a "${SESSION_LOG}"
GATE_RC=${PIPESTATUS[0]}
VERDICT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['verdict'])" \
          "${GATE_JSON}" 2>/dev/null || echo UNKNOWN)

# ---------------------------------------------------------------- 6. collect
rule
say "6/6 collect"
./collect.sh >> "${SESSION_LOG}" 2>&1 || say "collect.sh reported a problem, see ${SESSION_LOG}"

ELAPSED=$(( $(date +%s) - START_EPOCH ))
rule
say "session finished in $((ELAPSED/3600))h$((ELAPSED%3600/60))m — verdict=${VERDICT}"
say ""
say "For the agent writing this up:"
say "  gate verdict + numbers : ${GATE_JSON}"
say "  per-config statistics  : ${BENCH_RESULTS}/pilot_summary.csv"
say "  raw samples            : ${MATRIX}"
say "  Tier 0 raw             : ${BENCH_RESULTS}/tier0_facts.log, tier0_clc.log"
say "  full session log       : ${SESSION_LOG}"
say ""
say "Report conventions are in AGENTS.md section 7. The 'Claims that do NOT hold' section is"
say "mandatory, and the single-wave limitation belongs in it."
if [ -s "${BENCH_RESULTS}/failures.log" ]; then
    say ""
    say "steps that failed (session continued past them):"
    sed 's/^/    /' "${BENCH_RESULTS}/failures.log" | tee -a "${SESSION_LOG}"
fi

exit "${GATE_RC}"
