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
# signed results/<step>.done markers, fail-soft (a failed step is recorded and the session
# continues), and every raw number on disk. Re-running after a crash or a dropped connection
# is safe; markers match only the same mode, argv, executable, GPU and driver.
#
# Usage:
#   ./run_session.sh                 # full session
#   ./run_session.sh --fresh         # ignore previous .done markers, start over
#   FAST=1 ./run_session.sh          # minutes, not hours: proves the plumbing, not the science
#   RESULTS=results_run2 ./run_session.sh
#   SMOKE_RESULTS=smoke_run2 ./run_session.sh  # isolate/retry the smoke artefacts
#   STEP_TIMEOUT=1800 ./run_session.sh   # passed through to run_all.sh; bounds each step so a
#                                        # multi-wave hang is recorded, not left stalling
#
# Exit codes:
#   0  session completed and the gate was evaluated (read gate.json for the verdict)
#   1  preflight failed, or the smoke test failed -- nothing was measured
#   2  the gate could not be evaluated, or a strict Tier 0 validator rejected the campaign

set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

FAST="${FAST:-0}"
RESULTS="${RESULTS:-results}"
SMOKE_RESULTS="${SMOKE_RESULTS:-smoke}"
FRESH=""
[ "${1:-}" = "--fresh" ] && FRESH="--fresh"

BENCH_RESULTS="${ROOT}/bench/${RESULTS}"
mkdir -p "${BENCH_RESULTS}"
SESSION_LOG="${BENCH_RESULTS}/session.log"
SESSION_INVALID=0

archive_session_artifact() {
    local path="$1" n=1
    [ -e "${path}" ] || return 0
    while [ -e "${path}.retry${n}" ]; do n=$((n + 1)); done
    mv "${path}" "${path}.retry${n}"
}

# Tier 0.2 lives in the sibling survey and therefore does not go through run_all.sh's
# signed-step helper.  Give it the same resumability contract here: a marker is reusable
# only for the same source/scripts, built executables, architecture, GPU and driver.
xstream_signature() {
    local xs_dir="$1" xs_arch="$2"
    local device_fingerprint source_hash bench_hash diamond_hash

    command -v nvidia-smi >/dev/null 2>&1 || return 2
    command -v sha256sum >/dev/null 2>&1 || return 2
    device_fingerprint=$(nvidia-smi \
        --query-gpu=uuid,name,compute_cap,driver_version \
        --format=csv,noheader,nounits 2>/dev/null | head -1) || return 2
    [ -n "${device_fingerprint}" ] || return 2
    for required in build.sh run.sh pdl_bench.cu pdl_diamond.cu; do
        [ -f "${xs_dir}/${required}" ] || return 2
    done
    [ -x "${xs_dir}/pdl_bench" ] || return 2
    [ -x "${xs_dir}/pdl_diamond" ] || return 2
    source_hash=$(
        cd "${xs_dir}" && \
        sha256sum build.sh run.sh pdl_bench.cu pdl_diamond.cu | \
        sha256sum | awk '{print $1}'
    ) || return 2
    bench_hash=$(sha256sum -- "${xs_dir}/pdl_bench" | awk '{print $1}') || return 2
    diamond_hash=$(sha256sum -- "${xs_dir}/pdl_diamond" | awk '{print $1}') || return 2
    [ -n "${source_hash}" ] && [ -n "${bench_hash}" ] && [ -n "${diamond_hash}" ] || return 2

    printf 'marker_schema=2\nkind=xstream\nfast=%s\ndevice=%q\nsource_sha256=%s\n' \
        "${FAST}" "${device_fingerprint}" "${source_hash}"
    printf 'pdl_bench_sha256=%s\npdl_diamond_sha256=%s\n' \
        "${bench_hash}" "${diamond_hash}"
    printf 'argv=(cd\ %q\ \&\&\ ARCH=%q\ ./build.sh\ \&\&\ ./run.sh)' \
        "${xs_dir}" "${xs_arch}"
}

if [ -n "${FRESH}" ]; then
    for artifact in tier0_chain_validation.json tier0_background_validation.json \
                    pilot_analysis.json pilot_summary.csv gate.json; do
        archive_session_artifact "${BENCH_RESULTS}/${artifact}"
    done
fi

START_EPOCH=$(date +%s)
say() {
    local now elapsed
    now=$(date +%s); elapsed=$((now - START_EPOCH))
    printf '[%02d:%02d:%02d] %s\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)) "$*" \
        | tee -a "${SESSION_LOG}"
}
rule() { say "────────────────────────────────────────────────────────────────"; }

say "session start $(date -Iseconds)  FAST=${FAST}  RESULTS=${RESULTS}  SMOKE_RESULTS=${SMOKE_RESULTS}"

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
if ! ( cd bench && FAST=1 RESULTS="${SMOKE_RESULTS}" ./run_all.sh ${FRESH} all ) \
        >> "${SESSION_LOG}" 2>&1; then
    say "SMOKE FAILED — the harness does not run on this machine. See ${SESSION_LOG}."
    exit 1
fi
if [ -s "${ROOT}/bench/${SMOKE_RESULTS}/failures.log" ]; then
    say "SMOKE recorded failures:"
    sed 's/^/    /' "${ROOT}/bench/${SMOKE_RESULTS}/failures.log" | tee -a "${SESSION_LOG}"
    say "Refusing to spend GPU hours on a harness that already fails."
    exit 1
fi
say "smoke ok"

# ---------------------------------------------------------------- 3. Tier 0
rule
say "3/6 Tier 0 base facts (~1h) — overlap depth, resident-wait capacity/throughput, CLC, fence scopes"
( cd bench && RESULTS="${RESULTS}" ./run_all.sh ${FRESH} tier0 ) >> "${SESSION_LOG}" 2>&1
CHAIN_VALIDATION="${BENCH_RESULTS}/tier0_chain_validation.json"
rm -f "${CHAIN_VALIDATION}"
if python3 tools/validate_tier0_chain.py "${BENCH_RESULTS}" \
        --json "${CHAIN_VALIDATION}" >> "${SESSION_LOG}" 2>&1; then
    say "Tier 0.1 strict validation ok -> ${CHAIN_VALIDATION}"
else
    say "Tier 0.1 strict validation FAILED -> ${CHAIN_VALIDATION}"
    printf '%s\n' "Tier 0.1 strict validation failed (see ${CHAIN_VALIDATION})" \
        >> "${BENCH_RESULTS}/failures.log"
    SESSION_INVALID=1
fi
BACKGROUND_VALIDATION="${BENCH_RESULTS}/tier0_background_validation.json"
rm -f "${BACKGROUND_VALIDATION}"
if python3 tools/validate_tier0_background.py "${BENCH_RESULTS}" \
        --json "${BACKGROUND_VALIDATION}" >> "${SESSION_LOG}" 2>&1; then
    say "Tier 0.3 strict validation ok -> ${BACKGROUND_VALIDATION}"
else
    say "Tier 0.3 strict validation FAILED -> ${BACKGROUND_VALIDATION}"
    printf '%s\n' "Tier 0.3 strict validation failed (see ${BACKGROUND_VALIDATION})" \
        >> "${BENCH_RESULTS}/failures.log"
    SESSION_INVALID=1
fi
say "Tier 0 done"

# ---------------------------------------------------------------- 4. Tier 0.2 cross-stream
# This used to be a manual step. The whole repo is cloned here, so the sibling benchmark is
# present and there is no reason for a person to run it by hand.
rule
say "4/6 Tier 0.2 cross-stream / CUDA Graph re-verification"
XS_DIR="${ROOT}/../cross_stream_PDL_survey/bench/pdl_bench"
XS_LOG="${BENCH_RESULTS}/tier0_xstream.log"
XS_MARKER="${BENCH_RESULTS}/tier0_xstream.done"
if [ -d "${XS_DIR}" ]; then
    XS_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d '. ')
    XS_ARCH="sm_${XS_CC:-100}"
    XS_SIGNATURE=""
    XS_SIGNATURE=$(xstream_signature "${XS_DIR}" "${XS_ARCH}") || XS_SIGNATURE=""
    if [ -n "${XS_SIGNATURE}" ] && [ -s "${XS_LOG}" ] && [ -f "${XS_MARKER}" ] && \
       [ "$(cat "${XS_MARKER}")" = "${XS_SIGNATURE}" ]; then
        say "skip cross-stream (matching source/binary/device signature)"
    else
        XS_PREPARED=1
        if [ -e "${XS_MARKER}" ]; then
            say "rerun cross-stream (stale or legacy .done signature)"
            if ! archive_session_artifact "${XS_MARKER}"; then
                say "cross-stream marker archive FAILED; refusing to overwrite it"
                printf '%s\n' "Tier 0.2 marker archive failed (${XS_MARKER})" \
                    >> "${BENCH_RESULTS}/failures.log"
                XS_PREPARED=0
            fi
        fi
        if [ "${XS_PREPARED}" -eq 1 ] && ! archive_session_artifact "${XS_LOG}"; then
            say "cross-stream log archive FAILED; refusing to overwrite it"
            printf '%s\n' "Tier 0.2 log archive failed (${XS_LOG})" \
                >> "${BENCH_RESULTS}/failures.log"
            XS_PREPARED=0
        fi
        if [ "${XS_PREPARED}" -eq 1 ] && \
           ( cd "${XS_DIR}" && ARCH="${XS_ARCH}" ./build.sh && ./run.sh ) \
                > "${XS_LOG}" 2>&1; then
            XS_SIGNATURE=""
            XS_SIGNATURE=$(xstream_signature "${XS_DIR}" "${XS_ARCH}") || XS_SIGNATURE=""
            XS_MARKER_TMP="${XS_MARKER}.tmp.$$"
            if [ -n "${XS_SIGNATURE}" ] && \
               printf '%s\n' "${XS_SIGNATURE}" > "${XS_MARKER_TMP}" && \
               mv "${XS_MARKER_TMP}" "${XS_MARKER}"; then
                say "cross-stream ok -> ${XS_LOG}"
                say "  (expect eager cross-stream ~1.00x and captured graph ~2.00x; a change there"
                say "   would itself be a finding about Blackwell programmatic events)"
            else
                rm -f "${XS_MARKER_TMP}"
                say "cross-stream ran, but signed marker creation FAILED"
                printf '%s\n' "Tier 0.2 signed marker creation failed (${XS_MARKER})" \
                    >> "${BENCH_RESULTS}/failures.log"
            fi
        elif [ "${XS_PREPARED}" -eq 1 ]; then
            say "cross-stream FAILED (recorded, session continues) -> ${XS_LOG}"
            printf '%s\n' "Tier 0.2 cross-stream failed (see ${XS_LOG})" \
                >> "${BENCH_RESULTS}/failures.log"
        fi
    fi
else
    say "skip: ${XS_DIR} not present — clone the whole repo, not just this subtree"
    printf '%s\n' "Tier 0.2 cross-stream missing (${XS_DIR})" \
        >> "${BENCH_RESULTS}/failures.log"
fi

# ---------------------------------------------------------------- 5. Tier 1p + gate
rule
say "5/6 Tier 1p benefit map (~2h) — the decision point, corrected cta_dep_pilot"
( cd bench && RESULTS="${RESULTS}" ./run_all.sh tier1p ) >> "${SESSION_LOG}" 2>&1

MATRIX="${BENCH_RESULTS}/pilot_matrix.log"
EXPECTED="${BENCH_RESULTS}/pilot_expected_tags.txt"
ANALYSIS="${BENCH_RESULTS}/pilot_analysis.json"
GATE_JSON="${BENCH_RESULTS}/gate.json"
rm -f "${ANALYSIS}" "${BENCH_RESULTS}/pilot_summary.csv" "${GATE_JSON}"

if [ ! -s "${MATRIX}" ]; then
    say "NO PILOT DATA — ${MATRIX} is empty. Check ${BENCH_RESULTS}/failures.log."
    exit 2
fi

if ! python3 tools/analyze_pilot.py "${MATRIX}" --expected "${EXPECTED}" \
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
RESULTS="${RESULTS}" ./collect.sh >> "${SESSION_LOG}" 2>&1 \
    || say "collect.sh reported a problem, see ${SESSION_LOG}"

ELAPSED=$(( $(date +%s) - START_EPOCH ))
rule
say "session finished in $((ELAPSED/3600))h$((ELAPSED%3600/60))m — verdict=${VERDICT}"
say ""
say "For the agent writing this up:"
say "  gate verdict + numbers : ${GATE_JSON}"
say "  per-config statistics  : ${BENCH_RESULTS}/pilot_summary.csv"
say "  raw samples            : ${MATRIX}"
say "  Tier 0 summary rows    : ${BENCH_RESULTS}/summary.txt"
say "  Tier 0 base/CLC raw    : ${BENCH_RESULTS}/tier0_facts.log, tier0_clc.log"
say "  Tier 0.1 validation    : ${BENCH_RESULTS}/tier0_chain_validation.json"
say "  Tier 0.3 validation    : ${BENCH_RESULTS}/tier0_background_validation.json"
say "  Tier 0 background raw  : ${BENCH_RESULTS}/tier0_bg_*_smem*.log, tier0_bg_*_smem*_trace.csv"
say "  full session log       : ${SESSION_LOG}"
say ""
say "Report conventions are in AGENTS.md section 7. The 'Claims that do NOT hold' section is"
say "mandatory; include synthetic-workload, priority-stream scheduling, and rejected-attempt limits."
if [ -s "${BENCH_RESULTS}/failures.log" ]; then
    say ""
    say "steps that failed (session continued past them):"
    sed 's/^/    /' "${BENCH_RESULTS}/failures.log" | tee -a "${SESSION_LOG}"
fi

if [ "${SESSION_INVALID}" -ne 0 ]; then
    say "strict Tier 0 validation failed; returning INVALID even though Tier 1 gate=${VERDICT}"
    exit 2
fi
exit "${GATE_RC}"
