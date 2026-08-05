#!/usr/bin/env bash
# run_all.sh — unattended driver for the whole screening campaign.
#
# The experiment machine is a RENTED single B300/B200, separate from the dev box, so this
# script is built to run start-to-finish with no interaction and leave every raw number on
# disk. Analysis happens back on the dev box (tools/).
#
# Features required by that constraint:
#   - resumable: every step writes results/<step>.done; re-running skips finished steps
#   - fail-soft: a failing step is recorded and the campaign continues
#   - everything (stdout + SUMMARY lines) is teed to results/
#
# Usage:
#   ./run_all.sh                 # full campaign (~8 GPU-hours budget)
#   ./run_all.sh tier0           # one phase only
#   FAST=1 ./run_all.sh          # smoke test: tiny sweeps, minutes not hours
#   ./run_all.sh --fresh         # ignore previous .done markers
#   STEP_TIMEOUT=1800 ./run_all.sh   # bound each step so a hang is a failure, not a stall
#
# Phases: tier0 | tier1p | tier1 | tier23 | all       ('all' = tier0 + tier1p)
#
# HARNESS STATUS -- read before choosing a phase:
#   tier1p  drives cta_dep_pilot, the CORRECTED harness. This is the phase whose output
#           may be used for the Tier 1 gate. Supports single-wave and multi-wave (P,C > SM).
#   tier1   drives cta_dep_bench, whose trigger semantics were REJECTED (it publishes every
#   tier23  done[] flag before the PDL trigger, so the waits are pre-satisfied). Its timing
#           numbers are inadmissible as CTA-benefit evidence -- see reports/rejected/
#           fast_campaign.md and AGENTS.md section 4. Kept runnable only for re-audit.

set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

RESULTS="${RESULTS:-results}"
FAST="${FAST:-0}"
mkdir -p "${RESULTS}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${SELF}"
    exit 0
fi
if [ "${1:-}" = "--fresh" ]; then rm -f "${RESULTS}"/*.done; shift; fi
PHASE="${1:-all}"
case "${PHASE}" in
    all|tier0|tier1p|tier1|tier23) ;;
    *) echo "unknown phase '${PHASE}'"
       echo "usage: $0 [--fresh] [all|tier0|tier1p|tier1|tier23]"; exit 1 ;;
esac

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/campaign.log"; }
fail() { echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "${RESULTS}/campaign.log" \
                                              | tee -a "${RESULTS}/failures.log"; }

# STEP_TIMEOUT — seconds per benchmark invocation; 0 (the default) keeps the old behaviour.
#
# Fail-soft assumes a step terminates. Once P,C > SM the software wait paths have no
# forward-progress guarantee: a consumer CTA can be resident and spinning on a producer CTA
# belonging to a wave the scheduler has not placed yet, and no wait loop here has a timeout.
# `strided` is the worst case, since child 0's parents span the whole producer grid. A hang
# stalls the rented machine indefinitely, which is precisely the outcome the fail-soft
# contract exists to prevent, so bound the step and let it be recorded as a failure instead.
STEP_TIMEOUT="${STEP_TIMEOUT:-0}"
run_bounded() {
    if [ "${STEP_TIMEOUT}" != "0" ] && command -v timeout >/dev/null 2>&1; then
        timeout --kill-after=30s "${STEP_TIMEOUT}" "$@"
    else
        "$@"
    fi
}

# step <name> <command...> — run once, record completion, never abort the campaign
#
# The SUMMARY grep is anchored with a trailing space on purpose: cta_dep_pilot emits
# SUMMARY_PILOT records with a different schema, and tools/analyze.py accepts any line
# beginning with "SUMMARY". Letting pilot rows into summary.txt would silently corrupt the
# gate report rather than fail, so pilot output is kept out via pilot_step below.
step() {
    local name="$1"; shift
    if [ -f "${RESULTS}/${name}.done" ]; then log "skip ${name} (already done)"; return 0; fi
    log "run  ${name}"
    if run_bounded "$@" >> "${RESULTS}/${name}.log" 2>&1; then
        grep -h '^SUMMARY ' "${RESULTS}/${name}.log" >> "${RESULTS}/summary.txt" 2>/dev/null || true
        touch "${RESULTS}/${name}.done"
        log "ok   ${name}"
    else
        local rc=$?
        [ ${rc} -eq 124 ] && fail "${name} TIMED OUT after ${STEP_TIMEOUT}s" \
                          || fail "${name} (see ${RESULTS}/${name}.log)"
    fi
}

# pilot_step <name> <command...> — as step(), but for cta_dep_pilot. Its SAMPLE and
# SUMMARY_PILOT records go to tools/analyze_pilot.py via pilot_matrix.log, never to
# summary.txt.
pilot_step() {
    local name="$1"; shift
    if [ -f "${RESULTS}/${name}.done" ]; then log "skip ${name} (already done)"; return 0; fi
    log "run  ${name}"
    if run_bounded "$@" >> "${RESULTS}/${name}.log" 2>&1; then
        touch "${RESULTS}/${name}.done"
        log "ok   ${name}"
    else
        local rc=$?
        [ ${rc} -eq 124 ] && fail "${name} TIMED OUT after ${STEP_TIMEOUT}s" \
                          || fail "${name} (see ${RESULTS}/${name}.log)"
    fi
}

rejected_harness_warning() {
    log "!! WARNING: this phase drives cta_dep_bench, whose trigger semantics are REJECTED."
    log "!! Its timings are NOT admissible as CTA-benefit evidence and MUST NOT feed the"
    log "!! Tier 1 gate. See reports/rejected/fast_campaign.md and AGENTS.md section 4."
    log "!! For gate data use: ./run_all.sh tier1p (corrected cta_dep_pilot)."
}

if [ ! -x ./cta_dep_pilot ] || [ ! -x ./cta_dep_bench ] || [ ! -x ./tier0_facts ]; then
    log "binaries missing, building"
    ./build.sh 2>&1 | tee -a "${RESULTS}/build.log" || { fail "build"; exit 1; }
fi

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv \
    > "${RESULTS}/device.txt" 2>/dev/null || true

# ------------------------------------------------------------------ sweep parameter sets
if [ "${FAST}" = "1" ]; then
    DEGREES=(1 8 64)
    GRIDS=(256 1024)
    STRUCTS=(interval grouped)
    REPEATS=5
else
    DEGREES=(1 2 4 8 16 32 64 128 256 512 1024)
    GRIDS=(64 128 256 512 1024 2048 4096 8192)
    STRUCTS=(interval grouped strided random self)
    REPEATS=30
fi

# cta_dep_pilot accepts P,C > SM (multi-wave). Grids are sized from the device SM
# count unless PILOT_GRIDS is set explicitly. Plan §5.3 requires underfilled, =SM,
# and 2x/8x/32x SM. random/all/none structures remain excluded by the pilot.
detect_sms() {
    if [ -n "${PILOT_SMS:-}" ]; then
        echo "${PILOT_SMS}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local n
        n=$(nvidia-smi --query-gpu=multiprocessor_count --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then
            echo "$n"
            return
        fi
    fi
    # Fallback matches the B200/B300 campaigns so far; override with PILOT_SMS on other SKUs.
    echo 148
}
PILOT_SMS_COUNT="$(detect_sms)"
if [ "${FAST}" = "1" ]; then
    PILOT_DEGREES=(1 8)
    PILOT_STRUCTS=(interval strided)
    PILOT_REPEATS=5
    # underfilled + full + one multi-wave point (plumbing proof, not the science matrix)
    PILOT_DEFAULT_GRIDS="64 ${PILOT_SMS_COUNT} $((2 * PILOT_SMS_COUNT))"
else
    PILOT_DEGREES=(1 2 4 8 16 32 64)
    PILOT_STRUCTS=(interval grouped strided self)
    PILOT_REPEATS=31
    PILOT_DEFAULT_GRIDS="32 64 128 ${PILOT_SMS_COUNT} $((2 * PILOT_SMS_COUNT)) $((8 * PILOT_SMS_COUNT)) $((32 * PILOT_SMS_COUNT))"
fi
read -ra PILOT_GRIDS <<< "${PILOT_GRIDS:-${PILOT_DEFAULT_GRIDS}}"
log "pilot SM count=${PILOT_SMS_COUNT}; grids=${PILOT_GRIDS[*]}"

# ------------------------------------------------------------------ Tier 0: base facts (~1h)
run_tier0() {
    log "=== Tier 0: base facts ==="
    step tier0_facts ./tier0_facts --repeats "${REPEATS}"
    if [ -x ./clc_probe ]; then
        step tier0_clc ./clc_probe --clusters 4096 --repeats 10
    else
        log "skip tier0_clc (binary not built; needs sm_100+)"
    fi
    log "NOTE: Tier 0.2 (PDL eager cross-stream behaviour) = re-run"
    log "      ../../cross_stream_PDL_survey/bench/pdl_bench on this device and diff against the H100 result"
}

# ------------------------------------------------------------------ Tier 1p: corrected map
#
# THE decisive experiment, on the corrected harness. Degree and structure are swept as
# INDEPENDENT axes: BlockMaestro grew both together, so its "degree > 32 => no benefit"
# threshold cannot separate "too many edges" from "too complex a shape". LLM FFN GEMM chains
# and DSA indexer->topk are both high degree but contiguous, and would be wrongly excluded.
#
# Coverage: single-wave (P,C <= SM) plus multi-wave (2x/8x/32x SM) per EXPERIMENT_PLAN.md
# §5.3. Trigger semantics stay publish-after-ready per CTA; see cta_dep_pilot.cu header.
run_tier1_pilot() {
    log "=== Tier 1.1p: degree axis, structure PINNED to interval, tail=0 (conservative) ==="
    for g in "${PILOT_GRIDS[@]}"; do
        for d in "${PILOT_DEGREES[@]}"; do
            [ "$d" -gt "$g" ] && continue
            pilot_step "t11p_g${g}_d${d}" ./cta_dep_pilot \
                --producers "$g" --consumers "$g" --structure interval --degree "$d" \
                --tail 0 --repeats "${PILOT_REPEATS}" --tag "t11p_g${g}_d${d}"
        done
    done

    log "=== Tier 1.1p: structure axis, degree PINNED to 32 ==="
    for g in "${PILOT_GRIDS[@]}"; do
        [ "$g" -lt 32 ] && continue
        for s in "${PILOT_STRUCTS[@]}"; do
            pilot_step "t11ps_g${g}_${s}" ./cta_dep_pilot \
                --producers "$g" --consumers "$g" --structure "$s" --degree 32 \
                --tail 0 --repeats "${PILOT_REPEATS}" --tag "t11ps_g${g}_${s}"
        done
    done

    log "=== Tier 1.2p: tail/prologue ratio sweep, degree PINNED to 8 ==="
    for ratio in 1 2 4 8 16; do
        local tail=$((200000 * ratio))
        pilot_step "t12p_r${ratio}" ./cta_dep_pilot \
            --producers "${PILOT_SMS_COUNT}" --consumers "${PILOT_SMS_COUNT}" \
            --structure interval --degree 8 \
            --tail "${tail}" --prologue 200000 \
            --repeats "${PILOT_REPEATS}" --tag "t12p_r${ratio}"
    done

    # analyze_pilot.py needs SAMPLE and SUMMARY_PILOT for the SAME tag set in one file, and
    # aborts if the two sets differ. Only completed steps are concatenated, so a fail-soft
    # step cannot leave a half-written tag behind and break the analysis back on the dev box.
    : > "${RESULTS}/pilot_matrix.log"
    for marker in "${RESULTS}"/t11p_*.done "${RESULTS}"/t11ps_*.done "${RESULTS}"/t12p_*.done; do
        [ -e "${marker}" ] || continue
        cat "${marker%.done}.log" >> "${RESULTS}/pilot_matrix.log"
    done
    log "pilot matrix -> ${RESULTS}/pilot_matrix.log"
    log "analyze with tools/analyze_pilot.py (NOT analyze.py -- different schema)"
    log "NOTE: grids=${PILOT_GRIDS[*]} (SM=${PILOT_SMS_COUNT}). Re-check gate caveat for multi-wave coverage."
}

# ------------------------------------------------------------------ Tier 1: REJECTED harness
#
# Retained for re-audit only. See rejected_harness_warning().
run_tier1() {
    rejected_harness_warning
    log "=== Tier 1.1a: degree sweep, structure PINNED to interval ==="
    for g in "${GRIDS[@]}"; do
        for d in "${DEGREES[@]}"; do
            [ "$d" -gt "$g" ] && continue
            step "t11a_g${g}_d${d}" ./cta_dep_bench \
                --producers "$g" --consumers "$g" --structure interval --degree "$d" \
                --all-waits --repeats "${REPEATS}" --tag "t11a_g${g}_d${d}"
        done
    done

    log "=== Tier 1.1b: structure sweep, degree PINNED to 32 ==="
    for g in "${GRIDS[@]}"; do
        for s in "${STRUCTS[@]}"; do
            step "t11b_g${g}_${s}" ./cta_dep_bench \
                --producers "$g" --consumers "$g" --structure "$s" --degree 32 \
                --all-waits --repeats "${REPEATS}" --tag "t11b_g${g}_${s}"
        done
    done

    log "=== Tier 1.2: tail/prologue ratio sweep ==="
    for ratio in 1 2 4 8 16; do
        local tail=$((200000 * ratio))
        step "t12_r${ratio}" ./cta_dep_bench \
            --producers 1024 --consumers 1024 --structure interval --degree 8 \
            --tail "${tail}" --prologue 200000 \
            --all-waits --repeats "${REPEATS}" --tag "t12_r${ratio}"
    done
}

# ------------------------------------------------------------------ Tier 2/3 (~3h)
run_tier23() {
    rejected_harness_warning
    log "=== Tier 2.1: sync protocol shootout (1-to-1, isolates protocol cost) ==="
    for g in 256 1024 4096; do
        step "t21_g${g}" ./cta_dep_bench \
            --producers "$g" --consumers "$g" --structure self --degree 1 \
            --all-waits --repeats "${REPEATS}" --tag "t21_g${g}"
    done

    log "=== Tier 2.3: encoding cost -- interval over-approximation vs exact parent set ==="
    for s in interval strided random; do
        for d in 4 16 64; do
            step "t23_${s}_d${d}" ./cta_dep_bench \
                --producers 2048 --consumers 2048 --structure "$s" --degree "$d" \
                --all-waits --repeats "${REPEATS}" --tag "t23_${s}_d${d}"
        done
    done

    log "=== Tier 0.3 extension: occupancy cost under a REAL dependency wait ==="
    for smem in 0 8 16 32 64; do
        step "t03_smem${smem}" ./cta_dep_bench \
            --producers 2048 --consumers 2048 --structure interval --degree 8 \
            --smem-kb "${smem}" --all-waits --repeats "${REPEATS}" --tag "t03_smem${smem}"
    done

    log "=== per-CTA timeline traces (primitive 1) for the headline configs ==="
    for s in interval random; do
        step "trace_${s}" ./cta_dep_bench \
            --producers 1024 --consumers 1024 --structure "$s" --degree 16 \
            --wait cta-backoff --repeats 5 \
            --trace "${RESULTS}/trace_${s}.csv" --tag "trace_${s}"
    done
}

# 'all' deliberately excludes tier1/tier23: they drive the rejected harness, and spending the
# 2h decision-point budget on inadmissible data is exactly the failure this driver exists to
# prevent. Ask for them by name if you are re-auditing.
case "${PHASE}" in
    tier0)  run_tier0 ;;
    tier1p) run_tier1_pilot ;;
    tier1)  run_tier1 ;;
    tier23) run_tier23 ;;
    all)    run_tier0; run_tier1_pilot ;;
esac

log "=== campaign finished ==="
log "SUMMARY lines: ${RESULTS}/summary.txt ($(wc -l < "${RESULTS}/summary.txt" 2>/dev/null || echo 0) rows)"
if [ -s "${RESULTS}/failures.log" ]; then
    log "FAILURES occurred:"; cat "${RESULTS}/failures.log"
fi
log "Copy ${RESULTS}/ back to the dev box and run tools/analyze.py"
