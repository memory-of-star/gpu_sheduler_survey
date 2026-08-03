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
    all|tier0|tier1|tier23) ;;
    *) echo "unknown phase '${PHASE}'"; echo "usage: $0 [--fresh] [all|tier0|tier1|tier23]"; exit 1 ;;
esac

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/campaign.log"; }
fail() { echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "${RESULTS}/campaign.log" \
                                              | tee -a "${RESULTS}/failures.log"; }

# step <name> <command...> — run once, record completion, never abort the campaign
step() {
    local name="$1"; shift
    if [ -f "${RESULTS}/${name}.done" ]; then log "skip ${name} (already done)"; return 0; fi
    log "run  ${name}"
    if "$@" >> "${RESULTS}/${name}.log" 2>&1; then
        grep -h '^SUMMARY' "${RESULTS}/${name}.log" >> "${RESULTS}/summary.txt" 2>/dev/null || true
        touch "${RESULTS}/${name}.done"
        log "ok   ${name}"
    else
        fail "${name} (see ${RESULTS}/${name}.log)"
    fi
}

if [ ! -x ./cta_dep_bench ] || [ ! -x ./tier0_facts ]; then
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
    log "      ../../跨stream_PDL调研/bench/pdl_bench on this device and diff against the H100 result"
}

# ------------------------------------------------------------------ Tier 1: benefit map (~2h)
#
# THE decisive experiment. Degree and structure are swept as INDEPENDENT axes: BlockMaestro
# grew both together, so its "degree > 32 => no benefit" threshold cannot separate "too many
# edges" from "too complex a shape". LLM FFN GEMM chains and DSA indexer->topk are both high
# degree but contiguous, and would be wrongly excluded by that threshold.
run_tier1() {
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

case "${PHASE}" in
    tier0)  run_tier0 ;;
    tier1)  run_tier1 ;;
    tier23) run_tier23 ;;
    all)    run_tier0; run_tier1; run_tier23 ;;
esac

log "=== campaign finished ==="
log "SUMMARY lines: ${RESULTS}/summary.txt ($(wc -l < "${RESULTS}/summary.txt" 2>/dev/null || echo 0) rows)"
if [ -s "${RESULTS}/failures.log" ]; then
    log "FAILURES occurred:"; cat "${RESULTS}/failures.log"
fi
log "Copy ${RESULTS}/ back to the dev box and run tools/analyze.py"
