#!/usr/bin/env bash
# run_all.sh — unattended driver for the whole screening campaign.
#
# The experiment machine is a RENTED single B300/B200, separate from the dev box, so this
# script is built to run start-to-finish with no interaction and leave every raw number on
# disk. Analysis happens back on the dev box (tools/).
#
# Features required by that constraint:
#   - resumable: every step writes a command/binary signature to results/<step>.done;
#                re-running skips only an identical finished step
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
# Phases: tier0 | tier1p | tier1 | tier23 | tier23n | all
#         ('all' = tier0 + tier1p; tier23n is separately admitted after gate=GO)
#
# HARNESS STATUS -- read before choosing a phase:
#   tier1p  drives cta_dep_pilot, the CORRECTED harness. This is the phase whose output
#           may be used for the Tier 1 gate. Covers underfilled/=SM grid ratios and
#           trace-proven multi-wave points (P,C > SM); grid ratio alone never proves waves.
#   tier1   drives cta_dep_bench, whose trigger semantics were REJECTED (it publishes every
#   tier23  done[] flag before the PDL trigger, so the waits are pre-satisfied). Its timing
#           numbers are inadmissible as CTA-benefit evidence -- see reports/rejected/
#           fast_campaign.md and AGENTS.md section 4. Kept runnable only for re-audit.
#   tier23n drives run_tier23_native.sh, the admissible §7.1/§7.3-§7.6 matrix. It is an
#           independent post-GO campaign and never aliases the rejected tier23 phase.

set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

RESULTS="${RESULTS:-results}"
FAST="${FAST:-0}"
mkdir -p "${RESULTS}"

if command -v nvidia-smi >/dev/null 2>&1; then
    DEVICE_FINGERPRINT=$(nvidia-smi \
        --query-gpu=uuid,name,compute_cap,driver_version \
        --format=csv,noheader,nounits 2>/dev/null | head -1)
fi
DEVICE_FINGERPRINT="${DEVICE_FINGERPRINT:-unavailable}"

archive_result_file() {
    local path="$1" n=1
    [ -e "${path}" ] || return 0
    while [ -e "${path}.retry${n}" ]; do n=$((n + 1)); done
    mv "${path}" "${path}.retry${n}"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${SELF}"
    exit 0
fi
if [ "${1:-}" = "--fresh" ]; then
    for artifact in summary.txt failures.log pilot_matrix.log pilot_expected_tags.txt; do
        archive_result_file "${RESULTS}/${artifact}"
    done
    rm -f "${RESULTS}"/*.done "${RESULTS}"/*.invalid
    : > "${RESULTS}/summary.txt"
    shift
fi
PHASE="${1:-all}"
case "${PHASE}" in
    all|tier0|tier1p|tier1|tier23|tier23n) ;;
    *) echo "unknown phase '${PHASE}'"
       echo "usage: $0 [--fresh] [all|tier0|tier1p|tier1|tier23|tier23n]"; exit 1 ;;
esac

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/campaign.log"; }
fail() { echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "${RESULTS}/campaign.log" \
                                              | tee -a "${RESULTS}/failures.log"; }

# Preserve an incomplete prior attempt before retrying it. Appending a second run to the
# same file can create duplicate SUMMARY records and make a resumed campaign unanalyzable.
prepare_attempt_log() {
    local path="$1" n=1
    [ -s "${path}" ] || return 0
    while [ -e "${path}.retry${n}" ]; do n=$((n + 1)); done
    mv "${path}" "${path}.retry${n}"
}

# A bare marker is unsafe: FAST and formal runs reuse many tag names, and a rebuilt binary can
# change semantics without changing those names.  Bind every marker to the mode, full argv and
# executable content.  Historical empty markers intentionally mismatch and are rerun.
step_signature() {
    local kind="$1" command binary_hash="missing"
    shift
    command="${1:-}"
    if [ -f "${command}" ] && command -v sha256sum >/dev/null 2>&1; then
        binary_hash=$(sha256sum -- "${command}" | awk '{print $1}')
    fi
    printf 'marker_schema=2\nkind=%s\nfast=%s\ndevice=%q\nbinary_sha256=%s\nargv=' \
        "${kind}" "${FAST}" "${DEVICE_FINGERPRINT}" "${binary_hash}"
    printf '%q ' "$@"
}

marker_matches() {
    local path="$1" expected="$2"
    [ -f "${path}" ] && [ "$(cat "${path}")" = "${expected}" ]
}

write_marker() {
    printf '%s\n' "$2" > "$1"
}

# summary.txt is derived state.  Rebuild it from currently admitted step logs whenever a
# signature changes, so a FAST row cannot survive beside a rerun formal row.
rebuild_summary() {
    local marker name path
    : > "${RESULTS}/summary.txt"
    for marker in "${RESULTS}"/*.done; do
        [ -e "${marker}" ] || continue
        name="${marker##*/}"
        name="${name%.done}"
        path="${RESULTS}/${name}.log"
        [ -f "${path}" ] || continue
        grep -h '^SUMMARY ' "${path}" >> "${RESULTS}/summary.txt" 2>/dev/null || true
    done
}

# STEP_TIMEOUT — seconds per benchmark invocation; 0 (the default) keeps the old behaviour.
#
# cta_dep_pilot now reserves a checked producer resource slot and gives producers stream
# priority, but CUDA still makes no architecture-level fairness promise across kernels.
# `strided` is the worst case, since child 0's parents span the whole producer grid. Keep the
# outer process timeout in addition to the benchmark's event watchdog: a scheduler/driver
# regression must become a recorded failure rather than stall the rented machine indefinitely.
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
    local name="$1" marker signature; shift
    marker="${RESULTS}/${name}.done"
    signature="$(step_signature step "$@")"
    if marker_matches "${marker}" "${signature}"; then
        log "skip ${name} (matching command/binary signature)"
        return 0
    fi
    if [ -e "${marker}" ]; then
        log "rerun ${name} (stale or legacy .done signature)"
        rm -f "${marker}"
        rebuild_summary
    fi
    log "run  ${name}"
    prepare_attempt_log "${RESULTS}/${name}.log"
    if run_bounded "$@" >> "${RESULTS}/${name}.log" 2>&1; then
        write_marker "${marker}" "${signature}"
        rebuild_summary
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
    local name="$1" done_marker invalid_marker signature; shift
    done_marker="${RESULTS}/${name}.done"
    invalid_marker="${RESULTS}/${name}.invalid"
    signature="$(step_signature pilot "$@")"
    if marker_matches "${done_marker}" "${signature}"; then
        log "skip ${name} (matching command/binary signature)"
        return 0
    fi
    if [ -e "${done_marker}" ]; then
        log "rerun ${name} (stale or legacy .done signature)"
        rm -f "${done_marker}"
    fi
    if marker_matches "${invalid_marker}" "${signature}"; then
        log "skip ${name} (completed with an admissibility failure; gate remains INVALID)"
        return 0
    fi
    if [ -e "${invalid_marker}" ]; then
        log "rerun ${name} (stale or legacy .invalid signature)"
        rm -f "${invalid_marker}"
    fi
    log "run  ${name}"
    prepare_attempt_log "${RESULTS}/${name}.log"
    if run_bounded "$@" >> "${RESULTS}/${name}.log" 2>&1; then
        write_marker "${done_marker}" "${signature}"
        log "ok   ${name}"
    else
        local rc=$?
        # cta_dep_pilot intentionally exits non-zero after emitting a complete
        # SUMMARY_PILOT valid=0 record. Keep that record in the matrix so gate.py
        # returns INVALID; silently dropping it could turn an admissibility failure
        # into a false GO over the surviving configurations.
        if grep -q '^SUMMARY_PILOT .* valid=0$' "${RESULTS}/${name}.log"; then
            write_marker "${invalid_marker}" "${signature}"
            fail "${name} completed with an admissibility failure (kept for INVALID gate)"
        elif [ ${rc} -eq 124 ]; then
            fail "${name} TIMED OUT after ${STEP_TIMEOUT}s"
        else
            fail "${name} (see ${RESULTS}/${name}.log)"
        fi
    fi
}

rejected_harness_warning() {
    log "!! WARNING: this phase drives cta_dep_bench, whose trigger semantics are REJECTED."
    log "!! Its timings are NOT admissible as CTA-benefit evidence and MUST NOT feed the"
    log "!! Tier 1 gate. See reports/rejected/fast_campaign.md and AGENTS.md section 4."
    log "!! For gate data use: ./run_all.sh tier1p (corrected cta_dep_pilot)."
}

if [ ! -x ./cta_dep_pilot ] || [ ! -x ./cta_dep_bench ] || \
   [ ! -x ./tier0_facts ] || [ ! -x ./tier0_background ]; then
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
    CLC_REPEATS=10
    TIER0_FACTS_SHORT_ARG=(--allow-short)
else
    DEGREES=(1 2 4 8 16 32 64 128 256 512 1024)
    GRIDS=(64 128 256 512 1024 2048 4096 8192)
    STRUCTS=(interval grouped strided random self)
    REPEATS=31
    CLC_REPEATS=31
    TIER0_FACTS_SHORT_ARG=()
fi

# cta_dep_pilot accepts P,C > SM (multi-wave). Grids are sized from the device SM
# count unless PILOT_GRIDS is set explicitly. Plan §5.3 requires underfilled, =SM,
# and 2x/8x/32x SM. all/none structures remain excluded by the pilot.
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
    PILOT_DEGREES=(1 8 64)
    # Smoke every accepted structure so random uniqueness and the self degree
    # exception fail before the formal matrix spends GPU time.
    PILOT_STRUCTS=(interval grouped strided random self)
    PILOT_REPEATS=5
    # underfilled + full + one multi-wave point (plumbing proof, not the science matrix)
    PILOT_DEFAULT_GRIDS="64 ${PILOT_SMS_COUNT} $((2 * PILOT_SMS_COUNT))"
else
    PILOT_DEGREES=(1 2 4 8 16 32 64 128 256 512 1024)
    PILOT_STRUCTS=(interval grouped strided random self)
    PILOT_REPEATS=31
    PILOT_DEFAULT_GRIDS="32 64 128 ${PILOT_SMS_COUNT} $((2 * PILOT_SMS_COUNT)) $((8 * PILOT_SMS_COUNT)) $((32 * PILOT_SMS_COUNT))"
fi
read -ra PILOT_GRIDS <<< "${PILOT_GRIDS:-${PILOT_DEFAULT_GRIDS}}"
log "pilot SM count=${PILOT_SMS_COUNT}; grids=${PILOT_GRIDS[*]}"

# Tier 0.3 productive-background matrix. The formal path covers every planned smem point
# and three compile-time register tiers with >=31 paired repeats. FAST only proves the two
# resource extremes and must pass --allow-short explicitly so a short run cannot be mistaken
# for the formal experiment.
if [ "${FAST}" = "1" ]; then
    BG_SMEM_KB=(0 64)
    BG_REG_TIERS=(low high)
    BG_REPEATS=3
    BG_WARMUP=1
    BG_WAVES=2
    BG_ITERS=50000
    # Keep a long, auditable wait interval at the 64 KiB/high-register extreme;
    # the harness uses a single-CTA dependency holder so this interval is not
    # reduced to nondeterministic full-grid retirement skew.
    BG_PRODUCER_CYCLES=5000000
    BG_SHORT_ARG=(--allow-short)
else
    BG_SMEM_KB=(0 8 16 32 64)
    BG_REG_TIERS=(low mid high)
    BG_REPEATS=31
    BG_WARMUP=3
    BG_WAVES=8
    BG_ITERS=1000000
    BG_PRODUCER_CYCLES=4000000
    BG_SHORT_ARG=()
fi

# ------------------------------------------------------------------ Tier 0: base facts (~1h)
run_tier0() {
    log "=== Tier 0: base facts ==="
    step tier0_facts ./tier0_facts --repeats "${REPEATS}" \
        --warmup 3 --trace "${RESULTS}/tier0_chain_trace.csv" \
        "${TIER0_FACTS_SHORT_ARG[@]}"
    log "=== Tier 0.3: deferred gate vs resident griddepcontrol wait ==="
    for reg in "${BG_REG_TIERS[@]}"; do
        for smem in "${BG_SMEM_KB[@]}"; do
            local tag="tier0_bg_${reg}_smem${smem}"
            step "${tag}" ./tier0_background \
                --smem-kb "${smem}" --reg-tier "${reg}" \
                --repeats "${BG_REPEATS}" --warmup "${BG_WARMUP}" \
                --bg-waves "${BG_WAVES}" --bg-iters "${BG_ITERS}" \
                --producer-cycles "${BG_PRODUCER_CYCLES}" \
                --tag "${tag}" --trace "${RESULTS}/${tag}_trace.csv" \
                "${BG_SHORT_ARG[@]}"
        done
    done
    if [ -x ./clc_probe ]; then
        step tier0_clc ./clc_probe --clusters 4096 --repeats "${CLC_REPEATS}"
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
# Coverage: underfilled and =SM grid-ratio points plus trace-proven 2x/8x/32x SM
# multi-wave points per EXPERIMENT_PLAN.md.  P,C <= SM is not a residency/wave proof.
# §5.3. Floor stays publish-after-ready grid PDL; no-edge modes must pass the per-sample
# %globaltimer/progress proof emitted by cta_dep_pilot.cu before multi-wave coverage counts.
run_tier1_pilot() {
    local expected="${RESULTS}/pilot_expected_tags.txt"
    : > "${expected}"
    log "=== Tier 1.1p: degree axis, structure PINNED to interval, tail=1000000 ==="
    for g in "${PILOT_GRIDS[@]}"; do
        for d in "${PILOT_DEGREES[@]}"; do
            [ "$d" -gt "$g" ] && continue
            local tag="t11p_g${g}_d${d}"
            echo "${tag}" >> "${expected}"
            pilot_step "${tag}" ./cta_dep_pilot \
                --producers "$g" --consumers "$g" --structure interval --degree "$d" \
                --tail 1000000 --repeats "${PILOT_REPEATS}" --tag "${tag}"
        done
    done

    log "=== Tier 1.1p: structure axis, degree PINNED to 32 (self is semantic d=1 control) ==="
    for g in "${PILOT_GRIDS[@]}"; do
        [ "$g" -lt 32 ] && continue
        for s in "${PILOT_STRUCTS[@]}"; do
            # The degree axis already ran this exact physical coordinate as
            # t11p_g${g}_d32. Reuse it for the interval structure point instead
            # of measuring and double-weighting the same coordinate in the gate.
            [ "${s}" = "interval" ] && continue
            local tag="t11ps_g${g}_${s}"
            echo "${tag}" >> "${expected}"
            pilot_step "${tag}" ./cta_dep_pilot \
                --producers "$g" --consumers "$g" --structure "$s" --degree 32 \
                --tail 1000000 --repeats "${PILOT_REPEATS}" --tag "${tag}"
        done
    done

    log "=== Tier 1.2p: tail/prologue ratio sweep, degree PINNED to 8 ==="
    for ratio in 1 2 4 8 16; do
        local tail=$((200000 * ratio))
        local tag="t12p_r${ratio}"
        echo "${tag}" >> "${expected}"
        pilot_step "${tag}" ./cta_dep_pilot \
            --producers "${PILOT_SMS_COUNT}" --consumers "${PILOT_SMS_COUNT}" \
            --structure interval --degree 8 \
            --tail "${tail}" --prologue 200000 \
            --repeats "${PILOT_REPEATS}" --tag "${tag}"
    done

    # analyze_pilot.py needs SAMPLE and SUMMARY_PILOT for the SAME tag set in one file, and
    # aborts if the two sets differ. Only completed steps are concatenated, so a fail-soft
    # step cannot leave a half-written tag behind and break the analysis back on the dev box.
    : > "${RESULTS}/pilot_matrix.log"
    while IFS= read -r tag; do
        if [ -f "${RESULTS}/${tag}.done" ] || [ -f "${RESULTS}/${tag}.invalid" ]; then
            cat "${RESULTS}/${tag}.log" >> "${RESULTS}/pilot_matrix.log"
        fi
    done < "${expected}"
    log "pilot matrix -> ${RESULTS}/pilot_matrix.log"
    log "expected tags -> ${expected} ($(wc -l < "${expected}") configurations)"
    log "analyze with tools/analyze_pilot.py (NOT analyze.py -- different schema)"
    log "NOTE: grids=${PILOT_GRIDS[*]} (SM=${PILOT_SMS_COUNT}); gate accepts multi-wave only with trace_verified proof."
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

# ------------------------------------------------------------------ Tier 2/3 native (admissible)
run_tier23_native() {
    local native_results="${TIER23_RESULTS:-${RESULTS}/tier23_native}"
    local native_gate="${GATE_JSON:-${RESULTS}/gate.json}"
    log "=== Tier 2/3 native: §7.1/§7.3-§7.6 (separate strict campaign) ==="
    RESULTS="${native_results}" FAST="${FAST}" STEP_TIMEOUT="${STEP_TIMEOUT}" \
        T23_SMS="${PILOT_SMS_COUNT}" GATE_JSON="${native_gate}" \
        ./run_tier23_native.sh
}

# 'all' deliberately excludes tier1/tier23: they drive the rejected harness, and spending the
# 2h decision-point budget on inadmissible data is exactly the failure this driver exists to
# prevent. Ask for them by name if you are re-auditing.
case "${PHASE}" in
    tier0)  run_tier0 ;;
    tier1p) run_tier1_pilot ;;
    tier1)  run_tier1 ;;
    tier23) run_tier23 ;;
    tier23n) run_tier23_native ;;
    all)    run_tier0; run_tier1_pilot ;;
esac

log "=== campaign finished ==="
summary_rows=0
[ -f "${RESULTS}/summary.txt" ] && summary_rows="$(wc -l < "${RESULTS}/summary.txt")"
log "SUMMARY lines: ${RESULTS}/summary.txt (${summary_rows} rows)"
if [ -s "${RESULTS}/failures.log" ]; then
    log "FAILURES occurred:"; cat "${RESULTS}/failures.log"
fi
log "Copy ${RESULTS}/ back to the dev box and run tools/analyze.py"
