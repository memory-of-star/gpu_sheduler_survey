#!/usr/bin/env bash
# run_tier23_native.sh -- resumable/fail-soft formal matrix for Tier 2/3 §7.1-§7.6.

set -uo pipefail
cd "$(dirname "$0")"

RESULTS="${RESULTS:-results_tier23_native}"
FAST="${FAST:-0}"
STEP_TIMEOUT="${STEP_TIMEOUT:-600}"
T23_SMS="${T23_SMS:-148}"
GATE_JSON="${GATE_JSON:-results_20260805_b200_multiwave_v2/gate.json}"
FRESH=0
if [ "${1:-}" = "--fresh" ]; then FRESH=1; shift; fi
if [ "$#" -ne 0 ]; then
    echo "usage: RESULTS=... [FAST=1] $0 [--fresh]" >&2
    exit 2
fi

mkdir -p "${RESULTS}"
MANIFEST="${RESULTS}/tier23_manifest.tsv"
MATRIX="${RESULTS}/tier23_matrix.log"
FAILURES="${RESULTS}/failures.log"

archive_file() {
    local path="$1" n=1
    [ -e "${path}" ] || return 0
    while [ -e "${path}.retry${n}" ]; do n=$((n + 1)); done
    mv "${path}" "${path}.retry${n}"
}

if [ "${FRESH}" = "1" ]; then
    for path in "${MANIFEST}" "${MATRIX}" "${FAILURES}" \
                "${RESULTS}/tier23_validation.json" \
                "${RESULTS}/tier23_summary.csv" \
                "${RESULTS}/ncu_status.txt" "${RESULTS}/nsys_status.txt"; do
        archive_file "${path}"
    done
    rm -f "${RESULTS}"/*.done "${RESULTS}"/*.invalid
fi
: > "${FAILURES}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/campaign.log"; }
fail() {
    echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "${RESULTS}/campaign.log" \
        | tee -a "${FAILURES}"
}

if [ "${FAST}" != "1" ]; then
    if [ ! -f "${GATE_JSON}" ]; then
        echo "formal Tier 2/3 requires the complete Tier 1 gate JSON: ${GATE_JSON}" >&2
        exit 2
    fi
    gate_verdict=$(python3 - "${GATE_JSON}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(d.get("verdict", d.get("gate", "")))
except Exception:
    print("")
PY
)
    if [ "${gate_verdict}" != "GO" ]; then
        echo "formal Tier 2/3 admission requires gate=GO, got '${gate_verdict}'" >&2
        exit 2
    fi
fi

needs_build=0
for pair in \
    tier23_protocol_encoding.cu:tier23_protocol_encoding \
    tier23_diamond.cu:tier23_diamond \
    tier23_c1.cu:tier23_c1 \
    tier23_clc_scheduler.cu:tier23_clc_scheduler; do
    source_file="${pair%%:*}"
    binary="${pair##*:}"
    if [ ! -x "./${binary}" ] || [ "${source_file}" -nt "${binary}" ] || \
       [ common/tier23_native.cuh -nt "${binary}" ]; then
        needs_build=1
    fi
done
if [ "${needs_build}" -eq 1 ]; then
    log "Tier 2/3 binary missing/stale; building all benchmarks"
    ARCH="${ARCH:-sm_100}" ./build.sh >> "${RESULTS}/build.log" 2>&1 || {
        fail "build failed"; exit 2;
    }
fi
for pair in \
    tier23_protocol_encoding.cu:tier23_protocol_encoding \
    tier23_diamond.cu:tier23_diamond \
    tier23_c1.cu:tier23_c1 \
    tier23_clc_scheduler.cu:tier23_clc_scheduler; do
    source_file="${pair%%:*}"
    binary="${pair##*:}"
    if [ ! -x "./${binary}" ] || [ "${source_file}" -nt "${binary}" ] || \
       [ common/tier23_native.cuh -nt "${binary}" ]; then
        fail "required Tier 2/3 binary is absent/stale after build: ${binary}"
        exit 2
    fi
done

DEVICE_FINGERPRINT=$(nvidia-smi --query-gpu=uuid,name,compute_cap,driver_version \
    --format=csv,noheader,nounits 2>/dev/null | head -1)
DEVICE_FINGERPRINT="${DEVICE_FINGERPRINT:-unavailable}"
nvidia-smi -q > "${RESULTS}/device.txt" 2>&1 || true

signature() {
    local binary="$1" hash="missing"; shift
    [ -f "${binary}" ] && hash=$(sha256sum -- "${binary}" | awk '{print $1}')
    printf 'marker_schema=1\nfast=%s\ndevice=%q\nbinary_sha256=%s\nargv=' \
        "${FAST}" "${DEVICE_FINGERPRINT}" "${hash}"
    printf '%q ' "${binary}" "$@"
}

run_step() {
    local tag="$1"; shift
    local marker="${RESULTS}/${tag}.done" invalid="${RESULTS}/${tag}.invalid"
    local step_log="${RESULTS}/${tag}.log" expected rc
    expected=$(signature "$@")
    if [ -f "${marker}" ] && [ "$(cat "${marker}")" = "${expected}" ]; then
        log "skip ${tag} (matching signature)"
        return 0
    fi
    if [ -f "${invalid}" ] && [ "$(cat "${invalid}")" = "${expected}" ]; then
        log "skip ${tag} (matching semantic-invalid signature)"
        return 0
    fi
    [ -e "${marker}" ] && rm -f "${marker}"
    [ -e "${invalid}" ] && rm -f "${invalid}"
    archive_file "${step_log}"
    log "run ${tag}"
    if [ "${STEP_TIMEOUT}" != "0" ] && command -v timeout >/dev/null 2>&1; then
        timeout --kill-after=30s "${STEP_TIMEOUT}" "$@" > "${step_log}" 2>&1
        rc=$?
    else
        "$@" > "${step_log}" 2>&1
        rc=$?
    fi
    if [ "${rc}" -eq 0 ]; then
        printf '%s\n' "${expected}" > "${marker}"
        log "ok ${tag}"
    else
        printf '%s\n' "${expected}" > "${invalid}"
        fail "${tag} rc=${rc} (see ${step_log})"
    fi
}

manifest_header() {
    printf 'tag\texperiment\tlog\ttrace\tmodes\trepeats\twarmup\tP\tC\tstructure\tdegree\tratio\tbytes_per_tile\ttiles\n'
}
manifest_row() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@"
}

if [ "${FAST}" = "1" ]; then
    REPEATS=3
    WARMUP=1
    SHORT=(--allow-short)
    PROTOCOL_GRIDS=(32)
    ENCODING_DEGREES=(8)
    DIAMOND_RATIOS=(1 10)
    C1_BYTES_KB=(1 64)
    CLC_TILES=2048
    BG_BLOCKS=32
    BG_ITERS=20000
    READY=500000
    TAIL=300000
    PROLOGUE=10000
    EPILOGUE=10000
else
    REPEATS=31
    WARMUP=3
    SHORT=()
    PROTOCOL_GRIDS=("${T23_SMS}" "$((2 * T23_SMS))" "$((8 * T23_SMS))")
    ENCODING_DEGREES=(1 2 4 8 16 32 64)
    # Plan §7.4 requires the full 1:1 -> 1:10 scan, not selected endpoints.
    DIAMOND_RATIOS=(1 2 3 4 5 6 7 8 9 10)
    C1_BYTES_KB=(1 2 4 8 16 32 64)
    CLC_TILES=4096
    BG_BLOCKS="$((2 * T23_SMS))"
    BG_ITERS=50000
    READY=600000
    TAIL=800000
    PROLOGUE=100000
    EPILOGUE=400000
fi

manifest_header > "${MANIFEST}"

for grid in "${PROTOCOL_GRIDS[@]}"; do
    tag="t23_protocol_g${grid}"
    trace="${RESULTS}/${tag}_trace.csv"
    manifest_row "${tag}" protocol "${RESULTS}/${tag}.log" "${trace}" \
        grid,fixed-spin,backoff,monotonic-prefix,none "${REPEATS}" "${WARMUP}" \
        "${grid}" "${grid}" self 1 0 0 0 >> "${MANIFEST}"
    run_step "${tag}" ./tier23_protocol_encoding --experiment protocol --tag "${tag}" \
        --trace "${trace}" --producers "${grid}" --consumers "${grid}" \
        --structure self --degree 1 --repeats "${REPEATS}" --warmup "${WARMUP}" \
        --background-blocks "${BG_BLOCKS}" --background-iterations "${BG_ITERS}" \
        --ready "${READY}" --tail "${TAIL}" --prologue "${PROLOGUE}" \
        --epilogue "${EPILOGUE}" "${SHORT[@]}"
done

encoding_grid="$((2 * T23_SMS))"
if [ "${FAST}" = "1" ]; then encoding_grid=64; fi
for structure in interval strided; do
    for degree in "${ENCODING_DEGREES[@]}"; do
        tag="t23_encoding_${structure}_d${degree}"
        trace="${RESULTS}/${tag}_trace.csv"
        manifest_row "${tag}" encoding "${RESULTS}/${tag}.log" "${trace}" \
            grid,interval,bitmask,csr,none "${REPEATS}" "${WARMUP}" \
            "${encoding_grid}" "${encoding_grid}" "${structure}" "${degree}" 0 0 0 \
            >> "${MANIFEST}"
        run_step "${tag}" ./tier23_protocol_encoding --experiment encoding --tag "${tag}" \
            --trace "${trace}" --producers "${encoding_grid}" --consumers "${encoding_grid}" \
            --structure "${structure}" --degree "${degree}" \
            --repeats "${REPEATS}" --warmup "${WARMUP}" \
            --background-blocks "${BG_BLOCKS}" --background-iterations "${BG_ITERS}" \
            --ready "${READY}" --tail "${TAIL}" --prologue "${PROLOGUE}" \
            --epilogue "${EPILOGUE}" "${SHORT[@]}"
    done
done

diamond_blocks="${T23_SMS}"
[ "${FAST}" = "1" ] && diamond_blocks=32
for ratio in "${DIAMOND_RATIOS[@]}"; do
    tag="t23_diamond_r${ratio}"
    trace="${RESULTS}/${tag}_trace.csv"
    manifest_row "${tag}" diamond "${RESULTS}/${tag}.log" "${trace}" \
        grid-ordered,cta-ordered,cta-unordered,none "${REPEATS}" "${WARMUP}" \
        "${diamond_blocks}" "${diamond_blocks}" diamond 1 "${ratio}" 0 0 \
        >> "${MANIFEST}"
    run_step "${tag}" ./tier23_diamond --tag "${tag}" --trace "${trace}" \
        --blocks "${diamond_blocks}" --ratio "${ratio}" --repeats "${REPEATS}" \
        --warmup "${WARMUP}" "${SHORT[@]}"
done

c1_tiles="${T23_SMS}"
[ "${FAST}" = "1" ] && c1_tiles=32
for kb in "${C1_BYTES_KB[@]}"; do
    bytes=$((kb * 1024))
    tag="t23_c1_kb${kb}"
    trace="${RESULTS}/${tag}_trace.csv"
    manifest_row "${tag}" c1 "${RESULTS}/${tag}.log" "${trace}" \
        separate-default,separate-persist,none,fused-cluster,separate-cv \
        "${REPEATS}" "${WARMUP}" "${c1_tiles}" "${c1_tiles}" self 1 0 \
        "${bytes}" "${c1_tiles}" >> "${MANIFEST}"
    run_step "${tag}" ./tier23_c1 --tag "${tag}" --trace "${trace}" \
        --tiles "${c1_tiles}" --bytes-per-tile "${bytes}" \
        --repeats "${REPEATS}" --warmup "${WARMUP}" "${SHORT[@]}"
done

tag="t23_clc"
trace="${RESULTS}/${tag}_trace.csv"
manifest_row "${tag}" clc "${RESULTS}/${tag}.log" "${trace}" \
    producer-priority,consumer-priority,locality,none "${REPEATS}" "${WARMUP}" \
    "${CLC_TILES}" "${CLC_TILES}" self 1 0 0 "${CLC_TILES}" >> "${MANIFEST}"
run_step "${tag}" ./tier23_clc_scheduler --tag "${tag}" --trace "${trace}" \
    --tiles "${CLC_TILES}" --repeats "${REPEATS}" --warmup "${WARMUP}" \
    "${SHORT[@]}"

: > "${MATRIX}"
while IFS=$'\t' read -r tag experiment step_log trace_path rest; do
    [ "${tag}" = "tag" ] && continue
    if [ -f "${RESULTS}/${tag}.done" ] || [ -f "${RESULTS}/${tag}.invalid" ]; then
        cat "${step_log}" >> "${MATRIX}"
    fi
done < "${MANIFEST}"

# Profiler sidecars are diagnostics, never substitutes for raw globaltimer samples.  NCU
# permission failure is recorded verbatim and must not block the timing matrix.
run_profiler_sidecars() {
    local probe_dir="${RESULTS}/profiler_probe"
    mkdir -p "${probe_dir}"
    if command -v ncu >/dev/null 2>&1; then
        ncu --set basic --target-processes all --force-overwrite \
            -o "${probe_dir}/ncu_protocol" ./tier23_protocol_encoding \
            --experiment protocol --tag profiler_ncu --trace "${probe_dir}/ncu_trace.csv" \
            --producers 16 --consumers 16 --structure self --degree 1 \
            --background-blocks 16 --background-iterations 20000 \
            --ready 500000 --tail 300000 --prologue 10000 --epilogue 10000 \
            --repeats 1 --warmup 0 --allow-short \
            > "${probe_dir}/ncu.log" 2>&1
        ncu_rc=$?
        if rg -q 'ERR_NVGPUCTRPERM' "${probe_dir}/ncu.log"; then
            printf 'status=unavailable reason=ERR_NVGPUCTRPERM rc=%s blocking=0\n' "${ncu_rc}" \
                > "${RESULTS}/ncu_status.txt"
        else
            printf 'status=%s rc=%s blocking=0 report=%s\n' \
                "$([ "${ncu_rc}" -eq 0 ] && echo captured || echo failed)" "${ncu_rc}" \
                "${probe_dir}/ncu_protocol.ncu-rep" > "${RESULTS}/ncu_status.txt"
        fi
    else
        printf 'status=missing reason=ncu_not_installed blocking=0\n' \
            > "${RESULTS}/ncu_status.txt"
    fi

    if command -v nsys >/dev/null 2>&1; then
        nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
            --force-overwrite=true -o "${probe_dir}/nsys_protocol" \
            ./tier23_protocol_encoding --experiment protocol --tag profiler_nsys \
            --trace "${probe_dir}/nsys_trace.csv" --producers 16 --consumers 16 \
            --structure self --degree 1 --background-blocks 16 --background-iterations 20000 \
            --ready 500000 --tail 300000 --prologue 10000 --epilogue 10000 \
            --repeats 1 --warmup 0 --allow-short \
            > "${probe_dir}/nsys.log" 2>&1
        nsys_rc=$?
        printf 'status=%s rc=%s blocking=0 report=%s\n' \
            "$([ "${nsys_rc}" -eq 0 ] && echo captured || echo failed)" "${nsys_rc}" \
            "${probe_dir}/nsys_protocol.nsys-rep" > "${RESULTS}/nsys_status.txt"
    else
        printf 'status=missing reason=nsys_not_installed blocking=0\n' \
            > "${RESULTS}/nsys_status.txt"
    fi
}

run_profiler_sidecars

validator_args=("${RESULTS}" --manifest "${MANIFEST}" \
    --json "${RESULTS}/tier23_validation.json" \
    --csv "${RESULTS}/tier23_summary.csv")
[ "${FAST}" = "1" ] && validator_args+=(--allow-incomplete)
if python3 ../tools/validate_tier23_native.py "${validator_args[@]}"; then
    log "strict Tier 2/3 validation PASS"
else
    rc=$?
    fail "strict Tier 2/3 validation failed rc=${rc}"
fi

if [ -s "${FAILURES}" ]; then
    log "Tier 2/3 campaign finished INVALID; see ${FAILURES}"
    exit 2
fi
log "Tier 2/3 campaign finished PASS"
