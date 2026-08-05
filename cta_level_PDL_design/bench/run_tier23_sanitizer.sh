#!/usr/bin/env bash
# Independent, non-timing Compute Sanitizer evidence for the native Tier 2/3 binaries.

set -uo pipefail
cd "$(dirname "$0")"

RESULTS="${RESULTS:-results_20260805_b200_tier23_native_v2}"
STEP_TIMEOUT="${STEP_TIMEOUT:-900}"
SANITIZER_LABEL="${SANITIZER_LABEL:-sanitizer}"
VALIDATION="${RESULTS}/tier23_validation.json"
OUT="${RESULTS}/${SANITIZER_LABEL}"
STATUS="${RESULTS}/${SANITIZER_LABEL}_status.tsv"

python3 - "${VALIDATION}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"refusing sanitizer before strict formal validation: {exc}")
if data.get("status") != "PASS" or data.get("formal") is not True:
    raise SystemExit("refusing sanitizer before strict formal Tier 2/3 PASS")
PY

mkdir -p "${OUT}"
printf 'schema\tcase\tbinary\tbinary_sha256\tstatus\trc\terror_summary\tcoverage\tmemcheck_log\ttarget_log\n' \
    > "${STATUS}"

if ! command -v compute-sanitizer >/dev/null 2>&1; then
    printf '1\tall\tcompute-sanitizer\tmissing\tTOOL_UNAVAILABLE\t127\ttool_not_installed\t-\t-\t-\n' \
        >> "${STATUS}"
    exit 3
fi

nonpass=0
memory_error=0

run_case() {
    local name="$1" binary="$2"; shift 2
    local memlog="${OUT}/${name}.memcheck.log"
    local targetlog="${OUT}/${name}.target.log"
    local driverlog="${OUT}/${name}.driver.log"
    local trace="${OUT}/${name}.trace.csv"
    local rc status summary hash coverage safe_prefix=0
    local -a tool_extra=() target=()

    hash=$(sha256sum -- "${binary}" | awk '{print $1}')
    case "${name}" in
        protocol)
            safe_prefix=4
            coverage=grid,fixed-spin,backoff,monotonic-prefix
            ;;
        c1)
            safe_prefix=2
            coverage=separate-default,separate-persist
            ;;
        diamond)
            coverage=grid-ordered,cta-ordered,cta-unordered,none
            ;;
        clc)
            coverage=producer-priority,consumer-priority,locality,none
            ;;
        *) coverage=unknown ;;
    esac
    if [ "${safe_prefix}" -gt 0 ]; then
        # The frozen protocol binary deliberately deadlocks its unsafe adversarial sentinel
        # when a tool serializes PDL.  Audit all four safe validation modes, then let an
        # external wrapper stop the child before PE_NONE.  Memory errors still force rc=97.
        tool_extra=(--check-exit-code no)
        target=(./tier23_sanitizer_safe_prefix.sh "${safe_prefix}" "${targetlog}" "${binary}"
                --tag "sanitizer_${name}" --trace "${trace}" "$@")
    else
        target=("${binary}" --tag "sanitizer_${name}" --trace "${trace}" "$@")
    fi
    if command -v timeout >/dev/null 2>&1; then
        timeout --kill-after=30s "${STEP_TIMEOUT}" \
            compute-sanitizer --tool memcheck --target-processes all \
            --error-exitcode 97 --log-file "${memlog}" "${tool_extra[@]}" \
            "${target[@]}" > "${driverlog}" 2>&1
        rc=$?
    else
        compute-sanitizer --tool memcheck --target-processes all \
            --error-exitcode 97 --log-file "${memlog}" "${tool_extra[@]}" \
            "${target[@]}" > "${driverlog}" 2>&1
        rc=$?
    fi

    if [ "${safe_prefix}" -eq 0 ]; then
        mv "${driverlog}" "${targetlog}"
    fi

    summary=$(rg 'ERROR SUMMARY:' "${memlog}" 2>/dev/null | tail -1 | tr '\t' ' ')
    summary="${summary:-missing_error_summary}"
    if [ "${rc}" -eq 0 ] && printf '%s\n' "${summary}" | rg -q 'ERROR SUMMARY: 0 errors'; then
        if [ "${safe_prefix}" -gt 0 ] \
                && rg -q "^SANITIZER_SAFE_PREFIX status=PASS validations=${safe_prefix} " \
                    "${targetlog}"; then
            status=PASS_SAFE_MODES
        else
            status=PASS
        fi
    elif [ "${rc}" -eq 97 ] || printf '%s\n' "${summary}" | rg -q 'ERROR SUMMARY: [1-9][0-9]* error'; then
        status=MEMORY_ERROR
        memory_error=1
        nonpass=1
    elif [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
        status=TOOL_TIMEOUT
        nonpass=1
    elif printf '%s\n' "${summary}" | rg -q 'ERROR SUMMARY: 0 errors' \
            && rg -q '^VALIDATION_TIER23 .*mode=none .*status=FAIL' "${targetlog}"; then
        status=TARGET_SEMANTIC_FAILURE_UNDER_TOOL
        nonpass=1
    elif [ -f "${memlog}" ] && rg -qi \
        'not supported|unsupported|failed to instrument|internal error|terminated before' \
        "${memlog}" "${targetlog}"; then
        status=TOOL_UNSUPPORTED
        nonpass=1
    else
        status=TARGET_OR_TOOL_FAILURE
        nonpass=1
    fi

    summary=${summary//$'\n'/ }
    printf '1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${binary}" "${hash}" "${status}" "${rc}" "${summary}" "${coverage}" \
        "${memlog}" "${targetlog}" >> "${STATUS}"
}

# All four are intentionally tiny, admissible correctness invocations.  These samples are
# outside tier23_manifest.tsv and may never be used as formal timing observations.
run_case protocol ./tier23_protocol_encoding \
    --experiment protocol --producers 16 --consumers 16 --structure self --degree 1 \
    --background-blocks 16 --background-iterations 1000 \
    --ready 500000 --tail 10000 --prologue 10000 --epilogue 10000 \
    --repeats 1 --warmup 0 --allow-short

run_case diamond ./tier23_diamond \
    --blocks 8 --ratio 1 --base-cycles 10000 --tail-cycles 10000 \
    --repeats 1 --warmup 0 --allow-short

run_case c1 ./tier23_c1 \
    --tiles 2 --bytes-per-tile 1024 --ready-cycles 500000 --tail-cycles 10000 \
    --repeats 1 --warmup 0 --allow-short

# B200 reports 16 resident scheduler CTAs/SM, so 1200 tiles (2400 launch tokens) is the
# smallest rounded point beyond the 2368-token resident capacity required by this harness.
run_case clc ./tier23_clc_scheduler \
    --tiles 1200 --producer-cycles 10000 --consumer-cycles 10000 \
    --repeats 1 --warmup 0 --allow-short

if [ "${memory_error}" -ne 0 ]; then
    exit 2
fi
if [ "${nonpass}" -ne 0 ]; then
    exit 3
fi
exit 0
