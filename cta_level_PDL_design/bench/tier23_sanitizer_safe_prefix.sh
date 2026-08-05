#!/usr/bin/env bash
# Run a frozen binary under Compute Sanitizer until its declared safe validation prefix passes.
# The unsafe Ceiling that follows is deliberately not a memory-safety target: some PDL tools
# serialize it and thereby either deadlock a proof latch or make the omitted wait read correct.

set -uo pipefail

if [ "$#" -lt 4 ]; then
    echo "usage: $0 PASS_COUNT TARGET_LOG BINARY ARGS..." >&2
    exit 2
fi

PASS_COUNT="$1"
TARGET_LOG="$2"
BINARY="$3"
shift 3

if ! [[ "${PASS_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid safe-prefix pass count" >&2
    exit 2
fi

: > "${TARGET_LOG}"
safe_passes=0

coproc T23_TARGET { exec stdbuf -oL -eL "${BINARY}" "$@"; }
target_pid="${T23_TARGET_PID}"

while IFS= read -r line <&"${T23_TARGET[0]}"; do
    printf '%s\n' "${line}" >> "${TARGET_LOG}"
    case "${line}" in
        VALIDATION_TIER23*status=PASS*) safe_passes=$((safe_passes + 1)) ;;
    esac
    if [ "${safe_passes}" -ge "${PASS_COUNT}" ]; then
        kill -TERM "${target_pid}" 2>/dev/null || true
        wait "${target_pid}" 2>/dev/null || true
        printf 'SANITIZER_SAFE_PREFIX status=PASS validations=%s target_stopped_before_unsafe=1\n' \
            "${safe_passes}" >> "${TARGET_LOG}"
        exit 0
    fi
done

wait "${target_pid}" 2>/dev/null
target_rc=$?
printf 'SANITIZER_SAFE_PREFIX status=FAIL validations=%s required=%s target_rc=%s\n' \
    "${safe_passes}" "${PASS_COUNT}" "${target_rc}" >> "${TARGET_LOG}"
exit 2
