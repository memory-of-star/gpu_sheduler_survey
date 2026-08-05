#!/usr/bin/env bash
# run_production_tier5.sh -- failure-atomic production DSA driver.
#
# Default mode is a CPU-only dry run: no nvidia-smi, CUDA import, GPU query,
# nvcc, or timing sample.  Explicit execution requires both EXECUTE_GPU=1 and
# TIER5_PRODUCTION_GPU_ALLOWED=1.
#
# GPU execution reuses the native campaign's fail-closed protocol: a physical
# GPU identity snapshot, UUID-scoped global flock, idle lease, pre/post process
# snapshots, and a process-tree monitor whose SIGSTOP/READY/SIGCONT gate is
# durable before the harness can initialize CUDA.  Contamination creates a
# permanent formal_rejection.json at RESULTS; that directory is never resumable.
#
# A successful formal execution may set accepted_workload_timing=1.  It always
# keeps accepted_CTA_bracket=0 and the Tier-5 bracket PARTIAL because production
# APIs provide neither the CTA implementation nor an unordered Ceiling.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF="${SELF_DIR}/$(basename "$0")"
cd "${SELF_DIR}"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${SELF}"
    exit 0
fi
if [ "$#" -ne 0 ]; then
    echo "FAIL: unknown arguments: $*" >&2
    exit 2
fi

RESULTS_INPUT="${RESULTS:-results_dsa_production_dry_run}"
EXECUTE_GPU="${EXECUTE_GPU:-0}"
FAST="${FAST:-0}"
NVIDIA_SMI="${DSA_NVIDIA_SMI:-nvidia-smi}"
GPU_INDEX="${DSA_GPU_INDEX:-0}"
MONITOR_INTERVAL_MS="${DSA_MONITOR_INTERVAL_MS:-50}"
NVIDIA_SMI_TIMEOUT_MS="${DSA_NVIDIA_SMI_TIMEOUT_MS:-2000}"
STEP_TIMEOUT="${STEP_TIMEOUT:-0}"

case "${EXECUTE_GPU}" in
    0|1) ;;
    *) echo "FAIL: EXECUTE_GPU must be exactly 0 or 1" >&2; exit 2 ;;
esac
case "${FAST}" in
    0|1) ;;
    *) echo "FAIL: FAST must be exactly 0 or 1" >&2; exit 2 ;;
esac

# GPU execution is exclusively the resumable, independently sealed row path.
# The legacy monolithic body below remains reachable only for CPU dry-run.
if [ "${EXECUTE_GPU}" = "1" ]; then
    exec "${SELF_DIR}/run_production_tier5_fragments.sh"
fi

RESULTS_PARENT_INPUT="$(dirname -- "${RESULTS_INPUT}")"
RESULTS_NAME="$(basename -- "${RESULTS_INPUT}")"
mkdir -p -- "${RESULTS_PARENT_INPUT}"
RESULTS_PARENT="$(cd "${RESULTS_PARENT_INPUT}" && pwd -P)"
RESULTS="${RESULTS_PARENT}/${RESULTS_NAME}"

if [ -e "${RESULTS}" ]; then
    if [ -e "${RESULTS}/formal_rejection.json" ] || [ -e "${RESULTS}/REJECTED.md" ]; then
        echo "FAIL: RESULTS=${RESULTS} contains permanent rejection evidence; use a new directory" >&2
    else
        echo "FAIL: RESULTS already exists; choose a new failure-atomic destination: ${RESULTS}" >&2
    fi
    exit 2
fi

STAGE="$(mktemp -d "${RESULTS_PARENT}/${RESULTS_NAME}.inprogress.XXXXXX")"
PUBLISHED=0
PERMANENT_REJECTION=0

log() {
    echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${STAGE}/runner.log"
}

preserve_failure() {
    local rc="$?" failed
    trap - EXIT INT TERM
    if [ "${PUBLISHED}" = "0" ] && [ -d "${STAGE}" ]; then
        if [ "${PERMANENT_REJECTION}" = "1" ] && [ ! -e "${RESULTS}" ]; then
            mv -- "${STAGE}" "${RESULTS}"
            echo "FAIL: permanent rejected artifacts published at ${RESULTS}" >&2
        else
            failed="${RESULTS}.failed.$(date -u +%Y%m%dT%H%M%SZ).$$"
            mv -- "${STAGE}" "${failed}"
            echo "FAIL: incomplete artifacts preserved at ${failed}" >&2
        fi
    fi
    exit "${rc}"
}
trap preserve_failure EXIT INT TERM

sha_text() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }

write_exclusivity_rejection() {
    local phase="$1" evidence="$2"
    python3 - "${STAGE}/formal_rejection.json" "${phase}" "${evidence}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
phase = sys.argv[2]
evidence_path = Path(sys.argv[3])
try:
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
except (OSError, json.JSONDecodeError) as exc:
    evidence = {"status": "UNREADABLE", "error": str(exc)}
    evidence_sha = None
payload = {
    "schema": 1,
    "kind": "tier5_production_formal_rejection",
    "status": "REJECTED",
    "reason": "gpu_exclusivity_unproven_or_contaminated",
    "phase": phase,
    "accepted_timing": 0,
    "accepted_timing_semantics": "legacy_CTA_bracket_only",
    "accepted_workload_timing": 0,
    "accepted_CTA_bracket": 0,
    "formal_bracket_status": "PARTIAL",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "evidence_path": evidence_path.name,
    "evidence_sha256": evidence_sha,
    "evidence": evidence,
}
output.parent.mkdir(parents=True, exist_ok=True)
fd, temp = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output)
finally:
    try:
        os.unlink(temp)
    except FileNotFoundError:
        pass
PY
    printf '%s\n' \
        '# REJECTED' '' \
        'GPU exclusivity was not proven or contamination was observed.' \
        'This results directory is permanently inadmissible; use a new directory.' \
        > "${STAGE}/REJECTED.md"
    PERMANENT_REJECTION=1
}

HARNESS_ARGS=(
    --output-dir "${STAGE}"
    --publish-target "${RESULTS}"
    --runner-managed-stage
    --models "${MODELS:-deepseek_v32,glm5}"
    --seqs "${SEQS:-4096,32768,131072,1048576}"
    --workloads "${WORKLOADS:-operator_chain,single_layer,indexshare_fsss}"
    --warmup "${WARMUP:-5}"
    --repeats "${REPEATS:-31}"
    --max-logits-mb "${MAX_LOGITS_MB:-16384}"
    --max-query-chunk "${MAX_QUERY_CHUNK:-4096}"
    --moe-experts 32
    --moe-topk 8
    --moe-tokens "${MOE_TOKENS:-4096}"
    --seed "${SEED:-20260805}"
)

if [ "${FAST}" = "1" ]; then
    HARNESS_ARGS=(
        --output-dir "${STAGE}"
        --publish-target "${RESULTS}"
        --runner-managed-stage
        --models "${MODELS:-deepseek_v32}"
        --seqs "${SEQS:-4096}"
        --workloads "${WORKLOADS:-operator_chain,single_layer,indexshare_fsss}"
        --warmup "${WARMUP:-1}"
        --repeats "${REPEATS:-3}"
        --allow-short
        --max-logits-mb "${MAX_LOGITS_MB:-16384}"
        --max-query-chunk "${MAX_QUERY_CHUNK:-4096}"
        --moe-experts 32
        --moe-topk 8
        --moe-tokens "${MOE_TOKENS:-128}"
        --seed "${SEED:-20260805}"
    )
fi

if [ "${EXECUTE_GPU}" = "0" ]; then
    python3 ./production_tier5.py "${HARNESS_ARGS[@]}" \
        > "${STAGE}/harness.log" 2>&1 || {
            log "CPU dry-run harness failed"
            exit 2
        }
    python3 ./validate_production_tier5.py "${STAGE}" \
        --mode dry-run --json "${STAGE}/validation.json" \
        --marker "${STAGE}/production_dry_run.done.json" \
        >> "${STAGE}/runner.log" 2>&1 || {
            log "CPU dry-run validation failed"
            exit 2
        }
    mv -- "${STAGE}" "${RESULTS}"
    PUBLISHED=1
    trap - EXIT INT TERM
    echo "PRODUCTION_TIER5_PUBLISHED results=${RESULTS} mode=dry-run accepted_workload_timing=0 accepted_CTA_bracket=0 bracket=PARTIAL"
    exit 0
fi

if [ "${TIER5_PRODUCTION_GPU_ALLOWED:-0}" != "1" ]; then
    log "GPU mode needs TIER5_PRODUCTION_GPU_ALLOWED=1"
    exit 2
fi
for command in flock setsid timeout sha256sum; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        log "${command} unavailable; cannot establish fail-closed GPU execution"
        exit 2
    fi
done
case "${GPU_INDEX}" in
    ''|*[!0-9]*) log "DSA_GPU_INDEX must be a non-negative integer"; exit 2 ;;
esac
case "${MONITOR_INTERVAL_MS}" in
    ''|*[!0-9]*) log "DSA_MONITOR_INTERVAL_MS must be an integer in 10..100"; exit 2 ;;
esac
if [ "${MONITOR_INTERVAL_MS}" -lt 10 ] || [ "${MONITOR_INTERVAL_MS}" -gt 100 ]; then
    log "DSA_MONITOR_INTERVAL_MS must be bounded to 10..100ms"
    exit 2
fi
case "${NVIDIA_SMI_TIMEOUT_MS}" in
    ''|*[!0-9]*) log "DSA_NVIDIA_SMI_TIMEOUT_MS must be an integer in 100..5000"; exit 2 ;;
esac
if [ "${NVIDIA_SMI_TIMEOUT_MS}" -lt 100 ] || [ "${NVIDIA_SMI_TIMEOUT_MS}" -gt 5000 ]; then
    log "DSA_NVIDIA_SMI_TIMEOUT_MS must be bounded to 100..5000ms"
    exit 2
fi

IDENTITY="${STAGE}/gpu_identity.json"
LEASE="${STAGE}/gpu_exclusivity_lease.json"
GPU_PRE="${STAGE}/gpu_pre.json"
GPU_POST="${STAGE}/gpu_post.json"
GPU_MONITOR="${STAGE}/gpu_monitor.json"
GPU_OBSERVATIONS="${STAGE}/gpu_observations.ndjson"

python3 ./gpu_exclusivity.py identity --json "${IDENTITY}" \
    --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${GPU_INDEX}" \
    --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
    --owner-pid "$$" --phase production_global_lock_identity \
    >> "${STAGE}/runner.log" 2>&1 || {
        write_exclusivity_rejection production_global_lock_identity "${IDENTITY}"
        log "target GPU identity query failed; accepted_workload_timing=0"
        exit 2
    }

GPU_IDENTITY_BINDING="$(python3 - "${IDENTITY}" <<'PY'
import json
import re
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
target = value.get("target_gpu") or {}
uuid = target.get("uuid", "")
index = target.get("index")
requested_index = value.get("gpu_index")
if value.get("status") != "PASS" or value.get("errors") != []:
    raise SystemExit(2)
if not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid):
    raise SystemExit(2)
if (
    not isinstance(index, int)
    or isinstance(index, bool)
    or index < 0
    or index != requested_index
):
    raise SystemExit(2)
print(f"{uuid}\t{index}")
PY
)" || {
    write_exclusivity_rejection malformed_gpu_identity "${IDENTITY}"
    log "target GPU identity failed closed admission"
    exit 2
}
IFS=$'\t' read -r GPU_UUID RESOLVED_GPU_INDEX <<< "${GPU_IDENTITY_BINDING}"
if [ -z "${GPU_UUID}" ] || [ -z "${RESOLVED_GPU_INDEX}" ]; then
    write_exclusivity_rejection malformed_gpu_identity "${IDENTITY}"
    log "target GPU identity binding was incomplete"
    exit 2
fi

# This installed vLLM release parses CUDA_VISIBLE_DEVICES as an integer during
# import.  The selector therefore uses the physical index resolved by the
# identity probe; the UUID remains the lock/lease key and the harness rejects
# the run unless CUDA runtime ordinal 0 maps back to that exact UUID.
export CUDA_VISIBLE_DEVICES="${RESOLVED_GPU_INDEX}"
GPU_GLOBAL_LOCK="/tmp/cta_pdl_gpu_${GPU_UUID}.lock"
GPU_GLOBAL_LOCK_KEY_SHA256="$(sha_text "${GPU_UUID}")"
GPU_GLOBAL_LOCK_PATH_SHA256="$(sha_text "${GPU_GLOBAL_LOCK}")"
exec 9>"${GPU_GLOBAL_LOCK}"
if ! flock -n 9; then
    log "another campaign holds the global target-GPU lock ${GPU_GLOBAL_LOCK}"
    exit 2
fi

python3 ./gpu_exclusivity.py acquire --json "${LEASE}" \
    --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${RESOLVED_GPU_INDEX}" \
    --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
    --owner-pid "$$" --phase production_campaign_acquire \
    >> "${STAGE}/runner.log" 2>&1 || {
        write_exclusivity_rejection production_campaign_acquire "${LEASE}"
        log "GPU exclusivity acquisition failed"
        exit 2
    }

python3 ./gpu_exclusivity.py check --lease "${LEASE}" --json "${GPU_PRE}" \
    --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${RESOLVED_GPU_INDEX}" \
    --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
    --owner-pid "$$" --phase production_tier5_pre \
    >> "${STAGE}/runner.log" 2>&1 || {
        write_exclusivity_rejection production_tier5_pre "${GPU_PRE}"
        log "GPU pre-check failed"
        exit 2
    }

run_monitored() {
    local child_pid monitor_pid command_rc=0 monitor_rc=0 ready_status="" ready_wait
    local ready_file="${GPU_MONITOR}.ready"
    local -a command=("$@")
    if [ "${STEP_TIMEOUT}" != "0" ]; then
        command=(timeout --kill-after=30s "${STEP_TIMEOUT}" "${command[@]}")
    fi
    setsid bash -c 'kill -STOP "$$"; exec "$@"' production-monitor-gate \
        "${command[@]}" &
    child_pid=$!
    python3 ./gpu_exclusivity.py monitor --lease "${LEASE}" \
        --json "${GPU_MONITOR}" --observations "${GPU_OBSERVATIONS}" \
        --ready-file "${ready_file}" --nvidia-smi "${NVIDIA_SMI}" \
        --gpu-index "${RESOLVED_GPU_INDEX}" --watch-pid "${child_pid}" \
        --phase production_tier5 --interval-ms "${MONITOR_INTERVAL_MS}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --require-allowed-process --terminate-on-failure \
        >> "${STAGE}/runner.log" 2>&1 &
    monitor_pid=$!
    for ((ready_wait = 0; ready_wait < 3000; ready_wait++)); do
        [ -s "${ready_file}" ] && break
        sleep 0.01
    done
    if [ -s "${ready_file}" ]; then
        ready_status="$(python3 - "${ready_file}" <<'PY'
import json
import sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(2)
if value.get("schema") != 1 or value.get("kind") != "gpu_exclusivity_monitor_ready":
    raise SystemExit(2)
print(value.get("status", ""))
PY
)" || ready_status="INVALID"
    else
        ready_status="MISSING"
    fi
    if [ "${ready_status}" = "READY" ]; then
        kill -CONT "${child_pid}" 2>/dev/null || true
    else
        kill -TERM -- "-${child_pid}" 2>/dev/null || true
        kill -CONT -- "-${child_pid}" 2>/dev/null || true
    fi
    wait "${child_pid}" || command_rc=$?
    wait "${monitor_pid}" || monitor_rc=$?
    rm -f -- "${ready_file}"
    if [ "${ready_status}" != "READY" ]; then
        write_exclusivity_rejection production_tier5_monitor_start_gate "${GPU_MONITOR}"
        log "monitor start gate failed status=${ready_status}"
        exit 2
    fi
    if [ "${monitor_rc}" -ne 0 ]; then
        write_exclusivity_rejection production_tier5_monitor "${GPU_MONITOR}"
        log "runtime GPU monitor failed rc=${monitor_rc}"
        exit 2
    fi
    return "${command_rc}"
}

HARNESS_ARGS+=(
    --execute-gpu
    --expected-gpu-uuid "${GPU_UUID}"
    --expected-gpu-index "${RESOLVED_GPU_INDEX}"
)
HARNESS_RC=0
run_monitored python3 ./production_tier5.py "${HARNESS_ARGS[@]}" \
    > "${STAGE}/harness.log" 2>&1 || HARNESS_RC=$?

python3 ./gpu_exclusivity.py check --lease "${LEASE}" --json "${GPU_POST}" \
    --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${RESOLVED_GPU_INDEX}" \
    --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
    --owner-pid "$$" --phase production_tier5_post \
    >> "${STAGE}/runner.log" 2>&1 || {
        write_exclusivity_rejection production_tier5_post "${GPU_POST}"
        log "GPU post-check failed"
        exit 2
    }
if [ "${HARNESS_RC}" -ne 0 ]; then
    log "production harness failed rc=${HARNESS_RC}; workload timing not accepted"
    exit 2
fi

python3 ./validate_production_tier5.py "${STAGE}" \
    --mode execute --json "${STAGE}/validation.json" \
    --marker "${STAGE}/production_candidate.done.json" \
    --expected-gpu-uuid "${GPU_UUID}" \
    --expected-gpu-index "${RESOLVED_GPU_INDEX}" \
    --global-lock-key-sha256 "${GPU_GLOBAL_LOCK_KEY_SHA256}" \
    --global-lock-path-sha256 "${GPU_GLOBAL_LOCK_PATH_SHA256}" \
    --monitor-interval-ms "${MONITOR_INTERVAL_MS}" \
    --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
    >> "${STAGE}/runner.log" 2>&1 || {
        log "production artifact validation failed"
        exit 2
    }

ACCEPTED_WORKLOAD="$(python3 - "${STAGE}/validation.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "PASS" or value.get("accepted_CTA_bracket") != 0:
    raise SystemExit(2)
print(value.get("accepted_workload_timing", 0))
PY
)" || {
    log "validated acceptance fields are malformed"
    exit 2
}

mv -- "${STAGE}" "${RESULTS}"
PUBLISHED=1
trap - EXIT INT TERM
echo "PRODUCTION_TIER5_PUBLISHED results=${RESULTS} mode=execute accepted_workload_timing=${ACCEPTED_WORKLOAD} accepted_CTA_bracket=0 bracket=PARTIAL"
