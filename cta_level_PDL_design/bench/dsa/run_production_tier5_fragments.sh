#!/usr/bin/env bash
# Resumable Tier-5 production campaign: one independent GPU lease per row.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd "${SELF_DIR}"

if [ "$#" -ne 0 ]; then
    echo "FAIL: run_production_tier5_fragments.sh takes no arguments" >&2
    exit 2
fi
if [ "${EXECUTE_GPU:-0}" != "1" ]; then
    echo "FAIL: fragment runner requires EXECUTE_GPU=1" >&2
    exit 2
fi
if [ "${TIER5_PRODUCTION_GPU_ALLOWED:-0}" != "1" ]; then
    echo "FAIL: fragment runner requires TIER5_PRODUCTION_GPU_ALLOWED=1" >&2
    exit 2
fi

FAST="${FAST:-0}"
case "${FAST}" in 0|1) ;; *) echo "FAIL: FAST must be 0 or 1" >&2; exit 2 ;; esac
FRAGMENT_ONLY_ROW="${TIER5_FRAGMENT_ONLY_ROW:-}"
if [ -n "${FRAGMENT_ONLY_ROW}" ] && [ "${FAST}" != "1" ]; then
    echo "FAIL: TIER5_FRAGMENT_ONLY_ROW is nonformal and requires FAST=1" >&2
    exit 2
fi
NVIDIA_SMI="${DSA_NVIDIA_SMI:-nvidia-smi}"
GPU_INDEX="${DSA_GPU_INDEX:-0}"
MONITOR_INTERVAL_MS="${DSA_MONITOR_INTERVAL_MS:-50}"
NVIDIA_SMI_TIMEOUT_MS="${DSA_NVIDIA_SMI_TIMEOUT_MS:-2000}"
STEP_TIMEOUT="${STEP_TIMEOUT:-0}"
if [ "${STEP_TIMEOUT}" != "0" ]; then
    echo "FAIL: STEP_TIMEOUT is forbidden for formal fragment identity; use an external campaign policy" >&2
    exit 2
fi
for command in flock setsid timeout sha256sum ps; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "FAIL: ${command} is required" >&2
        exit 2
    }
done
case "${GPU_INDEX}" in ''|*[!0-9]*) echo "FAIL: DSA_GPU_INDEX must be non-negative" >&2; exit 2 ;; esac
case "${MONITOR_INTERVAL_MS}" in ''|*[!0-9]*) echo "FAIL: invalid monitor interval" >&2; exit 2 ;; esac
case "${NVIDIA_SMI_TIMEOUT_MS}" in ''|*[!0-9]*) echo "FAIL: invalid query timeout" >&2; exit 2 ;; esac
if [ "${MONITOR_INTERVAL_MS}" -lt 10 ] || [ "${MONITOR_INTERVAL_MS}" -gt 100 ]; then
    echo "FAIL: DSA_MONITOR_INTERVAL_MS must be 10..100" >&2
    exit 2
fi
if [ "${NVIDIA_SMI_TIMEOUT_MS}" -lt 100 ] || [ "${NVIDIA_SMI_TIMEOUT_MS}" -gt 5000 ]; then
    echo "FAIL: DSA_NVIDIA_SMI_TIMEOUT_MS must be 100..5000" >&2
    exit 2
fi

RESULTS_INPUT="${RESULTS:-results_dsa_production_campaign}"
RESULTS_PARENT_INPUT="$(dirname -- "${RESULTS_INPUT}")"
RESULTS_NAME="$(basename -- "${RESULTS_INPUT}")"
mkdir -p -- "${RESULTS_PARENT_INPUT}"
RESULTS_PARENT="$(cd "${RESULTS_PARENT_INPUT}" && pwd -P)"
RESULTS="${RESULTS_PARENT}/${RESULTS_NAME}"
if [ -L "${RESULTS}" ]; then
    echo "FAIL: campaign root may not be a symlink" >&2
    exit 2
fi
if [ -e "${RESULTS}" ] && [ ! -d "${RESULTS}" ]; then
    echo "FAIL: campaign root exists but is not a directory" >&2
    exit 2
fi
mkdir -p -- "${RESULTS}"
for directory in "${RESULTS}/rows" "${RESULTS}/failed_segments"; do
    if [ -L "${directory}" ] || { [ -e "${directory}" ] && [ ! -d "${directory}" ]; }; then
        echo "FAIL: unsafe campaign directory: ${directory}" >&2
        exit 2
    fi
    mkdir -p -- "${directory}"
done

CAMPAIGN_LOCK_HASH="$(printf '%s' "${RESULTS}" | sha256sum | awk '{print $1}')"
CAMPAIGN_LOCK="/tmp/cta_pdl_tier5_campaign_${CAMPAIGN_LOCK_HASH}.lock"
exec 8>"${CAMPAIGN_LOCK}"
if ! flock -n 8; then
    echo "FAIL: another runner holds campaign root lock ${CAMPAIGN_LOCK}" >&2
    exit 2
fi

CAMPAIGN_LOG="${RESULTS}/campaign_runner.log"
if [ -L "${CAMPAIGN_LOG}" ] || { [ -e "${CAMPAIGN_LOG}" ] && [ ! -f "${CAMPAIGN_LOG}" ]; }; then
    echo "FAIL: unsafe campaign log path" >&2
    exit 2
fi
campaign_log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${CAMPAIGN_LOG}"
}

CONTRACT="${RESULTS}/campaign_contract.json"
BINDING="${RESULTS}/campaign_binding.json"
for regular in "${CONTRACT}" "${BINDING}" "${RESULTS}/production_candidate.done.json"; do
    if [ -L "${regular}" ] || { [ -e "${regular}" ] && [ ! -f "${regular}" ]; }; then
        campaign_log "unsafe campaign file: ${regular}"
        exit 2
    fi
done
CONTRACT_ARGS=(
    --models "${MODELS:-deepseek_v32,glm5}"
    --seqs "${SEQS:-4096,32768,131072,1048576}"
    --workloads "${WORKLOADS:-operator_chain,single_layer,indexshare_fsss}"
    --warmup "${WARMUP:-5}"
    --repeats "${REPEATS:-31}"
    --seed "${SEED:-20260805}"
    --max-logits-mb "${MAX_LOGITS_MB:-16384}"
    --max-query-chunk "${MAX_QUERY_CHUNK:-4096}"
    --moe-experts 32 --moe-topk 8
    --moe-tokens "${MOE_TOKENS:-4096}"
    --backend flashinfer --required-device-substring B200
    --monitor-interval-ms "${MONITOR_INTERVAL_MS}"
    --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}"
)
if [ "${FAST}" = "1" ]; then
    CONTRACT_ARGS=(
        --models "${MODELS:-deepseek_v32}"
        --seqs "${SEQS:-4096}"
        --workloads "${WORKLOADS:-operator_chain}"
        --warmup "${WARMUP:-0}"
        --repeats "${REPEATS:-1}"
        --allow-short
        --seed "${SEED:-20260805}"
        --max-logits-mb "${MAX_LOGITS_MB:-16384}"
        --max-query-chunk "${MAX_QUERY_CHUNK:-4096}"
        --moe-experts 32 --moe-topk 8
        --moe-tokens "${MOE_TOKENS:-128}"
        --backend flashinfer --required-device-substring B200
        --monitor-interval-ms "${MONITOR_INTERVAL_MS}"
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}"
    )
fi
python3 ./production_tier5_campaign.py make-contract \
    --output "${CONTRACT}" "${CONTRACT_ARGS[@]}" | tee -a "${CAMPAIGN_LOG}" || exit 2

if [ -e "${RESULTS}/production_candidate.done.json" ]; then
    if [ ! -f "${BINDING}" ] || [ -L "${BINDING}" ]; then
        campaign_log "final marker exists without a safe campaign binding"
        exit 2
    fi
    python3 ./production_tier5_campaign.py check-final --root "${RESULTS}" \
        --contract "${CONTRACT}" --binding "${BINDING}" \
        | tee -a "${CAMPAIGN_LOG}" || exit 2
    echo "PRODUCTION_TIER5_FRAGMENT_CAMPAIGN_ALREADY_COMPLETE results=${RESULTS}"
    exit 0
fi

# A killed prior invocation leaves only an unsealed stage.  Quarantine it
# permanently and retry the missing canonical row; never treat it as resumable.
python3 - "${RESULTS}/rows" "${RESULTS}/failed_segments" <<'PY' || exit 2
import json, os, secrets, sys, tempfile
from pathlib import Path
rows, failed = map(Path, sys.argv[1:])
for stage in sorted(rows.iterdir()):
    if ".inprogress." not in stage.name:
        continue
    if stage.is_symlink() or not stage.is_dir():
        raise SystemExit(f"unsafe stale in-progress entry: {stage}")
    payload = {
        "schema": 1,
        "kind": "tier5_production_fragment_segment_rejection",
        "status": "REJECTED",
        "reason": "stale_unsealed_inprogress_recovered_on_resume",
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
    }
    output = stage / "segment_rejection.json"
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=stage)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output)
    destination = failed / f"{stage.name}.stale.{secrets.token_hex(8)}"
    os.rename(stage, destination)
# The rename mutates both directory entry sets.  Persist the source and destination
# directories so a power loss cannot resurrect a quarantined stage in ``rows``.
for parent in (rows, failed):
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(directory)
    os.close(directory)
PY

readarray -t ROW_SPECS < <(python3 - "${CONTRACT}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for ordinal, row in enumerate(value["ordered_matrix"]):
    print(f"{ordinal}\t{row['row_id']}")
PY
)
if [ "${#ROW_SPECS[@]}" -eq 0 ]; then
    campaign_log "contract has no rows"
    exit 2
fi

ACTIVE_STAGE=""
ACTIVE_INVOCATION=""
ACTIVE_ROW_ID=""
ACTIVE_ORDINAL=""
ACTIVE_CHILD_PID=""
ACTIVE_CHILD_START_TICKS=""
ACTIVE_CHILD_PGID=""
ACTIVE_CHILD_GROUP_VERIFIED=0
ACTIVE_MONITOR_PID=""
ACTIVE_MONITOR_START_TICKS=""
ACTIVE_READY_FILE=""
RUNNER_PGID="$(ps -o pgid= -p "$$" | tr -d '[:space:]')"

reject_stage() {
    local stage="$1" reason="$2" destination
    [ -d "${stage}" ] || return 0
    python3 - "${stage}" "${reason}" "${ACTIVE_ROW_ID}" \
        "${ACTIVE_ORDINAL}" "${ACTIVE_INVOCATION}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
root = Path(sys.argv[1])
artifacts = {}
for path in sorted(root.iterdir()):
    if path.is_file() and not path.is_symlink():
        artifacts[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
payload = {
    "schema": 1,
    "kind": "tier5_production_fragment_segment_rejection",
    "status": "REJECTED",
    "reason": sys.argv[2],
    "row_id": sys.argv[3],
    "ordinal": int(sys.argv[4]),
    "invocation_uuid": sys.argv[5],
    "accepted_timing": 0,
    "accepted_workload_timing": 0,
    "accepted_CTA_bracket": 0,
    "artifacts_before_rejection": artifacts,
}
output = root / "segment_rejection.json"
fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=root)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
os.fsync(directory)
os.close(directory)
PY
    destination="${RESULTS}/failed_segments/$(basename "${stage}").rejected.${ACTIVE_INVOCATION}"
    if [ -e "${destination}" ]; then
        campaign_log "rejection destination collision: ${destination}"
        return 2
    fi
    if ! mv -T -- "${stage}" "${destination}"; then
        campaign_log "failed to move rejected segment stage=${stage} destination=${destination}"
        return 2
    fi
    python3 - "$(dirname -- "${stage}")" "$(dirname -- "${destination}")" <<'PY' || {
import os
import sys
for raw in sys.argv[1:]:
    fd = os.open(raw, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
PY
        campaign_log "failed to persist rejected-segment directory rename"
        return 2
    }
    campaign_log "row=${ACTIVE_ROW_ID} segment=${ACTIVE_INVOCATION} permanently rejected reason=${reason} path=${destination}"
    ACTIVE_STAGE=""
}

cleanup_active() {
    local rc="$?"
    local cleanup_rc=0
    trap - EXIT
    trap '' INT TERM HUP
    terminate_active_processes || cleanup_rc=$?
    if [ -n "${ACTIVE_STAGE}" ] && [ -d "${ACTIVE_STAGE}" ]; then
        reject_stage "${ACTIVE_STAGE}" "runner_interrupted_or_unhandled_failure_rc_${rc}" || true
    fi
    if [ "${rc}" -eq 0 ] && [ "${cleanup_rc}" -ne 0 ]; then
        rc=2
    fi
    exit "${rc}"
}

proc_start_ticks() {
    local pid="$1" raw rest
    local -a fields=()
    case "${pid}" in ''|*[!0-9]*) return 1 ;; esac
    [ "${pid}" -gt 1 ] || return 1
    IFS= read -r raw < "/proc/${pid}/stat" || return 1
    rest="${raw##*) }"
    read -r -a fields <<< "${rest}"
    [ "${#fields[@]}" -gt 19 ] || return 1
    printf '%s\n' "${fields[19]}"
}

pid_identity_is_running() {
    local pid="$1" expected_start="$2" current_start state
    [ -n "${pid}" ] && [ -n "${expected_start}" ] || return 1
    current_start="$(proc_start_ticks "${pid}" 2>/dev/null)" || return 1
    [ "${current_start}" = "${expected_start}" ] || return 1
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    [ -n "${state}" ] && [ "${state#Z}" = "${state}" ]
}

active_child_group_is_safe() {
    case "${ACTIVE_CHILD_PGID}" in ''|*[!0-9]*) return 1 ;; esac
    [ "${ACTIVE_CHILD_GROUP_VERIFIED}" -eq 1 ] \
        && [ "${ACTIVE_CHILD_PGID}" -gt 1 ] \
        && [ "${ACTIVE_CHILD_PGID}" = "${ACTIVE_CHILD_PID}" ] \
        && [ "${ACTIVE_CHILD_PGID}" != "${RUNNER_PGID}" ]
}

active_child_is_running() {
    local line pgid state
    if active_child_group_is_safe; then
        while read -r pgid state; do
            [ "${pgid}" = "${ACTIVE_CHILD_PGID}" ] || continue
            [ -n "${state}" ] && [ "${state#Z}" = "${state}" ] && return 0
        done < <(ps -eo pgid=,stat= 2>/dev/null)
        return 1
    fi
    pid_identity_is_running "${ACTIVE_CHILD_PID}" "${ACTIVE_CHILD_START_TICKS}"
}

active_monitor_is_running() {
    pid_identity_is_running "${ACTIVE_MONITOR_PID}" "${ACTIVE_MONITOR_START_TICKS}"
}

clear_active_process_state() {
    local ready_file="${ACTIVE_READY_FILE}"
    ACTIVE_CHILD_PID=""
    ACTIVE_CHILD_START_TICKS=""
    ACTIVE_CHILD_PGID=""
    ACTIVE_CHILD_GROUP_VERIFIED=0
    ACTIVE_MONITOR_PID=""
    ACTIVE_MONITOR_START_TICKS=""
    ACTIVE_READY_FILE=""
    [ -z "${ready_file}" ] || rm -f -- "${ready_file}"
}

signal_active_processes() {
    local signal_name="$1"
    if active_child_group_is_safe; then
        kill -"${signal_name}" -- "-${ACTIVE_CHILD_PGID}" 2>/dev/null || true
        kill -CONT -- "-${ACTIVE_CHILD_PGID}" 2>/dev/null || true
    elif pid_identity_is_running \
        "${ACTIVE_CHILD_PID}" "${ACTIVE_CHILD_START_TICKS}"; then
        kill -"${signal_name}" "${ACTIVE_CHILD_PID}" 2>/dev/null || true
        kill -CONT "${ACTIVE_CHILD_PID}" 2>/dev/null || true
    fi
    if pid_identity_is_running \
        "${ACTIVE_MONITOR_PID}" "${ACTIVE_MONITOR_START_TICKS}"; then
        kill -"${signal_name}" "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    fi
}

terminate_active_processes() {
    local child_pid="${ACTIVE_CHILD_PID}" monitor_pid="${ACTIVE_MONITOR_PID}"
    local poll cleanup_failed=0
    if [ -z "${child_pid}" ] && [ -z "${monitor_pid}" ]; then
        clear_active_process_state
        return 0
    fi
    signal_active_processes TERM
    for ((poll = 0; poll < 100; poll++)); do
        if ! active_child_is_running && ! active_monitor_is_running; then
            break
        fi
        sleep 0.02
    done
    if active_child_is_running || active_monitor_is_running; then
        signal_active_processes KILL
        for ((poll = 0; poll < 100; poll++)); do
            if ! active_child_is_running && ! active_monitor_is_running; then
                break
            fi
            sleep 0.02
        done
    fi
    active_child_is_running && cleanup_failed=1
    active_monitor_is_running && cleanup_failed=1
    if [ -n "${child_pid}" ] && ! pid_identity_is_running \
        "${child_pid}" "${ACTIVE_CHILD_START_TICKS}"; then
        wait "${child_pid}" 2>/dev/null || true
    fi
    if [ -n "${monitor_pid}" ] && ! pid_identity_is_running \
        "${monitor_pid}" "${ACTIVE_MONITOR_START_TICKS}"; then
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    if [ "${cleanup_failed}" -ne 0 ]; then
        campaign_log "failed to reap active fragment process group=${ACTIVE_CHILD_PGID} monitor=${ACTIVE_MONITOR_PID}"
    fi
    clear_active_process_state
    [ "${cleanup_failed}" -eq 0 ]
}

trap cleanup_active EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
SELECTED_ROW_COUNT=0

run_monitored() {
    local stage="$1" lease="$2" gpu_monitor="$3" observations="$4"
    local resolved_index="$5" phase="$6"
    shift 6
    local child_pid monitor_pid command_rc=0 monitor_rc=0 ready_status="" ready_wait
    local observed_pgid=""
    local ready_file="${gpu_monitor}.ready"
    local -a command=("$@")
    if [ "${STEP_TIMEOUT}" != "0" ]; then
        command=(timeout --kill-after=30s "${STEP_TIMEOUT}" "${command[@]}")
    fi
    setsid bash -c 'kill -STOP "$$"; exec "$@"' production-fragment-gate \
        "${command[@]}" &
    ACTIVE_CHILD_PID=$!
    child_pid="${ACTIVE_CHILD_PID}"
    ACTIVE_CHILD_START_TICKS="$(proc_start_ticks "${child_pid}")" || {
        kill -TERM "${child_pid}" 2>/dev/null || true
        kill -CONT "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
        clear_active_process_state
        return 125
    }
    ACTIVE_CHILD_PGID="${child_pid}"
    ACTIVE_CHILD_GROUP_VERIFIED=0
    ACTIVE_READY_FILE="${ready_file}"
    python3 ./gpu_exclusivity.py monitor --lease "${lease}" \
        --json "${gpu_monitor}" --observations "${observations}" \
        --ready-file "${ready_file}" --nvidia-smi "${NVIDIA_SMI}" \
        --gpu-index "${resolved_index}" --watch-pid "${child_pid}" \
        --phase "${phase}" --interval-ms "${MONITOR_INTERVAL_MS}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --require-allowed-process --terminate-on-failure \
        >> "${stage}/runner.log" 2>&1 &
    ACTIVE_MONITOR_PID=$!
    monitor_pid="${ACTIVE_MONITOR_PID}"
    ACTIVE_MONITOR_START_TICKS="$(proc_start_ticks "${monitor_pid}")" || {
        terminate_active_processes || true
        return 125
    }
    for ((ready_wait = 0; ready_wait < 3000; ready_wait++)); do
        [ -s "${ready_file}" ] && break
        sleep 0.01
    done
    if [ -s "${ready_file}" ]; then
        ready_status="$(python3 - "${ready_file}" <<'PY'
import json, sys
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
        observed_pgid="$(ps -o pgid= -p "${child_pid}" 2>/dev/null | tr -d '[:space:]')"
        if [ "${observed_pgid}" = "${child_pid}" ]; then
            ACTIVE_CHILD_GROUP_VERIFIED=1
        else
            ready_status="INVALID_PROCESS_GROUP"
        fi
    fi
    if [ "${ready_status}" = "READY" ]; then
        kill -CONT "${child_pid}" 2>/dev/null || true
    else
        terminate_active_processes || true
        return 125
    fi
    while active_child_is_running && active_monitor_is_running; do
        sleep 0.02
    done
    if active_child_is_running && ! active_monitor_is_running; then
        wait "${monitor_pid}" || monitor_rc=$?
        terminate_active_processes || true
        return 124
    fi
    wait "${child_pid}" || command_rc=$?
    wait "${monitor_pid}" || monitor_rc=$?
    if active_child_is_running; then
        campaign_log "fragment command exited with a live descendant process group=${ACTIVE_CHILD_PGID}"
        terminate_active_processes || true
        [ "${command_rc}" -ne 0 ] || command_rc=123
    else
        clear_active_process_state
    fi
    [ "${ready_status}" = "READY" ] || return 125
    [ "${monitor_rc}" -eq 0 ] || return 124
    return "${command_rc}"
}

for row_spec in "${ROW_SPECS[@]}"; do
    IFS=$'\t' read -r ORDINAL ROW_ID <<< "${row_spec}"
    if [ -n "${FRAGMENT_ONLY_ROW}" ] && [ "${ROW_ID}" != "${FRAGMENT_ONLY_ROW}" ]; then
        continue
    fi
    SELECTED_ROW_COUNT=$((SELECTED_ROW_COUNT + 1))
    ROW_NAME="$(printf '%03d_%s' "${ORDINAL}" "${ROW_ID}")"
    ROW_FINAL="${RESULTS}/rows/${ROW_NAME}"
    if [ -L "${ROW_FINAL}" ] || { [ -e "${ROW_FINAL}" ] && [ ! -d "${ROW_FINAL}" ]; }; then
        campaign_log "unsafe completed row path: ${ROW_FINAL}"
        exit 2
    fi
    if [ -e "${ROW_FINAL}" ]; then
        if [ ! -f "${BINDING}" ]; then
            campaign_log "completed-looking row exists without campaign binding: ${ROW_FINAL}"
            exit 2
        fi
        python3 ./production_tier5_campaign.py check-fragment \
            --root "${ROW_FINAL}" --contract "${CONTRACT}" --binding "${BINDING}" \
            >> "${CAMPAIGN_LOG}" 2>&1 || {
                campaign_log "existing row is corrupt/stale and will not be overwritten: ${ROW_ID}"
                exit 2
            }
        campaign_log "resume verified and skipped sealed row=${ROW_ID} ordinal=${ORDINAL}"
        continue
    fi

    ACTIVE_ROW_ID="${ROW_ID}"
    ACTIVE_ORDINAL="${ORDINAL}"
    ACTIVE_INVOCATION="$(tr 'A-F' 'a-f' < /proc/sys/kernel/random/uuid)"
    ACTIVE_STAGE="$(mktemp -d "${RESULTS}/rows/${ROW_NAME}.inprogress.XXXXXX")"
    STAGE="${ACTIVE_STAGE}"
    : > "${STAGE}/runner.log"
    IDENTITY="${STAGE}/gpu_identity.json"
    LEASE="${STAGE}/gpu_exclusivity_lease.json"
    GPU_PRE="${STAGE}/gpu_pre.json"
    GPU_POST="${STAGE}/gpu_post.json"
    GPU_MONITOR="${STAGE}/gpu_monitor.json"
    GPU_OBSERVATIONS="${STAGE}/gpu_observations.ndjson"
    PHASE_BASE="production_tier5_fragment_${ORDINAL}_${ROW_ID}_${ACTIVE_INVOCATION}"

    python3 ./gpu_exclusivity.py identity --json "${IDENTITY}" \
        --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${GPU_INDEX}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" --owner-pid "$$" \
        --phase "${PHASE_BASE}_identity" >> "${STAGE}/runner.log" 2>&1 || {
            reject_stage "${STAGE}" "gpu_identity_failed"
            exit 2
        }
    python3 ./production_tier5_campaign.py bind-device \
        --contract "${CONTRACT}" --identity "${IDENTITY}" --output "${BINDING}" \
        >> "${STAGE}/runner.log" 2>&1 || {
            reject_stage "${STAGE}" "campaign_device_binding_failed"
            exit 2
        }
    GPU_BINDING="$(python3 - "${BINDING}" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
t=value["target_gpu"]
print(f"{t['uuid']}\t{t['index']}\t{value['campaign_fingerprint_sha256']}")
PY
)" || {
        reject_stage "${STAGE}" "campaign_device_binding_unreadable"
        exit 2
    }
    IFS=$'\t' read -r GPU_UUID RESOLVED_GPU_INDEX FINGERPRINT_SHA <<< "${GPU_BINDING}"
    CONTRACT_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["contract_sha256"])' "${CONTRACT}")" || {
        reject_stage "${STAGE}" "campaign_contract_unreadable"
        exit 2
    }
    export CUDA_VISIBLE_DEVICES="${RESOLVED_GPU_INDEX}"
    GPU_GLOBAL_LOCK="/tmp/cta_pdl_gpu_${GPU_UUID}.lock"
    exec {GPU_LOCK_FD}>"${GPU_GLOBAL_LOCK}"
    if ! flock -n "${GPU_LOCK_FD}"; then
        campaign_log "global target-GPU lock is busy: ${GPU_GLOBAL_LOCK}"
        reject_stage "${STAGE}" "target_gpu_lock_busy"
        exec {GPU_LOCK_FD}>&-
        exit 2
    fi

    python3 ./gpu_exclusivity.py acquire --json "${LEASE}" \
        --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${RESOLVED_GPU_INDEX}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" --owner-pid "$$" \
        --phase "${PHASE_BASE}_acquire" >> "${STAGE}/runner.log" 2>&1 || {
            reject_stage "${STAGE}" "gpu_lease_acquire_failed"
            exec {GPU_LOCK_FD}>&-
            exit 2
        }
    PRE_RC=0
    python3 ./gpu_exclusivity.py check --lease "${LEASE}" --json "${GPU_PRE}" \
        --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${RESOLVED_GPU_INDEX}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" --owner-pid "$$" \
        --phase "${PHASE_BASE}_pre" >> "${STAGE}/runner.log" 2>&1 || PRE_RC=$?

    HARNESS_RC=0
    if [ "${PRE_RC}" -eq 0 ]; then
        HARNESS_ARGS=(
            --output-dir "${STAGE}"
            --publish-target "${ROW_FINAL}"
            --runner-managed-stage
            --backend flashinfer
            --required-device-substring B200
            --models "${MODELS:-deepseek_v32,glm5}"
            --seqs "${SEQS:-4096,32768,131072,1048576}"
            --workloads "${WORKLOADS:-operator_chain,single_layer,indexshare_fsss}"
            --warmup "${WARMUP:-5}"
            --repeats "${REPEATS:-31}"
        )
        if [ "${FAST}" = "1" ]; then
            HARNESS_ARGS=(
                --output-dir "${STAGE}"
                --publish-target "${ROW_FINAL}"
                --runner-managed-stage
                --backend flashinfer
                --required-device-substring B200
                --models "${MODELS:-deepseek_v32}"
                --seqs "${SEQS:-4096}"
                --workloads "${WORKLOADS:-operator_chain}"
                --warmup "${WARMUP:-0}"
                --repeats "${REPEATS:-1}"
                --allow-short
            )
        fi
        HARNESS_ARGS+=(
            --seed "${SEED:-20260805}"
            --max-logits-mb "${MAX_LOGITS_MB:-16384}"
            --max-query-chunk "${MAX_QUERY_CHUNK:-4096}"
            --moe-experts 32 --moe-topk 8
            --moe-tokens "$( [ "${FAST}" = "1" ] && printf '%s' "${MOE_TOKENS:-128}" || printf '%s' "${MOE_TOKENS:-4096}" )"
            --execute-gpu
            --expected-gpu-uuid "${GPU_UUID}"
            --expected-gpu-index "${RESOLVED_GPU_INDEX}"
            --fragment-row-id "${ROW_ID}"
            --fragment-ordinal "${ORDINAL}"
            --campaign-contract-sha256 "${CONTRACT_SHA}"
            --campaign-fingerprint-sha256 "${FINGERPRINT_SHA}"
            --execution-segment-id "${ACTIVE_INVOCATION}"
        )
        run_monitored "${STAGE}" "${LEASE}" "${GPU_MONITOR}" \
            "${GPU_OBSERVATIONS}" "${RESOLVED_GPU_INDEX}" \
            "${PHASE_BASE}_monitor" python3 ./production_tier5.py \
            "${HARNESS_ARGS[@]}" > "${STAGE}/harness.log" 2>&1 || HARNESS_RC=$?
    else
        : > "${STAGE}/harness.log"
        HARNESS_RC=126
    fi

    POST_RC=0
    python3 ./gpu_exclusivity.py check --lease "${LEASE}" --json "${GPU_POST}" \
        --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${RESOLVED_GPU_INDEX}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" --owner-pid "$$" \
        --phase "${PHASE_BASE}_post" >> "${STAGE}/runner.log" 2>&1 || POST_RC=$?
    exec {GPU_LOCK_FD}>&-
    if [ "${PRE_RC}" -ne 0 ] || [ "${HARNESS_RC}" -ne 0 ] || [ "${POST_RC}" -ne 0 ]; then
        reject_stage "${STAGE}" "pre_${PRE_RC}_harness_or_monitor_${HARNESS_RC}_post_${POST_RC}"
        exit 2
    fi

    # Do not redirect this command into runner.log: the marker hashes runner.log.
    python3 ./production_tier5_campaign.py validate-fragment \
        --root "${STAGE}" --contract "${CONTRACT}" --binding "${BINDING}" || {
            reject_stage "${STAGE}" "fragment_semantic_validation_failed"
            exit 2
        }
    python3 ./production_tier5_campaign.py publish-fragment \
        --stage "${STAGE}" --final "${ROW_FINAL}" || {
            reject_stage "${STAGE}" "fragment_no_clobber_publish_failed"
            exit 2
        }
    ACTIVE_STAGE=""
    campaign_log "sealed row=${ROW_ID} ordinal=${ORDINAL} invocation=${ACTIVE_INVOCATION}"
done

if [ -n "${FRAGMENT_ONLY_ROW}" ]; then
    if [ "${SELECTED_ROW_COUNT}" -ne 1 ]; then
        campaign_log "fragment-only row is not an exact canonical contract row: ${FRAGMENT_ONLY_ROW}"
        exit 2
    fi
    python3 - "${RESULTS}" "${CONTRACT}" "${BINDING}" "${FRAGMENT_ONLY_ROW}" <<'PY' || exit 2
import hashlib, json, os, sys, tempfile
from pathlib import Path
root, contract_path, binding_path = map(Path, sys.argv[1:4])
row_id = sys.argv[4]
contract = json.load(open(contract_path, encoding="utf-8"))
binding = json.load(open(binding_path, encoding="utf-8"))
ordinal = next(i for i, row in enumerate(contract["ordered_matrix"]) if row["row_id"] == row_id)
fragment = root / "rows" / f"{ordinal:03d}_{row_id}" / "fragment.done.json"
payload = {
    "schema": 1,
    "kind": "tier5_production_nonformal_fragment_only_marker",
    "status": "PASS",
    "campaign_mode": "nonformal_short_profiler_input_only",
    "formal": False,
    "accepted_timing": 0,
    "accepted_workload_timing": 0,
    "accepted_CTA_bracket": 0,
    "row_id": row_id,
    "ordinal": ordinal,
    "campaign_contract_sha256": contract["contract_sha256"],
    "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
    "fragment_marker_sha256": hashlib.sha256(fragment.read_bytes()).hexdigest(),
}
output = root / "nonformal_fragment_only.done.json"
fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=root)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, output)
directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY); os.fsync(directory); os.close(directory)
PY
    trap - EXIT INT TERM HUP
    campaign_log "nonformal fragment-only campaign complete row=${FRAGMENT_ONLY_ROW} accepted_workload_timing=0"
    exit 0
fi

python3 ./production_tier5_campaign.py finalize --root "${RESULTS}" \
    --contract "${CONTRACT}" --binding "${BINDING}" | tee -a "${CAMPAIGN_LOG}" || exit 2
python3 ./production_tier5_campaign.py check-final --root "${RESULTS}" \
    --contract "${CONTRACT}" --binding "${BINDING}" \
    | tee -a "${CAMPAIGN_LOG}" || exit 2
trap - EXIT INT TERM HUP
campaign_log "campaign complete rows=${#ROW_SPECS[@]} fresh_check_final=PASS"
echo "PRODUCTION_TIER5_FRAGMENT_CAMPAIGN_PUBLISHED results=${RESULTS}"
