#!/usr/bin/env bash
# run_dsa_chain.sh -- resumable, fail-closed native Tier-5 DSA campaign.
#
# Formal mode is exactly 4K/32K/128K/1M with PROFILE=1.  4K/32K retain an exact
# tile-to-CTA dependency mapping; 128K/1M are explicitly labelled work-complete packed
# feasibility proxies.  FAST=1 is exactly one 4K, 3-repeat semantic smoke.
# The admissible claim is only a synthetic work-complete dependency proxy (survey section 9
# remains PARTIAL).  Nsight is an independent 4K mapping sidecar, never a timing sample.
#
# Usage:
#   ./run_dsa_chain.sh                 # exact four-point formal matrix + profiler proof
#   FAST=1 PROFILE=1 ./run_dsa_chain.sh # profiled 4K smoke
#   ./run_dsa_chain.sh --fresh         # archive prior derived attempt and rerun

set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${SELF}"
    exit 0
fi

RESULTS="${RESULTS:-results_dsa_native}"
FAST="${FAST:-0}"
PROFILE="${PROFILE:-$([ "${FAST}" = "1" ] && echo 0 || echo 1)}"
STEP_TIMEOUT="${STEP_TIMEOUT:-auto}"
ACTIVE_STEP_TIMEOUT=""
NVIDIA_SMI="${DSA_NVIDIA_SMI:-nvidia-smi}"
GPU_INDEX="${DSA_GPU_INDEX:-0}"
MONITOR_INTERVAL_MS="${DSA_MONITOR_INTERVAL_MS:-50}"
NVIDIA_SMI_TIMEOUT_MS="${DSA_NVIDIA_SMI_TIMEOUT_MS:-2000}"
mkdir -p "${RESULTS}"

# A rejected directory is permanently inadmissible.  This check intentionally precedes
# --fresh handling, marker replay, compilation, and every GPU action.
if [ -e "${RESULTS}/formal_rejection.json" ] || [ -e "${RESULTS}/REJECTED.md" ]; then
    echo "FAIL: RESULTS=${RESULTS} contains formal rejection evidence; use a new directory" >&2
    exit 2
fi

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/dsa.log"; }
warn() { echo "[$(date +%H:%M:%S)] WARN: $*" | tee -a "${RESULTS}/dsa.log"; }
fail() { echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "${RESULTS}/dsa.log" \
                                              | tee -a "${RESULTS}/failures.log"; }
sha()  { sha256sum -- "$1" | awk '{print $1}'; }
sha_text() { printf '%s' "$1" | sha256sum | awk '{print $1}'; }

archive_file() {
    local path="$1" n=1
    [ -e "${path}" ] || return 0
    while [ -e "${path}.retry${n}" ]; do n=$((n + 1)); done
    mv "${path}" "${path}.retry${n}"
}

# A prior failure ledger must never be mistaken for a failure in this invocation.
archive_file "${RESULTS}/failures.log"
archive_file "${RESULTS}/campaign_admission.json"
archive_file "${RESULTS}/terminal_status.json"

if [ "${1:-}" = "--fresh" ]; then
    for path in "${RESULTS}"/*.done "${RESULTS}"/*.invalid \
                "${RESULTS}"/dsa_matrix.log "${RESULTS}"/validation_matrix.json \
                "${RESULTS}"/campaign_admission.json \
                "${RESULTS}"/terminal_status.json \
                "${RESULTS}"/gpu_exclusivity_lease.json \
                "${RESULTS}"/gpu_exclusivity_preflight.json \
                "${RESULTS}"/dsa_build_manifest.json \
                "${RESULTS}"/dsa.log; do
        [ -e "${path}" ] && archive_file "${path}"
    done
    shift
fi
if [ "$#" -ne 0 ]; then
    fail "unknown arguments: $*"
    exit 2
fi
case "${FAST}" in
    0|1) ;;
    *) fail "FAST must be exactly 0 or 1"; exit 2 ;;
esac
case "${PROFILE}" in
    0|1) ;;
    *) fail "PROFILE must be exactly 0 or 1"; exit 2 ;;
esac
if [ "${STEP_TIMEOUT}" != "auto" ] \
   && [[ ! "${STEP_TIMEOUT}" =~ ^[1-9][0-9]*(s|m|h)?$ ]]; then
    fail "STEP_TIMEOUT must be auto or a positive integer with optional s/m/h suffix"
    exit 2
fi

step_timeout_for_seq() {
    local seq="$1"
    if [ "${STEP_TIMEOUT}" != "auto" ]; then
        printf '%s\n' "${STEP_TIMEOUT}"
        return
    fi
    case "${seq}" in
        4096) printf '%s\n' 120s ;;
        32768) printf '%s\n' 600s ;;
        131072) printf '%s\n' 1800s ;;
        1048576) printf '%s\n' 7200s ;;
        *) return 2 ;;
    esac
}

if [ "${FAST}" = "1" ]; then
    DEFAULT_SEQS="4096"
    REPEATS=3
    WARMUP=1
    SHORT_ARGS=(--allow-short)
else
    DEFAULT_SEQS="4096 32768 131072 1048576"
    REPEATS=31
    WARMUP=3
    SHORT_ARGS=()
    if [ "${PROFILE}" != "1" ]; then
        fail "formal FAST=0 requires PROFILE=1"
        exit 2
    fi
fi
PAIR_ARGS=()
read -ra REQUESTED_SEQS <<< "${DSA_SEQS:-${DEFAULT_SEQS}}"
declare -A DSA_SEEN=()
for seq in "${REQUESTED_SEQS[@]}"; do
    case "${seq}" in
        4096|32768|131072|1048576) ;;
        *) fail "unsupported Tier-5 context ${seq}"; exit 2 ;;
    esac
    if [ -n "${DSA_SEEN[${seq}]+present}" ]; then
        fail "duplicate Tier-5 context ${seq}"
        exit 2
    fi
    DSA_SEEN[${seq}]=1
done
if [ "${FAST}" = "1" ]; then
    if [ "${#REQUESTED_SEQS[@]}" -ne 1 ] || [ -z "${DSA_SEEN[4096]+present}" ]; then
        fail "FAST=1 requires exactly one context: 4096"
        exit 2
    fi
    SEQS=(4096)
else
    if [ "${#REQUESTED_SEQS[@]}" -ne 4 ]; then
        fail "formal FAST=0 requires exactly four unique contexts"
        exit 2
    fi
    for seq in 4096 32768 131072 1048576; do
        if [ -z "${DSA_SEEN[${seq}]+present}" ]; then
            fail "formal FAST=0 is missing required context ${seq}"
            exit 2
        fi
    done
    SEQS=(4096 32768 131072 1048576)
fi

case "${GPU_INDEX}" in
    ''|*[!0-9]*) fail "DSA_GPU_INDEX must be a non-negative integer"; exit 2 ;;
esac
case "${MONITOR_INTERVAL_MS}" in
    ''|*[!0-9]*) fail "DSA_MONITOR_INTERVAL_MS must be an integer in 10..100"; exit 2 ;;
esac
if [ "${MONITOR_INTERVAL_MS}" -lt 10 ] || [ "${MONITOR_INTERVAL_MS}" -gt 100 ]; then
    fail "DSA_MONITOR_INTERVAL_MS must be bounded to 10..100ms"
    exit 2
fi
case "${NVIDIA_SMI_TIMEOUT_MS}" in
    ''|*[!0-9]*) fail "DSA_NVIDIA_SMI_TIMEOUT_MS must be an integer in 100..5000"; exit 2 ;;
esac
if [ "${NVIDIA_SMI_TIMEOUT_MS}" -lt 100 ] || [ "${NVIDIA_SMI_TIMEOUT_MS}" -gt 5000 ]; then
    fail "DSA_NVIDIA_SMI_TIMEOUT_MS must be bounded to 100..5000ms"
    exit 2
fi
printf -v NVIDIA_SMI_TIMEOUT_SPEC '%d.%03ds' \
    "$((NVIDIA_SMI_TIMEOUT_MS / 1000))" "$((NVIDIA_SMI_TIMEOUT_MS % 1000))"
write_exclusivity_rejection() {
    local phase="$1" evidence="$2" marker
    for marker in "${RESULTS}"/*.done; do
        [ -e "${marker}" ] && archive_file "${marker}"
    done
    python3 - "${RESULTS}/formal_rejection.json" "${phase}" "${evidence}" <<'PY'
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
    "status": "REJECTED",
    "reason": "gpu_exclusivity_unproven_or_contaminated",
    "phase": phase,
    "accepted_timing": 0,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "evidence_path": str(evidence_path),
    "evidence_sha256": evidence_sha,
    "evidence": evidence,
}

output.parent.mkdir(parents=True, exist_ok=True)
fd, temp = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, output)
finally:
    try: os.unlink(temp)
    except FileNotFoundError: pass
PY
}

write_step_rejection() {
    local phase="$1" rc="$2" log_path="$3" timeout_spec="$4" kind="${5:-native}"
    python3 - "${RESULTS}/formal_rejection.json" "${phase}" "${rc}" \
        "${log_path}" "${timeout_spec}" "${kind}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
phase = sys.argv[2]
returncode = int(sys.argv[3])
log_path = Path(sys.argv[4])
timeout_spec = sys.argv[5]
kind = sys.argv[6]
try:
    log_sha256 = hashlib.sha256(log_path.read_bytes()).hexdigest()
except OSError:
    log_sha256 = None
payload = {
    "schema": 1,
    "status": "REJECTED",
    "reason": (
        "native_step_timeout" if kind == "native" and returncode in (124, 137)
        else "native_step_failed" if kind == "native"
        else "strict_validator_failed"
    ),
    "phase": phase,
    "failure_kind": kind,
    "native_returncode": returncode,
    "step_timeout_spec": timeout_spec,
    "accepted_timing": 0,
    "log_path": str(log_path),
    "log_sha256": log_sha256,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
output.parent.mkdir(parents=True, exist_ok=True)
fd, temp = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, output)
finally:
    try: os.unlink(temp)
    except FileNotFoundError: pass
PY
}

EXCLUSIVITY_LEASE="${RESULTS}/gpu_exclusivity_lease.json"
EXCLUSIVITY_PREFLIGHT="${RESULTS}/gpu_exclusivity_preflight.json"
GPU_IDENTITY_CURRENT="${RESULTS}/gpu_identity_current.json"
if ! command -v flock >/dev/null 2>&1; then
    fail "flock unavailable; cannot establish campaign lease"
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
    fail "setsid unavailable; cannot isolate monitored command process groups"
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    fail "timeout unavailable; cannot bound GPU identity/process queries"
    exit 2
fi
exec 8>"${RESULTS}/.campaign.lock"
if ! flock -n 8; then
    fail "another campaign holds this RESULTS directory lock"
    exit 2
fi
python3 ./gpu_exclusivity.py identity --json "${GPU_IDENTITY_CURRENT}" \
    --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${GPU_INDEX}" \
    --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
    --owner-pid "$$" --phase global_lock_identity \
    >> "${RESULTS}/dsa.log" 2>&1 || {
        write_exclusivity_rejection global_lock_identity "${GPU_IDENTITY_CURRENT}"
        fail "target GPU identity query failed; accepted_timing=0"
        exit 2
    }
GPU_UUID=$(python3 - "${GPU_IDENTITY_CURRENT}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
target = value.get("target_gpu") or {}
if value.get("status") != "PASS" or value.get("errors") != [] or not target.get("uuid"):
    raise SystemExit(2)
print(target["uuid"])
PY
) || {
    write_exclusivity_rejection malformed_gpu_identity "${GPU_IDENTITY_CURRENT}"
    fail "target GPU identity JSON failed closed admission; accepted_timing=0"
    exit 2
}
if [[ ! "${GPU_UUID}" =~ ^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
    write_exclusivity_rejection unsafe_gpu_uuid "${GPU_IDENTITY_CURRENT}"
    fail "target GPU UUID is unsafe for an explicit lock path; accepted_timing=0"
    exit 2
fi
# UUID visibility is invariant to CUDA ordinal reordering.  The native process must then
# observe this one visible physical GPU as runtime ordinal zero and report the same UUID.
export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
GPU_GLOBAL_LOCK="/tmp/cta_pdl_gpu_${GPU_UUID}.lock"
GPU_GLOBAL_LOCK_KEY_SHA256=$(sha_text "${GPU_UUID}")
GPU_GLOBAL_LOCK_PATH_SHA256=$(sha_text "${GPU_GLOBAL_LOCK}")
exec 9>"${GPU_GLOBAL_LOCK}"
if ! flock -n 9; then
    fail "another campaign holds the global target-GPU lock ${GPU_GLOBAL_LOCK}"
    exit 2
fi
if [ -e "${EXCLUSIVITY_LEASE}" ]; then
    python3 ./gpu_exclusivity.py check --lease "${EXCLUSIVITY_LEASE}" \
        --json "${EXCLUSIVITY_PREFLIGHT}" --nvidia-smi "${NVIDIA_SMI}" \
        --gpu-index "${GPU_INDEX}" --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --owner-pid "$$" \
        --phase campaign_resume_preflight >> "${RESULTS}/dsa.log" 2>&1 || {
            write_exclusivity_rejection campaign_resume_preflight "${EXCLUSIVITY_PREFLIGHT}"
            fail "GPU exclusivity resume preflight failed; accepted_timing=0"
            exit 2
        }
else
    python3 ./gpu_exclusivity.py acquire --json "${EXCLUSIVITY_LEASE}" \
        --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${GPU_INDEX}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --owner-pid "$$" --phase campaign_acquire \
        >> "${RESULTS}/dsa.log" 2>&1 || {
            write_exclusivity_rejection campaign_acquire "${EXCLUSIVITY_LEASE}"
            fail "GPU exclusivity acquisition failed; accepted_timing=0"
            exit 2
        }
    cp -- "${EXCLUSIVITY_LEASE}" "${EXCLUSIVITY_PREFLIGHT}"
fi
EXCLUSIVITY_LEASE_ID=$(python3 - "${EXCLUSIVITY_LEASE}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "PASS" or value.get("errors") != [] or not value.get("lease_id"):
    raise SystemExit(2)
print(value["lease_id"])
PY
) || {
    write_exclusivity_rejection malformed_lease "${EXCLUSIVITY_LEASE}"
    fail "GPU exclusivity lease JSON failed closed admission; accepted_timing=0"
    exit 2
}
EXCLUSIVITY_LEASE_SHA256=$(sha "${EXCLUSIVITY_LEASE}")

run_monitored() {
    local phase="$1" monitor_json="$2" observations="$3" require_allowed="$4"
    shift 4
    local command_rc=0 monitor_rc=0 child_pid monitor_pid ready_status=""
    local ready_file="${monitor_json}.ready" ready_wait
    local -a command=("$@") monitor_args=()
    if [ -z "${ACTIVE_STEP_TIMEOUT}" ]; then
        fail "${phase} has no positive ACTIVE_STEP_TIMEOUT"
        return 2
    fi
    command=(timeout --foreground --kill-after=30s "${ACTIVE_STEP_TIMEOUT}" "${command[@]}")
    if [ "${require_allowed}" = "1" ]; then
        monitor_args+=(--require-allowed-process)
    fi
    archive_file "${ready_file}"
    # The command cannot execute before the monitor has durably recorded a clean baseline.
    # This SIGSTOP/READY/SIGCONT gate eliminates the prior launch-before-monitor window.
    setsid bash -c 'kill -STOP "$$"; exec "$@"' dsa-monitor-gate \
        "${command[@]}" &
    child_pid=$!
    python3 ./gpu_exclusivity.py monitor --lease "${EXCLUSIVITY_LEASE}" \
        --json "${monitor_json}" --observations "${observations}" \
        --ready-file "${ready_file}" \
        --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${GPU_INDEX}" \
        --watch-pid "${child_pid}" --phase "${phase}" \
        --interval-ms "${MONITOR_INTERVAL_MS}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" --terminate-on-failure \
        "${monitor_args[@]}" >> "${RESULTS}/dsa.log" 2>&1 &
    monitor_pid=$!

    # Each nvidia-smi query is bounded; 30 seconds covers identity, preferred/fallback
    # process-name queries, and scheduler jitter without ever releasing an unproven child.
    for ((ready_wait = 0; ready_wait < 3000; ready_wait++)); do
        [ -s "${ready_file}" ] && break
        sleep 0.01
    done
    if [ -s "${ready_file}" ]; then
        ready_status=$(python3 - "${ready_file}" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(2)
if value.get("schema") != 1 or value.get("kind") != "gpu_exclusivity_monitor_ready":
    raise SystemExit(2)
print(value.get("status", ""))
PY
        ) || ready_status="INVALID"
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
        write_exclusivity_rejection "${phase}_monitor_start_gate" "${monitor_json}"
        fail "${phase} monitor start gate failed status=${ready_status}; accepted_timing=0"
        exit 2
    fi
    if [ "${monitor_rc}" -ne 0 ]; then
        write_exclusivity_rejection "${phase}_monitor" "${monitor_json}"
        fail "${phase} runtime GPU monitor failed rc=${monitor_rc}; accepted_timing=0"
        exit 2
    fi
    return "${command_rc}"
}

DEVICE_FINGERPRINT=$(timeout --kill-after=1s "${NVIDIA_SMI_TIMEOUT_SPEC}" \
    "${NVIDIA_SMI}" --id="${GPU_INDEX}" \
    --query-gpu=uuid,name,compute_cap,driver_version \
    --format=csv,noheader,nounits 2>/dev/null | head -1)
DEVICE_FINGERPRINT="${DEVICE_FINGERPRINT:-unavailable}"
printf '%s\n' "${DEVICE_FINGERPRINT}" > "${RESULTS}/device_fingerprint.txt"
DEVICE_CC=$(timeout --kill-after=1s "${NVIDIA_SMI_TIMEOUT_SPEC}" \
    "${NVIDIA_SMI}" --id="${GPU_INDEX}" \
    --query-gpu=compute_cap --format=csv,noheader,nounits \
    2>/dev/null | head -1 | tr -d '[:space:]')
if [[ "${DEVICE_CC}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    TARGET_ARCH="sm_${DEVICE_CC/./}"
else
    TARGET_ARCH="${ARCH:-sm_100}"
fi

BUILD_MANIFEST="${RESULTS}/dsa_build_manifest.json"
BUILD_LOG="${RESULTS}/build.log"
BUILD_PROVENANCE_ARGS=(
    --json "${BUILD_MANIFEST}" --binary ./dsa_native --source ./dsa_native.cu
    --bench-util ../common/bench_util.cuh --cta-trace ../common/cta_trace.cuh
    --build-script ../build.sh --build-log "${BUILD_LOG}" --target "${TARGET_ARCH}"
)
rebuild_reason=""
if [ ! -x ./dsa_native ]; then
    rebuild_reason="binary missing"
else
    for dependency in dsa_native.cu ../common/bench_util.cuh ../common/cta_trace.cuh ../build.sh; do
        if [ "${dependency}" -nt ./dsa_native ]; then
            rebuild_reason="${dependency} newer than binary"
            break
        fi
    done
    if [ -z "${rebuild_reason}" ]; then
        image_list=$(cuobjdump --list-elf ./dsa_native 2>/dev/null || true)
        if ! grep -Fq ".${TARGET_ARCH}.cubin" <<< "${image_list}"; then
            rebuild_reason="missing ${TARGET_ARCH} cubin"
        fi
    fi
    if [ -z "${rebuild_reason}" ]; then
        if ! python3 ./build_provenance.py verify "${BUILD_PROVENANCE_ARGS[@]}" \
            >> "${RESULTS}/dsa.log" 2>&1; then
            rebuild_reason="missing/stale source-to-binary build provenance"
        fi
    fi
fi
if [ -n "${rebuild_reason}" ]; then
    archive_file "${BUILD_LOG}"
    archive_file "${BUILD_MANIFEST}"
    NVCC_PATH=$(command -v "${NVCC:-nvcc}" 2>/dev/null || true)
    if [ -z "${NVCC_PATH}" ]; then
        fail "nvcc unavailable; cannot create source-to-binary provenance"
        exit 2
    fi
    RESOLVED_NVTX_INCLUDE="${NVTX_INCLUDE:-}"
    if [ -z "${RESOLVED_NVTX_INCLUDE}" ]; then
        for candidate in /usr/local/cuda/include \
            /opt/nvidia/nsight-systems/*/target-linux-x64/nvtx/include \
            /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; do
            if [ -d "${candidate}/nvtx3" ]; then
                RESOLVED_NVTX_INCLUDE="${candidate}"
                break
            fi
        done
    fi
    log "rebuild native binary for ${TARGET_ARCH}: ${rebuild_reason}"
    ARCH="${TARGET_ARCH}" NVCC="${NVCC_PATH}" NVTX_INCLUDE="${RESOLVED_NVTX_INCLUDE}" \
        ../build.sh >> "${BUILD_LOG}" 2>&1 || {
        fail "build failed (see ${BUILD_LOG})"
        exit 2
    }
    python3 ./build_provenance.py create "${BUILD_PROVENANCE_ARGS[@]}" \
        --nvcc "${NVCC_PATH}" --nvtx-include "${RESOLVED_NVTX_INCLUDE}" \
        >> "${RESULTS}/dsa.log" 2>&1 || {
            fail "source-to-binary build provenance creation failed"
            exit 2
        }
fi
python3 ./build_provenance.py verify "${BUILD_PROVENANCE_ARGS[@]}" \
    >> "${RESULTS}/dsa.log" 2>&1 || {
        fail "source-to-binary build provenance verification failed"
        exit 2
    }
image_list=$(cuobjdump --list-elf ./dsa_native 2>/dev/null || true)
if ! grep -Fq ".${TARGET_ARCH}.cubin" <<< "${image_list}"; then
    fail "binary admission failed: ${TARGET_ARCH} cubin absent after rebuild"
    exit 2
fi
timeout --kill-after=1s "${NVIDIA_SMI_TIMEOUT_SPEC}" \
    "${NVIDIA_SMI}" --id="${GPU_INDEX}" \
    --query-gpu=uuid,name,memory.total,compute_cap,driver_version \
    --format=csv > "${RESULTS}/device.txt" 2>/dev/null || true

# PTX is the semantic authority for PDL/acquire/release placement.  nvdisasm separately proves
# that the real target cubin is decodable and contains machine code for all three workers.
python3 ./verify_dsa_binary.py ./dsa_native \
    --ptx "${RESULTS}/dsa_native.ptx" \
    --resources "${RESULTS}/dsa_native_resources.txt" \
    --sass "${RESULTS}/dsa_native_${TARGET_ARCH}.sass" \
    --target "${TARGET_ARCH}" \
    --json "${RESULTS}/dsa_binary_proof.json" \
    > "${RESULTS}/dsa_binary_proof.log" 2>&1 || {
        fail "native PTX/resource/cubin proof failed"
        exit 2
    }
NVTX_RANGE_COUNT=$(python3 - "${RESULTS}/dsa_binary_proof.json" <<'PY'
import json
import sys
proof = json.load(open(sys.argv[1], encoding="utf-8"))
if proof.get("status") != "PASS" or proof.get("errors") != []:
    raise SystemExit(2)
print(sum(int(value) for value in proof.get("nvtx_ranges", {}).values()))
PY
) || {
    fail "binary proof JSON failed closed admission"
    exit 2
}
if [ "${NVTX_RANGE_COUNT}" -ne 9 ]; then
    fail "binary admission failed: NVTX range count=${NVTX_RANGE_COUNT}, expected=9"
    exit 2
fi

base_signature() {
    local seq="$1" tag="$2" argv_text
    argv_text=$(printf '%q ' ./dsa_native --seq "${seq}" --repeats "${REPEATS}" \
        --warmup "${WARMUP}" --tag "${tag}" \
        --trace "${RESULTS}/${tag}_trace.csv" "${SHORT_ARGS[@]}" "${PAIR_ARGS[@]}")
    printf '%s\n' \
        "marker_schema=2" \
        "fast=${FAST}" \
        "profile=${PROFILE}" \
        "tag=${tag}" \
        "seq=${seq}" \
        "mode_count=4" \
        "mode_order=floor,wave_floor,impl,ceiling" \
        "sample_order=cyclic_latin_4" \
        "invocations_per_point=$((4 + 4 * WARMUP + 4 * REPEATS))" \
        "step_timeout_spec=$(step_timeout_for_seq "${seq}")" \
        "gpu_index=${GPU_INDEX}" \
        "gpu_physical_uuid=${GPU_UUID}" \
        "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}" \
        "device_fingerprint_sha256=$(sha "${RESULTS}/device_fingerprint.txt")" \
        "target_arch=${TARGET_ARCH}" \
        "binary_sha256=$(sha ./dsa_native)" \
        "source_sha256=$(sha ./dsa_native.cu)" \
        "bench_util_sha256=$(sha ../common/bench_util.cuh)" \
        "cta_trace_sha256=$(sha ../common/cta_trace.cuh)" \
        "build_sha256=$(sha ../build.sh)" \
        "build_provenance_helper_sha256=$(sha ./build_provenance.py)" \
        "build_manifest_sha256=$(sha "${BUILD_MANIFEST}")" \
        "build_log_sha256=$(sha "${BUILD_LOG}")" \
        "validator_sha256=$(sha ./validate_dsa_native.py)" \
        "binary_verifier_sha256=$(sha ./verify_dsa_binary.py)" \
        "binary_proof_sha256=$(sha "${RESULTS}/dsa_binary_proof.json")" \
        "nvtx_range_count=${NVTX_RANGE_COUNT}" \
        "gpu_exclusivity_lease_id=${EXCLUSIVITY_LEASE_ID}" \
        "gpu_exclusivity_lease_sha256=${EXCLUSIVITY_LEASE_SHA256}" \
        "gpu_global_lock_scope=target_uuid" \
        "gpu_global_lock_key_sha256=${GPU_GLOBAL_LOCK_KEY_SHA256}" \
        "gpu_global_lock_path_sha256=${GPU_GLOBAL_LOCK_PATH_SHA256}" \
        "gpu_monitor_interval_ms=${MONITOR_INTERVAL_MS}" \
        "gpu_query_timeout_ms=${NVIDIA_SMI_TIMEOUT_MS}" \
        "gpu_monitor_coverage_model=bounded_interval_nvidia_smi_process_sampling" \
        "gpu_exclusivity_helper_sha256=$(sha ./gpu_exclusivity.py)" \
        "aggregator_sha256=$(sha ./aggregate_dsa_native.py)" \
        "profile_validator_sha256=$(sha ./validate_dsa_profile.py)" \
        "profiler_evidence_sha256=$(sha ./profiler_evidence.py)" \
        "campaign_finalizer_sha256=$(sha ./finalize_dsa_campaign.py)" \
        "runner_sha256=$(sha "${SELF}")" \
        "argv_sha256=$(sha_text "${argv_text}")"
}

validation_runtime_uuid() {
    python3 - "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
device = value.get("device") or {}
if value.get("status") != "PASS" or value.get("errors") != []:
    raise SystemExit(2)
if device.get("runtime_ordinal") != 0 or device.get("runtime_ordinal_zero") is not True:
    raise SystemExit(2)
uuid = device.get("runtime_uuid")
if not isinstance(uuid, str) or not uuid:
    raise SystemExit(2)
print(uuid)
PY
}

marker_content() {
    local seq="$1" tag="$2"
    base_signature "${seq}" "${tag}"
    printf '%s\n' \
        "native_runtime_uuid=$(validation_runtime_uuid "${RESULTS}/${tag}_validation.json")" \
        "log_sha256=$(sha "${RESULTS}/${tag}.log")" \
        "trace_sha256=$(sha "${RESULTS}/${tag}_trace.csv")" \
        "validation_sha256=$(sha "${RESULTS}/${tag}_validation.json")" \
        "gpu_pre_sha256=$(sha "${RESULTS}/${tag}_gpu_pre.json")" \
        "gpu_post_sha256=$(sha "${RESULTS}/${tag}_gpu_post.json")" \
        "gpu_monitor_sha256=$(sha "${RESULTS}/${tag}_gpu_monitor.json")" \
        "gpu_observations_sha256=$(sha "${RESULTS}/${tag}_gpu_observations.ndjson")"
}

run_point() {
    local seq="$1" mapping tag marker invalid expected resume_json resume_log
    local gpu_pre gpu_post gpu_monitor gpu_observations
    local rc=0 validator_rc=0 resume_rc=0
    if [ "${seq}" -le 32768 ]; then
        mapping=exact
    else
        mapping=work_complete_packed_proxy
    fi
    tag="dsa_${mapping}_seq${seq}"
    marker="${RESULTS}/${tag}.done"
    invalid="${RESULTS}/${tag}.invalid"
    resume_json="${RESULTS}/${tag}_resume_validation.json"
    resume_log="${RESULTS}/${tag}.resume_validator.log"
    gpu_pre="${RESULTS}/${tag}_gpu_pre.json"
    gpu_post="${RESULTS}/${tag}_gpu_post.json"
    gpu_monitor="${RESULTS}/${tag}_gpu_monitor.json"
    gpu_observations="${RESULTS}/${tag}_gpu_observations.ndjson"
    ACTIVE_STEP_TIMEOUT=$(step_timeout_for_seq "${seq}") || {
        fail "${tag} has no valid step timeout"
        exit 2
    }

    if [ -f "${marker}" ] \
       && [ -f "${RESULTS}/${tag}.log" ] \
       && [ -f "${RESULTS}/${tag}_trace.csv" ] \
       && [ -f "${RESULTS}/${tag}_validation.json" ] \
       && [ -f "${gpu_pre}" ] && [ -f "${gpu_post}" ] \
       && [ -f "${gpu_monitor}" ] && [ -f "${gpu_observations}" ]; then
        expected=$(marker_content "${seq}" "${tag}")
        if [ "$(cat "${marker}")" = "${expected}" ]; then
            archive_file "${resume_json}"
            archive_file "${resume_log}"
            python3 ./validate_dsa_native.py "${RESULTS}/${tag}.log" \
                --trace "${RESULTS}/${tag}_trace.csv" \
                --expected-gpu-uuid "${GPU_UUID}" \
                --json "${resume_json}" "${SHORT_ARGS[@]}" \
                > "${resume_log}" 2>&1 || resume_rc=$?
            if [ "${resume_rc}" -eq 0 ] \
               && cmp -s "${resume_json}" "${RESULTS}/${tag}_validation.json"; then
                log "skip ${tag} (artifact hashes match; strict validator replay matched)"
                return 0
            fi
            log "rerun ${tag} (strict validator replay failed/mismatched)"
        else
            log "rerun ${tag} (marker/artifact signature mismatch)"
        fi
    elif [ -e "${marker}" ]; then
        log "rerun ${tag} (marker has missing artifacts)"
    fi

    archive_file "${marker}"
    archive_file "${invalid}"
    for path in "${RESULTS}/${tag}.log" "${RESULTS}/${tag}.validator.log" \
                "${RESULTS}/${tag}_validation.json" "${RESULTS}/${tag}_trace.csv" \
                "${gpu_pre}" "${gpu_post}" "${gpu_monitor}" \
                "${gpu_observations}"; do
        archive_file "${path}"
    done
    python3 ./gpu_exclusivity.py check --lease "${EXCLUSIVITY_LEASE}" \
        --json "${gpu_pre}" --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${GPU_INDEX}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --owner-pid "$$" --phase "${tag}_pre" >> "${RESULTS}/dsa.log" 2>&1 || {
            write_exclusivity_rejection "${tag}_pre" "${gpu_pre}"
            fail "${tag} GPU pre-check failed; accepted_timing=0"
            exit 2
        }
    log "run  ${tag}: mapping=${mapping} repeats=${REPEATS} modes=4 "\
"order=cyclic_latin_4 invocations=$((4 + 4 * WARMUP + 4 * REPEATS)) "\
"timeout=${ACTIVE_STEP_TIMEOUT}"
    run_monitored "${tag}" "${gpu_monitor}" "${gpu_observations}" 1 \
        ./dsa_native --seq "${seq}" --repeats "${REPEATS}" --warmup "${WARMUP}" \
        --tag "${tag}" --trace "${RESULTS}/${tag}_trace.csv" \
        "${SHORT_ARGS[@]}" "${PAIR_ARGS[@]}" > "${RESULTS}/${tag}.log" 2>&1 || rc=$?
    python3 ./validate_dsa_native.py "${RESULTS}/${tag}.log" \
        --trace "${RESULTS}/${tag}_trace.csv" \
        --expected-gpu-uuid "${GPU_UUID}" \
        --json "${RESULTS}/${tag}_validation.json" \
        "${SHORT_ARGS[@]}" > "${RESULTS}/${tag}.validator.log" 2>&1 || validator_rc=$?
    python3 ./gpu_exclusivity.py check --lease "${EXCLUSIVITY_LEASE}" \
        --json "${gpu_post}" --nvidia-smi "${NVIDIA_SMI}" --gpu-index "${GPU_INDEX}" \
        --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --owner-pid "$$" --phase "${tag}_post" >> "${RESULTS}/dsa.log" 2>&1 || {
            write_exclusivity_rejection "${tag}_post" "${gpu_post}"
            fail "${tag} GPU post-check failed; accepted_timing=0"
            exit 2
        }
    if [ "${rc}" -ne 0 ]; then
        write_step_rejection "${tag}" "${rc}" "${RESULTS}/${tag}.log" \
            "${ACTIVE_STEP_TIMEOUT}"
        fail "${tag} native step failed rc=${rc}; campaign permanently rejected, accepted_timing=0"
        exit 2
    fi
    if [ "${validator_rc}" -ne 0 ]; then
        write_step_rejection "${tag}_validator" "${validator_rc}" \
            "${RESULTS}/${tag}.validator.log" "${ACTIVE_STEP_TIMEOUT}" validator
        fail "${tag} strict validator failed rc=${validator_rc}; campaign permanently rejected, accepted_timing=0"
        exit 2
    fi
    if [ "${rc}" -eq 0 ] && [ "${validator_rc}" -eq 0 ]; then
        local marker_tmp
        marker_tmp=$(mktemp "${marker}.tmp.XXXXXX")
        marker_content "${seq}" "${tag}" > "${marker_tmp}"
        mv "${marker_tmp}" "${marker}"
        log "ok   ${tag}"
        return 0
    fi
    base_signature "${seq}" "${tag}" > "${invalid}"
    fail "${tag} INVALID native_rc=${rc} validator_rc=${validator_rc}"
    return 1
}

log "=== Tier 5 native DSA chain ==="
log "contexts=${SEQS[*]} repeats=${REPEATS} warmup=${WARMUP} FAST=${FAST} PROFILE=${PROFILE}"
log "4K/32K exact mappings; 128K/1M work-complete packed proxy boundary"

# This CPU shape oracle is diagnostic only.  Its failure cannot create a false experimental
# failure because the native validator independently derives every executed shape.
for model in deepseek-v3.2-dsa glm-5.2-dsa; do
    for seq in 32768 1048576; do
        oracle="${RESULTS}/oracle_${model}_${seq}.json"
        if [ ! -f "${oracle}" ]; then
            python3 ../../tools/dep_oracle.py --model "${model}" --seq "${seq}" \
                --json "${oracle}" > "${RESULTS}/oracle_${model}_${seq}.log" 2>&1 || \
                warn "dependency oracle ${model}/${seq} unavailable (diagnostic only)"
        fi
    done
done

failed=0
for seq in "${SEQS[@]}"; do
    run_point "${seq}" || failed=1
done

archive_file "${RESULTS}/dsa_matrix.log"
archive_file "${RESULTS}/validation_matrix.json"
aggregate_rc=0
python3 ./aggregate_dsa_native.py --results "${RESULTS}" --fast "${FAST}" \
    --log-out "${RESULTS}/dsa_matrix.log" \
    --json-out "${RESULTS}/validation_matrix.json" \
    >> "${RESULTS}/dsa.log" 2>&1 || aggregate_rc=$?
if [ "${aggregate_rc}" -ne 0 ]; then
    fail "strict matrix aggregation failed rc=${aggregate_rc}"
    failed=1
fi

# Independent one-repeat profiler proof.  These results never enter the timing matrix.
if [ "${PROFILE}" = "1" ] && [ "${failed}" -eq 0 ]; then
    ACTIVE_STEP_TIMEOUT=300s
    profiler_failed=0
    profile_tag="dsa_profile_seq4096"
    profile_base="${RESULTS}/${profile_tag}"
    profile_gpu_pre="${profile_base}_gpu_pre.json"
    profile_gpu_post="${profile_base}_gpu_post.json"
    profile_gpu_monitor="${profile_base}_gpu_monitor.json"
    profile_gpu_observations="${profile_base}_gpu_observations.ndjson"
    ncu_gpu_pre="${RESULTS}/dsa_ncu_gpu_pre.json"
    ncu_gpu_post="${RESULTS}/dsa_ncu_gpu_post.json"
    ncu_gpu_monitor="${RESULTS}/dsa_ncu_gpu_monitor.json"
    ncu_gpu_observations="${RESULTS}/dsa_ncu_gpu_observations.ndjson"
    for path in "${profile_base}.nsys-rep" "${profile_base}.sqlite" \
                "${profile_base}.log" "${profile_base}_globaltimer.csv" \
                "${profile_base}_cuda_gpu_kern_sum.csv" \
                "${profile_base}_nvtx_gpu_proj_sum.csv" \
                "${profile_base}_nvtx_kern_sum.csv" \
                "${profile_base}_profile_validation.json" \
                "${profile_base}_stats.stderr" "${profile_gpu_pre}" \
                "${profile_gpu_post}" "${profile_gpu_monitor}" \
                "${profile_gpu_observations}" "${ncu_gpu_pre}" "${ncu_gpu_post}" \
                "${ncu_gpu_monitor}" "${ncu_gpu_observations}"; do
        archive_file "${path}"
    done
    python3 ./gpu_exclusivity.py check --lease "${EXCLUSIVITY_LEASE}" \
        --json "${profile_gpu_pre}" --nvidia-smi "${NVIDIA_SMI}" \
        --gpu-index "${GPU_INDEX}" --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --owner-pid "$$" --phase nsys_pre \
        >> "${RESULTS}/dsa.log" 2>&1 || {
            write_exclusivity_rejection nsys_pre "${profile_gpu_pre}"
            fail "nsys GPU pre-check failed; accepted_timing=0"
            exit 2
        }
    if command -v nsys >/dev/null 2>&1; then
        log "profile ${profile_tag}: NVTX launch ranges -> GPU kernels (not a timing sample)"
        nsys_rc=0
        run_monitored nsys_4k_sidecar "${profile_gpu_monitor}" \
            "${profile_gpu_observations}" 1 \
            nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
            --sample=none --cpuctxsw=none --force-overwrite=true \
            --output="${profile_base}" ./dsa_native --seq 4096 --repeats 1 --warmup 1 \
            --allow-short --tag "${profile_tag}" \
            --trace "${profile_base}_globaltimer.csv" \
            > "${profile_base}.log" 2>&1 || nsys_rc=$?
        if [ "${nsys_rc}" -ne 0 ]; then
                fail "nsys profile failed"
                profiler_failed=1
        fi
        if [ -f "${profile_base}.nsys-rep" ]; then
            : > "${profile_base}_stats.stderr"
            nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
                "${profile_base}.nsys-rep" > "${profile_base}_cuda_gpu_kern_sum.csv" \
                2>> "${profile_base}_stats.stderr" || profiler_failed=1
            nsys stats --force-export=true --report nvtx_gpu_proj_sum --format csv \
                "${profile_base}.nsys-rep" > "${profile_base}_nvtx_gpu_proj_sum.csv" \
                2>> "${profile_base}_stats.stderr" || profiler_failed=1
            nsys stats --force-export=true --report nvtx_kern_sum:base --format csv \
                "${profile_base}.nsys-rep" > "${profile_base}_nvtx_kern_sum.csv" \
                2>> "${profile_base}_stats.stderr" || profiler_failed=1
            python3 ./validate_dsa_profile.py \
                --projection "${profile_base}_nvtx_gpu_proj_sum.csv" \
                --range-kernels "${profile_base}_nvtx_kern_sum.csv" \
                --kernels "${profile_base}_cuda_gpu_kern_sum.csv" \
                --json "${profile_base}_profile_validation.json" \
                >> "${RESULTS}/dsa.log" 2>&1 || profiler_failed=1
        else
            fail "nsys did not produce ${profile_base}.nsys-rep"
            profiler_failed=1
        fi
    else
        fail "nsys unavailable; required NVTX-to-kernel profiler proof missing"
        profiler_failed=1
    fi
    python3 ./gpu_exclusivity.py check --lease "${EXCLUSIVITY_LEASE}" \
        --json "${profile_gpu_post}" --nvidia-smi "${NVIDIA_SMI}" \
        --gpu-index "${GPU_INDEX}" --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
        --owner-pid "$$" --phase nsys_post \
        >> "${RESULTS}/dsa.log" 2>&1 || {
            write_exclusivity_rejection nsys_post "${profile_gpu_post}"
            fail "nsys GPU post-check failed; accepted_timing=0"
            exit 2
        }

    # launch-count=1 is only a hardware-counter permission probe; never a timing sample.
    if command -v ncu >/dev/null 2>&1; then
        python3 ./gpu_exclusivity.py check --lease "${EXCLUSIVITY_LEASE}" \
            --json "${ncu_gpu_pre}" --nvidia-smi "${NVIDIA_SMI}" \
            --gpu-index "${GPU_INDEX}" --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
            --owner-pid "$$" --phase ncu_pre \
            >> "${RESULTS}/dsa.log" 2>&1 || {
                write_exclusivity_rejection ncu_pre "${ncu_gpu_pre}"
                fail "ncu GPU pre-check failed; accepted_timing=0"
                exit 2
            }
        ncu_stdout="${RESULTS}/ncu_permission.stdout"
        ncu_stderr="${RESULTS}/ncu_permission.stderr"
        ncu_command="ncu --set basic --launch-count 1 ./dsa_native --seq 4096 --repeats 1 --warmup 1 --allow-short --tag dsa_ncu_permission_probe"
        ncu_rc=0
        run_monitored ncu_permission_probe "${ncu_gpu_monitor}" \
            "${ncu_gpu_observations}" 0 \
            ncu --set basic --launch-count 1 ./dsa_native --seq 4096 --repeats 1 --warmup 1 \
            --allow-short --tag dsa_ncu_permission_probe \
            > "${ncu_stdout}" 2> "${ncu_stderr}" || ncu_rc=$?
        python3 ./profiler_evidence.py --tool "$(command -v ncu)" \
            --returncode "${ncu_rc}" --stdout "${ncu_stdout}" --stderr "${ncu_stderr}" \
            --command "${ncu_command}" --json "${RESULTS}/ncu_permission.json" \
            >> "${RESULTS}/dsa.log" 2>&1
        python3 ./gpu_exclusivity.py check --lease "${EXCLUSIVITY_LEASE}" \
            --json "${ncu_gpu_post}" --nvidia-smi "${NVIDIA_SMI}" \
            --gpu-index "${GPU_INDEX}" --query-timeout-ms "${NVIDIA_SMI_TIMEOUT_MS}" \
            --owner-pid "$$" --phase ncu_post \
            >> "${RESULTS}/dsa.log" 2>&1 || {
                write_exclusivity_rejection ncu_post "${ncu_gpu_post}"
                fail "ncu GPU post-check failed; accepted_timing=0"
                exit 2
            }
    else
        fail "ncu unavailable; hardware-counter permission evidence missing"
        profiler_failed=1
    fi
    if [ "${profiler_failed}" -ne 0 ]; then
        fail "required nsys NVTX-to-kernel proof failed"
        failed=1
    fi
fi

if [ "${failed}" -ne 0 ]; then
    log "Tier 5 campaign finished INVALID; inspect failures.log and strict JSON artifacts"
    exit 2
fi

python3 - "${RESULTS}/terminal_status.json" "${RESULTS}" "${FAST}" "${PROFILE}" \
    "$(sha "${SELF}")" "${SEQS[@]}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
results = Path(sys.argv[2])
fast = int(sys.argv[3])
profile = int(sys.argv[4])
runner_sha = sys.argv[5]
seqs = [int(value) for value in sys.argv[6:]]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

tags = [
    f"dsa_{'exact' if seq <= 32768 else 'work_complete_packed_proxy'}_seq{seq}"
    for seq in seqs
]
evidence = {
    "validation_matrix": digest(results / "validation_matrix.json"),
    "binary_proof": digest(results / "dsa_binary_proof.json"),
    "build_provenance": digest(results / "dsa_build_manifest.json"),
    "gpu_exclusivity_lease": digest(results / "gpu_exclusivity_lease.json"),
}
if profile:
    evidence["profile_validation"] = digest(
        results / "dsa_profile_seq4096_profile_validation.json"
    )
    evidence["ncu_permission"] = digest(results / "ncu_permission.json")
payload = {
    "schema": 1,
    "status": "PASS",
    "errors": [],
    "campaign": "tier5_native_dsa",
    "fast": fast,
    "profile": profile,
    "expected_tags": sorted(tags),
    "runner_sha256": runner_sha,
    "evidence_sha256": evidence,
    "created_at": datetime.now(timezone.utc).isoformat(),
}
output.parent.mkdir(parents=True, exist_ok=True)
fd, temp = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, output)
finally:
    try: os.unlink(temp)
    except FileNotFoundError: pass
PY

finalizer_rc=0
python3 ./finalize_dsa_campaign.py --results "${RESULTS}" --fast "${FAST}" \
    --profile "${PROFILE}" --binary ./dsa_native --runner "${SELF}" \
    --json "${RESULTS}/campaign_admission.json" \
    >> "${RESULTS}/dsa.log" 2>&1 || finalizer_rc=$?
if [ "${finalizer_rc}" -ne 0 ]; then
    fail "unified campaign admission failed rc=${finalizer_rc}; accepted_timing=0"
    exit 2
fi
log "Tier 5 campaign finished PASS; admission=${RESULTS}/campaign_admission.json"
exit 0
