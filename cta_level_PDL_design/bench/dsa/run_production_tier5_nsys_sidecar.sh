#!/usr/bin/env bash
# One-row, explicitly nonformal Nsight Systems sidecar.  Its profiler output is
# never part of the formal campaign inventory or timing aggregation.

set -uo pipefail
SELF_DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd "${SELF_DIR}"

if [ "$#" -ne 0 ]; then
    echo "FAIL: nsys sidecar takes no arguments" >&2
    exit 2
fi
if [ "${FAST:-0}" != "1" ]; then
    echo "FAIL: nsys sidecar is nonformal and requires FAST=1" >&2
    exit 2
fi
if [ "${EXECUTE_GPU:-0}" != "1" ] || [ "${TIER5_PRODUCTION_GPU_ALLOWED:-0}" != "1" ]; then
    echo "FAIL: explicit guarded GPU execution is required" >&2
    exit 2
fi

MODEL="${PROFILE_MODEL:-deepseek_v32}"
SEQ="${PROFILE_SEQ:-4096}"
WORKLOAD="${PROFILE_WORKLOAD:-operator_chain}"
ROW_ID="${MODEL}.${WORKLOAD}.seq${SEQ}"
python3 - "${MODEL}" "${SEQ}" "${WORKLOAD}" "${ROW_ID}" <<'PY' || exit 2
import sys
import production_tier5 as h
model, seq_text, workload, row_id = sys.argv[1:]
models = h.parse_csv_choice(model, h.MODEL_SPECS, "profile model")
seqs = h.parse_seqs(seq_text)
workloads = h.parse_csv_choice(workload, h.FORMAL_WORKLOADS, "profile workload")
matrix = h.expected_matrix(models, seqs, workloads)
matches = [row for row in matrix if row["row_id"] == row_id]
if len(matches) != 1 or matches[0]["workload"] == "moe32":
    raise SystemExit("profile row must be one exact canonical attention row")
PY

NSYS="${NSYS:-/usr/local/cuda/bin/nsys}"
if [ ! -x "${NSYS}" ]; then
    echo "FAIL: nsys executable unavailable: ${NSYS}" >&2
    exit 2
fi
NSYS_RESOLVED="$(readlink -f -- "${NSYS}")" || exit 2
NSYS_SHA256="$(sha256sum -- "${NSYS_RESOLVED}" | awk '{print $1}')" || exit 2
NSYS_VERSION="$("${NSYS_RESOLVED}" --version 2>&1)" || exit 2
SIDECAR_INPUT="${RESULTS:-results_dsa_production_nsys_sidecar}"
SIDECAR_PARENT_INPUT="$(dirname -- "${SIDECAR_INPUT}")"
SIDECAR_NAME="$(basename -- "${SIDECAR_INPUT}")"
mkdir -p -- "${SIDECAR_PARENT_INPUT}"
SIDECAR_PARENT="$(cd "${SIDECAR_PARENT_INPUT}" && pwd -P)"
SIDECAR_ROOT="${SIDECAR_PARENT}/${SIDECAR_NAME}"
if [ -e "${SIDECAR_ROOT}" ] || [ -L "${SIDECAR_ROOT}" ]; then
    echo "FAIL: sidecar destination already exists: ${SIDECAR_ROOT}" >&2
    exit 2
fi
SIDECAR_STAGE="$(mktemp -d "${SIDECAR_PARENT}/${SIDECAR_NAME}.inprogress.XXXXXX")"
SIDECAR_PUBLISHED=0
preserve_sidecar_failure() {
    local rc="$?" failed
    trap - EXIT INT TERM
    if [ "${SIDECAR_PUBLISHED}" = "0" ] && [ -d "${SIDECAR_STAGE}" ]; then
        failed="${SIDECAR_PARENT}/${SIDECAR_NAME}.failed.$(date -u +%Y%m%dT%H%M%SZ).$$"
        mv -T -- "${SIDECAR_STAGE}" "${failed}"
        echo "FAIL: incomplete nsys sidecar preserved at ${failed}" >&2
    fi
    exit "${rc}"
}
trap preserve_sidecar_failure EXIT INT TERM
CAMPAIGN_ROOT="${SIDECAR_STAGE}/fragment_campaign"
PROFILE_PREFIX="${SIDECAR_STAGE}/profile"
NSYS_PROFILE_ARGV=(
    "${NSYS_RESOLVED}" profile --trace=cuda,nvtx,osrt --sample=none
    --cpuctxsw=none --force-overwrite=false --output "${PROFILE_PREFIX}"
    bash ./run_production_tier5_fragments.sh
)

env \
    RESULTS="${CAMPAIGN_ROOT}" \
    EXECUTE_GPU=1 FAST=1 TIER5_PRODUCTION_GPU_ALLOWED=1 \
    MODELS="${MODEL}" SEQS="${SEQ}" WORKLOADS="${WORKLOAD}" \
    WARMUP="${WARMUP:-0}" REPEATS="${REPEATS:-1}" \
    MOE_TOKENS="${MOE_TOKENS:-128}" \
    TIER5_FRAGMENT_ONLY_ROW="${ROW_ID}" \
    DSA_NVIDIA_SMI="${DSA_NVIDIA_SMI:-nvidia-smi}" \
    DSA_GPU_INDEX="${DSA_GPU_INDEX:-0}" \
    DSA_MONITOR_INTERVAL_MS="${DSA_MONITOR_INTERVAL_MS:-50}" \
    DSA_NVIDIA_SMI_TIMEOUT_MS="${DSA_NVIDIA_SMI_TIMEOUT_MS:-2000}" \
    "${NSYS_PROFILE_ARGV[@]}" || exit 2

PROFILE_REPORT="${PROFILE_PREFIX}.nsys-rep"
if [ ! -f "${PROFILE_REPORT}" ] || [ -L "${PROFILE_REPORT}" ]; then
    echo "FAIL: nsys report was not produced" >&2
    exit 2
fi
if [ ! -s "${PROFILE_REPORT}" ]; then
    echo "FAIL: nsys report is empty" >&2
    exit 2
fi
NSYS_STATS="${SIDECAR_STAGE}/nsys_stats.txt"
NSYS_STATS_ARGV=(
    "${NSYS_RESOLVED}" stats
    --report cuda_gpu_kern_sum --report cuda_api_sum --report nvtx_sum
    --format csv --output - "${PROFILE_REPORT}"
)
"${NSYS_STATS_ARGV[@]}" > "${NSYS_STATS}" 2>&1 || exit 2
python3 - "${SIDECAR_STAGE}" "${ROW_ID}" "${NSYS_RESOLVED}" \
    "${NSYS_SHA256}" "${NSYS_VERSION}" \
    "$(printf '%s\0' "${NSYS_PROFILE_ARGV[@]}" | base64 -w0)" \
    "$(printf '%s\0' "${NSYS_STATS_ARGV[@]}" | base64 -w0)" <<'PY' || exit 2
import base64
import hashlib, json, os, sys, tempfile
from pathlib import Path
root = Path(sys.argv[1])
row_id = sys.argv[2]
nsys_path, nsys_sha, nsys_version = sys.argv[3:6]
decode_argv = lambda value: base64.b64decode(value).decode().rstrip("\0").split("\0")
profile_argv = decode_argv(sys.argv[6])
stats_argv = decode_argv(sys.argv[7])
campaign = root / "fragment_campaign"
contract = json.load(open(campaign / "campaign_contract.json", encoding="utf-8"))
binding = json.load(open(campaign / "campaign_binding.json", encoding="utf-8"))
ordinal = next(i for i, row in enumerate(contract["ordered_matrix"]) if row["row_id"] == row_id)
fragment_marker = campaign / "rows" / f"{ordinal:03d}_{row_id}" / "fragment.done.json"
fragment_value = json.load(open(fragment_marker, encoding="utf-8"))
invocation_uuid = fragment_value["invocation_uuid"]
report = root / "profile.nsys-rep"
stats = root / "nsys_stats.txt"
stats_text = stats.read_text(encoding="utf-8", errors="replace")
lower = stats_text.lower()
api_tokens = [token for token in ("cudalaunchkernel", "cudalaunchkernelexc") if token in lower]
kernel_tokens = [token for token in ("mqa", "mla", "top_k", "topk", "gemm", "trtllm", "flashinfer") if token in lower]
nvtx_token = f"tier5_fragment:{ordinal}:{row_id}:{invocation_uuid}"
if not api_tokens:
    raise SystemExit("nsys stats did not prove a CUDA kernel launch API")
if not kernel_tokens:
    raise SystemExit("nsys stats did not prove a target DSA/GEMM/MLA kernel")
if nvtx_token not in stats_text:
    raise SystemExit("nsys stats did not prove the exact fragment NVTX range")
payload = {
    "schema": 1,
    "kind": "tier5_production_nsys_sidecar",
    "status": "PASS",
    "campaign_mode": "nonformal_short_profiler_sidecar_only",
    "formal": False,
    "measurement_role": "profiler_diagnostic_only",
    "accepted_timing": 0,
    "accepted_workload_timing": 0,
    "accepted_CTA_bracket": 0,
    "row_id": row_id,
    "ordinal": ordinal,
    "campaign_contract_sha256": contract["contract_sha256"],
    "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
    "fragment_marker_sha256": hashlib.sha256(fragment_marker.read_bytes()).hexdigest(),
    "profiler": {
        "resolved_path": nsys_path,
        "sha256": nsys_sha,
        "version": nsys_version,
        "profile_argv": profile_argv,
        "profile_argv_sha256": hashlib.sha256("\0".join(profile_argv).encode()).hexdigest(),
        "stats_argv": stats_argv,
        "stats_argv_sha256": hashlib.sha256("\0".join(stats_argv).encode()).hexdigest(),
    },
    "capture_proof": {
        "cuda_api_tokens": api_tokens,
        "target_kernel_tokens": kernel_tokens,
        "nvtx_range_prefix": nvtx_token,
        "fragment_invocation_uuid": invocation_uuid,
        "stats_path": stats.name,
        "stats_size_bytes": stats.stat().st_size,
        "stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
        "report_parsed_by_nsys_stats": True,
    },
    "nsys_report": {
        "path": report.name,
        "size_bytes": report.stat().st_size,
        "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    },
}
output = root / "nsys_sidecar.json"
fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=root)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, output)
directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY); os.fsync(directory); os.close(directory)
PY
python3 ./production_tier5_campaign.py publish-fragment \
    --stage "${SIDECAR_STAGE}" --final "${SIDECAR_ROOT}" || exit 2
SIDECAR_PUBLISHED=1
trap - EXIT INT TERM
echo "PRODUCTION_TIER5_NSYS_SIDECAR status=PASS row=${ROW_ID} accepted_workload_timing=0 root=${SIDECAR_ROOT}"
