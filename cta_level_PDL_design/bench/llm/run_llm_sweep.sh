#!/usr/bin/env bash
# Formal Tier-4 Qwen3.6-27B runner.
#
# Required:
#   RESULTS=/persistent/new-or-resumed/tier4/root
# Optional:
#   MODEL=/workspace/models/Qwen3.6-27B
#   COHORTS=decode,prefill       (or only one name)
#   KV_OFFLOADING_SIZE=<GiB>     (native KV connector; never cpu_offload_gb)
#
# Each cohort/attempt is one persistent process with one model/worker cohort,
# three separately lowered variants, 31 adjacent Latin-3 timing triplets per
# point, and its own PTX/cubin/Nsight proof.  Failed attempts are preserved;
# rerunning selects a fresh attempt directory.  A completed raw candidate with
# an existing nsys report resumes at export/finalization without rerunning GPU.

set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")"

MODEL="${MODEL:-/workspace/models/Qwen3.6-27B}"
RESULTS="${RESULTS:-}"
COHORTS="${COHORTS:-decode,prefill}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,15p' "$(basename "$0")"
    exit 0
fi

if [[ -z "${RESULTS}" ]]; then
    echo "RUNNER status=blocked RESULTS must be explicit for safe resume" >&2
    exit 3
fi
if [[ "${MODEL}" != /* || "${RESULTS}" != /* ]]; then
    echo "RUNNER status=blocked MODEL and RESULTS must be absolute paths" >&2
    exit 3
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

manifest_args=(--results-root "${RESULTS}" --model "${MODEL}")
if [[ -n "${KV_OFFLOADING_SIZE:-}" ]]; then
    manifest_args+=(--kv-offloading-size "${KV_OFFLOADING_SIZE}")
fi
python3 tier4_manifest.py "${manifest_args[@]}"
MODEL_IDENTITY="${RESULTS}/model_identity.json"

mkdir -p "${RESULTS}/profiles" "${RESULTS}/logs" "${RESULTS}/preflight"

next_attempt_dir() {
    local cohort="$1"
    local base="${RESULTS}/cohorts/${cohort}"
    if [[ ! -e "${base}" ]]; then
        printf '%s\n' "${base}"
        return
    fi
    local index=2
    while [[ -e "${base}_attempt${index}" ]]; do
        index=$((index + 1))
    done
    printf '%s\n' "${base}_attempt${index}"
}

refresh_manifest() {
    python3 tier4_manifest.py "${manifest_args[@]}"
}

finish_candidate() {
    local cohort_dir="$1"
    local label
    label="$(basename "${cohort_dir}")"
    local report="${RESULTS}/profiles/${label}.nsys-rep"
    local sqlite="${cohort_dir}/profile.sqlite"
    if [[ ! -f "${report}" ]]; then
        return 4
    fi
    if [[ ! -f "${sqlite}" ]]; then
        nsys export --type sqlite --force-overwrite=true \
            --output="${sqlite}" "${report}"
    fi
    python3 tier4_finalize.py \
        --results "${cohort_dir}" \
        --nsys-sqlite "${sqlite}"
    python3 preflight_llm.py \
        --phase after \
        --model "${MODEL}" \
        --model-identity "${MODEL_IDENTITY}" \
        --results "${cohort_dir}" \
        --proof-root "${cohort_dir}/evidence" \
        --json "${RESULTS}/preflight/${label}_after.json"
    refresh_manifest
    python3 tier4_finalize.py \
        --results "${cohort_dir}" \
        --verify-admission
}

run_cohort() {
    local cohort="$1"
    local candidate
    for candidate in "${RESULTS}/cohorts/${cohort}"*; do
        [[ -d "${candidate}" ]] || continue
        if [[ -f "${candidate}/admission.json" ]]; then
            if python3 tier4_finalize.py \
                --results "${candidate}" \
                --verify-admission; then
                echo "RUNNER cohort=${cohort} status=resume-skip-admitted path=${candidate}"
                return 0
            fi
            echo "RUNNER cohort=${cohort} status=resume-admission-rejected path=${candidate}" >&2
            continue
        fi
        if [[ -f "${candidate}/raw_triplet.json" ]] && \
           [[ ! -f "${candidate}/finalize_in_progress.json" ]] && \
           [[ ! -f "${candidate}/finalize_failure.json" ]] && \
           [[ ! -f "${candidate}/driver_error.json" ]] && \
           [[ ! -f "${candidate}/warmup_output_mismatch.json" ]] && \
           [[ ! -f "${candidate}/sample_output_mismatch.json" ]]; then
            if finish_candidate "${candidate}"; then
                echo "RUNNER cohort=${cohort} status=resume-finalized path=${candidate}"
                return 0
            else
                local resume_status=$?
                if [[ ${resume_status} -ne 4 ]]; then
                    refresh_manifest
                    return "${resume_status}"
                fi
            fi
        fi
    done

    local cohort_dir
    cohort_dir="$(next_attempt_dir "${cohort}")"
    local label
    label="$(basename "${cohort_dir}")"
    local profile_base="${RESULTS}/profiles/${label}"
    local log="${RESULTS}/logs/${label}.log"

    python3 preflight_llm.py \
        --phase before \
        --model "${MODEL}" \
        --model-identity "${MODEL_IDENTITY}" \
        --results "${cohort_dir}" \
        --json "${RESULTS}/preflight/${label}_before.json"

    local -a common=(
        --model "${MODEL}"
        --model-identity "${MODEL_IDENTITY}"
        --formal-root-manifest "${RESULTS}/manifest.json"
        --results "${cohort_dir}"
        --repeats 31
        --warmups 3
        --bootstrap-samples 2000
        --max-num-batched-tokens 16384
        --variant-timeout 3600
    )
    if [[ -n "${KV_OFFLOADING_SIZE:-}" ]]; then
        common+=(--kv-offloading-size "${KV_OFFLOADING_SIZE}" --kv-offloading-backend native)
    fi

    local -a points
    local cohort_id gpu_mem proof_point
    case "${cohort}" in
        decode)
            cohort_id="decode_bs_scan_v1"
            gpu_mem="0.82"
            proof_point="decode_bs1:1:64:16:decode"
            points=(
                --point decode_bs1:1:64:16:decode
                --point decode_bs4:4:64:16:decode
                --point decode_bs16:16:64:16:decode
                --point decode_bs64:64:64:16:decode
            )
            ;;
        prefill)
            cohort_id="prefill_context_scan_v1"
            gpu_mem="0.90"
            proof_point="prefill_full_decode_proof:1:64:2:decode"
            points=(
                --point prefill_4k:1:4096:2:prefill
                --point prefill_32k:1:32768:2:prefill
                --point prefill_128k:1:131072:2:prefill
            )
            ;;
        *)
            echo "RUNNER status=blocked unknown cohort ${cohort}" >&2
            return 3
            ;;
    esac

    echo "RUNNER cohort=${cohort} status=starting path=${cohort_dir}"
    set -o pipefail
    nsys profile \
        --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=false \
        --output="${profile_base}" \
        python3 tier4_driver.py \
            "${common[@]}" \
            --cohort-id "${cohort_id}" \
            --gpu-mem-util "${gpu_mem}" \
            --proof-point "${proof_point}" \
            "${points[@]}" \
        2>&1 | tee "${log}"

    finish_candidate "${cohort_dir}"
    echo "RUNNER cohort=${cohort} status=admitted path=${cohort_dir}"
}

IFS=',' read -r -a requested_cohorts <<< "${COHORTS}"
for cohort in "${requested_cohorts[@]}"; do
    run_cohort "${cohort}"
done

refresh_manifest
echo "RUNNER status=ok results=${RESULTS}"
