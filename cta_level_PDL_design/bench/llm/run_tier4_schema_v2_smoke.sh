#!/usr/bin/env bash
# One-process schema-v2 plumbing smoke. This is diagnostic-only and never
# substitutes for the 31-repeat formal decode/prefill cohorts.

set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-/workspace/models/Qwen3.6-27B}"
RESULTS="${RESULTS:-}"
if [[ -z "${RESULTS}" || "${MODEL}" != /* || "${RESULTS}" != /* ]]; then
    echo "SMOKE status=blocked absolute MODEL and explicit absolute RESULTS required" >&2
    exit 3
fi
if [[ -e "${RESULTS}" ]]; then
    echo "SMOKE status=blocked RESULTS must be a fresh path: ${RESULTS}" >&2
    exit 3
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 tier4_manifest.py --results-root "${RESULTS}" --model "${MODEL}"
mkdir -p "${RESULTS}/profiles" "${RESULTS}/logs" "${RESULTS}/preflight"

COHORT_DIR="${RESULTS}/cohorts/schema_v2_mixed_smoke"
MODEL_IDENTITY="${RESULTS}/model_identity.json"
PROFILE_BASE="${RESULTS}/profiles/schema_v2_mixed_smoke"
REPORT="${PROFILE_BASE}.nsys-rep"
SQLITE="${COHORT_DIR}/profile.sqlite"

python3 preflight_llm.py \
    --phase before \
    --model "${MODEL}" \
    --model-identity "${MODEL_IDENTITY}" \
    --results "${COHORT_DIR}" \
    --json "${RESULTS}/preflight/schema_v2_mixed_smoke_before.json"

nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=false \
    --output="${PROFILE_BASE}" \
    python3 tier4_driver.py \
        --model "${MODEL}" \
        --model-identity "${MODEL_IDENTITY}" \
        --formal-root-manifest "${RESULTS}/manifest.json" \
        --results "${COHORT_DIR}" \
        --cohort-id diagnostic_schema_v2_mixed_smoke \
        --point decode_smoke:1:64:2:decode \
        --point prefill_4k_smoke:1:4096:2:prefill \
        --proof-point decode_smoke:1:64:2:decode \
        --repeats 1 \
        --warmups 1 \
        --bootstrap-samples 100 \
        --gpu-mem-util 0.82 \
        --max-num-batched-tokens 16384 \
        --variant-timeout 3600 \
        --allow-short \
    2>&1 | tee "${RESULTS}/logs/schema_v2_mixed_smoke.log"

nsys export --type sqlite --force-overwrite=true \
    --output="${SQLITE}" "${REPORT}"
python3 tier4_finalize.py \
    --results "${COHORT_DIR}" \
    --nsys-sqlite "${SQLITE}" \
    --allow-short
python3 preflight_llm.py \
    --phase after \
    --model "${MODEL}" \
    --model-identity "${MODEL_IDENTITY}" \
    --results "${COHORT_DIR}" \
    --proof-root "${COHORT_DIR}/evidence" \
    --json "${RESULTS}/preflight/schema_v2_mixed_smoke_after.json"

echo "SMOKE status=ok diagnostic_only=1 results=${RESULTS}"
