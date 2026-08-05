#!/usr/bin/env bash
# run_llm_sweep.sh — Tier 4: Qwen3.6-27B end-to-end PDL bracket on a single B300/B200.
#
# THE ONE NUMBER THIS PRODUCES
# ----------------------------
#   Ceiling - PDL_grid  =  how much headroom CTA-level dependency resolution still has
#                          AFTER production-grade grid-level PDL has taken its share.
# If that gap is small, the whole CTA-level direction needs re-evaluating.
#
# THREE RUNGS (see ../../EXPERIMENT_PLAN.md §8.2)
#   PDL_off   all PDL disabled
#   PDL_grid  current production config  <-- THE FLOOR. Not "no PDL"!
#             TRT-LLM / vLLM / SGLang already ship grid-level PDL, so measuring against
#             PDL-off would badly overstate the remaining opportunity.
#   Ceiling   gdc_wait removed (results are WRONG; timing only) = dependency costs nothing
#
# WHY Qwen3.6-27B
#   48 of its 64 layers are Gated DeltaNet (linear attention, recurrent, no KV cache).
#   Chunked DeltaNet is intra-chunk parallel + inter-chunk sequential state passing, i.e. a
#   low-degree 1-to-1 chain of length seq/chunk -- the single most favourable shape for
#   CTA-level dependencies. The model's DOMINANT compute pattern is the thing under study.
#   BF16 ~54 GB, so it fits one card with room to spare.
#
# Usage:
#   ./run_llm_sweep.sh                 # full sweep
#   FAST=1 ./run_llm_sweep.sh          # smoke test
#   ENGINE=vllm ./run_llm_sweep.sh     # pick the serving stack

set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${SELF}"
    exit 0
fi

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
ENGINE="${ENGINE:-vllm}"
RESULTS="${RESULTS:-results_llm}"
FAST="${FAST:-0}"
NSYS="${NSYS:-nsys}"
mkdir -p "${RESULTS}"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/llm.log"; }
fail() { echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "${RESULTS}/llm.log" \
                                              | tee -a "${RESULTS}/failures.log"; }

if [ "${FAST}" = "1" ]; then
    BATCHES=(1 8); SEQS=(4096); REQS=16
else
    # BS=1 decode is the most interesting point: smallest grids, GPU not full, largest
    # overlap headroom. vLLM's own note that PDL "never hurts in the low-batch scenario"
    # points the same way.
    BATCHES=(1 4 16 64); SEQS=(4096 32768 131072); REQS=64
fi

# ---- the three rungs, expressed as env deltas -------------------------------------------
# Ceiling needs gdc_wait removed. There is no supported switch for that, so it is applied by
# patching the Triton PDL helper to a no-op (see ceiling_patch.py). That is why it is gated
# behind CTA_PDL_ALLOW_CEILING: it produces WRONG RESULTS by construction.
rung_env() {
    case "$1" in
        pdl_off)
            echo "TRTLLM_ENABLE_PDL=0 TORCHINDUCTOR_ENABLE_PDL=0" ;;
        pdl_grid)
            echo "TRTLLM_ENABLE_PDL=1 TORCHINDUCTOR_ENABLE_PDL=1" ;;
        ceiling)
            echo "TRTLLM_ENABLE_PDL=1 TORCHINDUCTOR_ENABLE_PDL=1 CTA_PDL_CEILING=1" ;;
    esac
}

run_one() {
    local rung="$1" bs="$2" seq="$3"
    local name="${rung}_bs${bs}_seq${seq}"
    if [ -f "${RESULTS}/${name}.done" ]; then log "skip ${name}"; return 0; fi
    log "run  ${name}"

    local env_kv; env_kv="$(rung_env "${rung}")"
    # shellcheck disable=SC2086
    if env ${env_kv} \
        VLLM_USE_FULL_CUDA_GRAPH=1 \
        python3 bench_llm.py \
            --model "${MODEL}" --engine "${ENGINE}" \
            --batch "${bs}" --seq "${seq}" --requests "${REQS}" \
            --rung "${rung}" --tag "${name}" \
        >> "${RESULTS}/${name}.log" 2>&1
    then
        grep -h '^SUMMARY' "${RESULTS}/${name}.log" >> "${RESULTS}/summary_llm.txt" 2>/dev/null || true
        touch "${RESULTS}/${name}.done"
        log "ok   ${name}"
    else
        fail "${name} (see ${RESULTS}/${name}.log)"
    fi
}

log "=== Tier 4: LLM end-to-end PDL bracket ==="
log "model=${MODEL} engine=${ENGINE}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv \
    > "${RESULTS}/device.txt" 2>/dev/null || true

# NOTE: full CUDA graph mode matters. vLLM only enables PDL under FULL graphs because the
# host-side cost of PDL is a NET LOSS in prefill / piecewise modes.
for seq in "${SEQS[@]}"; do
    for bs in "${BATCHES[@]}"; do
        for rung in pdl_off pdl_grid ceiling; do
            run_one "${rung}" "${bs}" "${seq}"
        done
    done
done

# ---- one nsys capture at the headline config for kernel-level attribution ----------------
if command -v "${NSYS}" >/dev/null 2>&1; then
    for rung in pdl_grid ceiling; do
        prof="${RESULTS}/nsys_${rung}"
        if [ ! -f "${prof}.nsys-rep" ]; then
            log "nsys capture ${rung}"
            env $(rung_env "${rung}") VLLM_USE_FULL_CUDA_GRAPH=1 \
                "${NSYS}" profile --cuda-graph-trace=node -o "${prof}" --force-overwrite true \
                python3 bench_llm.py --model "${MODEL}" --engine "${ENGINE}" \
                    --batch 1 --seq 4096 --requests 8 --rung "${rung}" --tag "nsys_${rung}" \
                >> "${RESULTS}/nsys_${rung}.log" 2>&1 || fail "nsys ${rung}"
            "${NSYS}" export --type sqlite -o "${prof}.sqlite" "${prof}.nsys-rep" \
                >> "${RESULTS}/nsys_${rung}.log" 2>&1 || true
        fi
    done
else
    log "nsys not found, skipping kernel-level capture"
fi

log "=== done ==="
log "Next: python3 ../../tools/llm_bracket.py ${RESULTS}/summary_llm.txt"
if [ -s "${RESULTS}/failures.log" ]; then log "FAILURES:"; cat "${RESULTS}/failures.log"; fi
