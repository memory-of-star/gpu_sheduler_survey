#!/usr/bin/env bash
# run_dsa_chain.sh — Tier 5: DSA operator chain on a single B300/B200.
#
# The full DSA models do not fit one card (DeepSeek-V3.2 671B, GLM-5.2 744B; even FP4 needs
# 335/372 GB against 288 GB on B300), but the attention path does. This sweeps context
# length over the real-shape operator chain and brackets each point.
#
# What the context sweep is for: the lightning indexer is O(L^2), so its share of chain time
# climbs steeply with context. The benefit structure therefore MOVES as context grows --
# a single measurement at one length would be misleading.
#
# MoE is measured with a REDUCED expert count (32 instead of 256). The dependency shape is
# fixed by top-k routing and does not depend on the expert total, so this reproduces the
# structure at an eighth of the weights.
#
# Usage:
#   ./run_dsa_chain.sh              # full context sweep
#   FAST=1 ./run_dsa_chain.sh       # short lengths only

set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "${SELF}"
    exit 0
fi

RESULTS="${RESULTS:-results_dsa}"
FAST="${FAST:-0}"
mkdir -p "${RESULTS}"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "${RESULTS}/dsa.log"; }
fail() { echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "${RESULTS}/dsa.log" \
                                              | tee -a "${RESULTS}/failures.log"; }

if [ "${FAST}" = "1" ]; then
    SEQS=(4096 16384)
else
    # 1M is included but may OOM even for the attention path alone; the driver records the
    # failure and moves on rather than aborting.
    SEQS=(4096 32768 131072 1048576)
fi

step() {
    local name="$1"; shift
    if [ -f "${RESULTS}/${name}.done" ]; then log "skip ${name}"; return 0; fi
    log "run  ${name}"
    if "$@" >> "${RESULTS}/${name}.log" 2>&1; then
        grep -h '^SUMMARY' "${RESULTS}/${name}.log" >> "${RESULTS}/summary_dsa.txt" 2>/dev/null || true
        touch "${RESULTS}/${name}.done"
        log "ok   ${name}"
    else
        fail "${name} (see ${RESULTS}/${name}.log)"
    fi
}

log "=== Tier 5: DSA operator chain ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv \
    > "${RESULTS}/device.txt" 2>/dev/null || true

# ---- offline dependency derivation first: it needs no GPU and tells us what to expect ----
log "--- analytical dependency derivation (no GPU) ---"
for model in deepseek-v3.2-dsa glm-5.2-dsa; do
    for seq in 32768 1048576; do
        step "oracle_${model}_${seq}" \
            python3 ../../tools/dep_oracle.py --model "${model}" --seq "${seq}" \
                --json "${RESULTS}/oracle_${model}_${seq}.json"
    done
done

# ---- measured operator chain, GLM-5.2 shapes ----
log "--- DSA chain, real shapes, context sweep ---"
for seq in "${SEQS[@]}"; do
    for rung in floor ceiling; do
        step "dsa_${rung}_seq${seq}" \
            python3 dsa_chain.py --seq "${seq}" --rung "${rung}" \
                --tag "dsa_${rung}_seq${seq}"
    done
done

# ---- MoE dispatch/combine: the genuinely hard pattern ----
log "--- MoE dispatch/combine (reduced expert count) ---"
for toks in 1024 4096; do
    for rung in floor ceiling; do
        step "moe_${rung}_t${toks}" \
            python3 dsa_chain.py --moe --tokens "${toks}" --experts 32 --topk-experts 8 \
                --rung "${rung}" --tag "moe_${rung}_t${toks}"
    done
done

log "=== done ==="
log "summary: ${RESULTS}/summary_dsa.txt"
if [ -s "${RESULTS}/failures.log" ]; then log "FAILURES:"; cat "${RESULTS}/failures.log"; fi
