#!/usr/bin/env bash
# preflight.sh — verify the rented machine before spending money on experiments.
#
# Runs in about a minute and fails LOUDLY. The point is to discover a broken toolchain now
# rather than three hours into the campaign.
#
# Usage:  ./preflight.sh          (from the cta_level_PDL_design directory)

SELF="$(cd "$(dirname "$0")" && pwd)"
cd "${SELF}"

PASS=0; WARN=0; FAIL=0
ok()   { echo "  [ ok ] $*"; PASS=$((PASS+1)); }
warn() { echo "  [warn] $*"; WARN=$((WARN+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

echo "======================================================================"
echo " preflight — CTA-level PDL evaluation"
echo "======================================================================"

# ---------------------------------------------------------------- 1. GPU
echo
echo "1. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    NGPU=$(nvidia-smi --list-gpus | wc -l)
    ok "${NAME} | ${MEM} | CC ${CC} | ${NGPU} GPU(s)"

    CC_NUM=$(echo "${CC}" | tr -d '.')
    if [ "${CC_NUM}" -ge 100 ] 2>/dev/null; then
        ok "sm_${CC_NUM}: CLC (clusterlaunchcontrol) supported"
        ARCH_GUESS="sm_${CC_NUM}"
    elif [ "${CC_NUM}" -ge 90 ] 2>/dev/null; then
        warn "sm_${CC_NUM}: PDL ok, but CLC needs sm_100+ (Tier 0.4 will skip)"
        ARCH_GUESS="sm_${CC_NUM}"
    else
        bad "CC ${CC} < 9.0: PDL unavailable, the whole campaign is meaningless here"
        ARCH_GUESS="sm_${CC_NUM}"
    fi

    MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "${MEM_MB}" -lt 180000 ] 2>/dev/null; then
        warn "${MEM_MB} MB may be tight for Qwen3.6-27B BF16 (~54GB + KV + activations)"
    fi

    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${USED}" -gt 2000 ] 2>/dev/null; then
        warn "${USED} MB already in use — another process is on this GPU, timings will be noisy"
    else
        ok "GPU is idle"
    fi
else
    bad "nvidia-smi not found"
    ARCH_GUESS="sm_103"
fi

# ---------------------------------------------------------------- 2. CUDA toolkit
echo
echo "2. CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
    NVCC_V=$(nvcc --version | grep release | sed 's/.*release //;s/,.*//')
    ok "nvcc ${NVCC_V}"
    MAJ=${NVCC_V%%.*}; MIN=${NVCC_V#*.}; MIN=${MIN%%.*}
    if [ "${MAJ}" -lt 12 ] 2>/dev/null; then
        bad "CUDA ${NVCC_V} < 12.0: libcu++ atomic_ref unavailable (dep_wait.cuh needs it)"
    elif [ "${MAJ}" -eq 12 ] && [ "${MIN}" -lt 8 ] 2>/dev/null; then
        warn "CUDA ${NVCC_V} < 12.8: CLC PTX may not assemble (Tier 0.4 will skip)"
    else
        ok "CUDA ${NVCC_V} covers everything including CLC"
    fi
else
    bad "nvcc not found — try: export PATH=/usr/local/cuda/bin:\$PATH"
fi

# ---------------------------------------------------------------- 3. build
echo
echo "3. build (ARCH=${ARCH_GUESS})"
if command -v nvcc >/dev/null 2>&1; then
    if ARCH="${ARCH_GUESS}" ./bench/build.sh > /tmp/preflight_build.log 2>&1; then
        ok "all benchmarks compiled"
        # cta_dep_pilot is the corrected harness and the only admissible source of Tier 1
        # gate data, so a missing pilot binary is fatal, not cosmetic.
        for b in cta_dep_pilot tier0_facts; do
            [ -x "bench/${b}" ] && ok "  bench/${b}" || bad "  bench/${b} missing"
        done
        [ -x bench/cta_dep_bench ] && ok "  bench/cta_dep_bench (rejected harness, re-audit only)" \
            || warn "  bench/cta_dep_bench missing (only needed to re-audit old results)"
        if [ -x bench/clc_probe ]; then ok "  bench/clc_probe"; else
            warn "  bench/clc_probe not built (expected below sm_100)"; fi
    else
        bad "build failed — see /tmp/preflight_build.log"
        tail -15 /tmp/preflight_build.log | sed 's/^/        /'
    fi
else
    bad "skipped (no nvcc)"
fi

# ---------------------------------------------------------------- 4. Python
echo
echo "4. Python / analysis chain"
PY=$(python3 --version 2>&1) && ok "${PY}" || bad "python3 not found"
if python3 tools/make_test_fixtures.py --out /tmp/preflight_fix > /dev/null 2>&1 \
   && python3 tools/analyze.py      /tmp/preflight_fix/summary.txt     > /dev/null 2>&1 \
   && python3 tools/cta_timeline.py /tmp/preflight_fix/trace.csv       > /dev/null 2>&1 \
   && python3 tools/llm_bracket.py  /tmp/preflight_fix/summary_llm.txt > /dev/null 2>&1; then
    ok "analysis toolchain works on synthetic fixtures"
else
    bad "analysis toolchain broken — fix BEFORE running experiments"
fi
# The gate path is what makes the session unattended: if analyze_pilot.py or gate.py is
# broken, the run produces numbers that nothing can turn into a decision.
if python3 tools/analyze_pilot.py /tmp/preflight_fix/pilot_matrix.log \
        --json /tmp/preflight_fix/pilot_analysis.json \
        --csv  /tmp/preflight_fix/pilot_summary.csv > /dev/null 2>&1 \
   && python3 tools/gate.py /tmp/preflight_fix/pilot_analysis.json \
        --json /tmp/preflight_fix/gate.json > /dev/null 2>&1; then
    ok "gate path works (analyze_pilot.py -> gate.py)"
else
    bad "gate path broken — the session could run and still not reach a decision"
fi
if python3 tools/dep_oracle.py --model qwen3.6-27b --tokens 256 --seq 2048 > /dev/null 2>&1; then
    ok "dep_oracle (offline dependency derivation)"
else
    bad "dep_oracle failed"
fi

# ---------------------------------------------------------------- 5. LLM deps (optional)
echo
echo "5. LLM stack (Tier 4 only — NOT needed by run_session.sh)"
if python3 -c "import torch" 2>/dev/null; then
    TV=$(python3 -c "import torch;print(torch.__version__)")
    ok "torch ${TV}"
    python3 -c "import torch;exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
        && ok "torch sees CUDA" || warn "torch cannot see CUDA (Tier 4 only)"
else
    warn "torch missing (Tier 4/5 unavailable; the gate session is unaffected)"
fi
python3 -c "import vllm" 2>/dev/null && ok "vllm present" \
    || warn "vllm missing (Tier 4 only; install it in the later session if the gate says GO)"

# ---------------------------------------------------------------- 6. misc
echo
echo "6. environment"
command -v nsys >/dev/null 2>&1 && ok "nsys present" \
    || warn "nsys missing (kernel-level capture will be skipped)"
command -v ncu  >/dev/null 2>&1 && ok "ncu present"  \
    || warn "ncu missing (per-kernel L2/DRAM metrics unavailable)"

# The sibling cross-stream benchmark is Tier 0.2. run_session.sh runs it automatically, but
# only if the whole repo was cloned rather than this subtree alone.
if [ -d ../cross_stream_PDL_survey/bench/pdl_bench ]; then
    ok "cross_stream_PDL_survey present (Tier 0.2 will run)"
else
    warn "cross_stream_PDL_survey missing — Tier 0.2 will be skipped; clone the whole repo"
fi

AVAIL=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${AVAIL}" -lt 20 ] 2>/dev/null; then
    warn "${AVAIL} GB free; traces and logs may not fit"
else
    ok "${AVAIL} GB free"
fi

# ---------------------------------------------------------------- verdict
echo
echo "======================================================================"
echo " ${PASS} ok / ${WARN} warn / ${FAIL} fail"
echo "======================================================================"
if [ "${FAIL}" -gt 0 ]; then
    echo " NOT READY — fix the failures above before spending GPU time."
    exit 1
fi
echo " READY."
[ "${WARN}" -gt 0 ] && echo " (${WARN} warnings — some tiers may be skipped)"
echo
echo " This script is called by ./run_session.sh, which runs the whole session unattended."
echo " Running it directly is only useful to check the machine before committing to a run."
exit 0
