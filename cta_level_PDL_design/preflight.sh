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
        for b in cta_dep_bench tier0_facts; do
            [ -x "bench/${b}" ] && ok "  bench/${b}" || bad "  bench/${b} missing"
        done
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
if python3 tools/dep_oracle.py --model qwen3.6-27b --tokens 256 --seq 2048 > /dev/null 2>&1; then
    ok "dep_oracle (offline dependency derivation)"
else
    bad "dep_oracle failed"
fi

# ---------------------------------------------------------------- 5. LLM deps (optional)
echo
echo "5. LLM stack (only needed for Tier 4)"
if python3 -c "import torch" 2>/dev/null; then
    TV=$(python3 -c "import torch;print(torch.__version__)")
    ok "torch ${TV}"
    python3 -c "import torch;exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
        && ok "torch sees CUDA" || bad "torch cannot see CUDA"
else
    warn "torch missing (Tier 4/5 unavailable; Tier 0-3 unaffected)"
fi
python3 -c "import vllm" 2>/dev/null && ok "vllm present" \
    || warn "vllm missing — pip install vllm before Tier 4"

MODEL_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "${MODEL_DIR}" ]; then
    SZ=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)
    ok "HF cache at ${MODEL_DIR} (${SZ}) — Qwen3.6-27B BF16 needs ~54GB"
else
    warn "no HF cache yet; start the download NOW, it is usually the slowest step:"
    echo "         huggingface-cli download Qwen/Qwen3.6-27B &"
fi

# ---------------------------------------------------------------- 6. misc
echo
echo "6. environment"
command -v nsys >/dev/null 2>&1 && ok "nsys present" \
    || warn "nsys missing (kernel-level capture will be skipped)"
command -v ncu  >/dev/null 2>&1 && ok "ncu present"  \
    || warn "ncu missing (per-kernel L2/DRAM metrics unavailable)"
command -v tmux >/dev/null 2>&1 && ok "tmux present" \
    || warn "no tmux — a dropped connection will kill the campaign"

AVAIL=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${AVAIL}" -lt 150 ] 2>/dev/null; then
    warn "${AVAIL} GB free; model (54GB) + traces may not fit"
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
echo
echo " Next:"
echo "   huggingface-cli download Qwen/Qwen3.6-27B &   # background, Tier 0/1 do not need it"
echo "   cd bench && FAST=1 ./run_all.sh               # smoke test, ~5 min"
[ "${WARN}" -gt 0 ] && echo "   (${WARN} warnings — some tiers may be skipped)"
exit 0
