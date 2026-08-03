#!/usr/bin/env bash
# collect.sh — package every raw result for transfer back to the dev box.
#
# Analysis happens locally, so this only has to gather things reliably. It also prints a
# quick completeness report so a missing tier is noticed BEFORE the machine is released.
#
# Usage:  ./collect.sh [output.tar.gz]

SELF="$(cd "$(dirname "$0")" && pwd)"
cd "${SELF}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${1:-cta_pdl_results_${STAMP}.tar.gz}"
STAGE="/tmp/cta_collect_${STAMP}"

mkdir -p "${STAGE}"

echo "== collecting results"

# ---- raw result directories ----
for d in bench/results bench/llm/results_llm bench/dsa/results_dsa; do
    if [ -d "${d}" ]; then
        mkdir -p "${STAGE}/$(dirname "${d}")"
        cp -r "${d}" "${STAGE}/${d}"
        n=$(find "${d}" -name '*.done' 2>/dev/null | wc -l)
        echo "   ${d}: ${n} completed steps"
    else
        echo "   ${d}: MISSING"
    fi
done

# ---- merged SUMMARY stream (what the analysis actually consumes) ----
find bench -name 'summary*.txt' -exec cat {} \; 2>/dev/null > "${STAGE}/all_summary.txt"
# grep -c prints 0 AND exits non-zero on no-match, so `|| echo 0` would emit "0\n0".
count_matches() { local n; n=$(grep -c "$1" "$2" 2>/dev/null) || true; echo "${n:-0}"; }
NSUM=$(count_matches '^SUMMARY' "${STAGE}/all_summary.txt")
echo "   merged SUMMARY rows: ${NSUM}"

# ---- environment provenance ----
{
    echo "collected: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo
    echo "== nvidia-smi =="
    nvidia-smi 2>&1 || echo "(unavailable)"
    echo
    echo "== nvcc =="
    nvcc --version 2>&1 || echo "(unavailable)"
    echo
    echo "== python =="
    python3 --version 2>&1
    python3 -c "import torch;print('torch', torch.__version__)" 2>/dev/null || echo "torch: n/a"
    python3 -c "import vllm;print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: n/a"
} > "${STAGE}/environment.txt" 2>&1

# ---- nsys traces, if any (can be large) ----
NSYS_N=$(find bench -name '*.nsys-rep' -o -name '*.sqlite' 2>/dev/null | wc -l)
if [ "${NSYS_N}" -gt 0 ]; then
    SZ=$(find bench \( -name '*.nsys-rep' -o -name '*.sqlite' \) -exec du -ch {} + 2>/dev/null \
         | tail -1 | cut -f1)
    echo "   nsys artifacts: ${NSYS_N} files (${SZ})"
fi

# ---- completeness report ----
REPORT="${STAGE}/completeness.txt"
{
    echo "== completeness =="
    check() {
        local label="$1" pattern="$2" n
        n=$(count_matches "${pattern}" "${STAGE}/all_summary.txt")
        printf "  %-34s %5s rows  %s\n" "${label}" "${n}" \
               "$([ "${n}" -gt 0 ] && echo "ok" || echo "MISSING")"
    }
    check "Tier 0.1 chain overlap"     "tier0=chain"
    check "Tier 0.3 occupancy"         "tier0=occupancy"
    check "Tier 0.4 CLC"               "tier0=clc"
    check "Tier 0.5 fence"             "tier0=fence"
    check "Tier 1.1a degree sweep"     "tag=t11a_"
    check "Tier 1.1b structure sweep"  "tag=t11b_"
    check "Tier 1.2 tail/prologue"     "tag=t12_"
    check "Tier 2.1 protocols"         "tag=t21_"
    check "Tier 2.3 encoding"          "tag=t23_"
    check "Tier 0.3 under dependency"  "tag=t03_"
    check "Tier 4 LLM"                 "tier=4"
    check "Tier 5 DSA"                 "tier=5"
    echo
    echo "  CTA traces: $(find "${STAGE}" -name 'trace*.csv' 2>/dev/null | wc -l) files"
    FAILS=$(find "${STAGE}" -name 'failures.log' -size +0 2>/dev/null | wc -l)
    if [ "${FAILS}" -gt 0 ]; then
        echo
        echo "  !! failures were recorded:"
        find "${STAGE}" -name 'failures.log' -size +0 -exec cat {} \; 2>/dev/null | sed 's/^/     /'
    fi
} > "${REPORT}"
cat "${REPORT}"

# ---- pack ----
tar czf "${OUT}" -C "$(dirname "${STAGE}")" "$(basename "${STAGE}")"
rm -rf "${STAGE}"

echo
echo "== packed: ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo
echo "On the dev box:"
echo "  tar xzf ${OUT} && cd cta_collect_${STAMP}"
echo "  python3 tools/analyze.py      all_summary.txt --csv all.csv --json findings.json"
echo "  python3 tools/cta_timeline.py bench/results/trace_*.csv --plot concurrency.png"
echo "  python3 tools/llm_bracket.py  all_summary.txt"
