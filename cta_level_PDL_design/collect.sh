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

# ---- merged record streams (what the analysis actually consumes) ----
# Two schemas, two files. analyze.py reads SUMMARY (tier0_facts, and the rejected
# cta_dep_bench); analyze_pilot.py reads SAMPLE/SUMMARY_PILOT (cta_dep_pilot, the Tier 1
# gate). Merging them into one file would let pilot rows leak into the analyze.py CSV.
find bench -name 'summary*.txt' -exec cat {} \; 2>/dev/null > "${STAGE}/all_summary.txt"
find bench -name 'pilot_matrix.log' -exec cat {} \; 2>/dev/null > "${STAGE}/all_pilot.log"
# grep -c prints 0 AND exits non-zero on no-match, so `|| echo 0` would emit "0\n0".
count_matches() { local n; n=$(grep -c "$1" "$2" 2>/dev/null) || true; echo "${n:-0}"; }
NSUM=$(count_matches '^SUMMARY ' "${STAGE}/all_summary.txt")
NPILOT=$(count_matches '^SUMMARY_PILOT ' "${STAGE}/all_pilot.log")
echo "   merged SUMMARY rows: ${NSUM}"
echo "   merged pilot configs: ${NPILOT}"

# ---- the gate verdict, the single most important artefact of the session ----
GATE_SRC=$(find bench -name 'gate.json' 2>/dev/null | head -1)
if [ -n "${GATE_SRC}" ]; then
    cp "${GATE_SRC}" "${STAGE}/gate.json"
    VERDICT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['verdict'])" \
              "${GATE_SRC}" 2>/dev/null || echo "unparseable")
    echo "   gate verdict: ${VERDICT}"
else
    echo "   gate verdict: MISSING (tier1p did not reach a decision)"
fi

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
        local label="$1" pattern="$2" file="${3:-${STAGE}/all_summary.txt}" n
        n=$(count_matches "${pattern}" "${file}")
        printf "  %-34s %5s rows  %s\n" "${label}" "${n}" \
               "$([ "${n}" -gt 0 ] && echo "ok" || echo "MISSING")"
    }
    echo "  -- Tier 0 (valid harness) --"
    check "Tier 0.1 chain overlap"     "tier0=chain"
    check "Tier 0.3 occupancy"         "tier0=occupancy"
    check "Tier 0.4 CLC"               "tier0=clc"
    check "Tier 0.5 fence"             "tier0=fence"
    # Count SUMMARY_PILOT only: one per configuration. Matching a bare tag= would also hit
    # every SAMPLE line and report a count nobody can interpret.
    echo "  -- Tier 1p (corrected pilot: the gate data) --"
    check "Tier 1.1p degree axis"      "^SUMMARY_PILOT .*tag=t11p_g"  "${STAGE}/all_pilot.log"
    check "Tier 1.1p structure axis"   "^SUMMARY_PILOT .*tag=t11ps_"  "${STAGE}/all_pilot.log"
    check "Tier 1.2p tail/prologue"    "^SUMMARY_PILOT .*tag=t12p_"   "${STAGE}/all_pilot.log"
    echo "  -- rejected harness (present only if re-audited) --"
    check "Tier 1.1a (cta_dep_bench)"  "tag=t11a_"
    check "Tier 2.1 protocols"         "tag=t21_"
    check "Tier 2.3 encoding"          "tag=t23_"
    echo "  -- later sessions --"
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
echo "Re-analysing this archive anywhere (no GPU needed):"
echo "  tar xzf ${OUT} && cd cta_collect_${STAMP}"
echo "  cat gate.json                                  # the verdict"
echo "  python3 tools/analyze_pilot.py all_pilot.log \\"
echo "          --json pilot_analysis.json --csv pilot_summary.csv"
echo "  python3 tools/gate.py         pilot_analysis.json"
echo "  python3 tools/analyze.py      all_summary.txt --csv all.csv --json findings.json"
