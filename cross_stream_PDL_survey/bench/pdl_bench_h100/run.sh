#!/usr/bin/env bash
# Run the PDL benchmark: one detailed config, then a sweep over tail=prologue spin length.
set -euo pipefail
cd "$(dirname "$0")"

BIN=./pdl_bench
if [ ! -x "$BIN" ]; then
  echo "Binary not found; building first..."
  ./build.sh
fi

echo "==================== detailed run (tail=prologue=20e6 cyc) ===================="
"$BIN" --repeats 50 --tail 20000000

echo
echo "==================== sweep: tail = prologue (cycles) ====================="
printf "%-12s %9s %9s %9s %9s %9s %11s\n" "cycles" "BASE" "PDL_XS" "PDL_GRPH" "PDL_SS" "CONC" "graph_spd"
for c in 1000000 2000000 5000000 10000000 20000000 40000000 80000000; do
  line=$("$BIN" --repeats 30 --tail "$c" --prologue "$c" | grep '^SUMMARY')
  base=$(echo "$line"  | sed -n 's/.*BASE=\([0-9.]*\).*/\1/p')
  xs=$(echo "$line"    | sed -n 's/.*PDL_XS=\([0-9.]*\).*/\1/p')
  gr=$(echo "$line"    | sed -n 's/.*PDL_GRAPH=\([0-9.]*\).*/\1/p')
  ss=$(echo "$line"    | sed -n 's/.*PDL_SS=\([0-9.]*\).*/\1/p')
  conc=$(echo "$line"  | sed -n 's/.*CONC=\([0-9.]*\).*/\1/p')
  spg=$(echo "$line"   | sed -n 's/.*speedup_graph=\([0-9.]*\).*/\1/p')
  printf "%-12s %9s %9s %9s %9s %9s %10sx\n" "$c" "$base" "$xs" "$gr" "$ss" "$conc" "$spg"
done

echo
echo "Interpretation:"
echo " - PDL_SS (same-stream) is the reference: it should hit ~2x (== CONC ceiling)."
echo " - PDL_GRAPH tests cross-NODE PDL via a graph programmatic edge (built directly)."
echo " - PDL_CAPTURE captures the SAME literal sA/sB two-stream code into a graph via"
echo "   cudaStreamBeginCapture; expected to match PDL_GRAPH (proving capture is what unlocks it)."
echo " - PDL_XS is eager cross-stream via a programmatic event; on some drivers this does NOT"
echo "   overlap (behaves like BASE) because programmatic events are meant to be captured into a graph."

echo
echo "==================== diamond: PDL edges vs ordinary edges ====================="
if [ -x ./pdl_diamond ]; then
  ./pdl_diamond --repeats 50 --tail 20000000
else
  echo "(./pdl_diamond not built; run ./build.sh)"
fi
