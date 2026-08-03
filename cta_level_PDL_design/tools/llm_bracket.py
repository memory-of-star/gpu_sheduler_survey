#!/usr/bin/env python3
"""Compute the LLM PDL bracket — the decisive number of the whole project.

    Ceiling - PDL_grid = headroom still available to CTA-level dependency resolution
                         AFTER production grid-level PDL has taken its share.

Note the floor is PDL_grid, NOT PDL_off. TensorRT-LLM, vLLM and SGLang all ship grid-level
PDL today, so measuring against PDL-off would count gains that are already banked.

Published reference points for the PDL_off -> PDL_grid step:
    vLLM BS=1                  ~2-3% (up to ~10% in some configs)
    TRT-LLM DeepSeek-R1 / B200 ~3%   (168 -> 173 TPS/user)
    Triton simple kernels      ~15%
    Triton back-to-back layers up to 33%
If this campaign's PDL_off -> PDL_grid delta lands far outside that band, suspect the
configuration (e.g. PDL silently disabled because CUDA graphs were piecewise, not full).

Usage:
    python3 tools/llm_bracket.py bench/llm/results_llm/summary_llm.txt
"""

import argparse
import sys
from collections import defaultdict

sys.path.insert(0, __import__("os").path.dirname(__file__))
from analyze import parse_summary  # noqa: E402  (shared SUMMARY parser)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary")
    ap.add_argument("--metric", default="tok_per_s",
                    choices=["tok_per_s", "tok_per_s_per_user", "median_s"],
                    help="median_s is lower-is-better; the others higher-is-better")
    args = ap.parse_args()

    rows = [r for r in parse_summary(args.summary)
            if r.get("tier") == 4 and r.get("status") == "ok"]
    if not rows:
        print("no usable Tier-4 SUMMARY rows", file=sys.stderr)
        return 1

    lower_better = args.metric == "median_s"

    # (batch, seq) -> rung -> value
    grid = defaultdict(dict)
    for r in rows:
        grid[(r.get("batch"), r.get("seq"))][r.get("rung")] = r.get(args.metric)

    print(f"metric = {args.metric} ({'lower' if lower_better else 'higher'} is better)\n")
    print(f"{'batch':>6} {'seq':>8} {'pdl_off':>12} {'pdl_grid':>12} {'ceiling':>12} "
          f"{'grid_gain%':>11} {'headroom%':>11}")

    headrooms = []
    grid_gains = []
    for (bs, seq) in sorted(grid, key=lambda k: (k[1] or 0, k[0] or 0)):
        d = grid[(bs, seq)]
        off, flo, cei = d.get("pdl_off"), d.get("pdl_grid"), d.get("ceiling")
        if flo is None:
            continue

        def gain(a, b):
            """Improvement of b over a, sign-corrected for the metric direction."""
            if a is None or b is None or a == 0:
                return None
            return (a - b) / a * 100.0 if lower_better else (b - a) / a * 100.0

        g_grid = gain(off, flo)      # what production PDL already banked
        g_head = gain(flo, cei)      # what is LEFT for CTA-level
        if g_grid is not None:
            grid_gains.append(g_grid)
        if g_head is not None:
            headrooms.append(g_head)

        fmt = lambda v: f"{v:12.3f}" if v is not None else f"{'-':>12}"
        fmtp = lambda v: f"{v:11.2f}" if v is not None else f"{'-':>11}"
        print(f"{bs:>6} {seq:>8} {fmt(off)} {fmt(flo)} {fmt(cei)} "
              f"{fmtp(g_grid)} {fmtp(g_head)}")

    print("\n" + "=" * 74)
    print("Interpretation")
    print("=" * 74)

    if grid_gains:
        med = sorted(grid_gains)[len(grid_gains) // 2]
        print(f"grid-level PDL already banked a median {med:.2f}% "
              f"(range {min(grid_gains):.2f}% .. {max(grid_gains):.2f}%)")
        if not (0.0 <= med <= 40.0):
            print("  !! outside the published 2-33% band -- check that FULL CUDA graphs are on")
            print("     (vLLM only enables PDL under full graphs; piecewise mode is a net loss)")

    if headrooms:
        med = sorted(headrooms)[len(headrooms) // 2]
        print(f"\nHEADROOM for CTA-level: median {med:.2f}% "
              f"(range {min(headrooms):.2f}% .. {max(headrooms):.2f}%)")
        print()
        if med < 2.0:
            print("  VERDICT: headroom is small. Grid-level PDL has taken most of what is")
            print("  available on this workload; the CTA-level direction needs re-evaluating")
            print("  before investing in Tier 2/3.")
        elif med < 8.0:
            print("  VERDICT: modest headroom. Worth pursuing only where the microbenchmarks")
            print("  (Tier 1) show the pattern is favourable -- check dependency structure")
            print("  before committing.")
        else:
            print("  VERDICT: substantial headroom remains after production PDL. The")
            print("  CTA-level direction is worth pursuing on this workload.")

        best = max(zip(headrooms, sorted(grid, key=lambda k: (k[1] or 0, k[0] or 0))),
                   key=lambda t: t[0], default=None)
        if best:
            print(f"\n  Largest headroom at batch={best[1][0]} seq={best[1][1]} "
                  f"({best[0]:.2f}%).")
            if best[1][0] == 1:
                print("  BS=1 leading is the expected shape: smallest grids, GPU not full,")
                print("  most room for overlap. Matches vLLM's 'never hurts at low batch'.")
    else:
        print("no ceiling runs found -- rerun with CTA_PDL_CEILING=1 to get the headroom")

    return 0


if __name__ == "__main__":
    sys.exit(main())
