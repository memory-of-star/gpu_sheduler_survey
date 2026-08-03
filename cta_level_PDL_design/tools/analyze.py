#!/usr/bin/env python3
"""Analyze the SUMMARY lines produced by the CTA-PDL benchmark campaign.

Every benchmark prints exactly one machine-parsable line per configuration:

    SUMMARY key=value key=value ...

so this script never has to parse prose. It reconstructs the four-point bracket and answers
the decisive questions of the evaluation plan:

  * Tier 1.1  Where are the benefit boundaries on THIS device?
              BlockMaestro found "degree > 32 or grid > 2048 TB => no benefit" on a 28-SM
              Titan X. B300 has ~148 SMs. This script locates the boundaries empirically.
  * The degree/structure separation: because the sweep pins one axis while moving the other,
    the output can say whether a benefit loss came from MORE EDGES or from a HARDER SHAPE --
    which BlockMaestro's n-group injection could not distinguish.
  * Tier 2.1  Which sync protocol wins, and by how much.
  * Tier 0.3  Occupancy cost of waiting while resident (the pricing basis for B2).

Usage:
    python3 tools/analyze.py results/summary.txt
    python3 tools/analyze.py results/summary.txt --csv out.csv --json out.json
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict


def _coerce(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


# key=value, where value is either bare or double-quoted (quoted values may contain spaces,
# e.g. name="NVIDIA B300").
_KV = __import__("re").compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def parse_summary(path):
    """Parse every 'SUMMARY k=v ...' line into a dict, coercing numbers."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("SUMMARY"):
                continue
            d = {}
            for k, v in _KV.findall(line[len("SUMMARY"):]):
                if v.startswith('"') and v.endswith('"'):
                    d[k] = v[1:-1]
                else:
                    d[k] = _coerce(v)
            if d:
                rows.append(d)
    return rows


# --------------------------------------------------------------------- bracket

WAIT_COLS = {
    "spin_ms": "cta-spin",
    "backoff_ms": "cta-backoff",
    "counter_ms": "cta-counter",
    "exact_ms": "cta-exact",
}


def bracket(r):
    """Four-point bracket for one config. Returns None if the run is incomplete."""
    floor = r.get("floor_ms", 0.0)
    ceil = r.get("ceiling_ms", 0.0)
    if not floor or not ceil or floor <= 0:
        return None
    impls = {name: r[col] for col, name in WAIT_COLS.items()
             if r.get(col, 0.0) and r[col] > 0}
    best_name, best_ms = (None, None)
    if impls:
        best_name = min(impls, key=impls.get)
        best_ms = impls[best_name]
    space = (floor - ceil) / floor
    captured = (floor - best_ms) / floor if best_ms else 0.0
    return {
        "floor_ms": floor,
        "ceiling_ms": ceil,
        "space_pct": 100.0 * space,
        "best_impl": best_name,
        "best_impl_ms": best_ms,
        "captured_pct": 100.0 * captured,
        "captured_of_space_pct": (100.0 * captured / space) if space > 0 else 0.0,
        "impls": impls,
    }


# --------------------------------------------------------------------- Tier 1

def tier1_degree(rows, out):
    """Degree sweep with structure pinned -> where does the degree boundary sit?"""
    sel = [r for r in rows if str(r.get("tag", "")).startswith("t11a_")]
    if not sel:
        return
    print("\n" + "=" * 78)
    print("Tier 1.1a  DEGREE sweep (structure pinned to 'interval')")
    print("=" * 78)
    print("Isolates 'more edges' from 'harder shape'. Structure stays a contiguous interval,")
    print("so any benefit loss here is attributable to DEGREE ALONE.\n")

    by_grid = defaultdict(list)
    for r in sel:
        b = bracket(r)
        if b:
            by_grid[r.get("consumers", 0)].append((r.get("degree", 0), b, r))

    print(f"{'grid':>7} {'degree':>7} {'space%':>9} {'captured%':>10} "
          f"{'of space%':>10} {'tightness':>10} {'best':>13}")
    boundaries = {}
    for g in sorted(by_grid):
        prev_positive = None
        for deg, b, r in sorted(by_grid[g]):
            print(f"{g:>7} {deg:>7} {b['space_pct']:>9.2f} {b['captured_pct']:>10.2f} "
                  f"{b['captured_of_space_pct']:>10.1f} {r.get('tightness', 0):>10.3f} "
                  f"{str(b['best_impl']):>13}")
            # "Boundary" = first degree at which the available space drops below 2%.
            if b["space_pct"] >= 2.0:
                prev_positive = deg
            elif g not in boundaries and prev_positive is not None:
                boundaries[g] = (prev_positive, deg)
        print()

    if boundaries:
        print("Degree boundary (last degree with >=2% space -> first below):")
        for g, (lo, hi) in sorted(boundaries.items()):
            print(f"  grid={g:>6}: benefit persists to degree {lo}, gone by {hi}")
        print("\nCompare against BlockMaestro's 'degree > 32' on a 28-SM Titan X.")
        print("A boundary well above 32 means the threshold scales with SM count and the")
        print("high-degree-but-regular LLM patterns (GEMM chain, DSA indexer) are IN scope.")
    out["tier1_degree_boundaries"] = {str(k): v for k, v in boundaries.items()}


def tier1_structure(rows, out):
    """Structure sweep with degree pinned -> how much does SHAPE alone cost?"""
    sel = [r for r in rows if str(r.get("tag", "")).startswith("t11b_")]
    if not sel:
        return
    print("\n" + "=" * 78)
    print("Tier 1.1b  STRUCTURE sweep (degree pinned to 32)")
    print("=" * 78)
    print("Degree is constant, so differences here are attributable to SHAPE ALONE.")
    print("This is the comparison BlockMaestro's n-group injection could not make.\n")

    print(f"{'grid':>7} {'structure':>10} {'space%':>9} {'captured%':>10} "
          f"{'tightness':>10} {'eff_deg':>9} {'best':>13}")
    by_struct = defaultdict(list)
    for r in sorted(sel, key=lambda x: (x.get("consumers", 0), str(x.get("structure", "")))):
        b = bracket(r)
        if not b:
            continue
        print(f"{r.get('consumers', 0):>7} {str(r.get('structure')):>10} "
              f"{b['space_pct']:>9.2f} {b['captured_pct']:>10.2f} "
              f"{r.get('tightness', 0):>10.3f} {r.get('eff_degree', 0):>9.1f} "
              f"{str(b['best_impl']):>13}")
        by_struct[str(r.get("structure"))].append(b["captured_pct"])

    if by_struct:
        print("\nMedian captured% by structure (same degree throughout):")
        for s, vals in sorted(by_struct.items(), key=lambda kv: -statistics.median(kv[1])):
            print(f"  {s:>10}: {statistics.median(vals):6.2f}%")
        print("\n'tightness' is degree/interval_width: 1.0 means interval encoding is exact.")
        print("Low tightness with low captured% = the ENCODING is the bottleneck (dimension A3),")
        print("not the dependency degree.")
    out["tier1_structure"] = {k: statistics.median(v) for k, v in by_struct.items()}


def tier1_grid(rows, out):
    """Grid-size boundary: when the GPU fills up, run-ahead space disappears."""
    sel = [r for r in rows if str(r.get("tag", "")).startswith("t11a_")]
    if not sel:
        return
    by_deg = defaultdict(list)
    for r in sel:
        b = bracket(r)
        if b:
            by_deg[r.get("degree", 0)].append((r.get("consumers", 0), b["space_pct"]))
    if not by_deg:
        return
    print("\n" + "=" * 78)
    print("Tier 1.1c  GRID-SIZE boundary")
    print("=" * 78)
    print("BlockMaestro: benefit vanished past ~2048 TBs on 28 SMs (896 concurrent slots).")
    print("B300 has ~148 SMs, so the boundary should move if it tracks machine capacity.\n")
    print(f"{'degree':>7} {'grid where space<2%':>22}")
    boundaries = {}
    for d in sorted(by_deg):
        pts = sorted(by_deg[d])
        cut = next((g for g, sp in pts if sp < 2.0), None)
        boundaries[d] = cut
        print(f"{d:>7} {str(cut) if cut else '(still >2% at max)':>22}")
    out["tier1_grid_boundaries"] = {str(k): v for k, v in boundaries.items()}


# --------------------------------------------------------------------- Tier 2 / 0.3

def tier2_protocols(rows, out):
    sel = [r for r in rows if str(r.get("tag", "")).startswith(("t21_", "t23_"))]
    if not sel:
        return
    print("\n" + "=" * 78)
    print("Tier 2  sync protocol + encoding cost")
    print("=" * 78)
    print(f"{'tag':>18} {'struct':>10} {'deg':>5} "
          f"{'spin':>9} {'backoff':>9} {'counter':>9} {'exact':>9} {'floor':>9}")
    wins = defaultdict(int)
    for r in sorted(sel, key=lambda x: str(x.get("tag"))):
        b = bracket(r)
        if not b:
            continue
        print(f"{str(r.get('tag')):>18} {str(r.get('structure')):>10} "
              f"{r.get('degree', 0):>5} "
              f"{r.get('spin_ms', 0):>9.4f} {r.get('backoff_ms', 0):>9.4f} "
              f"{r.get('counter_ms', 0):>9.4f} {r.get('exact_ms', 0):>9.4f} "
              f"{r.get('floor_ms', 0):>9.4f}")
        if b["best_impl"]:
            wins[b["best_impl"]] += 1
    if wins:
        print("\nProtocol win counts:", dict(wins))
        print("cta-exact beating cta-spin means interval over-approximation is costing real")
        print("time -> the encoding choice (A3) matters for this pattern.")
    out["tier2_protocol_wins"] = dict(wins)


def tier0_occupancy(rows, out):
    occ = [r for r in rows if r.get("tier0") == "occupancy"]
    dep = [r for r in rows if str(r.get("tag", "")).startswith("t03_smem")]
    if not occ and not dep:
        return
    print("\n" + "=" * 78)
    print("Tier 0.3  occupancy cost of a resident-but-waiting CTA  (pricing basis for B2)")
    print("=" * 78)
    if occ:
        print(f"{'smem_kb':>8} {'threads':>8} {'ctas/SM':>9} {'concurrent':>11} {'median_ms':>11}")
        for r in sorted(occ, key=lambda x: (x.get("threads", 0), x.get("smem_kb", 0))):
            print(f"{r.get('smem_kb', 0):>8} {r.get('threads', 0):>8} "
                  f"{r.get('ctas_per_sm', 0):>9} {r.get('concurrent_ctas', 0):>11} "
                  f"{r.get('median_ms', 0):>11.4f}")
    if dep:
        print(f"\nUnder a real dependency wait:")
        print(f"{'smem_kb':>8} {'ctas/SM':>9} {'space%':>9} {'captured%':>10}")
        for r in sorted(dep, key=lambda x: x.get("smem_kb", 0)):
            b = bracket(r)
            if b:
                print(f"{r.get('smem_kb', 0):>8} {r.get('occ_per_sm', 0):>9} "
                      f"{b['space_pct']:>9.2f} {b['captured_pct']:>10.2f}")
        print("\nFalling captured% as smem grows = the cost of holding a slot while waiting.")
        print("That delta is what pre-dispatch gating ([H+], BlockMaestro's choice) would buy.")


def tier0_chain(rows, out):
    ch = [r for r in rows if r.get("tier0") == "chain"]
    if not ch:
        return
    print("\n" + "=" * 78)
    print("Tier 0.1  same-stream chain overlap depth  (which B3 options are reachable)")
    print("=" * 78)
    print(f"{'stages':>7} {'pdl_off_ms':>12} {'pdl_on_ms':>12} {'speedup':>9} {'implied_depth':>14}")
    depths = []
    for r in sorted(ch, key=lambda x: x.get("stages", 0)):
        print(f"{r.get('stages', 0):>7} {r.get('pdl_off_ms', 0):>12.4f} "
              f"{r.get('pdl_on_ms', 0):>12.4f} {r.get('speedup', 0):>9.3f} "
              f"{r.get('implied_depth', 0):>14.2f}")
        depths.append(r.get("implied_depth", 0))
    if depths:
        print(f"\nMax implied overlap depth: {max(depths):.2f}")
        print("If this saturates near 2, B300 only ever overlaps a PAIR of kernels and the")
        print("deeper-window options of dimension B3 are unreachable without new mechanisms.")
    out["tier0_max_overlap_depth"] = max(depths) if depths else None


def tier0_clc(rows, out):
    clc = [r for r in rows if r.get("tier0") == "clc"]
    if not clc:
        return
    print("\n" + "=" * 78)
    print("Tier 0.4  CLC try_cancel  (feasibility of the software-scheduling route for B4)")
    print("=" * 78)
    for r in clc:
        if r.get("status") == "skipped":
            print(f"  skipped (cc {r.get('cc')} < 10.0)")
            continue
        print(f"  clusters={r.get('clusters')} median={r.get('median_ms', 0):.4f} ms")
        print(f"  attempts={r.get('attempts')} successes={r.get('successes')} "
              f"success_rate={r.get('success_rate', 0):.3f}")
        print(f"  cycles/attempt={r.get('cyc_per_attempt', 0):.1f}")
        dup = r.get("duplicate_claims", 0)
        print(f"  single-winner arbiter: duplicates={dup} "
              f"{'OK' if dup == 0 else 'VIOLATION'}")
        out["tier0_clc_cyc_per_attempt"] = r.get("cyc_per_attempt")


# --------------------------------------------------------------------- main

def write_csv(rows, path):
    import csv as _csv
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", help="results/summary.txt")
    ap.add_argument("--csv", help="dump all parsed rows to CSV")
    ap.add_argument("--json", help="dump the derived findings to JSON")
    args = ap.parse_args()

    try:
        rows = parse_summary(args.summary)
    except OSError as e:
        print(f"cannot read {args.summary}: {e}", file=sys.stderr)
        return 1
    if not rows:
        print("no SUMMARY lines found", file=sys.stderr)
        return 1

    print(f"parsed {len(rows)} SUMMARY rows from {args.summary}")
    dev = next((r for r in rows if r.get("tier0") == "device"), None)
    if dev:
        print(f"device: {dev.get('name')} SMs={dev.get('sms')} "
              f"cc={dev.get('cc')} clock={dev.get('ghz')} GHz")

    out = {}
    tier0_chain(rows, out)
    tier0_clc(rows, out)
    tier1_degree(rows, out)
    tier1_structure(rows, out)
    tier1_grid(rows, out)
    tier2_protocols(rows, out)
    tier0_occupancy(rows, out)

    if args.csv:
        write_csv(rows, args.csv)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
