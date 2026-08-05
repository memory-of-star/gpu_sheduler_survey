#!/usr/bin/env python3
"""Turn the Tier 1 pilot result into a machine-readable gate verdict.

EXPERIMENT_PLAN.md section 6 defines the decision as a prose table. An unattended session
cannot read a prose table, so this computes the same verdict and exposes it three ways:
stdout for the log, JSON for the report, and an exit code for run_session.sh.

    verdict     typical space_pct   exit   meaning
    GO          >= 8               0       run everything, spend the full budget
    LLM_ONLY    2 .. 8             0       skip Tier 2/3, go straight to Tier 4 confirmation
    STOP        < 2                0       stop; re-evaluate the direction
    INVALID     any                2       a config failed validation -- no timing is usable

"space" is Ceiling - Floor as a percentage of Floor, i.e. the total headroom the workload
offers. The statistic is the MEDIAN across configurations, matching the plan's wording
("in most configurations"). Exit code 0 means the gate was evaluated, not that it passed;
the caller branches on the verdict string. Only a broken input exits non-zero.

Usage:
    python3 tools/gate.py bench/results/pilot_analysis.json
    python3 tools/gate.py bench/results/pilot_analysis.json --json bench/results/gate.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Thresholds are owned by EXPERIMENT_PLAN.md section 6. Changing them here without changing
# that section makes the code and the methodology disagree.
GO_THRESHOLD = 8.0
STOP_THRESHOLD = 2.0

# Plan §5.3 required multi-wave ratios relative to SM count.
PLAN_MULTI_RATIOS = (2, 8, 32)


def classify(space_pct: float) -> str:
    if space_pct >= GO_THRESHOLD:
        return "GO"
    if space_pct >= STOP_THRESHOLD:
        return "LLM_ONLY"
    return "STOP"


NEXT_STEP = {
    "GO": "run Tier 2/3 + Tier 4 + Tier 5, spend the full budget",
    "LLM_ONLY": "skip Tier 2/3; run Tier 4 end-to-end to see what real workloads leave",
    "STOP": "stop and re-evaluate the direction; at most confirm with the Tier 4 three rungs",
    "INVALID": "do not use any timing from this run; fix correctness first",
}


def _meta_int(meta: dict, key: str) -> int | None:
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def wave_coverage(configurations: dict) -> dict:
    """Classify P,C vs SM coverage against EXPERIMENT_PLAN.md §5.3."""
    ratios: list[float] = []
    for cfg in configurations.values():
        meta = cfg.get("metadata") or {}
        producers = _meta_int(meta, "producers")
        consumers = _meta_int(meta, "consumers")
        sms = _meta_int(meta, "sms")
        if producers is None or consumers is None or not sms:
            continue
        ratios.append(max(producers, consumers) / sms)

    if not ratios:
        return {
            "single_wave_only": False,
            "has_multi_wave": False,
            "plan_multi_ratios_present": [],
            "plan_multi_complete": False,
            "max_pc_over_sm": None,
        }

    present = []
    for req in PLAN_MULTI_RATIOS:
        if any(abs(r - req) <= 0.05 * req for r in ratios):
            present.append(req)

    return {
        "single_wave_only": all(r <= 1.0 + 1e-9 for r in ratios),
        "has_multi_wave": any(r > 1.0 + 1e-9 for r in ratios),
        "plan_multi_ratios_present": present,
        "plan_multi_complete": present == list(PLAN_MULTI_RATIOS),
        "max_pc_over_sm": max(ratios),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", type=Path, help="pilot_analysis.json from analyze_pilot.py")
    ap.add_argument("--json", type=Path, help="also write the verdict as JSON")
    args = ap.parse_args()

    try:
        data = json.loads(args.analysis.read_text())
    except FileNotFoundError:
        print(f"gate: {args.analysis} not found -- did tier1p produce a pilot matrix?")
        return 2
    except json.JSONDecodeError as exc:
        print(f"gate: {args.analysis} is not valid JSON: {exc}")
        return 2

    aggregates = data.get("aggregates") or {}
    if not aggregates:
        print(f"gate: {args.analysis} contains no configurations")
        return 2

    spaces, captures, invalid = [], [], []
    for base_tag, agg in sorted(aggregates.items()):
        across = agg.get("across_seed", {})
        space = across.get("space_pct", {}).get("median")
        captured = across.get("captured_pct", {}).get("median")
        if space is None:
            continue
        spaces.append(space)
        if captured is not None:
            captures.append(captured)
        if not agg.get("all_valid", False):
            invalid.append(base_tag)

    if not spaces:
        print(f"gate: {args.analysis} has no space_pct medians to judge")
        return 2

    median_space = statistics.median(spaces)
    median_captured = statistics.median(captures) if captures else 0.0
    verdict = "INVALID" if invalid else classify(median_space)
    waves = wave_coverage(data.get("configurations") or {})

    print("=" * 70)
    print(f"TIER 1 GATE  verdict={verdict}")
    print("=" * 70)
    print(f"  configurations      : {len(spaces)}")
    print(f"  median space%       : {median_space:.2f}   (Ceiling - Floor, over Floor)")
    print(f"  median captured%    : {median_captured:.2f}   (Floor - Impl, over Floor)")
    print(f"  range space%        : {min(spaces):.2f} .. {max(spaces):.2f}")
    print(f"  thresholds          : GO >= {GO_THRESHOLD}, STOP < {STOP_THRESHOLD}"
          f"  (EXPERIMENT_PLAN.md section 6)")
    if waves["max_pc_over_sm"] is not None:
        print(f"  max(P,C)/SM         : {waves['max_pc_over_sm']:.2f}"
              f"  multi={int(waves['has_multi_wave'])}"
              f"  plan_2/8/32x={waves['plan_multi_ratios_present']}")
    if invalid:
        print(f"  FAILED VALIDATION   : {', '.join(invalid)}")
    print()
    print(f"  next: {NEXT_STEP[verdict]}")
    if waves["single_wave_only"] and verdict != "INVALID":
        print()
        print("  CAVEAT: every configuration ran with P,C <= SM, i.e. single-wave and")
        print("  underfilled/full only. This verdict covers mechanism feasibility only.")
        print("  The multi-wave regime (plan §5.3: 2x/8x/32x SM) is NOT measured here, so")
        print("  a GO must not be read as permission to spend the full Tier 2/3 budget.")
    elif waves["has_multi_wave"] and not waves["plan_multi_complete"] and verdict != "INVALID":
        print()
        print("  CAVEAT: some multi-wave points exist, but plan §5.3's 2x/8x/32x SM set is")
        print(f"  incomplete (have {waves['plan_multi_ratios_present'] or 'none'}). Treat the")
        print("  verdict as provisional until the full multi-wave grid map is present.")
    print("=" * 70)

    if args.json:
        args.json.write_text(json.dumps({
            "verdict": verdict,
            "next_step": NEXT_STEP[verdict],
            "median_space_pct": median_space,
            "median_captured_pct": median_captured,
            "min_space_pct": min(spaces),
            "max_space_pct": max(spaces),
            "configuration_count": len(spaces),
            "failed_validation": invalid,
            "single_wave_only": waves["single_wave_only"],
            "has_multi_wave": waves["has_multi_wave"],
            "plan_multi_ratios_present": waves["plan_multi_ratios_present"],
            "plan_multi_complete": waves["plan_multi_complete"],
            "max_pc_over_sm": waves["max_pc_over_sm"],
            "thresholds": {"go": GO_THRESHOLD, "stop": STOP_THRESHOLD},
            "source": str(args.analysis),
        }, indent=2, sort_keys=True) + "\n")

    return 2 if verdict == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
