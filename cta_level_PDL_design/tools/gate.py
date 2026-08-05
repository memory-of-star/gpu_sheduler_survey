#!/usr/bin/env python3
"""Turn the Tier 1 pilot result into a machine-readable gate verdict.

EXPERIMENT_PLAN.md section 6 defines the decision as a prose table. An unattended session
cannot read a prose table, so this computes the same verdict and exposes it three ways:
stdout for the log, JSON for the report, and an exit code for run_session.sh.

    verdict     typical space_pct   exit   meaning
    GO          >= 8               0       run everything, spend the full budget
    LLM_ONLY    2 .. 8             0       skip Tier 2/3, go straight to Tier 4 confirmation
    STOP        < 2                0       stop; re-evaluate the direction
    INVALID     any                2       a required admission check failed -- no timing is usable

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
PLAN_DEGREES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
PLAN_STRUCTURES = ("self", "interval", "grouped", "strided", "random")


def is_tier11_tag(tag: str) -> bool:
    """Only EXPERIMENT_PLAN §5.1 rows contribute to the §6 median."""
    return tag.startswith("t11p_g") or tag.startswith("t11ps_g")


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
    "INVALID": "do not use any timing from this run; fix the admissibility failure first",
}


def _meta_int(meta: dict, key: str) -> int | None:
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def semantic_meta_errors(meta: dict, require_unstarted: bool = False) -> list[str]:
    """Return missing/incorrect §3 proof fields for one pilot configuration."""
    errors = []
    required_text = {
        "launch_gate": "trace_verified",
        "timer": "globaltimer",
        "floor_path": "programmatic_graph",
        "impl_path": "priority_streams",
        "ceiling_path": "priority_streams",
        "trigger_floor": "ready",
        "trigger_impl": "entry",
        "trigger_ceiling": "entry",
    }
    for key, expected in required_text.items():
        if meta.get(key) != expected:
            errors.append(f"{key}!={expected}")
    for key in (
        "producer_progress_complete",
        "floor_early_launch_proven",
        "ceiling_wrong",
        "producer_slot_reserved",
    ):
        if _meta_int(meta, key) != 1:
            errors.append(f"{key}!=1")
    if (_meta_int(meta, "consumer_smem_kb") or 0) <= 0:
        errors.append("consumer_smem_kb<=0")
    trace_retries = _meta_int(meta, "trace_retries")
    retry_limit = _meta_int(meta, "trace_retry_limit")
    max_attempts = _meta_int(meta, "trace_max_attempts")
    max_observed = _meta_int(meta, "trace_max_attempts_observed")
    if trace_retries is None or trace_retries < 0:
        errors.append("trace_retries<0-or-missing")
    if retry_limit != 3:
        errors.append("trace_retry_limit!=3")
    if max_attempts != 4:
        errors.append("trace_max_attempts!=4")
    if max_observed is None or not 1 <= max_observed <= 4:
        errors.append("trace_max_attempts_observed outside 1..4")
    if require_unstarted:
        if _meta_int(meta, "multiwave_overlap_proven") != 1:
            errors.append("multiwave_overlap_proven!=1")
        if (_meta_int(meta, "producers_unstarted_at_consumer") or 0) <= 0:
            errors.append("producers_unstarted_at_consumer<=0")
    return errors


def semantic_coverage(configurations: dict) -> dict:
    failures = {}
    for tag, cfg in configurations.items():
        meta = cfg.get("metadata") or {}
        producers = _meta_int(meta, "producers") or 0
        consumers = _meta_int(meta, "consumers") or 0
        sms = _meta_int(meta, "sms") or 0
        is_multi = sms > 0 and producers > sms and consumers > sms
        errors = semantic_meta_errors(meta, require_unstarted=is_multi)
        if errors:
            failures[tag] = errors
    return {
        "semantic_proof_complete": not failures,
        "semantic_proof_failures": failures,
    }


def wave_coverage(configurations: dict) -> dict:
    """Classify P,C vs SM coverage against EXPERIMENT_PLAN.md §5.3."""
    ratios: list[float] = []
    proven_ratios: list[float] = []
    for cfg in configurations.values():
        meta = cfg.get("metadata") or {}
        producers = _meta_int(meta, "producers")
        consumers = _meta_int(meta, "consumers")
        sms = _meta_int(meta, "sms")
        if producers is None or consumers is None or not sms:
            continue
        ratio = max(producers, consumers) / sms
        ratios.append(ratio)
        proven = not semantic_meta_errors(meta, require_unstarted=True)
        if proven:
            proven_ratios.append(ratio)

    if not ratios:
        return {
            "pc_ratio_lte_one_only": False,
            "has_multi_wave": False,
            "has_proven_multi_wave": False,
            "nominal_multi_ratios_present": [],
            "plan_multi_ratios_present": [],
            "plan_multi_complete": False,
            "max_pc_over_sm": None,
            "underfilled_ratios": [],
            "has_single_full": False,
            "plan_grid_complete": False,
        }

    nominal_present = []
    present = []
    for req in PLAN_MULTI_RATIOS:
        if any(abs(r - req) <= 0.05 * req for r in ratios):
            nominal_present.append(req)
        if any(abs(r - req) <= 0.05 * req for r in proven_ratios):
            present.append(req)

    underfilled = sorted({round(r, 6) for r in ratios if r < 1.0 - 1e-9})
    has_single_full = any(abs(r - 1.0) <= 0.01 for r in ratios)
    return {
        "pc_ratio_lte_one_only": all(r <= 1.0 + 1e-9 for r in ratios),
        "has_multi_wave": any(r > 1.0 + 1e-9 for r in ratios),
        "has_proven_multi_wave": any(r > 1.0 + 1e-9 for r in proven_ratios),
        "nominal_multi_ratios_present": nominal_present,
        "plan_multi_ratios_present": present,
        "plan_multi_complete": present == list(PLAN_MULTI_RATIOS),
        "max_pc_over_sm": max(ratios),
        "underfilled_ratios": underfilled,
        "has_single_full": has_single_full,
        "plan_grid_complete": len(underfilled) >= 2 and has_single_full
            and present == list(PLAN_MULTI_RATIOS),
    }


def axis_coverage(configurations: dict) -> dict:
    degrees, structures = set(), set()
    for tag, cfg in configurations.items():
        meta = cfg.get("metadata") or {}
        if tag.startswith("t11p_g"):
            degree = _meta_int(meta, "degree")
            if degree is not None:
                degrees.add(degree)
            if degree == 32 and meta.get("structure") == "interval":
                # This physical point also supplies the interval member of the
                # structure axis; run_all deliberately does not duplicate it.
                structures.add("interval")
        elif tag.startswith("t11ps_g"):
            structure = meta.get("structure")
            if structure:
                structures.add(str(structure))
    return {
        "plan_degrees_present": sorted(degrees),
        "plan_degree_complete": all(d in degrees for d in PLAN_DEGREES),
        "plan_structures_present": sorted(structures),
        "plan_structure_complete": all(s in structures for s in PLAN_STRUCTURES),
    }


def tail_coverage(configurations: dict) -> dict:
    """§5.1 gate points need a real post-ready overlap window; tail=0 is control only."""
    invalid = []
    for tag, cfg in configurations.items():
        tail = _meta_int(cfg.get("metadata") or {}, "tail")
        if tail is None or tail <= 0:
            invalid.append(tag)
    return {
        "tier11_positive_tail": not invalid,
        "tier11_nonpositive_tail": sorted(invalid),
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

    tier11_aggregates = {
        tag: agg for tag, agg in aggregates.items() if is_tier11_tag(tag)
    }
    if not tier11_aggregates:
        print(f"gate: {args.analysis} contains no Tier 1.1 configurations")
        return 2

    spaces, captures = [], []
    invalid = [tag for tag, agg in sorted(aggregates.items())
               if not agg.get("all_valid", False)]
    for base_tag, agg in sorted(tier11_aggregates.items()):
        across = agg.get("across_seed", {})
        space = across.get("space_pct", {}).get("median")
        captured = across.get("captured_pct", {}).get("median")
        if space is None:
            continue
        spaces.append(space)
        if captured is not None:
            captures.append(captured)

    if not spaces:
        print(f"gate: {args.analysis} has no space_pct medians to judge")
        return 2

    median_space = statistics.median(spaces)
    median_captured = statistics.median(captures) if captures else 0.0
    verdict = "INVALID" if invalid else classify(median_space)
    all_configurations = data.get("configurations") or {}
    tier11_configurations = {
        tag: cfg for tag, cfg in all_configurations.items() if is_tier11_tag(tag)
    }
    waves = wave_coverage(tier11_configurations)
    axes = axis_coverage(tier11_configurations)
    tails = tail_coverage(tier11_configurations)
    semantics = semantic_coverage(tier11_configurations)
    missing = list(data.get("missing_configurations") or [])
    unexpected = list(data.get("unexpected_configurations") or [])
    missing_tier11 = sorted(tag for tag in missing if is_tier11_tag(tag))
    unexpected_tier11 = sorted(tag for tag in unexpected if is_tier11_tag(tag))
    manifest_declared = data.get("expected_configuration_count") is not None
    tier11_manifest_complete = (manifest_declared and not missing_tier11
                                and not unexpected_tier11)
    statistics_complete = bool(data.get("statistics_complete", False))
    all_unique_parents = bool(data.get("all_unique_parents", False))
    minimum_repeats = data.get("minimum_repeats")
    plan_sweep_complete = (tier11_manifest_complete and waves["plan_grid_complete"]
                           and axes["plan_degree_complete"]
                           and axes["plan_structure_complete"]
                           and tails["tier11_positive_tail"]
                           and semantics["semantic_proof_complete"]
                           and statistics_complete and all_unique_parents)
    next_step = NEXT_STEP[verdict]
    if verdict == "GO" and not plan_sweep_complete:
        next_step = ("complete the declared Tier 1.1 sweep; Tier 4 may confirm the "
                     "workload, but Tier 2/3 remains closed")

    print("=" * 70)
    print(f"TIER 1 GATE  verdict={verdict}")
    print("=" * 70)
    print(f"  configurations      : {len(spaces)}")
    print("  statistic scope     : Tier 1.1 only (t11p/t11ps); Tier 1.2 excluded")
    print(f"  median space%       : {median_space:.2f}   (Ceiling - Floor, over Floor)")
    print(f"  median captured%    : {median_captured:.2f}   (Floor - Impl, over Floor)")
    print(f"  range space%        : {min(spaces):.2f} .. {max(spaces):.2f}")
    print(f"  thresholds          : GO >= {GO_THRESHOLD}, STOP < {STOP_THRESHOLD}"
          f"  (EXPERIMENT_PLAN.md section 6)")
    if waves["max_pc_over_sm"] is not None:
        print(f"  max(P,C)/SM         : {waves['max_pc_over_sm']:.2f}"
              f"  multi={int(waves['has_multi_wave'])}"
              f"  nominal_2/8/32x={waves['nominal_multi_ratios_present']}")
        print(f"  proven multi ratios : {waves['plan_multi_ratios_present']}"
              f"  proven={int(waves['has_proven_multi_wave'])}")
        print(f"  underfill/full grid : {len(waves['underfilled_ratios'])} / "
              f"{int(waves['has_single_full'])}"
              f"  grid_complete={int(waves['plan_grid_complete'])}")
    print(f"  degree 1..1024      : {int(axes['plan_degree_complete'])}")
    print(f"  structures 5/5      : {int(axes['plan_structure_complete'])}")
    print(f"  Tier 1.1 manifest   : declared={int(manifest_declared)} "
          f"missing={len(missing_tier11)} "
          f"unexpected={len(unexpected_tier11)} complete={int(tier11_manifest_complete)}")
    print(f"  repeats >= 31       : {int(statistics_complete)}"
          f"  minimum={minimum_repeats if minimum_repeats is not None else 'unknown'}")
    print(f"  unique parent sets  : {int(all_unique_parents)}")
    print(f"  Tier 1.1 tail > 0   : {int(tails['tier11_positive_tail'])}")
    print(f"  semantic trace proof: {int(semantics['semantic_proof_complete'])}"
          f"  failures={len(semantics['semantic_proof_failures'])}")
    print(f"  plan sweep complete : {int(plan_sweep_complete)}")
    if invalid:
        print(f"  FAILED VALIDATION   : {', '.join(invalid)}")
    print()
    print(f"  next: {next_step}")
    if waves["pc_ratio_lte_one_only"] and verdict != "INVALID":
        print()
        print("  CAVEAT: every configuration ran with P,C <= SM (underfilled/=SM ratio).")
        print("  That ratio does not prove a one-wave execution; this verdict covers")
        print("  mechanism feasibility only and has no trace-proven multi-wave point.")
        print("  The multi-wave regime (plan §5.3: 2x/8x/32x SM) is NOT measured here, so")
        print("  a GO must not be read as permission to spend the full Tier 2/3 budget.")
    elif waves["has_multi_wave"] and not waves["plan_multi_complete"] and verdict != "INVALID":
        print()
        print("  CAVEAT: nominal P,C>SM points are present, but plan §5.3 requires evidence")
        print("  that consumers started while producer logical waves were still unlaunched.")
        print(f"  Trace-proven ratios are {waves['plan_multi_ratios_present'] or 'none'}; treat")
        print("  the verdict as provisional until 2x/8x/32x are all proven.")
    if not plan_sweep_complete and verdict != "INVALID":
        print()
        print("  CAVEAT: the Tier 1.1 matrix does not fully cover the declared degree, structure,")
        print("  underfilled/full, 2x/8x/32x-SM, positive-tail, repeat-count, and")
        print("  unique-parent requirements.")
        print("  The numeric verdict is provisional;")
        print("  it does not open Tier 2/3. Missing expected tags are recorded in gate.json.")
    print("=" * 70)

    if args.json:
        args.json.write_text(json.dumps({
            "verdict": verdict,
            "next_step": next_step,
            "median_space_pct": median_space,
            "median_captured_pct": median_captured,
            "min_space_pct": min(spaces),
            "max_space_pct": max(spaces),
            "configuration_count": len(spaces),
            "failed_validation": invalid,
            "pc_ratio_lte_one_only": waves["pc_ratio_lte_one_only"],
            "has_multi_wave": waves["has_multi_wave"],
            "has_proven_multi_wave": waves["has_proven_multi_wave"],
            "nominal_multi_ratios_present": waves["nominal_multi_ratios_present"],
            "plan_multi_ratios_present": waves["plan_multi_ratios_present"],
            "plan_multi_complete": waves["plan_multi_complete"],
            "underfilled_ratios": waves["underfilled_ratios"],
            "has_single_full": waves["has_single_full"],
            "plan_grid_complete": waves["plan_grid_complete"],
            "max_pc_over_sm": waves["max_pc_over_sm"],
            **axes,
            **tails,
            **semantics,
            "missing_configurations": missing,
            "unexpected_configurations": unexpected,
            "missing_tier11": missing_tier11,
            "unexpected_tier11": unexpected_tier11,
            "manifest_declared": manifest_declared,
            "tier11_manifest_complete": tier11_manifest_complete,
            "minimum_repeats": minimum_repeats,
            "statistics_complete": statistics_complete,
            "all_unique_parents": all_unique_parents,
            "plan_sweep_complete": plan_sweep_complete,
            "thresholds": {"go": GO_THRESHOLD, "stop": STOP_THRESHOLD},
            "source": str(args.analysis),
        }, indent=2, sort_keys=True) + "\n")

    return 2 if verdict == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
