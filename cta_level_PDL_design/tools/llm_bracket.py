#!/usr/bin/env python3
"""Validate and compute the Tier-4 LLM PDL bracket.

This analyzer is intentionally fail closed: it prints no performance verdict
unless every (batch, seq) point has one complete, adjacent, same-process
``pdl_off / pdl_grid / ceiling`` triplet with >=31 repetitions, 95% CIs,
worker-side configuration proof, isolated PTX/cubin evidence, and observed
FULL CUDA-graph execution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import statistics
import sys
from typing import Any

sys.path.insert(0, __import__("os").path.dirname(__file__))
from analyze import parse_summary  # noqa: E402


RUNGS = ("pdl_off", "pdl_grid", "ceiling")
PROOF_SCOPE = "worker+isolated_ptx+cubin+nsys_graph"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def gain(a: float, b: float, lower_better: bool) -> float:
    return (a - b) / a * 100.0 if lower_better else (b - a) / a * 100.0


def validate_row(row: dict[str, Any], metric: str) -> list[str]:
    label = f"tag={row.get('tag', '?')} rung={row.get('rung', '?')}"
    errors: list[str] = []
    if row.get("status") != "ok" or row.get("kind") != "measurement":
        errors.append(f"{label}: status/kind must be ok/measurement")
    if row.get("rung") not in RUNGS:
        errors.append(f"{label}: unknown rung")
    if not is_number(row.get("repetitions")) or row["repetitions"] < 31:
        errors.append(f"{label}: repetitions must be >=31")
    if row.get("ci_method") != "bootstrap_95pct":
        errors.append(f"{label}: ci_method must be bootstrap_95pct")

    value = row.get(metric)
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if not all(is_number(item) for item in (value, low, high)):
        errors.append(f"{label}: {metric} and its 95% CI are required")
    elif value <= 0 or low <= 0 or high <= 0 or not low <= value <= high:
        errors.append(f"{label}: invalid {metric} value/CI ordering")

    required_equal = {
        "triplet_mode": "same_process_adjacent",
        "graph_mode": "FULL",
        "graph_execution_proof": "nsys_cuda_graph_node",
        "proof_scope": PROOF_SCOPE,
        "cache_fresh": 1,
    }
    for key, expected in required_equal.items():
        if row.get(key) != expected:
            errors.append(f"{label}: {key} must be {expected}")
    for key in ("triplet_id", "driver_pid", "worker_cohort", "model_fingerprint", "output_digest"):
        if row.get(key) in (None, ""):
            errors.append(f"{label}: missing {key}")

    rung = row.get("rung")
    expected = {
        "pdl_off": {"pdl_inductor": 0, "ceiling": 0, "worker_pdl": 0, "worker_ceiling_hook": 0,
                    "ptx_wait_count": 0, "launch": "zero", "verified": 1},
        "pdl_grid": {"pdl_inductor": 1, "ceiling": 0, "worker_pdl": 1, "worker_ceiling_hook": 0,
                     "ptx_wait_count": "positive", "launch": "positive", "verified": 1},
        "ceiling": {"pdl_inductor": 1, "ceiling": 1, "worker_pdl": 1, "worker_ceiling_hook": 1,
                    "ptx_wait_count": 0, "launch": "positive", "verified": 0},
    }.get(rung)
    if expected:
        for key in ("pdl_inductor", "ceiling", "worker_pdl", "worker_ceiling_hook", "verified"):
            if row.get(key) != expected[key]:
                errors.append(f"{label}: {key}={row.get(key)!r}, expected {expected[key]!r}")
        waits = row.get("ptx_wait_count")
        launches = row.get("ptx_launch_count")
        if expected["ptx_wait_count"] == "positive":
            if not is_number(waits) or waits <= 0:
                errors.append(f"{label}: PTX wait count must be positive")
        elif waits != expected["ptx_wait_count"]:
            errors.append(f"{label}: PTX wait count must be zero")
        if expected["launch"] == "positive":
            if not is_number(launches) or launches <= 0:
                errors.append(f"{label}: PTX launch_dependents count must be positive")
        elif launches != 0:
            errors.append(f"{label}: PTX launch_dependents count must be zero")
        if not is_number(row.get("ptx_files")) or row["ptx_files"] <= 0:
            errors.append(f"{label}: no isolated PTX files")
        if not is_number(row.get("cubin_files")) or row["cubin_files"] <= 0:
            errors.append(f"{label}: no isolated cubin files")
        if not is_number(row.get("paired_cubin_files")) or row["paired_cubin_files"] <= 0:
            errors.append(f"{label}: no same-stem PTX/cubin pair")
        if not is_number(row.get("nsys_graph_kernel_events")) or row["nsys_graph_kernel_events"] <= 0:
            errors.append(f"{label}: no Nsight CUDA-graph kernel events")
        if rung == "pdl_grid":
            for key in ("nsys_wait_entry_matches", "nsys_launch_entry_matches"):
                if not is_number(row.get(key)) or row[key] <= 0:
                    errors.append(f"{label}: {key} must be positive")
        elif rung == "ceiling":
            if not is_number(row.get("nsys_launch_entry_matches")) or row["nsys_launch_entry_matches"] <= 0:
                errors.append(f"{label}: nsys_launch_entry_matches must be positive")
        elif not is_number(row.get("nsys_off_entry_matches")) or row["nsys_off_entry_matches"] <= 0:
            errors.append(f"{label}: nsys_off_entry_matches must be positive")
    return errors


def validate_triplet(key: tuple[Any, Any], rungs: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    missing = [rung for rung in RUNGS if rung not in rungs]
    if missing:
        return [f"batch={key[0]} seq={key[1]}: incomplete triplet, missing {','.join(missing)}"]
    rows = [rungs[rung] for rung in RUNGS]
    for field in ("triplet_id", "driver_pid", "worker_cohort", "model_fingerprint"):
        values = {row.get(field) for row in rows}
        if len(values) != 1 or None in values or "" in values:
            errors.append(f"batch={key[0]} seq={key[1]}: rungs do not share one {field}")
    off_digest = rungs["pdl_off"].get("output_digest")
    grid_digest = rungs["pdl_grid"].get("output_digest")
    if off_digest != grid_digest:
        errors.append(f"batch={key[0]} seq={key[1]}: non-Ceiling output digests differ")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument(
        "--metric",
        default="tok_per_s",
        choices=["tok_per_s", "tok_per_s_per_user", "median_s"],
    )
    args = parser.parse_args()

    all_tier4 = [row for row in parse_summary(args.summary) if row.get("tier") == 4]
    if not all_tier4:
        print("BLOCKED: no Tier-4 SUMMARY rows", file=sys.stderr)
        return 3

    errors: list[str] = []
    grid: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    positions: dict[tuple[Any, Any], list[tuple[int, str]]] = defaultdict(list)
    for position, row in enumerate(all_tier4):
        errors.extend(validate_row(row, args.metric))
        key = (row.get("batch"), row.get("seq"))
        rung = row.get("rung")
        if None in key:
            errors.append(f"tag={row.get('tag', '?')}: missing batch/seq")
            continue
        if not all(is_number(value) and value > 0 for value in key):
            errors.append(f"tag={row.get('tag', '?')}: batch/seq must be positive numbers")
        if rung in grid[key]:
            errors.append(f"batch={key[0]} seq={key[1]} rung={rung}: duplicate row")
        elif rung in RUNGS:
            grid[key][rung] = row
            positions[key].append((position, rung))

    for key, rungs in grid.items():
        errors.extend(validate_triplet(key, rungs))
        ordered = sorted(positions[key])
        if len(ordered) == len(RUNGS):
            indices = [position for position, _ in ordered]
            order = tuple(rung for _, rung in ordered)
            if order != RUNGS or indices != list(range(indices[0], indices[0] + len(RUNGS))):
                errors.append(
                    f"batch={key[0]} seq={key[1]}: rungs are not adjacent in off/grid/ceiling order"
                )

    fingerprints = {row.get("model_fingerprint") for row in all_tier4}
    if len(fingerprints) != 1 or None in fingerprints or "" in fingerprints:
        errors.append("campaign rows do not share one model fingerprint")

    if errors:
        print("BLOCKED: Tier-4 bracket is not admissible", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 3

    lower_better = args.metric == "median_s"
    print(f"metric = {args.metric} ({'lower' if lower_better else 'higher'} is better)\n")
    print(
        f"{'batch':>6} {'seq':>8} {'pdl_off':>12} {'pdl_grid':>12} {'ceiling':>12} "
        f"{'grid_gain%':>11} {'headroom%':>11}"
    )

    grid_gains: list[float] = []
    headrooms: list[tuple[float, tuple[Any, Any]]] = []
    for key in sorted(grid, key=lambda item: (item[1], item[0])):
        rungs = grid[key]
        off = float(rungs["pdl_off"][args.metric])
        floor = float(rungs["pdl_grid"][args.metric])
        ceiling = float(rungs["ceiling"][args.metric])
        grid_gain = gain(off, floor, lower_better)
        headroom = gain(floor, ceiling, lower_better)
        grid_gains.append(grid_gain)
        headrooms.append((headroom, key))
        print(
            f"{key[0]:>6} {key[1]:>8} {off:>12.3f} {floor:>12.3f} {ceiling:>12.3f} "
            f"{grid_gain:>11.2f} {headroom:>11.2f}"
        )

    median_grid = statistics.median(grid_gains)
    print(
        f"\ngrid-level PDL median gain {median_grid:.2f}% "
        f"(range {min(grid_gains):.2f}% .. {max(grid_gains):.2f}%)"
    )
    if not 2.0 <= median_grid <= 33.0:
        print("WARNING: outside the 2-33% diagnostic band; investigate before publication")

    headroom_values = [value for value, _ in headrooms]
    median_headroom = statistics.median(headroom_values)
    print(
        f"HEADROOM for CTA-level: median {median_headroom:.2f}% "
        f"(range {min(headroom_values):.2f}% .. {max(headroom_values):.2f}%)"
    )
    best_value, best_key = max(headrooms, key=lambda item: item[0])
    print(f"Largest headroom at batch={best_key[0]} seq={best_key[1]} ({best_value:.2f}%).")

    if median_headroom < 2.0:
        print("VERDICT: small residual headroom; re-evaluate the CTA-level direction.")
    elif median_headroom < 8.0:
        print("VERDICT: modest residual headroom; pursue only favorable dependency structures.")
    else:
        print("VERDICT: substantial residual headroom remains after grid-level PDL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
