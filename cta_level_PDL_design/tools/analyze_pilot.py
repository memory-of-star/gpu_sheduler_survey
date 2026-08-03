#!/usr/bin/env python3
"""Robustly summarize cta_dep_pilot SAMPLE/SUMMARY_PILOT records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from pathlib import Path


SAMPLE_RE = re.compile(r"^SAMPLE\s+(.+)$")
SUMMARY_RE = re.compile(r"^SUMMARY_PILOT\s+(.+)$")


def fields(text: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def bootstrap(
    floor: list[float], ceiling: list[float], impl: list[float], iterations: int, label: str
) -> dict[str, list[float]]:
    rng = random.Random(stable_seed(label))
    n_floor, n_ceiling, n_impl = len(floor), len(ceiling), len(impl)
    results = {"floor_ms": [], "ceiling_ms": [], "impl_ms": [], "space_pct": [],
               "captured_pct": [], "of_space_pct": [], "speedup": []}
    for _ in range(iterations):
        f = statistics.median(floor[rng.randrange(n_floor)] for _ in range(n_floor))
        c = statistics.median(ceiling[rng.randrange(n_ceiling)] for _ in range(n_ceiling))
        i = statistics.median(impl[rng.randrange(n_impl)] for _ in range(n_impl))
        results["floor_ms"].append(f)
        results["ceiling_ms"].append(c)
        results["impl_ms"].append(i)
        results["space_pct"].append(100.0 * (f - c) / f)
        results["captured_pct"].append(100.0 * (f - i) / f)
        results["of_space_pct"].append(100.0 * (f - i) / (f - c))
        results["speedup"].append(f / i)
    return results


def ci(values: list[float]) -> list[float]:
    return [quantile(values, 0.025), quantile(values, 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    samples: dict[str, dict[str, list[float]]] = {}
    summaries: dict[str, dict[str, str]] = {}
    for line in args.log.read_text().splitlines():
        if match := SAMPLE_RE.match(line):
            record = fields(match.group(1))
            samples.setdefault(record["tag"], {}).setdefault(record["mode"], []).append(
                float(record["ms"])
            )
        elif match := SUMMARY_RE.match(line):
            record = fields(match.group(1))
            if record["tag"] in summaries:
                raise SystemExit(f"duplicate summary tag: {record['tag']}")
            summaries[record["tag"]] = record

    if set(samples) != set(summaries):
        raise SystemExit("SAMPLE and SUMMARY_PILOT tag sets differ")

    rows = []
    detailed: dict[str, object] = {}
    for tag in sorted(summaries):
        meta = summaries[tag]
        mode_samples = samples[tag]
        impl_name = meta["impl"]
        required = {"grid", "none", impl_name}
        if not required.issubset(mode_samples):
            raise SystemExit(f"{tag}: missing modes {sorted(required - set(mode_samples))}")
        counts = {mode: len(values) for mode, values in mode_samples.items()}
        if len(set(counts.values())) != 1:
            raise SystemExit(f"{tag}: unequal sample counts: {counts}")

        floor = mode_samples["grid"]
        ceiling = mode_samples["none"]
        impl = mode_samples[impl_name]
        medians = {
            "floor_ms": statistics.median(floor),
            "ceiling_ms": statistics.median(ceiling),
            "impl_ms": statistics.median(impl),
        }
        medians.update({
            "space_pct": 100.0 * (medians["floor_ms"] - medians["ceiling_ms"]) / medians["floor_ms"],
            "captured_pct": 100.0 * (medians["floor_ms"] - medians["impl_ms"]) / medians["floor_ms"],
            "of_space_pct": 100.0 * (medians["floor_ms"] - medians["impl_ms"]) /
                (medians["floor_ms"] - medians["ceiling_ms"]),
            "speedup": medians["floor_ms"] / medians["impl_ms"],
        })
        boot = bootstrap(floor, ceiling, impl, args.iterations, tag)
        cis = {name: ci(values) for name, values in boot.items()}
        seed_match = re.search(r"_s(\d+)$", tag)
        base_tag = tag[: seed_match.start()] if seed_match else tag
        seed = int(seed_match.group(1)) if seed_match else None
        row = {
            "tag": tag,
            "base_tag": base_tag,
            "seed": seed,
            "structure": meta["structure"],
            "degree": int(meta["degree"]),
            "effective_degree": float(meta["eff_degree"]),
            "tightness": float(meta["tightness"]),
            "producers": int(meta["producers"]),
            "consumers": int(meta["consumers"]),
            "tail_cycles": int(meta["tail"]),
            "repeats": int(meta["repeats"]),
            "valid": int(meta["valid"]),
            **medians,
        }
        for name, bounds in cis.items():
            row[f"{name}_ci_low"] = bounds[0]
            row[f"{name}_ci_high"] = bounds[1]
        rows.append(row)
        detailed[tag] = {
            "metadata": meta,
            "sample_counts": counts,
            "mode_medians_ms": {mode: statistics.median(values) for mode, values in mode_samples.items()},
            "estimates": medians,
            "bootstrap_95_ci": cis,
        }

    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["base_tag"]), []).append(row)
    aggregates = {}
    metrics = ["floor_ms", "ceiling_ms", "impl_ms", "space_pct", "captured_pct",
               "of_space_pct", "speedup"]
    for base_tag, group in sorted(groups.items()):
        aggregates[base_tag] = {
            "seed_count": len(group),
            "all_valid": all(row["valid"] == 1 for row in group),
            "across_seed": {
                metric: {
                    "median": statistics.median(float(row[metric]) for row in group),
                    "min": min(float(row[metric]) for row in group),
                    "max": max(float(row[metric]) for row in group),
                }
                for metric in metrics
            },
        }

    args.json.write_text(json.dumps({
        "source": str(args.log),
        "bootstrap_iterations": args.iterations,
        "configuration_count": len(rows),
        "all_valid": all(row["valid"] == 1 for row in rows),
        "configurations": detailed,
        "aggregates": aggregates,
    }, indent=2, sort_keys=True) + "\n")
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"configurations={len(rows)} all_valid={int(all(row['valid'] == 1 for row in rows))}")
    print("base_tag floor_ms impl_ms space_pct captured_pct of_space_pct speedup")
    for base_tag, aggregate in aggregates.items():
        values = aggregate["across_seed"]
        print(
            f"{base_tag} {values['floor_ms']['median']:.6f} {values['impl_ms']['median']:.6f} "
            f"{values['space_pct']['median']:.3f} {values['captured_pct']['median']:.3f} "
            f"{values['of_space_pct']['median']:.3f} {values['speedup']['median']:.4f}"
        )


if __name__ == "__main__":
    main()
