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
REJECTED_RE = re.compile(r"^REJECTED_ATTEMPT\s+(.+)$")
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
    if not (len(floor) == len(ceiling) == len(impl)):
        raise ValueError("paired bootstrap requires equal rung sample counts")
    rng = random.Random(stable_seed(label))
    n = len(floor)
    results = {"floor_ms": [], "ceiling_ms": [], "impl_ms": [], "space_pct": [],
               "captured_pct": [], "of_space_pct": [], "speedup": []}
    for _ in range(iterations):
        # Floor/Impl/Ceiling are executed adjacently inside each repeat.  Resample the
        # repeat identity once and carry all three rungs with it, preserving the paired
        # covariance instead of turning a same-process comparison into three independent
        # populations.
        indices = [rng.randrange(n) for _ in range(n)]
        f = statistics.median(floor[index] for index in indices)
        c = statistics.median(ceiling[index] for index in indices)
        i = statistics.median(impl[index] for index in indices)
        results["floor_ms"].append(f)
        results["ceiling_ms"].append(c)
        results["impl_ms"].append(i)
        results["space_pct"].append(100.0 * (f - c) / f)
        results["captured_pct"].append(100.0 * (f - i) / f)
        results["of_space_pct"].append(
            100.0 * (f - i) / (f - c) if f != c else 0.0
        )
        results["speedup"].append(f / i)
    return results


def ci(values: list[float]) -> list[float]:
    return [quantile(values, 0.025), quantile(values, 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expected", type=Path,
                        help="newline-delimited expected tags from bench/run_all.sh")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    samples: dict[str, dict[str, list[float]]] = {}
    sample_reps: dict[str, dict[str, list[int]]] = {}
    sample_attempts: dict[str, dict[str, list[int]]] = {}
    rejected: dict[str, list[dict[str, str]]] = {}
    summaries: dict[str, dict[str, str]] = {}
    for line in args.log.read_text().splitlines():
        if match := SAMPLE_RE.match(line):
            record = fields(match.group(1))
            if record.get("trace_complete") != "1":
                raise SystemExit(
                    f"{record.get('tag', '?')}: incomplete trace published as SAMPLE"
                )
            try:
                trace_attempts = int(record["trace_attempts"])
            except (KeyError, ValueError):
                raise SystemExit(
                    f"{record.get('tag', '?')}: SAMPLE missing valid trace_attempts"
                )
            if not 1 <= trace_attempts <= 4:
                raise SystemExit(
                    f"{record['tag']}: SAMPLE trace_attempts={trace_attempts} outside 1..4"
                )
            samples.setdefault(record["tag"], {}).setdefault(record["mode"], []).append(
                float(record["ms"])
            )
            sample_reps.setdefault(record["tag"], {}).setdefault(record["mode"], []).append(
                int(record["rep"])
            )
            sample_attempts.setdefault(record["tag"], {}).setdefault(
                record["mode"], []
            ).append(trace_attempts)
        elif match := REJECTED_RE.match(line):
            record = fields(match.group(1))
            rejected.setdefault(record.get("tag", ""), []).append(record)
        elif match := SUMMARY_RE.match(line):
            record = fields(match.group(1))
            if record["tag"] in summaries:
                raise SystemExit(f"duplicate summary tag: {record['tag']}")
            summaries[record["tag"]] = record

    # Fail before writing anything. An empty run would otherwise emit a JSON asserting
    # "all_valid": true over zero configurations and then die on the CSV, leaving a file
    # that claims everything passed when nothing ran.
    if not summaries:
        raise SystemExit(
            f"{args.log}: no SUMMARY_PILOT records. Either no cta_dep_pilot step completed "
            "(check failures.log), or this is a cta_dep_bench log -- that schema uses plain "
            "SUMMARY lines and is read by tools/analyze.py instead."
        )

    if set(samples) != set(summaries):
        missing_samples = sorted(set(summaries) - set(samples))
        missing_summary = sorted(set(samples) - set(summaries))
        raise SystemExit(
            "SAMPLE and SUMMARY_PILOT tag sets differ; a step probably died mid-run. "
            f"summary without samples: {missing_samples or 'none'}; "
            f"samples without summary: {missing_summary or 'none'}"
        )
    orphan_rejections = sorted(set(rejected) - set(summaries))
    if orphan_rejections:
        raise SystemExit(
            "REJECTED_ATTEMPT without completed SUMMARY_PILOT (retry exhaustion or "
            f"truncated run): {orphan_rejections}"
        )

    expected_tags: set[str] | None = None
    if args.expected:
        try:
            expected_lines = [line.strip() for line in args.expected.read_text().splitlines()
                              if line.strip()]
        except FileNotFoundError:
            raise SystemExit(f"expected-tag manifest not found: {args.expected}")
        if len(expected_lines) != len(set(expected_lines)):
            raise SystemExit(f"duplicate tags in expected manifest: {args.expected}")
        expected_tags = set(expected_lines)
    missing_configurations = sorted((expected_tags or set()) - set(summaries))
    unexpected_configurations = sorted(set(summaries) - (expected_tags or set())) \
        if expected_tags is not None else []

    rows = []
    detailed: dict[str, object] = {}
    for tag in sorted(summaries):
        meta = summaries[tag]
        if meta.get("semantics") != "2":
            raise SystemExit(f"{tag}: unsupported/missing pilot semantics={meta.get('semantics')!r}")
        mode_samples = samples[tag]
        impl_name = meta["impl"]
        required = {"grid", "none", impl_name}
        if not required.issubset(mode_samples):
            raise SystemExit(f"{tag}: missing modes {sorted(required - set(mode_samples))}")
        counts = {mode: len(values) for mode, values in mode_samples.items()}
        if len(set(counts.values())) != 1:
            raise SystemExit(f"{tag}: unequal sample counts: {counts}")
        declared_repeats = int(meta["repeats"])
        if any(count != declared_repeats for count in counts.values()):
            raise SystemExit(
                f"{tag}: sample counts {counts} do not match repeats={declared_repeats}"
            )
        expected_reps = list(range(declared_repeats))
        bad_reps = {
            mode: reps
            for mode, reps in sample_reps[tag].items()
            if reps != expected_reps
        }
        if bad_reps:
            raise SystemExit(
                f"{tag}: each mode must contain exactly one ordered sample for every "
                f"rep 0..{declared_repeats - 1}; got {bad_reps}"
            )

        try:
            declared_trace_retries = int(meta["trace_retries"])
            trace_retry_limit = int(meta["trace_retry_limit"])
            trace_max_attempts = int(meta["trace_max_attempts"])
            declared_max_observed = int(meta["trace_max_attempts_observed"])
        except (KeyError, ValueError):
            raise SystemExit(f"{tag}: missing/invalid trace retry metadata")
        if trace_retry_limit != 3 or trace_max_attempts != trace_retry_limit + 1:
            raise SystemExit(
                f"{tag}: invalid retry contract trace_retry_limit={trace_retry_limit}, "
                f"trace_max_attempts={trace_max_attempts}"
            )

        final_attempt_by_sample = {
            (mode, rep): attempt
            for mode, reps in sample_reps[tag].items()
            for rep, attempt in zip(reps, sample_attempts[tag][mode])
        }
        rejected_by_sample: dict[tuple[str, int], list[int]] = {}
        for record in rejected.get(tag, []):
            try:
                mode = record["mode"]
                rep = int(record["rep"])
                attempt = int(record["attempt"])
                rejected_max = int(record["max_attempts"])
                missing_or_invalid = sum(int(record[key]) for key in (
                    "missing_p_start", "missing_p_ready", "missing_p_end",
                    "missing_c_start", "missing_c_dep", "missing_c_end",
                    "invalid_p_order", "invalid_c_order",
                ))
            except (KeyError, ValueError):
                raise SystemExit(f"{tag}: malformed REJECTED_ATTEMPT: {record}")
            if record.get("reason") != "trace_incomplete" or missing_or_invalid <= 0:
                raise SystemExit(f"{tag}: unsupported/unproven rejected attempt: {record}")
            if rejected_max != trace_max_attempts:
                raise SystemExit(f"{tag}: inconsistent rejected max_attempts: {record}")
            rejected_by_sample.setdefault((mode, rep), []).append(attempt)

        for key, final_attempt in final_attempt_by_sample.items():
            actual_rejections = rejected_by_sample.pop(key, [])
            expected_rejections = list(range(1, final_attempt))
            if actual_rejections != expected_rejections:
                raise SystemExit(
                    f"{tag}: retry audit mismatch for mode={key[0]} rep={key[1]}: "
                    f"rejected={actual_rejections}, final_attempt={final_attempt}"
                )
        if rejected_by_sample:
            raise SystemExit(f"{tag}: rejected attempts without final SAMPLE: {rejected_by_sample}")
        if declared_trace_retries != len(rejected.get(tag, [])):
            raise SystemExit(
                f"{tag}: trace_retries={declared_trace_retries} but log contains "
                f"{len(rejected.get(tag, []))} REJECTED_ATTEMPT records"
            )
        observed_max = max(
            attempt
            for attempts_by_mode in sample_attempts[tag].values()
            for attempt in attempts_by_mode
        )
        if declared_max_observed != observed_max:
            raise SystemExit(
                f"{tag}: trace_max_attempts_observed={declared_max_observed}, "
                f"but SAMPLE records show {observed_max}"
            )

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
                (medians["floor_ms"] - medians["ceiling_ms"])
                if medians["floor_ms"] != medians["ceiling_ms"] else 0.0,
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
            "requested_degree": int(meta.get("requested_degree", meta["degree"])),
            "effective_degree": float(meta["eff_degree"]),
            "tightness": float(meta["tightness"]),
            "producers": int(meta["producers"]),
            "consumers": int(meta["consumers"]),
            "sms": int(meta.get("sms", 0) or 0),
            "wave": meta.get("wave", ""),
            "unique_parents": int(meta.get("unique_parents", 0) or 0),
            "tail_cycles": int(meta["tail"]),
            "repeats": declared_repeats,
            "trace_retries": declared_trace_retries,
            "trace_max_attempts_observed": declared_max_observed,
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
            "paired_repeat_ids": expected_reps,
            "rejected_attempt_count": len(rejected.get(tag, [])),
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
        "expected_source": str(args.expected) if args.expected else None,
        "expected_configuration_count": len(expected_tags) if expected_tags is not None else None,
        "missing_configurations": missing_configurations,
        "unexpected_configurations": unexpected_configurations,
        "coverage_complete": expected_tags is not None and not missing_configurations
            and not unexpected_configurations,
        "minimum_repeats": min(int(row["repeats"]) for row in rows),
        "statistics_complete": all(int(row["repeats"]) >= 31 for row in rows),
        "all_unique_parents": all(int(row["unique_parents"]) == 1 for row in rows),
        "bootstrap_iterations": args.iterations,
        "bootstrap_resampling": "paired_by_repeat",
        "configuration_count": len(rows),
        "total_trace_retries": sum(int(row["trace_retries"]) for row in rows),
        "max_trace_attempts_observed": max(
            int(row["trace_max_attempts_observed"]) for row in rows
        ),
        "all_valid": all(row["valid"] == 1 for row in rows),
        "configurations": detailed,
        "aggregates": aggregates,
    }, indent=2, sort_keys=True) + "\n")
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    coverage = "n/a" if expected_tags is None else str(int(not missing_configurations and not unexpected_configurations))
    print(f"configurations={len(rows)} all_valid={int(all(row['valid'] == 1 for row in rows))} "
          f"coverage_complete={coverage} missing={len(missing_configurations)} "
          f"unexpected={len(unexpected_configurations)}")
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
