#!/usr/bin/env python3
"""Fail-closed validator for the formal Tier 0.3 background-cost matrix.

The input is one ``bench/results_*`` directory produced by ``run_session.sh``.
An admissible directory contains the complete 3 register-tier x 5 dynamic-smem
matrix, at least 31 adjacent deferred/resident pairs per point, and one final-pair
``%globaltimer`` trace per point.  This validator checks both the declared schema
and the raw timing relationships; it does not compute or endorse performance
conclusions.

Campaign-level failure is contagious.  A non-empty ``failures.log``, any
``*.invalid`` marker, or ``gate.json`` with verdict ``INVALID`` rejects the input
even if its Tier 0 files happen to be locally complete.  This prevents a failed
formal campaign from being cited as successful Tier 0 timing evidence.

Examples::

    python3 tools/validate_tier0_background.py bench/results_run
    python3 tools/validate_tier0_background.py bench/results_run \
        --json bench/results_run/tier0_background_validation.json

Exit status is zero only for an admissible formal matrix.  A failed validation
still writes ``--json`` (when requested) so automation can retain the reasons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REG_TIERS = {
    "low": {"requested_reg_words": 8, "launch_bounds_min_blocks": 8},
    "mid": {"requested_reg_words": 40, "launch_bounds_min_blocks": 4},
    "high": {"requested_reg_words": 80, "launch_bounds_min_blocks": 2},
}
SMEM_KB = (0, 8, 16, 32, 64)
EXPECTED_TAGS = {
    f"tier0_bg_{reg}_smem{smem}": (reg, smem)
    for reg in REG_TIERS
    for smem in SMEM_KB
}

# Formal Tier 0.3 constants from bench/run_all.sh and tier0_background.cu.
EXPECTED_SMS = 148
THREADS_PER_BLOCK = 128
PRODUCER_BLOCKS = 1
BG_WAVES = 8
BG_ITERS = 1_000_000
PRODUCER_CYCLES = 4_000_000
EXPECTED_BG_BLOCKS = EXPECTED_SMS * BG_WAVES
EXPECTED_BG_TOTAL_THREADS = EXPECTED_BG_BLOCKS * THREADS_PER_BLOCK

TRACE_COLUMNS = {
    "tag",
    "mode",
    "kind",
    "block_id",
    "sm_id",
    "t_start",
    "t_wait_enter",
    "t_wait_exit",
    "t_end",
}
MODES = ("deferred_gate", "resident_wait")
KINDS = ("producer", "waiter", "background")

LINE_PREFIXES = {
    "config": "CONFIG tier0=background ",
    "resource": "RESOURCE tier0=background ",
    "semantics": "SEMANTICS tier0=background ",
    "sample": "SAMPLE_TIER0_BG ",
    "summary": "SUMMARY tier0=background ",
}


class Validation:
    """Collect all failures so one run exposes the complete repair list."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def expect(self, condition: bool, message: str) -> bool:
        if not condition:
            self.error(message)
            return False
        return True


def parse_fields(payload: str) -> dict[str, str]:
    """Parse shell-like key=value fields, retaining all values as strings."""
    fields: dict[str, str] = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def records_with_prefix(lines: Iterable[str], prefix: str) -> list[dict[str, str]]:
    return [parse_fields(line[len(prefix):]) for line in lines if line.startswith(prefix)]


def integer(
    record: dict[str, str], key: str, label: str, validation: Validation
) -> int | None:
    value = record.get(key)
    try:
        return int(value) if value is not None else None
    except ValueError:
        validation.error(f"{label}: {key} must be an integer, got {value!r}")
        return None


def number(
    record: dict[str, str], key: str, label: str, validation: Validation
) -> float | None:
    value = record.get(key)
    try:
        parsed = float(value) if value is not None else None
    except ValueError:
        validation.error(f"{label}: {key} must be numeric, got {value!r}")
        return None
    if parsed is None or not math.isfinite(parsed):
        validation.error(f"{label}: {key} is missing or non-finite")
        return None
    return parsed


def expect_int(
    record: dict[str, str], key: str, expected: int, label: str, validation: Validation
) -> int | None:
    value = integer(record, key, label, validation)
    if value is None:
        if key not in record:
            validation.error(f"{label}: missing {key}")
        return None
    validation.expect(value == expected, f"{label}: {key}={value}, expected {expected}")
    return value


def expect_text(
    record: dict[str, str], key: str, expected: str, label: str, validation: Validation
) -> None:
    value = record.get(key)
    validation.expect(value == expected, f"{label}: {key}={value!r}, expected {expected!r}")


def validate_triplet(
    record: dict[str, str], value_key: str, low_key: str, high_key: str,
    label: str, validation: Validation, *, positive: bool = False,
    nonnegative: bool = False
) -> None:
    value = number(record, value_key, label, validation)
    low = number(record, low_key, label, validation)
    high = number(record, high_key, label, validation)
    if None in (value, low, high):
        return
    assert value is not None and low is not None and high is not None
    validation.expect(
        low <= value <= high,
        f"{label}: {value_key}={value} not inside [{low_key}={low}, {high_key}={high}]",
    )
    if positive:
        validation.expect(low > 0, f"{label}: {value_key} and CI must be positive")
    if nonnegative:
        validation.expect(low >= 0, f"{label}: {value_key} and CI must be nonnegative")


def peak_concurrent(intervals: Iterable[tuple[int, int]]) -> int:
    events: list[tuple[int, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        events.append((start, +1))
        events.append((end, -1))
    # At equal timestamps, -1 sorts before +1: intervals are half-open [start, end).
    events.sort()
    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


def close_to_printed(actual: float, logged: float, *, decimals: int) -> bool:
    # SAMPLE fields are rounded before logging, whereas SUMMARY values use the original
    # doubles.  Allow two units in the last printed place plus a tiny relative allowance.
    return math.isclose(
        actual,
        logged,
        rel_tol=1.0e-9,
        abs_tol=2.0 * (10.0 ** -decimals),
    )


def validate_campaign(result_dir: Path, validation: Validation) -> str | None:
    failures = result_dir / "failures.log"
    if failures.exists():
        try:
            failure_text = failures.read_text(errors="replace").strip()
        except OSError as exc:
            validation.error(f"cannot read {failures}: {exc}")
        else:
            if failure_text:
                validation.error("campaign failures.log is non-empty")

    invalid_markers = sorted(path.name for path in result_dir.glob("*.invalid"))
    if invalid_markers:
        validation.error(
            "campaign contains validation-failure markers: " + ", ".join(invalid_markers)
        )

    verdict: str | None = None
    gate = result_dir / "gate.json"
    if gate.exists():
        try:
            gate_record = json.loads(gate.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            validation.error(f"cannot parse {gate}: {exc}")
        else:
            if not isinstance(gate_record, dict):
                validation.error(f"{gate}: top-level JSON value must be an object")
            else:
                verdict_value = gate_record.get("verdict")
                verdict = str(verdict_value) if verdict_value is not None else None
                if verdict == "INVALID":
                    validation.error("campaign gate.json verdict is INVALID")
    return verdict


def validate_summary_statistics(
    summary: dict[str, str], tag: str, validation: Validation
) -> None:
    label = f"{tag} SUMMARY"
    triplets = (
        ("control_gupdates_s", "control_gupdates_ci_low", "control_gupdates_ci_high", True, False),
        (
            "resident_gupdates_s", "resident_gupdates_ci_low",
            "resident_gupdates_ci_high", True, False,
        ),
        ("throughput_loss_pct", "throughput_loss_ci_low", "throughput_loss_ci_high", False, False),
        (
            "control_bg_active_ms", "control_bg_active_ci_low",
            "control_bg_active_ci_high", True, False,
        ),
        (
            "resident_bg_active_ms", "resident_bg_active_ci_low",
            "resident_bg_active_ci_high", True, False,
        ),
        (
            "control_bg_peak_ctas_median", "control_bg_peak_ctas_ci_low",
            "control_bg_peak_ctas_ci_high", True, False,
        ),
        (
            "resident_bg_peak_ctas_median", "resident_bg_peak_ctas_ci_low",
            "resident_bg_peak_ctas_ci_high", True, False,
        ),
        ("control_e2e_ms", "control_e2e_ci_low", "control_e2e_ci_high", True, False),
        ("resident_e2e_ms", "resident_e2e_ci_low", "resident_e2e_ci_high", True, False),
        ("e2e_delta_ms", "e2e_delta_ci_low", "e2e_delta_ci_high", False, False),
        ("control_wait_median_us", "control_wait_ci_low", "control_wait_ci_high", False, True),
        ("wait_median_us", "wait_ci_low", "wait_ci_high", True, False),
        (
            "deferred_waiters_median", "deferred_waiters_ci_low",
            "deferred_waiters_ci_high", True, False,
        ),
        ("early_waiters_median", "early_waiters_ci_low", "early_waiters_ci_high", True, False),
        ("peak_waiters_median", "peak_waiters_ci_low", "peak_waiters_ci_high", True, False),
    )
    for value, low, high, positive, nonnegative in triplets:
        validate_triplet(
            summary, value, low, high, label, validation,
            positive=positive, nonnegative=nonnegative,
        )


def validate_log(
    path: Path, tag: str, reg: str, smem: int, validation: Validation
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, str]]]:
    label = tag
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        validation.error(f"{tag}: cannot read {path}: {exc}")
        return {}, {}

    try:
        grouped = {
            kind: records_with_prefix(lines, prefix)
            for kind, prefix in LINE_PREFIXES.items()
        }
    except ValueError as exc:
        validation.error(f"{tag}: cannot parse key=value log records: {exc}")
        return {}, {}
    for kind in ("config", "resource", "semantics", "summary"):
        validation.expect(
            len(grouped[kind]) == 1,
            f"{tag}: expected exactly one {kind.upper()} record, got {len(grouped[kind])}",
        )
    if not all(grouped[kind] for kind in ("config", "resource", "semantics", "summary")):
        return {}, {}

    config = grouped["config"][0]
    resource = grouped["resource"][0]
    semantics = grouped["semantics"][0]
    summary = grouped["summary"][0]

    expect_text(summary, "tag", tag, f"{label} SUMMARY", validation)
    expect_text(summary, "reg_tier", reg, f"{label} SUMMARY", validation)
    expect_int(summary, "smem_kb", smem, f"{label} SUMMARY", validation)
    expect_int(summary, "semantics", 2, f"{label} SUMMARY", validation)
    expect_int(summary, "valid", 1, f"{label} SUMMARY", validation)
    expect_text(summary, "trigger", "entry", f"{label} SUMMARY", validation)
    expect_text(summary, "wait", "griddepcontrol", f"{label} SUMMARY", validation)
    expect_text(summary, "control_mode", "deferred_gate", f"{label} SUMMARY", validation)
    expect_text(summary, "resident_mode", "resident_wait", f"{label} SUMMARY", validation)
    expect_text(
        summary,
        "anchor",
        "earliest_producer_or_background_activity",
        f"{label} SUMMARY",
        validation,
    )

    repeats = integer(summary, "repeats", f"{label} SUMMARY", validation)
    if repeats is None:
        validation.error(f"{label} SUMMARY: missing repeats")
        repeats = 0
    else:
        validation.expect(repeats >= 31, f"{label} SUMMARY: repeats={repeats}, expected >=31")

    expected_reg = REG_TIERS[reg]
    expect_int(
        summary, "requested_reg_words", expected_reg["requested_reg_words"],
        f"{label} SUMMARY", validation,
    )
    expect_int(
        summary, "launch_bounds_min_blocks", expected_reg["launch_bounds_min_blocks"],
        f"{label} SUMMARY", validation,
    )
    actual_num_regs = integer(summary, "actual_num_regs", f"{label} SUMMARY", validation)
    if actual_num_regs is None:
        validation.error(f"{label} SUMMARY: missing actual_num_regs")
    else:
        validation.expect(actual_num_regs > 0, f"{label} SUMMARY: actual_num_regs must be >0")
    expect_int(summary, "local_bytes", 0, f"{label} SUMMARY", validation)
    expect_int(summary, "static_smem_bytes", 0, f"{label} SUMMARY", validation)
    expect_int(summary, "sms", EXPECTED_SMS, f"{label} SUMMARY", validation)
    expect_int(summary, "producer_blocks", PRODUCER_BLOCKS, f"{label} SUMMARY", validation)
    expect_int(summary, "bg_blocks", EXPECTED_BG_BLOCKS, f"{label} SUMMARY", validation)
    expect_int(summary, "threads_per_block", THREADS_PER_BLOCK, f"{label} SUMMARY", validation)
    expect_int(
        summary, "bg_total_threads", EXPECTED_BG_TOTAL_THREADS,
        f"{label} SUMMARY", validation,
    )
    expect_int(summary, "bg_waves", BG_WAVES, f"{label} SUMMARY", validation)
    expect_int(summary, "bg_iters", BG_ITERS, f"{label} SUMMARY", validation)
    expect_int(
        summary, "producer_cycles", PRODUCER_CYCLES, f"{label} SUMMARY", validation,
    )

    occupancy = integer(summary, "occ_per_sm", f"{label} SUMMARY", validation)
    waiter_blocks = integer(summary, "waiter_blocks", f"{label} SUMMARY", validation)
    if occupancy is None:
        validation.error(f"{label} SUMMARY: missing occ_per_sm")
        occupancy = 0
    else:
        validation.expect(occupancy > 0, f"{label} SUMMARY: occ_per_sm must be >0")
    if waiter_blocks is None:
        validation.error(f"{label} SUMMARY: missing waiter_blocks")
        waiter_blocks = 0
    else:
        validation.expect(
            waiter_blocks == occupancy * EXPECTED_SMS,
            f"{label} SUMMARY: waiter_blocks={waiter_blocks}, expected occ_per_sm*sms="
            f"{occupancy * EXPECTED_SMS}",
        )

    # CONFIG contains warmup, which is intentionally not repeated in SUMMARY.
    expect_int(config, "sms", EXPECTED_SMS, f"{label} CONFIG", validation)
    expect_int(config, "producer_blocks", PRODUCER_BLOCKS, f"{label} CONFIG", validation)
    expect_int(config, "smem_kb", smem, f"{label} CONFIG", validation)
    expect_text(config, "reg_tier", reg, f"{label} CONFIG", validation)
    expect_int(config, "repeats", repeats, f"{label} CONFIG", validation)
    expect_int(config, "warmup", 3, f"{label} CONFIG", validation)
    expect_int(config, "bg_waves", BG_WAVES, f"{label} CONFIG", validation)
    expect_int(config, "bg_iters", BG_ITERS, f"{label} CONFIG", validation)
    expect_int(config, "producer_cycles", PRODUCER_CYCLES, f"{label} CONFIG", validation)

    resource_fields = (
        "requested_reg_words", "launch_bounds_min_blocks", "actual_num_regs",
        "local_bytes", "static_smem_bytes", "smem_kb", "occ_per_sm", "waiter_blocks",
    )
    for key in resource_fields:
        validation.expect(
            resource.get(key) == summary.get(key),
            f"{label}: RESOURCE {key}={resource.get(key)!r} differs from "
            f"SUMMARY {summary.get(key)!r}",
        )
    expect_text(resource, "reg_tier", reg, f"{label} RESOURCE", validation)

    semantic_values = {
        "trigger": "producer_entry",
        "wait": "griddepcontrol",
        "producer_role": "single_cta_dependency_holder",
        "control": "deferred_gate",
        "test": "resident_wait",
        "same_waiter_kernel": "1",
        "poison": "every_repeat",
        "validation": "all_outputs",
        "timer": "globaltimer",
        "anchor": "earliest_producer_or_background_activity",
    }
    for key, expected in semantic_values.items():
        expect_text(semantics, key, expected, f"{label} SEMANTICS", validation)

    validate_summary_statistics(summary, tag, validation)
    deferred_median = number(summary, "deferred_waiters_median", label, validation)
    deferred_low = number(summary, "deferred_waiters_ci_low", label, validation)
    deferred_high = number(summary, "deferred_waiters_ci_high", label, validation)
    for key, value in (
        ("deferred_waiters_median", deferred_median),
        ("deferred_waiters_ci_low", deferred_low),
        ("deferred_waiters_ci_high", deferred_high),
    ):
        if value is not None:
            validation.expect(
                value == waiter_blocks,
                f"{tag} SUMMARY: {key}={value}, expected waiter_blocks={waiter_blocks}",
            )

    samples: dict[tuple[int, str], dict[str, str]] = {}
    sample_sequence: list[tuple[int, str]] = []
    for index, sample in enumerate(grouped["sample"], 1):
        sample_label = f"{tag} SAMPLE#{index}"
        expect_int(sample, "semantics", 2, sample_label, validation)
        expect_text(sample, "tag", tag, sample_label, validation)
        expect_int(sample, "valid", 1, sample_label, validation)
        rep = integer(sample, "rep", sample_label, validation)
        mode = sample.get("mode")
        if rep is None:
            validation.error(f"{sample_label}: missing rep")
            continue
        if mode not in MODES:
            validation.error(f"{sample_label}: mode={mode!r}, expected one of {MODES}")
            continue
        key = (rep, mode)
        if key in samples:
            validation.error(f"{tag}: duplicate SAMPLE rep={rep} mode={mode}")
            continue
        samples[key] = sample
        sample_sequence.append(key)

        for metric in (
            "bg_active_ms", "bg_effective_ms", "bg_gupdates_s", "e2e_ms",
            "wait_median_us", "early_waiters", "peak_waiters", "bg_peak_ctas",
        ):
            metric_value = number(sample, metric, sample_label, validation)
            if metric_value is not None:
                validation.expect(metric_value >= 0, f"{sample_label}: {metric} must be >=0")
        bg_peak = integer(sample, "bg_peak_ctas", sample_label, validation)
        if bg_peak is not None:
            validation.expect(
                0 < bg_peak <= EXPECTED_BG_BLOCKS,
                f"{sample_label}: bg_peak_ctas={bg_peak} outside 1..{EXPECTED_BG_BLOCKS}",
            )

        early = integer(sample, "early_waiters", sample_label, validation)
        peak = integer(sample, "peak_waiters", sample_label, validation)
        if mode == "deferred_gate":
            deferred = integer(sample, "deferred_waiters", sample_label, validation)
            validation.expect(
                deferred == waiter_blocks,
                f"{sample_label}: deferred_waiters={deferred}, expected {waiter_blocks}",
            )
            validation.expect(early == 0, f"{sample_label}: control early_waiters must be 0")
            validation.expect(peak == 0, f"{sample_label}: control peak_waiters must be 0")
        else:
            validation.expect(
                early is not None and 0 < early <= waiter_blocks,
                f"{sample_label}: resident early_waiters={early}, expected 1..{waiter_blocks}",
            )
            validation.expect(
                peak is not None and early is not None and 0 < peak <= early,
                f"{sample_label}: resident peak_waiters={peak}, expected 1..early_waiters",
            )

    expected_sample_keys = {
        (rep, mode) for rep in range(repeats) for mode in MODES
    }
    expected_sample_sequence = [
        (rep, mode) for rep in range(repeats) for mode in MODES
    ]
    missing_samples = sorted(expected_sample_keys - set(samples))
    extra_samples = sorted(set(samples) - expected_sample_keys)
    validation.expect(
        not missing_samples,
        f"{tag}: missing paired SAMPLE records: {missing_samples[:12]}"
        + (" ..." if len(missing_samples) > 12 else ""),
    )
    validation.expect(
        not extra_samples,
        f"{tag}: out-of-range SAMPLE records: {extra_samples[:12]}"
        + (" ..." if len(extra_samples) > 12 else ""),
    )
    validation.expect(
        sample_sequence == expected_sample_sequence,
        f"{tag}: SAMPLE records are not adjacent ordered deferred/resident pairs for "
        f"rep 0..{repeats - 1}",
    )

    matrix_row: dict[str, Any] = {
        "tag": tag,
        "reg_tier": reg,
        "smem_kb": smem,
        "actual_num_regs": actual_num_regs,
        "local_bytes": integer(summary, "local_bytes", label, validation),
        "occ_per_sm": occupancy,
        "waiter_blocks": waiter_blocks,
        "repeats": repeats,
        "sample_count": len(samples),
        "early_waiters_median": number(summary, "early_waiters_median", label, validation),
        "peak_waiters_median": number(summary, "peak_waiters_median", label, validation),
    }
    return matrix_row, samples


def parse_trace_integer(
    row: dict[str, str], key: str, label: str, validation: Validation
) -> int | None:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        validation.error(f"{label}: invalid/missing integer {key}={row.get(key)!r}")
        return None


def validate_trace_mode(
    tag: str,
    mode: str,
    rows: list[dict[str, int | str]],
    matrix_row: dict[str, Any],
    final_sample: dict[str, str] | None,
    validation: Validation,
) -> dict[str, int | float]:
    label = f"{tag} trace mode={mode}"
    waiter_blocks = int(matrix_row.get("waiter_blocks") or 0)
    expected_counts = {
        "producer": PRODUCER_BLOCKS,
        "waiter": waiter_blocks,
        "background": EXPECTED_BG_BLOCKS,
    }
    by_kind: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        kind = str(row["kind"])
        by_kind[kind].append(row)
    counts = {kind: len(by_kind.get(kind, [])) for kind in KINDS}
    validation.expect(
        counts == expected_counts,
        f"{label}: counts={counts}, expected={expected_counts}",
    )

    for kind, expected_count in expected_counts.items():
        block_ids = [int(row["block_id"]) for row in by_kind.get(kind, [])]
        validation.expect(
            sorted(block_ids) == list(range(expected_count)),
            f"{label}: {kind} block ids are not exactly 0..{expected_count - 1}",
        )

    producers = by_kind.get("producer", [])
    backgrounds = by_kind.get("background", [])
    waiters = by_kind.get("waiter", [])
    if not producers or not backgrounds or not waiters:
        return {"row_count": len(rows), "early_waiters": 0, "peak_waiters": 0}

    producer_start = min(int(row["t_start"]) for row in producers)
    producer_end = max(int(row["t_end"]) for row in producers)
    bg_start = min(int(row["t_start"]) for row in backgrounds)
    bg_end = max(int(row["t_end"]) for row in backgrounds)
    waiter_end = max(int(row["t_end"]) for row in waiters)
    anchor = min(producer_start, bg_start)

    early = [row for row in waiters if int(row["t_wait_enter"]) < producer_end]
    violations = [row for row in early if int(row["t_wait_exit"]) < producer_end]
    validation.expect(
        not violations,
        f"{label}: {len(violations)} early waiters exited before producer completion",
    )
    if mode == "deferred_gate":
        validation.expect(
            not early,
            f"{label}: control has {len(early)} waiters entering before producer completion",
        )
        wait_rows = waiters
        wait_intervals: list[tuple[int, int]] = []
        deferred_count = sum(
            int(row["t_wait_enter"]) >= producer_end for row in waiters
        )
        validation.expect(
            deferred_count == waiter_blocks,
            f"{label}: deferred_count={deferred_count}, expected {waiter_blocks}",
        )
    else:
        validation.expect(bool(early), f"{label}: no resident waiter entered early")
        wait_rows = early
        wait_intervals = [
            (int(row["t_wait_enter"]), int(row["t_wait_exit"])) for row in early
        ]

    peak_waiters = peak_concurrent(wait_intervals)
    bg_peak = peak_concurrent(
        (int(row["t_start"]), int(row["t_end"])) for row in backgrounds
    )
    wait_median_us = statistics.median(
        (int(row["t_wait_exit"]) - int(row["t_wait_enter"])) / 1000.0
        for row in wait_rows
    )
    bg_active_ms = (bg_end - bg_start) / 1.0e6
    bg_effective_ms = (bg_end - anchor) / 1.0e6
    bg_gupdates_s = EXPECTED_BG_TOTAL_THREADS * BG_ITERS / (bg_end - anchor)
    e2e_ms = (max(producer_end, bg_end, waiter_end) - anchor) / 1.0e6

    if final_sample is None:
        validation.error(f"{label}: no final-repeat SAMPLE to compare with trace")
    else:
        comparisons = (
            ("bg_active_ms", bg_active_ms, 6),
            ("bg_effective_ms", bg_effective_ms, 6),
            ("bg_gupdates_s", bg_gupdates_s, 6),
            ("e2e_ms", e2e_ms, 6),
            ("wait_median_us", wait_median_us, 3),
        )
        for key, computed, decimals in comparisons:
            logged = number(final_sample, key, label, validation)
            if logged is not None:
                validation.expect(
                    close_to_printed(computed, logged, decimals=decimals),
                    f"{label}: trace-derived {key}={computed} differs from final SAMPLE {logged}",
                )
        sample_early = integer(final_sample, "early_waiters", label, validation)
        sample_peak = integer(final_sample, "peak_waiters", label, validation)
        sample_bg_peak = integer(final_sample, "bg_peak_ctas", label, validation)
        validation.expect(
            sample_early == len(early),
            f"{label}: trace early_waiters={len(early)}, final SAMPLE={sample_early}",
        )
        validation.expect(
            sample_peak == peak_waiters,
            f"{label}: trace peak_waiters={peak_waiters}, final SAMPLE={sample_peak}",
        )
        validation.expect(
            sample_bg_peak == bg_peak,
            f"{label}: trace bg_peak_ctas={bg_peak}, final SAMPLE={sample_bg_peak}",
        )

    return {
        "row_count": len(rows),
        "early_waiters": len(early),
        "peak_waiters": peak_waiters,
        "bg_peak_ctas": bg_peak,
    }


def validate_trace(
    path: Path,
    tag: str,
    matrix_row: dict[str, Any],
    samples: dict[tuple[int, str], dict[str, str]],
    validation: Validation,
) -> dict[str, Any]:
    try:
        handle = path.open(newline="")
    except OSError as exc:
        validation.error(f"{tag}: cannot open trace {path}: {exc}")
        return {"row_count": 0, "modes": {}}

    parsed_rows: list[dict[str, int | str]] = []
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        validation.expect(
            fieldnames == TRACE_COLUMNS,
            f"{tag}: trace columns={sorted(fieldnames)}, expected={sorted(TRACE_COLUMNS)}",
        )
        seen: set[tuple[str, str, int]] = set()
        for row_number, row in enumerate(reader, 2):
            row_label = f"{tag} trace row {row_number}"
            if None in row or any(value is None for value in row.values()):
                validation.error(f"{row_label}: malformed CSV row")
                continue
            validation.expect(row.get("tag") == tag, f"{row_label}: tag={row.get('tag')!r}")
            mode = row.get("mode", "")
            kind = row.get("kind", "")
            if mode not in MODES:
                validation.error(f"{row_label}: unknown mode={mode!r}")
                continue
            if kind not in KINDS:
                validation.error(f"{row_label}: unknown kind={kind!r}")
                continue

            numeric: dict[str, int] = {}
            for key in (
                "block_id", "sm_id", "t_start", "t_wait_enter", "t_wait_exit", "t_end"
            ):
                value = parse_trace_integer(row, key, row_label, validation)
                if value is None:
                    break
                numeric[key] = value
            else:
                key = (mode, kind, numeric["block_id"])
                validation.expect(key not in seen, f"{row_label}: duplicate identity {key}")
                seen.add(key)
                validation.expect(
                    0 <= numeric["sm_id"] < EXPECTED_SMS,
                    f"{row_label}: sm_id={numeric['sm_id']} outside 0..{EXPECTED_SMS - 1}",
                )
                if kind == "waiter":
                    validation.expect(
                        0 < numeric["t_start"] <= numeric["t_wait_enter"]
                        <= numeric["t_wait_exit"] <= numeric["t_end"],
                        f"{row_label}: waiter timestamps are missing or out of order",
                    )
                else:
                    validation.expect(
                        0 < numeric["t_start"] < numeric["t_end"],
                        f"{row_label}: {kind} start/end are missing or out of order",
                    )
                    validation.expect(
                        numeric["t_wait_enter"] == 0 and numeric["t_wait_exit"] == 0,
                        f"{row_label}: non-waiter has wait timestamps",
                    )
                parsed_rows.append({"tag": tag, "mode": mode, "kind": kind, **numeric})

    repeats = int(matrix_row.get("repeats") or 0)
    modes: dict[str, Any] = {}
    for mode in MODES:
        mode_rows = [row for row in parsed_rows if row["mode"] == mode]
        modes[mode] = validate_trace_mode(
            tag,
            mode,
            mode_rows,
            matrix_row,
            samples.get((repeats - 1, mode)),
            validation,
        )
    return {"row_count": len(parsed_rows), "modes": modes}


def write_json(path: Path, report: dict[str, Any], validation: Validation) -> None:
    try:
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        validation.error(f"cannot write JSON {path}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the complete formal Tier 0.3 deferred/resident matrix.",
        epilog=(
            "The validator is fail closed: FAST/smoke matrices, partial traces, and any "
            "campaign carrying INVALID/failure evidence exit non-zero."
        ),
    )
    parser.add_argument("result_dir", type=Path, help="bench/results_* directory to validate")
    parser.add_argument("--json", type=Path, help="write the validation record to this path")
    args = parser.parse_args()

    validation = Validation()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        validation.error(f"result directory does not exist: {result_dir}")

    gate_verdict: str | None = None
    matrix: dict[str, dict[str, Any]] = {}
    sample_sets: dict[str, dict[tuple[int, str], dict[str, str]]] = {}
    traces: dict[str, dict[str, Any]] = {}

    if result_dir.is_dir():
        gate_verdict = validate_campaign(result_dir, validation)

        log_paths = {
            path.stem: path for path in result_dir.glob("tier0_bg_*.log")
            if path.is_file()
        }
        trace_paths = {
            path.name[:-len("_trace.csv")]: path
            for path in result_dir.glob("tier0_bg_*_trace.csv")
            if path.is_file()
        }
        log_tags = set(log_paths)
        trace_tags = set(trace_paths)
        expected_tags = set(EXPECTED_TAGS)
        validation.expect(
            log_tags == expected_tags,
            "Tier 0.3 log matrix mismatch: "
            f"missing={sorted(expected_tags - log_tags)}, "
            f"unexpected={sorted(log_tags - expected_tags)}",
        )
        validation.expect(
            trace_tags == expected_tags,
            "Tier 0.3 trace matrix mismatch: "
            f"missing={sorted(expected_tags - trace_tags)}, "
            f"unexpected={sorted(trace_tags - expected_tags)}",
        )

        for tag in sorted(expected_tags & log_tags):
            reg, smem = EXPECTED_TAGS[tag]
            matrix_row, samples = validate_log(
                log_paths[tag], tag, reg, smem, validation
            )
            matrix[tag] = matrix_row
            sample_sets[tag] = samples

        # Actual allocation, not the requested tier name, is the evidence.  It must be
        # stable across smem points and create three spill-free, strictly ordered tiers.
        regs_by_tier: dict[str, set[int]] = defaultdict(set)
        for row in matrix.values():
            actual = row.get("actual_num_regs")
            if isinstance(actual, int):
                regs_by_tier[str(row.get("reg_tier"))].add(actual)
        for reg in REG_TIERS:
            validation.expect(
                len(regs_by_tier.get(reg, set())) == 1,
                f"actual_num_regs is not one stable value for reg_tier={reg}: "
                f"{sorted(regs_by_tier.get(reg, set()))}",
            )
        if all(len(regs_by_tier.get(reg, set())) == 1 for reg in REG_TIERS):
            low = next(iter(regs_by_tier["low"]))
            mid = next(iter(regs_by_tier["mid"]))
            high = next(iter(regs_by_tier["high"]))
            validation.expect(
                low < mid < high,
                "actual register tiers are not strictly ordered: "
                f"low={low}, mid={mid}, high={high}",
            )

        for tag in sorted(expected_tags & trace_tags & set(matrix)):
            traces[tag] = validate_trace(
                trace_paths[tag], tag, matrix[tag], sample_sets[tag], validation
            )

    configuration_count = len(matrix)
    sample_count = sum(len(samples) for samples in sample_sets.values())
    paired_repetitions = 0
    for tag, samples in sample_sets.items():
        repeats = int(matrix.get(tag, {}).get("repeats") or 0)
        paired_repetitions += sum(
            all((rep, mode) in samples for mode in MODES) for rep in range(repeats)
        )
    trace_row_count = sum(int(trace.get("row_count", 0)) for trace in traces.values())
    repeats_values = [
        int(row["repeats"]) for row in matrix.values()
        if isinstance(row.get("repeats"), int) and int(row["repeats"]) > 0
    ]

    report: dict[str, Any] = {
        "schema": 1,
        "status": "PASS" if not validation.errors else "FAIL",
        "result_dir": str(result_dir),
        "gate_verdict": gate_verdict,
        "expected_configuration_count": len(EXPECTED_TAGS),
        "configuration_count": configuration_count,
        "minimum_repeats": min(repeats_values) if repeats_values else 0,
        "paired_repetitions": paired_repetitions,
        "sample_count": sample_count,
        "trace_file_count": len(traces),
        "trace_row_count": trace_row_count,
        "sms": EXPECTED_SMS,
        "matrix": [matrix[tag] for tag in sorted(matrix)],
        "traces": traces,
        "error_count": len(validation.errors),
        "errors": validation.errors,
    }

    if args.json:
        write_json(args.json, report, validation)
        # A write failure is itself fail-closed and must be reflected in the terminal state.
        if len(validation.errors) != report["error_count"]:
            report["status"] = "FAIL"
            report["error_count"] = len(validation.errors)
            report["errors"] = validation.errors

    print(
        "TIER0_BACKGROUND_VALIDATION "
        f"status={report['status']} configurations={configuration_count} "
        f"paired_repetitions={paired_repetitions} samples={sample_count} "
        f"trace_files={len(traces)} trace_rows={trace_row_count} "
        f"minimum_repeats={report['minimum_repeats']} errors={len(validation.errors)}"
    )
    if not validation.errors:
        for tag in sorted(matrix):
            row = matrix[tag]
            print(
                "TIER0_BACKGROUND_MATRIX "
                f"tag={tag} reg_tier={row.get('reg_tier')} smem_kb={row.get('smem_kb')} "
                f"actual_num_regs={row.get('actual_num_regs')} "
                f"occ_per_sm={row.get('occ_per_sm')} waiter_blocks={row.get('waiter_blocks')} "
                f"repeats={row.get('repeats')}"
            )
    else:
        print("Tier 0.3 background matrix is not admissible:", file=sys.stderr)
        for error in validation.errors:
            print(f"  - {error}", file=sys.stderr)
    return 0 if not validation.errors else 2


if __name__ == "__main__":
    sys.exit(main())
