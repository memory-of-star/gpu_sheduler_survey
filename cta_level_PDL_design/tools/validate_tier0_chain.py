#!/usr/bin/env python3
"""Fail-closed validator for the formal Tier 0.1 same-stream PDL chain.

The input is one ``bench/results_*`` directory produced through ``run_session.sh``.
An admissible Tier 0.1 result has six chain lengths, one independent all-edge
validation for each mode/configuration, 31 adjacent pairs in alternating order,
raw per-pair samples, reproducible bootstrap CIs, and a final off/on
``%globaltimer`` trace.  Monotonic invocation epochs bind those artifacts together;
the validator independently recomputes checkpoint/final digests and requires exact
integer trace metrics.  Model-implied chain depth is deliberately kept separate from
the number of simultaneously active grids observed in the trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODES = ("pdl_off", "pdl_on")
MAX_STAGES = 6
SEMANTICS = 3
WORK_CYCLES = 2_000_000
PROLOGUE_CYCLES = 1_000_000
TAIL_CYCLES = 1_000_000
BOOTSTRAP_REPS = 2000
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
TRACE_COLUMNS = {
    "tag", "rep", "mode", "epoch", "stage", "block_id", "sm_id", "t_launch",
    "t_dep_satisfied", "t_value_ready", "t_trigger", "t_end",
}

PREFIX_CONFIG = "CONFIG_TIER0_CHAIN "
PREFIX_VALIDATION = "VALIDATION_TIER0_CHAIN "
PREFIX_SAMPLE = "SAMPLE_TIER0_CHAIN "
PREFIX_SUMMARY = "SUMMARY tier0=chain "
PREFIX_TRACE = "TRACE_TIER0_CHAIN "


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def expect(self, condition: bool, message: str) -> bool:
        if not condition:
            self.errors.append(message)
            return False
        return True

    def error(self, message: str) -> None:
        self.errors.append(message)


def parse_fields(payload: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in shlex.split(payload):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def records(lines: Iterable[str], prefix: str) -> list[dict[str, str]]:
    return [parse_fields(line[len(prefix):]) for line in lines if line.startswith(prefix)]


def integer(record: dict[str, str], key: str, label: str,
            validation: Validation) -> int | None:
    try:
        return int(record[key])
    except (KeyError, TypeError, ValueError):
        validation.error(f"{label}: invalid/missing integer {key}={record.get(key)!r}")
        return None


def number(record: dict[str, str], key: str, label: str,
           validation: Validation) -> float | None:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        validation.error(f"{label}: invalid/missing number {key}={record.get(key)!r}")
        return None
    if not math.isfinite(value):
        validation.error(f"{label}: non-finite {key}={record.get(key)!r}")
        return None
    return value


def expect_int(record: dict[str, str], key: str, expected: int, label: str,
               validation: Validation) -> int | None:
    value = integer(record, key, label, validation)
    if value is not None:
        validation.expect(value == expected,
                          f"{label}: {key}={value}, expected {expected}")
    return value


def expect_text(record: dict[str, str], key: str, expected: str, label: str,
                validation: Validation) -> None:
    validation.expect(record.get(key) == expected,
                      f"{label}: {key}={record.get(key)!r}, expected {expected!r}")


def close_printed(actual: float, logged: float, decimals: int) -> bool:
    return math.isclose(actual, logged, rel_tol=2.0e-9,
                        abs_tol=2.0 * 10.0 ** -decimals)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def bootstrap_median(values: list[float], seed: int) -> tuple[float, float, float]:
    center = median(values)
    if len(values) == 1:
        return center, center, center
    state = seed & MASK64
    medians: list[float] = []
    count = len(values)
    for _ in range(BOOTSTRAP_REPS):
        sample: list[float] = []
        for _ in values:
            state, draw = splitmix64(state)
            sample.append(values[draw % count])
        medians.append(median(sample))
    medians.sort()
    return center, medians[int(0.025 * BOOTSTRAP_REPS)], \
        medians[int(0.975 * BOOTSTRAP_REPS)]


def model_depth(speedup: float, stages: int) -> float:
    return speedup / (2.0 - speedup) if 0.0 < speedup < 2.0 else float(stages)


def chain_mix(value: int) -> int:
    value &= MASK32
    value ^= value >> 16
    value = (value * 0x7FEB352D) & MASK32
    value ^= value >> 15
    value = (value * 0x846CA68B) & MASK32
    value ^= value >> 16
    return value & MASK32


def chain_initial(epoch: int, block: int) -> int:
    return chain_mix(epoch ^ 0xA5A5A5A5 ^
                     ((0x9E3779B9 * (block + 1)) & MASK32))


def chain_step(prior: int, epoch: int, stage: int, block: int) -> int:
    return chain_mix(prior ^ epoch ^
                     ((0x85EBCA6B * (stage + 1)) & MASK32) ^
                     ((0xC2B2AE35 * (block + 1)) & MASK32))


def digest_value(digest: int, value: int) -> int:
    return ((digest ^ (value & MASK32)) * 1_099_511_628_211) & MASK64


def expected_validation_digests(epoch: int, stages: int,
                                blocks: int) -> tuple[int, int]:
    checkpoint_digest = 1_469_598_103_934_665_603
    final_digest = 1_469_598_103_934_665_603
    for block in range(blocks):
        value = chain_initial(epoch, block)
        for stage in range(stages):
            value = chain_step(value, epoch, stage, block)
            checkpoint_digest = digest_value(checkpoint_digest, value)
        final_digest = digest_value(final_digest, value)
    return checkpoint_digest, final_digest


def stage_epoch_base(stages: int, warmup: int, repeats: int) -> int:
    invocations_per_stage = 2 + 2 * warmup + 2 * repeats
    return 1 + (stages - 1) * invocations_per_stage


def peak_concurrency(intervals: list[tuple[int, int, int]], stages: int) -> tuple[int, int]:
    events: list[tuple[int, int, int]] = []
    for start, end, stage in intervals:
        if end > start:
            events.append((start, +1, stage))
            events.append((end, -1, stage))
    # -1 precedes +1 at equal timestamps: half-open [start,end).
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    active_by_stage = [0] * stages
    peak_ctas = 0
    peak_grids = 0
    for _, delta, stage in events:
        active += delta
        active_by_stage[stage] += delta
        peak_ctas = max(peak_ctas, active)
        peak_grids = max(peak_grids, sum(value > 0 for value in active_by_stage))
    return peak_ctas, peak_grids


def trace_metrics(rows: list[dict[str, int | str]], stages: int, blocks: int,
                  mode: str, label: str, validation: Validation) -> dict[str, int | float]:
    by_stage: dict[int, list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        by_stage[int(row["stage"])].append(row)
    for stage in range(stages):
        stage_rows = by_stage.get(stage, [])
        validation.expect(len(stage_rows) == blocks,
                          f"{label}: stage={stage} rows={len(stage_rows)}, expected {blocks}")
        ids = sorted(int(row["block_id"]) for row in stage_rows)
        validation.expect(ids == list(range(blocks)),
                          f"{label}: stage={stage} block ids are incomplete/duplicated")

    intervals: list[tuple[int, int, int]] = []
    first = [2**64 - 1] * stages
    last = [0] * stages
    for row in rows:
        stage = int(row["stage"])
        launch = int(row["t_launch"])
        dep = int(row["t_dep_satisfied"])
        ready = int(row["t_value_ready"])
        trigger = int(row["t_trigger"])
        end = int(row["t_end"])
        ordered = 0 < launch <= dep <= ready <= end
        if mode == "pdl_on":
            ordered = ordered and ready <= trigger <= end and trigger > 0
        else:
            ordered = ordered and trigger == 0
        validation.expect(ordered, f"{label}: out-of-order timestamps in row {row}")
        first[stage] = min(first[stage], launch)
        last[stage] = max(last[stage], end)
        intervals.append((launch, end, stage))

    early = 0
    safe = 0
    serial = 0
    for stage in range(1, stages):
        predecessor_end = last[stage - 1]
        if first[stage] < predecessor_end:
            early += 1
        if first[stage] >= predecessor_end:
            serial += 1
        if all(int(row["t_dep_satisfied"]) >= predecessor_end
               for row in by_stage.get(stage, [])):
            safe += 1
    peak_ctas, peak_grids = peak_concurrency(intervals, stages)
    starts = [int(row["t_launch"]) for row in rows]
    ends = [int(row["t_end"]) for row in rows]
    makespan_ms = (max(ends) - min(starts)) / 1.0e6 if starts and ends else 0.0
    return {
        "peak_active_ctas": peak_ctas,
        "peak_active_grids": peak_grids,
        "early_links": early,
        "dependency_safe_links": safe,
        "serial_links": serial,
        "makespan_ms": makespan_ms,
    }


def campaign_checks(result_dir: Path, validation: Validation) -> str | None:
    failures = result_dir / "failures.log"
    if failures.exists():
        try:
            validation.expect(not failures.read_text(errors="replace").strip(),
                              "campaign failures.log is non-empty")
        except OSError as exc:
            validation.error(f"cannot read {failures}: {exc}")
    invalid = sorted(path.name for path in result_dir.glob("*.invalid"))
    validation.expect(not invalid, f"campaign has invalid markers: {invalid}")
    gate_verdict: str | None = None
    gate = result_dir / "gate.json"
    if gate.exists():
        try:
            value = json.loads(gate.read_text())
            gate_verdict = str(value.get("verdict"))
            validation.expect(gate_verdict in {"GO", "LLM_ONLY", "STOP"},
                              f"campaign gate.json verdict is not admissible: "
                              f"{gate_verdict!r}")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            validation.error(f"cannot parse {gate}: {exc}")
    return gate_verdict


def validate_log(path: Path, validation: Validation) -> dict[str, Any]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        validation.error(f"cannot read {path}: {exc}")
        return {}
    configs = records(lines, PREFIX_CONFIG)
    validations = records(lines, PREFIX_VALIDATION)
    samples = records(lines, PREFIX_SAMPLE)
    summaries = records(lines, PREFIX_SUMMARY)
    trace_records = records(lines, PREFIX_TRACE)
    validation.expect(len(configs) == 1,
                      f"expected one CONFIG_TIER0_CHAIN, got {len(configs)}")
    validation.expect(len(trace_records) == 1,
                      f"expected one TRACE_TIER0_CHAIN, got {len(trace_records)}")
    if not configs:
        return {}
    config = configs[0]
    label = "Tier 0.1 CONFIG"
    expect_int(config, "semantics", SEMANTICS, label, validation)
    sms = integer(config, "sms", label, validation) or 0
    blocks = integer(config, "blocks", label, validation) or 0
    threads = integer(config, "threads", label, validation) or 0
    repeats = integer(config, "repeats", label, validation) or 0
    warmup = integer(config, "warmup", label, validation) or 0
    validation.expect(sms > 0 and blocks == sms,
                      f"{label}: blocks={blocks}, sms={sms}; expected blocks==sms>0")
    validation.expect(threads == 128, f"{label}: threads={threads}, expected 128")
    expect_int(config, "stages_max", MAX_STAGES, label, validation)
    expect_int(config, "work_cycles", WORK_CYCLES, label, validation)
    expect_int(config, "prologue_cycles", PROLOGUE_CYCLES, label, validation)
    expect_int(config, "tail_cycles", TAIL_CYCLES, label, validation)
    validation.expect(repeats >= 31,
                      f"{label}: formal repeats={repeats}, expected >=31")
    validation.expect(warmup >= 3, f"{label}: warmup={warmup}, expected >=3")
    expect_int(config, "allow_short", 0, label, validation)
    for key, expected in {
        "pairing": "adjacent_alternating",
        "timer": "globaltimer_makespan",
        "trace_timer": "globaltimer",
        "trigger_pdl": "after_value_ready",
        "poison": "epoch_seed_every_invocation",
        "epoch_schedule": "monotonic_all_invocations",
        "validation": "independent_all_edges",
    }.items():
        expect_text(config, key, expected, label, validation)

    # One untimed full-edge validation for every chain length and mode.
    validation_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for index, record in enumerate(validations, 1):
        item_label = f"VALIDATION_TIER0_CHAIN#{index}"
        stages = integer(record, "stages", item_label, validation)
        mode = record.get("mode", "")
        if stages is None or mode not in MODES:
            validation.error(f"{item_label}: stages/mode invalid")
            continue
        key = (stages, mode)
        validation.expect(key not in validation_by_key,
                          f"duplicate validation record {key}")
        validation_by_key[key] = record
        expect_int(record, "semantics", SEMANTICS, item_label, validation)
        expect_text(record, "tag", f"t01_s{stages}", item_label, validation)
        expected_epoch = stage_epoch_base(stages, warmup, repeats) + MODES.index(mode)
        epoch = expect_int(record, "epoch", expected_epoch, item_label, validation)
        expect_int(record, "checked_edges", (stages - 1) * blocks,
                   item_label, validation)
        expect_int(record, "checked_stage_outputs", stages * blocks,
                   item_label, validation)
        expect_int(record, "checked_final_outputs", blocks, item_label, validation)
        expect_int(record, "mismatches", 0, item_label, validation)
        expect_int(record, "trace_complete", 1, item_label, validation)
        expect_int(record, "valid", 1, item_label, validation)
        observed = integer(record, "observed_digest", item_label, validation)
        expected = integer(record, "expected_digest", item_label, validation)
        observed_final = integer(record, "observed_final_digest", item_label, validation)
        expected_final = integer(record, "expected_final_digest", item_label, validation)
        if epoch is not None:
            recomputed, recomputed_final = expected_validation_digests(
                epoch, stages, blocks)
            if observed is not None and expected is not None:
                validation.expect(observed == expected == recomputed,
                                  f"{item_label}: checkpoint digest does not match "
                                  f"independent recomputation {recomputed}")
            if observed_final is not None and expected_final is not None:
                validation.expect(observed_final == expected_final == recomputed_final,
                                  f"{item_label}: final digest does not match independent "
                                  f"recomputation {recomputed_final}")
        links = stages - 1
        expect_int(record, "dependency_safe_links", links, item_label, validation)
        if mode == "pdl_on":
            early = integer(record, "early_links", item_label, validation)
            serial = integer(record, "serial_links", item_label, validation)
            if early is not None and serial is not None:
                validation.expect(0 <= early <= links and 0 <= serial <= links,
                                  f"{item_label}: overlap link counts out of range")
                validation.expect(early + serial == links,
                                  f"{item_label}: early+serial links != {links}")
        else:
            expect_int(record, "early_links", 0, item_label, validation)
            expect_int(record, "serial_links", links, item_label, validation)
    expected_validation_keys = {
        (stages, mode) for stages in range(1, MAX_STAGES + 1) for mode in MODES
    }
    validation.expect(set(validation_by_key) == expected_validation_keys,
                      "independent validation matrix is incomplete")

    sample_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    actual_sequence: list[tuple[int, int, int, str]] = []
    for index, record in enumerate(samples, 1):
        item_label = f"SAMPLE_TIER0_CHAIN#{index}"
        stages = integer(record, "stages", item_label, validation)
        rep = integer(record, "rep", item_label, validation)
        order = integer(record, "order", item_label, validation)
        mode = record.get("mode", "")
        if None in (stages, rep, order) or mode not in MODES:
            validation.error(f"{item_label}: invalid identity")
            continue
        assert stages is not None and rep is not None and order is not None
        identity_in_range = validation.expect(
            1 <= stages <= MAX_STAGES and 0 <= rep < repeats and order in (0, 1),
            f"{item_label}: identity out of range stages={stages} rep={rep} order={order}")
        key = (stages, rep, mode)
        unique = validation.expect(key not in sample_by_key, f"duplicate sample {key}")
        actual_sequence.append((stages, rep, order, mode))
        expect_int(record, "semantics", SEMANTICS, item_label, validation)
        expect_text(record, "tag", f"t01_s{stages}", item_label, validation)
        trace_complete = expect_int(record, "trace_complete", 1, item_label, validation)
        valid = expect_int(record, "valid", 1, item_label, validation)
        expected_epoch = (stage_epoch_base(stages, warmup, repeats) + 2 +
                          2 * warmup + 2 * rep + order)
        epoch = expect_int(record, "epoch", expected_epoch, item_label, validation)
        elapsed = number(record, "makespan_ms", item_label, validation)
        if elapsed is not None:
            validation.expect(elapsed > 0, f"{item_label}: makespan must be positive")
        peak_ctas = integer(record, "peak_active_ctas", item_label, validation)
        peak_grids = integer(record, "peak_active_grids", item_label, validation)
        if peak_ctas is not None:
            validation.expect(blocks <= peak_ctas <= stages * blocks,
                              f"{item_label}: peak_active_ctas={peak_ctas} out of range")
        if peak_grids is not None:
            validation.expect(1 <= peak_grids <= stages,
                              f"{item_label}: peak_active_grids={peak_grids} out of range")
        links = stages - 1
        safe = expect_int(record, "dependency_safe_links", links, item_label, validation)
        if mode == "pdl_on":
            early = integer(record, "early_links", item_label, validation)
            serial = integer(record, "serial_links", item_label, validation)
            if early is not None and serial is not None:
                validation.expect(0 <= early <= links and 0 <= serial <= links,
                                  f"{item_label}: overlap link counts out of range")
                validation.expect(early + serial == links,
                                  f"{item_label}: early+serial links != {links}")
            # Overlap is an observed outcome, not an admission condition. A complete,
            # dependency-safe PDL sample with peak_grids=1 must remain in the statistics.
        else:
            early = expect_int(record, "early_links", 0, item_label, validation)
            serial = expect_int(record, "serial_links", links, item_label, validation)
            if peak_grids is not None:
                validation.expect(peak_grids == 1,
                                  f"{item_label}: pdl_off must be serial")
        required = (epoch, elapsed, peak_ctas, peak_grids, early, safe, serial,
                    trace_complete, valid)
        if identity_in_range and unique and all(value is not None for value in required):
            sample_by_key[key] = {
                "epoch": epoch,
                "makespan_ms": elapsed,
                "peak_active_ctas": peak_ctas,
                "peak_active_grids": peak_grids,
                "early_links": early,
                "dependency_safe_links": safe,
                "serial_links": serial,
                "trace_complete": trace_complete,
                "valid": valid,
            }

    expected_sequence: list[tuple[int, int, int, str]] = []
    for stages in range(1, MAX_STAGES + 1):
        for rep in range(repeats):
            mode_order = ("pdl_off", "pdl_on") if rep % 2 == 0 \
                else ("pdl_on", "pdl_off")
            expected_sequence.extend((stages, rep, order, mode)
                                     for order, mode in enumerate(mode_order))
    validation.expect(actual_sequence == expected_sequence,
                      "SAMPLE records are not adjacent pairs in alternating execution order")
    actual_record_order: list[tuple[str, str, str, str, str]] = []
    for line in lines:
        if line.startswith(PREFIX_VALIDATION):
            item = parse_fields(line[len(PREFIX_VALIDATION):])
            actual_record_order.append(("validation", item.get("stages", ""), "", "",
                                        item.get("mode", "")))
        elif line.startswith(PREFIX_SAMPLE):
            item = parse_fields(line[len(PREFIX_SAMPLE):])
            actual_record_order.append(("sample", item.get("stages", ""),
                                        item.get("rep", ""), item.get("order", ""),
                                        item.get("mode", "")))
    expected_record_order: list[tuple[str, str, str, str, str]] = []
    for stages in range(1, MAX_STAGES + 1):
        expected_record_order.extend([
            ("validation", str(stages), "", "", "pdl_off"),
            ("validation", str(stages), "", "", "pdl_on"),
        ])
        for rep in range(repeats):
            mode_order = ("pdl_off", "pdl_on") if rep % 2 == 0 \
                else ("pdl_on", "pdl_off")
            expected_record_order.extend(
                ("sample", str(stages), str(rep), str(order), mode)
                for order, mode in enumerate(mode_order)
            )
    validation.expect(actual_record_order == expected_record_order,
                      "VALIDATION/SAMPLE records are not in the executed stage order")
    expected_sample_keys = {
        (stages, rep, mode)
        for stages in range(1, MAX_STAGES + 1)
        for rep in range(repeats) for mode in MODES
    }
    validation.expect(set(sample_by_key) == expected_sample_keys,
                      "formal SAMPLE matrix is incomplete or contains extras")

    summary_by_stage: dict[int, dict[str, str]] = {}
    for index, summary in enumerate(summaries, 1):
        item_label = f"SUMMARY tier0=chain#{index}"
        stages = integer(summary, "stages", item_label, validation)
        if stages is None:
            continue
        validation.expect(stages not in summary_by_stage,
                          f"duplicate SUMMARY stages={stages}")
        summary_by_stage[stages] = summary
        expect_int(summary, "semantics", SEMANTICS, item_label, validation)
        expect_text(summary, "tag", f"t01_s{stages}", item_label, validation)
        for key, expected in (("blocks", blocks), ("threads", threads), ("sms", sms),
                              ("work_cycles", WORK_CYCLES),
                              ("prologue_cycles", PROLOGUE_CYCLES),
                              ("tail_cycles", TAIL_CYCLES),
                              ("warmup", warmup), ("repeats", repeats), ("valid", 1)):
            expect_int(summary, key, expected, item_label, validation)
        for key, expected in {
            "pairing": "adjacent_alternating",
            "timer": "globaltimer_makespan",
            "trace_timer": "globaltimer",
            "trigger_pdl": "after_value_ready",
            "poison": "epoch_seed_every_invocation",
            "epoch_schedule": "monotonic_all_invocations",
            "validation": "independent_all_edges",
        }.items():
            expect_text(summary, key, expected, item_label, validation)

        if not all((stages, rep, mode) in sample_by_key
                   for rep in range(repeats) for mode in MODES):
            continue
        off = [float(sample_by_key[stages, rep, "pdl_off"]["makespan_ms"])
               for rep in range(repeats)]
        on = [float(sample_by_key[stages, rep, "pdl_on"]["makespan_ms"])
              for rep in range(repeats)]
        paired = [off[rep] / on[rep] for rep in range(repeats)]
        depths = [model_depth(value, stages) for value in paired]
        peak_grids = [float(sample_by_key[stages, rep, "pdl_on"]["peak_active_grids"])
                      for rep in range(repeats)]
        peak_ctas = [float(sample_by_key[stages, rep, "pdl_on"]["peak_active_ctas"])
                     for rep in range(repeats)]
        seed = 0xC7100000 + stages * 0x100
        expected_stats = {
            "pdl_off": bootstrap_median(off, seed + 1),
            "pdl_on": bootstrap_median(on, seed + 2),
            "paired_speedup": bootstrap_median(paired, seed + 3),
            "model_implied_chain_depth": bootstrap_median(depths, seed + 4),
            "pdl_on_peak_active_ctas": bootstrap_median(peak_ctas, seed + 5),
            "pdl_on_peak_active_grids": bootstrap_median(peak_grids, seed + 6),
        }
        for prefix, triplet in expected_stats.items():
            suffixes = (("_ms", "_ci_low", "_ci_high")
                        if prefix in ("pdl_off", "pdl_on")
                        else (("", "_ci_low", "_ci_high")
                              if prefix in ("paired_speedup", "model_implied_chain_depth")
                              else ("_median", "_ci_low", "_ci_high")))
            decimals = 1 if prefix.startswith("pdl_on_peak") else 6
            for suffix, expected in zip(suffixes, triplet):
                key = prefix + suffix
                logged = number(summary, key, item_label, validation)
                if logged is not None:
                    validation.expect(close_printed(expected, logged, decimals),
                                      f"{item_label}: {key}={logged}, recomputed {expected}")
        for key, expected in (
            ("pdl_on_peak_active_grids_max", max(peak_grids)),
            ("pdl_on_peak_active_ctas_max", max(peak_ctas)),
        ):
            logged = number(summary, key, item_label, validation)
            if logged is not None:
                validation.expect(close_printed(expected, logged, 1),
                                  f"{item_label}: {key}={logged}, recomputed {expected}")
    validation.expect(set(summary_by_stage) == set(range(1, MAX_STAGES + 1)),
                      "SUMMARY matrix must contain stages 1..6 exactly once")

    return {
        "sms": sms,
        "blocks": blocks,
        "threads": threads,
        "repeats": repeats,
        "warmup": warmup,
        "sample_count": len(sample_by_key),
        "validation_count": len(validation_by_key),
        "summary_count": len(summary_by_stage),
        "samples": sample_by_key,
        "trace_record": trace_records[0] if trace_records else {},
    }


def validate_trace(path: Path, log_data: dict[str, Any], validation: Validation) -> dict[str, Any]:
    blocks = int(log_data.get("blocks", 0))
    sms = int(log_data.get("sms", 0))
    repeats = int(log_data.get("repeats", 0))
    try:
        handle = path.open(newline="")
    except OSError as exc:
        validation.error(f"cannot read trace {path}: {exc}")
        return {}
    parsed: list[dict[str, int | str]] = []
    with handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        validation.expect(fields == TRACE_COLUMNS,
                          f"trace columns={sorted(fields)}, expected={sorted(TRACE_COLUMNS)}")
        seen: set[tuple[str, int, int]] = set()
        for line_number, row in enumerate(reader, 2):
            label = f"trace row {line_number}"
            if None in row or any(value is None for value in row.values()):
                validation.error(f"{label}: malformed CSV")
                continue
            validation.expect(row.get("tag") == "t01_s6",
                              f"{label}: tag={row.get('tag')!r}")
            mode = row.get("mode", "")
            if mode not in MODES:
                validation.error(f"{label}: invalid mode={mode!r}")
                continue
            numeric: dict[str, int] = {}
            for key in TRACE_COLUMNS - {"tag", "mode"}:
                try:
                    numeric[key] = int(row[key])
                except (KeyError, TypeError, ValueError):
                    validation.error(f"{label}: invalid {key}={row.get(key)!r}")
                    break
            else:
                row_ok = validation.expect(
                    numeric["rep"] == repeats - 1,
                    f"{label}: rep={numeric['rep']}, expected {repeats - 1}")
                row_ok = validation.expect(
                    0 <= numeric["stage"] < MAX_STAGES,
                    f"{label}: invalid stage") and row_ok
                row_ok = validation.expect(
                    0 <= numeric["block_id"] < blocks,
                    f"{label}: invalid block_id") and row_ok
                row_ok = validation.expect(
                    0 <= numeric["sm_id"] < sms,
                    f"{label}: invalid sm_id") and row_ok
                row_ok = validation.expect(
                    numeric["epoch"] > 0,
                    f"{label}: invalid epoch") and row_ok
                identity = (mode, numeric["stage"], numeric["block_id"])
                unique = validation.expect(identity not in seen,
                                           f"{label}: duplicate {identity}")
                if row_ok and unique:
                    seen.add(identity)
                    parsed.append({"tag": "t01_s6", "mode": mode, **numeric})

    modes: dict[str, Any] = {}
    samples = log_data.get("samples", {})
    for mode in MODES:
        mode_rows = [row for row in parsed if row["mode"] == mode]
        validation.expect(len(mode_rows) == MAX_STAGES * blocks,
                          f"trace mode={mode}: rows={len(mode_rows)}, "
                          f"expected {MAX_STAGES * blocks}")
        metrics = trace_metrics(mode_rows, MAX_STAGES, blocks, mode,
                                f"trace mode={mode}", validation)
        expected_sample = samples.get((MAX_STAGES, repeats - 1, mode))
        if expected_sample is None:
            validation.error(f"trace mode={mode}: final SAMPLE is missing")
        else:
            expected_epoch = int(expected_sample["epoch"])
            validation.expect(
                all(int(row["epoch"]) == expected_epoch for row in mode_rows),
                f"trace mode={mode}: row epoch does not match SAMPLE epoch={expected_epoch}")
            computed_ms = float(metrics["makespan_ms"])
            logged_ms = float(expected_sample["makespan_ms"])
            validation.expect(close_printed(computed_ms, logged_ms, 6),
                              f"trace mode={mode}: makespan_ms={computed_ms}, "
                              f"SAMPLE={logged_ms}")
            for key in ("peak_active_ctas", "peak_active_grids", "early_links",
                        "dependency_safe_links", "serial_links"):
                computed = int(metrics[key])
                logged = int(expected_sample[key])
                validation.expect(computed == logged,
                                  f"trace mode={mode}: {key}={computed}, SAMPLE={logged}")
        links = MAX_STAGES - 1
        if mode == "pdl_on":
            validation.expect(metrics["dependency_safe_links"] == links,
                              "pdl_on final trace returned before predecessor end")
            validation.expect(
                int(metrics["early_links"]) + int(metrics["serial_links"]) == links,
                "pdl_on final trace has an incomplete adjacent-link classification")
        else:
            validation.expect(metrics["serial_links"] == links,
                              "pdl_off final trace is not serial")
        modes[mode] = metrics
    return {"row_count": len(parsed), "modes": modes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path,
                        help="formal bench/results_* directory")
    parser.add_argument("--json", type=Path,
                        help="write structured validation result")
    args = parser.parse_args()

    validation = Validation()
    if args.json:
        try:
            args.json.unlink(missing_ok=True)
        except OSError as exc:
            validation.error(f"cannot clear stale validation JSON {args.json}: {exc}")
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        validation.error(f"result directory does not exist: {result_dir}")
    gate_verdict: str | None = None
    log_data: dict[str, Any] = {}
    trace_data: dict[str, Any] = {}
    if result_dir.is_dir():
        try:
            gate_verdict = campaign_checks(result_dir, validation)
            log_data = validate_log(result_dir / "tier0_facts.log", validation)
            trace_record = log_data.get("trace_record", {})
            expect_int(trace_record, "semantics", SEMANTICS,
                       "TRACE_TIER0_CHAIN", validation)
            expect_text(trace_record, "tag", "t01_s6", "TRACE_TIER0_CHAIN", validation)
            expect_text(trace_record, "modes", "pdl_off,pdl_on",
                        "TRACE_TIER0_CHAIN", validation)
            expect_text(trace_record, "timer", "globaltimer",
                        "TRACE_TIER0_CHAIN", validation)
            expect_int(trace_record, "epoch_in_rows", 1,
                       "TRACE_TIER0_CHAIN", validation)
            declared_trace = trace_record.get("path", "")
            declared_path = Path(declared_trace)
            if not declared_path.is_absolute():
                declared_path = result_dir.parent / declared_path
            validation.expect(
                bool(declared_trace) and declared_path.resolve() ==
                (result_dir / "tier0_chain_trace.csv").resolve(),
                f"TRACE_TIER0_CHAIN path={declared_trace!r} does not resolve to the "
                "validated trace artifact")
            if log_data:
                expect_int(trace_record, "rep", int(log_data.get("repeats", 0)) - 1,
                           "TRACE_TIER0_CHAIN", validation)
            trace_data = validate_trace(result_dir / "tier0_chain_trace.csv",
                                        log_data, validation)
        except Exception as exc:  # fail closed even for malformed adversarial artifacts
            validation.error(f"validator internal error: {type(exc).__name__}: {exc}")

    report: dict[str, Any] = {
        "schema": 1,
        "status": "PASS" if not validation.errors else "FAIL",
        "result_dir": str(result_dir),
        "gate_verdict": gate_verdict,
        "configuration_count": int(log_data.get("summary_count", 0)),
        "validation_count": int(log_data.get("validation_count", 0)),
        "paired_repetitions": int(log_data.get("repeats", 0)) * MAX_STAGES,
        "sample_count": int(log_data.get("sample_count", 0)),
        "trace_row_count": int(trace_data.get("row_count", 0)),
        "trace": trace_data,
        "error_count": len(validation.errors),
        "errors": validation.errors,
    }
    if args.json:
        temporary = args.json.with_name(f".{args.json.name}.tmp.{os.getpid()}")
        try:
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temporary.replace(args.json)
        except OSError as exc:
            validation.error(f"cannot write {args.json}: {exc}")
            report["status"] = "FAIL"
            report["error_count"] = len(validation.errors)
            report["errors"] = validation.errors
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    print(
        "TIER0_CHAIN_VALIDATION "
        f"status={report['status']} configurations={report['configuration_count']} "
        f"validation_records={report['validation_count']} "
        f"paired_repetitions={report['paired_repetitions']} "
        f"samples={report['sample_count']} trace_rows={report['trace_row_count']} "
        f"errors={len(validation.errors)}"
    )
    if validation.errors:
        for error in validation.errors:
            print(f"  - {error}", file=sys.stderr)
    return 0 if not validation.errors else 2


if __name__ == "__main__":
    sys.exit(main())
