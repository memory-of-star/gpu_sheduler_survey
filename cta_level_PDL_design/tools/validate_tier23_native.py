#!/usr/bin/env python3
"""Strict validator/analyzer for the native Tier 2/3 campaign.

The validator never trusts SUMMARY records alone.  It reparses the raw invocation stream,
recomputes epoch-bound correctness digests and deterministic bootstrap intervals, binds every
final SAMPLE to its declared CSV trace, and enforces the formal §7.1/§7.3-§7.6 matrix.  Any
exception becomes a current structured FAIL file; a stale PASS is removed before parsing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

MASK64 = (1 << 64) - 1
SEMANTICS = 1
BOOTSTRAPS = 2000


def mix64(x: int) -> int:
    x &= MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return x & MASK64


def value(epoch: int, index: int, stage: int = 0) -> int:
    return mix64(
        epoch * 0x9E3779B97F4A7C15
        ^ (index + 1) * 0xD1B54A32D192ED03
        ^ (stage + 1) * 0x94D049BB133111EB
    )


def fnv64_words(values: Iterable[int], width: int) -> int:
    h = 1469598103934665603
    for raw in values:
        v = raw & ((1 << (8 * width)) - 1)
        for b in range(width):
            h ^= (v >> (8 * b)) & 0xFF
            h = (h * 1099511628211) & MASK64
    return h


def median(values: Iterable[float]) -> float:
    vals = sorted(values)
    if not vals:
        raise ValueError("median of empty sequence")
    return vals[len(vals) // 2]


def bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap of empty sequence")
    meds: list[float] = []
    n = len(values)
    for b in range(BOOTSTRAPS):
        sample = []
        for i in range(n):
            r = mix64(
                seed
                ^ (b + 1) * 0x9E3779B97F4A7C15
                ^ (i + 1) * 0xD1B54A32D192ED03
            )
            sample.append(values[r % n])
        meds.append(median(sample))
    meds.sort()
    return meds[int(0.025 * (len(meds) - 1))], meds[int(0.975 * (len(meds) - 1))]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def parse_record(line: str, lineno: int) -> tuple[str, dict[str, str]] | None:
    stripped = line.strip()
    if not stripped:
        return None
    prefix = stripped.split(None, 1)[0]
    if prefix not in {
        "CONFIG_TIER23",
        "VALIDATION_TIER23",
        "WARMUP_TIER23",
        "SAMPLE_TIER23",
        "SUMMARY_TIER23",
        "TRACE_TIER23",
    }:
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError as exc:
        raise ValueError(f"line {lineno}: malformed quoting: {exc}") from exc
    out: dict[str, str] = {"_line": str(lineno)}
    for token in tokens[1:]:
        if "=" not in token:
            continue  # device names contain spaces; fields of interest remain key=value
        key, val = token.split("=", 1)
        if not key or key in out:
            raise ValueError(f"line {lineno}: duplicate/malformed key {key!r}")
        out[key] = val
    return prefix, out


def require(record: dict[str, str], *keys: str) -> None:
    missing = [k for k in keys if k not in record]
    if missing:
        raise ValueError(f"line {record.get('_line', '?')}: missing {','.join(missing)}")


def as_int(record: dict[str, str], key: str) -> int:
    require(record, key)
    try:
        return int(record[key], 10)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {record.get('_line', '?')}: {key} is not an integer: {record[key]!r}"
        ) from exc


def as_float(record: dict[str, str], key: str) -> float:
    require(record, key)
    try:
        result = float(record[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {record.get('_line', '?')}: {key} is not numeric: {record[key]!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"line {record.get('_line', '?')}: {key} is non-finite")
    return result


def ratio_value(record: dict[str, str]) -> int:
    """Read current numeric ratio; accept the pre-validator smoke's `1:N` label only."""
    require(record, "ratio")
    raw = record["ratio"]
    if raw.startswith("1:"):
        raw = raw.split(":", 1)[1]
    try:
        return int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"line {record.get('_line', '?')}: malformed ratio {raw!r}") from exc


def close(errors: list[str], label: str, actual: float, expected: float,
          tolerance: float = 5e-7) -> None:
    if abs(actual - expected) > max(tolerance, tolerance * abs(expected)):
        errors.append(f"{label}: got {actual}, recomputed {expected}")


def parents(structure: str, degree: int, p: int, c: int, child: int) -> list[int]:
    if structure == "self":
        return [child] if child < p else []
    d = min(max(degree, 1), p)
    if structure == "interval":
        span = p - d
        lo = (span * child) // (c - 1) if c > 1 else 0
        return list(range(lo, lo + d))
    if structure == "strided":
        stride = max(1, p // d)
        return [(child + k * stride) % p for k in range(d)]
    raise ValueError(f"unsupported strict-validator structure {structure!r}")


def pe_expected_digest(config: dict[str, str], epoch: int, validation: bool) -> int:
    p = as_int(config, "P")
    c = as_int(config, "C")
    degree = as_int(config, "degree")
    structure = config["structure"]
    outputs: list[int] = []
    for child in range(c):
        ps = parents(structure, degree, p, c, child)
        if validation:
            h = 1469598103934665603
            for parent in ps:
                h = mix64(h ^ value(epoch, parent, 0) ^ (parent + 1) * 0x9E3779B97F4A7C15)
            outputs.append(h)
        else:
            outputs.append(value(epoch, ps[-1], 0) if ps else 0)
    return fnv64_words(outputs, 8)


def diamond_stage(epoch: int, block: int, stage: int) -> int:
    x0 = value(epoch, block, 0)
    if stage == 0:
        return x0
    x1 = mix64(x0 ^ 0x1111111111111111)
    if stage == 1:
        return x1
    x2 = mix64(x0 ^ 0x3333333333333333)
    if stage == 2:
        return x2
    rot = ((x2 << 17) | (x2 >> 47)) & MASK64
    return mix64(x1 ^ rot ^ 0x4444444444444444)


def diamond_expected_digest(config: dict[str, str], epoch: int) -> int:
    blocks = as_int(config, "blocks")
    return fnv64_words(
        (diamond_stage(epoch, block, stage) for stage in range(4) for block in range(blocks)),
        8,
    )


def c1_word(epoch: int, tile: int, word: int) -> int:
    v = mix64(
        epoch * 0x9E3779B97F4A7C15
        ^ (tile + 1) * 0xD1B54A32D192ED03
        ^ (word + 1) * 0x94D049BB133111EB
    )
    return (v ^ (v >> 32)) & 0xFFFFFFFF


def c1_expected_digest(config: dict[str, str], epoch: int) -> int:
    tiles = as_int(config, "tiles")
    words = as_int(config, "words_per_tile")
    return fnv64_words((c1_word(epoch, t, w) for t in range(tiles) for w in range(words)), 4)


def c1_digest_job(job: tuple[dict[str, str], int]) -> tuple[int, int]:
    """Top-level worker used to recompute large all-word C1 digests in parallel."""
    config, epoch = job
    return epoch, c1_expected_digest(config, epoch)


def clc_expected_digest(config: dict[str, str], epoch: int) -> int:
    tiles = as_int(config, "tiles")
    vals = (mix64(value(epoch, i, 0) ^ 0xC6BC279692B5CC83) for i in range(tiles))
    return fnv64_words(vals, 8)


def expected_digest(experiment: str, config: dict[str, str], epoch: int,
                    validation: bool) -> int:
    if experiment in {"protocol", "encoding"}:
        return pe_expected_digest(config, epoch, validation)
    if experiment == "diamond":
        return diamond_expected_digest(config, epoch)
    if experiment == "c1":
        return c1_expected_digest(config, epoch)
    if experiment == "clc":
        return clc_expected_digest(config, epoch)
    raise ValueError(f"unknown experiment {experiment}")


TRACE_COLUMNS = [
    "tag", "experiment", "mode", "epoch", "kernel_id", "block_id", "sm_id",
    "t_start", "t_ready", "t_wait_begin", "t_dep", "t_end", "poll_loads",
    "metadata_loads", "decode_ns", "aux",
]


def load_trace(path: Path) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != TRACE_COLUMNS:
            raise ValueError(f"{path}: trace columns mismatch: {reader.fieldnames}")
        rows: list[dict[str, int | str]] = []
        for lineno, raw in enumerate(reader, 2):
            row: dict[str, int | str] = {}
            for key in ("tag", "experiment", "mode"):
                if raw.get(key) in {None, ""}:
                    raise ValueError(f"{path}:{lineno}: missing {key}")
                row[key] = raw[key]
            for key in TRACE_COLUMNS[3:]:
                try:
                    value_i = int(raw[key], 10)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: malformed integer {key}") from exc
                if value_i < 0:
                    raise ValueError(f"{path}:{lineno}: negative integer {key}")
                row[key] = value_i
            rows.append(row)
    return rows


def keyed(rows: list[dict[str, int | str]], kernel: int) -> dict[int, dict[str, int | str]]:
    result: dict[int, dict[str, int | str]] = {}
    for row in rows:
        if row["kernel_id"] != kernel:
            continue
        block = int(row["block_id"])
        if block in result:
            raise ValueError(f"duplicate trace kernel={kernel} block={block}")
        result[block] = row
    return result


def check_time_order(row: dict[str, int | str], fields: tuple[str, ...]) -> bool:
    vals = [int(row[f]) for f in fields]
    return all(v > 0 for v in vals) and all(a <= b for a, b in zip(vals, vals[1:]))


def interval_cover(config: dict[str, str], child: int) -> tuple[int, int]:
    ps = parents(config["structure"], as_int(config, "degree"), as_int(config, "P"),
                 as_int(config, "C"), child)
    return min(ps), max(ps)


def validate_pe_trace(errors: list[str], config: dict[str, str], sample: dict[str, str],
                      rows: list[dict[str, int | str]], mode: str) -> None:
    pcount, ccount = as_int(config, "P"), as_int(config, "C")
    bgcount = as_int(config, "background_blocks")
    p, c, bg = keyed(rows, 0), keyed(rows, 1), keyed(rows, 2)
    if set(p) != set(range(pcount)) or set(c) != set(range(ccount)) or set(bg) != set(range(bgcount)):
        errors.append(f"{config['tag']}/{mode}: incomplete producer/consumer/background trace")
        return
    for row in p.values():
        if not check_time_order(row, ("t_start", "t_ready", "t_end")):
            errors.append(f"{config['tag']}/{mode}: producer timestamp order")
            return
    for row in c.values():
        if not check_time_order(row, ("t_start", "t_wait_begin", "t_dep", "t_end")):
            errors.append(f"{config['tag']}/{mode}: consumer timestamp order")
            return
    for row in bg.values():
        if not check_time_order(row, ("t_start", "t_end")):
            errors.append(f"{config['tag']}/{mode}: background timestamp order")
            return
    first = min(min(int(r["t_start"]) for r in p.values()),
                min(int(r["t_start"]) for r in c.values()))
    last = max(max(int(r["t_end"]) for r in p.values()),
               max(int(r["t_end"]) for r in c.values()))
    waits = [float(int(r["t_dep"]) - int(r["t_wait_begin"])) for r in c.values()]
    decodes = [float(int(r["decode_ns"])) for r in c.values()]
    wakes: list[float] = []
    p_last = max(int(r["t_end"]) for r in p.values())
    for child, row in c.items():
        dep = int(row["t_dep"])
        if mode == "grid":
            satisfied = p_last
        elif mode == "none":
            continue
        elif mode == "monotonic-prefix":
            satisfied = max(int(p[i]["t_ready"]) for i in range(child + 1))
        elif mode in {"fixed-spin", "backoff"}:
            satisfied = int(p[child]["t_ready"])
        elif mode == "interval":
            lo, hi = interval_cover(config, child)
            satisfied = max(int(p[i]["t_ready"]) for i in range(lo, hi + 1))
        else:  # bitmask / CSR exact
            ps = parents(config["structure"], as_int(config, "degree"), pcount, ccount, child)
            satisfied = max(int(p[i]["t_ready"]) for i in ps)
        if dep < satisfied:
            errors.append(f"{config['tag']}/{mode}: wait returned before readiness")
            return
        if mode != "grid":
            wakes.append(float(dep - satisfied))
    if mode == "none":
        sentinel_parent = as_int(config, "ceiling_sentinel_parent")
        if int(c[0]["t_dep"]) > int(p[sentinel_parent]["t_ready"]):
            errors.append(
                f"{config['tag']}/{mode}: sentinel RAW snapshot was not before parent ready"
            )
            return
    bg_first = min(int(r["t_start"]) for r in bg.values())
    bg_last = max(int(r["t_end"]) for r in bg.values())
    overlap = sum(int(r["t_start"]) < last and int(r["t_end"]) > first for r in bg.values())
    poll_loads = sum(int(r["poll_loads"]) for r in p.values()) + sum(
        int(r["poll_loads"]) for r in c.values())
    metadata = sum(int(r["metadata_loads"]) for r in c.values())
    schedule_latch = sum(int(r["metadata_loads"]) for r in p.values())
    bg_bytes = bgcount * as_int(config, "threads") * as_int(config, "background_iterations") * 8
    checks = {
        "ms": (last - first) / 1e6,
        "wait_ns": median(waits),
        "wake_ns": median(wakes) if wakes else 0.0,
        "decode_ns": median(decodes),
        "background_ms": (bg_last - bg_first) / 1e6,
        "background_gbps": bg_bytes / (bg_last - bg_first),
    }
    for key, recomputed in checks.items():
        close(errors, f"{config['tag']}/{mode} trace {key}", as_float(sample, key), recomputed)
    for key, recomputed in {
        "poll_loads": poll_loads,
        "metadata_loads": metadata,
        "ceiling_schedule_latch_loads": schedule_latch,
        "background_overlap_rows": overlap,
        "trace_rows": len(rows),
    }.items():
        if as_int(sample, key) != recomputed:
            errors.append(f"{config['tag']}/{mode} trace {key}: got {sample[key]}, expected {recomputed}")
    if (mode == "none" and schedule_latch <= 0) or (mode != "none" and schedule_latch != 0):
        errors.append(f"{config['tag']}/{mode}: invalid Ceiling schedule-latch counter")


def validate_diamond_trace(errors: list[str], config: dict[str, str],
                           sample: dict[str, str], rows: list[dict[str, int | str]],
                           mode: str) -> None:
    blocks = as_int(config, "blocks")
    stages = [keyed(rows, stage) for stage in range(4)]
    if any(set(stage) != set(range(blocks)) for stage in stages):
        errors.append(f"{config['tag']}/{mode}: incomplete diamond trace")
        return
    for stage in stages:
        if any(not check_time_order(r, ("t_start", "t_wait_begin", "t_dep", "t_ready", "t_end"))
               for r in stage.values()):
            errors.append(f"{config['tag']}/{mode}: diamond timestamp order")
            return
    first = min(int(r["t_start"]) for stage in stages for r in stage.values())
    last = max(int(r["t_end"]) for stage in stages for r in stage.values())
    if mode == "grid-ordered":
        for stage in range(1, 4):
            prior_end = max(int(r["t_end"]) for r in stages[stage - 1].values())
            if any(int(r["t_dep"]) < prior_end for r in stages[stage].values()):
                errors.append(f"{config['tag']}/{mode}: grid edge returned early")
    elif mode != "none":
        for block in range(blocks):
            if int(stages[1][block]["t_dep"]) < int(stages[0][block]["t_ready"]):
                errors.append(f"{config['tag']}/{mode}: K1>K2 violation")
            parent = 1 if mode == "cta-ordered" else 0
            if int(stages[2][block]["t_dep"]) < int(stages[parent][block]["t_ready"]):
                errors.append(f"{config['tag']}/{mode}: K3 parent violation")
            if int(stages[3][block]["t_dep"]) < max(
                int(stages[1][block]["t_ready"]), int(stages[2][block]["t_ready"])
            ):
                errors.append(f"{config['tag']}/{mode}: K2+K3>K4 violation")
    k2_first = min(int(r["t_dep"]) for r in stages[1].values())
    k2_last = max(int(r["t_ready"]) for r in stages[1].values())
    k3_first = min(int(r["t_dep"]) for r in stages[2].values())
    k3_last = max(int(r["t_ready"]) for r in stages[2].values())
    overlap = max(0, min(k2_last, k3_last) - max(k2_first, k3_first)) / 1e6
    polls = sum(int(r["poll_loads"]) for stage in stages for r in stage.values())
    close(errors, f"{config['tag']}/{mode} trace ms", as_float(sample, "ms"),
          (last - first) / 1e6)
    close(errors, f"{config['tag']}/{mode} trace overlap",
          as_float(sample, "branch_overlap_ms"), overlap)
    if as_int(sample, "poll_loads") != polls or as_int(sample, "trace_rows") != len(rows):
        errors.append(f"{config['tag']}/{mode}: diamond trace counters mismatch")


def validate_c1_trace(errors: list[str], config: dict[str, str], sample: dict[str, str],
                      rows: list[dict[str, int | str]], mode: str) -> None:
    tiles = as_int(config, "tiles")
    prod, cons = keyed(rows, 0), keyed(rows, 1)
    if set(prod) != set(range(tiles)) or set(cons) != set(range(tiles)):
        errors.append(f"{config['tag']}/{mode}: incomplete C1 trace")
        return
    for row in list(prod.values()) + list(cons.values()):
        if not check_time_order(row, ("t_start", "t_wait_begin", "t_dep", "t_end")):
            errors.append(f"{config['tag']}/{mode}: C1 timestamp order")
            return
        if not (int(row["t_start"]) <= int(row["t_ready"]) <= int(row["t_end"])):
            errors.append(f"{config['tag']}/{mode}: C1 readiness timestamp order")
            return
    if mode not in {"fused-cluster", "none"}:
        p_end = max(int(r["t_end"]) for r in prod.values())
        if any(int(r["t_dep"]) < p_end for r in cons.values()):
            errors.append(f"{config['tag']}/{mode}: C1 grid wait returned early")
    first = min(int(r["t_start"]) for r in list(prod.values()) + list(cons.values()))
    last = max(int(r["t_end"]) for r in list(prod.values()) + list(cons.values()))
    close(errors, f"{config['tag']}/{mode} trace ms", as_float(sample, "ms"),
          (last - first) / 1e6)
    if as_int(sample, "trace_rows") != len(rows):
        errors.append(f"{config['tag']}/{mode}: C1 trace row count mismatch")


def validate_clc_trace(errors: list[str], config: dict[str, str], sample: dict[str, str],
                       rows: list[dict[str, int | str]], mode: str) -> None:
    tiles = as_int(config, "tiles")
    prod, cons, aggregate, tokens = (keyed(rows, i) for i in range(4))
    if set(prod) != set(range(tiles)) or set(cons) != set(range(tiles)) or set(aggregate) != {0} \
            or set(tokens) != set(range(2 * tiles)):
        errors.append(f"{config['tag']}/{mode}: incomplete CLC trace")
        return
    for kind in (prod, cons):
        for row in kind.values():
            if not check_time_order(row, ("t_start", "t_wait_begin", "t_dep", "t_end")):
                errors.append(f"{config['tag']}/{mode}: CLC task timestamp order")
                return
    if mode != "none":
        for tile in range(tiles):
            if int(cons[tile]["t_dep"]) < int(prod[tile]["t_ready"]):
                errors.append(f"{config['tag']}/{mode}: CLC consumer returned early")
                return
    coverage_errors = sum(
        int(row["poll_loads"]) + int(row["metadata_loads"]) != 1 for row in tokens.values()
    )
    if coverage_errors != as_int(sample, "token_coverage_errors") or coverage_errors != 0:
        errors.append(f"{config['tag']}/{mode}: CLC launch token coverage mismatch")
    agg = aggregate[0]
    close(errors, f"{config['tag']}/{mode} trace ms", as_float(sample, "ms"),
          (int(agg["t_end"]) - int(agg["t_start"])) / 1e6)
    polls = sum(int(row["poll_loads"]) for row in cons.values())
    if polls != as_int(sample, "wait_poll_loads"):
        errors.append(f"{config['tag']}/{mode}: CLC poll trace mismatch")
    if as_int(sample, "trace_rows") != len(rows):
        errors.append(f"{config['tag']}/{mode}: CLC trace row count mismatch")


def validate_trace(errors: list[str], experiment: str, config: dict[str, str],
                   sample: dict[str, str], rows: list[dict[str, int | str]], mode: str) -> None:
    if experiment in {"protocol", "encoding"}:
        validate_pe_trace(errors, config, sample, rows, mode)
    elif experiment == "diamond":
        validate_diamond_trace(errors, config, sample, rows, mode)
    elif experiment == "c1":
        validate_c1_trace(errors, config, sample, rows, mode)
    elif experiment == "clc":
        validate_clc_trace(errors, config, sample, rows, mode)


def validate_summary(errors: list[str], experiment: str, modes: list[str],
                     samples: list[dict[str, str]], summaries: list[dict[str, str]]) -> None:
    summary_by_mode = {s.get("mode", ""): s for s in summaries}
    sample_by_mode = {m: [s for s in samples if s.get("mode") == m] for m in modes}
    if set(summary_by_mode) != set(modes) or len(summaries) != len(modes):
        errors.append(f"{samples[0].get('tag', '?')}: summary mode set mismatch")
        return
    if experiment in {"protocol", "encoding"}:
        specs = [
            ("ms", "median_ms", "ci_ms_lo", "ci_ms_hi", 0x230000, 1e-7),
            ("wait_ns", "median_wait_ns", "ci_wait_lo", "ci_wait_hi", 0x231000, 1e-3),
            ("wake_ns", "median_wake_ns", "ci_wake_lo", "ci_wake_hi", 0x232000, 1e-3),
            ("background_gbps", "median_background_gbps", "ci_background_lo",
             "ci_background_hi", 0x233000, 1e-7),
            ("poll_loads", "median_poll_loads", "ci_poll_lo", "ci_poll_hi", 0x234000, 1e-3),
            ("metadata_loads", "median_metadata_loads", "ci_metadata_lo",
             "ci_metadata_hi", 0x235000, 1e-3),
            ("decode_ns", "median_decode_ns", "ci_decode_lo", "ci_decode_hi", 0x236000, 1e-3),
            ("ceiling_schedule_latch_loads", "median_ceiling_schedule_latch_loads",
             "ci_latch_lo", "ci_latch_hi", 0x237000, 1e-3),
        ]
        seed_index = {m: i for i, m in enumerate(modes)}
    elif experiment == "diamond":
        specs = [
            ("ms", "median_ms", "ci_ms_lo", "ci_ms_hi", 0x7400, 1e-7),
            ("branch_overlap_ms", "median_branch_overlap_ms", "ci_overlap_lo",
             "ci_overlap_hi", 0x7500, 1e-7),
            ("poll_loads", "median_poll_loads", "ci_poll_lo", "ci_poll_hi", 0x7600, 1e-3),
        ]
        seed_index = {m: i for i, m in enumerate(modes)}
    elif experiment == "c1":
        specs = [
            ("ms", "median_ms", "ci_ms_lo", "ci_ms_hi", 0xC100, 1e-7),
            ("software_transfer_gbps", "median_software_transfer_gbps", "ci_transfer_lo",
             "ci_transfer_hi", 0xC200, 1e-7),
        ]
        enums = {"fused-cluster": 0, "separate-persist": 1, "separate-default": 2,
                 "separate-cv": 3, "none": 4}
        seed_index = enums
    else:
        specs = [
            ("ms", "median_ms", "ci_ms_lo", "ci_ms_hi", 0xC1C000, 1e-7),
            ("clc_success_rate", "median_clc_success_rate", "ci_success_lo",
             "ci_success_hi", 0xC1C100, 1e-7),
            ("wait_poll_loads", "median_wait_poll_loads", "ci_poll_lo",
             "ci_poll_hi", 0xC1C200, 1e-3),
            ("locality_hits", "median_locality_hits", "ci_locality_lo",
             "ci_locality_hi", 0xC1C300, 1e-3),
            ("clc_cycles_per_attempt", "median_clc_cycles_per_attempt", "ci_cycles_lo",
             "ci_cycles_hi", 0xC1C400, 1e-7),
            ("clc_attempts_per_ms", "median_clc_attempts_per_ms", "ci_attempt_rate_lo",
             "ci_attempt_rate_hi", 0xC1C500, 1e-7),
        ]
        seed_index = {m: i for i, m in enumerate(modes)}
    for mode in modes:
        summary = summary_by_mode[mode]
        vals = sample_by_mode[mode]
        if as_int(summary, "repeats") != len(vals) or as_int(summary, "valid") != 1:
            errors.append(f"{summary.get('tag', '?')}/{mode}: invalid summary repeat/status")
        for raw, med_key, lo_key, hi_key, base, tol in specs:
            data = [as_float(s, raw) for s in vals]
            med = median(data)
            lo, hi = bootstrap_ci(data, base + seed_index[mode])
            close(errors, f"{summary.get('tag', '?')}/{mode} {med_key}",
                  as_float(summary, med_key), med, tol)
            close(errors, f"{summary.get('tag', '?')}/{mode} {lo_key}",
                  as_float(summary, lo_key), lo, tol)
            close(errors, f"{summary.get('tag', '?')}/{mode} {hi_key}",
                  as_float(summary, hi_key), hi, tol)


def validate_coverage(errors: list[str], configs: list[dict[str, str]]) -> None:
    by_exp: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in configs:
        by_exp[c["experiment"]].append(c)
    if set(by_exp) != {"protocol", "encoding", "diamond", "c1", "clc"}:
        errors.append(f"formal experiment set mismatch: {sorted(by_exp)}")
        return
    sms = {as_int(c, "sm") for c in configs}
    if len(sms) != 1:
        errors.append(f"formal config SM count mismatch: {sms}")
        return
    sm = next(iter(sms))
    if {as_int(c, "P") for c in by_exp["protocol"]} != {sm, 2 * sm, 8 * sm}:
        errors.append("§7.1 formal grids must be 1x/2x/8x SM")
    enc = {(c["structure"], as_int(c, "degree")) for c in by_exp["encoding"]}
    expected_enc = {(s, d) for s in ("interval", "strided")
                    for d in (1, 2, 4, 8, 16, 32, 64)}
    if enc != expected_enc or len(by_exp["encoding"]) != len(expected_enc):
        errors.append("§7.3 formal coverage must be interval/strided x degree 1..64")
    ratios = {ratio_value(c) for c in by_exp["diamond"]}
    if ratios != set(range(1, 11)) or len(by_exp["diamond"]) != 10:
        errors.append("§7.4 formal coverage must contain every ratio 1..10")
    sizes = {as_int(c, "bytes_per_tile") for c in by_exp["c1"]}
    if sizes != {1024 * (1 << i) for i in range(7)} or len(by_exp["c1"]) != 7:
        errors.append("§7.5 formal coverage must contain 1/2/4/8/16/32/64 KiB")
    if len(by_exp["clc"]) != 1:
        errors.append("§7.6 formal coverage requires exactly one CLC policy matrix")
    for c in configs:
        if as_int(c, "repeats") < 31 or as_int(c, "warmup") < 3:
            errors.append(f"{c['tag']}: formal repeat/warmup contract not met")


def validate_manifest(results: Path, manifest: Path, allow_incomplete: bool) -> dict[str, Any]:
    errors: list[str] = []
    config_outputs: list[dict[str, Any]] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f, delimiter="\t"))
    expected_header = ["tag", "experiment", "log", "trace", "modes", "repeats", "warmup",
                       "P", "C", "structure", "degree", "ratio", "bytes_per_tile", "tiles"]
    if not manifest_rows or list(manifest_rows[0]) != expected_header:
        raise ValueError("manifest missing/invalid header or empty")
    if len({r["tag"] for r in manifest_rows}) != len(manifest_rows):
        errors.append("manifest contains duplicate tags")

    all_configs: list[dict[str, str]] = []
    for mr in manifest_rows:
        tag, experiment = mr["tag"], mr["experiment"]
        modes = mr["modes"].split(",")
        log_path = Path(mr["log"])
        trace_path = Path(mr["trace"])
        if not log_path.is_absolute():
            log_path = (Path.cwd() / log_path).resolve()
        if not trace_path.is_absolute():
            trace_path = (Path.cwd() / trace_path).resolve()
        if not log_path.is_file():
            errors.append(f"{tag}: missing log {log_path}")
            continue
        records: dict[str, list[dict[str, str]]] = defaultdict(list)
        event_order: list[tuple[str, dict[str, str]]] = []
        for lineno, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
            parsed = parse_record(line, lineno)
            if parsed is None:
                continue
            kind, record = parsed
            records[kind].append(record)
            if kind in {"VALIDATION_TIER23", "WARMUP_TIER23", "SAMPLE_TIER23"}:
                event_order.append((kind, record))
        if len(records["CONFIG_TIER23"]) != 1:
            errors.append(f"{tag}: expected one CONFIG_TIER23")
            continue
        config = records["CONFIG_TIER23"][0]
        require(config, "semantics", "experiment", "tag", "sm", "cc", "warmup", "repeats",
                "timer", "bootstrap", "trace")
        all_configs.append(config)
        if as_int(config, "semantics") != SEMANTICS or config["experiment"] != experiment \
                or config["tag"] != tag:
            errors.append(f"{tag}: CONFIG identity/semantics mismatch")
        if config["timer"] != "globaltimer" or as_int(config, "bootstrap") != BOOTSTRAPS:
            errors.append(f"{tag}: timer/bootstrap contract mismatch")
        if as_int(config, "repeats") != int(mr["repeats"]) \
                or as_int(config, "warmup") != int(mr["warmup"]):
            errors.append(f"{tag}: manifest repeat/warmup mismatch")
        if Path(config["trace"]).resolve() != trace_path:
            errors.append(f"{tag}: CONFIG trace path does not bind manifest trace")
        if experiment in {"protocol", "encoding"}:
            for key in ("P", "C", "structure", "degree", "background_blocks",
                        "background_iterations", "threads", "ceiling_schedule",
                        "ceiling_proof",
                        "ceiling_sentinel_child", "ceiling_sentinel_parent",
                        "ceiling_proof_timing"):
                require(config, key)
            if (as_int(config, "P"), as_int(config, "C"), config["structure"],
                    as_int(config, "degree")) != (
                    int(mr["P"]), int(mr["C"]), mr["structure"], int(mr["degree"])):
                errors.append(f"{tag}: protocol/encoding manifest coordinate mismatch")
            if config.get("trigger_floor") != "ready" or config.get("trigger_impl") != "entry" \
                    or config.get("trigger_ceiling") != "entry" \
                    or config.get("publication_ceiling") != "none":
                errors.append(f"{tag}: trigger/publication declaration mismatch")
            if config.get("poll_counter_semantics") != \
                    "logical_acquire_loads_not_l2_requests":
                errors.append(f"{tag}: software poll counter is not explicitly scoped")
            sentinel_parents = parents(config["structure"], as_int(config, "degree"),
                                       as_int(config, "P"), as_int(config, "C"), 0)
            expected_sentinel = sentinel_parents[-1] if sentinel_parents else -1
            if config["ceiling_schedule"] != "deterministic_device_sentinel_RAW" \
                    or config["ceiling_proof"] != \
                    "adversarial_device_sentinel_raw_before_store" \
                    or as_int(config, "ceiling_sentinel_child") != 0 \
                    or as_int(config, "ceiling_sentinel_parent") != expected_sentinel \
                    or config["ceiling_proof_timing"] != "included":
                errors.append(f"{tag}: invalid deterministic Ceiling sentinel declaration")
        elif experiment == "diamond":
            require(config, "blocks", "ratio")
            if ratio_value(config) != int(mr["ratio"]):
                errors.append(f"{tag}: diamond ratio mismatch")
        elif experiment == "c1":
            require(config, "tiles", "bytes_per_tile", "words_per_tile", "bracket",
                    "cv_semantics", "software_bytes_not_dram_counters")
            if as_int(config, "bytes_per_tile") != int(mr["bytes_per_tile"]):
                errors.append(f"{tag}: C1 byte coordinate mismatch")
            if config["cv_semantics"] != "forced-refetch-pessimal-control":
                errors.append(f"{tag}: C1 CV path mislabeled")
            if as_int(config, "software_bytes_not_dram_counters") != 1:
                errors.append(f"{tag}: C1 software bytes mislabeled as profiler traffic")
        elif experiment == "clc":
            require(config, "tiles", "launch_clusters", "policies", "token_conservation",
                    "poll_counter_semantics")
            if as_int(config, "launch_clusters") != 2 * as_int(config, "tiles") \
                    or config["token_conservation"] != "executed_plus_canceled_equals_one":
                errors.append(f"{tag}: CLC launch-token conservation declaration mismatch")
            if config["poll_counter_semantics"] != "logical_acquire_loads_not_l2_requests":
                errors.append(f"{tag}: CLC poll counter is not explicitly scoped")

        validations = records["VALIDATION_TIER23"]
        warmups = records["WARMUP_TIER23"]
        samples = records["SAMPLE_TIER23"]
        summaries = records["SUMMARY_TIER23"]
        traces_decl = records["TRACE_TIER23"]

        # The 64-KiB C1 coordinate contains 2.4 million independently poisoned words per
        # epoch.  Recomputing every epoch serially is needlessly slow and previously left a
        # CPU-only strict pass running for many minutes after the GPU campaign ended.  Each
        # epoch is independent, so parallel workers retain the exact same all-word FNV proof.
        c1_digest_cache: dict[int, int] = {}
        if experiment == "c1":
            digest_epochs = sorted({as_int(r, "epoch") for r in validations + samples})
            try:
                requested_workers = int(os.environ.get("T23_VALIDATOR_WORKERS", "8"), 10)
            except ValueError as exc:
                raise ValueError("T23_VALIDATOR_WORKERS must be an integer") from exc
            worker_count = min(max(1, requested_workers), os.cpu_count() or 1,
                               len(digest_epochs))
            jobs = [(config, epoch) for epoch in digest_epochs]
            if worker_count == 1:
                c1_digest_cache.update(c1_digest_job(job) for job in jobs)
            else:
                with ProcessPoolExecutor(max_workers=worker_count) as pool:
                    c1_digest_cache.update(pool.map(c1_digest_job, jobs))

        def recompute(rec: dict[str, str], validation: bool) -> int:
            epoch = as_int(rec, "epoch")
            if experiment == "c1":
                return c1_digest_cache[epoch]
            return expected_digest(experiment, config, epoch, validation)

        repeats, warmup_count = as_int(config, "repeats"), as_int(config, "warmup")
        if len(validations) != len(modes) or len(warmups) != warmup_count * len(modes) \
                or len(samples) != repeats * len(modes) or len(summaries) != len(modes) \
                or len(traces_decl) != 1:
            errors.append(f"{tag}: record count mismatch")
            continue

        expected_events: list[tuple[str, int, str]] = []
        for mode in modes:
            expected_events.append(("VALIDATION_TIER23", -1, mode))
        for w in range(warmup_count):
            order = modes if w % 2 == 0 else list(reversed(modes))
            expected_events.extend(("WARMUP_TIER23", w, mode) for mode in order)
        for rep in range(repeats):
            order = modes if rep % 2 == 0 else list(reversed(modes))
            expected_events.extend(("SAMPLE_TIER23", rep, mode) for mode in order)
        if len(event_order) == len(expected_events):
            for epoch, ((kind, rec), (ekind, index, mode)) in enumerate(
                    zip(event_order, expected_events), 1):
                if kind != ekind or rec.get("mode") != mode or as_int(rec, "epoch") != epoch:
                    errors.append(f"{tag}: epoch/order mismatch at logical epoch {epoch}")
                    break
                if kind == "WARMUP_TIER23" and as_int(rec, "warmup") != index:
                    errors.append(f"{tag}: warmup index/order mismatch")
                if kind == "SAMPLE_TIER23" and as_int(rec, "rep") != index:
                    errors.append(f"{tag}: repeat index/order mismatch")
        else:
            errors.append(f"{tag}: event stream length mismatch")

        for rec in validations:
            require(rec, "status", "correct", "ceiling_wrong", "stale", "observed_digest",
                    "expected_digest", "trace_ok")
            mode = rec["mode"]
            ceiling = mode == "none"
            recomputed = recompute(rec, not ceiling)
            if as_int(rec, "expected_digest") != recomputed:
                errors.append(f"{tag}/{mode}: validation expected digest mismatch")
            if rec["status"] != "PASS" or as_int(rec, "trace_ok") != 1:
                errors.append(f"{tag}/{mode}: validation did not PASS")
            if ceiling:
                if as_int(rec, "ceiling_wrong") != 1 or as_int(rec, "stale") <= 0 \
                        or as_int(rec, "observed_digest") == as_int(rec, "expected_digest"):
                    errors.append(f"{tag}/{mode}: Ceiling is not demonstrably wrong")
            elif as_int(rec, "correct") != 1 \
                    or as_int(rec, "observed_digest") != as_int(rec, "expected_digest"):
                errors.append(f"{tag}/{mode}: correct validation digest mismatch")

        for rec in samples:
            mode, epoch = rec["mode"], as_int(rec, "epoch")
            ceiling = mode == "none"
            recomputed = recompute(rec, False)
            if as_int(rec, "expected_digest") != recomputed or as_int(rec, "trace_ok") != 1:
                errors.append(f"{tag}/{mode}: sample digest/trace declaration mismatch")
            if ceiling:
                if as_int(rec, "ceiling_wrong") != 1 or as_int(rec, "stale") <= 0 \
                        or as_int(rec, "observed_digest") == as_int(rec, "expected_digest"):
                    errors.append(f"{tag}/{mode}: timed Ceiling is not wrong")
            elif as_int(rec, "correct") != 1 \
                    or as_int(rec, "observed_digest") != as_int(rec, "expected_digest"):
                errors.append(f"{tag}/{mode}: timed correct digest mismatch")
            if "poll_bytes" in rec and as_int(rec, "poll_bytes") != as_int(rec, "poll_loads") * 8:
                errors.append(f"{tag}/{mode}: poll byte counter mismatch")
            if "metadata_bytes" in rec \
                    and as_int(rec, "metadata_bytes") != as_int(rec, "metadata_loads") * 4:
                errors.append(f"{tag}/{mode}: metadata byte counter mismatch")
            if experiment == "clc":
                attempts, successes = as_int(rec, "clc_attempts"), as_int(rec, "clc_successes")
                close(errors, f"{tag}/{mode} success rate", as_float(rec, "clc_success_rate"),
                      successes / attempts if attempts else 0.0)
                close(errors, f"{tag}/{mode} cycles/attempt",
                      as_float(rec, "clc_cycles_per_attempt"),
                      as_int(rec, "clc_cycles") / attempts if attempts else 0.0)
                close(errors, f"{tag}/{mode} attempts/ms",
                      as_float(rec, "clc_attempts_per_ms"),
                      attempts / as_float(rec, "ms") if as_float(rec, "ms") else 0.0)
                if as_int(rec, "tokens") != 2 * as_int(config, "tiles") \
                        or as_int(rec, "token_coverage_errors") != 0:
                    errors.append(f"{tag}/{mode}: CLC token conservation failure")

        validate_summary(errors, experiment, modes, samples, summaries)
        if not trace_path.is_file():
            errors.append(f"{tag}: missing trace {trace_path}")
            continue
        trace_rows = load_trace(trace_path)
        if {str(r["mode"]) for r in trace_rows} != set(modes):
            errors.append(f"{tag}: trace contains missing/extra modes")
        if len(trace_rows) != as_int(traces_decl[0], "rows_per_mode") * len(modes):
            errors.append(f"{tag}: total trace row count mismatch")
        final_samples = {s["mode"]: s for s in samples if as_int(s, "rep") == repeats - 1}
        declaration = traces_decl[0]
        if Path(declaration.get("path", "")).resolve() != trace_path \
                or as_int(declaration, "modes") != len(modes):
            errors.append(f"{tag}: TRACE declaration mismatch")
        for mode in modes:
            sample = final_samples.get(mode)
            if sample is None:
                errors.append(f"{tag}/{mode}: missing final sample")
                continue
            epoch = as_int(sample, "epoch")
            selected = [r for r in trace_rows if r["mode"] == mode]
            if any(r["tag"] != tag or r["experiment"] != experiment or r["epoch"] != epoch
                   for r in selected):
                errors.append(f"{tag}/{mode}: trace identity/epoch mismatch")
                continue
            if len(selected) != as_int(declaration, "rows_per_mode"):
                errors.append(f"{tag}/{mode}: trace row declaration mismatch")
                continue
            sm_count = as_int(config, "sm")
            real_rows = selected if experiment != "clc" else [
                r for r in selected if int(r["kernel_id"]) in (0, 1)
            ]
            if any(int(r["sm_id"]) >= sm_count for r in real_rows):
                errors.append(f"{tag}/{mode}: real task trace has out-of-range sm_id")
                continue
            validate_trace(errors, experiment, config, sample, selected, mode)

        config_outputs.append({
            "tag": tag,
            "experiment": experiment,
            "modes": modes,
            "repeats": repeats,
            "warmup": warmup_count,
            "samples": len(samples),
            "trace_rows": len(trace_rows),
            "coordinates": {
                key: config[key] for key in
                ("P", "C", "structure", "degree", "blocks", "ratio",
                 "bytes_per_tile", "tiles") if key in config
            },
            "summaries": [
                {key: val for key, val in summary.items() if key != "_line"}
                for summary in summaries
            ],
        })

    if not allow_incomplete:
        validate_coverage(errors, all_configs)
        ncu_status = results / "ncu_status.txt"
        nsys_status = results / "nsys_status.txt"
        if not ncu_status.is_file() or "blocking=0" not in ncu_status.read_text(encoding="utf-8"):
            errors.append("missing/non-fail-soft NCU sidecar status")
        if not nsys_status.is_file() or "status=captured" not in nsys_status.read_text(encoding="utf-8"):
            errors.append("formal campaign requires captured Nsight Systems sidecar")

    return {
        "schema": 1,
        "semantics": SEMANTICS,
        "status": "PASS" if not errors else "FAIL",
        "formal": not allow_incomplete,
        "manifest": str(manifest),
        "config_count": len(config_outputs),
        "sample_count": sum(c["samples"] for c in config_outputs),
        "trace_row_count": sum(c["trace_rows"] for c in config_outputs),
        "configs": config_outputs,
        "errors": errors,
    }


def write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config in result.get("configs", []):
        for summary in config.get("summaries", []):
            row: dict[str, Any] = {
                "tag": config["tag"],
                "experiment": config["experiment"],
                **config.get("coordinates", {}),
                **summary,
            }
            rows.append(row)
    preferred = ["tag", "experiment", "mode", "repeats", "P", "C", "structure",
                 "degree", "blocks", "ratio", "bytes_per_tile", "tiles", "median_ms",
                 "ci_ms_lo", "ci_ms_hi", "valid"]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extras
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    results = args.results.resolve()
    manifest = (args.manifest or results / "tier23_manifest.tsv").resolve()
    json_path = args.json_path.resolve()
    csv_path = args.csv_path.resolve()
    json_path.unlink(missing_ok=True)
    try:
        result = validate_manifest(results, manifest, args.allow_incomplete)
    except Exception as exc:  # fail closed, and atomically replace any stale PASS
        result = {
            "schema": 1,
            "semantics": SEMANTICS,
            "status": "FAIL",
            "formal": not args.allow_incomplete,
            "manifest": str(manifest),
            "config_count": 0,
            "sample_count": 0,
            "trace_row_count": 0,
            "configs": [],
            "errors": [f"validator exception: {type(exc).__name__}: {exc}"],
        }
    atomic_json(json_path, result)
    write_summary_csv(csv_path, result)
    print(
        f"TIER23_VALIDATION status={result['status']} configs={result['config_count']} "
        f"samples={result['sample_count']} trace_rows={result['trace_row_count']} "
        f"errors={len(result['errors'])} json={json_path}"
    )
    for error in result["errors"][:50]:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
