#!/usr/bin/env python3
"""Strict validator for dsa_native semantics=1 artifacts.

The validator does not trust self-reported PASS fields.  It rebuilds the invocation/epoch
schedule, bootstrap statistics, exact shape/memory accounting, validation checksums, and the
final retained %globaltimer traces.  Every failure replaces (rather than leaves behind) the
requested JSON result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
BOOTSTRAP = 2000
MODES = ("floor", "wave_floor", "impl", "ceiling")


def latin_order(rep: int) -> tuple[str, ...]:
    """Cyclic four-way Latin rotation used by the native timed loop."""
    return tuple(MODES[(position + rep) % len(MODES)] for position in range(len(MODES)))
PAIR_KEY_REGISTER_TILE = 8
PAIR_CONTRACT_STRINGS = {
    "pair_accumulator": "uint32_mod2p32",
    "pair_low16_equivalence": "mod2p32_then_low16_equals_uint64_low16",
    "pair_query_cache": "cta_shared_once",
    "pair_key_cache": "cta_shared_once_register_tile",
    "pair_iteration": "explicit_inline_ptx_add_u32_per_pair",
    "pair_closed_form": "0",
}
HISTORY_CONTRACT_STRINGS = {
    "history_loads_per_rank": "1",
    "history_load_count": "device_dynamic_exact",
    "history_load_work_parity": "floor_wave_floor_impl_ceiling_equal",
}
FLOOR_TRACE_CONTRACT_STRINGS = {
    "floor_overlap_metric": "consumer_start_before_upstream_kernel_end",
    "floor_dependency_metric": (
        "consumer_dep_after_upstream_programmatic_trigger"
    ),
}
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def canonical_gpu_uuid_from_bytes(value: bytes) -> str:
    if len(value) != 16:
        raise ValueError("CUDA UUID must contain exactly 16 bytes")
    encoded = value.hex()
    return (
        f"GPU-{encoded[:8]}-{encoded[8:12]}-{encoded[12:16]}-"
        f"{encoded[16:20]}-{encoded[20:]}"
    )


def fields(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.strip().split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


def as_int(row: dict[str, str], key: str) -> int:
    if key not in row:
        raise ValueError(f"missing integer field {key}")
    return int(row[key], 10)


def as_float(row: dict[str, str], key: str) -> float:
    if key not in row:
        raise ValueError(f"missing float field {key}")
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite field {key}")
    return value


def device_identity_errors(
    config: dict[str, str], device: dict[str, str], expected_gpu_uuid: str,
) -> list[str]:
    errors: list[str] = []
    if not GPU_UUID_RE.fullmatch(expected_gpu_uuid):
        errors.append("runner expected GPU UUID is not canonical")
    runtime_uuid = config.get("runtime_uuid", "")
    if not GPU_UUID_RE.fullmatch(runtime_uuid):
        errors.append("CONFIG_DSA runtime UUID is not canonical")
    if runtime_uuid != expected_gpu_uuid:
        errors.append("CONFIG_DSA runtime UUID does not match runner lease UUID")
    if device.get("runtime_uuid") != runtime_uuid:
        errors.append("DEVICE_DSA/CONFIG_DSA runtime UUID mismatch")
    try:
        if as_int(config, "runtime_ordinal") != 0:
            errors.append("CONFIG_DSA runtime ordinal is not zero")
        if as_int(config, "runtime_ordinal_zero") != 1:
            errors.append("CONFIG_DSA lacks cudaGetDevice()==0 proof")
        if as_int(device, "runtime_ordinal") != 0:
            errors.append("DEVICE_DSA runtime ordinal is not zero")
        if as_int(device, "runtime_ordinal_zero") != 1:
            errors.append("DEVICE_DSA lacks cudaGetDevice()==0 proof")
        pairs = (
            ("runtime_name_hex", "name_hex"),
            ("runtime_cc_major", "cc_major"),
            ("runtime_cc_minor", "cc_minor"),
            ("runtime_sms", "sms"),
        )
        for config_key, device_key in pairs:
            if config.get(config_key) != device.get(device_key):
                errors.append(
                    f"DEVICE_DSA/CONFIG_DSA {config_key}/{device_key} mismatch"
                )
        if as_int(config, "runtime_sms") != as_int(config, "sms"):
            errors.append("runtime SM count does not match benchmark SM count")
        if as_int(config, "runtime_cc_major") < 9:
            errors.append("runtime compute capability is below PDL minimum")
        if as_int(config, "runtime_cc_minor") < 0:
            errors.append("runtime compute capability minor is invalid")
        encoded_name = config.get("runtime_name_hex", "")
        if not encoded_name or len(encoded_name) % 2 or not re.fullmatch(
            r"[0-9a-f]+", encoded_name
        ):
            errors.append("runtime device name hex is malformed")
        else:
            bytes.fromhex(encoded_name).decode("utf-8")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        errors.append(f"runtime device identity malformed ({exc})")
    return errors


def mix32(x: int) -> int:
    x &= MASK32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK32
    x ^= x >> 16
    return x & MASK32


def history_value(key: int) -> int:
    return mix32(key ^ 0x6A09E667)


def history_sum(key_tiles: int, topk: int) -> int:
    value = 0
    for rank in range(topk):
        key = key_tiles - 1 - rank
        value = (value + history_value(key) + key * 65537) & MASK64
    return value


def attention_value(epoch: int, query: int, lane: int, aggregate: int) -> int:
    x = (
        (aggregate & MASK32)
        ^ ((aggregate >> 32) & MASK32)
        ^ ((epoch * 0x9E3779B9) & MASK32)
        ^ ((query * 0x85EBCA6B) & MASK32)
        ^ ((lane * 0xC2B2AE35) & MASK32)
    )
    return mix32(x)


def score_checksum(
    epoch: int, query_blocks: int, key_tiles: int, pair_query: int, pair_key: int
) -> int:
    qsum = 3 * pair_query * (pair_query - 1) // 2 + pair_query
    ksum = 5 * pair_key * (pair_key - 1) // 2 + pair_key
    work = pair_query * pair_key
    delta = (work * 7) & 65535
    period = 1 if delta == 0 else 65536 // math.gcd(delta, 65536)
    high_sum = 65536 * key_tiles * (key_tiles + 1) // 2
    total = 0
    for query in range(query_blocks):
        first = (
            pair_key * qsum
            + pair_query * ksum
            + work * (epoch * 17 + query * 131)
        ) & 65535
        cycle = sum((first + delta * key) & 65535 for key in range(period))
        full, remainder = divmod(key_tiles, period)
        low_sum = full * cycle + sum(
            (first + delta * key) & 65535 for key in range(remainder)
        )
        total = (total + high_sum + low_sum) & MASK64
    return total


def index_checksum(query_blocks: int, key_tiles: int, topk: int) -> int:
    per_row = topk * (2 * key_tiles - topk - 1) // 2
    return (query_blocks * per_row) & MASK64


def output_checksum(epoch: int, query_blocks: int, aggregate: int) -> int:
    total = 0
    for query in range(query_blocks):
        for lane in range(64):
            total = (total + attention_value(epoch, query, lane, aggregate)) & MASK64
    return total


def pair_low16_reference_u64(
    epoch: int, query: int, key_tile: int, pair_query: int, pair_key: int
) -> int:
    """Emulate the original uint64 accumulator (the base itself is uint32)."""
    base = (epoch * 17 + query * 131 + key_tile * 7) & MASK32
    accum = 0
    for q in range(pair_query):
        query_value = (q * 3 + 1) & MASK32
        for key in range(pair_key):
            key_value = (key * 5 + 1) & MASK32
            term = query_value + key_value + base
            accum = (accum + term) & MASK64
    return accum & 0xFFFF


def pair_low16_mod32(
    epoch: int, query: int, key_tile: int, pair_query: int, pair_key: int
) -> int:
    """Emulate the optimized per-pair uint32 modular accumulator."""
    base = (epoch * 17 + query * 131 + key_tile * 7) & MASK32
    accum = 0
    for q in range(pair_query):
        query_value = (q * 3 + 1) & MASK32
        for key in range(pair_key):
            key_value = (key * 5 + 1) & MASK32
            term = (query_value + key_value + base) & MASK32
            accum = (accum + term) & MASK32
    return accum & 0xFFFF


def pair_low16_equivalence_cases(
    query_blocks: int, key_tiles: int, epoch_last: int,
    pair_query: int, pair_key: int,
) -> list[dict[str, int | bool]]:
    """Representative boundary regression for the modulo-projection proof."""
    cases = {
        (0, 0, 0),
        (1, 0, max(0, key_tiles - 1)),
        (max(1, epoch_last), max(0, query_blocks - 1), max(0, key_tiles - 1)),
        (MASK32, max(0, query_blocks - 1), max(0, key_tiles - 1)),
    }
    out: list[dict[str, int | bool]] = []
    for epoch, query, key_tile in sorted(cases):
        old = pair_low16_reference_u64(
            epoch, query, key_tile, pair_query, pair_key
        )
        new = pair_low16_mod32(epoch, query, key_tile, pair_query, pair_key)
        out.append({
            "epoch": epoch,
            "query": query,
            "key_tile": key_tile,
            "reference_low16": old,
            "mod32_low16": new,
            "equivalent": old == new,
        })
    return out


def pair_contract_errors(
    row: dict[str, str], pair_query: int, pair_key: int, label: str
) -> list[str]:
    errors: list[str] = []
    for key, expected in PAIR_CONTRACT_STRINGS.items():
        if row.get(key) != expected:
            errors.append(f"{label} {key} mismatch")
    integer_contract = {
        "pair_key_register_tile": PAIR_KEY_REGISTER_TILE,
        "pair_lut_global_loads_per_cta": pair_query + pair_key,
        "pair_adds_per_score": pair_query * pair_key,
    }
    for key, expected in integer_contract.items():
        try:
            if as_int(row, key) != expected:
                errors.append(f"{label} {key} mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} malformed {key}")
    return errors


def history_load_contract_errors(
    row: dict[str, str], expected_loads: int, label: str,
    *, require_actual: bool,
) -> list[str]:
    errors: list[str] = []
    if require_actual:
        try:
            if as_int(row, "history_loads") != expected_loads:
                errors.append(f"{label} history_loads mismatch")
            if as_int(row, "expected_history_loads") != expected_loads:
                errors.append(f"{label} expected_history_loads mismatch")
            if as_int(row, "history_load_complete") != 1:
                errors.append(f"{label} history_load_complete != 1")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label} malformed history-load proof: {exc}")
    else:
        try:
            if as_int(row, "history_loads_expected_per_invocation") != expected_loads:
                errors.append(f"{label} history_loads_expected_per_invocation mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label} malformed expected history loads: {exc}")
        for key, expected in HISTORY_CONTRACT_STRINGS.items():
            if row.get(key) != expected:
                errors.append(f"{label} {key} mismatch")
    return errors


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    if n & 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def splitmix(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def bootstrap_median(values: list[float], seed: int) -> tuple[float, float, float]:
    draws: list[float] = []
    state = seed & MASK64
    n = len(values)
    for _ in range(BOOTSTRAP):
        sample: list[float] = []
        for _ in range(n):
            state, rnd = splitmix(state)
            sample.append(values[rnd % n])
        draws.append(median(sample))
    draws.sort()
    return median(values), draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP)]


def bootstrap_pair_delta(
    values: dict[str, list[float]], base: str, target: str, seed: int
) -> tuple[float, float, float]:
    floor = median(values[base])
    point = median(values[target])
    headline = 100.0 * (floor - point) / floor if floor else 0.0
    draws: list[float] = []
    state = seed & MASK64
    n = len(values[base])
    for _ in range(BOOTSTRAP):
        picked: dict[str, list[float]] = {mode: [] for mode in MODES}
        for _ in range(n):
            state, rnd = splitmix(state)
            index = rnd % n
            for mode in MODES:
                picked[mode].append(values[mode][index])
        bf = median(picked[base])
        bx = median(picked[target])
        draws.append(100.0 * (bf - bx) / bf if bf else 0.0)
    draws.sort()
    return headline, draws[int(0.025 * BOOTSTRAP)], draws[int(0.975 * BOOTSTRAP)]


def close(errors: list[str], name: str, observed: float, expected: float, tol: float = 1e-5) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tol):
        errors.append(f"{name}: observed={observed} expected={expected}")


def bind_identity(
    errors: list[str], records: dict[str, list[dict[str, str]]], tag: str, seq: int
) -> None:
    """Bind every textual ledger row to one schema/tag/sequence identity."""
    if not tag or any(char.isspace() for char in tag):
        errors.append("CONFIG_DSA tag is empty or contains whitespace")
    for prefix in (
        "DEVICE_DSA", "CONFIG_DSA", "RESOURCE_DSA", "VALIDATION_DSA", "CEILING_PROOF_DSA",
        "ADMISSION_DSA", "WARMUP_DSA", "SAMPLE_DSA", "PROGRESS_DSA",
        "SUMMARY_DSA", "TRACE_DSA",
    ):
        for index, row in enumerate(records[prefix]):
            try:
                if as_int(row, "semantics") != 1:
                    errors.append(f"{prefix}[{index}] semantics mismatch")
                if row.get("tag") != tag:
                    errors.append(f"{prefix}[{index}] tag mismatch")
                if as_int(row, "seq") != seq:
                    errors.append(f"{prefix}[{index}] seq mismatch")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{prefix}[{index}] identity malformed ({exc})")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def analyze_trace(
    rows: list[dict[str, str]], config: dict[str, int], mode: str, epoch: int, errors: list[str]
) -> dict[str, int | float]:
    producer_ctas = config["producer_ctas"]
    query_blocks = config["query_blocks"]
    physical_degree = config["physical_degree"]
    query_wave_size = config["query_wave_size"]
    expected_count = producer_ctas + 2 * query_blocks
    if len(rows) != expected_count:
        errors.append(f"trace {mode}: rows={len(rows)} expected={expected_count}")
        return {}
    by_stage: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    first = (1 << 64) - 1
    last = 0
    trigger_failures = 0
    for raw in rows:
        try:
            stage = raw["stage"]
            block = int(raw["block"])
            values = {
                key: int(raw[key])
                for key in ("sm", "t_start", "t_dep", "t_ready", "t_trigger", "t_end")
            }
            if int(raw["epoch"]) != epoch:
                raise ValueError("epoch mismatch")
            if stage not in ("indexer", "topk", "attention"):
                raise ValueError("bad stage")
            if block in by_stage[stage]:
                raise ValueError("duplicate block")
            limit = producer_ctas if stage == "indexer" else query_blocks
            if not 0 <= block < limit:
                raise ValueError("block out of range")
            s, d, r, t, e = (
                values["t_start"], values["t_dep"], values["t_ready"],
                values["t_trigger"], values["t_end"],
            )
            if not (
                0 < s <= d <= r <= e
                and 0 <= values["sm"] < config["sms"]
            ):
                raise ValueError("timestamp order")
            if mode in ("floor", "wave_floor"):
                if not r <= t <= e:
                    raise ValueError("floor trigger order")
                trigger_failures += not (r <= t < e)
            else:
                if not s <= t <= d:
                    raise ValueError("entry trigger order")
                trigger_failures += not (s <= t <= d)
            by_stage[stage][block] = values
            first = min(first, s)
            last = max(last, e)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"trace {mode}: malformed row ({exc})")
    expected_stage_counts = {
        "indexer": producer_ctas,
        "topk": query_blocks,
        "attention": query_blocks,
    }
    for stage, count in expected_stage_counts.items():
        if len(by_stage[stage]) != count:
            errors.append(f"trace {mode} {stage}: {len(by_stage[stage])}/{count} rows")
    if any(len(by_stage[s]) != n for s, n in expected_stage_counts.items()):
        return {}
    topk_early = attention_early = topk_waited = attention_waited = safety = 0
    progress_waves_verified = 0
    consumer_entry_order_failures = 0
    producer_forward_progress_failures = 0
    trace_wave_size = query_blocks if mode == "floor" else query_wave_size
    for wave_begin in range(0, query_blocks, trace_wave_size):
        wave_end = min(query_blocks, wave_begin + trace_wave_size)
        latest_consumer_start = max(
            by_stage[stage][query]["t_start"]
            for query in range(wave_begin, wave_end)
            for stage in ("topk", "attention")
        )
        first_consumer_end = min(
            by_stage[stage][query]["t_end"]
            for query in range(wave_begin, wave_end)
            for stage in ("topk", "attention")
        )
        first_producer_start = min(
            by_stage["indexer"][query * physical_degree + parent]["t_start"]
            for query in range(wave_begin, wave_end)
            for parent in range(physical_degree)
        )
        wave_ok = True
        if mode in ("impl", "ceiling") and latest_consumer_start > first_producer_start:
            consumer_entry_order_failures += 1
            wave_ok = False
        if mode == "impl" and first_producer_start >= first_consumer_end:
            producer_forward_progress_failures += 1
            wave_ok = False
        progress_waves_verified += int(wave_ok)
    dependency_wave_size = query_blocks if mode == "floor" else query_wave_size
    dependency_bounds: list[dict[str, int]] = []
    if mode in ("floor", "wave_floor"):
        for wave_begin in range(0, query_blocks, dependency_wave_size):
            wave_end = min(query_blocks, wave_begin + dependency_wave_size)
            dependency_bounds.append({
                "index_trigger": max(
                    by_stage["indexer"][wave_query * physical_degree + parent][
                        "t_trigger"
                    ]
                    for wave_query in range(wave_begin, wave_end)
                    for parent in range(physical_degree)
                ),
                "index_end": max(
                    by_stage["indexer"][wave_query * physical_degree + parent]["t_end"]
                    for wave_query in range(wave_begin, wave_end)
                    for parent in range(physical_degree)
                ),
                "topk_trigger": max(
                    by_stage["topk"][wave_query]["t_trigger"]
                    for wave_query in range(wave_begin, wave_end)
                ),
                "topk_end": max(
                    by_stage["topk"][wave_query]["t_end"]
                    for wave_query in range(wave_begin, wave_end)
                ),
            })

    for query in range(query_blocks):
        topk = by_stage["topk"][query]
        attention = by_stage["attention"][query]
        if mode in ("floor", "wave_floor"):
            bounds = dependency_bounds[query // dependency_wave_size]
            topk_overlap = topk["t_start"] < bounds["index_end"]
            attention_overlap = attention["t_start"] < bounds["topk_end"]
            topk_dependency_safe = topk["t_dep"] >= bounds["index_trigger"]
            attention_dependency_safe = (
                attention["t_dep"] >= bounds["topk_trigger"]
            )
            topk_early += topk_overlap
            attention_early += attention_overlap
            topk_waited += topk_overlap and topk_dependency_safe
            attention_waited += (
                attention_overlap and attention_dependency_safe
            )
            safety += not topk_dependency_safe
            safety += not attention_dependency_safe
        elif mode == "impl":
            base = query * physical_degree
            row_ready = max(
                by_stage["indexer"][base + parent]["t_ready"]
                for parent in range(physical_degree)
            )
            topk_early += topk["t_start"] < row_ready
            attention_early += attention["t_start"] < topk["t_ready"]
            topk_waited += topk["t_start"] < row_ready and topk["t_dep"] >= row_ready
            attention_waited += (
                attention["t_start"] < topk["t_ready"]
                and attention["t_dep"] >= topk["t_ready"]
            )
            safety += topk["t_dep"] < row_ready
            safety += attention["t_dep"] < topk["t_ready"]
        else:
            base = query * physical_degree
            row_ready = max(
                by_stage["indexer"][base + parent]["t_ready"]
                for parent in range(physical_degree)
            )
            topk_early += topk["t_start"] < row_ready
            attention_early += attention["t_start"] < topk["t_ready"]
    return {
        "ms": (last - first) / 1e6,
        "topk_early": topk_early,
        "attention_early": attention_early,
        "topk_waited": topk_waited,
        "attention_waited": attention_waited,
        "safety_failures": safety,
        "trigger_failures": trigger_failures,
        "progress_waves_verified": progress_waves_verified,
        "consumer_entry_order_failures": consumer_entry_order_failures,
        "producer_forward_progress_failures": producer_forward_progress_failures,
    }


def validate(
    log_path: Path, trace_path: Path | None, allow_short: bool,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    errors: list[str] = []
    records: dict[str, list[dict[str, str]]] = defaultdict(list)
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            for prefix in (
                "DEVICE_DSA", "CONFIG_DSA", "RESOURCE_DSA", "WARMUP_DSA", "SAMPLE_DSA",
                "VALIDATION_DSA", "CEILING_PROOF_DSA", "ADMISSION_DSA",
                "PROGRESS_DSA", "SUMMARY_DSA", "TRACE_DSA",
            ):
                if line.startswith(prefix + " "):
                    records[prefix].append(fields(line))
                    break
    except OSError as exc:
        errors.append(f"cannot read log: {exc}")

    for singleton in (
        "DEVICE_DSA", "CONFIG_DSA", "RESOURCE_DSA", "CEILING_PROOF_DSA", "ADMISSION_DSA",
        "SUMMARY_DSA", "TRACE_DSA",
    ):
        if len(records[singleton]) != 1:
            errors.append(f"{singleton} records={len(records[singleton])}, expected=1")
    if errors:
        return {"schema": 1, "status": "FAIL", "errors": errors}

    config_row = records["CONFIG_DSA"][0]
    device_row = records["DEVICE_DSA"][0]
    summary = records["SUMMARY_DSA"][0]
    try:
        seq = as_int(config_row, "seq")
        tag = config_row.get("tag", "")
        bind_identity(errors, records, tag, seq)
        errors.extend(device_identity_errors(config_row, device_row, expected_gpu_uuid))
        query_blocks = as_int(config_row, "query_blocks")
        key_tiles = as_int(config_row, "key_tiles")
        logical_degree = as_int(config_row, "logical_degree")
        physical_degree = as_int(config_row, "physical_cta_degree")
        topk = as_int(config_row, "topk")
        pair_query = as_int(config_row, "pair_query")
        pair_key = as_int(config_row, "pair_key")
        producer_ctas = as_int(config_row, "producer_ctas")
        topk_ctas = as_int(config_row, "topk_ctas")
        attention_ctas = as_int(config_row, "attention_ctas")
        sms = as_int(config_row, "sms")
        query_wave_size = as_int(config_row, "query_wave_size")
        query_wave_count = as_int(config_row, "query_wave_count")
        repeats = as_int(config_row, "repeats")
        warmup = as_int(config_row, "warmup")
        expected_physical = key_tiles if seq <= 32768 else min(key_tiles, 64)
        expected_mapping = (
            "exact" if expected_physical == key_tiles else "work_complete_packed_proxy"
        )
        exact = {
            "query_blocks": seq // 64,
            "key_tiles": seq // 128,
            "logical_degree": key_tiles,
            "physical_degree": expected_physical,
            "producer_ctas": query_blocks * physical_degree,
        }
        observed = {
            "query_blocks": query_blocks,
            "key_tiles": key_tiles,
            "logical_degree": logical_degree,
            "physical_degree": physical_degree,
            "producer_ctas": producer_ctas,
        }
        if seq <= 0 or seq % 128:
            errors.append("invalid seq")
        if observed != exact:
            errors.append(f"shape mismatch: observed={observed} expected={exact}")
        if topk != min(2048, key_tiles):
            errors.append("topk mismatch")
        if topk_ctas != query_blocks or attention_ctas != query_blocks or sms < 1:
            errors.append("consumer grid/SM coverage mismatch")
        expected_wave_size = min(query_blocks, max(1, (sms - 1) // 3))
        expected_wave_count = math.ceil(query_blocks / expected_wave_size)
        if query_wave_size != expected_wave_size:
            errors.append(
                f"query wave size mismatch: {query_wave_size}/{expected_wave_size}"
            )
        if query_wave_count != expected_wave_count:
            errors.append(
                f"query wave count mismatch: {query_wave_count}/{expected_wave_count}"
            )
        if as_int(config_row, "eff_degree") != physical_degree:
            errors.append("effective dependency degree mismatch")
        expected_tiles_per_cta = math.ceil(key_tiles / physical_degree)
        if as_int(config_row, "tiles_per_cta_max") != expected_tiles_per_cta:
            errors.append("tiles_per_cta_max mismatch")
        if not (1 <= pair_query <= 64 and 1 <= pair_key <= 128):
            errors.append("pair-work dimensions out of range")
        pair_complete = pair_query == 64 and pair_key == 128
        if not allow_short and not pair_complete:
            errors.append("formal artifacts require complete 64x128 pair work")
        if as_int(config_row, "pair_work_per_score") != pair_query * pair_key:
            errors.append("pair_work_per_score mismatch")
        pair_work_items = query_blocks * key_tiles * pair_query * pair_key
        if as_int(config_row, "pair_work_items") != pair_work_items:
            errors.append("pair_work_items mismatch")
        if as_int(config_row, "pair_work_complete") != int(pair_complete):
            errors.append("pair_work_complete mismatch")
        if pair_complete and pair_work_items != seq * seq:
            errors.append("complete pair work does not equal seq^2")
        errors.extend(pair_contract_errors(
            config_row, pair_query, pair_key, "CONFIG_DSA"
        ))
        expected_history_loads = query_blocks * topk
        errors.extend(history_load_contract_errors(
            config_row, expected_history_loads, "CONFIG_DSA", require_actual=False
        ))
        if config_row.get("mapping") != expected_mapping:
            errors.append("mapping label mismatch")
        expected_tag_fragment = "exact" if expected_mapping == "exact" else "packed"
        if expected_tag_fragment not in config_row.get("tag", ""):
            errors.append("tag does not encode exact/packed evidence boundary")
        close(
            errors,
            "interval_tightness",
            as_float(config_row, "interval_tightness"),
            logical_degree / (physical_degree * math.ceil(logical_degree / physical_degree)),
            1e-6,
        )
        if repeats < 31 and not allow_short:
            errors.append("formal validation requires >=31 repeats")
        if warmup < 1:
            errors.append("warmup must be >=1")
        if as_int(config_row, "nvtx") != 1:
            errors.append("official Tier-5 build requires NVTX annotations")
        expected_bytes = {
            "score_bytes": query_blocks * key_tiles * 4,
            "index_bytes": query_blocks * topk * 4,
            "output_bytes": query_blocks * 64 * 4,
            "trace_bytes": (producer_ctas + 2 * query_blocks) * 56,
        }
        for key, expected_value in expected_bytes.items():
            if as_int(config_row, key) != expected_value:
                errors.append(f"{key} mismatch")
        fixed_strings = {
            "timer": "globaltimer",
            "timer_scope": "first_cta_start_to_last_cta_end",
            "host_launch_before_first_cta_included": "0",
            "host_submission_path_differs": "1",
            "floor_graph_vs_impl_stream_submission": "outside_timer_not_normalized",
            "floor_path": "three_node_programmatic_graph",
            "wave_floor_path": "bounded_wave_three_node_programmatic_graph",
            "impl_path": "matched_priority_streams_epoch_flags",
            "ceiling_path": "matched_priority_streams_no_wait_no_publish",
            "stream_priority_pairing": "identical_by_stage",
            "launch_order": "attention_topk_indexer",
            "trigger_floor": "ready",
            "trigger_wave_floor": "ready",
            "trigger_impl": "entry",
            "trigger_ceiling": "entry",
            "floor_wait": "griddepcontrol",
            "wave_floor_wait": "griddepcontrol",
            "impl_wait": "per_producer_epoch_acquire",
            "ceiling_wait": "none",
            "topk_algorithm": "monotonic_analytical_proxy",
            "full_score_scan": "1",
            "validation": "untimed_device_full_element",
            "poison": "epoch_derived_full",
            "timed_reference_loops": "0",
            "expected_row_prep": "untimed_per_epoch",
            "structure": "interval",
            "long_context_boundary": "work_complete_packed_proxy",
            "forward_progress_protocol": "full_grid_floor_plus_bounded_query_waves",
            "wave_work_parity": "floor_wave_floor_impl_ceiling_equal",
            **FLOOR_TRACE_CONTRACT_STRINGS,
            "mode_order": "floor,wave_floor,impl,ceiling",
            "sample_order": "cyclic_latin_4",
        }
        for key, expected_value in fixed_strings.items():
            if config_row.get(key) != expected_value:
                errors.append(f"CONFIG_DSA {key} mismatch")
        if as_int(config_row, "mode_count") != 4:
            errors.append("CONFIG_DSA mode_count mismatch")
        if as_int(config_row, "full_grid_floor_single_launch") != 1:
            errors.append("CONFIG_DSA full-grid Floor is not a single launch")
        resource = records["RESOURCE_DSA"][0]
        for key in ("indexer_occ", "topk_occ", "attention_occ"):
            if as_int(resource, key) < 1:
                errors.append(f"RESOURCE_DSA {key} has no resident block")
        if (
            as_int(resource, "producer_progress_slot") != 1
            or as_int(resource, "topk_progress_with_attention") != 1
        ):
            errors.append("RESOURCE_DSA progress reservation missing")
        expected_reserved_sms = sms - 2 * query_wave_size
        if resource.get("progress_proof") != "global_consumer_cta_bound":
            errors.append("RESOURCE_DSA progress proof label mismatch")
        if as_int(resource, "query_wave_size") != query_wave_size:
            errors.append("RESOURCE_DSA query wave size mismatch")
        if as_int(resource, "consumer_ctas_per_wave") != 2 * query_wave_size:
            errors.append("RESOURCE_DSA consumer CTA bound mismatch")
        if as_int(resource, "reserved_producer_sms") != expected_reserved_sms:
            errors.append("RESOURCE_DSA reserved producer SM count mismatch")
        if as_int(resource, "total_sms") != sms or expected_reserved_sms < 1:
            errors.append("RESOURCE_DSA global consumer bound leaves no producer SM")
        least = as_int(resource, "stream_priority_least")
        greatest = as_int(resource, "stream_priority_greatest")
        topk_priority = as_int(resource, "stream_priority_topk")
        if as_int(resource, "distinct_priority_values") != 3:
            errors.append("RESOURCE_DSA must prove exactly three priority values")
        if len({least, topk_priority, greatest}) != 3:
            errors.append("RESOURCE_DSA reported priority values are not distinct")
        if topk_priority != greatest + (least - greatest) // 2:
            errors.append("topk priority is not the exact midpoint priority")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed CONFIG_DSA: {exc}")
        return {"schema": 1, "status": "FAIL", "errors": errors}

    expected_progress_modes: list[str] = ["floor", "wave_floor", "impl", "ceiling"]
    expected_progress_modes.extend(
        mode for _ in range(warmup) for mode in MODES
    )
    for rep in range(repeats):
        expected_progress_modes.extend(latin_order(rep))
    progress_records = records["PROGRESS_DSA"]
    progress_by_epoch: dict[int, dict[str, str]] = {}
    if len(progress_records) != len(expected_progress_modes):
        errors.append(
            f"progress audits={len(progress_records)}, "
            f"expected={len(expected_progress_modes)}"
        )
    for index, row in enumerate(progress_records):
        try:
            expected_epoch = index + 1
            expected_mode = expected_progress_modes[index]
            observed_epoch = as_int(row, "epoch")
            if observed_epoch != expected_epoch or row.get("mode") != expected_mode:
                errors.append(f"progress audit order mismatch at {index}")
            if row.get("tag") != tag or as_int(row, "seq") != seq:
                errors.append(f"progress audit identity mismatch at {index}")
            expected_protocol = (
                "full_grid_programmatic_graph"
                if expected_mode == "floor" else "bounded_query_waves"
            )
            expected_entry_gate = (
                "system_scope_mapped_counter"
                if expected_mode in ("impl", "ceiling")
                else "not_applicable_programmatic_graph"
            )
            fixed_progress = {
                "protocol": expected_protocol,
                "entry_gate": expected_entry_gate,
            }
            for key, expected_value in fixed_progress.items():
                if row.get(key) != expected_value:
                    errors.append(f"progress audit {index} {key} mismatch")
            exact_progress = {
                "entry_gate_timeout_ms": 5000,
                "progress_waves": 1 if expected_mode == "floor" else query_wave_count,
                "expected_progress_waves": (
                    1 if expected_mode == "floor" else query_wave_count
                ),
                "consumer_entries": 2 * query_blocks,
                "expected_consumer_entries": 2 * query_blocks,
                "consumer_completions": 2 * query_blocks,
                "expected_consumer_completions": 2 * query_blocks,
                "entry_gate_failures": 0,
                "impl_preproducer_completions": 0,
                "progress_waves_verified": (
                    1 if expected_mode == "floor" else query_wave_count
                ),
                "consumer_entry_order_failures": 0,
                "producer_forward_progress_failures": 0,
                "valid": 1,
            }
            for key, expected_value in exact_progress.items():
                if as_int(row, key) != expected_value:
                    errors.append(
                        f"progress audit {index} {key}="
                        f"{row.get(key)} expected={expected_value}"
                    )
            if observed_epoch in progress_by_epoch:
                errors.append(f"duplicate progress audit epoch {observed_epoch}")
            progress_by_epoch[observed_epoch] = row
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed progress audit {index}: {exc}")

    epoch = 0
    aggregate = history_sum(key_tiles, topk)
    expected_history_loads = query_blocks * topk
    validations = records["VALIDATION_DSA"]
    if len(validations) != 3:
        errors.append(f"validations={len(validations)}, expected=3")
    for index, row in enumerate(validations):
        try:
            epoch += 1
            mode = ("floor", "wave_floor", "impl")[index]
            if row.get("mode") != mode or as_int(row, "epoch") != epoch:
                errors.append(f"validation order/epoch mismatch at {index}")
            if row.get("checker") != "device_full_element" or row.get("poison") != "epoch_derived_full":
                errors.append(f"validation contract mismatch at {index}")
            if as_int(row, "trace_complete") != 1 or as_int(row, "valid") != 1:
                errors.append(f"validation did not pass at {index}")
            errors.extend(history_load_contract_errors(
                row, expected_history_loads, f"validation {mode}", require_actual=True
            ))
            expected_counts = {
                "score_elements": query_blocks * key_tiles,
                "index_elements": query_blocks * topk,
                "output_elements": query_blocks * 64,
            }
            for key, expected_value in expected_counts.items():
                if as_int(row, key) != expected_value:
                    errors.append(f"validation {mode} {key} mismatch")
            for key in (
                "score_mismatches", "index_mismatches", "output_mismatches",
                "row_mismatches", "flag_mismatches",
            ):
                if as_int(row, key) != 0:
                    errors.append(f"validation {mode} {key} != 0")
            expected_score = score_checksum(
                epoch, query_blocks, key_tiles, pair_query, pair_key
            )
            expected_index = index_checksum(query_blocks, key_tiles, topk)
            expected_output = output_checksum(epoch, query_blocks, aggregate)
            expected_flag = 0 if mode in ("floor", "wave_floor") else (
                epoch * (producer_ctas + query_blocks)
            ) & MASK64
            expected_sums = {
                "score_checksum": expected_score,
                "index_checksum": expected_index,
                "output_checksum": expected_output,
                "row_checksum": expected_score,
                "flag_checksum": expected_flag,
            }
            for prefix, expected_value in expected_sums.items():
                observed_value = as_int(row, prefix + "_observed")
                reported_expected = as_int(row, prefix + "_expected")
                if observed_value != expected_value or reported_expected != expected_value:
                    errors.append(
                        f"validation {mode} {prefix}: observed={observed_value} "
                        f"reported_expected={reported_expected} expected={expected_value}"
                    )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed validation {index}: {exc}")

    proof = records["CEILING_PROOF_DSA"][0]
    try:
        epoch += 1
        if proof.get("mode") != "ceiling" or as_int(proof, "epoch") != epoch:
            errors.append("Ceiling proof order/epoch mismatch")
        if proof.get("checker") != "device_wrongness_full_output_index":
            errors.append("Ceiling proof checker mismatch")
        if proof.get("poison") != "epoch_derived_full":
            errors.append("Ceiling proof poison mismatch")
        if as_int(proof, "trace_complete") != 1 or as_int(proof, "wrong") != 1:
            errors.append("Ceiling wrongness was not proven")
        if as_int(proof, "stale_rows") <= 0:
            errors.append("Ceiling proof has no stale rows")
        errors.extend(history_load_contract_errors(
            proof, expected_history_loads, "Ceiling proof", require_actual=True
        ))
        index_mismatch = as_int(proof, "index_mismatches")
        output_mismatch = as_int(proof, "output_mismatches")
        if index_mismatch <= 0 and output_mismatch <= 0:
            errors.append("Ceiling proof has no full-element mismatch")
        expected_index = index_checksum(query_blocks, key_tiles, topk)
        expected_output = output_checksum(epoch, query_blocks, aggregate)
        if as_int(proof, "index_checksum_expected") != expected_index:
            errors.append("Ceiling index expected checksum mismatch")
        if as_int(proof, "output_checksum_expected") != expected_output:
            errors.append("Ceiling output expected checksum mismatch")
        if (
            as_int(proof, "index_checksum_observed") == expected_index
            and as_int(proof, "output_checksum_observed") == expected_output
        ):
            errors.append("Ceiling observed checksums are both correct")
        admission = records["ADMISSION_DSA"][0]
        if (
            as_int(admission, "valid") != 1
            or as_int(admission, "validations") != 3
            or as_int(admission, "ceiling_proofs") != 1
            or as_int(admission, "history_load_complete") != 1
        ):
            errors.append("admission record did not pass")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed Ceiling/admission proof: {exc}")

    warmups = records["WARMUP_DSA"]
    if len(warmups) != warmup * 4:
        errors.append(f"warmups={len(warmups)}, expected={warmup * 4}")
    for index, row in enumerate(warmups):
        try:
            epoch += 1
            if as_int(row, "warmup") != index // 4 or row.get("mode") != MODES[index % 4]:
                errors.append(f"warmup order mismatch at {index}")
            if as_int(row, "epoch") != epoch or as_int(row, "trace_complete") != 1:
                errors.append(f"warmup epoch/trace mismatch at {index}")
            errors.extend(history_load_contract_errors(
                row, expected_history_loads, f"warmup {index}", require_actual=True
            ))
            stale = as_int(row, "stale_rows")
            expected_wrong = int(row.get("mode") == "ceiling" and stale > 0)
            if (row.get("mode") == "ceiling") != (stale > 0):
                errors.append(f"warmup stale semantics mismatch at {index}")
            if as_int(row, "ceiling_wrong") != expected_wrong:
                errors.append(f"warmup Ceiling wrongness mismatch at {index}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed warmup {index}: {exc}")

    samples = records["SAMPLE_DSA"]
    if len(samples) != repeats * 4:
        errors.append(f"samples={len(samples)}, expected={repeats * 4}")
    values: dict[str, list[float]] = {mode: [math.nan] * repeats for mode in MODES}
    samples_by_mode_rep: dict[tuple[str, int], dict[str, str]] = {}
    for index, row in enumerate(samples):
        try:
            epoch += 1
            rep = index // 4
            order_index = index % 4
            expected_order = latin_order(rep)
            mode = row["mode"]
            if as_int(row, "rep") != rep or as_int(row, "order") != order_index:
                errors.append(f"sample position mismatch at {index}")
            if mode != expected_order[order_index]:
                errors.append(f"sample mode order mismatch at {index}")
            if as_int(row, "epoch") != epoch:
                errors.append(f"sample epoch mismatch at {index}")
            if as_int(row, "trace_complete") != 1:
                errors.append(f"sample trace incomplete at {index}")
            errors.extend(history_load_contract_errors(
                row, expected_history_loads, f"sample {index}", require_actual=True
            ))
            if as_int(row, "safety_failures") != 0 or as_int(row, "trigger_failures") != 0:
                errors.append(f"sample semantic failure at {index}")
            expected_safety = "not_applicable" if mode == "ceiling" else "dependency_required"
            if row.get("safety_applicability") != expected_safety:
                errors.append(f"sample safety applicability mismatch at {index}")
            stale = as_int(row, "stale_rows")
            expected_wrong = int(mode == "ceiling" and stale > 0)
            if (mode == "ceiling") != (stale > 0):
                errors.append(f"sample stale semantics mismatch at {index}")
            if as_int(row, "ceiling_wrong") != expected_wrong:
                errors.append(f"sample Ceiling wrongness mismatch at {index}")
            if not allow_short and mode in ("floor", "wave_floor", "impl"):
                for key in (
                    "topk_early", "attention_early", "topk_waited", "attention_waited",
                ):
                    if as_int(row, key) <= 0:
                        errors.append(f"sample {mode}/{rep} has no {key} overlap")
            ms = as_float(row, "ms")
            if ms <= 0:
                errors.append(f"non-positive sample at {index}")
            values[mode][rep] = ms
            if (mode, rep) in samples_by_mode_rep:
                errors.append(f"duplicate sample {mode}/{rep}")
            samples_by_mode_rep[(mode, rep)] = row
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed sample {index}: {exc}")
    if any(not math.isfinite(v) for mode in MODES for v in values[mode]):
        errors.append("incomplete sample matrix")

    try:
        if as_int(summary, "epoch_first") != 1 or as_int(summary, "epoch_last") != epoch:
            errors.append("summary epoch range mismatch")
        if as_int(summary, "repeats") != repeats or as_int(summary, "warmup") != warmup:
            errors.append("summary repeat/warmup mismatch")
        if (
            as_int(summary, "samples") != repeats * 4
            or as_int(summary, "validations") != 3
            or as_int(summary, "ceiling_proofs") != 1
        ):
            errors.append("summary ledger mismatch")
        if as_int(summary, "valid") != 1:
            errors.append("summary validity mismatch")
        errors.extend(pair_contract_errors(
            summary, pair_query, pair_key, "SUMMARY_DSA"
        ))
        errors.extend(history_load_contract_errors(
            summary, expected_history_loads, "SUMMARY_DSA", require_actual=False
        ))
        if as_int(summary, "history_load_complete") != 1:
            errors.append("SUMMARY_DSA history_load_complete != 1")
        if (
            summary.get("ceiling_correctness") != "unsafe_not_validated"
            or as_int(summary, "ceiling_verified") != 0
            or as_int(summary, "ceiling_wrongness_verified") != 1
        ):
            errors.append("summary Ceiling correctness/wrongness contract mismatch")
        for key in (
            "seq", "query_blocks", "key_tiles", "logical_degree", "physical_cta_degree",
            "tiles_per_cta_max", "mapping", "interval_tightness", "eff_degree",
            "producer_ctas", "topk_ctas", "attention_ctas", "sms",
            "query_wave_size", "query_wave_count", "mode_count", "mode_order",
            "sample_order",
            "forward_progress_protocol", "wave_work_parity",
            "floor_overlap_metric", "floor_dependency_metric",
            "topk", "pair_query", "pair_key", "pair_work_items", "pair_work_complete",
        ):
            if summary.get(key) != config_row.get(key):
                errors.append(f"summary/config {key} mismatch")
        close(
            errors, "summary tightness", as_float(summary, "tightness"),
            as_float(config_row, "interval_tightness"), 1e-6,
        )
        resource = records["RESOURCE_DSA"][0]
        for key in ("indexer_occ", "topk_occ", "attention_occ"):
            if summary.get(key) != resource.get(key):
                errors.append(f"summary/resource {key} mismatch")
        for key, expected_value in {
            "timer": "globaltimer",
            "timer_scope": "first_cta_start_to_last_cta_end",
            "host_launch_before_first_cta_included": "0",
            "host_submission_path_differs": "1",
            "floor_graph_vs_impl_stream_submission": "outside_timer_not_normalized",
            "adjacent_rungs": "1",
            "floor_path": "three_node_programmatic_graph",
            "full_grid_floor_single_launch": "1",
            "wave_floor_path": "bounded_wave_three_node_programmatic_graph",
            "impl_path": "matched_priority_streams_epoch_flags",
            "ceiling_path": "matched_priority_streams_no_wait_no_publish",
            "stream_priority_pairing": "identical_by_stage",
            "launch_order": "attention_topk_indexer",
            "trigger_floor": "ready",
            "trigger_wave_floor": "ready",
            "trigger_impl": "entry",
            "trigger_ceiling": "entry",
            "floor_wait": "griddepcontrol",
            "wave_floor_wait": "griddepcontrol",
            "impl_wait": "per_producer_epoch_acquire",
            "ceiling_wait": "none",
            "timed_reference_loops": "0",
            "expected_row_prep": "untimed_per_epoch",
            "ceiling_trace_safety": "not_applicable",
            "structure": "interval",
            "long_context_boundary": "work_complete_packed_proxy",
            "captured_interpretation": (
                "bounded_wave_end_to_end_mechanism_envelope_not_pure_cta_headroom"
            ),
            "matched_protocol_interpretation": (
                "wave_boundary_matched_pdl_vs_epoch_flags"
            ),
            **FLOOR_TRACE_CONTRACT_STRINGS,
        }.items():
            if summary.get(key) != expected_value:
                errors.append(f"SUMMARY_DSA {key} mismatch")
        stats = {
            "floor": bootstrap_median(values["floor"], 0xD5A00001 + seq),
            "wave_floor": bootstrap_median(values["wave_floor"], 0xD5A00006 + seq),
            "impl": bootstrap_median(values["impl"], 0xD5A00002 + seq),
            "ceiling": bootstrap_median(values["ceiling"], 0xD5A00003 + seq),
        }
        for mode, (center, low, high) in stats.items():
            close(errors, f"{mode}_ms", as_float(summary, f"{mode}_ms"), center)
            close(errors, f"{mode}_ci_low", as_float(summary, f"{mode}_ci_low"), low)
            close(errors, f"{mode}_ci_high", as_float(summary, f"{mode}_ci_high"), high)
        space = bootstrap_pair_delta(values, "floor", "ceiling", 0xD5A00004 + seq)
        captured = bootstrap_pair_delta(values, "floor", "impl", 0xD5A00005 + seq)
        full_to_wave = bootstrap_pair_delta(
            values, "floor", "wave_floor", 0xD5A00007 + seq
        )
        matched_protocol = bootstrap_pair_delta(
            values, "wave_floor", "impl", 0xD5A00008 + seq
        )
        for prefix, triple in (
            ("space", space), ("captured", captured),
            ("full_to_wave", full_to_wave),
            ("matched_protocol", matched_protocol),
        ):
            close(errors, f"{prefix}_pct", as_float(summary, f"{prefix}_pct"), triple[0])
            close(errors, f"{prefix}_ci_low", as_float(summary, f"{prefix}_ci_low"), triple[1])
            close(errors, f"{prefix}_ci_high", as_float(summary, f"{prefix}_ci_high"), triple[2])
        expected_of_space = 100.0 * captured[0] / space[0] if space[0] else 0.0
        close(errors, "of_space_pct", as_float(summary, "of_space_pct"), expected_of_space)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed SUMMARY_DSA: {exc}")

    pair_arithmetic_cases = pair_low16_equivalence_cases(
        query_blocks, key_tiles, epoch, pair_query, pair_key
    )
    if not all(bool(case["equivalent"]) for case in pair_arithmetic_cases):
        errors.append("uint32 pair accumulator low16 equivalence regression failed")

    trace_decl = records["TRACE_DSA"][0]
    declared_path = Path(trace_decl.get("path", ""))
    selected_trace = trace_path or declared_path
    expected_trace_rows = 4 * (producer_ctas + 2 * query_blocks)
    try:
        if as_int(trace_decl, "rep") != repeats - 1:
            errors.append("TRACE_DSA rep mismatch")
        if as_int(trace_decl, "modes") != 4:
            errors.append("TRACE_DSA modes mismatch")
        if as_int(trace_decl, "rows") != expected_trace_rows:
            errors.append("TRACE_DSA rows mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"malformed TRACE_DSA declaration: {exc}")
    if trace_path is not None and declared_path != trace_path:
        errors.append(f"trace path mismatch: declared={declared_path} supplied={trace_path}")
    trace_rows: list[dict[str, str]] = []
    expected_trace_header = [
        "schema", "tag", "seq", "mode", "rep", "epoch", "stage", "block", "sm",
        "t_start", "t_dep", "t_ready", "t_trigger", "t_end",
    ]
    try:
        with selected_trace.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_trace_header:
                errors.append(
                    f"trace header mismatch: observed={reader.fieldnames} "
                    f"expected={expected_trace_header}"
                )
            trace_rows = list(reader)
    except OSError as exc:
        errors.append(f"cannot read trace: {exc}")
    if len(trace_rows) != expected_trace_rows:
        errors.append(f"trace rows={len(trace_rows)}, expected={expected_trace_rows}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        try:
            if None in row or any(value is None for value in row.values()):
                errors.append("trace row column count mismatch")
            if (
                int(row["schema"]) != 1
                or row["tag"] != tag
                or int(row["seq"]) != seq
            ):
                errors.append("trace schema/tag/seq mismatch")
            if int(row["rep"]) != repeats - 1:
                errors.append("trace rep mismatch")
            if row["mode"] not in MODES:
                errors.append("trace mode mismatch")
            else:
                grouped[row["mode"]].append(row)
        except (KeyError, TypeError, ValueError):
            errors.append("malformed trace identity row")
    trace_metrics: dict[str, dict[str, int | float | str]] = {}
    for mode in MODES:
        final_sample = samples_by_mode_rep.get((mode, repeats - 1))
        if final_sample is None:
            errors.append(f"missing final sample {mode}")
            continue
        try:
            final_epoch = as_int(final_sample, "epoch")
            metric = analyze_trace(
                grouped[mode],
                {
                    "producer_ctas": producer_ctas,
                    "query_blocks": query_blocks,
                    "physical_degree": physical_degree,
                    "query_wave_size": query_wave_size,
                    "sms": sms,
                },
                mode,
                final_epoch,
                errors,
            )
            metric["safety_applicability"] = (
                "not_applicable" if mode == "ceiling" else "dependency_required"
            )
            trace_metrics[mode] = metric
            if metric:
                close(errors, f"trace {mode} ms", as_float(final_sample, "ms"), float(metric["ms"]), 1e-6)
                for key in (
                    "topk_early", "attention_early", "topk_waited", "attention_waited",
                    "safety_failures", "trigger_failures",
                ):
                    if as_int(final_sample, key) != metric[key]:
                        errors.append(
                            f"trace {mode} {key}: sample={as_int(final_sample, key)} "
                            f"recomputed={metric[key]}"
                        )
                progress = progress_by_epoch.get(final_epoch)
                if progress is None:
                    errors.append(f"missing final progress audit {mode}")
                else:
                    for key in (
                        "progress_waves_verified", "consumer_entry_order_failures",
                        "producer_forward_progress_failures",
                    ):
                        if as_int(progress, key) != metric[key]:
                            errors.append(
                                f"trace {mode} {key}: progress={as_int(progress, key)} "
                                f"recomputed={metric[key]}"
                            )
                if not allow_short and mode in ("floor", "wave_floor", "impl"):
                    for key in (
                        "topk_early", "attention_early", "topk_waited", "attention_waited",
                    ):
                        if int(metric[key]) <= 0:
                            errors.append(f"final trace {mode} has no {key} overlap")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"cannot bind final trace {mode}: {exc}")

    return {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "log": str(log_path),
        "trace": str(selected_trace),
        "tag": tag,
        "seq": seq,
        "mapping": config_row.get("mapping"),
        "logical_degree": logical_degree,
        "physical_cta_degree": physical_degree,
        "effective_degree": physical_degree,
        "producer_ctas": producer_ctas,
        "topk_ctas": topk_ctas,
        "attention_ctas": attention_ctas,
        "sms": sms,
        "device": {
            "runtime_ordinal": as_int(config_row, "runtime_ordinal"),
            "runtime_ordinal_zero": as_int(config_row, "runtime_ordinal_zero") == 1,
            "runtime_uuid": config_row.get("runtime_uuid"),
            "name_hex": config_row.get("runtime_name_hex"),
            "cc_major": as_int(config_row, "runtime_cc_major"),
            "cc_minor": as_int(config_row, "runtime_cc_minor"),
            "sms": as_int(config_row, "runtime_sms"),
            "expected_lease_uuid": expected_gpu_uuid,
        },
        "pair_work_items": query_blocks * key_tiles * pair_query * pair_key,
        "pair_work_complete": pair_complete,
        "pair_arithmetic_proof": {
            "status": "PASS" if all(
                bool(case["equivalent"]) for case in pair_arithmetic_cases
            ) else "FAIL",
            "identity": "(S mod 2^32) mod 2^16 = S mod 2^16",
            "cases": pair_arithmetic_cases,
        },
        "expected_history_loads_per_invocation": expected_history_loads,
        "history_load_complete": summary.get("history_load_complete") == "1",
        "repeats": repeats,
        "mode_count": len(MODES),
        "samples": len(samples),
        "validations": len(validations),
        "progress_audits": len(progress_records),
        "trace_rows": len(trace_rows),
        "ceiling_correctness": summary.get("ceiling_correctness"),
        "ceiling_verified": summary.get("ceiling_verified") == "1",
        "ceiling_wrongness_verified": summary.get("ceiling_wrongness_verified") == "1",
        "summary": summary,
        "trace_metrics": trace_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()
    try:
        result = validate(
            args.log, args.trace, args.allow_short, args.expected_gpu_uuid
        )
    except Exception as exc:  # fail closed even for malformed/unanticipated artifacts
        result = {"schema": 1, "status": "FAIL", "errors": [f"internal validator error: {exc}"]}
    atomic_json(args.json, result)
    print(
        "DSA_VALIDATION "
        f"schema=1 status={result['status']} errors={len(result.get('errors', []))} "
        f"seq={result.get('seq', 'unknown')} samples={result.get('samples', 0)} "
        f"validations={result.get('validations', 0)} trace_rows={result.get('trace_rows', 0)}"
    )
    if result.get("errors"):
        for error in result["errors"][:20]:
            print(f"ERROR: {error}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
