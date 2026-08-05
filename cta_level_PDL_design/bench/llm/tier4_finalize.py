#!/usr/bin/env python3
"""Strictly admit and analyze a completed Tier-4 raw triplet.

This command never manufactures missing samples.  It first validates exact
adjacency, poison identity, worker cohort, output digests, and sample counts;
then attaches one exported Nsight SQLite to the three runtime records and runs
the independent PTX/cubin/NVTX graph-node validator.  Only after every check
passes does it write an admissible analysis with bootstrap confidence
intervals.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Callable

from model_identity import verify_model_identity
from pdl_evidence import (
    PDL_SCOPE,
    PREFILL_MODE,
    RUNGS,
    sha256_file,
    validate_evidence,
)
from tier4_manifest import (
    MATRIX,
    SOURCE_NAMES,
    first_difference,
    immutable_contract,
)


RAW_SCHEMA = "tier4.triplet.raw.v2"
ORDER_CYCLE = (
    ("pdl_off", "pdl_grid", "ceiling"),
    ("pdl_grid", "ceiling", "pdl_off"),
    ("ceiling", "pdl_off", "pdl_grid"),
)
FORMAL_REPEATS = 31
FORMAL_WARMUPS = 3
FORMAL_BOOTSTRAP_SAMPLES = 2000
def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def token_digest(output: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [record["token_ids"] for record in output], separators=(",", ":")
        ).encode()
    ).hexdigest()


def _expected_switch_bindings(
    workers: Any, rung: str, prefix: str, errors: list[str]
) -> list[dict[str, Any]]:
    """Validate build-time active identities and return compact bindings."""
    if not isinstance(workers, list) or not workers:
        errors.append(f"{prefix}: build probe worker list is absent")
        return []
    bindings: list[dict[str, Any]] = []
    digest_fields = (
        "compiled_callable_fingerprint",
        "graph_dictionary_fingerprint",
        "batch_graph_fingerprint",
        "activation_fingerprint",
    )
    for index, worker in enumerate(workers):
        worker_prefix = f"{prefix} worker[{index}]"
        if not isinstance(worker, dict):
            errors.append(f"{worker_prefix}: probe is not an object")
            continue
        active = worker.get("active_variant")
        if not isinstance(active, dict):
            errors.append(f"{worker_prefix}: active variant identity is absent")
            continue
        if active.get("schema") != "tier4.active_variant.v1":
            errors.append(f"{worker_prefix}: active variant schema mismatch")
        if active.get("rung") != rung or worker.get("rung") != rung:
            errors.append(f"{worker_prefix}: active variant rung mismatch")
        for field in digest_fields:
            value = active.get(field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                errors.append(f"{worker_prefix}: invalid {field}")
        graph_dictionary_id = active.get("graph_dictionary_id")
        if (
            not isinstance(graph_dictionary_id, int)
            or isinstance(graph_dictionary_id, bool)
            or graph_dictionary_id <= 0
        ):
            errors.append(f"{worker_prefix}: invalid graph_dictionary_id")
        entries = active.get("full_graph_entries")
        if (
            not isinstance(entries, int)
            or isinstance(entries, bool)
            or entries <= 0
            or worker.get("full_graph_entries") != entries
        ):
            errors.append(f"{worker_prefix}: active graph entry count mismatch")

        modules = worker.get("compiled_modules")
        if isinstance(modules, list):
            callable_records = sorted(
                (
                    {
                        "name": module.get("name"),
                        "compiled_callable_id": module.get("compiled_callable_id"),
                        "aot_compiled_fn_id": module.get("aot_compiled_fn_id"),
                        "compiled_bytecode_id": module.get("compiled_bytecode_id"),
                    }
                    for module in modules
                    if isinstance(module, dict)
                ),
                key=lambda item: str(item["name"]),
            )
            if active.get("compiled_callable_fingerprint") != digest_json(
                callable_records
            ):
                errors.append(f"{worker_prefix}: compiled callable fingerprint mismatch")
        activation_payload = {
            "rung": rung,
            "compiled_callable_fingerprint": active.get(
                "compiled_callable_fingerprint"
            ),
            "graph_dictionary_id": graph_dictionary_id,
            "graph_dictionary_fingerprint": active.get(
                "graph_dictionary_fingerprint"
            ),
            "batch_graph_fingerprint": active.get("batch_graph_fingerprint"),
            "full_graph_entries": entries,
        }
        if active.get("activation_fingerprint") != digest_json(activation_payload):
            errors.append(f"{worker_prefix}: activation fingerprint mismatch")
        pid = worker.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool):
            bindings.append({"pid": pid, **active})
    return sorted(bindings, key=lambda item: item["pid"])


def poison_tokens(point: dict[str, Any], epoch: int) -> list[list[int]]:
    return [
        [
            1000 + ((position * 17 + request * 31 + epoch * 7) % 5000)
            for position in range(point["seq"])
        ]
        for request in range(point["batch"])
    ]


def poison_digest(point: dict[str, Any], epoch: int) -> str:
    return hashlib.sha256(
        json.dumps(poison_tokens(point, epoch), separators=(",", ":")).encode()
    ).hexdigest()


def bootstrap_ci(
    values: list[float],
    samples: int,
    seed: int,
    statistic: Callable[[list[float]], float] = statistics.median,
) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistic([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(samples)
    )
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def paired_ratio_ci(
    numerator: list[float], denominator: list[float], samples: int, seed: int
) -> list[float]:
    rng = random.Random(seed)
    count = len(numerator)
    estimates: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        top = statistics.median(numerator[index] for index in indices)
        bottom = statistics.median(denominator[index] for index in indices)
        estimates.append(100.0 * (top - bottom) / top)
    estimates.sort()
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def _validate_output(
    output: Any,
    point: dict[str, Any],
    prefix: str,
    errors: list[str],
    allow_nonfinite: bool,
) -> bool | None:
    if not isinstance(output, list) or len(output) != point["batch"]:
        errors.append(f"{prefix}: output request count mismatch")
        return None
    nonfinite = False
    for request_index, record in enumerate(output):
        if not isinstance(record, dict):
            errors.append(f"{prefix}: output[{request_index}] is not an object")
            continue
        if record.get("request_index") != request_index:
            errors.append(f"{prefix}: output request order mismatch")
        token_ids = record.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != point["gen"]
            or any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids)
        ):
            errors.append(f"{prefix}: invalid generated token list")
        logprob = record.get("cumulative_logprob_hex")
        if not isinstance(logprob, str):
            errors.append(f"{prefix}: cumulative logprob hex is absent")
        else:
            try:
                value = float.fromhex(logprob)
                if not math.isfinite(value):
                    nonfinite = True
                    if not allow_nonfinite:
                        errors.append(
                            f"{prefix}: safe rung cumulative logprob is non-finite"
                        )
            except ValueError:
                errors.append(f"{prefix}: invalid cumulative logprob hex")
    return nonfinite


def _formal_matrix_entry(raw: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    matches = [
        (name, entry)
        for name, entry in MATRIX.items()
        if entry["cohort_id"] == raw.get("cohort_id")
    ]
    return matches[0] if len(matches) == 1 else None


def _validate_formal_contract(raw: dict[str, Any], errors: list[str]) -> None:
    matched = _formal_matrix_entry(raw)
    if matched is None:
        errors.append("formal cohort_id is not one exact declared Tier-4 cohort")
        return
    cohort_name, expected = matched
    if raw.get("points") != expected["points"]:
        errors.append(f"formal {cohort_name} point matrix differs from declaration")
    if raw.get("repeats") != FORMAL_REPEATS:
        errors.append(f"formal repeats must equal {FORMAL_REPEATS}")
    if raw.get("warmups") != FORMAL_WARMUPS:
        errors.append(f"formal warmups must equal {FORMAL_WARMUPS}")
    if raw.get("bootstrap_samples") != FORMAL_BOOTSTRAP_SAMPLES:
        errors.append(
            f"formal bootstrap_samples must equal {FORMAL_BOOTSTRAP_SAMPLES}"
        )

    formal = raw.get("formal_manifest")
    if not isinstance(formal, dict):
        errors.append("raw formal_manifest is absent")
        return
    all_engine_points = [*expected["points"], expected["proof_point"]]
    max_batch = max(point["batch"] for point in all_engine_points)
    expected_contract = {
        "cohort_id": expected["cohort_id"],
        "model_identity_manifest_sha256": raw.get(
            "model_identity_manifest_sha256"
        ),
        "model_identity_snapshot_sha256": raw.get(
            "model_identity_snapshot_sha256"
        ),
        "formal_root_manifest_sha256": raw.get("formal_root_manifest_sha256"),
        "semantic_rungs": list(RUNGS),
        "proof_rung_order": list(RUNGS),
        "timing_order_cycle": [list(order) for order in ORDER_CYCLE],
        "points": expected["points"],
        "proof_point": expected["proof_point"],
        "repeats_per_rung": FORMAL_REPEATS,
        "warmup_triplets_per_point": FORMAL_WARMUPS,
        "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
        "max_num_seqs": max_batch,
        "max_model_len": max(
            point["seq"] + point["gen"] for point in all_engine_points
        ),
        "max_num_batched_tokens": max(
            expected["max_num_batched_tokens"], max_batch
        ),
        "gpu_memory_utilization": expected["gpu_memory_utilization"],
        "graph_mode": "FULL_DECODE_ONLY",
        "pdl_scope": PDL_SCOPE,
        "prefill_mode": PREFILL_MODE,
        "trtllm_enable_pdl": False,
        "text_only_limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    for field, value in expected_contract.items():
        if formal.get(field) != value:
            errors.append(f"formal_manifest.{field} differs from declaration")
    if formal.get("kv_offloading_backend") != "native":
        errors.append("formal_manifest.kv_offloading_backend must be native")
    kv_size = formal.get("kv_offloading_size_gib")
    if kv_size is not None and (
        not isinstance(kv_size, (int, float))
        or isinstance(kv_size, bool)
        or not math.isfinite(kv_size)
        or kv_size <= 0
    ):
        errors.append("formal_manifest.kv_offloading_size_gib is invalid")


def validate_root_manifest(
    raw: dict[str, Any], results: Path
) -> tuple[list[str], dict[str, Any] | None, Path]:
    """Validate the immutable root manifest and return content to snapshot."""
    errors: list[str] = []
    manifest_path = results.parent.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read formal root manifest {manifest_path}: {exc}"], None, manifest_path
    if not isinstance(manifest, dict):
        return ["formal root manifest is not an object"], None, manifest_path
    if manifest.get("schema") != "tier4.formal.manifest.v3":
        errors.append("formal root manifest schema mismatch")
    if manifest.get("model") != raw.get("model"):
        errors.append("formal root manifest model path differs from raw candidate")
    if manifest.get("model_fingerprint") != raw.get("model_fingerprint"):
        errors.append("formal root manifest model fingerprint differs from candidate")
    identity_name = manifest.get("model_identity_manifest")
    identity_hash = manifest.get("model_identity_manifest_sha256")
    identity_object: dict[str, Any] | None = None
    actual_identity_hash: str | None = None
    if not isinstance(identity_name, str) or not identity_name:
        errors.append("formal root manifest model identity path is absent")
    else:
        identity_path = Path(identity_name).resolve()
        expected_identity_path = results.parent.parent / "model_identity.json"
        if identity_path != expected_identity_path.resolve():
            errors.append("formal root manifest model identity path mismatch")
        elif not identity_path.is_file():
            errors.append("formal root model identity file is absent")
        else:
            actual_identity_hash = sha256_file(identity_path)
            try:
                loaded_identity = json.loads(identity_path.read_text())
                if isinstance(loaded_identity, dict):
                    identity_object = loaded_identity
                else:
                    errors.append("formal root model identity is not an object")
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"cannot read formal root model identity: {exc}")
            if identity_hash != actual_identity_hash:
                errors.append("formal root model identity hash is stale")
            if raw.get("model_identity_manifest_sha256") != actual_identity_hash:
                errors.append("raw candidate is not bound to root model identity")
    exact_fields = {
        "semantic_rungs": list(RUNGS),
        "proof_rung_order": list(RUNGS),
        "timing_order_cycle": [list(order) for order in ORDER_CYCLE],
        "repeats_per_rung_per_point": FORMAL_REPEATS,
        "warmup_triplets_per_point": FORMAL_WARMUPS,
        "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
        "graph_mode": "FULL_DECODE_ONLY",
        "pdl_scope": PDL_SCOPE,
        "trtllm_enable_pdl": False,
        "cohorts": MATRIX,
    }
    for field, expected in exact_fields.items():
        if manifest.get(field) != expected:
            errors.append(f"formal root manifest {field} mismatch")

    source_root = Path(__file__).resolve().parent
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict):
        errors.append("formal root manifest source hashes are absent")
    else:
        for name in SOURCE_NAMES:
            if source_hashes.get(name) != sha256_file(source_root / name):
                errors.append(f"formal root manifest source hash mismatch: {name}")

    matched = _formal_matrix_entry(raw)
    if matched is not None:
        cohort_name, expected = matched
        cohorts = manifest.get("cohorts")
        if not isinstance(cohorts, dict) or cohorts.get(cohort_name) != expected:
            errors.append(f"root manifest {cohort_name} cohort declaration mismatch")
    formal = raw.get("formal_manifest")
    kv_strategy = manifest.get("kv_strategy")
    if isinstance(formal, dict) and isinstance(kv_strategy, dict):
        if kv_strategy.get("kv_offloading_size_gib") != formal.get(
            "kv_offloading_size_gib"
        ):
            errors.append("root/raw KV offloading size mismatch")
    else:
        errors.append("root/raw KV strategy metadata is absent")
    model_name = manifest.get("model")
    if (
        identity_object is not None
        and actual_identity_hash is not None
        and isinstance(model_name, str)
        and isinstance(kv_strategy, dict)
    ):
        expected_manifest = immutable_contract(
            results.parent.parent,
            Path(model_name).resolve(),
            identity_object,
            actual_identity_hash,
            kv_strategy.get("kv_offloading_size_gib"),
        )
        difference = first_difference(expected_manifest, manifest)
        if difference:
            errors.append(f"formal immutable root contract differs: {difference}")
    return errors, manifest, manifest_path


def validate_launch_manifest_snapshot(
    raw: dict[str, Any], results: Path, current_manifest: dict[str, Any] | None
) -> list[str]:
    errors: list[str] = []
    path = results / "launch_manifest_snapshot.json"
    try:
        launch = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read launch manifest snapshot: {exc}"]
    if not isinstance(launch, dict):
        return ["launch manifest snapshot is not an object"]
    if raw.get("formal_root_manifest_sha256") != sha256_file(path):
        errors.append("raw launch manifest snapshot hash mismatch")
    for field in (
        "schema",
        "model",
        "model_fingerprint",
        "model_identity_manifest_sha256",
        "semantic_rungs",
        "proof_rung_order",
        "timing_order_cycle",
        "repeats_per_rung_per_point",
        "warmup_triplets_per_point",
        "bootstrap_samples",
        "graph_mode",
        "pdl_scope",
        "trtllm_enable_pdl",
        "cohorts",
        "kv_strategy",
        "source_sha256",
    ):
        if current_manifest is None or launch.get(field) != current_manifest.get(field):
            errors.append(f"launch/current formal manifest {field} mismatch")
    if launch.get("model") != raw.get("model"):
        errors.append("launch manifest model path differs from raw candidate")
    if launch.get("model_fingerprint") != raw.get("model_fingerprint"):
        errors.append("launch manifest model fingerprint differs from raw candidate")
    if launch.get("model_identity_manifest_sha256") != raw.get(
        "model_identity_manifest_sha256"
    ):
        errors.append("launch manifest model identity differs from raw candidate")
    if current_manifest is not None and launch != current_manifest:
        errors.append("launch/current immutable formal manifest differs in full")
    return errors


def validate_raw(raw: dict[str, Any], allow_short: bool) -> list[str]:
    errors: list[str] = []
    if raw.get("schema") != RAW_SCHEMA:
        errors.append(f"raw schema is not {RAW_SCHEMA}")
    if raw.get("status") != "candidate" or raw.get("admissible") is not False:
        errors.append("raw input is not an immutable non-admitted candidate")
    if raw.get("triplet_mode") != "same_process_adjacent_latin3":
        errors.append("raw triplet mode is not adjacent Latin-3")
    if raw.get("resolved_graph_mode") != "FULL_DECODE_ONLY":
        errors.append("raw graph mode is not FULL_DECODE_ONLY")
    if raw.get("pdl_scope") != PDL_SCOPE or raw.get("prefill_mode") != PREFILL_MODE:
        errors.append("raw PDL/prefill scope is not explicit")
    if raw.get("trtllm_enable_pdl") is not False:
        errors.append("TRTLLM PDL must remain explicitly out of scope")
    for field in (
        "model_fingerprint",
        "model_identity_manifest_sha256",
        "model_identity_snapshot_sha256",
    ):
        value = raw.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append(f"raw {field} is not a SHA-256 digest")
    if raw.get("model_identity_manifest_sha256") != raw.get(
        "model_identity_snapshot_sha256"
    ):
        errors.append("raw root/snapshot model identity hashes differ")
    if not isinstance(raw.get("allow_short"), bool):
        errors.append("raw allow_short flag is not Boolean")
    elif not allow_short and raw.get("allow_short") is not False:
        errors.append("formal admission rejects allow_short candidates")
    if not allow_short:
        root_hash = raw.get("formal_root_manifest_sha256")
        if not isinstance(root_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", root_hash
        ) is None:
            errors.append("raw formal_root_manifest_sha256 is not a SHA-256 digest")

    points = raw.get("points")
    repeats = raw.get("repeats")
    warmups = raw.get("warmups")
    bootstrap_samples = raw.get("bootstrap_samples")
    cohort = raw.get("worker_cohort")
    if not isinstance(points, list) or not points:
        return errors + ["raw points are absent"]
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        errors.append("raw data requires at least one repeat")
    if (
        not isinstance(warmups, int)
        or isinstance(warmups, bool)
        or warmups < 1
    ):
        errors.append("raw data requires at least one warmup triplet per point")
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 100
    ):
        errors.append("bootstrap sample count is invalid")
    if (
        not isinstance(cohort, list)
        or not cohort
        or any(not isinstance(pid, int) or isinstance(pid, bool) for pid in cohort)
    ):
        errors.append("raw worker cohort is invalid")
    elif cohort != sorted(set(cohort)):
        errors.append("raw worker cohort is not sorted and unique")
    if errors and not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (repeats, warmups, bootstrap_samples)
    ):
        return errors

    if not allow_short:
        _validate_formal_contract(raw, errors)

    tags: set[str] = set()
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            errors.append(f"point[{index}] is not an object")
            continue
        if point.get("scope") not in {"decode", "prefill"}:
            errors.append(f"point[{index}] has invalid scope")
        if not isinstance(point.get("tag"), str) or not point["tag"]:
            errors.append(f"point[{index}] has invalid tag")
        elif point["tag"] in tags:
            errors.append(f"duplicate point tag {point['tag']}")
        else:
            tags.add(point["tag"])
        for field in ("batch", "seq", "gen"):
            if (
                not isinstance(point.get(field), int)
                or isinstance(point.get(field), bool)
                or point[field] <= 0
            ):
                errors.append(f"point[{index}] has invalid {field}")

    samples = raw.get("samples")
    if not isinstance(samples, list):
        return errors + ["raw samples are absent"]
    expected_count = len(points) * repeats * len(RUNGS)
    if len(samples) != expected_count:
        errors.append(f"raw sample count {len(samples)} != {expected_count}")

    warmup_records = raw.get("warmup_records")
    if not isinstance(warmup_records, list) or len(warmup_records) != len(points) * warmups:
        errors.append("warmup Latin-order record count mismatch")
    else:
        warmup_cursor = 0
        for point_index, point in enumerate(points):
            for warmup in range(warmups):
                record = warmup_records[warmup_cursor]
                warmup_cursor += 1
                order = ORDER_CYCLE[warmup % len(ORDER_CYCLE)]
                expected = {
                    "point_index": point_index,
                    "tag": point.get("tag"),
                    "warmup": warmup,
                    "epoch": 100_000 + point_index * 10_000 + warmup,
                    "order_pattern": ">".join(order),
                    "rung_sequence": list(order),
                    "off_grid_full_output_match": True,
                }
                if not isinstance(record, dict) or any(
                    record.get(field) != value for field, value in expected.items()
                ):
                    errors.append(
                        f"point={point.get('tag')} warmup={warmup}: Latin record mismatch"
                    )

    cursor = 0
    invocation = 0
    for point_index, point in enumerate(points):
        invocation += warmups * len(RUNGS)
        for repeat in range(repeats):
            epoch = 1_000_000 + point_index * 100_000 + repeat
            expected_poison = poison_digest(point, epoch)
            group: dict[str, dict[str, Any]] = {}
            timing_order = ORDER_CYCLE[repeat % len(ORDER_CYCLE)]
            order_pattern = ">".join(timing_order)
            for order_index, rung in enumerate(timing_order):
                rung_index = RUNGS.index(rung)
                if cursor >= len(samples):
                    break
                sample = samples[cursor]
                cursor += 1
                invocation += 1
                prefix = f"point={point.get('tag')} repeat={repeat} rung={rung}"
                if not isinstance(sample, dict):
                    errors.append(f"{prefix}: sample is not an object")
                    continue
                group[rung] = sample
                expected_fields = {
                    "point_index": point_index,
                    "tag": point.get("tag"),
                    "batch": point.get("batch"),
                    "seq": point.get("seq"),
                    "gen": point.get("gen"),
                    "scope": point.get("scope"),
                    "repeat": repeat,
                    "rung_index": rung_index,
                    "rung": rung,
                    "order_index": order_index,
                    "order_pattern": order_pattern,
                    "epoch": epoch,
                    "invocation": invocation,
                    "token_count": point.get("batch") * point.get("gen"),
                }
                for field, expected in expected_fields.items():
                    if sample.get(field) != expected:
                        errors.append(f"{prefix}: {field} mismatch")
                latency = sample.get("elapsed_s")
                if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
                    errors.append(f"{prefix}: invalid elapsed_s")
                if sample.get("worker_pids") != cohort:
                    errors.append(f"{prefix}: worker cohort changed")
                if sample.get("prompt_digest") != expected_poison:
                    errors.append(f"{prefix}: poison prompt digest mismatch")
                output = sample.get("output")
                observed_nonfinite = _validate_output(
                    output,
                    point,
                    prefix,
                    errors,
                    allow_nonfinite=rung == "ceiling",
                )
                if sample.get("nonfinite_logprob") is not observed_nonfinite:
                    errors.append(f"{prefix}: nonfinite_logprob flag mismatch")
                if isinstance(output, list):
                    if sample.get("output_digest") != digest_json(output):
                        errors.append(f"{prefix}: output digest mismatch")
                    try:
                        expected_token_digest = token_digest(output)
                    except (KeyError, TypeError):
                        expected_token_digest = None
                    if sample.get("token_digest") != expected_token_digest:
                        errors.append(f"{prefix}: token digest mismatch")
            if len(group) == len(RUNGS):
                if group["pdl_off"].get("output_digest") != group["pdl_grid"].get("output_digest"):
                    errors.append(
                        f"point={point.get('tag')} repeat={repeat}: off/grid exact "
                        "token+hex-logprob output differs"
                    )
                if len({sample.get("prompt_digest") for sample in group.values()}) != 1:
                    errors.append(
                        f"point={point.get('tag')} repeat={repeat}: rung poison differs"
                    )
                if group["pdl_off"].get("nonfinite_logprob") is not False:
                    errors.append(
                        f"point={point.get('tag')} repeat={repeat}: off logprob non-finite"
                    )
                if group["pdl_grid"].get("nonfinite_logprob") is not False:
                    errors.append(
                        f"point={point.get('tag')} repeat={repeat}: grid logprob non-finite"
                    )
    expected_invocations = len(points) * (warmups + repeats) * len(RUNGS)
    if raw.get("invocations") != expected_invocations or invocation != expected_invocations:
        errors.append("invocation count/order does not include exact warmup+sample triplets")

    validations = raw.get("validations")
    sample_groups: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for sample in samples:
        if (
            isinstance(sample, dict)
            and isinstance(sample.get("point_index"), int)
            and isinstance(sample.get("repeat"), int)
            and sample.get("rung") in RUNGS
        ):
            sample_groups.setdefault(
                (sample["point_index"], sample["repeat"]), {}
            )[sample["rung"]] = sample
    if not isinstance(validations, list) or len(validations) != len(points) * repeats:
        errors.append("raw correctness validation count mismatch")
    else:
        validation_cursor = 0
        for point_index, point in enumerate(points):
            for repeat_index in range(repeats):
                validation = validations[validation_cursor]
                validation_cursor += 1
                prefix = f"point={point.get('tag')} repeat={repeat_index}"
                if not isinstance(validation, dict):
                    errors.append(f"{prefix}: raw validation is not an object")
                    continue
                order = ORDER_CYCLE[repeat_index % len(ORDER_CYCLE)]
                expected = {
                    "point_index": point_index,
                    "tag": point.get("tag"),
                    "repeat": repeat_index,
                    "epoch": 1_000_000 + point_index * 100_000 + repeat_index,
                    "order_pattern": ">".join(order),
                    "rung_sequence": list(order),
                }
                for field, value in expected.items():
                    if validation.get(field) != value:
                        errors.append(f"{prefix}: validation {field} mismatch")
                if validation.get("off_grid_token_match") is not True:
                    errors.append(f"{prefix}: off/grid token validation failed")
                if validation.get("off_grid_full_output_match") is not True:
                    errors.append(f"{prefix}: off/grid logprob validation failed")
                if validation.get("ceiling_correctness") != "unsafe_not_validated":
                    errors.append(f"{prefix}: ceiling labeled correct")
                if validation.get("ceiling_verified") is not False:
                    errors.append(f"{prefix}: ceiling verified flag must be false")
                if validation.get("ceiling_unsafe") is not True:
                    errors.append(f"{prefix}: ceiling unsafe flag must be true")
                for field in (
                    "ceiling_token_differs",
                    "ceiling_full_output_differs",
                    "ceiling_nonfinite_logprob",
                    "ceiling_wrongness_evidence",
                ):
                    if not isinstance(validation.get(field), bool):
                        errors.append(f"{prefix}: {field} flag is absent")
                group = sample_groups.get((point_index, repeat_index), {})
                if set(group) == set(RUNGS):
                    token_differs = group["pdl_off"].get(
                        "token_digest"
                    ) != group["ceiling"].get("token_digest")
                    full_differs = group["pdl_off"].get(
                        "output_digest"
                    ) != group["ceiling"].get("output_digest")
                    ceiling_nonfinite = group["ceiling"].get(
                        "nonfinite_logprob"
                    )
                    wrongness = bool(
                        token_differs or full_differs or ceiling_nonfinite
                    )
                    expected_ceiling = {
                        "ceiling_token_differs": token_differs,
                        "ceiling_full_output_differs": full_differs,
                        "ceiling_nonfinite_logprob": ceiling_nonfinite,
                        "ceiling_wrongness_evidence": wrongness,
                    }
                    for field, value in expected_ceiling.items():
                        if validation.get(field) is not value:
                            errors.append(f"{prefix}: {field} mismatch")
                    if not wrongness:
                        errors.append(f"{prefix}: no ceiling wrongness evidence")

    probes = raw.get("build_probes")
    expected_switch_bindings: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(probes, dict) or set(probes) != set(RUNGS):
        errors.append("build probes do not contain exactly three rungs")
    else:
        for rung, workers in probes.items():
            expected_switch_bindings[rung] = _expected_switch_bindings(
                workers, rung, rung, errors
            )
            if not isinstance(workers, list):
                continue
            pids = sorted(
                worker.get("pid")
                for worker in workers
                if isinstance(worker, dict)
                and isinstance(worker.get("pid"), int)
                and not isinstance(worker.get("pid"), bool)
            )
            if pids != cohort:
                errors.append(f"{rung}: build probe worker cohort mismatch")
        for field in (
            "compiled_callable_fingerprint",
            "graph_dictionary_id",
            "graph_dictionary_fingerprint",
            "batch_graph_fingerprint",
            "activation_fingerprint",
        ):
            identities = {
                binding.get(field)
                for bindings in expected_switch_bindings.values()
                for binding in bindings
            }
            if len(identities) != len(RUNGS) * len(cohort):
                errors.append(
                    f"retained variants do not have distinct active {field} values"
                )

    if expected_switch_bindings:
        for record in warmup_records if isinstance(warmup_records, list) else []:
            if not isinstance(record, dict):
                continue
            bindings = record.get("switch_bindings")
            if not isinstance(bindings, dict) or set(bindings) != set(RUNGS):
                errors.append("warmup switch bindings do not contain exactly three rungs")
                continue
            for rung in RUNGS:
                if bindings.get(rung) != expected_switch_bindings.get(rung):
                    errors.append(f"warmup active variant binding mismatch: {rung}")
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("rung") not in RUNGS:
                continue
            rung = sample["rung"]
            if sample.get("switch_bindings") != expected_switch_bindings.get(rung):
                errors.append(
                    f"sample active variant binding mismatch: "
                    f"tag={sample.get('tag')} repeat={sample.get('repeat')} rung={rung}"
                )
    proof = raw.get("proof_records")
    if not isinstance(proof, list) or len(proof) != len(RUNGS):
        errors.append("proof records do not preserve off/grid/ceiling order")
    elif [item.get("rung") for item in proof if isinstance(item, dict)] != list(RUNGS):
        errors.append("proof records do not preserve off/grid/ceiling order")
    else:
        formal = raw.get("formal_manifest", {})
        proof_point = formal.get("proof_point") if isinstance(formal, dict) else None
        for index, (rung, record) in enumerate(zip(RUNGS, proof)):
            prefix = f"proof rung={rung}"
            if not isinstance(record, dict):
                errors.append(f"{prefix}: record is not an object")
                continue
            elapsed = record.get("elapsed_s_diagnostic")
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(elapsed)
                or elapsed <= 0
            ):
                errors.append(f"{prefix}: invalid diagnostic latency")
            output_hash = record.get("output_digest")
            if not isinstance(output_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", output_hash
            ):
                errors.append(f"{prefix}: output digest is invalid")
            if isinstance(proof_point, dict) and record.get(
                "prompt_digest"
            ) != poison_digest(proof_point, 10_000 + index):
                errors.append(f"{prefix}: proof poison digest mismatch")
            workers = record.get("worker_probes")
            worker_pids = (
                sorted(
                    worker.get("pid")
                    for worker in workers
                    if isinstance(worker, dict)
                    and isinstance(worker.get("pid"), int)
                    and not isinstance(worker.get("pid"), bool)
                )
                if isinstance(workers, list)
                else []
            )
            if worker_pids != cohort:
                errors.append(f"{prefix}: worker cohort mismatch")
            if record.get("switch_bindings") != expected_switch_bindings.get(rung):
                errors.append(f"{prefix}: active variant binding mismatch")
            proof_nonfinite = record.get("nonfinite_logprob")
            if not isinstance(proof_nonfinite, bool):
                errors.append(f"{prefix}: nonfinite_logprob flag is absent")
            elif rung != "ceiling" and proof_nonfinite:
                errors.append(f"{prefix}: safe proof logprob is non-finite")
            if rung == "ceiling":
                if record.get("ceiling_verified") is not False:
                    errors.append("proof ceiling verified flag must be false")
                if record.get("ceiling_unsafe") is not True:
                    errors.append("proof ceiling unsafe flag must be true")

    raw_runtimes = raw.get("runtimes")
    runtime_hashes = raw.get("runtime_candidate_sha256")
    if not isinstance(raw_runtimes, dict) or set(raw_runtimes) != set(RUNGS):
        errors.append("raw runtimes do not contain exactly three rungs")
    if not isinstance(runtime_hashes, dict) or set(runtime_hashes) != set(RUNGS):
        errors.append("raw runtime candidate hashes do not contain exactly three rungs")
    if isinstance(raw_runtimes, dict) and isinstance(runtime_hashes, dict):
        for rung in RUNGS:
            runtime = raw_runtimes.get(rung)
            expected_hash = runtime_hashes.get(rung)
            if not isinstance(runtime, dict):
                errors.append(f"raw {rung} runtime is not an object")
            elif expected_hash != digest_json(runtime):
                errors.append(f"raw {rung} runtime candidate hash mismatch")
    return errors


def analyze(raw: dict[str, Any]) -> dict[str, Any]:
    bootstrap_samples = raw["bootstrap_samples"]
    point_results: list[dict[str, Any]] = []
    for point_index, point in enumerate(raw["points"]):
        by_rung: dict[str, list[float]] = {rung: [] for rung in RUNGS}
        by_position: dict[str, dict[int, list[float]]] = {
            rung: {position: [] for position in range(len(RUNGS))}
            for rung in RUNGS
        }
        ceiling_differs = 0
        ceiling_nonfinite = 0
        for sample in raw["samples"]:
            if sample["point_index"] == point_index:
                latency = float(sample["elapsed_s"])
                by_rung[sample["rung"]].append(latency)
                by_position[sample["rung"]][sample["order_index"]].append(latency)
                if sample["rung"] == "ceiling" and sample["nonfinite_logprob"]:
                    ceiling_nonfinite += 1
        for validation in raw["validations"]:
            if (
                validation["point_index"] == point_index
                and validation["ceiling_full_output_differs"]
            ):
                ceiling_differs += 1
        rung_stats: dict[str, Any] = {}
        for rung_index, rung in enumerate(RUNGS):
            values = by_rung[rung]
            median = statistics.median(values)
            rung_stats[rung] = {
                "samples": len(values),
                "median_latency_s": median,
                "median_latency_bootstrap_95_ci_s": bootstrap_ci(
                    values,
                    bootstrap_samples,
                    seed=0x4A00 + point_index * 31 + rung_index,
                ),
                "mean_latency_s": statistics.fmean(values),
                "min_latency_s": min(values),
                "max_latency_s": max(values),
                "median_output_tokens_per_s": point["batch"] * point["gen"] / median,
                "latin_position_counts": [
                    len(by_position[rung][position])
                    for position in range(len(RUNGS))
                ],
                "latin_position_median_latency_s": [
                    (
                        statistics.median(by_position[rung][position])
                        if by_position[rung][position]
                        else None
                    )
                    for position in range(len(RUNGS))
                ],
                "correctness": (
                    "exact_match_to_off" if rung == "pdl_grid" else (
                        "reference" if rung == "pdl_off" else "unsafe_not_validated"
                    )
                ),
            }
        off, grid, ceiling = (by_rung[rung] for rung in RUNGS)
        off_median, grid_median, ceiling_median = (
            statistics.median(values) for values in (off, grid, ceiling)
        )
        point_results.append(
            {
                **point,
                "classification": (
                    "headline_full_decode"
                    if point["scope"] == "decode"
                    else PREFILL_MODE
                ),
                "rungs": rung_stats,
                "grid_vs_off_speedup_pct": 100.0 * (off_median - grid_median) / off_median,
                "grid_vs_off_bootstrap_95_ci_pct": paired_ratio_ci(
                    off, grid, bootstrap_samples, 0x5100 + point_index
                ),
                "wait_removal_headroom_pct": 100.0
                * (grid_median - ceiling_median)
                / grid_median,
                "wait_removal_headroom_bootstrap_95_ci_pct": paired_ratio_ci(
                    grid, ceiling, bootstrap_samples, 0x5200 + point_index
                ),
                "off_to_ceiling_pct": 100.0
                * (off_median - ceiling_median)
                / off_median,
                "ceiling_output_differs_repeats": ceiling_differs,
                "ceiling_nonfinite_logprob_repeats": ceiling_nonfinite,
                "ceiling_correctness": "unsafe_not_validated",
            }
        )
    return {
        "schema": "tier4.analysis.v2",
        "status": "ok",
        "admissible": True,
        "triplet_id": raw["triplet_id"],
        "triplet_mode": raw["triplet_mode"],
        "graph_mode": raw["resolved_graph_mode"],
        "pdl_scope": raw["pdl_scope"],
        "prefill_mode": raw["prefill_mode"],
        "repeats_per_rung": raw["repeats"],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence": 0.95,
        "semantic_rungs": list(RUNGS),
        "proof_rung_order": list(RUNGS),
        "timing_order_cycle": [list(order) for order in ORDER_CYCLE],
        "ceiling_correctness": "unsafe_not_validated",
        "points": point_results,
        "headline_decode": [
            item for item in point_results if item["scope"] == "decode"
        ],
        "production_mixed_mode_prefill": [
            item for item in point_results if item["scope"] == "prefill"
        ],
    }


def _read_json_object(
    path: Path, label: str, errors: list[str]
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        errors.append(f"missing {label}: {path}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label} {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} is not an object: {path}")
        return None
    return value


def _nsys_report_path(results: Path) -> Path:
    return results.parent.parent / "profiles" / f"{results.name}.nsys-rep"


def verify_candidate(results: Path, diagnostic: bool) -> list[str]:
    """Recompute one formal or diagnostic seal from the underlying files."""
    results = results.resolve()
    errors: list[str] = []
    for blocker in (
        "finalize_in_progress.json",
        "finalize_failure.json",
        "driver_error.json",
        "warmup_output_mismatch.json",
        "sample_output_mismatch.json",
    ):
        if (results / blocker).exists():
            errors.append(f"blocking failure artifact is present: {blocker}")

    root = results.parent.parent
    seal_path = results / (
        "diagnostic_validation.json" if diagnostic else "admission.json"
    )
    raw_path = results / "raw_triplet.json"
    analysis_path = results / "analysis.json"
    evidence_validation_path = results / "evidence_validation.json"
    sqlite_path = results / "profile.sqlite"
    report_path = _nsys_report_path(results)
    launch_path = results / "launch_manifest_snapshot.json"
    formal_snapshot_path = results / "formal_manifest_snapshot.json"
    identity_snapshot_path = results / "model_identity_snapshot.json"
    root_identity_path = root / "model_identity.json"
    root_manifest_path = root / "manifest.json"

    seal_label = "diagnostic validation" if diagnostic else "admission"
    seal = _read_json_object(seal_path, seal_label, errors)
    raw = _read_json_object(raw_path, "raw candidate", errors)
    analysis = _read_json_object(analysis_path, "analysis", errors)
    stored_evidence = _read_json_object(
        evidence_validation_path, "evidence validation", errors
    )
    launch = _read_json_object(launch_path, "launch manifest snapshot", errors)
    formal_snapshot = _read_json_object(
        formal_snapshot_path, "formal manifest snapshot", errors
    )
    identity_snapshot = _read_json_object(
        identity_snapshot_path, "model identity snapshot", errors
    )
    root_identity = _read_json_object(
        root_identity_path, "root model identity", errors
    )
    root_manifest = _read_json_object(
        root_manifest_path, "immutable root manifest", errors
    )
    for path, label in ((sqlite_path, "Nsight SQLite"), (report_path, "Nsight report")):
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing or empty {label}: {path}")

    if seal is not None:
        expected_schema = (
            "tier4.diagnostic.validation.v3"
            if diagnostic
            else "tier4.admission.v2"
        )
        if (
            seal.get("schema") != expected_schema
            or seal.get("status") != "ok"
            or seal.get("admissible") is not (not diagnostic)
            or (diagnostic and seal.get("diagnostic_only") is not True)
        ):
            errors.append(f"{seal_label} status/schema contract mismatch")
    if raw is not None:
        errors.extend(
            f"raw: {error}" for error in validate_raw(raw, diagnostic)
        )
        if raw.get("allow_short") is not diagnostic:
            errors.append("raw allow_short does not match verification mode")

    root_identity_hash = (
        sha256_file(root_identity_path) if root_identity_path.is_file() else None
    )
    root_manifest_hash = (
        sha256_file(root_manifest_path) if root_manifest_path.is_file() else None
    )
    snapshot_identity_hash = (
        sha256_file(identity_snapshot_path)
        if identity_snapshot_path.is_file()
        else None
    )
    sqlite_hash = sha256_file(sqlite_path) if sqlite_path.is_file() else None
    report_hash = sha256_file(report_path) if report_path.is_file() else None

    if root_identity is not None and root_manifest is not None:
        model_name = root_manifest.get("model")
        kv_strategy = root_manifest.get("kv_strategy")
        if not isinstance(model_name, str) or not isinstance(kv_strategy, dict):
            errors.append("root manifest lacks model/KV contract")
        elif root_identity_hash is not None:
            model_path = Path(model_name).resolve()
            errors.extend(
                f"model identity: {error}"
                for error in verify_model_identity(root_identity, model_path)
            )
            candidate = immutable_contract(
                root,
                model_path,
                root_identity,
                root_identity_hash,
                kv_strategy.get("kv_offloading_size_gib"),
            )
            difference = first_difference(candidate, root_manifest)
            if difference:
                errors.append(f"immutable root contract differs: {difference}")
    if root_identity is not None and identity_snapshot is not None:
        if identity_snapshot != root_identity:
            errors.append("cohort model identity snapshot differs from root identity")

    if raw is not None:
        root_errors, current_manifest, _manifest_path = validate_root_manifest(
            raw, results
        )
        errors.extend(f"root: {error}" for error in root_errors)
        errors.extend(
            f"launch: {error}"
            for error in validate_launch_manifest_snapshot(
                raw, results, current_manifest
            )
        )
        if launch is not None and root_manifest is not None and launch != root_manifest:
            errors.append("launch manifest snapshot differs from immutable root")
        if (
            formal_snapshot is not None
            and root_manifest is not None
            and formal_snapshot != root_manifest
        ):
            errors.append("formal manifest snapshot differs from immutable root")

    if seal is not None:
        file_hash_checks = {
            "raw_triplet_sha256": raw_path,
            "analysis_sha256": analysis_path,
            "evidence_validation_sha256": evidence_validation_path,
            "nsys_sqlite_sha256": sqlite_path,
            "nsys_report_sha256": report_path,
            "model_identity_snapshot_sha256": identity_snapshot_path,
            "model_identity_manifest_sha256": root_identity_path,
            "formal_root_manifest_sha256": launch_path,
            "formal_manifest_snapshot_sha256": formal_snapshot_path,
            "formal_root_manifest_observed_sha256": root_manifest_path,
        }
        for field, path in file_hash_checks.items():
            observed = sha256_file(path) if path.is_file() else None
            if seal.get(field) != observed:
                errors.append(f"{seal_label}/file hash mismatch: {field}")
        if seal.get("formal_root_manifest_sha256") != root_manifest_hash:
            errors.append("launch/root immutable manifest hash mismatch")
        if seal.get("model_identity_manifest_sha256") != root_identity_hash:
            errors.append(f"{seal_label}/root model identity hash mismatch")
        if seal.get("model_identity_snapshot_sha256") != snapshot_identity_hash:
            errors.append(f"{seal_label}/cohort model identity hash mismatch")
        if seal.get("nsys_sqlite_sha256") != sqlite_hash:
            errors.append(f"{seal_label}/profile SQLite hash mismatch")
        if seal.get("nsys_report_sha256") != report_hash:
            errors.append(f"{seal_label}/Nsight report hash mismatch")

    if raw is not None and seal is not None:
        for field in (
            "triplet_id",
            "model_fingerprint",
            "model_identity_manifest_sha256",
            "model_identity_snapshot_sha256",
            "formal_root_manifest_sha256",
        ):
            if seal.get(field) != raw.get(field):
                errors.append(f"{seal_label}/raw field mismatch: {field}")
        if raw.get("formal_root_manifest_sha256") != root_manifest_hash:
            errors.append("raw candidate is not bound to current immutable root")

    if raw is not None and analysis is not None:
        expected_analysis = analyze(raw)
        if diagnostic:
            expected_analysis.update(
                {
                    "status": "diagnostic",
                    "admissible": False,
                    "diagnostic_only": True,
                }
            )
        for field, value in expected_analysis.items():
            if analysis.get(field) != value:
                errors.append(f"analysis differs from recomputed raw metric: {field}")
        analysis_hash_fields = {
            "raw_triplet_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
            "nsys_sqlite_sha256": sqlite_hash,
            "nsys_report_sha256": report_hash,
            "model_identity_manifest_sha256": root_identity_hash,
            "model_identity_snapshot_sha256": snapshot_identity_hash,
            "formal_root_manifest_sha256": root_manifest_hash,
        }
        for field, value in analysis_hash_fields.items():
            if analysis.get(field) != value:
                errors.append(f"analysis/file hash mismatch: {field}")

    if raw is not None:
        runtime_hashes = seal.get("runtime_sha256") if seal else None
        if not isinstance(runtime_hashes, dict) or set(runtime_hashes) != set(RUNGS):
            errors.append(f"{seal_label} finalized runtime hashes are absent")
            runtime_hashes = {}
        for rung in RUNGS:
            runtime_path = results / "evidence" / rung / "runtime.json"
            runtime = _read_json_object(runtime_path, f"{rung} runtime", errors)
            observed_hash = sha256_file(runtime_path) if runtime_path.is_file() else None
            if runtime_hashes.get(rung) != observed_hash:
                errors.append(f"{seal_label}/{rung} runtime hash mismatch")
            candidate_runtime = raw.get("runtimes", {}).get(rung)
            candidate_hash = raw.get("runtime_candidate_sha256", {}).get(rung)
            if isinstance(candidate_runtime, dict) and runtime is not None:
                expected_runtime = copy.deepcopy(candidate_runtime)
                expected_runtime.update(
                    {
                        "graph_execution_proof": "nsys_cuda_graph_node_nvtx_window",
                        "nsys_sqlite": os.path.relpath(
                            sqlite_path, runtime_path.parent
                        ),
                        "nsys_sqlite_sha256": sqlite_hash,
                        "nsys_nvtx_label": (
                            f"TIER4_PROOF|rung={rung}|triplet={raw.get('triplet_id')}"
                        ),
                        "candidate_runtime_sha256": candidate_hash,
                    }
                )
                if runtime != expected_runtime:
                    errors.append(f"finalized {rung} runtime differs from raw binding")

        recomputed_evidence = validate_evidence(
            results / "evidence", raw.get("model_fingerprint")
        )
        if recomputed_evidence.get("status") != "ok":
            errors.extend(
                f"evidence: {error}"
                for error in recomputed_evidence.get("errors", [])
            )
        if (
            stored_evidence is not None
            and digest_json(stored_evidence) != digest_json(recomputed_evidence)
        ):
            errors.append("stored evidence validation differs from recomputation")
    return errors


def verify_admitted_candidate(results: Path) -> list[str]:
    """Recompute the entire formal admission before any resume-skip."""
    return verify_candidate(results, diagnostic=False)


def verify_diagnostic_candidate(results: Path) -> list[str]:
    """Recompute a short, explicitly non-admissible diagnostic chain."""
    return verify_candidate(results, diagnostic=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--nsys-sqlite", type=Path)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--verify-admission", action="store_true")
    parser.add_argument("--verify-diagnostic", action="store_true")
    args = parser.parse_args()
    results = args.results.resolve()
    if args.verify_admission or args.verify_diagnostic:
        if args.verify_admission and args.verify_diagnostic:
            parser.error("choose exactly one verification mode")
        if args.allow_short or args.nsys_sqlite is not None:
            parser.error("verification modes do not accept finalize-only options")
        try:
            verify_errors = (
                verify_admitted_candidate(results)
                if args.verify_admission
                else verify_diagnostic_candidate(results)
            )
        except Exception as exc:  # noqa: BLE001 - malformed artifacts fail closed
            verify_errors = [
                f"independent admission recomputation raised "
                f"{type(exc).__name__}: {exc}"
            ]
        if verify_errors:
            label = "VERIFY_ADMISSION" if args.verify_admission else "VERIFY_DIAGNOSTIC"
            print(f"{label} status=blocked", file=sys.stderr)
            for error in verify_errors:
                print(f"  - {error}", file=sys.stderr)
            return 3
        label = "VERIFY_ADMISSION" if args.verify_admission else "VERIFY_DIAGNOSTIC"
        print(f"{label} status=ok path={results}")
        return 0
    if args.nsys_sqlite is None:
        parser.error("finalization requires --nsys-sqlite")
    in_progress_path = results / "finalize_in_progress.json"
    atomic_json(
        in_progress_path,
        {
            "schema": "tier4.finalize.in_progress.v1",
            "status": "in_progress",
            "admissible": False,
            "note": "left in place on any interrupted or failed finalization",
        },
    )
    raw_path = results / "raw_triplet.json"
    try:
        raw = json.loads(raw_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        atomic_json(
            results / "finalize_failure.json",
            {
                "schema": "tier4.finalize.failure.v2",
                "status": "blocked",
                "admissible": False,
                "errors": [f"cannot read raw candidate: {exc}"],
            },
        )
        print(f"FINALIZE status=blocked cannot read raw candidate: {exc}", file=sys.stderr)
        return 3
    if not isinstance(raw, dict):
        atomic_json(
            results / "finalize_failure.json",
            {
                "schema": "tier4.finalize.failure.v2",
                "status": "blocked",
                "admissible": False,
                "errors": ["raw candidate is not an object"],
            },
        )
        print("FINALIZE status=blocked raw candidate is not an object", file=sys.stderr)
        return 3
    errors = validate_raw(raw, args.allow_short)
    identity_snapshot_path = results / "model_identity_snapshot.json"
    try:
        identity_snapshot = json.loads(identity_snapshot_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read model identity snapshot: {exc}")
    else:
        if raw.get("model_identity_snapshot_sha256") != sha256_file(
            identity_snapshot_path
        ):
            errors.append("raw model identity snapshot hash mismatch")
        if isinstance(identity_snapshot, dict):
            if identity_snapshot.get("model_fingerprint") != raw.get(
                "model_fingerprint"
            ):
                errors.append("model identity snapshot fingerprint mismatch")
            model_name = raw.get("model")
            if isinstance(model_name, str):
                errors.extend(
                    f"model identity: {error}"
                    for error in verify_model_identity(
                        identity_snapshot, Path(model_name).resolve()
                    )
                )
        else:
            errors.append("model identity snapshot is not an object")
    nsys_path = args.nsys_sqlite.resolve()
    if not nsys_path.is_file():
        errors.append(f"Nsight sqlite is absent: {nsys_path}")
    report_path = _nsys_report_path(results)
    root_bound = isinstance(raw.get("formal_root_manifest_sha256"), str)
    if (not args.allow_short or root_bound) and (
        not report_path.is_file() or report_path.stat().st_size <= 0
    ):
        errors.append(f"root-bound Nsight report is absent or empty: {report_path}")
    manifest_snapshot: dict[str, Any] | None = None
    manifest_path: Path | None = None
    if root_bound:
        manifest_errors, manifest_snapshot, manifest_path = validate_root_manifest(
            raw, results
        )
        errors.extend(manifest_errors)
        errors.extend(
            validate_launch_manifest_snapshot(raw, results, manifest_snapshot)
        )
    runtimes: dict[str, dict[str, Any]] = {}
    raw_runtimes = raw.get("runtimes")
    if not isinstance(raw_runtimes, dict):
        raw_runtimes = {}
    raw_runtime_hashes = raw.get("runtime_candidate_sha256")
    if not isinstance(raw_runtime_hashes, dict):
        raw_runtime_hashes = {}
    for rung in RUNGS:
        runtime_path = results / "evidence" / rung / "runtime.json"
        try:
            runtime = json.loads(runtime_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {rung} runtime before finalization: {exc}")
            continue
        if not isinstance(runtime, dict):
            errors.append(f"{rung} runtime is not an object")
            continue
        raw_runtime = raw_runtimes.get(rung)
        if runtime != raw_runtime:
            errors.append(f"{rung} on-disk runtime differs from raw runtime")
        raw_runtime_hash = raw_runtime_hashes.get(rung)
        if raw_runtime_hash != digest_json(runtime):
            errors.append(f"{rung} on-disk runtime candidate hash mismatch")
        for field in (
            "triplet_id",
            "cohort_id",
            "model_fingerprint",
            "model_identity_manifest_sha256",
            "model_identity_snapshot_sha256",
        ):
            if runtime.get(field) != raw.get(field):
                errors.append(f"{rung} runtime/raw {field} mismatch")
        runtimes[rung] = runtime
    if errors:
        failure = {
            "schema": "tier4.finalize.failure.v2",
            "status": "blocked",
            "admissible": False,
            "errors": errors,
        }
        atomic_json(results / "finalize_failure.json", failure)
        print("FINALIZE status=blocked", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 3

    if manifest_snapshot is not None and manifest_path is not None:
        snapshot_path = results / "formal_manifest_snapshot.json"
        atomic_json(snapshot_path, manifest_snapshot)

    nsys_hash = sha256_file(nsys_path)
    report_hash = sha256_file(report_path) if report_path.is_file() else None
    for rung in RUNGS:
        runtime_path = results / "evidence" / rung / "runtime.json"
        runtime = runtimes[rung]
        runtime["graph_execution_proof"] = "nsys_cuda_graph_node_nvtx_window"
        runtime["nsys_sqlite"] = os.path.relpath(nsys_path, runtime_path.parent)
        runtime["nsys_sqlite_sha256"] = nsys_hash
        runtime["nsys_nvtx_label"] = (
            f"TIER4_PROOF|rung={rung}|triplet={raw['triplet_id']}"
        )
        runtime["candidate_runtime_sha256"] = raw[
            "runtime_candidate_sha256"
        ][rung]
        atomic_json(runtime_path, runtime)

    evidence = validate_evidence(results / "evidence", raw["model_fingerprint"])
    atomic_json(results / "evidence_validation.json", evidence)
    if evidence["status"] != "ok":
        failure = {
            "schema": "tier4.finalize.failure.v2",
            "status": "blocked",
            "admissible": False,
            "errors": evidence["errors"],
        }
        atomic_json(results / "finalize_failure.json", failure)
        print("FINALIZE status=blocked evidence validation failed", file=sys.stderr)
        for error in evidence["errors"]:
            print(f"  - {error}", file=sys.stderr)
        return 3

    analysis = analyze(raw)
    if args.allow_short:
        analysis["status"] = "diagnostic"
        analysis["admissible"] = False
        analysis["diagnostic_only"] = True
    analysis["raw_triplet_sha256"] = sha256_file(raw_path)
    analysis["nsys_sqlite_sha256"] = nsys_hash
    analysis["nsys_report_sha256"] = report_hash
    analysis["model_fingerprint"] = raw["model_fingerprint"]
    analysis["model_identity_manifest_sha256"] = raw[
        "model_identity_manifest_sha256"
    ]
    analysis["model_identity_snapshot_sha256"] = raw[
        "model_identity_snapshot_sha256"
    ]
    analysis["formal_root_manifest_sha256"] = raw.get(
        "formal_root_manifest_sha256"
    )
    if manifest_snapshot is not None and manifest_path is not None:
        analysis["formal_manifest_snapshot_sha256"] = sha256_file(
            results / "formal_manifest_snapshot.json"
        )
        analysis["formal_root_manifest_observed_sha256"] = sha256_file(
            manifest_path
        )
    atomic_json(results / "analysis.json", analysis)
    if args.allow_short:
        diagnostic = {
            "schema": "tier4.diagnostic.validation.v3",
            "status": "ok",
            "admissible": False,
            "diagnostic_only": True,
            "triplet_id": raw["triplet_id"],
            "raw_triplet_sha256": analysis["raw_triplet_sha256"],
            "analysis_sha256": sha256_file(results / "analysis.json"),
            "evidence_validation_sha256": sha256_file(
                results / "evidence_validation.json"
            ),
            "nsys_sqlite_sha256": nsys_hash,
            "nsys_report_sha256": report_hash,
            "model_fingerprint": raw["model_fingerprint"],
            "model_identity_manifest_sha256": raw[
                "model_identity_manifest_sha256"
            ],
            "model_identity_snapshot_sha256": raw[
                "model_identity_snapshot_sha256"
            ],
            "formal_root_manifest_sha256": raw.get(
                "formal_root_manifest_sha256"
            ),
            "runtime_sha256": {
                rung: sha256_file(results / "evidence" / rung / "runtime.json")
                for rung in RUNGS
            },
        }
        if manifest_snapshot is not None and manifest_path is not None:
            diagnostic["formal_manifest_snapshot_sha256"] = sha256_file(
                results / "formal_manifest_snapshot.json"
            )
            diagnostic["formal_root_manifest_observed_sha256"] = sha256_file(
                manifest_path
            )
        atomic_json(results / "diagnostic_validation.json", diagnostic)
        in_progress_path.unlink()
        print(
            f"FINALIZE status=ok admissible=0 diagnostic_only=1 "
            f"triplet_id={raw['triplet_id']} points={len(raw['points'])} "
            f"repeats={raw['repeats']}"
        )
        return 0
    admission = {
        "schema": "tier4.admission.v2",
        "status": "ok",
        "admissible": True,
        "triplet_id": raw["triplet_id"],
        "raw_triplet_sha256": analysis["raw_triplet_sha256"],
        "analysis_sha256": sha256_file(results / "analysis.json"),
        "evidence_validation_sha256": sha256_file(
            results / "evidence_validation.json"
        ),
        "nsys_sqlite_sha256": nsys_hash,
        "nsys_report_sha256": report_hash,
        "model_fingerprint": raw["model_fingerprint"],
        "model_identity_manifest_sha256": raw[
            "model_identity_manifest_sha256"
        ],
        "model_identity_snapshot_sha256": raw[
            "model_identity_snapshot_sha256"
        ],
        "formal_root_manifest_sha256": raw["formal_root_manifest_sha256"],
        "runtime_sha256": {
            rung: sha256_file(results / "evidence" / rung / "runtime.json")
            for rung in RUNGS
        },
    }
    if manifest_snapshot is not None and manifest_path is not None:
        admission["formal_manifest_snapshot_sha256"] = sha256_file(
            results / "formal_manifest_snapshot.json"
        )
        admission["formal_root_manifest_observed_sha256"] = sha256_file(
            manifest_path
        )
    atomic_json(results / "admission.json", admission)
    in_progress_path.unlink()
    print(
        f"FINALIZE status=ok admissible=1 triplet_id={raw['triplet_id']} "
        f"points={len(raw['points'])} repeats={raw['repeats']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
