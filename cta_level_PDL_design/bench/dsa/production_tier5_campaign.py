#!/usr/bin/env python3
"""Freeze, validate, seal, resume, and aggregate Tier-5 row fragments.

This module is deliberately CPU-only.  It never imports CUDA libraries.  A
fragment is never accepted as workload timing on its own; only a fresh
revalidation of the exact formal 26-row inventory may set that field to one.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import production_tier5 as harness
import validate_production_tier5 as validator


SCHEMA = 1
FORMAL_MODELS = tuple(harness.MODEL_SPECS)
FORMAL_MATRIX = harness.expected_matrix(
    FORMAL_MODELS, harness.FORMAL_SEQS, harness.FORMAL_WORKLOADS
)
FORMAL_SAMPLE_COUNT = 2542
FORMAL_CORRECTNESS_ROW_COUNT = 26
FORMAL_SUMMARY_COUNT = 122
ROW_DIR_RE = re.compile(r"^[0-9]{3}_[A-Za-z0-9_.-]+$")

FRAGMENT_ARTIFACTS = (
    "manifest.json",
    "samples.jsonl",
    "correctness.json",
    "result.json",
    "terminal_status.json",
    "harness.log",
    "runner.log",
    "gpu_identity.json",
    "gpu_exclusivity_lease.json",
    "gpu_pre.json",
    "gpu_post.json",
    "gpu_monitor.json",
    "gpu_observations.ndjson",
    "fragment_validation.json",
)
FORBIDDEN_FRAGMENT_NAMES = (
    "failure.json",
    "formal_rejection.json",
    "REJECTED.md",
    "segment_rejection.json",
    "production_candidate.done.json",
)
MANIFEST_FIELDS = {
    "schema", "kind", "created_unix_ns", "mode", "status",
    "accepted_timing", "accepted_timing_semantics", "accepted_workload_timing",
    "accepted_CTA_bracket", "measurement_emitted", "random_seed",
    "random_weights", "backend", "required_device_substring",
    "expected_gpu_uuid", "expected_gpu_index", "models", "seqs", "workloads",
    "shape_records", "expected_matrix", "execution_scope", "fragment", "moe",
    "chunking", "long_context_execution", "correctness_contract", "warmup",
    "repeats", "allow_short",
    "formal_statistics_requested", "experiment_contract", "api_contracts",
    "static_api_checks", "packages", "sources", "argv", "argv_sha256",
    "publication", "environment", "device",
}
DEVICE_FIELDS = {
    "query_performed", "name", "uuid", "uuid_source", "runtime_ordinal",
    "runtime_ordinal_zero", "compute_capability", "total_memory_bytes",
    "multi_processor_count", "driver_version_raw", "torch_cuda_version",
    "device_index_inside_visible_set", "cuda_visible_devices_selector",
    "process_pid", "process_start_ticks",
}


def canonical_subsequence(values: Sequence[Any], full: Sequence[Any]) -> bool:
    positions = [full.index(value) for value in values if value in full]
    return (
        len(positions) == len(values)
        and positions == sorted(set(positions))
    )


def row_directory_name(ordinal: int, row_id: str) -> str:
    return f"{ordinal:03d}_{row_id}"


def atomic_publish_directory(stage: Path, final: Path) -> None:
    """Linux no-clobber directory rename followed by parent durability."""
    stage = stage.absolute()
    final = final.absolute()
    if stage.parent != final.parent:
        raise ValueError("stage and final must be siblings")
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("stage must be a regular directory")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 unavailable; refusing non-atomic publish")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    rc = renameat2(
        at_fdcwd,
        os.fsencode(stage),
        at_fdcwd,
        os.fsencode(final),
        rename_noreplace,
    )
    if rc != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"publish target already exists: {final}")
        raise OSError(error, os.strerror(error), str(final))
    directory_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def controls_to_harness_argv(controls: dict[str, Any]) -> list[str]:
    values = [
        "--output-dir", "/nonexistent/tier5-contract-plan",
        "--backend", str(controls["backend"]),
        "--required-device-substring", str(controls["required_device_substring"]),
        "--models", ",".join(controls["models"]),
        "--seqs", ",".join(str(value) for value in controls["seqs"]),
        "--workloads", ",".join(controls["workloads"]),
        "--warmup", str(controls["warmup"]),
        "--repeats", str(controls["repeats"]),
    ]
    if controls["allow_short"]:
        values.append("--allow-short")
    values.extend(
        [
            "--seed", str(controls["seed"]),
            "--max-logits-mb", str(controls["max_logits_mb"]),
            "--max-query-chunk", str(controls["max_query_chunk"]),
            "--moe-experts", str(controls["moe_experts"]),
            "--moe-topk", str(controls["moe_topk"]),
            "--moe-tokens", str(controls["moe_tokens"]),
        ]
    )
    return values


def normalize_controls(raw: dict[str, Any]) -> dict[str, Any]:
    args = harness.parse_args(controls_to_harness_argv(raw))
    controls = {
        "backend": args.backend,
        "required_device_substring": args.required_device_substring,
        "models": list(args.models),
        "seqs": list(args.seqs),
        "workloads": list(args.workloads),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "allow_short": args.allow_short,
        "seed": args.seed,
        "max_logits_mb": args.max_logits_mb,
        "max_query_chunk": args.max_query_chunk,
        "moe_experts": args.moe_experts,
        "moe_topk": args.moe_topk,
        "moe_tokens": args.moe_tokens,
        "monitor_interval_ms": raw.get("monitor_interval_ms", 50),
        "query_timeout_ms": raw.get("query_timeout_ms", 2000),
    }
    if (
        not isinstance(controls["monitor_interval_ms"], int)
        or isinstance(controls["monitor_interval_ms"], bool)
        or not 10 <= controls["monitor_interval_ms"] <= 100
    ):
        raise ValueError("monitor_interval_ms must be an integer in 10..100")
    if (
        not isinstance(controls["query_timeout_ms"], int)
        or isinstance(controls["query_timeout_ms"], bool)
        or not 100 <= controls["query_timeout_ms"] <= 5000
    ):
        raise ValueError("query_timeout_ms must be an integer in 100..5000")
    if not canonical_subsequence(controls["models"], list(FORMAL_MODELS)):
        raise ValueError("models must be a canonical ordered subset")
    if not canonical_subsequence(controls["seqs"], list(harness.FORMAL_SEQS)):
        raise ValueError("contexts must be a canonical ordered subset")
    if not canonical_subsequence(
        controls["workloads"], list(harness.FORMAL_WORKLOADS)
    ):
        raise ValueError("workloads must be a canonical ordered subset")
    return controls


def fragment_argv_template(controls: dict[str, Any]) -> list[str]:
    values = [
        str(Path(sys.executable).resolve()),
        str(Path(harness.__file__).resolve()),
        "--output-dir", "${ROW_STAGE}",
        "--publish-target", "${ROW_FINAL}",
        "--runner-managed-stage",
        "--backend", controls["backend"],
        "--required-device-substring", controls["required_device_substring"],
        "--models", ",".join(controls["models"]),
        "--seqs", ",".join(str(value) for value in controls["seqs"]),
        "--workloads", ",".join(controls["workloads"]),
        "--warmup", str(controls["warmup"]),
        "--repeats", str(controls["repeats"]),
    ]
    if controls["allow_short"]:
        values.append("--allow-short")
    values.extend(
        [
            "--seed", str(controls["seed"]),
            "--max-logits-mb", str(controls["max_logits_mb"]),
            "--max-query-chunk", str(controls["max_query_chunk"]),
            "--moe-experts", str(controls["moe_experts"]),
            "--moe-topk", str(controls["moe_topk"]),
            "--moe-tokens", str(controls["moe_tokens"]),
            "--execute-gpu",
            "--expected-gpu-uuid", "${GPU_UUID}",
            "--expected-gpu-index", "${GPU_INDEX}",
            "--fragment-row-id", "${ROW_ID}",
            "--fragment-ordinal", "${ROW_ORDINAL}",
            "--campaign-contract-sha256", "${CONTRACT_SHA256}",
            "--campaign-fingerprint-sha256", "${FINGERPRINT_SHA256}",
            "--execution-segment-id", "${INVOCATION_UUID}",
        ]
    )
    return values


def build_campaign_contract(raw_controls: dict[str, Any]) -> dict[str, Any]:
    controls = normalize_controls(raw_controls)
    matrix = harness.expected_matrix(
        controls["models"], controls["seqs"], controls["workloads"]
    )
    formal = not controls["allow_short"]
    if formal and matrix != FORMAL_MATRIX:
        raise ValueError("formal campaign must be the exact ordered 26-row matrix")
    model_specs = [asdict(harness.MODEL_SPECS[key]) for key in controls["models"]]
    shapes = [
        harness.shape_record(
            harness.MODEL_SPECS[model],
            seq,
            controls["max_logits_mb"],
            controls["max_query_chunk"],
            controls["moe_experts"],
            controls["moe_tokens"],
        )
        for model in controls["models"]
        for seq in controls["seqs"]
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_contract",
        "status": "FROZEN",
        "campaign_mode": "formal" if formal else "nonformal_short",
        "formal": formal,
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "controls": controls,
        "controls_sha256": harness.canonical_json_sha(controls),
        "ordered_matrix": matrix,
        "row_count": len(matrix),
        "formal_full_ordered_matrix": FORMAL_MATRIX,
        "formal_full_matrix_sha256": harness.canonical_json_sha(FORMAL_MATRIX),
        "is_exact_formal_matrix": formal and matrix == FORMAL_MATRIX,
        "model_specs": model_specs,
        "shape_records": shapes,
        "fragment_argv_template": fragment_argv_template(controls),
        "fragment_argv_template_sha256": harness.canonical_json_sha(
            fragment_argv_template(controls)
        ),
        "long_context_execution": harness.long_context_execution_contract(),
        "correctness_contract": harness.correctness_contract(),
        "experiment_contract": harness.experiment_contract(),
        "api_contracts": list(harness.API_CONTRACTS),
        "static_api_checks": harness.static_api_checks(),
        "packages": harness.package_manifest(hash_binaries=True),
        "sources": harness.local_source_manifest(),
        "formula_authority": harness.correctness_contract()["formula_authority"],
    }
    payload["package_manifest_sha256"] = harness.canonical_json_sha(
        payload["packages"]
    )
    payload["source_manifest_sha256"] = harness.canonical_json_sha(
        payload["sources"]
    )
    payload["contract_sha256"] = harness.canonical_json_sha(payload)
    return payload


def validate_campaign_contract(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["contract_not_object"]
    digest = value.get("contract_sha256")
    body = dict(value)
    body.pop("contract_sha256", None)
    if digest != harness.canonical_json_sha(body):
        errors.append("contract_self_hash_mismatch")
    controls = value.get("controls")
    if not isinstance(controls, dict):
        errors.append("contract_controls_missing")
        return errors
    try:
        rebuilt = build_campaign_contract(controls)
    except (ValueError, SystemExit, KeyError, TypeError) as exc:
        errors.append(f"contract_rebuild:{type(exc).__name__}:{exc}")
        return errors
    if rebuilt != value:
        errors.append("contract_current_environment_or_semantics_drift")
    if value.get("formal") is True:
        if value.get("campaign_mode") != "formal":
            errors.append("formal_campaign_mode_drift")
        if value.get("ordered_matrix") != FORMAL_MATRIX:
            errors.append("formal_matrix_not_exact_26")
        if value.get("row_count") != FORMAL_CORRECTNESS_ROW_COUNT:
            errors.append("formal_row_count_not_26")
        if controls.get("allow_short") is not False:
            errors.append("formal_allow_short_forbidden")
    elif value.get("campaign_mode") != "nonformal_short":
        errors.append("short_campaign_not_labelled_nonformal")
    return errors


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_regular_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return load_json(path)


def load_and_validate_contract(path: Path) -> dict[str, Any]:
    value = load_regular_json(path, "campaign contract")
    errors = validate_campaign_contract(value)
    if errors:
        raise ValueError("invalid campaign contract: " + ",".join(errors))
    return value


def binding_body(contract: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_device_binding",
        "status": "FROZEN",
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "contract_sha256": contract["contract_sha256"],
        "package_manifest_sha256": contract["package_manifest_sha256"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "controls_sha256": contract["controls_sha256"],
        "target_gpu": target,
    }


def bind_campaign_device(
    contract: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    if (
        identity.get("schema") != SCHEMA
        or identity.get("kind") != "gpu_identity"
        or identity.get("status") != "PASS"
        or identity.get("errors") != []
    ):
        raise ValueError("GPU identity is not a clean PASS")
    target = identity.get("target_gpu")
    if not isinstance(target, dict):
        raise ValueError("GPU identity target is missing")
    uuid = harness.canonical_gpu_uuid(target.get("uuid", ""))
    index = target.get("index")
    name = target.get("name")
    if (
        uuid is None
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or not isinstance(name, str)
        or contract["controls"]["required_device_substring"] not in name
    ):
        raise ValueError("GPU identity target is malformed or wrong device")
    canonical_target = {"index": index, "uuid": uuid, "name": name}
    value = binding_body(contract, canonical_target)
    value["campaign_fingerprint_sha256"] = harness.canonical_json_sha(value)
    return value


def validate_campaign_binding(
    binding: Any, contract: dict[str, Any]
) -> list[str]:
    if not isinstance(binding, dict):
        return ["binding_not_object"]
    errors: list[str] = []
    target = binding.get("target_gpu")
    if not isinstance(target, dict):
        return ["binding_target_missing"]
    expected = binding_body(contract, target)
    fingerprint = binding.get("campaign_fingerprint_sha256")
    if fingerprint != harness.canonical_json_sha(expected):
        errors.append("campaign_fingerprint_mismatch")
    rebuilt = dict(expected)
    rebuilt["campaign_fingerprint_sha256"] = fingerprint
    if binding != rebuilt:
        errors.append("binding_field_drift")
    if harness.canonical_gpu_uuid(target.get("uuid", "")) != target.get("uuid"):
        errors.append("binding_uuid_noncanonical")
    return errors


def instantiate_fragment_argv(
    contract: dict[str, Any],
    binding: dict[str, Any],
    row: dict[str, Any],
    ordinal: int,
    execution_root: Path,
    publish_target: Path,
    invocation_uuid: str,
) -> list[str]:
    replacements = {
        "${ROW_STAGE}": str(execution_root.resolve()),
        "${ROW_FINAL}": str(publish_target.resolve()),
        "${GPU_UUID}": binding["target_gpu"]["uuid"],
        "${GPU_INDEX}": str(binding["target_gpu"]["index"]),
        "${ROW_ID}": row["row_id"],
        "${ROW_ORDINAL}": str(ordinal),
        "${CONTRACT_SHA256}": contract["contract_sha256"],
        "${FINGERPRINT_SHA256}": binding["campaign_fingerprint_sha256"],
        "${INVOCATION_UUID}": invocation_uuid,
    }
    return [replacements.get(value, value) for value in contract["fragment_argv_template"]]


def runtime_build_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    device = manifest.get("device", {})
    return {
        "packages_sha256": harness.canonical_json_sha(manifest.get("packages")),
        "sources_sha256": harness.canonical_json_sha(manifest.get("sources")),
        "static_api_checks_sha256": harness.canonical_json_sha(
            manifest.get("static_api_checks")
        ),
        "api_contracts_sha256": harness.canonical_json_sha(
            manifest.get("api_contracts")
        ),
        "device_build": {
            key: device.get(key)
            for key in (
                "name", "uuid", "compute_capability", "total_memory_bytes",
                "multi_processor_count", "driver_version_raw", "torch_cuda_version",
                "cuda_visible_devices_selector",
            )
        },
    }


def _fragment_evidence(
    root: Path, binding: dict[str, Any], errors: list[str]
) -> dict[str, Any] | None:
    try:
        monitor = load_json(root / "gpu_monitor.json")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"monitor_load:{type(exc).__name__}:{exc}")
        return None
    uuid = binding["target_gpu"]["uuid"]
    lock_path = f"/tmp/cta_pdl_gpu_{uuid}.lock"
    return {
        "expected_gpu_uuid": uuid,
        "expected_gpu_index": binding["target_gpu"]["index"],
        "global_lock_scope": "target_uuid",
        "global_lock_key_sha256": harness.sha256_bytes(uuid.encode("utf-8")),
        "global_lock_path_sha256": harness.sha256_bytes(lock_path.encode("utf-8")),
        "monitor_interval_ms": monitor.get("poll_interval_ms"),
        "query_timeout_ms": monitor.get("query_timeout_ms"),
    }


def _validate_fragment_result(
    root: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    samples: list[dict[str, Any]],
    correctness: Any,
    errors: list[str],
) -> None:
    result = validator.load_json(root / "result.json", errors)
    terminal = validator.load_json(root / "terminal_status.json", errors)
    fragment = manifest.get("fragment")
    validator.require(
        set(manifest) == MANIFEST_FIELDS,
        "fragment_manifest_top_level_field_drift",
        errors,
    )
    if result is not None:
        expected_result = {
            "schema": SCHEMA,
            "kind": "tier5_production_dsa_result",
            "status": "CANDIDATE",
            "accepted_timing": 0,
            "accepted_timing_semantics": "legacy_CTA_bracket_only",
            "accepted_workload_timing": 0,
            "accepted_CTA_bracket": 0,
            "measurement_emitted": True,
            "claim_scope": "production_kernel_characterization_only",
            "production_timing_candidate": True,
            "tier5_bracket_admitted": False,
            "formal_bracket_status": "PARTIAL",
            "headroom_defined": False,
            "headroom_pct": None,
            "manifest_sha256": harness.sha256_file(root / "manifest.json"),
            "samples_sha256": harness.sha256_file(root / "samples.jsonl"),
            "correctness_sha256": harness.sha256_file(root / "correctness.json"),
            "sample_count": len(samples),
            "summaries": harness.summarize_samples(
                samples, int(manifest["random_seed"])
            ),
            "execution_scope": "row_fragment",
            "fragment": fragment,
        }
        validator.require(
            result == expected_result,
            "fragment_result_exact_reconstruction_drift",
            errors,
        )
    if terminal is not None:
        validator.require(
            terminal
            == {
                "schema": SCHEMA,
                "status": "CANDIDATE",
                "accepted_timing": 0,
                "accepted_workload_timing": 0,
                "accepted_CTA_bracket": 0,
                "measurement_emitted": True,
                "result_sha256": harness.sha256_file(root / "result.json"),
                "execution_scope": "row_fragment",
                "fragment": fragment,
            },
            "fragment_terminal_exact_reconstruction_drift",
            errors,
        )


def validate_fragment(
    root: Path, contract: dict[str, Any], binding: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    root_was_symlink = root.is_symlink()
    root = root.resolve()
    errors: list[str] = []
    if root_was_symlink:
        errors.append("fragment_root_symlink")
    allowed_entries = set(FRAGMENT_ARTIFACTS) | {"fragment.done.json"}
    for name in FORBIDDEN_FRAGMENT_NAMES:
        if (root / name).exists():
            errors.append(f"forbidden_fragment_artifact:{name}")
    for path in root.iterdir() if root.is_dir() else ():
        if ".partial" in path.name or path.is_symlink():
            errors.append(f"unsafe_fragment_entry:{path.name}")
        elif path.name not in allowed_entries:
            errors.append(f"unexpected_fragment_entry:{path.name}")

    contract_errors = validate_campaign_contract(contract)
    errors.extend(f"contract:{value}" for value in contract_errors)
    binding_errors = validate_campaign_binding(binding, contract)
    errors.extend(f"binding:{value}" for value in binding_errors)

    manifest, full_rows = validator.validate_common(root, "execute", errors)
    metadata: dict[str, Any] | None = None
    if manifest is None:
        return _fragment_validation_payload(errors, contract), None
    fragment = manifest.get("fragment")
    validator.require(manifest.get("execution_scope") == "row_fragment", "fragment_manifest_scope", errors)
    validator.require(isinstance(fragment, dict), "fragment_manifest_metadata", errors)
    if not isinstance(fragment, dict):
        return _fragment_validation_payload(errors, contract), None
    row_id = fragment.get("row_id")
    ordinal = fragment.get("ordinal")
    matrix = contract.get("ordered_matrix", [])
    valid_ordinal = isinstance(ordinal, int) and not isinstance(ordinal, bool) and 0 <= ordinal < len(matrix)
    validator.require(valid_ordinal, "fragment_ordinal_invalid", errors)
    row = matrix[ordinal] if valid_ordinal else None
    validator.require(isinstance(row, dict) and row.get("row_id") == row_id, "fragment_row_ordinal_drift", errors)
    if not isinstance(row, dict):
        return _fragment_validation_payload(errors, contract), None

    expected_fragment = {
        "row_id": row_id,
        "ordinal": ordinal,
        "expected_row_count": len(matrix),
        "row": row,
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
        "execution_segment_id": fragment.get("execution_segment_id"),
        "invocation_uuid": fragment.get("execution_segment_id"),
        "derived_row_seed": harness.canonical_row_seed(
            row, int(contract["controls"]["seed"])
        ),
    }
    validator.require(fragment == expected_fragment, "fragment_metadata_drift", errors)
    validator.require(
        harness.INVOCATION_UUID_RE.fullmatch(
            str(fragment.get("execution_segment_id", ""))
        ) is not None,
        "fragment_invocation_uuid_invalid",
        errors,
    )
    validator.require(manifest.get("expected_matrix") == matrix, "fragment_matrix_contract_drift", errors)
    validator.require(manifest.get("shape_records") == contract.get("shape_records"), "fragment_shape_contract_drift", errors)
    validator.require(manifest.get("packages") == contract.get("packages"), "fragment_package_contract_drift", errors)
    validator.require(manifest.get("sources") == contract.get("sources"), "fragment_source_contract_drift", errors)
    validator.require(manifest.get("api_contracts") == contract.get("api_contracts"), "fragment_api_contract_drift", errors)
    validator.require(manifest.get("static_api_checks") == contract.get("static_api_checks"), "fragment_static_api_contract_drift", errors)
    validator.require(manifest.get("long_context_execution") == contract.get("long_context_execution"), "fragment_long_context_execution_contract_drift", errors)
    validator.require(manifest.get("correctness_contract") == contract.get("correctness_contract"), "fragment_correctness_contract_drift", errors)
    validator.require(manifest.get("experiment_contract") == contract.get("experiment_contract"), "fragment_experiment_contract_drift", errors)
    device = manifest.get("device", {})
    validator.require(
        isinstance(device, dict) and set(device) == DEVICE_FIELDS,
        "fragment_device_field_drift",
        errors,
    )
    target = binding["target_gpu"]
    validator.require(device.get("uuid") == target["uuid"], "fragment_device_uuid_drift", errors)
    validator.require(manifest.get("expected_gpu_index") == target["index"], "fragment_device_index_drift", errors)
    validator.require(device.get("name") == target["name"], "fragment_device_name_drift", errors)

    publication = manifest.get("publication", {})
    try:
        expected_argv = instantiate_fragment_argv(
            contract,
            binding,
            row,
            ordinal,
            Path(publication["execution_output_dir"]),
            Path(publication["requested_publish_target"]),
            fragment["execution_segment_id"],
        )
    except (KeyError, TypeError) as exc:
        errors.append(f"fragment_argv_rebuild:{type(exc).__name__}:{exc}")
    else:
        validator.require(manifest.get("argv") == expected_argv, "fragment_exact_argv_drift", errors)

    evidence = _fragment_evidence(root, binding, errors)
    if evidence is not None:
        validator.require(
            evidence.get("monitor_interval_ms")
            == contract["controls"]["monitor_interval_ms"],
            "fragment_monitor_interval_contract_drift",
            errors,
        )
        validator.require(
            evidence.get("query_timeout_ms")
            == contract["controls"]["query_timeout_ms"],
            "fragment_query_timeout_contract_drift",
            errors,
        )
    evidence_summary = validator.validate_gpu_exclusivity(
        root, manifest, evidence, errors
    ) if evidence is not None else None
    row_map = {row_id: full_rows[row_id]} if row_id in full_rows else {}
    validator.require(bool(row_map), "fragment_row_absent_from_manifest", errors)
    samples = validator.load_samples(root / "samples.jsonl", errors)
    correctness = validator.load_json(root / "correctness.json", errors)
    if row_map:
        validator.validate_sample_matrix(manifest, row_map, samples, errors)
        validator.validate_correctness(manifest, row_map, correctness, errors)
    if isinstance(correctness, dict):
        validator.require(correctness.get("execution_scope") == "row_fragment", "fragment_correctness_scope", errors)
        validator.require(correctness.get("fragment") == fragment, "fragment_correctness_binding", errors)
        validator.require(
            set(correctness)
            == {
                "schema", "kind", "status", "execution_scope",
                "fragment_row_id", "fragment", "rows",
                "all_expected_rows_present",
            },
            "fragment_correctness_top_level_field_drift",
            errors,
        )
    for sample in samples:
        validator.require(sample.get("row_id") == row_id, "fragment_sample_cross_row", errors)
        validator.require(sample.get("fragment_row_ordinal") == ordinal, "fragment_sample_ordinal", errors)
        validator.require(sample.get("invocation_uuid") == fragment.get("execution_segment_id"), "fragment_sample_invocation", errors)
        validator.require(sample.get("campaign_contract_sha256") == contract["contract_sha256"], "fragment_sample_contract", errors)
        validator.require(sample.get("campaign_fingerprint_sha256") == binding["campaign_fingerprint_sha256"], "fragment_sample_fingerprint", errors)
        validator.require(sample.get("derived_row_seed") == expected_fragment["derived_row_seed"], "fragment_sample_seed", errors)
    _validate_fragment_result(root, manifest, row, samples, correctness, errors)

    build = runtime_build_identity(manifest)
    metadata = {
        "row_id": row_id,
        "ordinal": ordinal,
        "invocation_uuid": fragment.get("execution_segment_id"),
        "derived_row_seed": expected_fragment["derived_row_seed"],
        "device": device,
        "device_sha256": harness.canonical_json_sha(device),
        "runtime_build": build,
        "runtime_build_sha256": harness.canonical_json_sha(build),
        "gpu_exclusivity": evidence_summary,
        "sample_count": len(samples),
    }
    return _fragment_validation_payload(errors, contract, metadata), metadata


def _fragment_validation_payload(
    errors: list[str],
    contract: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": "tier5_production_fragment_validation",
        "status": "PASS" if not errors else "FAIL",
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "formal_campaign": contract.get("formal") is True,
        "fragment": metadata,
        "errors": errors,
    }


def artifact_manifest(root: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name in FRAGMENT_ARTIFACTS:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required regular fragment artifact missing: {name}")
        artifacts[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": harness.sha256_file(path),
        }
    return artifacts


def seal_fragment(
    root: Path, contract: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    validation, metadata = validate_fragment(root, contract, binding)
    harness.atomic_write_json(root / "fragment_validation.json", validation)
    if validation["status"] != "PASS" or metadata is None:
        raise ValueError("fragment validation failed: " + ",".join(validation["errors"]))
    artifacts = artifact_manifest(root)
    marker = {
        "schema": SCHEMA,
        "kind": "tier5_production_fragment_completion_marker",
        "status": "PASS",
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
        "controls_sha256": contract["controls_sha256"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "package_manifest_sha256": contract["package_manifest_sha256"],
        "row_id": metadata["row_id"],
        "ordinal": metadata["ordinal"],
        "invocation_uuid": metadata["invocation_uuid"],
        "derived_row_seed": metadata["derived_row_seed"],
        "device_sha256": metadata["device_sha256"],
        "runtime_build_sha256": metadata["runtime_build_sha256"],
        "artifacts": artifacts,
    }
    harness.atomic_write_json(root / "fragment.done.json", marker)
    return marker


def check_fragment(
    root: Path, contract: dict[str, Any], binding: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    validation, metadata = validate_fragment(root, contract, binding)
    if validation["status"] != "PASS" or metadata is None:
        return validation, metadata
    try:
        marker = load_json(root / "fragment.done.json")
        current_artifacts = artifact_manifest(root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        validation["status"] = "FAIL"
        validation["errors"].append(f"fragment_marker_load:{type(exc).__name__}:{exc}")
        return validation, metadata
    expected = {
        "schema": SCHEMA,
        "kind": "tier5_production_fragment_completion_marker",
        "status": "PASS",
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
        "controls_sha256": contract["controls_sha256"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "package_manifest_sha256": contract["package_manifest_sha256"],
        "row_id": metadata["row_id"],
        "ordinal": metadata["ordinal"],
        "invocation_uuid": metadata["invocation_uuid"],
        "derived_row_seed": metadata["derived_row_seed"],
        "device_sha256": metadata["device_sha256"],
        "runtime_build_sha256": metadata["runtime_build_sha256"],
        "artifacts": current_artifacts,
    }
    if marker != expected:
        validation["status"] = "FAIL"
        validation["errors"].append("fragment_marker_or_artifact_hash_drift")
    return validation, metadata


def formal_campaign_eligible(
    contract: dict[str, Any], row_count: int, sample_count: int, summary_count: int
) -> bool:
    return (
        contract.get("formal") is True
        and contract.get("ordered_matrix") == FORMAL_MATRIX
        and row_count == FORMAL_CORRECTNESS_ROW_COUNT
        and sample_count == FORMAL_SAMPLE_COUNT
        and summary_count == FORMAL_SUMMARY_COUNT
    )


def finalize_campaign(
    root: Path, contract: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError("campaign root may not be a symlink")
    root = root.resolve()
    for name, expected in (
        ("campaign_contract.json", contract),
        ("campaign_binding.json", binding),
    ):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"campaign root {name} must be a regular file")
        if load_json(path) != expected:
            raise ValueError(f"campaign root {name} differs from supplied value")
    rows_root = root / "rows"
    if rows_root.is_symlink() or not rows_root.is_dir():
        raise ValueError("campaign rows directory must be a regular directory")
    expected_names = [
        row_directory_name(ordinal, row["row_id"])
        for ordinal, row in enumerate(contract["ordered_matrix"])
    ]
    actual_entries = sorted(path.name for path in rows_root.iterdir())
    if actual_entries != sorted(expected_names):
        raise ValueError(
            f"campaign row inventory drift: actual={actual_entries} expected={sorted(expected_names)}"
        )
    all_samples: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    fragment_markers: list[dict[str, Any]] = []
    invocation_uuids: set[str] = set()
    common_build_sha: str | None = None
    row_runtime_identities: dict[str, dict[str, Any]] = {}
    aggregate_manifest: dict[str, Any] | None = None
    for ordinal, row in enumerate(contract["ordered_matrix"]):
        fragment_root = rows_root / row_directory_name(ordinal, row["row_id"])
        if not fragment_root.is_dir() or fragment_root.is_symlink():
            raise ValueError(f"fragment is not a regular directory: {fragment_root}")
        validation, metadata = check_fragment(fragment_root, contract, binding)
        if validation["status"] != "PASS" or metadata is None:
            raise ValueError(
                f"fragment revalidation failed {row['row_id']}: {validation['errors']}"
            )
        if metadata["row_id"] != row["row_id"] or metadata["ordinal"] != ordinal:
            raise ValueError(f"fragment row/ordinal drift: {row['row_id']}")
        invocation_uuid = str(metadata["invocation_uuid"])
        if invocation_uuid in invocation_uuids:
            raise ValueError(f"duplicate fragment invocation UUID: {invocation_uuid}")
        invocation_uuids.add(invocation_uuid)
        if common_build_sha is None:
            common_build_sha = metadata["runtime_build_sha256"]
        elif metadata["runtime_build_sha256"] != common_build_sha:
            raise ValueError("fragment runtime build identity drift")
        row_runtime_identities[row["row_id"]] = {
            "process_pid": metadata["device"].get("process_pid"),
            "process_start_ticks": metadata["device"].get(
                "process_start_ticks"
            ),
        }
        manifest = load_json(fragment_root / "manifest.json")
        if aggregate_manifest is None:
            aggregate_manifest = dict(manifest)
        samples = validator.load_samples(fragment_root / "samples.jsonl", [])
        correctness = load_json(fragment_root / "correctness.json")
        if not isinstance(correctness.get("rows"), list) or len(correctness["rows"]) != 1:
            raise ValueError(f"fragment correctness cardinality: {row['row_id']}")
        all_samples.extend(samples)
        correctness_rows.extend(correctness["rows"])
        marker_path = fragment_root / "fragment.done.json"
        fragment_markers.append(
            {
                "row_id": row["row_id"],
                "ordinal": ordinal,
                "path": str(marker_path.relative_to(root)),
                "sha256": harness.sha256_file(marker_path),
            }
        )
    assert aggregate_manifest is not None
    aggregate_manifest["execution_scope"] = "campaign_aggregate"
    aggregate_manifest["fragment"] = None
    aggregate_manifest["publication"] = {
        "execution_output_dir": str(root),
        "requested_publish_target": str(root),
        "failure_atomic_stage": False,
        "runner_managed_stage": True,
    }
    aggregate_manifest["argv"] = contract["fragment_argv_template"]
    aggregate_manifest["argv_sha256"] = harness.canonical_json_sha(
        aggregate_manifest["argv"]
    )
    aggregate_manifest["campaign_contract_sha256"] = contract["contract_sha256"]
    aggregate_manifest["campaign_fingerprint_sha256"] = binding[
        "campaign_fingerprint_sha256"
    ]
    aggregate_manifest["fragment_markers"] = fragment_markers
    aggregate_manifest["row_runtime_identities"] = row_runtime_identities

    errors: list[str] = []
    row_map = {
        row["row_id"]: row for row in contract["ordered_matrix"]
    }
    validator.validate_sample_matrix(aggregate_manifest, row_map, all_samples, errors)
    aggregate_correctness = {
        "schema": SCHEMA,
        "kind": "tier5_production_correctness",
        "status": "PASS",
        "execution_scope": "campaign_aggregate",
        "rows": correctness_rows,
        "all_expected_rows_present": True,
    }
    validator.validate_correctness(
        aggregate_manifest, row_map, aggregate_correctness, errors
    )
    summaries = harness.summarize_samples(
        all_samples, int(contract["controls"]["seed"])
    )
    eligible = formal_campaign_eligible(
        contract, len(correctness_rows), len(all_samples), len(summaries)
    )
    if contract.get("formal") is True and not eligible:
        errors.append(
            "formal_campaign_exact_cardinality_failed:"
            f"rows={len(correctness_rows)} samples={len(all_samples)} summaries={len(summaries)}"
        )
    if errors:
        raise ValueError("whole-campaign validation failed: " + ",".join(errors))

    samples_payload = b"".join(
        (json.dumps(sample, sort_keys=True, allow_nan=False) + "\n").encode()
        for sample in all_samples
    )
    harness.atomic_write_json(root / "manifest.json", aggregate_manifest)
    harness.atomic_write_bytes(root / "samples.jsonl", samples_payload)
    harness.atomic_write_json(root / "correctness.json", aggregate_correctness)
    result = {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_result",
        "status": "PASS",
        "campaign_mode": contract["campaign_mode"],
        "accepted_timing": 0,
        "accepted_workload_timing": int(eligible),
        "accepted_CTA_bracket": 0,
        "tier5_bracket_admitted": False,
        "formal_bracket_status": "PARTIAL",
        "headroom_defined": False,
        "headroom_pct": None,
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
        "manifest_sha256": harness.sha256_file(root / "manifest.json"),
        "samples_sha256": harness.sha256_file(root / "samples.jsonl"),
        "correctness_sha256": harness.sha256_file(root / "correctness.json"),
        "sample_count": len(all_samples),
        "correctness_row_count": len(correctness_rows),
        "summary_count": len(summaries),
        "summaries": summaries,
        "fragment_markers": fragment_markers,
        "device_binding_sha256": harness.canonical_json_sha(
            binding["target_gpu"]
        ),
        "runtime_build_sha256": common_build_sha,
    }
    harness.atomic_write_json(root / "result.json", result)
    validation = {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_validation",
        "status": "PASS",
        "formal_campaign": contract["formal"],
        "exact_inventory": True,
        "accepted_timing": 0,
        "accepted_workload_timing": int(eligible),
        "accepted_CTA_bracket": 0,
        "errors": [],
        "result_sha256": harness.sha256_file(root / "result.json"),
    }
    harness.atomic_write_json(root / "campaign_validation.json", validation)
    marker_artifacts = {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": harness.sha256_file(root / name),
        }
        for name in (
            "campaign_contract.json", "campaign_binding.json", "manifest.json",
            "samples.jsonl", "correctness.json", "result.json",
            "campaign_validation.json",
        )
    }
    marker = {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_completion_marker",
        "status": "PASS",
        "campaign_mode": contract["campaign_mode"],
        "accepted_timing": 0,
        "accepted_workload_timing": int(eligible),
        "accepted_CTA_bracket": 0,
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
        "fragment_markers": fragment_markers,
        "artifacts": marker_artifacts,
    }
    harness.atomic_write_json(root / "production_candidate.done.json", marker)
    return marker


def check_final_campaign(
    root: Path, contract: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    """Read-only fresh validation of a published aggregate and every fragment."""
    if root.is_symlink():
        raise ValueError("campaign root may not be a symlink")
    root = root.resolve()
    for name, expected in (
        ("campaign_contract.json", contract),
        ("campaign_binding.json", binding),
    ):
        path = root / name
        if not path.is_file() or path.is_symlink() or load_json(path) != expected:
            raise ValueError(f"unsafe or drifted campaign root file: {name}")
    required = (
        "manifest.json", "samples.jsonl", "correctness.json", "result.json",
        "campaign_validation.json", "production_candidate.done.json",
    )
    for name in required:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"final campaign artifact is not regular: {name}")
    rows_root = root / "rows"
    if not rows_root.is_dir() or rows_root.is_symlink():
        raise ValueError("final campaign rows directory is unsafe")
    expected_names = [
        row_directory_name(i, row["row_id"])
        for i, row in enumerate(contract["ordered_matrix"])
    ]
    if sorted(path.name for path in rows_root.iterdir()) != sorted(expected_names):
        raise ValueError("final campaign row inventory drift")

    gathered_samples: list[dict[str, Any]] = []
    gathered_correctness: list[dict[str, Any]] = []
    fragment_markers: list[dict[str, Any]] = []
    invocation_uuids: set[str] = set()
    runtime_build_sha: str | None = None
    row_runtime_identities: dict[str, dict[str, Any]] = {}
    expected_aggregate_manifest: dict[str, Any] | None = None
    for ordinal, row in enumerate(contract["ordered_matrix"]):
        fragment_root = rows_root / row_directory_name(ordinal, row["row_id"])
        validation, metadata = check_fragment(fragment_root, contract, binding)
        if validation["status"] != "PASS" or metadata is None:
            raise ValueError(
                f"final fragment revalidation failed {row['row_id']}: "
                f"{validation['errors']}"
            )
        if metadata["row_id"] != row["row_id"] or metadata["ordinal"] != ordinal:
            raise ValueError("final fragment row/ordinal mismatch")
        invocation = str(metadata["invocation_uuid"])
        if invocation in invocation_uuids:
            raise ValueError("duplicate final fragment invocation UUID")
        invocation_uuids.add(invocation)
        if runtime_build_sha is None:
            runtime_build_sha = metadata["runtime_build_sha256"]
        elif runtime_build_sha != metadata["runtime_build_sha256"]:
            raise ValueError("final fragment runtime build drift")
        row_runtime_identities[row["row_id"]] = {
            "process_pid": metadata["device"].get("process_pid"),
            "process_start_ticks": metadata["device"].get("process_start_ticks"),
        }
        sample_errors: list[str] = []
        gathered_samples.extend(
            validator.load_samples(fragment_root / "samples.jsonl", sample_errors)
        )
        if sample_errors:
            raise ValueError(f"fragment sample reload failed: {sample_errors}")
        correctness = load_json(fragment_root / "correctness.json")
        rows = correctness.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("fragment correctness row cardinality drift")
        gathered_correctness.extend(rows)
        if expected_aggregate_manifest is None:
            expected_aggregate_manifest = dict(
                load_json(fragment_root / "manifest.json")
            )
        marker_path = fragment_root / "fragment.done.json"
        fragment_markers.append(
            {
                "row_id": row["row_id"],
                "ordinal": ordinal,
                "path": str(marker_path.relative_to(root)),
                "sha256": harness.sha256_file(marker_path),
            }
        )

    sample_errors: list[str] = []
    published_samples = validator.load_samples(root / "samples.jsonl", sample_errors)
    if sample_errors or published_samples != gathered_samples:
        raise ValueError("published samples differ from canonical fragment concatenation")
    published_correctness = load_json(root / "correctness.json")
    if published_correctness != {
        "schema": SCHEMA,
        "kind": "tier5_production_correctness",
        "status": "PASS",
        "execution_scope": "campaign_aggregate",
        "rows": gathered_correctness,
        "all_expected_rows_present": True,
    }:
        raise ValueError("published correctness differs from fragments")
    assert expected_aggregate_manifest is not None
    expected_aggregate_manifest["execution_scope"] = "campaign_aggregate"
    expected_aggregate_manifest["fragment"] = None
    expected_aggregate_manifest["publication"] = {
        "execution_output_dir": str(root),
        "requested_publish_target": str(root),
        "failure_atomic_stage": False,
        "runner_managed_stage": True,
    }
    expected_aggregate_manifest["argv"] = contract["fragment_argv_template"]
    expected_aggregate_manifest["argv_sha256"] = harness.canonical_json_sha(
        expected_aggregate_manifest["argv"]
    )
    expected_aggregate_manifest["campaign_contract_sha256"] = contract[
        "contract_sha256"
    ]
    expected_aggregate_manifest["campaign_fingerprint_sha256"] = binding[
        "campaign_fingerprint_sha256"
    ]
    expected_aggregate_manifest["fragment_markers"] = fragment_markers
    expected_aggregate_manifest["row_runtime_identities"] = row_runtime_identities
    manifest = load_json(root / "manifest.json")
    if manifest != expected_aggregate_manifest:
        raise ValueError("published aggregate manifest binding drift")
    errors: list[str] = []
    row_map = {row["row_id"]: row for row in contract["ordered_matrix"]}
    validator.validate_sample_matrix(manifest, row_map, published_samples, errors)
    validator.validate_correctness(
        manifest, row_map, published_correctness, errors
    )
    summaries = harness.summarize_samples(
        published_samples, int(contract["controls"]["seed"])
    )
    eligible = formal_campaign_eligible(
        contract, len(gathered_correctness), len(published_samples), len(summaries)
    )
    if contract.get("formal") is True and not eligible:
        errors.append("formal_final_cardinality_not_exact")
    if errors:
        raise ValueError("final whole-matrix validation failed: " + ",".join(errors))
    result = load_json(root / "result.json")
    expected_result = {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_result",
        "status": "PASS",
        "campaign_mode": contract["campaign_mode"],
        "accepted_timing": 0,
        "accepted_workload_timing": int(eligible),
        "accepted_CTA_bracket": 0,
        "tier5_bracket_admitted": False,
        "formal_bracket_status": "PARTIAL",
        "headroom_defined": False,
        "headroom_pct": None,
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
        "manifest_sha256": harness.sha256_file(root / "manifest.json"),
        "samples_sha256": harness.sha256_file(root / "samples.jsonl"),
        "correctness_sha256": harness.sha256_file(root / "correctness.json"),
        "sample_count": len(published_samples),
        "correctness_row_count": len(gathered_correctness),
        "summary_count": len(summaries),
        "summaries": summaries,
        "fragment_markers": fragment_markers,
        "device_binding_sha256": harness.canonical_json_sha(
            binding["target_gpu"]
        ),
        "runtime_build_sha256": runtime_build_sha,
    }
    if result != expected_result:
        raise ValueError("final result exact reconstruction drift")
    validation = load_json(root / "campaign_validation.json")
    if validation != {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_validation",
        "status": "PASS",
        "formal_campaign": contract["formal"],
        "exact_inventory": True,
        "accepted_timing": 0,
        "accepted_workload_timing": int(eligible),
        "accepted_CTA_bracket": 0,
        "errors": [],
        "result_sha256": harness.sha256_file(root / "result.json"),
    }:
        raise ValueError("campaign validation artifact drift")
    marker_artifacts = {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": harness.sha256_file(root / name),
        }
        for name in (
            "campaign_contract.json", "campaign_binding.json", "manifest.json",
            "samples.jsonl", "correctness.json", "result.json",
            "campaign_validation.json",
        )
    }
    expected_marker = {
        "schema": SCHEMA,
        "kind": "tier5_production_campaign_completion_marker",
        "status": "PASS",
        "campaign_mode": contract["campaign_mode"],
        "accepted_timing": 0,
        "accepted_workload_timing": int(eligible),
        "accepted_CTA_bracket": 0,
        "campaign_contract_sha256": contract["contract_sha256"],
        "campaign_fingerprint_sha256": binding["campaign_fingerprint_sha256"],
        "fragment_markers": fragment_markers,
        "artifacts": marker_artifacts,
    }
    if load_json(root / "production_candidate.done.json") != expected_marker:
        raise ValueError("campaign completion marker or artifact hash drift")
    return expected_marker


def add_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default=",".join(FORMAL_MODELS))
    parser.add_argument("--seqs", default=",".join(str(x) for x in harness.FORMAL_SEQS))
    parser.add_argument("--workloads", default=",".join(harness.FORMAL_WORKLOADS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--max-logits-mb", type=int, default=harness.FORMAL_MAX_LOGITS_MB
    )
    parser.add_argument(
        "--max-query-chunk", type=int, default=harness.FORMAL_MAX_QUERY_CHUNK
    )
    parser.add_argument("--moe-experts", type=int, default=32)
    parser.add_argument("--moe-topk", type=int, default=8)
    parser.add_argument("--moe-tokens", type=int, default=4096)
    parser.add_argument("--backend", default="flashinfer")
    parser.add_argument("--required-device-substring", default="B200")
    parser.add_argument("--monitor-interval-ms", type=int, default=50)
    parser.add_argument("--query-timeout-ms", type=int, default=2000)


def controls_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "models": harness.parse_csv_choice(args.models, harness.MODEL_SPECS, "models"),
        "seqs": harness.parse_seqs(args.seqs),
        "workloads": harness.parse_csv_choice(args.workloads, harness.FORMAL_WORKLOADS, "workloads"),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "allow_short": args.allow_short,
        "seed": args.seed,
        "max_logits_mb": args.max_logits_mb,
        "max_query_chunk": args.max_query_chunk,
        "moe_experts": args.moe_experts,
        "moe_topk": args.moe_topk,
        "moe_tokens": args.moe_tokens,
        "backend": args.backend,
        "required_device_substring": args.required_device_substring,
        "monitor_interval_ms": args.monitor_interval_ms,
        "query_timeout_ms": args.query_timeout_ms,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-contract")
    make.add_argument("--output", type=Path, required=True)
    add_control_arguments(make)
    check = commands.add_parser("check-contract")
    check.add_argument("--contract", type=Path, required=True)
    bind = commands.add_parser("bind-device")
    bind.add_argument("--contract", type=Path, required=True)
    bind.add_argument("--identity", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    for name in ("validate-fragment", "check-fragment"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--binding", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--contract", type=Path, required=True)
    finalize.add_argument("--binding", type=Path, required=True)
    check_final = commands.add_parser("check-final")
    check_final.add_argument("--root", type=Path, required=True)
    check_final.add_argument("--contract", type=Path, required=True)
    check_final.add_argument("--binding", type=Path, required=True)
    publish = commands.add_parser("publish-fragment")
    publish.add_argument("--stage", type=Path, required=True)
    publish.add_argument("--final", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "publish-fragment":
            atomic_publish_directory(args.stage, args.final)
            print(f"TIER5_FRAGMENT_PUBLISH status=PASS final={args.final.absolute()}")
            return 0
        if args.command == "make-contract":
            value = build_campaign_contract(controls_from_cli(args))
            if args.output.exists():
                existing = load_regular_json(
                    args.output, "existing campaign contract"
                )
                if existing != value:
                    raise ValueError("existing campaign contract drift; refusing overwrite")
            else:
                harness.atomic_write_json(args.output, value)
            print(f"TIER5_CAMPAIGN_CONTRACT status=PASS rows={value['row_count']} formal={int(value['formal'])} sha256={value['contract_sha256']}")
            return 0
        contract = load_and_validate_contract(args.contract)
        if args.command == "check-contract":
            print(f"TIER5_CAMPAIGN_CONTRACT_CHECK status=PASS sha256={contract['contract_sha256']}")
            return 0
        if args.command == "bind-device":
            value = bind_campaign_device(
                contract, load_regular_json(args.identity, "GPU identity")
            )
            if args.output.exists():
                if load_regular_json(
                    args.output, "existing campaign binding"
                ) != value:
                    raise ValueError("existing campaign device binding drift")
            else:
                harness.atomic_write_json(args.output, value)
            print(f"TIER5_CAMPAIGN_BINDING status=PASS fingerprint={value['campaign_fingerprint_sha256']}")
            return 0
        binding = load_regular_json(args.binding, "campaign binding")
        binding_errors = validate_campaign_binding(binding, contract)
        if binding_errors:
            raise ValueError("invalid binding: " + ",".join(binding_errors))
        if args.command == "validate-fragment":
            marker = seal_fragment(args.root.absolute(), contract, binding)
            print(f"TIER5_FRAGMENT_SEAL status=PASS row={marker['row_id']} ordinal={marker['ordinal']} accepted_workload_timing=0")
            return 0
        if args.command == "check-fragment":
            validation, metadata = check_fragment(args.root.absolute(), contract, binding)
            print(f"TIER5_FRAGMENT_CHECK status={validation['status']} row={metadata.get('row_id') if metadata else None} accepted_workload_timing=0")
            if validation["status"] != "PASS":
                for error in validation["errors"]:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 2
            return 0
        if args.command == "finalize":
            marker = finalize_campaign(args.root.absolute(), contract, binding)
            print(f"TIER5_CAMPAIGN_FINAL status=PASS accepted_workload_timing={marker['accepted_workload_timing']} accepted_CTA_bracket=0")
            return 0
        if args.command == "check-final":
            marker = check_final_campaign(
                args.root.absolute(), contract, binding
            )
            print(
                "TIER5_CAMPAIGN_FINAL_CHECK status=PASS "
                f"accepted_workload_timing={marker['accepted_workload_timing']}"
            )
            return 0
        raise AssertionError(args.command)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
