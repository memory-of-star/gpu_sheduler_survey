#!/usr/bin/env python3
"""Close the Tier-5 evidence graph and emit the only final campaign admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import build_provenance


HEX256 = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
FORMAL = (
    (4096, "dsa_exact_seq4096"),
    (32768, "dsa_exact_seq32768"),
    (131072, "dsa_work_complete_packed_proxy_seq131072"),
    (1048576, "dsa_work_complete_packed_proxy_seq1048576"),
)
FAST = ((4096, "dsa_exact_seq4096"),)
PROFILE_RANGES = {
    "dsa.poison", "dsa.floor", "dsa.wave_floor", "dsa.impl", "dsa.ceiling",
    "dsa.validate.floor", "dsa.validate.wave_floor", "dsa.validate.impl",
    "dsa.validate.ceiling_wrongness",
}
EXIT_RESIDUAL_LIMIT = 4
EXIT_RESIDUAL_POLICY = (
    "previously_observed_allowed_same_pid_start_ticks_and_exact_No_data_"
    "with_proc_missing_only_bounded_per_pid"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def read_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level JSON is not an object")
        return value
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot read JSON {path.name}: {exc}")
        return {}


def parse_marker(path: Path, errors: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read marker {path.name}: {exc}")
        return values
    for line_number, line in enumerate(lines, 1):
        if "=" not in line:
            errors.append(f"{path.name}:{line_number}: malformed line")
            continue
        key, value = line.split("=", 1)
        if not key or key in values:
            errors.append(f"{path.name}:{line_number}: duplicate/empty key")
            continue
        values[key] = value
    return values


def hash_required(
    path: Path, errors: list[str], hashes: dict[str, str], label: str
) -> str | None:
    try:
        value = sha256(path)
        hashes[label] = value
        return value
    except OSError as exc:
        errors.append(f"{label}: cannot hash {path}: {exc}")
        return None


def clean_pass(value: dict[str, Any], label: str, errors: list[str]) -> None:
    if value.get("status") != "PASS" or value.get("errors") != []:
        errors.append(f"{label}: not a clean PASS")


def validate_checkpoint(
    path: Path, lease_id: str | None, target_gpu: dict[str, Any] | None,
    errors: list[str], hashes: dict[str, str],
    label: str, *, permit_lease: bool = False,
) -> None:
    hash_required(path, errors, hashes, label)
    value = read_json(path, errors, label)
    clean_pass(value, label, errors)
    allowed_kinds = {"gpu_exclusivity_checkpoint"}
    if permit_lease:
        allowed_kinds.add("gpu_exclusivity_lease")
    if value.get("schema") != 1 or value.get("kind") not in allowed_kinds:
        errors.append(f"{label}: schema/kind mismatch")
    if value.get("lease_id") != lease_id:
        errors.append(f"{label}: lease_id mismatch")
    if value.get("observation", {}).get("target_gpu") != target_gpu:
        errors.append(f"{label}: target GPU identity mismatch")
    if value.get("accepted_timing") != 0:
        errors.append(f"{label}: exclusivity evidence must not be timing")
    if value.get("observation", {}).get("target_compute_processes") != []:
        errors.append(f"{label}: target GPU compute process list is not empty")


def validate_monitor(
    manifest_path: Path, observations_path: Path, lease_id: str | None,
    target_gpu: dict[str, Any] | None, expected_phase: str,
    require_allowed: bool, expected_interval_ms: int | None,
    expected_query_timeout_ms: int | None,
    errors: list[str], hashes: dict[str, str], label: str,
) -> None:
    hash_required(manifest_path, errors, hashes, label + ":manifest")
    observations_sha = hash_required(
        observations_path, errors, hashes, label + ":observations"
    )
    manifest = read_json(manifest_path, errors, label)
    clean_pass(manifest, label, errors)
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "gpu_exclusivity_monitor"
        or manifest.get("lease_id") != lease_id
        or manifest.get("target_gpu") != target_gpu
        or manifest.get("phase") != expected_phase
        or manifest.get("accepted_timing") != 0
        or manifest.get("observations_sha256") != observations_sha
        or Path(str(manifest.get("observations_path", ""))) != observations_path
        or manifest.get("foreign_processes_detected") is not False
        or manifest.get("query_failure_detected") is not False
        or manifest.get("require_allowed_process") is not require_allowed
        or manifest.get("poll_interval_ms") != expected_interval_ms
        or manifest.get("query_timeout_ms") != expected_query_timeout_ms
        or manifest.get("coverage_model")
        != "bounded_interval_nvidia_smi_process_sampling"
        or manifest.get("coverage_limit")
        != "foreign GPU processes wholly between completed samples may not be observed"
        or manifest.get("start_barrier_complete") is not True
        or manifest.get("ready_record_written") is not True
        or manifest.get("baseline_observation_sequence") != 0
        or manifest.get("allowed_exit_residual_max_observations_per_pid")
        != EXIT_RESIDUAL_LIMIT
        or manifest.get("exit_residual_policy") != EXIT_RESIDUAL_POLICY
    ):
        errors.append(f"{label}: monitor manifest contract mismatch")
    if (
        not isinstance(expected_interval_ms, int)
        or not 10 <= expected_interval_ms <= 100
        or not isinstance(expected_query_timeout_ms, int)
        or not 100 <= expected_query_timeout_ms <= 5000
    ):
        errors.append(f"{label}: monitor polling/query bounds are unsafe")
    if require_allowed and (
        manifest.get("allowed_observation_count", 0) <= 0
        or not manifest.get("allowed_processes")
    ):
        errors.append(f"{label}: watched GPU process tree was never observed")

    line_count = 0
    allowed_line_count = 0
    residual_line_count = 0
    observed_allowed: set[tuple[int, int]] = set()
    residual_counts: dict[int, int] = {}
    residual_starts: dict[int, int] = {}
    previous_query_finished: int | None = None
    target_empty_by_line: list[bool] = []
    try:
        with observations_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    errors.append(f"{label}: blank observation line {line_number}")
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("observation is not an object")
                observation = record.get("observation", {})
                target_processes = observation.get("target_compute_processes", [])
                allowed_processes = record.get("allowed_target_processes", [])
                residual_processes = record.get(
                    "allowed_exit_residual_processes", []
                )
                query_started = record.get("query_started_monotonic_ns")
                query_finished = record.get("query_finished_monotonic_ns")
                query_duration = record.get("query_duration_ms")
                if (
                    record.get("schema") != 1
                    or record.get("sequence") != line_count
                    or record.get("query_errors") != []
                    or record.get("foreign_target_processes") != []
                    or observation.get("target_gpu") != target_gpu
                    or not isinstance(target_processes, list)
                    or not isinstance(allowed_processes, list)
                    or not isinstance(residual_processes, list)
                    or len(target_processes)
                    != len(allowed_processes) + len(residual_processes)
                    or not isinstance(query_started, int)
                    or not isinstance(query_finished, int)
                    or query_finished < query_started
                    or not isinstance(query_duration, (int, float))
                    or query_duration < 0
                    or (
                        previous_query_finished is not None
                        and query_started < previous_query_finished
                    )
                ):
                    errors.append(f"{label}: malformed/unsafe observation {line_number}")
                if line_count == 0 and (target_processes or allowed_processes):
                    errors.append(f"{label}: start-gate baseline was not idle")
                target_by_pid = {
                    process.get("pid"): process for process in target_processes
                    if isinstance(process, dict)
                    and isinstance(process.get("pid"), int)
                }
                allowed_by_pid = {
                    process.get("pid"): process for process in allowed_processes
                    if isinstance(process, dict)
                    and isinstance(process.get("pid"), int)
                }
                residual_by_pid = {
                    process.get("pid"): process for process in residual_processes
                    if isinstance(process, dict)
                    and isinstance(process.get("pid"), int)
                }
                if (
                    len(target_by_pid) != len(target_processes)
                    or len(allowed_by_pid) != len(allowed_processes)
                    or len(residual_by_pid) != len(residual_processes)
                    or set(allowed_by_pid) & set(residual_by_pid)
                    or set(target_by_pid)
                    != set(allowed_by_pid) | set(residual_by_pid)
                ):
                    errors.append(f"{label}: target/allowed process partition mismatch")
                if any(
                    not isinstance(process, dict)
                    or process.get("gpu_uuid")
                    != (target_gpu or {}).get("uuid")
                    for process in target_processes
                ):
                    errors.append(
                        f"{label}: target process GPU UUID mismatch at line "
                        f"{line_number}"
                    )
                for process in allowed_processes:
                    if not isinstance(process, dict):
                        errors.append(f"{label}: malformed allowed process {line_number}")
                        continue
                    pid = process.get("pid")
                    start_ticks = process.get("proc_start_ticks")
                    if not isinstance(pid, int) or not isinstance(start_ticks, int):
                        errors.append(f"{label}: unbound allowed process {line_number}")
                    else:
                        if pid in residual_counts:
                            errors.append(
                                f"{label}: retired PID returned as allowed at line "
                                f"{line_number}"
                            )
                        target_process = target_by_pid.get(pid)
                        if not isinstance(target_process, dict) or any(
                            process.get(key) != target_process.get(key)
                            for key in ("pid", "gpu_uuid", "name", "used_memory")
                        ):
                            errors.append(
                                f"{label}: allowed process/raw observation mismatch "
                                f"at line {line_number}"
                            )
                        observed_allowed.add((pid, start_ticks))
                for process in residual_processes:
                    if not isinstance(process, dict):
                        errors.append(
                            f"{label}: malformed exit residual {line_number}"
                        )
                        continue
                    pid = process.get("pid")
                    previous_start = process.get("previous_allowed_start_ticks")
                    residual_number = process.get("residual_observation_number")
                    target_process = target_by_pid.get(pid)
                    if (
                        not isinstance(pid, int)
                        or not isinstance(previous_start, int)
                        or process.get("proc_start_ticks") is not None
                        or process.get("name") != "[No data]"
                        or process.get("classification")
                        != "allowed_post_exit_nvidia_smi_residual"
                        or process.get("residual_observation_limit")
                        != EXIT_RESIDUAL_LIMIT
                        or not isinstance(residual_number, int)
                        or not 1 <= residual_number <= EXIT_RESIDUAL_LIMIT
                        or (pid, previous_start) not in observed_allowed
                        or (
                            pid in residual_starts
                            and residual_starts[pid] != previous_start
                        )
                        or residual_number != residual_counts.get(pid, 0) + 1
                        or not isinstance(target_process, dict)
                        or any(
                            process.get(key) != target_process.get(key)
                            for key in ("pid", "gpu_uuid", "name", "used_memory")
                        )
                    ):
                        errors.append(
                            f"{label}: unsafe/unbound exit residual at line "
                            f"{line_number}"
                        )
                        continue
                    residual_starts[pid] = previous_start
                    residual_counts[pid] = residual_number
                if allowed_processes:
                    allowed_line_count += 1
                if residual_processes:
                    residual_line_count += 1
                target_empty_by_line.append(not target_processes)
                if isinstance(query_finished, int):
                    previous_query_finished = query_finished
                line_count += 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot validate observations: {exc}")
    if line_count < 2 or manifest.get("observation_count") != line_count:
        errors.append(f"{label}: observation count mismatch")
    if manifest.get("allowed_observation_count") != allowed_line_count:
        errors.append(f"{label}: allowed observation count mismatch")
    if (
        manifest.get("allowed_exit_residual_observation_count")
        != residual_line_count
    ):
        errors.append(f"{label}: exit residual observation count mismatch")
    if len(target_empty_by_line) < 2 or target_empty_by_line[-2:] != [True, True]:
        errors.append(f"{label}: monitor did not finish with two empty drain samples")
    manifest_allowed = manifest.get("allowed_processes", [])
    manifest_allowed_set = {
        (process.get("pid"), process.get("proc_start_ticks"))
        for process in manifest_allowed if isinstance(process, dict)
    }
    if (
        not isinstance(manifest_allowed, list)
        or len(manifest_allowed_set) != len(manifest_allowed)
        or manifest_allowed_set != observed_allowed
    ):
        errors.append(f"{label}: allowed process identity set mismatch")
    expected_residual_manifest = [
        {
            "pid": pid,
            "previous_allowed_start_ticks": residual_starts[pid],
            "residual_observation_count": residual_counts[pid],
        }
        for pid in sorted(residual_counts)
    ]
    if manifest.get("allowed_exit_residual_processes") != expected_residual_manifest:
        errors.append(f"{label}: exit residual identity/count set mismatch")
    if manifest.get("max_allowed_exit_residual_observations_observed") != max(
        residual_counts.values(), default=0
    ):
        errors.append(f"{label}: exit residual maximum count mismatch")


def finalize_campaign(
    results: Path, fast: int, profile: int, binary: Path, runner: Path
) -> dict[str, Any]:
    expected = FAST if fast else FORMAL
    expected_tags = [tag for _, tag in expected]
    expected_tag_set = set(expected_tags)
    errors: list[str] = []
    hashes: dict[str, str] = {}
    script_dir = Path(__file__).resolve().parent
    if not fast and profile != 1:
        errors.append("formal four-point campaign requires profile=1")

    for rejected in ("formal_rejection.json", "REJECTED.md"):
        if (results / rejected).exists():
            errors.append(f"rejected results directory contains {rejected}")

    matrix_path = results / "validation_matrix.json"
    matrix = read_json(matrix_path, errors, "validation_matrix")
    hash_required(matrix_path, errors, hashes, "validation_matrix")
    hash_required(results / "dsa_matrix.log", errors, hashes, "timing_matrix_log")
    clean_pass(matrix, "validation_matrix", errors)
    if (
        matrix.get("schema") != 1
        or matrix.get("fast") != fast
        or set(matrix.get("expected_tags", [])) != expected_tag_set
        or matrix.get("admission_scope") != "timing_matrix_only"
        or matrix.get("campaign_complete") is not False
    ):
        errors.append("validation_matrix identity/scope mismatch")
    matrix_records = matrix.get("records", [])
    if not isinstance(matrix_records, list) or len(matrix_records) != len(expected):
        errors.append("validation_matrix record count mismatch")
        matrix_records = []
    record_identities = {
        (record.get("seq"), record.get("tag"))
        for record in matrix_records if isinstance(record, dict)
    }
    if record_identities != set(expected):
        errors.append("validation_matrix record identity set mismatch")
    for record in matrix_records:
        if not isinstance(record, dict) or record.get("status") != "PASS" or record.get("errors") != []:
            errors.append("validation_matrix contains a non-PASS strict record")
    matrix_by_identity = {
        (record.get("seq"), record.get("tag")): record
        for record in matrix_records if isinstance(record, dict)
    }

    observed_done = {path.stem for path in results.glob("dsa_*.done")}
    if observed_done != expected_tag_set:
        errors.append(
            f"done set mismatch: observed={sorted(observed_done)} "
            f"expected={sorted(expected_tag_set)}"
        )

    marker_values: dict[str, dict[str, str]] = {}
    marker_common_keys = (
        "profile", "gpu_index", "gpu_physical_uuid", "cuda_visible_devices",
        "native_runtime_uuid",
        "mode_count", "mode_order", "sample_order", "invocations_per_point",
        "device_fingerprint_sha256", "target_arch", "binary_sha256",
        "source_sha256", "bench_util_sha256", "cta_trace_sha256", "build_sha256",
        "build_provenance_helper_sha256", "build_manifest_sha256",
        "build_log_sha256",
        "validator_sha256", "binary_verifier_sha256", "binary_proof_sha256",
        "nvtx_range_count", "gpu_exclusivity_lease_id",
        "gpu_exclusivity_lease_sha256", "gpu_exclusivity_helper_sha256",
        "gpu_global_lock_scope", "gpu_global_lock_key_sha256",
        "gpu_global_lock_path_sha256",
        "gpu_monitor_interval_ms", "gpu_query_timeout_ms",
        "gpu_monitor_coverage_model",
        "aggregator_sha256", "profile_validator_sha256", "profiler_evidence_sha256",
        "campaign_finalizer_sha256", "runner_sha256",
    )
    common_reference: dict[str, str] | None = None
    artifact_specs: dict[str, list[tuple[str, Path]]] = {}
    for seq, tag in expected:
        marker = results / f"{tag}.done"
        values = parse_marker(marker, errors)
        marker_values[tag] = values
        hash_required(marker, errors, hashes, f"marker:{tag}")
        expected_header = {
            "marker_schema": "2", "fast": str(fast), "profile": str(profile),
            "tag": tag, "seq": str(seq), "nvtx_range_count": "9",
            "mode_count": "4",
            "mode_order": "floor,wave_floor,impl,ceiling",
            "sample_order": "cyclic_latin_4",
            "invocations_per_point": "20" if fast else "140",
        }
        for key, expected_value in expected_header.items():
            if values.get(key) != expected_value:
                errors.append(f"{tag}: marker {key} mismatch")
        if not re.fullmatch(r"[1-9][0-9]*(?:s|m|h)?", values.get("step_timeout_spec", "")):
            errors.append(f"{tag}: marker step_timeout_spec malformed")
        for key in (
            "device_fingerprint_sha256", "binary_sha256", "source_sha256",
            "bench_util_sha256", "cta_trace_sha256", "build_sha256",
            "build_provenance_helper_sha256", "build_manifest_sha256",
            "build_log_sha256",
            "validator_sha256", "binary_verifier_sha256", "binary_proof_sha256",
            "gpu_exclusivity_lease_sha256", "gpu_exclusivity_helper_sha256",
            "gpu_global_lock_key_sha256", "gpu_global_lock_path_sha256",
            "aggregator_sha256", "profile_validator_sha256",
            "profiler_evidence_sha256", "campaign_finalizer_sha256",
            "runner_sha256", "argv_sha256", "log_sha256", "trace_sha256",
            "validation_sha256", "gpu_pre_sha256", "gpu_post_sha256",
            "gpu_monitor_sha256", "gpu_observations_sha256",
        ):
            if not HEX256.fullmatch(values.get(key, "")):
                errors.append(f"{tag}: marker {key} is not sha256")
        if not HEX256.fullmatch(values.get("gpu_exclusivity_lease_id", "")):
            errors.append(f"{tag}: lease_id malformed")
        for key in (
            "gpu_physical_uuid", "cuda_visible_devices", "native_runtime_uuid",
        ):
            if not GPU_UUID_RE.fullmatch(values.get(key, "")):
                errors.append(f"{tag}: marker {key} is not a canonical GPU UUID")
        specs = [
            ("log_sha256", results / f"{tag}.log"),
            ("trace_sha256", results / f"{tag}_trace.csv"),
            ("validation_sha256", results / f"{tag}_validation.json"),
            ("gpu_pre_sha256", results / f"{tag}_gpu_pre.json"),
            ("gpu_post_sha256", results / f"{tag}_gpu_post.json"),
            ("gpu_monitor_sha256", results / f"{tag}_gpu_monitor.json"),
            ("gpu_observations_sha256", results / f"{tag}_gpu_observations.ndjson"),
        ]
        artifact_specs[tag] = specs
        for marker_key, path in specs:
            actual = hash_required(path, errors, hashes, f"{tag}:{path.name}")
            if values.get(marker_key) != actual:
                errors.append(f"{tag}: marker {marker_key} mismatch")
        validation = read_json(
            results / f"{tag}_validation.json", errors, f"validation:{tag}"
        )
        clean_pass(validation, f"validation:{tag}", errors)
        if validation.get("tag") != tag or validation.get("seq") != seq:
            errors.append(f"{tag}: strict validation identity mismatch")
        runtime_device = validation.get("device", {})
        if (
            runtime_device.get("runtime_ordinal") != 0
            or runtime_device.get("runtime_ordinal_zero") is not True
            or runtime_device.get("runtime_uuid") != values.get("native_runtime_uuid")
            or runtime_device.get("expected_lease_uuid")
            != values.get("gpu_physical_uuid")
        ):
            errors.append(f"{tag}: native runtime device identity mismatch")
        if matrix_by_identity.get((seq, tag)) != validation:
            errors.append(f"{tag}: validation_matrix record differs from bound validation JSON")
        common = {key: values.get(key, "") for key in marker_common_keys}
        if common_reference is None:
            common_reference = common
        elif common != common_reference:
            errors.append(f"{tag}: marker campaign-common signature mismatch")

    common_reference = common_reference or {}
    try:
        monitor_interval_ms: int | None = int(
            common_reference.get("gpu_monitor_interval_ms", "")
        )
        query_timeout_ms: int | None = int(
            common_reference.get("gpu_query_timeout_ms", "")
        )
    except ValueError:
        monitor_interval_ms = None
        query_timeout_ms = None
        errors.append("campaign monitor interval/query timeout marker is malformed")
    if (
        common_reference.get("gpu_monitor_coverage_model")
        != "bounded_interval_nvidia_smi_process_sampling"
    ):
        errors.append("campaign monitor coverage model marker mismatch")
    current_files = {
        "binary_sha256": binary,
        "source_sha256": script_dir / "dsa_native.cu",
        "bench_util_sha256": script_dir.parent / "common" / "bench_util.cuh",
        "cta_trace_sha256": script_dir.parent / "common" / "cta_trace.cuh",
        "build_sha256": script_dir.parent / "build.sh",
        "build_provenance_helper_sha256": script_dir / "build_provenance.py",
        "build_manifest_sha256": results / "dsa_build_manifest.json",
        "build_log_sha256": results / "build.log",
        "validator_sha256": script_dir / "validate_dsa_native.py",
        "binary_verifier_sha256": script_dir / "verify_dsa_binary.py",
        "binary_proof_sha256": results / "dsa_binary_proof.json",
        "gpu_exclusivity_helper_sha256": script_dir / "gpu_exclusivity.py",
        "aggregator_sha256": script_dir / "aggregate_dsa_native.py",
        "profile_validator_sha256": script_dir / "validate_dsa_profile.py",
        "profiler_evidence_sha256": script_dir / "profiler_evidence.py",
        "campaign_finalizer_sha256": Path(__file__).resolve(),
        "runner_sha256": runner,
    }
    for marker_key, path in current_files.items():
        actual = hash_required(path, errors, hashes, f"current:{marker_key}")
        if common_reference.get(marker_key) != actual:
            errors.append(f"campaign marker {marker_key} does not bind current file")

    build_manifest_path = results / "dsa_build_manifest.json"
    build_manifest, build_errors = build_provenance.validate_manifest(
        build_manifest_path, binary, script_dir / "dsa_native.cu",
        script_dir.parent / "common" / "bench_util.cuh",
        script_dir.parent / "common" / "cta_trace.cuh",
        script_dir.parent / "build.sh", results / "build.log",
        common_reference.get("target_arch", ""),
    )
    errors.extend(f"build_provenance: {error}" for error in build_errors)

    fingerprint = results / "device_fingerprint.txt"
    device = results / "device.txt"
    fingerprint_sha = hash_required(fingerprint, errors, hashes, "device_fingerprint")
    hash_required(device, errors, hashes, "device_identity")
    if common_reference.get("device_fingerprint_sha256") != fingerprint_sha:
        errors.append("device fingerprint hash mismatch")

    lease_path = results / "gpu_exclusivity_lease.json"
    lease_sha = hash_required(lease_path, errors, hashes, "gpu_exclusivity_lease")
    lease = read_json(lease_path, errors, "gpu_exclusivity_lease")
    clean_pass(lease, "gpu_exclusivity_lease", errors)
    lease_id = lease.get("lease_id")
    lease_target = lease.get("observation", {}).get("target_gpu")
    if (
        lease.get("schema") != 1
        or lease.get("kind") != "gpu_exclusivity_lease"
        or not HEX256.fullmatch(str(lease_id or ""))
        or lease.get("observation", {}).get("target_compute_processes") != []
    ):
        errors.append("GPU exclusivity lease contract mismatch")
    if common_reference.get("gpu_exclusivity_lease_id") != lease_id:
        errors.append("marker/lease id mismatch")
    if common_reference.get("gpu_exclusivity_lease_sha256") != lease_sha:
        errors.append("marker/lease hash mismatch")
    identity_path = results / "gpu_identity_current.json"
    identity = read_json(identity_path, errors, "gpu_identity_current")
    hash_required(identity_path, errors, hashes, "gpu_identity_current")
    clean_pass(identity, "gpu_identity_current", errors)
    if (
        identity.get("schema") != 1
        or identity.get("kind") != "gpu_identity"
        or identity.get("target_gpu") != lease_target
        or identity.get("accepted_timing") != 0
    ):
        errors.append("current GPU identity evidence mismatch")
    target_uuid_value = (lease_target or {}).get("uuid", "")
    target_uuid = target_uuid_value if isinstance(target_uuid_value, str) else ""
    expected_lock_path = f"/tmp/cta_pdl_gpu_{target_uuid}.lock"
    expected_lock_key_sha = hashlib.sha256(target_uuid.encode()).hexdigest()
    expected_lock_path_sha = hashlib.sha256(expected_lock_path.encode()).hexdigest()
    if (
        common_reference.get("gpu_global_lock_scope") != "target_uuid"
        or common_reference.get("gpu_global_lock_key_sha256") != expected_lock_key_sha
        or common_reference.get("gpu_global_lock_path_sha256") != expected_lock_path_sha
    ):
        errors.append("global target-GPU lock binding mismatch")
    target_index = lease.get("observation", {}).get("target_gpu", {}).get("index")
    if (
        not GPU_UUID_RE.fullmatch(target_uuid)
        or common_reference.get("gpu_index") != str(target_index)
        or common_reference.get("gpu_physical_uuid") != target_uuid
        or common_reference.get("cuda_visible_devices") != target_uuid
        or common_reference.get("native_runtime_uuid") != target_uuid
    ):
        errors.append(
            "nvidia-smi physical UUID, CUDA visibility UUID, and native runtime UUID mismatch"
        )
    try:
        device_text = device.read_text(encoding="utf-8")
        fingerprint_text = fingerprint.read_text(encoding="utf-8")
        if not target_uuid or target_uuid not in device_text or target_uuid not in fingerprint_text:
            errors.append("device identity is not bound to exclusivity target UUID")
    except OSError as exc:
        errors.append(f"cannot bind device identity to target UUID: {exc}")

    validate_checkpoint(
        results / "gpu_exclusivity_preflight.json", lease_id, lease_target,
        errors, hashes,
        "gpu_exclusivity_preflight", permit_lease=True,
    )
    for _, tag in expected:
        validate_checkpoint(
            results / f"{tag}_gpu_pre.json", lease_id, lease_target, errors, hashes,
            f"{tag}:gpu_pre",
        )
        validate_checkpoint(
            results / f"{tag}_gpu_post.json", lease_id, lease_target, errors, hashes,
            f"{tag}:gpu_post",
        )
        validate_monitor(
            results / f"{tag}_gpu_monitor.json",
            results / f"{tag}_gpu_observations.ndjson",
            lease_id, lease_target, tag, True,
            monitor_interval_ms, query_timeout_ms, errors, hashes,
            f"{tag}:gpu_monitor",
        )

    binary_proof_path = results / "dsa_binary_proof.json"
    binary_proof = read_json(binary_proof_path, errors, "binary_proof")
    clean_pass(binary_proof, "binary_proof", errors)
    if binary_proof.get("binary_sha256") != hashes.get("current:binary_sha256"):
        errors.append("binary proof does not bind current binary")
    kernels = binary_proof.get("kernels", {})
    if set(kernels) != {"dsaIndexer", "dsaTopk", "dsaAttention"}:
        errors.append("binary proof worker set mismatch")
    for name in ("dsaIndexer", "dsaTopk", "dsaAttention"):
        if not kernels.get(name, {}).get("ordering_pass"):
            errors.append(f"binary proof ordering failed for {name}")
    if not kernels.get("dsaIndexer", {}).get("pair_add_present"):
        errors.append("binary proof lacks explicit per-pair add")
    if not kernels.get("dsaIndexer", {}).get("shared_cache_present"):
        errors.append("binary proof lacks shared LUT cache")
    if not kernels.get("dsaIndexer", {}).get("register_tile_no_local_spill"):
        errors.append("binary proof lacks no-spill register-tile evidence")
    if not kernels.get("dsaIndexer", {}).get("register_tile_no_spill_complete"):
        errors.append("binary proof lacks PTX/resource/SASS no-spill closure")
    if not kernels.get("dsaAttention", {}).get("history_load_before_semantic_propagation"):
        errors.append("binary proof lacks history-load ordering")
    if not kernels.get("dsaAttention", {}).get("history_load_after_dependency_acquire"):
        errors.append("binary proof lacks post-acquire history-load evidence")
    if not kernels.get("dsaAttention", {}).get("history_load_to_semantic_straight_line"):
        errors.append("binary proof lacks branchless history-load evidence")
    if kernels.get("dsaAttention", {}).get("explicit_history_count_add", 0) <= 0:
        errors.append("binary proof lacks dynamic per-load history counter add")
    ptx = results / "dsa_native.ptx"
    resources = results / "dsa_native_resources.txt"
    target_arch = common_reference.get("target_arch", "")
    sass = results / f"dsa_native_{target_arch}.sass"
    ptx_sha = hash_required(ptx, errors, hashes, "binary_ptx")
    resource_sha = hash_required(resources, errors, hashes, "binary_resources")
    sass_sha = hash_required(sass, errors, hashes, "binary_sass")
    if binary_proof.get("ptx_sha256") != ptx_sha:
        errors.append("binary PTX hash mismatch")
    if binary_proof.get("resource_sha256") != resource_sha:
        errors.append("binary resource hash mismatch")
    if binary_proof.get("sass_proof", {}).get("sass_sha256") != sass_sha:
        errors.append("binary SASS hash mismatch")

    profile_summary: dict[str, Any] = {
        "required": bool(profile),
        "scope": "independent_4k_sidecar",
        "timing_matrix_included": False,
    }
    if profile:
        profile_base = results / "dsa_profile_seq4096"
        profile_validation_path = results / "dsa_profile_seq4096_profile_validation.json"
        profile_validation = read_json(
            profile_validation_path, errors, "profile_validation"
        )
        clean_pass(profile_validation, "profile_validation", errors)
        if (
            profile_validation.get("schema") != 1
            or set(profile_validation.get("required_ranges", [])) != PROFILE_RANGES
            or set(profile_validation.get("projection", {})) != PROFILE_RANGES
            or set(profile_validation.get("range_kernel_instances", {})) != PROFILE_RANGES
        ):
            errors.append("profile validation nine-range coverage mismatch")
        for range_name in PROFILE_RANGES:
            projected = profile_validation.get("projection", {}).get(range_name, {})
            mapped = profile_validation.get("range_kernel_instances", {}).get(range_name, {})
            if (
                projected.get("range_instances", 0) <= 0
                or projected.get("total_gpu_ops", 0) <= 0
                or not mapped
                or any(not isinstance(count, int) or count <= 0 for count in mapped.values())
            ):
                errors.append(f"profile range {range_name} has incomplete GPU mapping")
        global_workers = profile_validation.get("global_worker_instances", {})
        if (
            set(global_workers) != {"dsaIndexer", "dsaTopk", "dsaAttention"}
            or any(not isinstance(count, int) or count <= 0 for count in global_workers.values())
        ):
            errors.append("profile global worker coverage mismatch")
        profile_artifacts = {
            "nsys_rep": Path(str(profile_base) + ".nsys-rep"),
            "sqlite": Path(str(profile_base) + ".sqlite"),
            "profile_log": Path(str(profile_base) + ".log"),
            "globaltimer_trace": results / "dsa_profile_seq4096_globaltimer.csv",
            "cuda_gpu_kern_sum": results / "dsa_profile_seq4096_cuda_gpu_kern_sum.csv",
            "nvtx_gpu_proj_sum": results / "dsa_profile_seq4096_nvtx_gpu_proj_sum.csv",
            "nvtx_kern_sum": results / "dsa_profile_seq4096_nvtx_kern_sum.csv",
            "profile_validation": profile_validation_path,
        }
        profile_hashes: dict[str, str | None] = {}
        for label, path in profile_artifacts.items():
            profile_hashes[label] = hash_required(
                path, errors, hashes, f"profile:{label}"
            )
        reported = profile_validation.get("report_sha256", {})
        if reported.get("projection") != profile_hashes["nvtx_gpu_proj_sum"]:
            errors.append("profile projection hash mismatch")
        if reported.get("range_kernels") != profile_hashes["nvtx_kern_sum"]:
            errors.append("profile range-kernel hash mismatch")
        if reported.get("kernels") != profile_hashes["cuda_gpu_kern_sum"]:
            errors.append("profile kernel-summary hash mismatch")
        validate_checkpoint(
            results / "dsa_profile_seq4096_gpu_pre.json", lease_id, lease_target,
            errors, hashes,
            "profile:gpu_pre",
        )
        validate_checkpoint(
            results / "dsa_profile_seq4096_gpu_post.json", lease_id, lease_target,
            errors, hashes,
            "profile:gpu_post",
        )
        validate_monitor(
            results / "dsa_profile_seq4096_gpu_monitor.json",
            results / "dsa_profile_seq4096_gpu_observations.ndjson",
            lease_id, lease_target, "nsys_4k_sidecar", True,
            monitor_interval_ms, query_timeout_ms, errors, hashes,
            "profile:gpu_monitor",
        )

        ncu_path = results / "ncu_permission.json"
        ncu = read_json(ncu_path, errors, "ncu_permission")
        hash_required(ncu_path, errors, hashes, "ncu_permission")
        available = ncu.get("hardware_counters_available") is True
        denied = ncu.get("permission_denied") is True
        rc = ncu.get("returncode")
        if (
            ncu.get("schema") != 1
            or ncu.get("purpose") != "hardware_counter_permission_probe"
            or ncu.get("measurement_admission") != "not_a_timing_sample"
            or ncu.get("launch_count") != 1
            or ncu.get("timing_admission_affected") is not False
            or available == denied
            or not isinstance(rc, int)
            or (rc == 0 and not available)
            or (rc != 0 and not denied)
        ):
            errors.append("ncu permission evidence is missing/ambiguous")
        for label, key in (("ncu_stdout", "stdout_path"), ("ncu_stderr", "stderr_path")):
            expected_path = results / ("ncu_permission.stdout" if label.endswith("stdout") else "ncu_permission.stderr")
            if Path(str(ncu.get(key, ""))) != expected_path:
                errors.append(f"{label} path mismatch")
            hash_required(expected_path, errors, hashes, label)
        validate_checkpoint(
            results / "dsa_ncu_gpu_pre.json", lease_id, lease_target, errors, hashes,
            "ncu:gpu_pre",
        )
        validate_checkpoint(
            results / "dsa_ncu_gpu_post.json", lease_id, lease_target, errors, hashes,
            "ncu:gpu_post",
        )
        validate_monitor(
            results / "dsa_ncu_gpu_monitor.json",
            results / "dsa_ncu_gpu_observations.ndjson",
            lease_id, lease_target, "ncu_permission_probe", False,
            monitor_interval_ms, query_timeout_ms, errors, hashes,
            "ncu:gpu_monitor",
        )
        profile_summary.update({
            "status": profile_validation.get("status"),
            "required_ranges": sorted(PROFILE_RANGES),
            "ncu_classification": "available" if available else "permission_denied",
            "artifact_sha256": profile_hashes,
        })
    else:
        profile_summary["status"] = "NOT_REQUESTED_FAST_ONLY"

    terminal_path = results / "terminal_status.json"
    terminal = read_json(terminal_path, errors, "terminal_status")
    terminal_sha = hash_required(terminal_path, errors, hashes, "terminal_status")
    if (
        terminal.get("schema") != 1
        or terminal.get("status") != "PASS"
        or terminal.get("errors") != []
        or terminal.get("campaign") != "tier5_native_dsa"
        or terminal.get("fast") != fast
        or terminal.get("profile") != profile
        or set(terminal.get("expected_tags", [])) != expected_tag_set
    ):
        errors.append("terminal status contract mismatch")
    if terminal.get("runner_sha256") != hashes.get("current:runner_sha256"):
        errors.append("terminal status runner hash mismatch")
    terminal_evidence = terminal.get("evidence_sha256", {})
    expected_terminal_evidence = {
        "validation_matrix": hashes.get("validation_matrix"),
        "binary_proof": hashes.get("current:binary_proof_sha256"),
        "build_provenance": hashes.get("current:build_manifest_sha256"),
        "gpu_exclusivity_lease": lease_sha,
    }
    if profile:
        expected_terminal_evidence.update({
            "profile_validation": hashes.get("profile:profile_validation"),
            "ncu_permission": hashes.get("ncu_permission"),
        })
    if terminal_evidence != expected_terminal_evidence:
        errors.append("terminal status evidence hash closure mismatch")

    return {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "campaign": "tier5_native_dsa",
        "claim_boundary": "synthetic work-complete dependency proxy",
        "section_9_status": "PARTIAL",
        "fast": fast,
        "profile": profile,
        "expected_points": [
            {"seq": seq, "tag": tag} for seq, tag in expected
        ],
        "timing_matrix": {
            "status": matrix.get("status"),
            "sha256": hashes.get("validation_matrix"),
            "points": len(matrix_records),
        },
        "binary_proof": {
            "status": binary_proof.get("status"),
            "sha256": hashes.get("current:binary_proof_sha256"),
        },
        "build_provenance": {
            "status": build_manifest.get("status"),
            "sha256": hashes.get("current:build_manifest_sha256"),
        },
        "profile_sidecar": profile_summary,
        "device_identity": {
            "target_gpu": lease.get("observation", {}).get("target_gpu"),
            "cuda_visible_devices": common_reference.get("cuda_visible_devices"),
            "native_runtime_uuid": common_reference.get("native_runtime_uuid"),
            "runtime_ordinal": 0,
            "device_sha256": hashes.get("device_identity"),
            "fingerprint_sha256": hashes.get("device_fingerprint"),
        },
        "gpu_exclusivity": {
            "status": lease.get("status"),
            "lease_id": lease_id,
            "lease_sha256": lease_sha,
            "coverage_model": common_reference.get("gpu_monitor_coverage_model"),
            "coverage_limit": (
                "foreign GPU processes wholly between completed samples may not be observed"
            ),
            "poll_interval_ms": monitor_interval_ms,
            "query_timeout_ms": query_timeout_ms,
        },
        "terminal_status_sha256": terminal_sha,
        "artifact_sha256": hashes,
        "accepted_timing": 1 if not errors else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fast", type=int, choices=(0, 1), required=True)
    parser.add_argument("--profile", type=int, choices=(0, 1), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize_campaign(
        args.results, args.fast, args.profile, args.binary, args.runner
    )
    atomic_json(args.json, payload)
    print(
        "DSA_CAMPAIGN_ADMISSION "
        f"schema=1 status={payload['status']} errors={len(payload['errors'])} "
        f"accepted_timing={payload['accepted_timing']}"
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
