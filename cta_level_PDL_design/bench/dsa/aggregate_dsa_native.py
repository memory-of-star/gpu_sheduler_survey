#!/usr/bin/env python3
"""Fail-closed timing-matrix aggregation; this is not final campaign admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import finalize_dsa_campaign


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_marker(path: Path, errors: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read marker {path.name}: {exc}")
        return values
    for line_number, line in enumerate(lines, 1):
        if "=" not in line:
            errors.append(f"{path.name}:{line_number}: malformed marker line")
            continue
        key, value = line.split("=", 1)
        if not key or key in values:
            errors.append(f"{path.name}:{line_number}: duplicate/empty marker key {key!r}")
            continue
        values[key] = value
    return values


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fast", choices=("0", "1"), required=True)
    parser.add_argument("--log-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    expected = FAST if args.fast == "1" else FORMAL
    expected_tags = {tag for _, tag in expected}
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    logs: list[str] = []

    for rejected in ("formal_rejection.json", "REJECTED.md"):
        if (args.results / rejected).exists():
            errors.append(f"rejected results directory contains {rejected}")

    observed_done = {path.stem for path in args.results.glob("dsa_*.done")}
    if observed_done != expected_tags:
        errors.append(
            f"done set mismatch: observed={sorted(observed_done)} "
            f"expected={sorted(expected_tags)}"
        )
    observed_invalid = {path.stem for path in args.results.glob("dsa_*.invalid")}
    if observed_invalid & expected_tags:
        errors.append(f"expected points still invalid: {sorted(observed_invalid & expected_tags)}")

    required_signature_hashes = (
        "device_fingerprint_sha256", "binary_sha256", "source_sha256",
        "bench_util_sha256", "cta_trace_sha256",
        "build_sha256", "build_provenance_helper_sha256",
        "build_manifest_sha256", "build_log_sha256",
        "validator_sha256", "binary_verifier_sha256",
        "binary_proof_sha256", "aggregator_sha256", "profile_validator_sha256",
        "profiler_evidence_sha256", "campaign_finalizer_sha256",
        "gpu_exclusivity_lease_sha256", "gpu_exclusivity_helper_sha256",
        "gpu_global_lock_key_sha256", "gpu_global_lock_path_sha256",
        "runner_sha256", "argv_sha256",
    )
    for seq, tag in expected:
        marker = args.results / f"{tag}.done"
        log = args.results / f"{tag}.log"
        trace = args.results / f"{tag}_trace.csv"
        validation = args.results / f"{tag}_validation.json"
        gpu_pre = args.results / f"{tag}_gpu_pre.json"
        gpu_post = args.results / f"{tag}_gpu_post.json"
        gpu_monitor = args.results / f"{tag}_gpu_monitor.json"
        gpu_observations = args.results / f"{tag}_gpu_observations.ndjson"
        for path in (
            marker, log, trace, validation, gpu_pre, gpu_post,
            gpu_monitor, gpu_observations,
        ):
            if not path.is_file():
                errors.append(f"{tag}: missing {path.name}")
        if not all(path.is_file() for path in (
            marker, log, trace, validation, gpu_pre, gpu_post,
            gpu_monitor, gpu_observations,
        )):
            continue

        values = parse_marker(marker, errors)
        expected_marker = {
            "marker_schema": "2", "fast": args.fast, "tag": tag, "seq": str(seq),
            "mode_count": "4",
            "mode_order": "floor,wave_floor,impl,ceiling",
            "sample_order": "cyclic_latin_4",
            "invocations_per_point": "20" if args.fast == "1" else "140",
        }
        for key, value in expected_marker.items():
            if values.get(key) != value:
                errors.append(f"{tag}: marker {key} mismatch")
        if args.fast == "0" and values.get("profile") != "1":
            errors.append(f"{tag}: formal marker profile mismatch")
        if args.fast == "1" and values.get("profile") not in ("0", "1"):
            errors.append(f"{tag}: FAST marker profile mismatch")
        if not values.get("target_arch", "").startswith("sm_"):
            errors.append(f"{tag}: marker target_arch missing/malformed")
        if values.get("nvtx_range_count") != "9":
            errors.append(f"{tag}: marker nvtx_range_count mismatch")
        if not re.fullmatch(r"[1-9][0-9]*(?:s|m|h)?", values.get("step_timeout_spec", "")):
            errors.append(f"{tag}: marker step_timeout_spec malformed")
        if values.get("gpu_global_lock_scope") != "target_uuid":
            errors.append(f"{tag}: marker global GPU lock scope mismatch")
        physical_uuid = values.get("gpu_physical_uuid", "")
        if not GPU_UUID_RE.fullmatch(physical_uuid):
            errors.append(f"{tag}: marker physical GPU UUID malformed")
        if values.get("cuda_visible_devices") != physical_uuid:
            errors.append(f"{tag}: CUDA_VISIBLE_DEVICES is not the physical GPU UUID")
        if values.get("native_runtime_uuid") != physical_uuid:
            errors.append(f"{tag}: native runtime UUID is not the physical GPU UUID")
        try:
            monitor_interval_ms = int(values.get("gpu_monitor_interval_ms", ""))
            query_timeout_ms = int(values.get("gpu_query_timeout_ms", ""))
        except ValueError:
            monitor_interval_ms = -1
            query_timeout_ms = -1
        if not 10 <= monitor_interval_ms <= 100:
            errors.append(f"{tag}: marker monitor interval is outside 10..100ms")
        if not 100 <= query_timeout_ms <= 5000:
            errors.append(f"{tag}: marker query timeout is outside 100..5000ms")
        if (
            values.get("gpu_monitor_coverage_model")
            != "bounded_interval_nvidia_smi_process_sampling"
        ):
            errors.append(f"{tag}: marker monitor coverage model mismatch")
        if not HEX256.fullmatch(values.get("gpu_exclusivity_lease_id", "")):
            errors.append(f"{tag}: marker gpu_exclusivity_lease_id malformed")
        for key in required_signature_hashes:
            if not HEX256.fullmatch(values.get(key, "")):
                errors.append(f"{tag}: marker {key} is not sha256")
        for key, path in (
            ("build_provenance_helper_sha256", Path(__file__).resolve().parent / "build_provenance.py"),
            ("build_manifest_sha256", args.results / "dsa_build_manifest.json"),
            ("build_log_sha256", args.results / "build.log"),
            (
                "campaign_finalizer_sha256",
                Path(__file__).resolve().parent / "finalize_dsa_campaign.py",
            ),
        ):
            try:
                actual = sha256(path)
            except OSError as exc:
                errors.append(f"{tag}: cannot hash {path.name} ({exc})")
                continue
            if values.get(key) != actual:
                errors.append(f"{tag}: marker {key} mismatch")
        for key, path in (
            ("log_sha256", log), ("trace_sha256", trace),
            ("validation_sha256", validation),
            ("gpu_pre_sha256", gpu_pre), ("gpu_post_sha256", gpu_post),
            ("gpu_monitor_sha256", gpu_monitor),
            ("gpu_observations_sha256", gpu_observations),
        ):
            observed = values.get(key, "")
            actual = sha256(path)
            if observed != actual:
                errors.append(f"{tag}: marker {key} mismatch")

        monitor_target: dict[str, Any] | None = None
        for label, path in (("gpu_pre", gpu_pre), ("gpu_post", gpu_post)):
            try:
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                checkpoint_target = checkpoint.get("observation", {}).get(
                    "target_gpu"
                )
                if (
                    checkpoint.get("schema") != 1
                    or checkpoint.get("kind") != "gpu_exclusivity_checkpoint"
                    or checkpoint.get("status") != "PASS"
                    or checkpoint.get("errors") != []
                    or checkpoint.get("lease_id") != values.get("gpu_exclusivity_lease_id")
                    or checkpoint.get("observation", {}).get("target_gpu", {}).get("uuid")
                    != physical_uuid
                    or checkpoint.get("observation", {}).get("target_compute_processes") != []
                ):
                    errors.append(f"{tag}: {label} exclusivity checkpoint mismatch")
                if not isinstance(checkpoint_target, dict):
                    errors.append(f"{tag}: {label} target GPU identity missing")
                elif monitor_target is None:
                    monitor_target = checkpoint_target
                elif checkpoint_target != monitor_target:
                    errors.append(f"{tag}: pre/post target GPU identity mismatch")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{tag}: invalid {label} exclusivity JSON ({exc})")

        try:
            monitor = json.loads(gpu_monitor.read_text(encoding="utf-8"))
            observations_sha = sha256(gpu_observations)
            if (
                monitor.get("schema") != 1
                or monitor.get("kind") != "gpu_exclusivity_monitor"
                or monitor.get("status") != "PASS"
                or monitor.get("errors") != []
                or monitor.get("accepted_timing") != 0
                or monitor.get("phase") != tag
                or monitor.get("lease_id") != values.get("gpu_exclusivity_lease_id")
                or monitor.get("target_gpu", {}).get("uuid") != physical_uuid
                or monitor.get("observations_sha256") != observations_sha
                or Path(str(monitor.get("observations_path", ""))) != gpu_observations
                or monitor.get("observation_count", 0) < 2
                or monitor.get("allowed_observation_count", 0) <= 0
                or not monitor.get("allowed_processes")
                or monitor.get("require_allowed_process") is not True
                or monitor.get("foreign_processes_detected") is not False
                or monitor.get("query_failure_detected") is not False
                or monitor.get("poll_interval_ms") != monitor_interval_ms
                or monitor.get("query_timeout_ms") != query_timeout_ms
                or monitor.get("coverage_model")
                != "bounded_interval_nvidia_smi_process_sampling"
                or monitor.get("start_barrier_complete") is not True
                or monitor.get("ready_record_written") is not True
                or monitor.get("baseline_observation_sequence") != 0
            ):
                errors.append(f"{tag}: runtime GPU monitor evidence mismatch")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{tag}: invalid runtime GPU monitor JSON ({exc})")
        finalize_dsa_campaign.validate_monitor(
            gpu_monitor,
            gpu_observations,
            values.get("gpu_exclusivity_lease_id"),
            monitor_target,
            tag,
            True,
            monitor_interval_ms,
            query_timeout_ms,
            errors,
            {},
            f"{tag}:runtime_monitor_deep_contract",
        )

        try:
            record = json.loads(validation.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("top-level JSON is not an object")
            if (
                record.get("schema") != 1
                or record.get("status") != "PASS"
                or record.get("errors") != []
                or record.get("tag") != tag
                or record.get("seq") != seq
                or record.get("device", {}).get("runtime_ordinal") != 0
                or record.get("device", {}).get("runtime_ordinal_zero") is not True
                or record.get("device", {}).get("runtime_uuid") != physical_uuid
                or record.get("device", {}).get("expected_lease_uuid") != physical_uuid
            ):
                errors.append(f"{tag}: strict validation JSON identity/status mismatch")
            records.append(record)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{tag}: invalid validation JSON ({exc})")

        try:
            log_text = log.read_text(encoding="utf-8")
            if log_text.count("CONFIG_DSA ") != 1 or log_text.count("SUMMARY_DSA ") != 1:
                errors.append(f"{tag}: log singleton ledger mismatch")
            if f"tag={tag} seq={seq}" not in log_text:
                errors.append(f"{tag}: log identity missing")
            logs.append(log_text if log_text.endswith("\n") else log_text + "\n")
        except OSError as exc:
            errors.append(f"{tag}: cannot read log ({exc})")

    payload = {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "fast": int(args.fast),
        "admission_scope": "timing_matrix_only",
        "campaign_complete": False,
        "expected_tags": sorted(expected_tags),
        "records": records,
    }
    aggregate_log = "".join(logs)
    if errors:
        aggregate_log += "AGGREGATE_DSA schema=1 status=FAIL errors=%d\n" % len(errors)
        aggregate_log += "".join(f"AGGREGATE_ERROR {error}\n" for error in errors)
    else:
        aggregate_log += "AGGREGATE_DSA schema=1 status=PASS errors=0 points=%d\n" % len(records)
    atomic_text(args.log_out, aggregate_log)
    atomic_json(args.json_out, payload)
    print(
        f"DSA_AGGREGATION schema=1 status={payload['status']} "
        f"errors={len(errors)} points={len(records)}"
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
