#!/usr/bin/env python3
"""CPU-only negative tests for Tier-5 admission, proof, resume, and aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest
from unittest import mock
from pathlib import Path

import verify_dsa_binary
import profiler_evidence
import finalize_dsa_campaign
import build_provenance
from validate_dsa_native import (
    analyze_trace,
    canonical_gpu_uuid_from_bytes,
    device_identity_errors,
    history_load_contract_errors,
    latin_order,
    pair_contract_errors,
    pair_low16_equivalence_cases,
)
from validate_dsa_profile import RANGE_KERNELS


BASE = Path(__file__).resolve().parent
RUNNER = BASE / "run_dsa_chain.sh"
AGGREGATOR = BASE / "aggregate_dsa_native.py"
PROFILE_VALIDATOR = BASE / "validate_dsa_profile.py"
NATIVE_VALIDATOR = BASE / "validate_dsa_native.py"
TAG = "dsa_exact_seq4096"
GPU_TEST_UUID = "GPU-11111111-2222-3333-4444-555555555555"
GPU_MONITOR_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*argv: str, cwd: Path = BASE, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def make_aggregate_fixture(root: Path) -> None:
    log = root / f"{TAG}.log"
    trace = root / f"{TAG}_trace.csv"
    validation = root / f"{TAG}_validation.json"
    marker = root / f"{TAG}.done"
    gpu_pre = root / f"{TAG}_gpu_pre.json"
    gpu_post = root / f"{TAG}_gpu_post.json"
    gpu_monitor = root / f"{TAG}_gpu_monitor.json"
    gpu_observations = root / f"{TAG}_gpu_observations.ndjson"
    build_manifest = root / "dsa_build_manifest.json"
    build_log = root / "build.log"
    lease_id = "c" * 64
    target = {"index": 0, "uuid": GPU_TEST_UUID, "name": "Synthetic GPU"}
    log.write_text(
        f"CONFIG_DSA semantics=1 tag={TAG} seq=4096\n"
        f"SUMMARY_DSA semantics=1 tag={TAG} seq=4096\n",
        encoding="utf-8",
    )
    trace.write_text("synthetic-trace\n", encoding="utf-8")
    validation.write_text(
        json.dumps({
            "schema": 1, "status": "PASS", "errors": [], "tag": TAG, "seq": 4096,
            "device": {
                "runtime_ordinal": 0, "runtime_ordinal_zero": True,
                "runtime_uuid": GPU_TEST_UUID,
                "expected_lease_uuid": GPU_TEST_UUID,
            },
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "schema": 1, "kind": "gpu_exclusivity_checkpoint", "status": "PASS",
        "errors": [], "lease_id": lease_id,
        "accepted_timing": 0,
        "observation": {"target_gpu": target, "target_compute_processes": []},
    }
    gpu_pre.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
    gpu_post.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
    make_monitor_fixture(
        gpu_monitor, gpu_observations, TAG, lease_id, target,
        require_allowed=True, include_residual=True,
    )
    build_manifest.write_text('{"synthetic":"build provenance"}\n', encoding="utf-8")
    build_log.write_text("synthetic build log\n", encoding="utf-8")
    hash_fields = (
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
    lines = [
        "marker_schema=2", "fast=1", "profile=0", f"tag={TAG}", "seq=4096",
        "target_arch=sm_100", "nvtx_range_count=9",
        "mode_count=4", "mode_order=floor,wave_floor,impl,ceiling",
        "sample_order=cyclic_latin_4", "invocations_per_point=20",
        "step_timeout_spec=120s",
        "gpu_index=0", f"gpu_physical_uuid={GPU_TEST_UUID}",
        f"cuda_visible_devices={GPU_TEST_UUID}",
        f"native_runtime_uuid={GPU_TEST_UUID}",
        f"gpu_exclusivity_lease_id={lease_id}",
        "gpu_global_lock_scope=target_uuid",
        "gpu_monitor_interval_ms=50", "gpu_query_timeout_ms=2000",
        "gpu_monitor_coverage_model=bounded_interval_nvidia_smi_process_sampling",
    ]
    signature_values = {key: "a" * 64 for key in hash_fields}
    signature_values.update({
        "build_provenance_helper_sha256": sha(BASE / "build_provenance.py"),
        "build_manifest_sha256": sha(build_manifest),
        "build_log_sha256": sha(build_log),
        "campaign_finalizer_sha256": sha(BASE / "finalize_dsa_campaign.py"),
    })
    lines.extend(f"{key}={signature_values[key]}" for key in hash_fields)
    lines.extend((
        f"log_sha256={sha(log)}", f"trace_sha256={sha(trace)}",
        f"validation_sha256={sha(validation)}",
        f"gpu_pre_sha256={sha(gpu_pre)}", f"gpu_post_sha256={sha(gpu_post)}",
        f"gpu_monitor_sha256={sha(gpu_monitor)}",
        f"gpu_observations_sha256={sha(gpu_observations)}",
    ))
    marker.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_monitor_fake(path: Path) -> None:
    path.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        case "$1" in
          --query-gpu=index,uuid,name)
            echo '0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, Synthetic GPU'
            ;;
          --query-compute-apps=*)
            if [ "${FAKE_GPU_MODE:-allowed}" = "timeout" ] && [[ "$1" == *",name,"* ]]; then
              sleep 0.3
              exit 1
            fi
            pid=$(cat "${FAKE_WATCH_PID_FILE}")
            state="missing"
            if [ -r "/proc/${pid}/stat" ]; then
              state=$(awk '{print $3}' "/proc/${pid}/stat")
            fi
            if [ "${FAKE_GPU_MODE:-allowed}" = "allowed" ] \
               && [ "${state}" != "T" ] && [ "${state}" != "Z" ] \
               && [ "${state}" != "missing" ]; then
              echo "${pid}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, allowed-worker, 64"
            elif [ "${FAKE_GPU_MODE:-allowed}" = "persistent_residual" ] \
                 && [ "${state}" != "T" ] && [ "${state}" != "Z" ] \
                 && [ "${state}" != "missing" ]; then
              echo "${pid}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, allowed-worker, 64"
            elif [ "${FAKE_GPU_MODE:-allowed}" = "persistent_residual" ] \
                 && [ "${state}" = "missing" ]; then
              echo "${pid}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, [No data], 64"
            elif [ "${FAKE_GPU_MODE:-allowed}" = "exit_residual" ] \
                 && [ "${state}" != "T" ] && [ "${state}" != "Z" ] \
                 && [ "${state}" != "missing" ]; then
              echo "${pid}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, allowed-worker, 64"
            elif [ "${FAKE_GPU_MODE:-allowed}" = "exit_residual" ] \
                 && [ "${state}" = "missing" ]; then
              residual_count=0
              if [ -r "${FAKE_RESIDUAL_COUNT_FILE}" ]; then
                residual_count=$(cat "${FAKE_RESIDUAL_COUNT_FILE}")
              fi
              if [ "${residual_count}" -lt 2 ]; then
                echo "${pid}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, [No data], 64"
                residual_count=$((residual_count + 1))
                printf '%s\\n' "${residual_count}" > "${FAKE_RESIDUAL_COUNT_FILE}"
              fi
            elif [ "${FAKE_GPU_MODE:-allowed}" = "unknown_residual" ]; then
              echo "${FAKE_UNKNOWN_PID}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, [No data], 64"
            elif [ "${FAKE_GPU_MODE:-allowed}" = "foreign" ] \
                 && [ "${state}" != "T" ] && [ "${state}" != "Z" ] \
                 && [ "${state}" != "missing" ]; then
              echo "${FAKE_FOREIGN_PID}, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, foreign-worker, 64"
            fi
            ;;
          *)
            echo 'unexpected query' >&2
            exit 2
            ;;
        esac
    """), encoding="utf-8")
    path.chmod(0o755)


def make_monitor_lease(path: Path) -> tuple[str, dict[str, object]]:
    lease_id = "f" * 64
    target: dict[str, object] = {
        "index": 0, "uuid": GPU_MONITOR_UUID, "name": "Synthetic GPU",
    }
    write_json(path, {
        "schema": 1, "kind": "gpu_exclusivity_lease", "status": "PASS",
        "errors": [], "accepted_timing": 0, "lease_id": lease_id,
        "observation": {"target_gpu": target, "target_compute_processes": []},
    })
    return lease_id, target


def make_monitor_fixture(
    manifest: Path, observations: Path, phase: str, lease_id: str,
    target: dict[str, object], *, require_allowed: bool,
    include_residual: bool = False,
) -> None:
    raw_allowed = {
        "pid": 777, "gpu_uuid": target["uuid"], "name": "allowed-worker",
        "used_memory": "64",
    }
    allowed_process = {**raw_allowed, "proc_start_ticks": 12345}
    raw_residual = {**raw_allowed, "name": "[No data]"}
    records = []
    record_count = 6 if include_residual else 4
    for sequence in range(record_count):
        allowed = [allowed_process] if sequence == 1 and require_allowed else []
        residual = []
        target_processes = [raw_allowed] if allowed else []
        if include_residual and sequence in (2, 3):
            residual_number = sequence - 1
            residual = [{
                **raw_residual,
                "proc_start_ticks": None,
                "previous_allowed_start_ticks": 12345,
                "classification": "allowed_post_exit_nvidia_smi_residual",
                "residual_observation_number": residual_number,
                "residual_observation_limit": 4,
            }]
            target_processes = [raw_residual]
        records.append({
            "schema": 1, "sequence": sequence,
            "observed_at": f"synthetic-{sequence}",
            "query_started_monotonic_ns": sequence * 1_000_000,
            "query_finished_monotonic_ns": sequence * 1_000_000 + 500_000,
            "query_duration_ms": 0.5,
            "query_errors": [],
            "observation": {
                "target_gpu": target,
                "target_compute_processes": target_processes,
            },
            "allowed_target_processes": allowed,
            "allowed_exit_residual_processes": residual,
            "foreign_target_processes": [],
        })
    observations.write_text("".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ), encoding="utf-8")
    value = {
        "schema": 1, "kind": "gpu_exclusivity_monitor", "status": "PASS",
        "errors": [], "accepted_timing": 0, "phase": phase,
        "lease_id": lease_id, "target_gpu": target,
        "observation_count": record_count,
        "allowed_observation_count": 1 if require_allowed else 0,
        "allowed_exit_residual_observation_count": 2 if include_residual else 0,
        "allowed_processes": [
            {"pid": 777, "proc_start_ticks": 12345}
        ] if require_allowed else [],
        "allowed_exit_residual_processes": ([{
            "pid": 777,
            "previous_allowed_start_ticks": 12345,
            "residual_observation_count": 2,
        }] if include_residual else []),
        "allowed_exit_residual_max_observations_per_pid": 4,
        "max_allowed_exit_residual_observations_observed": (
            2 if include_residual else 0
        ),
        "exit_residual_policy": (
            "previously_observed_allowed_same_pid_start_ticks_and_exact_No_data_"
            "with_proc_missing_only_bounded_per_pid"
        ),
        "require_allowed_process": require_allowed,
        "foreign_processes_detected": False,
        "query_failure_detected": False,
        "poll_interval_ms": 50, "query_timeout_ms": 2000,
        "coverage_model": "bounded_interval_nvidia_smi_process_sampling",
        "coverage_limit": (
            "foreign GPU processes wholly between completed samples may not be observed"
        ),
        "start_barrier_complete": True,
        "ready_record_written": True,
        "baseline_observation_sequence": 0,
        "observations_path": str(observations),
        "observations_sha256": sha(observations),
    }
    write_json(manifest, value)


def make_final_admission_fixture(root: Path) -> Path:
    """Build a fully hash-closed synthetic FAST+profile evidence graph."""
    root.mkdir(parents=True, exist_ok=True)
    binary = root / "dsa_native"
    binary.write_bytes(b"synthetic-sm100-binary")
    lease_id = "d" * 64
    # Physical nvidia-smi index 3 intentionally differs from the one visible CUDA ordinal 0.
    target = {"index": 3, "uuid": GPU_TEST_UUID, "name": "Synthetic GPU"}
    build_log = root / "build.log"
    build_log.write_text(
        "-- dsa/dsa_native.cu -> dsa/dsa_native\n== build OK\n",
        encoding="utf-8",
    )
    synthetic_nvcc = Path("/synthetic/cuda/bin/nvcc")
    synthetic_version = "Cuda compilation tools, release synthetic\n"
    build_manifest = {
        "schema": 1, "kind": "dsa_native_build_provenance",
        "status": "PASS", "errors": [], "target_arch": "sm_100",
        "build_invocation": [str(BASE.parent / "build.sh")],
        "build_environment": {
            "ARCH": "sm_100", "NVCC": str(synthetic_nvcc), "NVTX_INCLUDE": "",
        },
        "nvcc_path": str(synthetic_nvcc),
        "nvcc_version": synthetic_version,
        "nvcc_version_sha256": hashlib.sha256(synthetic_version.encode()).hexdigest(),
        "dsa_compile_argv": build_provenance.compile_argv(
            synthetic_nvcc, "sm_100", ""
        ),
        "input_sha256": {
            "dsa_native.cu": sha(BASE / "dsa_native.cu"),
            "bench_util.cuh": sha(BASE.parent / "common" / "bench_util.cuh"),
            "cta_trace.cuh": sha(BASE.parent / "common" / "cta_trace.cuh"),
            "build.sh": sha(BASE.parent / "build.sh"),
        },
        "binary_path": str(binary), "binary_sha256": sha(binary),
        "build_log_path": str(build_log), "build_log_sha256": sha(build_log),
    }
    write_json(root / "dsa_build_manifest.json", build_manifest)
    observation = {
        "target_gpu": target,
        "all_compute_processes": [],
        "target_compute_processes": [],
    }
    lease = {
        "schema": 1, "kind": "gpu_exclusivity_lease", "status": "PASS",
        "errors": [], "accepted_timing": 0, "lease_id": lease_id,
        "observation": observation,
    }
    checkpoint = {
        "schema": 1, "kind": "gpu_exclusivity_checkpoint", "status": "PASS",
        "errors": [], "accepted_timing": 0, "lease_id": lease_id,
        "observation": observation,
    }
    write_json(root / "gpu_exclusivity_lease.json", lease)
    write_json(root / "gpu_exclusivity_preflight.json", lease)
    write_json(root / "gpu_identity_current.json", {
        "schema": 1, "kind": "gpu_identity", "status": "PASS",
        "errors": [], "accepted_timing": 0, "target_gpu": target,
    })
    for name in (
        f"{TAG}_gpu_pre.json", f"{TAG}_gpu_post.json",
        "dsa_profile_seq4096_gpu_pre.json", "dsa_profile_seq4096_gpu_post.json",
        "dsa_ncu_gpu_pre.json", "dsa_ncu_gpu_post.json",
    ):
        write_json(root / name, checkpoint)
    make_monitor_fixture(
        root / f"{TAG}_gpu_monitor.json",
        root / f"{TAG}_gpu_observations.ndjson",
        TAG, lease_id, target, require_allowed=True, include_residual=True,
    )
    make_monitor_fixture(
        root / "dsa_profile_seq4096_gpu_monitor.json",
        root / "dsa_profile_seq4096_gpu_observations.ndjson",
        "nsys_4k_sidecar", lease_id, target, require_allowed=True,
    )
    make_monitor_fixture(
        root / "dsa_ncu_gpu_monitor.json",
        root / "dsa_ncu_gpu_observations.ndjson",
        "ncu_permission_probe", lease_id, target, require_allowed=False,
    )
    (root / "device_fingerprint.txt").write_text(
        f"{GPU_TEST_UUID}, Synthetic GPU, 10.0, 999.0\n", encoding="utf-8"
    )
    (root / "device.txt").write_text(
        f"uuid, name\n{GPU_TEST_UUID}, Synthetic GPU\n", encoding="utf-8"
    )

    ptx = root / "dsa_native.ptx"
    resources = root / "dsa_native_resources.txt"
    sass = root / "dsa_native_sm_100.sass"
    ptx.write_text("synthetic ptx proof\n", encoding="utf-8")
    resources.write_text("synthetic resources\n", encoding="utf-8")
    sass.write_text("synthetic sass proof\n", encoding="utf-8")
    binary_proof = {
        "schema": 2, "status": "PASS", "errors": [],
        "binary_sha256": sha(binary), "ptx_sha256": sha(ptx),
        "resource_sha256": sha(resources),
        "kernels": {
            "dsaIndexer": {
                "ordering_pass": True, "pair_add_present": True,
                "shared_cache_present": True, "register_tile_no_local_spill": True,
                "register_tile_no_spill_complete": True,
            },
            "dsaTopk": {"ordering_pass": True},
            "dsaAttention": {
                "ordering_pass": True,
                "history_load_before_semantic_propagation": True,
                "history_load_after_dependency_acquire": True,
                "history_load_to_semantic_straight_line": True,
                "explicit_history_count_add": 1,
            },
        },
        "sass_proof": {"sass_sha256": sha(sass)},
    }
    write_json(root / "dsa_binary_proof.json", binary_proof)

    log = root / f"{TAG}.log"
    trace = root / f"{TAG}_trace.csv"
    validation_path = root / f"{TAG}_validation.json"
    log.write_text("synthetic admitted log\n", encoding="utf-8")
    trace.write_text("synthetic admitted trace\n", encoding="utf-8")
    validation = {
        "schema": 1, "status": "PASS", "errors": [], "tag": TAG, "seq": 4096,
        "device": {
            "runtime_ordinal": 0, "runtime_ordinal_zero": True,
            "runtime_uuid": GPU_TEST_UUID,
            "name_hex": "Synthetic GPU".encode().hex(),
            "cc_major": 10, "cc_minor": 0, "sms": 160,
            "expected_lease_uuid": GPU_TEST_UUID,
        },
    }
    write_json(validation_path, validation)
    matrix = {
        "schema": 1, "status": "PASS", "errors": [], "fast": 1,
        "expected_tags": [TAG], "records": [validation],
        "admission_scope": "timing_matrix_only", "campaign_complete": False,
    }
    write_json(root / "validation_matrix.json", matrix)
    (root / "dsa_matrix.log").write_text(
        "synthetic timing matrix only\n", encoding="utf-8"
    )

    profile_files = {
        "dsa_profile_seq4096.nsys-rep": b"synthetic nsys report",
        "dsa_profile_seq4096.sqlite": b"synthetic sqlite",
        "dsa_profile_seq4096.log": b"synthetic profile log",
        "dsa_profile_seq4096_globaltimer.csv": b"synthetic sidecar trace",
        "dsa_profile_seq4096_cuda_gpu_kern_sum.csv": b"kernel summary",
        "dsa_profile_seq4096_nvtx_gpu_proj_sum.csv": b"projection summary",
        "dsa_profile_seq4096_nvtx_kern_sum.csv": b"range kernel summary",
    }
    for name, contents in profile_files.items():
        (root / name).write_bytes(contents)
    profile_validation = {
        "schema": 1, "status": "PASS", "errors": [],
        "required_ranges": list(RANGE_KERNELS),
        "projection": {name: {"range_instances": 1, "total_gpu_ops": 1}
                       for name in RANGE_KERNELS},
        "range_kernel_instances": {
            name: {kernel: 1 for kernel in kernels}
            for name, kernels in RANGE_KERNELS.items()
        },
        "global_worker_instances": {
            "dsaIndexer": 1, "dsaTopk": 1, "dsaAttention": 1,
        },
        "report_sha256": {
            "projection": sha(root / "dsa_profile_seq4096_nvtx_gpu_proj_sum.csv"),
            "range_kernels": sha(root / "dsa_profile_seq4096_nvtx_kern_sum.csv"),
            "kernels": sha(root / "dsa_profile_seq4096_cuda_gpu_kern_sum.csv"),
        },
    }
    profile_validation_path = root / "dsa_profile_seq4096_profile_validation.json"
    write_json(profile_validation_path, profile_validation)

    ncu_stdout = root / "ncu_permission.stdout"
    ncu_stderr = root / "ncu_permission.stderr"
    ncu_stdout.write_text("ERR_NVGPUCTRPERM\n", encoding="utf-8")
    ncu_stderr.write_text("permission denied\n", encoding="utf-8")
    ncu = {
        "schema": 1, "purpose": "hardware_counter_permission_probe",
        "measurement_admission": "not_a_timing_sample", "launch_count": 1,
        "timing_admission_affected": False, "returncode": 1,
        "permission_denied": True, "hardware_counters_available": False,
        "stdout_path": str(ncu_stdout), "stderr_path": str(ncu_stderr),
    }
    write_json(root / "ncu_permission.json", ncu)

    current_hashes = {
        "binary_sha256": sha(binary),
        "source_sha256": sha(BASE / "dsa_native.cu"),
        "bench_util_sha256": sha(BASE.parent / "common" / "bench_util.cuh"),
        "cta_trace_sha256": sha(BASE.parent / "common" / "cta_trace.cuh"),
        "build_sha256": sha(BASE.parent / "build.sh"),
        "build_provenance_helper_sha256": sha(BASE / "build_provenance.py"),
        "build_manifest_sha256": sha(root / "dsa_build_manifest.json"),
        "build_log_sha256": sha(build_log),
        "validator_sha256": sha(BASE / "validate_dsa_native.py"),
        "binary_verifier_sha256": sha(BASE / "verify_dsa_binary.py"),
        "binary_proof_sha256": sha(root / "dsa_binary_proof.json"),
        "gpu_exclusivity_helper_sha256": sha(BASE / "gpu_exclusivity.py"),
        "aggregator_sha256": sha(BASE / "aggregate_dsa_native.py"),
        "profile_validator_sha256": sha(BASE / "validate_dsa_profile.py"),
        "profiler_evidence_sha256": sha(BASE / "profiler_evidence.py"),
        "campaign_finalizer_sha256": sha(BASE / "finalize_dsa_campaign.py"),
        "runner_sha256": sha(RUNNER),
    }
    marker_lines = [
        "marker_schema=2", "fast=1", "profile=1", f"tag={TAG}", "seq=4096",
        "gpu_index=3", f"gpu_physical_uuid={GPU_TEST_UUID}",
        f"cuda_visible_devices={GPU_TEST_UUID}",
        f"native_runtime_uuid={GPU_TEST_UUID}",
        "target_arch=sm_100", "nvtx_range_count=9",
        "mode_count=4", "mode_order=floor,wave_floor,impl,ceiling",
        "sample_order=cyclic_latin_4", "invocations_per_point=20",
        "step_timeout_spec=120s",
        f"device_fingerprint_sha256={sha(root / 'device_fingerprint.txt')}",
        f"gpu_exclusivity_lease_id={lease_id}",
        f"gpu_exclusivity_lease_sha256={sha(root / 'gpu_exclusivity_lease.json')}",
        "gpu_global_lock_scope=target_uuid",
        f"gpu_global_lock_key_sha256={hashlib.sha256(target['uuid'].encode()).hexdigest()}",
        f"gpu_global_lock_path_sha256={hashlib.sha256(('/tmp/cta_pdl_gpu_' + str(target['uuid']) + '.lock').encode()).hexdigest()}",
        "gpu_monitor_interval_ms=50", "gpu_query_timeout_ms=2000",
        "gpu_monitor_coverage_model=bounded_interval_nvidia_smi_process_sampling",
    ]
    marker_lines.extend(f"{key}={value}" for key, value in current_hashes.items())
    marker_lines.extend((
        f"log_sha256={sha(log)}", f"trace_sha256={sha(trace)}",
        f"validation_sha256={sha(validation_path)}",
        f"gpu_pre_sha256={sha(root / f'{TAG}_gpu_pre.json')}",
        f"gpu_post_sha256={sha(root / f'{TAG}_gpu_post.json')}",
        f"gpu_monitor_sha256={sha(root / f'{TAG}_gpu_monitor.json')}",
        f"gpu_observations_sha256={sha(root / f'{TAG}_gpu_observations.ndjson')}",
        f"argv_sha256={'e' * 64}",
    ))
    (root / f"{TAG}.done").write_text(
        "\n".join(marker_lines) + "\n", encoding="utf-8"
    )

    terminal = {
        "schema": 1, "status": "PASS", "errors": [],
        "campaign": "tier5_native_dsa", "fast": 1, "profile": 1,
        "expected_tags": [TAG], "runner_sha256": sha(RUNNER),
        "evidence_sha256": {
            "validation_matrix": sha(root / "validation_matrix.json"),
            "binary_proof": sha(root / "dsa_binary_proof.json"),
            "build_provenance": sha(root / "dsa_build_manifest.json"),
            "gpu_exclusivity_lease": sha(root / "gpu_exclusivity_lease.json"),
            "profile_validation": sha(profile_validation_path),
            "ncu_permission": sha(root / "ncu_permission.json"),
        },
    }
    write_json(root / "terminal_status.json", terminal)
    return binary


class RunnerPreflightTests(unittest.TestCase):
    def assert_rejected(self, **settings: str) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-runner-negative-") as temp:
            env = os.environ.copy()
            env.update(settings)
            env["RESULTS"] = temp
            result = run(str(RUNNER), env=env)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertFalse((Path(temp) / "device.txt").exists(), result.stdout)

    def test_formal_requires_profile(self) -> None:
        self.assert_rejected(FAST="0", PROFILE="0")

    def test_formal_rejects_subset(self) -> None:
        self.assert_rejected(FAST="0", PROFILE="1", DSA_SEQS="4096 32768")

    def test_formal_rejects_duplicate(self) -> None:
        self.assert_rejected(
            FAST="0", PROFILE="1", DSA_SEQS="4096 32768 131072 131072"
        )

    def test_fast_is_exactly_4k(self) -> None:
        self.assert_rejected(FAST="1", PROFILE="0", DSA_SEQS="4096 32768")

    def test_monitor_interval_cannot_disable_runtime_sampling(self) -> None:
        self.assert_rejected(
            FAST="1", PROFILE="0", DSA_MONITOR_INTERVAL_MS="3600000"
        )

    def test_rejected_results_directory_fails_before_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-runner-rejected-") as temp:
            root = Path(temp)
            (root / "formal_rejection.json").write_text(
                json.dumps({"status": "REJECTED", "accepted_timing": 0}) + "\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({"RESULTS": temp, "FAST": "1", "PROFILE": "0"})
            result = run(str(RUNNER), env=env)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("contains formal rejection evidence", result.stdout)
            self.assertFalse((root / f"{TAG}.done").exists())

    def test_foreign_compute_pid_creates_zero_timing_rejection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-runner-foreign-") as temp:
            root = Path(temp)
            fake = root / "nvidia-smi"
            fake.write_text(textwrap.dedent("""\
                #!/usr/bin/env bash
                case "$1" in
                  --query-gpu=index,uuid,name)
                    echo '0, GPU-11111111-2222-3333-4444-555555555555, Synthetic GPU'
                    ;;
                  --query-compute-apps=*)
                    echo '4242, GPU-11111111-2222-3333-4444-555555555555, foreign-worker, 64'
                    ;;
                  *)
                    echo 'unexpected query' >&2
                    exit 2
                    ;;
                esac
            """), encoding="utf-8")
            fake.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "RESULTS": temp, "FAST": "1", "PROFILE": "0",
                "DSA_NVIDIA_SMI": str(fake),
            })
            result = run(str(RUNNER), env=env)
            self.assertEqual(result.returncode, 2, result.stdout)
            rejection = json.loads(
                (root / "formal_rejection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(rejection["accepted_timing"], 0)
            self.assertFalse(any(root.glob("*.done")))

    def test_global_gpu_uuid_lock_blocks_different_results_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-global-lock-") as temp:
            root = Path(temp)
            uuid_hex = hashlib.sha256(temp.encode()).hexdigest()
            uuid = (
                f"GPU-{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-"
                f"{uuid_hex[16:20]}-{uuid_hex[20:32]}"
            )
            fake = root / "nvidia-smi"
            fake.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                case "$1" in
                  --query-gpu=index,uuid,name)
                    echo '0, {uuid}, Synthetic GPU'
                    ;;
                  *)
                    echo 'unexpected query' >&2
                    exit 2
                    ;;
                esac
            """), encoding="utf-8")
            fake.chmod(0o755)
            lock_path = Path(f"/tmp/cta_pdl_gpu_{uuid}.lock")
            handle = lock_path.open("w", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                results = root / "other-results"
                env = os.environ.copy()
                env.update({
                    "RESULTS": str(results), "FAST": "1", "PROFILE": "0",
                    "DSA_NVIDIA_SMI": str(fake),
                })
                result = run(str(RUNNER), env=env)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("global target-GPU lock", result.stdout)
                self.assertFalse(any(results.glob("*.done")))
            finally:
                handle.close()
                lock_path.unlink(missing_ok=True)


class GpuMonitorMockTests(unittest.TestCase):
    def test_exit_residual_requires_prior_identity_and_missing_proc(self) -> None:
        module = __import__("gpu_exclusivity")
        classify = module.is_allowed_exit_residual
        process = {
            "pid": 4242, "gpu_uuid": GPU_MONITOR_UUID,
            "name": "[No data]", "used_memory": "766",
        }
        known = {4242: 123456}
        self.assertTrue(classify(process, known, "missing", {}, {}))
        self.assertFalse(
            classify(process, {}, "missing", {}, {}), "never-allowed PID"
        )
        self.assertFalse(
            classify({**process, "pid": 4343}, known, "missing", {}, {}),
            "unknown [No data] PID",
        )
        self.assertFalse(
            classify(process, known, "present", {4242: 123456}, {}),
            "live PID reuse must not inherit prior trust",
        )
        self.assertFalse(
            classify(
                {**process, "name": "foreign-worker"}, known, "missing", {}, {}
            ),
            "missing /proc with a non-tombstone name remains foreign",
        )
        self.assertFalse(
            classify(process, known, "missing", {4242: 999999}, {}),
            "retired identity mismatch must not be accepted",
        )
        at_limit = {4242: module.MAX_EXIT_RESIDUAL_OBSERVATIONS_PER_PID}
        self.assertFalse(
            classify(process, known, "missing", {4242: 123456}, at_limit),
            "persistent [No data] must fail closed at the sample limit",
        )
        self.assertFalse(
            classify(process, known, "unreadable", {4242: 123456}, {}),
            "an unreadable /proc stat is not proof of process exit",
        )

    def test_proc_info_distinguishes_missing_from_unreadable(self) -> None:
        module = __import__("gpu_exclusivity")
        with mock.patch.object(
            module.Path, "read_text", side_effect=FileNotFoundError()
        ):
            self.assertEqual(module.proc_info(4242), (None, "missing", None))
        with mock.patch.object(
            module.Path, "read_text", side_effect=PermissionError("denied")
        ):
            info, status, detail = module.proc_info(4242)
            self.assertIsNone(info)
            self.assertEqual(status, "unreadable")
            self.assertIn("denied", detail)
        with mock.patch.object(module.Path, "read_text", return_value="malformed"):
            info, status, detail = module.proc_info(4242)
            self.assertIsNone(info)
            self.assertEqual(status, "unreadable")
            self.assertIn("malformed", detail)

    def run_gated_monitor(
        self, root: Path, *, mode: str, command: str,
        require_allowed: bool, reap_while_monitoring: bool = False,
        expect_ready: bool = True,
    ) -> tuple[int, dict[str, object], str]:
        fake = root / "nvidia-smi"
        lease = root / "lease.json"
        monitor_json = root / "monitor.json"
        observations = root / "observations.ndjson"
        ready = root / "ready.json"
        pid_file = root / "watch.pid"
        residual_count_file = root / "residual.count"
        write_monitor_fake(fake)
        make_monitor_lease(lease)
        child = subprocess.Popen(
            ["bash", "-c", f'kill -STOP "$$"; exec {command}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_file.write_text(f"{child.pid}\n", encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "FAKE_WATCH_PID_FILE": str(pid_file), "FAKE_GPU_MODE": mode,
            "FAKE_FOREIGN_PID": str(os.getpid()),
            "FAKE_UNKNOWN_PID": "2147483646",
            "FAKE_RESIDUAL_COUNT_FILE": str(residual_count_file),
        })
        argv = [
            "python3", str(BASE / "gpu_exclusivity.py"), "monitor",
            "--lease", str(lease), "--json", str(monitor_json),
            "--observations", str(observations), "--ready-file", str(ready),
            "--nvidia-smi", str(fake), "--gpu-index", "0",
            "--watch-pid", str(child.pid), "--phase", "mock",
            "--interval-ms", "10", "--query-timeout-ms", "100",
            "--terminate-on-failure",
        ]
        if require_allowed:
            argv.append("--require-allowed-process")
        monitor = subprocess.Popen(
            argv, cwd=BASE, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not ready.is_file():
                time.sleep(0.005)
            self.assertTrue(ready.is_file(), "monitor did not publish READY/FAIL")
            ready_value = json.loads(ready.read_text(encoding="utf-8"))
            expected_ready_status = "READY" if expect_ready else "FAIL"
            self.assertEqual(
                ready_value["status"], expected_ready_status, ready_value
            )
            if expect_ready:
                os.kill(child.pid, signal.SIGCONT)
                if reap_while_monitoring:
                    child.wait(timeout=3)
            stdout, _ = monitor.communicate(timeout=8)
            if child.poll() is None:
                child.wait(timeout=3)
            manifest = json.loads(monitor_json.read_text(encoding="utf-8"))
            return monitor.returncode, manifest, stdout
        finally:
            if monitor.poll() is None:
                monitor.kill()
                monitor.wait(timeout=3)
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=3)

    def test_ready_gate_observes_allowed_process_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-") as temp:
            rc, manifest, output = self.run_gated_monitor(
                Path(temp), mode="allowed", command="sleep 0.25",
                require_allowed=True,
            )
            self.assertEqual(rc, 0, output)
            self.assertEqual(manifest["status"], "PASS")
            self.assertTrue(manifest["start_barrier_complete"])
            self.assertGreater(manifest["allowed_observation_count"], 0)

    def test_unobserved_extremely_short_timing_child_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-short-") as temp:
            rc, manifest, _ = self.run_gated_monitor(
                Path(temp), mode="empty", command="true", require_allowed=True,
            )
            self.assertEqual(rc, 2)
            self.assertEqual(manifest["status"], "FAIL")
            self.assertIn(
                "no allowed target-GPU process was observed",
                "\n".join(manifest["errors"]),
            )

    def test_extremely_short_non_timing_probe_can_finish_after_ready(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-probe-") as temp:
            rc, manifest, output = self.run_gated_monitor(
                Path(temp), mode="empty", command="true", require_allowed=False,
            )
            self.assertEqual(rc, 0, output)
            self.assertEqual(manifest["status"], "PASS")
            self.assertTrue(manifest["start_barrier_complete"])

    def test_foreign_process_after_release_fails_and_terminates_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-foreign-") as temp:
            rc, manifest, _ = self.run_gated_monitor(
                Path(temp), mode="foreign", command="sleep 2",
                require_allowed=True,
            )
            self.assertEqual(rc, 2)
            self.assertEqual(manifest["status"], "FAIL")
            self.assertTrue(manifest["foreign_processes_detected"])
            self.assertTrue(manifest["terminated_on_failure"])

    def test_persistent_exit_residual_exceeds_bound_and_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-residual-") as temp:
            rc, manifest, output = self.run_gated_monitor(
                Path(temp), mode="persistent_residual", command="sleep 0.2",
                require_allowed=True, reap_while_monitoring=True,
            )
            self.assertEqual(rc, 2, output)
            self.assertEqual(manifest["status"], "FAIL")
            self.assertTrue(manifest["foreign_processes_detected"])
            limit = manifest["allowed_exit_residual_max_observations_per_pid"]
            self.assertEqual(limit, 4)
            self.assertEqual(
                manifest["max_allowed_exit_residual_observations_observed"], limit
            )
            self.assertEqual(
                manifest["allowed_exit_residual_processes"][0][
                    "residual_observation_count"
                ],
                limit,
            )
            self.assertIn(
                "post_exit_residual_observation_limit_exceeded",
                "\n".join(manifest["errors"]),
            )

    def test_known_exit_residual_clears_and_drains_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-residual-pass-") as temp:
            root = Path(temp)
            rc, manifest, output = self.run_gated_monitor(
                root, mode="exit_residual", command="sleep 0.2",
                require_allowed=True, reap_while_monitoring=True,
            )
            self.assertEqual(rc, 0, output)
            self.assertEqual(manifest["status"], "PASS")
            self.assertFalse(manifest["foreign_processes_detected"])
            self.assertGreater(manifest["allowed_observation_count"], 0)
            self.assertEqual(manifest["allowed_exit_residual_observation_count"], 2)
            residual = manifest["allowed_exit_residual_processes"]
            self.assertEqual(len(residual), 1)
            self.assertEqual(residual[0]["residual_observation_count"], 2)
            records = [
                json.loads(line)
                for line in (root / "observations.ndjson").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            residual_records = [
                record for record in records
                if record["allowed_exit_residual_processes"]
            ]
            self.assertEqual(len(residual_records), 2)
            self.assertEqual(
                [
                    record["allowed_exit_residual_processes"][0][
                        "residual_observation_number"
                    ]
                    for record in residual_records
                ],
                [1, 2],
            )
            self.assertEqual(
                records[-2]["observation"]["target_compute_processes"], []
            )
            self.assertEqual(
                records[-1]["observation"]["target_compute_processes"], []
            )

    def test_unknown_no_data_process_fails_integration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-unknown-residual-") as temp:
            rc, manifest, _ = self.run_gated_monitor(
                Path(temp), mode="unknown_residual", command="sleep 2",
                require_allowed=False, expect_ready=False,
            )
            self.assertEqual(rc, 2)
            self.assertEqual(manifest["status"], "FAIL")
            self.assertTrue(manifest["foreign_processes_detected"])
            self.assertIn("[No data]", "\n".join(manifest["errors"]))

    def test_timeout_is_not_silently_retried_as_field_compatibility(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-timeout-") as temp:
            root = Path(temp)
            fake = root / "nvidia-smi"
            pid_file = root / "watch.pid"
            pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
            write_monitor_fake(fake)
            old_pid_file = os.environ.get("FAKE_WATCH_PID_FILE")
            old_mode = os.environ.get("FAKE_GPU_MODE")
            os.environ["FAKE_WATCH_PID_FILE"] = str(pid_file)
            os.environ["FAKE_GPU_MODE"] = "timeout"
            try:
                _, errors = __import__("gpu_exclusivity").observe(
                    str(fake), 0, 100
                )
            finally:
                if old_pid_file is None:
                    os.environ.pop("FAKE_WATCH_PID_FILE", None)
                else:
                    os.environ["FAKE_WATCH_PID_FILE"] = old_pid_file
                if old_mode is None:
                    os.environ.pop("FAKE_GPU_MODE", None)
                else:
                    os.environ["FAKE_GPU_MODE"] = old_mode
            self.assertTrue(any("timed out" in error for error in errors), errors)


class AggregationTests(unittest.TestCase):
    def aggregate(self, root: Path) -> subprocess.CompletedProcess[str]:
        return run(
            "python3", str(AGGREGATOR), "--results", str(root), "--fast", "1",
            "--log-out", str(root / "matrix.log"),
            "--json-out", str(root / "matrix.json"),
        )

    def test_pass_then_trace_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-aggregate-") as temp:
            root = Path(temp)
            make_aggregate_fixture(root)
            self.assertEqual(self.aggregate(root).returncode, 0)
            with (root / f"{TAG}_trace.csv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            result = self.aggregate(root)
            self.assertEqual(result.returncode, 2, result.stdout)

    def test_missing_json_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-aggregate-") as temp:
            root = Path(temp)
            make_aggregate_fixture(root)
            (root / f"{TAG}_validation.json").unlink()
            self.assertEqual(self.aggregate(root).returncode, 2)

    def test_marker_hash_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-aggregate-") as temp:
            root = Path(temp)
            make_aggregate_fixture(root)
            marker = root / f"{TAG}.done"
            marker.write_text(
                marker.read_text(encoding="utf-8").replace(
                    f"trace_sha256={sha(root / f'{TAG}_trace.csv')}",
                    f"trace_sha256={'b' * 64}",
                ),
                encoding="utf-8",
            )
            self.assertEqual(self.aggregate(root).returncode, 2)

    def test_legacy_three_mode_marker_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-aggregate-legacy-mode-") as temp:
            root = Path(temp)
            make_aggregate_fixture(root)
            marker = root / f"{TAG}.done"
            marker.write_text(
                marker.read_text(encoding="utf-8")
                .replace("mode_count=4", "mode_count=3")
                .replace(
                    "mode_order=floor,wave_floor,impl,ceiling",
                    "mode_order=floor,impl,ceiling",
                ),
                encoding="utf-8",
            )
            result = self.aggregate(root)
            self.assertEqual(result.returncode, 2, result.stdout)

    def test_rehashed_exit_residual_semantic_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-aggregate-residual-") as temp:
            root = Path(temp)
            make_aggregate_fixture(root)
            observations = root / f"{TAG}_gpu_observations.ndjson"
            records = [
                json.loads(line)
                for line in observations.read_text(encoding="utf-8").splitlines()
            ]
            records[2]["allowed_exit_residual_processes"][0][
                "previous_allowed_start_ticks"
            ] = 99999
            observations.write_text(
                "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            monitor_path = root / f"{TAG}_gpu_monitor.json"
            monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
            monitor["observations_sha256"] = sha(observations)
            write_json(monitor_path, monitor)
            marker = root / f"{TAG}.done"
            values: dict[str, str] = {}
            for line in marker.read_text(encoding="utf-8").splitlines():
                key, value = line.split("=", 1)
                values[key] = value
            values["gpu_observations_sha256"] = sha(observations)
            values["gpu_monitor_sha256"] = sha(monitor_path)
            marker.write_text(
                "".join(f"{key}={value}\n" for key, value in values.items()),
                encoding="utf-8",
            )
            result = self.aggregate(root)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn(
                "unsafe/unbound exit residual",
                (root / "matrix.log").read_text(encoding="utf-8"),
            )

    def test_extra_done_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-aggregate-") as temp:
            root = Path(temp)
            make_aggregate_fixture(root)
            (root / "dsa_extra.done").write_text("marker_schema=2\n", encoding="utf-8")
            self.assertEqual(self.aggregate(root).returncode, 2)

    def test_rejection_sentinel_overrides_complete_matrix(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-aggregate-") as temp:
            root = Path(temp)
            make_aggregate_fixture(root)
            (root / "formal_rejection.json").write_text(
                '{"status":"REJECTED","accepted_timing":0}\n', encoding="utf-8"
            )
            self.assertEqual(self.aggregate(root).returncode, 2)


class WorkParityContractTests(unittest.TestCase):
    def test_floor_trace_uses_trigger_for_safety_and_end_for_overlap(self) -> None:
        # The dependent grid starts after the producer's ready/trigger point but
        # before its tail ends.  This is the legal PDL overlap that the old
        # start<ready definition incorrectly rejected.
        rows = [
            {
                "stage": "indexer", "block": "0", "epoch": "7", "sm": "0",
                "t_start": "100", "t_dep": "100", "t_ready": "200",
                "t_trigger": "210", "t_end": "400",
            },
            {
                "stage": "topk", "block": "0", "epoch": "7", "sm": "1",
                "t_start": "250", "t_dep": "300", "t_ready": "350",
                "t_trigger": "360", "t_end": "500",
            },
            {
                "stage": "attention", "block": "0", "epoch": "7", "sm": "2",
                "t_start": "380", "t_dep": "420", "t_ready": "450",
                "t_trigger": "460", "t_end": "600",
            },
        ]
        errors: list[str] = []
        metrics = analyze_trace(
            rows,
            {
                "producer_ctas": 1, "query_blocks": 1,
                "physical_degree": 1, "query_wave_size": 1, "sms": 3,
            },
            "floor", 7, errors,
        )
        self.assertEqual(errors, [])
        self.assertEqual(metrics["topk_early"], 1)
        self.assertEqual(metrics["attention_early"], 1)
        self.assertEqual(metrics["topk_waited"], 1)
        self.assertEqual(metrics["attention_waited"], 1)
        self.assertEqual(metrics["safety_failures"], 0)

        unsafe = [dict(row) for row in rows]
        unsafe[1]["t_start"] = "205"
        unsafe[1]["t_dep"] = "205"
        errors = []
        metrics = analyze_trace(
            unsafe,
            {
                "producer_ctas": 1, "query_blocks": 1,
                "physical_degree": 1, "query_wave_size": 1, "sms": 3,
            },
            "floor", 7, errors,
        )
        self.assertEqual(metrics["safety_failures"], 1)
        self.assertFalse(metrics["topk_waited"])

    def test_wave_floor_extrema_are_precomputed_per_wave_with_partial_tail(self) -> None:
        config = {
            "producer_ctas": 6, "query_blocks": 3,
            "physical_degree": 2, "query_wave_size": 2, "sms": 3,
        }
        rows: list[dict[str, str]] = []

        producer_times = (
            (10, 100, 110, 400), (11, 105, 115, 410),
            (12, 110, 120, 420), (13, 115, 125, 430),
            # The partial second wave has deliberately much larger extrema.  A
            # global-max regression would incorrectly mark queries 0 and 1 unsafe.
            (900, 990, 1000, 1400), (901, 995, 1010, 1410),
        )
        for block, (start, ready, trigger, end) in enumerate(producer_times):
            rows.append({
                "stage": "indexer", "block": str(block), "epoch": "9",
                "sm": str(block % 3), "t_start": str(start),
                "t_dep": str(start), "t_ready": str(ready),
                "t_trigger": str(trigger), "t_end": str(end),
            })

        for block, values in enumerate((
            (150, 200, 250, 260, 600),
            (160, 210, 255, 270, 610),
            (1050, 1100, 1150, 1160, 1500),
        )):
            start, dep, ready, trigger, end = values
            rows.append({
                "stage": "topk", "block": str(block), "epoch": "9",
                "sm": str(block % 3), "t_start": str(start),
                "t_dep": str(dep), "t_ready": str(ready),
                "t_trigger": str(trigger), "t_end": str(end),
            })
        for block, values in enumerate((
            (300, 300, 350, 360, 700),
            (310, 320, 370, 380, 710),
            (1200, 1200, 1250, 1260, 1600),
        )):
            start, dep, ready, trigger, end = values
            rows.append({
                "stage": "attention", "block": str(block), "epoch": "9",
                "sm": str(block % 3), "t_start": str(start),
                "t_dep": str(dep), "t_ready": str(ready),
                "t_trigger": str(trigger), "t_end": str(end),
            })

        def slow_reference(mode: str) -> dict[str, int]:
            stages = {
                stage: {
                    int(row["block"]): {key: int(row[key]) for key in (
                        "t_start", "t_dep", "t_trigger", "t_end"
                    )}
                    for row in rows if row["stage"] == stage
                }
                for stage in ("indexer", "topk", "attention")
            }
            wave_size = config["query_blocks"] if mode == "floor" else 2
            result = {
                "topk_early": 0, "attention_early": 0,
                "topk_waited": 0, "attention_waited": 0,
                "safety_failures": 0,
            }
            for query in range(config["query_blocks"]):
                begin = query // wave_size * wave_size
                finish = min(config["query_blocks"], begin + wave_size)
                producers = [
                    stages["indexer"][wave_query * 2 + parent]
                    for wave_query in range(begin, finish) for parent in range(2)
                ]
                topks = [stages["topk"][wave_query] for wave_query in range(begin, finish)]
                topk = stages["topk"][query]
                attention = stages["attention"][query]
                topk_overlap = topk["t_start"] < max(row["t_end"] for row in producers)
                attention_overlap = attention["t_start"] < max(
                    row["t_end"] for row in topks
                )
                topk_safe = topk["t_dep"] >= max(
                    row["t_trigger"] for row in producers
                )
                attention_safe = attention["t_dep"] >= max(
                    row["t_trigger"] for row in topks
                )
                result["topk_early"] += topk_overlap
                result["attention_early"] += attention_overlap
                result["topk_waited"] += topk_overlap and topk_safe
                result["attention_waited"] += attention_overlap and attention_safe
                result["safety_failures"] += not topk_safe
                result["safety_failures"] += not attention_safe
            return result

        for mode in ("wave_floor", "floor"):
            errors: list[str] = []
            observed = analyze_trace(rows, config, mode, 9, errors)
            self.assertEqual(errors, [])
            expected = slow_reference(mode)
            self.assertEqual(
                {key: observed[key] for key in expected}, expected,
            )
        self.assertEqual(slow_reference("wave_floor")["safety_failures"], 0)
        self.assertEqual(slow_reference("floor")["safety_failures"], 4)

    def test_four_way_latin_rotation_balances_every_position(self) -> None:
        orders = [latin_order(rep) for rep in range(4)]
        expected = {"floor", "wave_floor", "impl", "ceiling"}
        self.assertTrue(all(set(order) == expected for order in orders))
        for position in range(4):
            self.assertEqual({order[position] for order in orders}, expected)

    def test_cuda_uuid_bytes_use_nvidia_smi_canonical_grouping(self) -> None:
        self.assertEqual(
            canonical_gpu_uuid_from_bytes(
                bytes.fromhex("00112233445566778899aabbccddeeff")
            ),
            "GPU-00112233-4455-6677-8899-aabbccddeeff",
        )

    def test_runtime_uuid_must_match_runner_lease(self) -> None:
        name_hex = "Synthetic GPU".encode().hex()
        config = {
            "runtime_uuid": GPU_TEST_UUID, "runtime_ordinal": "0",
            "runtime_ordinal_zero": "1", "runtime_name_hex": name_hex,
            "runtime_cc_major": "10", "runtime_cc_minor": "0",
            "runtime_sms": "160", "sms": "160",
        }
        device = {
            "runtime_uuid": GPU_TEST_UUID, "runtime_ordinal": "0",
            "runtime_ordinal_zero": "1", "name_hex": name_hex,
            "cc_major": "10", "cc_minor": "0", "sms": "160",
        }
        self.assertEqual(device_identity_errors(config, device, GPU_TEST_UUID), [])
        wrong = "GPU-99999999-2222-3333-4444-555555555555"
        self.assertTrue(device_identity_errors(config, device, wrong))

    def test_uint32_pair_accumulator_preserves_old_low16(self) -> None:
        cases = pair_low16_equivalence_cases(16384, 8192, 4096, 64, 128)
        self.assertTrue(cases)
        self.assertTrue(all(case["equivalent"] for case in cases))

    def test_pair_contract_tamper_fails(self) -> None:
        row = {
            "pair_accumulator": "uint32_mod2p32",
            "pair_low16_equivalence": "mod2p32_then_low16_equals_uint64_low16",
            "pair_query_cache": "cta_shared_once",
            "pair_key_cache": "cta_shared_once_register_tile",
            "pair_iteration": "explicit_inline_ptx_add_u32_per_pair",
            "pair_closed_form": "0",
            "pair_key_register_tile": "8",
            "pair_lut_global_loads_per_cta": "192",
            "pair_adds_per_score": "8192",
        }
        self.assertEqual(pair_contract_errors(row, 64, 128, "fixture"), [])
        row["pair_adds_per_score"] = "8191"
        self.assertTrue(pair_contract_errors(row, 64, 128, "fixture"))

    def test_history_load_count_tamper_fails(self) -> None:
        row = {
            "history_loads": "65536",
            "expected_history_loads": "65536",
            "history_load_complete": "1",
        }
        self.assertEqual(
            history_load_contract_errors(row, 65536, "fixture", require_actual=True),
            [],
        )
        row["history_loads"] = "65535"
        self.assertTrue(
            history_load_contract_errors(row, 65536, "fixture", require_actual=True)
        )


class MonitorFinalizerContractTests(unittest.TestCase):
    TARGET = {"index": 3, "uuid": GPU_TEST_UUID, "name": "Synthetic GPU"}
    LEASE_ID = "d" * 64

    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        manifest = root / "monitor.json"
        observations = root / "observations.ndjson"
        make_monitor_fixture(
            manifest, observations, "semantic-test", self.LEASE_ID,
            self.TARGET, require_allowed=True, include_residual=True,
        )
        return manifest, observations

    def validate(self, manifest: Path, observations: Path) -> list[str]:
        errors: list[str] = []
        finalize_dsa_campaign.validate_monitor(
            manifest, observations, self.LEASE_ID, self.TARGET,
            "semantic-test", True, 50, 2000, errors, {}, "monitor",
        )
        return errors

    def test_bounded_known_exit_residual_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-monitor-finalizer-") as temp:
            manifest, observations = self.make_fixture(Path(temp))
            self.assertEqual(self.validate(manifest, observations), [])

    def test_exit_residual_contract_mutations_fail(self) -> None:
        expected = {
            "unknown_prior": "unsafe/unbound exit residual",
            "wrong_name": "unsafe/unbound exit residual",
            "wrong_classification": "unsafe/unbound exit residual",
            "nonmonotonic_number": "unsafe/unbound exit residual",
            "over_limit": "unsafe/unbound exit residual",
            "normal_pid_reappears": "retired PID returned as allowed",
            "manifest_mismatch": "exit residual identity/count set mismatch",
            "wrong_raw_gpu_uuid": "target process GPU UUID mismatch",
        }
        for mutation, expected_error in expected.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="dsa-monitor-finalizer-tamper-"
            ) as temp:
                manifest_path, observations = self.make_fixture(Path(temp))
                records = [
                    json.loads(line)
                    for line in observations.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                first = records[2]["allowed_exit_residual_processes"][0]
                second = records[3]["allowed_exit_residual_processes"][0]
                if mutation == "unknown_prior":
                    first["previous_allowed_start_ticks"] = 99999
                elif mutation == "wrong_name":
                    first["name"] = "No data"
                    records[2]["observation"]["target_compute_processes"][0][
                        "name"
                    ] = "No data"
                elif mutation == "wrong_classification":
                    first["classification"] = "trusted_without_proof"
                elif mutation == "nonmonotonic_number":
                    second["residual_observation_number"] = 1
                elif mutation == "over_limit":
                    first["residual_observation_number"] = 5
                elif mutation == "normal_pid_reappears":
                    raw = {
                        "pid": 777, "gpu_uuid": GPU_TEST_UUID,
                        "name": "allowed-worker", "used_memory": "64",
                    }
                    records[4]["observation"]["target_compute_processes"] = [raw]
                    records[4]["allowed_target_processes"] = [
                        {**raw, "proc_start_ticks": 12345}
                    ]
                    manifest["allowed_observation_count"] = 2
                elif mutation == "manifest_mismatch":
                    manifest["allowed_exit_residual_processes"][0][
                        "residual_observation_count"
                    ] = 3
                elif mutation == "wrong_raw_gpu_uuid":
                    wrong_uuid = "GPU-99999999-2222-3333-4444-555555555555"
                    first["gpu_uuid"] = wrong_uuid
                    records[2]["observation"]["target_compute_processes"][0][
                        "gpu_uuid"
                    ] = wrong_uuid
                observations.write_text(
                    "".join(
                        json.dumps(record, sort_keys=True, separators=(",", ":"))
                        + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )
                manifest["observations_sha256"] = sha(observations)
                write_json(manifest_path, manifest)
                errors = self.validate(manifest_path, observations)
                self.assertTrue(errors)
                self.assertIn(expected_error, "\n".join(errors))


class FinalCampaignAdmissionTests(unittest.TestCase):
    def admission(self, root: Path, binary: Path) -> dict[str, object]:
        return finalize_dsa_campaign.finalize_campaign(root, 1, 1, binary, RUNNER)

    def test_complete_hash_closed_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            payload = self.admission(root, binary)
            self.assertEqual(payload["status"], "PASS", payload["errors"])
            self.assertEqual(payload["accepted_timing"], 1)
            self.assertEqual(payload["device_identity"]["target_gpu"]["index"], 3)
            self.assertEqual(payload["device_identity"]["runtime_ordinal"], 0)
            self.assertEqual(
                payload["device_identity"]["native_runtime_uuid"], GPU_TEST_UUID
            )

    def test_missing_profile_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            (root / "dsa_profile_seq4096.nsys-rep").unlink()
            self.assertEqual(self.admission(root, binary)["status"], "FAIL")

    def test_ambiguous_ncu_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            path = root / "ncu_permission.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["permission_denied"] = False
            value["hardware_counters_available"] = False
            write_json(path, value)
            self.assertEqual(self.admission(root, binary)["status"], "FAIL")

    def test_exclusivity_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            path = root / f"{TAG}_gpu_post.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["observation"]["target_compute_processes"] = [
                {"pid": 99, "gpu_uuid": GPU_TEST_UUID, "name": "foreign"}
            ]
            write_json(path, value)
            self.assertEqual(self.admission(root, binary)["status"], "FAIL")

    def test_monitor_observation_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            path = root / f"{TAG}_gpu_observations.ndjson"
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"tampered":true}\n')
            self.assertEqual(self.admission(root, binary)["status"], "FAIL")

    def test_build_provenance_binary_binding_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            path = root / "dsa_build_manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["binary_sha256"] = "0" * 64
            write_json(path, value)
            self.assertEqual(self.admission(root, binary)["status"], "FAIL")

    def test_terminal_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            path = root / "terminal_status.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["status"] = "FAIL"
            write_json(path, value)
            self.assertEqual(self.admission(root, binary)["status"], "FAIL")

    def test_marker_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-finalizer-") as temp:
            root = Path(temp)
            binary = make_final_admission_fixture(root)
            marker = root / f"{TAG}.done"
            marker.write_text(
                marker.read_text(encoding="utf-8").replace(
                    f"source_sha256={sha(BASE / 'dsa_native.cu')}",
                    f"source_sha256={'0' * 64}",
                ),
                encoding="utf-8",
            )
            self.assertEqual(self.admission(root, binary)["status"], "FAIL")


class BinaryPtxTests(unittest.TestCase):
    def test_attention_acquire_cfg_rejects_control_flow_tampering(self) -> None:
        section = textwrap.dedent("""\
            .entry synthetic_dsaAttention(
            .param .u64 synthetic_dsaAttention_param_0,
            .param .u64 synthetic_dsaAttention_param_1,
            .param .u64 synthetic_dsaAttention_param_3,
            .param .u64 synthetic_dsaAttention_param_6,
            .param .u32 synthetic_dsaAttention_param_10,
            .param .u32 synthetic_dsaAttention_param_12,
            .param .u32 synthetic_dsaAttention_param_13
            )
            {
            .reg .pred %p<16>;
            .reg .b32 %r<96>;
            .reg .b64 %rd<128>;
            ld.param.u64 %rd30, [synthetic_dsaAttention_param_0];
            ld.param.u64 %rd31, [synthetic_dsaAttention_param_1];
            ld.param.u64 %rd41, [synthetic_dsaAttention_param_3];
            ld.param.u64 %rd44, [synthetic_dsaAttention_param_6];
            ld.param.u32 %r18, [synthetic_dsaAttention_param_10];
            ld.param.u32 %r20, [synthetic_dsaAttention_param_12];
            ld.param.u32 %r41, [synthetic_dsaAttention_param_13];
            ld.param.u32 %r22, [synthetic_dsaAttention_param_16];
            cvta.to.global.u64 %rd52, %rd44;
            ld.global.nc.u32 %r2, [%rd52];
            mov.u32 %r23, %ctaid.x;
            add.s32 %r1, %r22, %r23;
            mov.u32 %r3, %tid.x;
            cvta.to.global.u64 %rd12, %rd31;
            cvta.to.global.u64 %rd62, %rd30;
            griddepcontrol.launch_dependents;
            setp.ne.s32 %p8, %r3, 0;
            @%p8 bra $L__BB2_15;
            setp.eq.s32 %p9, %r41, 1;
            @%p9 bra $L__BB2_30;
            setp.ne.s32 %p10, %r41, 0;
            @%p10 bra $L__BB2_14;
            griddepcontrol.wait;
        $L__BB2_14:
            mov.u64 %rd106, %globaltimer;
        $L__BB2_15:
            bar.sync 0;
            cvt.u64.u32 %rd10, %r1;
            cvt.u64.u32 %rd58, %r3;
            cvt.s64.s32 %rd59, %r20;
            mad.lo.s64 %rd60, %rd59, %rd10, %rd58;
            shl.b64 %rd61, %rd60, 2;
            add.s64 %rd107, %rd62, %rd61;
            mov.u32 %r8, %ntid.x;
            mov.u32 %r76, %r3;
        $L__BB2_17:
            ld.global.u32 %r34, [%rd107];
            setp.lt.u32 %p14, %r34, %r18;
            rem.u32 %r44, %r43, %r18;
            selp.b32 %r45, %r34, %r44, %p14;
            mul.wide.u32 %rd65, %r45, 4;
            add.s64 %rd66, %rd12, %rd65;
            ld.global.u32 %r29, [%rd66];
            .reg .u32 dsa_history_loaded_value;
            mov.u32 dsa_history_loaded_value, %r29;
            mov.u32 %r28, dsa_history_loaded_value;
            .reg .u32 dsa_history_count_dependency;
            .reg .u64 dsa_history_load_count;
            mov.u32 dsa_history_count_dependency, %r28;
            mov.u64 dsa_history_load_count, %rd111;
            add.u64 dsa_history_load_count, dsa_history_load_count, 1;
            mov.u64 %rd111, dsa_history_load_count;
            add.s32 %r46, %r44, %r18;
            selp.b32 %r32, %r34, %r46, %p14;
            .reg .u32 dsa_semantic_index;
            .reg .u32 dsa_history_dependency;
            mov.u32 dsa_history_dependency, %r28;
            mov.u32 dsa_semantic_index, %r32;
            mov.u32 %r31, dsa_semantic_index;
            cvt.u64.u32 %rd67, %r28;
            mul.wide.u32 %rd68, %r31, 65537;
            add.s64 %rd69, %rd110, %rd67;
            add.s64 %rd110, %rd69, %rd68;
            add.s32 %r76, %r76, %r8;
            mul.wide.u32 %rd70, %r8, 4;
            add.s64 %rd107, %rd107, %rd70;
            setp.lt.s32 %p15, %r76, %r20;
            @%p15 bra $L__BB2_17;
        $L__BB2_16:
            membar.gl;
            griddepcontrol.launch_dependents;
            ret;
        $L__BB2_30:
            mul.wide.u32 %rd63, %r1, 4;
            add.s64 %rd9, %rd41, %rd63;
            mov.u32 %r75, 32;
        $L__BB2_31:
            ld.acquire.gpu.b32 %r25, [%rd9];
            setp.eq.s32 %p11, %r25, %r2;
            @%p11 bra $L__BB2_14;
            nanosleep.u32 %r75;
            bra.uni $L__BB2_31;
            }
        """)

        def proof(candidate: str) -> dict[str, object]:
            return verify_dsa_binary.attention_acquire_cfg_proof(
                candidate,
                candidate.index("ld.acquire.gpu.b32"),
                candidate.index("ld.global.u32 %r29"),
            )

        baseline = proof(section)
        self.assertTrue(baseline["pass"], baseline)
        removed_first_site = section
        for marker_name in (
            "dsa_history_loaded_value", "dsa_history_load_count",
            "dsa_history_count_dependency", "dsa_semantic_index",
            "dsa_history_dependency",
        ):
            removed_first_site = removed_first_site.replace(
                marker_name, "removed_" + marker_name
            )
        tampered = {
            "selector_uses_non_mode_parameter_register": section.replace(
                "setp.eq.s32 %p9, %r41, 1;",
                "setp.eq.s32 %p9, %r40, 1;",
            ),
            "success_compares_non_epoch_register": section.replace(
                "setp.eq.s32 %p11, %r25, %r2;",
                "setp.eq.s32 %p11, %r25, %r3;",
            ),
            "acquire_uses_unbound_address": section.replace(
                "ld.acquire.gpu.b32 %r25, [%rd9];",
                "ld.acquire.gpu.b32 %r25, [%rd1];",
            ),
            "pre_acquire_branch_bypasses_wait": section.replace(
                "add.s64 %rd9, %rd41, %rd63;",
                "add.s64 %rd9, %rd41, %rd63;\n@%p9 bra $L__BB2_15;",
            ),
            "unresolved_conditional_branch": section.replace(
                "add.s64 %rd9, %rd41, %rd63;",
                "add.s64 %rd9, %rd41, %rd63;\n@%p9 bra $L__BB2_99;",
            ),
            "duplicate_join_label": section.replace(
                "$L__BB2_30:", "$L__BB2_14:\n$L__BB2_30:"
            ),
            "success_retargets_post_history": section.replace(
                "@%p11 bra $L__BB2_14;", "@%p11 bra $L__BB2_16;"
            ),
            "success_predicate_inverted": section.replace(
                "@%p11 bra $L__BB2_14;", "@!%p11 bra $L__BB2_14;"
            ),
            "impl_selector_value_changed": section.replace(
                "setp.eq.s32 %p9, %r41, 1;",
                "setp.eq.s32 %p9, %r41, 2;",
            ),
            "failure_retry_bypasses_acquire": section.replace(
                "bra.uni $L__BB2_31;", "bra.uni $L__BB2_14;"
            ),
            "malformed_indirect_retry": section.replace(
                "bra.uni $L__BB2_31;", "bra.uni %r7;"
            ),
            "nonzero_selector_uses_non_mode_register": section.replace(
                "setp.ne.s32 %p10, %r41, 0;",
                "setp.ne.s32 %p10, %r40, 0;",
            ),
            "nonzero_branch_uses_impl_predicate": section.replace(
                "@%p10 bra $L__BB2_14;", "@%p9 bra $L__BB2_14;"
            ),
            "nonzero_branch_inverted": section.replace(
                "@%p10 bra $L__BB2_14;", "@!%p10 bra $L__BB2_14;"
            ),
            "nonzero_branch_skips_floor_wait_join": section.replace(
                "@%p10 bra $L__BB2_14;", "@%p10 bra $L__BB2_15;"
            ),
            "thread_gate_uses_epoch_not_tid": section.replace(
                "setp.ne.s32 %p8, %r3, 0;",
                "setp.ne.s32 %p8, %r2, 0;",
            ),
            "thread_gate_uses_impl_predicate": section.replace(
                "@%p8 bra $L__BB2_15;", "@%p9 bra $L__BB2_15;"
            ),
            "thread_gate_inverted": section.replace(
                "@%p8 bra $L__BB2_15;", "@!%p8 bra $L__BB2_15;"
            ),
            "thread_gate_targets_floor_join": section.replace(
                "@%p8 bra $L__BB2_15;", "@%p8 bra $L__BB2_14;"
            ),
            "consumer_barrier_removed": section.replace(
                "bar.sync 0;", "membar.cta;"
            ),
            "history_load_uses_nonhistory_address": section.replace(
                "ld.global.u32 %r29, [%rd66];",
                "ld.global.u32 %r29, [%rd62];",
            ),
            "extra_history_load_before_marker": section.replace(
                "ld.global.u32 %r29, [%rd66];",
                "ld.global.u32 %r29, [%rd66];\n"
                "ld.global.u32 %r30, [%rd66];",
            ),
            "safe_index_replaced_by_constant": section.replace(
                "selp.b32 %r45, %r34, %r44, %p14;",
                "mov.u32 %r45, 0;",
            ),
            "semantic_candidate_uses_unrelated_index": section.replace(
                "selp.b32 %r32, %r34, %r46, %p14;",
                "selp.b32 %r32, %r33, %r46, %p14;",
            ),
            "rank_index_load_uses_wrong_address": section.replace(
                "ld.global.u32 %r34, [%rd107];",
                "ld.global.u32 %r34, [%rd106];",
            ),
            "rank_index_load_removed": section.replace(
                "ld.global.u32 %r34, [%rd107];",
                "mov.u32 %r34, 0;",
            ),
            "rank_stride_address_update_removed": section.replace(
                "add.s64 %rd107, %rd107, %rd70;",
                "mov.u64 %rd107, %rd107;",
            ),
            "loaded_marker_uses_wrong_source": section.replace(
                "mov.u32 dsa_history_loaded_value, %r29;",
                "mov.u32 dsa_history_loaded_value, %r30;",
            ),
            "loaded_marker_uses_wrong_output": section.replace(
                "mov.u32 %r28, dsa_history_loaded_value;",
                "mov.u32 %r27, dsa_history_loaded_value;",
            ),
            "count_dependency_uses_wrong_value": section.replace(
                "mov.u32 dsa_history_count_dependency, %r28;",
                "mov.u32 dsa_history_count_dependency, %r27;",
            ),
            "semantic_dependency_uses_wrong_value": section.replace(
                "mov.u32 dsa_history_dependency, %r28;",
                "mov.u32 dsa_history_dependency, %r27;",
            ),
            "semantic_marker_uses_wrong_candidate": section.replace(
                "mov.u32 dsa_semantic_index, %r32;",
                "mov.u32 dsa_semantic_index, %r33;",
            ),
            "semantic_marker_uses_wrong_output": section.replace(
                "mov.u32 %r31, dsa_semantic_index;",
                "mov.u32 %r30, dsa_semantic_index;",
            ),
            "whole_history_marker_site_removed": removed_first_site,
        }
        for name, candidate in tampered.items():
            with self.subTest(name=name):
                candidate_proof = proof(candidate)
                self.assertFalse(candidate_proof["pass"], candidate_proof)

    def test_exact_ptx_and_new_work_proofs_fail_closed_on_tamper(self) -> None:
        binary = BASE / "dsa_native"
        dependencies = (BASE / "dsa_native.cu", BASE / "verify_dsa_binary.py")
        if not binary.is_file() or any(
            dependency.stat().st_mtime_ns > binary.stat().st_mtime_ns
            for dependency in dependencies
        ):
            self.skipTest("native binary is stale; run this proof after the authorized rebuild")
        dump = run("cuobjdump", "--dump-ptx", str(binary))
        self.assertEqual(dump.returncode, 0, dump.stdout)
        errors: list[str] = []
        verify_dsa_binary.verify_ptx(dump.stdout, errors)
        self.assertEqual(errors, [])
        for needle, replacement in (
            ("ld.acquire.gpu.b32", "ld.relaxed.gpu.b32"),
            ("dsa_pair_term", "removed_pair_term"),
            ("dsa_semantic_index", "removed_semantic_index"),
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, dump.stdout)
                tampered = dump.stdout.replace(needle, replacement)
                tamper_errors: list[str] = []
                verify_dsa_binary.verify_ptx(tampered, tamper_errors)
                self.assertTrue(tamper_errors)


class ProfileMappingTests(unittest.TestCase):
    def write_reports(self, root: Path, omit: tuple[str, str] | None = None) -> tuple[Path, Path, Path]:
        projection = root / "projection.csv"
        mapping = root / "mapping.csv"
        kernels = root / "kernels.csv"
        projection.write_text(
            '"Range","Style","Range Instances","Total GPU Ops"\n'
            + "".join(f'"{name}","PushPop",1,10\n' for name in RANGE_KERNELS),
            encoding="utf-8",
        )
        mapping_rows = ['"NVTX Range","Kern Inst","Kernel Name"\n']
        for range_name, names in RANGE_KERNELS.items():
            for kernel in names:
                if omit != (range_name, kernel):
                    mapping_rows.append(f'"{range_name}",1,"{kernel}"\n')
        mapping.write_text("".join(mapping_rows), encoding="utf-8")
        kernels.write_text(
            '"Instances","Name"\n1,"dsaIndexer"\n1,"dsaTopk"\n1,"dsaAttention"\n',
            encoding="utf-8",
        )
        return projection, mapping, kernels

    def validate(self, root: Path, omit: tuple[str, str] | None = None) -> int:
        projection, mapping, kernels = self.write_reports(root, omit)
        result = run(
            "python3", str(PROFILE_VALIDATOR),
            "--projection", str(projection), "--range-kernels", str(mapping),
            "--kernels", str(kernels), "--json", str(root / "proof.json"),
        )
        return result.returncode

    def test_all_nine_mappings_required(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dsa-profile-") as temp:
            root = Path(temp)
            self.assertEqual(self.validate(root), 0)
            self.assertEqual(self.validate(root, ("dsa.impl", "dsaTopk")), 2)


class ProfilerPermissionEvidenceTests(unittest.TestCase):
    def test_stdout_only_err_nvgpuctrperm_is_denied(self) -> None:
        result = profiler_evidence.classify_permission(
            "==ERROR== ERR_NVGPUCTRPERM - no permission", "", 1, 0
        )
        self.assertTrue(result["permission_denied"])
        self.assertTrue(result["explicit_permission_denial"])
        self.assertIn("stdout", result["permission_evidence_sources"])

    def test_stderr_only_permission_message_is_denied(self) -> None:
        result = profiler_evidence.classify_permission(
            "", "Permission denied while accessing performance counters", 1, 0
        )
        self.assertTrue(result["permission_denied"])
        self.assertIn("stderr", result["permission_evidence_sources"])

    def test_admin_only_failed_probe_is_denied_fallback(self) -> None:
        result = profiler_evidence.classify_permission("", "", 1, 1)
        self.assertTrue(result["permission_denied"])
        self.assertTrue(result["admin_policy_fallback"])
        self.assertIn("rm_profiling_admin_only", result["permission_evidence_sources"])

    def test_successful_privileged_probe_is_available(self) -> None:
        result = profiler_evidence.classify_permission("profile complete", "", 0, 1)
        self.assertFalse(result["permission_denied"])
        self.assertFalse(result["admin_policy_fallback"])

    def test_unknown_nonpermission_failure_is_not_misclassified(self) -> None:
        result = profiler_evidence.classify_permission("application failed", "", 1, 0)
        self.assertFalse(result["permission_denied"])


@unittest.skipUnless(os.environ.get("DSA_SMOKE_RESULTS"), "no strict smoke artifact supplied")
class SmokeArtifactTamperTests(unittest.TestCase):
    def test_ledger_and_trace_identity_tampering_fail_closed(self) -> None:
        source = Path(os.environ["DSA_SMOKE_RESULTS"])
        source_log = source / f"{TAG}.log"
        source_trace = source / f"{TAG}_trace.csv"
        with tempfile.TemporaryDirectory(prefix="dsa-validator-tamper-") as temp:
            root = Path(temp)
            trace = root / "trace.csv"
            trace.write_bytes(source_trace.read_bytes())
            base_log = source_log.read_text(encoding="utf-8").replace(
                str(source_trace), str(trace)
            )
            device_line = next(
                line for line in base_log.splitlines() if line.startswith("DEVICE_DSA ")
            )
            expected_uuid = next(
                token.split("=", 1)[1] for token in device_line.split()
                if token.startswith("runtime_uuid=")
            )
            variants = {
                "resource_semantics": base_log.replace(
                    "RESOURCE_DSA semantics=1", "RESOURCE_DSA semantics=2", 1
                ),
                "ceiling_wrong": base_log.replace("ceiling_wrong=1", "ceiling_wrong=0", 1),
            }
            for name, text in variants.items():
                log = root / f"{name}.log"
                out = root / f"{name}.json"
                log.write_text(text, encoding="utf-8")
                result = run(
                    "python3", str(NATIVE_VALIDATOR), str(log), "--trace", str(trace),
                    "--allow-short", "--expected-gpu-uuid", expected_uuid,
                    "--json", str(out),
                )
                self.assertEqual(result.returncode, 2, f"{name}: {result.stdout}")

            rows = trace.read_text(encoding="utf-8").splitlines()
            header = rows[0].split(",")
            sm_index = header.index("sm")
            first = rows[1].split(",")
            config = next(line for line in base_log.splitlines() if line.startswith("CONFIG_DSA "))
            sms = int(next(token.split("=", 1)[1] for token in config.split() if token.startswith("sms=")))
            first[sm_index] = str(sms)
            rows[1] = ",".join(first)
            trace.write_text("\n".join(rows) + "\n", encoding="utf-8")
            log = root / "trace_sm.log"
            out = root / "trace_sm.json"
            log.write_text(base_log, encoding="utf-8")
            result = run(
                "python3", str(NATIVE_VALIDATOR), str(log), "--trace", str(trace),
                "--allow-short", "--expected-gpu-uuid", expected_uuid,
                "--json", str(out),
            )
            self.assertEqual(result.returncode, 2, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
