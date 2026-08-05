#!/usr/bin/env python3
"""CPU-only tests for the production Tier-5 plan, validator, and runner."""

from __future__ import annotations

import json
import contextlib
import fcntl
import inspect
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import production_tier5 as harness
import production_tier5_campaign as campaign
import validate_production_tier5 as validator


HARNESS = BASE / "production_tier5.py"
VALIDATOR = BASE / "validate_production_tier5.py"
RUNNER = BASE / "run_production_tier5.sh"
FRAGMENT_RUNNER = BASE / "run_production_tier5_fragments.sh"
NSYS_SIDECAR = BASE / "run_production_tier5_nsys_sidecar.sh"
FAKE_GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
FAKE_FRAGMENT_INVOCATION_UUID = "11111111-1111-4111-8111-111111111111"


def make_attention_mode_record(mode: str, *, start: int = 0, count: int = 4096) -> dict:
    spec = harness.MODEL_SPECS["deepseek_v32"]
    enabled = mode == "on"
    valid_topk = harness.valid_topk_entry_count(start, count, spec.index_topk)
    output_slots = count * spec.index_topk
    diagnostics = {
        "pdl_mode": mode,
        "invalid_valid_entries": 0,
        "topk_tail_contract": harness.TOPK_TAIL_CONTRACT,
        "tail_slots_ignored": output_slots - valid_topk,
        "tail_nonminus_one_observed": 0,
        "duplicate_entries": 0,
        "score_reference": "deepgemm_full_n_logits_causal_masked",
        "score_violations": 0,
        "valid_set_symmetric_difference": 0,
        "exact_set_difference_role": "diagnostic_not_acceptance",
        "logits_quality_failure": 0,
        "acceptance_mismatches": 0,
        "indexer_logits_quality": {
            "actual_source": (
                "full-N vllm.utils.deep_gemm.fp8_fp4_mqa_logits("
                "clean_logits=False), then causal-valid-cell mask"
            ),
            "native_replay_pdl_mode": mode,
            "native_replay_calls": 1,
            "replay_total_seq_lens": 4096,
            "invalid_logits_contract": "UNSPECIFIED_IGNORED",
            "manual_reference": (
                "sum_h(relu(q_fp8_dequant@k_fp8_dequant.T)*weights)"
            ),
            "manual_reference_reuse": (
                "computed_once_per_chunk_and_shared_read_only_across_off_on"
            ),
            "quality_reduction": "streamed_query_rows_all_causal_valid_cells",
            "quality_query_row_batch": harness.LOGITS_QUALITY_QUERY_BATCH,
            "quality_reduction_dtype": "float64",
            "valid_elements": count * (2 * start + count + 1) // 2,
            "kernel_valid_nonfinite": 0,
            "manual_valid_nonfinite": 0,
            "row_quality_failures": 0,
            "calc_diff_limit_exclusive": harness.DEEPGEMM_CALC_DIFF_LIMIT,
            "calc_diff": 0.0,
            "row_calc_diff_limit_exclusive": harness.DEEPGEMM_ROW_CALC_DIFF_LIMIT,
            "row_calc_diff_max": 0.0,
            "row_calc_diff_p99": 0.0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "rms_abs_diff": 0.0,
            "manual_rms": 1.0,
            "status": "PASS",
        },
    }
    return {
        "pdl_mode": mode,
        "status": "PASS",
        "control": {
            "requested": enabled,
            "deep_gemm_readback": enabled,
            "deep_gemm_readback_after_validation": enabled,
            "flashinfer_enable_pdl": enabled,
            "topk_control": None,
        },
        "actual_indexer_calls": 1,
        "native_replay_calls": 1,
        "sparse_mla_calls": 1,
        "manual_reference_shared_read_only": True,
        "topk_mismatches": 0,
        "topk_diagnostics": diagnostics,
        "topk_valid_elements_checked": valid_topk,
        "topk_output_slots_observed": output_slots,
        "topk_tail_slots_ignored": output_slots - valid_topk,
        "topk_tail_contract": harness.TOPK_TAIL_CONTRACT,
        "attention_elements_checked": count * spec.attention_heads * spec.kv_lora_rank,
        "validation_scope": "all_rows_valid_topk_prefix_and_all_attention_elements",
    }


def make_attention_correctness_fixture() -> tuple[dict, dict, dict]:
    row_id = "deepseek_v32.operator_chain.seq4096"
    manifest = {
        "moe": {"tokens": 128},
        "shape_records": [
            {
                "model": "deepseek_v32",
                "seq": 4096,
                "num_query_chunks": 1,
                "query_chunk_tokens": 4096,
            }
        ],
    }
    rows = {
        row_id: {
            "row_id": row_id,
            "model": "deepseek_v32",
            "seq": 4096,
            "workload": "operator_chain",
            "pdl_modes": list(harness.PDL_MODES),
        }
    }
    pairs = 4096 * 4097 // 2
    correctness = {
        "schema": 1,
        "kind": "tier5_production_correctness",
        "status": "PASS",
        "all_expected_rows_present": True,
        "rows": [
            {
                "row_id": row_id,
                "status": "PASS",
                "all_query_rows_executed": 4096,
                "indexer_workload_geometry": "exact_causal_lower_triangle",
                "query_sampling": "NONE",
                "indexer_causal_pairs_executed": pairs,
                "indexer_causal_pair_formula": "S*(S+1)/2",
                "causal_pair_sampling": "NONE",
                "chunk_causal_pairs_sum": pairs,
                "chunk_pair_partition_verified": True,
                "correctness_pdl_modes": list(harness.PDL_MODES),
                "per_chunk_mode_correctness_complete": True,
                "chunks": [
                    {
                        "query_start": 0,
                        "query_count": 4096,
                        "indexer_workload_geometry": "exact_causal_lower_triangle",
                        "first_query_causal_key_count": 1,
                        "last_query_causal_key_count": 4096,
                        "indexer_causal_pairs_executed": pairs,
                        "query_sampling": "NONE",
                        "causal_pair_sampling": "NONE",
                        "manual_indexer_reference": {
                            "computation_count": 1,
                            "shared_read_only_across_modes": True,
                            "modes_using_reference": list(harness.PDL_MODES),
                        },
                        "mode_correctness": [
                            make_attention_mode_record("off"),
                            make_attention_mode_record("on"),
                        ],
                    }
                ],
            }
        ],
    }
    return manifest, rows, correctness


def make_gpu_exclusivity_fixture(root: Path) -> tuple[dict, dict]:
    target = {"index": 0, "uuid": FAKE_GPU_UUID, "name": "NVIDIA B200"}
    lease_id = "fixture-lease"
    harness.atomic_write_json(
        root / "gpu_identity.json",
        {
            "schema": 1,
            "kind": "gpu_identity",
            "status": "PASS",
            "errors": [],
            "phase": "production_global_lock_identity",
            "target_gpu": target,
        },
    )
    harness.atomic_write_json(
        root / "gpu_exclusivity_lease.json",
        {
            "schema": 1,
            "kind": "gpu_exclusivity_lease",
            "status": "PASS",
            "errors": [],
            "phase": "production_campaign_acquire",
            "lease_id": lease_id,
            "observation": {"target_gpu": target, "target_compute_processes": []},
        },
    )
    for label in ("pre", "post"):
        harness.atomic_write_json(
            root / f"gpu_{label}.json",
            {
                "schema": 1,
                "kind": "gpu_exclusivity_checkpoint",
                "status": "PASS",
                "errors": [],
                "phase": f"production_tier5_{label}",
                "lease_id": lease_id,
                "observation": {
                    "target_gpu": target,
                    "target_compute_processes": [],
                },
            },
        )
    raw_allowed = {
        "pid": 777,
        "gpu_uuid": FAKE_GPU_UUID,
        "name": "allowed-worker",
        "used_memory": "64",
    }
    allowed = {**raw_allowed, "proc_start_ticks": 12345}
    raw_residual = {**raw_allowed, "name": "[No data]"}
    records = []
    for sequence in range(6):
        allowed_processes = [allowed] if sequence == 1 else []
        residual_processes = []
        target_processes = [raw_allowed] if allowed_processes else []
        if sequence in (2, 3):
            residual_processes = [
                {
                    **raw_residual,
                    "proc_start_ticks": None,
                    "previous_allowed_start_ticks": 12345,
                    "classification": "allowed_post_exit_nvidia_smi_residual",
                    "residual_observation_number": sequence - 1,
                    "residual_observation_limit": validator.EXIT_RESIDUAL_LIMIT,
                }
            ]
            target_processes = [raw_residual]
        records.append(
            {
                "schema": 1,
                "sequence": sequence,
                "observed_at": f"synthetic-{sequence}",
                "query_started_monotonic_ns": sequence * 1_000_000,
                "query_finished_monotonic_ns": sequence * 1_000_000 + 500_000,
                "query_duration_ms": 0.5,
                "query_errors": [],
                "observation": {
                    "target_gpu": target,
                    "target_compute_processes": target_processes,
                },
                "allowed_target_processes": allowed_processes,
                "allowed_exit_residual_processes": residual_processes,
                "foreign_target_processes": [],
            }
        )
    observations = root / "gpu_observations.ndjson"
    observations.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    harness.atomic_write_json(
        root / "gpu_monitor.json",
        {
            "schema": 1,
            "kind": "gpu_exclusivity_monitor",
            "status": "PASS",
            "errors": [],
            "accepted_timing": 0,
            "measurement_role": "gpu_exclusivity_monitor_only",
            "phase": "production_tier5",
            "lease_id": lease_id,
            "target_gpu": target,
            "poll_interval_ms": 50,
            "query_timeout_ms": 2000,
            "start_barrier_complete": True,
            "ready_record_written": True,
            "baseline_observation_sequence": 0,
            "coverage_model": "bounded_interval_nvidia_smi_process_sampling",
            "coverage_limit": (
                "foreign GPU processes wholly between completed samples may not be observed"
            ),
            "require_allowed_process": True,
            "allowed_observation_count": 1,
            "allowed_exit_residual_observation_count": 2,
            "allowed_processes": [{"pid": 777, "proc_start_ticks": 12345}],
            "allowed_exit_residual_processes": [
                {
                    "pid": 777,
                    "previous_allowed_start_ticks": 12345,
                    "residual_observation_count": 2,
                }
            ],
            "allowed_exit_residual_max_observations_per_pid": (
                validator.EXIT_RESIDUAL_LIMIT
            ),
            "max_allowed_exit_residual_observations_observed": 2,
            "exit_residual_policy": validator.EXIT_RESIDUAL_POLICY,
            "foreign_processes_detected": False,
            "query_failure_detected": False,
            "terminated_on_failure": False,
            "watch_pid": 777,
            "watch_root_start_ticks": 12345,
            "observation_count": len(records),
            "observations_path": str(observations),
            "observations_sha256": harness.sha256_file(observations),
        },
    )
    manifest = {
        "expected_gpu_uuid": FAKE_GPU_UUID,
        "expected_gpu_index": 0,
        "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        "device": {
            "uuid": FAKE_GPU_UUID,
            "runtime_ordinal": 0,
            "runtime_ordinal_zero": True,
            "cuda_visible_devices_selector": "0",
            "process_pid": 777,
            "process_start_ticks": 12345,
        },
    }
    lock_path = f"/tmp/cta_pdl_gpu_{FAKE_GPU_UUID}.lock"
    evidence = {
        "expected_gpu_uuid": FAKE_GPU_UUID,
        "expected_gpu_index": 0,
        "global_lock_scope": "target_uuid",
        "global_lock_key_sha256": harness.sha256_bytes(FAKE_GPU_UUID.encode()),
        "global_lock_path_sha256": harness.sha256_bytes(lock_path.encode()),
        "monitor_interval_ms": 50,
        "query_timeout_ms": 2000,
    }
    return manifest, evidence


def short_campaign_contract() -> dict:
    return campaign.build_campaign_contract(
        {
            "models": ["deepseek_v32"],
            "seqs": [4096],
            "workloads": ["operator_chain"],
            "warmup": 0,
            "repeats": 1,
            "allow_short": True,
            "seed": 20260805,
            "max_logits_mb": harness.FORMAL_MAX_LOGITS_MB,
            "max_query_chunk": 4096,
            "moe_experts": 32,
            "moe_topk": 8,
            "moe_tokens": 128,
            "backend": "flashinfer",
            "required_device_substring": "B200",
            "monitor_interval_ms": 50,
            "query_timeout_ms": 2000,
        }
    )


def make_fragment_fixture(
    campaign_root: Path,
    contract: dict,
    binding: dict,
    ordinal: int,
    invocation_uuid: str,
) -> tuple[Path, Path]:
    row = contract["ordered_matrix"][ordinal]
    rows_root = campaign_root / "rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    final = rows_root / campaign.row_directory_name(ordinal, row["row_id"])
    stage = rows_root / (final.name + ".inprogress.fixture")
    stage.mkdir()
    argv = campaign.instantiate_fragment_argv(
        contract, binding, row, ordinal, stage, final, invocation_uuid
    )
    args = harness.parse_args(argv[2:])
    device = {
        "query_performed": True,
        "name": "NVIDIA B200",
        "uuid": FAKE_GPU_UUID,
        "uuid_source": "synthetic_fixture",
        "runtime_ordinal": 0,
        "runtime_ordinal_zero": True,
        "compute_capability": "10.0",
        "total_memory_bytes": 192_000_000_000,
        "multi_processor_count": 160,
        "driver_version_raw": 12080,
        "torch_cuda_version": "12.8",
        "device_index_inside_visible_set": 0,
        "cuda_visible_devices_selector": "0",
        "process_pid": 777,
        "process_start_ticks": 12345,
    }
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_guard = os.environ.get("TIER5_PRODUCTION_GPU_ALLOWED")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TIER5_PRODUCTION_GPU_ALLOWED"] = "1"
    try:
        manifest = harness.make_manifest(args, runtime_device=device)
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        if old_guard is None:
            os.environ.pop("TIER5_PRODUCTION_GPU_ALLOWED", None)
        else:
            os.environ["TIER5_PRODUCTION_GPU_ALLOWED"] = old_guard
    harness.atomic_write_json(stage / "manifest.json", manifest)

    make_gpu_exclusivity_fixture(stage)
    phase = (
        f"production_tier5_fragment_{ordinal}_{row['row_id']}_"
        f"{invocation_uuid}"
    )
    for name, suffix in (
        ("gpu_identity.json", "identity"),
        ("gpu_exclusivity_lease.json", "acquire"),
        ("gpu_pre.json", "pre"),
        ("gpu_post.json", "post"),
        ("gpu_monitor.json", "monitor"),
    ):
        path = stage / name
        value = json.loads(path.read_text())
        value["phase"] = f"{phase}_{suffix}"
        harness.atomic_write_json(path, value)

    fragment = manifest["fragment"]
    row_seed = fragment["derived_row_seed"]
    if row["workload"] == "moe32":
        samples = [
            {
                "schema": 1,
                "row_id": row["row_id"],
                "model": row["model"],
                "seq": None,
                "workload": "moe32",
                "component": "fused_topk_plus_fused_experts",
                "pdl_mode": "framework_default_uncontrolled",
                "repeat": 0,
                "elapsed_ms": 1.0,
                "poison_epoch": 1,
                "poison_verified": True,
                "fresh_output_allocation": True,
                "timed_validation": False,
            }
        ]
        correctness_row = {
            "row_id": row["row_id"],
            "status": "PASS",
            "tokens_checked": 128,
            "output_elements_checked": 128 * 7168,
            "routing_assignments_checked": 128 * 8,
            "experts": 32,
            "topk": 8,
            "max_abs": 0.0,
        }
    else:
        samples = []
        for event, repeat, component, enabled, pair_order in harness.paired_timing_schedule(
            1, validator.expected_components(row["workload"])
        ):
            samples.append(
                {
                    "schema": 1,
                    "row_id": row["row_id"],
                    "model": row["model"],
                    "seq": row["seq"],
                    "workload": row["workload"],
                    "component": component,
                    "pdl_mode": "on" if enabled else "off",
                    "repeat": repeat,
                    "elapsed_ms": 1.0 + event / 100,
                    "poison_epoch": repeat + 1,
                    "poison_verified": True,
                    "timed_validation": False,
                    "timing_event_ordinal": event,
                    "timing_pair_ordinal": event // 2,
                    "pair_order": pair_order,
                    "pair_same_process_pid": 777,
                    "pair_same_process_start_ticks": 12345,
                }
            )
        _, _, correctness_fixture = make_attention_correctness_fixture()
        correctness_row = correctness_fixture["rows"][0]
    for sample in samples:
        sample.update(
            fragment_row_ordinal=ordinal,
            invocation_uuid=invocation_uuid,
            campaign_contract_sha256=contract["contract_sha256"],
            campaign_fingerprint_sha256=binding[
                "campaign_fingerprint_sha256"
            ],
            derived_row_seed=row_seed,
        )
    correctness = {
        "schema": 1,
        "kind": "tier5_production_correctness",
        "status": "PASS",
        "execution_scope": "row_fragment",
        "fragment_row_id": row["row_id"],
        "fragment": fragment,
        "rows": [correctness_row],
        "all_expected_rows_present": True,
    }
    sample_payload = "".join(
        json.dumps(sample, sort_keys=True) + "\n" for sample in samples
    ).encode()
    harness.atomic_write_bytes(stage / "samples.jsonl", sample_payload)
    harness.atomic_write_json(stage / "correctness.json", correctness)
    result = {
        "schema": 1,
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
        "manifest_sha256": harness.sha256_file(stage / "manifest.json"),
        "samples_sha256": harness.sha256_file(stage / "samples.jsonl"),
        "correctness_sha256": harness.sha256_file(stage / "correctness.json"),
        "sample_count": len(samples),
        "summaries": harness.summarize_samples(samples, 20260805),
        "execution_scope": "row_fragment",
        "fragment": fragment,
    }
    harness.atomic_write_json(stage / "result.json", result)
    harness.atomic_write_json(
        stage / "terminal_status.json",
        {
            "schema": 1,
            "status": "CANDIDATE",
            "accepted_timing": 0,
            "accepted_workload_timing": 0,
            "accepted_CTA_bracket": 0,
            "measurement_emitted": True,
            "result_sha256": harness.sha256_file(stage / "result.json"),
            "execution_scope": "row_fragment",
            "fragment": fragment,
        },
    )
    (stage / "runner.log").write_text("synthetic runner\n")
    (stage / "harness.log").write_text("synthetic harness\n")
    return stage, final


def run(*argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=BASE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_fake_nvidia_smi(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -eu
case "${1:-}" in
  --query-gpu=index,uuid,name)
    echo "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, NVIDIA B200"
    ;;
  --query-compute-apps=*)
    state="${FAKE_GPU_STATE:?}"
    count=0
    [ ! -f "${state}" ] || count="$(cat "${state}")"
    count=$((count + 1))
    echo "${count}" > "${state}"
    if [ "${FAKE_GPU_MODE:-idle}" = "foreign_after_pre" ] && [ "${count}" -ge 3 ]; then
      echo "999999, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, foreign-worker, 64"
    fi
    ;;
  *)
    echo "unexpected fake nvidia-smi query: ${1:-}" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_fake_fragment_process_stack(root: Path) -> Path:
    """Install CPU-only python/monitor/harness shims for runner lifecycle tests."""
    shim_dir = root / "shim-bin"
    shim_dir.mkdir()
    python_shim = shim_dir / "python3"
    monitor = root / "fake-fragment-monitor"
    child = root / "fake-fragment-child"
    python_shim.write_text(
        r'''#!/usr/bin/env bash
set -eu
if [ "${1:-}" = "./gpu_exclusivity.py" ] && [ "${2:-}" = "monitor" ]; then
    shift 2
    exec "${FAKE_FRAGMENT_MONITOR:?}" "$@"
fi
if [ "${1:-}" = "./production_tier5.py" ]; then
    shift
    exec "${FAKE_FRAGMENT_CHILD:?}" "$@"
fi
exec "${REAL_PYTHON:?}" "$@"
''',
        encoding="utf-8",
    )
    monitor.write_text(
        r'''#!/usr/bin/env bash
set -uo pipefail
ready_file=""
watch_pid=""
json_file=""
observations=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --ready-file) ready_file="$2"; shift 2 ;;
        --watch-pid) watch_pid="$2"; shift 2 ;;
        --json) json_file="$2"; shift 2 ;;
        --observations) observations="$2"; shift 2 ;;
        *) shift ;;
    esac
done
printf '%s\n' "$$" > "${FAKE_MONITOR_PID_FILE:?}"
: > "${observations:?}"
printf '{"schema":1,"kind":"gpu_exclusivity_monitor_ready","status":"READY","watch_pid":%s}\n' \
    "${watch_pid:?}" > "${ready_file:?}"
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
if [ "${FAKE_FRAGMENT_MONITOR_MODE:-follow}" = "exit_after_child_start" ]; then
    for _ in $(seq 1 500); do
        if [ -s "${FAKE_CHILD_READY_FILE:?}" ]; then
            sleep 0.5
            exit 9
        fi
        sleep 0.01
    done
    exit 10
fi
while kill -0 "${watch_pid}" 2>/dev/null; do
    state="$(ps -o stat= -p "${watch_pid}" 2>/dev/null | tr -d '[:space:]')"
    [ -n "${state}" ] && [ "${state#Z}" = "${state}" ] || break
    sleep 0.02
done
printf '{"schema":1,"kind":"gpu_exclusivity_monitor","status":"FAIL","errors":["cpu fixture"]}\n' \
    > "${json_file:?}"
exit 11
''',
        encoding="utf-8",
    )
    child.write_text(
        r'''#!/usr/bin/env bash
set -uo pipefail
grandchild=""
cleanup() {
    trap - TERM INT HUP
    if [ -n "${grandchild}" ]; then
        kill -TERM "${grandchild}" 2>/dev/null || true
        wait "${grandchild}" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup TERM INT HUP
(
    trap 'exit 0' TERM INT HUP
    while :; do sleep 0.1; done
) &
grandchild=$!
printf '%s %s %s\n' "$$" "${grandchild}" \
    "$(ps -o pgid= -p "$$" | tr -d '[:space:]')" \
    > "${FAKE_CHILD_READY_FILE:?}"
while :; do sleep 0.1; done
''',
        encoding="utf-8",
    )
    for executable in (python_shim, monitor, child):
        executable.chmod(0o755)
    return shim_dir


def process_identity_is_live(pid: int, start_ticks: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        raw = stat.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    fields = raw.rsplit(") ", 1)[1].split()
    return int(fields[19]) == start_ticks and fields[0] != "Z"


def proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(raw.rsplit(") ", 1)[1].split()[19])


def wait_for_process_fixture(
    path: Path, process: subprocess.Popen[str], timeout: float = 15.0
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return path.read_text(encoding="utf-8").strip()
        if process.poll() is not None:
            output = process.communicate()[0]
            raise AssertionError(
                f"fixture runner exited before {path.name}: "
                f"rc={process.returncode}\n{output}"
            )
        time.sleep(0.01)
    process.kill()
    output = process.communicate()[0]
    raise AssertionError(f"timed out waiting for {path}:\n{output}")


def assert_process_identities_stop(
    test: unittest.TestCase, identities: list[tuple[int, int]]
) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not any(process_identity_is_live(*identity) for identity in identities):
            return
        time.sleep(0.02)
    live = [
        pid for pid, start in identities if process_identity_is_live(pid, start)
    ]
    test.fail(f"runner left live fixture processes: {live}")


def write_fake_nsys(path: Path) -> None:
    """Write a CPU-only nsys double that materializes the sidecar inputs."""
    path.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("NVIDIA Nsight Systems version fake-test")
    raise SystemExit(0)
if args and args[0] == "profile":
    prefix = Path(args[args.index("--output") + 1])
    Path(str(prefix) + ".nsys-rep").write_bytes(b"fake-nsys-report")
    campaign = Path(os.environ["RESULTS"])
    model = os.environ["MODELS"]
    seq = int(os.environ["SEQS"])
    workload = os.environ["WORKLOADS"]
    row_id = f"{model}.{workload}.seq{seq}"
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / "campaign_contract.json").write_text(json.dumps({
        "contract_sha256": "c" * 64,
        "ordered_matrix": [{
            "model": model,
            "seq": seq,
            "workload": workload,
            "row_id": row_id,
            "pdl_modes": ["off", "on"],
        }],
    }))
    (campaign / "campaign_binding.json").write_text(json.dumps({
        "campaign_fingerprint_sha256": "f" * 64,
    }))
    fragment = campaign / "rows" / f"000_{row_id}"
    fragment.mkdir(parents=True)
    (fragment / "fragment.done.json").write_text(json.dumps({
        "status": "PASS", "accepted_workload_timing": 0,
        "invocation_uuid": "11111111-1111-4111-8111-111111111111",
    }))
    raise SystemExit(0)
if args and args[0] == "stats":
    print(os.environ.get("FAKE_NSYS_STATS", ""))
    raise SystemExit(0)
print("unexpected fake nsys invocation", args, file=sys.stderr)
raise SystemExit(2)
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


class ShapePlanTests(unittest.TestCase):
    def test_exact_model_shapes(self) -> None:
        ds = harness.MODEL_SPECS["deepseek_v32"]
        glm = harness.MODEL_SPECS["glm5"]
        self.assertEqual(
            (ds.hidden_size, ds.q_lora_rank, ds.attention_heads, ds.index_heads),
            (7168, 1536, 128, 64),
        )
        self.assertEqual(
            (glm.hidden_size, glm.q_lora_rank, glm.attention_heads, glm.index_heads),
            (6144, 2048, 64, 32),
        )
        for spec in (ds, glm):
            self.assertEqual(spec.kv_lora_rank, 512)
            self.assertEqual(spec.qk_rope_head_dim, 64)
            self.assertEqual(spec.index_head_dim, 128)
            self.assertEqual(spec.index_topk, 2048)
            self.assertEqual(spec.experts_per_token, 8)

    def test_query_chunk_matrix(self) -> None:
        self.assertEqual(
            [
                harness.query_chunk_tokens(
                    seq,
                    harness.FORMAL_MAX_LOGITS_MB,
                    harness.FORMAL_MAX_QUERY_CHUNK,
                )
                for seq in harness.FORMAL_SEQS
            ],
            [4096, 4096, 4096, 4096],
        )

    def test_one_million_is_extreme_and_out_of_range(self) -> None:
        for spec in harness.MODEL_SPECS.values():
            record = harness.shape_record(
                spec,
                1048576,
                harness.FORMAL_MAX_LOGITS_MB,
                harness.FORMAL_MAX_QUERY_CHUNK,
                32,
                4096,
            )
            self.assertTrue(record["extreme"])
            self.assertFalse(record["within_official_position_range"])
            self.assertEqual(record["all_query_rows"], 1048576)
            self.assertEqual(record["query_sampling"], "NONE")
            self.assertEqual(record["causal_pair_sampling"], "NONE")
            self.assertEqual(
                record["indexer_causal_pairs"], 1048576 * 1048577 // 2
            )
            self.assertEqual(record["query_chunk_tokens"], 4096)
            self.assertEqual(record["num_query_chunks"], 256)

    def test_exact_causal_pair_partition_for_formal_matrix(self) -> None:
        spec = harness.MODEL_SPECS["deepseek_v32"]
        for seq in harness.FORMAL_SEQS:
            record = harness.shape_record(
                spec,
                seq,
                harness.FORMAL_MAX_LOGITS_MB,
                harness.FORMAL_MAX_QUERY_CHUNK,
                32,
                4096,
            )
            chunk = record["query_chunk_tokens"]
            expected_pairs = seq * (seq + 1) // 2
            independent_partition = sum(
                count * (2 * start + count + 1) // 2
                for start in range(0, seq, chunk)
                for count in (min(chunk, seq - start),)
            )
            self.assertEqual(record["indexer_causal_pairs"], expected_pairs)
            self.assertEqual(record["chunk_causal_pairs_sum"], expected_pairs)
            self.assertEqual(independent_partition, expected_pairs)
            self.assertTrue(record["chunk_pair_partition_verified"])
            self.assertEqual(record["indexer_workload_geometry"], "exact_causal_lower_triangle")
            self.assertNotIn("full_query_rows", record)
            self.assertNotIn("full_indexer_fma_flops", record)

    def test_valid_topk_prefix_formula(self) -> None:
        topk = 2048
        for start, count in ((0, 4096), (4096, 4096), (0, 128), (2047, 4)):
            independent = sum(
                min(position + 1, topk)
                for position in range(start, start + count)
            )
            self.assertEqual(
                harness.valid_topk_entry_count(start, count, topk), independent
            )

    def test_deepgemm_weighted_relu_authority_is_hash_bound(self) -> None:
        self.assertEqual(
            harness.sha256_file(harness.DEEPGEMM_MQA_HEADER),
            harness.DEEPGEMM_MQA_HEADER_SHA256,
        )
        header = harness.DEEPGEMM_MQA_HEADER.read_text(encoding="utf-8")
        self.assertIn("Accumulate weighted ReLU", header)
        self.assertIn("fmaxf(accum[j], 0)", header)
        reference_source = inspect.getsource(
            harness.ProductionRuntime.indexer_reference
        )
        self.assertIn("head_scores.relu_()", reference_source)
        self.assertIn("clean_logits=False", reference_source)
        self.assertIn('state["k_quant"].shape[0]', reference_source)

    def test_memory_formulas(self) -> None:
        spec = harness.MODEL_SPECS["deepseek_v32"]
        record = harness.shape_record(
            spec,
            1048576,
            harness.FORMAL_MAX_LOGITS_MB,
            harness.FORMAL_MAX_QUERY_CHUNK,
            32,
            4096,
        )
        tensors = record["tensor_bytes"]
        self.assertEqual(tensors["indexer_cache"], 1048576 * 132)
        self.assertEqual(tensors["mla_bf16_cache"], 1048576 * 1152)
        self.assertEqual(tensors["indexer_workspace_capacity"], 40 * 1048576 * 132)
        self.assertEqual(tensors["fp32_logits_chunk"], 4096 * 1048576 * 4)
        self.assertEqual(
            tensors["moe32_weights_bf16"], 32 * 3 * 7168 * 2048 * 2
        )

    def test_streamed_logits_quality_matches_full_matrix_formula(self) -> None:
        import math
        import torch

        generator = torch.Generator().manual_seed(20260805)
        kernel = torch.randn(13, 37, generator=generator, dtype=torch.float32)
        manual = kernel + 1e-3 * torch.randn(
            13, 37, generator=generator, dtype=torch.float32
        )
        limits = torch.tensor(
            [1, 3, 7, 11, 13, 17, 19, 23, 29, 31, 33, 35, 37]
        )
        mask = torch.arange(37).unsqueeze(0) < limits.unsqueeze(1)
        kernel[~mask] = float("nan")
        manual[~mask] = float("inf")
        original_kernel = kernel.clone()
        original_manual = manual.clone()

        streamed = harness.streamed_logits_quality_statistics(
            torch, kernel, manual, mask, query_row_batch=3
        )
        kernel_quality = kernel.masked_fill(~mask, 0.0).double()
        manual_quality = manual.masked_fill(~mask, 0.0).double()
        denominator = (kernel_quality.square() + manual_quality.square()).sum()
        numerator = 2 * (kernel_quality * manual_quality).sum()
        expected_calc_diff = float((1 - numerator / denominator).item())
        row_denominator = (
            kernel_quality.square() + manual_quality.square()
        ).sum(dim=1)
        row_numerator = 2 * (kernel_quality * manual_quality).sum(dim=1)
        row_similarity = torch.where(
            row_denominator == 0,
            torch.ones_like(row_denominator),
            row_numerator / row_denominator,
        )
        row_calc_diff = 1 - row_similarity
        valid_kernel64 = kernel.masked_select(mask).double()
        valid_manual64 = manual.masked_select(mask).double()
        valid_abs64 = (valid_kernel64 - valid_manual64).abs()

        self.assertEqual(streamed["valid_elements"], int(mask.sum().item()))
        self.assertEqual(streamed["kernel_valid_nonfinite"], 0)
        self.assertEqual(streamed["manual_valid_nonfinite"], 0)
        self.assertAlmostEqual(streamed["calc_diff"], expected_calc_diff, places=14)
        self.assertAlmostEqual(
            streamed["row_calc_diff_max"], float(row_calc_diff.max().item()), places=14
        )
        self.assertAlmostEqual(
            streamed["row_calc_diff_p99"],
            float(torch.quantile(row_calc_diff, 0.99).item()),
            places=14,
        )
        self.assertEqual(streamed["row_quality_failures"], 0)
        self.assertAlmostEqual(
            streamed["max_abs_diff"], float(valid_abs64.max()), places=14
        )
        self.assertAlmostEqual(
            streamed["mean_abs_diff"], float(valid_abs64.mean()), places=14
        )
        self.assertAlmostEqual(
            streamed["rms_abs_diff"],
            float(valid_abs64.square().mean().sqrt()),
            places=14,
        )
        self.assertAlmostEqual(
            streamed["manual_rms"],
            float(valid_manual64.square().mean().sqrt()),
            places=14,
        )
        torch.testing.assert_close(kernel, original_kernel, equal_nan=True)
        torch.testing.assert_close(manual, original_manual, equal_nan=True)

        valid_nonfinite = original_kernel.clone()
        valid_nonfinite[0, 0] = float("inf")
        boundary = harness.streamed_logits_quality_statistics(
            torch, valid_nonfinite, original_manual, mask, query_row_batch=4
        )
        self.assertEqual(boundary["kernel_valid_nonfinite"], 1)
        self.assertTrue(math.isnan(boundary["calc_diff"]))
        self.assertGreaterEqual(boundary["row_quality_failures"], 1)

        zero_mask = torch.ones(2, 5, dtype=torch.bool)
        zeros = torch.zeros(2, 5)
        zero_boundary = harness.streamed_logits_quality_statistics(
            torch, zeros, zeros, zero_mask, query_row_batch=1
        )
        self.assertTrue(math.isnan(zero_boundary["calc_diff"]))
        self.assertEqual(zero_boundary["row_calc_diff_max"], 0.0)
        self.assertEqual(zero_boundary["row_quality_failures"], 0)

    def test_long_context_execution_policy_is_frozen(self) -> None:
        contract = harness.long_context_execution_contract()
        self.assertEqual(
            contract["attention_reference"]["row_batch"],
            harness.ATTENTION_REFERENCE_ROW_BATCH,
        )
        self.assertEqual(
            contract["logits_quality"]["query_row_batch"],
            harness.LOGITS_QUALITY_QUERY_BATCH,
        )
        self.assertEqual(contract["logits_quality"]["sampling"], "NONE")
        self.assertEqual(
            contract["cuda_cache_release"]["cadence_chunks"], 1
        )
        self.assertFalse(
            contract["cuda_cache_release"]["inside_cuda_event_timing"]
        )
        args = harness.parse_args(["--output-dir", "/tmp/never-created"])
        self.assertEqual(args.max_logits_mb, harness.FORMAL_MAX_LOGITS_MB)
        self.assertEqual(args.max_query_chunk, harness.FORMAL_MAX_QUERY_CHUNK)
        manifest = harness.make_manifest(args, runtime_device=None)
        self.assertEqual(manifest["long_context_execution"], contract)
        self.assertEqual(
            manifest["correctness_contract"]["attention_reference_row_batch"],
            32,
        )
        chain_source = inspect.getsource(
            harness.ProductionRuntime.benchmark_operator_chain
        )
        layer_source = inspect.getsource(
            harness.ProductionRuntime.benchmark_layer_like
        )
        release_source = inspect.getsource(
            harness.ProductionRuntime.release_completed_chunk_cache
        )
        self.assertIn("del chain_call, mla_call, index_call", chain_source)
        self.assertIn("del manual_indexer_reference, result", layer_source)
        self.assertIn("del forward", layer_source)
        self.assertIn("self.torch.cuda.empty_cache()", release_source)
        self.assertIn("CUDA_CACHE_RELEASE_CADENCE_CHUNKS", release_source)

    def test_exact_attention_weight_count(self) -> None:
        ds = sum(harness.attention_weight_elements(harness.MODEL_SPECS["deepseek_v32"]).values())
        glm = sum(harness.attention_weight_elements(harness.MODEL_SPECS["glm5"]).values())
        self.assertEqual(ds, 201_064_448)
        self.assertEqual(glm, 174_391_296)

    def test_formal_gpu_budget_is_not_mislabelled_as_zero(self) -> None:
        coordinates = harness.experiment_contract()["coordinates"]
        self.assertNotIn("gpu_budget_this_revision_minutes", coordinates)
        self.assertEqual(
            coordinates["gpu_budget"]["formal_execution_budget"],
            "UNBOUNDED_BY_HARNESS",
        )

    def test_formal_parser_rejects_short_matrix(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                harness.parse_args(
                    [
                        "--output-dir",
                        "/tmp/never-created",
                        "--seqs",
                        "4096",
                    ]
                )

    def test_short_parser_requires_explicit_allow(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                harness.parse_args(
                    ["--output-dir", "/tmp/never-created", "--repeats", "3"]
                )
        args = harness.parse_args(
            [
                "--output-dir",
                "/tmp/never-created",
                "--models",
                "deepseek_v32",
                "--seqs",
                "4096",
                "--workloads",
                "operator_chain",
                "--repeats",
                "3",
                "--allow-short",
            ]
        )
        self.assertTrue(args.allow_short)

    def test_formal_parser_replays_every_frozen_run_control(self) -> None:
        for extra in (
            ("--warmup", "1"),
            ("--repeats", "32"),
            ("--moe-tokens", "128"),
            ("--max-logits-mb", "512"),
            ("--max-query-chunk", "2048"),
            ("--seed", "7"),
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    harness.parse_args(
                        ["--output-dir", "/tmp/never-created", *extra]
                    )


class DualModeCorrectnessTests(unittest.TestCase):
    def test_validator_accepts_exact_off_on_full_reference_fixture(self) -> None:
        manifest, rows, correctness = make_attention_correctness_fixture()
        errors: list[str] = []
        validator.validate_correctness(manifest, rows, correctness, errors)
        self.assertEqual(errors, [])

    def test_operator_emitter_seals_realistic_mode_scoped_schema(self) -> None:
        manifest, rows, correctness = make_attention_correctness_fixture()
        ideal_chunk = correctness["rows"][0]["chunks"][0]
        raw_runtime_evidence = {
            "manual_indexer_reference": ideal_chunk[
                "manual_indexer_reference"
            ],
            "mode_correctness": [
                {
                    key: value
                    for key, value in mode_record.items()
                    if key != "validation_scope"
                }
                for mode_record in ideal_chunk["mode_correctness"]
            ],
        }
        emitted = harness.seal_operator_chain_chunk_correctness(
            raw_runtime_evidence,
            chunk_index=0,
            start=0,
            count=4096,
        )
        self.assertNotIn("validation_scope", emitted)
        self.assertEqual(
            [record["validation_scope"] for record in emitted["mode_correctness"]],
            [harness.OPERATOR_VALIDATION_SCOPE] * 2,
        )
        correctness["rows"][0]["chunks"] = [emitted]
        errors: list[str] = []
        validator.validate_correctness(manifest, rows, correctness, errors)
        self.assertEqual(errors, [])

    def test_validator_rejects_exact_historical_runtime_schema_drift(self) -> None:
        manifest, rows, correctness = make_attention_correctness_fixture()
        chunk = correctness["rows"][0]["chunks"][0]
        chunk["validation_scope"] = harness.OPERATOR_VALIDATION_SCOPE
        for mode_record in chunk["mode_correctness"]:
            del mode_record["validation_scope"]
        errors: list[str] = []
        validator.validate_correctness(manifest, rows, correctness, errors)
        self.assertEqual(
            errors,
            [
                "correctness_legacy_flat_mode_evidence:"
                "deepseek_v32.operator_chain.seq4096:0",
                "operator_validation_scope:"
                "deepseek_v32.operator_chain.seq4096:0:off",
                "operator_validation_scope:"
                "deepseek_v32.operator_chain.seq4096:0:on",
            ],
        )

    def test_operator_emitter_rejects_flat_scope_input(self) -> None:
        _, _, correctness = make_attention_correctness_fixture()
        chunk = correctness["rows"][0]["chunks"][0]
        with self.assertRaisesRegex(RuntimeError, "legacy flat mode evidence"):
            harness.seal_operator_chain_chunk_correctness(
                {
                    "manual_indexer_reference": chunk[
                        "manual_indexer_reference"
                    ],
                    "mode_correctness": chunk["mode_correctness"],
                    "validation_scope": harness.OPERATOR_VALIDATION_SCOPE,
                },
                chunk_index=0,
                start=0,
                count=4096,
            )

    def test_validator_rejects_missing_off_mode(self) -> None:
        manifest, rows, correctness = make_attention_correctness_fixture()
        modes = correctness["rows"][0]["chunks"][0]["mode_correctness"]
        modes[:] = [record for record in modes if record["pdl_mode"] == "on"]
        errors: list[str] = []
        validator.validate_correctness(manifest, rows, correctness, errors)
        self.assertIn(
            "correctness_mode_set:deepseek_v32.operator_chain.seq4096:0",
            errors,
        )

    def test_validator_rejects_duplicate_mode(self) -> None:
        manifest, rows, correctness = make_attention_correctness_fixture()
        modes = correctness["rows"][0]["chunks"][0]["mode_correctness"]
        modes[0] = dict(modes[1])
        errors: list[str] = []
        validator.validate_correctness(manifest, rows, correctness, errors)
        self.assertIn(
            "correctness_mode_set:deepseek_v32.operator_chain.seq4096:0",
            errors,
        )

    def test_validator_rejects_off_readback_tamper(self) -> None:
        manifest, rows, correctness = make_attention_correctness_fixture()
        off = correctness["rows"][0]["chunks"][0]["mode_correctness"][0]
        off["control"]["deep_gemm_readback_after_validation"] = True
        errors: list[str] = []
        validator.validate_correctness(manifest, rows, correctness, errors)
        self.assertIn(
            "control_indexer_final_readback:deepseek_v32.operator_chain.seq4096:0:off",
            errors,
        )

    def test_validator_rejects_mode_specific_replay_tamper(self) -> None:
        manifest, rows, correctness = make_attention_correctness_fixture()
        off = correctness["rows"][0]["chunks"][0]["mode_correctness"][0]
        off["topk_diagnostics"]["indexer_logits_quality"][
            "native_replay_pdl_mode"
        ] = "on"
        errors: list[str] = []
        validator.validate_correctness(manifest, rows, correctness, errors)
        self.assertIn(
            "indexer_logits_mode:deepseek_v32.operator_chain.seq4096:0:off",
            errors,
        )

    def test_harness_executes_both_modes_before_timing(self) -> None:
        chain_source = inspect.getsource(
            harness.ProductionRuntime.validate_chain_chunk
        )
        layer_source = inspect.getsource(
            harness.ProductionRuntime.benchmark_layer_like
        )
        reference_source = inspect.getsource(
            harness.ProductionRuntime.indexer_reference
        )
        for source in (chain_source, layer_source):
            self.assertIn("for mode in PDL_MODES", source)
            self.assertIn('"mode_correctness"', source)
            self.assertIn('"deep_gemm_readback_after_validation"', source)
        self.assertIn("manual_reference=manual_reference", chain_source)
        self.assertIn("native replay PDL control mismatch", reference_source)
        self.assertIn('"native_replay_pdl_mode"', reference_source)


class DryRunTests(unittest.TestCase):
    def make_dry_run(self, root: Path, *, formal: bool = False) -> subprocess.CompletedProcess[str]:
        argv = [sys.executable, str(HARNESS), "--output-dir", str(root)]
        if not formal:
            argv.extend(
                [
                    "--models",
                    "deepseek_v32",
                    "--seqs",
                    "4096",
                    "--workloads",
                    "operator_chain",
                    "--repeats",
                    "3",
                    "--allow-short",
                    "--moe-tokens",
                    "128",
                ]
            )
        return run(*argv)

    def test_import_is_cuda_free(self) -> None:
        script = (
            "import sys; import production_tier5; "
            "print(int('torch' in sys.modules), int('vllm' in sys.modules), "
            "int('flashinfer' in sys.modules))"
        )
        completed = run(sys.executable, "-c", script)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(completed.stdout.strip(), "0 0 0")

    def test_dry_run_and_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dry"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("status=NOT_EXECUTED", completed.stdout)
            self.assertFalse((root / "samples.jsonl").exists())
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertFalse(manifest["device"]["query_performed"])
            self.assertEqual(manifest["accepted_timing"], 0)
            self.assertEqual(manifest["accepted_workload_timing"], 0)
            self.assertEqual(manifest["accepted_CTA_bracket"], 0)
            self.assertEqual(
                manifest["correctness_contract"], harness.correctness_contract()
            )
            self.assertEqual(
                manifest["correctness_contract"]["topk_tail"],
                "UNSPECIFIED_IGNORED",
            )
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "PASS", result["errors"])

    def test_formal_dry_run_has_twenty_six_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            completed = self.make_dry_run(root, formal=True)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(manifest["expected_matrix"]), 26)
            self.assertEqual(len(manifest["shape_records"]), 8)

    def test_double_gpu_guard_fails_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "guard"
            env = os.environ.copy()
            env.pop("TIER5_PRODUCTION_GPU_ALLOWED", None)
            completed = run(
                sys.executable,
                str(HARNESS),
                "--output-dir",
                str(root),
                "--models",
                "deepseek_v32",
                "--seqs",
                "4096",
                "--workloads",
                "operator_chain",
                "--repeats",
                "3",
                "--allow-short",
                "--execute-gpu",
                env=env,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("monolithic GPU execution is inadmissible", completed.stdout)
            self.assertFalse(root.exists())

    def test_validator_rejects_fabricated_cta_impl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tamper"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["experiment_contract"]["rungs"]["cta_impl"].update(
                status="COMPLETE", available=True
            )
            harness.atomic_write_json(manifest_path, manifest)
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("rung_not_partial:cta_impl", result["errors"])
            self.assertIn("cta_impl_must_be_unavailable", result["errors"])

    def test_validator_rejects_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tamper"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["experiment_contract"]["headroom_defined"] = True
            manifest["experiment_contract"]["headroom_pct"] = 12.5
            harness.atomic_write_json(manifest_path, manifest)
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("headroom_must_be_undefined", result["errors"])
            self.assertIn("headroom_value_forbidden", result["errors"])

    def test_validator_rejects_one_million_relabel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            completed = self.make_dry_run(root, formal=True)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for record in manifest["shape_records"]:
                if record["seq"] == 1048576:
                    record["extreme"] = False
                    break
            harness.atomic_write_json(manifest_path, manifest)
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any(error.startswith("shape_formula_mismatch:") for error in result["errors"])
            )

    def test_validator_independently_rejects_causal_pair_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tamper"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            record = manifest["shape_records"][0]
            record["indexer_causal_pairs"] -= 1
            harness.atomic_write_json(manifest_path, manifest)
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(
                "shape_causal_pairs:deepseek_v32:4096", result["errors"]
            )

    def test_validator_rejects_static_api_evidence_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tamper"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["static_api_checks"][0]["tokens"].append("fabricated_api")
            harness.atomic_write_json(manifest_path, manifest)
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("static_api_checks_drift", result["errors"])

    def test_validator_rejects_correctness_contract_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tamper"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["correctness_contract"]["topk_tail"] = "FABRICATED"
            harness.atomic_write_json(manifest_path, manifest)
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("correctness_contract_drift", result["errors"])
            self.assertIn("correctness_topk_tail_contract", result["errors"])

    def test_validator_rejects_long_context_execution_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tamper"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["long_context_execution"]["attention_reference"][
                "row_batch"
            ] = 4
            harness.atomic_write_json(manifest_path, manifest)
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("long_context_execution_contract_drift", result["errors"])
            self.assertIn("long_context_attention_reference_contract", result["errors"])

    def test_validator_rejects_malformed_manifest_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "malformed"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            (root / "manifest.json").write_text("{not-json", encoding="utf-8")
            result = validator.validate(root, "dry-run")
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any(error.startswith("unreadable:manifest.json:") for error in result["errors"]),
                result["errors"],
            )

    def test_no_temporary_files_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dry"
            completed = self.make_dry_run(root)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertFalse(any(path.suffix == ".tmp" for path in root.iterdir()))


class RunnerTests(unittest.TestCase):
    def test_formal_dry_runner_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal-dry"
            env = os.environ.copy()
            env.update(RESULTS=str(root), FAST="0", EXECUTE_GPU="0")
            completed = run("bash", str(RUNNER), env=env)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest = json.loads((root / "manifest.json").read_text())
            validation = json.loads((root / "validation.json").read_text())
            marker = json.loads((root / "production_dry_run.done.json").read_text())
            self.assertEqual(len(manifest["expected_matrix"]), 26)
            self.assertTrue(manifest["formal_statistics_requested"])
            self.assertEqual(
                manifest["publication"]["requested_publish_target"], str(root)
            )
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["accepted_workload_timing"], 0)
            self.assertEqual(marker["accepted_CTA_bracket"], 0)

    def test_runner_publishes_only_validated_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "published"
            env = os.environ.copy()
            env.update(RESULTS=str(root), FAST="1", EXECUTE_GPU="0")
            completed = run("bash", str(RUNNER), env=env)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertTrue((root / "production_dry_run.done.json").is_file())
            validation = json.loads((root / "validation.json").read_text())
            marker = json.loads((root / "production_dry_run.done.json").read_text())
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(marker["status"], "PASS")
            self.assertEqual(marker["accepted_timing"], 0)
            self.assertEqual(marker["accepted_workload_timing"], 0)
            self.assertEqual(marker["accepted_CTA_bracket"], 0)
            self.assertFalse(any(root.parent.glob(root.name + ".inprogress.*")))

    def test_fast_runner_env_overrides_one_long_attention_workload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "long-probe"
            env = os.environ.copy()
            env.update(
                RESULTS=str(root),
                FAST="1",
                EXECUTE_GPU="0",
                MODELS="glm5",
                SEQS="1048576",
                WORKLOADS="indexshare_fsss",
                WARMUP="0",
                REPEATS="1",
                MOE_TOKENS="128",
            )
            completed = run("bash", str(RUNNER), env=env)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual([model["key"] for model in manifest["models"]], ["glm5"])
            self.assertEqual(manifest["seqs"], [1048576])
            self.assertEqual(manifest["workloads"], ["indexshare_fsss"])
            self.assertEqual((manifest["warmup"], manifest["repeats"]), (0, 1))
            self.assertEqual(
                manifest["shape_records"][0]["query_chunk_tokens"], 4096
            )
            self.assertEqual(manifest["shape_records"][0]["num_query_chunks"], 256)
            self.assertEqual(
                [row["row_id"] for row in manifest["expected_matrix"]],
                ["glm5.indexshare_fsss.seq1048576", "glm5.moe32"],
            )

    def test_runner_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "existing"
            root.mkdir()
            env = os.environ.copy()
            env.update(RESULTS=str(root), FAST="1", EXECUTE_GPU="0")
            completed = run("bash", str(RUNNER), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("RESULTS already exists", completed.stdout)

    def test_dry_runner_never_invokes_nvidia_smi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "dry"
            fake = base / "nvidia-smi"
            state = base / "gpu-query-count"
            write_fake_nvidia_smi(fake)
            env = os.environ.copy()
            env.update(
                RESULTS=str(root),
                FAST="1",
                EXECUTE_GPU="0",
                DSA_NVIDIA_SMI=str(fake),
                FAKE_GPU_STATE=str(state),
            )
            completed = run("bash", str(RUNNER), env=env)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertFalse(state.exists(), "CPU dry-run queried the fake GPU")

    def test_foreign_gpu_process_creates_permanent_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "rejected"
            fake = base / "nvidia-smi"
            state = base / "gpu-query-count"
            write_fake_nvidia_smi(fake)
            env = os.environ.copy()
            env.update(
                RESULTS=str(root),
                FAST="1",
                EXECUTE_GPU="1",
                TIER5_PRODUCTION_GPU_ALLOWED="1",
                DSA_NVIDIA_SMI=str(fake),
                DSA_GPU_INDEX="0",
                DSA_MONITOR_INTERVAL_MS="10",
                DSA_NVIDIA_SMI_TIMEOUT_MS="500",
                FAKE_GPU_STATE=str(state),
                FAKE_GPU_MODE="foreign_after_pre",
            )
            completed = run("bash", str(RUNNER), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            rejected = list((root / "failed_segments").glob("*.rejected.*"))
            self.assertEqual(len(rejected), 1)
            rejection = json.loads(
                (rejected[0] / "segment_rejection.json").read_text()
            )
            self.assertEqual(rejection["status"], "REJECTED")
            self.assertEqual(rejection["accepted_workload_timing"], 0)
            self.assertEqual(rejection["accepted_CTA_bracket"], 0)
            query_count = state.read_text()
            replay = run("bash", str(RUNNER), env=env)
            self.assertEqual(replay.returncode, 2, replay.stdout)
            self.assertGreater(int(state.read_text()), int(query_count))
            self.assertEqual(
                len(list((root / "failed_segments").glob("*.rejected.*"))), 2
            )

    def test_uuid_global_lock_blocks_production_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "blocked"
            fake = base / "nvidia-smi"
            state = base / "gpu-query-count"
            write_fake_nvidia_smi(fake)
            env = os.environ.copy()
            env.update(
                RESULTS=str(root),
                FAST="1",
                EXECUTE_GPU="1",
                TIER5_PRODUCTION_GPU_ALLOWED="1",
                DSA_NVIDIA_SMI=str(fake),
                FAKE_GPU_STATE=str(state),
            )
            lock_path = Path(f"/tmp/cta_pdl_gpu_{FAKE_GPU_UUID}.lock")
            with lock_path.open("w") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed = run("bash", str(RUNNER), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("global target-GPU lock", completed.stdout)
            self.assertTrue(root.is_dir())
            self.assertEqual(
                len(list((root / "failed_segments").glob("*.rejected.*"))), 1
            )

    def test_runner_contains_native_gate_and_monitor_contract(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for token in (
            '/tmp/cta_pdl_gpu_${GPU_UUID}.lock',
            'export CUDA_VISIBLE_DEVICES="${RESOLVED_GPU_INDEX}"',
            "flock -n 9",
            "kill -STOP",
            "--ready-file",
            "kill -CONT",
            "--require-allowed-process",
            "--terminate-on-failure",
            "production_tier5_pre",
            "production_tier5_post",
            "formal_rejection.json",
        ):
            self.assertIn(token, source)
        harness_source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("runtime GPU UUID mismatch", harness_source)
        self.assertIn("one numeric CUDA_VISIBLE_DEVICES", harness_source)

    def test_validator_rejects_identity_target_index_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_index = 3
            identity = {
                "schema": 1,
                "kind": "gpu_identity",
                "status": "PASS",
                "errors": [],
                "phase": "production_global_lock_identity",
                "target_gpu": {
                    "index": expected_index - 1,
                    "uuid": FAKE_GPU_UUID,
                    "name": "NVIDIA B200",
                },
            }
            harness.atomic_write_json(root / "gpu_identity.json", identity)
            manifest = {
                "expected_gpu_uuid": FAKE_GPU_UUID,
                "expected_gpu_index": expected_index,
                "environment": {"CUDA_VISIBLE_DEVICES": str(expected_index)},
                "device": {
                    "uuid": FAKE_GPU_UUID,
                    "runtime_ordinal": 0,
                    "runtime_ordinal_zero": True,
                    "cuda_visible_devices_selector": str(expected_index),
                },
            }
            lock_path = f"/tmp/cta_pdl_gpu_{FAKE_GPU_UUID}.lock"
            evidence = {
                "expected_gpu_uuid": FAKE_GPU_UUID,
                "expected_gpu_index": expected_index,
                "global_lock_scope": "target_uuid",
                "global_lock_key_sha256": harness.sha256_bytes(
                    FAKE_GPU_UUID.encode("utf-8")
                ),
                "global_lock_path_sha256": harness.sha256_bytes(
                    lock_path.encode("utf-8")
                ),
                "monitor_interval_ms": 50,
                "query_timeout_ms": 2000,
            }
            errors: list[str] = []
            validator.validate_gpu_exclusivity(root, manifest, evidence, errors)
            self.assertIn("identity_gpu_index_drift", errors)


class MonitorExitResidualTests(unittest.TestCase):
    @staticmethod
    def validate(root: Path, manifest: dict, evidence: dict) -> list[str]:
        errors: list[str] = []
        validator.validate_gpu_exclusivity(root, manifest, evidence, errors)
        return errors

    @staticmethod
    def rewrite_observations(root: Path, records: list[dict]) -> None:
        observations = root / "gpu_observations.ndjson"
        observations.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        monitor_path = root / "gpu_monitor.json"
        monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
        monitor["observations_sha256"] = harness.sha256_file(observations)
        harness.atomic_write_json(monitor_path, monitor)

    def test_validator_accepts_bounded_bound_identity_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evidence = make_gpu_exclusivity_fixture(root)
            self.assertEqual(self.validate(root, manifest, evidence), [])

    def test_validator_rejects_target_partition_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evidence = make_gpu_exclusivity_fixture(root)
            observations = root / "gpu_observations.ndjson"
            records = [json.loads(line) for line in observations.read_text().splitlines()]
            records[2]["allowed_exit_residual_processes"] = []
            self.rewrite_observations(root, records)
            errors = self.validate(root, manifest, evidence)
            self.assertIn("monitor_process_partition:2", errors)

    def test_validator_rejects_unbound_or_over_limit_residual(self) -> None:
        for mutation in ("prior_identity", "classification", "over_limit"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, evidence = make_gpu_exclusivity_fixture(root)
                observations = root / "gpu_observations.ndjson"
                records = [
                    json.loads(line) for line in observations.read_text().splitlines()
                ]
                residual = records[2]["allowed_exit_residual_processes"][0]
                if mutation == "prior_identity":
                    residual["previous_allowed_start_ticks"] = 99999
                elif mutation == "classification":
                    residual["classification"] = "trusted_without_missing_proc"
                else:
                    residual["residual_observation_number"] = (
                        validator.EXIT_RESIDUAL_LIMIT + 1
                    )
                self.rewrite_observations(root, records)
                errors = self.validate(root, manifest, evidence)
                self.assertTrue(
                    any(error.startswith("monitor_unsafe_exit_residual:2:") for error in errors),
                    errors,
                )

    def test_validator_rejects_residual_manifest_count_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evidence = make_gpu_exclusivity_fixture(root)
            monitor_path = root / "gpu_monitor.json"
            monitor = json.loads(monitor_path.read_text())
            monitor["allowed_exit_residual_processes"][0][
                "residual_observation_count"
            ] = 3
            harness.atomic_write_json(monitor_path, monitor)
            errors = self.validate(root, manifest, evidence)
            self.assertIn("monitor_residual_identity_count_drift", errors)

    def test_validator_rejects_retired_pid_reappearance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evidence = make_gpu_exclusivity_fixture(root)
            observations = root / "gpu_observations.ndjson"
            records = [json.loads(line) for line in observations.read_text().splitlines()]
            raw = {
                "pid": 777,
                "gpu_uuid": FAKE_GPU_UUID,
                "name": "reused-worker",
                "used_memory": "64",
            }
            records[4]["observation"]["target_compute_processes"] = [raw]
            records[4]["allowed_target_processes"] = [
                {**raw, "proc_start_ticks": 54321}
            ]
            self.rewrite_observations(root, records)
            errors = self.validate(root, manifest, evidence)
            self.assertIn("monitor_retired_pid_reappeared:4:777", errors)


class FragmentCampaignTests(unittest.TestCase):
    def contract_binding(self) -> tuple[dict, dict]:
        contract = short_campaign_contract()
        identity = {
            "schema": 1,
            "kind": "gpu_identity",
            "status": "PASS",
            "errors": [],
            "target_gpu": {
                "index": 0,
                "uuid": FAKE_GPU_UUID,
                "name": "NVIDIA B200",
            },
        }
        return contract, campaign.bind_campaign_device(contract, identity)

    def launch_fake_process_campaign(
        self, base: Path, monitor_mode: str
    ) -> tuple[subprocess.Popen[str], Path, Path, Path]:
        root = base / "process-lifecycle-campaign"
        fake_smi = base / "nvidia-smi"
        gpu_state = base / "gpu-query-count"
        child_ready = base / "child-ready"
        monitor_pid = base / "monitor-pid"
        write_fake_nvidia_smi(fake_smi)
        shim_dir = write_fake_fragment_process_stack(base)
        env = os.environ.copy()
        env.update(
            RESULTS=str(root),
            FAST="1",
            EXECUTE_GPU="1",
            TIER5_PRODUCTION_GPU_ALLOWED="1",
            TIER5_FRAGMENT_ONLY_ROW=(
                "deepseek_v32.operator_chain.seq4096"
            ),
            DSA_NVIDIA_SMI=str(fake_smi),
            DSA_GPU_INDEX="0",
            DSA_MONITOR_INTERVAL_MS="10",
            DSA_NVIDIA_SMI_TIMEOUT_MS="500",
            FAKE_GPU_STATE=str(gpu_state),
            FAKE_FRAGMENT_MONITOR=str(base / "fake-fragment-monitor"),
            FAKE_FRAGMENT_CHILD=str(base / "fake-fragment-child"),
            FAKE_FRAGMENT_MONITOR_MODE=monitor_mode,
            FAKE_CHILD_READY_FILE=str(child_ready),
            FAKE_MONITOR_PID_FILE=str(monitor_pid),
            REAL_PYTHON=sys.executable,
            PATH=str(shim_dir) + os.pathsep + env["PATH"],
        )
        process = subprocess.Popen(
            ["bash", str(FRAGMENT_RUNNER)],
            cwd=BASE,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return process, child_ready, monitor_pid, root

    def test_formal_contract_freezes_exact_ordered_26_rows_and_seeds(self) -> None:
        controls = short_campaign_contract()["controls"] | {
            "models": list(harness.MODEL_SPECS),
            "seqs": list(harness.FORMAL_SEQS),
            "workloads": list(harness.FORMAL_WORKLOADS),
            "warmup": 5,
            "repeats": 31,
            "allow_short": False,
            "moe_tokens": 4096,
        }
        contract = campaign.build_campaign_contract(controls)
        self.assertTrue(contract["formal"])
        self.assertEqual(
            contract["controls"]["max_logits_mb"],
            harness.FORMAL_MAX_LOGITS_MB,
        )
        self.assertEqual(
            contract["long_context_execution"],
            harness.long_context_execution_contract(),
        )
        self.assertEqual(contract["ordered_matrix"], campaign.FORMAL_MATRIX)
        self.assertEqual(contract["row_count"], 26)
        seeds = [
            harness.canonical_row_seed(row, 20260805)
            for row in contract["ordered_matrix"]
        ]
        self.assertEqual(seeds[0], 20261805)
        self.assertEqual(seeds[9], 20561805)
        self.assertEqual(seeds[12], 21160805)
        self.assertEqual(seeds[13], 21261805)
        self.assertEqual(seeds[25], 22160805)

    def test_subset_seed_uses_full_canonical_ordinals(self) -> None:
        contract = campaign.build_campaign_contract(
            short_campaign_contract()["controls"]
            | {"models": ["glm5"], "seqs": [1048576]}
        )
        row = contract["ordered_matrix"][0]
        self.assertEqual(row["row_id"], "glm5.operator_chain.seq1048576")
        self.assertEqual(
            harness.canonical_row_seed(row, 20260805), 21561805
        )

    def test_timing_schedule_is_component_adjacent_and_alternating(self) -> None:
        components = ("indexer_topk", "sparse_mla", "chain_total")
        schedule = harness.paired_timing_schedule(3, components)
        for pair in range(9):
            left, right = schedule[2 * pair : 2 * pair + 2]
            self.assertEqual(left[2], right[2])
            self.assertEqual(left[1], right[1])
            self.assertNotEqual(left[3], right[3])
            expected = "off_then_on" if left[1] % 2 == 0 else "on_then_off"
            self.assertEqual((left[4], right[4]), (expected, expected))

    def test_gpu_parser_requires_complete_fragment_binding(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                harness.parse_args(
                    [
                        "--output-dir", "/tmp/never-created",
                        "--execute-gpu", "--allow-short", "--repeats", "1",
                        "--models", "deepseek_v32", "--seqs", "4096",
                        "--workloads", "operator_chain",
                    ]
                )

    def test_single_moe_fragment_seals_and_is_always_accept_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            stage, _ = make_fragment_fixture(
                root, contract, binding, 1,
                "11111111-1111-4111-8111-111111111111",
            )
            marker = campaign.seal_fragment(stage, contract, binding)
            self.assertEqual(marker["row_id"], "deepseek_v32.moe32")
            self.assertEqual(marker["accepted_workload_timing"], 0)
            validation, metadata = campaign.check_fragment(
                stage, contract, binding
            )
            self.assertEqual(validation["status"], "PASS", validation["errors"])
            self.assertEqual(metadata["ordinal"], 1)

    def test_single_attention_fragment_does_not_require_other_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            stage, _ = make_fragment_fixture(
                root, contract, binding, 0,
                "22222222-2222-4222-8222-222222222222",
            )
            marker = campaign.seal_fragment(stage, contract, binding)
            self.assertEqual(marker["ordinal"], 0)
            self.assertEqual(marker["accepted_workload_timing"], 0)

    def test_generic_validator_uses_selected_row_and_never_accepts_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            stage, _ = make_fragment_fixture(
                root, contract, binding, 1,
                "12121212-1212-4212-8212-121212121212",
            )
            evidence_errors: list[str] = []
            evidence = campaign._fragment_evidence(
                stage, binding, evidence_errors
            )
            self.assertEqual(evidence_errors, [])
            result = validator.validate(stage, "execute", evidence)
            self.assertEqual(result["status"], "PASS", result["errors"])
            self.assertEqual(result["accepted_workload_timing"], 0)

    def test_fragment_semantic_tamper_is_rejected_before_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            stage, _ = make_fragment_fixture(
                root, contract, binding, 1,
                "33333333-3333-4333-8333-333333333333",
            )
            result_path = stage / "result.json"
            result = json.loads(result_path.read_text())
            result["accepted_workload_timing"] = 1
            harness.atomic_write_json(result_path, result)
            with self.assertRaisesRegex(ValueError, "fragment validation failed"):
                campaign.seal_fragment(stage, contract, binding)
            self.assertFalse((stage / "fragment.done.json").exists())

    def test_fragment_marker_hash_detects_post_seal_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            stage, _ = make_fragment_fixture(
                root, contract, binding, 1,
                "44444444-4444-4444-8444-444444444444",
            )
            campaign.seal_fragment(stage, contract, binding)
            with (stage / "harness.log").open("a") as handle:
                handle.write("tamper\n")
            validation, _ = campaign.check_fragment(stage, contract, binding)
            self.assertEqual(validation["status"], "FAIL")
            self.assertIn(
                "fragment_marker_or_artifact_hash_drift", validation["errors"]
            )

    def test_fragment_exact_artifact_allowlist_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            stage, _ = make_fragment_fixture(
                root, contract, binding, 1,
                "45454545-4545-4545-8545-454545454545",
            )
            (stage / "unbound-profile.sqlite").write_text("extra")
            validation, _ = campaign.validate_fragment(stage, contract, binding)
            self.assertIn(
                "unexpected_fragment_entry:unbound-profile.sqlite",
                validation["errors"],
            )

    def test_monitor_control_and_process_identity_tamper_fail(self) -> None:
        for filename, field, value, error in (
            (
                "gpu_monitor.json", "poll_interval_ms", 51,
                "fragment_monitor_interval_contract_drift",
            ),
            (
                "gpu_monitor.json", "watch_root_start_ticks", 999,
                "monitor_watch_start_runtime_drift",
            ),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                contract, binding = self.contract_binding()
                stage, _ = make_fragment_fixture(
                    root, contract, binding, 1,
                    "55555555-5555-4555-8555-555555555555",
                )
                path = stage / filename
                data = json.loads(path.read_text())
                data[field] = value
                harness.atomic_write_json(path, data)
                validation, _ = campaign.validate_fragment(
                    stage, contract, binding
                )
                self.assertEqual(validation["status"], "FAIL")
                self.assertIn(error, validation["errors"])

    def test_no_clobber_publish_refuses_existing_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "row.inprogress.fixture"
            final = root / "row"
            stage.mkdir()
            final.mkdir()
            with self.assertRaises(FileExistsError):
                campaign.atomic_publish_directory(stage, final)
            self.assertTrue(stage.is_dir())

    def test_formal_acceptance_requires_exact_final_cardinalities(self) -> None:
        controls = short_campaign_contract()["controls"] | {
            "models": list(harness.MODEL_SPECS),
            "seqs": list(harness.FORMAL_SEQS),
            "workloads": list(harness.FORMAL_WORKLOADS),
            "warmup": 5,
            "repeats": 31,
            "allow_short": False,
            "moe_tokens": 4096,
        }
        contract = campaign.build_campaign_contract(controls)
        self.assertTrue(campaign.formal_campaign_eligible(contract, 26, 2542, 122))
        for values in ((25, 2542, 122), (26, 2541, 122), (26, 2542, 121)):
            self.assertFalse(campaign.formal_campaign_eligible(contract, *values))
        short = short_campaign_contract()
        self.assertFalse(campaign.formal_campaign_eligible(short, 26, 2542, 122))

    def test_short_campaign_finalizes_and_fresh_check_stays_accept_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            harness.atomic_write_json(root / "campaign_contract.json", contract)
            harness.atomic_write_json(root / "campaign_binding.json", binding)
            for ordinal, invocation in enumerate(
                (
                    "66666666-6666-4666-8666-666666666666",
                    "77777777-7777-4777-8777-777777777777",
                )
            ):
                stage, final = make_fragment_fixture(
                    root, contract, binding, ordinal, invocation
                )
                campaign.seal_fragment(stage, contract, binding)
                campaign.atomic_publish_directory(stage, final)
            marker = campaign.finalize_campaign(root, contract, binding)
            self.assertEqual(marker["accepted_workload_timing"], 0)
            checked = campaign.check_final_campaign(root, contract, binding)
            self.assertEqual(checked, marker)
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["correctness_row_count"], 2)
            self.assertEqual(result["sample_count"], 7)

    def test_finalizer_rejects_duplicate_invocation_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            harness.atomic_write_json(root / "campaign_contract.json", contract)
            harness.atomic_write_json(root / "campaign_binding.json", binding)
            invocation = "88888888-8888-4888-8888-888888888888"
            for ordinal in range(2):
                stage, final = make_fragment_fixture(
                    root, contract, binding, ordinal, invocation
                )
                campaign.seal_fragment(stage, contract, binding)
                campaign.atomic_publish_directory(stage, final)
            with self.assertRaisesRegex(ValueError, "duplicate fragment invocation"):
                campaign.finalize_campaign(root, contract, binding)

    def test_fresh_final_check_detects_aggregate_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding = self.contract_binding()
            harness.atomic_write_json(root / "campaign_contract.json", contract)
            harness.atomic_write_json(root / "campaign_binding.json", binding)
            for ordinal, invocation in enumerate(
                (
                    "99999999-9999-4999-8999-999999999999",
                    "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                )
            ):
                stage, final = make_fragment_fixture(
                    root, contract, binding, ordinal, invocation
                )
                campaign.seal_fragment(stage, contract, binding)
                campaign.atomic_publish_directory(stage, final)
            campaign.finalize_campaign(root, contract, binding)
            result_path = root / "result.json"
            result = json.loads(result_path.read_text())
            result["accepted_workload_timing"] = 1
            harness.atomic_write_json(result_path, result)
            with self.assertRaises(ValueError):
                campaign.check_final_campaign(root, contract, binding)

    def test_runner_and_profiler_static_separation(self) -> None:
        runner = FRAGMENT_RUNNER.read_text(encoding="utf-8")
        sidecar = NSYS_SIDECAR.read_text(encoding="utf-8")
        self.assertNotIn("nsys profile", runner)
        self.assertIn('${STEP_TIMEOUT}" != "0"', runner)
        self.assertIn("check-final", runner)
        self.assertIn("NSYS_PROFILE_ARGV", sidecar)
        self.assertIn("cuda_gpu_kern_sum", sidecar)
        self.assertIn("nvtx_sum", sidecar)
        self.assertIn("accepted_workload_timing", sidecar)

    def test_fragment_runner_fresh_finalize_is_immediately_revalidated(self) -> None:
        runner = FRAGMENT_RUNNER.read_text(encoding="utf-8")
        finalizer = "production_tier5_campaign.py finalize"
        checker = "production_tier5_campaign.py check-final"
        finalizer_offset = runner.rindex(finalizer)
        completion_offset = runner.rindex(
            'campaign_log "campaign complete rows=${#ROW_SPECS[@]}'
        )
        tail = runner[finalizer_offset:completion_offset]
        self.assertIn(checker, tail)
        self.assertLess(tail.index(finalizer), tail.index(checker))
        self.assertIn("fresh_check_final=PASS", runner[completion_offset:])

    def test_nsys_sidecar_rejects_formal_before_any_gpu_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fake_nsys = base / "nsys"
            fake_nsys.write_text("#!/bin/sh\ntouch \"$NSYS_CALLED\"\n")
            fake_nsys.chmod(0o755)
            state = base / "called"
            env = os.environ.copy()
            env.update(
                FAST="0",
                EXECUTE_GPU="1",
                TIER5_PRODUCTION_GPU_ALLOWED="1",
                NSYS=str(fake_nsys),
                NSYS_CALLED=str(state),
                RESULTS=str(base / "output"),
            )
            completed = run("bash", str(NSYS_SIDECAR), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("requires FAST=1", completed.stdout)
            self.assertFalse(state.exists())

    def test_nsys_sidecar_proof_gate_fails_closed(self) -> None:
        row_id = "deepseek_v32.operator_chain.seq4096"
        cases = {
            "complete": (
                f"cudaLaunchKernel,mqa,tier5_fragment:0:{row_id}:"
                f"{FAKE_FRAGMENT_INVOCATION_UUID}",
                0,
                None,
            ),
            "missing_exact_nvtx": (
                "cudaLaunchKernel,mqa",
                2,
                "exact fragment NVTX range",
            ),
            "missing_target_kernel": (
                f"cudaLaunchKernel,tier5_fragment:0:{row_id}:"
                f"{FAKE_FRAGMENT_INVOCATION_UUID}",
                2,
                "target DSA/GEMM/MLA kernel",
            ),
        }
        for name, (stats, expected_rc, error_text) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                fake_nsys = base / "nsys"
                write_fake_nsys(fake_nsys)
                output = base / "sidecar"
                env = os.environ.copy()
                env.update(
                    FAST="1",
                    EXECUTE_GPU="1",
                    TIER5_PRODUCTION_GPU_ALLOWED="1",
                    NSYS=str(fake_nsys),
                    RESULTS=str(output),
                    FAKE_NSYS_STATS=stats,
                )
                completed = run("bash", str(NSYS_SIDECAR), env=env)
                self.assertEqual(completed.returncode, expected_rc, completed.stdout)
                if expected_rc == 0:
                    self.assertTrue(output.is_dir())
                    proof = json.loads((output / "nsys_sidecar.json").read_text())
                    self.assertEqual(proof["status"], "PASS")
                    self.assertEqual(proof["accepted_workload_timing"], 0)
                    self.assertEqual(proof["accepted_CTA_bracket"], 0)
                else:
                    self.assertFalse(output.exists())
                    self.assertIn(error_text, completed.stdout)
                    self.assertEqual(
                        len(list(base.glob("sidecar.failed.*"))), 1,
                        completed.stdout,
                    )

    def test_nonformal_heredoc_proofs_are_explicitly_fail_closed(self) -> None:
        sidecar = NSYS_SIDECAR.read_text(encoding="utf-8")
        fragment_runner = FRAGMENT_RUNNER.read_text(encoding="utf-8")
        self.assertIn("<<'PY' || exit 2", sidecar)
        self.assertIn("<<'PY' || exit 2", fragment_runner)

    def test_fragment_runner_rejects_step_timeout_before_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fake = base / "nvidia-smi"
            state = base / "gpu-query-count"
            write_fake_nvidia_smi(fake)
            env = os.environ.copy()
            env.update(
                RESULTS=str(base / "campaign"), EXECUTE_GPU="1", FAST="1",
                TIER5_PRODUCTION_GPU_ALLOWED="1", STEP_TIMEOUT="60",
                DSA_NVIDIA_SMI=str(fake), FAKE_GPU_STATE=str(state),
            )
            completed = run("bash", str(FRAGMENT_RUNNER), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("STEP_TIMEOUT is forbidden", completed.stdout)
            self.assertFalse(state.exists())

    def test_fragment_runner_interrupt_reaps_child_group_before_reject(self) -> None:
        for sent_signal, expected_rc in (
            (signal.SIGINT, 130),
            (signal.SIGTERM, 143),
        ):
            with self.subTest(signal=sent_signal), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                process, child_ready, monitor_file, root = (
                    self.launch_fake_process_campaign(base, "follow")
                )
                child_state = wait_for_process_fixture(child_ready, process)
                monitor_state = wait_for_process_fixture(monitor_file, process)
                child_pid, grandchild_pid, child_pgid = map(
                    int, child_state.split()
                )
                monitor_pid = int(monitor_state)
                self.assertEqual(child_pgid, child_pid)
                identities = [
                    (pid, proc_start_ticks(pid))
                    for pid in (child_pid, grandchild_pid, monitor_pid)
                ]
                process.send_signal(sent_signal)
                output = process.communicate(timeout=15)[0]
                self.assertEqual(process.returncode, expected_rc, output)
                assert_process_identities_stop(self, identities)
                rejected = list(
                    (root / "failed_segments").glob("*.rejected.*")
                )
                self.assertEqual(len(rejected), 1, output)
                rejection = json.loads(
                    (rejected[0] / "segment_rejection.json").read_text()
                )
                self.assertEqual(
                    rejection["reason"],
                    "runner_interrupted_or_unhandled_failure_rc_"
                    f"{expected_rc}",
                )

    def test_fragment_runner_monitor_failure_reaps_live_child_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            process, child_ready, monitor_file, root = (
                self.launch_fake_process_campaign(
                    base, "exit_after_child_start"
                )
            )
            child_state = wait_for_process_fixture(child_ready, process)
            monitor_state = wait_for_process_fixture(monitor_file, process)
            child_pid, grandchild_pid, child_pgid = map(
                int, child_state.split()
            )
            monitor_pid = int(monitor_state)
            self.assertEqual(child_pgid, child_pid)
            identities = [
                (pid, proc_start_ticks(pid))
                for pid in (child_pid, grandchild_pid, monitor_pid)
            ]
            output = process.communicate(timeout=15)[0]
            self.assertEqual(process.returncode, 2, output)
            assert_process_identities_stop(self, identities)
            rejected = list((root / "failed_segments").glob("*.rejected.*"))
            self.assertEqual(len(rejected), 1, output)
            rejection = json.loads(
                (rejected[0] / "segment_rejection.json").read_text()
            )
            self.assertEqual(
                rejection["reason"],
                "pre_0_harness_or_monitor_124_post_0",
            )

    def test_campaign_root_lock_blocks_before_gpu_query(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "campaign"
            fake = base / "nvidia-smi"
            state = base / "gpu-query-count"
            write_fake_nvidia_smi(fake)
            digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()
            lock = Path(f"/tmp/cta_pdl_tier5_campaign_{digest}.lock")
            env = os.environ.copy()
            env.update(
                RESULTS=str(root), EXECUTE_GPU="1", FAST="1",
                TIER5_PRODUCTION_GPU_ALLOWED="1", DSA_NVIDIA_SMI=str(fake),
                FAKE_GPU_STATE=str(state),
            )
            with lock.open("w") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed = run("bash", str(FRAGMENT_RUNNER), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertIn("campaign root lock", completed.stdout)
            self.assertFalse(state.exists())

    def test_resume_quarantines_stale_stage_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "campaign"
            stale = root / "rows" / "000_row.inprogress.dead"
            stale.mkdir(parents=True)
            (root / "failed_segments").mkdir()
            (stale / "partial.txt").write_text("partial")
            fake = base / "nvidia-smi"
            state = base / "gpu-query-count"
            write_fake_nvidia_smi(fake)
            env = os.environ.copy()
            env.update(
                RESULTS=str(root), EXECUTE_GPU="1", FAST="1",
                TIER5_PRODUCTION_GPU_ALLOWED="1", DSA_NVIDIA_SMI=str(fake),
                FAKE_GPU_STATE=str(state),
            )
            lock_path = Path(f"/tmp/cta_pdl_gpu_{FAKE_GPU_UUID}.lock")
            with lock_path.open("w") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                completed = run("bash", str(FRAGMENT_RUNNER), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertFalse(stale.exists())
            recovered = list((root / "failed_segments").glob("*.stale.*"))
            self.assertEqual(len(recovered), 1)
            rejection = json.loads(
                (recovered[0] / "segment_rejection.json").read_text()
            )
            self.assertEqual(
                rejection["reason"],
                "stale_unsealed_inprogress_recovered_on_resume",
            )

    def test_corrupt_existing_row_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "campaign"
            row = (
                root / "rows" /
                "000_deepseek_v32.operator_chain.seq4096"
            )
            row.mkdir(parents=True)
            (root / "failed_segments").mkdir()
            sentinel = row / "user-sentinel"
            sentinel.write_text("preserve")
            fake = base / "nvidia-smi"
            state = base / "gpu-query-count"
            write_fake_nvidia_smi(fake)
            env = os.environ.copy()
            env.update(
                RESULTS=str(root), EXECUTE_GPU="1", FAST="1",
                TIER5_PRODUCTION_GPU_ALLOWED="1", DSA_NVIDIA_SMI=str(fake),
                FAKE_GPU_STATE=str(state),
            )
            completed = run("bash", str(FRAGMENT_RUNNER), env=env)
            self.assertEqual(completed.returncode, 2, completed.stdout)
            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertFalse(state.exists())


class StatisticsTests(unittest.TestCase):
    def test_paired_summary_has_ci_but_no_headroom(self) -> None:
        samples = []
        for repeat in range(31):
            for mode, value in (("off", 10.0 + repeat / 100), ("on", 9.0 + repeat / 100)):
                samples.append(
                    {
                        "row_id": "deepseek_v32.operator_chain.seq4096",
                        "component": "chain_total",
                        "pdl_mode": mode,
                        "repeat": repeat,
                        "elapsed_ms": value,
                    }
                )
        summaries = harness.summarize_samples(samples, 7)
        paired = [item for item in summaries if "comparison" in item]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["sample_count"], 31)
        self.assertFalse(paired[0]["formal_tier5_headroom"])
        self.assertNotIn("headroom_pct", paired[0])


if __name__ == "__main__":
    unittest.main()
