#!/usr/bin/env python3
"""Create and verify fail-closed GPU compute-process exclusivity evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


MIN_MONITOR_INTERVAL_MS = 10
MAX_MONITOR_INTERVAL_MS = 100
MIN_QUERY_TIMEOUT_MS = 100
MAX_QUERY_TIMEOUT_MS = 5000
MAX_EXIT_RESIDUAL_OBSERVATIONS_PER_PID = 4
GPU_UUID_RE = re.compile(
    r"^GPU-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


def canonical_gpu_uuid(value: str) -> str | None:
    match = GPU_UUID_RE.fullmatch(value.strip())
    return "GPU-" + match.group(1).lower() if match else None


def run_query(
    executable: str, query: str, timeout_ms: int
) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            [executable, query, "--format=csv,noheader,nounits"],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_ms / 1000.0,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        detail = f"query timed out after {timeout_ms}ms"
        if stderr.strip():
            detail += f": {stderr.strip()}"
        return 124, stdout, detail
    except OSError as exc:
        return 127, "", str(exc)


def csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in csv.reader(text.splitlines()):
        cleaned = [field.strip() for field in row]
        if cleaned and any(cleaned):
            rows.append(cleaned)
    return rows


def query_identity(
    executable: str, gpu_index: int, query_timeout_ms: int
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    gpu_rc, gpu_stdout, gpu_stderr = run_query(
        executable, "--query-gpu=index,uuid,name", query_timeout_ms
    )
    gpu_rows = csv_rows(gpu_stdout)
    target: dict[str, Any] | None = None
    if gpu_rc != 0:
        errors.append(f"GPU identity query returncode={gpu_rc}: {gpu_stderr.strip()}")
    for row in gpu_rows:
        if len(row) != 3:
            errors.append(f"malformed GPU identity row: {row!r}")
            continue
        try:
            index = int(row[0], 10)
        except ValueError:
            errors.append(f"malformed GPU index: {row[0]!r}")
            continue
        uuid = canonical_gpu_uuid(row[1])
        if uuid is None:
            errors.append(f"malformed/noncanonicalizable GPU UUID: {row[1]!r}")
            continue
        if index == gpu_index:
            target = {"index": index, "uuid": uuid, "name": row[2]}
    if target is None:
        errors.append(f"target GPU index {gpu_index} not found")
    return target, errors


def observe(
    executable: str, gpu_index: int, query_timeout_ms: int,
    process_query_override: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    target, errors = query_identity(executable, gpu_index, query_timeout_ms)

    # Driver releases have exposed this column under both names.  Prefer the audit's
    # requested `name`, then use the documented compatibility spelling if necessary.
    process_query = process_query_override or "--query-compute-apps=pid,gpu_uuid,name,used_memory"
    proc_rc, proc_stdout, proc_stderr = run_query(
        executable, process_query, query_timeout_ms
    )
    process_name_field = "process_name" if "process_name" in process_query else "name"
    compatibility_error = proc_stderr.lower()
    may_retry_compatibility_name = (
        proc_rc != 0
        and process_query_override is None
        and "name" in compatibility_error
        and any(fragment in compatibility_error for fragment in (
            "not a valid field", "unknown field", "unsupported field",
            "invalid field",
        ))
    )
    if may_retry_compatibility_name:
        process_query = "--query-compute-apps=pid,gpu_uuid,process_name,used_memory"
        proc_rc, proc_stdout, proc_stderr = run_query(
            executable, process_query, query_timeout_ms
        )
        process_name_field = "process_name"
    if proc_rc != 0:
        errors.append(
            f"compute-process query returncode={proc_rc}: {proc_stderr.strip()}"
        )

    all_processes: list[dict[str, Any]] = []
    for row in csv_rows(proc_stdout):
        if len(row) == 1 and "no running processes" in row[0].lower():
            continue
        if len(row) != 4:
            errors.append(f"malformed compute-process row: {row!r}")
            continue
        try:
            pid = int(row[0], 10)
        except ValueError:
            errors.append(f"malformed compute PID: {row[0]!r}")
            continue
        if pid <= 0:
            errors.append(f"non-positive compute PID: {pid}")
            continue
        process_uuid = canonical_gpu_uuid(row[1])
        if process_uuid is None:
            errors.append(f"malformed compute-process GPU UUID: {row[1]!r}")
            continue
        all_processes.append({
            "pid": pid,
            "gpu_uuid": process_uuid,
            "name": row[2],
            "used_memory": row[3],
        })
    target_uuid = target.get("uuid") if target else None
    target_processes = [
        process for process in all_processes
        if process.get("gpu_uuid") == target_uuid
    ]
    observation = {
        "target_gpu": target,
        "query_gpu": "--query-gpu=index,uuid,name",
        "query_compute_apps": process_query,
        "process_name_field": process_name_field,
        "all_compute_processes": all_processes,
        "target_compute_processes": target_processes,
    }
    return observation, errors


def load_lease(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"cannot read lease {path}: {exc}"]
    errors: list[str] = []
    if value.get("schema") != 1 or value.get("kind") != "gpu_exclusivity_lease":
        errors.append("lease schema/kind mismatch")
    if value.get("status") != "PASS" or value.get("errors") != []:
        errors.append("lease is not a clean PASS")
    if not value.get("lease_id"):
        errors.append("lease_id missing")
    target = value.get("observation", {}).get("target_gpu")
    if not isinstance(target, dict) or not target.get("uuid"):
        errors.append("lease target GPU identity missing")
    if value.get("observation", {}).get("target_compute_processes") != []:
        errors.append("lease acquisition observed target compute processes")
    return value, errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ProcStatus = Literal["present", "missing", "unreadable"]
ProcInfo = tuple[int, int, str]


def proc_info(pid: int) -> tuple[ProcInfo | None, ProcStatus, str | None]:
    """Read ``/proc/PID/stat`` without conflating absence with read failure.

    ``missing`` is returned only for ``FileNotFoundError``.  Permission failures,
    other I/O errors, and malformed stat contents are explicitly ``unreadable`` so
    callers can fail closed rather than treating them as proof of process exit.
    """
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        text = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "missing", None
    except OSError as exc:
        return None, "unreadable", f"cannot read {stat_path}: {exc}"
    try:
        suffix = text[text.rfind(")") + 2 :].split()
        if text.rfind(")") < 0 or len(suffix) <= 19:
            raise ValueError("missing comm terminator or required stat fields")
        return (int(suffix[1]), int(suffix[19]), suffix[0]), "present", None
    except (ValueError, IndexError) as exc:
        return None, "unreadable", f"malformed {stat_path}: {exc}"


def is_descendant(
    pid: int, start_ticks: int, root_pid: int, root_start_ticks: int
) -> bool:
    current = pid
    expected_start = start_ticks
    seen: set[int] = set()
    for _ in range(256):
        if current in seen or current <= 1:
            return False
        seen.add(current)
        info, status, _ = proc_info(current)
        if status != "present" or info is None:
            return False
        parent, actual_start, _ = info
        if actual_start != expected_start:
            return False
        if current == root_pid:
            return actual_start == root_start_ticks
        current = parent
        parent_info, parent_status, _ = proc_info(current)
        if parent_status != "present" or parent_info is None:
            return False
        expected_start = parent_info[1]
    return False


def terminate_process_group(pgid: int) -> str | None:
    try:
        os.killpg(pgid, signal.SIGTERM)
        # A gated command may still be job-control stopped.  SIGTERM remains pending until
        # the group is continued, so always release it after requesting termination.
        os.killpg(pgid, signal.SIGCONT)
        return None
    except ProcessLookupError:
        return None
    except OSError as exc:
        return str(exc)


def is_allowed_exit_residual(
    process: dict[str, Any], known_allowed: dict[int, int],
    current_proc_status: ProcStatus,
    retired_allowed: dict[int, int],
    residual_counts: dict[int, int],
) -> bool:
    """Recognize only an already-proven child's post-exit nvidia-smi tombstone.

    NVIDIA may retain a compute-app row briefly after /proc has removed the process,
    reporting the exact prior PID with name ``[No data]``.  This is safe to classify as
    drain evidence only after that PID/start-time identity was observed as a descendant.
    A live/reused PID, an unknown PID, or any other name remains foreign.  The
    bounded sample allowance prevents a persistent/unverifiable driver row from
    being trusted indefinitely.
    """
    pid = process.get("pid")
    if (
        not isinstance(pid, int)
        or current_proc_status != "missing"
        or process.get("name") != "[No data]"
        or pid not in known_allowed
        or residual_counts.get(pid, 0) >= MAX_EXIT_RESIDUAL_OBSERVATIONS_PER_PID
    ):
        return False
    prior_start = known_allowed[pid]
    return retired_allowed.get(pid, prior_start) == prior_start


def run_monitor(args: argparse.Namespace) -> int:
    lease, errors = load_lease(args.lease)
    lease_id = lease.get("lease_id") if lease else None
    lease_target = lease.get("observation", {}).get("target_gpu") if lease else None
    # The runner forks the gate before this Python process.  Briefly wait for the child to
    # execute its immediate SIGSTOP so scheduler ordering cannot cause a false rejection.
    root: tuple[int, int, str] | None = None
    candidate_start: int | None = None
    gate_deadline = time.monotonic() + 2.0
    while time.monotonic() < gate_deadline:
        current, current_status, current_error = proc_info(args.watch_pid)
        if current_status == "unreadable":
            errors.append(current_error or "watch PID /proc stat is unreadable")
            break
        if current_status == "present" and current is not None:
            if candidate_start is None:
                candidate_start = current[1]
            elif current[1] != candidate_start:
                errors.append("watch PID was reused while entering the stopped start gate")
                break
            if current[2] == "T":
                root = current
                break
            if current[2] == "Z":
                break
        time.sleep(0.001)
    if root is None:
        errors.append(
            f"watch PID {args.watch_pid} did not enter the required stopped start gate"
        )
        root_start = candidate_start if candidate_start is not None else -1
    else:
        root_start = root[1]

    args.observations.parent.mkdir(parents=True, exist_ok=True)
    known_allowed: dict[int, int] = {}
    retired_allowed: dict[int, int] = {}
    exit_residual_counts: dict[int, int] = {}
    allowed_observation_count = 0
    allowed_exit_residual_observation_count = 0
    observation_count = 0
    empty_after_exit = 0
    selected_process_query: str | None = None
    monitor_start = datetime.now(timezone.utc).isoformat()
    terminated_on_failure = False
    start_barrier_ready = False
    command_release_observed = False
    ready_record_written = False
    with args.observations.open("w", encoding="utf-8") as handle:
        while not errors:
            query_started_ns = time.monotonic_ns()
            observation, query_errors = observe(
                args.nvidia_smi, args.gpu_index, args.query_timeout_ms,
                selected_process_query,
            )
            query_finished_ns = time.monotonic_ns()
            selected_process_query = observation.get("query_compute_apps")
            if observation.get("target_gpu") != lease_target:
                query_errors.append("target GPU identity changed from lease acquisition")
            allowed: list[dict[str, Any]] = []
            allowed_exit_residuals: list[dict[str, Any]] = []
            foreign: list[dict[str, Any]] = []
            for process in observation.get("target_compute_processes", []):
                pid = process.get("pid")
                if isinstance(pid, int):
                    info, proc_status, proc_error = proc_info(pid)
                else:
                    info, proc_status, proc_error = None, "unreadable", (
                        "compute process PID is not an integer"
                    )
                if proc_status == "unreadable":
                    query_errors.append(
                        proc_error or f"cannot establish /proc identity for PID {pid!r}"
                    )
                start_ticks = info[1] if info else None
                already_allowed = (
                    isinstance(pid, int)
                    and start_ticks is not None
                    and known_allowed.get(pid) == start_ticks
                )
                descendant = (
                    isinstance(pid, int)
                    and start_ticks is not None
                    and pid not in retired_allowed
                    and root_start >= 0
                    and is_descendant(
                        pid, start_ticks, args.watch_pid, root_start
                    )
                )
                enriched = {**process, "proc_start_ticks": start_ticks}
                exit_residual = is_allowed_exit_residual(
                    process, known_allowed, proc_status, retired_allowed,
                    exit_residual_counts,
                )
                if exit_residual:
                    prior_start = known_allowed[pid]
                    retired_allowed[pid] = prior_start
                    exit_residual_counts[pid] = exit_residual_counts.get(pid, 0) + 1
                    allowed_exit_residuals.append({
                        **enriched,
                        "previous_allowed_start_ticks": prior_start,
                        "classification": "allowed_post_exit_nvidia_smi_residual",
                        "residual_observation_number": exit_residual_counts[pid],
                        "residual_observation_limit": (
                            MAX_EXIT_RESIDUAL_OBSERVATIONS_PER_PID
                        ),
                    })
                elif isinstance(pid, int) and pid in retired_allowed:
                    # Once /proc disappearance proves retirement, a live row with the
                    # same numeric PID is a reuse/new process and cannot inherit trust.
                    # Likewise, a persistent post-exit driver row fails closed after its
                    # small, explicit sample budget is exhausted.
                    classification = (
                        "post_exit_residual_observation_limit_exceeded"
                        if (
                            proc_status == "missing"
                            and process.get("name") == "[No data]"
                            and exit_residual_counts.get(pid, 0)
                            >= MAX_EXIT_RESIDUAL_OBSERVATIONS_PER_PID
                        )
                        else "retired_pid_reappeared_or_reused"
                    )
                    foreign.append({**enriched, "classification": classification})
                elif already_allowed or descendant:
                    known_allowed[pid] = start_ticks
                    allowed.append(enriched)
                else:
                    foreign.append(enriched)
            if allowed:
                allowed_observation_count += 1
            if allowed_exit_residuals:
                allowed_exit_residual_observation_count += 1
            record = {
                "schema": 1,
                "sequence": observation_count,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "query_started_monotonic_ns": query_started_ns,
                "query_finished_monotonic_ns": query_finished_ns,
                "query_duration_ms": (query_finished_ns - query_started_ns) / 1_000_000.0,
                "query_errors": query_errors,
                "observation": observation,
                "allowed_target_processes": allowed,
                "allowed_exit_residual_processes": allowed_exit_residuals,
                "foreign_target_processes": foreign,
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            observation_count += 1
            if query_errors:
                errors.extend(f"observation {observation_count - 1}: {error}" for error in query_errors)
            if foreign:
                errors.append(
                    f"observation {observation_count - 1}: foreign target-GPU processes {foreign!r}"
                )

            # The watched command starts SIGSTOP'ed.  The first clean, empty snapshot is
            # durable before the runner receives READY and sends SIGCONT, closing the old
            # child-start/monitor-start race.  A missed very short timing command then fails
            # closed through --require-allowed-process rather than being admitted unseen.
            if observation_count == 1:
                if observation.get("target_compute_processes"):
                    errors.append("start-gate baseline was not target-GPU idle")
                ready_payload = {
                    "schema": 1,
                    "kind": "gpu_exclusivity_monitor_ready",
                    "status": "READY" if not errors else "FAIL",
                    "errors": list(errors),
                    "phase": args.phase,
                    "watch_pid": args.watch_pid,
                    "watch_root_start_ticks": root_start,
                    "baseline_observation_sequence": 0,
                    "baseline_observed_at": record["observed_at"],
                }
                atomic_json(args.ready_file, ready_payload)
                ready_record_written = True
                if not errors:
                    start_barrier_ready = True
            if errors:
                if args.terminate_on_failure:
                    termination_error = terminate_process_group(args.watch_pid)
                    terminated_on_failure = True
                    if termination_error:
                        errors.append(f"cannot terminate watched process group: {termination_error}")
                break

            if observation_count == 1:
                release_deadline = time.monotonic() + 10.0
                while time.monotonic() < release_deadline:
                    root_now, root_status, root_error = proc_info(args.watch_pid)
                    if root_status == "unreadable":
                        errors.append(root_error or "watch PID /proc stat is unreadable")
                        break
                    if (
                        root_status == "missing"
                        or root_now is None
                        or root_now[1] != root_start
                        or root_now[2] != "T"
                    ):
                        command_release_observed = True
                        break
                    time.sleep(0.001)
                if errors:
                    if args.terminate_on_failure:
                        termination_error = terminate_process_group(args.watch_pid)
                        terminated_on_failure = True
                        if termination_error:
                            errors.append(
                                f"cannot terminate watched process group: {termination_error}"
                            )
                    break
                if not command_release_observed:
                    errors.append("runner did not release the stopped command start gate")
                    if args.terminate_on_failure:
                        termination_error = terminate_process_group(args.watch_pid)
                        terminated_on_failure = True
                        if termination_error:
                            errors.append(
                                f"cannot terminate watched process group: {termination_error}"
                            )
                    break
                # Sample immediately after SIGCONT rather than sleeping for one interval.
                continue

            root_now, root_status, root_error = proc_info(args.watch_pid)
            if root_status == "unreadable":
                errors.append(root_error or "watch PID /proc stat is unreadable")
                if args.terminate_on_failure:
                    termination_error = terminate_process_group(args.watch_pid)
                    terminated_on_failure = True
                    if termination_error:
                        errors.append(
                            f"cannot terminate watched process group: {termination_error}"
                        )
                break
            root_alive = (
                root_status == "present"
                and root_now is not None
                and root_now[1] == root_start
                and root_now[2] != "Z"
            )
            if not root_alive and not observation.get("target_compute_processes"):
                empty_after_exit += 1
            else:
                empty_after_exit = 0
            if empty_after_exit >= 2:
                break
            time.sleep(args.interval_ms / 1000.0)

    if args.require_allowed_process and not known_allowed:
        errors.append("no allowed target-GPU process was observed during watched command")
    observations_sha = sha256_file(args.observations)
    payload = {
        "schema": 1,
        "kind": "gpu_exclusivity_monitor",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "measurement_role": "gpu_exclusivity_monitor_only",
        "accepted_timing": 0,
        "phase": args.phase,
        "started_at": monitor_start,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "lease_path": str(args.lease),
        "lease_id": lease_id,
        "target_gpu": lease_target,
        "watch_pid": args.watch_pid,
        "watch_root_start_ticks": root_start,
        "poll_interval_ms": args.interval_ms,
        "query_timeout_ms": args.query_timeout_ms,
        "coverage_model": "bounded_interval_nvidia_smi_process_sampling",
        "coverage_limit": (
            "foreign GPU processes wholly between completed samples may not be observed"
        ),
        "start_barrier_complete": bool(
            start_barrier_ready and command_release_observed
        ),
        "ready_record_written": ready_record_written,
        "baseline_observation_sequence": 0 if ready_record_written else None,
        "observation_count": observation_count,
        "allowed_observation_count": allowed_observation_count,
        "allowed_exit_residual_observation_count": (
            allowed_exit_residual_observation_count
        ),
        "allowed_processes": [
            {"pid": pid, "proc_start_ticks": start}
            for pid, start in sorted(known_allowed.items())
        ],
        "allowed_exit_residual_processes": [
            {
                "pid": pid,
                "previous_allowed_start_ticks": start,
                "residual_observation_count": exit_residual_counts.get(pid, 0),
            }
            for pid, start in sorted(retired_allowed.items())
        ],
        "allowed_exit_residual_max_observations_per_pid": (
            MAX_EXIT_RESIDUAL_OBSERVATIONS_PER_PID
        ),
        "max_allowed_exit_residual_observations_observed": max(
            exit_residual_counts.values(), default=0
        ),
        "exit_residual_policy": (
            "previously_observed_allowed_same_pid_start_ticks_and_exact_No_data_"
            "with_proc_missing_only_bounded_per_pid"
        ),
        "require_allowed_process": bool(args.require_allowed_process),
        "foreign_processes_detected": any(
            "foreign target-GPU" in error for error in errors
        ),
        "query_failure_detected": any(
            error.startswith("observation ") and "foreign target-GPU" not in error
            for error in errors
        ),
        "terminated_on_failure": terminated_on_failure,
        "observations_path": str(args.observations),
        "observations_sha256": observations_sha,
    }
    atomic_json(args.json, payload)
    print(
        "GPU_EXCLUSIVITY_MONITOR "
        f"phase={args.phase} status={payload['status']} "
        f"observations={observation_count} allowed_pids={len(known_allowed)} "
        f"errors={len(errors)}"
    )
    return 0 if not errors else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("identity", "acquire", "check"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--json", type=Path, required=True)
        sub.add_argument("--nvidia-smi", default=os.environ.get("DSA_NVIDIA_SMI", "nvidia-smi"))
        sub.add_argument("--gpu-index", type=int, default=0)
        sub.add_argument("--query-timeout-ms", type=int, default=2000)
        sub.add_argument("--owner-pid", type=int, default=os.getppid())
        sub.add_argument("--phase", default=action)
        if action == "check":
            sub.add_argument("--lease", type=Path, required=True)
    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--lease", type=Path, required=True)
    monitor.add_argument("--json", type=Path, required=True)
    monitor.add_argument("--observations", type=Path, required=True)
    monitor.add_argument("--ready-file", type=Path, required=True)
    monitor.add_argument("--nvidia-smi", default=os.environ.get("DSA_NVIDIA_SMI", "nvidia-smi"))
    monitor.add_argument("--gpu-index", type=int, default=0)
    monitor.add_argument("--watch-pid", type=int, required=True)
    monitor.add_argument("--phase", required=True)
    monitor.add_argument("--interval-ms", type=int, default=50)
    monitor.add_argument("--query-timeout-ms", type=int, default=2000)
    monitor.add_argument("--require-allowed-process", action="store_true")
    monitor.add_argument("--terminate-on-failure", action="store_true")
    args = parser.parse_args()

    if not MIN_QUERY_TIMEOUT_MS <= args.query_timeout_ms <= MAX_QUERY_TIMEOUT_MS:
        parser.error(
            f"--query-timeout-ms must be {MIN_QUERY_TIMEOUT_MS}.."
            f"{MAX_QUERY_TIMEOUT_MS}"
        )

    if args.action == "monitor":
        if not MIN_MONITOR_INTERVAL_MS <= args.interval_ms <= MAX_MONITOR_INTERVAL_MS:
            parser.error(
                f"--interval-ms must be {MIN_MONITOR_INTERVAL_MS}.."
                f"{MAX_MONITOR_INTERVAL_MS}"
            )
        return run_monitor(args)

    now = datetime.now(timezone.utc).isoformat()
    if args.action == "identity":
        target, errors = query_identity(
            args.nvidia_smi, args.gpu_index, args.query_timeout_ms
        )
        payload = {
            "schema": 1,
            "kind": "gpu_identity",
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "measurement_role": "gpu_identity_only",
            "accepted_timing": 0,
            "phase": args.phase,
            "observed_at": now,
            "owner_pid": args.owner_pid,
            "nvidia_smi": args.nvidia_smi,
            "gpu_index": args.gpu_index,
            "target_gpu": target,
        }
        atomic_json(args.json, payload)
        print(
            "GPU_IDENTITY "
            f"phase={args.phase} status={payload['status']} "
            f"uuid={(target or {}).get('uuid', 'missing')}"
        )
        return 0 if not errors else 2

    observation, errors = observe(
        args.nvidia_smi, args.gpu_index, args.query_timeout_ms
    )
    lease: dict[str, Any] | None = None
    if args.action == "check":
        lease, lease_errors = load_lease(args.lease)
        errors.extend(lease_errors)
        if lease is not None:
            old_target = lease.get("observation", {}).get("target_gpu")
            if old_target != observation.get("target_gpu"):
                errors.append("target GPU identity changed from lease acquisition")
    foreign = observation.get("target_compute_processes", [])
    if foreign:
        errors.append(f"foreign target-GPU compute processes present: {foreign!r}")

    observation_sha = canonical_sha(observation)
    if args.action == "acquire":
        lease_material = {
            "owner_pid": args.owner_pid,
            "observed_at": now,
            "observation_sha256": observation_sha,
            "nonce": secrets.token_hex(16),
        }
        lease_id = canonical_sha(lease_material)
        kind = "gpu_exclusivity_lease"
    else:
        lease_id = lease.get("lease_id") if lease else None
        kind = "gpu_exclusivity_checkpoint"

    payload = {
        "schema": 1,
        "kind": kind,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "measurement_role": "gpu_exclusivity_only",
        "accepted_timing": 0,
        "phase": args.phase,
        "observed_at": now,
        "owner_pid": args.owner_pid,
        "nvidia_smi": args.nvidia_smi,
        "gpu_index": args.gpu_index,
        "observation": observation,
        "observation_sha256": observation_sha,
        "lease_id": lease_id,
    }
    if args.action == "check":
        payload["lease_path"] = str(args.lease)
    atomic_json(args.json, payload)
    print(
        "GPU_EXCLUSIVITY "
        f"action={args.action} phase={args.phase} status={payload['status']} "
        f"foreign_processes={len(foreign)} lease_id={lease_id or 'missing'}"
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
