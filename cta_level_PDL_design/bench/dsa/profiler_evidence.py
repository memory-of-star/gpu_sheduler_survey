#!/usr/bin/env python3
"""Write structured, non-blocking profiler admission evidence for Tier 5."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


PERMISSION_MARKERS = (
    "err_nvgpuctrperm",
    "does not have permission to access nvidia gpu performance counters",
    "permission to access nvidia gpu performance counters",
    "permission denied",
    "profiling permission",
)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def classify_permission(
    stdout: str, stderr: str, returncode: int, rm_profiling_admin_only: int | None
) -> dict[str, object]:
    """Classify the dedicated counter-permission probe without assuming stderr-only tools."""
    stdout_lower = stdout.lower()
    stderr_lower = stderr.lower()
    stdout_markers = [marker for marker in PERMISSION_MARKERS if marker in stdout_lower]
    stderr_markers = [marker for marker in PERMISSION_MARKERS if marker in stderr_lower]
    explicit_denial = bool(stdout_markers or stderr_markers)
    admin_policy_fallback = (
        rm_profiling_admin_only == 1 and returncode != 0 and not explicit_denial
    )
    permission_denied = explicit_denial or admin_policy_fallback
    sources: list[str] = []
    if stdout_markers:
        sources.append("stdout")
    if stderr_markers:
        sources.append("stderr")
    if rm_profiling_admin_only == 1:
        sources.append("rm_profiling_admin_only")
    return {
        "permission_denied": permission_denied,
        "explicit_permission_denial": explicit_denial,
        "admin_policy_fallback": admin_policy_fallback,
        "permission_evidence_sources": sources,
        "stdout_permission_markers": stdout_markers,
        "stderr_permission_markers": stderr_markers,
        "profiling_restricted_to_admin": rm_profiling_admin_only == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--returncode", type=int, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    stderr = read_text(args.stderr)
    stdout = read_text(args.stdout)
    params = read_text(Path("/proc/driver/nvidia/params"))
    status = read_text(Path("/proc/self/status"))
    admin = None
    match = re.search(r"^RmProfilingAdminOnly:\s*(\d+)", params, re.MULTILINE)
    if match:
        admin = int(match.group(1))
    cap_bnd = None
    match = re.search(r"^CapBnd:\s*([0-9a-fA-F]+)", status, re.MULTILINE)
    if match:
        cap_bnd = match.group(1)
    try:
        version = subprocess.run(
            [args.tool, "--version"], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.strip()
    except OSError as exc:
        version = str(exc)
    permission = classify_permission(stdout, stderr, args.returncode, admin)
    permission_denied = bool(permission["permission_denied"])
    payload = {
        "schema": 1,
        "purpose": "hardware_counter_permission_probe",
        "measurement_admission": "not_a_timing_sample",
        "launch_count": 1,
        "tool": args.tool,
        "version": version,
        "command": args.command,
        "returncode": args.returncode,
        **permission,
        "hardware_counters_available": args.returncode == 0 and not permission_denied,
        "rm_profiling_admin_only": admin,
        "cap_bnd": cap_bnd,
        "stdout_path": str(args.stdout),
        "stderr_path": str(args.stderr),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "timing_admission_affected": False,
        "fallback_evidence": ["globaltimer_trace", "nsys_cuda_trace"],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.json.unlink()
    except FileNotFoundError:
        pass
    fd, temp = tempfile.mkstemp(prefix=args.json.name + ".", dir=args.json.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, args.json)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
    print(
        "PROFILER_EVIDENCE "
        f"schema=1 tool={args.tool} returncode={args.returncode} "
        f"permission_denied={int(permission_denied)} "
        f"hardware_counters_available={int(payload['hardware_counters_available'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
