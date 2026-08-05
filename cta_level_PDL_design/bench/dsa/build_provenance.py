#!/usr/bin/env python3
"""Create/verify a hash-closed source-to-binary build provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def nvcc_version(executable: Path) -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            [str(executable), "--version"], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    if result.returncode != 0:
        return result.stdout, f"nvcc --version returncode={result.returncode}"
    return result.stdout, None


def compile_argv(
    nvcc: Path, target: str, nvtx_include: str,
) -> list[str]:
    argv = [
        str(nvcc), "-O3", "-std=c++17", f"-arch={target}", "-lineinfo",
        "-I.", "--expt-relaxed-constexpr",
    ]
    if nvtx_include and (Path(nvtx_include) / "nvtx3").is_dir():
        argv.append(f"-I{nvtx_include}")
    argv.extend(["dsa/dsa_native.cu", "-o", "dsa/dsa_native"])
    return argv


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    version, version_error = nvcc_version(args.nvcc)
    errors: list[str] = []
    if version_error:
        errors.append(version_error)
    paths = {
        "dsa_native.cu": args.source,
        "bench_util.cuh": args.bench_util,
        "cta_trace.cuh": args.cta_trace,
        "build.sh": args.build_script,
    }
    inputs: dict[str, str] = {}
    for label, path in paths.items():
        try:
            inputs[label] = sha256(path)
        except OSError as exc:
            errors.append(f"cannot hash {label}: {exc}")
    try:
        binary_sha = sha256(args.binary)
    except OSError as exc:
        binary_sha = None
        errors.append(f"cannot hash binary: {exc}")
    try:
        build_log_sha = sha256(args.build_log)
        build_log_text = args.build_log.read_text(encoding="utf-8", errors="replace")
        if "-- dsa/dsa_native.cu -> dsa/dsa_native" not in build_log_text:
            errors.append("build log lacks the dsa_native compilation ledger")
        if "== build OK" not in build_log_text:
            errors.append("build log lacks successful terminal status")
    except OSError as exc:
        build_log_sha = None
        errors.append(f"cannot bind build log: {exc}")
    return {
        "schema": 1,
        "kind": "dsa_native_build_provenance",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_arch": args.target,
        "build_invocation": [str(args.build_script.resolve())],
        "build_environment": {
            "ARCH": args.target,
            "NVCC": str(args.nvcc),
            "NVTX_INCLUDE": args.nvtx_include,
        },
        "nvcc_path": str(args.nvcc),
        "nvcc_version": version,
        "nvcc_version_sha256": hashlib.sha256(version.encode()).hexdigest(),
        "dsa_compile_argv": compile_argv(args.nvcc, args.target, args.nvtx_include),
        "input_sha256": inputs,
        "binary_path": str(args.binary.resolve()),
        "binary_sha256": binary_sha,
        "build_log_path": str(args.build_log.resolve()),
        "build_log_sha256": build_log_sha,
    }


def validate_manifest(
    manifest_path: Path, binary: Path, source: Path, bench_util: Path,
    cta_trace: Path, build_script: Path, build_log: Path, target: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level JSON is not an object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"cannot read build manifest: {exc}"]
    if (
        value.get("schema") != 1
        or value.get("kind") != "dsa_native_build_provenance"
        or value.get("status") != "PASS"
        or value.get("errors") != []
        or value.get("target_arch") != target
    ):
        errors.append("build manifest identity/status/target mismatch")
    expected_inputs = {
        "dsa_native.cu": source,
        "bench_util.cuh": bench_util,
        "cta_trace.cuh": cta_trace,
        "build.sh": build_script,
    }
    for label, path in expected_inputs.items():
        try:
            actual = sha256(path)
        except OSError as exc:
            errors.append(f"cannot hash current {label}: {exc}")
            continue
        if value.get("input_sha256", {}).get(label) != actual:
            errors.append(f"build manifest current-input mismatch: {label}")
    for label, path, field in (
        ("binary", binary, "binary_sha256"),
        ("build log", build_log, "build_log_sha256"),
    ):
        try:
            actual = sha256(path)
        except OSError as exc:
            errors.append(f"cannot hash current {label}: {exc}")
            continue
        if value.get(field) != actual:
            errors.append(f"build manifest current-{label} mismatch")
    if Path(str(value.get("binary_path", ""))) != binary.resolve():
        errors.append("build manifest binary path mismatch")
    if Path(str(value.get("build_log_path", ""))) != build_log.resolve():
        errors.append("build manifest build-log path mismatch")
    nvcc_path = Path(str(value.get("nvcc_path", "")))
    nvtx_include = str(value.get("build_environment", {}).get("NVTX_INCLUDE", ""))
    if value.get("build_environment", {}).get("ARCH") != target:
        errors.append("build manifest ARCH environment mismatch")
    if value.get("build_environment", {}).get("NVCC") != str(nvcc_path):
        errors.append("build manifest NVCC environment mismatch")
    if value.get("build_invocation") != [str(build_script.resolve())]:
        errors.append("build manifest build invocation mismatch")
    if value.get("dsa_compile_argv") != compile_argv(nvcc_path, target, nvtx_include):
        errors.append("build manifest dsa nvcc argv mismatch")
    version = value.get("nvcc_version")
    if not isinstance(version, str) or not version.strip():
        errors.append("build manifest nvcc version missing")
    elif value.get("nvcc_version_sha256") != hashlib.sha256(version.encode()).hexdigest():
        errors.append("build manifest nvcc version hash mismatch")
    return value, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("create", "verify"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--json", type=Path, required=True)
        sub.add_argument("--binary", type=Path, required=True)
        sub.add_argument("--source", type=Path, required=True)
        sub.add_argument("--bench-util", type=Path, required=True)
        sub.add_argument("--cta-trace", type=Path, required=True)
        sub.add_argument("--build-script", type=Path, required=True)
        sub.add_argument("--build-log", type=Path, required=True)
        sub.add_argument("--target", required=True)
        if action == "create":
            sub.add_argument("--nvcc", type=Path, required=True)
            sub.add_argument("--nvtx-include", default="")
    args = parser.parse_args()
    if args.action == "create":
        payload = create_manifest(args)
        atomic_json(args.json, payload)
        errors = payload["errors"]
    else:
        _, errors = validate_manifest(
            args.json, args.binary, args.source, args.bench_util,
            args.cta_trace, args.build_script, args.build_log, args.target,
        )
    print(
        f"DSA_BUILD_PROVENANCE action={args.action} "
        f"status={'PASS' if not errors else 'FAIL'} errors={len(errors)}"
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
