#!/usr/bin/env python3
"""Validate that all nine Tier-5 NVTX ranges map to their launched GPU kernels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


RANGE_KERNELS = {
    "dsa.poison": ("poisonDsaBuffers", "prepareExpectedRowSums"),
    "dsa.floor": ("dsaIndexer", "dsaTopk", "dsaAttention"),
    "dsa.wave_floor": ("dsaIndexer", "dsaTopk", "dsaAttention"),
    "dsa.impl": ("dsaIndexer", "dsaTopk", "dsaAttention"),
    "dsa.ceiling": ("dsaIndexer", "dsaTopk", "dsaAttention"),
    "dsa.validate.floor": (
        "validateScores", "validateIndices", "validateOutput", "validateRowsAndFlags",
    ),
    "dsa.validate.wave_floor": (
        "validateScores", "validateIndices", "validateOutput", "validateRowsAndFlags",
    ),
    "dsa.validate.impl": (
        "validateScores", "validateIndices", "validateOutput", "validateRowsAndFlags",
    ),
    "dsa.validate.ceiling_wrongness": ("validateIndices", "validateOutput"),
}


def normalize(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value.strip()).lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def parse_report(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    header_index = -1
    headers: list[str] = []
    for index, row in enumerate(rows):
        candidate = [normalize(cell) for cell in row]
        if required <= set(candidate):
            header_index = index
            headers = candidate
            break
    if header_index < 0:
        raise ValueError(f"{path}: cannot locate CSV header containing {sorted(required)}")
    out: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(headers):
            raise ValueError(f"{path}: data row has {len(row)} columns, expected {len(headers)}")
        out.append(dict(zip(headers, row)))
    return out


def integer(row: dict[str, str], key: str) -> int:
    return int(row[key].replace(",", "").strip(), 10)


def canonical_range(value: str) -> str:
    # Nsight Systems 2025 CSV prefixes default-domain Push/Pop names with a colon.
    return value.strip().lstrip(":")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--range-kernels", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    projection_rows: list[dict[str, str]] = []
    range_rows: list[dict[str, str]] = []
    kernel_rows: list[dict[str, str]] = []
    for label, path, required in (
        ("projection", args.projection, {"range", "range_instances", "total_gpu_ops"}),
        ("range kernels", args.range_kernels, {"nvtx_range", "kern_inst", "kernel_name"}),
        ("kernels", args.kernels, {"instances", "name"}),
    ):
        try:
            parsed = parse_report(path, required)
            if label == "projection":
                projection_rows = parsed
            elif label == "range kernels":
                range_rows = parsed
            else:
                kernel_rows = parsed
        except (OSError, UnicodeError, ValueError, KeyError) as exc:
            errors.append(f"{label}: {exc}")

    projected: dict[str, dict[str, int]] = {}
    for name in RANGE_KERNELS:
        matches = [
            row for row in projection_rows if canonical_range(row.get("range", "")) == name
        ]
        if len(matches) != 1:
            errors.append(f"{name}: projection rows={len(matches)}, expected=1")
            continue
        try:
            instances = integer(matches[0], "range_instances")
            gpu_ops = integer(matches[0], "total_gpu_ops")
            projected[name] = {"range_instances": instances, "total_gpu_ops": gpu_ops}
            if instances <= 0 or gpu_ops <= 0:
                errors.append(f"{name}: non-positive range instances/GPU ops")
        except (KeyError, ValueError) as exc:
            errors.append(f"{name}: malformed projection counters ({exc})")

    mapped: dict[str, dict[str, int]] = {}
    for nvtx_range, required_kernels in RANGE_KERNELS.items():
        mapped[nvtx_range] = {}
        relevant = [
            row for row in range_rows
            if canonical_range(row.get("nvtx_range", "")) == nvtx_range
        ]
        for kernel in required_kernels:
            count = 0
            try:
                count = sum(
                    integer(row, "kern_inst")
                    for row in relevant
                    if kernel in row.get("kernel_name", "")
                )
            except (KeyError, ValueError) as exc:
                errors.append(f"{nvtx_range}/{kernel}: malformed instance count ({exc})")
            mapped[nvtx_range][kernel] = count
            if count <= 0:
                errors.append(f"{nvtx_range}: no mapped {kernel} kernel")

    global_workers: dict[str, int] = {}
    for kernel in ("dsaIndexer", "dsaTopk", "dsaAttention"):
        count = 0
        try:
            count = sum(
                integer(row, "instances")
                for row in kernel_rows
                if kernel in row.get("name", "")
            )
        except (KeyError, ValueError) as exc:
            errors.append(f"global {kernel}: malformed instance count ({exc})")
        global_workers[kernel] = count
        if count <= 0:
            errors.append(f"global kernel summary missing {kernel}")

    report_hashes: dict[str, str] = {}
    for label, path in (
        ("projection", args.projection),
        ("range_kernels", args.range_kernels),
        ("kernels", args.kernels),
    ):
        try:
            report_hashes[label] = sha256(path)
        except OSError as exc:
            errors.append(f"cannot hash {path}: {exc}")
    payload = {
        "schema": 1,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "proof_scope": "nvtx_cpu_launch_range_to_gpu_kernel_mapping",
        "required_ranges": list(RANGE_KERNELS),
        "projection": projected,
        "range_kernel_instances": mapped,
        "global_worker_instances": global_workers,
        "report_sha256": report_hashes,
    }
    atomic_json(args.json, payload)
    print(
        f"DSA_PROFILE_VALIDATION schema=1 status={payload['status']} "
        f"errors={len(errors)} ranges={len(projected)}"
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
