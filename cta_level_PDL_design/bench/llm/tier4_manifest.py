#!/usr/bin/env python3
"""Create or verify the immutable formal Tier-4 root contract.

The root ``manifest.json`` and ``model_identity.json`` are write-once.  Later
invocations reconstruct the complete contract in memory and require exact
equality before writing only the separate, non-contract ``attempts.json``
ledger.  This prevents a resumed decode/prefill campaign from silently mixing
models, sources, packages, matrices, or KV strategies in one result root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any

from model_identity import build_model_identity


MATRIX = {
    "decode": {
        "cohort_id": "decode_bs_scan_v1",
        "classification": "headline_full_decode",
        "points": [
            {"tag": f"decode_bs{batch}", "batch": batch, "seq": 64, "gen": 16, "scope": "decode"}
            for batch in (1, 4, 16, 64)
        ],
        "proof_point": {"tag": "decode_bs1", "batch": 1, "seq": 64, "gen": 16, "scope": "decode"},
        "gpu_memory_utilization": 0.82,
        "max_num_batched_tokens": 16384,
    },
    "prefill": {
        "cohort_id": "prefill_context_scan_v1",
        "classification": "production_mixed_mode_non_headline",
        "points": [
            {"tag": "prefill_4k", "batch": 1, "seq": 4096, "gen": 2, "scope": "prefill"},
            {"tag": "prefill_32k", "batch": 1, "seq": 32768, "gen": 2, "scope": "prefill"},
            {"tag": "prefill_128k", "batch": 1, "seq": 131072, "gen": 2, "scope": "prefill"},
        ],
        "proof_point": {"tag": "prefill_full_decode_proof", "batch": 1, "seq": 64, "gen": 2, "scope": "decode"},
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 16384,
    },
}

SOURCE_NAMES = (
    "tier4_driver.py",
    "pdl_evidence.py",
    "tier4_finalize.py",
    "preflight_llm.py",
    "model_identity.py",
    "tier4_manifest.py",
    "run_llm_sweep.sh",
    "run_tier4_schema_v3_strict_smoke.sh",
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_create_json(path: Path, value: dict[str, Any]) -> None:
    """Install a fully written contract file without ever replacing a peer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RuntimeError(f"immutable contract file appeared concurrently: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packages() -> dict[str, str]:
    result = {}
    for name in ("vllm", "torch", "triton", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "MISSING"
    return result


def immutable_contract(
    root: Path,
    model: Path,
    identity: dict[str, Any],
    identity_sha256: str,
    kv_offloading_size: float | None,
) -> dict[str, Any]:
    """Build the complete deterministic root contract without writing files."""
    source_root = Path(__file__).resolve().parent
    return {
        "schema": "tier4.formal.manifest.v3",
        "host": socket.gethostname(),
        "model": str(model),
        "model_fingerprint": identity["model_fingerprint"],
        "model_identity_manifest": str((root / "model_identity.json").resolve()),
        "model_identity_manifest_sha256": identity_sha256,
        "weight_files": identity["weight_file_count"],
        "weight_bytes": identity["weight_bytes"],
        "packages": packages(),
        "semantic_rungs": ["pdl_off", "pdl_grid", "ceiling"],
        "proof_rung_order": ["pdl_off", "pdl_grid", "ceiling"],
        "timing_order_cycle": [
            ["pdl_off", "pdl_grid", "ceiling"],
            ["pdl_grid", "ceiling", "pdl_off"],
            ["ceiling", "pdl_off", "pdl_grid"],
        ],
        "repeats_per_rung_per_point": 31,
        "warmup_triplets_per_point": 3,
        "bootstrap_samples": 2000,
        "graph_mode": "FULL_DECODE_ONLY",
        "pdl_scope": "inductor_generated_full_decode_graph_kernels",
        "trtllm_enable_pdl": False,
        "ceiling": {"unsafe": True, "verified": False},
        "kv_strategy": {
            "kv_offloading_size_gib": kv_offloading_size,
            "kv_offloading_backend": "native",
            "cpu_offload_gb_is_weight_offload_and_not_used_as_kv_proof": True,
            "resolved_gpu_cpu_blocks_pages_and_connector_required_in_worker_probe": True,
        },
        "cohorts": MATRIX,
        "source_sha256": {
            name: sha256_file(source_root / name) for name in SOURCE_NAMES
        },
    }


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not an object: {path}")
    return value


def first_difference(expected: Any, observed: Any, prefix: str = "manifest") -> str:
    """Return one stable path explaining an exact-contract mismatch."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            path = f"{prefix}.{key}"
            if key not in expected:
                return f"unexpected field {path}"
            if key not in observed:
                return f"missing field {path}"
            difference = first_difference(expected[key], observed[key], path)
            if difference:
                return difference
        return ""
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            return f"length mismatch at {prefix}: expected {len(expected)}, got {len(observed)}"
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            difference = first_difference(left, right, f"{prefix}[{index}]")
            if difference:
                return difference
        return ""
    if expected != observed or type(expected) is not type(observed):
        return f"value mismatch at {prefix}: expected {expected!r}, got {observed!r}"
    return ""


def update_attempt_ledger(root: Path, manifest_hash: str) -> None:
    cohorts_root = root / "cohorts"
    cohorts_root.mkdir(exist_ok=True)
    attempts = {
        cohort: [
            attempt_record(path)
            for path in sorted(cohorts_root.glob(f"{cohort}*"))
            if path.is_dir()
        ]
        for cohort in MATRIX
    }
    atomic_json(
        root / "attempts.json",
        {
            "schema": "tier4.formal.attempts.v1",
            "contract_manifest_sha256": manifest_hash,
            "attempts": attempts,
        },
    )


def attempt_record(path: Path) -> dict[str, Any]:
    admission = path / "admission.json"
    candidate = path / "raw_triplet.json"
    failure = path / "finalize_failure.json"
    driver_error = path / "driver_error.json"
    in_progress = path / "finalize_in_progress.json"
    if driver_error.is_file() or failure.is_file() or in_progress.is_file():
        status = "failed_preserved"
    elif admission.is_file():
        status = "admission_present_requires_full_verification"
    elif candidate.is_file():
        status = "candidate_pending_or_rejected"
    else:
        status = "incomplete_preserved"
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "status": status,
    }
    for name, artifact in (
        ("admission", admission),
        ("raw_triplet", candidate),
        ("finalize_failure", failure),
        ("driver_error", driver_error),
        ("finalize_in_progress", in_progress),
    ):
        if artifact.is_file():
            record[f"{name}_sha256"] = sha256_file(artifact)
    if candidate.is_file():
        try:
            raw = json.loads(candidate.read_text())
            record["triplet_id"] = raw.get("triplet_id")
            record["cohort_id"] = raw.get("cohort_id")
            record["repeats"] = raw.get("repeats")
            record["points"] = raw.get("points")
        except (OSError, json.JSONDecodeError):
            record["candidate_parse_error"] = True
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--kv-offloading-size", type=float)
    args = parser.parse_args()
    if args.kv_offloading_size is not None and (
        not math.isfinite(args.kv_offloading_size)
        or args.kv_offloading_size <= 0
    ):
        print(
            "MANIFEST status=blocked KV offloading size must be finite and positive",
            file=sys.stderr,
        )
        return 3
    root = args.results_root.resolve()
    model = args.model.resolve()
    try:
        identity = build_model_identity(model)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 3
    identity_path = root / "model_identity.json"
    manifest_path = root / "manifest.json"
    existing_identity = identity_path.exists()
    existing_manifest = manifest_path.exists()
    if existing_identity != existing_manifest:
        print(
            "MANIFEST status=blocked partial root contract: identity/manifest "
            "must either both exist or both be absent",
            file=sys.stderr,
        )
        return 3
    if not existing_manifest and root.exists() and any(root.iterdir()):
        print(
            f"MANIFEST status=blocked nonempty root lacks contract: {root}",
            file=sys.stderr,
        )
        return 3

    # Serialize in memory so the identity digest used by the candidate exactly
    # matches the bytes that would be atomically installed for a fresh root.
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode()
    identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
    candidate = immutable_contract(
        root, model, identity, identity_sha256, args.kv_offloading_size
    )

    if existing_manifest:
        try:
            observed_identity = load_json_object(identity_path, "model identity")
            observed_manifest = load_json_object(manifest_path, "formal root manifest")
        except RuntimeError as exc:
            print(f"MANIFEST status=blocked {exc}", file=sys.stderr)
            return 3
        if observed_identity != identity or sha256_file(identity_path) != identity_sha256:
            print(
                "MANIFEST status=blocked immutable model identity differs from "
                "the currently resolved model",
                file=sys.stderr,
            )
            return 3
        difference = first_difference(candidate, observed_manifest)
        if difference:
            print(
                f"MANIFEST status=blocked immutable root contract differs: {difference}",
                file=sys.stderr,
            )
            return 3
        status = "verified"
    else:
        root.mkdir(parents=True, exist_ok=True)
        try:
            atomic_create_json(identity_path, identity)
            atomic_create_json(manifest_path, candidate)
        except RuntimeError as exc:
            print(f"MANIFEST status=blocked {exc}", file=sys.stderr)
            return 3
        status = "created"

    manifest_hash = sha256_file(manifest_path)
    update_attempt_ledger(root, manifest_hash)
    print(
        f"MANIFEST status={status} immutable=1 sha256={manifest_hash} "
        f"path={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
