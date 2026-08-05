#!/usr/bin/env python3
"""CPU-only preflight for the Tier-4 persistent triplet runner."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from model_identity import (
    build_model_identity,
    sha256_file,
    verify_model_identity,
)
from pdl_evidence import validate_evidence


WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.gguf")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def local_model_files(model: Path) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    if not model.is_dir():
        return [], [f"model is not a local directory: {model}"]
    if not (model / "config.json").is_file():
        errors.append(f"missing local config: {model / 'config.json'}")
    weights = sorted(
        {
            path
            for pattern in WEIGHT_PATTERNS
            for path in model.glob(pattern)
            if path.is_file()
        }
    )
    if not weights:
        errors.append("local model has no staged weight files")
    return weights, errors


def model_fingerprint(model: Path, weights: list[Path]) -> str:
    del weights
    return str(build_model_identity(model)["model_fingerprint"])


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def api_errors() -> list[str]:
    errors: list[str] = []
    try:
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
    except Exception as exc:  # noqa: BLE001 - report exact blocker
        return [f"cannot import vLLM: {type(exc).__name__}: {exc}"]
    generate = inspect.signature(LLM.generate).parameters
    constructor = inspect.signature(LLM.__init__).parameters
    if "prompts" not in generate:
        errors.append("installed LLM.generate lacks prompts=")
    engine_args = inspect.signature(EngineArgs).parameters
    for parameter in (
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_offloading_size",
        "kv_offloading_backend",
    ):
        if parameter not in engine_args:
            errors.append(f"installed EngineArgs lacks {parameter}")
    if "compilation_config" not in constructor:
        errors.append("installed LLM constructor lacks compilation_config")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--phase", choices=["before", "after"], default="before")
    parser.add_argument("--proof-root", type=Path)
    parser.add_argument("--model-identity", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    model = args.model.resolve()
    results = args.results.resolve()
    weights, errors = local_model_files(model)
    identity = None
    identity_sha256 = None
    if not errors:
        if args.model_identity is not None:
            try:
                identity = json.loads(args.model_identity.resolve().read_text())
                identity_sha256 = sha256_file(args.model_identity.resolve())
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"cannot read declared model identity: {exc}")
            else:
                errors.extend(verify_model_identity(identity, model))
        else:
            try:
                identity = build_model_identity(model)
            except RuntimeError as exc:
                errors.append(str(exc))
    fingerprint = (
        identity.get("model_fingerprint") if isinstance(identity, dict) else None
    )
    versions = {
        name: package_version(name)
        for name in ("vllm", "torch", "triton", "transformers")
    }
    if "MISSING" in versions.values():
        errors.append(f"required package missing: {versions}")
    if shutil.which("nsys") is None:
        errors.append("nsys is missing")
    errors.extend(api_errors())

    evidence = None
    if args.phase == "before":
        if results.exists() and (
            not results.is_dir() or any(results.iterdir())
        ):
            errors.append(f"cohort result path must be absent or empty: {results}")
    else:
        if args.proof_root is None:
            errors.append("after phase requires --proof-root")
        elif fingerprint is not None:
            evidence = validate_evidence(args.proof_root, fingerprint)
            if evidence["status"] != "ok":
                errors.extend(f"proof: {message}" for message in evidence["errors"])

    result = {
        "schema": "tier4.preflight.v2",
        "status": "PASS" if not errors else "BLOCKED",
        "phase": args.phase,
        "errors": errors,
        "model": str(model),
        "model_fingerprint": fingerprint,
        "model_identity_manifest": (
            str(args.model_identity.resolve()) if args.model_identity else None
        ),
        "model_identity_manifest_sha256": identity_sha256,
        "weight_files": len(weights),
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "packages": versions,
        "results": str(results),
        "runner_mode": "same_process_adjacent_latin3",
        "graph_mode": "FULL_DECODE_ONLY",
        "evidence": evidence,
        "notes": [
            "model registry access is disabled by the runner",
            "before phase is intentionally non-circular; target proof is checked after capture",
        ],
    }
    if args.json:
        atomic_json(args.json, result)
    stream = sys.stdout if not errors else sys.stderr
    print(f"PREFLIGHT status={result['status']} phase={args.phase}", file=stream)
    for error in errors:
        print(f"  - {error}", file=stream)
    return 0 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())
