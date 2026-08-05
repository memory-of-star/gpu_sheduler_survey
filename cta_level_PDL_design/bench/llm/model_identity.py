#!/usr/bin/env python3
"""Deterministic local-model identity without rereading weight payloads.

The identity hashes every top-level non-weight artifact (including config,
tokenizer, and the safetensors index), records the complete weight-shard
name/size inventory, and captures any locally available revision evidence.
It deliberately does not hash tens of GiB of weight contents for every cohort.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


WEIGHT_SUFFIXES = {".safetensors", ".bin", ".gguf"}
SCHEMA = "tier4.model.identity.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_revision(model: Path) -> str | None:
    head = model / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(errors="replace").strip()
    if value.startswith("ref: "):
        ref = model / ".git" / value[5:]
        if ref.is_file():
            return ref.read_text(errors="replace").strip() or None
    return value or None


def _declared_revision(config: Path) -> str | None:
    try:
        value = json.loads(config.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    for key in ("_commit_hash", "commit_hash", "revision"):
        revision = value.get(key)
        if isinstance(revision, str) and revision:
            return revision
    return None


def _snapshot_revision(model: Path) -> str | None:
    resolved = model.resolve()
    if resolved.parent.name == "snapshots" and resolved.name:
        return resolved.name
    return None


def build_model_identity(model: Path) -> dict[str, Any]:
    model = model.resolve()
    if not model.is_dir():
        raise RuntimeError(f"model is not a local directory: {model}")
    config = model / "config.json"
    if not config.is_file():
        raise RuntimeError(f"missing model config: {config}")

    files = sorted(path for path in model.iterdir() if path.is_file())
    weights = [path for path in files if path.suffix in WEIGHT_SUFFIXES]
    if not weights:
        raise RuntimeError(f"model has no local weight shards: {model}")
    metadata = [path for path in files if path.suffix not in WEIGHT_SUFFIXES]
    metadata_records = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in metadata
    ]
    weight_records = [
        {"path": path.name, "size_bytes": path.stat().st_size}
        for path in weights
    ]

    index_path = model / "model.safetensors.index.json"
    indexed_shards: set[str] = set()
    index_error: str | None = None
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                index_error = "safetensors index has no nonempty weight_map"
            elif any(not isinstance(value, str) for value in weight_map.values()):
                index_error = "safetensors index contains a non-string shard name"
            else:
                indexed_shards = set(weight_map.values())
        except (OSError, json.JSONDecodeError) as exc:
            index_error = f"cannot parse safetensors index: {exc}"
    actual_safetensors = {
        record["path"] for record in weight_records if record["path"].endswith(".safetensors")
    }
    if index_path.is_file() and index_error is None and indexed_shards != actual_safetensors:
        index_error = (
            "safetensors index shard set differs from local shard inventory: "
            f"indexed={sorted(indexed_shards)} actual={sorted(actual_safetensors)}"
        )
    if index_error is not None:
        raise RuntimeError(index_error)

    inventory_hash = digest_json(weight_records)
    metadata_hash = digest_json(metadata_records)
    local_revision = digest_json(
        {"metadata_manifest_sha256": metadata_hash, "weight_inventory_sha256": inventory_hash}
    )
    identity: dict[str, Any] = {
        "schema": SCHEMA,
        "model": str(model),
        "revision": {
            "declared_commit": _declared_revision(config),
            "git_commit": _git_revision(model),
            "hf_snapshot_revision": _snapshot_revision(model),
            "local_content_revision": local_revision,
        },
        "metadata_files": metadata_records,
        "metadata_manifest_sha256": metadata_hash,
        "weight_files": weight_records,
        "weight_inventory_sha256": inventory_hash,
        "weight_file_count": len(weight_records),
        "weight_bytes": sum(record["size_bytes"] for record in weight_records),
        "safetensors_index_present": index_path.is_file(),
        "safetensors_index_shards": sorted(indexed_shards),
        "weight_contents_hashed_per_cohort": False,
    }
    identity["model_fingerprint"] = digest_json(identity)
    return identity


def verify_model_identity(identity: Any, model: Path) -> list[str]:
    if not isinstance(identity, dict):
        return ["model identity is not an object"]
    if identity.get("schema") != SCHEMA:
        return [f"model identity schema is not {SCHEMA}"]
    try:
        current = build_model_identity(model)
    except RuntimeError as exc:
        return [str(exc)]
    if identity != current:
        return ["staged model metadata/revision/weight inventory changed"]
    return []
