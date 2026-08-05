#!/usr/bin/env python3
"""Fail-closed target evidence validator for the Tier-4 LLM triplet.

The validator independently scans the three isolated compile caches and then
proves that their PTX entries executed as CUDA-graph nodes inside each rung's
own exact ``TIER4_PROOF`` NVTX interval.  A kernel with the same name outside
that interval, or in another rung's interval in the shared profile, cannot
satisfy the contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any


RUNGS = ("pdl_off", "pdl_grid", "ceiling")
WAIT = re.compile(r"(?m)^\s*(?:@\S+\s+)?griddepcontrol\.wait\s*;\s*(?://.*)?$")
LAUNCH = re.compile(
    r"(?m)^\s*(?:@\S+\s+)?griddepcontrol\.launch_dependents\s*;\s*(?://.*)?$"
)
ENTRY = re.compile(r"(?m)^\s*(?:\.visible\s+)?\.entry\s+([^\s(]+)\s*\(")
TARGET = re.compile(r"(?m)^\s*\.target\s+([^\s,]+)")
GENERIC_KERNEL_NAMES = {"", "kernel", "triton_"}
MIN_DISTINCT_PROOF_NAMES = 2
MIN_PTX_NAME_COVERAGE = 0.50
PDL_SCOPE = "inductor_generated_full_decode_graph_kernels"
PREFILL_MODE = "production_mixed_mode_non_headline"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal_kernel_name(value: str) -> str:
    """Controlled normalization only; deliberately no substring matching."""
    name = value.strip()
    if name.startswith("void "):
        name = name[5:].lstrip()
    if "(" in name:
        name = name.split("(", 1)[0].rstrip()
    return name


def _descriptive(values: set[str]) -> list[str]:
    return sorted({
        name
        for value in values
        if (name := _normal_kernel_name(value)) not in GENERIC_KERNEL_NAMES
    })


def scan_ptx(cache: Path) -> dict[str, Any]:
    ptx_files = sorted(cache.rglob("*.ptx")) if cache.is_dir() else []
    cubin_files = sorted(cache.rglob("*.cubin")) if cache.is_dir() else []
    waits = 0
    launches = 0
    entries: set[str] = set()
    name_occurrences: dict[str, list[dict[str, Any]]] = {}
    targets: set[str] = set()
    paired = 0
    files: list[dict[str, Any]] = []
    manifest = hashlib.sha256()

    for path in sorted(ptx_files + cubin_files):
        relative = str(path.relative_to(cache))
        digest = sha256_file(path)
        manifest.update(relative.encode())
        manifest.update(digest.encode())

    for path in ptx_files:
        text = path.read_text(errors="replace")
        file_waits = len(WAIT.findall(text))
        file_launches = len(LAUNCH.findall(text))
        file_entries = set(ENTRY.findall(text))
        file_target = TARGET.search(text)
        if file_target:
            targets.add(file_target.group(1))
        waits += file_waits
        launches += file_launches
        entries.update(file_entries)
        candidate_names = set(file_entries)
        candidate_names.add(path.stem)
        for candidate in candidate_names:
            normalized = _normal_kernel_name(candidate)
            if normalized in GENERIC_KERNEL_NAMES:
                continue
            name_occurrences.setdefault(normalized, []).append(
                {
                    "path": str(path.relative_to(cache)),
                    "wait_count": file_waits,
                    "launch_count": file_launches,
                }
            )
        cubin = path.with_suffix(".cubin")
        if cubin.is_file():
            paired += 1

        metadata_path = path.with_suffix(".json")
        metadata: dict[str, Any] | None = None
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text())
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, json.JSONDecodeError):
                metadata = None
        record = {
            "path": str(path.relative_to(cache)),
            "stem": path.stem,
            "entries": sorted(file_entries),
            "wait_count": file_waits,
            "launch_count": file_launches,
            "target": file_target.group(1) if file_target else None,
            "paired_cubin": cubin.is_file(),
            "metadata_launch_pdl": (
                metadata.get("launch_pdl") if metadata is not None else None
            ),
            "metadata_arch": metadata.get("arch") if metadata is not None else None,
        }
        if file_waits or file_launches:
            files.append(record)

    # A short name is eligible only when *every* PTX file in this isolated
    # cache that exposes the entry/stem has the corresponding semantics.  This
    # prevents one wait-bearing file from laundering a same-name collision in
    # another file through Nsight's short-name-only kernel table.
    name_coverage = {
        name: {
            "occurrences": len(occurrences),
            "all_wait": all(item["wait_count"] > 0 for item in occurrences),
            "all_launch": all(item["launch_count"] > 0 for item in occurrences),
            "files": occurrences,
        }
        for name, occurrences in sorted(name_occurrences.items())
    }
    wait_entries = {
        name for name, coverage in name_coverage.items() if coverage["all_wait"]
    }
    launch_entries = {
        name for name, coverage in name_coverage.items() if coverage["all_launch"]
    }
    ambiguous_wait_entries = {
        name
        for name, occurrences in name_occurrences.items()
        if any(item["wait_count"] > 0 for item in occurrences)
        and not all(item["wait_count"] > 0 for item in occurrences)
    }
    ambiguous_launch_entries = {
        name
        for name, occurrences in name_occurrences.items()
        if any(item["launch_count"] > 0 for item in occurrences)
        and not all(item["launch_count"] > 0 for item in occurrences)
    }
    # Off has no GDC-bearing subset, so use all descriptive target entries to
    # tie multiple independently compiled kernels to its own graph.
    off_candidates = set(entries) | {path.stem for path in ptx_files}
    return {
        "ptx_files": len(ptx_files),
        "cubin_files": len(cubin_files),
        "paired_cubin_files": paired,
        "wait_count": waits,
        "launch_count": launches,
        "targets": sorted(targets),
        "manifest_sha256": manifest.hexdigest(),
        "entries": _descriptive(off_candidates),
        "wait_entries": sorted(wait_entries),
        "launch_entries": sorted(launch_entries),
        "ambiguous_wait_entries": sorted(ambiguous_wait_entries),
        "ambiguous_launch_entries": sorted(ambiguous_launch_entries),
        "short_name_collisions": sorted(
            name
            for name, coverage in name_coverage.items()
            if coverage["occurrences"] > 1
        ),
        "name_coverage": name_coverage,
        "semantic_ptx": files,
    }


def inspect_nsys_sqlite(
    path: Path,
    entry_groups: dict[str, list[str]],
    rung: str,
    triplet_id: str,
) -> dict[str, Any]:
    """Join target entries to graph kernels in exactly one rung NVTX window."""
    label = f"TIER4_PROOF|rung={rung}|triplet={triplet_id}"
    result: dict[str, Any] = {
        "nvtx_label": label,
        "nvtx_range": None,
        "kernel_events_in_window": 0,
        "graph_kernel_events_in_window": 0,
        "known_graph_node_kernel_events": 0,
        "graph_kernel_names": [],
        "graph_kernel_name_counts": {},
        "matched_entries": {},
        "entry_group_coverage": {},
        "errors": [],
    }
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        result["errors"].append(f"cannot open Nsight sqlite: {exc}")
        return result
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {
            "CUDA_GRAPH_NODE_EVENTS",
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "NVTX_EVENTS",
            "StringIds",
        }
        if not required <= tables:
            result["errors"].append(
                "Nsight sqlite lacks required tables: "
                + ",".join(sorted(required - tables))
            )
            return result

        kernel_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
        }
        if not {"start", "end", "graphNodeId", "shortName"} <= kernel_columns:
            result["errors"].append(
                "Nsight kernel table lacks start/end/graphNodeId/shortName"
            )
            return result
        ranges = list(
            connection.execute(
                "SELECT start,end FROM NVTX_EVENTS WHERE text=?", (label,)
            )
        )
        if len(ranges) != 1:
            result["errors"].append(
                f"expected exactly one exact NVTX proof range, found {len(ranges)}"
            )
            return result
        start, end = ranges[0]
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            result["errors"].append(f"invalid NVTX proof interval: {ranges[0]}")
            return result
        result["nvtx_range"] = {"start": start, "end": end}

        result["kernel_events_in_window"] = connection.execute(
            """
            SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start>=? AND end<=?
            """,
            (start, end),
        ).fetchone()[0]
        graph_rows = list(
            connection.execute(
                """
                SELECT kernels.graphNodeId, strings.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
                JOIN StringIds AS strings ON strings.id=kernels.shortName
                WHERE kernels.start>=? AND kernels.end<=?
                  AND kernels.graphNodeId IS NOT NULL
                """,
                (start, end),
            )
        )
        result["graph_kernel_events_in_window"] = len(graph_rows)
        node_ids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                "WHERE graphNodeId IS NOT NULL"
            )
        }
        result["known_graph_node_kernel_events"] = sum(
            graph_node_id in node_ids for graph_node_id, _name in graph_rows
        )
        counts = Counter(str(name) for _node, name in graph_rows)
        result["graph_kernel_names"] = sorted(counts)
        result["graph_kernel_name_counts"] = dict(sorted(counts.items()))

        normalized_nsys: dict[str, list[str]] = {}
        for raw_name in counts:
            normalized_nsys.setdefault(_normal_kernel_name(raw_name), []).append(raw_name)
        for group, raw_entries in entry_groups.items():
            matches: list[dict[str, str]] = []
            for raw_entry in sorted(set(raw_entries)):
                normalized = _normal_kernel_name(raw_entry)
                for nsys_name in normalized_nsys.get(normalized, []):
                    matches.append(
                        {
                            "ptx_or_stem": raw_entry,
                            "normalized": normalized,
                            "nsys_short_name": nsys_name,
                        }
                    )
            result["matched_entries"][group] = matches
            eligible_names = {
                _normal_kernel_name(entry)
                for entry in raw_entries
                if _normal_kernel_name(entry) not in GENERIC_KERNEL_NAMES
            }
            matched_names = {item["normalized"] for item in matches}
            coverage = (
                len(matched_names) / len(eligible_names) if eligible_names else 0.0
            )
            result["entry_group_coverage"][group] = {
                "eligible_distinct_names": len(eligible_names),
                "matched_distinct_names": len(matched_names),
                "coverage": coverage,
                "minimum_distinct_names": MIN_DISTINCT_PROOF_NAMES,
                "minimum_coverage": MIN_PTX_NAME_COVERAGE,
            }
            if len(eligible_names) < MIN_DISTINCT_PROOF_NAMES:
                result["errors"].append(
                    f"{group} has fewer than {MIN_DISTINCT_PROOF_NAMES} "
                    "collision-safe descriptive PTX names"
                )
            if len(matched_names) < MIN_DISTINCT_PROOF_NAMES:
                result["errors"].append(
                    f"fewer than {MIN_DISTINCT_PROOF_NAMES} distinct {group} PTX "
                    f"names executed as graph nodes in {label}"
                )
            if coverage < MIN_PTX_NAME_COVERAGE:
                result["errors"].append(
                    f"{group} PTX-to-graph name coverage {coverage:.3f} is below "
                    f"{MIN_PTX_NAME_COVERAGE:.3f}"
                )

        if result["kernel_events_in_window"] <= 0:
            result["errors"].append("NVTX proof interval contains no CUDA kernels")
        if not graph_rows:
            result["errors"].append("NVTX proof interval contains no graph-node kernels")
        if result["known_graph_node_kernel_events"] != len(graph_rows):
            result["errors"].append(
                "some in-window graphNodeId kernel records are absent from "
                "CUDA_GRAPH_NODE_EVENTS"
            )
    except sqlite3.Error as exc:
        result["errors"].append(f"cannot query Nsight sqlite: {exc}")
    finally:
        connection.close()
    return result


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None, f"missing {path}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read {path}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path} is not a JSON object"
    return value, None


def _worker_errors(rung: str, workers: Any, cache_root: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(workers, list) or not workers:
        return [f"{rung}: no worker-side build probes"]
    expected_pdl = rung != "pdl_off"
    expected_ceiling = rung == "ceiling"
    for index, worker in enumerate(workers):
        prefix = f"{rung}: worker[{index}]"
        if not isinstance(worker, dict):
            errors.append(f"{prefix} is not an object")
            continue
        if not isinstance(worker.get("pid"), int) or isinstance(worker.get("pid"), bool):
            errors.append(f"{prefix} has no integer pid")
        if worker.get("pdl_env") is not expected_pdl:
            errors.append(f"{prefix} PDL environment mismatch")
        if worker.get("pdl_inductor_config") is not expected_pdl:
            errors.append(f"{prefix} Inductor PDL config mismatch")
        if worker.get("ceiling_hook") is not expected_ceiling:
            errors.append(f"{prefix} ceiling hook mismatch")
        if worker.get("trtllm_enable_pdl") is not False:
            errors.append(f"{prefix} TRTLLM PDL must remain explicitly out of scope")
        if worker.get("configured_graph_mode") != "FULL_DECODE_ONLY":
            errors.append(f"{prefix} configured graph mode is not FULL_DECODE_ONLY")
        if worker.get("dispatcher_mode") != "FULL_DECODE_ONLY":
            errors.append(f"{prefix} dispatcher graph mode is not FULL_DECODE_ONLY")
        if not isinstance(worker.get("full_graph_entries"), int) or worker["full_graph_entries"] <= 0:
            errors.append(f"{prefix} has no retained FULL graph entry")
        if worker.get("cache_root") != str(cache_root.resolve()):
            errors.append(f"{prefix} cache root mismatch")
        endpoints = worker.get("compile_ranges_endpoints")
        if not isinstance(endpoints, list) or not endpoints or not all(
            isinstance(value, int) and value > 0 for value in endpoints
        ):
            errors.append(f"{prefix} has invalid compile range endpoints")
        compiled = worker.get("compiled_modules")
        if not isinstance(compiled, list) or not compiled:
            errors.append(f"{prefix} has no actual compiled-module probe")
        else:
            for module in compiled:
                if not isinstance(module, dict) or module.get("compiled") is not True:
                    errors.append(f"{prefix} contains an uncompiled module probe")
                    continue
                if not isinstance(module.get("name"), str) or not module["name"]:
                    errors.append(f"{prefix} compiled module path is absent")
                ids = (
                    module.get("compiled_callable_id"),
                    module.get("aot_compiled_fn_id"),
                    module.get("compiled_bytecode_id"),
                )
                if not any(isinstance(value, int) and value > 0 for value in ids):
                    errors.append(f"{prefix} compiled module has no callable identity")
        active = worker.get("active_variant")
        if not isinstance(active, dict):
            errors.append(f"{prefix} lacks active callable/graph identity")
        else:
            if active.get("schema") != "tier4.active_variant.v1":
                errors.append(f"{prefix} active variant schema mismatch")
            if active.get("rung") != rung:
                errors.append(f"{prefix} active variant rung mismatch")
            for field in (
                "compiled_callable_fingerprint",
                "graph_dictionary_fingerprint",
                "batch_graph_fingerprint",
                "activation_fingerprint",
            ):
                value = active.get(field)
                if not isinstance(value, str) or re.fullmatch(
                    r"[0-9a-f]{64}", value
                ) is None:
                    errors.append(f"{prefix} invalid active {field}")
            if (
                not isinstance(active.get("graph_dictionary_id"), int)
                or isinstance(active.get("graph_dictionary_id"), bool)
                or active["graph_dictionary_id"] <= 0
            ):
                errors.append(f"{prefix} invalid active graph dictionary id")
            if active.get("full_graph_entries") != worker.get("full_graph_entries"):
                errors.append(f"{prefix} active graph entry count mismatch")
        cache_scan = worker.get("cache_scan")
        if not isinstance(cache_scan, dict):
            errors.append(f"{prefix} lacks post-build cache scan")
        else:
            if not isinstance(cache_scan.get("ptx_files"), int) or cache_scan["ptx_files"] <= 0:
                errors.append(f"{prefix} post-build cache scan has no PTX")
        backends = worker.get("attention_implementation_counts")
        if not isinstance(backends, dict) or not backends or any(
            not isinstance(name, str)
            or not isinstance(count, int)
            or count <= 0
            for name, count in backends.items()
        ):
            errors.append(f"{prefix} lacks concrete attention backend evidence")
        kv_cache = worker.get("kv_cache")
        if not isinstance(kv_cache, dict):
            errors.append(f"{prefix} lacks resolved KV cache metadata")
        else:
            if not isinstance(kv_cache.get("gpu_num_blocks"), int) or kv_cache["gpu_num_blocks"] <= 0:
                errors.append(f"{prefix} has no positive GPU KV block capacity")
            if not isinstance(kv_cache.get("block_size"), int) or kv_cache["block_size"] <= 0:
                errors.append(f"{prefix} has invalid KV block size")
            if not isinstance(kv_cache.get("groups"), list) or not kv_cache["groups"]:
                errors.append(f"{prefix} lacks KV group/page metadata")
    return errors


def validate_evidence(
    root: Path, expected_model_fingerprint: str | None = None
) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    details: dict[str, Any] = {}
    cross_rung_correspondence: dict[str, list[str]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return {
            "status": "blocked",
            "errors": [f"evidence root is not a directory: {root}"],
            "rungs": {},
        }

    directories = {path.name for path in root.iterdir() if path.is_dir()}
    if directories != set(RUNGS):
        errors.append(
            f"evidence directories must be exactly {list(RUNGS)}, got {sorted(directories)}"
        )

    for rung_index, rung in enumerate(RUNGS):
        rung_root = root / rung
        runtime, problem = _load_json(rung_root / "runtime.json")
        cache_root = rung_root / "cache"
        scan = scan_ptx(cache_root)
        details[rung] = {"runtime": runtime, "ptx": scan}
        if problem:
            errors.append(problem)
            continue
        assert runtime is not None
        runtimes[rung] = runtime

        if runtime.get("schema") != "tier4.runtime.v2":
            errors.append(f"{rung}: runtime schema is not tier4.runtime.v2")
        if runtime.get("rung") != rung or runtime.get("rung_index") != rung_index:
            errors.append(f"{rung}: rung/order mismatch")
        if runtime.get("triplet_mode") != "same_process_adjacent_latin3":
            errors.append(f"{rung}: triplet mode is not adjacent Latin-3")
        if runtime.get("cache_fresh") is not True:
            errors.append(f"{rung}: cache is not stamped fresh")
        if runtime.get("cache_root") != str(cache_root.resolve()):
            errors.append(f"{rung}: runtime cache root mismatch")
        if runtime.get("graph_mode") != "FULL_DECODE_ONLY":
            errors.append(f"{rung}: graph mode is not FULL_DECODE_ONLY")
        if runtime.get("executed_full_decode") is not True:
            errors.append(f"{rung}: actual FULL-decode execution is not asserted")
        if runtime.get("pdl_scope") != PDL_SCOPE:
            errors.append(f"{rung}: PDL scope mismatch")
        if runtime.get("prefill_mode") != PREFILL_MODE:
            errors.append(f"{rung}: prefill is not marked separate production mixed-mode")
        if runtime.get("trtllm_enable_pdl") is not False:
            errors.append(f"{rung}: TRTLLM PDL must be explicitly disabled/out of scope")
        candidate_runtime_hash = runtime.get("candidate_runtime_sha256")
        if not isinstance(candidate_runtime_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", candidate_runtime_hash
        ) is None:
            errors.append(f"{rung}: candidate runtime hash binding is absent")

        packages = runtime.get("packages")
        required_packages = {"vllm", "torch", "triton", "transformers"}
        if not isinstance(packages, dict) or not required_packages <= set(packages):
            errors.append(f"{rung}: incomplete package versions")
        errors.extend(_worker_errors(rung, runtime.get("workers"), cache_root))

        if scan["ptx_files"] <= 0 or scan["cubin_files"] <= 0:
            errors.append(f"{rung}: cache has no PTX/cubin")
        if scan["paired_cubin_files"] != scan["ptx_files"]:
            errors.append(f"{rung}: every PTX must have a same-stem cubin")
        if not any(target.startswith("sm_100") for target in scan["targets"]):
            errors.append(f"{rung}: cache has no B200 sm_100 target PTX")
        waits, launches = scan["wait_count"], scan["launch_count"]
        if rung == "pdl_off" and (waits != 0 or launches != 0):
            errors.append("pdl_off: expected wait=0 and launch=0")
        elif rung == "pdl_grid" and (waits <= 0 or launches <= 0):
            errors.append("pdl_grid: expected wait>0 and launch>0")
        elif rung == "ceiling" and (waits != 0 or launches <= 0):
            errors.append("ceiling: expected wait=0 and launch>0")

        if runtime.get("graph_execution_proof") != "nsys_cuda_graph_node_nvtx_window":
            errors.append(f"{rung}: graph execution proof is not finalized")
        nsys_name = runtime.get("nsys_sqlite")
        nsys_hash = runtime.get("nsys_sqlite_sha256")
        triplet_id = runtime.get("triplet_id")
        if not all(isinstance(value, str) and value for value in (nsys_name, nsys_hash, triplet_id)):
            errors.append(f"{rung}: incomplete Nsight path/hash/triplet identity")
            continue
        expected_label = f"TIER4_PROOF|rung={rung}|triplet={triplet_id}"
        if runtime.get("nsys_nvtx_label") != expected_label:
            errors.append(f"{rung}: runtime exact NVTX label mismatch")
        nsys_path = (rung_root / nsys_name).resolve()
        if not nsys_path.is_file():
            errors.append(f"{rung}: missing Nsight sqlite {nsys_path}")
            continue
        if sha256_file(nsys_path) != nsys_hash:
            errors.append(f"{rung}: Nsight sqlite hash mismatch")
            continue
        groups = (
            {"off-compiled": scan["entries"]}
            if rung == "pdl_off"
            else (
                {
                    "wait-bearing": scan["wait_entries"],
                    "launch-bearing": scan["launch_entries"],
                }
                if rung == "pdl_grid"
                else {"launch-bearing": scan["launch_entries"]}
            )
        )
        nsys = inspect_nsys_sqlite(nsys_path, groups, rung, triplet_id)
        details[rung]["nsys"] = nsys
        errors.extend(f"{rung}: {message}" for message in nsys["errors"])

    if len(runtimes) == len(RUNGS):
        def same_nonempty(field: str, expected_type: type) -> list[Any]:
            values = [runtimes[rung].get(field) for rung in RUNGS]
            if any(
                not isinstance(value, expected_type)
                or isinstance(value, bool)
                or (expected_type is str and not value)
                for value in values
            ) or len(set(values)) != 1:
                errors.append(f"rungs do not share one valid {field}")
            return values

        same_nonempty("triplet_id", str)
        same_nonempty("driver_pid", int)
        fingerprints = same_nonempty("model_fingerprint", str)
        same_nonempty("model_identity_manifest_sha256", str)
        same_nonempty("model_identity_snapshot_sha256", str)
        if expected_model_fingerprint is not None and any(
            value != expected_model_fingerprint for value in fingerprints
        ):
            errors.append("model fingerprint does not match requested model")
        package_sets = {
            json.dumps(runtimes[rung].get("packages"), sort_keys=True) for rung in RUNGS
        }
        if len(package_sets) != 1:
            errors.append("package versions differ across rungs")
        same_nonempty("cohort_id", str)
        device_sets = {
            json.dumps(runtimes[rung].get("device"), sort_keys=True) for rung in RUNGS
        }
        if len(device_sets) != 1:
            errors.append("device identity differs across rungs")
        endpoints = {
            tuple(worker.get("compile_ranges_endpoints", []))
            for rung in RUNGS
            for worker in runtimes[rung].get("workers", [])
            if isinstance(worker, dict)
        }
        if len(endpoints) != 1:
            errors.append("compile ranges differ across rung workers")
        cohorts: list[tuple[int, ...]] = []
        module_topologies: list[tuple[tuple[str, str], ...]] = []
        callable_identities: list[tuple[tuple[Any, Any, Any], ...]] = []
        for rung in RUNGS:
            workers = runtimes[rung].get("workers", [])
            cohorts.append(
                tuple(
                    sorted(
                        worker["pid"]
                        for worker in workers
                        if isinstance(worker, dict)
                        and isinstance(worker.get("pid"), int)
                        and not isinstance(worker.get("pid"), bool)
                    )
                )
            )
            modules = workers[0].get("compiled_modules", []) if workers else []
            module_topologies.append(
                tuple(
                    (module.get("name"), module.get("type"))
                    for module in modules
                    if isinstance(module, dict)
                )
            )
            callable_identities.append(
                tuple(
                    (
                        module.get("compiled_callable_id"),
                        module.get("aot_compiled_fn_id"),
                        module.get("compiled_bytecode_id"),
                    )
                    for module in modules
                    if isinstance(module, dict)
                )
            )
        if len(set(cohorts)) != 1 or not cohorts[0]:
            errors.append("worker cohort is not stable across rungs")
        if len(set(module_topologies)) != 1 or not module_topologies[0]:
            errors.append("compiled-module topology changed across rungs")
        if len(set(callable_identities)) != len(RUNGS):
            errors.append("rungs did not retain three distinct compiled callable states")
        for field in (
            "compiled_callable_fingerprint",
            "graph_dictionary_id",
            "graph_dictionary_fingerprint",
            "batch_graph_fingerprint",
            "activation_fingerprint",
        ):
            active_identities = {
                worker.get("active_variant", {}).get(field)
                for rung in RUNGS
                for worker in runtimes[rung].get("workers", [])
                if isinstance(worker, dict)
                and isinstance(worker.get("active_variant"), dict)
            }
            if len(active_identities) != len(RUNGS) * len(cohorts[0]):
                errors.append(f"rungs lack distinct active {field} identities")
        manifests = {details[rung]["ptx"]["manifest_sha256"] for rung in RUNGS}
        if len(manifests) != len(RUNGS):
            errors.append("rung compile-cache manifests are not distinct")

        # Tie the off variant to the *same named Inductor kernels* whose grid
        # versions bear waits, and tie Ceiling to grid's launch-bearing set.
        # This rules out satisfying the off proof with an unrelated helper.
        off_graph_names = {
            _normal_kernel_name(name)
            for name in details["pdl_off"].get("nsys", {}).get("graph_kernel_names", [])
        }
        ceiling_graph_names = {
            _normal_kernel_name(name)
            for name in details["ceiling"].get("nsys", {}).get("graph_kernel_names", [])
        }
        grid_wait_names = {
            _normal_kernel_name(name)
            for name in details["pdl_grid"]["ptx"]["wait_entries"]
        }
        grid_launch_names = {
            _normal_kernel_name(name)
            for name in details["pdl_grid"]["ptx"]["launch_entries"]
        }
        off_corresponding = sorted(off_graph_names & grid_wait_names)
        ceiling_corresponding = sorted(ceiling_graph_names & grid_launch_names)
        cross_rung_correspondence = {
            "off_graph_nodes_matching_grid_wait_entries": off_corresponding,
            "ceiling_graph_nodes_matching_grid_launch_entries": ceiling_corresponding,
        }
        if len(off_corresponding) < MIN_DISTINCT_PROOF_NAMES:
            errors.append(
                "off proof did not execute multiple kernels corresponding to grid wait PTX"
            )
        if len(ceiling_corresponding) < MIN_DISTINCT_PROOF_NAMES:
            errors.append(
                "ceiling proof did not execute multiple kernels corresponding to grid launch PTX"
            )
        if grid_wait_names and (
            len(off_corresponding) / len(grid_wait_names) < MIN_PTX_NAME_COVERAGE
        ):
            errors.append("off/grid-wait cross-rung name coverage is below threshold")
        if grid_launch_names and (
            len(ceiling_corresponding) / len(grid_launch_names)
            < MIN_PTX_NAME_COVERAGE
        ):
            errors.append("ceiling/grid-launch cross-rung name coverage is below threshold")

        nsys_paths = {
            (root / rung / str(runtimes[rung].get("nsys_sqlite"))).resolve()
            for rung in RUNGS
        }
        if len(nsys_paths) != 1:
            errors.append("rungs must refer to one shared same-process Nsight profile")
        windows: list[tuple[int, int, str]] = []
        for rung in RUNGS:
            window = details[rung].get("nsys", {}).get("nvtx_range")
            if isinstance(window, dict):
                windows.append((window.get("start"), window.get("end"), rung))
        if len(windows) == len(RUNGS):
            for index, (start, end, rung) in enumerate(windows):
                if rung != RUNGS[index] or not isinstance(start, int) or not isinstance(end, int):
                    errors.append("Nsight proof windows lack fixed rung order")
                    break
                if index and windows[index - 1][1] > start:
                    errors.append("Nsight proof windows overlap or are out of order")
                    break

    return {
        "schema": "tier4.evidence.validation.v2",
        "status": "ok" if not errors else "blocked",
        "errors": errors,
        "rungs": details,
        "cross_rung_correspondence": cross_rung_correspondence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = validate_evidence(args.root, args.model_fingerprint)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] == "ok":
        print("EVIDENCE status=ok scope=isolated-ptx+cubin+per-rung-nvtx-graph-node")
        return 0
    print("EVIDENCE status=blocked", file=sys.stderr)
    for error in result["errors"]:
        print(f"  - {error}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
