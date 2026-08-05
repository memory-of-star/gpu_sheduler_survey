#!/usr/bin/env python3
"""CPU-only positive and tamper tests for the Tier-4 admission pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any

from model_identity import build_model_identity, sha256_file, verify_model_identity
from pdl_evidence import (
    PDL_SCOPE,
    PREFILL_MODE,
    RUNGS,
    inspect_nsys_sqlite,
    scan_ptx,
    validate_evidence,
)
from tier4_finalize import (
    ORDER_CYCLE,
    SOURCE_NAMES,
    digest_json,
    poison_digest,
    token_digest,
    validate_raw,
)
from tier4_manifest import MATRIX


HERE = Path(__file__).resolve().parent


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fake_model(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "synthetic", "_commit_hash": "local-test-rev"})
    )
    (model / "tokenizer.json").write_text(json.dumps({"tokens": ["a", "b"]}))
    (model / "tokenizer_config.json").write_text(json.dumps({"eos_token": "b"}))
    shard = "model-00001-of-00001.safetensors"
    (model / shard).write_bytes(b"synthetic-weight-inventory-only")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard}})
    )
    identity = build_model_identity(model)
    identity_path = root / "model_identity.json"
    write_json(identity_path, identity)
    return model, identity_path, identity


def worker_probe(
    rung: str, cache: Path, callable_id: int, pid: int = 4242
) -> dict[str, Any]:
    enabled = rung != "pdl_off"
    callable_records = [
        {
            "name": "language_model.model",
            "compiled_callable_id": callable_id,
            "aot_compiled_fn_id": callable_id + 1000,
            "compiled_bytecode_id": None,
        }
    ]
    compiled_callable_fingerprint = digest_json(callable_records)
    graph_dictionary_id = callable_id + 2000
    graph_dictionary_fingerprint = digest_json(
        {"graph_dictionary_id": graph_dictionary_id, "rung": rung}
    )
    batch_graph_fingerprint = digest_json(
        {"rung": rung, "batch_descriptors": [1]}
    )
    active_payload = {
        "rung": rung,
        "compiled_callable_fingerprint": compiled_callable_fingerprint,
        "graph_dictionary_id": graph_dictionary_id,
        "graph_dictionary_fingerprint": graph_dictionary_fingerprint,
        "batch_graph_fingerprint": batch_graph_fingerprint,
        "full_graph_entries": 1,
    }
    active_variant = {
        "schema": "tier4.active_variant.v1",
        **active_payload,
        "activation_fingerprint": digest_json(active_payload),
    }
    return {
        "pid": pid,
        "rung": rung,
        "pdl_env": enabled,
        "pdl_inductor_config": enabled,
        "ceiling_hook": rung == "ceiling",
        "trtllm_enable_pdl": False,
        "configured_graph_mode": "FULL_DECODE_ONLY",
        "dispatcher_mode": "FULL_DECODE_ONLY",
        "compile_ranges_endpoints": [1, 16385],
        "compiled_modules": [
            {
                "name": "language_model.model",
                "type": "TorchCompileWithNoGuardsWrapper",
                "compiled": True,
                "compiled_callable_id": callable_id,
                "aot_compiled_fn_id": callable_id + 1000,
                "compiled_bytecode_id": None,
            }
        ],
        "active_variant": active_variant,
        "full_graph_entries": 1,
        "cache_root": str(cache.resolve()),
        "cache_scan": {"ptx_files": 1, "cubin_files": 1},
        "attention_implementation_counts": {
            "vllm.synthetic.SyntheticAttentionImpl": 1
        },
        "kv_cache": {
            "gpu_num_blocks": 1024,
            "cpu_num_blocks": 0,
            "block_size": 16,
            "kv_cache_memory_bytes": 1024 * 1024,
            "kv_offloading_size_gib": None,
            "kv_offloading_backend": "native",
            "mamba_cache_mode": "align",
            "connector_type": None,
            "groups": [
                {
                    "spec_type": "FullAttentionSpec",
                    "layer_count": 1,
                    "block_size": 16,
                    "page_size_bytes": 4096,
                }
            ],
        },
    }


def make_cache(cache: Path, rung: str) -> None:
    cache.mkdir(parents=True)
    instructions = {
        "pdl_off": "",
        "pdl_grid": (
            "    griddepcontrol.wait;\n"
            "    griddepcontrol.launch_dependents;\n"
        ),
        "ceiling": "    griddepcontrol.launch_dependents;\n",
    }[rung]
    for index in range(2):
        name = f"triton_poi_fused_{index}"
        ptx = (
            ".version 8.8\n"
            ".target sm_100a\n"
            ".address_size 64\n"
            f".visible .entry {name}()\n"
            "{\n"
            f"{instructions}"
            "    ret;\n"
            "}\n"
        )
        (cache / f"{name}.ptx").write_text(ptx)
        (cache / f"{name}.cubin").write_bytes(
            f"synthetic-{rung}-{index}".encode()
        )


def make_sqlite(path: Path, triplet_id: str) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT);
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER, end INTEGER, graphNodeId INTEGER, shortName INTEGER
        );
        CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER);
        """
    )
    connection.execute(
        "INSERT INTO StringIds(id,value) VALUES(?,?)",
        (1, "triton_poi_fused_0"),
    )
    connection.execute(
        "INSERT INTO StringIds(id,value) VALUES(?,?)",
        (2, "triton_poi_fused_1"),
    )
    for index, rung in enumerate(RUNGS):
        start = 100 + index * 200
        end = start + 100
        connection.execute(
            "INSERT INTO NVTX_EVENTS(start,end,text) VALUES(?,?,?)",
            (start, end, f"TIER4_PROOF|rung={rung}|triplet={triplet_id}"),
        )
        for kernel_index in range(2):
            node = 900 + index * 10 + kernel_index
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL"
                "(start,end,graphNodeId,shortName) VALUES(?,?,?,?)",
                (start + 10 + kernel_index, start + 20 + kernel_index, node, kernel_index + 1),
            )
            connection.execute(
                "INSERT INTO CUDA_GRAPH_NODE_EVENTS(graphNodeId) VALUES(?)", (node,)
            )
    connection.commit()
    connection.close()


def canonical_output(point: dict[str, Any], repeat: int, rung: str) -> list[dict[str, Any]]:
    records = []
    for request in range(point["batch"]):
        base = 9000 if rung == "ceiling" else 100 + repeat + request
        records.append(
            {
                "request_index": request,
                "token_ids": [base + token for token in range(point["gen"])],
                "cumulative_logprob_hex": (
                    "nan"
                    if rung == "ceiling"
                    else float(-repeat - request - 0.25).hex()
                ),
            }
        )
    return records


def switch_binding(probe: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"pid": probe["pid"], **probe["active_variant"]}]


def make_raw(
    points: list[dict[str, Any]],
    repeats: int,
    warmups: int,
    bootstrap_samples: int,
    allow_short: bool,
    identity: dict[str, Any],
    identity_hash: str,
    model: Path,
    evidence: Path,
    cohort_id: str,
    proof_point: dict[str, Any],
) -> dict[str, Any]:
    triplet_id = "synthetictriplet"
    probes: dict[str, list[dict[str, Any]]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    for rung_index, rung in enumerate(RUNGS):
        cache = evidence / rung / "cache"
        probe = worker_probe(rung, cache, 100 + rung_index)
        probes[rung] = [probe]
        runtimes[rung] = {
            "schema": "tier4.runtime.v2",
            "rung": rung,
            "rung_index": rung_index,
            "triplet_mode": "same_process_adjacent_latin3",
            "triplet_id": triplet_id,
            "cohort_id": cohort_id,
            "driver_pid": 4242,
            "model_fingerprint": identity["model_fingerprint"],
            "model_identity_manifest_sha256": identity_hash,
            "model_identity_snapshot_sha256": identity_hash,
            "packages": {
                "vllm": "synthetic",
                "torch": "synthetic",
                "triton": "synthetic",
                "transformers": "synthetic",
            },
            "device": {
                "name": "NVIDIA B200",
                "compute_capability": [10, 0],
                "total_memory_bytes": 1,
                "sm_count": 1,
            },
            "cache_fresh": True,
            "cache_root": str(cache.resolve()),
            "graph_mode": "FULL_DECODE_ONLY",
            "executed_full_decode": True,
            "pdl_scope": PDL_SCOPE,
            "prefill_mode": PREFILL_MODE,
            "trtllm_enable_pdl": False,
            "workers": [probe],
            "graph_execution_proof": "pending_nsys_cuda_graph_node",
            "nsys_sqlite": None,
            "nsys_sqlite_sha256": None,
        }

    warmup_records = []
    samples = []
    validations = []
    invocation = 0
    for point_index, point in enumerate(points):
        for warmup in range(warmups):
            order = ORDER_CYCLE[warmup % len(ORDER_CYCLE)]
            invocation += len(RUNGS)
            warmup_records.append(
                {
                    "point_index": point_index,
                    "tag": point["tag"],
                    "warmup": warmup,
                    "epoch": 100_000 + point_index * 10_000 + warmup,
                    "order_pattern": ">".join(order),
                    "rung_sequence": list(order),
                    "off_grid_full_output_match": True,
                    "switch_bindings": {
                        rung: switch_binding(probes[rung][0]) for rung in RUNGS
                    },
                }
            )
        for repeat in range(repeats):
            epoch = 1_000_000 + point_index * 100_000 + repeat
            order = ORDER_CYCLE[repeat % len(ORDER_CYCLE)]
            repeat_outputs = {
                rung: canonical_output(point, repeat, rung) for rung in RUNGS
            }
            for order_index, rung in enumerate(order):
                invocation += 1
                output = repeat_outputs[rung]
                samples.append(
                    {
                        "point_index": point_index,
                        "tag": point["tag"],
                        "batch": point["batch"],
                        "seq": point["seq"],
                        "gen": point["gen"],
                        "scope": point["scope"],
                        "repeat": repeat,
                        "rung_index": RUNGS.index(rung),
                        "rung": rung,
                        "order_index": order_index,
                        "order_pattern": ">".join(order),
                        "epoch": epoch,
                        "prompt_digest": poison_digest(point, epoch),
                        "invocation": invocation,
                        "elapsed_s": 0.01 + RUNGS.index(rung) * 0.001 + repeat * 1e-5,
                        "token_count": point["batch"] * point["gen"],
                        "output": output,
                        "output_digest": digest_json(output),
                        "token_digest": token_digest(output),
                        "nonfinite_logprob": rung == "ceiling",
                        "worker_pids": [4242],
                        "switch_bindings": switch_binding(probes[rung][0]),
                    }
                )
            off_output = repeat_outputs["pdl_off"]
            grid_output = repeat_outputs["pdl_grid"]
            ceiling_output = repeat_outputs["ceiling"]
            validations.append(
                {
                    "point_index": point_index,
                    "tag": point["tag"],
                    "repeat": repeat,
                    "epoch": epoch,
                    "order_pattern": ">".join(order),
                    "rung_sequence": list(order),
                    "off_grid_token_match": token_digest(off_output)
                    == token_digest(grid_output),
                    "off_grid_full_output_match": digest_json(off_output)
                    == digest_json(grid_output),
                    "ceiling_token_differs": token_digest(off_output)
                    != token_digest(ceiling_output),
                    "ceiling_full_output_differs": digest_json(off_output)
                    != digest_json(ceiling_output),
                    "ceiling_nonfinite_logprob": True,
                    "ceiling_wrongness_evidence": True,
                    "ceiling_verified": False,
                    "ceiling_unsafe": True,
                    "ceiling_correctness": "unsafe_not_validated",
                }
            )

    all_points = [*points, proof_point]
    formal_manifest = {
        "cohort_id": cohort_id,
        "model_identity_manifest_sha256": identity_hash,
        "model_identity_snapshot_sha256": identity_hash,
        "formal_root_manifest_sha256": "b" * 64 if not allow_short else None,
        "semantic_rungs": list(RUNGS),
        "proof_rung_order": list(RUNGS),
        "timing_order_cycle": [list(order) for order in ORDER_CYCLE],
        "points": points,
        "proof_point": proof_point,
        "repeats_per_rung": repeats,
        "warmup_triplets_per_point": warmups,
        "bootstrap_samples": bootstrap_samples,
        "max_num_seqs": max(point["batch"] for point in all_points),
        "max_model_len": max(point["seq"] + point["gen"] for point in all_points),
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.82,
        "graph_mode": "FULL_DECODE_ONLY",
        "pdl_scope": PDL_SCOPE,
        "prefill_mode": PREFILL_MODE,
        "trtllm_enable_pdl": False,
        "kv_offloading_size_gib": None,
        "kv_offloading_backend": "native",
        "text_only_limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    proof_records = [
        {
            "rung": rung,
            "elapsed_s_diagnostic": 0.01,
            "output_digest": "a" * 64,
            "nonfinite_logprob": rung == "ceiling",
            "prompt_digest": poison_digest(proof_point, 10_000 + index),
            "worker_probes": probes[rung],
            "switch_bindings": switch_binding(probes[rung][0]),
            "ceiling_verified": False if rung == "ceiling" else None,
            "ceiling_unsafe": True if rung == "ceiling" else None,
        }
        for index, rung in enumerate(RUNGS)
    ]
    runtime_candidate_sha256 = {
        rung: digest_json(runtime) for rung, runtime in runtimes.items()
    }
    return {
        "schema": "tier4.triplet.raw.v2",
        "status": "candidate",
        "admissible": False,
        "triplet_id": triplet_id,
        "triplet_mode": "same_process_adjacent_latin3",
        "cohort_id": cohort_id,
        "driver_pid": 4242,
        "worker_cohort": [4242],
        "model": str(model.resolve()),
        "model_fingerprint": identity["model_fingerprint"],
        "model_identity_manifest_sha256": identity_hash,
        "model_identity_snapshot_sha256": identity_hash,
        "formal_root_manifest_sha256": "b" * 64 if not allow_short else None,
        "packages": runtimes["pdl_off"]["packages"],
        "device": runtimes["pdl_off"]["device"],
        "resolved_graph_mode": "FULL_DECODE_ONLY",
        "pdl_scope": PDL_SCOPE,
        "prefill_mode": PREFILL_MODE,
        "trtllm_enable_pdl": False,
        "points": points,
        "warmups": warmups,
        "repeats": repeats,
        "bootstrap_samples": bootstrap_samples,
        "allow_short": allow_short,
        "proof_records": proof_records,
        "build_probes": probes,
        "warmup_records": warmup_records,
        "samples": samples,
        "validations": validations,
        "invocations": invocation,
        "formal_manifest": formal_manifest,
        "runtimes": runtimes,
        "runtime_candidate_sha256": runtime_candidate_sha256,
    }


def assert_contains(errors: list[str], needle: str) -> None:
    if not any(needle in error for error in errors):
        raise AssertionError(f"expected error containing {needle!r}; got {errors}")


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="tier4_pipeline_test.") as name:
        root = Path(name)
        model, identity_path, identity = fake_model(root)
        identity_hash = sha256_file(identity_path)

        # The local identity detects tokenizer/config/revision/inventory tampering
        # without ever hashing the weight payload itself.
        assert verify_model_identity(identity, model) == []
        tokenizer = model / "tokenizer_config.json"
        original_tokenizer = tokenizer.read_text()
        tokenizer.write_text(json.dumps({"eos_token": "tampered"}))
        assert_contains(verify_model_identity(identity, model), "changed")
        tokenizer.write_text(original_tokenizer)
        assert verify_model_identity(identity, model) == []

        results = root / "diagnostic"
        evidence = results / "evidence"
        for rung in RUNGS:
            make_cache(evidence / rung / "cache", rung)
        sqlite_path = results / "profile.sqlite"
        results.mkdir(parents=True, exist_ok=True)
        make_sqlite(sqlite_path, "synthetictriplet")
        point = {"tag": "decode_smoke", "batch": 1, "seq": 64, "gen": 2, "scope": "decode"}
        raw = make_raw(
            [point], 3, 1, 100, True, identity, identity_hash, model,
            evidence, "diagnostic_smoke", point,
        )
        write_json(results / "model_identity_snapshot.json", identity)
        write_json(results / "raw_triplet.json", raw)
        for rung in RUNGS:
            write_json(evidence / rung / "runtime.json", raw["runtimes"][rung])

        completed = subprocess.run(
            [
                sys.executable,
                str(HERE / "tier4_finalize.py"),
                "--results",
                str(results),
                "--nsys-sqlite",
                str(sqlite_path),
                "--allow-short",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        diagnostic = json.loads((results / "diagnostic_validation.json").read_text())
        assert diagnostic["schema"] == "tier4.diagnostic.validation.v3"
        assert diagnostic["admissible"] is False
        assert not (results / "admission.json").exists()
        diagnostic_analysis = json.loads((results / "analysis.json").read_text())
        diagnostic_point = diagnostic_analysis["points"][0]
        assert "floor_to_ceiling_pct" not in diagnostic_point
        off_median = diagnostic_point["rungs"]["pdl_off"]["median_latency_s"]
        grid_median = diagnostic_point["rungs"]["pdl_grid"]["median_latency_s"]
        ceiling_median = diagnostic_point["rungs"]["ceiling"]["median_latency_s"]
        assert abs(
            diagnostic_point["off_to_ceiling_pct"]
            - 100.0 * (off_median - ceiling_median) / off_median
        ) < 1e-12
        assert abs(
            diagnostic_point["wait_removal_headroom_pct"]
            - 100.0 * (grid_median - ceiling_median) / grid_median
        ) < 1e-12
        evidence_result = validate_evidence(evidence, identity["model_fingerprint"])
        assert evidence_result["status"] == "ok", evidence_result["errors"]

        # Raw tamper cases must fail closed before evidence is consulted.
        reordered = copy.deepcopy(raw)
        reordered["samples"][0], reordered["samples"][1] = (
            reordered["samples"][1], reordered["samples"][0]
        )
        assert validate_raw(reordered, True)
        missing = copy.deepcopy(raw)
        missing["samples"].pop()
        assert_contains(validate_raw(missing, True), "sample count")
        mismatch = copy.deepcopy(raw)
        mismatch["samples"][1]["output"][0]["token_ids"][0] += 1
        mismatch["samples"][1]["output_digest"] = digest_json(
            mismatch["samples"][1]["output"]
        )
        mismatch["samples"][1]["token_digest"] = token_digest(
            mismatch["samples"][1]["output"]
        )
        assert_contains(validate_raw(mismatch, True), "off/grid exact")
        safe_nonfinite = copy.deepcopy(raw)
        safe_nonfinite["samples"][1]["output"][0][
            "cumulative_logprob_hex"
        ] = "nan"
        safe_nonfinite["samples"][1]["output_digest"] = digest_json(
            safe_nonfinite["samples"][1]["output"]
        )
        safe_nonfinite["samples"][1]["nonfinite_logprob"] = True
        assert_contains(
            validate_raw(safe_nonfinite, True),
            "safe rung cumulative logprob is non-finite",
        )
        ceiling_flag_tamper = copy.deepcopy(raw)
        ceiling_flag_tamper["samples"][2]["nonfinite_logprob"] = False
        assert_contains(
            validate_raw(ceiling_flag_tamper, True),
            "nonfinite_logprob flag mismatch",
        )
        active_binding_tamper = copy.deepcopy(raw)
        active_binding_tamper["samples"][0]["switch_bindings"][0][
            "batch_graph_fingerprint"
        ] = "f" * 64
        assert_contains(
            validate_raw(active_binding_tamper, True),
            "active variant binding mismatch",
        )
        runtime_binding_tamper = copy.deepcopy(raw)
        runtime_binding_tamper["runtimes"]["pdl_off"]["driver_pid"] += 1
        assert_contains(
            validate_raw(runtime_binding_tamper, True),
            "runtime candidate hash mismatch",
        )

        # Exact NVTX labels/windows are required; a lookalike label is rejected.
        wrong_label = inspect_nsys_sqlite(
            sqlite_path,
            {"off-compiled": ["triton_poi_fused_0"]},
            "pdl_off",
            "wrong-triplet",
        )
        assert wrong_label["errors"]

        # PTX semantic tampering invalidates otherwise finalized evidence.
        grid_ptx = evidence / "pdl_grid" / "cache" / "triton_poi_fused_0.ptx"
        original_grid = grid_ptx.read_text()
        grid_ptx.write_text(original_grid.replace("    griddepcontrol.wait;\n", ""))
        assert_contains(validate_evidence(evidence)["errors"], "fewer than 2")
        grid_ptx.write_text(original_grid)
        collision_ptx = evidence / "pdl_grid" / "cache" / "triton_poi_fused_1.ptx"
        original_collision = collision_ptx.read_text()
        collision_ptx.write_text(
            original_collision.replace("triton_poi_fused_1", "triton_poi_fused_0")
            .replace("    griddepcontrol.wait;\n", "")
        )
        collision_scan = scan_ptx(evidence / "pdl_grid" / "cache")
        assert "triton_poi_fused_0" in collision_scan["ambiguous_wait_entries"]
        assert "triton_poi_fused_0" not in collision_scan["wait_entries"]
        assert_contains(validate_evidence(evidence)["errors"], "fewer than 2")
        collision_ptx.write_text(original_collision)
        ceiling_ptx = evidence / "ceiling" / "cache" / "triton_poi_fused_0.ptx"
        original_ceiling = ceiling_ptx.read_text()
        ceiling_ptx.write_text(
            original_ceiling.replace(
                "    griddepcontrol.launch_dependents;\n",
                "    griddepcontrol.wait;\n    griddepcontrol.launch_dependents;\n",
            )
        )
        assert_contains(validate_evidence(evidence)["errors"], "expected wait=0")
        ceiling_ptx.write_text(original_ceiling)

        # Formal contract is exact, not merely >= a loose minimum.
        formal_results = root / "formal"
        formal_evidence = formal_results / "evidence"
        formal_points = copy.deepcopy(MATRIX["decode"]["points"])
        formal_raw = make_raw(
            formal_points,
            31,
            3,
            2000,
            False,
            identity,
            identity_hash,
            model,
            formal_evidence,
            MATRIX["decode"]["cohort_id"],
            copy.deepcopy(MATRIX["decode"]["proof_point"]),
        )
        assert validate_raw(formal_raw, False) == []
        warmup_tamper = copy.deepcopy(formal_raw)
        warmup_tamper["warmups"] = 2
        assert_contains(validate_raw(warmup_tamper, False), "formal warmups must equal 3")
        bootstrap_tamper = copy.deepcopy(formal_raw)
        bootstrap_tamper["bootstrap_samples"] = 1999
        assert_contains(
            validate_raw(bootstrap_tamper, False),
            "formal bootstrap_samples must equal 2000",
        )

        # Exercise a complete formal finalization, including the launch-time
        # immutable root contract, source hashes, and resume verification.
        formal_root = root / "formal_root"
        formal_results = formal_root / "cohorts" / "decode"
        formal_evidence = formal_results / "evidence"
        manifest_command = [
            sys.executable,
            str(HERE / "tier4_manifest.py"),
            "--results-root",
            str(formal_root),
            "--model",
            str(model),
        ]
        manifest_created = subprocess.run(
            manifest_command,
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        if manifest_created.returncode != 0:
            raise AssertionError(manifest_created.stdout + manifest_created.stderr)
        formal_identity_path = formal_root / "model_identity.json"
        formal_identity_hash = sha256_file(formal_identity_path)
        root_manifest_path = formal_root / "manifest.json"
        root_manifest = json.loads(root_manifest_path.read_text())
        assert root_manifest["schema"] == "tier4.formal.manifest.v3"
        immutable_manifest_bytes = root_manifest_path.read_bytes()
        immutable_identity_bytes = formal_identity_path.read_bytes()
        launch_hash = sha256_file(root_manifest_path)

        # A valid refresh may update only attempts.json; contract bytes stay fixed.
        manifest_verified = subprocess.run(
            manifest_command,
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert manifest_verified.returncode == 0, manifest_verified.stderr
        assert root_manifest_path.read_bytes() == immutable_manifest_bytes
        assert formal_identity_path.read_bytes() == immutable_identity_bytes

        # KV/model/source contract differences fail without rewriting either
        # immutable root file.
        kv_rejected = subprocess.run(
            [*manifest_command, "--kv-offloading-size", "1"],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert kv_rejected.returncode != 0
        assert root_manifest_path.read_bytes() == immutable_manifest_bytes
        assert formal_identity_path.read_bytes() == immutable_identity_bytes
        tokenizer_bytes = tokenizer.read_bytes()
        tokenizer.write_text(json.dumps({"eos_token": "root-contract-tamper"}))
        model_rejected = subprocess.run(
            manifest_command,
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert model_rejected.returncode != 0
        assert root_manifest_path.read_bytes() == immutable_manifest_bytes
        assert formal_identity_path.read_bytes() == immutable_identity_bytes
        tokenizer.write_bytes(tokenizer_bytes)
        source_tamper = copy.deepcopy(root_manifest)
        source_tamper["source_sha256"][SOURCE_NAMES[0]] = "0" * 64
        write_json(root_manifest_path, source_tamper)
        tampered_manifest_bytes = root_manifest_path.read_bytes()
        source_rejected = subprocess.run(
            manifest_command,
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert source_rejected.returncode != 0
        assert root_manifest_path.read_bytes() == tampered_manifest_bytes
        assert formal_identity_path.read_bytes() == immutable_identity_bytes
        root_manifest_path.write_bytes(immutable_manifest_bytes)
        matrix_tamper = copy.deepcopy(root_manifest)
        matrix_tamper["cohorts"]["decode"]["points"][0]["batch"] = 2
        write_json(root_manifest_path, matrix_tamper)
        tampered_matrix_bytes = root_manifest_path.read_bytes()
        matrix_rejected = subprocess.run(
            manifest_command,
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert matrix_rejected.returncode != 0
        assert root_manifest_path.read_bytes() == tampered_matrix_bytes
        assert formal_identity_path.read_bytes() == immutable_identity_bytes
        root_manifest_path.write_bytes(immutable_manifest_bytes)

        formal_raw = make_raw(
            copy.deepcopy(MATRIX["decode"]["points"]),
            31,
            3,
            2000,
            False,
            identity,
            formal_identity_hash,
            model,
            formal_evidence,
            MATRIX["decode"]["cohort_id"],
            copy.deepcopy(MATRIX["decode"]["proof_point"]),
        )
        formal_raw["formal_root_manifest_sha256"] = launch_hash
        formal_raw["formal_manifest"]["formal_root_manifest_sha256"] = launch_hash
        for rung in RUNGS:
            make_cache(formal_evidence / rung / "cache", rung)
            write_json(
                formal_evidence / rung / "runtime.json",
                formal_raw["runtimes"][rung],
            )
        formal_results.mkdir(parents=True, exist_ok=True)
        write_json(formal_results / "model_identity_snapshot.json", identity)
        write_json(formal_results / "launch_manifest_snapshot.json", root_manifest)
        write_json(formal_results / "raw_triplet.json", formal_raw)
        formal_sqlite = formal_results / "profile.sqlite"
        make_sqlite(formal_sqlite, "synthetictriplet")
        formal_report = formal_root / "profiles" / "decode.nsys-rep"
        formal_report.parent.mkdir(parents=True)
        formal_report.write_bytes(b"synthetic-nsys-report")
        formal_completed = subprocess.run(
            [
                sys.executable,
                str(HERE / "tier4_finalize.py"),
                "--results",
                str(formal_results),
                "--nsys-sqlite",
                str(formal_sqlite),
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        if formal_completed.returncode != 0:
            raise AssertionError(formal_completed.stdout + formal_completed.stderr)
        formal_admission = json.loads(
            (formal_results / "admission.json").read_text()
        )
        assert formal_admission["admissible"] is True
        assert formal_admission["formal_root_manifest_sha256"] == launch_hash
        post_admission_refresh = subprocess.run(
            manifest_command,
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert post_admission_refresh.returncode == 0
        assert root_manifest_path.read_bytes() == immutable_manifest_bytes
        assert formal_identity_path.read_bytes() == immutable_identity_bytes
        formal_analysis = json.loads((formal_results / "analysis.json").read_text())
        assert all(
            "off_to_ceiling_pct" in item
            and "floor_to_ceiling_pct" not in item
            for item in formal_analysis["points"]
        )

        verify_command = [
            sys.executable,
            str(HERE / "tier4_finalize.py"),
            "--results",
            str(formal_results),
            "--verify-admission",
        ]

        def verify_returncode() -> int:
            return subprocess.run(
                verify_command,
                cwd=HERE,
                text=True,
                capture_output=True,
                check=False,
            ).returncode

        assert verify_returncode() == 0

        # Resume-skip is fail-closed across missing files, JSON tampering,
        # profile/report mutation, evidence mutation, and stale failure state.
        sqlite_backup = formal_sqlite.with_suffix(".sqlite.saved")
        formal_sqlite.rename(sqlite_backup)
        assert verify_returncode() != 0
        sqlite_backup.rename(formal_sqlite)

        raw_path = formal_results / "raw_triplet.json"
        raw_bytes = raw_path.read_bytes()
        raw_tamper = json.loads(raw_bytes)
        raw_tamper["triplet_id"] = "json-tamper"
        write_json(raw_path, raw_tamper)
        assert verify_returncode() != 0
        raw_path.write_bytes(raw_bytes)

        launch_path = formal_results / "launch_manifest_snapshot.json"
        launch_bytes = launch_path.read_bytes()
        launch_tamper = json.loads(launch_bytes)
        launch_tamper["model_fingerprint"] = "0" * 64
        write_json(launch_path, launch_tamper)
        assert verify_returncode() != 0
        launch_path.write_bytes(launch_bytes)

        report_bytes = formal_report.read_bytes()
        formal_report.write_bytes(report_bytes + b"tamper")
        assert verify_returncode() != 0
        formal_report.write_bytes(report_bytes)

        evidence_ptx = (
            formal_evidence / "pdl_grid" / "cache" / "triton_poi_fused_0.ptx"
        )
        evidence_ptx_bytes = evidence_ptx.read_bytes()
        evidence_ptx.write_bytes(evidence_ptx_bytes + b"// tamper\n")
        assert verify_returncode() != 0
        evidence_ptx.write_bytes(evidence_ptx_bytes)

        write_json(
            formal_results / "finalize_in_progress.json",
            {
                "schema": "tier4.finalize.in_progress.v1",
                "status": "in_progress",
                "admissible": False,
            },
        )
        assert verify_returncode() != 0
        (formal_results / "finalize_in_progress.json").unlink()

        write_json(
            formal_results / "finalize_failure.json",
            {
                "schema": "tier4.finalize.failure.v2",
                "status": "blocked",
                "admissible": False,
                "errors": ["synthetic stale-admission failure"],
            },
        )
        assert verify_returncode() != 0
        (formal_results / "finalize_failure.json").unlink()
        assert verify_returncode() == 0

        # Driver independently rejects the same two formal downgrades before
        # importing/initializing the GPU engine.
        base = [
            sys.executable,
            str(HERE / "tier4_driver.py"),
            "--model",
            str(model),
            "--model-identity",
            str(formal_identity_path),
            "--formal-root-manifest",
            str(root_manifest_path),
            "--cohort-id",
            "synthetic-driver-contract",
            "--point",
            "decode_smoke:1:64:2:decode",
            "--repeats",
            "31",
        ]
        driver_warmup = subprocess.run(
            [
                *base,
                "--results",
                str(formal_root / "cohorts" / "driver_warmup"),
                "--warmups",
                "2",
                "--bootstrap-samples",
                "2000",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert driver_warmup.returncode != 0
        assert "warmups>=3" in driver_warmup.stderr
        driver_bootstrap = subprocess.run(
            [
                *base,
                "--results",
                str(formal_root / "cohorts" / "driver_bootstrap"),
                "--warmups",
                "3",
                "--bootstrap-samples",
                "1999",
            ],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
        )
        assert driver_bootstrap.returncode != 0
        assert "bootstrap-samples>=2000" in driver_bootstrap.stderr

    print("TIER4_PIPELINE_TEST status=PASS positive=2 tamper_cases=27")


if __name__ == "__main__":
    run()
