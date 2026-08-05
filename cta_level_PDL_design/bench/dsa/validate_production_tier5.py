#!/usr/bin/env python3
"""CPU-only fail-closed validator for production Tier-5 artifacts."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import production_tier5 as harness


SCHEMA = 1
EXIT_RESIDUAL_LIMIT = 4
EXIT_RESIDUAL_POLICY = (
    "previously_observed_allowed_same_pid_start_ticks_and_exact_No_data_"
    "with_proc_missing_only_bounded_per_pid"
)


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing:{path.name}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"unreadable:{path.name}:{exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"not_object:{path.name}")
        return None
    return value


def require(condition: bool, code: str, errors: list[str]) -> None:
    if not condition:
        errors.append(code)


def exact_causal_pair_count(seq: int) -> int:
    """Validator-owned formula; deliberately independent of the harness helper."""
    return seq * (seq + 1) // 2


def exact_chunk_causal_pairs(start: int, count: int) -> int:
    """Validator-owned streamed partition formula."""
    return count * (2 * start + count + 1) // 2


def exact_valid_topk_entries(start: int, count: int, topk: int) -> int:
    """Independent valid-prefix cardinality for causal top-k output."""
    growing_rows = min(count, max(0, topk - start))
    growing_sum = growing_rows * (2 * start + growing_rows + 1) // 2
    return growing_sum + (count - growing_rows) * topk


def validate_correctness_contract(
    manifest: dict[str, Any], errors: list[str]
) -> None:
    contract = manifest.get("correctness_contract")
    require(isinstance(contract, dict), "correctness_contract_missing", errors)
    if not isinstance(contract, dict):
        return
    require(
        contract == harness.correctness_contract(),
        "correctness_contract_drift",
        errors,
    )
    require(
        contract.get("topk_tail") == harness.TOPK_TAIL_CONTRACT,
        "correctness_topk_tail_contract",
        errors,
    )
    require(
        contract.get("validated_pdl_modes") == list(harness.PDL_MODES),
        "correctness_mode_contract",
        errors,
    )
    require(
        contract.get("per_mode_untimed_full_reference") is True,
        "correctness_per_mode_reference_contract",
        errors,
    )
    require(
        contract.get("mode_specific_native_logits_replay") is True,
        "correctness_mode_replay_contract",
        errors,
    )
    require(
        contract.get("manual_logits_reference_reuse")
        == "computed_once_per_chunk_and_shared_read_only_across_off_on",
        "correctness_manual_reuse_contract",
        errors,
    )
    require(
        contract.get("manual_logits_quality_metric")
        == "streamed_fp64_formula_equivalent_to_vllm.utils.deep_gemm.calc_diff",
        "correctness_quality_metric_contract",
        errors,
    )
    require(
        contract.get("manual_logits_quality_reduction")
        == {
            "query_row_batch": harness.LOGITS_QUALITY_QUERY_BATCH,
            "scope": "all_causal_valid_cells",
            "sampling": "NONE",
            "input_mutation": False,
        },
        "correctness_quality_reduction_contract",
        errors,
    )
    require(
        contract.get("attention_reference_row_batch")
        == harness.ATTENTION_REFERENCE_ROW_BATCH,
        "correctness_attention_reference_batch_contract",
        errors,
    )
    require(
        contract.get("attention_reference_sampling") == "NONE",
        "correctness_attention_reference_sampling_contract",
        errors,
    )
    require(
        contract.get("manual_logits_calc_diff_limit_exclusive") == 5e-6,
        "correctness_calc_diff_limit",
        errors,
    )
    require(
        contract.get("manual_logits_row_calc_diff_limit_exclusive") == 1e-3,
        "correctness_row_calc_diff_limit",
        errors,
    )
    authority = contract.get("formula_authority")
    require(isinstance(authority, dict), "correctness_formula_authority", errors)
    if not isinstance(authority, dict):
        return
    require(
        authority.get("path") == str(harness.DEEPGEMM_MQA_HEADER),
        "correctness_formula_authority_path",
        errors,
    )
    require(
        authority.get("expected_sha256")
        == harness.DEEPGEMM_MQA_HEADER_SHA256,
        "correctness_formula_expected_sha",
        errors,
    )
    actual_sha = (
        harness.sha256_file(harness.DEEPGEMM_MQA_HEADER)
        if harness.DEEPGEMM_MQA_HEADER.is_file()
        else None
    )
    require(
        actual_sha == harness.DEEPGEMM_MQA_HEADER_SHA256,
        "installed_deepgemm_formula_sha",
        errors,
    )
    require(
        authority.get("sha256") == actual_sha,
        "correctness_formula_manifest_sha",
        errors,
    )
    require(
        authority.get("weighted_relu_lines") == "357-374",
        "correctness_formula_lines",
        errors,
    )


def validate_long_context_execution(
    manifest: dict[str, Any], errors: list[str]
) -> None:
    execution = manifest.get("long_context_execution")
    require(
        execution == harness.long_context_execution_contract(),
        "long_context_execution_contract_drift",
        errors,
    )
    if not isinstance(execution, dict):
        return
    attention = execution.get("attention_reference")
    quality = execution.get("logits_quality")
    release = execution.get("cuda_cache_release")
    require(
        isinstance(attention, dict)
        and attention.get("row_batch") == harness.ATTENTION_REFERENCE_ROW_BATCH
        and attention.get("sampling") == "NONE",
        "long_context_attention_reference_contract",
        errors,
    )
    require(
        isinstance(quality, dict)
        and quality.get("query_row_batch") == harness.LOGITS_QUALITY_QUERY_BATCH
        and quality.get("reduction_dtype") == "float64"
        and quality.get("sampling") == "NONE",
        "long_context_logits_quality_contract",
        errors,
    )
    require(
        isinstance(release, dict)
        and release.get("cadence_chunks")
        == harness.CUDA_CACHE_RELEASE_CADENCE_CHUNKS
        and release.get("inside_cuda_event_timing") is False,
        "long_context_cache_release_contract",
        errors,
    )


def validate_sources(manifest: dict[str, Any], errors: list[str]) -> None:
    sources = manifest.get("sources")
    require(isinstance(sources, list) and len(sources) >= 6, "sources_missing", errors)
    if not isinstance(sources, list):
        return
    require(
        sources == harness.local_source_manifest(),
        "source_manifest_drift",
        errors,
    )
    for record in sources:
        if not isinstance(record, dict):
            errors.append("source_record_not_object")
            continue
        path = Path(str(record.get("path", "")))
        require(record.get("exists") is True, f"source_missing:{path}", errors)
        if not path.is_file():
            continue
        require(
            record.get("size_bytes") == path.stat().st_size,
            f"source_size_mismatch:{path}",
            errors,
        )
        expected_sha = record.get("sha256")
        if expected_sha is not None:
            require(
                expected_sha == harness.sha256_file(path),
                f"source_sha_mismatch:{path}",
                errors,
            )


def validate_packages(
    manifest: dict[str, Any], mode: str, errors: list[str]
) -> None:
    packages = manifest.get("packages")
    require(isinstance(packages, dict), "packages_missing", errors)
    if not isinstance(packages, dict):
        return
    require(
        packages == harness.package_manifest(hash_binaries=mode == "execute"),
        "package_manifest_drift",
        errors,
    )
    distributions = packages.get("distributions")
    require(isinstance(distributions, list), "distributions_missing", errors)
    if isinstance(distributions, list):
        by_name = {
            item.get("name"): item for item in distributions if isinstance(item, dict)
        }
        for name in (
            "vllm",
            "torch",
            "flashinfer-python",
            "flashinfer-cubin",
            "triton",
            "transformers",
        ):
            record = by_name.get(name)
            require(isinstance(record, dict), f"distribution_absent:{name}", errors)
            if isinstance(record, dict):
                require(record.get("installed") is True, f"not_installed:{name}", errors)
                require(bool(record.get("version")), f"version_missing:{name}", errors)
    artifacts = packages.get("artifacts")
    require(isinstance(artifacts, list) and len(artifacts) >= 5, "binary_artifacts_missing", errors)
    if isinstance(artifacts, list):
        for record in artifacts:
            if not isinstance(record, dict):
                errors.append("binary_record_not_object")
                continue
            require(record.get("exists") is True, f"binary_missing:{record.get('path')}", errors)
    api_sources = packages.get("api_sources")
    require(isinstance(api_sources, list) and len(api_sources) >= 7, "api_sources_missing", errors)
    if isinstance(api_sources, list):
        for record in api_sources:
            if not isinstance(record, dict):
                errors.append("api_source_not_object")
                continue
            path = Path(str(record.get("path", "")))
            require(path.is_file() and record.get("exists") is True, f"api_source_missing:{path}", errors)
            if path.is_file():
                require(record.get("sha256") == harness.sha256_file(path), f"api_source_sha:{path}", errors)


def validate_model_specs(manifest: dict[str, Any], errors: list[str]) -> list[str]:
    models = manifest.get("models")
    require(isinstance(models, list) and bool(models), "models_missing", errors)
    if not isinstance(models, list):
        return []
    keys: list[str] = []
    for value in models:
        if not isinstance(value, dict):
            errors.append("model_spec_not_object")
            continue
        key = value.get("key")
        require(key in harness.MODEL_SPECS, f"unknown_model:{key}", errors)
        if key not in harness.MODEL_SPECS:
            continue
        keys.append(key)
        expected = harness.asdict(harness.MODEL_SPECS[key])
        require(value == expected, f"model_shape_drift:{key}", errors)
    require(len(keys) == len(set(keys)), "duplicate_model", errors)
    return keys


def validate_contract(manifest: dict[str, Any], errors: list[str]) -> None:
    contract = manifest.get("experiment_contract")
    require(isinstance(contract, dict), "experiment_contract_missing", errors)
    if not isinstance(contract, dict):
        return
    require(contract == harness.experiment_contract(), "experiment_contract_drift", errors)
    require(contract.get("tier5_bracket_admitted") is False, "bracket_must_be_partial", errors)
    require(contract.get("headroom_defined") is False, "headroom_must_be_undefined", errors)
    require(contract.get("headroom_pct") is None, "headroom_value_forbidden", errors)
    rungs = contract.get("rungs")
    require(isinstance(rungs, dict), "rungs_missing", errors)
    if isinstance(rungs, dict):
        for name in ("floor", "cta_impl", "ceiling"):
            rung = rungs.get(name)
            require(isinstance(rung, dict), f"rung_missing:{name}", errors)
            if isinstance(rung, dict):
                require(rung.get("status") == "PARTIAL", f"rung_not_partial:{name}", errors)
        require(
            isinstance(rungs.get("cta_impl"), dict)
            and rungs["cta_impl"].get("available") is False,
            "cta_impl_must_be_unavailable",
            errors,
        )
        require(
            isinstance(rungs.get("ceiling"), dict)
            and rungs["ceiling"].get("available") is False,
            "ceiling_must_be_unavailable",
            errors,
        )
    controls = contract.get("pdl_controls")
    require(isinstance(controls, dict), "pdl_controls_missing", errors)
    if isinstance(controls, dict):
        require(
            controls.get("indexer", {}).get("api")
            == "vllm.third_party.deep_gemm.set_pdl/get_pdl",
            "indexer_pdl_api_drift",
            errors,
        )
        require(controls.get("topk", {}).get("status") == "UNAVAILABLE", "topk_pdl_false_claim", errors)
        require("enable_pdl" in str(controls.get("sparse_mla", {}).get("api")), "mla_pdl_api_missing", errors)
    reuse = contract.get("benchmark_reuse")
    require(isinstance(reuse, dict), "benchmark_reuse_missing", errors)
    if isinstance(reuse, dict):
        require(reuse.get("mock_indexer_used") is False, "mock_indexer_used", errors)
        require(reuse.get("replaced_component") == "MockIndexer/fill_random_indices", "mock_indexer_replacement_missing", errors)


def validate_shape_records(
    manifest: dict[str, Any], models: Sequence[str], errors: list[str]
) -> None:
    seqs = manifest.get("seqs")
    require(isinstance(seqs, list) and bool(seqs), "seqs_missing", errors)
    if not isinstance(seqs, list):
        return
    records = manifest.get("shape_records")
    require(isinstance(records, list), "shape_records_missing", errors)
    if not isinstance(records, list):
        return
    by_key = {
        (record.get("model"), record.get("seq")): record
        for record in records
        if isinstance(record, dict)
    }
    require(len(by_key) == len(models) * len(seqs), "shape_record_cardinality", errors)
    chunking = manifest.get("chunking", {})
    require(isinstance(chunking, dict), "chunking_missing", errors)
    if not isinstance(chunking, dict):
        return
    require(
        chunking.get("indexer_workload_geometry")
        == "exact_causal_lower_triangle",
        "chunking_geometry_drift",
        errors,
    )
    require(chunking.get("all_query_rows") is True, "chunking_rows_incomplete", errors)
    require(chunking.get("query_sampling") == "NONE", "chunking_query_sampling", errors)
    require(
        chunking.get("causal_pair_sampling") == "NONE",
        "chunking_causal_pair_sampling",
        errors,
    )
    for legacy_key in ("complete_work", "sampled_proxy"):
        require(
            legacy_key not in chunking,
            f"chunking_legacy_field:{legacy_key}",
            errors,
        )
    moe = manifest.get("moe", {})
    for model in models:
        spec = harness.MODEL_SPECS[model]
        for seq in seqs:
            record = by_key.get((model, seq))
            require(isinstance(record, dict), f"shape_record_missing:{model}:{seq}", errors)
            if not isinstance(record, dict):
                continue
            expected = harness.shape_record(
                spec,
                seq,
                int(chunking.get("max_logits_mb", 0)),
                int(chunking.get("max_query_chunk", 0)),
                int(moe.get("experts", 0)),
                int(moe.get("tokens", 0)),
            )
            require(record == expected, f"shape_formula_mismatch:{model}:{seq}", errors)
            causal_pairs = exact_causal_pair_count(seq)
            query_chunk = record.get("query_chunk_tokens")
            require(
                record.get("indexer_workload_geometry")
                == "exact_causal_lower_triangle",
                f"shape_geometry:{model}:{seq}",
                errors,
            )
            require(record.get("all_query_rows") == seq, f"incomplete_rows:{model}:{seq}", errors)
            require(record.get("query_sampling") == "NONE", f"query_sampling:{model}:{seq}", errors)
            require(
                record.get("indexer_causal_pairs") == causal_pairs,
                f"shape_causal_pairs:{model}:{seq}",
                errors,
            )
            require(
                record.get("indexer_causal_pair_formula") == "S*(S+1)/2",
                f"shape_causal_formula:{model}:{seq}",
                errors,
            )
            require(
                record.get("causal_pair_sampling") == "NONE",
                f"causal_pair_sampling:{model}:{seq}",
                errors,
            )
            require(
                record.get("chunk_causal_pair_formula")
                == "count*(2*start+count+1)/2",
                f"chunk_causal_formula:{model}:{seq}",
                errors,
            )
            if isinstance(query_chunk, int) and query_chunk > 0:
                partition_pairs = sum(
                    exact_chunk_causal_pairs(start, min(query_chunk, seq - start))
                    for start in range(0, seq, query_chunk)
                )
                require(
                    partition_pairs == causal_pairs,
                    f"validator_chunk_partition:{model}:{seq}",
                    errors,
                )
                require(
                    record.get("chunk_causal_pairs_sum") == partition_pairs,
                    f"shape_chunk_pair_sum:{model}:{seq}",
                    errors,
                )
                require(
                    record.get("num_query_chunks") == math.ceil(seq / query_chunk),
                    f"shape_chunk_count:{model}:{seq}",
                    errors,
                )
            else:
                errors.append(f"shape_query_chunk_invalid:{model}:{seq}")
            require(
                record.get("chunk_pair_partition_verified") is True,
                f"shape_chunk_partition_unverified:{model}:{seq}",
                errors,
            )
            require(
                record.get("indexer_causal_fma_flops")
                == 2 * causal_pairs * spec.index_heads * spec.index_head_dim,
                f"shape_causal_flops:{model}:{seq}",
                errors,
            )
            for legacy_key in (
                "full_query_rows",
                "sampled_proxy",
                "full_indexer_fma_flops",
            ):
                require(
                    legacy_key not in record,
                    f"shape_legacy_field:{model}:{seq}:{legacy_key}",
                    errors,
                )
            if seq == 1048576:
                require(record.get("extreme") is True, f"1m_not_extreme:{model}", errors)
                require(
                    record.get("within_official_position_range") is False,
                    f"1m_not_out_of_range:{model}",
                    errors,
                )


def expected_row_map(manifest: dict[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("expected_matrix")
    require(isinstance(rows, list) and bool(rows), "expected_matrix_missing", errors)
    if not isinstance(rows, list):
        return {}
    model_keys = [
        item.get("key") for item in manifest.get("models", []) if isinstance(item, dict)
    ]
    expected = harness.expected_matrix(
        model_keys,
        manifest.get("seqs", []),
        manifest.get("workloads", []),
    )
    require(rows == expected, "expected_matrix_drift", errors)
    by_id = {
        row.get("row_id"): row for row in rows if isinstance(row, dict) and row.get("row_id")
    }
    require(len(by_id) == len(rows), "expected_matrix_duplicate", errors)
    return by_id


def validate_argv(
    root: Path, manifest: dict[str, Any], mode: str, errors: list[str]
) -> None:
    argv = manifest.get("argv")
    require(isinstance(argv, list) and len(argv) >= 3, "argv_missing", errors)
    if not isinstance(argv, list) or len(argv) < 3:
        return
    require(
        argv[0] == str(Path(sys.executable).resolve()),
        "argv_python_drift",
        errors,
    )
    require(
        argv[1] == str(Path(harness.__file__).resolve()),
        "argv_harness_drift",
        errors,
    )
    require(
        manifest.get("argv_sha256")
        == harness.sha256_bytes("\0".join(argv).encode("utf-8")),
        "argv_sha_mismatch",
        errors,
    )
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parsed = harness.parse_args(argv[2:])
    except (SystemExit, Exception) as exc:
        errors.append(f"argv_parse:{type(exc).__name__}:{exc}")
        return
    execution_root = Path(parsed.output_dir).resolve()
    publish_target = Path(parsed.publish_target or parsed.output_dir).resolve()
    require(
        root in {execution_root, publish_target},
        "argv_output_or_publish_root_drift",
        errors,
    )
    if execution_root != publish_target:
        require(
            execution_root.parent == publish_target.parent
            and execution_root.name.startswith(publish_target.name + ".inprogress."),
            "failure_atomic_stage_name_drift",
            errors,
        )
    require(
        manifest.get("publication")
        == {
            "execution_output_dir": str(execution_root),
            "requested_publish_target": str(publish_target),
            "failure_atomic_stage": parsed.publish_target is not None,
            "runner_managed_stage": parsed.runner_managed_stage,
        },
        "publication_manifest_drift",
        errors,
    )
    require(parsed.execute_gpu is (mode == "execute"), "argv_execute_mode_drift", errors)
    expected_fields = {
        "backend": manifest.get("backend"),
        "required_device_substring": manifest.get("required_device_substring"),
        "seqs": manifest.get("seqs"),
        "workloads": manifest.get("workloads"),
        "warmup": manifest.get("warmup"),
        "repeats": manifest.get("repeats"),
        "allow_short": manifest.get("allow_short"),
        "seed": manifest.get("random_seed"),
        "max_logits_mb": manifest.get("chunking", {}).get("max_logits_mb"),
        "max_query_chunk": manifest.get("chunking", {}).get("max_query_chunk"),
        "moe_experts": manifest.get("moe", {}).get("experts"),
        "moe_topk": manifest.get("moe", {}).get("topk"),
        "moe_tokens": manifest.get("moe", {}).get("tokens"),
    }
    for field, expected in expected_fields.items():
        require(getattr(parsed, field) == expected, f"argv_field_drift:{field}", errors)
    manifest_models = [
        item.get("key") for item in manifest.get("models", []) if isinstance(item, dict)
    ]
    require(parsed.models == manifest_models, "argv_field_drift:models", errors)
    require(
        parsed.expected_gpu_uuid == manifest.get("expected_gpu_uuid"),
        "argv_field_drift:expected_gpu_uuid",
        errors,
    )
    require(
        parsed.expected_gpu_index == manifest.get("expected_gpu_index"),
        "argv_field_drift:expected_gpu_index",
        errors,
    )
    fragment = manifest.get("fragment")
    if manifest.get("execution_scope") == "row_fragment":
        require(isinstance(fragment, dict), "argv_fragment_metadata_missing", errors)
        if isinstance(fragment, dict):
            fragment_fields = {
                "fragment_row_id": "row_id",
                "fragment_ordinal": "ordinal",
                "campaign_contract_sha256": "campaign_contract_sha256",
                "campaign_fingerprint_sha256": "campaign_fingerprint_sha256",
                "execution_segment_id": "execution_segment_id",
            }
            for parsed_field, manifest_field in fragment_fields.items():
                require(
                    getattr(parsed, parsed_field) == fragment.get(manifest_field),
                    f"argv_fragment_field_drift:{parsed_field}",
                    errors,
                )
    else:
        require(parsed.fragment_row_id is None, "argv_unexpected_fragment", errors)


def validate_common(root: Path, mode: str, errors: list[str]) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    manifest = load_json(root / "manifest.json", errors)
    if manifest is None:
        return None, {}
    require(manifest.get("schema") == SCHEMA, "manifest_schema", errors)
    require(manifest.get("kind") == "tier5_production_dsa_manifest", "manifest_kind", errors)
    expected_mode = "cpu_dry_run" if mode == "dry-run" else "execute_gpu"
    require(manifest.get("mode") == expected_mode, "manifest_mode", errors)
    require(manifest.get("accepted_timing") == 0, "manifest_accepts_timing", errors)
    require(
        manifest.get("accepted_timing_semantics") == "legacy_CTA_bracket_only",
        "manifest_legacy_timing_semantics",
        errors,
    )
    require(
        manifest.get("accepted_workload_timing") == 0,
        "manifest_premature_workload_acceptance",
        errors,
    )
    require(
        manifest.get("accepted_CTA_bracket") == 0,
        "manifest_CTA_bracket_acceptance",
        errors,
    )
    require(manifest.get("random_weights") is True, "random_weights_missing", errors)
    require(manifest.get("backend") == "flashinfer", "backend_not_flashinfer", errors)
    require(manifest.get("required_device_substring") == "B200", "device_target_not_b200", errors)
    formal_statistics = (
        manifest.get("repeats") == 31
        and manifest.get("allow_short") is False
        and manifest.get("warmup") == 5
        and manifest.get("moe", {}).get("tokens") == 4096
        and manifest.get("chunking", {}).get("max_logits_mb")
        == harness.FORMAL_MAX_LOGITS_MB
        and manifest.get("chunking", {}).get("max_query_chunk")
        == harness.FORMAL_MAX_QUERY_CHUNK
        and manifest.get("random_seed") == 20260805
    )
    require(
        manifest.get("formal_statistics_requested") is formal_statistics,
        "formal_statistics_flag_drift",
        errors,
    )
    validate_argv(root, manifest, mode, errors)
    models = validate_model_specs(manifest, errors)
    validate_contract(manifest, errors)
    validate_long_context_execution(manifest, errors)
    validate_correctness_contract(manifest, errors)
    validate_shape_records(manifest, models, errors)
    validate_sources(manifest, errors)
    validate_packages(manifest, mode, errors)
    api = manifest.get("api_contracts")
    require(
        isinstance(api, list) and len(api) == len(harness.API_CONTRACTS),
        "api_contracts_missing",
        errors,
    )
    if isinstance(api, list):
        require(api == list(harness.API_CONTRACTS), "api_contracts_drift", errors)
        text = json.dumps(api, sort_keys=True)
        for symbol in (
            "SparseAttnIndexer",
            "top_k_per_row_prefill",
            "trtllm_batch_decode_with_kv_cache_mla",
            "concat_and_cache_mla",
            "enable_pdl",
            "fused_topk",
            "fused_experts",
            "set_pdl",
            "get_pdl",
            "fp8_fp4_mqa_logits",
            "calc_diff",
            "weighted ReLU",
        ):
            require(symbol in text, f"api_symbol_missing:{symbol}", errors)
    static_checks = manifest.get("static_api_checks")
    require(static_checks == harness.static_api_checks(), "static_api_checks_drift", errors)
    if isinstance(static_checks, list):
        for check in static_checks:
            require(
                isinstance(check, dict)
                and check.get("status") == "PASS"
                and check.get("missing_tokens") == [],
                f"static_api_check_fail:{check.get('stage') if isinstance(check, dict) else 'malformed'}",
                errors,
            )
    return manifest, expected_row_map(manifest, errors)


def validate_dry_run(root: Path, errors: list[str]) -> None:
    manifest, _ = validate_common(root, "dry-run", errors)
    if manifest is None:
        return
    require(manifest.get("status") == "NOT_EXECUTED", "dry_manifest_status", errors)
    require(manifest.get("measurement_emitted") is False, "dry_measurement_emitted", errors)
    device = manifest.get("device")
    require(isinstance(device, dict) and device.get("query_performed") is False, "dry_device_queried", errors)
    plan = load_json(root / "plan.json", errors)
    terminal = load_json(root / "terminal_status.json", errors)
    if plan is not None:
        require(plan.get("schema") == SCHEMA, "dry_plan_schema", errors)
        require(plan.get("kind") == "tier5_production_dsa_plan", "dry_plan_kind", errors)
        require(plan.get("status") == "NOT_EXECUTED", "dry_plan_status", errors)
        require(plan.get("accepted_timing") == 0, "dry_plan_accepts_timing", errors)
        require(plan.get("accepted_workload_timing") == 0, "dry_plan_workload_timing", errors)
        require(plan.get("accepted_CTA_bracket") == 0, "dry_plan_CTA_bracket", errors)
        require(plan.get("measurement_emitted") is False, "dry_plan_measurement", errors)
        require(
            plan.get("manifest_sha256") == harness.sha256_file(root / "manifest.json"),
            "dry_plan_manifest_sha",
            errors,
        )
        require(
            plan.get("expected_matrix") == manifest.get("expected_matrix"),
            "dry_plan_matrix_drift",
            errors,
        )
        require(
            plan.get("shape_records") == manifest.get("shape_records"),
            "dry_plan_shapes_drift",
            errors,
        )
    if terminal is not None:
        require(terminal.get("status") == "NOT_EXECUTED", "dry_terminal_status", errors)
        require(terminal.get("accepted_timing") == 0, "dry_terminal_accepts", errors)
        require(terminal.get("accepted_workload_timing") == 0, "dry_terminal_workload_timing", errors)
        require(terminal.get("accepted_CTA_bracket") == 0, "dry_terminal_CTA_bracket", errors)
        require(terminal.get("measurement_emitted") is False, "dry_terminal_measurement", errors)
        require(
            terminal.get("manifest_sha256") == harness.sha256_file(root / "manifest.json"),
            "dry_terminal_manifest_sha",
            errors,
        )
        require(
            terminal.get("plan_sha256") == harness.sha256_file(root / "plan.json"),
            "dry_terminal_plan_sha",
            errors,
        )
    for forbidden in ("samples.jsonl", "result.json", "correctness.json"):
        require(not (root / forbidden).exists(), f"dry_timing_artifact:{forbidden}", errors)


def load_samples(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        errors.append("missing:samples.jsonl")
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"sample_json:{line_number}:{exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"sample_not_object:{line_number}")
            continue
        records.append(value)
    return records


def expected_components(workload: str) -> tuple[str, ...]:
    if workload == "operator_chain":
        return ("indexer_topk", "sparse_mla", "chain_total")
    if workload == "single_layer":
        return ("attention_layer_total",)
    if workload == "indexshare_fsss":
        return ("four_layer_fsss_total",)
    if workload == "moe32":
        return ("fused_topk_plus_fused_experts",)
    raise ValueError(workload)


def validate_sample_matrix(
    manifest: dict[str, Any], rows: dict[str, dict[str, Any]], samples: list[dict[str, Any]], errors: list[str]
) -> None:
    repeats = int(manifest.get("repeats", 0))
    require(repeats >= 31 or manifest.get("allow_short") is True, "repeat_contract", errors)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for ordinal, sample in enumerate(samples):
        row_id = sample.get("row_id")
        require(row_id in rows, f"unexpected_sample_row:{row_id}", errors)
        require(sample.get("schema") == SCHEMA, f"sample_schema:{ordinal}", errors)
        value = sample.get("elapsed_ms")
        require(
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0,
            f"invalid_elapsed:{ordinal}",
            errors,
        )
        require(sample.get("timed_validation") is False, f"validation_in_timing:{ordinal}", errors)
        repeat = sample.get("repeat")
        require(isinstance(repeat, int) and not isinstance(repeat, bool), f"repeat_type:{ordinal}", errors)
        require(sample.get("poison_epoch") == int(repeat if isinstance(repeat, int) else -1) + 1, f"poison_epoch:{ordinal}", errors)
        require(sample.get("poison_verified") is True, f"poison_unverified:{ordinal}", errors)
        require(sample.get("pdl_mode") not in {"ceiling", "cta_impl"}, f"fabricated_rung:{ordinal}", errors)
        row = rows.get(row_id)
        if isinstance(row, dict):
            for field in ("model", "seq", "workload"):
                require(
                    sample.get(field) == row.get(field),
                    f"sample_{field}_drift:{ordinal}",
                    errors,
                )
            if row.get("workload") in {"single_layer", "indexshare_fsss"}:
                require(
                    sample.get("indexer_calls_per_invocation") == 1,
                    f"sample_indexer_calls:{ordinal}",
                    errors,
                )
            if row.get("workload") == "moe32":
                require(
                    sample.get("fresh_output_allocation") is True,
                    f"sample_moe_output_not_fresh:{ordinal}",
                    errors,
                )
        key = (str(row_id), str(sample.get("component")), str(sample.get("pdl_mode")))
        grouped.setdefault(key, []).append(sample)
    for row_id, row in rows.items():
        workload = str(row.get("workload"))
        modes = tuple(row.get("pdl_modes", []))
        if modes == tuple(harness.PDL_MODES):
            expected_schedule = harness.paired_timing_schedule(
                repeats, expected_components(workload)
            )
            observed_schedule = [
                sample for sample in samples if sample.get("row_id") == row_id
            ]
            require(
                len(observed_schedule) == len(expected_schedule),
                f"timing_schedule_count:{row_id}",
                errors,
            )
            runtime_device = manifest.get("device", {})
            if manifest.get("execution_scope") == "campaign_aggregate":
                runtime_device = manifest.get("row_runtime_identities", {}).get(
                    row_id, {}
                )
            expected_pid = runtime_device.get("process_pid")
            expected_start = runtime_device.get("process_start_ticks")
            for expected_event, sample in zip(
                expected_schedule, observed_schedule, strict=False
            ):
                event, repeat, component, enabled, pair_order = expected_event
                mode = "on" if enabled else "off"
                require(
                    sample.get("timing_event_ordinal") == event,
                    f"timing_event_ordinal:{row_id}:{event}",
                    errors,
                )
                require(
                    sample.get("timing_pair_ordinal") == event // 2,
                    f"timing_pair_ordinal:{row_id}:{event}",
                    errors,
                )
                require(
                    sample.get("repeat") == repeat
                    and sample.get("component") == component
                    and sample.get("pdl_mode") == mode,
                    f"timing_adjacent_pair_schedule:{row_id}:{event}",
                    errors,
                )
                require(
                    sample.get("pair_order") == pair_order,
                    f"timing_pair_order:{row_id}:{event}",
                    errors,
                )
                require(
                    isinstance(expected_pid, int)
                    and not isinstance(expected_pid, bool)
                    and sample.get("pair_same_process_pid") == expected_pid,
                    f"timing_pair_pid:{row_id}:{event}",
                    errors,
                )
                require(
                    isinstance(expected_start, int)
                    and not isinstance(expected_start, bool)
                    and sample.get("pair_same_process_start_ticks")
                    == expected_start,
                    f"timing_pair_start_ticks:{row_id}:{event}",
                    errors,
                )
        for component in expected_components(workload):
            for mode in modes:
                records = grouped.get((row_id, component, mode), [])
                require(len(records) == repeats, f"sample_count:{row_id}:{component}:{mode}", errors)
                repeat_ids = sorted(int(record.get("repeat", -1)) for record in records)
                require(repeat_ids == list(range(repeats)), f"repeat_ids:{row_id}:{component}:{mode}", errors)
        allowed = {
            (row_id, component, mode)
            for component in expected_components(workload)
            for mode in modes
        }
        extras = [key for key in grouped if key[0] == row_id and key not in allowed]
        require(not extras, f"extra_sample_groups:{row_id}:{extras}", errors)


def validate_attention_mode_correctness(
    *,
    model: str,
    seq: int,
    workload: str,
    row_id: str,
    start: int,
    count: int,
    mode: str,
    record: dict[str, Any],
    errors: list[str],
) -> None:
    """Fail closed unless one attention chunk fully proves one real PDL mode."""
    label = f"{row_id}:{start}:{mode}"
    enabled = mode == "on"
    require(record.get("pdl_mode") == mode, f"correctness_mode_label:{label}", errors)
    require(record.get("status") == "PASS", f"correctness_mode_status:{label}", errors)
    require(
        record.get("manual_reference_shared_read_only") is True,
        f"correctness_manual_reference_reuse:{label}",
        errors,
    )
    require(record.get("actual_indexer_calls") == 1, f"actual_indexer_calls:{label}", errors)
    require(record.get("native_replay_calls") == 1, f"native_replay_calls:{label}", errors)

    control = record.get("control")
    require(isinstance(control, dict), f"correctness_control:{label}", errors)
    if isinstance(control, dict):
        require(control.get("requested") is enabled, f"control_requested:{label}", errors)
        require(
            control.get("deep_gemm_readback") is enabled,
            f"control_indexer_readback:{label}",
            errors,
        )
        require(
            control.get("deep_gemm_readback_after_validation") is enabled,
            f"control_indexer_final_readback:{label}",
            errors,
        )
        require(
            control.get("flashinfer_enable_pdl") is enabled,
            f"control_mla_argument:{label}",
            errors,
        )
        require(control.get("topk_control") is None, f"control_topk_claim:{label}", errors)

    diagnostics = record.get("topk_diagnostics")
    require(isinstance(diagnostics, dict), f"topk_diagnostics:{label}", errors)
    if isinstance(diagnostics, dict):
        require(diagnostics.get("pdl_mode") == mode, f"topk_mode:{label}", errors)
        for field in (
            "invalid_valid_entries",
            "duplicate_entries",
            "score_violations",
            "logits_quality_failure",
            "acceptance_mismatches",
        ):
            require(
                diagnostics.get(field) == 0,
                f"topk_diagnostic_{field}:{label}",
                errors,
            )
        require(
            diagnostics.get("topk_tail_contract") == harness.TOPK_TAIL_CONTRACT,
            f"topk_tail_contract:{label}",
            errors,
        )
        tail_slots = diagnostics.get("tail_slots_ignored")
        tail_nonminus_one = diagnostics.get("tail_nonminus_one_observed")
        require(
            isinstance(tail_slots, int) and tail_slots >= 0,
            f"topk_tail_slots:{label}",
            errors,
        )
        require(
            isinstance(tail_nonminus_one, int)
            and isinstance(tail_slots, int)
            and 0 <= tail_nonminus_one <= tail_slots,
            f"topk_tail_observation:{label}",
            errors,
        )
        require(
            diagnostics.get("score_reference")
            == "deepgemm_full_n_logits_causal_masked",
            f"topk_score_reference:{label}",
            errors,
        )
        require(
            isinstance(diagnostics.get("valid_set_symmetric_difference"), int)
            and diagnostics.get("valid_set_symmetric_difference", -1) >= 0,
            f"topk_set_symmetric_difference:{label}",
            errors,
        )
        require(
            diagnostics.get("exact_set_difference_role")
            == "diagnostic_not_acceptance",
            f"topk_set_diagnostic_role:{label}",
            errors,
        )
        quality = diagnostics.get("indexer_logits_quality")
        require(isinstance(quality, dict), f"indexer_logits_quality:{label}", errors)
        if isinstance(quality, dict):
            require(
                quality.get("actual_source")
                == (
                    "full-N vllm.utils.deep_gemm.fp8_fp4_mqa_logits("
                    "clean_logits=False), then causal-valid-cell mask"
                ),
                f"indexer_logits_source:{label}",
                errors,
            )
            require(
                quality.get("native_replay_pdl_mode") == mode,
                f"indexer_logits_mode:{label}",
                errors,
            )
            require(
                quality.get("native_replay_calls") == 1,
                f"indexer_logits_replay_calls:{label}",
                errors,
            )
            require(
                quality.get("manual_reference")
                == "sum_h(relu(q_fp8_dequant@k_fp8_dequant.T)*weights)",
                f"indexer_logits_manual_formula:{label}",
                errors,
            )
            require(
                quality.get("manual_reference_reuse")
                == "computed_once_per_chunk_and_shared_read_only_across_off_on",
                f"indexer_logits_manual_reuse:{label}",
                errors,
            )
            require(
                quality.get("quality_reduction")
                == "streamed_query_rows_all_causal_valid_cells",
                f"indexer_logits_quality_reduction:{label}",
                errors,
            )
            require(
                quality.get("quality_query_row_batch")
                == harness.LOGITS_QUALITY_QUERY_BATCH,
                f"indexer_logits_quality_batch:{label}",
                errors,
            )
            require(
                quality.get("quality_reduction_dtype") == "float64",
                f"indexer_logits_quality_dtype:{label}",
                errors,
            )
            expected_replay_tokens = seq if workload == "operator_chain" else start + count
            require(
                quality.get("replay_total_seq_lens") == expected_replay_tokens,
                f"indexer_logits_replay_extent:{label}",
                errors,
            )
            require(
                quality.get("invalid_logits_contract") == "UNSPECIFIED_IGNORED",
                f"indexer_logits_invalid_contract:{label}",
                errors,
            )
            require(
                quality.get("valid_elements") == exact_chunk_causal_pairs(start, count),
                f"indexer_logits_valid_scope:{label}",
                errors,
            )
            for field in (
                "kernel_valid_nonfinite",
                "manual_valid_nonfinite",
                "row_quality_failures",
            ):
                require(
                    quality.get(field) == 0,
                    f"indexer_logits_{field}:{label}",
                    errors,
                )
            require(
                quality.get("calc_diff_limit_exclusive")
                == harness.DEEPGEMM_CALC_DIFF_LIMIT,
                f"indexer_logits_calc_limit:{label}",
                errors,
            )
            calc_diff = quality.get("calc_diff")
            require(
                isinstance(calc_diff, (int, float))
                and not isinstance(calc_diff, bool)
                and math.isfinite(calc_diff)
                and 0 <= calc_diff
                and calc_diff < harness.DEEPGEMM_CALC_DIFF_LIMIT,
                f"indexer_logits_calc_diff:{label}",
                errors,
            )
            require(
                quality.get("row_calc_diff_limit_exclusive")
                == harness.DEEPGEMM_ROW_CALC_DIFF_LIMIT,
                f"indexer_logits_row_limit:{label}",
                errors,
            )
            row_calc_diff_max = quality.get("row_calc_diff_max")
            require(
                isinstance(row_calc_diff_max, (int, float))
                and not isinstance(row_calc_diff_max, bool)
                and math.isfinite(row_calc_diff_max)
                and 0 <= row_calc_diff_max
                and row_calc_diff_max < harness.DEEPGEMM_ROW_CALC_DIFF_LIMIT,
                f"indexer_logits_row_diff:{label}",
                errors,
            )
            row_calc_diff_p99 = quality.get("row_calc_diff_p99")
            require(
                isinstance(row_calc_diff_p99, (int, float))
                and not isinstance(row_calc_diff_p99, bool)
                and math.isfinite(row_calc_diff_p99)
                and 0 <= row_calc_diff_p99
                and row_calc_diff_p99 < harness.DEEPGEMM_ROW_CALC_DIFF_LIMIT,
                f"indexer_logits_row_p99:{label}",
                errors,
            )
            for field in (
                "max_abs_diff",
                "mean_abs_diff",
                "rms_abs_diff",
                "manual_rms",
            ):
                numeric = quality.get(field)
                require(
                    isinstance(numeric, (int, float))
                    and not isinstance(numeric, bool)
                    and math.isfinite(numeric)
                    and numeric >= 0,
                    f"indexer_logits_{field}:{label}",
                    errors,
                )
            require(
                quality.get("status") == "PASS",
                f"indexer_logits_quality_status:{label}",
                errors,
            )

    spec = harness.MODEL_SPECS[model]
    expected_valid_topk = exact_valid_topk_entries(start, count, spec.index_topk)
    expected_output_slots = count * spec.index_topk
    require(
        record.get("topk_valid_elements_checked") == expected_valid_topk,
        f"topk_valid_scope:{label}",
        errors,
    )
    require(
        record.get("topk_output_slots_observed") == expected_output_slots,
        f"topk_output_slots:{label}",
        errors,
    )
    require(
        record.get("topk_tail_slots_ignored")
        == expected_output_slots - expected_valid_topk,
        f"topk_tail_scope:{label}",
        errors,
    )
    require(
        record.get("topk_tail_contract") == harness.TOPK_TAIL_CONTRACT,
        f"chunk_topk_tail_contract:{label}",
        errors,
    )
    if isinstance(diagnostics, dict):
        require(
            diagnostics.get("tail_slots_ignored")
            == expected_output_slots - expected_valid_topk,
            f"topk_diagnostic_tail_scope:{label}",
            errors,
        )

    expected_layers = 1 if workload in {"operator_chain", "single_layer"} else 4
    require(
        record.get("sparse_mla_calls") == expected_layers,
        f"sparse_mla_calls:{label}",
        errors,
    )
    require(record.get("topk_mismatches") == 0, f"topk_mismatch:{label}", errors)
    if workload == "operator_chain":
        require(
            record.get("validation_scope")
            == "all_rows_valid_topk_prefix_and_all_attention_elements",
            f"operator_validation_scope:{label}",
            errors,
        )
        require(
            record.get("attention_elements_checked")
            == count * spec.attention_heads * spec.kv_lora_rank,
            f"attention_check_scope:{label}",
            errors,
        )
    else:
        require(
            record.get("validation_scope")
            == "all_rows_valid_topk_prefix_attention_and_layer_outputs",
            f"layer_validation_scope:{label}",
            errors,
        )
        require(record.get("expected_indexer_calls") == 1, f"expected_indexer_calls:{label}", errors)
        expected_pattern = "F" if workload == "single_layer" else "FSSS"
        require(record.get("pattern") == expected_pattern, f"indexshare_pattern:{label}", errors)
        require(record.get("attention_layers") == expected_layers, f"attention_layers:{label}", errors)
        require(
            record.get("attention_elements_checked")
            == expected_layers * count * spec.attention_heads * spec.kv_lora_rank,
            f"layer_attention_scope:{label}",
            errors,
        )
        require(
            record.get("layer_output_elements_checked")
            == expected_layers * count * spec.hidden_size,
            f"layer_output_scope:{label}",
            errors,
        )
        require(
            record.get("output_elements_checked") == count * spec.hidden_size,
            f"layer_final_output_scope:{label}",
            errors,
        )


def validate_correctness(
    manifest: dict[str, Any], rows: dict[str, dict[str, Any]], value: dict[str, Any] | None, errors: list[str]
) -> None:
    if value is None:
        return
    require(value.get("schema") == SCHEMA, "correctness_schema", errors)
    require(value.get("kind") == "tier5_production_correctness", "correctness_kind", errors)
    require(value.get("status") == "PASS", "correctness_status", errors)
    require(value.get("all_expected_rows_present") is True, "correctness_completeness", errors)
    records = value.get("rows")
    require(isinstance(records, list), "correctness_rows_missing", errors)
    if not isinstance(records, list):
        return
    by_id = {record.get("row_id"): record for record in records if isinstance(record, dict)}
    require(set(by_id) == set(rows), "correctness_row_set", errors)
    require(len(by_id) == len(records), "correctness_duplicate_or_malformed_row", errors)
    shape_by_key = {
        (item["model"], item["seq"]): item
        for item in manifest.get("shape_records", [])
        if isinstance(item, dict)
    }
    for row_id, expected in rows.items():
        record = by_id.get(row_id)
        if not isinstance(record, dict):
            continue
        require(record.get("status") == "PASS", f"correctness_fail:{row_id}", errors)
        workload = expected.get("workload")
        if workload == "moe32":
            require(record.get("experts") == 32, f"moe_experts:{row_id}", errors)
            require(record.get("topk") == 8, f"moe_topk:{row_id}", errors)
            require(record.get("tokens_checked") == manifest["moe"]["tokens"], f"moe_tokens:{row_id}", errors)
            require(
                record.get("output_elements_checked")
                == manifest["moe"]["tokens"] * harness.MODEL_SPECS[expected["model"]].hidden_size,
                f"moe_output_scope:{row_id}",
                errors,
            )
            require(
                record.get("routing_assignments_checked")
                == manifest["moe"]["tokens"] * 8,
                f"moe_routing_scope:{row_id}",
                errors,
            )
            continue
        model, seq = expected.get("model"), expected.get("seq")
        shape = shape_by_key.get((model, seq))
        require(isinstance(shape, dict), f"shape_for_correctness:{row_id}", errors)
        if not isinstance(shape, dict):
            continue
        causal_pairs = exact_causal_pair_count(seq)
        require(record.get("all_query_rows_executed") == seq, f"rows_not_executed:{row_id}", errors)
        require(
            record.get("indexer_workload_geometry")
            == "exact_causal_lower_triangle",
            f"correctness_geometry:{row_id}",
            errors,
        )
        require(record.get("query_sampling") == "NONE", f"correctness_query_sampling:{row_id}", errors)
        require(
            record.get("indexer_causal_pairs_executed") == causal_pairs,
            f"correctness_causal_pairs:{row_id}",
            errors,
        )
        require(
            record.get("indexer_causal_pair_formula") == "S*(S+1)/2",
            f"correctness_causal_formula:{row_id}",
            errors,
        )
        require(
            record.get("causal_pair_sampling") == "NONE",
            f"correctness_causal_pair_sampling:{row_id}",
            errors,
        )
        require(
            "sampled_proxy" not in record,
            f"correctness_legacy_proxy:{row_id}",
            errors,
        )
        require(
            record.get("correctness_pdl_modes") == list(harness.PDL_MODES),
            f"correctness_row_mode_set:{row_id}",
            errors,
        )
        require(
            record.get("per_chunk_mode_correctness_complete") is True,
            f"correctness_row_mode_completeness:{row_id}",
            errors,
        )
        chunks = record.get("chunks")
        require(isinstance(chunks, list), f"correctness_chunks:{row_id}", errors)
        if not isinstance(chunks, list):
            continue
        require(
            len(chunks) == shape.get("num_query_chunks"),
            f"correctness_chunk_cardinality:{row_id}",
            errors,
        )
        cursor = 0
        reported_pair_sum = 0
        for chunk in chunks:
            if not isinstance(chunk, dict):
                errors.append(f"correctness_chunk_not_object:{row_id}")
                continue
            start = chunk.get("query_start")
            count = chunk.get("query_count")
            require(start == cursor, f"correctness_chunk_gap:{row_id}:{cursor}", errors)
            require(isinstance(count, int) and count > 0, f"correctness_chunk_count:{row_id}:{cursor}", errors)
            if not isinstance(count, int) or count <= 0:
                continue
            if isinstance(start, int):
                expected_chunk_pairs = exact_chunk_causal_pairs(start, count)
                reported_chunk_pairs = chunk.get(
                    "indexer_causal_pairs_executed"
                )
                require(
                    reported_chunk_pairs == expected_chunk_pairs,
                    f"correctness_chunk_causal_pairs:{row_id}:{start}",
                    errors,
                )
                if isinstance(reported_chunk_pairs, int):
                    reported_pair_sum += reported_chunk_pairs
                require(
                    chunk.get("first_query_causal_key_count") == start + 1,
                    f"correctness_chunk_first_key_count:{row_id}:{start}",
                    errors,
                )
                require(
                    chunk.get("last_query_causal_key_count") == start + count,
                    f"correctness_chunk_last_key_count:{row_id}:{start}",
                    errors,
                )
            require(
                chunk.get("indexer_workload_geometry")
                == "exact_causal_lower_triangle",
                f"correctness_chunk_geometry:{row_id}:{start}",
                errors,
            )
            require(
                chunk.get("query_sampling") == "NONE",
                f"correctness_chunk_query_sampling:{row_id}:{start}",
                errors,
            )
            require(
                chunk.get("causal_pair_sampling") == "NONE",
                f"correctness_chunk_pair_sampling:{row_id}:{start}",
                errors,
            )
            cursor += count
            require(
                count <= shape.get("query_chunk_tokens", 0),
                f"correctness_chunk_oversize:{row_id}:{start}",
                errors,
            )
            manual_reference = chunk.get("manual_indexer_reference")
            require(
                isinstance(manual_reference, dict),
                f"correctness_manual_reference:{row_id}:{start}",
                errors,
            )
            if isinstance(manual_reference, dict):
                require(
                    manual_reference.get("computation_count") == 1,
                    f"correctness_manual_reference_count:{row_id}:{start}",
                    errors,
                )
                require(
                    manual_reference.get("shared_read_only_across_modes") is True,
                    f"correctness_manual_reference_read_only:{row_id}:{start}",
                    errors,
                )
                require(
                    manual_reference.get("modes_using_reference")
                    == list(harness.PDL_MODES),
                    f"correctness_manual_reference_modes:{row_id}:{start}",
                    errors,
                )
            legacy_mode_fields = (
                "topk_diagnostics",
                "topk_mismatches",
                "attention_elements_checked",
                "layer_output_elements_checked",
                "validation_scope",
                "indexer_calls",
            )
            require(
                not any(field in chunk for field in legacy_mode_fields),
                f"correctness_legacy_flat_mode_evidence:{row_id}:{start}",
                errors,
            )
            mode_records = chunk.get("mode_correctness")
            require(
                isinstance(mode_records, list),
                f"correctness_mode_records:{row_id}:{start}",
                errors,
            )
            if isinstance(mode_records, list):
                by_mode = {
                    item.get("pdl_mode"): item
                    for item in mode_records
                    if isinstance(item, dict)
                }
                require(
                    len(mode_records) == len(harness.PDL_MODES)
                    and len(by_mode) == len(mode_records)
                    and set(by_mode) == set(harness.PDL_MODES),
                    f"correctness_mode_set:{row_id}:{start}",
                    errors,
                )
                if isinstance(start, int):
                    for mode in harness.PDL_MODES:
                        mode_record = by_mode.get(mode)
                        if not isinstance(mode_record, dict):
                            continue
                        validate_attention_mode_correctness(
                            model=model,
                            seq=seq,
                            workload=workload,
                            row_id=row_id,
                            start=start,
                            count=count,
                            mode=mode,
                            record=mode_record,
                            errors=errors,
                        )
        require(cursor == seq, f"correctness_not_full_seq:{row_id}:{cursor}:{seq}", errors)
        require(
            reported_pair_sum == causal_pairs,
            f"correctness_chunk_pair_total:{row_id}",
            errors,
        )
        require(
            record.get("chunk_causal_pairs_sum") == causal_pairs,
            f"correctness_row_chunk_pair_sum:{row_id}",
            errors,
        )
        require(
            record.get("chunk_pair_partition_verified") is True,
            f"correctness_chunk_partition_unverified:{row_id}",
            errors,
        )


def validate_gpu_exclusivity(
    root: Path,
    manifest: dict[str, Any],
    evidence: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any] | None:
    initial_error_count = len(errors)
    if not isinstance(evidence, dict):
        errors.append("exclusivity_evidence_missing")
        return None
    expected_uuid = harness.canonical_gpu_uuid(evidence.get("expected_gpu_uuid", ""))
    expected_index = evidence.get("expected_gpu_index")
    require(expected_uuid is not None, "exclusivity_expected_uuid_invalid", errors)
    require(
        isinstance(expected_index, int)
        and not isinstance(expected_index, bool)
        and expected_index >= 0,
        "exclusivity_expected_index_invalid",
        errors,
    )
    if expected_uuid is None:
        return None
    fragment = manifest.get("fragment")
    if manifest.get("execution_scope") == "row_fragment" and isinstance(
        fragment, dict
    ):
        phase_base = (
            "production_tier5_fragment_"
            f"{fragment.get('ordinal')}_{fragment.get('row_id')}_"
            f"{fragment.get('execution_segment_id')}"
        )
        expected_phases = {
            "identity": phase_base + "_identity",
            "lease": phase_base + "_acquire",
            "pre": phase_base + "_pre",
            "post": phase_base + "_post",
            "monitor": phase_base + "_monitor",
        }
    else:
        expected_phases = {
            "identity": "production_global_lock_identity",
            "lease": "production_campaign_acquire",
            "pre": "production_tier5_pre",
            "post": "production_tier5_post",
            "monitor": "production_tier5",
        }
    require(
        manifest.get("expected_gpu_uuid") == expected_uuid,
        "manifest_expected_uuid_drift",
        errors,
    )
    require(
        manifest.get("expected_gpu_index") == expected_index,
        "manifest_expected_index_drift",
        errors,
    )
    device = manifest.get("device", {})
    require(
        harness.canonical_gpu_uuid(device.get("uuid", "")) == expected_uuid,
        "runtime_uuid_drift",
        errors,
    )
    require(device.get("runtime_ordinal") == 0, "runtime_ordinal_drift", errors)
    require(device.get("runtime_ordinal_zero") is True, "runtime_ordinal_zero_missing", errors)

    expected_lock_path = f"/tmp/cta_pdl_gpu_{expected_uuid}.lock"
    expected_key_sha = harness.sha256_bytes(expected_uuid.encode("utf-8"))
    expected_path_sha = harness.sha256_bytes(expected_lock_path.encode("utf-8"))
    require(
        evidence.get("global_lock_scope") == "target_uuid",
        "global_lock_scope_drift",
        errors,
    )
    require(
        evidence.get("global_lock_key_sha256") == expected_key_sha,
        "global_lock_key_drift",
        errors,
    )
    require(
        evidence.get("global_lock_path_sha256") == expected_path_sha,
        "global_lock_path_drift",
        errors,
    )
    interval_ms = evidence.get("monitor_interval_ms")
    query_timeout_ms = evidence.get("query_timeout_ms")
    require(
        isinstance(interval_ms, int) and 10 <= interval_ms <= 100,
        "monitor_interval_invalid",
        errors,
    )
    require(
        isinstance(query_timeout_ms, int) and 100 <= query_timeout_ms <= 5000,
        "query_timeout_invalid",
        errors,
    )

    paths = {
        "identity": root / "gpu_identity.json",
        "lease": root / "gpu_exclusivity_lease.json",
        "pre": root / "gpu_pre.json",
        "post": root / "gpu_post.json",
        "monitor": root / "gpu_monitor.json",
        "observations": root / "gpu_observations.ndjson",
    }
    identity = load_json(paths["identity"], errors)
    lease = load_json(paths["lease"], errors)
    pre = load_json(paths["pre"], errors)
    post = load_json(paths["post"], errors)
    monitor = load_json(paths["monitor"], errors)
    target: dict[str, Any] | None = None
    if identity is not None:
        require(identity.get("kind") == "gpu_identity", "identity_kind", errors)
        require(identity.get("status") == "PASS" and identity.get("errors") == [], "identity_not_clean", errors)
        require(identity.get("phase") == expected_phases["identity"], "identity_phase_drift", errors)
        target_value = identity.get("target_gpu")
        require(isinstance(target_value, dict), "identity_target_missing", errors)
        if isinstance(target_value, dict):
            target = target_value
            target_index = target.get("index")
            require(
                isinstance(target_index, int) and target_index >= 0,
                "identity_gpu_index_invalid",
                errors,
            )
            require(
                target_index == expected_index,
                "identity_gpu_index_drift",
                errors,
            )
            require(
                harness.canonical_gpu_uuid(target.get("uuid", "")) == expected_uuid,
                "identity_uuid_drift",
                errors,
            )
            visible_selector = manifest.get("environment", {}).get(
                "CUDA_VISIBLE_DEVICES"
            )
            require(
                visible_selector == str(target_index),
                "manifest_visible_index_drift",
                errors,
            )
            require(
                device.get("cuda_visible_devices_selector")
                == visible_selector,
                "runtime_visible_index_drift",
                errors,
            )

    lease_id: str | None = None
    if lease is not None:
        require(lease.get("kind") == "gpu_exclusivity_lease", "lease_kind", errors)
        require(lease.get("status") == "PASS" and lease.get("errors") == [], "lease_not_clean", errors)
        require(lease.get("phase") == expected_phases["lease"], "lease_phase_drift", errors)
        lease_id = lease.get("lease_id")
        require(isinstance(lease_id, str) and bool(lease_id), "lease_id_missing", errors)
        observation = lease.get("observation", {})
        require(observation.get("target_gpu") == target, "lease_target_drift", errors)
        require(observation.get("target_compute_processes") == [], "lease_not_idle", errors)

    for label, checkpoint in (("pre", pre), ("post", post)):
        if checkpoint is None:
            continue
        require(
            checkpoint.get("kind") == "gpu_exclusivity_checkpoint",
            f"{label}_checkpoint_kind",
            errors,
        )
        require(
            checkpoint.get("status") == "PASS" and checkpoint.get("errors") == [],
            f"{label}_checkpoint_not_clean",
            errors,
        )
        require(checkpoint.get("lease_id") == lease_id, f"{label}_lease_id_drift", errors)
        require(
            checkpoint.get("phase") == expected_phases[label],
            f"{label}_phase_drift",
            errors,
        )
        observation = checkpoint.get("observation", {})
        require(observation.get("target_gpu") == target, f"{label}_target_drift", errors)
        require(
            observation.get("target_compute_processes") == [],
            f"{label}_not_idle",
            errors,
        )

    observation_records: list[dict[str, Any]] = []
    allowed_observation_count = 0
    residual_observation_count = 0
    observed_allowed: set[tuple[int, int]] = set()
    residual_counts: dict[int, int] = {}
    residual_starts: dict[int, int] = {}
    previous_query_finished: int | None = None
    target_empty_by_observation: list[bool] = []
    try:
        observation_lines = paths["observations"].read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"unreadable:gpu_observations.ndjson:{exc}")
        observation_lines = []
    require(bool(observation_lines), "monitor_observations_empty", errors)
    for sequence, line in enumerate(observation_lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"monitor_observation_json:{sequence}:{exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"monitor_observation_not_object:{sequence}")
            continue
        observation_records.append(record)
        require(record.get("schema") == SCHEMA, f"monitor_observation_schema:{sequence}", errors)
        require(record.get("sequence") == sequence, f"monitor_observation_sequence:{sequence}", errors)
        require(record.get("query_errors") == [], f"monitor_query_error:{sequence}", errors)
        require(record.get("foreign_target_processes") == [], f"monitor_foreign:{sequence}", errors)
        observation = record.get("observation", {})
        require(
            observation.get("target_gpu") == target,
            f"monitor_target_drift:{sequence}",
            errors,
        )
        target_processes = observation.get("target_compute_processes")
        allowed_processes = record.get("allowed_target_processes")
        residual_processes = record.get("allowed_exit_residual_processes")
        require(
            isinstance(target_processes, list),
            f"monitor_target_process_list:{sequence}",
            errors,
        )
        require(
            isinstance(allowed_processes, list),
            f"monitor_allowed_process_list:{sequence}",
            errors,
        )
        require(
            isinstance(residual_processes, list),
            f"monitor_residual_process_list:{sequence}",
            errors,
        )
        if not all(
            isinstance(value, list)
            for value in (target_processes, allowed_processes, residual_processes)
        ):
            continue
        query_started = record.get("query_started_monotonic_ns")
        query_finished = record.get("query_finished_monotonic_ns")
        query_duration = record.get("query_duration_ms")
        require(
            isinstance(query_started, int)
            and isinstance(query_finished, int)
            and query_finished >= query_started,
            f"monitor_query_interval:{sequence}",
            errors,
        )
        require(
            isinstance(query_duration, (int, float))
            and not isinstance(query_duration, bool)
            and math.isfinite(query_duration)
            and query_duration >= 0,
            f"monitor_query_duration:{sequence}",
            errors,
        )
        if previous_query_finished is not None and isinstance(query_started, int):
            require(
                query_started >= previous_query_finished,
                f"monitor_query_overlap:{sequence}",
                errors,
            )
        if isinstance(query_finished, int):
            previous_query_finished = query_finished

        target_by_pid = {
            process.get("pid"): process
            for process in target_processes
            if isinstance(process, dict) and isinstance(process.get("pid"), int)
        }
        allowed_by_pid = {
            process.get("pid"): process
            for process in allowed_processes
            if isinstance(process, dict) and isinstance(process.get("pid"), int)
        }
        residual_by_pid = {
            process.get("pid"): process
            for process in residual_processes
            if isinstance(process, dict) and isinstance(process.get("pid"), int)
        }
        require(
            len(target_by_pid) == len(target_processes)
            and len(allowed_by_pid) == len(allowed_processes)
            and len(residual_by_pid) == len(residual_processes)
            and not (set(allowed_by_pid) & set(residual_by_pid))
            and set(target_by_pid)
            == set(allowed_by_pid) | set(residual_by_pid)
            and len(target_processes)
            == len(allowed_processes) + len(residual_processes),
            f"monitor_process_partition:{sequence}",
            errors,
        )
        require(
            all(
                isinstance(process, dict)
                and harness.canonical_gpu_uuid(process.get("gpu_uuid", ""))
                == expected_uuid
                for process in target_processes
            ),
            f"monitor_process_gpu_uuid:{sequence}",
            errors,
        )
        if sequence == 0:
            require(
                not target_processes
                and not allowed_processes
                and not residual_processes,
                "monitor_start_gate_not_idle",
                errors,
            )

        for process in allowed_processes:
            if not isinstance(process, dict):
                continue
            pid = process.get("pid")
            start_ticks = process.get("proc_start_ticks")
            require(
                isinstance(pid, int)
                and pid > 0
                and isinstance(start_ticks, int)
                and start_ticks >= 0,
                f"monitor_allowed_identity:{sequence}",
                errors,
            )
            require(
                not isinstance(pid, int) or pid not in residual_counts,
                f"monitor_retired_pid_reappeared:{sequence}:{pid}",
                errors,
            )
            target_process = target_by_pid.get(pid)
            require(
                isinstance(target_process, dict)
                and all(
                    process.get(key) == target_process.get(key)
                    for key in ("pid", "gpu_uuid", "name", "used_memory")
                ),
                f"monitor_allowed_raw_identity:{sequence}:{pid}",
                errors,
            )
            if isinstance(pid, int) and isinstance(start_ticks, int):
                observed_allowed.add((pid, start_ticks))

        for process in residual_processes:
            if not isinstance(process, dict):
                continue
            pid = process.get("pid")
            previous_start = process.get("previous_allowed_start_ticks")
            residual_number = process.get("residual_observation_number")
            target_process = target_by_pid.get(pid)
            safe_residual = (
                isinstance(pid, int)
                and pid > 0
                and isinstance(previous_start, int)
                and previous_start >= 0
                and process.get("proc_start_ticks") is None
                and process.get("name") == "[No data]"
                and process.get("classification")
                == "allowed_post_exit_nvidia_smi_residual"
                and process.get("residual_observation_limit")
                == EXIT_RESIDUAL_LIMIT
                and isinstance(residual_number, int)
                and 1 <= residual_number <= EXIT_RESIDUAL_LIMIT
                and (pid, previous_start) in observed_allowed
                and (
                    pid not in residual_starts
                    or residual_starts[pid] == previous_start
                )
                and residual_number == residual_counts.get(pid, 0) + 1
                and isinstance(target_process, dict)
                and all(
                    process.get(key) == target_process.get(key)
                    for key in ("pid", "gpu_uuid", "name", "used_memory")
                )
            )
            require(
                safe_residual,
                f"monitor_unsafe_exit_residual:{sequence}:{pid}",
                errors,
            )
            if safe_residual:
                residual_starts[pid] = previous_start
                residual_counts[pid] = residual_number

        if allowed_processes:
            allowed_observation_count += 1
        if residual_processes:
            residual_observation_count += 1
        target_empty_by_observation.append(not target_processes)

    if monitor is not None:
        require(monitor.get("schema") == SCHEMA, "monitor_schema", errors)
        require(monitor.get("kind") == "gpu_exclusivity_monitor", "monitor_kind", errors)
        require(monitor.get("status") == "PASS" and monitor.get("errors") == [], "monitor_not_clean", errors)
        require(monitor.get("accepted_timing") == 0, "monitor_accepts_timing", errors)
        require(
            monitor.get("measurement_role") == "gpu_exclusivity_monitor_only",
            "monitor_measurement_role",
            errors,
        )
        require(monitor.get("lease_id") == lease_id, "monitor_lease_id_drift", errors)
        require(monitor.get("target_gpu") == target, "monitor_target_drift", errors)
        require(monitor.get("phase") == expected_phases["monitor"], "monitor_phase_drift", errors)
        runtime_device = manifest.get("device", {})
        runtime_pid = runtime_device.get("process_pid")
        runtime_start = runtime_device.get("process_start_ticks")
        require(
            isinstance(runtime_pid, int)
            and not isinstance(runtime_pid, bool)
            and monitor.get("watch_pid") == runtime_pid,
            "monitor_watch_pid_runtime_drift",
            errors,
        )
        require(
            isinstance(runtime_start, int)
            and not isinstance(runtime_start, bool)
            and monitor.get("watch_root_start_ticks") == runtime_start,
            "monitor_watch_start_runtime_drift",
            errors,
        )
        allowed_identities = {
            (item.get("pid"), item.get("proc_start_ticks"))
            for item in monitor.get("allowed_processes", [])
            if isinstance(item, dict)
        }
        require(
            (runtime_pid, runtime_start) in allowed_identities,
            "monitor_runtime_identity_not_allowed",
            errors,
        )
        require(monitor.get("poll_interval_ms") == interval_ms, "monitor_interval_drift", errors)
        require(monitor.get("query_timeout_ms") == query_timeout_ms, "monitor_timeout_drift", errors)
        require(monitor.get("start_barrier_complete") is True, "monitor_start_barrier_missing", errors)
        require(monitor.get("ready_record_written") is True, "monitor_ready_missing", errors)
        require(monitor.get("baseline_observation_sequence") == 0, "monitor_baseline_drift", errors)
        require(
            monitor.get("coverage_model")
            == "bounded_interval_nvidia_smi_process_sampling",
            "monitor_coverage_model",
            errors,
        )
        require(
            monitor.get("coverage_limit")
            == "foreign GPU processes wholly between completed samples may not be observed",
            "monitor_coverage_limit",
            errors,
        )
        require(monitor.get("require_allowed_process") is True, "monitor_allowed_process_not_required", errors)
        require(monitor.get("allowed_observation_count", 0) > 0, "monitor_allowed_process_unseen", errors)
        require(bool(monitor.get("allowed_processes")), "monitor_allowed_pid_missing", errors)
        require(
            monitor.get("allowed_observation_count") == allowed_observation_count,
            "monitor_allowed_observation_count_drift",
            errors,
        )
        require(
            monitor.get("allowed_exit_residual_observation_count")
            == residual_observation_count,
            "monitor_residual_observation_count_drift",
            errors,
        )
        require(
            monitor.get("allowed_exit_residual_max_observations_per_pid")
            == EXIT_RESIDUAL_LIMIT,
            "monitor_residual_limit_drift",
            errors,
        )
        require(
            monitor.get("exit_residual_policy") == EXIT_RESIDUAL_POLICY,
            "monitor_residual_policy_drift",
            errors,
        )
        manifest_allowed = monitor.get("allowed_processes")
        manifest_allowed_set = {
            (process.get("pid"), process.get("proc_start_ticks"))
            for process in manifest_allowed
            if isinstance(process, dict)
        } if isinstance(manifest_allowed, list) else set()
        require(
            isinstance(manifest_allowed, list)
            and len(manifest_allowed_set) == len(manifest_allowed)
            and manifest_allowed_set == observed_allowed,
            "monitor_allowed_identity_set_drift",
            errors,
        )
        expected_residual_manifest = [
            {
                "pid": pid,
                "previous_allowed_start_ticks": residual_starts[pid],
                "residual_observation_count": residual_counts[pid],
            }
            for pid in sorted(residual_counts)
        ]
        require(
            monitor.get("allowed_exit_residual_processes")
            == expected_residual_manifest,
            "monitor_residual_identity_count_drift",
            errors,
        )
        require(
            monitor.get("max_allowed_exit_residual_observations_observed")
            == max(residual_counts.values(), default=0),
            "monitor_residual_max_count_drift",
            errors,
        )
        require(monitor.get("foreign_processes_detected") is False, "monitor_foreign_flag", errors)
        require(monitor.get("query_failure_detected") is False, "monitor_query_failure_flag", errors)
        require(monitor.get("terminated_on_failure") is False, "monitor_terminated", errors)
        require(
            monitor.get("observation_count") == len(observation_records),
            "monitor_observation_count_drift",
            errors,
        )
        if paths["observations"].is_file():
            require(
                monitor.get("observations_sha256")
                == harness.sha256_file(paths["observations"]),
                "monitor_observations_sha",
                errors,
            )
        reported_observations = Path(str(monitor.get("observations_path", "")))
        actual_observations = paths["observations"].resolve()
        reported_parent = reported_observations.parent
        require(
            reported_observations.name == actual_observations.name
            and (
                reported_observations.resolve() == actual_observations
                or (
                    reported_parent.parent.resolve() == root.parent.resolve()
                    and reported_parent.name.startswith(root.name + ".inprogress.")
                )
            ),
            "monitor_observations_path_drift",
            errors,
        )

    require(
        len(target_empty_by_observation) >= 2
        and target_empty_by_observation[-2:] == [True, True],
        "monitor_final_drain_not_empty",
        errors,
    )

    if not all(path.is_file() for path in paths.values()):
        return None
    return {
        "schema": SCHEMA,
        "kind": "production_gpu_exclusivity_evidence",
        "status": "PASS" if len(errors) == initial_error_count else "FAIL",
        "gpu_uuid": expected_uuid,
        "lease_id": lease_id,
        "global_lock_scope": "target_uuid",
        "global_lock_key_sha256": expected_key_sha,
        "global_lock_path_sha256": expected_path_sha,
        "monitor_interval_ms": interval_ms,
        "query_timeout_ms": query_timeout_ms,
        "artifacts": {
            name: {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": harness.sha256_file(path),
            }
            for name, path in paths.items()
        },
    }


def validate_execute(
    root: Path, errors: list[str], evidence: dict[str, Any] | None
) -> dict[str, Any] | None:
    manifest, rows = validate_common(root, "execute", errors)
    if manifest is None:
        return None
    require(manifest.get("status") == "RUNNING", "execute_manifest_status", errors)
    require(manifest.get("measurement_emitted") is False, "manifest_premature_measurement", errors)
    device = manifest.get("device")
    require(isinstance(device, dict) and device.get("query_performed") is True, "device_missing", errors)
    if isinstance(device, dict):
        cc = str(device.get("compute_capability", ""))
        require(cc.startswith("10."), "device_not_sm10x", errors)
        require("B200" in str(device.get("name", "")), "device_not_b200", errors)

    evidence_summary = validate_gpu_exclusivity(root, manifest, evidence, errors)

    validation_rows = rows
    if manifest.get("execution_scope") == "row_fragment":
        fragment = manifest.get("fragment")
        require(isinstance(fragment, dict), "execute_fragment_metadata", errors)
        if isinstance(fragment, dict):
            row_id = fragment.get("row_id")
            ordinal = fragment.get("ordinal")
            matrix = manifest.get("expected_matrix", [])
            valid_ordinal = (
                isinstance(ordinal, int)
                and not isinstance(ordinal, bool)
                and 0 <= ordinal < len(matrix)
            )
            require(valid_ordinal, "execute_fragment_ordinal", errors)
            require(
                valid_ordinal
                and matrix[ordinal].get("row_id") == row_id
                and fragment.get("row") == matrix[ordinal],
                "execute_fragment_row_binding",
                errors,
            )
            require(
                fragment.get("expected_row_count") == len(matrix),
                "execute_fragment_expected_count",
                errors,
            )
            validation_rows = {row_id: rows[row_id]} if row_id in rows else {}
            require(bool(validation_rows), "execute_fragment_unknown_row", errors)

    result = load_json(root / "result.json", errors)
    correctness = load_json(root / "correctness.json", errors)
    terminal = load_json(root / "terminal_status.json", errors)
    samples = load_samples(root / "samples.jsonl", errors)
    validate_sample_matrix(manifest, validation_rows, samples, errors)
    validate_correctness(manifest, validation_rows, correctness, errors)
    if result is not None:
        require(result.get("status") == "CANDIDATE", "result_status", errors)
        require(result.get("production_timing_candidate") is True, "production_candidate_missing", errors)
        require(result.get("accepted_timing") == 0, "result_accepts_timing", errors)
        require(result.get("accepted_workload_timing") == 0, "result_premature_workload_acceptance", errors)
        require(result.get("accepted_CTA_bracket") == 0, "result_CTA_bracket_acceptance", errors)
        require(result.get("measurement_emitted") is True, "result_no_measurement", errors)
        require(result.get("formal_bracket_status") == "PARTIAL", "result_not_partial", errors)
        require(result.get("tier5_bracket_admitted") is False, "result_bracket_admitted", errors)
        require(result.get("headroom_defined") is False, "result_headroom_defined", errors)
        require(result.get("headroom_pct") is None, "result_headroom_value", errors)
        require(result.get("sample_count") == len(samples), "result_sample_count", errors)
        require(result.get("manifest_sha256") == harness.sha256_file(root / "manifest.json"), "result_manifest_sha", errors)
        require(result.get("samples_sha256") == harness.sha256_file(root / "samples.jsonl"), "result_samples_sha", errors)
        require(result.get("correctness_sha256") == harness.sha256_file(root / "correctness.json"), "result_correctness_sha", errors)
        summaries = result.get("summaries")
        require(isinstance(summaries, list) and bool(summaries), "summaries_missing", errors)
        if isinstance(summaries, list):
            require(
                summaries == harness.summarize_samples(samples, int(manifest.get("random_seed", 0))),
                "summary_recompute_mismatch",
                errors,
            )
            for summary in summaries:
                if not isinstance(summary, dict):
                    errors.append("summary_not_object")
                    continue
                require(summary.get("sample_count") == manifest.get("repeats"), f"summary_count:{summary.get('row_id')}", errors)
                require("headroom_pct" not in summary, f"summary_headroom_forbidden:{summary.get('row_id')}", errors)
    if terminal is not None:
        require(terminal.get("status") == "CANDIDATE", "execute_terminal_status", errors)
        require(terminal.get("accepted_timing") == 0, "execute_terminal_accepts", errors)
        require(terminal.get("accepted_workload_timing") == 0, "execute_terminal_workload_timing", errors)
        require(terminal.get("accepted_CTA_bracket") == 0, "execute_terminal_CTA_bracket", errors)
        require(terminal.get("measurement_emitted") is True, "execute_terminal_measurement", errors)
        require(
            terminal.get("result_sha256") == harness.sha256_file(root / "result.json"),
            "execute_terminal_result_sha",
            errors,
        )
    return evidence_summary


def validate(
    root: Path, mode: str, evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    evidence_summary: dict[str, Any] | None = None
    formal_workload = False
    try:
        if (root / "failure.json").exists():
            errors.append("harness_failure_present")
        if mode == "dry-run":
            validate_dry_run(root, errors)
        else:
            evidence_summary = validate_execute(root, errors, evidence)
            if not errors:
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                formal_workload = (
                    manifest.get("formal_statistics_requested") is True
                    and manifest.get("execution_scope") != "row_fragment"
                )
    except Exception as exc:
        errors.append(f"validator_exception:{type(exc).__name__}:{exc}")
    accepted_workload_timing = int(
        mode == "execute" and not errors and formal_workload
    )
    return {
        "schema": SCHEMA,
        "kind": "tier5_production_validation",
        "mode": mode,
        "status": "PASS" if not errors else "FAIL",
        "accepted_timing": 0,
        "accepted_timing_semantics": "legacy_CTA_bracket_only",
        "accepted_workload_timing": accepted_workload_timing,
        "accepted_CTA_bracket": 0,
        "formal_workload_statistics": formal_workload,
        "tier5_bracket_admitted": False,
        "formal_bracket_status": "PARTIAL",
        "gpu_exclusivity": evidence_summary,
        "errors": errors,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--mode", choices=["dry-run", "execute"], required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--marker")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--expected-gpu-index", type=int)
    parser.add_argument("--global-lock-key-sha256")
    parser.add_argument("--global-lock-path-sha256")
    parser.add_argument("--monitor-interval-ms", type=int)
    parser.add_argument("--query-timeout-ms", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    evidence = None
    if args.mode == "execute":
        evidence = {
            "expected_gpu_uuid": args.expected_gpu_uuid,
            "expected_gpu_index": args.expected_gpu_index,
            "global_lock_scope": "target_uuid",
            "global_lock_key_sha256": args.global_lock_key_sha256,
            "global_lock_path_sha256": args.global_lock_path_sha256,
            "monitor_interval_ms": args.monitor_interval_ms,
            "query_timeout_ms": args.query_timeout_ms,
        }
    result = validate(root, args.mode, evidence)
    validation_path = Path(args.json).resolve()
    harness.atomic_write_json(validation_path, result)
    if result["status"] == "PASS" and args.marker:
        artifact_names = (
            ("manifest.json", "plan.json", "terminal_status.json", validation_path.name)
            if args.mode == "dry-run"
            else (
                "manifest.json",
                "samples.jsonl",
                "correctness.json",
                "result.json",
                "terminal_status.json",
                "harness.log",
                "gpu_identity.json",
                "gpu_exclusivity_lease.json",
                "gpu_pre.json",
                "gpu_post.json",
                "gpu_monitor.json",
                "gpu_observations.ndjson",
                validation_path.name,
            )
        )
        artifacts = {}
        for name in artifact_names:
            path = root / name
            if not path.is_file():
                result["status"] = "FAIL"
                result["accepted_workload_timing"] = 0
                result["errors"].append(f"marker_artifact_missing:{name}")
                harness.atomic_write_json(validation_path, result)
                break
            artifacts[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": harness.sha256_file(path),
            }
        if result["status"] == "PASS":
            harness.atomic_write_json(
                Path(args.marker).resolve(),
                {
                    "schema": SCHEMA,
                    "kind": "tier5_production_completion_marker",
                    "mode": args.mode,
                    "status": "PASS",
                    "accepted_timing": 0,
                    "accepted_timing_semantics": "legacy_CTA_bracket_only",
                    "accepted_workload_timing": result[
                        "accepted_workload_timing"
                    ],
                    "accepted_CTA_bracket": 0,
                    "production_characterization_validated": args.mode == "execute",
                    "tier5_bracket_admitted": False,
                    "formal_bracket_status": "PARTIAL",
                    "artifacts": artifacts,
                },
            )
    print(
        "VALIDATE_PRODUCTION_TIER5 "
        f"schema={SCHEMA} mode={args.mode} status={result['status']} "
        f"errors={len(result['errors'])} "
        f"accepted_workload_timing={result['accepted_workload_timing']} "
        "accepted_CTA_bracket=0 bracket=PARTIAL"
    )
    if result["status"] != "PASS":
        for error in result["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
