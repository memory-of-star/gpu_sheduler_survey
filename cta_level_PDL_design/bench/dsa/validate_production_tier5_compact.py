#!/usr/bin/env python3
"""Fail-closed admission for the bounded Tier-5 production compact campaign.

The compact campaign is not the exact 26-row production campaign and is not a
CTA Floor/Impl/Ceiling bracket.  This validator therefore leaves every legacy
acceptance field at zero and emits a separate, deliberately narrow
``accepted_compact_workload_timing`` field only after the existing campaign
validator has freshly reconstructed every fragment and aggregate artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import production_tier5 as harness
import production_tier5_campaign as campaign
import validate_production_tier5 as validator


SCHEMA = 1
COMPACT_MODELS = ("deepseek_v32", "glm5")
COMPACT_SEQS = (4096, 131072)
EXCLUDED_SEQS = (32768, 1048576)
COMPACT_WORKLOADS = ("operator_chain", "single_layer", "indexshare_fsss")
COMPACT_MATRIX = harness.expected_matrix(
    COMPACT_MODELS, COMPACT_SEQS, COMPACT_WORKLOADS
)
COMPACT_CORRECTNESS_ROW_COUNT = 14
COMPACT_SAMPLE_COUNT = 1302
COMPACT_SUMMARY_COUNT = 62

COMPACT_CONTROLS: dict[str, Any] = {
    "backend": "flashinfer",
    "required_device_substring": "B200",
    "models": list(COMPACT_MODELS),
    "seqs": list(COMPACT_SEQS),
    "workloads": list(COMPACT_WORKLOADS),
    "warmup": 5,
    "repeats": 31,
    # The base harness reserves allow_short=False for its exact 26-row scope.
    # Compact admission is independent and verifies all formal statistics here.
    "allow_short": True,
    "seed": 20260805,
    "max_logits_mb": 16384,
    "max_query_chunk": 4096,
    "moe_experts": 32,
    "moe_topk": 8,
    "moe_tokens": 4096,
    "monitor_interval_ms": 50,
    "query_timeout_ms": 2000,
}

CLAIM_SCOPE: dict[str, Any] = {
    "name": "compact_production_workload_component_timing",
    "models": list(COMPACT_MODELS),
    "contexts": list(COMPACT_SEQS),
    "workloads": list(COMPACT_WORKLOADS),
    "moe32": {
        "models": list(COMPACT_MODELS),
        "rows_per_model": 1,
        "experts": 32,
        "topk": 8,
        "tokens": 4096,
    },
    "statistics": {
        "warmup": 5,
        "timed_repeats": 31,
        "paired_pdl_off_on": True,
        "correctness_sampling": "NONE",
    },
    "excluded": {
        "context_timing": list(EXCLUDED_SEQS),
        "exact_26_row_campaign": True,
        "cta_bracket": True,
        "tier5_headroom": True,
    },
}

STRICT_ARTIFACTS = (
    "campaign_contract.json",
    "campaign_binding.json",
    "manifest.json",
    "samples.jsonl",
    "correctness.json",
    "result.json",
    "campaign_validation.json",
    "production_candidate.done.json",
)


def _base_result() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": "tier5_production_compact_campaign_admission",
        "status": "FAIL",
        "claim_scope": CLAIM_SCOPE,
        "included_models": list(COMPACT_MODELS),
        "included_seqs": list(COMPACT_SEQS),
        "excluded_seqs": list(EXCLUDED_SEQS),
        "included_workloads": list(COMPACT_WORKLOADS),
        "accepted_compact_workload_timing": 0,
        "accepted_exact26_workload_timing": 0,
        "accepted_timing": 0,
        "accepted_timing_semantics": "legacy_CTA_bracket_only",
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "tier5_bracket_admitted": False,
        "formal_bracket_status": "PARTIAL",
        "headroom_defined": False,
        "headroom_pct": None,
        "exact26_campaign_completed": False,
        "strict_final_validation": {
            "status": "NOT_RUN",
            "checker": "production_tier5_campaign.check_final_campaign",
        },
        "expected_cardinalities": {
            "correctness_rows": COMPACT_CORRECTNESS_ROW_COUNT,
            "samples": COMPACT_SAMPLE_COUNT,
            "summaries": COMPACT_SUMMARY_COUNT,
        },
        "observed_cardinalities": {
            "correctness_rows": None,
            "samples": None,
            "summaries": None,
        },
        "bindings": {},
        "artifacts": {},
        "errors": [],
    }


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    passed = not result["errors"]
    result["status"] = "PASS" if passed else "FAIL"
    result["accepted_compact_workload_timing"] = int(passed)
    # Hash the complete body before the self-hash field is inserted.
    result["admission_body_sha256"] = harness.canonical_json_sha(result)
    return result


def _require(condition: bool, error: str, errors: list[str]) -> None:
    if not condition:
        errors.append(error)


def _load_binding(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    binding = campaign.load_regular_json(
        root / "campaign_binding.json", "campaign binding"
    )
    errors = campaign.validate_campaign_binding(binding, contract)
    if errors:
        raise ValueError("invalid campaign binding: " + ",".join(errors))
    return binding


def _artifact_manifest(root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name in STRICT_ARTIFACTS:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"compact artifact is not a regular file: {name}")
        artifacts[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": harness.sha256_file(path),
        }
    return artifacts


def validate_compact_campaign(root: Path) -> dict[str, Any]:
    """Freshly validate ``root`` and return a narrow compact admission."""
    result = _base_result()
    errors: list[str] = result["errors"]
    if root.is_symlink() or not root.is_dir():
        errors.append("campaign_root_not_regular_directory")
        return _finish(result)
    root = root.resolve()

    # This is the first evidence gate.  Scope/cardinality checks below are not
    # evaluated unless the existing strict final checker passes in this call.
    try:
        contract = campaign.load_and_validate_contract(
            root / "campaign_contract.json"
        )
        binding = _load_binding(root, contract)
        completion_marker = campaign.check_final_campaign(
            root, contract, binding
        )
        if completion_marker.get("status") != "PASS":
            raise ValueError("strict final checker did not return PASS")
    except Exception as exc:  # fail closed across malformed/tampered evidence
        result["strict_final_validation"] = {
            "status": "FAIL",
            "checker": "production_tier5_campaign.check_final_campaign",
            "error": f"{type(exc).__name__}:{exc}",
        }
        errors.append(f"strict_final_check:{type(exc).__name__}:{exc}")
        return _finish(result)

    result["strict_final_validation"] = {
        "status": "PASS",
        "checker": "production_tier5_campaign.check_final_campaign",
        "fragment_count": len(completion_marker.get("fragment_markers", [])),
        "completion_marker_sha256": harness.sha256_file(
            root / "production_candidate.done.json"
        ),
        "closure": [
            "exact_fragment_inventory_and_marker_hashes",
            "fragment_semantics_and_aggregate_reconstruction",
            "gpu_identity_and_exclusivity_binding",
            "uniform_runtime_build_identity",
            "package_and_source_manifests",
        ],
    }

    controls = contract.get("controls")
    _require(controls == COMPACT_CONTROLS, "compact_controls_not_exact", errors)
    _require(contract.get("formal") is False, "compact_contract_formal_flag", errors)
    _require(
        contract.get("campaign_mode") == "nonformal_short",
        "compact_campaign_mode",
        errors,
    )
    _require(
        contract.get("ordered_matrix") == COMPACT_MATRIX,
        "compact_ordered_matrix_not_exact",
        errors,
    )
    _require(
        contract.get("row_count") == COMPACT_CORRECTNESS_ROW_COUNT,
        "compact_contract_row_count",
        errors,
    )
    _require(
        contract.get("is_exact_formal_matrix") is False,
        "compact_mislabeled_exact26",
        errors,
    )

    compact_validator_path = Path(__file__).resolve()
    expected_source_record = harness.source_record(str(compact_validator_path))
    sources = contract.get("sources")
    compact_source_records = [
        record
        for record in sources if isinstance(record, dict)
        and record.get("path") == str(compact_validator_path)
    ] if isinstance(sources, list) else []
    _require(
        compact_source_records == [expected_source_record],
        "compact_validator_not_bound_in_source_manifest",
        errors,
    )

    sample_errors: list[str] = []
    samples = validator.load_samples(root / "samples.jsonl", sample_errors)
    errors.extend(f"compact_samples:{error}" for error in sample_errors)
    try:
        correctness = campaign.load_regular_json(
            root / "correctness.json", "compact correctness"
        )
        campaign_result = campaign.load_regular_json(
            root / "result.json", "compact campaign result"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"compact_aggregate_load:{type(exc).__name__}:{exc}")
        return _finish(result)

    correctness_rows = correctness.get("rows")
    summaries = harness.summarize_samples(samples, COMPACT_CONTROLS["seed"])
    observed = {
        "correctness_rows": (
            len(correctness_rows) if isinstance(correctness_rows, list) else None
        ),
        "samples": len(samples),
        "summaries": len(summaries),
    }
    result["observed_cardinalities"] = observed
    _require(
        observed["correctness_rows"] == COMPACT_CORRECTNESS_ROW_COUNT,
        "compact_correctness_row_count",
        errors,
    )
    _require(
        observed["samples"] == COMPACT_SAMPLE_COUNT,
        "compact_sample_count",
        errors,
    )
    _require(
        observed["summaries"] == COMPACT_SUMMARY_COUNT,
        "compact_summary_count",
        errors,
    )
    _require(
        campaign_result.get("correctness_row_count")
        == COMPACT_CORRECTNESS_ROW_COUNT,
        "compact_result_correctness_count",
        errors,
    )
    _require(
        campaign_result.get("sample_count") == COMPACT_SAMPLE_COUNT,
        "compact_result_sample_count",
        errors,
    )
    _require(
        campaign_result.get("summary_count") == COMPACT_SUMMARY_COUNT,
        "compact_result_summary_count",
        errors,
    )
    _require(
        campaign_result.get("summaries") == summaries,
        "compact_result_summary_recompute",
        errors,
    )
    _require(
        campaign_result.get("accepted_timing") == 0
        and campaign_result.get("accepted_workload_timing") == 0
        and campaign_result.get("accepted_CTA_bracket") == 0,
        "compact_base_acceptance_boundary",
        errors,
    )
    _require(
        campaign_result.get("headroom_defined") is False
        and campaign_result.get("headroom_pct") is None,
        "compact_headroom_boundary",
        errors,
    )

    try:
        result["artifacts"] = _artifact_manifest(root)
    except ValueError as exc:
        errors.append(str(exc))
    result["bindings"] = {
        "campaign_contract_sha256": contract.get("contract_sha256"),
        "campaign_fingerprint_sha256": binding.get(
            "campaign_fingerprint_sha256"
        ),
        "controls_sha256": contract.get("controls_sha256"),
        "source_manifest_sha256": contract.get("source_manifest_sha256"),
        "package_manifest_sha256": contract.get("package_manifest_sha256"),
        "compact_matrix_sha256": harness.canonical_json_sha(COMPACT_MATRIX),
        "runtime_build_sha256": campaign_result.get("runtime_build_sha256"),
        "target_gpu": binding.get("target_gpu"),
        "compact_validator": expected_source_record,
    }
    return _finish(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="finalized compact fragment-campaign root")
    parser.add_argument(
        "--json",
        help="output path (default: ROOT/compact_campaign_admission.json)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).absolute()
    result = validate_compact_campaign(root)
    output = (
        Path(args.json).absolute()
        if args.json
        else root / "compact_campaign_admission.json"
    )
    harness.atomic_write_json(output, result)
    print(
        "VALIDATE_PRODUCTION_TIER5_COMPACT "
        f"status={result['status']} errors={len(result['errors'])} "
        "accepted_timing=0 accepted_workload_timing=0 "
        f"accepted_compact_workload_timing="
        f"{result['accepted_compact_workload_timing']} "
        "accepted_exact26_workload_timing=0 accepted_CTA_bracket=0"
    )
    if result["status"] != "PASS":
        for error in result["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
