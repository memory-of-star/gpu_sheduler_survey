#!/usr/bin/env python3
"""CPU-only admission-boundary tests for the compact Tier-5 campaign."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import production_tier5 as harness
import production_tier5_campaign as campaign
import validate_production_tier5 as validator
import validate_production_tier5_compact as compact


def compact_samples() -> list[dict]:
    samples: list[dict] = []
    for row in compact.COMPACT_MATRIX:
        if row["workload"] == "moe32":
            for repeat in range(31):
                samples.append(
                    {
                        "row_id": row["row_id"],
                        "component": "fused_topk_plus_fused_experts",
                        "pdl_mode": "framework_default_uncontrolled",
                        "repeat": repeat,
                        "elapsed_ms": 1.0 + repeat / 1000,
                    }
                )
            continue
        components = validator.expected_components(row["workload"])
        for event, repeat, component, enabled, _ in harness.paired_timing_schedule(
            31, components
        ):
            samples.append(
                {
                    "row_id": row["row_id"],
                    "component": component,
                    "pdl_mode": "on" if enabled else "off",
                    "repeat": repeat,
                    "elapsed_ms": 1.0 + event / 10000,
                }
            )
    return samples


def make_post_strict_fixture(root: Path) -> tuple[dict, dict, dict]:
    samples = compact_samples()
    summaries = harness.summarize_samples(samples, 20260805)
    contract = {
        "controls": dict(compact.COMPACT_CONTROLS),
        "formal": False,
        "campaign_mode": "nonformal_short",
        "ordered_matrix": compact.COMPACT_MATRIX,
        "row_count": 14,
        "is_exact_formal_matrix": False,
        "sources": [harness.source_record(str(Path(compact.__file__).resolve()))],
        "contract_sha256": "a" * 64,
        "controls_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "package_manifest_sha256": "d" * 64,
    }
    binding = {
        "campaign_fingerprint_sha256": "e" * 64,
        "target_gpu": {"index": 0, "uuid": "GPU-fixture", "name": "NVIDIA B200"},
    }
    marker = {
        "status": "PASS",
        "fragment_markers": [
            {"row_id": row["row_id"]} for row in compact.COMPACT_MATRIX
        ],
    }
    for name in compact.STRICT_ARTIFACTS:
        (root / name).write_text("{}\n", encoding="utf-8")
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    harness.atomic_write_json(
        root / "correctness.json",
        {"rows": [{"row_id": row["row_id"]} for row in compact.COMPACT_MATRIX]},
    )
    harness.atomic_write_json(
        root / "result.json",
        {
            "accepted_timing": 0,
            "accepted_workload_timing": 0,
            "accepted_CTA_bracket": 0,
            "headroom_defined": False,
            "headroom_pct": None,
            "correctness_row_count": 14,
            "sample_count": 1302,
            "summary_count": 62,
            "summaries": summaries,
            "runtime_build_sha256": "f" * 64,
        },
    )
    return contract, binding, marker


class CompactAdmissionTests(unittest.TestCase):
    def validate_fixture(self, root: Path, contract: dict, binding: dict, marker: dict):
        with (
            mock.patch.object(
                campaign, "load_and_validate_contract", return_value=contract
            ),
            mock.patch.object(compact, "_load_binding", return_value=binding),
            mock.patch.object(
                campaign, "check_final_campaign", return_value=marker
            ) as strict_check,
        ):
            result = compact.validate_compact_campaign(root)
        strict_check.assert_called_once_with(root.resolve(), contract, binding)
        return result

    def test_exact_compact_scope_passes_with_separate_acceptance_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding, marker = make_post_strict_fixture(root)
            result = self.validate_fixture(root, contract, binding, marker)
        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertEqual(result["accepted_compact_workload_timing"], 1)
        self.assertEqual(result["accepted_exact26_workload_timing"], 0)
        self.assertEqual(result["accepted_timing"], 0)
        self.assertEqual(result["accepted_workload_timing"], 0)
        self.assertEqual(result["accepted_CTA_bracket"], 0)
        self.assertEqual(result["excluded_seqs"], [32768, 1048576])
        self.assertEqual(
            result["observed_cardinalities"],
            {"correctness_rows": 14, "samples": 1302, "summaries": 62},
        )

    def test_post_strict_sample_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, binding, marker = make_post_strict_fixture(root)
            lines = (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            (root / "samples.jsonl").write_text(
                "\n".join(lines[:-1]) + "\n", encoding="utf-8"
            )
            result = self.validate_fixture(root, contract, binding, marker)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["accepted_compact_workload_timing"], 0)
        self.assertIn("compact_sample_count", result["errors"])

    def test_validator_is_part_of_campaign_source_closure(self) -> None:
        expected = str(Path(compact.__file__).resolve())
        records = harness.local_source_manifest()
        self.assertEqual(
            [record for record in records if record["path"] == expected],
            [harness.source_record(expected)],
        )


if __name__ == "__main__":
    unittest.main()
