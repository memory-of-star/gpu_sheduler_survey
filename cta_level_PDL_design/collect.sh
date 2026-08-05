#!/usr/bin/env bash
# collect.sh — package every raw result for transfer back to the dev box.
#
# Analysis happens locally, so this only has to gather things reliably. It also prints a
# quick completeness report so a missing tier is noticed BEFORE the machine is released.
#
# Usage:  RESULTS=results_<tag> ./collect.sh [output.tar.gz]
#
# Optional colon-separated, project-root-relative paths:
#   EXTRA_RESULT_DIRS=bench/results_t23:results/tier4:bench/dsa/results_t5
#   EXTRA_SUPPORT_PATHS=reports:EXPERIMENT_REPORT_INDEX.md:codex/state/coordinates.md

set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)"
cd "${SELF}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${1:-cta_pdl_results_${STAMP}.tar.gz}"
STAGE="/tmp/cta_collect_${STAMP}"
RESULTS="${RESULTS:-results}"
case "${RESULTS}" in
    ""|.|..|*/*)
        echo "collect.sh: RESULTS must name one directory directly under bench/: ${RESULTS}" >&2
        exit 2
        ;;
esac
BENCH_RESULTS="bench/${RESULTS}"

echo "== collecting results"
echo "   selected bench results: ${BENCH_RESULTS}"

canonical_relative_path() {
    local item="$1" kind="$2" relative
    case "${item}" in
        ""|/*|..|../*|*/../*|*/..)
            echo "collect.sh: ${kind} path must be a non-empty project-relative path without '..': ${item}" >&2
            return 2
            ;;
    esac
    if [ ! -e "${item}" ]; then
        echo "collect.sh: ${kind} path does not exist: ${item}" >&2
        return 2
    fi
    relative=$(realpath --canonicalize-existing --relative-to="${SELF}" -- "${item}") || {
        echo "collect.sh: cannot canonicalize ${kind} path: ${item}" >&2
        return 2
    }
    case "${relative}" in
        .|..|../*|/*)
            echo "collect.sh: ${kind} path resolves outside the project or to its root: ${item} -> ${relative}" >&2
            return 2
            ;;
    esac
    if [ "${kind}" = "result" ] && [ ! -d "${relative}" ]; then
        echo "collect.sh: result path must be a directory: ${relative}" >&2
        return 2
    fi
    printf '%s\n' "${relative}"
}

append_relative_paths() {
    local encoded="$1" kind="$2" item relative
    local -n destination="$3"
    [ -n "${encoded}" ] || return 0
    local -a parsed=()
    IFS=: read -r -a parsed <<< "${encoded}"
    for item in "${parsed[@]}"; do
        relative=$(canonical_relative_path "${item}" "${kind}") || exit 2
        destination+=("${relative}")
    done
}

copy_to_stage() {
    local path="$1" target="${STAGE}/$1"
    if [ -e "${target}" ]; then
        echo "collect.sh: duplicate archive destination: ${path}" >&2
        exit 2
    fi
    mkdir -p "${STAGE}/$(dirname "${path}")"
    cp -a -- "${path}" "${target}"
}

# ---- raw result directories ----
# The Tier 0/1 directory is selected explicitly. Do not scan bench/results* here: a rented
# machine may contain historical campaigns with compatible-looking logs and gate.json files.
if [ ! -d "${BENCH_RESULTS}" ]; then
    echo "collect.sh: selected bench results directory is missing: ${BENCH_RESULTS}" >&2
    exit 2
fi
BENCH_RESULTS=$(canonical_relative_path "${BENCH_RESULTS}" "result") || exit 2
RESULT_DIRS=("${BENCH_RESULTS}")
for legacy in bench/llm/results_llm bench/dsa/results_dsa; do
    if [ -d "${legacy}" ]; then
        RESULT_DIRS+=("$(canonical_relative_path "${legacy}" "result")")
    fi
done
append_relative_paths "${EXTRA_RESULT_DIRS:-}" "result" RESULT_DIRS

# Reports, source snapshots, and index files are deliberately opt-in so historical session
# archives retain their old shape.  A completion archive can bind every raw directory to the
# exact reports and validators that interpret it.
SUPPORT_PATHS=()
append_relative_paths "${EXTRA_SUPPORT_PATHS:-}" "support" SUPPORT_PATHS

# A repeated path or a parent/child pair would either overwrite a stage target or copy the
# same evidence twice under nested paths.  Reject it before the first copy, across both kinds.
ALL_PATHS=("${RESULT_DIRS[@]}" "${SUPPORT_PATHS[@]}")
for ((i = 0; i < ${#ALL_PATHS[@]}; ++i)); do
    for ((j = i + 1; j < ${#ALL_PATHS[@]}; ++j)); do
        left="${ALL_PATHS[i]}"
        right="${ALL_PATHS[j]}"
        case "${right}" in
            "${left}"|"${left}"/*)
                echo "collect.sh: duplicate or parent/child archive paths: ${left} <> ${right}" >&2
                exit 2
                ;;
        esac
        case "${left}" in
            "${right}"/*)
                echo "collect.sh: parent/child archive paths: ${left} <> ${right}" >&2
                exit 2
                ;;
        esac
    done
done

if [ -e "${STAGE}" ]; then
    echo "collect.sh: staging path already exists; refusing a mixed archive: ${STAGE}" >&2
    exit 2
fi
mkdir -- "${STAGE}"

for d in "${RESULT_DIRS[@]}"; do
    copy_to_stage "${d}"
    n=$(find "${d}" -name '*.done' 2>/dev/null | wc -l)
    echo "   ${d}: ${n} completed steps"
done
for path in "${SUPPORT_PATHS[@]}"; do
    copy_to_stage "${path}"
    echo "   support: ${path}"
done

# ---- merged record streams (what the analysis actually consumes) ----
# Two schemas, two files. analyze.py reads SUMMARY (tier0_facts, and the rejected
# cta_dep_bench); analyze_pilot.py reads SAMPLE/SUMMARY_PILOT (cta_dep_pilot, the Tier 1
# gate). Merging them into one file would let pilot rows leak into the analyze.py CSV. The
# pilot stream is deliberately limited to this campaign's selected bench results directory.
find "${BENCH_RESULTS}" -name 'summary*.txt' -exec cat {} \; 2>/dev/null \
    > "${STAGE}/all_summary.txt"
if [ -d "${BENCH_RESULTS}" ]; then
    find "${BENCH_RESULTS}" -name 'pilot_matrix.log' -exec cat {} \; 2>/dev/null \
        > "${STAGE}/all_pilot.log"
    if [ -s "${BENCH_RESULTS}/pilot_expected_tags.txt" ]; then
        cp "${BENCH_RESULTS}/pilot_expected_tags.txt" "${STAGE}/pilot_expected_tags.txt"
    fi
else
    : > "${STAGE}/all_pilot.log"
fi
# grep -c prints 0 AND exits non-zero on no-match, so `|| echo 0` would emit "0\n0".
count_matches() { local n; n=$(grep -c "$1" "$2" 2>/dev/null) || true; echo "${n:-0}"; }
NSUM=$(count_matches '^SUMMARY ' "${STAGE}/all_summary.txt")
NPILOT=$(count_matches '^SUMMARY_PILOT ' "${STAGE}/all_pilot.log")
echo "   merged SUMMARY rows: ${NSUM}"
echo "   merged pilot configs: ${NPILOT}"

# New Tier 2/3, Tier 4, and Tier 5 runners are JSON-admitted and intentionally do not emit
# the legacy SUMMARY schema.  Count their strict artifacts directly instead of calling them
# missing merely because all_summary.txt cannot consume them.
read -r T23_STRICT_PASS T4_ADMITTED T5_NATIVE_FORMAL_PASS T5_PRODUCTION_FORMAL_PASS T5_PRODUCTION_COMPACT_PASS < <(
    python3 - "${STAGE}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
t23_hashes: set[str] = set()
t4_cohorts: set[str] = set()
t5_native_campaigns: set[str] = set()
t5_production_campaigns: set[str] = set()
t5_production_compact_campaigns: set[str] = set()
expected_t4 = {"decode_bs_scan_v1", "prefill_context_scan_v1"}
expected_native = {
    (4096, "dsa_exact_seq4096", "exact"),
    (32768, "dsa_exact_seq32768", "exact"),
    (131072, "dsa_work_complete_packed_proxy_seq131072", "work_complete_packed_proxy"),
    (1048576, "dsa_work_complete_packed_proxy_seq1048576", "work_complete_packed_proxy"),
}
expected_production = []
for model in ("deepseek_v32", "glm5"):
    for seq in (4096, 32768, 131072, 1048576):
        for workload in ("operator_chain", "single_layer", "indexshare_fsss"):
            expected_production.append({
                "row_id": f"{model}.{workload}.seq{seq}",
                "model": model,
                "seq": seq,
                "workload": workload,
                "pdl_modes": ["off", "on"],
            })
    expected_production.append({
        "row_id": f"{model}.moe32",
        "model": model,
        "seq": None,
        "workload": "moe32",
        "pdl_modes": ["framework_default_uncontrolled"],
    })
expected_compact = []
for model in ("deepseek_v32", "glm5"):
    for seq in (4096, 131072):
        for workload in ("operator_chain", "single_layer", "indexshare_fsss"):
            expected_compact.append({
                "row_id": f"{model}.{workload}.seq{seq}",
                "model": model,
                "seq": seq,
                "workload": workload,
                "pdl_modes": ["off", "on"],
            })
    expected_compact.append({
        "row_id": f"{model}.moe32",
        "model": model,
        "seq": None,
        "workload": "moe32",
        "pdl_modes": ["framework_default_uncontrolled"],
    })
expected_compact_controls = {
    "backend": "flashinfer",
    "required_device_substring": "B200",
    "models": ["deepseek_v32", "glm5"],
    "seqs": [4096, 131072],
    "workloads": ["operator_chain", "single_layer", "indexshare_fsss"],
    "warmup": 5,
    "repeats": 31,
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
expected_compact_scope = {
    "name": "compact_production_workload_component_timing",
    "models": ["deepseek_v32", "glm5"],
    "contexts": [4096, 131072],
    "workloads": ["operator_chain", "single_layer", "indexshare_fsss"],
    "moe32": {
        "models": ["deepseek_v32", "glm5"],
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
        "context_timing": [32768, 1048576],
        "exact_26_row_campaign": True,
        "cta_bracket": True,
        "tier5_headroom": True,
    },
}
compact_artifact_names = (
    "campaign_contract.json", "campaign_binding.json", "manifest.json",
    "samples.jsonl", "correctness.json", "result.json",
    "campaign_validation.json", "production_candidate.done.json",
)
completion_artifact_names = compact_artifact_names[:-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    payload = json.dumps(
        value, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def regular_json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe JSON path: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def exact_artifact_closure(
    candidate: Path, records, names: tuple[str, ...]
) -> bool:
    if not isinstance(records, dict) or set(records) != set(names):
        return False
    for name in names:
        path = candidate / name
        record = records.get(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(record, dict)
            or set(record) != {"size_bytes", "sha256"}
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != sha256(path)
        ):
            return False
    return True


def declared_artifact_closure(candidate: Path, records) -> bool:
    """Re-hash every regular file named by one fragment completion marker."""
    if not isinstance(records, dict) or not records:
        return False
    required = {
        "correctness.json", "fragment_validation.json", "gpu_identity.json",
        "gpu_monitor.json", "gpu_post.json", "gpu_pre.json", "manifest.json",
        "result.json", "samples.jsonl", "terminal_status.json",
    }
    if not required.issubset(records):
        return False
    for name, record in records.items():
        # Marker artifact names are intentionally flat.  Refuse traversal even
        # though the enclosing campaign tree has already passed the symlink scan.
        if not isinstance(name, str) or Path(name).name != name:
            return False
        path = candidate / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(record, dict)
            or set(record) != {"size_bytes", "sha256"}
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != sha256(path)
        ):
            return False
    return True


def sealed_production_failed_segments(candidate: Path) -> bool:
    """Accept only explicitly quarantined, zero-admission fragment attempts.

    A resumable production campaign keeps failed invocations for audit under a
    directory that is never read by the aggregate finalizer.  Those records
    must not make a later exact-26 PASS look like a rejected campaign, but a
    rejection sentinel anywhere outside this one namespace must remain fatal.
    """
    failed_root = candidate / "failed_segments"
    if failed_root.is_symlink() or not failed_root.is_dir():
        return False
    try:
        contract = regular_json(candidate / "campaign_contract.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    ordered = contract.get("ordered_matrix")
    if not isinstance(ordered, list):
        return False
    prefixes = [
        (ordinal, row.get("row_id"), f"{ordinal:03d}_{row.get('row_id')}.inprogress.")
        for ordinal, row in enumerate(ordered)
        if isinstance(row, dict) and isinstance(row.get("row_id"), str)
    ]
    if len(prefixes) != len(ordered):
        return False
    for attempt in failed_root.iterdir():
        if attempt.is_symlink() or not attempt.is_dir():
            return False
        if any(path.is_symlink() for path in attempt.rglob("*")):
            return False
        matched = [entry for entry in prefixes if attempt.name.startswith(entry[2])]
        if len(matched) != 1:
            return False
        ordinal, row_id, prefix = matched[0]
        suffix = attempt.name[len(prefix):]
        marker_path = attempt / "segment_rejection.json"
        try:
            marker = regular_json(marker_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not (
            marker.get("schema") == 1
            and marker.get("kind") == "tier5_production_fragment_segment_rejection"
            and marker.get("status") == "REJECTED"
            and isinstance(marker.get("reason"), str)
            and bool(marker.get("reason"))
            and marker.get("accepted_timing") == 0
            and marker.get("accepted_workload_timing") == 0
            and marker.get("accepted_CTA_bracket") == 0
        ):
            return False
        reason = marker["reason"]
        if ".stale." in suffix:
            if (
                suffix.count(".stale.") != 1
                or reason != "stale_unsealed_inprogress_recovered_on_resume"
            ):
                return False
        elif ".rejected." in suffix:
            if suffix.count(".rejected.") != 1:
                return False
            invocation = suffix.rsplit(".rejected.", 1)[1]
            if not (
                invocation
                and marker.get("row_id") == row_id
                and marker.get("ordinal") == ordinal
                and marker.get("invocation_uuid") == invocation
            ):
                return False
            expected_artifacts = marker.get("artifacts_before_rejection")
            if not isinstance(expected_artifacts, dict):
                return False
            actual_artifacts = {}
            for artifact in attempt.iterdir():
                if artifact == marker_path:
                    continue
                if artifact.is_symlink() or not artifact.is_file():
                    return False
                actual_artifacts[artifact.name] = {
                    "size_bytes": artifact.stat().st_size,
                    "sha256": sha256(artifact),
                }
            if actual_artifacts != expected_artifacts:
                return False
        else:
            return False
    return True


def clean_root(candidate: Path, *, production_resumable: bool = False) -> bool:
    if candidate.is_symlink() or not candidate.is_dir():
        return False
    failed_root = candidate / "failed_segments"
    if production_resumable and not sealed_production_failed_segments(candidate):
        return False
    forbidden = {"formal_rejection.json", "REJECTED.md", "failure.json", "segment_rejection.json"}
    for path in candidate.rglob("*"):
        if path.is_symlink():
            return False
        quarantined = production_resumable and failed_root in path.parents
        if not quarantined and path.name in forbidden:
            return False
        if (
            not quarantined
            and path.name == "failures.log"
            and path.is_file()
            and path.stat().st_size > 0
        ):
            return False
    return True


def native_formal_pass(candidate: Path, admission: dict) -> bool:
    expected_points = {(seq, tag) for seq, tag, _ in expected_native}
    expected_tags = {tag for _, tag, _ in expected_native}
    if not clean_root(candidate):
        return False
    if not (
        admission.get("schema") == 1
        and admission.get("campaign") == "tier5_native_dsa"
        and admission.get("status") == "PASS"
        and admission.get("errors") == []
        and admission.get("accepted_timing") == 1
        and admission.get("fast") == 0
        and admission.get("profile") == 1
        and {
            (point.get("seq"), point.get("tag"))
            for point in admission.get("expected_points", [])
            if isinstance(point, dict)
        } == expected_points
        and len(admission.get("expected_points", [])) == 4
        and admission.get("timing_matrix", {}).get("status") == "PASS"
        and admission.get("timing_matrix", {}).get("points") == 4
        and admission.get("profile_sidecar", {}).get("required") is True
        and admission.get("profile_sidecar", {}).get("status") == "PASS"
    ):
        return False
    matrix_path = candidate / "validation_matrix.json"
    terminal_path = candidate / "terminal_status.json"
    matrix = regular_json(matrix_path)
    terminal = regular_json(terminal_path)
    if not (
        admission.get("timing_matrix", {}).get("sha256") == sha256(matrix_path)
        and admission.get("terminal_status_sha256") == sha256(terminal_path)
        and matrix.get("schema") == 1
        and matrix.get("status") == "PASS"
        and matrix.get("errors") == []
        and matrix.get("fast") == 0
        and set(matrix.get("expected_tags", [])) == expected_tags
        and len(matrix.get("records", [])) == 4
        and terminal.get("schema") == 1
        and terminal.get("campaign") == "tier5_native_dsa"
        and terminal.get("status") == "PASS"
        and terminal.get("errors") == []
        and terminal.get("fast") == 0
        and terminal.get("profile") == 1
        and set(terminal.get("expected_tags", [])) == expected_tags
    ):
        return False
    record_points = {
        (record.get("seq"), record.get("tag"))
        for record in matrix.get("records", []) if isinstance(record, dict)
    }
    if record_points != expected_points or any(
        not isinstance(record, dict)
        or record.get("status") != "PASS"
        or record.get("errors") != []
        for record in matrix.get("records", [])
    ):
        return False
    if {path.stem for path in candidate.glob("dsa_*.done")} != expected_tags:
        return False
    artifact_hashes = admission.get("artifact_sha256", {})
    if not isinstance(artifact_hashes, dict):
        return False
    for seq, tag, mapping in expected_native:
        validation_path = candidate / f"{tag}_validation.json"
        validation = regular_json(validation_path)
        if not (
            validation.get("schema") == 1
            and validation.get("status") == "PASS"
            and validation.get("errors") == []
            and validation.get("seq") == seq
            and validation.get("mapping") == mapping
            and validation.get("repeats") == 31
            and validation.get("samples") == 124
            and validation.get("mode_count") == 4
            and validation.get("pair_work_complete") is True
            and artifact_hashes.get(f"{tag}:{validation_path.name}") == sha256(validation_path)
        ):
            return False
    return True


def production_formal_pass(candidate: Path, validation: dict) -> bool:
    if not clean_root(candidate, production_resumable=True):
        return False
    contract_path = candidate / "campaign_contract.json"
    binding_path = candidate / "campaign_binding.json"
    result_path = candidate / "result.json"
    marker_path = candidate / "production_candidate.done.json"
    contract = regular_json(contract_path)
    binding = regular_json(binding_path)
    result = regular_json(result_path)
    marker = regular_json(marker_path)
    ordered = contract.get("ordered_matrix", [])
    row_ids = [row.get("row_id") for row in ordered if isinstance(row, dict)]
    summaries = result.get("summaries", [])
    controls = contract.get("controls", {})
    experiment = contract.get("experiment_contract", {})
    if not (
        contract.get("schema") == 1
        and contract.get("kind") == "tier5_production_campaign_contract"
        and contract.get("status") == "FROZEN"
        and contract.get("formal") is True
        and contract.get("campaign_mode") == "formal"
        and contract.get("is_exact_formal_matrix") is True
        and contract.get("row_count") == 26
        and ordered == expected_production
        and len(set(row_ids)) == 26
        and contract.get("formal_full_ordered_matrix") == ordered
        and controls.get("allow_short") is False
        and controls.get("warmup") == 5
        and controls.get("repeats") == 31
        and contract.get("accepted_timing") == 0
        and contract.get("accepted_workload_timing") == 0
        and contract.get("accepted_CTA_bracket") == 0
        and experiment.get("headroom_defined") is False
        and experiment.get("headroom_pct") is None
    ):
        return False
    contract_body = dict(contract)
    contract_digest = contract_body.pop("contract_sha256", None)
    canonical = json.dumps(contract_body, separators=(",", ":"), sort_keys=True).encode()
    if contract_digest != hashlib.sha256(canonical).hexdigest():
        return False
    if not (
        binding.get("schema") == 1
        and binding.get("kind") == "tier5_production_campaign_device_binding"
        and binding.get("status") == "FROZEN"
        and binding.get("accepted_timing") == 0
        and binding.get("accepted_workload_timing") == 0
        and binding.get("accepted_CTA_bracket") == 0
        and binding.get("contract_sha256") == contract_digest
        and binding.get("controls_sha256") == contract.get("controls_sha256")
        and binding.get("package_manifest_sha256") == contract.get("package_manifest_sha256")
        and binding.get("source_manifest_sha256") == contract.get("source_manifest_sha256")
    ):
        return False
    if not (
        validation.get("schema") == 1
        and validation.get("kind") == "tier5_production_campaign_validation"
        and validation.get("status") == "PASS"
        and validation.get("errors") == []
        and validation.get("formal_campaign") is True
        and validation.get("exact_inventory") is True
        and validation.get("accepted_timing") == 0
        and validation.get("accepted_workload_timing") == 1
        and validation.get("accepted_CTA_bracket") == 0
        and validation.get("result_sha256") == sha256(result_path)
        and result.get("schema") == 1
        and result.get("kind") == "tier5_production_campaign_result"
        and result.get("status") == "PASS"
        and result.get("campaign_mode") == "formal"
        and result.get("accepted_timing") == 0
        and result.get("accepted_workload_timing") == 1
        and result.get("accepted_CTA_bracket") == 0
        and result.get("correctness_row_count") == 26
        and result.get("sample_count") == 2542
        and result.get("summary_count") == 122
        and isinstance(summaries, list)
        and len(summaries) == 122
        and all(
            isinstance(summary, dict)
            and "headroom_pct" not in summary
            and summary.get("formal_tier5_headroom") is not True
            for summary in summaries
        )
        and result.get("tier5_bracket_admitted") is False
        and result.get("formal_bracket_status") == "PARTIAL"
        and result.get("headroom_defined") is False
        and result.get("headroom_pct") is None
        and result.get("campaign_contract_sha256") == contract_digest
        and result.get("campaign_fingerprint_sha256") == binding.get("campaign_fingerprint_sha256")
        and marker.get("schema") == 1
        and marker.get("kind") == "tier5_production_campaign_completion_marker"
        and marker.get("status") == "PASS"
        and marker.get("campaign_mode") == "formal"
        and marker.get("accepted_timing") == 0
        and marker.get("accepted_workload_timing") == 1
        and marker.get("accepted_CTA_bracket") == 0
        and marker.get("campaign_contract_sha256") == contract_digest
        and marker.get("campaign_fingerprint_sha256") == binding.get("campaign_fingerprint_sha256")
    ):
        return False
    for name, field in (
        ("manifest.json", "manifest_sha256"),
        ("samples.jsonl", "samples_sha256"),
        ("correctness.json", "correctness_sha256"),
    ):
        path = candidate / name
        if path.is_symlink() or not path.is_file() or result.get(field) != sha256(path):
            return False
    marker_fragments = marker.get("fragment_markers", [])
    if result.get("fragment_markers") != marker_fragments or len(marker_fragments) != 26:
        return False
    rows_root = candidate / "rows"
    if rows_root.is_symlink() or not rows_root.is_dir() or len(list(rows_root.iterdir())) != 26:
        return False
    for ordinal, (row_id, fragment) in enumerate(zip(row_ids, marker_fragments)):
        expected_relative = Path("rows") / f"{ordinal:03d}_{row_id}" / "fragment.done.json"
        fragment_path = candidate / expected_relative
        if not (
            isinstance(fragment, dict)
            and fragment.get("row_id") == row_id
            and fragment.get("ordinal") == ordinal
            and fragment.get("path") == str(expected_relative)
            and fragment_path.is_file()
            and not fragment_path.is_symlink()
            and fragment.get("sha256") == sha256(fragment_path)
        ):
            return False
        fragment_marker = regular_json(fragment_path)
        if not (
            fragment_marker.get("kind") == "tier5_production_fragment_completion_marker"
            and fragment_marker.get("status") == "PASS"
            and fragment_marker.get("accepted_timing") == 0
            and fragment_marker.get("accepted_workload_timing") == 0
            and fragment_marker.get("accepted_CTA_bracket") == 0
        ):
            return False
    artifacts = marker.get("artifacts", {})
    required_artifacts = (
        "campaign_contract.json", "campaign_binding.json", "manifest.json",
        "samples.jsonl", "correctness.json", "result.json", "campaign_validation.json",
    )
    return all(
        isinstance(artifacts.get(name), dict)
        and artifacts[name].get("size_bytes") == (candidate / name).stat().st_size
        and artifacts[name].get("sha256") == sha256(candidate / name)
        for name in required_artifacts
    )


def production_compact_pass(candidate: Path, admission: dict) -> bool:
    """Admit only the independent compact-14 scope, never exact-26 or CTA timing."""
    if not clean_root(candidate):
        return False
    failed_root = candidate / "failed_segments"
    if failed_root.exists() and (
        failed_root.is_symlink()
        or not failed_root.is_dir()
        or any(failed_root.iterdir())
    ):
        return False

    body = dict(admission)
    body_digest = body.pop("admission_body_sha256", None)
    expected_cardinalities = {
        "correctness_rows": 14,
        "samples": 1302,
        "summaries": 62,
    }
    if not (
        admission.get("schema") == 1
        and admission.get("kind")
        == "tier5_production_compact_campaign_admission"
        and admission.get("status") == "PASS"
        and admission.get("errors") == []
        and body_digest == canonical_sha(body)
        and admission.get("claim_scope") == expected_compact_scope
        and admission.get("included_models") == ["deepseek_v32", "glm5"]
        and admission.get("included_seqs") == [4096, 131072]
        and admission.get("excluded_seqs") == [32768, 1048576]
        and admission.get("included_workloads")
        == ["operator_chain", "single_layer", "indexshare_fsss"]
        and admission.get("accepted_compact_workload_timing") == 1
        and admission.get("accepted_exact26_workload_timing") == 0
        and admission.get("accepted_timing") == 0
        and admission.get("accepted_timing_semantics")
        == "legacy_CTA_bracket_only"
        and admission.get("accepted_workload_timing") == 0
        and admission.get("accepted_CTA_bracket") == 0
        and admission.get("tier5_bracket_admitted") is False
        and admission.get("formal_bracket_status") == "PARTIAL"
        and admission.get("headroom_defined") is False
        and admission.get("headroom_pct") is None
        and admission.get("exact26_campaign_completed") is False
        and admission.get("expected_cardinalities") == expected_cardinalities
        and admission.get("observed_cardinalities") == expected_cardinalities
        and exact_artifact_closure(
            candidate, admission.get("artifacts"), compact_artifact_names
        )
    ):
        return False

    contract_path = candidate / "campaign_contract.json"
    binding_path = candidate / "campaign_binding.json"
    manifest_path = candidate / "manifest.json"
    samples_path = candidate / "samples.jsonl"
    correctness_path = candidate / "correctness.json"
    result_path = candidate / "result.json"
    validation_path = candidate / "campaign_validation.json"
    completion_path = candidate / "production_candidate.done.json"
    contract = regular_json(contract_path)
    binding = regular_json(binding_path)
    manifest = regular_json(manifest_path)
    correctness = regular_json(correctness_path)
    campaign_result = regular_json(result_path)
    validation = regular_json(validation_path)
    completion = regular_json(completion_path)

    contract_body = dict(contract)
    contract_digest = contract_body.pop("contract_sha256", None)
    controls = contract.get("controls")
    ordered = contract.get("ordered_matrix")
    row_ids = [row.get("row_id") for row in ordered if isinstance(row, dict)] \
        if isinstance(ordered, list) else []
    experiment = contract.get("experiment_contract")
    if not (
        contract.get("schema") == 1
        and contract.get("kind") == "tier5_production_campaign_contract"
        and contract.get("status") == "FROZEN"
        and contract.get("formal") is False
        and contract.get("campaign_mode") == "nonformal_short"
        and contract.get("is_exact_formal_matrix") is False
        and contract.get("row_count") == 14
        and ordered == expected_compact
        and len(row_ids) == 14
        and len(set(row_ids)) == 14
        and contract.get("formal_full_ordered_matrix") == expected_production
        and contract.get("formal_full_matrix_sha256")
        == canonical_sha(expected_production)
        and controls == expected_compact_controls
        and contract.get("controls_sha256") == canonical_sha(controls)
        and contract.get("accepted_timing") == 0
        and contract.get("accepted_workload_timing") == 0
        and contract.get("accepted_CTA_bracket") == 0
        and isinstance(experiment, dict)
        and experiment.get("tier5_bracket_admitted") is False
        and experiment.get("headroom_defined") is False
        and experiment.get("headroom_pct") is None
        and contract_digest == canonical_sha(contract_body)
        and contract.get("package_manifest_sha256")
        == canonical_sha(contract.get("packages"))
        and contract.get("source_manifest_sha256")
        == canonical_sha(contract.get("sources"))
        and contract.get("fragment_argv_template_sha256")
        == canonical_sha(contract.get("fragment_argv_template"))
    ):
        return False

    target = binding.get("target_gpu")
    if not (
        isinstance(target, dict)
        and isinstance(target.get("index"), int)
        and not isinstance(target.get("index"), bool)
        and target.get("index") >= 0
        and isinstance(target.get("uuid"), str)
        and target.get("uuid").startswith("GPU-")
        and isinstance(target.get("name"), str)
        and "B200" in target.get("name")
    ):
        return False
    binding_body = {
        "schema": 1,
        "kind": "tier5_production_campaign_device_binding",
        "status": "FROZEN",
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "contract_sha256": contract_digest,
        "package_manifest_sha256": contract.get("package_manifest_sha256"),
        "source_manifest_sha256": contract.get("source_manifest_sha256"),
        "controls_sha256": contract.get("controls_sha256"),
        "target_gpu": target,
    }
    expected_binding = dict(binding_body)
    expected_binding["campaign_fingerprint_sha256"] = canonical_sha(binding_body)
    if binding != expected_binding:
        return False
    fingerprint = binding["campaign_fingerprint_sha256"]

    summaries = campaign_result.get("summaries")
    result_fragments = campaign_result.get("fragment_markers")
    completion_fragments = completion.get("fragment_markers")
    runtime_build = campaign_result.get("runtime_build_sha256")
    if not (
        campaign_result.get("schema") == 1
        and campaign_result.get("kind") == "tier5_production_campaign_result"
        and campaign_result.get("status") == "PASS"
        and campaign_result.get("campaign_mode") == "nonformal_short"
        and campaign_result.get("accepted_timing") == 0
        and campaign_result.get("accepted_workload_timing") == 0
        and campaign_result.get("accepted_CTA_bracket") == 0
        and campaign_result.get("tier5_bracket_admitted") is False
        and campaign_result.get("formal_bracket_status") == "PARTIAL"
        and campaign_result.get("headroom_defined") is False
        and campaign_result.get("headroom_pct") is None
        and campaign_result.get("correctness_row_count") == 14
        and campaign_result.get("sample_count") == 1302
        and campaign_result.get("summary_count") == 62
        and isinstance(summaries, list)
        and len(summaries) == 62
        and all(
            isinstance(summary, dict)
            and "headroom_pct" not in summary
            and summary.get("formal_tier5_headroom") is not True
            for summary in summaries
        )
        and campaign_result.get("campaign_contract_sha256") == contract_digest
        and campaign_result.get("campaign_fingerprint_sha256") == fingerprint
        and campaign_result.get("manifest_sha256") == sha256(manifest_path)
        and campaign_result.get("samples_sha256") == sha256(samples_path)
        and campaign_result.get("correctness_sha256") == sha256(correctness_path)
        and campaign_result.get("device_binding_sha256") == canonical_sha(target)
        and is_sha256(runtime_build)
        and isinstance(result_fragments, list)
        and result_fragments == completion_fragments
        and len(result_fragments) == 14
    ):
        return False

    expected_validation = {
        "schema": 1,
        "kind": "tier5_production_campaign_validation",
        "status": "PASS",
        "formal_campaign": False,
        "exact_inventory": True,
        "accepted_timing": 0,
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "errors": [],
        "result_sha256": sha256(result_path),
    }
    if validation != expected_validation:
        return False
    if not (
        completion.get("schema") == 1
        and completion.get("kind")
        == "tier5_production_campaign_completion_marker"
        and completion.get("status") == "PASS"
        and completion.get("campaign_mode") == "nonformal_short"
        and completion.get("accepted_timing") == 0
        and completion.get("accepted_workload_timing") == 0
        and completion.get("accepted_CTA_bracket") == 0
        and completion.get("campaign_contract_sha256") == contract_digest
        and completion.get("campaign_fingerprint_sha256") == fingerprint
        and exact_artifact_closure(
            candidate, completion.get("artifacts"), completion_artifact_names
        )
    ):
        return False

    strict_final = admission.get("strict_final_validation")
    if not (
        isinstance(strict_final, dict)
        and strict_final.get("status") == "PASS"
        and strict_final.get("checker")
        == "production_tier5_campaign.check_final_campaign"
        and strict_final.get("fragment_count") == 14
        and strict_final.get("completion_marker_sha256") == sha256(completion_path)
    ):
        return False

    try:
        sample_lines = samples_path.read_text(encoding="utf-8").splitlines()
        samples = [json.loads(line) for line in sample_lines if line]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        len(sample_lines) != 1302
        or len(samples) != 1302
        or any(not isinstance(sample, dict) for sample in samples)
    ):
        return False
    expected_sample_counts = {
        row["row_id"]: (
            31 if row["workload"] == "moe32"
            else 186 if row["workload"] == "operator_chain"
            else 62
        )
        for row in expected_compact
    }
    sample_counts = {row_id: 0 for row_id in row_ids}
    for sample in samples:
        row_id = sample.get("row_id")
        if row_id not in sample_counts:
            return False
        sample_counts[row_id] += 1
    correctness_rows = correctness.get("rows")
    if not (
        sample_counts == expected_sample_counts
        and correctness.get("schema") == 1
        and correctness.get("kind") == "tier5_production_correctness"
        and correctness.get("status") == "PASS"
        and correctness.get("execution_scope") == "campaign_aggregate"
        and correctness.get("all_expected_rows_present") is True
        and isinstance(correctness_rows, list)
        and len(correctness_rows) == 14
        and all(isinstance(row, dict) for row in correctness_rows)
        and [row.get("row_id") for row in correctness_rows] == row_ids
    ):
        return False

    if not (
        manifest.get("schema") == 1
        and manifest.get("kind") == "tier5_production_dsa_manifest"
        and manifest.get("execution_scope") == "campaign_aggregate"
        and manifest.get("campaign_contract_sha256") == contract_digest
        and manifest.get("campaign_fingerprint_sha256") == fingerprint
        and manifest.get("fragment_markers") == completion_fragments
    ):
        return False

    rows_root = candidate / "rows"
    expected_row_names = [
        f"{ordinal:03d}_{row_id}" for ordinal, row_id in enumerate(row_ids)
    ]
    if (
        rows_root.is_symlink()
        or not rows_root.is_dir()
        or sorted(path.name for path in rows_root.iterdir())
        != sorted(expected_row_names)
    ):
        return False
    invocation_uuids: set[str] = set()
    for ordinal, (row_id, compact_fragment) in enumerate(
        zip(row_ids, completion_fragments)
    ):
        fragment_root = rows_root / expected_row_names[ordinal]
        fragment_path = fragment_root / "fragment.done.json"
        expected_relative = Path("rows") / expected_row_names[ordinal] / "fragment.done.json"
        if (
            fragment_root.is_symlink()
            or not fragment_root.is_dir()
            or fragment_path.is_symlink()
            or not fragment_path.is_file()
        ):
            return False
        expected_compact_fragment = {
            "row_id": row_id,
            "ordinal": ordinal,
            "path": str(expected_relative),
            "sha256": sha256(fragment_path),
        }
        if compact_fragment != expected_compact_fragment:
            return False
        fragment = regular_json(fragment_path)
        invocation_uuid = fragment.get("invocation_uuid")
        if not (
            fragment.get("schema") == 1
            and fragment.get("kind")
            == "tier5_production_fragment_completion_marker"
            and fragment.get("status") == "PASS"
            and fragment.get("accepted_timing") == 0
            and fragment.get("accepted_workload_timing") == 0
            and fragment.get("accepted_CTA_bracket") == 0
            and fragment.get("row_id") == row_id
            and fragment.get("ordinal") == ordinal
            and fragment.get("campaign_contract_sha256") == contract_digest
            and fragment.get("campaign_fingerprint_sha256") == fingerprint
            and fragment.get("controls_sha256") == contract.get("controls_sha256")
            and fragment.get("package_manifest_sha256")
            == contract.get("package_manifest_sha256")
            and fragment.get("runtime_build_sha256") == runtime_build
            and is_sha256(fragment.get("source_manifest_sha256"))
            and is_sha256(fragment.get("device_sha256"))
            and isinstance(invocation_uuid, str)
            and bool(invocation_uuid)
            and invocation_uuid not in invocation_uuids
            and declared_artifact_closure(
                fragment_root, fragment.get("artifacts")
            )
        ):
            return False
        invocation_uuids.add(invocation_uuid)

    bindings = admission.get("bindings")
    compact_validator = bindings.get("compact_validator") \
        if isinstance(bindings, dict) else None
    sources = contract.get("sources")
    if not (
        isinstance(bindings, dict)
        and bindings.get("campaign_contract_sha256") == contract_digest
        and bindings.get("campaign_fingerprint_sha256") == fingerprint
        and bindings.get("controls_sha256") == contract.get("controls_sha256")
        and bindings.get("source_manifest_sha256")
        == contract.get("source_manifest_sha256")
        and bindings.get("package_manifest_sha256")
        == contract.get("package_manifest_sha256")
        and bindings.get("compact_matrix_sha256") == canonical_sha(expected_compact)
        and bindings.get("runtime_build_sha256") == runtime_build
        and bindings.get("target_gpu") == target
        and isinstance(compact_validator, dict)
        and compact_validator.get("exists") is True
        and is_sha256(compact_validator.get("sha256"))
        and isinstance(compact_validator.get("size_bytes"), int)
        and compact_validator.get("size_bytes") > 0
        and isinstance(sources, list)
        and [source for source in sources if source == compact_validator]
        == [compact_validator]
    ):
        return False
    return True


for path in root.rglob("*.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if path.name == "tier23_validation.json":
        if (
            value.get("schema") == 1
            and value.get("status") == "PASS"
            and value.get("formal") is True
            and value.get("errors") == []
            and value.get("config_count") == 35
            and value.get("sample_count") == 5084
            and value.get("trace_row_count") == 182460
        ):
            canonical = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
            t23_hashes.add(hashlib.sha256(canonical).hexdigest())
    elif path.name == "admission.json":
        raw_path = path.parent / "raw_triplet.json"
        try:
            raw_bytes = raw_path.read_bytes()
            raw = json.loads(raw_bytes)
        except Exception:
            continue
        if (
            value.get("schema") == "tier4.admission.v2"
            and value.get("admissible") is True
            and value.get("status") == "ok"
            and value.get("raw_triplet_sha256") == hashlib.sha256(raw_bytes).hexdigest()
            and raw.get("schema") == "tier4.triplet.raw.v2"
            and raw.get("repeats") == 31
            and raw.get("warmups") == 3
            and raw.get("bootstrap_samples") == 2000
            and raw.get("allow_short") is False
            and raw.get("cohort_id") in expected_t4
        ):
            t4_cohorts.add(raw["cohort_id"])
    elif path.name == "campaign_admission.json":
        try:
            if native_formal_pass(path.parent, value):
                t5_native_campaigns.add(sha256(path))
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            pass
    elif path.name == "campaign_validation.json" and value.get("kind") == "tier5_production_campaign_validation":
        try:
            if production_formal_pass(path.parent, value):
                t5_production_campaigns.add(sha256(path))
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            pass
    elif path.name == "compact_campaign_admission.json":
        try:
            if production_compact_pass(path.parent, value):
                t5_production_compact_campaigns.add(sha256(path))
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            pass
print(
    len(t23_hashes), len(t4_cohorts),
    len(t5_native_campaigns), len(t5_production_campaigns),
    len(t5_production_compact_campaigns),
)
PY
)

# ---- the gate verdict, the single most important artefact of the session ----
# Never choose a gate by `find ... | head`: directory traversal order is not provenance.
GATE_SRC="${BENCH_RESULTS}/gate.json"
if [ -s "${GATE_SRC}" ]; then
    cp "${GATE_SRC}" "${STAGE}/gate.json"
    VERDICT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['verdict'])" \
              "${GATE_SRC}" 2>/dev/null || echo "unparseable")
    echo "   gate verdict: ${VERDICT}"
else
    echo "   gate verdict: MISSING in ${BENCH_RESULTS} (tier1p did not reach a decision)"
fi

# ---- environment provenance ----
{
    echo "collected: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "results: ${RESULTS}"
    echo "bench_results: ${BENCH_RESULTS}"
    echo
    echo "== nvidia-smi =="
    nvidia-smi 2>&1 || echo "(unavailable)"
    echo
    echo "== nvcc =="
    nvcc --version 2>&1 || echo "(unavailable)"
    echo
    echo "== python =="
    python3 --version 2>&1
    python3 -c "import torch;print('torch', torch.__version__)" 2>/dev/null || echo "torch: n/a"
    python3 -c "import vllm;print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm: n/a"
} > "${STAGE}/environment.txt" 2>&1

# ---- nsys traces, if any (can be large) ----
NSYS_N=0
for d in "${RESULT_DIRS[@]}"; do
    [ -d "${d}" ] || continue
    n=$(find "${d}" \( -name '*.nsys-rep' -o -name '*.sqlite' \) 2>/dev/null | wc -l)
    NSYS_N=$((NSYS_N + n))
done
if [ "${NSYS_N}" -gt 0 ]; then
    echo "   nsys artifacts in selected result directories: ${NSYS_N} files"
fi

# ---- completeness report ----
REPORT="${STAGE}/completeness.txt"
{
    echo "== completeness =="
    check() {
        local label="$1" pattern="$2" file="${3:-${STAGE}/all_summary.txt}" n
        n=$(count_matches "${pattern}" "${file}")
        printf "  %-34s %5s rows  %s\n" "${label}" "${n}" \
               "$([ "${n}" -gt 0 ] && echo "ok" || echo "MISSING")"
    }
    echo "  -- Tier 0 (valid harness) --"
    check "Tier 0.1 chain overlap"     "tier0=chain"
    check "Tier 0.3 occupancy"         "tier0=occupancy"
    check "Tier 0.4 CLC"               "tier0=clc"
    check "Tier 0.5 fence"             "tier0=fence"
    # Count SUMMARY_PILOT only: one per configuration. Matching a bare tag= would also hit
    # every SAMPLE line and report a count nobody can interpret.
    echo "  -- Tier 1p (corrected pilot: the gate data) --"
    check "Tier 1.1p degree axis"      "^SUMMARY_PILOT .*tag=t11p_g"  "${STAGE}/all_pilot.log"
    check "Tier 1.1p structure axis"   "^SUMMARY_PILOT .*tag=t11ps_"  "${STAGE}/all_pilot.log"
    check "Tier 1.2p tail/prologue"    "^SUMMARY_PILOT .*tag=t12p_"   "${STAGE}/all_pilot.log"
    echo "  -- rejected harness (present only if re-audited) --"
    check "Tier 1.1a (cta_dep_bench)"  "tag=t11a_"
    check "Tier 2.1 protocols"         "tag=t21_"
    check "Tier 2.3 encoding"          "tag=t23_"
    echo "  -- JSON-admitted later sessions --"
    printf "  %-34s %5s artefacts  %s\n" "Tier 2/3 strict campaigns" "${T23_STRICT_PASS}" \
           "$([ "${T23_STRICT_PASS}" -eq 1 ] && echo "ok" || echo "INCOMPLETE(expected 1)")"
    printf "  %-34s %5s cohorts    %s\n" "Tier 4 admitted cohorts" "${T4_ADMITTED}" \
           "$([ "${T4_ADMITTED}" -eq 2 ] && echo "ok" || echo "INCOMPLETE(expected 2)")"
    printf "  %-34s %5s campaigns  %s\n" "Tier 5 native formal admission" "${T5_NATIVE_FORMAL_PASS}" \
           "$([ "${T5_NATIVE_FORMAL_PASS}" -eq 1 ] && echo "ok" || echo "INCOMPLETE(expected 1)")"
    printf "  %-34s %5s campaigns  %s\n" "Tier 5 production exact-26" "${T5_PRODUCTION_FORMAL_PASS}" \
           "$(case "${T5_PRODUCTION_FORMAL_PASS}" in 0) echo "NOT COMPLETED (allowed by compact-14 scope)" ;; 1) echo "ok" ;; *) echo "INVALID(expected at most 1)" ;; esac)"
    printf "  %-34s %5s campaigns  %s\n" "Tier 5 production compact-14" "${T5_PRODUCTION_COMPACT_PASS}" \
           "$([ "${T5_PRODUCTION_COMPACT_PASS}" -eq 1 ] && echo "ok (SCOPED: 4K/128K only)" || echo "INCOMPLETE(expected 1 scoped)")"
    echo
    echo "  CTA traces: $(find "${STAGE}" -name '*trace*.csv' 2>/dev/null | wc -l) files"
    QUARANTINED_SEGMENTS=$(find "${STAGE}" -path '*/failed_segments/*/segment_rejection.json' \
        -type f 2>/dev/null | wc -l)
    echo "  raw production rejection markers: ${QUARANTINED_SEGMENTS} (validated separately before admission)"
    FAILS=$(find "${STAGE}" -path '*/failed_segments/*' -prune -o \
        -name 'failures.log' -type f -size +0 -print 2>/dev/null | wc -l)
    if [ "${FAILS}" -gt 0 ]; then
        echo
        echo "  !! failures were recorded outside quarantine:"
        find "${STAGE}" -path '*/failed_segments/*' -prune -o \
            -name 'failures.log' -type f -size +0 -exec cat {} \; 2>/dev/null \
            | sed 's/^/     /'
    fi
} > "${REPORT}"
cat "${REPORT}"

# ---- pack ----
tar czf "${OUT}" -C "$(dirname "${STAGE}")" "$(basename "${STAGE}")"
rm -rf "${STAGE}"

echo
echo "== packed: ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo
echo "Re-analysing this archive anywhere (no GPU needed):"
echo "  tar xzf ${OUT} && cd cta_collect_${STAMP}"
echo "  cat gate.json                                  # the verdict"
echo "  python3 tools/analyze_pilot.py all_pilot.log \\"
echo "          --expected pilot_expected_tags.txt \\"
echo "          --json pilot_analysis.json --csv pilot_summary.csv"
echo "  python3 tools/gate.py         pilot_analysis.json"
echo "  python3 tools/analyze.py      all_summary.txt --csv all.csv --json findings.json"
