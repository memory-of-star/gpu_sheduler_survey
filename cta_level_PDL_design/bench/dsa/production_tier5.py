#!/usr/bin/env python3
"""Production-kernel Tier-5 DSA harness.

The default mode is a CPU-only dry run.  It imports neither torch nor vLLM,
does not inspect a CUDA device, and writes only an execution plan.  GPU work is
guarded by both ``--execute-gpu`` and ``TIER5_PRODUCTION_GPU_ALLOWED=1``.

The executable path deliberately composes the installed production primitives:

* vLLM ``SparseAttnIndexer`` (DeepGEMM MQA logits plus vLLM top-k),
* FlashInfer ``trtllm_batch_decode_with_kv_cache_mla`` sparse MLA, and
* vLLM ``fused_topk`` plus ``fused_experts`` for the 32-expert MoE row.

Random weights use the exact DeepSeek-V3.2 and GLM-5 attention dimensions.
The 4K/32K/128K/1M attention workload is an exact causal lower triangle: all
S query rows and all S*(S+1)/2 causal indexer query-key pairs are executed,
with no query or causal-pair sampling.  Query streaming changes only the
workspace partition.  One million tokens is always labelled an extreme point
outside both models' official position range.

This is a production workload harness, not a fabricated CTA implementation.
The installed APIs expose a real DeepGEMM PDL switch and FlashInfer
``enable_pdl``.  They do not expose a CTA-readiness implementation or a truly
unordered Ceiling.  Consequently every artifact keeps the formal Tier-5
bracket status ``PARTIAL`` and never emits CTA headroom.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = 1
KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30
FORMAL_SEQS = (4096, 32768, 131072, 1048576)
FORMAL_WORKLOADS = ("operator_chain", "single_layer", "indexshare_fsss")
PDL_MODES = ("off", "on")
INDEX_BLOCK_SIZE = 64
INDEX_CACHE_BYTES_PER_TOKEN = 132
MLA_CACHE_BYTES_PER_TOKEN_BF16 = 576 * 2
FLASHINFER_WORKSPACE_BYTES = 128 * MIB
TOPK_TAIL_CONTRACT = "UNSPECIFIED_IGNORED"
DEEPGEMM_CALC_DIFF_LIMIT = 5e-6
DEEPGEMM_ROW_CALC_DIFF_LIMIT = 1e-3
FORMAL_MAX_LOGITS_MB = 16384
FORMAL_MAX_QUERY_CHUNK = 4096
ATTENTION_REFERENCE_ROW_BATCH = 32
LOGITS_QUALITY_QUERY_BATCH = 8
CUDA_CACHE_RELEASE_CADENCE_CHUNKS = 1
DEEPGEMM_MQA_HEADER = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/"
    "include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh"
)
DEEPGEMM_MQA_HEADER_SHA256 = (
    "66629c2eeca18e419edc76d57fc5c50f761db8de52bbc3f80c747b8bd873b464"
)
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
INVOCATION_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def canonical_gpu_uuid(value: object) -> str | None:
    text = str(value).strip()
    if not GPU_UUID_RE.fullmatch(text):
        return None
    return "GPU-" + text[4:].lower()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    architecture: str
    hidden_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    attention_heads: int
    index_heads: int
    index_head_dim: int
    index_topk: int
    num_layers: int
    max_position_embeddings: int
    moe_intermediate_size: int
    routed_experts_full_model: int
    experts_per_token: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def sparse_mla_qk_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


MODEL_SPECS: dict[str, ModelSpec] = {
    "deepseek_v32": ModelSpec(
        key="deepseek_v32",
        label="DeepSeek-V3.2",
        architecture="DeepseekV32ForCausalLM",
        hidden_size=7168,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        attention_heads=128,
        index_heads=64,
        index_head_dim=128,
        index_topk=2048,
        num_layers=61,
        max_position_embeddings=163840,
        moe_intermediate_size=2048,
        routed_experts_full_model=256,
        experts_per_token=8,
    ),
    "glm5": ModelSpec(
        key="glm5",
        label="GLM-5",
        architecture="GlmMoeDsaForCausalLM",
        hidden_size=6144,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        attention_heads=64,
        index_heads=32,
        index_head_dim=128,
        index_topk=2048,
        num_layers=78,
        max_position_embeddings=202752,
        moe_intermediate_size=2048,
        routed_experts_full_model=256,
        experts_per_token=8,
    ),
}


API_CONTRACTS = (
    {
        "stage": "indexer_and_topk",
        "module": "vllm.model_executor.layers.sparse_attn_indexer",
        "symbols": ["SparseAttnIndexer", "sparse_attn_indexer"],
        "source": "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py",
        "required_calls": [
            "ops.indexer_k_quant_and_cache",
            "ops.cp_gather_indexer_k_quant_cache",
            "fp8_fp4_mqa_logits",
            "ops.top_k_per_row_prefill",
        ],
    },
    {
        "stage": "sparse_mla",
        "module": "flashinfer.mla",
        "symbols": ["trtllm_batch_decode_with_kv_cache_mla"],
        "source": "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py",
        "required_kwargs": ["sparse_mla_top_k", "enable_pdl", "backend"],
    },
    {
        "stage": "mla_cache_insert",
        "module": "vllm._custom_ops",
        "symbols": ["concat_and_cache_mla"],
        "source": "/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py",
    },
    {
        "stage": "moe32",
        "module": "vllm.model_executor.layers.fused_moe",
        "symbols": ["fused_topk", "fused_experts"],
        "sources": [
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py",
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py",
        ],
    },
    {
        "stage": "indexer_pdl_control",
        "module": "vllm.third_party.deep_gemm",
        "symbols": ["set_pdl", "get_pdl"],
        "source": "/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/__init__.py",
    },
    {
        "stage": "indexer_logits_reference",
        "module": "vllm.utils.deep_gemm",
        "symbols": ["fp8_fp4_mqa_logits", "calc_diff"],
        "sources": [
            "/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py",
            str(DEEPGEMM_MQA_HEADER),
        ],
        "required_calls": ["clean_logits", "weighted ReLU", "fmaxf(accum[j], 0)"],
    },
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def canonical_json_sha(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def process_start_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = text.rfind(")")
    if close < 0:
        raise RuntimeError(f"malformed /proc stat for pid {pid}")
    fields_after_comm = text[close + 2 :].split()
    # fields_after_comm[0] is field 3 (state); starttime is field 22.
    value = int(fields_after_comm[19])
    if value <= 0:
        raise RuntimeError(f"invalid /proc start ticks for pid {pid}")
    return value


def paired_timing_schedule(
    repeats: int, components: Sequence[str]
) -> list[tuple[int, int, str, bool, str]]:
    """Frozen repeat/component schedule with adjacent, alternating PDL pairs."""
    events: list[tuple[int, int, str, bool, str]] = []
    event_ordinal = 0
    for repeat in range(repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        order_label = "off_then_on" if repeat % 2 == 0 else "on_then_off"
        for component in components:
            for enabled in order:
                events.append(
                    (event_ordinal, repeat, component, enabled, order_label)
                )
                event_ordinal += 1
    return events


def gib(value: int) -> float:
    return round(value / GIB, 6)


def query_chunk_tokens(seq: int, max_logits_mb: int, max_query_chunk: int) -> int:
    """Largest full-work query chunk respecting the FP32 logits budget."""
    by_logits = max(1, (max_logits_mb * MIB) // (seq * 4))
    return min(seq, max_query_chunk, by_logits)


def causal_pair_count(seq: int) -> int:
    """Exact number of query-key pairs in an inclusive causal sequence."""
    return seq * (seq + 1) // 2


def causal_pairs_for_chunk(start: int, count: int) -> int:
    """Pairs for query rows ``start`` through ``start + count - 1``."""
    return count * (2 * start + count + 1) // 2


def chunk_causal_pair_sum(seq: int, chunk: int) -> int:
    """Sum the causal pair partition used by the streamed execution."""
    return sum(
        causal_pairs_for_chunk(start, min(chunk, seq - start))
        for start in range(0, seq, chunk)
    )


def long_context_execution_contract() -> dict[str, Any]:
    """Frozen, correctness-preserving long-context execution policy."""
    return {
        "attention_reference": {
            "row_batch": ATTENTION_REFERENCE_ROW_BATCH,
            "scope": "all_query_rows_all_sparse_attention_output_elements",
            "sampling": "NONE",
        },
        "logits_quality": {
            "query_row_batch": LOGITS_QUALITY_QUERY_BATCH,
            "reduction_dtype": "float64",
            "reduction": "streamed_query_rows_all_causal_valid_cells",
            "calc_diff_formula": "1-2*sum(x*y)/sum(x*x+y*y)",
            "row_calc_diff_formula": "1-2*sum_j(x_j*y_j)/sum_j(x_j*x_j+y_j*y_j)",
            "invalid_logits_contract": TOPK_TAIL_CONTRACT,
            "sampling": "NONE",
        },
        "cuda_cache_release": {
            "cadence_chunks": CUDA_CACHE_RELEASE_CADENCE_CHUNKS,
            "operation": "explicit_del_tensor_locals_then_torch.cuda.empty_cache",
            "timing_scope": "after_correctness_warmup_and_all_timed_events",
            "inside_cuda_event_timing": False,
        },
    }


def streamed_logits_quality_statistics(
    torch_module,
    kernel_logits,
    manual_scores,
    valid_logits_mask,
    *,
    query_row_batch: int = LOGITS_QUALITY_QUERY_BATCH,
) -> dict[str, Any]:
    """Full-cell calc-diff and absolute-error statistics with bounded FP64 memory.

    The formulas are the same as ``vllm.utils.deep_gemm.calc_diff`` and the
    pre-v2 per-row checks.  Only the reduction partition changes: every valid
    cell is consumed in deterministic contiguous query-row batches.  Inputs
    remain read-only so one manual oracle can be shared by the off/on replays.
    """
    torch = torch_module
    if query_row_batch <= 0:
        raise ValueError("query_row_batch must be positive")
    if (
        kernel_logits.ndim != 2
        or manual_scores.shape != kernel_logits.shape
        or valid_logits_mask.shape != kernel_logits.shape
    ):
        raise ValueError("logits and validity mask must have one identical 2D shape")
    if valid_logits_mask.dtype != torch.bool:
        raise ValueError("valid_logits_mask must be boolean")

    device = kernel_logits.device
    global_denominator = torch.zeros((), dtype=torch.float64, device=device)
    global_numerator = torch.zeros((), dtype=torch.float64, device=device)
    absolute_sum = torch.zeros((), dtype=torch.float64, device=device)
    absolute_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    manual_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    kernel_nonfinite = torch.zeros((), dtype=torch.int64, device=device)
    manual_nonfinite = torch.zeros((), dtype=torch.int64, device=device)
    row_calc_diff_parts = []
    absolute_max_parts = []
    valid_elements = 0

    for row_start in range(0, kernel_logits.shape[0], query_row_batch):
        row_end = min(kernel_logits.shape[0], row_start + query_row_batch)
        local_mask = valid_logits_mask[row_start:row_end]
        local_kernel = kernel_logits[row_start:row_end]
        local_manual = manual_scores[row_start:row_end]

        kernel_valid = local_kernel.masked_select(local_mask)
        manual_valid = local_manual.masked_select(local_mask)
        valid_elements += kernel_valid.numel()
        kernel_nonfinite.add_((~torch.isfinite(kernel_valid)).sum())
        manual_nonfinite.add_((~torch.isfinite(manual_valid)).sum())
        # Every reported quality moment, not only its accumulator, is evaluated in
        # FP64.  Keeping this conversion scoped to eight query rows bounds memory while
        # making ``quality_reduction_dtype=float64`` literal and unambiguous.
        kernel_valid64 = kernel_valid.double()
        manual_valid64 = manual_valid.double()
        local_absolute64 = (kernel_valid64 - manual_valid64).abs()
        if local_absolute64.numel():
            absolute_max_parts.append(local_absolute64.max())
            absolute_sum.add_(local_absolute64.sum())
            absolute_square_sum.add_(
                local_absolute64.square().sum()
            )
            manual_square_sum.add_(
                manual_valid64.square().sum()
            )
        del kernel_valid, manual_valid
        del kernel_valid64, manual_valid64, local_absolute64

        kernel64 = local_kernel.double()
        manual64 = local_manual.double()
        kernel64.masked_fill_(~local_mask, 0.0)
        manual64.masked_fill_(~local_mask, 0.0)
        row_denominator = (
            kernel64.square().sum(dim=1) + manual64.square().sum(dim=1)
        )
        row_numerator = 2 * (kernel64 * manual64).sum(dim=1)
        global_denominator.add_(row_denominator.sum())
        global_numerator.add_(row_numerator.sum())
        row_similarity = torch.where(
            row_denominator == 0,
            torch.ones_like(row_denominator),
            row_numerator / row_denominator,
        )
        row_calc_diff_parts.append(1 - row_similarity)
        del kernel64, manual64, row_denominator, row_numerator, row_similarity

    if valid_elements == 0:
        raise ValueError("logits quality requires at least one valid cell")
    row_calc_diff = torch.cat(row_calc_diff_parts)
    logits_calc_diff = 1 - global_numerator / global_denominator
    max_abs_diff = torch.stack(absolute_max_parts).max()
    row_quality_failure_mask = (
        ~torch.isfinite(row_calc_diff)
        | (row_calc_diff >= DEEPGEMM_ROW_CALC_DIFF_LIMIT)
    )
    return {
        "valid_elements": valid_elements,
        "kernel_valid_nonfinite": int(kernel_nonfinite.item()),
        "manual_valid_nonfinite": int(manual_nonfinite.item()),
        "calc_diff": float(logits_calc_diff.item()),
        "row_calc_diff_max": float(row_calc_diff.max().item()),
        "row_calc_diff_p99": float(torch.quantile(row_calc_diff, 0.99).item()),
        "row_quality_failures": int(row_quality_failure_mask.sum().item()),
        "max_abs_diff": float(max_abs_diff.item()),
        "mean_abs_diff": float((absolute_sum / valid_elements).item()),
        "rms_abs_diff": float(
            torch.sqrt(absolute_square_sum / valid_elements).item()
        ),
        "manual_rms": float(torch.sqrt(manual_square_sum / valid_elements).item()),
    }


OPERATOR_VALIDATION_SCOPE = (
    "all_rows_valid_topk_prefix_and_all_attention_elements"
)


def seal_operator_chain_chunk_correctness(
    evidence: dict[str, Any],
    *,
    chunk_index: int,
    start: int,
    count: int,
) -> dict[str, Any]:
    """Assemble an operator-chain chunk without legacy flat mode evidence.

    This CPU-testable boundary is shared by the real runtime emitter and the
    schema regression tests, so a hand-written fixture cannot hide drift in
    where mode-specific evidence is placed.
    """
    legacy_flat_fields = {
        "topk_diagnostics",
        "topk_mismatches",
        "attention_elements_checked",
        "layer_output_elements_checked",
        "validation_scope",
        "indexer_calls",
    }
    unexpected = sorted(legacy_flat_fields.intersection(evidence))
    if unexpected:
        raise RuntimeError(
            "operator correctness contains legacy flat mode evidence: "
            + ",".join(unexpected)
        )
    mode_records = evidence.get("mode_correctness")
    if not isinstance(mode_records, list):
        raise RuntimeError("operator correctness mode records are missing")
    by_mode = {
        record.get("pdl_mode"): record
        for record in mode_records
        if isinstance(record, dict)
    }
    if (
        len(mode_records) != len(PDL_MODES)
        or len(by_mode) != len(mode_records)
        or set(by_mode) != set(PDL_MODES)
    ):
        raise RuntimeError("operator correctness requires exact off/on mode records")
    sealed_modes: list[dict[str, Any]] = []
    for mode in PDL_MODES:
        record = dict(by_mode[mode])
        existing_scope = record.get("validation_scope")
        if existing_scope not in (None, OPERATOR_VALIDATION_SCOPE):
            raise RuntimeError(
                f"operator correctness mode={mode} has invalid validation scope"
            )
        record["validation_scope"] = OPERATOR_VALIDATION_SCOPE
        sealed_modes.append(record)
    return {
        **evidence,
        "mode_correctness": sealed_modes,
        "chunk_index": chunk_index,
        "query_start": start,
        "query_count": count,
        "indexer_workload_geometry": "exact_causal_lower_triangle",
        "first_query_causal_key_count": start + 1,
        "last_query_causal_key_count": start + count,
        "indexer_causal_pairs_executed": causal_pairs_for_chunk(start, count),
        "query_sampling": "NONE",
        "causal_pair_sampling": "NONE",
    }


def valid_topk_entry_count(start: int, count: int, topk: int) -> int:
    """Valid causal top-k prefix slots across a contiguous query chunk."""
    growing_rows = min(count, max(0, topk - start))
    growing_sum = growing_rows * (2 * start + growing_rows + 1) // 2
    return growing_sum + (count - growing_rows) * topk


def attention_weight_elements(spec: ModelSpec) -> dict[str, int]:
    """Exact random matrices used by the attention-only layer algebra."""
    h, lq, lkv = spec.hidden_size, spec.q_lora_rank, spec.kv_lora_rank
    n, p, r, v = (
        spec.attention_heads,
        spec.qk_nope_head_dim,
        spec.qk_rope_head_dim,
        spec.v_head_dim,
    )
    ih, idim = spec.index_heads, spec.index_head_dim
    return {
        "fused_qkv_a": h * (lq + lkv + r),
        "q_b": lq * n * (p + r),
        "w_uk": lkv * n * p,
        "w_uv": lkv * n * v,
        "o_proj": n * v * h,
        "indexer_wq_b": lq * ih * idim,
        "indexer_wk_weights": h * (idim + ih),
    }


def shape_record(
    spec: ModelSpec,
    seq: int,
    max_logits_mb: int,
    max_query_chunk: int,
    moe_experts: int,
    moe_tokens: int,
) -> dict[str, Any]:
    chunk = query_chunk_tokens(seq, max_logits_mb, max_query_chunk)
    causal_pairs = causal_pair_count(seq)
    partition_pairs = chunk_causal_pair_sum(seq, chunk)
    logits = chunk * seq * 4
    index_cache = seq * INDEX_CACHE_BYTES_PER_TOKEN
    mla_cache = seq * MLA_CACHE_BYTES_PER_TOKEN_BF16
    indexer_workspace_capacity = 40 * seq * INDEX_CACHE_BYTES_PER_TOKEN
    layer_weights = sum(attention_weight_elements(spec).values()) * 2
    working = {
        "indexer_cache": index_cache,
        "indexer_k_history_bf16": seq * 128 * 2,
        "indexer_slot_mapping_i64": seq * 8,
        "indexer_token_to_seq_i32": seq * 4,
        "mla_bf16_cache": mla_cache,
        "indexer_workspace_capacity": indexer_workspace_capacity,
        "flashinfer_workspace": FLASHINFER_WORKSPACE_BYTES,
        "fp32_logits_chunk": logits,
        "index_q_fp8_chunk": chunk * spec.index_heads * spec.index_head_dim,
        "index_weights_fp32_chunk": chunk * spec.index_heads * 4,
        "topk_indices_i32_chunk": chunk * spec.index_topk * 4,
        "mla_query_bf16_chunk":
            chunk * spec.attention_heads * spec.sparse_mla_qk_dim * 2,
        "mla_output_bf16_chunk":
            chunk * spec.attention_heads * spec.kv_lora_rank * 2,
        "hidden_bf16_chunk": chunk * spec.hidden_size * 2,
        "attention_layer_weights_bf16": layer_weights,
        "indexshare_four_layer_weights_bf16": 4 * layer_weights,
        "indexshare_four_layer_mla_caches_bf16": 4 * mla_cache,
        "moe32_weights_bf16":
            moe_experts
            * 3
            * spec.hidden_size
            * spec.moe_intermediate_size
            * 2,
        "moe_tokens_bf16": moe_tokens * spec.hidden_size * 2,
        "moe_router_fp32_base_and_epoch_delta": moe_tokens * moe_experts * 4 * 2,
    }
    return {
        "model": spec.key,
        "label": spec.label,
        "seq": seq,
        "query_chunk_tokens": chunk,
        "num_query_chunks": math.ceil(seq / chunk),
        "indexer_workload_geometry": "exact_causal_lower_triangle",
        "all_query_rows": seq,
        "query_sampling": "NONE",
        "indexer_causal_pairs": causal_pairs,
        "indexer_causal_pair_formula": "S*(S+1)/2",
        "causal_pair_sampling": "NONE",
        "chunk_causal_pair_formula": "count*(2*start+count+1)/2",
        "chunk_causal_pairs_sum": partition_pairs,
        "chunk_pair_partition_verified": partition_pairs == causal_pairs,
        "indexer_causal_fma_flops": 2
        * causal_pairs
        * spec.index_heads
        * spec.index_head_dim,
        "official_max_position_embeddings": spec.max_position_embeddings,
        "within_official_position_range": seq <= spec.max_position_embeddings,
        "extreme": seq == 1048576,
        "extreme_reason": (
            "one_million_tokens_outside_official_position_range"
            if seq == 1048576
            else None
        ),
        "tensor_bytes": working,
        "tensor_gib": {name: gib(value) for name, value in working.items()},
        "memory_accounting_note": (
            "explicit resident tensors and named workspaces; not an allocator-peak claim"
        ),
    }


def expected_matrix(
    models: Sequence[str], seqs: Sequence[int], workloads: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for seq in seqs:
            for workload in workloads:
                rows.append(
                    {
                        "row_id": f"{model}.{workload}.seq{seq}",
                        "model": model,
                        "seq": seq,
                        "workload": workload,
                        "pdl_modes": list(PDL_MODES),
                    }
                )
        rows.append(
            {
                "row_id": f"{model}.moe32",
                "model": model,
                "seq": None,
                "workload": "moe32",
                "pdl_modes": ["framework_default_uncontrolled"],
            }
        )
    return rows


def canonical_row_seed(row: dict[str, Any], campaign_seed: int) -> int:
    """Return the frozen full-canonical seed, independent of subset selection."""
    model = str(row["model"])
    model_ordinal = tuple(MODEL_SPECS).index(model)
    workload = str(row["workload"])
    base = campaign_seed + model_ordinal * 1_000_000
    if workload == "moe32":
        return base + 900_000
    seq = int(row["seq"])
    seq_ordinal = FORMAL_SEQS.index(seq)
    offsets = {
        "operator_chain": 1000,
        "single_layer": 2000,
        "indexshare_fsss": 3000,
    }
    return base + seq_ordinal * 100_000 + offsets[workload]


def source_record(path_text: str, *, hash_binary: bool = False) -> dict[str, Any]:
    path = Path(path_text)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": None,
    }
    if path.is_file() and (
        hash_binary or path.suffix in {".py", ".sh", ".json", ".cuh"}
    ):
        record["sha256"] = sha256_file(path)
    return record


def distribution_record(name: str) -> dict[str, Any]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "installed": False, "version": None}
    direct_url_path = Path(dist._path) / "direct_url.json"  # type: ignore[attr-defined]
    direct_url: object | None = None
    if direct_url_path.is_file():
        try:
            direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            direct_url = {"unreadable": True}
    return {
        "name": name,
        "installed": True,
        "version": dist.version,
        "metadata_path": str(dist._path),  # type: ignore[attr-defined]
        "direct_url": direct_url,
    }


def package_manifest(hash_binaries: bool) -> dict[str, Any]:
    artifacts = [
        "/usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so",
        "/usr/local/lib/python3.12/dist-packages/vllm/_flashmla_C.abi3.so",
        "/usr/local/lib/python3.12/dist-packages/vllm/_flashmla_extension_C.abi3.so",
        "/usr/local/lib/python3.12/dist-packages/vllm/_moe_C.abi3.so",
        "/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so",
    ]
    api_sources = sorted(
        {
            path
            for contract in API_CONTRACTS
            for path in (
                [contract["source"]]
                if "source" in contract
                else contract.get("sources", [])
            )
        }
        | {
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py",
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py",
        }
    )
    return {
        "distributions": [
            distribution_record(name)
            for name in (
                "vllm",
                "torch",
                "flashinfer-python",
                "flashinfer-cubin",
                "triton",
                "tilelang",
                "transformers",
            )
        ],
        "artifacts": [source_record(path, hash_binary=hash_binaries) for path in artifacts],
        "api_sources": [source_record(path) for path in api_sources],
        "binary_hash_policy": (
            "sha256_during_explicit_gpu_execution"
            if hash_binaries
            else "size_only_in_cpu_dry_run_to_avoid_large_io"
        ),
    }


def static_api_checks() -> list[dict[str, Any]]:
    """Bind declared symbols to the installed Python sources without imports."""
    checks: list[dict[str, Any]] = []
    for contract in API_CONTRACTS:
        paths = (
            [contract["source"]]
            if "source" in contract
            else list(contract.get("sources", []))
        )
        source_text = "\n".join(
            Path(path).read_text(encoding="utf-8", errors="replace")
            for path in paths
            if Path(path).is_file()
        )
        tokens = list(contract.get("symbols", []))
        tokens.extend(contract.get("required_calls", []))
        tokens.extend(contract.get("required_kwargs", []))
        missing = [token for token in tokens if token not in source_text]
        checks.append(
            {
                "stage": contract["stage"],
                "sources": paths,
                "tokens": tokens,
                "missing_tokens": missing,
                "status": "PASS" if not missing and len(paths) > 0 else "FAIL",
            }
        )
    return checks


def local_source_manifest() -> list[dict[str, Any]]:
    base = Path(__file__).resolve().parent
    paths = [
        Path(__file__).resolve(),
        base / "validate_production_tier5.py",
        base / "validate_production_tier5_compact.py",
        base / "run_production_tier5.sh",
        base / "production_tier5_campaign.py",
        base / "run_production_tier5_fragments.sh",
        base / "run_production_tier5_nsys_sidecar.sh",
        base / "gpu_exclusivity.py",
        Path("/workspace/benchmarks/attention_benchmarks/mla_runner.py"),
        Path("/workspace/benchmarks/attention_benchmarks/common.py"),
        Path("/workspace/benchmarks/kernels/benchmark_moe.py"),
    ]
    return [source_record(str(path)) for path in paths]


def correctness_contract() -> dict[str, Any]:
    return {
        "validated_pdl_modes": list(PDL_MODES),
        "per_mode_untimed_full_reference": True,
        "mode_specific_actual_paths": [
            "SparseAttnIndexer/top-k",
            "FlashInfer sparse MLA",
            "attention layer output when applicable",
        ],
        "mode_specific_native_logits_replay": True,
        "manual_logits_reference_reuse": (
            "computed_once_per_chunk_and_shared_read_only_across_off_on"
        ),
        "topk_valid_prefix": (
            "all min(causal_key_count,index_topk) entries per query"
        ),
        "topk_tail": TOPK_TAIL_CONTRACT,
        "topk_reference": (
            "full-N vllm.utils.deep_gemm.fp8_fp4_mqa_logits(clean_logits=False), "
            "then causal-valid-cell mask"
        ),
        "manual_logits_reference": (
            "sum_h(relu(q_fp8_dequant@k_fp8_dequant.T)*weights)"
        ),
        "manual_logits_quality_metric": (
            "streamed_fp64_formula_equivalent_to_vllm.utils.deep_gemm.calc_diff"
        ),
        "manual_logits_quality_reduction": {
            "query_row_batch": LOGITS_QUALITY_QUERY_BATCH,
            "scope": "all_causal_valid_cells",
            "sampling": "NONE",
            "input_mutation": False,
        },
        "attention_reference_row_batch": ATTENTION_REFERENCE_ROW_BATCH,
        "attention_reference_sampling": "NONE",
        "manual_logits_calc_diff_limit_exclusive": DEEPGEMM_CALC_DIFF_LIMIT,
        "manual_logits_row_calc_diff_limit_exclusive":
            DEEPGEMM_ROW_CALC_DIFF_LIMIT,
        "exact_set_difference_role": "diagnostic_not_acceptance",
        "formula_authority": {
            "path": str(DEEPGEMM_MQA_HEADER),
            "expected_sha256": DEEPGEMM_MQA_HEADER_SHA256,
            "sha256": (
                sha256_file(DEEPGEMM_MQA_HEADER)
                if DEEPGEMM_MQA_HEADER.is_file()
                else None
            ),
            "weighted_relu_lines": "357-374",
        },
    }


def experiment_contract() -> dict[str, Any]:
    return {
        "coordinates": {
            "dimensions": {
                "A1": ["adjacent_indexer_topk_mla", "span_4_indexshare_FSSS"],
                "A2": ["high_degree_contiguous_indexer_row", "historical_kv_indirection"],
            },
            "decision": (
                "whether production DSA stages justify a future CTA wrapper and which "
                "stage dominates; this harness cannot choose a CTA mechanism"
            ),
            "gpu_budget": {
                "cpu_static_validation_minutes": 0,
                "formal_execution_budget": "UNBOUNDED_BY_HARNESS",
                "execution_control": (
                    "externally scheduled under the UUID-scoped global lease; the "
                    "runner never truncates a formal matrix to satisfy a time budget"
                ),
            },
        },
        "rungs": {
            "pdl_off_diagnostic": {
                "available": True,
                "formal_rung": False,
                "controls": ["DeepGEMM.set_pdl(False)", "FlashInfer enable_pdl=False"],
            },
            "floor": {
                "status": "PARTIAL",
                "available_components": ["indexer_deepgemm", "sparse_mla_flashinfer"],
                "missing_components": ["topk_explicit_pdl_control", "worker_binary_proof"],
                "label": "production_pdl_on_diagnostic",
            },
            "cta_impl": {
                "status": "PARTIAL",
                "available": False,
                "reason": "installed production APIs expose no identity-preserving CTA readiness path",
            },
            "ceiling": {
                "status": "PARTIAL",
                "available": False,
                "reason": "closed production APIs expose no truly unordered dependency-free chain",
            },
            "ideal": {"status": "NOT_IMPLEMENTED"},
        },
        "tier5_bracket_admitted": False,
        "headroom_defined": False,
        "headroom_pct": None,
        "claim_scope": "production_kernel_characterization_only",
        "workload_semantics": {
            "operator_chain": "preprojected full-work SparseAttnIndexer/top-k/sparse-MLA",
            "single_layer": (
                "attention-only DeepSeek/GLM algebra: random exact-shape projections, "
                "RMSNorm, RoPE, production MLA cache insertion, coherent causal "
                "indexer/KV history, production DSA primitives, W_UV and output "
                "projection; no decoder residual or MLP"
            ),
            "indexshare_fsss": (
                "four causally coherent attention-only layers with exactly one production "
                "indexer call and the same logical top-k reused by FSSS"
            ),
            "moe32": "production fused_topk plus fused_experts, 32 resident experts, top-8",
        },
        "benchmark_reuse": {
            "source": "/workspace/benchmarks/attention_benchmarks/mla_runner.py",
            "reused_contracts": [
                "FlashInfer sparse MLA tensor layout",
                "paged BF16 KV cache shape",
                "production backend invocation dimensions",
            ],
            "replaced_component": "MockIndexer/fill_random_indices",
            "replacement": "vLLM SparseAttnIndexer with DeepGEMM logits and vLLM top-k",
            "mock_indexer_used": False,
        },
        "pdl_controls": {
            "indexer": {
                "api": "vllm.third_party.deep_gemm.set_pdl/get_pdl",
                "readback_required": True,
            },
            "topk": {"api": None, "status": "UNAVAILABLE"},
            "sparse_mla": {
                "api": "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(enable_pdl=bool)",
                "backend": "trtllm-gen",
            },
        },
        "statistics": {
            "minimum_timed_repeats": 31,
            "default_warmup": 5,
            "pairing": "off/on adjacent in one process; order alternates by repeat",
            "reported": ["median", "bootstrap_95pct_ci", "paired_median_delta_ci"],
        },
        "correctness": {
            "timed_repeat_poison": True,
            "untimed_reference": (
                "off and on independently for all query rows, every API-defined "
                "valid top-k prefix entry, all MLA outputs, and all applicable "
                "layer outputs"
            ),
            "mode_coverage": "exactly off and on for every attention chunk",
            "manual_reference_reuse": (
                "one read-only manual logits oracle per chunk; native replay, "
                "actual indexer/top-k, MLA, and layer outputs remain mode-specific"
            ),
            "indexer_reference": (
                "FP8-cache-dequantized FP32 row scores; valid/range/uniqueness and "
                "kth-score-threshold completeness are mandatory, with exact top-k set "
                "difference retained as a diagnostic for boundary ties"
            ),
            "attention_reference": "FP32 gathered sparse softmax/value for every output element",
            "moe_reference": "all token/expert assignments and full BF16 expert algebra",
            "failure_policy": "any mismatch invalidates every sample in the invocation",
        },
    }


def make_manifest(args: argparse.Namespace, *, runtime_device: dict[str, Any] | None) -> dict[str, Any]:
    model_specs = [asdict(MODEL_SPECS[key]) for key in args.models]
    shapes = [
        shape_record(
            MODEL_SPECS[model],
            seq,
            args.max_logits_mb,
            args.max_query_chunk,
            args.moe_experts,
            args.moe_tokens,
        )
        for model in args.models
        for seq in args.seqs
    ]
    argv = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *args.invocation_argv,
    ]
    matrix = expected_matrix(args.models, args.seqs, args.workloads)
    fragment = None
    if args.fragment_row_id is not None:
        fragment = {
            "row_id": args.fragment_row_id,
            "ordinal": args.fragment_ordinal,
            "expected_row_count": len(matrix),
            "row": matrix[args.fragment_ordinal],
            "campaign_contract_sha256": args.campaign_contract_sha256,
            "campaign_fingerprint_sha256": args.campaign_fingerprint_sha256,
            "execution_segment_id": args.execution_segment_id,
            "invocation_uuid": args.execution_segment_id,
            "derived_row_seed": canonical_row_seed(
                matrix[args.fragment_ordinal], args.seed
            ),
        }
    return {
        "schema": SCHEMA,
        "kind": "tier5_production_dsa_manifest",
        "created_unix_ns": time.time_ns(),
        "mode": "execute_gpu" if args.execute_gpu else "cpu_dry_run",
        "status": "RUNNING" if args.execute_gpu else "NOT_EXECUTED",
        "accepted_timing": 0,
        "accepted_timing_semantics": "legacy_CTA_bracket_only",
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "measurement_emitted": False,
        "random_seed": args.seed,
        "random_weights": True,
        "backend": args.backend,
        "required_device_substring": args.required_device_substring,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "expected_gpu_index": args.expected_gpu_index,
        "models": model_specs,
        "seqs": list(args.seqs),
        "workloads": list(args.workloads),
        "shape_records": shapes,
        "expected_matrix": matrix,
        "execution_scope": (
            "row_fragment" if fragment is not None else "full_matrix_cpu_plan"
        ),
        "fragment": fragment,
        "moe": {
            "experts": args.moe_experts,
            "topk": args.moe_topk,
            "tokens": args.moe_tokens,
            "full_model_expert_count": 256,
            "reduced_expert_claim": "dependency_shape_only",
        },
        "chunking": {
            "max_logits_mb": args.max_logits_mb,
            "max_query_chunk": args.max_query_chunk,
            "indexer_workload_geometry": "exact_causal_lower_triangle",
            "all_query_rows": True,
            "query_sampling": "NONE",
            "causal_pair_sampling": "NONE",
        },
        "long_context_execution": long_context_execution_contract(),
        "correctness_contract": correctness_contract(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "allow_short": args.allow_short,
        "formal_statistics_requested": args.repeats >= 31 and not args.allow_short,
        "experiment_contract": experiment_contract(),
        "api_contracts": list(API_CONTRACTS),
        "static_api_checks": static_api_checks(),
        "packages": package_manifest(hash_binaries=args.execute_gpu),
        "sources": local_source_manifest(),
        "argv": argv,
        "argv_sha256": sha256_bytes("\0".join(argv).encode("utf-8")),
        "publication": {
            "execution_output_dir": str(Path(args.output_dir).resolve()),
            "requested_publish_target": str(
                Path(args.publish_target or args.output_dir).resolve()
            ),
            "failure_atomic_stage": bool(args.publish_target),
            "runner_managed_stage": args.runner_managed_stage,
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": os.environ.get(
                "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", str(args.max_logits_mb)
            ),
            "TIER5_PRODUCTION_GPU_ALLOWED": os.environ.get(
                "TIER5_PRODUCTION_GPU_ALLOWED"
            ),
        },
        "device": runtime_device
        if runtime_device is not None
        else {
            "query_performed": False,
            "reason": "CPU dry-run contract forbids CUDA/device initialization",
            "required_compute_capability_major": 10,
        },
    }


def bootstrap_ci(
    samples: Sequence[float], *, seed: int, draws: int = 4000
) -> tuple[float, float]:
    if not samples:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    n = len(samples)
    values = sorted(
        statistics.median(samples[rng.randrange(n)] for _ in range(n))
        for _ in range(draws)
    )
    return values[int(0.025 * (draws - 1))], values[int(0.975 * (draws - 1))]


def summarize_samples(samples: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        key = (sample["row_id"], sample["component"], sample["pdl_mode"])
        grouped.setdefault(key, []).append(sample)
    summaries: list[dict[str, Any]] = []
    for ordinal, (key, records) in enumerate(sorted(grouped.items())):
        values = [float(record["elapsed_ms"]) for record in records]
        lo, hi = bootstrap_ci(values, seed=seed + ordinal)
        summaries.append(
            {
                "row_id": key[0],
                "component": key[1],
                "pdl_mode": key[2],
                "sample_count": len(values),
                "median_ms": statistics.median(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "bootstrap_95pct_ci_ms": [lo, hi],
            }
        )

    by_pair: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for (row_id, component, mode), records in grouped.items():
        by_pair.setdefault((row_id, component), {})[mode] = records
    for ordinal, (key, modes) in enumerate(sorted(by_pair.items()), start=len(summaries)):
        if set(modes) != {"off", "on"}:
            continue
        off = {int(record["repeat"]): float(record["elapsed_ms"]) for record in modes["off"]}
        on = {int(record["repeat"]): float(record["elapsed_ms"]) for record in modes["on"]}
        if set(off) != set(on):
            raise ValueError(f"unpaired PDL samples for {key}")
        deltas = [on[i] - off[i] for i in sorted(off)]
        lo, hi = bootstrap_ci(deltas, seed=seed + ordinal)
        summaries.append(
            {
                "row_id": key[0],
                "component": key[1],
                "comparison": "pdl_on_minus_pdl_off_diagnostic",
                "sample_count": len(deltas),
                "paired_median_delta_ms": statistics.median(deltas),
                "paired_bootstrap_95pct_ci_ms": [lo, hi],
                "formal_tier5_headroom": False,
            }
        )
    return summaries


class ProductionRuntime:
    """Late-imported CUDA runtime.

    No name from torch, vLLM, FlashInfer, or DeepGEMM is imported until the
    double GPU guard has passed.
    """

    def __init__(self, args: argparse.Namespace):
        if not args.execute_gpu or os.environ.get("TIER5_PRODUCTION_GPU_ALLOWED") != "1":
            raise RuntimeError(
                "GPU execution requires --execute-gpu and TIER5_PRODUCTION_GPU_ALLOWED=1"
            )
        visible_device_selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", visible_device_selector) is None:
            raise RuntimeError(
                "production runner requires one numeric CUDA_VISIBLE_DEVICES "
                "selector resolved by the UUID-scoped lease"
            )
        if (
            args.expected_gpu_index is None
            or args.expected_gpu_index < 0
            or visible_device_selector != str(args.expected_gpu_index)
        ):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES does not match the canonical physical "
                "index resolved by the UUID-scoped lease"
            )

        import torch
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
        from vllm import _custom_ops as ops
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.forward_context import ForwardContext, override_forward_context
        from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.platforms import current_platform
        from vllm.third_party import deep_gemm
        from vllm.utils.deep_gemm import calc_diff, fp8_fp4_mqa_logits
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadata,
            DeepseekV32IndexerPrefillChunkMetadata,
            DeepseekV32IndexerPrefillMetadata,
        )
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            init_workspace_manager,
        )

        self.args = args
        self.torch = torch
        self.ops = ops
        self.vllm_config = VllmConfig()
        self.set_current_vllm_config = set_current_vllm_config
        self.ForwardContext = ForwardContext
        self.override_forward_context = override_forward_context
        self.fused_experts = fused_experts
        self.fused_topk = fused_topk
        self.FusedMoEQuantConfig = FusedMoEQuantConfig
        self.per_token_group_quant_fp8 = per_token_group_quant_fp8
        self.SparseAttnIndexer = SparseAttnIndexer
        self.current_platform = current_platform
        self.deep_gemm = deep_gemm
        self.calc_diff = calc_diff
        self.fp8_fp4_mqa_logits = fp8_fp4_mqa_logits
        self.trtllm_mla = trtllm_batch_decode_with_kv_cache_mla
        self.DeepseekV32IndexerMetadata = DeepseekV32IndexerMetadata
        self.DeepseekV32IndexerPrefillChunkMetadata = (
            DeepseekV32IndexerPrefillChunkMetadata
        )
        self.DeepseekV32IndexerPrefillMetadata = DeepseekV32IndexerPrefillMetadata

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "production runner requires exactly one lease-scoped visible GPU"
            )
        self.device = torch.device("cuda", 0)
        prop = torch.cuda.get_device_properties(self.device)
        if prop.major != 10:
            raise RuntimeError(
                f"production sparse MLA requires compute capability major 10, got {prop.major}.{prop.minor}"
            )
        if args.required_device_substring not in prop.name:
            raise RuntimeError(
                f"device name {prop.name!r} does not contain required substring "
                f"{args.required_device_substring!r}"
            )
        expected_uuid = canonical_gpu_uuid(args.expected_gpu_uuid or "")
        if expected_uuid is None:
            raise RuntimeError(
                "GPU execution requires a canonical --expected-gpu-uuid from the "
                "UUID-scoped exclusivity lease"
            )
        runtime_uuid: str | None = None
        uuid_source: str | None = None
        property_uuid = getattr(prop, "uuid", None)
        if property_uuid:
            candidate = canonical_gpu_uuid(property_uuid)
            if candidate is not None:
                runtime_uuid = candidate
                uuid_source = "torch_device_properties"
        if runtime_uuid is None:
            try:
                raw_uuids = torch.cuda._raw_device_uuid_nvml()
                physical_index = torch.cuda._get_nvml_device_index(self.device)
                candidate = (
                    canonical_gpu_uuid(raw_uuids[physical_index])
                    if raw_uuids
                    else None
                )
            except (AttributeError, IndexError, RuntimeError, TypeError):
                candidate = None
            if candidate is not None:
                runtime_uuid = candidate
                uuid_source = "torch_nvml_runtime_ordinal_mapping"
        if runtime_uuid != expected_uuid:
            raise RuntimeError(
                f"runtime GPU UUID mismatch: expected={expected_uuid}, got={runtime_uuid}"
            )
        init_workspace_manager(self.device)
        self.workspace_manager = current_workspace_manager()
        self.flashinfer_workspace = torch.zeros(
            FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        get_driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
        driver_version = (
            int(get_driver_version()) if callable(get_driver_version) else None
        )
        self.device_manifest = {
            "query_performed": True,
            "name": prop.name,
            "uuid": runtime_uuid,
            "uuid_source": uuid_source,
            "runtime_ordinal": 0,
            "runtime_ordinal_zero": True,
            "compute_capability": f"{prop.major}.{prop.minor}",
            "total_memory_bytes": prop.total_memory,
            "multi_processor_count": prop.multi_processor_count,
            "driver_version_raw": driver_version,
            "torch_cuda_version": torch.version.cuda,
            "device_index_inside_visible_set": 0,
            "cuda_visible_devices_selector": visible_device_selector,
            "process_pid": os.getpid(),
            "process_start_ticks": process_start_ticks(os.getpid()),
        }

    def set_pdl(self, enabled: bool) -> dict[str, Any]:
        self.deep_gemm.set_pdl(enabled)
        readback = bool(self.deep_gemm.get_pdl())
        if readback != enabled:
            raise RuntimeError(
                f"DeepGEMM PDL readback mismatch: requested={enabled}, got={readback}"
            )
        return {
            "requested": enabled,
            "deep_gemm_readback": readback,
            "flashinfer_enable_pdl": enabled,
            "topk_control": None,
        }

    def elapsed_ms(self, function) -> float:
        torch = self.torch
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        value = float(start.elapsed_time(stop))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"invalid CUDA event duration {value}")
        return value

    def release_completed_chunk_cache(self, completed_chunks: int) -> None:
        """Release dead chunk allocations only after every timed event ends."""
        if completed_chunks <= 0:
            raise ValueError("completed_chunks must be positive")
        if completed_chunks % CUDA_CACHE_RELEASE_CADENCE_CHUNKS == 0:
            self.torch.cuda.empty_cache()

    def _rng(self, seed: int):
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    def randn(self, shape: tuple[int, ...], seed: int, dtype=None):
        torch = self.torch
        return torch.randn(
            shape,
            generator=self._rng(seed),
            device=self.device,
            dtype=dtype or torch.bfloat16,
        )

    def rms_norm(self, value, eps: float):
        torch = self.torch
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        return (value.float() * torch.rsqrt(variance + eps)).to(value.dtype)

    def rope(self, q, k, positions, rope_dim: int):
        """Shape-preserving interleaved RoPE reference used by random layers."""
        torch = self.torch
        inv = 1.0 / (
            10000.0
            ** (
                torch.arange(0, rope_dim, 2, device=self.device, dtype=torch.float32)
                / rope_dim
            )
        )
        angles = positions.float().unsqueeze(-1) * inv
        cos = angles.cos()
        sin = angles.sin()

        def rotate(value):
            original_shape = value.shape
            paired = value.float().reshape(*original_shape[:-1], rope_dim // 2, 2)
            a, b = paired[..., 0], paired[..., 1]
            while cos.ndim < a.ndim:
                local_cos = cos.unsqueeze(-2)
                local_sin = sin.unsqueeze(-2)
                cos_view, sin_view = local_cos, local_sin
                break
            else:
                cos_view, sin_view = cos, sin
            out = torch.stack(
                [a * cos_view - b * sin_view, a * sin_view + b * cos_view], dim=-1
            )
            return out.reshape(original_shape).to(value.dtype)

        return rotate(q), rotate(k)

    def make_indexer_state(
        self,
        spec: ModelSpec,
        seq: int,
        chunk: int,
        seed: int,
        *,
        prefill_random_history: bool = True,
    ):
        torch = self.torch
        num_blocks = math.ceil(seq / INDEX_BLOCK_SIZE)
        index_cache = torch.empty(
            num_blocks,
            INDEX_BLOCK_SIZE,
            INDEX_CACHE_BYTES_PER_TOKEN,
            dtype=torch.uint8,
            device=self.device,
        )
        topk_buffer = torch.empty(
            chunk, spec.index_topk, dtype=torch.int32, device=self.device
        )
        cache_holder = type("ProductionIndexerCache", (), {})()
        cache_holder.prefix = f"production.{spec.key}.indexer.k_cache"
        cache_holder.kv_cache = index_cache
        with self.set_current_vllm_config(self.vllm_config):
            indexer = self.SparseAttnIndexer(
                cache_holder,
                128,
                "ue8m0",
                spec.index_topk,
                spec.index_head_dim,
                seq,
                40 * seq,
                topk_buffer,
                skip_k_cache_insert=prefill_random_history,
            )

        fp8_dtype = self.current_platform.fp8_dtype()
        k_quant_full, k_scale_full = self.workspace_manager.get_simultaneous(
            ((40 * seq, spec.index_head_dim), fp8_dtype),
            ((40 * seq, 4), torch.uint8),
        )
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        cu_seq_lens = torch.tensor([0, seq], dtype=torch.int32, device=self.device)
        if prefill_random_history:
            k_history = self.randn((seq, spec.index_head_dim), seed + 1)
            slot_mapping = torch.arange(seq, dtype=torch.int64, device=self.device)
            self.ops.indexer_k_quant_and_cache(
                k_history, index_cache, slot_mapping, 128, "ue8m0"
            )
            self.ops.cp_gather_indexer_k_quant_cache(
                index_cache,
                k_quant_full[:seq],
                k_scale_full[:seq],
                block_table,
                cu_seq_lens,
            )
        else:
            index_cache.zero_()
            k_quant_full.zero_()
            k_scale_full.zero_()
        return {
            "cache_holder": cache_holder,
            "index_cache": index_cache,
            "indexer": indexer,
            "topk": topk_buffer,
            "k_quant": k_quant_full[:seq],
            "k_scale_bytes": k_scale_full[:seq],
            "block_table": block_table,
            "cu_seq_lens": cu_seq_lens,
            "token_to_seq": torch.zeros(seq, dtype=torch.int32, device=self.device),
            "prefill_random_history": prefill_random_history,
        }

    def make_indexer_metadata(self, state, seq: int, start: int, count: int):
        torch = self.torch
        prefilled = state["prefill_random_history"]
        history_tokens = seq if prefilled else start + count
        cu_ks = torch.zeros(count, dtype=torch.int32, device=self.device)
        cu_ke = torch.arange(
            start + 1, start + count + 1, dtype=torch.int32, device=self.device
        )
        chunk = self.DeepseekV32IndexerPrefillChunkMetadata(
            block_table=state["block_table"],
            cu_seqlen_ks=cu_ks,
            cu_seqlen_ke=cu_ke,
            cu_seq_lens=(
                state["cu_seq_lens"]
                if prefilled
                else torch.tensor(
                    [0, history_tokens], dtype=torch.int32, device=self.device
                )
            ),
            token_to_seq=state["token_to_seq"][:history_tokens],
            total_seq_lens=history_tokens,
            token_start=0,
            token_end=count,
            num_reqs=1,
            skip_kv_gather=prefilled,
        )
        return self.DeepseekV32IndexerMetadata(
            seq_lens=torch.tensor([seq], dtype=torch.int32, device=self.device),
            max_seq_len=seq,
            slot_mapping=(
                torch.empty(0, dtype=torch.int64, device=self.device)
                if prefilled
                else torch.arange(
                    start,
                    start + count,
                    dtype=torch.int64,
                    device=self.device,
                )
            ),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            num_prefill_tokens=count,
            prefill=self.DeepseekV32IndexerPrefillMetadata([chunk]),
            decode=None,
        )

    def run_indexer(self, state, metadata, hidden_sentinel, q_fp8, k, weights):
        context = self.ForwardContext(
            no_compile_layers={},
            attn_metadata={state["cache_holder"].prefix: metadata},
            slot_mapping={},
        )
        with self.override_forward_context(context):
            return state["indexer"](hidden_sentinel, q_fp8, k, weights)

    def topk_valid_prefix_is_invalid(self, topk, start: int) -> bool:
        """Check only API-defined causal prefix slots; tail is unspecified."""
        torch = self.torch
        count, width = topk.shape
        valid_counts = torch.minimum(
            torch.arange(
                start + 1,
                start + count + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            torch.tensor(width, dtype=torch.int32, device=self.device),
        )
        columns = torch.arange(width, device=self.device).unsqueeze(0)
        valid_mask = columns < valid_counts.unsqueeze(1)
        causal_limits = torch.arange(
            start + 1, start + count + 1, device=self.device
        ).unsqueeze(1)
        return bool(
            (
                valid_mask
                & ((topk < 0) | (topk >= causal_limits))
            ).any()
        )

    def run_sparse_mla(
        self,
        spec: ModelSpec,
        query,
        kv_cache,
        indices,
        valid_counts,
        output,
        enable_pdl: bool,
    ):
        return self.trtllm_mla(
            query=query.unsqueeze(1),
            kv_cache=kv_cache,
            workspace_buffer=self.flashinfer_workspace,
            qk_nope_head_dim=spec.qk_nope_head_dim,
            kv_lora_rank=spec.kv_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            block_tables=indices.unsqueeze(1),
            seq_lens=valid_counts,
            max_seq_len=spec.index_topk,
            sparse_mla_top_k=spec.index_topk,
            out=output,
            bmm1_scale=spec.qk_head_dim**-0.5,
            bmm2_scale=1.0,
            enable_pdl=enable_pdl,
            backend="trtllm-gen",
        )

    def indexer_reference(
        self,
        state,
        q_fp8,
        weights,
        start: int,
        topk,
        *,
        pdl_mode: str,
        manual_reference=None,
    ):
        """Validate one real PDL mode against a mode-specific native replay.

        The expensive FP32 weighted-ReLU oracle is immutable and may be reused
        by the off/on checks for the same chunk.  The production indexer output
        and native DeepGEMM replay are nevertheless executed and checked once
        under each mode.
        """
        torch = self.torch
        if pdl_mode not in PDL_MODES:
            raise ValueError(f"invalid PDL correctness mode {pdl_mode!r}")
        expected_enabled = pdl_mode == "on"
        if bool(self.deep_gemm.get_pdl()) != expected_enabled:
            raise RuntimeError(
                f"native replay PDL control mismatch for mode={pdl_mode}"
            )

        history_tokens = start + q_fp8.shape[0]
        replay_tokens = (
            state["k_quant"].shape[0]
            if state["prefill_random_history"]
            else history_tokens
        )
        k_quant = state["k_quant"][:replay_tokens]
        k_scales = (
            state["k_scale_bytes"][:replay_tokens]
            .contiguous()
            .view(torch.float32)
            .reshape(-1)
        )
        count, heads, _ = q_fp8.shape
        row_starts = torch.zeros(count, dtype=torch.int32, device=self.device)
        row_ends = torch.arange(
            start + 1,
            start + count + 1,
            dtype=torch.int32,
            device=self.device,
        )

        if manual_reference is None:
            k = k_quant.float() * k_scales.unsqueeze(1)
            qf = q_fp8.float()
            wf = weights.float()
            manual_scores = torch.zeros(
                count, replay_tokens, dtype=torch.float32, device=self.device
            )
            for head in range(heads):
                head_scores = qf[:, head] @ k.T
                head_scores.relu_()
                manual_scores.add_(head_scores * wf[:, head].unsqueeze(1))
            positions = torch.arange(replay_tokens, device=self.device).unsqueeze(0)
            limits = row_ends.long().unsqueeze(1)
            valid_logits_mask = positions < limits
            manual_reference = {
                "start": start,
                "count": count,
                "replay_tokens": replay_tokens,
                "topk_width": int(topk.shape[1]),
                "q_fp8_binding": q_fp8,
                "weights_binding": weights,
                "manual_scores": manual_scores,
                "valid_logits_mask": valid_logits_mask,
                "q_fp8_abs_max": float(qf.abs().max().item()),
                "k_dequant_abs_max": float(k.abs().max().item()),
                "weight_abs_max": float(wf.abs().max().item()),
            }
            del head_scores, qf, wf, k, positions, limits
        else:
            binding = (
                manual_reference.get("start") == start
                and manual_reference.get("count") == count
                and manual_reference.get("replay_tokens") == replay_tokens
                and manual_reference.get("topk_width") == int(topk.shape[1])
                and bool(
                    torch.equal(manual_reference.get("q_fp8_binding"), q_fp8)
                )
                and bool(
                    torch.equal(manual_reference.get("weights_binding"), weights)
                )
            )
            if not binding:
                raise RuntimeError(
                    "shared manual indexer reference input binding mismatch"
                )

        manual_scores = manual_reference["manual_scores"]
        valid_logits_mask = manual_reference["valid_logits_mask"]
        kernel_logits = self.fp8_fp4_mqa_logits(
            (q_fp8, None),
            (k_quant, k_scales),
            weights,
            row_starts,
            row_ends,
            clean_logits=False,
        )
        quality_statistics = streamed_logits_quality_statistics(
            torch,
            kernel_logits,
            manual_scores,
            valid_logits_mask,
            query_row_batch=LOGITS_QUALITY_QUERY_BATCH,
        )
        kernel_valid_nonfinite = quality_statistics["kernel_valid_nonfinite"]
        manual_valid_nonfinite = quality_statistics["manual_valid_nonfinite"]
        logits_calc_diff = quality_statistics["calc_diff"]
        row_quality_failures = quality_statistics["row_quality_failures"]
        logits_quality_failure = int(
            kernel_valid_nonfinite != 0
            or manual_valid_nonfinite != 0
            or not math.isfinite(logits_calc_diff)
            or logits_calc_diff >= DEEPGEMM_CALC_DIFF_LIMIT
            or row_quality_failures != 0
        )

        # The mode-specific kernel output is no longer needed unmasked.  Mask
        # it in place only after the full-cell quality pass so the shared
        # manual oracle remains immutable and no third full logits matrix is
        # allocated.
        kernel_logits.masked_fill_(~valid_logits_mask, -torch.inf)
        canonical_kernel_logits = kernel_logits

        valid_counts = torch.minimum(
            row_ends,
            torch.tensor(topk.shape[1], dtype=torch.int32, device=self.device),
        )
        ref = torch.full_like(topk, -1)
        max_k = min(topk.shape[1], start + count)
        ref_values, values = torch.topk(
            canonical_kernel_logits, max_k, dim=1, sorted=True
        )
        ref[:, :max_k] = values.to(torch.int32)
        columns = torch.arange(max_k, device=self.device).unsqueeze(0)
        ref[:, :max_k].masked_fill_(columns >= valid_counts.unsqueeze(1), -1)
        column_ids = torch.arange(
            topk.shape[1], dtype=topk.dtype, device=self.device
        ).unsqueeze(0)
        valid_mask = column_ids < valid_counts.unsqueeze(1)
        limits = row_ends.long().unsqueeze(1)
        invalid_valid_entries = int(
            (valid_mask & ((topk < 0) | (topk >= limits))).sum().item()
        )
        tail_slots_ignored = int((~valid_mask).sum().item())
        tail_nonminus_one_observed = int(
            ((~valid_mask) & (topk != -1)).sum().item()
        )

        # Ignore padding while proving that each selected token identity is
        # unique.  Padding receives distinct sentinels above the sequence.
        padded_unique = torch.where(valid_mask, topk, replay_tokens + column_ids)
        sorted_unique = torch.sort(padded_unique, dim=1).values
        duplicate_entries = int(
            (
                (sorted_unique[:, 1:] == sorted_unique[:, :-1])
                & (sorted_unique[:, 1:] < replay_tokens)
            ).sum().item()
        )

        actual_gather = canonical_kernel_logits.gather(
            1, topk.clamp(min=0, max=replay_tokens - 1).long()
        )
        actual_gather.masked_fill_(~valid_mask, torch.inf)
        actual_min = actual_gather.min(dim=1).values
        threshold = ref_values.gather(
            1, (valid_counts.long() - 1).unsqueeze(1)
        ).squeeze(1)
        tolerance = torch.zeros_like(threshold)
        score_violations = int((actual_min < threshold).sum().item())
        score_violation_mask = actual_min < threshold
        score_margin = actual_min - threshold

        # Diagnostic only: compute the true symmetric difference between the
        # two valid-prefix sets.  Tail slots have no API contract and never
        # participate in either correctness acceptance or this diagnostic.
        if invalid_valid_entries == 0 and duplicate_entries == 0:
            sentinels = replay_tokens + column_ids
            actual_set_sorted = torch.sort(
                torch.where(valid_mask, topk, sentinels), dim=1
            ).values
            ref_set_sorted = torch.sort(
                torch.where(valid_mask, ref, sentinels), dim=1
            ).values
            insertion = torch.searchsorted(ref_set_sorted, actual_set_sorted)
            insertion_clamped = insertion.clamp(max=topk.shape[1] - 1)
            found = (
                ref_set_sorted.gather(1, insertion_clamped)
                == actual_set_sorted
            ) & (column_ids < valid_counts.unsqueeze(1))
            intersection = found.sum(dim=1)
            per_row_symmetric_difference = 2 * (
                valid_counts.to(torch.int64) - intersection
            )
            valid_set_symmetric_difference = int(
                per_row_symmetric_difference.sum().item()
            )
            rows_with_set_symmetric_difference = int(
                (per_row_symmetric_difference != 0).sum().item()
            )
        else:
            per_row_symmetric_difference = torch.full(
                (count,), -1, dtype=torch.int64, device=self.device
            )
            valid_set_symmetric_difference = -1
            rows_with_set_symmetric_difference = -1
        quantile_points = torch.tensor(
            [0.0, 0.01, 0.5, 0.99, 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        margin_quantiles = torch.quantile(score_margin.float(), quantile_points)
        violation_rows = torch.nonzero(score_violation_mask).flatten()
        violation_examples = []
        for row in violation_rows[:8].tolist():
            valid = int(valid_counts[row].item())
            violation_examples.append(
                {
                    "row": int(row),
                    "absolute_query_position": start + int(row),
                    "valid_count": valid,
                    "actual_min": float(actual_min[row].item()),
                    "reference_threshold": float(threshold[row].item()),
                    "tolerance": float(tolerance[row].item()),
                    "actual_minus_threshold": float(score_margin[row].item()),
                    "valid_set_symmetric_difference": int(
                        per_row_symmetric_difference[row].item()
                    ),
                    "actual_indices_head": [
                        int(value)
                        for value in topk[row, : min(valid, 8)].tolist()
                    ],
                    "reference_indices_head": [
                        int(value)
                        for value in ref[row, : min(valid, 8)].tolist()
                    ],
                }
            )
        diagnostics = {
            "pdl_mode": pdl_mode,
            "invalid_valid_entries": invalid_valid_entries,
            "topk_tail_contract": TOPK_TAIL_CONTRACT,
            "tail_slots_ignored": tail_slots_ignored,
            "tail_nonminus_one_observed": tail_nonminus_one_observed,
            "duplicate_entries": duplicate_entries,
            "score_reference": "deepgemm_full_n_logits_causal_masked",
            "score_violations": score_violations,
            "score_violation_rows_first_half": int(
                score_violation_mask[: count // 2].sum().item()
            ),
            "score_violation_rows_second_half": int(
                score_violation_mask[count // 2 :].sum().item()
            ),
            "score_margin_quantiles": {
                name: float(value)
                for name, value in zip(
                    ("min", "p01", "median", "p99", "max"),
                    margin_quantiles.tolist(),
                )
            },
            "score_scale": {
                "q_fp8_abs_max": manual_reference["q_fp8_abs_max"],
                "k_dequant_abs_max": manual_reference["k_dequant_abs_max"],
                "weight_abs_max": manual_reference["weight_abs_max"],
                "threshold_abs_max": float(threshold.abs().max().item()),
            },
            "score_violation_examples": violation_examples,
            "valid_set_symmetric_difference": valid_set_symmetric_difference,
            "rows_with_valid_set_symmetric_difference":
                rows_with_set_symmetric_difference,
            "exact_set_difference_role": "diagnostic_not_acceptance",
            "indexer_logits_quality": {
                "actual_source": (
                    "full-N vllm.utils.deep_gemm.fp8_fp4_mqa_logits("
                    "clean_logits=False), then causal-valid-cell mask"
                ),
                "native_replay_pdl_mode": pdl_mode,
                "native_replay_calls": 1,
                "replay_total_seq_lens": replay_tokens,
                "invalid_logits_contract": "UNSPECIFIED_IGNORED",
                "manual_reference": (
                    "sum_h(relu(q_fp8_dequant@k_fp8_dequant.T)*weights)"
                ),
                "manual_reference_reuse": (
                    "computed_once_per_chunk_and_shared_read_only_across_off_on"
                ),
                "quality_reduction": "streamed_query_rows_all_causal_valid_cells",
                "quality_query_row_batch": LOGITS_QUALITY_QUERY_BATCH,
                "quality_reduction_dtype": "float64",
                "valid_elements": quality_statistics["valid_elements"],
                "kernel_valid_nonfinite": kernel_valid_nonfinite,
                "manual_valid_nonfinite": manual_valid_nonfinite,
                "calc_diff": logits_calc_diff,
                "calc_diff_limit_exclusive": DEEPGEMM_CALC_DIFF_LIMIT,
                "row_calc_diff_limit_exclusive":
                    DEEPGEMM_ROW_CALC_DIFF_LIMIT,
                "row_calc_diff_max": quality_statistics["row_calc_diff_max"],
                "row_calc_diff_p99": quality_statistics["row_calc_diff_p99"],
                "row_quality_failures": row_quality_failures,
                "max_abs_diff": quality_statistics["max_abs_diff"],
                "mean_abs_diff": quality_statistics["mean_abs_diff"],
                "rms_abs_diff": quality_statistics["rms_abs_diff"],
                "manual_rms": quality_statistics["manual_rms"],
                "status": "PASS" if logits_quality_failure == 0 else "FAIL",
            },
            "logits_quality_failure": logits_quality_failure,
            "acceptance_mismatches": invalid_valid_entries
            + duplicate_entries
            + score_violations
            + logits_quality_failure,
        }
        return ref, valid_counts, diagnostics, manual_reference

    def attention_reference(
        self, spec: ModelSpec, query, kv_cache, indices, valid_counts
    ):
        torch = self.torch
        cache = kv_cache.reshape(-1, spec.sparse_mla_qk_dim)
        count = query.shape[0]
        reference = torch.empty(
            count,
            1,
            spec.attention_heads,
            spec.kv_lora_rank,
            dtype=torch.bfloat16,
            device=self.device,
        )
        scale = spec.qk_head_dim**-0.5
        # Bound the reference workspace while checking every row and element.
        # Thirty-two GLM rows need about 144 MiB for gathered FP32 KV at
        # top-k 2048, reducing launch/Python overhead eightfold versus v1.
        columns = torch.arange(indices.shape[1], device=self.device).view(1, 1, -1)
        for row_start in range(0, count, ATTENTION_REFERENCE_ROW_BATCH):
            row_end = min(count, row_start + ATTENTION_REFERENCE_ROW_BATCH)
            local_indices = indices[row_start:row_end]
            local_valid_counts = valid_counts[row_start:row_end]
            valid_index_mask = (
                torch.arange(indices.shape[1], device=self.device).unsqueeze(0)
                < local_valid_counts.unsqueeze(1)
            )
            safe_indices = torch.where(
                valid_index_mask, local_indices, torch.zeros_like(local_indices)
            )
            selected = cache[safe_indices.long()].float()
            q = query[row_start:row_end].float()
            logits = torch.einsum("rhd,rkd->rhk", q, selected) * scale
            valid = local_valid_counts.view(-1, 1, 1)
            logits.masked_fill_(columns >= valid, -torch.inf)
            probs = torch.softmax(logits, dim=-1)
            out = torch.einsum(
                "rhk,rkl->rhl", probs, selected[..., : spec.kv_lora_rank]
            )
            reference[row_start:row_end, 0] = out.to(torch.bfloat16)
        return reference

    def compact_tensor_digest(self, value) -> dict[str, Any]:
        """Small output binding; avoids copying full 1M validation tensors."""
        torch = self.torch
        fp = value.float()
        metrics = torch.stack(
            (
                fp.sum(),
                fp.square().sum(),
                fp.abs().max(),
                torch.isfinite(fp).sum().float(),
            )
        )
        payload = metrics.cpu().numpy().tobytes()
        return {
            "sha256": sha256_bytes(payload),
            "sum": float(metrics[0].item()),
            "sum_squares": float(metrics[1].item()),
            "abs_max": float(metrics[2].item()),
            "finite_elements": int(metrics[3].item()),
        }

    def validate_chain_chunk(
        self,
        spec: ModelSpec,
        state,
        metadata,
        start: int,
        hidden,
        q_fp8,
        k,
        weights,
        mla_query,
        mla_cache,
    ) -> dict[str, Any]:
        torch = self.torch
        manual_reference = None
        mode_correctness: list[dict[str, Any]] = []
        for mode in PDL_MODES:
            enabled = mode == "on"
            control = self.set_pdl(enabled)
            state["topk"].fill_(-31337 if not enabled else -31338)
            topk = self.run_indexer(
                state, metadata, hidden, q_fp8, k, weights
            )[: hidden.shape[0]].clone()
            ref_topk, valid_counts, topk_diagnostics, manual_reference = (
                self.indexer_reference(
                    state,
                    q_fp8,
                    weights,
                    start,
                    topk,
                    pdl_mode=mode,
                    manual_reference=manual_reference,
                )
            )
            topk_mismatches = topk_diagnostics["acceptance_mismatches"]
            if topk_mismatches:
                raise RuntimeError(
                    f"indexer correctness failure model={spec.key} start={start} "
                    f"mode={mode}: topk_mismatches={topk_mismatches}, "
                    f"topk_diagnostics={json.dumps(topk_diagnostics, sort_keys=True)}"
                )
            physical = topk.clone()
            output = torch.full(
                (
                    hidden.shape[0],
                    1,
                    spec.attention_heads,
                    spec.kv_lora_rank,
                ),
                float("nan"),
                dtype=torch.bfloat16,
                device=self.device,
            )
            actual = self.run_sparse_mla(
                spec,
                mla_query,
                mla_cache,
                physical,
                valid_counts,
                output,
                enabled,
            )
            reference = self.attention_reference(
                spec, mla_query, mla_cache, physical, valid_counts
            )
            diff = (actual.float() - reference.float()).abs()
            max_abs = float(diff.max().item())
            denominator = reference.float().abs().clamp_min(1e-3)
            max_rel = float((diff / denominator).max().item())
            attention_ok = bool(
                torch.allclose(
                    actual, reference, rtol=0.08, atol=0.08, equal_nan=False
                )
            )
            if not attention_ok:
                raise RuntimeError(
                    f"correctness failure model={spec.key} start={start} "
                    f"mode={mode}: topk_mismatches={topk_mismatches}, "
                    f"max_abs={max_abs}, max_rel={max_rel}, topk_diagnostics="
                    f"{json.dumps(topk_diagnostics, sort_keys=True)}"
                )
            readback_after = bool(self.deep_gemm.get_pdl())
            if readback_after != enabled:
                raise RuntimeError(
                    f"PDL control changed during correctness mode={mode}"
                )
            mode_correctness.append(
                {
                    "pdl_mode": mode,
                    "status": "PASS",
                    "control": {
                        **control,
                        "deep_gemm_readback_after_validation": readback_after,
                    },
                    "actual_indexer_calls": 1,
                    "native_replay_calls": 1,
                    "sparse_mla_calls": 1,
                    "manual_reference_shared_read_only": True,
                    "topk_mismatches": topk_mismatches,
                    "topk_diagnostics": topk_diagnostics,
                    "topk_valid_elements_checked": int(valid_counts.sum().item()),
                    "topk_output_slots_observed": int(topk.numel()),
                    "topk_tail_slots_ignored": int(
                        topk.numel() - valid_counts.sum().item()
                    ),
                    "topk_tail_contract": TOPK_TAIL_CONTRACT,
                    "attention_elements_checked": int(actual.numel()),
                    "attention_max_abs": max_abs,
                    "attention_max_rel": max_rel,
                    "reference_topk_digest": self.compact_tensor_digest(ref_topk),
                    "actual_output_digest": self.compact_tensor_digest(actual),
                }
            )
        return {
            "manual_indexer_reference": {
                "computation_count": 1,
                "shared_read_only_across_modes": True,
                "modes_using_reference": list(PDL_MODES),
            },
            "mode_correctness": mode_correctness,
        }

    def make_chain_inputs(
        self, spec: ModelSpec, count: int, seed: int
    ) -> dict[str, Any]:
        torch = self.torch
        q = self.randn(
            (count, spec.index_heads, spec.index_head_dim), seed + 1
        ).contiguous()
        flat_q = q.reshape(-1, spec.index_head_dim)
        q_fp8, q_scale = self.per_token_group_quant_fp8(
            flat_q, 128, column_major_scales=False, use_ue8m0=True
        )
        q_fp8 = q_fp8.reshape(count, spec.index_heads, spec.index_head_dim)
        q_scale = q_scale.reshape(count, spec.index_heads)
        base_weights = self.randn((count, spec.index_heads), seed + 2)
        weights = (
            base_weights.float()
            * q_scale.float()
            * spec.index_head_dim**-0.5
            * spec.index_heads**-0.5
        )
        return {
            "hidden": torch.empty(count, 1, dtype=torch.bfloat16, device=self.device),
            "q_fp8": q_fp8,
            "k": self.randn((count, spec.index_head_dim), seed + 3),
            "weights": weights,
            "mla_query": self.randn(
                (count, spec.attention_heads, spec.sparse_mla_qk_dim), seed + 4
            ),
        }

    def benchmark_operator_chain(
        self, spec: ModelSpec, seq: int, chunk: int, seed: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        torch = self.torch
        state = self.make_indexer_state(spec, seq, chunk, seed)
        num_blocks = math.ceil(seq / INDEX_BLOCK_SIZE)
        mla_cache = self.randn(
            (num_blocks, INDEX_BLOCK_SIZE, spec.sparse_mla_qk_dim), seed + 100
        )
        totals = {
            component: {mode: [0.0] * self.args.repeats for mode in PDL_MODES}
            for component in ("indexer_topk", "sparse_mla", "chain_total")
        }
        correctness: list[dict[str, Any]] = []
        for chunk_index, start in enumerate(range(0, seq, chunk)):
            count = min(chunk, seq - start)
            inputs = self.make_chain_inputs(spec, count, seed + 1000 + chunk_index * 17)
            metadata = self.make_indexer_metadata(state, seq, start, count)

            evidence = seal_operator_chain_chunk_correctness(
                self.validate_chain_chunk(
                    spec,
                    state,
                    metadata,
                    start,
                    inputs["hidden"],
                    inputs["q_fp8"],
                    inputs["k"],
                    inputs["weights"],
                    inputs["mla_query"],
                    mla_cache,
                ),
                chunk_index=chunk_index,
                start=start,
                count=count,
            )
            correctness.append(evidence)

            def index_call():
                self.run_indexer(
                    state,
                    metadata,
                    inputs["hidden"],
                    inputs["q_fp8"],
                    inputs["k"],
                    inputs["weights"],
                )

            # Establish correct top-k for the standalone MLA samples.
            self.set_pdl(True)
            index_call()
            physical = state["topk"][:count].clone()
            valid_counts = torch.minimum(
                torch.arange(
                    start + 1,
                    start + count + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.tensor(spec.index_topk, dtype=torch.int32, device=self.device),
            )
            output = torch.empty(
                count,
                1,
                spec.attention_heads,
                spec.kv_lora_rank,
                dtype=torch.bfloat16,
                device=self.device,
            )

            def mla_call(enabled: bool):
                self.run_sparse_mla(
                    spec,
                    inputs["mla_query"],
                    mla_cache,
                    physical,
                    valid_counts,
                    output,
                    enabled,
                )

            def chain_call(enabled: bool):
                index_call()
                local_indices = state["topk"][:count]
                self.run_sparse_mla(
                    spec,
                    inputs["mla_query"],
                    mla_cache,
                    local_indices,
                    valid_counts,
                    output,
                    enabled,
                )

            for _ in range(self.args.warmup):
                for enabled in (False, True):
                    self.set_pdl(enabled)
                    index_call()
                    mla_call(enabled)
                    chain_call(enabled)
            torch.cuda.synchronize()

            for _, repeat, component, enabled, _ in paired_timing_schedule(
                self.args.repeats,
                ("indexer_topk", "sparse_mla", "chain_total"),
            ):
                mode = "on" if enabled else "off"
                control = self.set_pdl(enabled)
                if component == "indexer_topk":
                    state["topk"].fill_(-100000 - repeat)
                    totals["indexer_topk"][mode][repeat] += self.elapsed_ms(index_call)
                    if self.topk_valid_prefix_is_invalid(
                        state["topk"][:count], start
                    ):
                        raise RuntimeError(
                            "indexer timed sample left an invalid valid-prefix entry"
                        )

                elif component == "sparse_mla":
                    output.fill_(float("nan"))
                    totals["sparse_mla"][mode][repeat] += self.elapsed_ms(
                        lambda enabled=enabled: mla_call(enabled)
                    )
                    if not bool(torch.isfinite(output).all()):
                        raise RuntimeError(
                            "sparse MLA timed sample did not overwrite its poison output"
                        )

                elif component == "chain_total":
                    state["topk"].fill_(-200000 - repeat)
                    output.fill_(float("nan"))
                    totals["chain_total"][mode][repeat] += self.elapsed_ms(
                        lambda enabled=enabled: chain_call(enabled)
                    )
                    if self.topk_valid_prefix_is_invalid(
                        state["topk"][:count], start
                    ) or not bool(torch.isfinite(output).all()):
                        raise RuntimeError(
                            "chain timed sample left an invalid valid-prefix/output buffer"
                        )
                else:
                    raise AssertionError(component)
                if control["deep_gemm_readback"] != enabled:
                    raise RuntimeError("PDL control changed during timed sample")

            # All correctness, warmup, and timed work for this chunk is now
            # complete.  Break closure references before the deterministic
            # cache release; neither deletion nor empty_cache is CUDA-timed.
            del chain_call, mla_call, index_call
            del output, valid_counts, physical, metadata, inputs
            self.release_completed_chunk_cache(chunk_index + 1)

        row_id = f"{spec.key}.operator_chain.seq{seq}"
        samples = []
        schedule = paired_timing_schedule(
            self.args.repeats,
            ("indexer_topk", "sparse_mla", "chain_total"),
        )
        for event_ordinal, repeat, component, enabled, pair_order in schedule:
            mode = "on" if enabled else "off"
            samples.append(
                {
                    "schema": SCHEMA,
                    "row_id": row_id,
                    "model": spec.key,
                    "seq": seq,
                    "workload": "operator_chain",
                    "component": component,
                    "pdl_mode": mode,
                    "repeat": repeat,
                    "elapsed_ms": totals[component][mode][repeat],
                    "poison_epoch": repeat + 1,
                    "poison_verified": True,
                    "timed_validation": False,
                    "timing_event_ordinal": event_ordinal,
                    "timing_pair_ordinal": event_ordinal // 2,
                    "pair_order": pair_order,
                    "pair_same_process_pid": os.getpid(),
                    "pair_same_process_start_ticks": process_start_ticks(
                        os.getpid()
                    ),
                }
            )
        return samples, correctness

    def random_layer_weights(self, spec: ModelSpec, seed: int):
        h, lq, lkv = spec.hidden_size, spec.q_lora_rank, spec.kv_lora_rank
        n, p, r, v = (
            spec.attention_heads,
            spec.qk_nope_head_dim,
            spec.qk_rope_head_dim,
            spec.v_head_dim,
        )
        ih, idim = spec.index_heads, spec.index_head_dim
        scale = 1.0 / math.sqrt(h)
        return {
            "fused_qkv_a": self.randn((h, lq + lkv + r), seed + 1) * scale,
            "q_b": self.randn((lq, n * (p + r)), seed + 2) / math.sqrt(lq),
            "w_uk": self.randn((lkv, n, p), seed + 3) / math.sqrt(lkv),
            "w_uv": self.randn((lkv, n, v), seed + 4) / math.sqrt(lkv),
            "o_proj": self.randn((n * v, h), seed + 5) / math.sqrt(n * v),
            "index_wq": self.randn((lq, ih * idim), seed + 6) / math.sqrt(lq),
            "index_wk_weights": self.randn((h, idim + ih), seed + 7) * scale,
        }

    def layer_preprocess(self, spec: ModelSpec, weights, hidden, positions):
        torch = self.torch
        lq, lkv, r = spec.q_lora_rank, spec.kv_lora_rank, spec.qk_rope_head_dim
        qkv = hidden @ weights["fused_qkv_a"]
        q_c, kv = qkv.split([lq, lkv + r], dim=-1)
        q_c = self.rms_norm(q_c, 1e-6)
        kv_c, k_pe = kv.split([lkv, r], dim=-1)
        kv_c = self.rms_norm(kv_c, 1e-6)
        q_uncompressed = (q_c @ weights["q_b"]).view(
            -1, spec.attention_heads, spec.qk_head_dim
        )
        q_nope, q_pe = q_uncompressed.split(
            [spec.qk_nope_head_dim, r], dim=-1
        )
        q_pe, k_pe_rope = self.rope(q_pe, k_pe, positions, r)
        ql_nope = torch.einsum("mhp,lhp->mhl", q_nope, weights["w_uk"])
        mla_query = torch.cat([ql_nope, q_pe], dim=-1).contiguous()

        index_q = (q_c @ weights["index_wq"]).view(
            -1, spec.index_heads, spec.index_head_dim
        )
        index_kw = hidden @ weights["index_wk_weights"]
        index_k, index_weights = index_kw.split(
            [spec.index_head_dim, spec.index_heads], dim=-1
        )
        index_k = self.rms_norm(index_k, 1e-6)
        iq_rope, iq_nope = index_q.split(
            [r, spec.index_head_dim - r], dim=-1
        )
        ik_rope, ik_nope = index_k.split([r, spec.index_head_dim - r], dim=-1)
        iq_rope, ik_rope = self.rope(iq_rope, ik_rope, positions, r)
        index_q = torch.cat([iq_rope, iq_nope], dim=-1).contiguous()
        index_k = torch.cat([ik_rope, ik_nope], dim=-1).contiguous()
        flat = index_q.view(-1, spec.index_head_dim)
        index_q_fp8, index_q_scale = self.per_token_group_quant_fp8(
            flat, 128, column_major_scales=False, use_ue8m0=True
        )
        index_q_fp8 = index_q_fp8.view(
            -1, spec.index_heads, spec.index_head_dim
        )
        index_q_scale = index_q_scale.view(-1, spec.index_heads)
        index_weights = (
            index_weights.float()
            * index_q_scale.float()
            * spec.index_head_dim**-0.5
            * spec.index_heads**-0.5
        )
        return {
            "kv_c": kv_c.contiguous(),
            "k_pe": k_pe_rope.contiguous(),
            "mla_query": mla_query,
            "index_q_fp8": index_q_fp8,
            "index_k": index_k,
            "index_weights": index_weights,
        }

    def layer_postprocess(self, spec: ModelSpec, weights, latent_output):
        torch = self.torch
        latent = latent_output.squeeze(1)
        expanded = torch.einsum("mhl,lhv->mhv", latent, weights["w_uv"])
        return expanded.reshape(expanded.shape[0], -1) @ weights["o_proj"]

    def benchmark_layer_like(
        self,
        spec: ModelSpec,
        seq: int,
        chunk: int,
        seed: int,
        layers: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Attention-only exact algebra for one layer or FSSS four layers."""
        torch = self.torch
        if layers not in (1, 4):
            raise ValueError("layers must be 1 or 4")
        workload = "single_layer" if layers == 1 else "indexshare_fsss"
        layer_weights = [
            self.random_layer_weights(spec, seed + 10000 * layer) for layer in range(layers)
        ]
        index_state = self.make_indexer_state(
            spec,
            seq,
            chunk,
            seed + 50000,
            prefill_random_history=False,
        )
        num_blocks = math.ceil(seq / INDEX_BLOCK_SIZE)
        layer_caches = [
            torch.zeros(
                num_blocks,
                INDEX_BLOCK_SIZE,
                spec.sparse_mla_qk_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(layers)
        ]
        layer_cache_scales = [
            torch.ones(1, dtype=torch.float32, device=self.device)
            for _ in range(layers)
        ]
        totals = {mode: [0.0] * self.args.repeats for mode in PDL_MODES}
        correctness: list[dict[str, Any]] = []

        for chunk_index, start in enumerate(range(0, seq, chunk)):
            count = min(chunk, seq - start)
            positions = torch.arange(
                start, start + count, dtype=torch.int64, device=self.device
            )
            original_hidden = self.randn(
                (count, spec.hidden_size), seed + 70000 + chunk_index
            )
            metadata = self.make_indexer_metadata(index_state, seq, start, count)
            valid_counts = torch.minimum(
                torch.arange(
                    start + 1,
                    start + count + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.tensor(spec.index_topk, dtype=torch.int32, device=self.device),
            )
            outputs = [
                torch.empty(
                    count,
                    1,
                    spec.attention_heads,
                    spec.kv_lora_rank,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                for _ in range(layers)
            ]
            call_counter = {"indexer": 0}

            def forward(
                enabled: bool,
                validate: bool = False,
                manual_indexer_reference=None,
            ):
                hidden = original_hidden
                physical = None
                validation = {
                    "topk_mismatches": 0,
                    "topk_valid_elements_checked": 0,
                    "topk_output_slots_observed": 0,
                    "topk_tail_slots_ignored": 0,
                    "topk_tail_contract": TOPK_TAIL_CONTRACT,
                    "attention_elements_checked": 0,
                    "layer_output_elements_checked": 0,
                    "attention_max_abs": 0.0,
                    "layer_output_max_abs": 0.0,
                }
                for layer in range(layers):
                    values = self.layer_preprocess(
                        spec, layer_weights[layer], hidden, positions
                    )
                    self.ops.concat_and_cache_mla(
                        values["kv_c"],
                        values["k_pe"],
                        layer_caches[layer],
                        positions,
                        "auto",
                        layer_cache_scales[layer],
                    )
                    if layer == 0:
                        call_counter["indexer"] += 1
                        self.run_indexer(
                            index_state,
                            metadata,
                            hidden,
                            values["index_q_fp8"],
                            values["index_k"],
                            values["index_weights"],
                        )
                        physical = index_state["topk"][:count]
                        if validate:
                            (
                                _,
                                reference_counts,
                                index_diagnostics,
                                manual_indexer_reference,
                            ) = self.indexer_reference(
                                index_state,
                                values["index_q_fp8"],
                                values["index_weights"],
                                start,
                                physical,
                                pdl_mode="on" if enabled else "off",
                                manual_reference=manual_indexer_reference,
                            )
                            if not bool(torch.equal(reference_counts, valid_counts)):
                                raise RuntimeError("layer indexer valid-count mismatch")
                            validation["topk_mismatches"] = index_diagnostics[
                                "acceptance_mismatches"
                            ]
                            validation["topk_diagnostics"] = index_diagnostics
                            validation["topk_valid_elements_checked"] = int(
                                valid_counts.sum().item()
                            )
                            validation["topk_output_slots_observed"] = int(
                                physical.numel()
                            )
                            validation["topk_tail_slots_ignored"] = int(
                                physical.numel() - valid_counts.sum().item()
                            )
                            if index_diagnostics["acceptance_mismatches"]:
                                raise RuntimeError(
                                    f"{workload} top-k reference mismatch at query_start={start}"
                                )
                    assert physical is not None
                    latent = self.run_sparse_mla(
                        spec,
                        values["mla_query"],
                        layer_caches[layer],
                        physical,
                        valid_counts,
                        outputs[layer],
                        enabled,
                    )
                    actual_hidden = self.layer_postprocess(
                        spec, layer_weights[layer], latent
                    )
                    if validate:
                        reference_latent = self.attention_reference(
                            spec,
                            values["mla_query"],
                            layer_caches[layer],
                            physical,
                            valid_counts,
                        )
                        latent_diff = (latent.float() - reference_latent.float()).abs()
                        latent_max = float(latent_diff.max().item())
                        validation["attention_max_abs"] = max(
                            validation["attention_max_abs"], latent_max
                        )
                        validation["attention_elements_checked"] += int(latent.numel())
                        if not bool(
                            torch.allclose(
                                latent,
                                reference_latent,
                                rtol=0.08,
                                atol=0.08,
                                equal_nan=False,
                            )
                        ):
                            raise RuntimeError(
                                f"{workload} sparse MLA reference mismatch "
                                f"layer={layer} query_start={start} max_abs={latent_max}"
                            )
                        reference_hidden = self.layer_postprocess(
                            spec, layer_weights[layer], reference_latent
                        )
                        output_diff = (
                            actual_hidden.float() - reference_hidden.float()
                        ).abs()
                        output_max = float(output_diff.max().item())
                        validation["layer_output_max_abs"] = max(
                            validation["layer_output_max_abs"], output_max
                        )
                        validation["layer_output_elements_checked"] += int(
                            actual_hidden.numel()
                        )
                        if not bool(
                            torch.allclose(
                                actual_hidden,
                                reference_hidden,
                                rtol=0.1,
                                atol=0.12,
                                equal_nan=False,
                            )
                        ):
                            raise RuntimeError(
                                f"{workload} layer output reference mismatch "
                                f"layer={layer} query_start={start} max_abs={output_max}"
                            )
                    hidden = actual_hidden
                return hidden, validation, manual_indexer_reference

            # Every chunk gets independent off/on execution and full reference
            # checks.  Only the mode-independent FP32 manual logits oracle is
            # shared read-only across the two checks.
            manual_indexer_reference = None
            mode_correctness: list[dict[str, Any]] = []
            for mode in PDL_MODES:
                enabled = mode == "on"
                control = self.set_pdl(enabled)
                call_counter["indexer"] = 0
                result, validation, manual_indexer_reference = forward(
                    enabled,
                    validate=True,
                    manual_indexer_reference=manual_indexer_reference,
                )
                expected_calls = 1
                observed_calls = call_counter["indexer"]
                if observed_calls != expected_calls or not bool(
                    torch.isfinite(result).all()
                ):
                    raise RuntimeError(
                        f"{workload} validation failed mode={mode}: "
                        f"indexer_calls={observed_calls}"
                    )
                if validation["topk_mismatches"]:
                    raise RuntimeError(
                        f"{workload} validation retained top-k mismatches "
                        f"mode={mode}: {validation['topk_mismatches']}"
                    )
                readback_after = bool(self.deep_gemm.get_pdl())
                if readback_after != enabled:
                    raise RuntimeError(
                        f"PDL control changed during {workload} correctness "
                        f"mode={mode}"
                    )
                mode_correctness.append(
                    {
                        "pdl_mode": mode,
                        "status": "PASS",
                        "control": {
                            **control,
                            "deep_gemm_readback_after_validation": readback_after,
                        },
                        "actual_indexer_calls": observed_calls,
                        "native_replay_calls": 1,
                        "sparse_mla_calls": layers,
                        "manual_reference_shared_read_only": True,
                        "expected_indexer_calls": expected_calls,
                        "attention_layers": layers,
                        "pattern": "F" if layers == 1 else "FSSS",
                        "output_elements_checked": int(result.numel()),
                        "output_digest": self.compact_tensor_digest(result),
                        "validation_scope": (
                            "all_rows_valid_topk_prefix_attention_and_layer_outputs"
                        ),
                        **validation,
                    }
                )
            correctness.append(
                {
                    "query_start": start,
                    "query_count": count,
                    "indexer_workload_geometry": "exact_causal_lower_triangle",
                    "first_query_causal_key_count": start + 1,
                    "last_query_causal_key_count": start + count,
                    "indexer_causal_pairs_executed": causal_pairs_for_chunk(
                        start, count
                    ),
                    "query_sampling": "NONE",
                    "causal_pair_sampling": "NONE",
                    "manual_indexer_reference": {
                        "computation_count": 1,
                        "shared_read_only_across_modes": True,
                        "modes_using_reference": list(PDL_MODES),
                    },
                    "mode_correctness": mode_correctness,
                }
            )
            # Both mode-specific correctness passes have consumed the one
            # immutable manual oracle.  It is not part of warmup or timing.
            del manual_indexer_reference, result

            for _ in range(self.args.warmup):
                for enabled in (False, True):
                    self.set_pdl(enabled)
                    call_counter["indexer"] = 0
                    forward(enabled)
                    if call_counter["indexer"] != 1:
                        raise RuntimeError("IndexShare warmup executed wrong indexer count")
            torch.cuda.synchronize()

            for _, repeat, _, enabled, _ in paired_timing_schedule(
                self.args.repeats, ("layer_total",)
            ):
                mode = "on" if enabled else "off"
                self.set_pdl(enabled)
                index_state["topk"].fill_(-300000 - repeat)
                for output in outputs:
                    output.fill_(float("nan"))
                call_counter["indexer"] = 0
                totals[mode][repeat] += self.elapsed_ms(
                    lambda enabled=enabled: forward(enabled)
                )
                if call_counter["indexer"] != 1:
                    raise RuntimeError("IndexShare timed path executed wrong indexer count")
                if self.topk_valid_prefix_is_invalid(
                    index_state["topk"][:count], start
                ) or any(
                    not bool(torch.isfinite(output).all()) for output in outputs
                ):
                    raise RuntimeError(
                        "layer timed sample left an invalid valid-prefix/output buffer"
                    )

            # The shared manual oracle is immutable across off/on but scoped
            # to exactly this chunk.  Drop every tensor-bearing local and its
            # closure only after the final paired timed event.
            del forward
            del output, outputs, valid_counts, metadata, original_hidden, positions
            self.release_completed_chunk_cache(chunk_index + 1)

        row_id = f"{spec.key}.{workload}.seq{seq}"
        samples = []
        component = (
            "attention_layer_total" if layers == 1 else "four_layer_fsss_total"
        )
        for event_ordinal, repeat, _, enabled, pair_order in paired_timing_schedule(
            self.args.repeats, ("layer_total",)
        ):
            mode = "on" if enabled else "off"
            samples.append({
                "schema": SCHEMA,
                "row_id": row_id,
                "model": spec.key,
                "seq": seq,
                "workload": workload,
                "component": component,
                "pdl_mode": mode,
                "repeat": repeat,
                "elapsed_ms": totals[mode][repeat],
                "poison_epoch": repeat + 1,
                "poison_verified": True,
                "timed_validation": False,
                "indexer_calls_per_invocation": 1,
                "timing_event_ordinal": event_ordinal,
                "timing_pair_ordinal": event_ordinal // 2,
                "pair_order": pair_order,
                "pair_same_process_pid": os.getpid(),
                "pair_same_process_start_ticks": process_start_ticks(
                    os.getpid()
                ),
            })
        return samples, correctness

    def benchmark_moe(self, spec: ModelSpec, seed: int):
        torch = self.torch
        e = self.args.moe_experts
        topk = self.args.moe_topk
        tokens = self.args.moe_tokens
        h = spec.hidden_size
        inter = spec.moe_intermediate_size
        x = self.randn((tokens, h), seed + 1)
        gating_base = torch.randn(
            tokens,
            e,
            generator=self._rng(seed + 2),
            dtype=torch.float32,
            device=self.device,
        )
        gating_delta = torch.randn(
            tokens,
            e,
            generator=self._rng(seed + 20),
            dtype=torch.float32,
            device=self.device,
        )
        gating = torch.empty(tokens, e, dtype=torch.float32, device=self.device)
        gating.copy_(gating_base)
        w1 = self.randn((e, 2 * inter, h), seed + 3) / math.sqrt(h)
        w2 = self.randn((e, h, inter), seed + 4) / math.sqrt(inter)
        quant_config = self.FusedMoEQuantConfig.make(quant_dtype=None)

        def forward():
            weights, ids, _ = self.fused_topk(x, gating, topk, renormalize=True)
            output = self.fused_experts(
                x, w1, w2, weights, ids, quant_config=quant_config
            )
            return output, weights, ids

        actual, route_weights, route_ids = forward()
        reference = torch.zeros_like(actual, dtype=torch.float32)
        for expert in range(e):
            locations = (route_ids == expert).nonzero(as_tuple=False)
            if locations.numel() == 0:
                continue
            token_ids = locations[:, 0]
            route_slots = locations[:, 1]
            selected = x[token_ids].float()
            gate_up = selected @ w1[expert].float().T
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            expert_out = activated @ w2[expert].float().T
            reference.index_add_(
                0,
                token_ids,
                expert_out * route_weights[token_ids, route_slots].float().unsqueeze(1),
            )
        moe_ok = bool(
            torch.allclose(actual.float(), reference, rtol=0.08, atol=0.12)
        )
        max_abs = float((actual.float() - reference).abs().max().item())
        if not moe_ok:
            raise RuntimeError(f"MoE-32 full reference mismatch max_abs={max_abs}")

        for _ in range(self.args.warmup):
            forward()
        torch.cuda.synchronize()
        values = []
        for repeat in range(self.args.repeats):
            # Routing input changes by epoch outside the timed region; fused_experts
            # allocates a fresh output, so stale output cannot satisfy validation.
            gating.copy_(gating_base).add_(gating_delta, alpha=(repeat + 1) * 1e-3)
            captured: list[Any] = []

            def timed_forward():
                captured[:] = forward()

            values.append(self.elapsed_ms(timed_forward))
            if len(captured) != 3:
                raise RuntimeError("MoE timed sample did not return all production outputs")
            timed_output, timed_weights, timed_ids = captured
            sorted_ids = torch.sort(timed_ids, dim=1).values
            if (
                not bool(torch.isfinite(timed_output).all())
                or not bool(torch.isfinite(timed_weights).all())
                or bool(((timed_ids < 0) | (timed_ids >= e)).any())
                or bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any())
            ):
                raise RuntimeError("MoE timed sample failed poison/routing validation")
        row_id = f"{spec.key}.moe32"
        samples = [
            {
                "schema": SCHEMA,
                "row_id": row_id,
                "model": spec.key,
                "seq": None,
                "workload": "moe32",
                "component": "fused_topk_plus_fused_experts",
                "pdl_mode": "framework_default_uncontrolled",
                "repeat": repeat,
                "elapsed_ms": value,
                "poison_epoch": repeat + 1,
                "poison_verified": True,
                "fresh_output_allocation": True,
                "timed_validation": False,
            }
            for repeat, value in enumerate(values)
        ]
        correctness = {
            "tokens_checked": tokens,
            "output_elements_checked": int(actual.numel()),
            "routing_assignments_checked": int(route_ids.numel()),
            "experts": e,
            "topk": topk,
            "max_abs": max_abs,
            "status": "PASS",
        }
        return samples, correctness

    def execute_fragment(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute exactly one campaign row while preserving frozen seed mapping."""
        matrix = expected_matrix(
            self.args.models, self.args.seqs, self.args.workloads
        )
        ordinal = self.args.fragment_ordinal
        if not isinstance(ordinal, int) or not 0 <= ordinal < len(matrix):
            raise RuntimeError("row-fragment ordinal is outside the campaign matrix")
        row = matrix[ordinal]
        if row["row_id"] != self.args.fragment_row_id:
            raise RuntimeError("row-fragment id/ordinal binding drift")

        model = str(row["model"])
        spec = MODEL_SPECS[model]
        workload = str(row["workload"])
        row_id = str(row["row_id"])
        row_seed = canonical_row_seed(row, self.args.seed)
        nvtx_label = (
            f"tier5_fragment:{self.args.fragment_ordinal}:{row_id}:"
            f"{self.args.execution_segment_id}"
        )
        self.torch.cuda.nvtx.range_push(nvtx_label)

        if workload == "moe32":
            samples, evidence = self.benchmark_moe(spec, row_seed)
            correctness_row = {"row_id": row_id, "status": "PASS", **evidence}
        else:
            seq = int(row["seq"])
            chunk = query_chunk_tokens(
                seq, self.args.max_logits_mb, self.args.max_query_chunk
            )
            if workload == "operator_chain":
                samples, evidence = self.benchmark_operator_chain(
                    spec, seq, chunk, row_seed
                )
            elif workload == "single_layer":
                samples, evidence = self.benchmark_layer_like(
                    spec, seq, chunk, row_seed, 1
                )
            elif workload == "indexshare_fsss":
                samples, evidence = self.benchmark_layer_like(
                    spec, seq, chunk, row_seed, 4
                )
            else:
                raise AssertionError(workload)
            partition_pairs = sum(
                int(chunk_evidence["indexer_causal_pairs_executed"])
                for chunk_evidence in evidence
            )
            expected_pairs = causal_pair_count(seq)
            if partition_pairs != expected_pairs:
                raise RuntimeError(
                    f"causal pair partition mismatch model={model} "
                    f"workload={workload} seq={seq}: "
                    f"chunks={partition_pairs} expected={expected_pairs}"
                )
            correctness_row = {
                "row_id": row_id,
                "status": "PASS",
                "chunks": evidence,
                "all_query_rows_executed": seq,
                "indexer_workload_geometry": "exact_causal_lower_triangle",
                "query_sampling": "NONE",
                "indexer_causal_pairs_executed": expected_pairs,
                "indexer_causal_pair_formula": "S*(S+1)/2",
                "causal_pair_sampling": "NONE",
                "chunk_causal_pairs_sum": partition_pairs,
                "chunk_pair_partition_verified": True,
                "correctness_pdl_modes": list(PDL_MODES),
                "per_chunk_mode_correctness_complete": True,
            }
        correctness = {
            "schema": SCHEMA,
            "kind": "tier5_production_correctness",
            "status": "PASS",
            "execution_scope": "row_fragment",
            "fragment_row_id": row_id,
            "rows": [correctness_row],
            "all_expected_rows_present": True,
        }
        self.torch.cuda.nvtx.range_pop()
        return samples, correctness

    def execute(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise RuntimeError(
            "monolithic GPU execution is inadmissible; execute one sealed row "
            "fragment through the campaign runner"
        )


def parse_csv_choice(value: str, choices: Iterable[str], label: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    allowed = set(choices)
    bad = [part for part in values if part not in allowed]
    if not values or bad or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            f"invalid {label} list {value!r}; allowed unique values: {sorted(allowed)}"
        )
    return values


def parse_seqs(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seqs must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seqs must be non-empty and unique")
    bad = [seq for seq in values if seq not in FORMAL_SEQS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unsupported contexts {bad}; allowed={list(FORMAL_SEQS)}"
        )
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    invocation_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--publish-target",
        help="final sibling destination used by the failure-atomic runner",
    )
    parser.add_argument(
        "--runner-managed-stage",
        action="store_true",
        help="allow only the runner's named logs/exclusivity evidence to pre-exist",
    )
    parser.add_argument(
        "--execute-gpu",
        action="store_true",
        help="execute CUDA only when TIER5_PRODUCTION_GPU_ALLOWED=1 is also set",
    )
    parser.add_argument("--backend", choices=["flashinfer"], default="flashinfer")
    parser.add_argument("--required-device-substring", default="B200")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--expected-gpu-index", type=int)
    parser.add_argument(
        "--fragment-row-id",
        help="exact canonical row id selected by the campaign runner",
    )
    parser.add_argument("--fragment-ordinal", type=int)
    parser.add_argument("--campaign-contract-sha256")
    parser.add_argument("--campaign-fingerprint-sha256")
    parser.add_argument("--execution-segment-id")
    parser.add_argument(
        "--models",
        type=lambda value: parse_csv_choice(value, MODEL_SPECS, "models"),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--seqs", type=parse_seqs, default=list(FORMAL_SEQS))
    parser.add_argument(
        "--workloads",
        type=lambda value: parse_csv_choice(value, FORMAL_WORKLOADS, "workloads"),
        default=list(FORMAL_WORKLOADS),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--max-logits-mb", type=int, default=FORMAL_MAX_LOGITS_MB
    )
    parser.add_argument(
        "--max-query-chunk", type=int, default=FORMAL_MAX_QUERY_CHUNK
    )
    parser.add_argument("--moe-experts", type=int, default=32)
    parser.add_argument("--moe-topk", type=int, default=8)
    parser.add_argument("--moe-tokens", type=int, default=4096)
    args = parser.parse_args(invocation_argv)
    args.invocation_argv = invocation_argv
    for name in (
        "max_logits_mb",
        "max_query_chunk",
        "moe_experts",
        "moe_topk",
        "moe_tokens",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.moe_experts != 32 or args.moe_topk != 8:
        parser.error("formal production contract requires --moe-experts 32 --moe-topk 8")
    if not args.required_device_substring:
        parser.error("--required-device-substring must be non-empty")
    if args.expected_gpu_uuid is not None:
        args.expected_gpu_uuid = canonical_gpu_uuid(args.expected_gpu_uuid)
        if args.expected_gpu_uuid is None:
            parser.error("--expected-gpu-uuid must be a canonical full GPU UUID")
    if args.expected_gpu_index is not None and args.expected_gpu_index < 0:
        parser.error("--expected-gpu-index must be non-negative")
    fragment_values = (
        args.fragment_row_id,
        args.fragment_ordinal,
        args.campaign_contract_sha256,
        args.campaign_fingerprint_sha256,
        args.execution_segment_id,
    )
    has_fragment = any(value is not None for value in fragment_values)
    if has_fragment and not all(value is not None for value in fragment_values):
        parser.error(
            "row-fragment execution requires row id, ordinal, contract hash, "
            "fingerprint hash, and segment id together"
        )
    if args.execute_gpu and not has_fragment:
        parser.error(
            "monolithic GPU execution is inadmissible; a bound row fragment is required"
        )
    if has_fragment and not args.execute_gpu:
        parser.error("row-fragment arguments are valid only with --execute-gpu")
    if has_fragment:
        if args.fragment_ordinal < 0:
            parser.error("--fragment-ordinal must be non-negative")
        if not SHA256_RE.fullmatch(args.campaign_contract_sha256):
            parser.error("--campaign-contract-sha256 must be lowercase sha256")
        if not SHA256_RE.fullmatch(args.campaign_fingerprint_sha256):
            parser.error("--campaign-fingerprint-sha256 must be lowercase sha256")
        if not INVOCATION_UUID_RE.fullmatch(args.execution_segment_id):
            parser.error("--execution-segment-id must be a lowercase UUIDv4")
        matrix = expected_matrix(args.models, args.seqs, args.workloads)
        if args.fragment_ordinal >= len(matrix):
            parser.error("--fragment-ordinal is outside the canonical matrix")
        if matrix[args.fragment_ordinal]["row_id"] != args.fragment_row_id:
            parser.error("--fragment-row-id does not match its canonical ordinal")
    if args.repeats < 31 and not args.allow_short:
        parser.error("fewer than 31 repeats requires --allow-short")
    if not args.allow_short:
        if args.warmup != 5 or args.repeats != 31:
            parser.error("formal mode requires exactly --warmup 5 --repeats 31")
        if args.moe_tokens != 4096:
            parser.error("formal mode requires --moe-tokens 4096")
        if (
            args.max_logits_mb != FORMAL_MAX_LOGITS_MB
            or args.max_query_chunk != FORMAL_MAX_QUERY_CHUNK
        ):
            parser.error(
                "formal mode requires --max-logits-mb 16384 --max-query-chunk 4096"
            )
        if args.seed != 20260805:
            parser.error("formal mode requires the frozen --seed 20260805")
        if tuple(args.seqs) != FORMAL_SEQS:
            parser.error(
                "formal mode requires exact ordered contexts 4096,32768,131072,1048576"
            )
        if tuple(args.workloads) != FORMAL_WORKLOADS:
            parser.error("formal mode requires operator_chain,single_layer,indexshare_fsss")
        if tuple(args.models) != tuple(MODEL_SPECS):
            parser.error("formal mode requires deepseek_v32,glm5")
    return args


def write_failure(output: Path, args: argparse.Namespace | None, exc: BaseException) -> None:
    payload = {
        "schema": SCHEMA,
        "kind": "tier5_production_failure",
        "status": "FAIL",
        "accepted_timing": 0,
        "accepted_timing_semantics": "legacy_CTA_bracket_only",
        "accepted_workload_timing": 0,
        "accepted_CTA_bracket": 0,
        "measurement_emitted": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "mode": "execute_gpu" if args is not None and args.execute_gpu else "cpu_dry_run",
        "created_unix_ns": time.time_ns(),
    }
    atomic_write_json(output / "failure.json", payload)
    atomic_write_json(
        output / "terminal_status.json",
        {
            "schema": SCHEMA,
            "status": "FAIL",
            "accepted_timing": 0,
            "accepted_workload_timing": 0,
            "accepted_CTA_bracket": 0,
            "failure_sha256": sha256_file(output / "failure.json"),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    output = Path(".")
    try:
        args = parse_args(argv)
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        existing = {path.name for path in output.iterdir()}
        if args.runner_managed_stage:
            if not args.publish_target:
                raise RuntimeError("runner-managed staging requires --publish-target")
            allowed_existing = {
                "runner.log",
                "harness.log",
                "gpu_identity.json",
                "gpu_exclusivity_lease.json",
                "gpu_pre.json",
                "gpu_monitor.json",
                "gpu_monitor.json.ready",
                "gpu_observations.ndjson",
            }
            unexpected = sorted(existing - allowed_existing)
            if unexpected:
                raise RuntimeError(
                    f"runner-managed output contains unexpected files: {unexpected}"
                )
        elif existing:
            raise RuntimeError(f"output directory must be empty: {output}")

        runtime: ProductionRuntime | None = None
        if args.execute_gpu:
            runtime = ProductionRuntime(args)
            device = runtime.device_manifest
        else:
            device = None
        manifest = make_manifest(args, runtime_device=device)
        atomic_write_json(output / "manifest.json", manifest)

        if runtime is None:
            plan = {
                "schema": SCHEMA,
                "kind": "tier5_production_dsa_plan",
                "status": "NOT_EXECUTED",
                "accepted_timing": 0,
                "accepted_timing_semantics": "legacy_CTA_bracket_only",
                "accepted_workload_timing": 0,
                "accepted_CTA_bracket": 0,
                "measurement_emitted": False,
                "manifest_sha256": sha256_file(output / "manifest.json"),
                "expected_matrix": manifest["expected_matrix"],
                "shape_records": manifest["shape_records"],
                "gpu_guard": {
                    "required_cli": "--execute-gpu",
                    "required_environment": "TIER5_PRODUCTION_GPU_ALLOWED=1",
                },
            }
            atomic_write_json(output / "plan.json", plan)
            atomic_write_json(
                output / "terminal_status.json",
                {
                    "schema": SCHEMA,
                    "status": "NOT_EXECUTED",
                    "accepted_timing": 0,
                    "accepted_workload_timing": 0,
                    "accepted_CTA_bracket": 0,
                    "measurement_emitted": False,
                    "manifest_sha256": sha256_file(output / "manifest.json"),
                    "plan_sha256": sha256_file(output / "plan.json"),
                },
            )
            print(
                "PRODUCTION_TIER5_DRY_RUN "
                f"schema={SCHEMA} rows={len(plan['expected_matrix'])} "
                "status=NOT_EXECUTED accepted_timing=0 bracket=PARTIAL"
            )
            return 0

        samples, correctness = runtime.execute_fragment()
        for sample in samples:
            sample.update(
                fragment_row_ordinal=args.fragment_ordinal,
                invocation_uuid=args.execution_segment_id,
                campaign_contract_sha256=args.campaign_contract_sha256,
                campaign_fingerprint_sha256=args.campaign_fingerprint_sha256,
                derived_row_seed=manifest["fragment"]["derived_row_seed"],
            )
        correctness["fragment"] = manifest["fragment"]
        sample_payload = b"".join(
            (
                json.dumps(sample, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            for sample in samples
        )
        atomic_write_bytes(output / "samples.jsonl", sample_payload)
        atomic_write_json(output / "correctness.json", correctness)
        summaries = summarize_samples(samples, args.seed)
        result = {
            "schema": SCHEMA,
            "kind": "tier5_production_dsa_result",
            "status": "CANDIDATE",
            "accepted_timing": 0,
            "accepted_timing_semantics": "legacy_CTA_bracket_only",
            "accepted_workload_timing": 0,
            "accepted_CTA_bracket": 0,
            "measurement_emitted": True,
            "claim_scope": "production_kernel_characterization_only",
            "production_timing_candidate": True,
            "tier5_bracket_admitted": False,
            "formal_bracket_status": "PARTIAL",
            "headroom_defined": False,
            "headroom_pct": None,
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "samples_sha256": sha256_file(output / "samples.jsonl"),
            "correctness_sha256": sha256_file(output / "correctness.json"),
            "sample_count": len(samples),
            "summaries": summaries,
            "execution_scope": "row_fragment",
            "fragment": manifest["fragment"],
        }
        atomic_write_json(output / "result.json", result)
        atomic_write_json(
            output / "terminal_status.json",
            {
                "schema": SCHEMA,
                "status": "CANDIDATE",
                "accepted_timing": 0,
                "accepted_workload_timing": 0,
                "accepted_CTA_bracket": 0,
                "measurement_emitted": True,
                "result_sha256": sha256_file(output / "result.json"),
                "execution_scope": "row_fragment",
                "fragment": manifest["fragment"],
            },
        )
        print(
            "PRODUCTION_TIER5_CANDIDATE "
            f"schema={SCHEMA} samples={len(samples)} status=CANDIDATE "
            "accepted_timing=0 bracket=PARTIAL"
        )
        return 0
    except Exception as exc:
        try:
            output.mkdir(parents=True, exist_ok=True)
            write_failure(output, args, exc)
        except Exception:
            traceback.print_exc()
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
