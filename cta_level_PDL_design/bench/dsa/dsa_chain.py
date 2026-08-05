#!/usr/bin/env python3
"""Tier-5 DSA harness admission audit (fail closed).

This file deliberately does *not* time the old PyTorch operator chain.  That chain cannot
realise any of the dependency rungs required by EXPERIMENT_PLAN.md:

* ``torch.cuda.synchronize`` is a host wait, not grid-level PDL;
* deleting that host wait leaves all PyTorch kernels ordered in one CUDA stream, so it is
  not the dependency-free Ceiling;
* there is no CTA-granular implementation rung; and
* Floor/Impl/Ceiling were formerly measured in different processes.

``--audit-only`` writes machine-readable semantic and allocation evidence and exits zero.
Any request to measure exits 2 without emitting a ``SUMMARY`` timing record.  This is the
intended fail-closed behaviour until a native harness provides real PDL launches, an
actually unordered Ceiling, adjacent rungs, and full validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCHEMA = 2
BF16_BYTES = 2


def gib(nbytes: int) -> float:
    return nbytes / (1 << 30)


def dsa_allocations(args: argparse.Namespace) -> dict[str, int]:
    """Explicit tensor sizes created by the rejected implementation.

    These are not presented as an allocator peak estimate.  Each entry is the size of a
    tensor whose shape follows directly from the former PyTorch expressions, and is enough
    to prove that the declared 128K/1M full-sequence points cannot use that implementation.
    """

    s = args.seq
    h = args.index_n_heads
    d = args.index_head_dim
    k = min(args.index_topk, s)
    r = args.kv_lora_rank
    qlr = args.q_lora_rank
    return {
        "q_lat_bf16": s * qlr * BF16_BYTES,
        "projected_q_bf16": s * h * d * BF16_BYTES,
        # torch.einsum("thd,sd->ths", ...)
        "indexer_ths_bf16": s * h * s * BF16_BYTES,
        # torch.einsum("ths,h->ts", ...)
        "indexer_scores_bf16": s * s * BF16_BYTES,
        "topk_indices_i64": s * k * 8,
        # kv_lat[idx] -> (S, topk, kv_lora_rank)
        "sparse_gather_bf16": s * k * r * BF16_BYTES,
        "kv_lat_bf16": s * r * BF16_BYTES,
        "attention_q_bf16": s * r * BF16_BYTES,
    }


def moe_allocations(args: argparse.Namespace) -> dict[str, int]:
    e, h, inter, t = args.experts, args.hidden, args.moe_inter, args.tokens
    return {
        "expert_w1_bf16": e * h * inter * BF16_BYTES,
        "expert_w2_bf16": e * inter * h * BF16_BYTES,
        "tokens_bf16": t * h * BF16_BYTES,
        "router_bf16": h * e * BF16_BYTES,
    }


def visible_device() -> dict[str, object]:
    info: dict[str, object] = {
        "cuda_available": False,
        "name": None,
        "total_memory_bytes": None,
        "compute_capability": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            info.update(
                cuda_available=True,
                name=prop.name,
                total_memory_bytes=prop.total_memory,
                compute_capability=f"{prop.major}.{prop.minor}",
            )
    except (ImportError, RuntimeError):
        pass
    return info


def build_audit(args: argparse.Namespace) -> dict[str, object]:
    kind = "moe" if args.moe else "dsa"
    allocations = moe_allocations(args) if args.moe else dsa_allocations(args)
    device = visible_device()
    total = device["total_memory_bytes"]
    over_device = []
    if isinstance(total, int):
        over_device = [name for name, size in allocations.items() if size > total]

    semantic_blockers = [
        {
            "code": "floor_not_grid_pdl",
            "evidence": "former Floor used torch.cuda.synchronize between ordinary same-stream ops",
        },
        {
            "code": "ceiling_not_unordered",
            "evidence": "removing host synchronization leaves CUDA stream order unchanged",
        },
        {
            "code": "impl_missing",
            "evidence": "no CTA/query-row readiness protocol exists in the PyTorch chain",
        },
        {
            "code": "rungs_not_adjacent",
            "evidence": "former --rung invocations used separate processes",
        },
        {
            "code": "validation_missing",
            "evidence": "former verified=1 was a label; no poisoned full-edge reference comparison ran",
        },
        {
            "code": "statistics_incomplete",
            "evidence": "former output had a median/min only, with 10 repeats and no confidence interval",
        },
    ]
    if args.moe:
        semantic_blockers.append(
            {
                "code": "moe_host_scalar_sync",
                "evidence": "int(mask.sum()) synchronizes the host once per expert inside the timed path",
            }
        )

    allocation_blockers = []
    if over_device:
        allocation_blockers.append(
            {
                "code": "single_tensor_exceeds_device",
                "tensors": over_device,
                "device_memory_bytes": total,
            }
        )

    return {
        "schema": SCHEMA,
        "tier": 5,
        "kind": kind,
        "tag": args.tag,
        "status": "blocked",
        "runnable": False,
        "measurement_emitted": False,
        "requested_rung": args.rung,
        "requested_warmup": args.warmup,
        "requested_repeats": args.iters,
        "minimum_repeats": 31,
        "shape": (
            {
                "seq": args.seq,
                "hidden": args.hidden,
                "kv_lora_rank": args.kv_lora_rank,
                "q_lora_rank": args.q_lora_rank,
                "index_head_dim": args.index_head_dim,
                "index_n_heads": args.index_n_heads,
                "index_topk": args.index_topk,
            }
            if not args.moe
            else {
                "tokens": args.tokens,
                "hidden": args.hidden,
                "experts": args.experts,
                "topk_experts": args.topk_experts,
                "moe_inter": args.moe_inter,
            }
        ),
        "explicit_tensor_bytes": allocations,
        "explicit_tensor_gib": {name: round(gib(size), 6) for name, size in allocations.items()},
        "device": device,
        "semantic_blockers": semantic_blockers,
        "allocation_blockers": allocation_blockers,
        "required_replacement_contract": {
            "floor": "cudaLaunchAttributeProgrammaticStreamSerialization + device gdc wait",
            "impl": "identity-preserving per-query/CTA readiness, if implemented",
            "ceiling": "kernels launched without dependency/order; results deliberately unverified",
            "validation": "poison every repeat; separate full reference comparison for Floor/Impl",
            "statistics": "all rungs adjacent in one process, >=31 samples each, confidence intervals",
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-only", action="store_true", help="emit blocking evidence and exit 0")
    ap.add_argument("--json", help="write the audit object to this file")
    ap.add_argument("--tag", default="dsa")
    ap.add_argument("--rung", default="all", choices=["all", "floor", "impl", "ceiling"])
    ap.add_argument("--iters", type=int, default=31)
    ap.add_argument("--warmup", type=int, default=5)

    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--kv-lora-rank", type=int, default=512)
    ap.add_argument("--q-lora-rank", type=int, default=2048)
    ap.add_argument("--index-head-dim", type=int, default=128)
    ap.add_argument("--index-n-heads", type=int, default=32)
    ap.add_argument("--index-topk", type=int, default=2048)

    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--topk-experts", type=int, default=8)
    ap.add_argument("--moe-inter", type=int, default=2048)
    args = ap.parse_args()

    positive = {
        name: getattr(args, name)
        for name in (
            "iters",
            "seq",
            "hidden",
            "kv_lora_rank",
            "q_lora_rank",
            "index_head_dim",
            "index_n_heads",
            "index_topk",
            "tokens",
            "experts",
            "topk_experts",
            "moe_inter",
        )
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if args.warmup < 0:
        bad.append("warmup")
    if bad:
        ap.error("arguments must be positive: " + ", ".join(bad))
    return args


def main() -> int:
    args = parse_args()
    audit = build_audit(args)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    largest_name, largest_bytes = max(
        audit["explicit_tensor_bytes"].items(), key=lambda item: item[1]
    )
    print(
        "AUDIT_DSA "
        f"schema={SCHEMA} tier=5 kind={audit['kind']} status=blocked runnable=0 "
        f"tag={args.tag} largest_tensor={largest_name} largest_gib={gib(largest_bytes):.6f} "
        f"semantic_blockers={len(audit['semantic_blockers'])} "
        f"allocation_blockers={len(audit['allocation_blockers'])}"
    )
    print(
        "BLOCKED: Tier-5 timing is disabled until a native harness proves real grid PDL, "
        "a truly unordered Ceiling, adjacent rungs, and full non-Ceiling validation.",
        file=sys.stderr,
    )
    return 0 if args.audit_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
