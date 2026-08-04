#!/usr/bin/env python3
"""Tier 5: measure the DSA operator chain on a SINGLE GPU with real shapes.

The full models (DeepSeek-V3.2 671B, GLM-5.2 744B) do not fit one card, but the DSA
attention path is small. This reproduces

    lightning indexer  ->  top-k selection  ->  sparse MLA attention

at production shapes and brackets it the same way everything else is bracketed:

    Floor    per-op boundaries enforced (each kernel waits for the whole predecessor grid)
    Ceiling  boundaries removed (WRONG RESULTS, timing only) = dependency costs nothing

MoE dispatch/combine is reproduced with a REDUCED expert count. The dependency SHAPE is set
by top-k routing and is independent of the expert total, so 32 experts exercise the same
structure as 256 at an eighth of the weights.

Shapes default to GLM-5.2: hidden 6144, kv_lora_rank 512, q_lora_rank 2048,
index_head_dim 128, index_n_heads 32, index_topk 2048.

Usage:
    python3 dsa_chain.py --seq 32768
    python3 dsa_chain.py --seq 131072 --rung ceiling
    python3 dsa_chain.py --moe --experts 32
"""

import argparse
import json
import statistics
import sys
import time


def require_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("PyTorch required for Tier 5 operator-chain measurement", file=sys.stderr)
        sys.exit(2)


# --------------------------------------------------------------------- DSA chain

def build_dsa(torch, args, dev, dtype):
    """Allocate the tensors of one DSA attention layer at real shapes."""
    S, H = args.seq, args.hidden
    g = torch.Generator(device=dev).manual_seed(0)
    t = lambda *s: torch.randn(*s, device=dev, dtype=dtype, generator=g)
    return {
        # indexer works off the low-rank query latent, per the reference implementation
        "q_lat": t(S, args.q_lora_rank),
        "wq_idx": t(args.q_lora_rank, args.index_n_heads * args.index_head_dim),
        "k_idx": t(S, args.index_head_dim),
        "head_w": t(args.index_n_heads),
        # MLA latent KV cache (MQA mode: one latent shared by all query heads)
        "kv_lat": t(S, args.kv_lora_rank),
        "q": t(S, args.kv_lora_rank),
    }


def dsa_indexer(torch, b, args):
    """Lightning indexer: ReLU-activated low-rank scores, per-head learned weighting.

    Cost is O(L^2) in the sequence length, which is why the chain's share of total time
    grows sharply with context.
    """
    S = args.seq
    qi = (b["q_lat"] @ b["wq_idx"]).view(S, args.index_n_heads, args.index_head_dim)
    # scores[t, s] = sum_h w_h * relu(q_{t,h} . k_s)
    scores = torch.einsum("thd,sd->ths", qi, b["k_idx"])
    scores = torch.relu(scores)
    return torch.einsum("ths,h->ts", scores, b["head_w"])


def dsa_topk(torch, scores, args):
    """Fine-grained token selection: keep the top-k highest-scoring predecessors."""
    k = min(args.index_topk, scores.shape[-1])
    _, idx = torch.topk(scores, k, dim=-1)
    return idx


def dsa_sparse_attn(torch, b, idx, args):
    """Sparse MLA over the selected latents.

    NOTE the dependency structure this embodies: the gather target `kv_lat` was written by
    EARLIER decode steps, not by topk. The only inter-kernel RAW edge here is on `idx`,
    which is 1-to-1 per query block. See archive/dsa_dependency_analysis.md §3.2.
    """
    sel = b["kv_lat"][idx]                                  # (S, k, kv_lora_rank)
    logits = torch.einsum("sd,skd->sk", b["q"], sel)
    w = torch.softmax(logits.float(), dim=-1).to(sel.dtype)
    return torch.einsum("sk,skd->sd", w, sel)


def run_dsa(torch, args, dev, dtype, rung):
    b = build_dsa(torch, args, dev, dtype)

    def one_iter():
        scores = dsa_indexer(torch, b, args)
        if rung == "floor":
            torch.cuda.synchronize()      # hard boundary = whole-grid wait
        idx = dsa_topk(torch, scores, args)
        if rung == "floor":
            torch.cuda.synchronize()
        return dsa_sparse_attn(torch, b, idx, args)

    for _ in range(args.warmup):
        one_iter()
    torch.cuda.synchronize()

    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_iter()
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)

    # Per-stage attribution, so we can see how the indexer's O(L^2) share grows.
    stage = {}
    for name, fn in (
        ("indexer", lambda: dsa_indexer(torch, b, args)),
        ("full_chain", one_iter),
    ):
        ts = []
        for _ in range(max(3, args.iters // 2)):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        stage[name] = statistics.median(ts)

    return {
        "median_s": statistics.median(lat),
        "min_s": min(lat),
        "indexer_s": stage["indexer"],
        "indexer_share": stage["indexer"] / stage["full_chain"] if stage["full_chain"] else 0.0,
    }


# --------------------------------------------------------------------- MoE chain

def run_moe(torch, args, dev, dtype, rung):
    """MoE dispatch/combine with a reduced expert count.

    This is the one DSA-family pattern that is genuinely hostile to CTA-level dependency
    resolution: the permute indices come from the IMMEDIATELY PRECEDING router, so the
    structure is dynamic and no prologue inspector can precompute it.
    """
    T, H, E, K = args.tokens, args.hidden, args.experts, args.topk_experts
    inter = args.moe_inter
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(T, H, device=dev, dtype=dtype, generator=g)
    wr = torch.randn(H, E, device=dev, dtype=dtype, generator=g)
    w1 = torch.randn(E, H, inter, device=dev, dtype=dtype, generator=g)
    w2 = torch.randn(E, inter, H, device=dev, dtype=dtype, generator=g)

    def one_iter():
        logits = x @ wr                                  # router
        _, sel = torch.topk(logits, K, dim=-1)           # top-k experts per token
        if rung == "floor":
            torch.cuda.synchronize()
        out = torch.zeros_like(x)
        for e in range(E):                               # grouped GEMM per expert
            mask = (sel == e).any(dim=-1)
            n = int(mask.sum())
            if n == 0:
                continue
            xi = x[mask]                                 # gather/permute
            hi = torch.relu(xi @ w1[e])
            out[mask] += hi @ w2[e]                      # scatter/unpermute
        return out

    for _ in range(args.warmup):
        one_iter()
    torch.cuda.synchronize()

    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_iter()
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(lat), "min_s": min(lat),
            "indexer_s": 0.0, "indexer_share": 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--kv-lora-rank", type=int, default=512)
    ap.add_argument("--q-lora-rank", type=int, default=2048)
    ap.add_argument("--index-head-dim", type=int, default=128)
    ap.add_argument("--index-n-heads", type=int, default=32)
    ap.add_argument("--index-topk", type=int, default=2048)
    ap.add_argument("--rung", default="floor", choices=["floor", "ceiling"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default="dsa")
    ap.add_argument("--json", help="write the result as JSON")
    # MoE mode
    ap.add_argument("--moe", action="store_true", help="measure MoE dispatch/combine instead")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=32,
                    help="REDUCED from 256; dependency shape is set by top-k, not by the total")
    ap.add_argument("--topk-experts", type=int, default=8)
    ap.add_argument("--moe-inter", type=int, default=2048)
    args = ap.parse_args()

    torch = require_torch()
    if not torch.cuda.is_available():
        print("SUMMARY tier=5 status=no_cuda", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    name = torch.cuda.get_device_name(0)
    print(f"[dev] {name}")

    kind = "moe" if args.moe else "dsa"
    try:
        res = (run_moe if args.moe else run_dsa)(torch, args, dev, dtype, args.rung)
    except RuntimeError as e:
        # OOM at long context is an expected outcome, not a campaign-stopping failure.
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"SUMMARY tier=5 kind={kind} tag={args.tag} rung={args.rung} status=error "
              f"seq={args.seq}")
        return 1

    extra = (f"tokens={args.tokens} experts={args.experts} topk_experts={args.topk_experts}"
             if args.moe else
             f"seq={args.seq} index_topk={args.index_topk} index_n_heads={args.index_n_heads}")

    print(f"SUMMARY tier=5 kind={kind} tag={args.tag} rung={args.rung} status=ok "
          f"{extra} hidden={args.hidden} "
          f"median_s={res['median_s']:.6f} min_s={res['min_s']:.6f} "
          f"indexer_s={res['indexer_s']:.6f} indexer_share={res['indexer_share']:.4f} "
          f"verified={0 if args.rung == 'ceiling' else 1}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "result": res, "device": name}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
