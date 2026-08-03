#!/usr/bin/env python3
"""Derive CTA-level inter-kernel dependencies analytically — no GPU, no instrumentation.

WHY THIS WORKS WITHOUT PROFILING
--------------------------------
For the kernels that dominate LLM inference (GEMM, RMSNorm, SwiGLU, elementwise), the
CTA -> data mapping is PUBLIC: it is the tile decomposition chosen by CUTLASS/Triton. So the
read/write set of every CTA can be computed in closed form, and the exact dependency
bipartite graph follows from intersecting them. NVBit instrumentation (100x+ overhead) is
only needed for opaque kernels.

WHAT IT ANSWERS
---------------
Dimension A2 (dependency provenance) and A3 (encoding) need two numbers per kernel pair:

  interval tightness = |true parents| / |interval cover|
      1.0 => an interval-encoded wait is EXACT
      low => interval encoding drags in many false parents

  false-edge rate    = fraction of the interval cover that is not a true parent
      this is the performance tax of choosing a cheap encoding

They also settle the question E1 raises: LLM patterns have HIGH DEGREE but CONTIGUOUS
structure, so the BlockMaestro "degree > 32 => no benefit" threshold should not apply.

Usage:
    python3 tools/dep_oracle.py                       # analyze the built-in Qwen3.6-27B chain
    python3 tools/dep_oracle.py --model qwen3.6-27b --tokens 4096
    python3 tools/dep_oracle.py --model glm-5.2-dsa --seq 131072
    python3 tools/dep_oracle.py --json out.json
"""

import argparse
import json
from dataclasses import dataclass, field


# --------------------------------------------------------------------- tile model

@dataclass
class Kernel:
    """A kernel described by how its CTAs map onto a linear element space.

    Each CTA c owns a contiguous output range and reads one or more contiguous input ranges.
    That is enough to express GEMM / norm / elementwise / gather-with-known-structure tiles.
    """
    name: str
    n_cta: int
    # cta -> (lo, hi) inclusive range of the OUTPUT buffer it writes
    write: callable
    # cta -> list of (buffer_name, lo, hi) it reads
    read: callable
    out_buf: str = "out"
    note: str = ""


def ranges_overlap(a_lo, a_hi, b_lo, b_hi):
    return a_lo <= b_hi and b_lo <= a_hi


def sample_consumers(n_cta, limit):
    """Uniformly sample consumer CTA ids when the grid is too large to enumerate.

    Every statistic below is a mean over consumers, so sampling is statistically sound.
    This matters because DSA at 1M context has ~1.3e8 producer CTAs and 16K consumers;
    the exact sweep is pure-Python O(consumers x candidates) and would take hours.
    """
    if limit <= 0 or n_cta <= limit:
        return list(range(n_cta)), False
    stride = n_cta / limit
    return sorted({int(i * stride) for i in range(limit)}), True


def build_graph(producer: Kernel, consumer: Kernel, consumer_ids=None):
    """Exact CTA-level dependency graph: consumer CTA j depends on producer CTA i iff
    j reads any element that i writes.

    Producer write ranges are almost always monotonically increasing in CTA id (that is what
    a tile decomposition gives you), so the common case is resolved with bisect in
    O(M log N) instead of the naive O(M*N). The general case falls back to a linear sweep,
    which matters because DSA at 1M context has ~10^6 producer CTAs.
    """
    import bisect

    n_prod = producer.n_cta
    write = producer.write

    # Lazy view of the producer write-start array so bisect never materializes it. DSA at 1M
    # context has ~1.3e8 producer CTAs; a real list would be hopeless.
    class _LazyLos:
        __slots__ = ()
        def __len__(self): return n_prod
        def __getitem__(self, i): return write(i)[0]

    los = _LazyLos()

    # Monotonic write starts are what a tile decomposition naturally produces. Probe instead
    # of scanning all N, then verify while walking candidates.
    def _is_sorted(probe=4096):
        stride = max(1, n_prod // probe)
        prev = None
        for i in range(0, n_prod, stride):
            cur = write(i)[0]
            if prev is not None and cur < prev:
                return False
            prev = cur
        return True

    sorted_writes = _is_sorted()
    # Widest write range, sampled; used to bound how far back a candidate can start.
    max_w = 0
    if sorted_writes:
        stride = max(1, n_prod // 4096)
        for i in range(0, n_prod, stride):
            lo, hi = write(i)
            max_w = max(max_w, hi - lo + 1)

    ids = consumer_ids if consumer_ids is not None else range(consumer.n_cta)

    parents = []
    for j in ids:
        reads = [r for r in consumer.read(j) if r[0] == producer.out_buf]
        if not reads:
            parents.append([])
            continue

        ps = set()
        if sorted_writes:
            for (_buf, rlo, rhi) in reads:
                # Candidates start no later than rhi; walk back far enough to cover the
                # widest possible write range, then forward while ranges still overlap.
                hi_idx = bisect.bisect_right(los, rhi) - 1
                if hi_idx < 0:
                    continue
                lo_idx = bisect.bisect_left(los, rlo - max_w)
                for i in range(max(0, lo_idx), hi_idx + 1):
                    wlo, whi = write(i)
                    if ranges_overlap(wlo, whi, rlo, rhi):
                        ps.add(i)
        else:
            for i in range(n_prod):
                wlo, whi = write(i)
                for (_buf, rlo, rhi) in reads:
                    if ranges_overlap(wlo, whi, rlo, rhi):
                        ps.add(i)
                        break
        parents.append(sorted(ps))
    return parents


def graph_stats(parents, n_producer):
    """Degree / interval-tightness / false-edge statistics for one bipartite graph."""
    degs, widths, tights = [], [], []
    contiguous = 0
    n_with_deps = 0
    for ps in parents:
        if not ps:
            continue
        n_with_deps += 1
        d = len(ps)
        lo, hi = min(ps), max(ps)
        w = hi - lo + 1
        degs.append(d)
        widths.append(w)
        tights.append(d / w)
        if w == d:
            contiguous += 1
    if not degs:
        return None
    n = len(degs)
    return {
        "consumers_with_deps": n_with_deps,
        "degree_min": min(degs),
        "degree_max": max(degs),
        "degree_mean": sum(degs) / n,
        "interval_width_mean": sum(widths) / n,
        "interval_tightness_mean": sum(tights) / n,
        "false_edge_rate": 1.0 - (sum(degs) / sum(widths)),
        "contiguous_pct": 100.0 * contiguous / n,
        "encoding": {
            "exact_adjacency_entries": sum(degs),
            "interval_entries": 2 * n,
            "storage_ratio_interval_vs_exact":
                (2 * n) / sum(degs) if sum(degs) else 0.0,
        },
    }


# --------------------------------------------------------------------- model builders

def ceil_div(a, b):
    return -(-a // b)


def qwen_ffn_chain(tokens, hidden=5120, inter=17408, bm=128, bn=128):
    """Qwen3.6-27B FFN: RMSNorm -> gate/up GEMM -> SwiGLU -> down GEMM.

    The interesting pair is norm -> GEMM: a GEMM output tile (m, n) needs the FULL hidden
    row for its BM tokens, so it depends on every norm CTA covering token rows
    [m*BM, (m+1)*BM). High degree, contiguous.
    """
    # RMSNorm: one CTA per token, writes that token's whole hidden row.
    norm = Kernel(
        name="rmsnorm",
        n_cta=tokens,
        write=lambda c: (c * hidden, (c + 1) * hidden - 1),
        read=lambda c: [("x", c * hidden, (c + 1) * hidden - 1)],
        out_buf="normed",
        note="1 CTA per token",
    )
    # gate/up GEMM: 2D tiling over (tokens, inter). CTA id = m * n_tiles_n + n.
    m_tiles = ceil_div(tokens, bm)
    n_tiles = ceil_div(inter, bn)

    def gemm_read(c):
        m = c // n_tiles
        lo = m * bm
        hi = min(tokens, lo + bm) - 1
        # needs the entire hidden row for each of its BM tokens
        return [("normed", lo * hidden, (hi + 1) * hidden - 1)]

    def gemm_write(c):
        m, n = c // n_tiles, c % n_tiles
        base = m * bm * inter + n * bn
        return (base, base + bn - 1)

    gemm = Kernel(
        name=f"gate_up_gemm(BM={bm},BN={bn})",
        n_cta=m_tiles * n_tiles,
        write=gemm_write,
        read=gemm_read,
        out_buf="gu",
        note=f"{m_tiles}x{n_tiles} tiles",
    )
    return [("rmsnorm -> gate/up GEMM", norm, gemm)]


def qwen_deltanet_chain(seq, chunk=64):
    """Gated DeltaNet chunked recurrence: intra-chunk kernel -> inter-chunk scan.

    48 of Qwen3.6-27B's 64 layers are DeltaNet, so this is the DOMINANT pattern in that
    model. Chunk i+1's state depends on chunk i's -> a long 1-to-1 chain, the single most
    favourable shape for CTA-level dependencies (O(1) interval, and the chain length means
    the benefit accumulates).
    """
    n_chunks = ceil_div(seq, chunk)
    intra = Kernel(
        name="deltanet_intra_chunk",
        n_cta=n_chunks,
        write=lambda c: (c, c),                    # one partial state per chunk
        read=lambda c: [("x", c, c)],
        out_buf="partial",
        note=f"{n_chunks} chunks of {chunk}",
    )
    # Sequential scan: chunk c's final state needs its own partial plus the running state,
    # i.e. it depends on the immediately preceding chunk. 1-to-1 chain.
    scan = Kernel(
        name="deltanet_inter_chunk_scan",
        n_cta=n_chunks,
        write=lambda c: (c, c),
        read=lambda c: [("partial", max(0, c - 1), c)],
        out_buf="state",
        note="sequential chunk recurrence",
    )
    return [("DeltaNet intra-chunk -> inter-chunk scan", intra, scan)]


def dsa_chain(seq, key_block=128, query_block=64, topk=2048):
    """DeepSeek/GLM DSA: lightning indexer -> top-k selection -> sparse MLA attention.

    Two pairs matter, for opposite reasons:

    indexer -> topk : topk for query block j needs ALL key-direction scores for j, so it
        depends on the whole row of indexer CTAs. Degree = seq/key_block (thousands at 1M
        context) but the parents are a CONTIGUOUS RANGE -> O(1) interval. The textbook
        "high degree, simple structure" case.

    topk -> attention : LOOKS like indirect access (KV[idx[i]]), but the KV entries were
        written by EARLIER decode steps, not by topk. The only real inter-kernel RAW edge is
        on idx itself, which is 1-to-1 per query block. This is why BlockMaestro's
        Algorithm 1 over-approximates here: it bails out on "address derives from a global
        load" without asking WHEN that data was produced.
    """
    q_blocks = ceil_div(seq, query_block)
    k_blocks = ceil_div(seq, key_block)

    # indexer: 2D grid (query_block, key_block), CTA id = q * k_blocks + k
    indexer = Kernel(
        name="lightning_indexer",
        n_cta=q_blocks * k_blocks,
        write=lambda c: (c, c),                    # score tile
        read=lambda c: [("h", 0, 0)],
        out_buf="scores",
        note=f"{q_blocks}x{k_blocks} score tiles",
    )
    topk = Kernel(
        name=f"topk_select(k={topk})",
        n_cta=q_blocks,
        write=lambda c: (c, c),
        # query block c needs its entire row of score tiles
        read=lambda c: [("scores", c * k_blocks, (c + 1) * k_blocks - 1)],
        out_buf="idx",
        note="one CTA per query block",
    )
    attn = Kernel(
        name="sparse_mla_attention",
        n_cta=q_blocks,
        write=lambda c: (c, c),
        # depends ONLY on its own idx entry; the KV it gathers is historical data
        read=lambda c: [("idx", c, c)],
        out_buf="o",
        note="indirect KV gather, but KV is NOT produced by topk",
    )
    return [
        ("DSA indexer -> topk", indexer, topk),
        ("DSA topk -> sparse attention", topk, attn),
    ]


def glm_indexshare_chain(seq, query_block=64, layers=4):
    """GLM-5.2 IndexShare: one indexer feeds 4 consecutive attention layers.

    The index array is produced SEVERAL kernels upstream, which is exactly the constraint a
    prologue-inspector design needs ("the structure array must not be written by the
    immediately preceding producer"). Also a real sample of A1 span > 1.
    """
    q_blocks = ceil_div(seq, query_block)
    topk = Kernel(
        name="topk_select(shared)",
        n_cta=q_blocks,
        write=lambda c: (c, c),
        read=lambda c: [("scores", c, c)],
        out_buf="idx",
        note="computed once for 4 layers",
    )
    pairs = []
    for L in range(layers):
        attn = Kernel(
            name=f"sparse_attn_L{L}",
            n_cta=q_blocks,
            write=lambda c: (c, c),
            read=lambda c: [("idx", c, c)],
            out_buf=f"o{L}",
            note=f"span={L+1} from the shared indexer",
        )
        pairs.append((f"IndexShare topk -> attn(L{L}) [span={L+1}]", topk, attn))
    return pairs


MODELS = {
    "qwen3.6-27b": lambda a: (
        qwen_deltanet_chain(a.seq) + qwen_ffn_chain(a.tokens)
    ),
    "glm-5.2-dsa": lambda a: (
        dsa_chain(a.seq) + glm_indexshare_chain(a.seq)
    ),
    "deepseek-v3.2-dsa": lambda a: dsa_chain(a.seq),
}


# --------------------------------------------------------------------- reporting

def report_pair(label, prod, cons, stats, sampled=False, n_sampled=0):
    print(f"\n--- {label}")
    print(f"    producer: {prod.name:34s} {prod.n_cta:>8} CTAs   {prod.note}")
    print(f"    consumer: {cons.name:34s} {cons.n_cta:>8} CTAs   {cons.note}")
    if sampled:
        print(f"    (stats sampled over {n_sampled} of {cons.n_cta} consumer CTAs)")
    if not stats:
        print("    (no dependencies)")
        return
    s = stats
    print(f"    degree      : min={s['degree_min']} max={s['degree_max']} "
          f"mean={s['degree_mean']:.1f}")
    print(f"    interval    : mean width={s['interval_width_mean']:.1f}  "
          f"tightness={s['interval_tightness_mean']:.4f}  "
          f"contiguous={s['contiguous_pct']:.1f}%")
    print(f"    false edges : {100.0 * s['false_edge_rate']:.2f}%")
    e = s["encoding"]
    print(f"    encoding    : exact adjacency {e['exact_adjacency_entries']} entries  vs  "
          f"interval {e['interval_entries']} entries "
          f"({e['storage_ratio_interval_vs_exact']:.3f}x)")

    d, t = s["degree_mean"], s["interval_tightness_mean"]
    if d > 32 and t > 0.99:
        print(f"    >> HIGH DEGREE ({d:.0f}) but EXACT interval encoding.")
        print(f"       BlockMaestro's 'degree > 32 => no benefit' would wrongly exclude this.")
    elif t < 0.5:
        print(f"    >> Interval encoding is lossy here (tightness {t:.3f});")
        print(f"       a mask or exact adjacency is needed (dimension A3).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.6-27b", choices=sorted(MODELS))
    ap.add_argument("--tokens", type=int, default=4096, help="batch tokens for GEMM tiling")
    ap.add_argument("--seq", type=int, default=32768, help="sequence length")
    ap.add_argument("--sample", type=int, default=256,
                    help="max consumer CTAs to analyze per pair (0 = exact, no sampling). "
                         "Stats are consumer means, so sampling is sound; the default keeps "
                         "even 1M-context DSA fast.")
    ap.add_argument("--json", help="write stats as JSON")
    args = ap.parse_args()

    print(f"=== CTA-level dependency oracle: {args.model} ===")
    print(f"tokens={args.tokens} seq={args.seq}")
    print("\nDerived analytically from the tile decomposition — no GPU, no instrumentation.")

    pairs = MODELS[args.model](args)
    out = []
    for label, prod, cons in pairs:
        ids, sampled = sample_consumers(cons.n_cta, args.sample)
        parents = build_graph(prod, cons, ids)
        stats = graph_stats(parents, prod.n_cta)
        if stats and sampled:
            stats["sampled_consumers"] = len(ids)
        report_pair(label, prod, cons, stats, sampled=sampled, n_sampled=len(ids))
        out.append({
            "pair": label,
            "producer": prod.name, "producer_ctas": prod.n_cta,
            "consumer": cons.name, "consumer_ctas": cons.n_cta,
            "sampled": sampled,
            "stats": stats,
        })

    print("\n" + "=" * 74)
    print("Interpretation")
    print("=" * 74)
    print("tightness ~= 1.0  -> interval encoding [lo,hi] is EXACT; O(1) storage suffices")
    print("tightness << 1.0  -> interval wait pulls in false parents; needs mask/adjacency")
    print("high degree + tightness 1.0 -> the case E1 says must be measured SEPARATELY from")
    print("                               structural complexity, because the two are")
    print("                               conflated in BlockMaestro's n-group injection.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
