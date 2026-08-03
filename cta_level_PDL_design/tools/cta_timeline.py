#!/usr/bin/env python3
"""Reconstruct the CTA-level timeline from cta_trace.cuh CSV dumps.

Consumes the per-CTA records written by primitive 1 and produces the four metrics the
evaluation plan needs:

  1. pre-dependency phase distribution (prologue + any dependency wait)
  2. CTA concurrency over time       -> BlockMaestro Fig.10 analogue
  3. producer/consumer SM affinity   -> how often a consumer lands on its parent's SM
  4. grid overlap ratio              -> how much the two kernels actually overlapped

All timestamps come from %globaltimer (nanoseconds, consistent across SMs), which is why
cross-SM ordering here is meaningful at all. clock64() would NOT be valid for this.

Usage:
    python3 tools/cta_timeline.py results/trace_interval.csv
    python3 tools/cta_timeline.py results/*.csv --json out.json
    python3 tools/cta_timeline.py results/trace.csv --plot timeline.png

Only stdlib is required; matplotlib is optional and used solely by --plot.
"""

import argparse
import csv
import glob
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    """Read one trace CSV into a list of record dicts."""
    recs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                recs.append({
                    "tag": row["tag"],
                    "kernel_id": int(row["kernel_id"]),
                    "block_id": int(row["block_id"]),
                    "sm_id": int(row["sm_id"]),
                    "t_launch": int(row["t_launch"]),
                    "t_dep": int(row["t_dep_satisfied"]),
                    "t_end": int(row["t_end"]),
                })
            except (KeyError, ValueError):
                continue
    # Records whose t_end is 0 were never written (unused CTA slot).
    return [r for r in recs if r["t_end"] > 0]


def normalize(recs):
    """Shift timestamps so the first CTA launch is t=0. Keeps numbers readable."""
    if not recs:
        return recs
    t0 = min(r["t_launch"] for r in recs)
    for r in recs:
        r["t_launch"] -= t0
        r["t_dep"] -= t0
        r["t_end"] -= t0
    return recs


def stall_stats(recs):
    """Time from CTA entry to the post-wait timestamp.

    The trace schema has no timestamp between the independent prologue and dependency
    wait, so this quantity is *not* an isolated dependency stall. Keep the historical
    JSON field names for compatibility, but label the metric honestly in the report.
    """
    out = {}
    for kid in sorted({r["kernel_id"] for r in recs}):
        ks = [r for r in recs if r["kernel_id"] == kid]
        abs_stall, rel_stall = [], []
        for r in ks:
            stall = r["t_dep"] - r["t_launch"]
            exec_t = r["t_end"] - r["t_dep"]
            abs_stall.append(stall)
            if exec_t > 0:
                rel_stall.append(stall / exec_t)
        if not abs_stall:
            continue
        s = sorted(abs_stall)
        out[kid] = {
            "n_ctas": len(ks),
            "stall_ns": {
                "min": s[0],
                "p25": s[len(s) // 4],
                "median": statistics.median(s),
                "p75": s[3 * len(s) // 4],
                "max": s[-1],
                "mean": statistics.fmean(s),
            },
            "stall_over_exec": {
                "median": statistics.median(rel_stall) if rel_stall else 0.0,
                "mean": statistics.fmean(rel_stall) if rel_stall else 0.0,
                "max": max(rel_stall) if rel_stall else 0.0,
            },
        }
    return out


def concurrency(recs, buckets=200):
    """Compute exact peak/time-weighted mean CTA residency and a sampled plot series.

    Intervals are half-open [t_launch, t_end). Exact endpoint deltas avoid the previous
    coarse-bin overcount, where CTAs ending and starting in the same bucket were counted
    as simultaneously resident.
    """
    if not recs:
        return {}
    t_max = max(r["t_end"] for r in recs)
    if t_max <= 0:
        return {}
    sample_step = t_max / buckets

    groups = {"all": recs}
    for kid in sorted({r["kernel_id"] for r in recs}):
        groups[str(kid)] = [r for r in recs if r["kernel_id"] == kid]

    peaks, means, series = {}, {}, {}
    for key, group in groups.items():
        deltas = defaultdict(int)
        for r in group:
            deltas[r["t_launch"]] += 1
            deltas[r["t_end"]] -= 1
        event_times = sorted(deltas)

        active = 0
        peak = 0
        area = 0
        previous = 0
        for timestamp in event_times:
            area += active * (timestamp - previous)
            active += deltas[timestamp]
            peak = max(peak, active)
            previous = timestamp
        peaks[key] = peak
        means[key] = area / t_max

        sampled = []
        active = 0
        event_index = 0
        for bucket in range(buckets + 1):
            timestamp = bucket * sample_step
            while (event_index < len(event_times) and
                   event_times[event_index] <= timestamp):
                active += deltas[event_times[event_index]]
                event_index += 1
            sampled.append(active)
        series[key] = sampled

    return {
        "method": "exact_half_open_endpoint_sweep",
        "bucket_ns": sample_step,
        "buckets": buckets,
        "peak": peaks,
        "mean": means,
        "series": series,
    }


def overlap(recs):
    """Wall-clock overlap between kernel 0 and kernel 1."""
    spans = {}
    for kid in {r["kernel_id"] for r in recs}:
        ks = [r for r in recs if r["kernel_id"] == kid]
        spans[kid] = (min(r["t_launch"] for r in ks), max(r["t_end"] for r in ks))
    if 0 not in spans or 1 not in spans:
        return None
    (a0, a1), (b0, b1) = spans[0], spans[1]
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return {
        "producer_span_ns": a1 - a0,
        "consumer_span_ns": b1 - b0,
        "overlap_ns": inter,
        "overlap_pct_of_union": 100.0 * inter / union if union else 0.0,
        "overlap_pct_of_producer": 100.0 * inter / (a1 - a0) if a1 > a0 else 0.0,
    }


def sm_affinity(recs):
    """How often does consumer CTA j run on the same SM as producer CTA j?

    Only meaningful for self-dependency-like patterns, but the SM histogram is informative
    regardless -- it shows whether the scheduler spreads or clusters CTAs.
    """
    prod = {r["block_id"]: r["sm_id"] for r in recs if r["kernel_id"] == 0}
    cons = {r["block_id"]: r["sm_id"] for r in recs if r["kernel_id"] == 1}
    if not prod or not cons:
        return None
    shared = set(prod) & set(cons)
    if not shared:
        return None
    same = sum(1 for b in shared if prod[b] == cons[b])
    hist = defaultdict(int)
    for sm in cons.values():
        hist[sm] += 1
    return {
        "compared_ctas": len(shared),
        "same_sm": same,
        "same_sm_pct": 100.0 * same / len(shared),
        "random_baseline_pct": 100.0 / len(set(prod.values())) if prod else 0.0,
        "consumer_sms_used": len(hist),
        "max_ctas_on_one_sm": max(hist.values()),
    }


def analyze(path):
    recs = normalize(load(path))
    if not recs:
        return {"file": path, "error": "no usable records"}
    return {
        "file": path,
        "tag": recs[0]["tag"],
        "n_records": len(recs),
        "span_ns": max(r["t_end"] for r in recs),
        "pre_dependency_phase_note": (
            "t_dep-t_launch includes the independent prologue and any wait; "
            "it is not an isolated dependency-stall measurement"
        ),
        "stalls": stall_stats(recs),
        "concurrency": concurrency(recs),
        "overlap": overlap(recs),
        "sm_affinity": sm_affinity(recs),
    }


def report(a):
    print(f"\n=== {a['file']} ===")
    if "error" in a:
        print(f"  {a['error']}")
        return
    print(f"tag={a['tag']}  records={a['n_records']}  span={a['span_ns']/1e6:.3f} ms")

    print("\n-- pre-dependency phase (independent prologue + any wait; NOT pure stall)")
    for kid, s in a["stalls"].items():
        name = {0: "producer", 1: "consumer"}.get(kid, f"kernel{kid}")
        st = s["stall_ns"]
        print(f"  {name:9s} n={s['n_ctas']:6d}  stall p25/med/p75 = "
              f"{st['p25']/1e3:8.1f} / {st['median']/1e3:8.1f} / {st['p75']/1e3:8.1f} us"
              f"   stall/exec median={s['stall_over_exec']['median']:.2f}")

    c = a["concurrency"]
    if c:
        print("\n-- CTA concurrency (exact half-open endpoint sweep)")
        for k in sorted(c["peak"], key=lambda x: (x != "all", x)):
            print(f"  {k:9s} peak={c['peak'][k]:6d}  mean={c['mean'][k]:8.1f}")

    if a["overlap"]:
        o = a["overlap"]
        print("\n-- grid overlap")
        print(f"  producer span {o['producer_span_ns']/1e6:.3f} ms, "
              f"consumer span {o['consumer_span_ns']/1e6:.3f} ms")
        print(f"  overlap {o['overlap_ns']/1e6:.3f} ms "
              f"({o['overlap_pct_of_producer']:.1f}% of producer, "
              f"{o['overlap_pct_of_union']:.1f}% of union)")

    if a["sm_affinity"]:
        s = a["sm_affinity"]
        print("\n-- producer/consumer SM affinity (C1: is reuse even possible?)")
        print(f"  same SM for {s['same_sm']}/{s['compared_ctas']} CTAs "
              f"= {s['same_sm_pct']:.1f}%  (random baseline {s['random_baseline_pct']:.1f}%)")
        print(f"  consumer used {s['consumer_sms_used']} SMs, "
              f"max {s['max_ctas_on_one_sm']} CTAs on one SM")


def plot(a, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping --plot", file=sys.stderr)
        return
    c = a["concurrency"]
    if not c:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = [i * c["bucket_ns"] / 1e6 for i in range(c["buckets"] + 1)]
    for k, series in sorted(c["series"].items()):
        if k == "all":
            ax.plot(xs, series, label="all", lw=2, color="black")
        else:
            name = {"0": "producer", "1": "consumer"}.get(k, f"kernel{k}")
            ax.fill_between(xs, series, alpha=0.4, label=name)
    ax.set_xlabel("time (ms, from first CTA launch)")
    ax.set_ylabel("resident CTAs")
    ax.set_title(f"CTA concurrency — {a['tag']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="trace CSV files (globs ok)")
    ap.add_argument("--json", help="write all results as JSON")
    ap.add_argument("--plot", help="write a concurrency plot (first file only)")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) or [f])

    results = []
    for p in paths:
        try:
            a = analyze(p)
        except OSError as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        results.append(a)
        report(a)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")
    if args.plot and results:
        plot(results[0], args.plot)


if __name__ == "__main__":
    main()
