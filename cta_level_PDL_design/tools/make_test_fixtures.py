#!/usr/bin/env python3
"""Generate synthetic benchmark output so the analysis toolchain can be validated offline.

Rent time is expensive and the analysis scripts must work on the FIRST try. This produces
plausible SUMMARY lines and CTA trace CSVs with the shapes the real benchmarks emit, so
analyze.py / cta_timeline.py / llm_bracket.py can be exercised end-to-end without a GPU.

The numbers are invented. Only the FORMAT is meaningful.

Usage:
    python3 tools/make_test_fixtures.py --out /tmp/ctafix
    python3 tools/analyze.py      /tmp/ctafix/summary.txt
    python3 tools/cta_timeline.py /tmp/ctafix/trace.csv
    python3 tools/llm_bracket.py  /tmp/ctafix/summary_llm.txt
"""

import argparse
import os
import random


def gen_summary(path, sms=148):
    L = [f'SUMMARY tier0=device name="NVIDIA B300" sms={sms} cc=10.3 ghz=1.900']

    # Tier 0.1 chain overlap: saturating near depth 2 (the interesting negative result).
    for st in range(1, 7):
        off = st * 2.0
        on = off / (2 * st / (st + 1))
        L.append(f"SUMMARY tier0=chain stages={st} pdl_off_ms={off:.5f} "
                 f"pdl_on_ms={on:.5f} speedup={off/on:.4f} implied_depth={min(st,2.1):.3f}")

    L.append("SUMMARY tier0=clc status=ok clusters=4096 median_ms=0.812 attempts=5104 "
             "successes=4096 success_rate=0.8025 cyc_per_attempt=431.20 "
             "duplicate_claims=0 unclaimed=0")

    for smem in (0, 8, 16, 32, 64):
        occ = max(1, 32 - smem // 2)
        L.append(f"SUMMARY tier0=occupancy smem_kb={smem} threads=128 ctas_per_sm={occ} "
                 f"concurrent_ctas={occ*sms} median_ms={1.0+smem*0.01:.5f}")

    for scope, ns in (("none", 0.0), ("cta", 12.3), ("gpu", 48.7), ("sys", 191.4)):
        L.append(f"SUMMARY tier0=fence scope={scope} median_ms={1.0+ns/1000:.5f} "
                 f"ns_per_fence={ns:.3f}")

    def row(tag, struct, deg, g, tight, space):
        floor = 10.0
        ceil = floor * (1 - space / 100)
        return (f"SUMMARY tag={tag} structure={struct} degree={deg} "
                f"eff_degree={deg/tight:.2f} tightness={tight:.4f} "
                f"producers={g} consumers={g} threads=128 smem_kb=0 occ_per_sm=16 sms={sms} "
                f"tail=200000 prologue=200000 pdl=1 "
                f"floor_ms={floor:.5f} ceiling_ms={ceil:.5f} "
                f"spin_ms={floor-space*0.060:.5f} backoff_ms={floor-space*0.070:.5f} "
                f"counter_ms={floor-space*0.050:.5f} exact_ms={floor-space*0.065:.5f} "
                f"space_pct={space:.3f}")

    # Tier 1.1a: degree sweep, structure pinned. Benefit decays with degree AND grid.
    for g in (64, 256, 1024, 4096, 8192):
        for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            if d > g:
                continue
            space = max(0.3, 30.0 * (1 - d / 900) * (1 - g / 11000))
            L.append(row(f"t11a_g{g}_d{d}", "interval", d, g, 1.0, space))

    # Tier 1.1b: structure sweep, degree pinned -> tightness drives the benefit.
    for g in (256, 1024, 4096):
        for s, tight in (("interval", 1.00), ("self", 1.00), ("grouped", 0.90),
                         ("strided", 0.05), ("random", 0.03)):
            L.append(row(f"t11b_g{g}_{s}", s, 32, g, tight, 20.0 * tight))

    # Tier 1.2 tail/prologue ratio
    for r in (1, 2, 4, 8, 16):
        L.append(row(f"t12_r{r}", "interval", 8, 1024, 1.0, min(45.0, 8.0 * r)))

    # Tier 2.1 protocol shootout
    for g in (256, 1024, 4096):
        L.append(row(f"t21_g{g}", "self", 1, g, 1.0, 25.0))

    # Tier 2.3 encoding cost
    for s, tight in (("interval", 1.00), ("strided", 0.05), ("random", 0.03)):
        for d in (4, 16, 64):
            L.append(row(f"t23_{s}_d{d}", s, d, 2048, tight, 22.0 * tight))

    # Tier 0.3 occupancy under a real dependency wait
    for smem in (0, 8, 16, 32, 64):
        L.append(row(f"t03_smem{smem}", "interval", 8, 2048, 1.0, max(1.0, 20.0 - smem * 0.22))
                 .replace("smem_kb=0", f"smem_kb={smem}")
                 .replace("occ_per_sm=16", f"occ_per_sm={max(1, 32 - smem//2)}"))

    open(path, "w").write("\n".join(L) + "\n")
    return len(L)


def gen_trace(path, n_cta=512, sms=148, seed=7):
    rnd = random.Random(seed)
    rows = ["tag,kernel_id,block_id,sm_id,t_launch,t_dep_satisfied,t_end"]
    # producer: no wait, staggered launches
    for b in range(n_cta):
        st = b * 1000
        rows.append(f"synth,0,{b},{b % sms},{st},{st},{st + 900000}")
    # consumer: launches during the producer's tail, then stalls on its dependency
    for b in range(n_cta):
        st = 300000 + b * 1000
        dep = st + rnd.randint(50000, 400000)
        rows.append(f"synth,1,{b},{(b + 3) % sms},{st},{dep},{dep + 700000}")
    open(path, "w").write("\n".join(rows) + "\n")
    return len(rows) - 1


def gen_llm(path):
    L = []
    # Headroom shrinks as batch grows: BS=1 has the smallest grids and the most slack.
    for seq in (4096, 32768):
        for bs, off, grid, ceil in ((1, 100.0, 103.0, 116.0),
                                    (4, 340.0, 349.0, 378.0),
                                    (16, 620.0, 631.0, 654.0),
                                    (64, 800.0, 813.0, 820.0)):
            scale = 1.0 if seq == 4096 else 0.55
            for rung, tps, ceilflag in (("pdl_off", off, 0), ("pdl_grid", grid, 0),
                                        ("ceiling", ceil, 1)):
                L.append(f"SUMMARY tier=4 tag={rung}_bs{bs}_seq{seq} rung={rung} status=ok "
                         f"engine=vllm batch={bs} seq={seq} gen=64 iters=8 "
                         f"median_s={64*bs/(tps*scale):.5f} min_s={64*bs/(tps*scale):.5f} "
                         f"tok_per_s={tps*scale:.3f} "
                         f"tok_per_s_per_user={tps*scale/bs:.3f} "
                         f"pdl_trt={0 if rung=='pdl_off' else 1} "
                         f"pdl_inductor={0 if rung=='pdl_off' else 1} "
                         f"ceiling={ceilflag} verified={1-ceilflag}")
    open(path, "w").write("\n".join(L) + "\n")
    return len(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/ctafix", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    s = os.path.join(args.out, "summary.txt")
    t = os.path.join(args.out, "trace.csv")
    m = os.path.join(args.out, "summary_llm.txt")

    print(f"summary.txt      {gen_summary(s):5d} SUMMARY rows  -> {s}")
    print(f"trace.csv        {gen_trace(t):5d} CTA records   -> {t}")
    print(f"summary_llm.txt  {gen_llm(m):5d} SUMMARY rows  -> {m}")
    print("\nValidate the toolchain with:")
    print(f"  python3 tools/analyze.py      {s}")
    print(f"  python3 tools/cta_timeline.py {t}")
    print(f"  python3 tools/llm_bracket.py  {m}")
    print("\nNumbers are invented; only the format is real.")


if __name__ == "__main__":
    main()
