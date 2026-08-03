#!/usr/bin/env python3
"""Render an nsys report (exported to SQLite) to PNG images with matplotlib.

Outputs:
  diamond_pdl_timeline.png  - measured GPU timeline, one bar per CUDA-graph launch (2 lanes)
  diamond_pdl_model.png     - to-scale schematic showing WHERE the PDL speedup comes from

Usage:  python3 nsys_plot.py [diamond_pdl.sqlite]
"""
import sqlite3, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DB = sys.argv[1] if len(sys.argv) > 1 else "diamond_pdl.sqlite"
RED, GRN, BLU, ORG = "#e5534b", "#3fb950", "#58a6ff", "#d29922"


def load():
    c = sqlite3.connect(DB)
    dev = "unknown"
    try: dev = c.execute("select name from TARGET_INFO_GPU limit 1").fetchone()[0]
    except Exception: pass
    rows = c.execute("select start,end,graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                     "order by start").fetchall()
    t0 = rows[0][0]
    L = [{"start": (s-t0)/1e6, "dur": (e-s)/1e6, "g": g} for s, e, g, in rows]
    grp = {}
    for x in L: grp.setdefault(x["g"], []).append(x["dur"])
    med = {g: st.median(v) for g, v in grp.items()}
    order = sorted(med, key=lambda g: med[g])
    pdl_g, plain_g = order[0], order[-1]
    return dev, L, plain_g, pdl_g, med[plain_g], med[pdl_g], med[plain_g]/med[pdl_g]


def plot_timeline(dev, L, plain_g, pdl_g, mp, md, sp):
    fig, ax = plt.subplots(figsize=(13, 3.6), dpi=130)
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    laneY = {plain_g: 1.0, pdl_g: 0.0}
    for x in L:
        c = RED if x["g"] == plain_g else GRN
        ax.add_patch(Rectangle((x["start"], laneY[x["g"]]-0.32), x["dur"], 0.64,
                               facecolor=c, edgecolor="#0d1117", linewidth=0.6, alpha=0.9))
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([f"PLAIN (ordinary)\n~{mp:.1f} ms", f"PDL (programmatic)\n~{md:.1f} ms"],
                       color="#c9d1d9", fontsize=11)
    ax.set_ylim(-0.6, 1.6)
    total = max(x["start"]+x["dur"] for x in L)
    ax.set_xlim(0, total*1.005)
    ax.set_xlabel("time (ms)", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values(): s.set_color("#30363d")
    ax.set_title(f"{dev} — diamond CUDA graph on GPU   |   PDL speedup = {sp:.2f}x",
                 color="#e6edf3", fontsize=13, pad=10)
    ax.grid(axis="x", color="#21262d", linewidth=0.6)
    fig.tight_layout()
    fig.savefig("diamond_pdl_timeline.png", facecolor=fig.get_facecolor())
    print("wrote diamond_pdl_timeline.png")


def bar(ax, x, y, w, c, label):
    ax.add_patch(Rectangle((x, y-0.4), w, 0.8, facecolor=c, edgecolor="#0d1117", linewidth=1))
    ax.text(x+0.04, y, label, va="center", ha="left", fontsize=9, color="#0d1117", fontweight="bold")


def plot_model(sp):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.4), dpi=130, sharex=True)
    for ax in (a1, a2):
        ax.set_facecolor("#0d1117"); ax.set_xlim(-0.05, 4.2); ax.set_ylim(-0.7, 3.7)
        ax.set_yticks([]); ax.tick_params(colors="#8b949e")
        for s in ax.spines.values(): s.set_color("#30363d")
        for k in range(5):
            ax.axvline(k, color="#30363d", ls="--", lw=0.7)
    fig.patch.set_facecolor("#0d1117")
    # PLAIN = 4T
    bar(a1, 0, 3, 1, BLU, "producer  T")
    bar(a1, 1, 2, 2, GRN, "midA  prologue T + tail T")
    bar(a1, 1, 1, 2, GRN, "midB  (∥ midA)")
    bar(a1, 3, 0, 1, ORG, "final  T")
    a1.set_title("PLAIN — ordinary edges  ≈ 4T  (each stage completes before the next)",
                 color=RED, fontsize=12, loc="left")
    # PDL = 2T
    bar(a2, 0, 3, 1, BLU, "producer  T")
    bar(a2, 0, 2, 2, GRN, "midA  prologue ‖ P.tail, then tail")
    bar(a2, 0, 1, 2, GRN, "midB  (∥ midA)")
    bar(a2, 1, 0, 1, ORG, "final  prologue ‖ mid.tail")
    a2.set_title(f"PDL — programmatic edges  ≈ 2T  ({sp:.2f}x)  "
                 f"(each prologue overlaps the previous tail)", color=GRN, fontsize=12, loc="left")
    a2.set_xlabel("time (units of T)", color="#8b949e")
    a2.set_xticks(range(5)); a2.set_xticklabels([f"{k}T" for k in range(5)])
    fig.suptitle("Where the 2x comes from: vertical overlap (midA∥midB is present in both)",
                 color="#e6edf3", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("diamond_pdl_model.png", facecolor=fig.get_facecolor())
    print("wrote diamond_pdl_model.png")


def load_nodes():
    """Per-kernel (node granularity). Returns dev + {variant: {total,T,kernels}} for one launch each."""
    c = sqlite3.connect(DB)
    dev = "unknown"
    try: dev = c.execute("select name from TARGET_INFO_GPU limit 1").fetchone()[0]
    except Exception: pass
    def nm(i):
        r = c.execute("select value from StringIds where id=?", (i,)).fetchone()
        return r[0] if r else "?"
    rows = c.execute("select start,end,shortName,streamId,graphId from "
                     "CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    byg = {}
    for s, e, sn, stid, g in rows:
        byg.setdefault(g, []).append((s, e, nm(sn), stid))
    variants = {}
    tot = {}
    for g, rr in byg.items():
        ll = sorted(rr)[-4:]                       # last launch = last 4 kernels
        t0 = min(r[0] for r in ll); t1 = max(r[1] for r in ll)
        tot[g] = (t1 - t0) / 1e6
        prod_dur = next((e - s) for s, e, n, st in ll if n == "k_prod") / 1e6
        variants[g] = {"total": tot[g], "T": prod_dur,
                       "kernels": [(n, (s-t0)/1e6, (e-t0)/1e6, st) for s, e, n, st in sorted(ll)]}
    order = sorted(tot, key=lambda g: tot[g])
    pdl_g, plain_g = order[0], order[-1]
    return dev, {"PLAIN (ordinary edges)": variants[plain_g],
                 "PDL (programmatic edges)": variants[pdl_g]}


def plot_kernels(dev, data):
    xmax = max(v["total"] for v in data.values()) * 1.03
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.6), dpi=130, sharex=True)
    fig.patch.set_facecolor("#0d1117")
    color = {"k_prod": BLU, "k_mid": GRN, "k_fin": ORG}
    yrow = {"k_prod": 3, "k_fin": 0}
    titles = {"PLAIN (ordinary edges)": RED, "PDL (programmatic edges)": GRN}
    for ax, (label, v) in zip(axes, data.items()):
        ax.set_facecolor("#0d1117")
        midrow = [2, 1]; mi = 0
        seen = {}
        for n, s, e, st in v["kernels"]:
            if n == "k_mid":
                y = midrow[mi]; mi += 1; disp = f"midA" if y == 2 else "midB"
            else:
                y = yrow[n]; disp = n.replace("k_", "")
            ax.add_patch(Rectangle((s, y-0.4), max(e-s, 0.05), 0.8,
                         facecolor=color[n], edgecolor="#0d1117", lw=1))
            ax.text(s+0.15, y, f"{disp}  (stream {st})", va="center", ha="left",
                    fontsize=9, color="#0d1117", fontweight="bold")
        T = v["T"]
        for k in range(int(xmax/T)+2):
            ax.axvline(k*T, color="#30363d", ls="--", lw=0.7)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.7, 3.7); ax.set_yticks([])
        ax.tick_params(colors="#8b949e")
        for sp_ in ax.spines.values(): sp_.set_color("#30363d")
        ax.set_title(f"{label}   —   total {v['total']:.2f} ms   (T≈{T:.1f} ms)",
                     color=titles[label], fontsize=12, loc="left")
    axes[1].set_xlabel("time (ms)", color="#8b949e")
    fig.suptitle(f"{dev} — diamond graph, per-kernel timeline (measured, node granularity)",
                 color="#e6edf3", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("diamond_node_kernels.png", facecolor=fig.get_facecolor())
    print("wrote diamond_node_kernels.png")


def has_kernels():
    c = sqlite3.connect(DB)
    try:
        return c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] > 0
    except Exception:
        return False


if __name__ == "__main__":
    if has_kernels():
        dev, data = load_nodes()
        for lab, v in data.items():
            print(f"{lab}: total {v['total']:.2f} ms")
        plot_kernels(dev, data)
    else:
        dev, L, plain_g, pdl_g, mp, md, sp = load()
        print(f"device={dev}  PLAIN~{mp:.2f}ms  PDL~{md:.2f}ms  speedup={sp:.2f}x  ({len(L)} launches)")
        plot_timeline(dev, L, plain_g, pdl_g, mp, md, sp)
        plot_model(sp)
