"""Figures of the first-look calibration (publication drafts), from the result files only (no sampling).

    python plot_firstlook.py <dir with cal1.jsonl, cal2.jsonl, validate3/4.jsonl, study2.jsonl> <out dir>
Figures:
  fl_dlnz_hist.pdf    dlnZ of the reference runs: no-moon vs moon light curves (the Occam penalty and the threshold)
  fl_recall_snr.pdf   detected fraction vs moon photometric S/N and vs M_moon/M_planet, per regime (reference dlnZ >= 8
                      and preset 0.95 detection)
  fl_cost_recall.pdf  measured recall vs CPU-s per light curve of the presets and of single-run settings
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402


def load(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def main(d, out):
    os.makedirs(out, exist_ok=True)
    ref = [r for f in ("cal1.jsonl", "cal2.jsonl") for r in load(os.path.join(d, f)) if r["config"] == "ref"]
    # 1. dlnZ histogram
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    nul = np.array([r["dlnz"] for r in ref if not r["moon"]])
    mo = np.array([r["dlnz"] for r in ref if r["moon"]])
    bins = np.concatenate([np.linspace(-12, 20, 65), [1e5]])
    ax.hist(np.clip(nul, -12, 1e5), bins=bins, histtype="step", color="k", label=f"no moon (n = {len(nul)})")
    ax.hist(np.clip(mo, -12, 1e5), bins=bins, histtype="stepfilled", alpha=0.4, color="C0",
            label=f"moon injected (n = {len(mo)})")
    ax.axvline(3, color="C3", ls="--", label="threshold 3")
    ax.axvline(8, color="C2", ls=":", label="well detectable (8)")
    ax.set_xlim(-12, 21)
    ax.set_xlabel(r"$\Delta \ln Z$ (reference run; values > 20 in the last bin)")
    ax.set_ylabel("light curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fl_dlnz_hist.pdf"))
    plt.close(fig)
    # 2. detected fraction vs moon S/N and mass ratio, per regime group
    val = {}
    for f in ("validate3.jsonl", "validate4.jsonl"):
        for r in load(os.path.join(d, f)):
            if r["preset"] == 0.95 and r["has_moon"]:
                val.setdefault((r["regime"], r["index"]), []).append(r["interesting"])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for grp, col in (("kepler", "C0"), ("plato", "C1")):
        rr = [r for r in ref if r["moon"] and r["regime"].startswith(grp)]
        for ax, key, xl, lg in ((axs[0], "snr", "moon photometric S/N", True),
                                (axs[1], "q", r"$M_\mathrm{moon} / M_\mathrm{planet}$", True)):
            x = np.array([r["snr"] if key == "snr" else r["truth"]["M_moon"] / r["truth"]["M_planet"] for r in rr])
            det = np.array([r["dlnz"] >= 8 for r in rr], float)
            fl = np.array([np.mean(val.get((r["regime"], r["index"]), [np.nan])) if r["dlnz"] >= 8 else 0.0 for r in rr])
            edges = np.geomspace(max(x.min(), 1e-4), x.max() * 1.001, 9)
            c = np.sqrt(edges[1:] * edges[:-1])
            k = np.digitize(x, edges) - 1
            f_det = [det[k == i].mean() if np.any(k == i) else np.nan for i in range(len(c))]
            f_fl = [np.nanmean(fl[k == i]) if np.any(k == i) else np.nan for i in range(len(c))]
            ax.plot(c, f_det, "o-", color=col, label=f"{grp}-like: reference dlnZ ≥ 8")
            ax.plot(c, f_fl, "s--", color=col, alpha=0.6, label=f"{grp}-like: first look (0.95) flags")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(xl)
    axs[0].set_ylabel("fraction of injected moons")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fl_recall_snr.pdf"))
    plt.close(fig)
    # 3. cost vs recall
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    pts = {0.9: (0.35, None), 0.95: (0.61, None), 0.99: (1.58, None), 0.999: (3.07, None)}
    for f in ("validate3.jsonl", "validate4.jsonl"):
        rows = load(os.path.join(d, f))
        for p in pts:
            rr = [r["interesting"] for r in rows if r["preset"] == p and r["has_moon"]]
            if rr and (f == "validate4.jsonl" or p in (0.9, 0.95)):
                pts[p] = (pts[p][0], (1 - np.mean(rr), len(rr) - sum(rr), len(rr)))
    for p, (cpu, m) in pts.items():
        if m is None:
            continue
        miss = max(m[0], 0.5 / m[2])
        ax.errorbar(cpu, 100 * miss, fmt="o", color="C3" if m[1] else "C2")
        ax.annotate(f"recall={p}", (cpu, 100 * miss), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU seconds per light curve (one core)")
    ax.set_ylabel("missed well-detectable moons [%]")
    ax.set_title("first look: measured miss rate vs cost (0 misses plotted at 0.5/n)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fl_cost_recall.pdf"))
    plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
