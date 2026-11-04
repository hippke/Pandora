"""Generate reference fingerprints (reference.json) and inspection PDFs.

Run ONLY when the model output is intentionally changed (e.g. after a
verified bug fix), never to make a failing regression test pass blindly.

    python generate_reference.py            # reference.json + plots/*.pdf
    python generate_reference.py --plots    # PDFs only, reference untouched
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

import cases as C

HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = os.path.join(HERE, "reference.json")
PLOT_DIR = os.path.join(HERE, "plots")

PARAM_KEYS = None  # all keys except description are stored


def _git_commit(path):
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "pandoramoon"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def case_params(c):
    return {k: v for k, v in c.items() if k != "description"}


def plot_case(c, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ep = c["epochs"]
    per_ep = len(r["time"]) // n_ep
    fp, fm, ft = r["flux_planet"], r["flux_moon"], r["flux_total"]
    lo = min(fp.min(), fm.min(), ft.min())
    hi = max(fp.max(), fm.max(), ft.max())
    span = hi - lo if hi > lo else 1e-6
    ylim = (lo - 0.1 * span, hi + 0.1 * span)

    fig, axes = plt.subplots(2, n_ep, figsize=(5.5 * n_ep, 6.4), sharey="row", squeeze=False,
                             gridspec_kw=dict(height_ratios=[2.2, 1]))
    sparse = per_ep < 400
    style = dict(lw=1.0, marker="." if sparse else None, ms=3)
    mlo, mhi = fm.min(), fm.max()
    mspan = mhi - mlo if mhi > mlo else 1e-9
    for e in range(n_ep):
        ax = axes[0, e]
        sl = slice(e * per_ep, (e + 1) * per_ep)
        centre = c["t0_bary"] + e * c["epoch_distance"]
        hours = (r["time"][sl] - centre) * 24
        ax.plot(hours, fp[sl], color="blue", label="planet", **style)
        ax.plot(hours, fm[sl], color="red", label="moon", **style)
        ax.plot(hours, ft[sl], color="black", label="total", **style)
        ax.set_ylim(*ylim)
        ax.set_title(f"epoch {e}", fontsize=9)
        ax.grid(alpha=0.3)
        axm = axes[1, e]
        axm.plot(hours, (fm[sl] - 1) * 1e6, color="red", **style)
        axm.set_ylim((mlo - 1 - 0.1 * mspan) * 1e6, (mhi - 1 + 0.1 * mspan) * 1e6)
        axm.set_xlabel(f"time from epoch {e} centre [h]")
        axm.grid(alpha=0.3)
    axes[0, 0].set_ylabel("relative flux")
    axes[1, 0].set_ylabel("moon only,\nown scale [ppm]")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(c["name"], fontsize=11)
    fig.text(0.5, 0.005, c["description"], ha="center", va="bottom", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main():
    plots_only = "--plots" in sys.argv
    os.makedirs(PLOT_DIR, exist_ok=True)
    cases = C.build_cases()
    out = dict(
        generated=datetime.datetime.now().isoformat(timespec="seconds"),
        pandora_commit=_git_commit(C._repo),
        numpy=np.__version__,
        numba=__import__("numba").__version__,
        python=sys.version.split()[0],
        cases={},
    )
    for c in cases:
        r = C.run_case(c)
        inv = C.invariants(c, r)
        if inv:
            raise SystemExit(f"{c['name']}: invariants violated, refusing to write reference:\n  "
                             + "\n  ".join(inv))
        out["cases"][c["name"]] = dict(description=c["description"],
                                       params=case_params(c),
                                       fingerprint=C.fingerprint(c, r))
        plot_case(c, r, os.path.join(PLOT_DIR, c["name"] + ".pdf"))
        print(f"{c['name']:48s} n={len(r['time']):5d}  eclipse points="
              f"{C.mutual_eclipse_points(c, r):4d}  plot written")
    if not plots_only:
        with open(REF_FILE, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"reference written: {REF_FILE} (Pandora {out['pandora_commit']})")


if __name__ == "__main__":
    main()
