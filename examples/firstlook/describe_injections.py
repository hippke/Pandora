"""Markdown description of the injected light curves and of the first-look cost (for README.md / publication).

    python describe_injections.py cal1.jsonl,cal2.jsonl study1.jsonl > injections_table.md
Reads the reference rows of the calibration files (one per light curve) and regenerates the system parameters
with injections.draw (no sampling). The second file (recall_study.py output) gives evaluations and CPU per run.
"""
import json
import sys

import numpy as np

import injections as I

R_SUN, R_E, R_J, M_E = 696342e3, 6371e3, 69911e3, 5.972e24


def q(a, fmt="{:.3g}"):
    return " / ".join(fmt.format(x) for x in np.percentile(a, [0, 50, 100]))


def main(cal, study=None):
    rows = [r for f in cal.split(",") for r in map(json.loads, open(f)) if r["config"] == "ref"]
    regs = [g for g in ("kepler", "plato", "kepler-faint", "plato-faint") if any(r["regime"] == g for r in rows)]
    tab = {}
    for g in regs:
        rr = [r for r in rows if r["regime"] == g]
        S = [I.draw(r["index"], g, moon=r["moon"]) for r in rr]
        cad = S[0]["cadences_per_day"]
        P = np.array([s["per_bary"] for s in S])
        ep = np.array([s["epochs"] for s in S])
        rp = np.array([s["r_planet"] for s in S])
        b = np.array([s["b_bary"] for s in S])
        a = np.array([s["a_bary"] for s in S])
        noise = np.array([s["noise_ppm"] for s in S])
        dur = P / np.pi * np.sqrt((1 + rp) ** 2 - b ** 2) / a
        m = [s for s in S if s["moon"]]
        rm = np.array([s["r_moon"] for s in m])
        qq = np.array([s["M_moon"] / s["M_planet"] for s in m])
        Mm = np.array([s["M_moon"] for s in m])
        ref = np.array([r["dlnz"] for r in rr if r["moon"]])
        tab[g] = [
            ("light curves (with moon)", f"{len(rr)} ({len(m)})"),
            ("cadence", f"{1440 / cad:.0f} min"),
            ("planet period [d]", q(P)),
            ("planet transits", q(ep, "{:.0f}")),
            ("data window per transit [d]", q([s["epoch_duration"] for s in S])),
            ("total cadences", q([r["ndata"] for r in rr], "{:.0f}")),
            ("white noise per cadence [ppm]", q(noise)),
            ("planet transit duration [h]", q(24 * dur)),
            ("planet radius [R_Jup]", q(rp * R_SUN / R_J)),
            ("planet depth [ppm]", q(1e6 * rp ** 2)),
            ("planet S/N", q(rp ** 2 * np.sqrt(ep * dur * cad) / (noise * 1e-6))),
            ("moon radius [R_Earth]", q(rm * R_SUN / R_E)),
            ("moon depth [ppm]", q(1e6 * rm ** 2)),
            ("moon mass [M_Earth]", q(Mm / M_E)),
            ("M_moon / M_planet", q(qq)),
            ("moon period [d]", q([s["per_moon"] for s in m])),
            ("moon photometric S/N", q([r["snr"] for r in rr if r["moon"]])),
            ("moons: ref dlnZ >= 8 / 2-8 / < 2", f"{np.sum(ref >= 8)} / {np.sum((ref >= 2) & (ref < 8))} / {np.sum(ref < 2)}"),
        ]
    print("| quantity (min / median / max) | " + " | ".join(regs) + " |")
    print("|---|" + "---|" * len(regs))
    for i in range(len(tab[regs[0]])):
        print(f"| {tab[regs[0]][i][0]} | " + " | ".join(tab[g][i][1] for g in regs) + " |")
    if study:
        s = [json.loads(line) for line in open(study)]
        print("\n| run (one nss run) | evaluations (median) | CPU-s (median; 5-95 %) | us / evaluation |")
        print("|---|---|---|---|")
        for lab, src, key in (("planet only, 100 live, nsteps 16", "planet", "p100"),
                              ("planet only, 200 live, nsteps 28", "planet", "p200"),
                              ("planet+moon, 50 live, nsteps 16", "moon", "m50"),
                              ("planet+moon, 100 live, nsteps 16", "moon", "m100"),
                              ("planet+moon, 200 live, nsteps 32", "moon", "m200")):
            c = np.array([r[src][key]["cpu"] for r in s])
            n = np.array([r[src][key]["ncall"] for r in s])
            print(f"| {lab} | {np.median(n) / 1e3:.0f} k | {np.median(c):.2f} ({np.percentile(c, 5):.2f}-"
                  f"{np.percentile(c, 95):.2f}) | {1e6 * np.median(c / n):.1f} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
