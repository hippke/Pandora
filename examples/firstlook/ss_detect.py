"""Is supersampling required for DETECTION (first look) in Kepler long-cadence data? (SAMPLING.md backlog 7)

Injections (injections.py, regimes kepler / kepler-faint: 30-min cadence, 3-9 transits) with real exposure
smearing (truth averaged over 30 sub-exposures per cadence), with and without a moon. Each light curve gets a
first look at several fit supersampling factors (same data and noise), and the moon light curves a reference
fit (long runs, factor 7) that labels the detectable moons (reference dlnZ >= 8).

    python ss_detect.py --n-moon 60 --n-null 60 --factors 1,3,5 --recall 0.95 --threads 20 --out ss_detect.jsonl
    python ss_detect.py --summary ss_detect.jsonl

Rows (jsonl): regime, index, moon, factor, kind ("fl" = first look with the recall preset, "ref" = reference),
dlnz, interesting (dlnz > threshold), cpu, ncall, planet_snr, moon_snr, settings.
The summary gives per factor: false-positive rate of no-moon light curves (dlnz > threshold) overall and vs
planet S/N, the dlnZ shift relative to factor 5 / 7 (paired), the recall of detectable moons, CPU-s per
light curve.
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REF_SETTINGS = dict(recall=None, nlive_planet=1000, nsteps_planet=28, nlive_moon=1000, nsteps_moon=64, repeats=1,
                    threshold=8.0)
REF_FACTOR = 7


def task(job):
    import firstlook as F
    import injections as I
    regime, idx, moon, factor, kind, recall, true_ssf = job
    s = I.draw(idx, regime, moon=moon)
    t, f, e = I.simulate(s, true_ssf=true_ssf, fit_ssf=factor)
    fl = F.FirstLook(t, f, e, t0_bary=s["t0_bary"], per_bary=s["priors"]["per_bary"], a_bary=s["priors"]["a_bary"],
                     epoch_distance=s["epoch_distance"], supersampling_factor=factor)
    seed = 1 + 1000 * idx + 17 * int(moon)
    if kind == "ref":
        r = fl.run(seed=seed, **REF_SETTINGS)
    else:
        r = fl.run(recall=recall, seed=seed)
    t_, _, e_ = I.simulate(s)                        # instantaneous grid (for the moon S/N proxy only)
    return dict(regime=regime, index=idx, moon=moon, factor=factor, kind=kind, recall=recall, true_ssf=true_ssf,
                dlnz=r["dlnz"], dlnz_err=r["dlnz_err"], interesting=r["interesting"], cpu=r["cpu"],
                cpu_planet=r["cpu_planet"], cpu_moon=r["cpu_moon"], ncall=r["ncall"], planet_snr=I.planet_snr(s),
                moon_snr=I.moon_snr(s, t_, e_), epochs=s["epochs"], noise_ppm=s["noise_ppm"], settings=r["settings"],
                date=time.strftime("%Y-%m-%d %H:%M"))


def summary(path, threshold=3.0):
    rows = [json.loads(line) for line in open(path)]
    fl = [r for r in rows if r["kind"] == "fl"]
    ref = {(r["regime"], r["index"]): r for r in rows if r["kind"] == "ref"}
    factors = sorted({r["factor"] for r in fl})
    key = lambda r: (r["regime"], r["index"], r["moon"])        # noqa: E731
    by = {k: {} for k in {key(r) for r in fl}}
    for r in fl:
        by[key(r)][r["factor"]] = r
    print(f"{len(fl)} first-look rows, {len(ref)} reference fits; factors {factors}; threshold {threshold}")
    fmax = max(factors)
    for regime in sorted({k[0] for k in by}) + ["all"]:
        ks = [k for k in by if regime == "all" or k[0] == regime]
        print(f"\n== {regime}: {sum(not k[2] for k in ks)} no-moon, {sum(k[2] for k in ks)} moon light curves")
        for fac in factors:
            null = [by[k][fac] for k in ks if not k[2] and fac in by[k]]
            moon = [by[k][fac] for k in ks if k[2] and fac in by[k]]
            det = [r for r in moon if (r["regime"], r["index"]) in ref and ref[(r["regime"], r["index"])]["dlnz"] >= 8]
            fp = np.mean([r["dlnz"] > threshold for r in null]) if null else np.nan
            rec = np.mean([r["dlnz"] > threshold for r in det]) if det else np.nan
            sh = [by[k][fac]["dlnz"] - by[k][fmax]["dlnz"] for k in ks if fac in by[k] and fmax in by[k]]
            shn = [by[k][fac]["dlnz"] - by[k][fmax]["dlnz"] for k in ks if not k[2] and fac in by[k] and fmax in by[k]]
            cpu = np.mean([by[k][fac]["cpu"] for k in ks if fac in by[k]])
            print(f"  factor {fac:2d}: false positives {np.sum([r['dlnz'] > threshold for r in null])}/{len(null)} "
                  f"({100 * fp:.1f} %); recall of ref-detectable moons {np.sum([r['dlnz'] > threshold for r in det])}"
                  f"/{len(det)} ({100 * rec:.1f} %); dlnZ - dlnZ(factor {fmax}): all mean {np.mean(sh):+.2f}, "
                  f"no-moon mean {np.mean(shn) if shn else np.nan:+.2f} (max {np.max(shn) if shn else np.nan:+.2f}); "
                  f"cpu {cpu:.2f} s/LC")
        # no-moon dlnZ shift vs planet S/N (the expected driver of the inflation)
        null_k = [k for k in ks if not k[2] and all(f in by[k] for f in (min(factors), fmax))]
        if null_k:
            snr = np.array([by[k][fmax]["planet_snr"] for k in null_k])
            sh = np.array([by[k][min(factors)]["dlnz"] - by[k][fmax]["dlnz"] for k in null_k])
            edges = np.quantile(snr, [0, 0.25, 0.5, 0.75, 1.0])
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (snr >= lo) & (snr <= hi)
                print(f"    no-moon, planet S/N {lo:6.0f}-{hi:6.0f}: dlnZ(factor {min(factors)}) - dlnZ(factor {fmax}) "
                      f"mean {sh[m].mean():+.2f}, max {sh[m].max():+.2f} (n {m.sum()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-moon", type=int, default=60, help="moon light curves per regime")
    ap.add_argument("--n-null", type=int, default=60, help="no-moon light curves per regime")
    ap.add_argument("--start", type=int, default=0, help="first injection index")
    ap.add_argument("--regimes", default="kepler,kepler-faint")
    ap.add_argument("--factors", default="1,3,5")
    ap.add_argument("--recall", type=float, default=0.95)
    ap.add_argument("--true-ssf", type=int, default=30)
    ap.add_argument("--no-ref", action="store_true", help="skip the reference fits of the moon light curves")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="ss_detect.jsonl")
    ap.add_argument("--summary", default="")
    ap.add_argument("--threshold", type=float, default=3.0)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary, a.threshold)
        return
    import firstlook as F
    F.warmup()
    jobs = []
    for regime in a.regimes.split(","):
        for moon, n in ((True, a.n_moon), (False, a.n_null)):
            for i in range(a.start, a.start + n):
                if moon and not a.no_ref:
                    jobs.append((regime, i, True, REF_FACTOR, "ref", None, a.true_ssf))
                for fac in map(int, a.factors.split(",")):
                    jobs.append((regime, i, moon, fac, "fl", a.recall, a.true_ssf))
    done = set()
    if os.path.exists(a.out):                                    # resume: skip finished jobs
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["regime"], r["index"], r["moon"], r["factor"], r["kind"]))
    jobs = [j for j in jobs if j[:5] not in done]
    jobs.sort(key=lambda j: j[4] != "ref")                       # the expensive reference fits first
    print(f"{len(jobs)} jobs on {a.threads} threads", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.threads) as ex, open(a.out, "a") as fo:
        for k, r in enumerate(ex.map(task, jobs)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(jobs)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"done: {len(jobs)} jobs in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
