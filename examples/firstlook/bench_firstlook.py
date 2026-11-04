"""Calibration and speed benchmark of the first look (white-noise injections, injections.py).

    # calibration: reference (long) runs + first-look configurations on the same light curves
    python bench_firstlook.py --n-moon 100 --n-null 50 --configs ref:1000:28:1000:64,A:200:28:200:32 --out cal.jsonl
    # summary (threshold for <= 2 % false passes on the no-moon set, recall per class, cost)
    python bench_firstlook.py --summary cal.jsonl
    # speed benchmark of the default first look on the fixed benchmark set (labels from cal.jsonl)
    python bench_firstlook.py --bench --labels cal.jsonl

Each task = one light curve x one configuration, one thread; cost = CPU seconds of that thread.
Configurations: name:nlive_planet:nsteps_planet:nlive_moon:nsteps_moon.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def summarize(path, ref="ref", fpr=0.02, classes=((8, math.inf), (5, 8), (2, 5), (-math.inf, 2))):
    rows = [json.loads(line) for line in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[(r["regime"], r["index"], r["moon"])] = r
    R = by.get(ref, {})
    out = {}
    print(f"{len(rows)} rows; configs {sorted(by)}; reference '{ref}': {len(R)} light curves")
    for cfg, d in sorted(by.items()):
        for regime in sorted({k[0] for k in d}) + ["all"]:
            keys = [k for k in d if regime == "all" or k[0] == regime]
            null = np.array([d[k]["dlnz"] for k in keys if not k[2]])
            if len(null) == 0:
                continue
            thr = float(np.quantile(null, 1 - fpr)) if len(null) >= 10 else float(np.max(null))
            s = f"{cfg:5s} {regime:6s}: thr({100 * (1 - fpr):.0f} % of nulls) {thr:6.2f} (null max {null.max():.2f}); "
            for lo, hi in classes:
                ks = [k for k in keys if k[2] and k in R and lo <= R[k]["dlnz"] < hi]
                if ks:
                    rec = np.mean([d[k]["dlnz"] > thr for k in ks])
                    s += f"ref dlnZ [{lo:g},{hi:g}): {rec * 100:5.1f} % of {len(ks):3d}; "
            cpu = np.array([d[k]["cpu"] for k in keys])
            nc = np.array([d[k]["ncall"] for k in keys])
            s += f"cpu {cpu.mean():.1f} s/LC (median {np.median(cpu):.1f}), {nc.mean() / 1e6:.1f} M calls/LC"
            print(s)
            out[(cfg, regime)] = dict(threshold=thr, cpu_mean=float(cpu.mean()), ncall_mean=float(nc.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-moon", type=int, default=60)
    ap.add_argument("--n-null", type=int, default=30)
    ap.add_argument("--regimes", default="kepler,plato")
    ap.add_argument("--configs", default="ref:1000:28:1000:64,A:200:28:200:32")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "cal.jsonl"))
    ap.add_argument("--summary", default="")
    ap.add_argument("--bench", action="store_true", help="speed benchmark: default first look on the labelled set")
    ap.add_argument("--labels", default="", help="jsonl with the reference rows (for --bench recall)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--preset", type=float, default=0.95, help="--bench: recall preset of firstlook.run")
    args = ap.parse_args()
    if args.summary:
        summarize(args.summary)
        return
    import firstlook
    import injections
    if args.bench:
        args.configs = "bench:0:0:0:0"            # firstlook.run(recall=args.preset)
    cfgs = []
    for c in args.configs.split(","):
        n, a, b, cc, d = c.split(":")
        cfgs.append((n, int(a), int(b), int(cc), int(d)))
    lcs = []
    for regime in args.regimes.split(","):
        for i in range(args.start, args.start + args.n_moon):
            lcs.append(injections.draw(i, regime, moon=True))
        for i in range(args.start, args.start + args.n_null):
            lcs.append(injections.draw(i, regime, moon=False))
    print(f"{len(lcs)} light curves x {len(cfgs)} configurations on {args.threads} threads", flush=True)
    firstlook.warmup()
    lock = threading.Lock()
    fout = open(args.out, "a")
    t_start = time.perf_counter()

    def task(s, cfg):
        name, a, b, c, d = cfg
        t, f, e = injections.simulate(s)
        fl = firstlook.FirstLook(t, f, e, t0_bary=s["t0_bary"], per_bary=s["priors"]["per_bary"],
                                 a_bary=s["priors"]["a_bary"], epoch_distance=s["epoch_distance"])
        if name == "bench":
            r = fl.run(recall=args.preset, seed=1 + s["index"])
        else:
            r = fl.run(recall=None, nlive_planet=a, nsteps_planet=b, nlive_moon=c, nsteps_moon=d, seed=1 + s["index"])
        row = dict(config=name, regime=s["regime"], index=s["index"], moon=s["moon"], dlnz=r["dlnz"],
                   dlnz_err=r["dlnz_err"], dmaxL=r["dmaxL"], logz_planet=r["logz_planet"], logz_moon=r["logz_moon"],
                   cpu=r["cpu"], wall=r["wall"], ncall=r["ncall"], ndata=r["ndata"], moon_summary=r["moon"],
                   snr=injections.moon_snr(s, t, e), truth={k: s[k] for k in ("per_bary", "epochs", "noise_ppm",
                   "r_moon", "per_moon", "M_moon", "M_planet", "i_moon", "Omega_moon", "cadences_per_day")},
                   settings=[a, b, c, d], date=time.strftime("%Y-%m-%d %H:%M"))
        with lock:
            fout.write(json.dumps(row) + "\n")
            fout.flush()
        return row

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(task, s, cfg) for cfg in cfgs for s in lcs]
        rows = [f.result() for f in futs]
    wall = time.perf_counter() - t_start
    fout.close()
    for name, *_ in cfgs:
        rr = [r for r in rows if r["config"] == name]
        cpu = sum(r["cpu"] for r in rr)
        print(f"config {name}: {len(rr)} light curves, {cpu / len(rr):.1f} cpu-s per light curve, "
              f"{np.mean([r['ncall'] for r in rr]) / 1e6:.1f} M calls per light curve", flush=True)
    print(f"total wall {wall:.0f} s on {args.threads} threads", flush=True)
    if args.bench:
        rr = [r for r in rows if r["config"] == "bench"]
        cpu = np.array([r["cpu"] for r in rr])
        res = dict(bench="firstlook", date=time.strftime("%Y-%m-%d %H:%M"), n=len(rr), preset=args.preset,
                   settings=firstlook.RECALL_PRESETS[args.preset],
                   cpu_per_lc=float(cpu.mean()), cpu_per_lc_median=float(np.median(cpu)),
                   lc_per_hour_per_core=float(3600 / cpu.mean()), wall=wall, threads=args.threads,
                   ncall_per_lc=float(np.mean([r["ncall"] for r in rr])))
        if args.labels:
            ref = {(r["regime"], r["index"], r["moon"]): r["dlnz"] for r in map(json.loads, open(args.labels))
                   if r["config"] == "ref"}
            thr = firstlook.RECALL_PRESETS[args.preset]["threshold"]
            null = [r["dlnz"] > thr for r in rr if not r["moon"]]
            good = [r for r in rr if r["moon"] and ref.get((r["regime"], r["index"], True), -1) >= 8]
            res.update(threshold=thr, recall_ref8=float(np.mean([r["dlnz"] > thr for r in good])) if good else None,
                       n_ref8=len(good), false_passes=int(sum(null)), n_null=len(null))
        print("BENCH", json.dumps(res), flush=True)
        open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "bench_log.jsonl"), "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
