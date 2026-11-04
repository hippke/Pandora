"""Recall vs cost of the first look (README.md "Choosing the recall").

Misses of well-detectable moons come from moon runs that never find the moon's mode; their probability
falls with more live points and with repeated independent moon runs (both miss: ~p^2). This script
reruns labelled light curves (reference rows of bench_firstlook.py) with several seeds and moon-run
sizes; `--analyse` then computes, for every (size, repeats) combination, the recall of moons with
reference dlnZ >= 8 (mean over light curves of 1 - prod over repeats of the per-run miss), the false
passes on no-moon light curves and the CPU cost.

    python recall_study.py --labels cal1.jsonl,cal2.jsonl --seeds-pos 10 --seeds-null 4 --out study.jsonl
    python recall_study.py --analyse study.jsonl
"""
import argparse
import itertools
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

PLANET = {"p200": (200, 28), "p100": (100, 16)}
MOON = {"m50": (50, 16), "m100": (100, 16), "m100s32": (100, 32), "m200": (200, 32)}
# Round 12 (cheaper moon runs): --sizes small
PLANET_SMALL = {"p100": (100, 16), "p50": (50, 12)}
MOON_SMALL = {"m50": (50, 16), "m50s8": (50, 8), "m50s12": (50, 12), "m30s16": (30, 16), "m30s8": (30, 8),
              "m70s8": (70, 8)}


def analyse(path, thresholds=(2.0, 3.0), max_repeats=8, nboot=2000):
    rows = [json.loads(line) for line in open(path)]
    lcs = {}
    for r in rows:
        lcs.setdefault((r["regime"], r["index"], r["has_moon"]), []).append(r)
    pos = {k: v for k, v in lcs.items() if k[2] and v[0]["ref"] >= 8}
    null = {k: v for k, v in lcs.items() if not k[2]}
    print(f"{len(rows)} rows: {len(pos)} light curves with ref dlnZ >= 8 ({sum(len(v) for v in pos.values())} seeds), "
          f"{len(null)} no-moon ({sum(len(v) for v in null.values())} seeds)")
    rng = np.random.default_rng(0)
    res = []
    for pk, mk, m in itertools.product(PLANET, MOON, range(1, max_repeats + 1)):
        cpu = np.mean([r["planet"][pk]["cpu"] + m * r["moon"][mk]["cpu"] for v in lcs.values() for r in v])
        ncall = np.mean([r["planet"][pk]["ncall"] + m * r["moon"][mk]["ncall"] for v in lcs.values() for r in v])
        for thr in thresholds:
            # per light curve: miss probability of one moon run = fraction of seeds with dlnz <= thr (planet run of
            # the same seed); m repeats: log-mean-Z of m runs -> approximated by: miss iff all m runs miss
            # (a run that finds the mode exceeds the threshold by far: dlnZ >= 8 - ln m)
            p_miss = np.array([np.mean([r["moon"][mk]["logz"] - r["planet"][pk]["logz"] <= thr for r in v])
                               for v in pos.values()])
            rec = 1.0 - np.mean(p_miss ** m)
            # bootstrap over light curves for the uncertainty of the recall
            bs = [1.0 - np.mean(p_miss[rng.integers(0, len(p_miss), len(p_miss))] ** m) for _ in range(nboot)]
            # false passes: log-mean-Z of m moon runs from different seeds of the same light curve
            fp = []
            for v in null.values():
                for i in range(len(v)):
                    js = [(i + j) % len(v) for j in range(m)]
                    lz = np.logaddexp.reduce([v[j]["moon"][mk]["logz"] for j in js]) - math.log(m)
                    fp.append(lz - v[i]["planet"][pk]["logz"] > thr)
            res.append(dict(planet=pk, moon=mk, repeats=m, threshold=thr, recall=rec,
                            recall_lo=float(np.quantile(bs, 0.05)), fp=float(np.mean(fp)), n_fp=int(np.sum(fp)),
                            n_null=len(fp), cpu=float(cpu), ncall=float(ncall),
                            n_lc_any_miss=int(np.sum(p_miss > 0))))
    res.sort(key=lambda d: d["cpu"])
    print("planet moon     rep thr  recall   (5 % bootstrap)  miss-LC  false passes      cpu-s/LC  calls/LC")
    for d in res:
        print(f"{d['planet']:6s} {d['moon']:8s} {d['repeats']:3d} {d['threshold']:3.0f}  {100 * d['recall']:7.3f} %"
              f" ({100 * d['recall_lo']:7.3f} %)  {d['n_lc_any_miss']:5d}   {d['n_fp']:3d}/{d['n_null']} "
              f"({100 * d['fp']:.1f} %)  {d['cpu']:6.2f}  {d['ncall'] / 1e6:5.2f} M")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="")
    ap.add_argument("--seeds-pos", type=int, default=10)
    ap.add_argument("--seeds-null", type=int, default=4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="study.jsonl")
    ap.add_argument("--analyse", default="")
    ap.add_argument("--validate", default="", help="comma list of recall presets: end-to-end first looks, fresh seeds")
    ap.add_argument("--seed0", type=int, default=50_000)
    ap.add_argument("--sizes", default="default", choices=["default", "small"])
    ap.add_argument("--vseeds", type=int, default=0, help="--validate: seeds per moon light curve (0: preset default)")
    args = ap.parse_args()
    global PLANET, MOON
    if args.sizes == "small":
        PLANET, MOON = PLANET_SMALL, MOON_SMALL
    if args.analyse:
        analyse(args.analyse)
        return
    import firstlook
    import injections
    ref = {}
    for path in args.labels.split(","):
        for r in map(json.loads, open(path)):
            if r["config"] == "ref":
                ref[(r["regime"], r["index"], r["moon"])] = r["dlnz"]
    if args.validate:
        presets = [float(x) for x in args.validate.split(",")]
        jobs = []
        for p in presets:
            nsp = args.vseeds or {0.9: 3, 0.95: 5, 0.99: 8, 0.999: 13}.get(p, 5)
            for k, d in ref.items():
                if (k[2] and d >= 8) or not k[2]:
                    jobs += [(p, k, s) for s in range(nsp if k[2] else 3)]
        print(f"validate {presets}: {len(jobs)} first looks on {args.threads} threads", flush=True)
        firstlook.warmup()
        lock = threading.Lock()
        fout = open(args.out, "a")

        def vjob(p, k, s):
            sysd = injections.draw(k[1], k[0], moon=k[2])
            t, f, e = injections.simulate(sysd)
            fl = firstlook.FirstLook(t, f, e, t0_bary=sysd["t0_bary"], per_bary=sysd["priors"]["per_bary"],
                                     a_bary=sysd["priors"]["a_bary"], epoch_distance=sysd["epoch_distance"])
            r = fl.run(recall=p, seed=args.seed0 + 1000 * s + k[1])
            row = dict(preset=p, regime=k[0], index=k[1], has_moon=k[2], ref=ref[k], seed=s, dlnz=r["dlnz"],
                       interesting=r["interesting"], cpu=r["cpu"], ncall=r["ncall"])
            with lock:
                fout.write(json.dumps(row) + "\n")
                fout.flush()
            return row

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            rows = [f.result() for f in [ex.submit(vjob, *j) for j in jobs]]
        for p in presets:
            rr = [r for r in rows if r["preset"] == p]
            pos = [r["interesting"] for r in rr if r["has_moon"]]
            nul = [r["interesting"] for r in rr if not r["has_moon"]]
            nm = len(pos) - sum(pos)
            ub = (3.0 if nm == 0 else nm + 2.0 * math.sqrt(nm) + 1.0) / len(pos)
            print(f"preset {p}: recall {100 * np.mean(pos):.3f} % ({nm} misses of {len(pos)}; miss rate <~ {100 * ub:.2f} % "
                  f"at 95 %), false passes {sum(nul)}/{len(nul)}, cpu {np.mean([r['cpu'] for r in rr]):.2f} s/LC, "
                  f"{np.mean([r['ncall'] for r in rr]) / 1e6:.2f} M calls/LC", flush=True)
        print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)
        return
    jobs = []
    for k, d in ref.items():
        if k[2] and d < 2:
            continue                                  # moons the reference does not detect: not needed here
        ns = args.seeds_pos if k[2] else args.seeds_null
        jobs += [(k, s) for s in range(ns)]
    print(f"{len(jobs)} jobs ({len(ref)} labelled light curves) on {args.threads} threads", flush=True)
    firstlook.warmup()
    lock = threading.Lock()
    fout = open(args.out, "a")

    def job(k, s):
        sysd = injections.draw(k[1], k[0], moon=k[2])
        t, f, e = injections.simulate(sysd)
        fl = firstlook.FirstLook(t, f, e, t0_bary=sysd["t0_bary"], per_bary=sysd["priors"]["per_bary"],
                                 a_bary=sysd["priors"]["a_bary"], epoch_distance=sysd["epoch_distance"])
        seed = 10_000 + 1000 * s + k[1]
        row = dict(regime=k[0], index=k[1], has_moon=k[2], ref=ref[k], seed=s, planet={}, moon={})
        for name, (nl, nst) in PLANET.items():
            r = fl.run_planet(nl, nst, seed=seed)
            row["planet"][name] = dict(logz=r["logz"], cpu=r["cpu"], ncall=r["ncall"])
        for name, (nl, nst) in MOON.items():
            r = fl.run_moon(nl, nst, seed=seed + 7)
            row["moon"][name] = dict(logz=r["logz"], cpu=r["cpu"], ncall=r["ncall"], maxL=r["maxL"])
        with lock:
            fout.write(json.dumps(row) + "\n")
            fout.flush()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for f in [ex.submit(job, k, s) for k, s in jobs]:
            f.result()
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
