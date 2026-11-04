"""Sensitivity check: does the test suite catch real regressions?

Runs every case against (a) older Pandora commits that still contain bugs
listed in Pandora/BUGS.md and (b) synthetic code mutations of the current
commit, and reports per case whether the change is detected by
  - SUM   : comparing only sum(flux) per component (= the final value of the
            cumulative sum) with the same relative tolerance,
  - SUITE : the full light-curve test (invariants + fingerprint),
  - UNIT  : number of failing unit tests (unit_tests.py).

    python check_sensitivity.py              # synthetic mutations of the current checkout (default)
    python check_sensitivity.py --commits    # also the old commits below
    python check_sensitivity.py --mutations  # mutations only

The old commits predate the 2.0 call signatures (numerical_grid), so on them every case fails with a call
error and is counted as detected: --commits no longer says anything about the suite. The mutations do.

Needs git and a Pandora checkout (default: the parent folder of this directory; PANDORA_PATH to override). Every variant is
checked out into a temporary git worktree, so the first run of each includes
numba compilation (~10 s per variant; ~2-3 min in total).
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.environ.get("PANDORA_PATH", os.path.join(HERE, "..")))   # a git checkout of Pandora

# old commits -> what they still contain relative to the reference version
COMMITS = {
    "baa7788": "original upstream code (bugs #1-#27)",
    "0b3166d": "bug #2 (eccentric solver) and #19-#27 open",
    "c91995a": "bugs #20-#27 open",
    "a45cf72": "bugs #21-#27 open",
    "d937537": "bugs #22-#27 open",
    "a655f9b": "bugs #23-#27 open",
    "92bc63b": "bugs #24-#27 open",
    "e1568ef": "bugs #25-#27 open",
    "01b61c6": "bugs #26-#27 open (timegrid spacing, count)",
    "33e9850": "bug #27 open (timegrid cadence count round-off)",
    "fa77161": "pre-analytic eclipse (pixelart default on the limb)",
}

# synthetic mutations of the current commit: name -> (file, old, new, note[, replace_all])
# old must occur exactly once, or (replace_all=True) at least once: then every occurrence is changed
MUTATIONS = {
    "time_shift_2s": ("pandoramoon/grids.py", "t0_shift_planet = t0_bary_offset / (tdur / 2)\n",
                      "t0_shift_planet = t0_bary_offset / (tdur / 2) + 1e-4\n",
                      "barycentre shifted by 1e-4 R_star (~2 s in time; all x_bary paths)", True),
    "mirror_ecc_moon": ("pandoramoon/ellipse.py", "    w_rad = w / 180 * pi\n",
                        "    w_rad = w / 180 * pi + pi\n", "re-introduce bug #8 (w_moon + 180 deg)"),
    "moon_phase_180": ("pandoramoon/ellipse.py", "k = pi * (time - tau * per) / per",
                       "k = pi * (time - tau * per) / per + pi / 2",
                       "circular moon on the opposite side of the planet"),
    "no_mutual_eclipse": ("pandoramoon/eclipse.py", "if er > 0:", "if er > 2:",
                          "planet-moon eclipses switched off"),
    "ld_norm_1e-6": ("pandoramoon/occult.py", "s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)\n    s_inv",
                     "s = 1.000001 * 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)\n    s_inv",
                     "occult_small normalisation changed by 1 ppm"),
    "analytic_er_1e-4": ("pandoramoon/overlap_analytic.py",
                         "return min(1.0, a_smp / a_sm)",
                         "return min(1.0, 1.0001 * a_smp / a_sm)",
                         "analytic limb overlap fraction scaled by 1e-4"),
    "analytic_disabled": ("pandoramoon/eclipse.py",
                          "er = er_fast(xp[idx], yp[idx], r_planet, xm[idx], ym[idx], r_moon)",
                          "er = eclipse_ratio(distance_p_m, r_planet, r_moon)",
                          "limb eclipses without the stellar limb (analytic three-circle overlap disabled)"),
}

WORKER = r"""
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])          # test_cases dir (cases.py)
import cases as C
out = {}
for c in C.build_cases():
    try:
        r = C.run_case(c)
        out[c["name"]] = dict(
            ok=True,
            sums={k: float(np.sum(r["flux_" + k])) for k in ("planet", "moon", "total")},
            n=int(len(r["time"])),
            invariants=C.invariants(c, r),
            fingerprint=C.fingerprint(c, r))
    except Exception as e:
        out[c["name"]] = dict(ok=False, error=f"{type(e).__name__}: {e}")
import os
os.environ["TQDM_DISABLE"] = "1"
unit_fail = []
try:
    import unit_tests
    for fn in unit_tests.TESTS:
        try:
            fn()
        except Exception:
            unit_fail.append(fn.__name__)
except Exception as e:
    unit_fail.append(f"import: {e}")
out["__unit__"] = unit_fail
print("JSON" + json.dumps(out))
"""


def run_version(path):
    env = dict(os.environ, PANDORA_PATH=path, MPLBACKEND="Agg")
    p = subprocess.run([sys.executable, "-c", WORKER, HERE], capture_output=True,
                       text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(line[0][4:])


def run_variant(commit, mutation=None):
    with tempfile.TemporaryDirectory() as tmp:
        wt = os.path.join(tmp, "wt")
        subprocess.run(["git", "-C", REPO, "worktree", "add", "-q", "--detach", wt, commit],
                       check=True, capture_output=True)
        try:
            if mutation:
                fname, old, new = mutation[:3]
                replace_all = len(mutation) > 4 and mutation[4]
                path = os.path.join(wt, fname)
                src = open(path).read()
                if not (src.count(old) >= 1 if replace_all else src.count(old) == 1):
                    raise RuntimeError(f"mutation target not unique/found in {fname}: {old!r}")
                open(path, "w").write(src.replace(old, new))
            return run_version(wt)
        finally:
            subprocess.run(["git", "-C", REPO, "worktree", "remove", "--force", wt],
                           capture_output=True)


def evaluate(C, ref, res):
    rows = []
    for name in ref:
        if name == "__unit__":
            continue
        a, b = ref[name], res[name]
        if not b["ok"]:
            rows.append((name, True, True, b["error"]))
            continue
        if a["n"] != b["n"]:
            sum_det = True
        else:
            sum_det = any(abs(a["sums"][k] - b["sums"][k]) >
                          1e-7 * max(abs(a["fingerprint"][k]["deficit_sum"]), 1e-300)
                          for k in ("planet", "moon", "total"))
        errs = b["invariants"] + C.compare_fingerprints(a["fingerprint"], b["fingerprint"])
        rows.append((name, sum_det, bool(errs), errs[0] if errs else ""))
    return rows


def main():
    sys.path.insert(0, HERE)
    import cases as C
    do_commits = "--commits" in sys.argv
    do_mut = "--commits" not in sys.argv or "--mutations" in sys.argv
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    ref = run_version(REPO)                      # current checkout = reference
    variants = []
    if do_commits:
        variants += [(f"commit {k}", k, None, v) for k, v in COMMITS.items()]
    if do_mut:
        variants += [(f"mut {k}", head, m, m[3]) for k, m in MUTATIONS.items()]
    summary = []
    for label, commit, mutation, note in variants:
        res = run_variant(commit, mutation)
        rows = evaluate(C, ref, res)
        unit = res.get("__unit__", [])
        print(f"== {label}: {note}")
        print(f"   unit tests failing: {', '.join(unit) if unit else '-'}")
        for name, s, f, why in rows:
            print(f"   {name:48s} SUM={'yes' if s else '-':3s} SUITE={'yes' if f else '-':3s} {why[:60]}")
        summary.append((label, sum(r[1] for r in rows), sum(r[2] for r in rows), len(unit), note))
    n_lc = len(ref) - 1
    print(f"\nsummary: light-curve cases flagged out of {n_lc}; failing unit tests")
    print(f"   {'variant':28s} {'SUM':>4s} {'SUITE':>6s} {'UNIT':>5s}")
    for label, s, f, u, note in summary:
        print(f"   {label:28s} {s:4d} {f:6d} {u:5d}   {note}")


if __name__ == "__main__":
    main()
