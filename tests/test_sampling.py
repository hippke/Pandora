"""Tests of the public fit API pandoramoon.sampling (NestedSampler, priors) -- SAMPLING.md Round 15.

    python test_sampling.py            # all tests (~2-5 min on the laptop: one small planet-only fit)
    python test_sampling.py --quick    # without the fit

Tests:
  T1 the compiled likelihood of the API equals a direct Likelihood call at the mapped parameters (+ the
     coordinate weight), planet+moon and planet-only, for random prior points (independent path: the
     Likelihood object's own entry point);
  T2 prior transforms: the inverse CDF (ppf) reproduces the prior's CDF (numerical integral of its density),
     for Uniform, LogUniform, Normal, truncated Normal;
  T3 moon-phase coordinates: importance-weighted cube samples reproduce the user's prior on per_moon and
     tau_moon (flat / log-flat), and the mapping equals reparam.Reparam.to_native (the validated module);
  T4 limb darkening: (u1, u2) Normal priors with correlation, truncated to the physical triangle: the
     mapped samples have the truncated prior's mean / sd / correlation (direct rejection sampling) and the
     normalisation constant;
  T5 a small planet-only fit end to end: result keys (UltraNest-like), files in log_dir, resume (a second
     run with resume="resume" reuses the stored runs and gives the same logZ without new runs);
  T6 resume after a crash: a fit in a subprocess is killed after some runs are stored; resume="resume"
     reloads them, continues and passes;
  T8 (Round 17) priors reaching outside Pandora's valid range (a Normal on q1, on r_planet) are truncated and
     renormalised (no prior mass lost to -inf), with a warning; the unbiased thinning of weighted samples;
  T7 nss itself on a problem with a known evidence (8-d: a banana pair, a dimension cut at the prior edge,
     plain Gaussians; lnZ = ln 1/2 exactly): nss.fit recovers it within its error (+ 0.1 systematic), and a
     fit started with too short chains (2 slice steps) doubles them by its nsteps check.
  T9 (Round 20) the rejection init (default): a Gaussian with a LOGZERO plateau on 75 % of the cube (lnZ = 0):
     no plateau points in the initial live set, no stuck chains, ln X0 = ln 0.25, lnZ recovered; the old init
     (init_reject=False) still runs and shows the plateau.
  T10 (Round 20/21) the n_live escalation of nss.fit: a compact mode with 99.9 % of the mass in 10 % of the volume
     (8-d, lnZ = 0): runs that miss it are ~7 nats low with a lower maxL; fit doubles n_live (not nsteps), flags the
     result multimodal and recovers lnZ; the unimodal known-Z problem of T7 does not escalate.
  T11 (Round 21) posterior repartitioning: removed with the option (Round 23 decision b; in git).
  T13 (2026-10-14) planet-only evidence by importance sampling: agrees with the full nested fit (lnZ, posterior) with
     < 1/10 of the calls; a failed check falls back to the nested fit.
  T14 (2026-10-14) a/R* prior: default_priors' wide default; with a wrong narrow prior a false moon and the
     prior-edge warning, with the new default neither.
  T12 (Round 22) the search API: resample_likelihood = a directly supersampled Likelihood; NestedSampler
     supersampling="auto" (30-min data -> 5 sub-exposures) and None; default_priors gives valid models; detect()
     flags the injected moon (candidate, jackknife passed) and not a planet-only light curve; the jackknife of two
     identical models is 0; prepare_lightcurve recovers the white noise level and beta ~ 1 (if wotan is installed).
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np

import pandoramoon as pandora
from pandoramoon import sampling as S
from pandoramoon.sampling import LogUniform, NestedSampler, Normal, Uniform

R_SUN = 696_342_000.0
FAIL = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def make_data(seed=0):
    """the notebook's injection (Jupiter + Earth-sized moon, 3 transits, 30 min, 500 ppm)"""
    p = pandora.model_params()
    p.R_star, p.u1, p.u2 = R_SUN, 0.4089, 0.2556
    p.per_bary, p.a_bary, p.r_planet, p.b_bary = 365.25, 215.0, 0.1, 0.3
    p.t0_bary, p.t0_bary_offset, p.M_planet, p.w_bary, p.ecc_bary = 11.0, 0.01, 1.8986e27, 0.0, 0.0
    p.r_moon, p.per_moon, p.tau_moon, p.Omega_moon, p.i_moon = 0.03526, 5.0, 0.07, 0.0, 90.0
    p.M_moon, p.ecc_moon, p.w_moon = 0.05395 * 1.8986e27, 0.0, 0.0
    p.epochs, p.epoch_duration, p.cadences_per_day, p.epoch_distance = 3, 2.0, 48, 365.26
    p.supersampling_factor, p.occult_small_threshold, p.hill_sphere_threshold = 1, 0, 1.0
    t = pandora.time(p).grid()
    flux, _, _ = pandora.moon_model(p).light_curve(t)
    rng = np.random.default_rng(seed)
    data = flux + rng.normal(0, 500e-6, len(t))
    like = pandora.Likelihood(t, data, np.full(len(t), 500e-6), epoch_distance=365.26, hill_sphere_threshold=1.0)
    like.yerr_input = np.full(len(t), 500e-6)          # for T6's child process (exact copy of the data)
    return like


PRIORS = dict(q1=Uniform(0, 1), q2=Uniform(0, 1), R_star=Uniform(0.4 * R_SUN, 1.2 * R_SUN),
              per_bary=Uniform(360.25, 370.25), a_bary=Uniform(210, 220), r_planet=Uniform(0.07, 0.11),
              b_bary=Uniform(0, 1), t0_bary=11.0, t0_bary_offset=Uniform(-0.05, 0.05),
              M_planet=Uniform(1.5e27, 2.5e27), r_moon=Uniform(0.01, 0.07), per_moon=Uniform(0, 50),
              tau_moon=Uniform(0, 1), Omega_moon=Uniform(0, 180), i_moon=Uniform(0, 180), M_moon=Uniform(1, 1e27))


def t1(like):
    rng = np.random.default_rng(1)
    for model, priors in (("moon", PRIORS), ("planet", PRIORS),
                          ("moon", dict(PRIORS, ecc_moon=Uniform(0, 0.5), w_moon=Uniform(0, 360))),
                          ("planet", dict(PRIORS, ecc_bary=Uniform(0, 0.3), w_bary=Uniform(0, 360)))):
        m = S._Model(like, priors, model, "auto", 0.0)
        U = rng.uniform(size=(400, m.ndim))
        th, lw = m.native(U)
        dmax, nval = 0.0, 0
        for i in range(len(U)):
            v = S._loglike(U[i], m.D)
            if not lw[i] > S.LOGZERO:
                assert v == S.LOGZERO
                continue
            if model == "planet":
                ref = like(*th[i][:12], 0.0, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                ref = like(*th[i])
            if np.isfinite(ref):
                dmax = max(dmax, abs((v - lw[i]) - ref) / max(1.0, abs(ref)))
                nval += 1
        extra = " (eccentric)" if ("ecc_moon" in priors or "ecc_bary" in priors) else ""
        check(nval > 200 and dmax < 1e-9, f"T1 {model}{extra}: compiled logL - log w = Likelihood(theta) for {nval} "
              f"prior points, max rel. diff {dmax:.1e}")


def t2():
    rng = np.random.default_rng(2)
    u = rng.uniform(size=200_000)
    for p in (Uniform(-2, 3), LogUniform(0.01, 50), Normal(1.0, 0.5), Normal(1.0, 0.5, lo=0.8, hi=3.0),
              Normal(0.0, 1.0, lo=-np.inf, hi=-1.0)):
        x = S._ppf_np(p, u)
        lo, hi = (x.min(), x.max())
        grid = np.geomspace(lo, hi, 4001) if isinstance(p, LogUniform) else np.linspace(lo, hi, 4001)
        pdf = np.exp([S._logpdf(p.kind, *p.par(), g) for g in grid])
        cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
        norm = cdf[-1]
        ecdf = np.searchsorted(np.sort(x), grid, side="right") / len(x)
        D = np.max(np.abs(ecdf - cdf / norm))
        check(abs(norm - 1) < 2e-3 and D < 0.006, f"T2 {p!r}: density integrates to {norm:.4f}, "
              f"KS(ppf samples, CDF) {D:.4f}")


def t3(like):
    rng = np.random.default_rng(3)
    for pm_prior in (Uniform(0, 50), LogUniform(0.5, 40), Normal(10, 3, lo=0.0, hi=30)):
        m = S._Model(like, dict(PRIORS, per_moon=pm_prior), "moon", True, 0.0)
        U = rng.uniform(size=(400_000, m.ndim))
        th, lw = m.native(U)
        ok = lw > S.LOGZERO
        # weight relative to the user's prior: exp(lw) contains the moon-phase weight x prior density ratio and
        # nothing else here (q1, q2 uniform)
        w = np.exp(lw[ok])
        pm, tau = th[ok, 13], th[ok, 14]
        meanw = w.sum() / len(U)
        F = m.F_excluded
        # expected: E_cube[w] = 1 - F (the excluded short-period part)
        edges = np.linspace(max(m.p_lo, pm_prior.support()[0]), min(30.0, pm_prior.support()[1]), 21)
        h, _ = np.histogram(pm, edges, weights=w)
        h = h / len(U)
        a, b, c, d = pm_prior.par()
        cdfs = np.array([math.exp(S._logcdf_np(pm_prior, e)) if e > 0 else 0.0 for e in edges])
        exp_h = np.diff(cdfs)
        rel = np.max(np.abs(h - exp_h) / exp_h)
        ht, _ = np.histogram(tau, np.linspace(0, 1, 11), weights=w)
        tflat = np.max(np.abs(ht / ht.sum() * 10 - 1))
        check(abs(meanw - (1 - F)) < 0.01 and rel < 0.05 and tflat < 0.02,
              f"T3 per_moon ~ {pm_prior!r}: mean weight {meanw:.4f} (expected {1 - F:.4f}), weighted per_moon "
              f"histogram vs prior max rel. dev. {rel:.3f}, tau flat to {tflat:.3f}")
    # the mapping equals the validated reparam module
    from pandoramoon.reparam import Reparam
    m = S._Model(like, PRIORS, "moon", True, 0.0)
    D = m.D
    rp = Reparam(360.25, 370.25, p_max=50.0, p_lo=m.p_lo, e_c=m.e_c)
    U = rng.uniform(size=(2000, m.ndim))
    th, lw = m.native(U)
    c = list(D[19])        # mp_c (cube indices of phi_c, delta, u_m); see the D layout in sampling.py
    dmax = 0.0
    for i in range(len(U)):
        pm, tau, lwr = rp.to_native(th[i, 3], th[i, 9], th[i, 10], U[i, c[0]], U[i, c[1]], U[i, c[2]], m.e_c)
        if lw[i] > S.LOGZERO and lwr > S.LOGZERO:
            dmax = max(dmax, abs(pm - th[i, 13]), abs(tau - th[i, 14]),
                       abs((lw[i] - (math.log(1 / 50) + math.log(50))) - lwr))
    check(dmax < 1e-12, f"T3 moon-phase mapping = reparam.Reparam.to_native (e_c {m.e_c}), max diff {dmax:.1e}")


def t4(like):
    rng = np.random.default_rng(4)
    mu1, mu2, s1, s2, rho = 0.45, 0.20, 0.10, 0.12, -0.6
    pri = {k: v for k, v in PRIORS.items() if k not in ("q1", "q2")}
    pri.update(u1=Normal(mu1, s1), u2=Normal(mu2, s2))
    m = S._Model(like, pri, "planet", False, rho)
    U = rng.uniform(size=(400_000, m.ndim))
    th, lw = m.native(U)
    ok = lw > S.LOGZERO
    u1, u2 = th[ok, 0], th[ok, 1]
    z = rng.standard_normal((2_000_000, 2))
    v1 = mu1 + s1 * z[:, 0]
    v2 = mu2 + s2 * (rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1])
    inside = (v1 >= 0) & (v1 + v2 <= 1) & (v1 + 2 * v2 >= 0)
    v1, v2 = v1[inside], v2[inside]
    d = max(abs(u1.mean() - v1.mean()), abs(u2.mean() - v2.mean()), abs(u1.std() - v1.std()),
            abs(u2.std() - v2.std()), abs(np.corrcoef(u1, u2)[0, 1] - np.corrcoef(v1, v2)[0, 1]))
    lnorm = float(lw[ok][0])
    check(d < 0.004 and abs(lnorm + math.log(inside.mean())) < 0.003,
          f"T4 correlated Normal (u1, u2) prior, truncated: moments vs rejection sampling max diff {d:.4f}; "
          f"log normalisation {lnorm:.4f} vs {-math.log(inside.mean()):.4f}")


def t5(like):
    d = tempfile.mkdtemp(prefix="pandora_t5_")
    try:
        pri = {k: v for k, v in PRIORS.items()}
        s = NestedSampler(like, pri, model="planet", log_dir=d, resume="overwrite", verbose=False,
                          planet_evidence="nested")
        r = s.run(accuracy="medium")
        keys = ("logz", "logzerr", "samples", "weighted_samples", "posterior", "maxlikelihood", "ncall", "paramnames",
                "ess", "H", "passed")
        check(all(k in r for k in keys) and r["samples"].shape[1] == len(r["paramnames"]) == 7,
              f"T5 planet fit: result keys, {r['samples'].shape[0]} samples of {r['paramnames']}; logZ "
              f"{r['logz']:.3f} +- {r['logzerr']:.3f}, passed {r['passed']}, {r['wall']:.0f} s")
        files = ("info/results.json", "info/post_summary.csv", "chains/equal_weighted_post.txt",
                 "chains/weighted_post.npz", "info/fit_state.json")
        check(all(os.path.exists(os.path.join(d, f)) for f in files) and len(os.listdir(os.path.join(d, "runs"))) > 0,
              "T5 files in log_dir: " + ", ".join(files) + f", runs/ ({len(os.listdir(os.path.join(d, 'runs')))})")
        vals = s.parameters()
        lv = s.likelihood(*[vals[k] for k in S.PARAMS])     # the likelihood the fit used (supersampled: "auto")
        check(abs(lv - r["maxlikelihood"]["logl"]) < 1e-6 * max(1.0, abs(lv)),
              f"T5 sampler.parameters(): Likelihood at the returned max-likelihood parameters {lv:.6f} = "
              f"maxlikelihood logl {r['maxlikelihood']['logl']:.6f}")
        fig = s.plot_run()
        check(os.path.exists(os.path.join(d, "plots", "run.pdf")), "T5 plot_run() writes plots/run.pdf")
        import matplotlib.pyplot as plt
        plt.close("all")
        s2 = NestedSampler(like, pri, model="planet", log_dir=d, resume="resume", verbose=False,
                           planet_evidence="nested")
        r2 = s2.run(accuracy="medium")
        check(r2["logz"] == r["logz"] and r2["runs"] == r["runs"],
              f"T5 resume of a finished fit: logZ {r2['logz']:.4f} (was {r['logz']:.4f}), {r2['runs']} runs, "
              f"same: {r2['logz'] == r['logz']}")
        # truth inside the posterior (r_planet 0.1, b 0.3, a 215)
        p = r["posterior"]
        j = r["paramnames"].index("r_planet")
        check(p["errlo"][j] - 0.01 < 0.1 < p["errup"][j] + 0.01,
              f"T5 r_planet {p['median'][j]:.4f} [{p['errlo'][j]:.4f}, {p['errup'][j]:.4f}] (truth 0.1)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


from numba import njit  # noqa: E402

_KD, _S, _S1, _S2 = 8, 0.02, 0.06, 0.004
_BB = 10.0 * _S2 / _S1 ** 2
_MU = np.full(_KD, 0.5)
_MU[1] = 0.6


@njit
def _known_ll(u):
    """lnZ = ln(1/2): banana (u0, u1), half-Gaussian at the edge u2 = 0, Gaussians in u3..u7"""
    y1 = u[0] - _MU[0]
    y2 = u[1] - _MU[1]
    z1 = y1 / _S1
    z2 = (y2 + _BB * (y1 * y1 - _S1 * _S1)) / _S2
    t = -0.5 * (z1 * z1 + z2 * z2) - math.log(_S1 * _S2) - math.log(2 * math.pi)
    z = u[2] / _S
    t += -0.5 * z * z - math.log(_S) - 0.5 * math.log(2 * math.pi)
    for i in range(3, _KD):
        z = (u[i] - _MU[i]) / _S
        t += -0.5 * z * z - math.log(_S) - 0.5 * math.log(2 * math.pi)
    return t


def t7():
    from pandoramoon import nss
    truth = math.log(0.5)
    th = min(8, os.cpu_count() or 1)
    r = nss.fit(_known_ll, _KD, nlive=300, threads=th, lz_tol=0.08, ks_tol=0.1, min_runs=8, max_runs=96, seed=5,
                verbose=False)
    dz = r["logz"] - truth
    check(abs(dz) <= 3 * r["logzerr"] + 0.1 and r["passed"],
          f"T7 known evidence (8-d banana + edge + Gaussians): logZ {r['logz']:.3f} +- {r['logzerr']:.3f} vs "
          f"truth {truth:.3f} (diff {dz:+.3f}), passed {r['passed']}, nsteps {r['nsteps']}, {len(r['runs'])} runs, "
          f"{r['wall']:.0f} s")
    # a deliberately broken start: chains of 2 slice steps. The nsteps check must notice and double the chains.
    rb = nss.fit(_known_ll, _KD, nlive=300, nsteps=2, threads=th, lz_tol=0.08, ks_tol=0.1, min_runs=8, max_runs=96,
                 seed=9, verbose=False)
    check(rb["n_escalations"] >= 1 and abs(rb["logz"] - truth) <= 3 * rb["logzerr"] + 0.15,
          f"T7 started with too short chains (nsteps 2): the check doubled them {rb['n_escalations']}x to "
          f"{rb['nsteps']}; logZ {rb['logz']:.3f} +- {rb['logzerr']:.3f} (diff {rb['logz'] - truth:+.3f})")


_PF = 0.75


@njit
def _plateau_ll(u):
    """8-d Gaussian (sigma 0.02) in the allowed part u0 >= 0.75 of the cube, LOGZERO below: lnZ = 0"""
    if u[0] < _PF:
        return -1e300
    t = 0.0
    for i in range(8):
        z = (u[i] - (_PF + 0.5 * (1 - _PF) if i == 0 else 0.5)) / 0.02
        t -= 0.5 * z * z
    return t - 8 * (math.log(0.02) + 0.5 * math.log(2 * math.pi))


def t9():
    from pandoramoon import nss
    rs = [nss.run(_plateau_ll, 8, nlive=200, nsteps=32, seed=s, dirs="pairs", kpair=4) for s in range(1, 9)]
    lz, _ = nss.merge_weights(rs)
    err = math.sqrt(np.mean([r["logzerr"] ** 2 for r in rs]) / len(rs))
    npl = sum(r["n_plateau"] for r in rs)
    nst = sum(r["nstuck"] for r in rs)
    lx0 = np.mean([r["logx0"] for r in rs])
    check(npl == 0 and nst == 0 and abs(lx0 - math.log(1 - _PF)) < 0.02 and abs(lz) < 4 * err + 0.05,
          f"T9 rejection init (default) on a 75 % LOGZERO plateau: lnZ {lz:+.3f} +- {err:.3f} (truth 0), ln X0 {lx0:.4f} "
          f"(ln 0.25 = {math.log(0.25):.4f}), plateau points {npl}, stuck steps {nst}")
    r0 = nss.run(_plateau_ll, 8, nlive=200, nsteps=32, seed=1, dirs="pairs", kpair=4, init_reject=False)
    check(r0["n_plateau"] > 100 and r0["logx0"] == 0.0,
          f"T9 old init (init_reject=False): {r0['n_plateau']} plateau points of 200, {r0['nstuck']} stuck steps")


def _make_twomode(vr, ndim=8, s1=0.01, w2=0.5):
    s2 = s1 * vr ** (1.0 / ndim)
    lw1 = math.log(1 - w2) - ndim * (math.log(s1) + 0.5 * math.log(2 * math.pi))
    lw2 = math.log(w2) - ndim * (math.log(s2) + 0.5 * math.log(2 * math.pi))

    @njit
    def ll(u):
        a = 0.0
        b = 0.0
        for i in range(ndim):
            z1 = (u[i] - 0.35) / s1
            z2 = (u[i] - 0.65) / s2
            a -= 0.5 * z1 * z1
            b -= 0.5 * z2 * z2
        a += lw1
        b += lw2
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))
    return ll


def t10():
    from pandoramoon import nss
    th = min(8, os.cpu_count() or 1)
    # a compact mode with 99.9 % of the mass in 10 % of the volume: runs that miss it are ~7 nats low and have a
    # lower maxL -> n_live escalation, and no nsteps escalation (Round 21). (An equal-mass compact mode at this n_live
    # is detected only statistically, by the scatter criterion: borderline, not a reliable unit test.)
    r = nss.fit(_make_twomode(1e-1, w2=0.999), 8, nlive=200, threads=th, lz_tol=0.2, ks_tol=0.08, min_runs=8, seed=3,
                verbose=False)
    check(r["n_nlive_escalations"] >= 1 and r["multimodal"] and r["nlive"] > 200 and r["n_escalations"] == 0
          and abs(r["logz"]) < 3 * r["logzerr"] + 0.2,
          f"T10 dominant compact mode (volume 1e-1, 99.9 % of the mass): n_live escalated {r['n_nlive_escalations']}x to "
          f"{r['nlive']}, nsteps escalations {r['n_escalations']}, multimodal {r['multimodal']}, logZ {r['logz']:+.3f} +- "
          f"{r['logzerr']:.3f} (truth 0)")
    u = nss.fit(_known_ll, _KD, nlive=300, threads=th, lz_tol=0.08, ks_tol=0.1, min_runs=8, seed=5, verbose=False)
    check(u["n_nlive_escalations"] == 0 and not u["multimodal"],
          f"T10 unimodal known-Z problem: no n_live escalation ({u['n_nlive_escalations']}), multimodal {u['multimodal']}")


def t13(like):
    """planet-only evidence by importance sampling (2026-10-14): agrees with the full nested fit; a failed check falls
    back to the nested fit"""
    from pandoramoon._config import settings
    th = min(8, os.cpu_count() or 1)
    ri = NestedSampler(like, PRIORS, model="planet", verbose=False, threads=th, seed=5).run(accuracy="high",
                                                                                          progress=False)
    rn = NestedSampler(like, PRIORS, model="planet", verbose=False, threads=th, seed=5,
                       planet_evidence="nested").run(accuracy="high", progress=False)
    d = ri["logz"] - rn["logz"]
    s = math.hypot(ri["logzerr"], rn["logzerr"])
    j = ri["paramnames"].index("r_planet")
    dr = abs(ri["posterior"]["median"][j] - rn["posterior"]["median"][j]) / rn["posterior"]["stdev"][j]
    check(ri.get("method") == "importance" and abs(d) < 3 * s + 0.05 and dr < 0.2 and ri["ncall"] < rn["ncall"] / 10,
          f"T13 planet lnZ by importance sampling {ri['logz']:.3f} +- {ri['logzerr']:.3f} vs nested {rn['logz']:.3f} +- "
          f"{rn['logzerr']:.3f} ({d / s:+.1f} sigma), r_planet median differs by {dr:.2f} sd, calls {ri['ncall']:,} vs "
          f"{rn['ncall']:,}")
    S = settings()["planet_is"]
    old = S["ess_min"]
    S["ess_min"] = 1e12                       # force a failed check
    try:
        rf = NestedSampler(like, PRIORS, model="planet", verbose=False, threads=th, seed=5).run(accuracy="medium",
                                                                                              progress=False)
    finally:
        S["ess_min"] = old
    check(rf.get("method", "nested") == "nested" and rf["runs"] >= 8,
          f"T13 failed importance-sampling check -> nested-sampling fit ({rf['runs']} runs, lnZ {rf['logz']:.3f})")


def t14():
    """a/R* prior (2026-10-14): default_priors' wide log-uniform default; with a wrong catalogue star (true a = 0.6 x
    Kepler's law, no moon) the old narrow prior gives a false dlnZ and the prior-edge warning of detect; the new
    default neither"""
    from pandoramoon import detect, default_priors
    from pandoramoon.search import model_flux
    from pandoramoon.kernel import PLANET_ONLY_MOON
    from pandoramoon.sampling import LogUniform as _LU
    from pandoramoon.constants import G, DAY, R_SUN, M_SUN, M_JUP
    P, t0 = 200.0, 10.0
    ak = (G * (M_SUN + M_JUP) * (P * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / R_SUN
    pr = default_priors(t0, P, 0.1)
    check(isinstance(pr["a_bary"], _LU) and abs(pr["a_bary"].lo / ak - 0.4) < 1e-6 and abs(pr["a_bary"].hi / ak - 2.5) < 1e-6,
          f"T14 default a_bary prior log-uniform 0.4-2.5 x Kepler's law ({pr['a_bary']})")
    t = np.concatenate([np.arange(t0 + k * P - 1.5, t0 + k * P + 1.5, 1 / 48) for k in range(4)])
    p = dict(u1=0.4, u2=0.25, R_star=R_SUN, per_bary=P, a_bary=0.6 * ak, r_planet=0.1, b_bary=0.3, w_bary=0.0,
             ecc_bary=0.0, t0_bary=t0, t0_bary_offset=0.0, M_planet=M_JUP)
    p.update(zip(S.MOON_NAMES, PLANET_ONLY_MOON))
    lk0 = pandora.Likelihood(t, np.ones_like(t), np.ones_like(t), epoch_distance=P)
    f = model_flux(lk0, p) + np.random.default_rng(3).normal(0, 300e-6, len(t))
    like = pandora.Likelihood(t, f, np.full(len(t), 300e-6), epoch_distance=P)
    old = detect(like, default_priors(t0, P, 0.1, a_bary=(0.977 * ak, 1.023 * ak)), recall=0.9, seed=1)
    new = detect(like, pr, recall=0.9, seed=1)
    w_old = [w for w in old["warnings"] if "a_bary" in w]
    w_new = [w for w in new["warnings"] if "a_bary" in w]
    check(len(w_old) >= 1 and old["dlnz"] > 3 and not w_new and new["dlnz"] < 3,
          f"T14 wrong star (no moon): old narrow prior -> a_bary edge warning ({len(w_old)}), false dlnZ {old['dlnz']:+.1f}; "
          f"new default -> warnings {len(w_new)}, dlnZ {new['dlnz']:+.1f}")


def t12(like):
    import pandoramoon as P
    from pandoramoon import search as SR
    p = pandora.model_params()
    p.epochs, p.epoch_duration, p.cadences_per_day, p.epoch_distance, p.t0_bary = 3, 2.0, 48, 365.26, 11.0
    p.supersampling_factor = 1
    t1 = pandora.time(p).grid()
    p.supersampling_factor = 5
    t5 = pandora.time(p).grid()
    l5 = SR.resample_likelihood(like, 5)
    l5b = pandora.Likelihood(t5, like.data, 1 / like.inv_err, epoch_distance=like.epoch_distance,
                             supersampling_factor=5, hill_sphere_threshold=1.0)
    th = [0.4089, 0.2556, R_SUN, 365.25, 215.0, 0.1, 0.3, 0, 0, 11.0, 0.01, 1.8986e27, 0.03526, 5.0, 0.07, 0.0, 90.0,
          0, 0, 0.05395 * 1.8986e27]
    dg = float(np.abs(SR.subexposure_times(t1, 5) - t5).max())
    check(dg < 1e-9 and abs(l5(*th) - l5b(*th)) < 1e-9,
          f"T12 resample_likelihood: sub-exposure grid = pandora.time (max diff {dg:.1e} d), logL {l5(*th):.6f} = "
          f"{l5b(*th):.6f}")
    pr = P.default_priors(t0_bary=11.0, per_bary=365.25, r_planet=0.1)
    s_auto = NestedSampler(like, pr, verbose=False)
    s_none = NestedSampler(like, pr, verbose=False, supersampling=None)
    check(s_auto.likelihood.supersampling_factor == 5 and s_none.likelihood.supersampling_factor == 1
          and s_auto.models["moon"].moon_phase,
          "T12 NestedSampler(supersampling='auto') uses 5 sub-exposures for 30-min data, None keeps 1; "
          "default_priors gives the moon-phase coordinates")
    d = P.detect(like, pr, seed=3)
    # planet-only light curve: the same system without the moon
    pl = pandora.model_params()
    pl.R_star, pl.u1, pl.u2 = R_SUN, 0.4089, 0.2556
    pl.per_bary, pl.a_bary, pl.r_planet, pl.b_bary = 365.25, 215.0, 0.1, 0.3
    pl.t0_bary, pl.t0_bary_offset, pl.M_planet, pl.w_bary, pl.ecc_bary = 11.0, 0.01, 1.8986e27, 0.0, 0.0
    pl.r_moon, pl.per_moon, pl.tau_moon, pl.Omega_moon, pl.i_moon = 1e-6, 5.0, 0.07, 0.0, 90.0
    pl.M_moon, pl.ecc_moon, pl.w_moon = 1.0, 0.0, 0.0
    pl.epochs, pl.epoch_duration, pl.cadences_per_day, pl.epoch_distance = 3, 2.0, 48, 365.26
    pl.supersampling_factor, pl.occult_small_threshold, pl.hill_sphere_threshold = 1, 0, 1.0
    fl, _, _ = pandora.moon_model(pl).light_curve(t1)
    rng = np.random.default_rng(1)
    like0 = pandora.Likelihood(t1, fl + rng.normal(0, 500e-6, len(t1)), np.full(len(t1), 500e-6), epoch_distance=365.26,
                               hill_sphere_threshold=1.0)
    d0 = P.detect(like0, pr, seed=3)
    check(d["candidate"] and d["dlnz"] > 50 and d["jackknife"]["passed"] and not d0["interesting"],
          f"T12 detect: moon light curve dlnZ {d['dlnz']:+.1f} (jackknife {d['jackknife']['dchi2_without_top']:.0f}, "
          f"candidate {d['candidate']}), planet-only light curve dlnZ {d0['dlnz']:+.1f} (interesting {d0['interesting']})")
    jk = P.epoch_jackknife(like, d["best"]["planet"], d["best"]["planet"])
    check(abs(jk["dchi2"]) < 1e-9 and not jk["passed"] and len(jk["epochs"]) == 3,
          f"T12 epoch_jackknife of identical models: dchi2 {jk['dchi2']:.1e}, epochs {jk['epochs']}")
    try:
        import wotan  # noqa: F401
    except ImportError:
        print("SKIP  T12 prepare_lightcurve (wotan not installed)")
        return
    tt = np.arange(0, 40, 1 / 48.0)
    ff = 1 + 300e-6 * np.sin(2 * np.pi * tt / 9.0) + rng.normal(0, 400e-6, len(tt))
    lc = P.prepare_lightcurve(tt, ff, t0=5.0, period=10.0, duration=0.4)
    check(abs(lc["sigma"] / 400e-6 - 1) < 0.1 and lc["beta"] < 1.3 and len(lc["epochs"]) == 4,
          f"T12 prepare_lightcurve (white noise 400 ppm + slow variability): sigma {1e6 * lc['sigma']:.0f} ppm, "
          f"beta {lc['beta']:.2f}, epochs {lc['epochs']}")


def t8(like):
    import warnings
    # (1) Normal prior on q1 reaching outside [0, 1]: truncated, so no cube point is outside the prior any more
    pri = {k: v for k, v in PRIORS.items()}
    pri["q1"] = Normal(0.5, 0.4)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        m = S._Model(like, pri, "planet", False, 0.0)
    rng = np.random.default_rng(8)
    U = rng.uniform(size=(200_000, m.ndim))
    th, lw = m.native(U)
    lost = float(np.mean(~(lw > S.LOGZERO)))
    q1 = m.to_params(U)[:, m.paramnames().index("q1")]
    # reference: rejection sampling of N(0.5, 0.4) restricted to [0, 1]
    z = rng.normal(0.5, 0.4, 2_000_000)
    z = z[(z >= 0) & (z <= 1)]
    d = max(abs(q1.mean() - z.mean()), abs(q1.std() - z.std()))
    check(lost == 0.0 and d < 0.003 and any("q1" in str(w.message) for w in wl) and 0.2 < m.truncated["q1"] < 0.23,
          f"T8 Normal(0.5, 0.4) on q1: truncated to [0, 1] ({100 * m.truncated['q1']:.1f} % of its mass outside, warned), "
          f"cube points outside the prior {lost:.4f} (was ~0.21 before Round 17), moments vs rejection sampling {d:.4f}")
    # (2) Normal on r_planet reaching below 0
    pri = {k: v for k, v in PRIORS.items()}
    pri["r_planet"] = Normal(0.02, 0.02)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        m = S._Model(like, pri, "planet", False, 0.0)
    th, lw = m.native(rng.uniform(size=(50_000, m.ndim)))
    check(th[:, S.IDX["r_planet"]].min() >= 0 and any("r_planet" in str(w.message) for w in wl),
          f"T8 Normal(0.02, 0.02) on r_planet: truncated at 0 ({100 * m.truncated['r_planet']:.1f} % outside, warned), "
          f"min r_planet {th[:, S.IDX['r_planet']].min():.2e}")
    # (3) unbiased thinning of weighted points (Horvitz-Thompson)
    w = rng.lognormal(0, 2, 100_000)
    w /= w.sum()
    x = rng.normal(size=len(w)) + 3 * (w > np.quantile(w, 0.999))
    full = float(np.sum(w * x))
    est, ns = [], []
    for sd in range(200):
        i, nw = S._thin(w, 5000, seed=sd)
        est.append(float(np.sum(nw * x[i])))
        ns.append(len(i))
    bias = np.mean(est) - full
    se = np.std(est) / math.sqrt(len(est))
    check(abs(bias) < 4 * se + 1e-4 and 4500 < np.mean(ns) < 5500,
          f"T8 thinning 1e5 -> {np.mean(ns):.0f} points: weighted mean {np.mean(est):.4f} vs full {full:.4f} "
          f"(bias {bias:+.4f} +- {se:.4f})")


T6_CHILD = """
import sys
import numpy as np
sys.path.insert(0, %r)
import pandoramoon as pandora
import test_sampling as T
from pandoramoon import NestedSampler
# the parent's data from a file, not regenerated: simulated flux can differ in the last bit between processes whose
# numba code was compiled fresh vs loaded from the cache (fastmath), and the resume compares the data's hash
z = np.load(%r)
like = pandora.Likelihood(z["time"], z["data"], z["yerr"], epoch_distance=365.26, hill_sphere_threshold=1.0)
NestedSampler(like, T.PRIORS, model="planet", log_dir=%r, resume="overwrite", verbose=False,
              planet_evidence="nested", threads=2).run(accuracy="medium")
"""


def t6(like):
    import subprocess
    import time as _t
    d = tempfile.mkdtemp(prefix="pandora_t6_")
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        dfile = os.path.join(d, "t6_data.npz")
        np.savez(dfile, time=like.time, data=like.data, yerr=like.yerr_input)
        pr = subprocess.Popen([sys.executable, "-c", T6_CHILD % (here, dfile, d)])
        runs = os.path.join(d, "runs")
        t0 = _t.time()
        while _t.time() - t0 < 600:
            if os.path.isdir(runs) and len([f for f in os.listdir(runs) if ".tmp" not in f]) >= 3:
                break
            _t.sleep(0.5)
        pr.kill()
        pr.wait()
        stored = sorted(f for f in os.listdir(runs) if ".tmp" not in f)
        idx_pre = {int(f[3:8]) for f in stored}
        lost = {i for i in range(max(idx_pre)) if i not in idx_pre}       # in flight at the crash
        s = NestedSampler(like, PRIORS, model="planet", log_dir=d, resume="resume", verbose=True, threads=2,
                          planet_evidence="nested")
        r = s.run(accuracy="medium")
        now = sorted(f for f in os.listdir(runs) if ".tmp" not in f)
        kept = all(f in now for f in stored)
        idx_now = {int(f[3:8]) for f in now}
        check(lost <= idx_now, f"T6 the runs lost in flight at the crash (indices {sorted(lost)}) are redone with their "
              f"seeds after the resume (stored indices now {sorted(idx_now)[:12]}...)")
        d2 = tempfile.mkdtemp(prefix="pandora_t6b_")
        ref = NestedSampler(like, PRIORS, model="planet", log_dir=d2, resume="overwrite", verbose=False,
                            planet_evidence="nested",
                            threads=2).run(accuracy="medium")
        shutil.rmtree(d2, ignore_errors=True)
        check(len(stored) >= 3 and kept and r["passed"],
              f"T6 crash after {len(stored)} stored runs: resumed, the stored runs kept and used, {len(now)} runs "
              f"stored at the end, passed {r['passed']}, logZ {r['logz']:.3f} +- {r['logzerr']:.3f}")
        # same seeds and the same thread count: the resumed fit continues the uninterrupted one's run sequence;
        # the stop can differ by the runs in flight at the crash (they are redone), so compare within the error
        check(abs(r["logz"] - ref["logz"]) <= 2 * math.hypot(r["logzerr"], ref["logzerr"]),
              f"T6 resumed vs uninterrupted fit (2 threads): logZ {r['logz']:.3f} vs {ref['logz']:.3f} "
              f"({r['runs']} vs {ref['runs']} runs)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    like = make_data()
    t1(like)
    t2()
    t3(like)
    t4(like)
    if "--quick" not in sys.argv:
        t5(like)
        t6(like)
        t7()
        t9()
        t10()
    t12(like)
    if "--quick" not in sys.argv:
        t13(like)
        t14()
    t8(like)
    print(f"\n{'OK' if not FAIL else 'FAILED'}: {len(FAIL)} failing")
    sys.exit(1 if FAIL else 0)
