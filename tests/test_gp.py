"""Tests of the simultaneous GP detrending (pandoramoon/gp.py, Likelihood(gp=...)). Standalone:

    python test_gp.py            # CPU tests, ~1 min (numba compile included)
    python test_gp.py --gpu      # also the GPU likelihood against the CPU one (needs pandoramoon[gpu])

1. the GP log-likelihood equals dense linear algebra (Cholesky of K + diag(s^2 yerr^2)) for the moon and the
   planet-only model, with and without supersampling and with a missing point;
2. amp -> 0 with err_scale 1 gives the white-noise likelihood; gp=None leaves the white-noise likelihood unchanged;
3. the exact gradient of the hyperparameter likelihood equals finite differences; GP.fit recovers the
   hyperparameters of an SHO draw;
4. NestedSampler's compiled likelihood (the unit-cube function nss samples) uses the GP, resampling keeps it;
5. (--gpu) the GPU likelihood with the GP equals the CPU one.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("PANDORA_PATH"):
    sys.path.insert(0, os.environ["PANDORA_PATH"])
import pandoramoon as pandora                                          # noqa: E402
from pandoramoon.gp import GP, _nll_grad, time_table, _prepare        # noqa: E402

FAILS = []
P = 365.25
T0 = 20.0
BASE = dict(u1=0.4, u2=0.25, R_star=696342e3, per_bary=P, a_bary=215.0, r_planet=0.1, b_bary=0.3, w_bary=0.0,
            ecc_bary=0.0, t0_bary=T0, t0_bary_offset=0.01, M_planet=1.0e26, r_moon=0.035, per_moon=1.28,
            tau_moon=0.3, Omega_moon=0.0, i_moon=80.0, ecc_moon=0.0, w_moon=0.0, M_moon=5.5e24)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def sho_k(dt, amp, tau):
    x = np.abs(dt) / (math.sqrt(2) * tau)
    return amp ** 2 * np.exp(-x) * (np.cos(x) + np.sin(x))


def dataset(ss, rng, amp=4e-4, tau=0.3, s=1.2):
    tc = np.concatenate([T0 + e * P + np.arange(-1, 1, 1 / 144) for e in range(4)])
    tc = tc[rng.random(len(tc)) > 0.05]
    time_ = tc if ss == 1 else (tc[:, None] + (np.arange(ss) - (ss - 1) / 2) * (1 / 144 / ss)).ravel()
    yerr = np.full(len(tc), 1e-4)
    C = sho_k(tc[:, None] - tc[None, :], amp, tau) + np.diag((s * yerr) ** 2)
    _, _, f = pandora.pandora_flux(*BASE.values(), P + 0.01, ss, 0.0, 1.2, time_)
    data = np.asarray(f) + np.linalg.cholesky(C) @ rng.standard_normal(len(tc))
    data[5] = np.nan
    return tc, time_, data, yerr, C


def test_dense():
    rng = np.random.default_rng(3)
    for ss in (1, 3):
        tc, time_, data, yerr, C = dataset(ss, rng)
        g = GP(4e-4, 0.3, 1.2)
        L = pandora.Likelihood(time_, data, yerr, epoch_distance=P + 0.01, supersampling_factor=ss,
                               hill_sphere_threshold=1.2, gp=g)
        valid = np.isfinite(data)
        Lv = np.linalg.cholesky(C[np.ix_(valid, valid)])
        ld = np.sum(np.log(np.diag(Lv))) - np.sum(np.log(yerr[valid]))
        worst = 0.0
        for _ in range(12):
            p = dict(BASE, t0_bary_offset=rng.uniform(-0.05, 0.05), r_planet=rng.uniform(0.05, 0.12),
                     tau_moon=rng.uniform(0, 1), Omega_moon=rng.uniform(-30, 30), per_moon=rng.uniform(0.8, 2.0))
            _, _, fm = pandora.pandora_flux(*p.values(), P + 0.01, ss, 0.0, 1.2, time_)
            z = np.linalg.solve(Lv, (data - np.asarray(fm))[valid])
            worst = max(worst, abs(L(*p.values()) - (-0.5 * z @ z - ld)))
        check(f"moon model = dense algebra (supersampling {ss})", worst < 1e-6, f"max error {worst:.1e}")
        from pandoramoon.gp import pandora_loglike_planet_gp
        pp = list(BASE.values())[:12]
        _, _, fp = pandora.pandora_flux(*pp, 0.0, 1e-6, 0, 0, 0, 0, 0, 0, P + 0.01, ss, 0.0, 1.2, time_)
        z = np.linalg.solve(Lv, (data - np.asarray(fp))[valid])
        got = pandora_loglike_planet_gp(*pp, P + 0.01, ss, 0.0, 1.2, L.time, *L.gp_tables, None, True)
        err = abs(got - (-0.5 * z @ z - ld))
        check(f"planet-only model = dense algebra (supersampling {ss})", err < 1e-6, f"error {err:.1e}")


def test_white_limit():
    rng = np.random.default_rng(4)
    tc, time_, data, yerr, C = dataset(1, rng)
    Lw = pandora.Likelihood(time_, data, yerr, epoch_distance=P + 0.01, hill_sphere_threshold=1.2)
    L0 = pandora.Likelihood(time_, data, yerr, epoch_distance=P + 0.01, hill_sphere_threshold=1.2,
                            gp=GP(1e-10, 0.3, 1.0))
    a, b = L0(*BASE.values()), Lw(*BASE.values())
    check("amp -> 0: the white-noise likelihood", abs(a - b) < 1e-3 * max(1.0, abs(b)) * 1e-3, f"{a:.6f} vs {b:.6f}")
    from pandoramoon.kernel import pandora_loglike
    ref = pandora_loglike(*[float(x) for x in BASE.values()], *Lw._tail)
    check("gp=None: Likelihood unchanged (the compiled white-noise likelihood)", Lw(*BASE.values()) == ref)


def test_fit():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 40, 6000))
    import gp_trend_case as G
    y = 1 + G.sho_draw(t, 8e-4, 0.6, rng) + rng.normal(0, 2e-4, len(t))
    t_, y1, e, v = _prepare(t, y, 2e-4, None)
    idx, steps = time_table(t_, 1e-6)
    z = np.array([math.log(5e-4), math.log(0.3), 0.1])

    def f(z):
        return _nll_grad(idx, steps, y1, (math.exp(z[2]) * e) ** 2, e, math.exp(z[0]), math.exp(z[1]))
    g = f(z)[1]
    num = np.array([(f(z + h)[0] - f(z - h)[0]) / 2e-6 for h in np.eye(3) * 1e-6])
    err = np.max(np.abs(g - num) / (np.abs(num) + 1))
    check("hyperparameter gradient = finite differences", err < 1e-5, f"max rel error {err:.1e}")
    gf = GP.fit(t, y, 2e-4, bin_width=None)
    ok = abs(gf.amp / 8e-4 - 1) < 0.25 and abs(gf.tau / 0.6 - 1) < 0.35 and abs(gf.err_scale - 1) < 0.1
    check("GP.fit recovers amp, tau, err_scale of an SHO draw", ok, repr(gf))


def test_sampler():
    import gp_trend_case as G
    c = G.make(moon=True, seed=0)
    g = GP.fit(c["time"], c["flux"], c["yerr"], mask=c["mask"])
    L = pandora.Likelihood(c["time"], c["flux"], c["yerr"], epoch_distance=c["epoch_distance"],
                           hill_sphere_threshold=1.0, gp=g)
    pri = pandora.default_priors(t0_bary=c["t0_bary"], per_bary=c["per_bary"], r_planet=c["r_planet"])
    S = pandora.NestedSampler(L, pri, verbose=False, supersampling=3)
    m = S.models["moon"]
    check("NestedSampler keeps the GP through supersampling='3'", S.likelihood.gp is g)
    from pandoramoon.sampling import _loglike_gp
    check("NestedSampler's moon model samples the GP likelihood", m.ll is _loglike_gp)
    U = np.random.default_rng(1).uniform(size=(20, m.ndim))
    th, lw = m.native(U)
    ok = True
    for i in range(len(U)):
        if not lw[i] > -1e299:
            continue
        a = m.ll(U[i], m.D)
        b = S.likelihood(*th[i]) + lw[i]
        ok &= bool(abs(a - b) < 1e-6 * max(1.0, abs(b)))
    check("unit-cube likelihood = Likelihood(theta) + coordinate weight", ok)
    mp = S.models["planet"]
    check("planet-only model samples the GP likelihood too", mp.ll is _loglike_gp)


def test_gpu():
    from pandoramoon import gpu
    if not gpu.available():
        check("GPU available", False, "no GPU / CUDA packages")
        return
    import gp_trend_case as G
    from pandoramoon.gpu.sampler import PandoraEvaluator
    from numba import cuda
    for ss in (1, 3):
        c = G.make(moon=True, seed=0)
        g = GP.fit(c["time"], c["flux"], c["yerr"], mask=c["mask"])
        L = pandora.Likelihood(c["time"], c["flux"], c["yerr"], epoch_distance=c["epoch_distance"],
                               hill_sphere_threshold=1.0, gp=g)
        pri = pandora.default_priors(t0_bary=c["t0_bary"], per_bary=c["per_bary"], r_planet=c["r_planet"])
        S = pandora.NestedSampler(L, pri, verbose=False, supersampling=ss)
        m = S.models["moon"]
        ev = PandoraEvaluator(m, S.likelihood, t0_ref=c["t0_bary"])
        rng = np.random.default_rng(7)
        U = rng.uniform(size=(4096, m.ndim))
        th, lw = m.native(U)
        ok = lw > -1e299
        n = len(U)
        dU = cuda.to_device(np.ascontiguousarray(U))
        dACT = cuda.to_device(np.ones(n, np.int32))
        dLW = cuda.device_array(n, np.float64)
        dLLK = cuda.device_array(n, np.float64)
        ev.evaluate(dU, dACT, n, dLW, dLLK)
        got = dLLK.copy_to_host()
        ref = np.array([S.likelihood(*th[i]) if ok[i] else -np.inf for i in range(n)])
        # random prior points are poor fits (logL -1e5 ... -2e4): FP32 rounding of the large terms, relative
        # (GPU_LOG Round 9: at posterior points the error is 3e-4 nats median, 1.2e-3 max, below the white-noise path's)
        fin = np.isfinite(ref) & (ref > -1e6)
        rel = np.abs(got[fin] - ref[fin]) / np.abs(ref[fin])
        check(f"GPU GP likelihood = CPU (supersampling {ss}, {fin.sum()} points)", np.max(rel) < 1e-5,
              f"max |GPU - CPU| / |logL| {np.max(rel):.1e}, median {np.median(rel):.1e}")
        inv = ~np.isfinite(ref)
        check(f"GPU: -inf where the CPU has -inf (supersampling {ss})", bool(np.all(got[inv] == -np.inf)))


if __name__ == "__main__":
    test_dense()
    test_white_limit()
    test_fit()
    test_sampler()
    if "--gpu" in sys.argv:
        test_gpu()
    print(f"\n{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
