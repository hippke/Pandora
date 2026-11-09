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


def kern_sho(dt, amp, tau, Q):
    """celerite's SHOTerm (S0 w0 Q = amp^2), written independently of pandoramoon.gp"""
    s = np.abs(dt) / tau
    if Q > 0.5:
        eta = math.sqrt(1 - 1 / (4 * Q * Q))
        return amp ** 2 * np.exp(-s / (2 * Q)) * (np.cos(eta * s) + np.sin(eta * s) / (2 * eta * Q))
    if Q == 0.5:
        return amp ** 2 * np.exp(-s) * (1 + s)
    f = math.sqrt(1 - 4 * Q * Q)
    return amp ** 2 * 0.5 * np.exp(-s / (2 * Q)) * ((1 + 1 / f) * np.exp(f * s / (2 * Q))
                                                     + (1 - 1 / f) * np.exp(-f * s / (2 * Q)))


def dense_cov(g, t):
    """the covariance of the GP terms + baseline at t (segments independent), without the white noise"""
    seg = np.cumsum(g.segment_starts(t))
    dt = t[:, None] - t[None, :]
    C = np.zeros((len(t), len(t)))
    for s in g.sho:
        C += kern_sho(dt, s.amp, s.tau, s.Q)
    if g.baseline:
        C += g.baseline_sd[0] ** 2
    if g.baseline == "linear":
        t0 = np.array([t[seg == k][0] for k in seg])
        C += g.baseline_sd[1] ** 2 * np.outer(t - t0, t - t0)
    return np.where(seg[:, None] == seg[None, :], C, 0.0)


def general_gps():
    from pandoramoon.gp import SHO, Rotation
    return {
        "SHO Q=3": GP(4e-4, 0.3, 1.2, Q=3.0),
        "SHO Q=0.5": GP(4e-4, 0.3, 1.2, Q=0.5),
        "SHO Q=0.3": GP(4e-4, 0.3, 1.2, Q=0.3),
        "SHO, gap": GP(4e-4, 0.3, 1.2, gap=0.5),
        "SHO + offset": GP(4e-4, 0.3, 1.2, baseline="offset"),
        "SHO + linear, gap": GP(4e-4, 0.3, 1.2, baseline="linear", gap=0.5),
        "2 SHO + offset, gap": GP(terms=[SHO(4e-4, 0.8), SHO(1.5e-4, 0.05)], err_scale=1.1, baseline="offset",
                                  gap=0.5),
        "Rotation + SHO + linear, gap (d=8)": GP(terms=[Rotation(5e-4, 1.5, 1.0, 2.0, 0.4), SHO(1e-4, 0.05)],
                                                err_scale=0.9, baseline="linear", gap=0.5),
    }


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


def test_dense_general():
    """the general kernels (any Q, sums of terms, Rotation, baselines, segments): moon and planet model = dense"""
    from pandoramoon.gp import pandora_loglike_planet_gp
    rng = np.random.default_rng(13)
    for ss in (1, 3):
        tc, time_, data, yerr, _ = dataset(ss, rng)
        valid = np.isfinite(data)
        for name, g in general_gps().items():
            L = pandora.Likelihood(time_, data, yerr, epoch_distance=P + 0.01, supersampling_factor=ss,
                                   hill_sphere_threshold=1.2, gp=g)
            C = dense_cov(g, tc) + np.diag((g.err_scale * yerr) ** 2)
            Lv = np.linalg.cholesky(C[np.ix_(valid, valid)])
            ld = np.sum(np.log(np.diag(Lv))) - np.sum(np.log(yerr[valid]))
            worst = 0.0
            for _ in range(6):
                p = dict(BASE, t0_bary_offset=rng.uniform(-0.05, 0.05), r_planet=rng.uniform(0.05, 0.12),
                         tau_moon=rng.uniform(0, 1), Omega_moon=rng.uniform(-30, 30), per_moon=rng.uniform(0.8, 2.0))
                _, _, fm = pandora.pandora_flux(*p.values(), P + 0.01, ss, 0.0, 1.2, time_)
                z = np.linalg.solve(Lv, (data - np.asarray(fm))[valid])
                worst = max(worst, abs(L(*p.values()) - (-0.5 * z @ z - ld)))
            pp = list(BASE.values())[:12]
            _, _, fp = pandora.pandora_flux(*pp, 0.0, 1e-6, 0, 0, 0, 0, 0, 0, P + 0.01, ss, 0.0, 1.2, time_)
            z = np.linalg.solve(Lv, (data - np.asarray(fp))[valid])
            got = pandora_loglike_planet_gp(*pp, P + 0.01, ss, 0.0, 1.2, L.time, *L.gp_tables, None, True)
            worst = max(worst, abs(got - (-0.5 * z @ z - ld)))
            check(f"{name}: moon + planet model = dense algebra (supersampling {ss}, d={g.dim})", worst < 1e-6,
                  f"max error {worst:.1e}")


def test_predict_sample_general():
    """GP.predict = the dense conditional mean / sd; GP.sample has the kernel's covariance (general kernels)"""
    rng = np.random.default_rng(17)
    t = np.sort(np.concatenate([rng.uniform(0, 2, 200), rng.uniform(4, 6, 200)]))
    yerr = np.full(len(t), 2e-4)
    r = 1e-3 * np.sin(2 * t) + rng.normal(0, 2e-4, len(t))
    mask = (t > 0.8) & (t < 1.1)
    gps = general_gps()
    worst = 0.0
    for g in [gps["SHO Q=3"], gps["SHO + linear, gap"], gps["Rotation + SHO + linear, gap (d=8)"]]:
        mean, sd = g.predict(t, r, yerr, mask=mask)
        K = dense_cov(g, t)
        o = ~mask
        C = K[np.ix_(o, o)] + np.diag((g.err_scale * yerr[o]) ** 2)
        A = np.linalg.solve(C, K[o]).T
        dm = A @ r[o]
        dv = np.diag(K) - np.sum(A * K[:, o], 1)
        worst = max(worst, np.max(np.abs(mean - dm)) / 1e-3, np.max(np.abs(sd - np.sqrt(dv))) / 1e-3)
    check("GP.predict (general) = dense conditional mean and sd", worst < 1e-6, f"max error / 1000 ppm {worst:.1e}")
    from pandoramoon.gp import SHO, Rotation
    ts = np.arange(0, 3000, 0.05)
    worst = 0.0
    for term in (SHO(1e-3, 0.4, 3.0), SHO(1e-3, 0.4, 0.3), Rotation(1e-3, 2.0, 1.0, 1.0, 0.5)):
        xs = GP(terms=[term]).sample(ts, 3)
        lags = np.array([0, 4, 10, 20])
        acf = np.array([np.mean(xs[:len(xs) - k] * xs[k:]) for k in lags])
        worst = max(worst, np.max(np.abs(acf - term.kernel(lags * 0.05))) / 1e-6)
    check("GP.sample (Q != 1/sqrt 2, Rotation) has the kernel's covariance", worst < 0.08, f"max {worst:.3f}")


def test_fit_general():
    """GP.fit of the general kernels recovers the hyperparameters of draws of them"""
    from pandoramoon.gp import SHO, Rotation
    rng = np.random.default_rng(21)
    t = np.arange(0, 60, 1 / 48)
    tq = GP(4e-4, 0.5, Q=4.0)
    y = 1 + tq.sample(t, rng) + rng.normal(0, 2e-4, len(t))
    g = GP.fit(t, y, 2e-4, kernel="sho_q", bin_width=None)
    ok = abs(g.amp / 4e-4 - 1) < 0.3 and abs(g.tau / 0.5 - 1) < 0.2 and abs(g.Q / 4 - 1) < 0.35
    check("GP.fit(kernel='sho_q') recovers amp, tau, Q", ok, repr(g))
    tr = GP(terms=[Rotation(1e-3, 3.3, 2.0, 3.0, 0.5), SHO(1.5e-4, 0.05)])
    y = 1 + tr.sample(t, rng) + rng.normal(0, 2e-4, len(t))
    g = GP.fit(t, y, 2e-4, kernel="rotation", bin_width=None)
    rot = g.terms[0]
    ok = abs(rot.period / 3.3 - 1) < 0.05 and abs(rot.amp / 1e-3 - 1) < 0.35
    check("GP.fit(kernel='rotation') recovers the period and amplitude (periodogram start)", ok, repr(g))
    off = np.where(t > 30, 3e-3, -2e-3)
    t2 = np.concatenate([t[t < 29], t[t > 31]])
    y2 = 1 + GP(4e-4, 0.5).sample(t2, rng) + np.interp(t2, t, off) + rng.normal(0, 2e-4, len(t2))
    g = GP.fit(t2, y2, 2e-4, baseline="offset", gap=1.0, bin_width=None)
    ok = abs(g.amp / 4e-4 - 1) < 0.3 and abs(g.tau / 0.5 - 1) < 0.3
    g0 = GP.fit(t2, y2, 2e-4, bin_width=None)
    check("GP.fit(baseline='offset', gap): segment offsets do not inflate amp / tau", ok,
          f"{g} (without baseline: {g0})")


def test_outliers():
    """flag_outliers: injected flares (above the trend, also inside the transit mask) and outliers are flagged,
    a transit dip inside the mask is not, few false flags"""
    rng = np.random.default_rng(23)
    t = np.arange(0, 20, 1 / 48)
    g = GP(5e-4, 0.4)
    y = 1 + g.sample(t, rng) + rng.normal(0, 1e-4, len(t))
    mask = np.abs(t - 10) < 0.5                        # the transit window with the moon's reach
    y[np.abs(t - 10) < 0.15] -= 3e-3                   # the planet's transit
    fl = []
    for t0 in (3.0, 7.0, 10.3, 15.0):                  # flares: 2.5e-3 peak, 1-h decay; one in the mask (moon reach)
        k = np.argmin(np.abs(t - t0))
        y[k:k + 8] += 2.5e-3 * np.exp(-np.arange(8) / 2.0)
        fl.append(k)
    out_lo = [100, 600]
    y[out_lo] -= 1.5e-3                                # single low outliers outside the mask
    bad = g.flag_outliers(t, y, 1e-4, mask=mask)
    caught = all(bad[k] and bad[k + 1] for k in fl) and all(bad[k] for k in out_lo)
    inj = np.zeros(len(t), bool)
    for k in fl:
        inj[k:k + 8] = True
    inj[out_lo] = True
    false = int(np.sum(bad & ~inj))
    check("flag_outliers: flares (incl. inside the transit mask) and low outliers flagged", caught,
          f"{int(np.sum(bad & inj))} of {int(inj.sum())} injected cadences flagged")
    check("flag_outliers: no transit cadence and <= 2 others falsely flagged", false <= 2 and not np.any(
        bad & (np.abs(t - 10) < 0.15)), f"{false} false flags")
    gf = GP.fit(t, y, 1e-4, mask=mask, clip=True)
    check("GP.fit(clip=True): returns the flagged points as .outliers", bool(np.all(gf.outliers[fl])),
          f"{gf}, {int(gf.outliers.sum())} flagged")


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
    ts = np.arange(0, 2000, 0.02)
    xs = GP(1e-3, 0.3).sample(ts, 1)
    acf = [np.mean(xs[:len(xs) - k] * xs[k:]) for k in (0, 5, 15)]
    ref = sho_k(np.array([0.0, 0.1, 0.3]), 1e-3, 0.3)
    err = np.max(np.abs(np.array(acf) - ref)) / 1e-6
    check("GP.sample has the SHO covariance", err < 0.08, f"max |acf - k| / amp^2 {err:.3f}")


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


def test_residual_scatter():
    """search.residual_scatter (NestedSampler's noise-model warning): ~1 for the GP and for white noise, >> 1 for
    the biweight-flattened trend case with white-noise errors"""
    import gp_trend_case as G
    from pandoramoon.search import residual_scatter
    from pandoramoon.sampling import PARAMS
    c = G.make(moon=True, seed=0)
    p = {k: float(c["truth"].get(k, 0.0)) for k in PARAMS}
    flat, _ = G.biweight(c, 1.5)
    clean = G.make(moon=True, seed=0, trend=dict(a1_ppm=0.0, a2_ppm=0.0, sho_ppm=0.0))["flux"]
    out = {}
    for name, f, g in (("biweight", flat, None), ("GP", c["flux"], GP.fit(c["time"], c["flux"], c["yerr"],
                                                                         mask=c["mask"])), ("white", clean, None)):
        L = pandora.Likelihood(c["time"], f, c["yerr"], epoch_distance=c["epoch_distance"], hill_sphere_threshold=1.0,
                               gp=g)
        out[name] = residual_scatter(L, p)[0]
    check("residual_scatter: GP and white noise ~1, biweight with white errors >> 1",
          out["biweight"] > 5 and abs(out["GP"] - 1) < 0.15 and abs(out["white"] - 1) < 0.15,
          ", ".join(f"{k} {v:.2f}" for k, v in out.items()))


def test_gpu():
    from pandoramoon import gpu
    if not gpu.available():
        check("GPU available", False, "no GPU / CUDA packages")
        return
    import gp_trend_case as G
    from pandoramoon.gpu.sampler import PandoraEvaluator
    from numba import cuda
    from pandoramoon.gp import SHO, Rotation
    c = G.make(moon=True, seed=0)
    gps = [("SHO", 1, GP.fit(c["time"], c["flux"], c["yerr"], mask=c["mask"])),
           ("SHO", 3, GP.fit(c["time"], c["flux"], c["yerr"], mask=c["mask"])),
           ("SHO Q=3, gap (d=2)", 1, GP(1e-3, 0.5, Q=3.0, gap=0.5)),
           ("SHO + offset, gap (d=3)", 3, GP.fit(c["time"], c["flux"], c["yerr"], mask=c["mask"], baseline="offset",
                                                 gap=0.5)),
           ("2 SHO + linear, gap (d=6)", 1, GP.fit(c["time"], c["flux"], c["yerr"], mask=c["mask"], kernel="sho2",
                                                   baseline="linear", gap=0.5)),
           ("Rotation + SHO + linear, gap (d=8)", 3,
            GP(terms=[Rotation(1.2e-3, 6.0, 1.0, 1.0, 0.4), SHO(2e-4, 0.15)], baseline="linear", gap=0.5))]
    for name, ss, g in gps:
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
        check(f"GPU GP likelihood = CPU ({name}, supersampling {ss}, {fin.sum()} points)", np.max(rel) < 1e-5,
              f"max |GPU - CPU| / |logL| {np.max(rel):.1e}, median {np.median(rel):.1e}")
        inv = ~np.isfinite(ref)
        check(f"GPU: -inf where the CPU has -inf ({name}, supersampling {ss})", bool(np.all(got[inv] == -np.inf)))
        # near the posterior (the truth, perturbed): absolute error in nats
        tr = c["truth"]
        from pandoramoon.sampling import PARAMS as names
        base = np.array([float(tr.get(p, getattr(pandora.model_params(), p, 0.0))) for p in names])
        TH = np.repeat(base[None, :], 256, 0)
        r_ = np.random.default_rng(3)
        for j, p in enumerate(names):
            if p in ("t0_bary_offset",):
                TH[:, j] += r_.normal(0, 2e-3, 256)
            elif p in ("r_moon", "r_planet"):
                TH[:, j] *= 1 + r_.normal(0, 0.01, 256)
        gl = ev.like(TH)
        cl = np.array([S.likelihood(*TH[i]) for i in range(len(TH))])
        err = np.abs(gl - cl)
        check(f"GPU = CPU near the truth ({name}, supersampling {ss})", np.max(err) < 0.02,
              f"|GPU - CPU| median {np.median(err):.1e}, max {np.max(err):.1e} nats")


if __name__ == "__main__":
    test_dense()
    test_dense_general()
    test_predict_sample_general()
    test_white_limit()
    test_fit()
    test_fit_general()
    test_outliers()
    test_residual_scatter()
    test_sampler()
    if "--gpu" in sys.argv:
        test_gpu()
    print(f"\n{len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
