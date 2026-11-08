"""Component tests of the FP32 GPU model (pandoramoon/gpu/model.cu) against the FP64 CPU model.

    python tests/gpu/test_model.py        # skipped (exit 0) without a GPU

Budgets (GPU_PLAN.md section 6): deficit per sample <= 1 ppm (measured ~1e-8), the eclipse ratio on the deficit
scale <= 1e-7, logL at posterior-like points |dlogL| <= 0.02 nats, the -inf pattern identical.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _gpu():
    from pandoramoon import gpu
    return gpu.available()


def test_deficit():
    from pandoramoon import occult as O
    from pandoramoon.gpu.likelihood import deficit
    worst = 0.0
    for k in (0.0006, 0.002, 0.01, 0.03, 0.07, 0.1, 0.15, 0.2, 0.25):
        z = np.concatenate([np.linspace(0, 1 + k + 0.01, 3001),
                            1 - k + k * np.logspace(-6, 0, 200), 1 - k - k * np.logspace(-6, 0, 200),
                            1 + k - k * np.logspace(-6, 0, 200), [k, 1.0]])
        z = z[z >= 0]
        for u1, u2 in ((0.4, 0.26), (0.0, 0.0), (1.0, 0.0), (0.0737, 0.8973), (0.8, -0.1), (0.2, 0.6)):
            ex = 1 - O.occult_exact(z, k, u1, u2)
            g = deficit(k, u1, u2, z * z).astype(np.float64)
            d = np.abs(g - ex).max()
            worst = max(worst, d)
            assert d <= 1e-7, (k, u1, u2, d, z[np.argmax(np.abs(g - ex))])
    return f"max |D_gpu - D_exact| {worst:.2e}"


def test_er3():
    from pandoramoon.overlap_analytic import er_fast, _lens_area
    from pandoramoon.gpu.likelihood import er3
    import area3_ref as A
    X = A.fuzz(20000, seed=7)
    ref = np.array([er_fast(*x) for x in X])
    g = er3(X).astype(np.float64)
    ok = np.isfinite(ref)
    a_sm = np.array([_lens_area(1.0, x[5], math.hypot(x[3], x[4])) for x in X])
    dd = np.abs(g - ref)[ok] * a_sm[ok] / np.pi            # on the scale of the moon deficit
    assert dd.max() <= 1e-7, dd.max()
    return f"deficit-scale max {dd.max():.2e}, |d er| 99.9% {np.quantile(np.abs(g - ref)[ok], 0.999):.2e}"


def test_loglike_tutorial():
    import pandoramoon as pandora
    from pandoramoon.sampling import _thetas, LOGZERO
    from pandoramoon.gpu import GPULikelihood
    import tutorial_data as td
    params, like, priors = td.make()
    S = pandora.NestedSampler(like, priors, verbose=False)
    L = S.likelihood
    m = S.models["moon"]
    rng = np.random.default_rng(3)
    # posterior-like points: the truth with small perturbations in the unit cube of the sampler's coordinates
    U = rng.uniform(size=(20000, m.ndim))
    TH_prior, lw = _thetas(U, m.D)
    TH_prior = TH_prior[lw > LOGZERO]
    truth = td.truth_theta(params)
    TH_post = np.repeat(truth[None, :], 20000, axis=0)
    scale = np.zeros(20)
    scale[[0, 1]] = 0.05                    # u1, u2
    scale[3] = 0.005                        # per_bary [d]
    scale[4] = 5.0                          # a_bary
    scale[5] = 0.002                        # r_planet
    scale[6] = 0.1                          # b_bary
    scale[10] = 0.01                        # t0_bary_offset [d]
    scale[11] = 3e26                        # M_planet
    scale[12] = 0.002                       # r_moon
    scale[13] = 0.5                         # per_moon [d]
    scale[14] = 0.3                         # tau_moon
    scale[15] = 20.0                        # Omega_moon [deg]
    scale[16] = 3.0                         # i_moon [deg]
    scale[19] = 1e26                        # M_moon
    TH_post = TH_post + rng.normal(size=TH_post.shape) * scale
    TH_post[:, [5, 12, 19]] = np.abs(TH_post[:, [5, 12, 19]])
    G = GPULikelihood(L, t0_ref=params.t0_bary)
    out = []
    for name, TH, tol_abs, tol_rel in (("prior", TH_prior, None, 1e-5), ("posterior-like", TH_post, 0.02, None)):
        TH = np.ascontiguousarray(TH)
        ref = np.array([L(*th) for th in TH])
        g, nl = G(TH, return_nlimb=True)
        fin = np.isfinite(ref)
        assert np.array_equal(fin, np.isfinite(g)), f"{name}: -inf pattern differs"
        d = np.abs(g - ref)[fin]
        if tol_abs is not None:
            assert d.max() <= tol_abs, (name, d.max())
        if tol_rel is not None:
            rel = d / np.maximum(1.0, np.abs(ref[fin]))
            assert rel.max() <= tol_rel, (name, rel.max())
        out.append(f"{name}: n={fin.sum()}, max |dlogL| {d.max():.2e}, 99.9% {np.quantile(d, 0.999):.2e}, "
                   f"calls with limb-eclipse samples {np.mean(nl > 0):.3f}")
    return "; ".join(out)


def _system_like(per, epochs, cad_per_day, ss, t0, r_p, a_b, M_p, r_m, per_m, M_m, seed=1, noise=300e-6, window=None):
    """a synthetic light curve (model + white noise) of a planet+moon system, cut to windows around the transits,
    and its supersampled Likelihood"""
    import pandoramoon as pandora
    p = pandora.model_params()
    p.R_star = 696_342_000; p.u1 = 0.45; p.u2 = 0.2
    p.per_bary = per; p.a_bary = a_b; p.r_planet = r_p; p.b_bary = 0.25; p.t0_bary = t0; p.t0_bary_offset = 0.003
    p.M_planet = M_p; p.w_bary = 0; p.ecc_bary = 0
    p.r_moon = r_m; p.per_moon = per_m; p.tau_moon = 0.3; p.Omega_moon = 10; p.i_moon = 88; p.ecc_moon = 0
    p.w_moon = 0; p.M_moon = M_m
    p.epochs = epochs; p.epoch_duration = window or 2.0; p.cadences_per_day = cad_per_day; p.epoch_distance = per
    p.occult_small_threshold = 0; p.hill_sphere_threshold = 1.0
    p.supersampling_factor = ss
    tg = pandora.time(p).grid()
    f, _, _ = pandora.moon_model(p).light_curve(tg)
    rng = np.random.default_rng(seed)
    data = f + rng.normal(0, noise, len(f))
    like = pandora.Likelihood(tg, data, np.full(len(data), noise), epoch_distance=per, supersampling_factor=ss,
                              hill_sphere_threshold=1.0)
    return p, like


def _compare(p, like, n=1500, seed=2):
    from pandoramoon.gpu import GPULikelihood
    import tutorial_data as td
    rng = np.random.default_rng(seed)
    th0 = td.truth_theta(p)
    TH = np.repeat(th0[None, :], n, axis=0)
    sc = np.zeros(20)
    sc[[0, 1]] = 0.03; sc[3] = 2e-5 * p.per_bary; sc[4] = 0.02 * p.a_bary; sc[5] = 0.02 * p.r_planet; sc[6] = 0.1
    sc[10] = 0.003; sc[11] = 0.1 * p.M_planet; sc[12] = 0.1 * p.r_moon; sc[13] = 0.02 * p.per_moon; sc[14] = 0.2
    sc[15] = 20; sc[16] = 3; sc[19] = 0.3 * p.M_moon
    f = np.where(np.arange(n) < n // 2, 1.0, 0.05)[:, None]      # half wide, half narrow (near the maximum)
    TH = TH + rng.normal(size=TH.shape) * sc * f
    TH[:, [5, 12, 19]] = np.abs(TH[:, [5, 12, 19]])
    TH[0] = th0
    G = GPULikelihood(like, t0_ref=p.t0_bary)
    ref = np.array([like(*th) for th in TH])
    g = G(TH)
    fin = np.isfinite(ref)
    assert np.array_equal(fin, np.isfinite(g)), "-inf pattern differs"
    d = np.abs(g - ref)[fin]
    return d, ref[fin], len(G.te), TH[fin]


DEFICIT_TOL = 3e-8      # GPU vs CPU deficit per sample: the FP32 position floor near the limb (GPU_MODEL.md 3.4)


def _bound(like, TH):
    """per point: the logL change that a deficit perturbation of DEFICIT_TOL in every in-transit cadence can cause,
    sum over cadences with a model deficit of IV |r| DEFICIT_TOL (r the CPU residual), plus 0.01 nats"""
    import pandoramoon as pandora
    e = 1.0 - like.data
    iv = like.inv_err ** 2
    out = np.empty(len(TH))
    for i, th in enumerate(TH):
        _, _, ft = pandora.pandora_flux(*th, like.epoch_distance, like.supersampling_factor,
                                        like.occult_small_threshold, like.hill_sphere_threshold, like.time)
        dm = 1.0 - ft
        on = dm > 0
        out[i] = 0.01 + DEFICIT_TOL * np.sum(iv[on] * np.abs(dm[on] - e[on]) + 0.5 * iv[on] * DEFICIT_TOL)
    return out


def _check_long(like, TH, d, ref):
    b = _bound(like, TH)
    lmax = ref.max()
    near = ref > lmax - 50
    assert d[near].max() <= 0.02, ("near the maximum", d[near].max())
    bad = d > b
    assert not bad.any(), ("beyond the deficit-tolerance bound", d[bad].max(), b[bad][np.argmax(d[bad])])
    return (f"near the maximum (n={near.sum()}): max |dlogL| {d[near].max():.2e}; all: max {d.max():.2e}, "
            f"max |dlogL| / bound {np.max(d / b):.2f}")


def test_long_kepler():
    """Kepler-like: P = 10 d, 146 transits over 4 years, 30-min cadence, supersampling 5 (epochs far from TE_0)"""
    p, like = _system_like(10.0, 146, 48, 5, 5.0, 0.05, 20.0, 1.0e26, 0.02, 1.5, 3e24)
    d, ref, ne, TH = _compare(p, like)
    return f"{ne} epochs, {like.inv_var.size} cadences: " + _check_long(like, TH, d, ref)


def test_long_baseline():
    """long baseline and fast moon: P = 300 d, 12 transits over 10 years, P_moon = 0.6 d (~6000 moon orbits from
    TE_0 to the last epoch), 2-min cadence, no supersampling"""
    p, like = _system_like(300.0, 12, 720, 1, 100.0, 0.08, 180.0, 1.9e27, 0.025, 0.6, 2e25, window=1.0)
    d, ref, ne, TH = _compare(p, like)
    return f"{ne} epochs, {like.inv_var.size} cadences: " + _check_long(like, TH, d, ref)


def test_edge_cases():
    """validity (-inf), unphysical moons (Hill sphere, inside the planet), no moon, grazing / missing planet,
    retrograde and face-on moons, a moon on the planet (i = 90, Omega = 0), parameters at the prior edges"""
    import pandoramoon as pandora
    from pandoramoon.gpu import GPULikelihood
    import tutorial_data as td
    params, like, priors = td.make()
    L = pandora.NestedSampler(like, priors, verbose=False).likelihood
    G = GPULikelihood(L, t0_ref=params.t0_bary)
    t0 = td.truth_theta(params)
    cases = []

    def add(label, **kw):
        th = t0.copy()
        names = ["u1", "u2", "R_star", "per_bary", "a_bary", "r_planet", "b_bary", "w_bary", "ecc_bary", "t0_bary",
                 "t0_bary_offset", "M_planet", "r_moon", "per_moon", "tau_moon", "Omega_moon", "i_moon", "ecc_moon",
                 "w_moon", "M_moon"]
        for k_, v in kw.items():
            th[names.index(k_)] = v
        cases.append((label, th))
    add("truth")
    add("u1 < 0 (invalid)", u1=-0.1)
    add("u1 + u2 > 1 (invalid)", u1=0.7, u2=0.5)
    add("NaN r_moon (invalid)", r_moon=np.nan)
    add("inf per_moon (invalid)", per_moon=np.inf)
    add("per_moon <= 0 (invalid)", per_moon=0.0)
    add("M_moon < 0 (invalid)", M_moon=-1.0)
    add("beyond the Hill sphere", per_moon=60.0)
    add("orbit inside the planet", per_moon=0.05)
    add("no moon (r_moon = 0)", r_moon=0.0, M_moon=0.0)
    add("grazing planet b = 1.05", b_bary=1.05)
    add("no transit b = 1.5", b_bary=1.5)
    add("retrograde moon i = 150", i_moon=150.0)
    add("face-on moon i = 0", i_moon=0.0)
    add("i = 90, Omega = 0 (eclipses)", i_moon=90.0, Omega_moon=0.0, tau_moon=0.0)
    add("i = 90, Omega = 90", i_moon=90.0, Omega_moon=90.0)
    add("large moon r_moon = 0.07", r_moon=0.07)
    add("heavy moon M_moon = M_planet / 2", M_moon=t0[11] / 2)
    add("large offset t0_bary_offset = 0.05", t0_bary_offset=0.05)
    add("tiny moon r_moon = 0.001", r_moon=0.001)
    TH = np.array([c[1] for c in cases])
    ref = np.array([L(*th) for th in TH])
    g = G(TH, check=False)
    msg = []
    for (lab, _), a, b in zip(cases, ref, g):
        if not np.isfinite(a):
            assert not np.isfinite(b) and b < 0, (lab, a, b)
        else:
            assert abs(a - b) <= 0.02 + 1e-6 * abs(a), (lab, a, b)
            msg.append(abs(a - b))
    return f"{len(cases)} cases ({np.sum(~np.isfinite(ref))} invalid): max |dlogL| {max(msg):.2e}"


TESTS = [test_deficit, test_er3, test_loglike_tutorial, test_long_kepler, test_long_baseline, test_edge_cases]

if __name__ == "__main__":
    if not _gpu():
        print("SKIP: no usable GPU")
        sys.exit(0)
    bad = 0
    for t in TESTS:
        try:
            msg = t()
            print(f"PASS  {t.__name__}: {msg}", flush=True)
        except AssertionError as ex:
            bad += 1
            print(f"FAIL  {t.__name__}: {ex}", flush=True)
    sys.exit(1 if bad else 0)
