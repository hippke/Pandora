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


def test_eclipse_dm():
    """eclipse_dm (moon deficit after the planet-moon overlap, T4 term included) against the CPU eclipse() on
    random overlaps of every kind (partial, planet inside the moon disk, moon behind the planet), on the star
    and on the limb, and at d = 0 (planet concentric with a larger moon disk: 0 / 0 gave a moon deficit of 0,
    Pandora bug #32). Exact tangencies are left out (FP32 lens area ill-conditioned; CPU limb er too, BUGS.md)."""
    from pandoramoon.eclipse import eclipse
    from pandoramoon.occult import occult_hybrid
    from pandoramoon.gpu.likelihood import eclipse_dm
    rng = np.random.default_rng(5)
    rows, kind = [], []
    while len(rows) < 12000:
        rp, rm = np.exp(rng.uniform(np.log(0.005), np.log(0.25), 2))
        zm = rng.uniform(0, 0.9) if rng.uniform() < 0.5 else rng.uniform(0.9, 1.05)
        ph, th = rng.uniform(0, 2 * np.pi, 2)
        xm, ym = zm * np.cos(ph), zm * np.sin(ph)
        k = int(rng.integers(0, 3))
        lo, hi = abs(rp - rm), rp + rm
        d = (rng.uniform(lo + 1e-4 * hi, hi * (1 - 1e-4)), rng.uniform(0, lo * (1 - 1e-4)), 0.0)[k]
        q1, q2 = rng.uniform(size=2)
        u1, u2 = 2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)
        dm = 1 - occult_hybrid(np.array([zm]), rm, u1, u2)[0]
        if dm > 0:
            rows.append((xm + d * np.cos(th), ym + d * np.sin(th), xm, ym, rp, rm, dm, u1, u2))
            kind.append(k)
    X = np.array(rows, dtype=np.float32).astype(np.float64)     # the CPU gets the GPU's (FP32) inputs
    kind = np.array(kind)
    g = eclipse_dm(X).astype(np.float64)
    a = np.array
    c = np.array([1 - eclipse(a([x[0]]), a([x[1]]), a([x[2]]), a([x[3]]), x[4], x[5], a([1 - x[6]]), x[7], x[8])[0]
                  for x in X])
    err = np.abs(g - c)
    msg = []
    for k, nm in enumerate(("partial", "inside", "d = 0")):
        e = err[kind == k].max()
        assert e <= 1e-6, f"{nm}: max |dm_gpu - dm_cpu| {e:.2e}"
        msg.append(f"{nm} {e:.1e}")
    d0 = (kind == 2) & (c > 0)          # d = 0 with a larger moon disk (else the moon is hidden: 0)
    assert d0.sum() > 1000 and np.all(g[d0] > 0.5 * c[d0]), "d = 0: moon deficit lost"
    return "max |dm_gpu - dm_cpu| " + ", ".join(msg)


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


def _system_like(per, epochs, cad_per_day, ss, t0, r_p, a_b, M_p, r_m, per_m, M_m, seed=1, noise=300e-6, window=None,
                 ecc_moon=0.0, w_moon=0.0):
    """a synthetic light curve (model + white noise) of a planet+moon system, cut to windows around the transits,
    and its supersampled Likelihood"""
    import pandoramoon as pandora
    p = pandora.model_params()
    p.R_star = 696_342_000; p.u1 = 0.45; p.u2 = 0.2
    p.per_bary = per; p.a_bary = a_b; p.r_planet = r_p; p.b_bary = 0.25; p.t0_bary = t0; p.t0_bary_offset = 0.003
    p.M_planet = M_p; p.w_bary = 0; p.ecc_bary = 0
    p.r_moon = r_m; p.per_moon = per_m; p.tau_moon = 0.3; p.Omega_moon = 10; p.i_moon = 88; p.ecc_moon = ecc_moon
    p.w_moon = w_moon; p.M_moon = M_m
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


def test_transform():
    """the prior transform (sampling._theta) on the GPU vs the CPU: default priors with moon-phase coordinates, and
    a prior with normal / log-uniform parameters and correlated limb darkening"""
    import pandoramoon as pandora
    from pandoramoon.sampling import _thetas, LOGZERO
    from pandoramoon.gpu.transform import GPUTransform
    import tutorial_data as td
    params, like, priors = td.make()
    pr2 = dict(priors)
    pr2["R_star"] = pandora.Normal(696_342_000, 3e7, lo=5e8, hi=9e8)
    pr2["M_planet"] = pandora.LogUniform(5e26, 5e27)
    del pr2["q1"], pr2["q2"]
    pr2["u1"] = pandora.Normal(0.41, 0.1)
    pr2["u2"] = pandora.Normal(0.26, 0.1)
    out = []
    for lab, pr, kw in (("default", priors, {}), ("normal / log-uniform, correlated LD", pr2,
                                                   dict(ld_correlation=0.5))):
        S = pandora.NestedSampler(like, pr, verbose=False, **kw)
        m = S.models["moon"]
        U = np.random.default_rng(5).uniform(size=(50000, m.ndim))
        th_c, lw_c = _thetas(U, m.D)
        th_g, lw_g = GPUTransform(m)(U)
        okc = lw_c > LOGZERO
        assert np.array_equal(okc, lw_g > LOGZERO), (lab, "LOGZERO pattern differs")
        # relative, and absolute below 1 (tau_moon = x - floor(x) of x ~ 100: FMA contraction differs at 1e-14)
        dth = np.max(np.abs(th_g[okc] - th_c[okc]) / np.maximum(1.0, np.abs(th_c[okc])))
        dlw = np.max(np.abs(lw_g[okc] - lw_c[okc]))
        assert dth <= 1e-12 and dlw <= 1e-10, (lab, dth, dlw)
        out.append(f"{lab}: {okc.mean():.3f} inside, max rel |d theta| {dth:.1e}, max |d lw| {dlw:.1e}")
    return "; ".join(out)


def test_eccentric_moon():
    """eccentric moon orbits (e = 0.1 ... 0.9, several w): 3 transits of a Jupiter with a moon, 10-min cadence, ss 3;
    points near the truth with e and w perturbed too"""
    from pandoramoon.gpu import GPULikelihood
    import tutorial_data as td
    out = []
    worst = 0.0
    for e, w in ((0.1, 30.0), (0.3, 120.0), (0.6, 250.0), (0.9, 75.0)):
        p, like = _system_like(365.25, 3, 144, 3, 11.0, 0.1, 215.0, 1.9e27, 0.03, 8.0, 1e26, window=3.0,
                               ecc_moon=e, w_moon=w)
        rng = np.random.default_rng(7)
        th0 = td.truth_theta(p)
        TH = np.repeat(th0[None, :], 400, axis=0)
        sc = np.zeros(20)
        sc[10] = 0.003; sc[12] = 0.002; sc[13] = 0.05; sc[14] = 0.05; sc[15] = 5; sc[16] = 2; sc[19] = 2e25
        sc[17] = 0.02 * (1 - e); sc[18] = 10
        TH = TH + rng.normal(size=TH.shape) * sc
        TH[:, 17] = np.clip(TH[:, 17], 0.0, 0.9)
        TH[:, [12, 19]] = np.abs(TH[:, [12, 19]])
        TH[0] = th0
        G = GPULikelihood(like, t0_ref=p.t0_bary)
        ref = np.array([like(*th) for th in TH])
        g = G(TH)
        fin = np.isfinite(ref)
        assert np.array_equal(fin, np.isfinite(g)), (e, "-inf pattern differs")
        d, b = np.abs(g - ref)[fin], _bound(like, TH[fin])
        near = ref[fin] > ref[fin].max() - 50
        assert d[near].max() <= 0.02 and np.all(d <= b), (e, w, d[near].max(), np.max(d / b))
        worst = max(worst, np.max(d / b))
        out.append(f"e={e}: near max {d[near].max():.1e}")
    return "; ".join(out) + f"; max |dlogL| / bound {worst:.2f}"


def _ptx_f64(kernel="loglike", ss=5):
    """FP64 arithmetic instructions in the PTX of a model.cu kernel (its whole body, inlined device functions
    included)"""
    import re
    import cuda.core as cc
    from pandoramoon.gpu import _cuda
    from pandoramoon import occult as O
    import math as m_
    from pandoramoon.constants import G, DAY
    dev = cc.Device()
    dev.set_current()
    cap = dev.compute_capability
    defs = [("SS", str(ss)), ("WARPS", "4"), ("MINB", "8"),
            ("LY0", f"{float(np.float32(O.LIMB_TAB_Y0))!r}f"), ("LSC", f"{float(np.float32(O.LIMB_TAB_SCALE))!r}f"),
            ("IY0", f"{float(np.float32(O.INT_TAB_Y0))!r}f"), ("ISC", f"{float(np.float32(O.INT_TAB_SCALE))!r}f"),
            ("GDAY", f"{G * DAY ** 2 / (4 * m_.pi ** 2)!r}f")]
    opts = cc.ProgramOptions(arch=f"sm_{cap.major}{cap.minor}", use_fast_math=True, define_macro=defs, std="c++17")
    ptx = cc.Program(_cuda._source(), code_type="c++", options=opts).compile("ptx").code
    ptx = ptx.decode() if isinstance(ptx, bytes) else ptx
    a = ptx.index(f".entry {kernel}(")
    b = ptx.find(".entry ", a + 1)
    body = ptx[a:b if b > 0 else len(ptx)]
    ops = re.findall(r"^\s*(?:@%p\d+\s+)?((?:add|sub|mul|fma|div|rcp|sqrt|neg|abs|min|max|setp|cvt)[\w.]*\.f64)\b",
                     body, flags=re.M)
    return len(ops), body


# FP64 arithmetic in kernel 1 (`loglike`): validity comparisons, the reference offsets of the epoch and the moon
# phase, the conversions of the 20 parameters and the warp reduction of chi^2 (GPU_MODEL.md). Measured 2026-11-08:
# 63 (21 conversions, 22 comparisons, 20 add / sub / mul / fma / rcp / floor). Many more mean FP64 slipped into the
# per-sample code (e.g. a float literal without f).
F64_LOGLIKE_MAX = 80


def test_fp64_lint():
    n, body = _ptx_f64("loglike")
    assert n <= F64_LOGLIKE_MAX, (n, "FP64 instructions in loglike: FP64 in the sample loop?")
    return f"{n} FP64 arithmetic instructions in loglike (limit {F64_LOGLIKE_MAX})"


TESTS = [test_fp64_lint, test_deficit, test_er3, test_eclipse_dm, test_loglike_tutorial, test_long_kepler, test_long_baseline, test_edge_cases,
         test_transform, test_eccentric_moon]

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
