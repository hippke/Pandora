"""Unit tests for Pandora bug fixes (see ../BUGS.md).

These complement the light-curve regression cases in cases.py: they target
failure modes that realistic light curves cannot reliably expose (exact
boundary points, the plausibility check, the video renderer) and pin each
fixed bug with a direct check. No stored reference is needed.

    python unit_tests.py         # standalone; exit code 0 = pass
    pytest unit_tests.py         # if pytest is installed
"""
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")  # silence video() progress bars

import numpy as np

import cases as C  # sets up the Pandora import path
from pandoramoon.pandora import pandora as pandora_fn
from pandoramoon.occult import (occult, occult_hybrid, occult_exact, occult_small, ellpicb,
                                create_occult_cache, read_occult_cache)
from pandoramoon.ellipse import ellipse, ellipse_ecc
from pandoramoon.grids import timegrid, x_bary_grid
from pandoramoon.helpers import resample, ld_convert, ld_invert

U1, U2 = C.LD_SUN


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def ref_flux(z, k, u1=U1, u2=U2, n=200000):
    """Independent occultation reference: radial integration of the
    quadratic limb-darkened disk (midpoint rule, ~0.01-0.05 ppm)."""
    r = (np.arange(n) + 0.5) / n
    mu = np.sqrt(1 - r * r)
    intensity = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    if z == 0:
        theta = np.where(r < k, 2 * np.pi, 0.0)
    else:
        c = (r * r + z * z - k * k) / (2 * r * z)
        theta = np.where(c <= -1, 2 * np.pi, np.where(c >= 1, 0.0, 2 * np.arccos(np.clip(c, -1, 1))))
    blocked = np.sum(intensity * r * theta) / n
    return 1 - blocked / (np.pi * (1 - u1 / 3 - u2 / 6))


def kepler_E_bisect(M, e):
    M = np.asarray(M, float) % (2 * np.pi)
    lo = np.maximum(M - e - 0.1, 0.0)
    hi = M + e + 0.1
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f = mid - e * np.sin(mid) - M
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)


def _vector_double_loops(ir):
    """Number of vectorised loops with double arithmetic in LLVM IR: basic
    blocks labelled vector.body* that hold an fmul / fadd / fsub / fdiv on
    <N x double> (a vector.body alone can be an integer count loop, P18)"""
    import re
    n = 0
    for blk in re.split(r"\n(?=[\w.\"-]+:)", ir):
        if re.match(r"\"?vector\.body", blk) and re.search(r"= f(mul|add|sub|div) [^\n]*<\d+ x double>", blk):
            n += 1
    return n


def _pandora_moon_kept(ecc_moon, per_moon, r_moon=0.035, i_moon=0.0):
    """Run pandora() for Sun/Jupiter + moon; True if the moon is kept
    (i.e. the orbit passed the plausibility check)."""
    t = np.linspace(10.7, 11.3, 50)
    out = pandora_fn(U1, U2, C.R_SUN_M, 365.25, 215.0, 0.1, 0.3, 0.0, 0.0, 11.0, 0.0,
                     C.M_JUPITER, r_moon, per_moon, 0.07, 0.0, i_moon, ecc_moon, 0.0,
                     0.05395 * C.M_JUPITER, 365.26, 1, 0.01, 1.2, t)
    xm = out[5]
    return bool(np.all(np.abs(xm) < 1e3))


# ----------------------------------------------------------------------------
# second-pass bugs #20, #21, #23, #24
# ----------------------------------------------------------------------------
def test_occult_finite_at_boundaries():
    """Bug #20 (and #4, #5, #6): no NaN/inf/exception at the exact geometric
    boundaries z = 1-k, z = k, z = k-1 and their float neighbourhoods."""
    bad = []
    for k in np.round(np.arange(0.005, 1.0, 0.005), 3):
        zs = [1 - k, k, k - 1e-6, k + 1e-6, 1 - k - 1e-9, 1 - k + 1e-9,
              np.nextafter(1 - k, 0), np.nextafter(1 - k, 2), 1 + k - 1e-12]
        zs = np.array([z for z in zs if z >= 0])
        for fn in (occult, occult_hybrid):
            try:
                f = fn(zs.copy(), float(k), U1, U2)
                if not np.all(np.isfinite(f)):
                    bad.append((fn.__name__, float(k), zs[~np.isfinite(f)].tolist()))
            except Exception as e:
                bad.append((fn.__name__, float(k), type(e).__name__))
    for k in (1.2, 1.5):  # occulter larger than the star: z = k - 1 boundary
        for fn in (occult, occult_hybrid):
            f = fn(np.array([k - 1.0, k - 1.0 - 1e-9]), k, U1, U2)
            if not np.all(np.isfinite(f)):
                bad.append((fn.__name__, k, "z=k-1"))
    assert not bad, f"{len(bad)} failures, e.g. {bad[:3]}"


def test_occult_first_contact_is_unity():
    """Bug #25: at exactly z = 1+k (first/last contact) the flux is 1.
    Previously occult returned uninitialised memory there (non-deterministic,
    seen as -5e98) and occult_hybrid the previous element's limb-darkening
    terms (e.g. 0.960 for k=0.3)."""
    for k in (0.01, 0.05, 0.1, 0.3, 0.6):
        z1 = np.float64(1 + k)
        for fn in (occult, occult_hybrid):
            for prev in (0.0, 0.5, 1 - k, 1 + k - 1e-3):   # vary the preceding element
                np.random.uniform(-5, 5, 1000)               # dirty the heap
                f = fn(np.array([prev, z1]), k, U1, U2)[1]
                assert f == 1.0, f"{fn.__name__}(z=1+k, k={k}) = {f!r} after z={prev}"


def test_occult_accuracy_near_limb_contact():
    """Bug #21: within 2e-4 of z = 1-k both functions must match an
    independent reference to < 1 ppm (the old 1e-4 special-case band gave up
    to 21 ppm and a jump)."""
    worst = 0.0
    for k in (0.05, 0.2, 0.45):
        for dz in (2e-4, 1.01e-4, 0.99e-4, 5e-5, 1e-6, 0.0, -5e-5):
            z = 1 - k - dz
            r = ref_flux(z, k)
            for fn in (occult, occult_hybrid):
                worst = max(worst, abs(fn(np.array([z]), k, U1, U2)[0] - r))
    assert worst < 1e-6, f"max error {worst * 1e6:.3f} ppm"


def test_plausibility_uses_periapsis_and_apoapsis():
    """Bug #23: moon eccentricity must enter the plausibility checks."""
    assert _pandora_moon_kept(0.0, 0.3), "circular close moon must be kept"
    assert _pandora_moon_kept(0.1, 0.3), "mildly eccentric moon must be kept"
    # e=0.9: periapsis a(1-e) = 0.019 R_star < r_planet + r_moon = 0.135
    assert not _pandora_moon_kept(0.9, 0.3), "moon passing through the planet was accepted"
    # P=212 d: a ~ 15 R_star, inside 1.2 r_Hill (~22 R_star) for e=0, outside for e=0.5
    assert _pandora_moon_kept(0.0, 212.0), "wide circular moon inside Hill limit must be kept"
    assert not _pandora_moon_kept(0.5, 212.0), "apoapsis outside the Hill limit was accepted"


def test_video_disk_full_size():
    """Bug #24: the limb-darkened stellar disk in video() has radius 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = C.build_cases()[0]
    model = C.pandora.moon_model(C._params(c))
    t = C.pandora.time(C._params(c)).grid()[:5]
    try:
        for n in (10, 100):
            plt.close("all")
            ani = model.video(time=t, limb_darkening=True, teff=5800, ld_circles=n)
            ax = plt.gcf().axes[0]
            radii = sorted(a.get_radius() for a in list(ax.patches) + list(ax.artists)
                           if isinstance(a, matplotlib.patches.Circle) and a.get_radius() <= 1.0)
            del ani
            assert len(radii) == n, f"{len(radii)} disk circles for ld_circles={n}"
            assert abs(radii[-1] - 1.0) < 1e-12, f"disk radius {radii[-1]} != 1 (ld_circles={n})"
            assert radii[0] > 0, "zero-radius circle drawn"
    finally:
        plt.close("all")   # unsaved animations + tqdm segfault at exit otherwise


# ----------------------------------------------------------------------------
# first-pass bugs
# ----------------------------------------------------------------------------
def test_resample():
    """Bugs #1, #17: full-group means, partial tail kept."""
    assert np.allclose(resample(np.arange(10.0), 5), [2.0, 7.0])
    assert np.allclose(resample(np.arange(6.0), 2), [0.5, 2.5, 4.5])
    assert np.allclose(resample(np.arange(11.0), 5), [2.0, 7.0, 10.0])


def test_eccentric_orbit_matches_kepler():
    """Bugs #2, #8: ellipse_ecc follows a reference Kepler orbit over the
    full orbit; periapsis direction = Omega + w; e->0 matches ellipse().
    P7b: also near periapsis (M down to 1e-9 on both sides) and for e up to
    0.995 (the vectorised solver up to ECC_FAST_MAX = 0.95, Markley's above);
    ellipse_ecc_z gives the same positions plus z^2 and vectorises."""
    a, per, tau, Om, w, inc, q = 0.1888, 0.3, 0.07, 30.0, 20.0, 80.0, 0.05395
    # P7b: also e at / above ECC_FAST_MAX (fast solver / Markley fallback)
    # and a small-M family on both sides of periapsis (M = 1e-9 ... 1e-2)
    near = per / (2 * np.pi) * np.logspace(-9, -2, 200)
    t = np.concatenate([np.linspace(0, 2 * per, 3000), tau * per + near, tau * per - near])
    M = (2 * np.pi / per) * (t - tau * per) % (2 * np.pi)
    for e in (0.1, 0.5, 0.9, 0.93, 0.95, 0.97, 0.995):
        E = kepler_E_bisect(M, e)
        nu = np.arctan2(np.sqrt(1 - e * e) * np.sin(E), np.cos(E) - e)
        wf = np.deg2rad(w) + nu
        r = (1 - e * np.cos(E)) * a / (1 + q)
        O, ii = np.deg2rad(Om), np.deg2rad(inc)
        refx = (np.cos(O) * np.cos(wf) - np.sin(O) * np.sin(wf) * np.cos(ii)) * r
        refy = (np.sin(O) * np.cos(wf) + np.cos(O) * np.sin(wf) * np.cos(ii)) * r
        xm, ym, _, _ = ellipse_ecc(a, per, e, tau, Om, w, inc, t, q, np.zeros(len(t)), 0.0)
        err = max(np.abs(xm - refx).max(), np.abs(ym - refy).max())
        assert err < 1e-12, f"e={e}: position error {err:.2e} R_star"
    xm, ym, _, _ = ellipse_ecc(a, per, 0.5, tau, Om, w, 0.0, np.array([tau * per]), q, np.zeros(1), 0.0)
    ang = np.degrees(np.arctan2(ym[0], xm[0]))
    assert abs(ang - (Om + w)) < 1e-6, f"periapsis angle {ang} != Omega+w"
    xe, ye, _, _ = ellipse_ecc(a, per, 1e-9, tau, Om, 0.0, inc, t, q, np.zeros(len(t)), 0.0)
    xc, yc, _, _ = ellipse(a, per, tau, Om, inc, t, np.zeros(len(t)), q, 0.0)
    assert max(np.abs(xe - xc).max(), np.abs(ye - yc).max()) < 1e-6
    # P7b: ellipse_ecc_z returns the same positions plus z^2, and its fast
    # solver loop (e <= ECC_FAST_MAX) vectorises (uncached copy)
    import numba
    from pandoramoon.ellipse import ellipse_ecc_z
    xb = np.linspace(-1, 1, len(t))
    for e in (0.3, 0.97):
        o = ellipse_ecc_z(a, per, e, tau, Om, w, inc, t, q, xb, 0.2)
        x4 = ellipse_ecc(a, per, e, tau, Om, w, inc, t, q, xb, 0.2)
        assert all(np.array_equal(o[k], x4[k]) for k in range(4))
        assert np.allclose(o[4], o[0] ** 2 + o[1] ** 2, rtol=1e-14, atol=0)
        assert np.allclose(o[5], o[2] ** 2 + o[3] ** 2, rtol=1e-14, atol=0)
    fresh = numba.njit(fastmath=True, error_model="numpy")(ellipse_ecc_z.py_func)
    fresh(a, per, 0.3, tau, Om, w, inc, t[:64], q, xb[:64], 0.2)
    ir = "".join(fresh.inspect_llvm().values())
    assert _vector_double_loops(ir) >= 2, "P7b: eccentric solver / position loops not vectorised"
    # P7b: the branch-free solver itself on a dense (e, M) grid up to its
    # cap, vs bisection: (cos E, sin E) within 5e-15 (measured 2.4e-15)
    from pandoramoon.ellipse import ECC_FAST_MAX, _kepler_cs_fast
    kcs = numba.njit(fastmath=True)(lambda M, e: _kepler_cs_fast(M, e))
    Mg = np.concatenate([np.logspace(-15, 0, 400), np.linspace(0, np.pi, 1501), np.pi - np.logspace(-15, 0, 200)])
    for e in np.concatenate([np.linspace(0, ECC_FAST_MAX, 39), [ECC_FAST_MAX]]):
        E = kepler_E_bisect(Mg, e)
        err = max(max(abs(kcs(m, e)[0] - np.cos(x)), abs(kcs(m, e)[1] - np.sin(x))) for m, x in zip(Mg, E))
        assert err < 5e-15, f"P7b solver: e={e}: {err:.1e}"


def test_eccentric_runs_block_anchors():
    """P7d: ellipse_ecc_z_runs (eccentric anomaly from block anchors with
    the difference-form Halley steps, checked per block, per-sample
    fallback) gives the positions of ellipse_ecc_z (per-sample solver) on
    ascending runs: dense and coarse cadence, gaps, |t| up to ~1000 d, e up
    to 0.99 (incl. periapsis passages inside blocks), short and long moon
    periods, and a high-e family near t = 0 where the per-block checks
    reject blocks; within 1e-13 (1 + |t| / (P (1 - e))) R_star (the phase ulp),
    and the anchors are really used (outputs not bit-identical) in most
    dense draws."""
    el = sys.modules["pandoramoon.ellipse"]
    rng = np.random.default_rng(20261021)
    n_anchor = 0
    n_dense = 0
    for j in range(1500):
        per_b = rng.uniform(5, 30)
        dtc = 10 ** rng.uniform(np.log10(0.0005), np.log10(0.03))
        t = np.sort(np.concatenate([np.arange(-1.5, 1.5, dtc) + per_b * k + rng.uniform(-1, 1) * 1000
                                    for k in range(3)]))
        if j >= 1000:      # high e near t = 0 (tight tolerance): the per-block checks matter
            dtc = rng.uniform(0.0005, 0.003)
            t = np.arange(-1.5, 1.5, dtc)
        if j % 5 == 0:                                   # a gap inside the run
            t = np.delete(t, np.arange(len(t) // 3, len(t) // 3 + len(t) // 10))
        e = [rng.uniform(0, 0.3), rng.uniform(0.3, 0.9), rng.uniform(0.9, 0.99)][j % 3] if j < 1000 else rng.uniform(0.8, 0.99)
        per_m = rng.uniform(0.2, 20)
        tau, Om, w, inc = rng.uniform(), rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(0, 180)
        a, q, b = rng.uniform(0.01, 0.5), 10 ** rng.uniform(-4, -1), rng.uniform(-1, 1)
        cut = len(t) // 2                                # two runs
        los = np.array([0, cut], np.int64)
        his = np.array([cut, len(t)], np.int64)
        ct = np.array([0.1, -0.2])
        cd = np.array([0.01, -0.02])
        o1 = el.ellipse_ecc_z_runs(a, per_m, e, tau, Om, w, inc, t, los, his, ct, cd, 2.0, 0.003, q, b)
        xb = np.concatenate([2 * (t[:cut] - 0.1) / 2.0 - 0.003 - 0.01, 2 * (t[cut:] + 0.2) / 2.0 - 0.003 + 0.02])
        o2 = el.ellipse_ecc_z(a, per_m, e, tau, Om, w, inc, t, q, xb, b)
        d = max(np.abs(o1[k] - o2[k]).max() for k in range(4))
        tol = 1e-13 * (1 + np.abs(t).max() / (per_m * (1 - e)))
        assert d <= tol, f"draw {j}: e={e:.3f} P={per_m:.2f} dt={dtc:.4f}: {d:.1e} > {tol:.1e}"
        assert np.allclose(o1[4], o1[0] ** 2 + o1[1] ** 2, rtol=1e-14, atol=0)
        if dtc < 0.002 and e < 0.9 and per_m > 2:
            n_dense += 1
            n_anchor += d > 0
    assert n_anchor > 0.8 * n_dense, f"block anchors used in only {n_anchor} of {n_dense} dense draws"


def test_cache_below_grid_keeps_transit():
    """Bug #3: radius ratios below the cache grid (k < 0.001) keep their transit."""
    cache = create_occult_cache(U1, U2, 300)
    zs = np.linspace(0, 1.05, 400)
    for k in (0.001, 0.002, 0.05, 0.099):   # cache accuracy inside its grid
        d = np.max(np.abs(read_occult_cache(zs.copy(), k, cache) - occult(zs.copy(), k, U1, U2)))
        assert d < 5e-6, f"cache error {d:.2e} at k={k}"
    # through the full pipeline (moon_model uses the cache)
    t = np.linspace(10.7, 11.3, 200)
    for k in (0.0005, 0.0009):
        out = pandora_fn(U1, U2, C.R_SUN_M, 365.25, 215.0, k, 0.0, 0.0, 0.0, 11.0, 0.0,
                         C.M_JUPITER, 1e-8, 5.0, 0.07, 0.0, 90.0, 0.0, 0.0, 1e-8,
                         365.26, 1, 0.001, 1.2, t, cache)
        depth = 1 - out[0].min()
        expect = k * k / (1 - U1 / 3 - U2 / 6)
        assert abs(depth - expect) / expect < 0.02, f"k={k}: depth {depth} vs {expect}"


def test_hybrid_matches_exact():
    """Bugs #6, #7, #11, #12: occult_hybrid agrees with occult everywhere,
    including length-1 arrays and the former interpolation zone (k <= 0.05,
    removed for bug #28)."""
    rng = np.random.default_rng(1)
    for k in (0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.95):
        zs = rng.uniform(0, 1 + k, 4000)
        d = np.abs(occult(zs.copy(), k, U1, U2) - occult_hybrid(zs.copy(), k, U1, U2)).max()
        assert d < 1e-6, f"k={k}: max|occult-hybrid|={d:.2e}"
    for z, k in ((0.9, 0.1), (0.4, 0.6), (0.3, 0.7), (0.5, 0.5)):
        f = occult_hybrid(np.array([z]), k, U1, U2)[0]
        assert abs(f - ref_flux(z, k)) < 1e-6, f"hybrid({z},{k})={f}"
    assert occult_hybrid(np.linspace(0, 0.65, 2000), 0.05, U1, U2).max() <= 1.0


def test_moon_exact_by_default():
    """Bug #9: the small-body approximation occult_small (4.4 ppm off at
    k = 0.0099, 105 ppm at 0.035) is opt-in only. model_params, Likelihood
    and all examples/notebooks use occult_small_threshold = 0 (exact), and
    with it a k = 0.008 moon matches the independent reference."""
    import inspect
    import glob
    import re
    import pandoramoon
    from pandoramoon.pandora import pandora_window
    from pandoramoon._config import settings
    assert pandoramoon.model_params().occult_small_threshold == 0
    assert settings()["model"]["occult_small_threshold"] == 0       # settings.cfg: the default of Likelihood and the fits
    sig = inspect.signature(pandoramoon.Likelihood.__init__)
    assert sig.parameters["occult_small_threshold"].default is None  # None: the settings.cfg value
    like = pandoramoon.Likelihood(np.linspace(10, 12, 50), np.ones(50), 1e-4, epoch_distance=365.25)
    assert like.occult_small_threshold == 0
    # the examples: next to the package (source checkout) or in ../examples of this tests folder (repository);
    # a wheel installation has none, then only the default values are checked
    for ex in (os.path.join(os.path.dirname(pandoramoon.__file__), "..", "examples"),
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")):
        files = glob.glob(os.path.join(ex, "*.py")) + glob.glob(os.path.join(ex, "*.ipynb"))
        if files:
            break
    for f in files:
        for v in re.findall(r"occult_small_threshold\s*=\s*([0-9.eE+-]+)", open(f).read()):
            assert float(v) == 0, f"{os.path.basename(f)} sets occult_small_threshold = {v}"
    # behaviour: with the default threshold the moon flux of a k = 0.008
    # moon equals the exact model at the moon's z (no eclipse: planet far)
    from pandoramoon.pandora import pandora as pandora_full
    k = 0.008
    t = np.linspace(99.6, 100.4, 400)
    args = (0.4, 0.25, 696342000.0, 365.25, 215.0, 0.1, 0.3, 0.0, 0.0, 100.0, 0.0, 1.9e27,
            k, 8.0, 0.3, 0.0, 90.0, 0.0, 0.0, 6e24, 365.25, 1,
            pandoramoon.model_params().occult_small_threshold, 1.2, t)
    fp, fm, ftot, xp, yp, xm, ym = pandora_full(*args)
    zm = np.hypot(xm, ym)
    sep = np.hypot(xm - xp, ym - yp)
    on = (zm < 1 + k) & (sep > 0.1 + k)
    assert on.sum() > 20, "moon not on the star in the test geometry"
    exact = occult_hybrid(zm[on].copy(), k, 0.4, 0.25)
    small = occult_small(zm[on].copy(), k, 0.4, 0.25)
    assert np.max(np.abs(fm[on] - exact)) < 1e-12, "default moon path is not the exact model"
    assert np.max(np.abs(small - exact)) > 1e-6, "test cannot tell occult_small from exact"


def test_eclipse_skip_is_exact():
    """P5: eclipse_possible() is False only if eclipse() leaves flux_moon
    unchanged: random sky positions (on star, on the limb, off star) with
    separations around r_p + r_m incl. exact tangency and +- a few ulp; and
    its loop is vectorised."""
    import numba
    from pandoramoon.eclipse import eclipse, eclipse_possible
    rng = np.random.default_rng(29)
    n_skip = n_call = 0
    for trial in range(1500):
        rp = rng.uniform(0.01, 0.2)
        rm = rng.uniform(0.002, 0.05)
        n = int(rng.integers(1, 6))
        ang = rng.uniform(0, 2 * np.pi, n)
        r0 = rng.choice([rng.uniform(0, 0.9), rng.uniform(0.9, 1.1), rng.uniform(1.1, 1.3)], n)
        xp, yp = r0 * np.cos(ang), r0 * np.sin(ang)
        phi = rng.uniform(0, 2 * np.pi, n)
        sep = (rp + rm) * rng.choice([1.0, 1 - 1e-15, 1 + 1e-15, 1 + 1e-12, 1 - 1e-9, 1.001, 1.5, 0.5], n)
        sep = sep * (1 + rng.integers(-3, 4, n) * np.finfo(float).eps)
        xm, ym = xp + sep * np.cos(phi), yp + sep * np.sin(phi)
        f0 = np.full(n, 0.9995)
        f = eclipse(xp, yp, xm, ym, rp, rm, f0.copy())
        if not eclipse_possible(xp, yp, xm, ym, rp, rm):
            n_skip += 1
            assert np.array_equal(f, f0), f"skipped but eclipse() changed the flux (trial {trial})"
        else:
            n_call += 1
    assert n_skip > 60 and n_call > 600, (n_skip, n_call)
    # cached code cannot be inspected: compile an uncached copy of the source
    fresh = numba.njit(fastmath=True)(eclipse_possible.py_func)
    fresh(xp, yp, xm, ym, rp, rm)
    assert "vector.body" in "".join(fresh.inspect_llvm().values()), "eclipse_possible not vectorised"


def test_eclipse_impossible_circular():
    """P24b: eclipse_impossible_circular(a, i, r_p, r_m) (the likelihood
    path then computes z only, ellipse_z_only_runs, and skips eclipse())
    is True only if no sample of ellipse_z_runs() can pass
    eclipse_possible(): draws with a |cos i| = (r_p + r_m)(1 + delta),
    delta down to +-1e-8 and 0, sampled at the conjunctions (sin Q = +-1,
    where the separation is a |cos i|) and at random times. It must also be
    tight: for delta <= -1e-7 the conjunction samples do overlap (catches
    a bound that is off by more than the margins). ellipse_z_only_runs()
    gives the same z as ellipse_z_runs(). P7e: the same for eccentric
    moons with a (1 - e) (the likelihood path skips eclipse_possible()),
    tight at periapsis conjunctions (w = 90 / 270 deg)."""
    from pandoramoon.eclipse import eclipse_possible, eclipse_impossible_circular
    from pandoramoon.ellipse import ellipse_z_runs, ellipse_z_only_runs
    rng = np.random.default_rng(2410)
    n_imp = n_tight = 0
    for trial in range(3000):
        rp, rm = rng.uniform(0.01, 0.3), rng.uniform(0.002, 0.05)
        r = rp + rm
        a = r * 10 ** rng.uniform(0.01, 2.5)
        delta = rng.choice([1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6, 0.0, 1e-3, -1e-3, rng.uniform(-1, 3)])
        c = min(r * (1 + delta) / a, 1.0)
        inc = np.degrees(np.arccos(c))
        if trial % 2:
            inc = 180 - inc
        per, tau, Om = rng.uniform(0.2, 30), rng.uniform(), rng.uniform(0, 360)
        q, b = 10 ** rng.uniform(-5, -0.3), rng.uniform(-1.2, 1.2)
        n_c = 8
        tc = per * (tau + 0.25 + 0.5 * rng.integers(-50, 50, n_c))
        tc = tc * (1 + rng.integers(-2, 3, n_c) * np.finfo(float).eps)
        t = np.sort(np.concatenate([tc, rng.uniform(-200, 200, 200)]))
        los, his = np.array([0]), np.array([len(t)])
        ct, cd = np.array([rng.uniform(-1, 1)]), np.array([0.0])
        tdur = rng.uniform(0.5, 500)      # x_bary = 2 (t - ct) / tdur - cd
        xm, ym, xp, yp, zm2, zp2 = ellipse_z_runs(a, per, tau, Om, inc, t, los, his, ct, cd, tdur, 0.0, q, b)
        zmo, zpo = ellipse_z_only_runs(a, per, tau, Om, inc, t, los, his, ct, cd, tdur, 0.0, q, b)
        assert np.allclose(zmo, zm2, rtol=1e-15, atol=1e-15) and np.allclose(zpo, zp2, rtol=1e-15, atol=1e-15), trial
        imp = eclipse_impossible_circular(a, inc, rp, rm)
        if imp:
            n_imp += 1
            assert not eclipse_possible(xp, yp, xm, ym, rp, rm), \
                f"trial {trial}: impossible claimed, but eclipse_possible (delta {delta:g})"
        if delta <= -1e-7:
            assert not imp, f"trial {trial}: delta {delta:g} claimed impossible"
            assert eclipse_possible(xp, yp, xm, ym, rp, rm), f"trial {trial}: conjunctions miss (delta {delta:g})"
            n_tight += 1
    assert n_imp > 900 and n_tight > 600, (n_imp, n_tight)
    assert not eclipse_impossible_circular(np.nan, 90.0, 0.1, 0.01)
    # P7e: eccentric moons with a (1 - e): the closest sky separation
    # a (1 - e) |cos i| is reached at periapsis when w = 90 or 270 deg
    # (periapsis at conjunction); positions from ellipse_ecc_z_runs (P7d)
    from pandoramoon.ellipse import ellipse_ecc_z_runs
    n_imp = n_tight = 0
    for trial in range(3000):
        rp, rm = rng.uniform(0.01, 0.3), rng.uniform(0.002, 0.05)
        r = rp + rm
        ecc = rng.uniform(0.01, 0.95)
        a = r * 10 ** rng.uniform(0.01, 2.5) / (1 - ecc)
        delta = rng.choice([1e-8, -1e-8, 1e-7, -1e-7, 1e-6, -1e-6, 0.0, 1e-3, -1e-3, rng.uniform(-1, 3)])
        c = min(r * (1 + delta) / (a * (1 - ecc)), 1.0)
        inc = np.degrees(np.arccos(c))
        if trial % 2:
            inc = 180 - inc
        per, tau, Om = rng.uniform(0.2, 30), rng.uniform(), rng.uniform(0, 360)
        w = rng.choice([90.0, 270.0]) if trial % 3 else rng.uniform(0, 360)
        q, b = 10 ** rng.uniform(-5, -0.3), rng.uniform(-1.2, 1.2)
        tc = per * (tau + rng.integers(-50, 50, 8))
        tc = tc * (1 + rng.integers(-2, 3, 8) * np.finfo(float).eps)
        t = np.sort(np.concatenate([tc, rng.uniform(-200, 200, 200)]))
        los, his = np.array([0]), np.array([len(t)])
        ct, cd = np.array([rng.uniform(-1, 1)]), np.array([0.0])
        xm, ym, xp, yp, _, _ = ellipse_ecc_z_runs(a, per, ecc, tau, Om, w, inc, t, los, his, ct, cd,
                                                  rng.uniform(0.5, 500), 0.0, q, b)
        imp = eclipse_impossible_circular(a * (1 - ecc), inc, rp, rm)
        if imp:
            n_imp += 1
            assert not eclipse_possible(xp, yp, xm, ym, rp, rm), \
                f"P7e trial {trial}: impossible claimed, but eclipse_possible (e {ecc:.3f}, delta {delta:g})"
        if delta <= -1e-7 and trial % 3:
            assert not imp, f"P7e trial {trial}: delta {delta:g} claimed impossible"
            assert eclipse_possible(xp, yp, xm, ym, rp, rm), f"P7e trial {trial}: periapsis misses (delta {delta:g})"
            n_tight += 1
    assert n_imp > 900 and n_tight > 400, (n_imp, n_tight)


def _ref_union_deficit(xp, yp, rp, xm, ym, rm, u1, u2, n=200000):
    """Independent reference: deficit of the union of two opaque disks on a
    quadratic limb-darkened star, by radial midpoint integration of
    I(r) r * (angular length of the union on the circle of radius r)."""
    def arc(bx, by, R, r):
        b = np.hypot(bx, by)
        c = (r * r + b * b - R * R) / (2 * r * b)
        half = np.where(c <= -1, np.pi, np.where(c >= 1, 0.0, np.arccos(np.clip(c, -1, 1))))
        return np.arctan2(by, bx), half
    zp, zm = np.hypot(xp, yp), np.hypot(xm, ym)
    lo = max(0.0, min(zp - rp, zm - rm))
    hi = min(1.0, max(zp + rp, zm + rm))
    r = lo + (hi - lo) * (np.arange(n) + 0.5) / n
    c1, h1 = arc(xp, yp, rp, r)
    c2, h2 = arc(xm, ym, rm, r)
    # union length of two arcs [c - h, c + h] on the circle
    dc = np.abs((c2 - c1 + np.pi) % (2 * np.pi) - np.pi)
    inter = np.clip(np.minimum(h1, h2) * 2, 0, None)
    gap = dc - h1 - h2                                   # > 0: disjoint
    overl = np.clip(np.minimum(h1 + h2 - dc, 2 * np.minimum(h1, h2)), 0, None)
    wrap = np.clip(h1 + h2 - (2 * np.pi - dc), 0, None)   # overlap on the far side
    length = np.minimum(2 * h1 + 2 * h2 - np.where(gap < 0, overl, 0) - wrap, 2 * np.pi)
    mu = np.sqrt(np.clip(1 - r * r, 0, 1))
    intensity = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    blocked = np.sum(intensity * r * length) * (hi - lo) / n
    return blocked / (np.pi * (1 - u1 / 3 - u2 / 6))


def test_eclipse_limb_darkening_gradient():
    """T4: partial planet-moon overlaps get the first-order limb-darkening
    correction (lens_gradient_term): against an independent radial
    integration of the union of both disks, the total deficit error of an
    Earth-sized moon next to Jupiter drops from the constant-intensity
    treatment's ~1.3 ppm to <= 0.25 ppm (moon fully on the star) and
    <= 0.65 ppm (moon on the limb), and the rms by > 5x. Also: full
    overlaps stay exact (flux 1), the term is continuous where the taper
    starts and ends (z_m = 1 - r_m and z_m = 1), and u1 = u2 = None gives the
    constant-intensity treatment."""
    from pandoramoon.eclipse import eclipse, lens_gradient_term
    u1, u2 = 0.4, 0.26
    rp, rm = 0.1005, 0.0092
    rng = np.random.default_rng(44)
    err_old = {"in": [], "limb": []}
    err_new = {"in": [], "limb": []}
    while min(len(err_new["in"]), len(err_new["limb"])) < 12:
        zone = "in" if len(err_new["in"]) < 12 else "limb"
        zmt = rng.uniform(0.3, 1 - rm) if zone == "in" else rng.uniform(1 - rm, 1 + 0.5 * rm)
        ang = rng.uniform(0, 2 * np.pi)
        xm_, ym_ = zmt * np.cos(ang), zmt * np.sin(ang)
        d = rng.uniform(rp - rm + 1e-4, rp + rm - 1e-4)
        phi = rng.uniform(0, 2 * np.pi)
        xp_, yp_ = xm_ + d * np.cos(phi), ym_ + d * np.sin(phi)
        if np.hypot(xp_, yp_) > 1 + rp:
            continue
        xp, yp, xm, ym = (np.array([v]) for v in (xp_, yp_, xm_, ym_))
        fp = occult_hybrid(np.hypot(xp, yp), rp, u1, u2)
        fm = occult_hybrid(np.hypot(xm, ym), rm, u1, u2)
        ref = _ref_union_deficit(xp_, yp_, rp, xm_, ym_, rm, u1, u2)
        old = (1 - fp) + (1 - eclipse(xp, yp, xm, ym, rp, rm, fm.copy()))
        new = (1 - fp) + (1 - eclipse(xp, yp, xm, ym, rp, rm, fm.copy(), u1, u2))
        err_old[zone].append(abs(old[0] - ref) * 1e6)
        err_new[zone].append(abs(new[0] - ref) * 1e6)
    for zone, lim in (("in", 0.25), ("limb", 0.65)):
        e_old, e_new = np.array(err_old[zone]), np.array(err_new[zone])
        assert e_new.max() <= lim, f"{zone}: T4 error {e_new.max():.3f} ppm > {lim}"
        rms_old, rms_new = np.sqrt(np.mean(e_old ** 2)), np.sqrt(np.mean(e_new ** 2))
        assert rms_new < rms_old / 2, f"{zone}: rms {rms_new:.3f} not < half of {rms_old:.3f} ppm"
    # full overlap stays exact; None = constant-intensity treatment
    xp, yp = np.array([0.3]), np.array([0.1])
    xm, ym = xp + 0.02, yp
    f0 = occult_hybrid(np.hypot(xm, ym), rm, u1, u2)
    assert eclipse(xp, yp, xm, ym, rp, rm, f0.copy(), u1, u2)[0] == 1.0
    xm = xp + 0.1
    old = eclipse(xp, yp, xm, ym, rp, rm, f0.copy())
    assert eclipse(xp, yp, xm, ym, rp, rm, f0.copy(), u1, u2)[0] != old[0]
    # continuity of the taper: at z_m = 1 - r_m (w = 1) and z_m -> 1 (w -> 0)
    for zt in (1 - rm, 1.0):
        vals = []
        for eps in (-1e-9, 1e-9):
            zz = zt + eps
            vals.append(lens_gradient_term(zz - 0.1, 0.0, zz, 0.0, 0.1, rp, rm, u1, u2))
        assert abs(vals[0] - vals[1]) < 1e-9, f"taper discontinuous at z_m = {zt}: {vals}"


def test_complete_occultation():
    """Bug #16: k > 1 and z <= k-1 gives zero flux in both functions."""
    zs = np.array([0.0, 0.05, 0.1, 0.19])
    assert np.all(occult(zs.copy(), 1.2, U1, U2) == 0.0)
    assert np.all(occult_hybrid(zs.copy(), 1.2, U1, U2) == 0.0)


def test_grid_helpers():
    """Bugs #10, #14, #15, #22."""
    xb = x_bary_grid(np.array([11.0 - 365.26]), 215.0, 365.25, 11.0, 0.0, 365.26, 0.0, 0.0)
    assert abs(xb[0]) < 1.1, "epoch before t0 mis-rounded"
    try:
        timegrid(11.0, 3, 0.6, 250.0, 365.26, 0)
        raise AssertionError("supersampling_factor=0 accepted")
    except ValueError:
        pass
    for cpd, dur, ssf in ((48.0, 0.6, 5), (30.0, 0.7, 3), (250.0, 0.6, 5)):
        t = timegrid(11.0, 3, dur, cpd, 365.26, ssf)
        groups = t.reshape(-1, ssf)
        assert len(t) % (3 * ssf) == 0
        assert (groups.max(1) - groups.min(1)).max() < dur, "supersample group straddles epochs"
    q1, q2 = ld_invert(0.0, 0.0)
    assert ld_convert(q1, q2) == (0.0, 0.0)
    for u in ((0.4, 0.25), (0.6, -0.1)):
        assert np.allclose(ld_convert(*ld_invert(*u)), u)


def test_timegrid_cadence_spacing():
    """Bug #26: cadences are spaced 1/cadences_per_day, sub-exposures are
    centred within each cadence, and the block is centred on the epoch."""
    for cpd, dur, ssf in ((250.0, 0.6, 1), (250.0, 0.6, 5), (1440 / 29.4244, 1.0, 5),
                          (48.0, 0.6, 3), (720.0, 0.37, 1)):
        n = int(cpd * dur)
        t = timegrid(11.0, 3, dur, cpd, 365.26, ssf).reshape(3, n, ssf)
        for e in range(3):
            mid = 11.0 + 365.26 * e
            # independent reference: exposure j covers [s + j/cpd, s + (j+1)/cpd]
            s = mid - n / cpd / 2
            ref = s + (np.arange(n)[:, None] + (np.arange(ssf)[None, :] + 0.5) / ssf) / cpd
            err = np.abs(t[e] - ref).max()
            assert err < 1e-9, f"cpd={cpd} dur={dur} ssf={ssf}: max |t - ref| = {err * 86400:.3g} s"
            assert t[e].min() > mid - dur / 2 and t[e].max() < mid + dur / 2
    # integer cpd*dur: the exposures tile the window [t_start, t_end] exactly
    t = timegrid(11.0, 1, 0.6, 250.0, 365.26, 1)
    assert abs(t[0] - (10.7 + 0.5 / 250)) < 1e-12 and abs(t[-1] - (11.3 - 0.5 / 250)) < 1e-12


def test_timegrid_cadence_count_roundoff():
    """Bug #27: float round-off in cadences_per_day * epoch_duration must not
    drop a cadence (int(720 * 0.35) == 251); non-integer products still floor."""
    from fractions import Fraction
    for cpd, dur, ssf in ((100.0, 0.29, 1), (720.0, 0.35, 1), (720.0, 0.7, 3),
                          (4320.0, 0.15, 1), (250.0, 0.6, 5), (48.0, 0.61, 5),
                          (1440 / 29.4244, 1.0, 5)):
        # exact count from the decimal values (independent of float rounding)
        exact = Fraction(repr(cpd)) * Fraction(repr(dur))
        n = int(exact)          # floor of the exact decimal product
        t = timegrid(11.0, 2, dur, cpd, 365.26, ssf)
        assert len(t) == 2 * n * ssf, f"cpd={cpd} dur={dur}: {len(t) // (2 * ssf)} cadences, expected {n}"


def test_ellpicb_converges():
    """Bug #18: no silent zero; extreme but valid arguments converge."""
    for n, k in ((0.5, 0.3), (1e12, 0.6), (5.25, 1 - 1e-8), (0.0, 0.0)):
        v = ellpicb(n, k)
        assert np.isfinite(v) and v > 0, f"ellpicb({n},{k})={v}"


# ----------------------------------------------------------------------------
# analytic mutual-eclipse overlap (the only method since 2.0)
# ----------------------------------------------------------------------------
def _disk_intersection_area(circles, n=40_000):
    """Independent reference: area of the intersection of disks, by midpoint
    integration over x of the exact intersection of the per-disk y-intervals
    (~1e-8 relative). Shares no code with overlap_analytic."""
    lo = max(cx - r for cx, cy, r in circles)
    hi = min(cx + r for cx, cy, r in circles)
    if hi <= lo:
        return 0.0
    x = lo + (np.arange(n) + 0.5) * (hi - lo) / n
    ylo = np.full(n, -np.inf)
    yhi = np.full(n, np.inf)
    for cx, cy, r in circles:
        h = np.sqrt(np.clip(r * r - (x - cx) ** 2, 0.0, None))
        ylo = np.maximum(ylo, cy - h)
        yhi = np.minimum(yhi, cy + h)
    return float(np.sum(np.clip(yhi - ylo, 0.0, None)) * (hi - lo) / n)


def _limb_eclipse_configs(n, seed=20261006):
    """Random planet-moon overlaps with the moon or the planet on the limb."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rp = 10 ** rng.uniform(-2, -0.5)
        rm = 10 ** rng.uniform(-2.7, -0.5)
        th, ph = rng.uniform(0, 2 * np.pi, 2)
        s = rng.uniform(0, 1) * (rp + rm)
        if rng.random() < 0.6:          # moon on the limb
            zm = 1 + rng.uniform(-1, 1) * rm
            xm, ym = zm * np.cos(th), zm * np.sin(th)
            xp, yp = xm + s * np.cos(ph), ym + s * np.sin(ph)
        else:                           # planet on the limb
            zp = 1 + rng.uniform(-1, 1) * rp
            xp, yp = zp * np.cos(th), zp * np.sin(th)
            xm, ym = xp + s * np.cos(ph), yp + s * np.sin(ph)
        out.append((xp, yp, rp, xm, ym, rm))
    return out


def test_analytic_eclipse_matches_reference():
    """Analytic eclipse: er = A(star&planet&moon)/A(star&moon) from
    overlap_analytic.er_fast matches an independent numerical integration
    (< 1e-6 of the moon area) on random limb geometries and exact
    symmetric/tangent inputs."""
    from pandoramoon.overlap_analytic import er_fast
    worst_a = 0.0
    for xp, yp, rp, xm, ym, rm in _limb_eclipse_configs(80):
        a_m = np.pi * rm * rm
        a_sm = _disk_intersection_area([(0, 0, 1), (xm, ym, rm)])
        a_smp = _disk_intersection_area([(0, 0, 1), (xm, ym, rm), (xp, yp, rp)])
        er_ref = 1.0 if a_sm <= 0 else min(1.0, a_smp / a_sm)
        er_a = er_fast(xp, yp, rp, xm, ym, rm)
        assert np.isfinite(er_a), f"NaN for {(xp, yp, rp, xm, ym, rm)}"
        worst_a = max(worst_a, abs(er_a - er_ref) * a_sm / a_m)
    assert worst_a < 1e-6, f"analytic overlap error {worst_a:.2e} of moon area"
    # exact inputs: coincident, concentric, tangent, centre exactly on the limb
    exact = [((1.0, 0, 0.1, 1.0, 0, 0.1), 1.0), ((1.0, 0, 0.1, 1.0, 0, 0.05), 1.0),
             ((1.0, 0.1, 0.1, 1.0, -0.1, 0.1), 0.0), ((1.0, 0.05, 0.05, 1.0, -0.05, 0.05), 0.0),
             ((0.0, 0.0, 1.5, 1.0, 0.0, 0.05), 1.0), ((1.0, 0.0, 1.0, 1.0, 0.0, 0.01), 1.0)]
    for args, want in exact:
        assert er_fast(*args) == want, f"er_fast{args} = {er_fast(*args)} != {want}"
    for args in [(1.0, 0, 0.05, 1.0, 0, 0.1), (0.95, 0, 0.05, 1.0, 0, 0.05), (1.0, 0.0, 0.1, 1.0, 0.1, 0.1)]:
        xp, yp, rp, xm, ym, rm = args
        ref = (_disk_intersection_area([(0, 0, 1), (xm, ym, rm), (xp, yp, rp)], 400_000)
               / _disk_intersection_area([(0, 0, 1), (xm, ym, rm)], 400_000))
        assert abs(er_fast(*args) - ref) < 1e-7, f"er_fast{args} = {er_fast(*args)}, ref {ref}"


def test_pixelmap_removed():
    """2.0: the pixel-raster eclipse (eclipse_method="pixelmap", numerical_grid) is gone. The analytic overlap is
    the only method: model_params has no such attributes, moon_model and Likelihood reject "pixelmap" with a
    ValueError, Likelihood(numerical_grid=...) only warns and is ignored, and positional callers that still pass
    the grid get an error (numba TypingError), never a silently misaligned result."""
    import warnings
    import numba
    from pandoramoon import Likelihood
    p = C.pandora.model_params()
    assert not hasattr(p, "numerical_grid") and not hasattr(p, "eclipse_method")
    p.eclipse_method = "pixelmap"
    try:
        C.pandora.moon_model(p)
        raise AssertionError("moon_model accepted eclipse_method='pixelmap'")
    except ValueError:
        pass
    t = np.linspace(10, 12, 300)
    try:
        Likelihood(t, np.ones(300), np.ones(300), epoch_distance=365.25, eclipse_method="pixelmap")
        raise AssertionError("Likelihood accepted eclipse_method='pixelmap'")
    except ValueError:
        pass
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Likelihood(t, np.ones(300), np.ones(300), epoch_distance=365.25, numerical_grid=25)
    assert any("numerical_grid" in str(x.message) for x in w), "numerical_grid was not reported as ignored"
    args = (U1, U2, C.R_SUN_M, 365.25, 215.0, 0.1, 0.3, 0.0, 0.0, 11.0, 0.0, C.M_JUPITER, 0.01, 3.0, 0.1, 0.0,
            90.0, 0.0, 0.0, 6e24, 365.26, 1, 0.0, 1.0)
    try:
        pandora_fn(*args, 25, t)
        raise AssertionError("pandora() accepted the removed numerical_grid argument")
    except (TypeError, numba.core.errors.TypingError):    # numba reports the misaligned call as a TypingError
        pass


def test_light_curve_exact():
    """2.0: moon_model.light_curve() uses the exact path (no float32 occultation cache any more), like
    coordinates() and video(): it equals pandora() without a cache bit for bit, and differs from the cached path
    of Pandora <= 1.x by at most a few ppm."""
    c = [c for c in C.build_cases() if c["name"] == "sun_earth_luna_plato_1min"][0]
    p = C._params(c)
    t = C.pandora.time(p).grid()
    ft, fp, fm = C.pandora.moon_model(p).light_curve(t)
    args = (c["u1"], c["u2"], c["R_star"], c["per_bary"], c["a_bary"], c["r_planet"], c["b_bary"], c["w_bary"],
            c["ecc_bary"], c["t0_bary"], c["t0_bary_offset"], c["M_planet"], c["r_moon"], c["per_moon"],
            c["tau_moon"], c["Omega_moon"], c["i_moon"], c["ecc_moon"], c["w_moon"], c["M_moon"],
            c["epoch_distance"], c["supersampling_factor"], c["occult_small_threshold"], c["hill_sphere_threshold"], t)
    exact = pandora_fn(*args)
    assert np.array_equal(fp, exact[0]) and np.array_equal(fm, exact[1]) and np.array_equal(ft, exact[2])
    cached = pandora_fn(*args, create_occult_cache(c["u1"], c["u2"], dim=300))
    d = max(np.max(np.abs(a - b)) for a, b in zip(cached[:3], exact[:3]))
    assert d < 5e-6, f"cached path differs by {d:.2e}"


def _unculled_fluxes(args, cache, coords):
    """Fluxes from pandora()'s coordinates with the pre-P1 pipeline: every
    sample goes through z, the occultation routines and eclipse() (no
    culling), then resampling. Independent of the culling bound."""
    (u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0,
     t0_off, M_planet, r_moon, per_moon, tau, Omega, i_moon, ecc_moon, w_moon,
     M_moon, epoch_distance, ssf, small_threshold, hill, t) = args
    from pandoramoon.eclipse import eclipse
    xp, yp, xm, ym = coords
    unphysical = bool(np.all(xm == 1e8))
    zp = np.sqrt(xp ** 2 + yp ** 2)
    zm = xm.copy() if unphysical else np.sqrt(xm ** 2 + ym ** 2)
    if cache is not None and r_planet < 0.1:
        fp = (occult_small(zp, r_planet, u1, u2) if r_planet < cache[1][0]
              else read_occult_cache(zp, r_planet, cache))
    else:
        fp = occult_hybrid(zp, r_planet, u1, u2)
    if r_moon < small_threshold:
        fm = occult_small(zm, r_moon, u1, u2)
    elif cache is not None and r_moon < 0.1:
        fm = (occult_small(zm, r_moon, u1, u2) if r_moon < cache[1][0]
              else read_occult_cache(zm, r_moon, cache))
    else:
        fm = occult_hybrid(zm, r_moon, u1, u2)
    eclipsed = False
    if not unphysical:
        before = fm.copy()
        fm = eclipse(xp, yp, xm, ym, r_planet, r_moon, fm, u1, u2)
        eclipsed = bool(np.any(fm != before))
    ft = fm + fp - 1
    if ssf > 1:
        fp, fm, ft = resample(fp, ssf), resample(fm, ssf), resample(ft, ssf)
    return (fp, fm, ft), eclipsed


def _random_draws(seed=20261007, n_draws=80):
    """Random pandora() argument tuples for the P1/float32/P9 tests: yields
    (j, args, cache, info). Covers unphysical moons, eccentric
    moon and planet orbits, supersampling 1/3/5 (also with a partial last
    cadence), the occultation cache, sorted and unsorted
    time with gaps, near-coplanar eclipsing moons, and edge-on low-mass
    moons sampled right at the culling bound ("tight")."""
    rng = np.random.default_rng(seed)
    for j in range(n_draws):
        u1, u2 = ld_convert(rng.uniform(), rng.uniform())
        per_bary, a_bary = rng.uniform(5, 400), rng.uniform(15, 250)
        r_planet = rng.choice([rng.uniform(0.005, 0.15), rng.uniform(0.0005, 0.01)])
        r_moon = rng.choice([rng.uniform(0.001, 0.05), rng.uniform(0.0002, 0.01)])
        ecc_moon = 0.0 if j % 2 else rng.uniform(0, 0.6)
        ecc_bary = 0.0 if j % 3 else rng.uniform(0, 0.5)
        M_planet = 10 ** rng.uniform(25, 28)
        M_moon = M_planet * 10 ** rng.uniform(-4, -0.3)
        per_moon = rng.uniform(0.3, 40)
        tau, Omega, i_moon = rng.uniform(), rng.uniform(0, 180), rng.uniform(0, 180)
        b_bary, t0, t0_off = rng.uniform(0, 1.2), 50.0, rng.uniform(-0.05, 0.05)
        ssf = int(rng.choice([1, 3, 5]))
        epoch_distance = per_bary + rng.uniform(-0.01, 0.01)
        tight = (j % 5 == 4)
        if tight:   # edge-on, tiny moon mass: moon elongation ~ a_moon
            ecc_moon, ecc_bary, b_bary, i_moon, Omega = 0.0, 0.0, 0.0, 90.0, 0.0
            M_moon = M_planet * 1e-9
            per_moon = rng.uniform(0.5, 3)
        if j % 5 == 2:  # near-coplanar, short-period moon: mutual eclipses likely
            i_moon, Omega, b_bary = rng.uniform(88, 92), rng.uniform(-3, 3), rng.uniform(0, 0.5)
            per_moon = rng.uniform(0.3, 1.5)
        n = 3001 * ssf + (2 if j % 4 == 0 else 0)          # partial last cadence
        mode = j % 3
        if tight:
            a_m = (6.67408e-11 * (M_planet + M_moon) / (2 * np.pi / (per_moon * 86400)) ** 2) ** (1 / 3) / 696342000.0
            tdur = per_bary / np.pi * np.arcsin(1 / a_bary)
            edge = (1 + max(r_planet, r_moon) + a_m) * tdur / 2   # [d] from mid-transit
            t = t0 + t0_off + np.sign(rng.uniform(-1, 1, n)) * (edge + rng.uniform(-0.02, 0.002, n) * tdur)
        elif mode == 0:
            t = np.sort(rng.uniform(t0 - 1.5 * per_bary, t0 + 2.5 * per_bary, n))
        elif mode == 1:   # unsorted, with gaps
            t = np.concatenate([t0 + e * per_bary + rng.uniform(-2, 2, n // 4 + 1) for e in range(4)])[:n]
            rng.shuffle(t)
        else:
            t = t0 + rng.uniform(-1.0, 1.0, n)
        use_cache = (j % 4 == 1)
        args = (u1, u2, 696342000.0, per_bary, a_bary, r_planet, b_bary, 0.0, ecc_bary, t0,
                t0_off, M_planet, r_moon, per_moon, tau, Omega, i_moon, ecc_moon, 0.0,
                M_moon, epoch_distance, ssf, 0.01 if j % 2 else 0.0, 1.2, t)
        cache = create_occult_cache(u1, u2, dim=300) if use_cache else None
        yield j, args, cache, dict(tight=tight, ecc_moon=ecc_moon, ssf=ssf,
                                             use_cache=use_cache, sorted=(mode == 0 and not tight))


def _call(fn, args, cache):
    if cache is not None:
        return fn(*args, cache)
    return fn(*args)


def test_pandora_flux_matches_pandora():
    """P1 (sample culling): pandora_flux() returns the same planet/moon/total
    flux as pandora(), and both match the unculled pipeline evaluated on all
    samples. Random draws cover unphysical moons (moved to 1e8), eccentric
    moon and planet orbits, supersampling 1/3/5 (also with a partial last
    cadence), the occultation cache, unsorted time with
    gaps, and edge-on low-mass moons sampled right at the culling bound
    (moon elongation ~ a, so the bound is tight). Tolerance 1e-12: fastmath
    code compiled in different contexts differs by up to ~4e-14."""
    from pandoramoon.pandora import pandora_flux
    seen = dict(unphysical=0, eccentric=0, ssf=0, cache=0, eclipse=0, culled=0, tight=0)
    for j, args, cache, info in _random_draws():
        tight, ecc_moon, ssf, use_cache = info["tight"], info["ecc_moon"], info["ssf"], info["use_cache"]
        u1, u2, r_moon = args[0], args[1], args[12]
        full = _call(pandora_fn, args, cache)
        flux = _call(pandora_flux, args, cache)
        ref, eclipsed = _unculled_fluxes(args, cache, full[3:7])
        for name, a, b, c in zip(("planet", "moon", "total"), flux, full[:3], ref):
            assert a.shape == b.shape == c.shape, f"draw {j} {name}: shapes {a.shape} {b.shape} {c.shape}"
            assert np.all(np.isfinite(a)), f"draw {j} {name}: non-finite flux"
            for lab, other in (("pandora()", b), ("unculled", c)):
                d = np.max(np.abs(a - other))
                assert d <= 1e-12, f"draw {j} {name}: pandora_flux vs {lab} differ by {d:.2e}"
            # flux == 1 pattern: a culled sample must have had flux exactly 1.
            # Deficits <= 1e-13 may round to exactly 0 differently (P7 sincos
            # moves positions by ~1e-13; e.g. 1.4e-15 near a dark limb), so
            # only mismatches with a larger deficit count (still kills the
            # mutations "bound 1 % too small" and "no 1.1 cache limit").
            mism = (a == 1.0) != (c == 1.0)
            big = np.maximum(np.abs(1.0 - a), np.abs(1.0 - c)) > 1e-13
            assert not np.any(mism & big), f"draw {j} {name}: flux == 1 pattern differs"
        xm = full[5]
        seen["unphysical"] += bool(np.all(xm == 1e8))
        seen["eccentric"] += ecc_moon > 0 and not np.all(xm == 1e8)
        seen["ssf"] += ssf > 1
        seen["cache"] += use_cache
        seen["eclipse"] += eclipsed
        seen["culled"] += bool(np.mean(flux[2] == 1.0) > 0.5)
        if tight and not np.all(xm == 1e8):
            # in transit at the samples closest to the bound (flux < 1 there)
            seen["tight"] += bool(np.any(flux[1] < 1.0))
    for k, v in seen.items():
        assert v >= 3, f"coverage: only {v} draws of kind {k!r}"
    # The cache's bilinear interpolation gives flux < 1 slightly beyond
    # z = 1 + k (grid bin straddling the contact point): the bound must use
    # z = 1.1 with a cache. Planet only (moon unphysical), b = 0, so z = |x|.
    cache = create_occult_cache(U1, U2, dim=300)
    tdur = 365.25 / np.pi * np.arcsin(1 / 215.0)
    t = 11.0 + np.linspace(1.0, 1.12, 2000) * tdur / 2
    args = (U1, U2, C.R_SUN_M, 365.25, 215.0, 0.05, 0.0, 0.0, 0.0, 11.0, 0.0, C.M_JUPITER,
            0.01, 300.0, 0.0, 0.0, 90.0, 0.0, 0.0, 6e24, 365.25, 1, 0.01, 1.2, t)
    full = pandora_fn(*args, cache)
    flux = pandora_flux(*args, cache)
    assert np.all(full[5] == 1e8), "cache-edge geometry: moon should be unphysical"
    ref, _ = _unculled_fluxes(args, cache, full[3:7])
    z = np.hypot(full[3], full[4])
    beyond = (z > 1.05) & (ref[0] < 1)
    assert beyond.sum() > 10, "cache-edge geometry: no flux < 1 beyond z = 1 + k"
    for a, b, c in zip(flux, full[:3], ref):
        assert np.max(np.abs(a - c)) <= 1e-12, "pandora_flux culls cache samples with flux < 1"
        assert np.max(np.abs(b - c)) <= 1e-12, "pandora() culls cache samples with flux < 1"
    # keyword call as in the retrieval notebook
    ft_kw = pandora_flux(u1=U1, u2=U2, R_star=C.R_SUN_M, per_bary=365.25, a_bary=215.0,
                         r_planet=0.1, b_bary=0.3, w_bary=0, ecc_bary=0, t0_bary=11.0,
                         t0_bary_offset=0.0, M_planet=C.M_JUPITER, r_moon=0.01, per_moon=3.0,
                         tau_moon=0.1, Omega_moon=10.0, i_moon=80.0, ecc_moon=0, w_moon=0,
                         M_moon=6e24, epoch_distance=365.26, supersampling_factor=1,
                         occult_small_threshold=0.01, hill_sphere_threshold=1.2,
                         time=np.linspace(10, 12, 500))[2]
    assert 0.98 < ft_kw.min() < 1.0 and ft_kw[0] == 1.0


def test_pandora_deficit_float32():
    """float32 step: pandora_deficit() returns float32 arrays equal to
    1 - pandora_flux() rounded to float32 (round-to-nearest of the float64
    deficit: error <= 2^-24 relative to the deficit), exactly 0 where the
    flux is exactly 1, for all random draws of the P1 test. The logL of a
    sampler-like chi^2 changes by << 0.01 against float64 flux."""
    from pandoramoon.pandora import pandora_flux, pandora_deficit
    worst_rel, worst_dll = 0.0, 0.0
    for j, args, cache, info in _random_draws(n_draws=40):
        flux = _call(pandora_flux, args, cache)
        dfc = _call(pandora_deficit, args, cache)
        for name, f, d in zip(("planet", "moon", "total"), flux, dfc):
            assert d.dtype == np.float32, f"draw {j} {name}: dtype {d.dtype}"
            assert d.shape == f.shape, f"draw {j} {name}: shape"
            exact = 1.0 - f
            assert np.array_equal(d == 0, f == 1.0), f"draw {j} {name}: zero pattern differs"
            err = np.abs(d.astype(np.float64) - exact)
            # float32 rounding of a float64 value (plus ~1e-15 float64 noise)
            assert np.all(err <= 2.0 ** -24 * np.abs(exact) + 1e-14), \
                f"draw {j} {name}: max err {err.max():.2e}"
            nz = exact != 0
            if nz.any():
                worst_rel = max(worst_rel, float(np.max(err[nz] / np.abs(exact[nz]))))
        # chi^2 with noise 100 ppm against data = model + noise
        rng = np.random.default_rng(j)
        data = flux[2] + rng.normal(0, 1e-4, len(flux[2]))
        l64 = -0.5 * np.sum(((flux[2] - data) / 1e-4) ** 2)
        l32 = -0.5 * np.sum((((1.0 - data) - dfc[2]) / 1e-4) ** 2)
        worst_dll = max(worst_dll, abs(l32 - l64))
    assert 0 < worst_rel <= 2.0 ** -24 * 1.01, f"relative rounding {worst_rel:.2e}"
    assert worst_dll < 0.01, f"logL changed by {worst_dll:.3g}"


def test_params_valid():
    """Bug #31: parameters outside their physical range are rejected before
    any computation: pandora(), pandora_flux(), pandora_deficit() and
    pandora_window() raise ValueError, Likelihood returns -inf, and
    check_params() names the rule. Before, per_moon < 0 gave a Likelihood
    up to 175 ppm off pandora_flux() (P24d's block anchors assumed
    per > 0). Every rule is tested just inside (accepted, finite result) and
    just outside its boundary, and NaN / +inf / -inf in every parameter; the
    jitted check (no fastmath, also when inlined into fastmath callers) and
    check_params() agree on random draws."""
    import pandoramoon
    from pandoramoon import Likelihood, check_params
    from pandoramoon.pandora import params_valid
    names = ("u1 u2 R_star per_bary a_bary r_planet b_bary w_bary ecc_bary t0_bary "
             "t0_bary_offset M_planet r_moon per_moon tau_moon Omega_moon i_moon ecc_moon "
             "w_moon M_moon").split()
    base = dict(u1=0.4, u2=0.26, R_star=C.R_SUN_M, per_bary=365.25, a_bary=215.0, r_planet=0.1,
                b_bary=0.3, w_bary=0.0, ecc_bary=0.0, t0_bary=11.0, t0_bary_offset=0.0,
                M_planet=C.M_JUPITER, r_moon=0.01, per_moon=3.0, tau_moon=0.3,
                Omega_moon=10.0, i_moon=88.0, ecc_moon=0.0, w_moon=0.0, M_moon=C.M_EARTH)
    t = np.linspace(10.0, 12.0, 500)
    rng = np.random.default_rng(31)
    data = 1 + rng.normal(0, 1e-4, len(t))
    like = Likelihood(t, data, 1e-4, epoch_distance=365.25)

    def call_all(p):
        a = [p[k] for k in names]
        tail = (365.25, 1, 0.0, 1.2, t)
        ll = like(*a)
        raised = []
        for f in (pandoramoon.pandora, pandoramoon.pandora_flux, pandoramoon.pandora_deficit,
                  pandoramoon.pandora_window):
            try:
                out = f(*a, *tail)
                assert np.all(np.isfinite(out[0 if f is not pandoramoon.pandora_window else 1]))
                raised.append(False)
            except ValueError:
                raised.append(True)
        return ll, raised, check_params(*a)

    ll, raised, bad = call_all(base)
    assert np.isfinite(ll) and not any(raised) and bad == [], (ll, raised, bad)
    # (parameter, value just inside, value just outside, rule name)
    edges = [("per_moon", 1e-6, -3.0, "per_moon > 0"), ("per_moon", 1e-6, 0.0, "per_moon > 0"),
             ("per_bary", 1.0, 0.0, "per_bary > 0"), ("a_bary", 215.0, -1.0, "a_bary > 0"),
             ("R_star", C.R_SUN_M, 0.0, "R_star > 0"), ("M_planet", 1e20, 0.0, "M_planet > 0"),
             ("r_planet", 0.0, -1e-9, "r_planet >= 0"), ("r_moon", 0.0, -1e-9, "r_moon >= 0"),
             ("M_moon", 0.0, -1.0, "M_moon >= 0"),
             ("ecc_bary", 0.0, -1e-9, "0 <= ecc_bary < 1"), ("ecc_bary", 0.5, 1.0, "0 <= ecc_bary < 1"),
             ("ecc_moon", 0.0, -1e-9, "0 <= ecc_moon < 1"), ("ecc_moon", 0.3, 1.0, "0 <= ecc_moon < 1"),
             ("u1", 0.0, -1e-9, "u1 >= 0"), ("u2", 0.6, 0.6 + 1e-9, "u1 + u2 <= 1"),
             ("u2", -0.2, -0.2 - 1e-9, "u1 + 2 u2 >= 0")]
    for k, ok_v, bad_v, rule in edges:
        p = dict(base)
        if k == "ecc_moon" and ok_v > 0:
            p["per_moon"] = 10.0     # keep the eccentric moon physical (no planet-only fallback needed)
        p[k] = ok_v
        ll, raised, bad = call_all(p)
        assert np.isfinite(ll) and not any(raised) and bad == [], (k, ok_v, ll, raised, bad)
        p[k] = bad_v
        ll, raised, bad = call_all(p)
        assert ll == -np.inf and all(raised) and rule in bad, (k, bad_v, ll, raised, bad)
    for k in names:
        for v in (np.nan, np.inf, -np.inf):
            p = dict(base)
            p[k] = v
            ll, raised, bad = call_all(p)
            assert ll == -np.inf and all(raised) and f"{k} finite" in bad, (k, v, ll, raised, bad)
    # settings: supersampling_factor / epoch_distance
    a = [base[k] for k in names]
    for tail in ((365.25, 0, 0.0, 1.2, t), (0.0, 1, 0.0, 1.2, t), (np.nan, 1, 0.0, 1.2, t)):
        try:
            pandoramoon.pandora_flux(*a, *tail)
            raise AssertionError(f"no ValueError for {tail[:4]}")
        except ValueError:
            pass
    # jitted check == Python check on random draws around the boundaries
    for trial in range(3000):
        p = dict(base)
        for k in ("u1", "u2", "per_moon", "ecc_moon", "ecc_bary", "r_moon", "M_moon", "r_planet"):
            if rng.random() < 0.3:
                p[k] = rng.choice([rng.uniform(-1, 1.5), 0.0, 1.0, -1e-15, 1 - 1e-16, np.nan])
        a = [p[k] for k in names]
        assert params_valid(*a, 365.25, 1) == (check_params(*a) == []), (trial, check_params(*a))
        assert (like(*a) == -np.inf) == (check_params(*a) != []), (trial, check_params(*a))


def test_pandora_window_and_likelihood():
    """P9: pandora_window() densified equals pandora_flux() (<= 1e-12; 1 at
    all other cadences); Likelihood equals -0.5 * nansum(((flux_total -
    data) / yerr)^2) with pandora_flux() (relative 1e-10), including NaN
    data/yerr (weight 0), supersampling with a partial last cadence, the
    cache; invalid inputs raise ValueError. P8: the call goes
    through the compiled entry point, bit-identical to the dispatcher with
    float parameters (numpy scalars, ints and 0-d arrays act as float(x);
    a non-numeric parameter raises TypeError)."""
    from pandoramoon.pandora import pandora_flux, pandora_window, pandora_loglike
    from pandoramoon import Likelihood
    for j, args, cache, info in _random_draws(n_draws=40):
        flux = _call(pandora_flux, args, cache)
        idx, fp, fm, ft = _call(pandora_window, args, cache)
        n_out = len(flux[0])
        assert idx.dtype == np.int64 and np.all(np.diff(idx) > 0), f"draw {j}: idx not ascending"
        for name, dense, w in zip(("planet", "moon", "total"), flux, (fp, fm, ft)):
            full = np.ones(n_out)
            full[idx] = w
            d = np.max(np.abs(full - dense))
            assert d <= 1e-12, f"draw {j} {name}: window vs pandora_flux differ by {d:.2e}"
        rng = np.random.default_rng(100 + j)
        sigma = rng.uniform(5e-5, 1e-3)
        data = flux[2] + rng.normal(0, sigma, n_out)
        yerr = np.full(n_out, sigma) * rng.uniform(0.8, 1.2, n_out)
        if j % 3 == 0:   # missing values
            data[rng.integers(0, n_out, 20)] = np.nan
            yerr[rng.integers(0, n_out, 5)] = np.nan
        ref = -0.5 * np.nansum(((flux[2] - data) / yerr) ** 2)
        (u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0, t0_off,
         M_planet, r_moon, per_moon, tau, Omega, i_moon, ecc_moon, w_moon, M_moon,
         epoch_distance, ssf, small, hill, t) = args
        like = Likelihood(t, data, yerr, epoch_distance=epoch_distance, supersampling_factor=ssf,
                          occult_small_threshold=small, hill_sphere_threshold=hill, cache=cache)
        val = like(u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0,
                   t0_off, M_planet, r_moon, per_moon, tau, Omega, i_moon, ecc_moon, w_moon,
                   M_moon)
        assert np.isfinite(val) and abs(val - ref) <= 1e-10 * abs(ref) + 1e-9, \
            f"draw {j}: Likelihood {val!r} vs nansum {ref!r}"
        # P8: Likelihood calls the compiled entry point (no numba dispatch);
        # same value as the dispatcher with float parameters, and parameters
        # given as np.float64 / int / 0-d arrays are taken as float(x)
        params = [u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0,
                  t0_off, M_planet, r_moon, per_moon, tau, Omega, i_moon, ecc_moon, w_moon, M_moon]
        assert like._loglike is not pandora_loglike, "P8: no compiled entry point"
        via_dispatcher = pandora_loglike(*[float(x) for x in params], *like._tail)
        assert val == via_dispatcher, f"draw {j}: entry point {val!r} vs dispatcher {via_dispatcher!r}"
        mixed = [np.float64(x) if q % 3 == 0 else (np.array(float(x)) if q % 3 == 1 else x)
                 for q, x in enumerate(params)]
        mixed[8] = 0 if float(ecc_bary) == 0 else mixed[8]   # an int parameter
        assert like(*mixed) == pandora_loglike(*[float(x) for x in mixed], *like._tail), \
            f"draw {j}: numpy / int parameters"
    try:
        like("x", *params[1:])
        raise AssertionError("P8: a non-numeric parameter was accepted")
    except TypeError:
        pass
    t = np.linspace(10, 12, 300)
    for bad in (dict(data=np.ones(299), yerr=np.ones(299)),
                dict(data=np.ones(300), yerr=np.zeros(300)),
                dict(data=np.ones(300), yerr=np.ones(300), eclipse_method="pixelmap")):
        try:
            Likelihood(t, epoch_distance=365.25, **bad)
            raise AssertionError(f"accepted invalid input {sorted(bad)}")
        except ValueError:
            pass


def test_sorted_time_window_search():
    """Binary-search window (time_sorted): x_bary_window_sorted() returns the
    same indices and x_bary as the full scan x_bary_window() for ascending
    time (P11: up to samples at |x| = reach within round-off) over random
    geometries: period drift (per_bary != epoch_distance),
    t0 offsets, eccentric planet, duplicate times, samples placed exactly at
    and around the window edges, single/many epochs, windows too wide for
    the search (fallback), more epochs than samples (fallback), no samples.
    pandora_window(time_sorted=True) and Likelihood (auto-detects
    sorted time) agree with the unsorted paths."""
    from pandoramoon.grids import (x_bary_window, x_bary_window_sorted,
                                   x_bary_window_sorted_t, x_bary_grid)
    from pandoramoon.pandora import pandora_window
    from pandoramoon import Likelihood
    rng = np.random.default_rng(20261008)
    n_fallback = 0
    n_edge = 0
    for j in range(300):
        per = rng.choice([rng.uniform(0.3, 5), rng.uniform(5, 500)])
        a = rng.uniform(1.5, 300)
        ed = per + rng.uniform(-0.02, 0.02) * (j % 2)
        t0, off = rng.uniform(-100, 100), rng.uniform(-0.1, 0.1) * (j % 3 > 0)
        ecc, w = (rng.uniform(0, 0.6), rng.uniform(0, 360)) if j % 4 == 0 else (0.0, 0.0)
        reach = rng.choice([rng.uniform(1.0, 3.0), rng.uniform(3, 50), rng.uniform(50, 2000)])
        n = int(rng.integers(0, 4000)) if j % 25 else 0
        span = per * rng.choice([0.5, 3, 40]) if j % 7 else per * 5000
        t = np.sort(t0 + rng.uniform(-span / 2, span / 2, n))
        extra = None
        if n > 10:  # put samples exactly at/around the window edges
            tdur = per / np.pi * np.arcsin(1 / a)
            if ecc > 0:
                tdur /= 1 / np.sqrt(1 - ecc ** 2) * (1 + ecc * np.cos((w - 90) / 180 * np.pi))
            e = np.round((t[n // 2] - t0) / per)
            mid = t0 + per * e + off
            edge = mid + np.array([-1, 1]) * reach * tdur / 2
            extra = np.concatenate([edge, np.nextafter(edge, np.inf), np.nextafter(edge, -np.inf)])
            # just outside/inside the edges, but inside the search margin
            # (pad >= 1e-6 tdur): the sorted search must trim/keep these
            near = np.concatenate([edge - 3e-7 * tdur, edge + 3e-7 * tdur])
            t = np.sort(np.concatenate([t[:-10], extra, near]))
            t[3] = t[2]                                   # duplicate time
        i_full, x_full = x_bary_window(t, a, per, t0, off, ed, ecc, w, reach)
        i_bin, x_bin = x_bary_window_sorted(t, a, per, t0, off, ed, ecc, w, reach)
        i_bin3, x_bin3, t_bin3 = x_bary_window_sorted_t(t, a, per, t0, off, ed, ecc, w, reach)
        assert np.array_equal(i_bin, i_bin3) and np.array_equal(x_bin, x_bin3, equal_nan=True)
        assert np.array_equal(t_bin3, t[i_bin3]), f"case {j}: window times differ"
        # P11: the sorted search computes x with the epoch terms hoisted
        # (closer to exact than the scan's fastmath form, by ~1e-13), so a
        # sample with |x| = reach within round-off may fall either way
        # (flux is exactly 1 there: reach has a 1e-6 margin). Any other
        # difference of the index sets is a bug.
        tdur_j = per / np.pi * np.arcsin(1 / a)
        if ecc > 0:
            tdur_j /= 1 / np.sqrt(1 - ecc ** 2) * (1 + ecc * np.cos((w - 90) / 180 * np.pi))
        # round-off of the scan's x (~ulp of t, scaled by 2 / tdur)
        tol = lambda tt: 16 * np.finfo(float).eps * ((np.abs(tt) + abs(t0)) * 2 / tdur_j + reach)
        diff = np.setxor1d(i_full, i_bin)
        if len(diff):
            xg = x_bary_grid(t[diff], a, per, t0, off, ed, ecc, w)
            assert np.all(np.abs(np.abs(xg) - reach) <= tol(t[diff])), \
                f"case {j}: index sets differ away from the edge ({len(i_full)} vs {len(i_bin)})"
            assert extra is not None and np.all(np.isin(t[diff], extra)), \
                f"case {j}: a random (non-edge) sample changed window membership"
            n_edge += len(diff)
        common, a_full, a_bin = np.intersect1d(i_full, i_bin, return_indices=True)
        dx = np.abs(x_full[a_full] - x_bin[a_bin])
        assert np.all((dx <= np.maximum(1e-12, tol(t[common]))) | np.isnan(dx)), f"case {j}: x differs"
        if n and not (reach * per / np.pi * np.arcsin(1 / a) / 2 + abs(off) < 0.45 * per):
            n_fallback += 1
    assert n_fallback >= 10, f"fallback path hit only {n_fallback} times"
    assert n_edge > 0, "the edge samples never hit the round-off case (test lost its edge)"
    # full model: sorted-time search == scan; Likelihood sorted == shuffled
    for j, args, cache, info in _random_draws(n_draws=30):
        if info["ssf"] != 1:
            continue
        t = np.sort(args[-1])
        a_sorted = args[:-1] + (t,)
        w0 = _call(pandora_window, a_sorted, cache)
        kw = dict(time_sorted=True)
        w1 = pandora_window(*a_sorted, cache, **kw) if cache is not None else pandora_window(*a_sorted, **kw)
        assert np.array_equal(w0[0], w1[0]), f"draw {j}: window cadences differ"
        for x, y in zip(w0[1:], w1[1:]):
            assert np.max(np.abs(x - y), initial=0) <= 1e-12, f"draw {j}: fluxes differ"
        rng = np.random.default_rng(j)
        data = 1 + rng.normal(0, 1e-4, len(t))
        perm = rng.permutation(len(t))
        (u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0, t0_off,
         M_planet, r_moon, per_moon, tau, Omega, i_moon, ecc_moon, w_moon, M_moon,
         epoch_distance, ssf, small, hill, _) = args
        params = (u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0,
                  t0_off, M_planet, r_moon, per_moon, tau, Omega, i_moon, ecc_moon, w_moon, M_moon)
        common = dict(epoch_distance=epoch_distance, occult_small_threshold=small,
                      hill_sphere_threshold=hill, cache=cache)
        l_sorted = Likelihood(t, data, 1e-4, **common)
        l_shuffled = Likelihood(t[perm], data[perm], 1e-4, **common)
        assert l_sorted.time_sorted and (len(t) < 3 or not l_shuffled.time_sorted)
        v1, v2 = l_sorted(*params), l_shuffled(*params)
        assert abs(v1 - v2) <= 1e-10 * abs(v1) + 1e-9, f"draw {j}: sorted {v1} vs shuffled {v2}"


def test_sincos_turns_and_fused_orbit():
    """P7: trig.sincos_turns(u) = (sin 2 pi u, cos 2 pi u) to <= 2.5e-16
    (since P24c; 4e-16 before) for |u| up to 1e14 turns (long-double reference with exact turn
    reduction), at quadrant boundaries and for NaN/inf; the fused circular
    orbit ellipse_z() (squared distances since P26; compared as z) matches
    ellipse() + sqrt to the ulp of ellipse()'s
    phase argument (measured 4.3e-12 R_star for |t| <= 3000 d, P >= 0.3 d,
    where 2 pi u reaches 6e4 rad with an ulp of 7e-12; tolerance 1e-11),
    and its loop is vectorised by LLVM."""
    from pandoramoon.trig import sincos_turns
    from pandoramoon.ellipse import ellipse_z
    import numba

    @numba.njit(fastmath=True)
    def batch(u):
        s, c = np.empty(len(u)), np.empty(len(u))
        for i in range(len(u)):
            s[i], c[i] = sincos_turns(u[i])
        return s, c

    rng = np.random.default_rng(7)
    PI = np.longdouble("3.14159265358979323846264338327950288")
    edges = np.arange(-4, 4.001, 0.125)
    edges = np.concatenate([edges, np.nextafter(edges, 10), np.nextafter(edges, -10), [0.0, -0.0, 1e-300]])
    for span in (1.0, 1e3, 1e8, 1e14):
        u = np.concatenate([rng.uniform(-span, span, 200_000), edges])
        s, c = batch(u)
        ul = u.astype(np.longdouble)
        th = 2 * PI * (ul - np.floor(ul + np.longdouble(0.5)))
        err = max(float(np.max(np.abs(s - np.sin(th)))), float(np.max(np.abs(c - np.cos(th)))))
        # P24c (quarter-turn reduction from u directly): measured 1.9e-16
        # (P7's two-step reduction: 3.5e-16)
        assert err <= 2.5e-16, f"|u| <= {span:g}: sincos_turns error {err:.2e}"
    s, c = batch(np.array([np.nan, np.inf, -np.inf]))
    assert np.all(np.isnan(s)) and np.all(np.isnan(c)), "non-finite phase must give NaN"
    worst = 0.0
    for _ in range(200):
        a, per, tau = rng.uniform(0.005, 0.5), rng.uniform(0.3, 40), rng.uniform()
        Om, inc, q, b = rng.uniform(0, 360), rng.uniform(0, 180), 10 ** rng.uniform(-5, -0.3), rng.uniform(0, 1.2)
        t = np.sort(rng.uniform(-1000, 3000, 2000))
        xb = rng.uniform(-2, 2, len(t))
        xm, ym, xp, yp = ellipse(a, per, tau, Om, inc, t, xb, q, b)
        out = list(ellipse_z(a, per, tau, Om, inc, t, xb, q, b))
        out[4:] = np.sqrt(out[4]), np.sqrt(out[5])     # P26: z^2
        ref = (xm, ym, xp, yp, np.hypot(xm, ym), np.hypot(xp, yp))
        worst = max(worst, max(float(np.max(np.abs(x - y))) for x, y in zip(out, ref)))
    assert worst <= 1e-11, f"ellipse_z vs ellipse: {worst:.2e} R_star"
    # cached code cannot be inspected: compile an uncached copy of the source
    fresh = numba.njit(fastmath=True)(ellipse_z.py_func)
    fresh(0.1, 3.0, 0.2, 30.0, 80.0, t, xb, 0.01, 0.3)
    assert "vector.body" in "".join(fresh.inspect_llvm().values()), "ellipse_z loop not vectorised"
    # P18: ellipse_z_runs (x_bary inline from the window runs) == ellipse_z
    # on the materialised window (runs_x_t), and vectorised
    from pandoramoon.ellipse import ellipse_z_runs
    from pandoramoon.grids import x_bary_runs_sorted, runs_x_t
    n_runs = 0
    for _ in range(100):
        per_b = rng.uniform(1, 400)
        a_b = rng.uniform(3, 300)
        ed = per_b + rng.uniform(-0.01, 0.01)
        t0 = rng.uniform(-50, 50)
        t = np.sort(rng.uniform(t0 - 3 * per_b, t0 + 5 * per_b, 5000))
        ok, los, his, ct, cd, tdur, sh = x_bary_runs_sorted(
            t, a_b, per_b, t0, rng.uniform(-0.05, 0.05), ed, 0.0, 0.0, rng.uniform(1.1, 5))
        if not ok:
            continue
        n_runs += 1
        _, xw, tw = runs_x_t(t, los, his, ct, cd, tdur, sh)
        a, per, tau = rng.uniform(0.005, 0.5), rng.uniform(0.3, 40), rng.uniform()
        Om, inc, q, b = rng.uniform(0, 360), rng.uniform(0, 180), 10 ** rng.uniform(-5, -0.3), rng.uniform(0, 1.2)
        ref = list(ellipse_z(a, per, tau, Om, inc, tw, xw, q, b))
        out = list(ellipse_z_runs(a, per, tau, Om, inc, t, los, his, ct, cd, tdur, sh, q, b))
        for v in (ref, out):
            v[4:] = np.sqrt(v[4]), np.sqrt(v[5])        # P26: z^2
        d = max(float(np.max(np.abs(x - y), initial=0)) for x, y in zip(out, ref))
        # same x_bary expression in both, but LLVM may re-associate it
        # differently per function under fastmath (P11 pitfall): allow a few
        # ulp of 2 t / tdur (found in P19: up to 1.2e-12 R_star over 3000
        # draws; the earlier 1e-13 passed by luck of these 100 draws)
        tol = 32 * np.finfo(float).eps * (2 * (np.max(np.abs(t)) + np.max(np.abs(ct), initial=0)) / tdur + 10)
        assert d <= tol, f"ellipse_z_runs vs ellipse_z: {d:.2e} (tol {tol:.1e})"
    assert n_runs > 50
    fresh = numba.njit(fastmath=True)(ellipse_z_runs.py_func)
    fresh(0.1, 3.0, 0.2, 30.0, 80.0, t, los, his, ct, cd, tdur, sh, 0.01, 0.3)
    import re
    ir = "".join(fresh.inspect_llvm().values())
    assert _vector_double_loops(ir) >= 2, "ellipse_z_runs loops not vectorised"
    # P24a: the quadrant select stays in float lanes (AVX2 has no vector
    # float -> int64 conversion; LLVM scalarised it, ~0.4 ns per sample)
    assert not re.search(r"fptosi <\d+ x double>", ir), "vector float->int conversion in the orbit loop"


def test_orbit_block_anchors():
    """P24d: trig.sincos_small(phi) = (sin phi, cos phi) to <= 1.5e-16 on
    |phi| <= SMALL_PHI_MAX (long-double reference; measured 8.8e-17), and
    the orbit loops on the runs (ellipse_z_runs, ellipse_z_only_runs:
    sincos_turns at block anchors, sincos_small of omega (t - t_anchor)
    per sample) match a long-double evaluation of the circular orbit to
    eps (2 pi a |u| + |x_bary|) (the ulp of the phase u and of x_bary;
    measured 0.27 of it, as for 54be54e's per-sample sincos_turns), with
    dense and coarse sampling, gaps and repeated times inside runs (blocks
    whose span exceeds the polynomial range use sincos_turns), empty and
    single-sample runs; both loops of each function vectorise (since P26
    they return z^2 without a sqrt: counted as vector.body blocks with
    double vector arithmetic).
    Mutations caught (scratch_p24d/mutate.py): SS4 or SC5 dropped, range
    0.5 with the 0.3 coefficients, no / one-sided / halved span check,
    wrong sign in the folded projection."""
    import re
    import numba
    from pandoramoon.trig import sincos_small, SMALL_PHI_MAX
    from pandoramoon import ellipse as el

    L = np.longdouble
    PI = L("3.14159265358979323846264338327950288")

    @numba.njit(fastmath=True)
    def batch(p):
        s, c = np.empty(len(p)), np.empty(len(p))
        for i in range(len(p)):
            s[i], c[i] = sincos_small(p[i])
        return s, c

    rng = np.random.default_rng(11)
    m = SMALL_PHI_MAX
    p = np.concatenate([rng.uniform(-m, m, 200_000), np.linspace(-m, m, 10001), [0.0, -0.0, 1e-300]])
    s, c = batch(p)
    pl = p.astype(L)
    err = max(float(np.max(np.abs(s - np.sin(pl)))), float(np.max(np.abs(c - np.cos(pl)))))
    assert err <= 1.5e-16, f"sincos_small error {err:.2e}"

    def ref(a, per, tau, Om, inc, t, los, his, ct, cd, tdur, sh, q, b):
        O, inc_ = L(Om) / 180 * PI, L(inc) / 180 * PI
        ap = (L(a) * L(q)) / (1 + L(q))
        am = L(a) - ap
        out = [[] for _ in range(6)]
        for k in range(len(los)):
            tk = t[los[k]:his[k]].astype(L)
            xb = (2 * (tk - L(ct[k]))) / L(tdur) - L(sh) - L(cd[k])
            u = (tk - L(tau) * L(per)) / L(per)
            Q = 2 * PI * (u - np.floor(u + L(0.5)))
            vx = np.cos(O) * np.cos(Q) - np.sin(O) * np.sin(Q) * np.cos(inc_)
            vy = np.sin(O) * np.cos(Q) + np.cos(O) * np.sin(Q) * np.cos(inc_)
            xm, ym, xp, yp = vx * am + xb, vy * am + L(b), -vx * ap + xb, -vy * ap + L(b)
            for o, v in zip(out, (xm, ym, xp, yp, np.sqrt(xm ** 2 + ym ** 2), np.sqrt(xp ** 2 + yp ** 2))):
                o.append(v)
        return [np.concatenate(o) for o in out]

    worst = 0.0
    n_blocks = [0, 0]  # sincos_turns blocks, sincos_small blocks
    for _ in range(300):
        a, per, tau = rng.uniform(0.005, 0.5), 10 ** rng.uniform(-0.5, 1.5), rng.uniform()
        Om, inc, q, b = rng.uniform(0, 360), rng.uniform(0, 180), 10 ** rng.uniform(-5, -0.3), rng.uniform(0, 1.2)
        dt = 10 ** rng.uniform(-3.5, -1.3)
        nr = int(rng.integers(1, 6))
        t0 = rng.uniform(-1000, 3000)
        segs, los, his, n = [], [], [], 0
        for r in range(nr):
            cnt = int(rng.integers(0 if r else 1, 600))
            tt = t0 + r * 50 + dt * np.arange(cnt)
            if cnt > 10 and rng.uniform() < 0.3:  # gap inside the run
                tt[rng.integers(1, cnt):] += rng.uniform(0, 30) * dt * rng.integers(1, 20)
            if cnt > 10 and rng.uniform() < 0.2:  # repeated time
                g = rng.integers(1, cnt)
                tt[g] = tt[g - 1]
            segs.append(tt)
            los.append(n)
            his.append(n + cnt)
            n += cnt
        t, los, his = np.concatenate(segs), np.array(los), np.array(his)
        ct = t0 + 50 * np.arange(nr) + rng.uniform(0, 1, nr)
        cd = rng.uniform(-0.01, 0.01, nr)
        tdur, sh = rng.uniform(0.1, 1), rng.uniform(-0.01, 0.01)
        omega = 2 * np.pi / per
        for k in range(nr):
            cnt = his[k] - los[k]
            bl = el._anchor_block(t[los[k]:his[k]], cnt, omega)
            for b0 in range(0, cnt, bl):
                tb = t[los[k] + b0:los[k] + min(b0 + bl, cnt)]
                ta = tb[len(tb) // 2]
                n_blocks[int(max(ta - tb[0], tb[-1] - ta) * omega <= SMALL_PHI_MAX)] += 1
        out = list(el.ellipse_z_runs(a, per, tau, Om, inc, t, los, his, ct, cd, tdur, sh, q, b))
        zo = el.ellipse_z_only_runs(a, per, tau, Om, inc, t, los, his, ct, cd, tdur, sh, q, b)
        assert np.array_equal(zo[0], out[4]) and np.array_equal(zo[1], out[5]), "z-only loop differs"
        out[4:] = np.sqrt(out[4]), np.sqrt(out[5])      # P26: z^2
        R = ref(a, per, tau, Om, inc, t, los, his, ct, cd, tdur, sh, q, b)
        u_max = np.max(np.abs(t)) / per + abs(tau)
        x_max = 2 * (np.max(np.abs(t)) + np.max(np.abs(ct))) / tdur + 2
        tol = np.finfo(float).eps * (2 * np.pi * a * u_max + x_max)
        d = max(float(np.max(np.abs(x - y), initial=0)) for x, y in zip(out, R))
        worst = max(worst, d / tol)
    assert worst <= 1.0, f"orbit with block anchors: error {worst:.2f} x tolerance"
    # measured 419 / 1742 (ANCHOR_BMIN = 32)
    assert n_blocks[1] > 1000 and n_blocks[0] > 100, f"blocks (sincos_turns, sincos_small): {n_blocks}"
    # both loops (anchored and per-sample sincos_turns) of each function vectorise
    tt = np.linspace(0.0, 1.0, 100)
    for f in (el.ellipse_z_runs, el.ellipse_z_only_runs):
        fresh = numba.njit(fastmath=True)(f.py_func)
        fresh(0.1, 3.0, 0.2, 30.0, 80.0, tt, np.array([0]), np.array([100]), np.array([0.5]), np.array([0.0]), 0.5, 0.0, 0.01, 0.3)
        ir = "".join(fresh.inspect_llvm().values())
        n_vec = _vector_double_loops(ir)
        assert n_vec >= 2, f"{f.__name__}: {n_vec} vectorised double loops (expected 2)"
        assert not re.search(r"fptosi <\d+ x double>", ir), f"{f.__name__}: vector float->int conversion"


def test_acos_poly():
    """P16: trig.acos_poly(x), the branch-free fdlibm arccos used for the
    limb terms of _occult_exact_batch, is within 1 ulp of libm on [-1, 1]
    (uniform, near +-1 down to 1 ulp, at +-1/2 and 0), within 1.5 ulp of a
    long-double reference (measured 1.05), exact at +-1, and a loop over it
    is vectorised."""
    from pandoramoon.trig import acos_poly
    import numba

    def batch_src(x, out):
        for i in range(len(x)):
            out[i] = acos_poly(x[i])
    batch = numba.njit(fastmath=True, error_model="numpy")(batch_src)

    rng = np.random.default_rng(16)
    t = 10 ** rng.uniform(-16, 0, 50_000)
    edges = np.array([-1.0, 1.0, -0.5, 0.5, 0.0, -0.0, 1e-300, -1e-300])
    edges = np.concatenate([edges, np.nextafter(edges, 2), np.nextafter(edges, -2)])
    x = np.concatenate([rng.uniform(-1, 1, 200_000), 1 - t, -1 + t, edges])
    x = x[np.abs(x) <= 1]
    out = np.empty_like(x)
    batch(x, out)
    ref = np.arccos(x)
    ulp = np.abs(out - ref) / np.spacing(np.maximum(ref, 1e-300))
    assert ulp.max() <= 1.0, f"acos_poly vs libm: {ulp.max():.1f} ulp at x = {x[ulp.argmax()]!r}"
    assert out[x == 1.0].max() == 0.0 and np.all(out[x == -1.0] == np.pi), "acos_poly(+-1)"
    xs = x[::97]
    refl = np.arccos(xs.astype(np.longdouble))
    ulpl = np.abs(out[::97] - refl) / np.spacing(np.maximum(refl, 1e-300).astype(np.float64))
    assert float(ulpl.max()) <= 1.5, f"acos_poly vs long double: {float(ulpl.max()):.2f} ulp"
    # cached code cannot be inspected: compile an uncached copy
    fresh = numba.njit(fastmath=True, error_model="numpy")(batch_src)
    fresh(x[:64], out[:64])
    assert "vector.body" in "".join(fresh.inspect_llvm().values()), "acos_poly loop not vectorised"


# mpmath 1.4.1 references (dps 30), generated by scratch_p3/gen_test_values.py;
# embedded so that the tests do not need mpmath.
ELL_REF = [  # (q, n, K, E, Pi) from mpmath 1.4.1, dps 30
    (0.0, 10000.0, 1.5707963267948966, 1.5707963267948966, 0.015707177928685524),
    (1e-06, 40.0, 1.5707963267952894, 1.570796326794504, 0.2453171715163798),
    (0.1, 3.0, 1.574745561517356, 1.5668619420216683, 0.786713748823911),
    (0.3, 0.5, 1.6080486199305128, 1.5348334649232491, 1.3098448759814962),
    (0.5, 0.5, 1.685750354812596, 1.4674622093394272, 1.3664739530045968),
    (0.7, 0.001, 1.8456939983747234, 1.3556611355719554, 1.8446947024281666),
    (0.9, 0.001, 2.2805491384227703, 1.1716970527816142, 2.2791812851258992),
    (0.99, 0.001, 3.356600523361192, 1.028475809028804, 3.3542271841774034),
    (0.999, 0.001, 4.495596395842144, 1.0039944099655078, 4.492100965128268),
    (0.99999, 10000.0, 6.7962149844353315, 1.0000629619736887, 0.01628595322424193),
    (0.99999999, 40.0, 10.250061189054028, 1.000000097500612, 0.46811853377686746),
]
FLUX_REF = [  # (z, k, u1, u2, flux) from mpmath radial quadrature, dps 30
    (0.0, 0.003, 0.4, 0.25, 0.9999890909189091),
    (0.2991, 0.003, 0.4, 0.25, 0.9999892963960456),
    (0.0045000000000000005, 0.003, 0.4, 0.25, 0.9999890909630919),
    (0.7976000000000001, 0.003, 0.4, 0.25, 0.9999912519526962),
    (0.996999, 0.003, 0.4, 0.25, 0.9999954684324002),
    (0.997, 0.003, 0.4, 0.25, 0.9999954685788042),
    (0.997001, 0.003, 0.4, 0.25, 0.9999954687393463),
    (1.0, 0.003, 0.4, 0.25, 0.999997863827276),
    (1.0029, 0.003, 0.4, 0.25, 0.9999999858470948),
    (0.0, 0.003, 0.2, 0.8, 0.9999887500050626),
    (0.2991, 0.003, 0.2, 0.8, 0.9999888718714227),
    (0.0045000000000000005, 0.003, 0.2, 0.8, 0.9999887500278458),
    (0.7976000000000001, 0.003, 0.2, 0.8, 0.9999910600778085),
    (0.996999, 0.003, 0.2, 0.8, 0.9999985488529927),
    (0.997, 0.003, 0.2, 0.8, 0.9999985491482212),
    (0.997001, 0.003, 0.2, 0.8, 0.9999985494438023),
    (1.0, 0.003, 0.2, 0.8, 0.9999995334266583),
    (1.0029, 0.003, 0.2, 0.8, 0.9999999993899137),
    (0.0, 0.003, 0.9, -0.1, 0.9999874418858954),
    (0.2991, 0.003, 0.9, -0.1, 0.9999879566565592),
    (0.0045000000000000005, 0.003, 0.9, -0.1, 0.9999874420003323),
    (0.7976000000000001, 0.003, 0.9, -0.1, 0.9999917290988406),
    (0.996999, 0.003, 0.9, -0.1, 0.9999968274994618),
    (0.997, 0.003, 0.9, -0.1, 0.9999968276379178),
    (0.997001, 0.003, 0.9, -0.1, 0.9999968277857143),
    (1.0, 0.003, 0.9, -0.1, 0.999998535880199),
    (1.0029, 0.003, 0.9, -0.1, 0.9999999906186628),
    (0.0, 0.01, 0.4, 0.25, 0.9998787890909545),
    (0.297, 0.01, 0.4, 0.25, 0.9998810388148404),
    (0.015, 0.01, 0.4, 0.25, 0.9998787945468043),
    (0.792, 0.01, 0.4, 0.25, 0.9999022752002927),
    (0.989999, 0.01, 0.4, 0.25, 0.9999434135503777),
    (0.99, 0.01, 0.4, 0.25, 0.999943414411551),
    (0.990001, 0.01, 0.4, 0.25, 0.9999434152986707),
    (1.0, 0.01, 0.4, 0.25, 0.9999742720553754),
    (1.0099, 0.01, 0.4, 0.25, 0.999999974159542),
    (0.0, 0.01, 0.2, 0.8, 0.9998750006250937),
    (0.297, 0.01, 0.2, 0.8, 0.9998763328431083),
    (0.015, 0.01, 0.2, 0.8, 0.9998750034402838),
    (0.792, 0.01, 0.2, 0.8, 0.9998999183256039),
    (0.989999, 0.01, 0.2, 0.8, 0.999971534896904),
    (0.99, 0.01, 0.2, 0.8, 0.9999715365988238),
    (0.990001, 0.01, 0.2, 0.8, 0.9999715383014761),
    (1.0, 0.01, 0.2, 0.8, 0.9999907514922328),
    (1.0099, 0.01, 0.2, 0.8, 0.9999999988867427),
    (0.0, 0.01, 0.9, -0.1, 0.9998604682558546),
    (0.297, 0.01, 0.9, -0.1, 0.9998661066342175),
    (0.015, 0.01, 0.9, -0.1, 0.9998604823849294),
    (0.792, 0.01, 0.9, -0.1, 0.9999072678281691),
    (0.989999, 0.01, 0.9, -0.1, 0.9999585972835243),
    (0.99, 0.01, 0.9, -0.1, 0.9999585981359235),
    (0.990001, 0.01, 0.9, -0.1, 0.999958599005475),
    (1.0, 0.01, 0.9, -0.1, 0.9999818192362964),
    (1.0099, 0.01, 0.9, -0.1, 0.9999999828715752),
    (0.0, 0.03, 0.4, 0.25, 0.9989091891240567),
    (0.291, 0.03, 0.4, 0.25, 0.998928605088892),
    (0.045, 0.03, 0.4, 0.25, 0.9989096318946732),
    (0.776, 0.03, 0.4, 0.25, 0.9991079081338963),
    (0.969999, 0.03, 0.4, 0.25, 0.9994057055276507),
    (0.97, 0.03, 0.4, 0.25, 0.9994057097370896),
    (0.970001, 0.03, 0.4, 0.25, 0.9994057139919977),
    (1.0, 0.03, 0.4, 0.25, 0.9997412006749816),
    (1.0299, 0.03, 0.4, 0.25, 0.999999955635431),
    (0.0, 0.03, 0.2, 0.8, 0.9988750506933668),
    (0.291, 0.03, 0.2, 0.8, 0.9988865162833399),
    (0.045, 0.03, 0.2, 0.8, 0.9988752804702635),
    (0.776, 0.03, 0.2, 0.8, 0.999081674251161),
    (0.969999, 0.03, 0.2, 0.8, 0.9995815626805293),
    (0.97, 0.03, 0.2, 0.8, 0.9995815707077473),
    (0.970001, 0.03, 0.2, 0.8, 0.999581578736385),
    (1.0, 0.03, 0.2, 0.8, 0.9998616911814185),
    (1.0299, 0.03, 0.2, 0.8, 0.9999999980890001),
    (0.0, 0.03, 0.9, -0.1, 0.9987444403785158),
    (0.291, 0.03, 0.9, -0.1, 0.9987931367167987),
    (0.045, 0.03, 0.9, -0.1, 0.9987455855917423),
    (0.776, 0.03, 0.9, -0.1, 0.9991450282919769),
    (0.969999, 0.03, 0.9, -0.1, 0.9995369092318271),
    (0.97, 0.03, 0.9, -0.1, 0.9995369137180918),
    (0.970001, 0.03, 0.9, -0.1, 0.9995369182344308),
    (1.0, 0.03, 0.9, -0.1, 0.9998087193109083),
    (1.0299, 0.03, 0.9, -0.1, 0.9999999705928498),
    (0.0, 0.06, 0.4, 0.25, 0.99563793666905),
    (0.282, 0.06, 0.4, 0.25, 0.9957108835173012),
    (0.09, 0.06, 0.4, 0.25, 0.9956450670718215),
    (0.752, 0.06, 0.4, 0.25, 0.9963649585906936),
    (0.9399989999999999, 0.06, 0.4, 0.25, 0.9973195770798886),
    (0.94, 0.06, 0.4, 0.25, 0.9973195882231154),
    (0.940001, 0.06, 0.4, 0.25, 0.9973195994317239),
    (1.0, 0.06, 0.4, 0.25, 0.9988675219891824),
    (1.0599, 0.06, 0.4, 0.25, 0.9999999381378952),
    (0.0, 0.06, 0.2, 0.8, 0.9955008143799156),
    (0.282, 0.06, 0.2, 0.8, 0.9955439098544336),
    (0.09, 0.06, 0.2, 0.8, 0.9955045858356321),
    (0.752, 0.06, 0.2, 0.8, 0.9962368666933056),
    (0.9399989999999999, 0.06, 0.2, 0.8, 0.997776340265269),
    (0.94, 0.06, 0.2, 0.8, 0.9977763607101756),
    (0.940001, 0.06, 0.2, 0.8, 0.9977763811572439),
    (1.0, 0.06, 0.2, 0.8, 0.999252380524641),
    (1.0599, 0.06, 0.2, 0.8, 0.9999999973354039),
    (0.0, 0.06, 0.9, -0.1, 0.9949808149246145),
    (0.282, 0.06, 0.9, -0.1, 0.9951637493445603),
    (0.09, 0.06, 0.9, -0.1, 0.9949991795875612),
    (0.752, 0.06, 0.9, -0.1, 0.9964681730265512),
    (0.9399989999999999, 0.06, 0.9, -0.1, 0.997793372976925),
    (0.94, 0.06, 0.9, -0.1, 0.9977933857346498),
    (0.940001, 0.06, 0.9, -0.1, 0.9977933985356354),
    (1.0, 0.06, 0.9, -0.1, 0.9991287794640135),
    (1.0599, 0.06, 0.9, -0.1, 0.9999999589945834),
    (0.0, 0.1, 0.4, 0.25, 0.9878909547167664),
    (0.27, 0.1, 0.4, 0.25, 0.9880773895795162),
    (0.15000000000000002, 0.1, 0.4, 0.25, 0.9879468328723794),
    (0.7200000000000001, 0.1, 0.4, 0.25, 0.9896925496195617),
    (0.899999, 0.1, 0.4, 0.25, 0.9918036898788049),
    (0.9, 0.1, 0.4, 0.25, 0.9918037122033105),
    (0.900001, 0.1, 0.4, 0.25, 0.991803734614136),
    (1.0, 0.1, 0.4, 0.25, 0.9966143500998468),
    (1.0999, 0.1, 0.4, 0.25, 0.9999999215941832),
    (0.0, 0.1, 0.2, 0.8, 0.9875063441033306),
    (0.27, 0.1, 0.2, 0.8, 0.9876174910077522),
    (0.15000000000000002, 0.1, 0.2, 0.8, 0.9875372177992824),
    (0.7200000000000001, 0.1, 0.2, 0.8, 0.989278133590346),
    (0.899999, 0.1, 0.2, 0.8, 0.9925388671250313),
    (0.9, 0.1, 0.2, 0.8, 0.9925389064193278),
    (0.900001, 0.1, 0.2, 0.8, 0.9925389457165904),
    (1.0, 0.1, 0.2, 0.8, 0.9974491327957676),
    (1.0999, 0.1, 0.2, 0.8, 0.9999999966228691),
    (0.0, 0.1, 0.9, -0.1, 0.9860779478278024),
    (0.27, 0.1, 0.9, -0.1, 0.9865443836932878),
    (0.15000000000000002, 0.1, 0.9, -0.1, 0.9862204201299211),
    (0.7200000000000001, 0.1, 0.9, -0.1, 0.9898208641942587),
    (0.899999, 0.1, 0.9, -0.1, 0.992909110286841),
    (0.9, 0.1, 0.9, -0.1, 0.9929091376704059),
    (0.900001, 0.1, 0.9, -0.1, 0.9929091651111032),
    (1.0, 0.1, 0.9, -0.1, 0.9972976122249572),
    (1.0999, 0.1, 0.9, -0.1, 0.9999999480285582),
    (0.0, 0.2, 0.4, 0.25, 0.951712044529993),
    (0.24, 0.2, 0.4, 0.25, 0.9523170122993064),
    (0.30000000000000004, 0.2, 0.4, 0.25, 0.9526763119111793),
    (0.6400000000000001, 0.2, 0.4, 0.25, 0.9572083848543652),
    (0.799999, 0.2, 0.4, 0.25, 0.9626214855955035),
    (0.8, 0.2, 0.4, 0.25, 0.9626215402798207),
    (0.8000010000000001, 0.2, 0.4, 0.25, 0.9626215950937256),
    (1.0, 0.2, 0.4, 0.25, 0.9850185384656939),
    (1.1999, 0.2, 0.4, 0.25, 0.9999998938304475),
    (0.0, 0.2, 0.2, 0.8, 0.9501060918431106),
    (0.24, 0.2, 0.2, 0.8, 0.9504898378673196),
    (0.30000000000000004, 0.2, 0.2, 0.8, 0.9507448933168073),
    (0.6400000000000001, 0.2, 0.2, 0.8, 0.9552982937621283),
    (0.799999, 0.2, 0.2, 0.8, 0.9630068380407111),
    (0.8, 0.2, 0.2, 0.8, 0.9630069268271303),
    (0.8000010000000001, 0.2, 0.2, 0.8, 0.9630070156182021),
    (1.0, 0.2, 0.2, 0.8, 0.9869843289385613),
    (1.1999, 0.2, 0.2, 0.8, 0.999999995427073),
    (0.0, 0.2, 0.9, -0.1, 0.9446910166140635),
    (0.24, 0.2, 0.9, -0.1, 0.9461792784540302),
    (0.30000000000000004, 0.2, 0.9, -0.1, 0.9470334420600094),
    (0.6400000000000001, 0.2, 0.9, -0.1, 0.9563435693851099),
    (0.799999, 0.2, 0.9, -0.1, 0.9649765751494789),
    (0.8, 0.2, 0.9, -0.1, 0.9649766504055759),
    (0.8000010000000001, 0.2, 0.9, -0.1, 0.9649767257474887),
    (1.0, 0.2, 0.9, -0.1, 0.9872814264731766),
    (1.1999, 0.2, 0.9, -0.1, 0.9999999296253209),
    (0.0, 0.5, 0.4, 0.25, 0.7053320384278756),
    (0.15, 0.5, 0.4, 0.25, 0.7072043628947146),
    (0.75, 0.5, 0.4, 0.25, 0.81016582053068),
    (0.4, 0.5, 0.4, 0.25, 0.720477808390602),
    (0.499999, 0.5, 0.4, 0.25, 0.7319834254418621),
    (0.5, 0.5, 0.4, 0.25, 0.7319835811800197),
    (0.500001, 0.5, 0.4, 0.25, 0.7319837371775435),
    (1.0, 0.5, 0.4, 0.25, 0.8969882617777193),
    (1.4999, 0.5, 0.4, 0.25, 0.9999998498479715),
    (0.0, 0.5, 0.2, 0.8, 0.6930285792574935),
    (0.15, 0.5, 0.2, 0.8, 0.6947808734703489),
    (0.75, 0.5, 0.2, 0.8, 0.8093603865983434),
    (0.4, 0.5, 0.2, 0.8, 0.7092667298056168),
    (0.499999, 0.5, 0.2, 0.8, 0.7245596587237468),
    (0.5, 0.5, 0.2, 0.8, 0.7245598861837906),
    (0.500001, 0.5, 0.2, 0.8, 0.7245601136534059),
    (1.0, 0.5, 0.2, 0.8, 0.8996436303562566),
    (1.4999, 0.5, 0.2, 0.8, 0.9999999935327112),
    (0.0, 0.5, 0.9, -0.1, 0.6714891506854235),
    (0.15, 0.5, 0.9, -0.1, 0.6754765807313142),
    (0.75, 0.5, 0.9, -0.1, 0.8116403650282319),
    (0.4, 0.5, 0.9, -0.1, 0.701484006737322),
    (0.499999, 0.5, 0.9, -0.1, 0.7210296309189792),
    (0.5, 0.5, 0.9, -0.1, 0.7210298730720331),
    (0.500001, 0.5, 0.9, -0.1, 0.7210301153970292),
    (1.0, 0.5, 0.9, -0.1, 0.9032332946481151),
    (1.4999, 0.5, 0.9, -0.1, 0.9999999004714633),
    (0.0, 0.7, 0.4, 0.25, 0.44183661807686214),
    (0.09000000000000001, 0.7, 0.4, 0.25, 0.4436865504347832),
    (1.0499999999999998, 0.7, 0.4, 0.25, 0.8199065174486454),
    (0.24000000000000005, 0.7, 0.4, 0.25, 0.456111621669793),
    (0.29999900000000007, 0.7, 0.4, 0.25, 0.46590913918023374),
    (0.30000000000000004, 0.7, 0.4, 0.25, 0.4659093471258367),
    (0.300001, 0.7, 0.4, 0.25, 0.4659095554676112),
    (1.0, 0.7, 0.4, 0.25, 0.7975779764026434),
    (1.6999, 0.7, 0.4, 0.25, 0.9999998331145145),
    (0.0, 0.7, 0.2, 0.8, 0.4162692747835281),
    (0.09000000000000001, 0.7, 0.2, 0.8, 0.4185963852717403),
    (1.0499999999999998, 0.7, 0.2, 0.8, 0.8217178117588055),
    (0.24000000000000005, 0.7, 0.2, 0.8, 0.435127469693948),
    (0.29999900000000007, 0.7, 0.2, 0.8, 0.44938145980904826),
    (0.30000000000000004, 0.7, 0.2, 0.8, 0.4493817784470381),
    (0.300001, 0.7, 0.2, 0.8, 0.4493820970995332),
    (1.0, 0.7, 0.2, 0.8, 0.7987309621253518),
    (1.6999, 0.7, 0.2, 0.8, 0.9999999928119848),
    (0.0, 0.7, 0.9, -0.1, 0.39763394874323704),
    (0.09000000000000001, 0.7, 0.9, -0.1, 0.4009209579474222),
    (1.0499999999999998, 0.7, 0.9, -0.1, 0.82577387637495),
    (0.24000000000000005, 0.7, 0.9, -0.1, 0.4220109635880705),
    (0.29999900000000007, 0.7, 0.9, -0.1, 0.43730573638250364),
    (0.30000000000000004, 0.7, 0.9, -0.1, 0.4373060433568538),
    (0.300001, 0.7, 0.9, -0.1, 0.4373063505939237),
    (1.0, 0.7, 0.9, -0.1, 0.8023748146757033),
    (1.6999, 0.7, 0.9, -0.1, 0.999999889379663),
]


CONTACT_REF = [  # (z, k, flux) at u = (0.4, 0.25), mpmath radial quadrature, dps 30 (scratch_p15)
    (0.001000000000001, 0.001, 0.9999987878791515),
    (0.000999999999999, 0.001, 0.9999987878791515),
    (0.999000000000001, 0.001, 0.9999995295279969),
    (0.998999999999999, 0.001, 0.9999995295279969),
    (0.001000000001, 0.001, 0.9999987878791515),
    (0.000999999999, 0.001, 0.9999987878791515),
    (0.999000000001, 0.001, 0.9999995295279969),
    (0.998999999999, 0.001, 0.9999995295279968),
    (0.001000001, 0.001, 0.9999987878791515),
    (0.0009999990000000001, 0.001, 0.9999987878791515),
    (0.999000001, 0.001, 0.9999995295280257),
    (0.998999999, 0.001, 0.9999995295279681),
    (0.001001, 0.001, 0.999998787879152),
    (0.000999, 0.001, 0.9999987878791511),
    (0.999001, 0.001, 0.9999995295647575),
    (0.998999, 0.001, 0.9999995294993778),
    (0.002, 0.001, 0.9999987878798788),
    (1.0, 0.001, 0.9999997731730672),
    (0.998, 0.001, 0.9999995085727309),
    (1.000999, 0.001, 0.9999999999919358),
    (0.9999999999999999, 0.001, 0.9999997731730672),
    (0.009200000000000999, 0.0092, 0.999897408665947),
    (0.009199999999999, 0.0092, 0.999897408665947),
    (0.990800000000001, 0.0092, 0.9999525722904858),
    (0.990799999999999, 0.0092, 0.9999525722904858),
    (0.009200000001, 0.0092, 0.999897408665947),
    (0.009199999999, 0.0092, 0.999897408665947),
    (0.990800000001, 0.0092, 0.9999525722904866),
    (0.990799999999, 0.0092, 0.9999525722904851),
    (0.009200001, 0.0092, 0.9998974086659473),
    (0.009199999, 0.0092, 0.9998974086659467),
    (0.990800001, 0.0092, 0.9999525722912491),
    (0.990799999, 0.0092, 0.9999525722897232),
    (0.009200999999999999, 0.0092, 0.9998974086663247),
    (0.009199, 0.0092, 0.9998974086655694),
    (0.990801, 0.0092, 0.9999525730777249),
    (0.990799, 0.0092, 0.999952571528121),
    (0.0102, 0.0092, 0.9998974090640916),
    (0.008199999999999999, 0.0092, 0.9998974083088557),
    (0.9918, 0.0092, 0.9999541501460757),
    (0.9898, 0.0092, 0.9999518643946211),
    (1.0091990000000002, 0.0092, 0.9999999999756362),
    (1.0082000000000002, 0.0092, 0.999999192681262),
    (0.100000000000001, 0.1, 0.9879156121362547),
    (0.099999999999999, 0.1, 0.9879156121362547),
    (0.900000000000001, 0.1, 0.9918037122033105),
    (0.899999999999999, 0.1, 0.9918037122033104),
    (0.10000000000100001, 0.1, 0.9879156121362552),
    (0.099999999999, 0.1, 0.9879156121362542),
    (0.900000000001, 0.1, 0.9918037122033329),
    (0.899999999999, 0.1, 0.9918037122032881),
    (0.100000001, 0.1, 0.9879156121367506),
    (0.099999999, 0.1, 0.9879156121357587),
    (0.900000001, 0.1, 0.9918037122256385),
    (0.899999999, 0.1, 0.9918037121809853),
    (0.100001, 0.1, 0.9879156126322243),
    (0.099999, 0.1, 0.9879156116402902),
    (0.900001, 0.1, 0.991803734614136),
    (0.899999, 0.1, 0.9918036898788049),
    (0.101, 0.1, 0.9879161106403518),
    (0.099, 0.1, 0.987915118705161),
    (0.901, 0.1, 0.9918290384419695),
    (0.899, 0.1, 0.9917817086217675),
    (1.0999990000000002, 0.1, 0.9999999999230603),
    (1.0990000000000002, 0.1, 0.9999974122850465),
    (0.200000000000001, 0.2, 0.952127708253069),
    (0.199999999999999, 0.2, 0.952127708253069),
    (0.800000000000001, 0.2, 0.9626215402798208),
    (0.799999999999999, 0.2, 0.9626215402798207),
    (0.20000000000100002, 0.2, 0.9521277082530732),
    (0.199999999999, 0.2, 0.9521277082530647),
    (0.800000000001, 0.2, 0.9626215402798755),
    (0.7999999999990001, 0.2, 0.9626215402797661),
    (0.200000001, 0.2, 0.9521277082573258),
    (0.199999999, 0.2, 0.9521277082488122),
    (0.800000001, 0.2, 0.9626215403345102),
    (0.7999999990000001, 0.2, 0.9626215402251352),
    (0.200001, 0.2, 0.9521277125098745),
    (0.199999, 0.2, 0.9521277039962869),
    (0.8000010000000001, 0.2, 0.9626215950937256),
    (0.799999, 0.2, 0.9626214855955035),
    (0.201, 0.2, 0.9521319767191858),
    (0.199, 0.2, 0.9521234631208438),
    (0.801, 0.2, 0.9626807843406644),
    (0.799, 0.2, 0.962567390872914),
    (1.199999, 0.2, 0.9999999998958231),
    (1.199, 0.2, 0.9999964937292187),
]


K05_REF = [  # (z, k, u1, u2, flux) near k = 0.5, mpmath radial quadrature, dps 30 (scratch_k05, bug #30)
    (0.0, 0.49991, 0.4, 0.25, 0.7054347857246447),
    (0.3, 0.49991, 0.4, 0.25, 0.7133751424733266),
    (0.4999099, 0.49991, 0.4, 0.25, 0.7320566423364765),
    (0.4999101, 0.49991, 0.4, 0.25, 0.7320566733798075),
    (0.5000899999999999, 0.49991, 0.4, 0.25, 0.7320846386721803),
    (0.8, 0.49991, 0.4, 0.25, 0.8281101481337687),
    (1.2, 0.49991, 0.4, 0.25, 0.954730446784811),
    (0.0, 0.49991, 2.0, -1.0, 0.562634995948542),
    (0.3, 0.49991, 2.0, -1.0, 0.607618797406542),
    (0.4999099, 0.49991, 2.0, -1.0, 0.6875449702726263),
    (0.4999101, 0.49991, 2.0, -1.0, 0.6875450702186361),
    (0.5000899999999999, 0.49991, 2.0, -1.0, 0.6876349878485422),
    (0.8, 0.49991, 2.0, -1.0, 0.8412168691276356),
    (1.2, 0.49991, 2.0, -1.0, 0.9762519359069428),
    (0.0, 0.499999, 0.4, 0.25, 0.7053331801515802),
    (0.3, 0.499999, 0.4, 0.25, 0.7132773878817719),
    (0.4999989, 0.499999, 0.4, 0.25, 0.7319843770964096),
    (0.49999910000000003, 0.499999, 0.4, 0.25, 0.7319844082427253),
    (0.5000009999999999, 0.499999, 0.4, 0.25, 0.7319847041409898),
    (0.8, 0.499999, 0.4, 0.25, 0.8280598458045004),
    (1.2, 0.499999, 0.4, 0.25, 0.9547052426311802),
    (0.0, 0.499999, 2.0, -1.0, 0.5625014999994999),
    (0.3, 0.499999, 2.0, -1.0, 0.6075013199996799),
    (0.4999989, 0.499999, 2.0, -1.0, 0.687500450002805),
    (0.49999910000000003, 0.499999, 2.0, -1.0, 0.687500550002205),
    (0.5000009999999999, 0.499999, 2.0, -1.0, 0.6875014999985),
    (0.8, 0.499999, 2.0, -1.0, 0.8411613273713953),
    (1.2, 0.499999, 2.0, -1.0, 0.9762337734411515),
    (0.0, 0.5, 0.4, 0.25, 0.7053320384278756),
    (0.3, 0.5, 0.4, 0.25, 0.7132762894369031),
    (0.4999999, 0.5, 0.4, 0.25, 0.7319835656059951),
    (0.5000001, 0.5, 0.4, 0.25, 0.731983596762152),
    (0.5, 0.5, 0.4, 0.25, 0.7319835811800197),
    (0.8, 0.5, 0.4, 0.25, 0.8280592805773233),
    (1.2, 0.5, 0.4, 0.25, 0.9547049594002633),
    (0.0, 0.5, 2.0, -1.0, 0.5625),
    (0.3, 0.5, 2.0, -1.0, 0.6075),
    (0.4999999, 0.5, 2.0, -1.0, 0.687499950000005),
    (0.5000001, 0.5, 2.0, -1.0, 0.687500050000005),
    (0.5, 0.5, 2.0, -1.0, 0.6875),
    (0.8, 0.5, 2.0, -1.0, 0.8411607032540964),
    (1.2, 0.5, 2.0, -1.0, 0.9762335693213314),
    (0.0, 0.500001, 0.4, 0.25, 0.7053308967022144),
    (0.3, 0.500001, 0.4, 0.25, 0.7132751909902864),
    (0.5000009000000001, 0.500001, 0.4, 0.25, 0.7319827547994092),
    (0.5000011, 0.500001, 0.4, 0.25, 0.7319827860573241),
    (0.49999899999999997, 0.500001, 0.4, 0.25, 0.7319824582168036),
    (0.8, 0.500001, 0.4, 0.25, 0.8280587153494245),
    (1.2, 0.500001, 0.4, 0.25, 0.9547046761684981),
    (0.0, 0.500001, 2.0, -1.0, 0.5624984999995),
    (0.3, 0.500001, 2.0, -1.0, 0.60749867999968),
    (0.5000009000000001, 0.500001, 2.0, -1.0, 0.6874994500022003),
    (0.5000011, 0.500001, 2.0, -1.0, 0.6874995500027988),
    (0.49999899999999997, 0.500001, 2.0, -1.0, 0.6874984999985),
    (0.8, 0.500001, 2.0, -1.0, 0.8411600791356282),
    (1.2, 0.500001, 2.0, -1.0, 0.976233365200463),
    (0.0, 0.50009, 0.4, 0.25, 0.7052292752822982),
    (0.3, 0.50009, 0.4, 0.25, 0.7131774222405073),
    (0.5000899000000001, 0.50009, 0.4, 0.25, 0.7319111901228029),
    (0.5000901, 0.50009, 0.4, 0.25, 0.7319112224005928),
    (0.49990999999999997, 0.50009, 0.4, 0.25, 0.7318825054958732),
    (0.8, 0.50009, 0.4, 0.25, 0.8280084071771546),
    (1.2, 0.50009, 0.4, 0.25, 0.9546794651435186),
    (0.0, 0.50009, 2.0, -1.0, 0.5623649959514581),
    (0.3, 0.50009, 2.0, -1.0, 0.6073811974094581),
    (0.5000899000000001, 0.50009, 2.0, -1.0, 0.6874549698104738),
    (0.5000901, 0.50009, 2.0, -1.0, 0.6874550698633238),
    (0.49990999999999997, 0.50009, 2.0, -1.0, 0.6873649878514582),
    (0.8, 0.50009, 2.0, -1.0, 0.84110452790891),
    (1.2, 0.50009, 2.0, -1.0, 0.9762151942446282),
]


def test_ell_combo_and_occultation_exact():
    """P3, bug #28: ell_combo(q, n, A, B, C) = A K + B E + C Pi in one
    Bulirsch loop is exact (<= 2e-15 relative vs mpmath; ellk/ellec were
    ~2e-8), and occult_hybrid matches mpmath's radial-quadrature flux to
    1e-10 for every k incl. the former small-body interpolation zone
    (k <= 0.05, z <= 0.65 ... 0.98; bug #28, up to 2.6 ppm off before), away
    from the contact points; within 2e-6 of z = 1 - k and z = k to 2e-9
    (P15: the q -= 1e-8 and z -> k + 1e-6 nudges are gone, which were up
    to 4.8e-7 off at k = 0.5, z = k - 1e-6; what is left, <= 1.5e-9 at z = 1 - k
    exactly, is the limb area's arccos at argument +-1). Since P14 this holds
    for occult_exact; occult_hybrid (interior series) to within SERIES_TOL more."""
    from pandoramoon.occult import ell_combo, SERIES_TOL
    for q, n, K, E, P in ELL_REF:
        for (A, B, Cc), ref in (((1.0, 0.0, 0.0), K), ((0.0, 1.0, 0.0), E), ((0.0, 0.0, 1.0), P)):
            got = ell_combo(q, n, A, B, Cc)
            assert abs(got - ref) <= 2e-15 * abs(ref), f"ell_combo q={q} n={n} ABC={A, B, Cc}: {got!r} vs {ref!r}"
    for z, k, u1, u2, ref in FLUX_REF:
        boundary = abs(z - (1 - k)) < 2e-6 or abs(z - k) < 2e-6
        tol = 2e-9 if boundary else 1e-10
        got = occult_exact(np.array([z]), k, u1, u2)[0]
        assert abs(got - ref) <= tol, f"occult_exact z={z} k={k} u=({u1},{u2}): {got!r} vs mpmath {ref!r}"
        got = occult_hybrid(np.array([z]), k, u1, u2)[0]   # P14 series where it applies
        assert abs(got - ref) <= tol + SERIES_TOL, f"occult_hybrid z={z} k={k} u=({u1},{u2}): {got!r} vs mpmath {ref!r}"


def test_hybrid_second_contact_gap():
    """Bug #29: for some k, z = 1 - k satisfies neither the old Case III
    condition (z >= 0.5 + |k - 0.5|, one ulp higher in floating point) nor
    Case IV (z < 1 - k); occult_hybrid then reused the previous sample's
    limb-darkening terms and occult() left them uninitialised (np.empty).
    occult_hybrid must match the independent reference there, whatever
    sample came before; occult (still on the ellk/ellec polynomials) must be
    continuous with the next float and within their ~2 ppm, also after
    dirtying the heap."""
    found = 0
    for k in np.linspace(0.001, 0.499, 4000):
        z = 1 - k
        if z >= 0.5 + abs(k - 0.5):
            continue
        found += 1
        if found > 4:
            break
        ref = ref_flux(z, k)
        got = occult_hybrid(np.array([0.3, z]), k, U1, U2)[1]
        assert abs(got - ref) < 1e-7, f"occult_hybrid k={k!r} z={z!r}: {got!r} vs reference {ref!r}"
        # occult() still uses the ellk/ellec polynomials (~2 ppm near z = 1 - k
        # for k ~ 0.001, see P3a): require continuity with the next float
        # (Case III in old and new code) and the polynomial accuracy
        nxt = occult(np.array([np.nextafter(z, 2.0)]), k, U1, U2)[0]
        for _ in range(3):   # occult's buffers come from np.empty: dirty the heap first
            np.empty(4).fill(-1e300)
            got = occult(np.array([0.3, z]), k, U1, U2)[1]
            assert abs(got - nxt) < 2e-8 and abs(got - ref) < 3e-6, \
                f"occult k={k!r} z={z!r}: {got!r} vs next float {nxt!r}, reference {ref!r}"
    assert found > 0, "no k with the gap found"


def test_occult_batch_matches_scalar():
    """P3b: occult_hybrid's vectorised exact pass (_occult_exact_batch, fixed
    Bulirsch iteration count) equals the scalar per-sample model
    (_occult_point, early-exit ell_combo) to 2e-11 on dense z grids incl.
    one ulp around 0, k, 1 - k, 1 + k (5e-9 within 1e-6 of first contact),
    also for lanes that need the scalar fallback (q -> 1 near second
    contact); and the batch loop is vectorised. Tested on occult_exact
    (occult_hybrid without the P14 interior series)."""
    import numba
    from pandoramoon.occult import _occult_point, _occult_exact_batch, ELL_BATCH_ITERATIONS
    for k in (0.001, 0.01, 0.051, 0.1, 0.2, 0.5, 0.7, 0.95, 1.2):
        for u1, u2 in ((0.4, 0.25), (0.9, -0.1), (0.2, 0.8)):
            zs = list(np.linspace(0, 1 + k, 3001))
            for b in (0.0, k, abs(1 - k), 1 + k):
                for d in (0.0, 1e-12, 1e-9, 1e-8, 1e-7, 1e-5):
                    zs += [b - d, b + d]
                zs += [np.nextafter(b, 2), np.nextafter(b, -2)]
            zs = np.array([z for z in zs if 0 <= z])
            got = occult_exact(zs.copy(), k, u1, u2)   # P14: series off
            assert np.all(np.isfinite(got)), f"k={k}: non-finite flux"
            kk = 0.5 if abs(k - 0.5) < 1e-4 else k
            for z, f in zip(zs, got):
                # occult_hybrid moves only z == k exactly, by one ulp (P15)
                zz = kk + max(kk * 2.220446049250313e-16, 1e-150) if abs(z - kk) < 1e-150 else z
                if zz >= 1 + kk or (kk >= 1 and zz <= kk - 1):     # off star / fully occulted
                    continue
                ref = min(_occult_point(zz, kk, u1, u2), 1.0)
                # within 1e-6 of first contact both carry the formula's own
                # cancellation noise (<= 1.2e-9 vs mpmath, either way)
                tol = 2e-11 if 1 + kk - zz > 1e-6 else 5e-9
                assert abs(f - ref) <= tol, f"k={k} z={z!r} u=({u1},{u2}): batch {f!r} vs scalar {ref!r}"
    # lanes beyond ELL_BATCH_ITERATIONS: limb samples with q -> 1 (z just above 1 - k)
    k = 0.1
    zs = 1 - k + np.logspace(-12, -3, 50)
    fb = np.empty(50)
    dev = np.empty(50)
    _occult_exact_batch(zs, 50, k, 0.1, 0.9, 0.25, 1.1, fb, dev)
    assert np.any(dev > 0), "no non-converged lane tested"
    got = occult_exact(zs.copy(), k, 0.4, 0.25)
    for z, f in zip(zs, got):
        assert abs(f - min(_occult_point(z, k, 0.4, 0.25), 1.0)) <= 1e-13, f"fallback lane z={z!r}"
    assert ELL_BATCH_ITERATIONS >= 4
    # cached code cannot be inspected: compile an uncached copy of the source
    fresh = numba.njit(fastmath=True, error_model="numpy")(_occult_exact_batch.py_func)
    fresh(zs, 50, k, 0.1, 0.9, 0.25, 1.1, fb, dev)
    assert "vector.body" in "".join(fresh.inspect_llvm().values()), "_occult_exact_batch not vectorised"


def test_occult_contact_points_exact():
    """P15: with the elliptic operands in closed form (kc^2, p, C, C/p from
    (z, k); no q, no nudges) occult_hybrid is exact at z = k +- 1e-15 ...
    1e-3, 1 - k -+ 1e-15 ... 1e-3 and 1 + k - 1e-6, 1e-3 for k = 0.001 ...
    0.2 (mpmath, CONTACT_REF): <= 1e-12, and <= 5e-10 on the limb side
    within 1e-6 of z = 1 - k (the limb area's arccos near argument 1, not
    the elliptic part). Before P15 the z -> k + 1e-6 and q -> 1 - 1e-8
    nudges were up to 4.3e-9 / 2.9e-8 off here."""
    for z, k, ref in CONTACT_REF:
        got = occult_exact(np.array([z]), k, 0.4, 0.25)[0]   # (P14 series, removed in 2.0)
        tol = 5e-10 if 0 <= z - (1 - k) < 1e-6 else 1e-12
        assert abs(got - ref) <= tol, f"k={k} z={z!r}: {got!r} vs mpmath {ref!r} ({abs(got - ref):.1e})"
    # z == k exactly is moved by one ulp: finite and continuous
    for k in (0.0, 0.0092, 0.1, 0.5, 1.2):
        for fn in (occult_exact, occult_hybrid):
            f = fn(np.array([k, np.nextafter(k, 2.0)]), k, 0.4, 0.25)
            assert np.all(np.isfinite(f)) and abs(f[0] - f[1]) <= 1e-14, f"z == k = {k}: {f!r}"

def test_no_k_snap_near_one_half():
    """Bug #30: occult_hybrid and occult replaced k by 0.5 for
    |k - 0.5| < 1e-4, i.e. evaluated a body of the wrong size: -114 / +114
    ppm at k = 0.5 -+ 1e-4, z = 0 for u = (0.4, 0.25), up to 150 ppm for
    u = (2, -1). occult_exact must match mpmath (K05_REF) to 1e-10 (5e-9
    at z = 1 - k, the limb area's arccos; occult_hybrid with the P14 series
    within 1e-8 more), occult to its polynomial
    accuracy (1 ppm), and both must be finite at k = 0.5 exactly."""
    for z, k, u1, u2, ref in K05_REF:
        tol = 5e-9 if abs(z - (1 - k)) < 1e-12 else 1e-10
        got = occult_exact(np.array([z]), k, u1, u2)[0]
        assert abs(got - ref) <= tol, f"occult_exact k={k!r} z={z!r} u=({u1},{u2}): {got!r} vs {ref!r}"
        got = occult_hybrid(np.array([z]), k, u1, u2)[0]   # P14: + series tolerance 1e-8
        assert abs(got - ref) <= tol + 1e-8, f"occult_hybrid k={k!r} z={z!r} u=({u1},{u2}): {got!r} vs {ref!r}"
        got = occult(np.array([z]), k, u1, u2)[0]
        assert abs(got - ref) <= 1e-6, f"occult k={k!r} z={z!r} u=({u1},{u2}): {got!r} vs {ref!r}"


# P14: mpmath disk quadrature of the quadratic-LD flux, at z = 0, 0.5, 0.9 and
# (1 - 1e-9) times the series cut z_c of each (k, u) (scratch_p14/gen_p14_ref.py)
SERIES_REF = [
    (0.0, 0.001, 0.4, 0.26, 0.9999987854252227),
    (0.499494952046, 0.001, 0.4, 0.26, 0.9999988560161609),
    (0.899090913682, 0.001, 0.4, 0.26, 0.9999991584037362),
    (0.998989903092, 0.001, 0.4, 0.26, 0.9999995394096318),
    (0.0, 0.001, 0.2, 0.6, 0.99999880000006),
    (0.499494952046, 0.001, 0.2, 0.6, 0.9999988449514372),
    (0.899090913682, 0.001, 0.2, 0.6, 0.99999916254012),
    (0.998989903092, 0.001, 0.2, 0.6, 0.9999996888769734),
    (0.0, 0.001, 1.0, 0.0, 0.999998500000375),
    (0.499494952046, 0.001, 1.0, 0.0, 0.9999987005253093),
    (0.899090913682, 0.001, 1.0, 0.0, 0.9999993433597314),
    (0.998989903092, 0.001, 1.0, 0.0, 0.999999935199579),
    (0.0, 0.0092, 0.4, 0.26, 0.999897199250708),
    (0.494191105978, 0.0092, 0.4, 0.26, 0.9999030284163232),
    (0.88954399076, 0.0092, 0.4, 0.26, 0.9999274298617247),
    (0.988382210968, 0.0092, 0.4, 0.26, 0.9999516168409094),
    (0.0, 0.0092, 0.2, 0.6, 0.9998984324298782),
    (0.494486751727, 0.0092, 0.2, 0.6, 0.9999021336139444),
    (0.890076153109, 0.0092, 0.2, 0.6, 0.9999275500995902),
    (0.988973502466, 0.0092, 0.2, 0.6, 0.9999605035768931),
    (0.0, 0.0092, 1.0, 0.0, 0.9998730426865114),
    (0.494453498482, 0.0092, 1.0, 0.0, 0.9998896494405589),
    (0.890016297268, 0.0092, 1.0, 0.0, 0.999942132425749),
    (0.988906995975, 0.0092, 1.0, 0.0, 0.9999816198941193),
    (0.0, 0.05, 0.4, 0.26, 0.9969643225905108),
    (0.46490494041, 0.05, 0.4, 0.26, 0.9971144354283091),
    (0.836828892738, 0.05, 0.4, 0.26, 0.9976808874252211),
    (0.92980987989, 0.05, 0.4, 0.26, 0.9980670286641052),
    (0.0, 0.05, 0.2, 0.6, 0.9970003760947767),
    (0.463916788971, 0.05, 0.2, 0.6, 0.9970930378729228),
    (0.835050220147, 0.05, 0.2, 0.6, 0.9976422522039242),
    (0.927833577013, 0.05, 0.2, 0.6, 0.998112720584482),
    (0.0, 0.05, 1.0, 0.0, 0.9962523447274791),
    (0.464196238452, 0.05, 1.0, 0.0, 0.996681515846024),
    (0.835553229214, 0.05, 1.0, 0.0, 0.9979490226998993),
    (0.928392475976, 0.05, 1.0, 0.0, 0.998634613499135),
    (0.0, 0.1, 0.4, 0.26, 0.9878664434953113),
    (0.423237403785, 0.1, 0.4, 0.26, 0.9883572495125204),
    (0.761827326814, 0.1, 0.4, 0.26, 0.9900159492338796),
    (0.846474806724, 0.1, 0.4, 0.26, 0.9909266335140795),
    (0.0, 0.1, 0.2, 0.6, 0.9880060702638203),
    (0.421062980279, 0.1, 0.2, 0.6, 0.988300813709658),
    (0.757913364502, 0.1, 0.2, 0.6, 0.9897786084320822),
    (0.842125959716, 0.1, 0.2, 0.6, 0.990780898471229),
    (0.0, 0.1, 1.0, 0.0, 0.9850375627355538),
    (0.421675310891, 0.1, 1.0, 0.0, 0.9864447720358712),
    (0.759015559604, 0.1, 1.0, 0.0, 0.9903326961657857),
    (0.843350620939, 0.1, 1.0, 0.0, 0.9921045040663103),
    (0.0, 0.2, 0.4, 0.26, 0.951614361401167),
    (0.336726653064, 0.2, 0.4, 0.26, 0.952852522811894),
    (0.606107975516, 0.2, 0.4, 0.26, 0.9564164560671808),
    (0.673453305455, 0.2, 0.4, 0.26, 0.9579866308278869),
    (0.0, 0.2, 0.2, 0.6, 0.9521005485761892),
    (0.332527635679, 0.2, 0.2, 0.6, 0.9528280966106254),
    (0.598549744221, 0.2, 0.2, 0.6, 0.9555769082083176),
    (0.665055270692, 0.2, 0.2, 0.6, 0.957013311414393),
    (0.0, 0.2, 1.0, 0.0, 0.9406040612287404),
    (0.333705053497, 0.2, 1.0, 0.0, 0.9441226044688104),
    (0.600669096295, 0.2, 1.0, 0.0, 0.95302036131956),
    (0.667410106327, 0.2, 1.0, 0.0, 0.95649851860077),
    (0.0, 0.4, 0.4, 0.26, 0.8089804853541495),
    (0.175235782559, 0.4, 0.4, 0.26, 0.8104560974266033),
    (0.315424408605, 0.4, 0.4, 0.26, 0.8140153434516404),
    (0.350471564767, 0.4, 0.4, 0.26, 0.8153123032204651),
    (0.0, 0.4, 0.2, 0.6, 0.8098414427628908),
    (0.169161366927, 0.4, 0.2, 0.6, 0.8107829904563377),
    (0.304490460468, 0.4, 0.2, 0.6, 0.8132208141442633),
    (0.338322733515, 0.4, 0.2, 0.6, 0.8141626145283694),
    (0.0, 0.4, 1.0, 0.0, 0.7698727167525811),
    (0.170851100785, 0.4, 1.0, 0.0, 0.7737334560496436),
    (0.307531981413, 0.4, 1.0, 0.0, 0.7826876347955286),
    (0.341702201228, 0.4, 1.0, 0.0, 0.7858327068737229),
]


def test_window_chi2_matches_cadences():
    """P13: _window_chi2() (cadence block means fused with the chi^2, fixed-
    length sums over contiguous runs) equals the chi^2 of the general
    cadence loop _window_cadences() (block means to a few ulp), for random index
    patterns: several contiguous runs starting aligned and unaligned with
    respect to the blocks, single samples, gaps inside a block, runs that
    end at n with a partial last block, empty windows, and factors 1..13.
    P18: the function is now _window_chi2_runs() on (los, his) runs; also
    with the runs split into adjacent pieces (sharing a block) and an empty
    run inserted. P13b: it takes inv_var, lin_terms (Delta chi^2 =
    D (inv_var D - lin_terms))."""
    pm = sys.modules["pandoramoon.pandora"]
    rng = np.random.default_rng(20261007)
    n_fast = 0
    n_split = 0
    for j in range(400):
        factor = int(rng.choice([1, 2, 3, 5, 7, 13])) if j % 10 else 5
        n = int(rng.integers(1, 3000))
        parts = []
        pos = 0
        for _ in range(int(rng.integers(0, 6))):
            pos += int(rng.integers(0, 80))
            length = int(rng.choice([1, 2, factor, rng.integers(1, 400)]))
            if j % 4 == 0:                       # run aligned to a block start
                pos = -(-pos // factor) * factor
            parts.append(np.arange(pos, pos + length))
            pos += length
        if j % 5 == 0 and n > 10:                # a run up to the end of the samples
            parts.append(np.arange(n - int(rng.integers(1, 10)), n))
        idx = np.unique(np.concatenate(parts).astype(np.int64)) if parts else np.empty(0, np.int64)
        if j % 7 == 0 and len(idx) > 4:          # holes inside runs
            idx = np.delete(idx, rng.integers(0, len(idx), 3))
            idx = np.unique(idx)
        idx = idx[idx < n]
        m = len(idx)
        fp = 1 - rng.uniform(0, 1e-2, m) * (rng.uniform(size=m) < 0.8)
        fm = 1 - rng.uniform(0, 1e-3, m) * (rng.uniform(size=m) < 0.5)
        n_out = -(-n // factor)
        data = 1 + rng.normal(0, 1e-3, n_out)
        inv_err = 1 / rng.uniform(1e-4, 1e-3, n_out)
        flat = ((1 - data) * inv_err) ** 2
        oi, _, _, ft = pm._window_cadences(n, idx, fp, fm, factor)
        ref = np.sum(((ft - data[oi]) * inv_err[oi]) ** 2 - flat[oi])
        # P18: the window as runs; split the maximal runs at random points
        # (adjacent runs, as consecutive epochs of the search can give) and
        # insert empty runs
        los, his = pm._idx_runs(idx)
        assert np.array_equal(np.concatenate([np.arange(a, b) for a, b in zip(los, his)]
                                             or [np.empty(0, np.int64)]), idx)
        if j % 3 == 0 and m:
            cuts = np.unique(rng.integers(0, m, int(rng.integers(1, 6))))
            bounds = np.concatenate([[0], cuts[cuts > 0], [m]])
            bounds = np.unique(bounds)
            pieces = [(idx[b0], idx[b1 - 1] + 1) for b0, b1 in zip(bounds[:-1], bounds[1:])]
            # a piece may span a hole: re-split it at the holes
            split = []
            for a0, b0 in pieces:
                sub = idx[(idx >= a0) & (idx < b0)]
                sl, sh = pm._idx_runs(sub)
                split += list(zip(sl, sh))
            split.insert(int(rng.integers(0, len(split) + 1)), (split[0][0], split[0][0]))
            split.sort(key=lambda ab: (ab[0], ab[1] - ab[0] > 0))
            los = np.array([a0 for a0, _ in split], np.int64)
            his = np.array([b0 for _, b0 in split], np.int64)
            n_split += 1
        # P13b: chi^2 - flat term per cadence as D (inv_var D - lin_terms)
        got = pm._window_chi2_runs(n, los, his, fp, fm, factor, inv_err ** 2,
                                   2 * (1 - data) * inv_err ** 2)
        # the two block-mean formulas differ by a few ulp of f, i.e. r by
        # ~1e-16 inv_err (inv_err up to 1e4): tolerance from that, plus the
        # chi^2 summation order
        r = (ft - data[oi]) * inv_err[oi]
        tol = 8 * np.finfo(float).eps * np.sum(2 * np.abs(r) * inv_err[oi] + r ** 2 + flat[oi])
        assert abs(got - ref) <= tol + 1e-300, f"case {j}: {got!r} vs {ref!r} (tol {tol:.1e})"
        if factor > 1 and m:
            n_fast += int(np.sum(np.diff(idx) == 1) >= factor)
    assert n_fast > 50, f"contiguous runs too rare ({n_fast})"
    assert n_split > 50, f"split runs too rare ({n_split})"


INTERIOR_REF = [  # (z, k, u1, u2, flux) on 0 <= z < 1 - k, mpmath radial quadrature, dps 30 (scratch_p21c)
    (0.0, 0.001, 0.0, 1.0, 0.9999988),
    (0.001, 0.001, 1.0, 0.0, 0.999998500001125),
    (0.4995, 0.001, 0.4, 0.25, 0.9999988581122636),
    (0.998999999999001, 0.001, 2.0, -1.0, 0.999999996003),
    (0.99899999001, 0.001, 1.2, -0.2, 0.9999999451383357),
    (0.9989900100000001, 0.001, 0.0, 1.0, 0.9999998987482301),
    (0.998001, 0.001, 1.0, 0.0, 0.9999999059803468),
    (0.98901, 0.001, 0.4, 0.25, 0.9999994211351773),
    (0.96903, 0.001, 2.0, -1.0, 0.9999998780392818),
    (0.8991, 0.001, 1.2, -0.2, 0.9999993865521822),
    (0.0, 0.0092, 0.0, 1.0, 0.9998984320000607),
    (0.0092, 0.0092, 1.0, 0.0, 0.9998730480597998),
    (0.4954, 0.0092, 0.4, 0.25, 0.999903239669086),
    (0.9907999999990093, 0.0092, 2.0, -1.0, 0.9999969067397885),
    (0.9907999900919999, 0.0092, 1.2, -0.2, 0.9999856301520397),
    (0.990790092, 0.0092, 0.0, 1.0, 0.9999754655348648),
    (0.9898092000000001, 0.0092, 1.0, 0.0, 0.9999824817050815),
    (0.980892, 0.0092, 0.4, 0.25, 0.9999472407996405),
    (0.961076, 0.0092, 2.0, -1.0, 0.9999870855268556),
    (0.8917200000000001, 0.0092, 1.2, -0.2, 0.9999461531214322),
    (0.0, 0.05, 0.0, 1.0, 0.9970000015639667),
    (0.05, 0.05, 1.0, 0.0, 0.9962570410478134),
    (0.475, 0.05, 0.4, 0.25, 0.9971273547646897),
    (0.9499999999990499, 0.05, 2.0, -1.0, 0.999518749999991),
    (0.9499999905, 0.05, 1.2, -0.2, 0.9989823017647853),
    (0.9499905, 0.05, 0.0, 1.0, 0.9984993419192647),
    (0.94905, 0.05, 1.0, 0.0, 0.9988687821200771),
    (0.9405, 0.05, 0.4, 0.25, 0.9981267120482871),
    (0.9215, 0.05, 2.0, -1.0, 0.99925206125),
    (0.855, 0.05, 1.2, -0.2, 0.9981599836025613),
    (0.0, 0.1, 0.0, 1.0, 0.988000100376886),
    (0.1, 0.1, 1.0, 0.0, 0.9851131333540168),
    (0.45, 0.1, 0.4, 0.25, 0.9884514994251203),
    (0.8999999999991001, 0.1, 2.0, -1.0, 0.9962999999999677),
    (0.8999999909999999, 0.1, 1.2, -0.2, 0.9941885085263388),
    (0.8999910000000001, 0.1, 0.0, 1.0, 0.9922878113131528),
    (0.8991, 0.1, 1.0, 0.0, 0.9937585951472834),
    (0.891, 0.1, 0.4, 0.25, 0.9916188219649883),
    (0.873, 0.1, 2.0, -1.0, 0.99534258),
    (0.81, 0.1, 1.2, -0.2, 0.9916313605092293),
    (0.0, 0.2, 0.0, 1.0, 0.9520064979659846),
    (0.2, 0.2, 1.0, 0.0, 0.9418422655146366),
    (0.4, 0.2, 0.4, 0.25, 0.9535041687478941),
    (0.7999999999992, 0.2, 2.0, -1.0, 0.9727999999998976),
    (0.799999992, 0.2, 1.2, -0.2, 0.9673021335111975),
    (0.799992, 0.2, 0.0, 1.0, 0.9623533660180227),
    (0.7992, 0.2, 1.0, 0.0, 0.9661902965611994),
    (0.792, 0.2, 0.4, 0.25, 0.9622069971165791),
    (0.776, 0.2, 2.0, -1.0, 0.96977408),
    (0.7200000000000001, 0.2, 1.2, -0.2, 0.9603031033198584),
    (0.0, 0.3, 0.0, 1.0, 0.8920754772630729),
    (0.3, 0.3, 1.0, 0.0, 0.8746328151838496),
    (0.35, 0.3, 0.4, 0.25, 0.895136424190794),
    (0.6999999999993, 0.3, 2.0, -1.0, 0.9162999999998236),
    (0.6999999929999999, 0.3, 1.2, -0.2, 0.9118842318090897),
    (0.699993, 0.3, 0.0, 1.0, 0.9079091409999179),
    (0.6992999999999999, 0.3, 1.0, 0.0, 0.9109342894890141),
    (0.693, 0.3, 0.4, 0.25, 0.9092151881580525),
    (0.6789999999999999, 0.3, 2.0, -1.0, 0.91108738),
    (0.63, 0.3, 1.2, -0.2, 0.9006290124828319),
    (0.0, 0.5, 0.0, 1.0, 0.7017304845413264),
    (0.5, 0.5, 1.0, 0.0, 0.7122065907891938),
    (0.25, 0.5, 0.4, 0.25, 0.7107136252082533),
    (0.4999999999995, 0.5, 2.0, -1.0, 0.68749999999975),
    (0.499999995, 0.5, 1.2, -0.2, 0.7083055484032795),
    (0.499995, 0.5, 0.0, 1.0, 0.7270294988950307),
    (0.4995, 0.5, 1.0, 0.0, 0.7120479043736507),
    (0.495, 0.5, 0.4, 0.25, 0.7312271503461749),
    (0.485, 0.5, 2.0, -1.0, 0.6801125),
    (0.45, 0.5, 1.2, -0.2, 0.6928526331709409),
    (0.0, 0.8, 0.0, 1.0, 0.2678399999999999),
    (0.8, 0.8, 1.0, 0.0, 0.6215072098802313),
    (0.09999999999999998, 0.8, 0.4, 0.25, 0.29416114104291735),
    (0.19999999999979995, 0.8, 2.0, -1.0, 0.18079999999989751),
    (0.19999999799999996, 0.8, 1.2, -0.2, 0.23993959023986602),
    (0.19999799999999995, 0.8, 0.0, 1.0, 0.29316453370708667),
    (0.19979999999999995, 0.8, 1.0, 0.0, 0.25094694362944),
    (0.19799999999999995, 0.8, 0.4, 0.25, 0.3074700249371411),
    (0.19399999999999995, 0.8, 2.0, -1.0, 0.17777407999999992),
    (0.17999999999999997, 0.8, 1.2, -0.2, 0.23222408144976595),
]


def test_interior_interpolant():
    """P21c: occult_hybrid's per-call interior interpolant (0 <= z < 1 - k,
    polynomial of degree INT_N in x = 2 sqrt((1 - k)^2 - z^2) / (1 - k) - 1;
    _int_coeffs / _occult_int_batch / _occult_int_select; replaces the P14
    series and the exact band) is within 2e-9 of an independent mpmath
    radial quadrature (INTERIOR_REF: z = 0, z = k, 1e-12 ... 0.1 from
    internal tangency, k = 0.001 ... 0.8) and within 5e-9 (measured
    <= 3.7e-9, scratch_p21c/interior_err.py) of the exact kernel
    (occult_exact) on dense grids for k = 1e-6 ... 0.99 and 15 LD laws incl.
    unphysical ones, on both paths; it is really used (equals a direct
    _occult_int_batch call) on the per-sample and the stretch path, not by
    occult_exact or for k >= 1; the per-call build's end conditions hold
    (closed forms at z = 0 and z = 1 - k, zero slope at x = -1); the
    offline table (INT_TAB, interior_table.csv) equals a fresh
    _int_table_build() and reproduces the per-call build (sum of
    coefficient errors <= 2e-10 for k in [INT_TAB_KMIN, INT_TAB_KMAX], the
    range ends included) and is used there; flux <= 1; and both loops are
    vectorised."""
    import numba
    from pandoramoon.occult import (INT_N, INT_TAB, INT_TAB_KMIN, INT_TAB_KMAX, _int_coeffs,
                                    _int_coeffs_exact, _int_table_build, _occult_int_batch,
                                    _occult_int_select, _occult_hybrid_samples, _occult_point,
                                    _use_interior)
    assert np.max(np.abs(INT_TAB - _int_table_build())) <= 1e-13, "interior_table.csv is stale"
    for k in np.concatenate([[INT_TAB_KMIN, INT_TAB_KMAX], np.exp(np.random.default_rng(7).uniform(
            np.log(INT_TAB_KMIN), np.log(INT_TAB_KMAX), 200))]):
        for u1, u2 in ((0.4, 0.26), (0.0, 1.0), (2.0, -1.0)):
            d = np.sum(np.abs(_int_coeffs(k, u1, u2) - _int_coeffs_exact(k, u1, u2)))
            assert d <= 2e-10 * max(1.0, abs((u1 + 2 * u2) / (1 - u1 / 3 - u2 / 6))), (k, u1, u2, d)
    for k in (INT_TAB_KMIN / 2, INT_TAB_KMAX * 1.1):   # outside the table: the per-call build
        assert np.array_equal(_int_coeffs(k, 0.4, 0.26), _int_coeffs_exact(k, 0.4, 0.26))
    for z, k, u1, u2, ref in INTERIOR_REF:
        got = occult_hybrid(np.array([z]), k, u1, u2)[0]
        assert abs(got - ref) <= 2e-9, f"z={z!r} k={k} u=({u1},{u2}): {got!r} vs mpmath {ref!r}"
    rng = np.random.default_rng(2110)
    lds = [(0.4, 0.26), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, -1.0), (0.8, -0.1),
           (-0.5, 0.3), (1.5, -0.8)]
    lds += [(2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)) for q1, q2 in rng.uniform(0, 1, (7, 2))]
    n_used = 0
    for k in (1e-6, 1e-4, 1e-3, 0.0092, 0.03, 0.1, 0.2, 0.3, 0.5, 0.8, 0.99):
        omk = 1 - k
        zi = np.concatenate([np.linspace(0, omk, 1500, endpoint=False), omk * (1 - np.logspace(-15, -0.3, 60)),
                             k * (1 + np.linspace(-1e-9, 1e-9, 5)), [k, np.nextafter(k, 0), np.nextafter(k, 2)],
                             rng.uniform(0, omk, 300)])
        zi = zi[(zi >= 0) & (zi < omk)]
        zs = np.concatenate([zi, np.linspace(omk, 1.2 + k, 50), [-0.1]])
        smooth = np.sort(np.concatenate([zs, np.linspace(0, 1.3 + k, 2500)]))   # stretch path
        for u1, u2 in lds:
            coef = _int_coeffs(k, u1, u2)
            cex = _int_coeffs_exact(k, u1, u2)
            for z in (zs, smooth):
                hyb = occult_hybrid(z, k, u1, u2)
                d = np.abs(hyb - occult_exact(z, k, u1, u2)).max()
                assert d <= 5e-9, f"k={k} u=({u1:.3f},{u2:.3f}) n={len(z)}: {d:.2e}"
                assert hyb.max() <= 1.0
                ins = (z >= 0) & (z < omk)
                direct = np.empty(ins.sum())
                _occult_int_batch(np.ascontiguousarray(z[ins]) ** 2, 0, len(direct), coef, omk, direct)
                assert np.max(np.abs(hyb[ins] - direct)) <= 1e-15, f"k={k} n={len(z)}: interpolant not used"
                n_used += 1
            # end conditions: D(z = 0) and D_t (closed forms), dD/dx = 0 at x = -1
            om = 1 / (1 - u1 / 3 - u2 / 6)
            P1 = np.sum(cex[:INT_N + 1])
            Pm1 = np.sum(cex[:INT_N + 1] * (-1.0) ** np.arange(INT_N + 1))
            dPm1 = np.sum(cex[1:INT_N + 1] * np.arange(1, INT_N + 1) * (-1.0) ** np.arange(INT_N))
            assert abs(P1 - (1 - _occult_point(1e-300, k, u1, u2))) <= 1e-11, (k, u1, u2)
            if k >= 1e-3:
                assert abs(Pm1 - (1 - _occult_point(omk * (1 - 1e-16), k, u1, u2))) <= 1e-11, (k, u1, u2)
            assert abs(dPm1) <= 1e-10, (k, u1, u2, dPm1)   # (monomial rounding)
    assert n_used >= 300, n_used
    # not used: occult_exact, k >= 1, no approximations
    assert not _use_interior(1.0, 1e-8) and not _use_interior(0.1, 0.0) and not _use_interior(0.0, 1e-8)
    zi = np.linspace(0, 0.85, 3000)
    assert np.array_equal(occult_exact(zi, 0.1, 0.4, 0.26), _occult_hybrid_samples(zi * zi, 0.1, 0.4, 0.26, False, False))
    for name, fn, args in (("_occult_int_batch", _occult_int_batch, ()),
                           ("_occult_int_select", _occult_int_select, (1.1,))):
        fresh = numba.njit(fastmath=True, error_model="numpy")(fn.py_func)
        fresh(np.linspace(0, 0.8, 64), 0, 64, _int_coeffs(0.1, 0.4, 0.26), 0.9, *args, np.empty(64))
        ir = "".join(fresh.inspect_llvm().values())
        assert "vector.body" in ir and "llvm.sqrt.v4f64" in ir, f"{name} not vectorised"


LIMB_REF = [  # (z, k, u1, u2, flux) on 1 - k <= z < 1 + k, mpmath radial quadrature, dps 30 (scratch_p21b)
    (0.999, 0.001, 0.4, 0.25, 0.9999995295279969),
    (1.0, 0.001, 1.0, 0.0, 0.9999999795410058),
    (1.0009999999999988, 0.001, 0.0, 1.0, 1.0),
    (0.999000000000002, 0.001, 2.0, -1.0, 0.999999996003),
    (1.000999999999998, 0.001, 1.2, -0.2, 1.0),
    (0.999000002, 0.001, 0.4, 0.25, 0.9999995295280548),
    (1.000999998, 0.001, 1.0, 0.0, 1.0),
    (0.999002, 0.001, 0.0, 1.0, 0.9999998994851232),
    (1.0009979999999998, 0.001, 2.0, -1.0, 0.9999999999999998),
    (0.9991, 0.001, 1.2, -0.2, 0.9999999487933647),
    (0.9994, 0.001, 0.4, 0.25, 0.9999996028832945),
    (0.9997, 0.001, 1.0, 0.0, 0.9999999673293396),
    (1.0, 0.001, 0.0, 1.0, 0.9999999677746052),
    (1.0003, 0.001, 2.0, -1.0, 0.9999999996376516),
    (1.0006, 0.001, 1.2, -0.2, 0.9999999969563031),
    (1.0009, 0.001, 0.4, 0.25, 0.9999999919035696),
    (0.9908, 0.0092, 1.0, 0.0, 0.9999835158035406),
    (1.0, 0.0092, 0.0, 1.0, 0.9999920176004313),
    (1.009199999999991, 0.0092, 2.0, -1.0, 1.0),
    (0.9908000000000184, 0.0092, 1.2, -0.2, 0.9999856301618957),
    (1.0091999999999817, 0.0092, 0.4, 0.25, 1.0),
    (0.9908000184, 0.0092, 1.0, 0.0, 0.9999835158241187),
    (1.0091999816000001, 0.0092, 0.0, 1.0, 1.0),
    (0.9908184, 0.0092, 2.0, -1.0, 0.9999969129119043),
    (1.0091816, 0.0092, 1.2, -0.2, 0.9999999999795681),
    (0.99172, 0.0092, 0.4, 0.25, 0.9999539935687062),
    (0.99448, 0.0092, 1.0, 0.0, 0.9999881462267313),
    (0.99724, 0.0092, 0.0, 1.0, 0.9999873464917959),
    (1.0, 0.0092, 2.0, -1.0, 0.9999993426068658),
    (1.00276, 0.0092, 1.2, -0.2, 0.9999976856367387),
    (1.00552, 0.0092, 0.4, 0.25, 0.9999942471575788),
    (1.00828, 0.0092, 1.0, 0.0, 0.9999999401473416),
    (0.95, 0.05, 0.0, 1.0, 0.998499498591064),
    (1.0, 0.05, 2.0, -1.0, 0.9998970018094973),
    (1.04999999999995, 0.05, 1.2, -0.2, 1.0),
    (0.9500000000001, 0.05, 0.4, 0.25, 0.9981990145110905),
    (1.0499999999999001, 0.05, 1.0, 0.0, 1.0),
    (0.9500000999999999, 0.05, 0.0, 1.0, 0.9984995002410453),
    (1.0499999, 0.05, 2.0, -1.0, 1.0),
    (0.9501, 0.05, 1.2, -0.2, 0.9989836236167645),
    (1.0499, 0.05, 0.4, 0.25, 0.9999999432624112),
    (0.955, 0.05, 1.0, 0.0, 0.9989554836429043),
    (0.97, 0.05, 0.0, 1.0, 0.9988937596925193),
    (0.985, 0.05, 2.0, -1.0, 0.9998107760633976),
    (1.0, 0.05, 1.2, -0.2, 0.9996849963281301),
    (1.015, 0.05, 0.4, 0.25, 0.9995479803909687),
    (1.03, 0.05, 1.0, 0.0, 0.9999377929754587),
    (1.045, 0.05, 0.0, 1.0, 0.999993752399994),
    (0.9, 0.1, 2.0, -1.0, 0.9963),
    (1.0, 0.1, 1.2, -0.2, 0.9982172925003785),
    (1.0999999999999002, 0.1, 0.4, 0.25, 1.0),
    (0.9000000000002, 0.1, 1.0, 0.0, 0.9937926042795595),
    (1.0999999999998, 0.1, 0.0, 1.0, 1.0),
    (0.9000002, 0.1, 2.0, -1.0, 0.9963000072000008),
    (1.0999998000000002, 0.1, 1.2, -0.2, 0.9999999999999923),
    (0.9002, 0.1, 0.4, 0.25, 0.9918084338189979),
    (1.0998, 0.1, 1.0, 0.0, 0.9999999909567706),
    (0.91, 0.1, 0.0, 1.0, 0.9927249934015804),
    (0.9400000000000001, 0.1, 2.0, -1.0, 0.9976739574975839),
    (0.97, 0.1, 1.2, -0.2, 0.9971184312290305),
    (1.0, 0.1, 0.4, 0.25, 0.9966143500998468),
    (1.03, 0.1, 1.0, 0.0, 0.9989892120344624),
    (1.06, 0.1, 0.0, 1.0, 0.9995020881156866),
    (1.09, 0.1, 2.0, -1.0, 0.9999971389844455),
    (0.8, 0.2, 1.2, -0.2, 0.9673021343590907),
    (1.0, 0.2, 0.4, 0.25, 0.9850185384656939),
    (1.1999999999998, 0.2, 1.0, 0.0, 1.0),
    (0.8000000000004001, 0.2, 0.0, 1.0, 0.9623540552823068),
    (1.1999999999996, 0.2, 2.0, -1.0, 1.0),
    (0.8000004000000001, 0.2, 1.2, -0.2, 0.9673021767539856),
    (1.1999996, 0.2, 0.4, 0.25, 0.9999999999736658),
    (0.8004, 0.2, 1.0, 0.0, 0.966312166124191),
    (1.1996, 0.2, 0.0, 1.0, 0.9999999223869447),
    (0.8200000000000001, 0.2, 2.0, -1.0, 0.9753651512943308),
    (0.8800000000000001, 0.2, 1.2, -0.2, 0.9768140161094097),
    (0.9400000000000001, 0.2, 0.4, 0.25, 0.9778704845022844),
    (1.0, 0.2, 1.0, 0.0, 0.9893192209948879),
    (1.06, 0.2, 0.0, 1.0, 0.9927823713210181),
    (1.12, 0.2, 2.0, -1.0, 0.9993297153032198),
    (1.1800000000000002, 0.2, 1.2, -0.2, 0.9998946090049384),
    (0.7, 0.3, 0.4, 0.25, 0.9098136558758796),
    (1.0, 0.3, 1.0, 0.0, 0.9717878213091774),
    (1.2999999999997, 0.3, 0.0, 1.0, 1.0),
    (0.7000000000005999, 0.3, 2.0, -1.0, 0.9163000000001512),
    (1.2999999999994, 0.3, 1.2, -0.2, 1.0),
    (0.7000006, 0.3, 0.4, 0.25, 0.9098137094265756),
    (1.2999994000000001, 0.3, 1.0, 0.0, 0.9999999999998703),
    (0.7006, 0.3, 0.0, 1.0, 0.9079879771664322),
    (1.2994, 0.3, 2.0, -1.0, 0.9999999959332142),
    (0.73, 0.3, 1.2, -0.2, 0.9178914331428294),
    (0.82, 0.3, 0.4, 0.25, 0.9294670247222471),
    (0.9099999999999999, 0.3, 1.0, 0.0, 0.954820505737976),
    (1.0, 0.3, 0.0, 1.0, 0.9662744453100681),
    (1.0899999999999999, 0.3, 2.0, -1.0, 0.9917357079265864),
    (1.18, 0.3, 1.2, -0.2, 0.9955055929161019),
    (1.27, 0.3, 0.4, 0.25, 0.9991480061435207),
]


def test_limb_interpolant():
    """P21b: occult_hybrid's per-call limb interpolant (1 - k <= z < 1 + k,
    used for 0 < k <= LIMB_K_MAX when >= LIMB_MIN_LANES samples of a call
    lie on the band; _limb_coeffs / _occult_limb_batch) is within
    SERIES_TOL = 1e-8 (0.01 ppm) of an independent mpmath radial quadrature
    (LIMB_REF, incl. both contacts and 1e-12 from them) and of the exact
    kernel (occult_exact) on dense grids for k = 1e-3 ... 0.25 and 12 LD
    laws; it is really used (equals a direct _occult_limb_batch call) on
    both the per-sample and the stretch path, also for a short call of an
    Earth-sized moon (P21d), not for too few band samples, k > LIMB_K_MAX
    or occult_exact; flux <= 1; and the loop is vectorised. P21d: the
    offline table (LIMB_TAB, limb_table.csv) equals a fresh
    _limb_table_build(), reproduces the per-call build (_limb_coeffs_exact)
    on the band to <= 5e-11 (scaled by the LD weights; measured 8e-12) for
    k in [LIMB_TAB_KMIN, LIMB_TAB_KMAX] incl. the range ends, and below the
    range the per-call build is used."""
    import numba
    from pandoramoon.occult import (SERIES_TOL, LIMB_K_MAX, LIMB_MIN_LANES, LIMB_W, LIMB_TAB,
                                    LIMB_TAB_KMIN, LIMB_TAB_KMAX, _limb_coeffs, _limb_coeffs_exact,
                                    _limb_table_build, _occult_limb_batch)
    assert np.max(np.abs(LIMB_TAB - _limb_table_build())) <= 1e-13, "limb_table.csv is stale"
    xg = np.linspace(-1, 1, 401)
    for k in np.concatenate([[LIMB_TAB_KMIN, LIMB_TAB_KMAX], np.exp(np.random.default_rng(8).uniform(
            np.log(LIMB_TAB_KMIN), np.log(LIMB_TAB_KMAX), 150))]):
        for u1, u2 in ((0.4, 0.26), (0.0, 1.0), (2.0, -1.0), (1.0, 0.0)):
            dc = _limb_coeffs(k, u1, u2) - _limb_coeffs_exact(k, u1, u2)
            d = max(np.abs(np.polynomial.polynomial.polyval(xg, dc[:LIMB_W])).max(),
                    np.abs(np.polynomial.polynomial.polyval(xg, dc[LIMB_W:])).max())
            om = 1 / (1 - u1 / 3 - u2 / 6)
            assert d <= 5e-11 * om * (abs(1 - u1 - 2 * u2) + abs(u1 + 2 * u2) + abs(u2)), (k, u1, u2, d)
    for k in (LIMB_TAB_KMIN / 2, LIMB_TAB_KMAX * 1.1):   # outside the table: the per-call build
        assert np.array_equal(_limb_coeffs(k, 0.4, 0.26), _limb_coeffs_exact(k, 0.4, 0.26))
    rng = np.random.default_rng(21)
    for z, k, u1, u2, ref in LIMB_REF:
        if k > LIMB_K_MAX:
            continue
        n = 4 * LIMB_MIN_LANES
        zs = np.concatenate([np.linspace(1 - k, 1 + k, n, endpoint=False), [z]])
        got = occult_hybrid(zs, k, u1, u2)[-1]
        assert abs(got - ref) <= SERIES_TOL, f"z={z!r} k={k} u=({u1},{u2}): {got!r} vs mpmath {ref!r}"
    lds = [(0.4, 0.26), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, -1.0), (0.8, -0.1)]
    lds += [(2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)) for q1, q2 in rng.uniform(0, 1, (6, 2))]
    n_used = 0
    for k in (1e-3, 0.0092, 0.03, 0.1, 0.2, LIMB_K_MAX):
        zl = np.concatenate([np.linspace(1 - k, 1 + k, 600, endpoint=False), rng.uniform(1 - k, 1 + k, 300),
                             1 - k + 2 * k * np.logspace(-14, -1, 27), 1 + k - 2 * k * np.logspace(-14, -1, 27)])
        zs = np.concatenate([np.linspace(0, 1 - k, 300, endpoint=False), zl, [1.5]])
        smooth = np.sort(np.concatenate([zs, np.linspace(0, 1.4, 3000)]))   # stretch path
        dense = np.sort(np.concatenate([zs, np.linspace(1 - k, 1 + k, int(64 / k))]))
        for u1, u2 in lds:
            for z in (zs, smooth, dense):
                hyb = occult_hybrid(z, k, u1, u2)
                d = np.abs(hyb - occult_exact(z, k, u1, u2)).max()
                assert d <= SERIES_TOL, f"k={k} u=({u1:.3f},{u2:.3f}) n={len(z)}: {d:.2e}"
                assert hyb.max() <= 1.0
                on = (z >= 1 - k) & (z < 1 + k)
                direct = np.empty(on.sum())
                _occult_limb_batch(z[on] ** 2, len(direct), k, _limb_coeffs(k, u1, u2), direct)
                assert np.max(np.abs(hyb[on] - direct)) <= 1e-15, f"k={k} n={len(z)}: interpolant not used"
                n_used += 1
    assert n_used == 216, n_used
    # P21d: a Kepler-like moon call (k ~ 0.0094, ~700 samples, 16 band samples)
    k, u1, u2 = 0.0094, 0.4, 0.26
    zm = np.concatenate([np.linspace(0, 1 - k, 600, endpoint=False), np.linspace(1 - k, 1 + k, 16, endpoint=False),
                         np.full(80, 1.3)])
    on = (zm >= 1 - k) & (zm < 1 + k)
    direct = np.empty(on.sum())
    _occult_limb_batch(zm[on] ** 2, len(direct), k, _limb_coeffs(k, u1, u2), direct)
    hyb = occult_hybrid(zm, k, u1, u2)
    assert np.max(np.abs(hyb[on] - direct)) <= 1e-15, "moon call: interpolant not used"
    assert np.max(np.abs(hyb - occult_exact(zm, k, u1, u2))) <= SERIES_TOL
    # not used: occult_exact, too few band samples, k > LIMB_K_MAX
    k, u1, u2 = 0.1, 0.4, 0.26
    zl = np.linspace(1 - k, 1 + k, 2000, endpoint=False)
    from pandoramoon.occult import _occult_hybrid_samples
    assert np.array_equal(occult_exact(zl, k, u1, u2), _occult_hybrid_samples(zl * zl, k, u1, u2, False, False))
    few = np.concatenate([zl[::4][:LIMB_MIN_LANES - 1], np.full(1000, 2.0)])
    assert np.max(np.abs(occult_hybrid(few, k, u1, u2) - occult_exact(few, k, u1, u2))) <= 1e-15
    k = LIMB_K_MAX + 0.05
    zl = np.linspace(1 - k, 1 + k, 2000, endpoint=False)
    assert np.max(np.abs(occult_hybrid(zl, k, u1, u2) - occult_exact(zl, k, u1, u2))) <= 1e-15
    fresh = numba.njit(fastmath=True, error_model="numpy")(_occult_limb_batch.py_func)
    zl = np.linspace(0.9, 1.1, 64) ** 2
    fresh(zl, 64, 0.1, _limb_coeffs(0.1, 0.4, 0.26), np.empty(64))
    ir = "".join(fresh.inspect_llvm().values())
    assert "vector.body" in ir and "llvm.sqrt.v4f64" in ir, "_occult_limb_batch not vectorised"


def test_occult_blocks_match_single_samples():
    """P23: the per-sample path of occult_hybrid (_occult_hybrid_samples:
    interior approximation (P21c interpolant; P14 series until P21c)
    evaluated in place per block of BLOCK_N samples with a select,
    exact samples collected only from blocks that hold any, exact batch
    padded to 4 lanes, P25) gives every sample the value it gets alone
    (length-1 call), for smooth, shuffled and block-boundary inputs with
    off-star (incl. z < 0), interior, limb, z == k and fully occulted
    samples, any length (not multiples of BLOCK_N, single blocks with one
    exact sample), with and without the interpolant; limb interpolant off (its
    use depends on the call's sample count, see test_limb_interpolant)."""
    from pandoramoon.occult import SERIES_TOL, BLOCK_N, _use_interior, _occult_hybrid_samples
    rng = np.random.default_rng(20261023)
    n_one = 0
    for t in range(400):
        k = float(rng.choice([10 ** rng.uniform(-3, 0.3), 0.1, 0.0092, 1.0, 1.5]))
        u1, u2 = rng.uniform(0, 1), rng.uniform(-0.3, 0.6)
        tol = SERIES_TOL if t % 3 else 0.0
        n = int(rng.choice([1, 2, BLOCK_N - 1, BLOCK_N, BLOCK_N + 1, int(rng.integers(3, 700))]))
        kind = t % 4
        if kind == 0:
            z = np.sqrt(rng.uniform(0, 1.2) ** 2 + np.linspace(-1.6, 1.6, n) ** 2)
        elif kind == 1:
            z = rng.uniform(-0.1, 1.3 + k, n)
        elif kind == 2:     # one exact sample per block, rest series / off
            z = np.where(np.arange(n) % BLOCK_N == BLOCK_N // 2, 1.0, rng.choice([0.2, 5.0], n))
        else:
            z = np.linspace(-0.05, 1.1 + k, n)
            for p in (k, abs(1 - k), 1 + k, max(k - 1, 0.0)):
                if n > 10:
                    z[rng.integers(0, n)] = p
        interior = _use_interior(k, tol)
        z = z * np.abs(z)       # P26: the internal paths take z^2 (z < 0 stays negative)
        got = _occult_hybrid_samples(z, k, u1, u2, interior, False)
        one = np.array([_occult_hybrid_samples(z[i:i + 1], k, u1, u2, interior, False)[0] for i in range(n)])
        d = np.max(np.abs(got - one), initial=0)
        assert d <= 1e-14, f"case {t} (kind {kind}, k {k}, n {n}): blocks vs single samples {d:.1e}"
        n_one += kind == 2 and n > BLOCK_N
    assert n_one > 20

def test_occult_stretches_match_samples():
    """P12: the stretch pass of occult_hybrid (_occult_hybrid_stretches:
    off-star stretches filled, interior stretches in place (P21c
    interpolant; P14 series until P21c), exact stretches gathered) equals
    the per-sample pass (_occult_hybrid_samples) to <= 1e-15 (rounding, in
    place vs per block), and occult_hybrid /
    occult_exact equal the per-sample pass whichever path they take, on
    smooth transits (single and multiple, with a wobble), random order,
    contact points and z == k planted in smooth arrays (since P26 the
    stretch pass handles z == k as the per-sample pass, in _sqrt_lanes),
    fully occulted samples (k >= 1, these make the stretch pass give up),
    empty and short arrays, with and without the interpolants. Both passes
    take z^2 (P26). _long_stretches() picks the
    stretch path for smooth and the per-sample path for shuffled input,
    and both paths are exercised."""
    from pandoramoon.occult import (SERIES_TOL, _use_interior, _occult_hybrid_stretches,
                                    _occult_hybrid_samples, _long_stretches, occult_hybrid,
                                    occult_exact, LIMB_K_MAX, LIMB_MIN_LANES)
    rng = np.random.default_rng(20261012)
    n_ok = n_bail = n_long = n_short = 0
    for t in range(1500):
        k = float(rng.choice([10 ** rng.uniform(-4, 0.3), 0.1, 0.0092, 1.0, 1.5, 0.5]))
        u1, u2 = rng.uniform(0, 1), rng.uniform(-0.3, 0.6)
        tol = SERIES_TOL if t % 4 else 0.0
        n = int(rng.choice([0, 1, 5, int(rng.integers(10, 3000))]))
        kind = t % 5
        if kind == 0:      # one transit
            z = np.sqrt(rng.uniform(0, 1.2) ** 2 + np.linspace(-2, 2, n) ** 2)
        elif kind == 1:    # several transits with a wobble
            x = np.linspace(-3, 3, n)
            z = np.abs(np.abs(np.sin(x * rng.uniform(1, 5))) * 1.3 + 0.01 * np.sin(x * 40))
        elif kind == 2:    # random order
            z = rng.uniform(0, 1.3, n)
        elif kind == 3:    # contact points and z == k planted in a smooth array
            z = np.linspace(0, 1.3, n)
            for p in (k, abs(1 - k), 1 + k, max(k - 1, 0.0)):
                if n > 10:
                    for v in (p, np.nextafter(p, 2), np.nextafter(p, -2)):
                        z[rng.integers(0, n)] = abs(v)
        else:              # long off-star stretch, then interior
            z = np.concatenate([np.full(n // 2, 2.0), np.linspace(0, 0.5, n - n // 2)])
        interior = _use_interior(k, tol)
        z_c = 1 - k if interior else 0.0
        limb = bool(tol) and 0 < k <= LIMB_K_MAX   # as _occult_hybrid
        z2 = z * z              # P26: the internal paths take z^2
        ref = _occult_hybrid_samples(z2, k, u1, u2, interior, limb)
        ok, got = _occult_hybrid_stretches(z2, k, u1, u2, interior, limb)
        if ok:
            n_ok += 1
            d = np.max(np.abs(got - ref), initial=0)
            assert d <= 1e-15, f"case {t} (kind {kind}, k {k}): stretches vs samples {d:.1e}"
        else:
            n_bail += 1
        hyb = (occult_hybrid if tol else occult_exact)(z, k, u1, u2)
        d = np.max(np.abs(hyb - ref), initial=0)
        assert d <= 1e-15, f"case {t}: occult_hybrid vs samples {d:.1e}"
        if n >= 1000 and kind in (0, 2, 4):
            cls = np.where(z >= 1 + k, 0, np.where(z < z_c, 1, 2))
            mixed = np.max(np.bincount(cls, minlength=3)) < 0.8 * n
            if _long_stretches(z2, k, z_c * z_c):
                n_long += 1
                assert not (kind == 2 and mixed), f"case {t}: shuffled input taken as long stretches"
            else:
                n_short += 1
    assert n_ok > 300 and n_bail > 100, f"paths: stretches {n_ok}, gave up {n_bail}"
    assert n_long > 50 and n_short > 20, f"_long_stretches: {n_long} long, {n_short} short"


def test_moon_subruns_exact():
    """P10: the likelihood path (_window_model_runs) evaluates only the
    sub-runs of the window where the planet or the circular moon can be on
    the star (ellipse_subruns: planet |x_bary| bound, moon chord distance
    minus a curvature bound per cell). Over random systems with dense and
    coarse sampling (incl. grazing / non-transiting planets, short moon
    periods, large moons, mutual eclipses): every window sample outside
    the sub-runs has planet and moon flux exactly 1 in the full-window
    model (_window_model), the kept samples have the same fluxes
    (<= 1e-13; since P21b <= SERIES_TOL in a few draws (10 of 1400), where only
    one of the two calls used the limb interpolant), the sub-runs are ascending and disjoint, and on dense
    sampling they are tight (on average < 70 % of the window, so that a
    silently disabled P10 shows). Extra draws with moon periods of 0.2-1.5 d
    exercise the curvature bound (dropping it is caught). P7c: two thirds
    of the draws have an eccentric moon (e up to 0.97, random w), which
    takes the same runs / sub-runs path since P7c."""
    pm = sys.modules["pandoramoon.pandora"]
    rng = np.random.default_rng(20261013)
    from pandoramoon.occult import SERIES_TOL
    kept = []
    n_split = 0
    n_limb_diff = 0
    for j in range(1400):
        dense = j % 2 == 0 or j >= 800     # j >= 800: short moon periods
        ecc = 0.0 if j % 3 == 0 else rng.uniform(0, 0.97)   # P7c: eccentric moons too
        per_b = rng.uniform(2, 30) if dense else rng.uniform(5, 400)
        if dense:
            dtc = rng.uniform(0.0007, 0.004)
            t = np.concatenate([np.arange(-1.5, 1.5, dtc) + per_b * e for e in range(3)])
        else:
            t = np.sort(rng.uniform(0, 3 * per_b, int(rng.integers(100, 20000))))
        a_b = rng.uniform(5, 300)
        r_p = rng.uniform(0.01, 0.3)
        args = (rng.uniform(0, 1), rng.uniform(-0.2, 0.5), 6.96e8 * rng.uniform(0.3, 2), per_b, a_b, r_p,
                rng.uniform(-1.5, 1.5), 0.0, 0.0, rng.uniform(-0.3, 0.3) if dense else rng.uniform(0, per_b),
                rng.uniform(-0.05, 0.05), 10 ** rng.uniform(25, 28.5), rng.uniform(0.001, 0.15),
                rng.uniform(0.3, 30) if j < 800 else rng.uniform(0.2, 1.5), rng.uniform(),
                rng.uniform(0, 360), rng.uniform(0, 180), ecc, rng.uniform(0, 360),
                10 ** rng.uniform(22, 27), per_b + rng.uniform(-0.01, 0.01), 0.0, 1e9, t, None, True)
        idx, fp, fm = pm._window_model(*args)
        los, his, fp2, fm2 = pm._window_model_runs(*args)
        assert np.all(his >= los) and np.all(los[1:] >= his[:-1]), f"draw {j}: sub-runs not ascending/disjoint"
        idx2 = np.concatenate([np.arange(a, b) for a, b in zip(los, his)] or [np.empty(0, np.int64)])
        pos = np.searchsorted(idx, idx2)
        assert np.array_equal(idx[np.minimum(pos, max(len(idx) - 1, 0))], idx2), f"draw {j}: sample outside the window"
        d = max(np.max(np.abs(fp[pos] - fp2), initial=0), np.max(np.abs(fm[pos] - fm2), initial=0))
        # (P21b: the limb interpolant is used per call by the number of band
        # samples, so the full window and the sub-runs may differ by its
        # error, <= SERIES_TOL; else <= 1e-13)
        assert d <= SERIES_TOL, f"draw {j}: kept fluxes differ by {d:.1e}"
        n_limb_diff += d > 1e-13
        skipped = np.ones(len(idx), bool)
        skipped[pos] = False
        bad = np.flatnonzero(skipped & ((fp != 1) | (fm != 1)))
        assert len(bad) == 0, f"draw {j}: {len(bad)} skipped samples with flux != 1"
        if dense and len(idx) and j < 800:
            kept.append(len(idx2) / len(idx))
            n_split += len(idx2) < len(idx)
    assert n_split > 100, f"sub-runs split the window only {n_split} times"
    assert n_limb_diff < 50, f"{n_limb_diff} draws differ by > 1e-13"
    assert np.mean(kept) < 0.7, f"sub-runs keep {np.mean(kept):.0%} of the dense windows"


def test_moon_subruns_geometric():
    """P10 / P20: ellipse_subruns() directly against the orbit loop, over
    many cheap draws: every window sample with the moon (z_m < 1 + r_m) or
    the planet (z_p < 1 + r_p) on the star lies in a sub-run, and the
    sub-runs are ascending, disjoint and inside the window runs. Half the
    draws are high-curvature moons (small star, massive planet, moon
    periods of 0.15-0.5 d, near face-on orbits, |b| up to 1.6), where the
    moon's sky track loops through the limb: dropping the curvature term
    of the cell bound, the A_x / A_y range of the projected circle (P20) or
    the cells next to the planet interval is caught. The other half are
    wide random systems (incl. non-transiting barycentres). P7c: two thirds
    of the draws have an eccentric moon (e up to 0.99, random w; orbit from
    ellipse_ecc_z_runs)."""
    g = sys.modules["pandoramoon.grids"]
    el = sys.modules["pandoramoon.ellipse"]
    pm = sys.modules["pandoramoon.pandora"]
    rng = np.random.default_rng(20261020)
    n_moon = 0
    n_ecc = 0
    for j in range(16000):
        per_b = rng.uniform(5, 30)
        dtc = rng.uniform(0.0007, 0.004)
        t = np.concatenate([np.arange(-1.5, 1.5, dtc) + per_b * e for e in range(3)])
        if j % 2 == 0:
            inc = rng.uniform(0, 25) + (155 if rng.uniform() < 0.5 else 0)
            R_star = 6.96e8 * rng.uniform(0.2, 0.6)
            M_p = 10 ** rng.uniform(27.5, 28.5)
            per_m = rng.uniform(0.15, 0.5)
            b = rng.choice([-1, 1]) * rng.uniform(0.6, 1.6)
        else:
            inc = rng.uniform(0, 180)
            R_star = 6.96e8 * rng.uniform(0.3, 2)
            M_p = 10 ** rng.uniform(25, 28.5)
            per_m = rng.uniform(0.2, 30)
            b = rng.uniform(-1.6, 1.6)
        r_p = rng.uniform(0.01, 0.2)
        r_m = rng.uniform(0.005, 0.1)
        a_b = rng.uniform(10, 100)
        M_m = 10 ** rng.uniform(23, 26.5)
        Om = rng.uniform(0, 360)
        tau = rng.uniform()
        t0 = rng.uniform(-0.3, 0.3)
        t0o = rng.uniform(-0.05, 0.05)
        # P7c: two thirds of the draws with an eccentric moon: e up to 0.9,
        # and in the high-curvature half e = 0.6-0.9 (the periapsis term
        # 1 / (1 - e)^2 of the curvature bound: dropping it is caught by ~2 %
        # of these draws, scratch_p7b/fam.py), in the wide half 0.9-0.99
        # (Markley cell points above ECC_FAST_MAX)
        if j % 3 == 0:
            ecc = 0.0
        elif j % 3 == 1:
            ecc = rng.uniform(0, 0.9)
        else:
            ecc = rng.uniform(0.6, 0.9) if j % 2 == 0 else rng.uniform(0.9, 0.99)
        w_m = rng.uniform(0, 360)
        a_moon, unph = pm._moon_setup(R_star, per_b, a_b, M_p, M_m, per_m, ecc, r_p, r_m, 1e9)
        if unph:
            continue
        reach = pm._cull_reach(a_moon, ecc, r_p, r_m, unph, None)
        ok, los, his, ct, cd, tdur, sh = g.x_bary_runs_sorted(t, a_b, per_b, t0, t0o, per_b, 0.0, 0.0, reach)
        if not ok:
            continue
        q = M_m / M_p
        if ecc == 0:
            _, _, _, _, zm2, zp2 = el.ellipse_z_runs(a_moon, per_m, tau, Om, inc, t, los, his, ct, cd, tdur, sh, q, b)
        else:
            _, _, _, _, zm2, zp2 = el.ellipse_ecc_z_runs(a_moon, per_m, ecc, tau, Om, w_m, inc, t, los, his, ct, cd,
                                                        tdur, sh, q, b)
        zm, zp = np.sqrt(zm2), np.sqrt(zp2)     # P26: z^2
        l2, h2, _, _ = el.ellipse_subruns(a_moon, per_m, tau, Om, inc, t, los, his, ct, cd, tdur, sh, q, b,
                                          r_p, r_m, ecc, w_m)
        n_ecc += ecc > 0 and bool(np.any(zm < 1 + r_m))
        assert np.all(h2 > l2) and np.all(l2[1:] >= h2[:-1]), f"draw {j}: sub-runs not ascending/disjoint"
        widx = np.concatenate([np.arange(a, c) for a, c in zip(los, his)] or [np.empty(0, np.int64)])
        kidx = np.concatenate([np.arange(a, c) for a, c in zip(l2, h2)] or [np.empty(0, np.int64)])
        assert np.all(np.isin(kidx, widx)), f"draw {j}: sub-run outside the window"
        on = (zm < 1 + r_m) | (zp < 1 + r_p)
        miss = np.flatnonzero(on & ~np.isin(widx, kidx))
        assert len(miss) == 0, f"draw {j}: {len(miss)} on-star samples outside the sub-runs"
        n_moon += bool(np.any(zm < 1 + r_m))
    assert n_moon > 8000, f"only {n_moon} draws with the moon on the star"
    assert n_ecc > 3000, f"only {n_ecc} eccentric draws with the moon on the star"


def test_planet_only_loglike_matches_general():
    """pandora_loglike_planet() (fused planet-only likelihood, SAMPLING.md
    Round 12) equals pandora_loglike() with the planet-only moon parameters
    PLANET_ONLY_MOON to rounding, for random planets on several data sets
    (2-9 transits, 30 / 10 min), including -inf outside the physical
    ranges, and it falls back to pandora_loglike() exactly for an eccentric
    planet."""
    import pandoramoon as pandora
    pm = sys.modules["pandoramoon.pandora"]
    rng = np.random.default_rng(20261009)
    worst = 0.0
    for case in range(6):
        per = float(rng.uniform(150, 450))
        epochs = int(rng.integers(2, 10))
        cad = 48 if case % 2 else 144
        t = []
        for e in range(epochs):
            t.append(11.0 + e * (per + 0.01) + np.arange(-1.0, 1.0, 1 / cad))
        t = np.concatenate(t)
        y = 1 + rng.normal(0, 3e-4, len(t))
        like = pandora.Likelihood(t, y, np.full(len(t), 3e-4), epoch_distance=per + 0.01)
        tail = like._tail
        for j in range(300):
            u1 = float(rng.uniform(0, 1))
            u2 = float(rng.uniform(-0.5, 0.5)) if j % 25 else -0.9        # j % 25 == 0: invalid LD
            ecc = 0.2 if j % 50 == 1 else 0.0
            a = float(rng.uniform(150, 300))
            args = (u1, u2, 696342e3, per * float(rng.uniform(0.99, 1.01)), a, float(rng.uniform(0.002, 0.15)),
                    float(rng.uniform(0, 1.2)), 30.0, ecc, 11.0, float(rng.uniform(-0.05, 0.05)), 1.9e27)
            ref = pm.pandora_loglike(*args, *pm.PLANET_ONLY_MOON, *tail)
            new = pm.pandora_loglike_planet(*args, *tail)
            if not np.isfinite(ref):
                assert not np.isfinite(new), (case, j, ref, new)
                continue
            d = abs(new - ref)
            if ecc != 0.0:
                assert d == 0.0, (case, j, d)
            worst = max(worst, d / max(1.0, abs(ref)))
    assert worst < 1e-12, worst
    print(f"     planet-only likelihood: max rel |dlogL| {worst:.1e}")


def test_planet_only_loglike_supersampled():
    """pandora_loglike_planet() with supersampling (SAMPLING.md backlog 8): the fast path now averages the planet
    flux over the sub-exposures of each cadence. It must equal pandora_loglike() with PLANET_ONLY_MOON (to
    rounding) and an independent dense reference: pandora_flux() on the supersampled grid (resampled to the
    cadences by pandora's own resample) and chi^2 = sum((f - y)^2 / sigma^2), for factors 2-11, partial blocks at
    the window edges and random planets."""
    import pandoramoon as pandora
    pm = sys.modules["pandoramoon.pandora"]
    rng = np.random.default_rng(20261011)
    worst_gen = 0.0
    worst_ref = 0.0
    for case, ssf in enumerate((2, 3, 5, 7, 11)):
        per = float(rng.uniform(150, 450))
        epochs = int(rng.integers(2, 6))
        cad = 48
        # Pandora's supersampled grid: ssf sub-exposure mid-times per 30-min cadence
        p = pandora.model_params()
        p.t0_bary, p.epochs, p.epoch_duration, p.cadences_per_day = 11.0, epochs, 2.0, cad
        p.epoch_distance, p.supersampling_factor = per + 0.01, ssf
        t = pandora.time(p).grid()
        n_out = len(t) // ssf
        sig = 3e-4
        y = 1 + rng.normal(0, sig, n_out)
        like = pandora.Likelihood(t, y, np.full(n_out, sig), epoch_distance=per + 0.01, supersampling_factor=ssf)
        tail = like._tail
        for j in range(60):
            u1 = float(rng.uniform(0, 1))
            u2 = float(rng.uniform(-0.4, 0.4))
            a = float(rng.uniform(150, 300))
            args = (u1, u2, 696342e3, per * float(rng.uniform(0.999, 1.001)), a, float(rng.uniform(0.002, 0.15)),
                    float(rng.uniform(0, 1.1)), 30.0, 0.0, 11.0, float(rng.uniform(-0.05, 0.05)), 1.9e27)
            ref = pm.pandora_loglike(*args, *pm.PLANET_ONLY_MOON, *tail)
            new = pm.pandora_loglike_planet(*args, *tail)
            if not np.isfinite(ref):                     # unphysical limb darkening: both -inf
                assert not np.isfinite(new), (ssf, j, ref, new)
                continue
            worst_gen = max(worst_gen, abs(new - ref) / max(1.0, abs(ref)))
            if j % 10 == 0:
                _, _, f = pm.pandora_flux(*args, *pm.PLANET_ONLY_MOON, per + 0.01, ssf, 0.0, 1.1, t)
                dense = -0.5 * float(np.sum(((f - y) / sig) ** 2))
                worst_ref = max(worst_ref, abs(new - dense) / max(1.0, abs(dense)))
    assert worst_gen < 1e-12, worst_gen
    assert worst_ref < 1e-9, worst_ref
    print(f"     supersampled planet-only likelihood: max rel |dlogL| {worst_gen:.1e} vs general, "
          f"{worst_ref:.1e} vs dense reference")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def run(verbose=False):
    failures = 0
    t_all = time.perf_counter()
    for fn in TESTS:
        t0 = time.perf_counter()
        try:
            fn()
            status, msg = "PASS", ""
        except Exception as e:
            status, msg = "FAIL", f"{type(e).__name__}: {e}"
            failures += 1
        timing = f"  ({(time.perf_counter() - t0) * 1e3:.0f} ms)" if verbose else ""
        print(f"{status}  unit:{fn.__name__}{timing}")
        if msg:
            print(f"        {msg}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'}: {failures} failing unit test(s); "
          f"total {time.perf_counter() - t_all:.2f} s")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run(verbose="-v" in sys.argv) else 0)
