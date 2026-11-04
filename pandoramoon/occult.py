import numpy as np
from numpy import sqrt, pi, arccos, abs, log
from numba import jit

from pandoramoon.trig import acos_poly


def _table_path(name):
    """Path of an offline coefficient table shipped with the package"""
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _table_load(name, shape):
    """Offline table `name` (CSV, '#' header lines) as a C-contiguous float64
    array of the given shape. A missing file is an error (zeros would give wrong fluxes), except while the
    tables are being generated (environment variable PANDORA_BUILDING_TABLES=1). Module-level arrays are
    compile-time constants of the numba functions that read them; pandoramoon's cache guard (_config.py) clears
    the numba cache when a table file changes."""
    import os
    path = _table_path(name)
    if not os.path.exists(path):
        if os.environ.get("PANDORA_BUILDING_TABLES"):
            return np.zeros(shape)
        raise FileNotFoundError(f"pandoramoon: occultation table {path} is missing (reinstall the package)")
    T = np.ascontiguousarray(np.loadtxt(path, delimiter=",", ndmin=2), dtype=np.float64)
    assert T.shape == shape, (name, T.shape, shape)
    return T


def _table_save(name, T, header):
    """Write an offline table as CSV; %.17g round-trips every float64 exactly"""
    np.savetxt(_table_path(name), T, delimiter=",", fmt="%.17g", header=header)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def create_occult_cache(u1, u2, dim):
    """2D Cache of size dim*dim with Mandel-Agol occultation values
    as a function of quadratic limb darkening values (u1, u2), k (=radius ratio)"""
    z_max = 1.1
    k_min = 0.001
    k_max = 0.1
    zs = np.linspace(0, z_max, dim)
    ks = np.linspace(k_min, k_max, dim)
    fs = np.empty((dim, dim), dtype="float32")
    for count, k in enumerate(ks):
        fs[count] = occult(zs, k, u1, u2)
    return (fs, ks, zs)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def read_occult_cache(zs_target, k, cache):
    """Read nearest neighbors from 2D cache and perform bilinear interpolation"""
    fs, ks, zs = cache
    flux = np.ones(len(zs_target))
    idx_k = int(np.ceil((k - ks[0]) / (ks[1] - ks[0])))
    if idx_k < 1:
        idx_k = 1
    if idx_k > len(ks) - 1:
        idx_k = len(ks) - 1
    ratio_k = (ks[idx_k] - k) / (ks[idx_k] - ks[idx_k - 1])
    curve = fs[idx_k - 1] * ratio_k + fs[idx_k] * (1 - ratio_k)
    for idx in range(len(zs_target)):
        if zs_target[idx] == 0:
            idx_z = 1
        else:
            idx_z = int(np.ceil(zs_target[idx] / zs[1]))
        if zs_target[idx] >= 1.1:
            continue

        ratio_z = (zs[idx_z] - zs_target[idx]) / (zs[idx_z] - zs[idx_z - 1])
        res = curve[idx_z - 1] * ratio_z + curve[idx_z] * (1 - ratio_z)
        if res > 1:
            res = 1
        if res < 0:
            res = 0
        flux[idx] = res

        # Some combinations of (u1,u2) cause flux>1 which is unphysical
        if flux[idx] > 1:
            flux[idx] = 1
    return flux


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def cci(r1, r2, d):
    """Circle-Circle-Intersect to calculate the area of asymmetric "lens"
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    if r1 < d - r2:
        return 0
    elif r1 >= d + r2:
        return pi * r2**2
    elif d - r2 <= -r1:
        return pi * r1**2
    else:
        return (
            r2**2 * arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            + r1**2 * arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            - 0.5 * sqrt((-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1))
        )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_small(zs, k, u1, u2):
    """Small body approximation by Mandel-Agol. Adequate for k<~0.01"""
    i = zs.size
    f = np.ones(i)
    b = abs(zs)
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    s_inv = 1 / s
    for j in range(i):
        if b[j] < 1 + k:
            m = sqrt(1 - min(b[j] ** 2, 1))
            limb_darkening = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
            area = cci(1, k, b[j])
            f[j] = (s - limb_darkening * area) * s_inv

            # Some combinations of (u1,u2) cause flux>1 which is unphysical
            if f[j] > 1:
                f[j] = 1
    return f


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellpicb(n, k):
    """The complete elliptical integral of the third kind
    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353
    Adapted from L. Kreidbergs C version in BATMAN
    (Kreidberg, L. 2015, PASP 957, 127)
    (https://github.com/lkreidberg/batman)
    which is translated from J. Eastman's IDL routine
    in EXOFAST (Eastman et al. 2013, PASP 125, 83)"""
    HALF_PI = 0.5 * pi
    kc = sqrt(1 - k**2)
    e = kc
    p = sqrt(n + 1)
    m0 = 1
    c = 1
    d = 1 / p
    for nit in range(1000):
        f = c
        c = d / p + c
        g = e / p
        d = 2 * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(1 - kc / g) > 1e-8:
            kc = 2 * sqrt(e)
            e = kc * m0
        else:
            return HALF_PI * (c * m0 + d) / (m0 * (m0 + p))
    raise ValueError("ellpicb did not converge")


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ell_combo(q, n, A, B, C):
    """A K(q) + B E(q) + C Pi(n, q) in one Bulirsch iteration (P3a).

    q is the modulus (K(q) = ellk(q), E(q) = ellec(q) up to their ~2e-8
    polynomial error), Pi(n, q) = ellpicb(n, q) = Bulirsch cel(kc, n+1, 1, 1).
    All three share the AGM sequence (kc, m0, e) of Bulirsch's cel
    algorithm (Bulirsch 1969, Numer. Math. 13, 305; as in ellpicb).
    K and E: cel(kc, 1, A + B, A + B kc^2); for p = 1 the cel p-sequence
    equals m0, so this pair needs no p update. Pi: cel(kc, n+1, C, C).
    Exact to ~1e-16 relative (quadratic convergence after the 1e-8 test),
    instead of ellk/ellec's ~2e-8."""
    p = sqrt(n + 1)
    return ell_combo_kc(sqrt(1 - q * q), p, A, B, C, C / p)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ell_combo_kc(kc, p, A, B, C, Cp):
    """ell_combo() with its operands given directly (P15): the complementary
    modulus kc = sqrt(1 - q^2) > 0, p = sqrt(n + 1) > 0 and Cp = C / p.
    Lets the caller form kc, p and C / p from (z, k) in closed form, without
    the cancellation of 1 - q^2 near q = 1 (Agol, Luger & Foreman-Mackey
    2020, eq. 41)."""
    a1 = A + B
    b1 = A + B * kc * kc
    a2 = C
    b2 = Cp
    e = kc
    m0 = 1.0
    for nit in range(1000):
        # K/E pair (its p equals m0)
        f1 = a1
        a1 = a1 + b1 / m0
        b1 = 2 * (b1 + f1 * kc)
        # Pi pair (as ellpicb)
        inv_p = 1 / p
        f2 = a2
        a2 = a2 + b2 * inv_p
        g = e * inv_p
        b2 = 2 * (b2 + f2 * g)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(g - kc) > 1e-8 * g:
            kc = 2 * sqrt(e)
            e = kc * m0
        else:
            return 0.5 * pi * (
                (b1 + a1 * m0) / (2 * m0 * m0) + (b2 + a2 * m0) / (m0 * (m0 + p))
            )
    raise ValueError("ell_combo did not converge")


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellec(k):
    a1 = 0.443251414630
    a2 = 0.062606012200
    a3 = 0.047573835460
    a4 = 0.017365064510
    b1 = 0.249983683100
    b2 = 0.092001800370
    b3 = 0.040696975260
    b4 = 0.005264496390
    m1 = 1 - k * k
    epsilon = 1e-14
    return (1 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) + (
        m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1 / (m1 + epsilon))
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellk(k):
    a0 = 1.386294361120
    a1 = 0.096663442590
    a2 = 0.035900923830
    a3 = 0.037425637130
    a4 = 0.014511962120
    b0 = 0.50
    b1 = 0.124985935970
    b2 = 0.068802485760
    b3 = 0.033283553460
    b4 = 0.004417870120
    m1 = 1 - k * k
    epsilon = 1e-14
    return (a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) - (
        (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1 + epsilon)
    )


# Routine originally from:
# PyTransit: fast and easy exoplanet transit modelling in Python
# Copyright (C) 2010-2020  Hannu Parviainen
# Modified by Michael Hippke 2021, based on a GPL3 license
# Modifications:
# - Increased numerical stability for k=0.6 (exactly), see comments below
# - Fixed an error for missing ed[i], must be >= in "if (z >= 0.5 + abs(k - 0.5)"
# - Restricted to quadratic limb-darkening
# - Caching a few expensive values, e.g. from sqrt
# - Replaced some array generations with floats or copies
# - Overall speed improvements: ~10%
@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult(zs, k, u1, u2):
    """Evaluates the transit model for an array of normalized distances.
    Parameters
    ----------
    z: 1D array
        Normalized distances
    k: float
        Planet-star radius ratio
    u1, u2: float
        Limb darkening coefficients
    Returns
    -------
    Transit model evaluated at `z`.
    """
    # No k -> 0.5 snap for |k - 0.5| < 1e-4 any more (bug #30: up to 150 ppm). The Case V/VI branch (edge of the
    # body at the disk centre) was removed in the 2026-10-13 cleanup: its values were always overwritten by
    # Case III/IV
    INV_PI = 1 / pi
    k2 = k**2
    lzs = len(zs)
    epsilon = 1e-14
    flux = np.empty(lzs)
    le = np.empty(lzs)
    ld = np.empty(lzs)
    ed = np.empty(lzs)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2

    for i in range(lzs):
        z = zs[i]
        if abs(z - k) < 1e-6:
            z = k + 1e-6

        # The source is unocculted
        if z >= 1 + k or z < 0:
            flux[i] = 1
            le[i] = 0
            ld[i] = 0
            ed[i] = 0
            continue

        # The source is completely occulted
        elif k >= 1 and z <= k - 1:
            flux[i] = 0
            le[i] = 1
            ld[i] = 1
            ed[i] = 1
            continue

        z2 = z**2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k**2 - z2

        # Star partially occulted and the occulting object crosses the limb
        if z >= abs(1 - k) and z <= 1 + k:
            kap1 = arccos(max(min((1 - k2 + z2) / (2 * z + epsilon), 1), -1))
            kap0 = arccos(max(min((k2 + z2 - 1) / (2 * k * z + epsilon), 1), -1))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
        # Occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            le[i] = k2

        # Occulting body partly occults the source and crosses the limb:
        # Table 3, Case III. Written as the complement of Case IV below
        # (z < 1 - k): the former condition (z >= 0.5 + |k - 0.5|, or
        # k > 0.5 and |1 - k| <= z < k) is the same set mathematically, but
        # 0.5 + |k - 0.5| can round one ulp above 1 - k, and z = 1 - k then
        # matched neither case (ld, ed left uninitialised, bug #29)
        if z >= 1 - k and z < 1 + k:
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)

            # Numerical stability: if k=0.06 then q is close to 1, ellpicb goes to hell
            if (1 - q) < 1e-8:
                q -= 1e-8
            ld[i] = (
                1
                / 9
                * INV_PI
                / sqrt(k * z)
                * (
                    ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)) * ellk(q)
                    + 4 * k * z * (z2 + 7 * k2 - 4) * ellec(q)
                    - 3 * x3 / x1 * ellpicb((1 / x1 - 1), q)
                )
            )
            if z < k:
                ld[i] = ld[i] + 2 / 3
            ed[i] = (
                1
                / 2
                * INV_PI
                * (
                    kap1
                    + k2 * (k2 + 2 * z2) * kap0
                    - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0))
                )
            )

        # Occulting body transits the source:
        if k <= 1 and z < (1 - k):
            q = sqrt((x2 - x1) / (1 - x1))  # re-calc because different condition
            if q >= 1:
                q = 1 - 1e-8  # float rounding guard; ld is overwritten by the special case below
            ld[i] = (
                2
                / 9
                * INV_PI
                / sqrt(1 - x1)
                * (
                    (1 - 5 * z2 + k2 + x3 * x3) * ellk(q)
                    + (1 - x1) * (z2 + 7 * k2 - 4) * ellec(q)
                    - 3 * x3 / (x1) * ellpicb((x2 / (x1) - 1), q)
                )
            )
            if z < k:
                ld[i] = ld[i] + 2 / 3
            if abs(k + z - 1) < 1e-8:
                ld[i] = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            ed[i] = k2 / 2 * (k2 + 2 * z2)
        flux[i] = 1 - (c1 * le[i] + c2 * ld[i] + u2 * ed[i]) * omega

        # Some combinations of (u1,u2) cause flux>1 which is unphysical
        if flux[i] > 1:
                flux[i] = 1
    return flux


# Fixed iteration count of the vectorised Bulirsch loop (P3b). 5 steps pass
# ell_combo's 1e-8 test for kc >= ~0.45 (q <= ~0.9); lanes with smaller kc
# (<= 0.5 % of the on-star samples for any k, mostly near second contact)
# fail the test after the last step and are recomputed with the scalar
# ell_combo_kc, so the result is always the converged one.
ELL_BATCH_ITERATIONS = 5

# Floor of the complementary modulus kc (P15). kc = 0 only at z = 1 - k
# exactly (or below it by rounding), where K and Pi diverge
# logarithmically but A K + B E + C Pi stays finite; with kc >= 1e-150 the
# result is off by O(kc^2 log kc), i.e. not at all, and the AGM converges
# in ~15 steps (scalar fallback).
KC_MIN = 1e-150

# Smallest |z - k| passed to the batch (occult_hybrid, pass 1), see there.
Z_K_MIN = 1e-150


# Flux accuracy target of occult_hybrid's interpolants (limb band P21b, interior P21c): their measured errors are
# <= SERIES_TOL = 1e-8 (0.01 ppm) of the exact kernel; > 0 switches them on (0: exact kernel everywhere, tests).
# (The rigorous P14 interior series that this name comes from was replaced by the P21c interpolant and removed
# in the 2026-10-13 cleanup; see git history.)
SERIES_TOL = 1e-8

# P23: block length of the per-sample path's pass 1 (vectorised counts and
# interior approximation per block; blocks without exact samples are not scanned again)
BLOCK_N = 32


# ---------------------------------------------------------------------------
# P21b: per-call polynomial interpolant of the limb band 1 - k <= z < 1 + k
#
# On the limb the exact kernel costs ~10.5 ns per lane (elliptic integrals,
# two arccos, ~12 divisions). For fixed (k, u1, u2) the deficit
# D(z) = (c1 lex + c2 ldx + u2 edx) omega is a smooth function of z on the
# band except at its ends: near first contact (h = 1 + k - z -> 0) it is a
# series in sqrt(h) starting at h^(3/2), near second contact
# (g = z - (1 - k) -> 0) D_t + a g + b g^(3/2) + ... plus g^2 ln g terms
# from the elliptic integrals. So the band is split at z = 1 and each half
# is interpolated in its own variable, u = sqrt(h / k) (outer half, z >= 1)
# or u = sqrt(g / k) (inner half), both in [0, 1], by a polynomial of
# degree LIMB_N in x = 2 u - 1. In u the outer half is analytic and the
# inner half has a u^4 ln u endpoint singularity (error ~N^-8). The
# conditions are the values at the Lobatto nodes x_j = cos(pi j / N),
# except that the nodes next to x = -1 are replaced by conditions known
# for free: outer D = dD/du = d2D/du2 = 0 at u = 0 (D ~ u^3), so nodes
# N-1, N-2 are dropped; inner D = D_t (second contact, closed form) and
# dD/du = 0 at u = 0, so node N-1 (kc ~ 0.012: a scalar fallback lane) is
# dropped. That leaves LIMB_NODES = 20 exact lanes per call (z = 1 is
# shared; one of them, inner node N-2, needs the scalar fallback); a fixed
# matrix per half turns the values into monomial coefficients (Horner per
# lane, ~2 ns, vectorised).
# Measured (scratch_p21b/acc3.py, mpmath reference for lex / edx, the
# converged kernel for ldx; dense z incl. 1e-14 from both contacts): max
# component error 2.2e-10 (k = 0.1), 1.2e-9 (k = 0.2), 2.5e-9 (k = 0.25),
# i.e. <= ~7e-9 in flux for any physical LD (test_limb_interpolant: <= 1e-8
# vs mpmath quadrature and occult_exact); near the contacts the
# interpolant is closer to mpmath than the double closed forms of lex (up
# to 7e-9 off within 1e-14 of first contact). Used by occult_hybrid (not
# occult_exact) for k <= LIMB_K_MAX when at least LIMB_MIN_LANES samples of
# a call lie on the band. Since P21d the coefficients come from an offline
# table in sqrt(k) (~65 ns instead of the ~0.4 us per-call build), so the
# break-even is ~10 lanes: LIMB_MIN_LANES 64 -> 16, and the pre-check
# len(zs) k >= 32 (LIMB_FILL) is gone (Earth-sized moons and Kepler 30-min
# cadence use the interpolant too).
# ---------------------------------------------------------------------------
LIMB_N = 12             # degree per half (the kernels below are unrolled for 12)
LIMB_K_MAX = 0.25       # validated range of k (error grows ~k^2)
LIMB_MIN_LANES = 16     # fewer band samples per call: exact kernel (P21d: was 64)
LIMB_W = 16             # coefficient slots per half in _limb_coeffs' output (>= LIMB_N + 1)
LIMB_NO = LIMB_N - 2    # outer nodes x_0 .. x_{N-3}
LIMB_NI = LIMB_N - 1    # inner nodes x_0 .. x_{N-2}
LIMB_NODES = LIMB_NO + LIMB_NI - 1   # exact lanes (z = 1 shared): 20, a multiple of 4
                                     # (a scalar remainder lane of the batch costs ~25 ns)


def _limb_tables(N):
    """(u_j^2 at the Lobatto nodes x_j = cos(pi j / N), u = (x + 1) / 2;
    transposed matrices MO, MI: MO[j, i] = monomial coefficient i (in x) per
    unit value at outer node j (j < LIMB_NO; D, D', D'' = 0 at x = -1),
    MI[j, i] likewise for inner node j (j < LIMB_NI) and, at j = LIMB_NI,
    per unit second-contact value D_t (D' = 0 at x = -1))"""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    u = (x + 1) / 2

    def deriv_row(d):        # d-th derivative of x^i at x = -1
        r = np.zeros(N + 1)
        for i in range(d, N + 1):
            c = 1.0
            for q in range(d):
                c *= i - q
            r[i] = c * (-1.0) ** (i - d)
        return r

    V = np.vander(x, N + 1, increasing=True)
    no, ni = N - 2, N - 1
    A_o = np.vstack([V[:no], V[N], deriv_row(1), deriv_row(2)])
    A_i = np.vstack([V[:ni], V[N], deriv_row(1)])
    MO = np.linalg.inv(A_o)[:, :no]          # right-hand side zero in the other rows
    MI = np.linalg.inv(A_i)[:, :ni + 1]
    return u * u, np.ascontiguousarray(MO.T), np.ascontiguousarray(MI.T)


LIMB_U2, LIMB_MO, LIMB_MI = _limb_tables(LIMB_N)
assert LIMB_N == 12 and LIMB_NODES % 4 == 0  # _occult_limb_batch is unrolled for 12


# P21d: offline table of the limb interpolant. Its coefficients are linear in
# the node values, i.e. coef = omega (c1 C_lex(k) + c2 C_ldx(k) + u2 C_edx(k))
# with LD-independent component vectors C_*(k) (2 LIMB_W slots each). Each is
# stored as a Chebyshev series of degree LIMB_TAB_M in
# t = 2 (sqrt(k) - sqrt(k_min)) / (sqrt(k_max) - sqrt(k_min)) - 1, fitted at the
# first-kind Chebyshev nodes in t to per-call builds (_limb_coeffs_exact) and
# stored in limb_table.csv (_limb_table_write(); test_limb_interpolant
# rebuilds it). Reconstruction vs the per-call build (scratch_p21d/proto.py,
# 200 random k, 4 LD laws incl. unphysical ones): max |delta D| on the band
# 6.3e-12 (M = 10: 1.5e-10; in k instead of sqrt(k) the ldx part needs
# M > 24). Per call ~65 ns instead of ~410 ns (20 exact lanes, one scalar
# fallback, matvec). Outside [k_min, k_max] the per-call build is used.
LIMB_TAB_KMIN = 1e-4
LIMB_TAB_KMAX = LIMB_K_MAX
LIMB_TAB_M = 12
LIMB_TAB_Y0 = float(np.sqrt(LIMB_TAB_KMIN))
LIMB_TAB_SCALE = float(2 / (np.sqrt(LIMB_TAB_KMAX) - np.sqrt(LIMB_TAB_KMIN)))


def _limb_table_build(M=LIMB_TAB_M, kmin=LIMB_TAB_KMIN, kmax=LIMB_TAB_KMAX):
    """(3 (M + 1), 2 LIMB_W) table: rows c (M + 1) + m hold the Chebyshev
    coefficient m (in t) of the component vector C_c(k), c = 0: lex
    (u = (0, 0)), 1: ldx (u = (1, 0), omega = 1.5), 2: edx (u = (-2, 1):
    omega (lex + edx) = (lex + edx) / 1.5). Run
    `PANDORA_BUILDING_TABLES=1 python -c "import pandoramoon.occult as o; o._limb_table_write()"` to
    regenerate limb_table.csv (only needed if LIMB_N, the nodes, the range or
    M change), then clear the numba cache (__pycache__/*.nb*)."""
    from numpy.polynomial import chebyshev
    y0, y1 = np.sqrt(kmin), np.sqrt(kmax)
    t = np.cos(np.pi * (np.arange(M + 1) + 0.5) / (M + 1))
    ks = (y0 + (t + 1) / 2 * (y1 - y0)) ** 2
    C = np.empty((3, M + 1, 2 * LIMB_W))
    for i, k in enumerate(ks):
        lex = _limb_coeffs_exact(float(k), 0.0, 0.0)
        C[0, i] = lex
        C[1, i] = _limb_coeffs_exact(float(k), 1.0, 0.0) / 1.5
        C[2, i] = _limb_coeffs_exact(float(k), -2.0, 1.0) * 1.5 - lex
    return np.concatenate([chebyshev.chebfit(t, C[c], M) for c in range(3)])


def _limb_table_write():
    _table_save("limb_table.csv", _limb_table_build(),
                "P21d: offline table of the limb interpolant (see occult.py, LIMB_TAB_*).\n"
                "Generated by occult._limb_table_write(); do not edit.\n"
                f"LIMB_TAB_M = {LIMB_TAB_M}, k in [{LIMB_TAB_KMIN}, {LIMB_TAB_KMAX}], LIMB_N = {LIMB_N}; "
                f"rows c (M + 1) + m, c = 0: lex, 1: ldx, 2: edx; columns: 2 LIMB_W = {2 * LIMB_W} slots")


LIMB_TAB = _table_load("limb_table.csv", (3 * (LIMB_TAB_M + 1), 2 * LIMB_W))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _limb_coeffs(k, u1, u2):
    """Monomial coefficients in x (outer half [0:N+1], inner half
    [LIMB_W:LIMB_W+N+1]) of the P21b limb interpolant of the deficit
    1 - flux for (k, u1, u2): from the offline table (P21d) for
    LIMB_TAB_KMIN <= k <= LIMB_TAB_KMAX, else built per call."""
    if k >= LIMB_TAB_KMIN and k <= LIMB_TAB_KMAX:
        return _limb_coeffs_tab(k, u1, u2)
    return _limb_coeffs_exact(k, u1, u2)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _limb_coeffs_tab(k, u1, u2):
    """_limb_coeffs() from the offline table: coef_j = sum_m T_m(t)
    (w0 LIMB_TAB[m, j] + w1 LIMB_TAB[M+1+m, j] + w2 LIMB_TAB[2(M+1)+m, j]),
    (w0, w1, w2) = omega (c1, c2, u2), t linear in sqrt(k)."""
    M1 = LIMB_TAB_M + 1
    W2 = 2 * LIMB_W
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    w0 = (1 - u1 - 2 * u2) * omega
    w1 = (u1 + 2 * u2) * omega
    w2 = u2 * omega
    t = (np.sqrt(k) - LIMB_TAB_Y0) * LIMB_TAB_SCALE - 1
    t2 = 2 * t
    coef = np.empty(W2)
    for j in range(W2):
        coef[j] = (w0 * (LIMB_TAB[0, j] + t * LIMB_TAB[1, j])
                   + w1 * (LIMB_TAB[M1, j] + t * LIMB_TAB[M1 + 1, j])
                   + w2 * (LIMB_TAB[2 * M1, j] + t * LIMB_TAB[2 * M1 + 1, j]))
    tm1 = 1.0                              # T_0
    tm = t                                 # T_1
    for m in range(2, M1):
        tn = t2 * tm - tm1
        tm1 = tm
        tm = tn
        a = w0 * tn
        b = w1 * tn
        c = w2 * tn
        for j in range(W2):
            coef[j] += a * LIMB_TAB[m, j] + b * LIMB_TAB[M1 + m, j] + c * LIMB_TAB[2 * M1 + m, j]
    return coef


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _limb_coeffs_exact(k, u1, u2):
    """_limb_coeffs() built per call: exact kernel at the nodes
    (non-converged lanes scalar), closed form at second contact (~0.4 us;
    also the source of the offline table)."""
    N = LIMB_N
    W = LIMB_W
    NO = LIMB_NO
    NI = LIMB_NI
    nn = LIMB_NODES
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    k2 = k * k
    # lanes: z = 1 (x_0 of both halves), outer x_1..x_{NO-1}, inner x_1..x_{NI-1}
    buf = np.empty(3 * nn)
    zn = buf[:nn]
    fn = buf[nn:2 * nn]
    dv = buf[2 * nn:]
    zn[0] = 1.0
    for j in range(1, NO):
        zn[j] = 1 + k - k * LIMB_U2[j]
    for j in range(1, NI):
        zn[NO - 1 + j] = 1 - k + k * LIMB_U2[j]
    _occult_exact_batch(zn, nn, k, c1, c2, u2, omega, fn, dv)
    for j in range(nn):
        if not dv[j] <= 0:
            fn[j] = _occult_point(zn[j], k, u1, u2)
    # second contact z = 1 - k in closed form (Case IV with z + k = 1)
    ldx_t = (2 / 3 / pi) * np.arccos(1 - 2 * k) - (4 / 9 / pi) * np.sqrt(k * (1 - k)) * (3 + 2 * k - 8 * k2)
    edx_t = k2 / 2 * (k2 + 2 * (1 - k) * (1 - k))
    d_t = (c1 * k2 + c2 * ldx_t + u2 * edx_t) * omega
    # coefficients = M @ conditions, as reductions per coefficient (44 ns;
    # row updates of both halves: 130 ns)
    coef = np.zeros(2 * W)
    for i in range(N + 1):
        d0 = 1 - fn[0]
        so = LIMB_MO[0, i] * d0
        si = LIMB_MI[0, i] * d0 + LIMB_MI[NI, i] * d_t
        for j in range(1, NO):
            so += LIMB_MO[j, i] * (1 - fn[j])
        for j in range(1, NI):
            si += LIMB_MI[j, i] * (1 - fn[NO - 1 + j])
        coef[i] = so
        coef[W + i] = si
    return coef


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_limb_batch(zl2, m, k, coef, flux):
    """P21b limb interpolant for the squared distances zl2[:m] (all
    1 - k <= z < 1 + k; P26: z = sqrt(z^2) here, vectorised): flux =
    1 - D(z), clamped at 1; coef from _limb_coeffs(). Unrolled Horner with
    the half selected per lane, so that LLVM vectorises the loop."""
    inv_k = 1 / k
    o0 = coef[0]; o1 = coef[1]; o2 = coef[2]; o3 = coef[3]; o4 = coef[4]
    o5 = coef[5]; o6 = coef[6]; o7 = coef[7]; o8 = coef[8]; o9 = coef[9]
    o10 = coef[10]; o11 = coef[11]; o12 = coef[12]
    i0 = coef[16]; i1 = coef[17]; i2 = coef[18]; i3 = coef[19]; i4 = coef[20]
    i5 = coef[21]; i6 = coef[22]; i7 = coef[23]; i8 = coef[24]; i9 = coef[25]
    i10 = coef[26]; i11 = coef[27]; i12 = coef[28]
    for j in range(m):
        t = np.sqrt(zl2[j]) - 1
        out = t >= 0
        r = ((k - t) if out else (k + t)) * inv_k
        x = 2 * np.sqrt(max(r, 0.0)) - 1
        P = o12 if out else i12
        P = P * x + (o11 if out else i11)
        P = P * x + (o10 if out else i10)
        P = P * x + (o9 if out else i9)
        P = P * x + (o8 if out else i8)
        P = P * x + (o7 if out else i7)
        P = P * x + (o6 if out else i6)
        P = P * x + (o5 if out else i5)
        P = P * x + (o4 if out else i4)
        P = P * x + (o3 if out else i3)
        P = P * x + (o2 if out else i2)
        P = P * x + (o1 if out else i1)
        P = P * x + (o0 if out else i0)
        f = 1 - P
        flux[j] = f if f < 1 else 1.0


# ---------------------------------------------------------------------------
# P21c: per-call polynomial interpolant of the interior 0 <= z <= 1 - k
#
# For fixed (k, u1, u2) the deficit D(z) = (c1 k^2 + c2 ldx + u2 edx) omega of a
# body fully on the star is analytic in z^2 on [0, (1 - k)^2) (edx is a
# polynomial in z^2, ldx = k^2 <mu>); its only singularity is at internal
# tangency, D = D_t + a delta + b delta^2 ln delta + ... (delta = 1 - k - z).
# In s = sqrt((1 - k)^2 - z^2) (s^2 ~ 2 (1 - k) delta, analytic in z^2) that is
# D_t + a' s^2 + b' s^4 ln s + ... (only even powers plus s^(2n) ln s terms),
# so a polynomial in x = 2 s / (1 - k) - 1 converges like N^-8 (transit
# project, PERFORMANCE_LOG.md §28; Option A of its ASSESSMENT.md). Conditions:
# the values at the Lobatto nodes x_j = cos(pi j / N), with the end values in
# closed form (x_0 = 1: z = 0, D_0 from <mu> = 2 (1 - (1 - k^2)^(3/2)) / (3 k^2);
# x_N = -1: z = 1 - k, D_t as in _limb_coeffs) and node N-1 (next to contact,
# a scalar-fallback lane) replaced by dD/dx = 0 at x = -1: INT_NODES = N - 2
# = 12 exact lanes (one of them, x_{N-2}, takes the scalar fallback for
# k >~ 0.05). lex = k^2 and edx are polynomials in x, so only the ldx part
# depends on the nodes: for 0.0005 <= k <= 0.5 its coefficients come from
# an offline table in sqrt(k) (INT_TAB below, ~0.05 us per call), else
# from the 12 exact lanes per call (~0.3 us). Replaces the P14 series
# (z < z_c) *and* the exact band z_c <= z < 1 - k: one sqrt and a 15-term
# Horner per interior sample, no division, no z_c search, no z == k
# nudge. Measured (scratch_p21c/acc.py, interior_err.py; converged kernel,
# which agrees with an mpmath rim quadrature to ~1e-14; dense z incl.
# 1e-15 from contact and z = k): max ldx error 1.3e-11 (k = 1e-4), 5.5e-10
# (0.0092), 1.3e-9 (0.1), 1.5e-9 (0.2 ... 0.3), 1.2e-9 (0.5), 6.5e-10
# (0.7), so <= 3.7e-9 in flux for the LD laws tested, unphysical ones
# included (c2 omega <= 2.4 for physical LD); the error is measured, not
# proven (P14's series had a rigorous bound at 1e-8; it is kept below as
# the reference of the unit tests).
# ---------------------------------------------------------------------------
INT_N = 14              # degree (the kernels below are unrolled for 14)
INT_NODES = INT_N - 2   # exact lanes x_1 .. x_{N-2}: 12, a multiple of 4
INT_W = 16              # coefficient slots (>= INT_N + 1)


def _int_tables(N):
    """(z_j / (1 - k) at the exact nodes x_1 .. x_{N-2}; transposed matrix M:
    M[j, i] = monomial coefficient i (in x) per unit value of condition j,
    j = 0: x_0 (z = 0), j = 1 .. N-2: x_j, j = N-1: x_N (z = 1 - k); the
    dD/dx(-1) = 0 condition has right-hand side 0 and no column)"""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    sf = (x + 1) / 2                                  # s / (1 - k)
    d1 = np.zeros(N + 1)                              # d/dx x^i at x = -1
    for i in range(1, N + 1):
        d1[i] = i * (-1.0) ** (i - 1)
    V = np.vander(x, N + 1, increasing=True)
    A = np.vstack([V[:N - 1], V[N], d1])
    M = np.linalg.inv(A)[:, :N]
    zf = np.sqrt(1 - sf[1:N - 1] ** 2)
    return np.ascontiguousarray(zf), np.ascontiguousarray(M.T)


INT_ZF, INT_M = _int_tables(INT_N)
assert INT_N == 14 and INT_NODES % 4 == 0 and INT_M.shape == (INT_N, INT_N + 1)

# Offline table of the ldx part (LD-independent) of the interior
# interpolant: its monomial coefficients c_j(k) as Chebyshev series of
# degree INT_TAB_M in t = 2 (sqrt(k) - sqrt(k_min)) / (sqrt(k_max) - sqrt(k_min)) - 1
# (sqrt(k): the coefficients have a sqrt(k) behaviour at small k). Built by
# _int_table_build() from _int_coeffs_exact (interpolation at the
# first-kind Chebyshev nodes in t) and stored in interior_table.csv;
# test_interior_interpolant rebuilds it. Reconstruction vs the per-call
# build at 400 random k (scratch_p21c/gen_table.py): coefficient error
# <= 1.3e-11, sum of |errors| <= 6.3e-11 (a bound of the ldx change on the
# whole interior). The per-call build then costs ~0.05 instead of ~0.3 us
# (12 exact lanes). Outside [k_min, k_max] the per-call build is used.
INT_TAB_KMIN = 0.0005
INT_TAB_KMAX = 0.5
INT_TAB_M = 32
INT_TAB_Y0 = float(np.sqrt(INT_TAB_KMIN))
INT_TAB_SCALE = float(2 / (np.sqrt(INT_TAB_KMAX) - np.sqrt(INT_TAB_KMIN)))


def _int_table_build(M=INT_TAB_M, kmin=INT_TAB_KMIN, kmax=INT_TAB_KMAX):
    """(M + 1, INT_W) table of the Chebyshev coefficients (in t) of the ldx
    monomial coefficients c_j(k) (u = (1, 0): deficit = 1.5 ldx), from
    per-call builds at the M + 1 first-kind Chebyshev nodes in t. Run
    `PANDORA_BUILDING_TABLES=1 python -c "import pandoramoon.occult as o; o._int_table_write()"` to
    regenerate interior_table.csv (only needed if INT_N, the nodes, the
    range or M change), then clear the numba cache (__pycache__/*.nb*): the
    cached functions do not notice a changed table module."""
    from numpy.polynomial import chebyshev
    y0, y1 = np.sqrt(kmin), np.sqrt(kmax)
    t = np.cos(np.pi * (np.arange(M + 1) + 0.5) / (M + 1))
    ks = (y0 + (t + 1) / 2 * (y1 - y0)) ** 2
    C = np.array([_int_coeffs_exact(float(k), 1.0, 0.0) / 1.5 for k in ks])   # (M + 1, INT_W)
    T = chebyshev.chebfit(t, C, M)
    T[:, INT_N + 1:] = 0.0
    return T


def _int_table_write():
    _table_save("interior_table.csv", _int_table_build(),
                "P21c: offline table of the interior interpolant (see occult.py, INT_TAB_*).\n"
                "Generated by occult._int_table_write(); do not edit.\n"
                f"INT_TAB_M = {INT_TAB_M}, k in [{INT_TAB_KMIN}, {INT_TAB_KMAX}], INT_N = {INT_N}; "
                f"rows m (Chebyshev term), columns: INT_W = {INT_W} slots")


INT_TAB = _table_load("interior_table.csv", (INT_TAB_M + 1, INT_W))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _int_coeffs(k, u1, u2):
    """Monomial coefficients in x (coef[0:INT_N+1]) of the P21c interior
    interpolant of the deficit 1 - flux for (k, u1, u2), 0 < k < 1: from
    the offline table for INT_TAB_KMIN <= k <= INT_TAB_KMAX, else built
    from the exact kernel (_int_coeffs_exact)."""
    if k >= INT_TAB_KMIN and k <= INT_TAB_KMAX:
        return _int_coeffs_tab(k, u1, u2)
    return _int_coeffs_exact(k, u1, u2)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _int_coeffs_tab(k, u1, u2):
    """_int_coeffs() from the offline table (INT_TAB_KMIN <= k <= INT_TAB_KMAX):
    the ldx coefficients c_j(k) = sum_m INT_TAB[m, j] T_m(t), t linear in
    sqrt(k), plus the exact polynomials in x of lex = k^2 and
    edx = k^2 / 2 (k^2 + 2 z^2), z^2 = (1 - k)^2 (3 - 2 x - x^2) / 4."""
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    k2 = k * k
    t = (np.sqrt(k) - INT_TAB_Y0) * INT_TAB_SCALE - 1
    t2 = 2 * t
    coef = np.zeros(INT_W)
    tm1 = 1.0                              # T_0
    tm = t                                 # T_1
    for j in range(INT_W):
        coef[j] = INT_TAB[0, j] + INT_TAB[1, j] * t
    for m in range(2, INT_TAB_M + 1):
        tn = t2 * tm - tm1
        tm1 = tm
        tm = tn
        for j in range(INT_W):
            coef[j] += INT_TAB[m, j] * tn
    # deficit = omega (c1 k^2 + c2 ldx + u2 edx)
    q = k2 * (1 - k) * (1 - k)
    for j in range(INT_W):
        coef[j] *= c2 * omega
    coef[0] += omega * (c1 * k2 + u2 * (0.5 * k2 * k2 + 0.75 * q))
    coef[1] -= omega * u2 * 0.5 * q
    coef[2] -= omega * u2 * 0.25 * q
    return coef


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _int_coeffs_exact(k, u1, u2):
    """_int_coeffs() built per call: exact kernel at INT_NODES nodes
    (non-converged lanes scalar), closed forms at z = 0 and z = 1 - k
    (~0.3 us; also the source of the offline table)."""
    N = INT_N
    nn = INT_NODES
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    k2 = k * k
    omk = 1 - k
    buf = np.empty(3 * nn)
    zn = buf[:nn]
    fn = buf[nn:2 * nn]
    dv = buf[2 * nn:]
    for j in range(nn):
        z = omk * INT_ZF[j]
        if abs(z - k) < Z_K_MIN:           # as occult_hybrid's pass 2
            z = k + max(k * 2.220446049250313e-16, Z_K_MIN)
        zn[j] = z
    # The trip count must not be a compile-time constant: with m = nn
    # LLVM compiles the inlined batch loop for exactly 12 lanes without
    # vectorising it (466 instead of 131 ns, scratch_p21c/batch_probe.py).
    # k > 0 always holds here, but LLVM cannot know that.
    m = nn if k > 0 else 0
    _occult_exact_batch(zn, m, k, c1, c2, u2, omega, fn, dv)
    for j in range(nn):
        if not dv[j] <= 0:
            fn[j] = _occult_point(zn[j], k, u1, u2)
    # z = 0: ldx = k^2 <mu> = 2/3 (1 - (1 - k^2)^(3/2)), edx = k^4 / 2
    ldx_0 = -(2 / 3) * np.expm1(1.5 * np.log1p(-k2))
    d_0 = (c1 * k2 + c2 * ldx_0 + u2 * (0.5 * k2 * k2)) * omega
    # z = 1 - k: closed form (Case IV with z + k = 1), as in _limb_coeffs
    ldx_t = (2 / 3 / pi) * np.arccos(1 - 2 * k) - (4 / 9 / pi) * np.sqrt(k * omk) * (3 + 2 * k - 8 * k2)
    edx_t = k2 / 2 * (k2 + 2 * omk * omk)
    d_t = (c1 * k2 + c2 * ldx_t + u2 * edx_t) * omega
    coef = np.zeros(INT_W)
    for i in range(N + 1):
        s = INT_M[0, i] * d_0 + INT_M[N - 1, i] * d_t
        for j in range(nn):
            s += INT_M[j + 1, i] * (1 - fn[j])
        coef[i] = s
    return coef


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_int_batch(z2s, lo, hi, coef, omk, flux):
    """P21c interior interpolant for the squared distances z2s[lo:hi] (all
    0 <= z < 1 - k = omk; P26: s^2 = omk^2 - z^2 needs no z): flux =
    1 - D(z), clamped at 1; coef from _int_coeffs(). Unrolled Horner
    (vectorised); the arrays are sliced here, not by the caller: views
    passed into a jitted callee made P23's per-block kernel ~1.5x slower
    (PERFORMANCE_LOG.md §32.2)."""
    zr = z2s[lo:hi]
    fr = flux[lo:hi]
    inv_h = 2 / omk
    omk2 = omk * omk
    c0 = coef[0]; c1 = coef[1]; c2 = coef[2]; c3 = coef[3]; c4 = coef[4]
    c5 = coef[5]; c6 = coef[6]; c7 = coef[7]; c8 = coef[8]; c9 = coef[9]
    c10 = coef[10]; c11 = coef[11]; c12 = coef[12]; c13 = coef[13]; c14 = coef[14]
    for j in range(hi - lo):
        x = np.sqrt(max(omk2 - zr[j], 0.0)) * inv_h - 1
        P = c14
        P = P * x + c13
        P = P * x + c12
        P = P * x + c11
        P = P * x + c10
        P = P * x + c9
        P = P * x + c8
        P = P * x + c7
        P = P * x + c6
        P = P * x + c5
        P = P * x + c4
        P = P * x + c3
        P = P * x + c2
        P = P * x + c1
        P = P * x + c0
        f = 1 - P
        fr[j] = f if f < 1 else 1.0


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_int_select(z2s, lo, hi, coef, omk, one_k, flux):
    """P21c in the per-sample path (replaces P23's _occult_series_select),
    on squared distances z2s (P26): flux[i] = 1 off the star (z >= 1 + k or
    z^2 < 0), else the interior interpolant at min(z, 1 - k) (s clamped at
    0; samples with z >= 1 - k are overwritten by the limb pass), for
    lo <= i < hi, branch-free. Returns the number of limb samples
    (1 - k <= z < 1 + k)."""
    zr = z2s[lo:hi]
    fr = flux[lo:hi]
    inv_h = 2 / omk
    omk2 = omk * omk
    one_k2 = one_k * one_k
    c0 = coef[0]; c1 = coef[1]; c2 = coef[2]; c3 = coef[3]; c4 = coef[4]
    c5 = coef[5]; c6 = coef[6]; c7 = coef[7]; c8 = coef[8]; c9 = coef[9]
    c10 = coef[10]; c11 = coef[11]; c12 = coef[12]; c13 = coef[13]; c14 = coef[14]
    c = 0
    for j in range(hi - lo):
        z2 = zr[j]
        x = np.sqrt(max(omk2 - z2, 0.0)) * inv_h - 1
        P = c14
        P = P * x + c13
        P = P * x + c12
        P = P * x + c11
        P = P * x + c10
        P = P * x + c9
        P = P * x + c8
        P = P * x + c7
        P = P * x + c6
        P = P * x + c5
        P = P * x + c4
        P = P * x + c3
        P = P * x + c2
        P = P * x + c1
        P = P * x + c0
        f = 1 - P
        f = f if f < 1 else 1.0
        off = (z2 >= one_k2) | (z2 < 0)
        fr[j] = 1.0 if off else f
        c += 1 if (z2 >= omk2) & (z2 < one_k2) else 0
    return c


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _use_interior(k, series_tol):
    """True if occult_hybrid uses the P21c interior interpolant: with the
    approximations on (series_tol > 0; occult_exact passes 0) and an
    interior to interpolate (0 < k < 1)"""
    return series_tol > 0 and k > 0 and k < 1


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _occult_exact_batch(zb, m, k, c1, c2, u2, omega, flux, dev):
    """Exact Mandel-Agol flux for the m on-star samples zb[:m] (P3b);
    branch-free, so LLVM vectorises it.

    Per sample: Case IV (z < 1 - k) or Case III (else; the limb), the
    coefficients A, B, C of A K + B E + C Pi selected per case, then the
    Bulirsch iteration of ell_combo with a fixed ELL_BATCH_ITERATIONS steps
    and no early exit (iterating past convergence is harmless: the AGM
    state only scales by 2 per step). flux[j] holds
    1 - (c1 lex + c2 ldx + u2 edx) omega (unclamped); dev[j] > 0 marks a
    lane whose last step fails ell_combo's 1e-8 test (recompute it with the
    scalar code).

    Since P16 the limb's area terms lex, edx (two arccos, Case III) are
    computed here for every lane with the branch-free trig.acos_poly and
    selected for limb lanes, instead of a scalar libm loop in the caller
    (16.4 ns per limb sample; acos_poly is <= 1 ulp from libm). Their
    divisors are floored at 1e-300 so that interior lanes (z = 0 or k = 0
    possible) stay finite under fastmath.

    Operands in closed form (P15; Agol, Luger & Foreman-Mackey 2020, eq. 41):
    with s = z + k, d = z - k, sg = sign(d)
      Case IV: kc^2 = (1 - s)(1 + s) / (1 - d^2), p = sqrt(n + 1) = s / |d|
      Case III: kc^2 = (s - 1)(s + 1) / (4 z k),  p = 1 / |d|
      C = -3 x3 / x1 = 3 s / d;  C / p = 3 sg (IV) or 3 s sg (III);
      prefactor 2 / (9 pi sqrt(den)) in both cases (den = 1 - d^2 or 4 z k).
    No modulus q, no 1 - q^2 cancellation, no q -> 1 - 1e-8 nudges (which
    were up to 2.6e-8 off in ldx at z -> 1 - k), 3 divisions and 2 square
    roots fewer per sample, and exact down to |z - k| = 1 ulp (the old
    z -> k + 1e-6 nudge is gone; only z == k exactly is moved by one ulp in
    occult_hybrid).

    error_model="numpy" is required for vectorisation: numba's default
    ("python") inserts a zero-divisor check before every division, and that
    branch stops LLVM. All divisors are > 0 here (z != k, den > 0 on the
    star, m0 >= 1, p > 0)."""
    INV_PI = 1 / pi
    k2 = k * k
    for j in range(m):
        z = zb[j]
        z2 = z * z
        s = z + k
        d = z - k
        x1 = d * d
        x2 = s * s
        x3 = k2 - z2
        inner = z < 1 - k
        # Case IV (inner) / Case III (limb): kc^2 = num / den
        num = (1 - s) * (1 + s) if inner else (s - 1) * (s + 1)
        den = 1 - x1 if inner else 4 * z * k
        inv_sden = 1 / sqrt(den)
        kc = max(sqrt(max(num, 0.0)) * inv_sden, KC_MIN)
        sg = 1.0 if d > 0 else -1.0
        inv_ad = 1 / abs(d)
        p = s * inv_ad if inner else inv_ad
        a2 = 3 * s * sg * inv_ad
        b2 = 3 * sg if inner else 3 * s * sg
        A = (1 - 5 * z2 + k2 + x3 * x3) if inner else ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2))
        B = den * (z2 + 7 * k2 - 4)
        # A K + B E + C Pi: ell_combo's iteration, one reciprocal
        # r = 1 / (m0 p) per step instead of two divisions
        a1 = A + B
        b1 = A + B * kc * kc
        e = kc
        m0 = 1.0
        dd = 0.0
        for it in range(ELL_BATCH_ITERATIONS):
            r = 1 / (m0 * p)
            f1 = a1
            a1 = a1 + b1 * (p * r)
            b1 = 2 * (b1 + f1 * kc)
            inv_p = m0 * r
            f2 = a2
            a2 = a2 + b2 * inv_p
            g = e * inv_p
            b2 = 2 * (b2 + f2 * g)
            p = g + p
            g = m0
            m0 = kc + m0
            dd = abs(g - kc) - 1e-8 * g
            kc = 2 * sqrt(e)
            e = kc * m0
        # (b1 + a1 m0) / (2 m0^2) + (b2 + a2 m0) / (m0 (m0 + p)), one division
        mp = m0 + p
        ell = 0.25 * pi * ((b1 + a1 * m0) * mp + 2 * m0 * (b2 + a2 * m0)) / (m0 * m0 * mp)
        ldx = (2 / 9 * INV_PI) * inv_sden * ell + (2 / 3 if z < k else 0.0)
        # Case III area terms (P16): computed for all lanes, used on the limb
        kap1 = acos_poly(max(min((1 - k2 + z2) / max(2 * z, 1e-300), 1.0), -1.0))
        kap0 = acos_poly(max(min((k2 + z2 - 1) / max(2 * k * z, 1e-300), 1.0), -1.0))
        lex_l = (k2 * kap0 + kap1 - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0.0))) * INV_PI
        edx_l = 0.5 * INV_PI * (kap1 + k2 * (k2 + 2 * z2) * kap0
                                - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0.0)))
        lex = k2 if inner else lex_l
        edx = k2 / 2 * (k2 + 2 * z2) if inner else edx_l
        flux[j] = 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega
        dev[j] = dd


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_point(z, k, u1, u2):
    """Unclamped exact Mandel-Agol flux at one on-star z (z != k,
    0 <= z < 1 + k, not fully occulted), scalar ell_combo_kc. Same cases and
    closed-form operands as _occult_exact_batch (P15): Case IV for
    z < 1 - k, else Case III. Used for the batch's non-converged lanes."""
    INV_PI = 1 / pi
    k2 = k**2
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    z2 = z**2
    s = z + k
    d = z - k
    x1 = d * d
    x2 = s * s
    x3 = k**2 - z2
    sg = 1.0 if d > 0 else -1.0
    if z < 1 - k:
        # Occulting body transits the source: Table 3, Case IV
        lex = k2
        edx = k2 / 2 * (k2 + 2 * z2)
        num = (1 - s) * (1 + s)
        den = 1 - x1
        A = 1 - 5 * z2 + k2 + x3 * x3
        p = s / abs(d)
        Cp = 3 * sg
    else:
        # Partly occults the source and crosses the limb: Table 3, Case III
        kap1 = arccos(max(min((1 - k2 + z2) / (2 * z), 1), -1))
        kap0 = arccos(max(min((k2 + z2 - 1) / (2 * k * z), 1), -1))
        lex = (k2 * kap0 + kap1 - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
        edx = 1 / 2 * INV_PI * (kap1 + k2 * (k2 + 2 * z2) * kap0
                                - (1 + 5 * k2 + z2) / 4 * sqrt(max((1 - x1) * (x2 - 1), 0)))
        num = (s - 1) * (s + 1)
        den = 4 * z * k
        A = (1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)
        p = 1 / abs(d)
        Cp = 3 * s * sg
    sden = sqrt(den)
    kc = max(sqrt(max(num, 0.0)) / sden, KC_MIN)
    ldx = 2 / 9 * INV_PI / sden * ell_combo_kc(kc, p, A, den * (z2 + 7 * k2 - 4), 3 * s / d, Cp)
    if z < k:
        ldx = ldx + 2 / 3
    return 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_hybrid(zs, k, u1, u2):
    """Transit model for an array of normalized distances, for any k: the
    exact Mandel & Agol model (see _occult_hybrid), with per-call
    interpolants of the interior (P21c, z < 1 - k) and of the limb band
    (P21b) that are within SERIES_TOL = 1e-8 (0.01 ppm) of it (measured
    <= ~4e-9 for physical limb darkening). occult_hybrid_z2() takes z^2."""
    return _occult_hybrid(_squared(zs), k, u1, u2, SERIES_TOL)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_hybrid_z2(z2s, k, u1, u2):
    """occult_hybrid() for the squared distances z2s = z^2 (P26): the model
    needs z itself only on the limb band, so the orbit loops skip the sqrt
    per sample. A negative z2 counts as off the star (as z < 0)."""
    return _occult_hybrid(z2s, k, u1, u2, SERIES_TOL)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_exact(zs, k, u1, u2):
    """occult_hybrid() without the interior / limb interpolants: the exact
    model for every sample (~10.5 instead of ~0.8 ns per interior sample)."""
    return _occult_hybrid(_squared(zs), k, u1, u2, 0.0)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _squared(zs):
    """z |z| per sample: z^2 with the sign kept (z < 0 stays off the star)"""
    z2s = np.empty(len(zs))
    for j in range(len(zs)):
        z2s[j] = zs[j] * abs(zs[j])
    return z2s


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def _sqrt_lanes(zb, lo, hi, k):
    """zb[lo:hi] = sqrt(zb[lo:hi]) in place (squared distances of the exact
    lanes -> z, P26; vectorised), with z == k moved by one ulp: the
    closed-form operands (P15) are exact down to |z - k| = 1 ulp; only
    z == k itself (C = 3 (z + k) / (z - k)) moves (k + k 2^-52 is at least
    the next float above k). Z_K_MIN keeps 1 / |z - k| finite for
    k < ~1e-134, where ulps are smaller."""
    zr = zb[lo:hi]
    zk = k + max(k * 2.220446049250313e-16, Z_K_MIN)
    for q in range(hi - lo):
        z = np.sqrt(zr[q])
        zr[q] = zk if abs(z - k) < Z_K_MIN else z


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_hybrid(zs, k, u1, u2, series_tol):
    """Exact Mandel & Agol transit model (quadratic limb darkening) for an
    array of normalized distances, for any k, with the P21c interior
    interpolant for z < 1 - k and the P21b limb interpolant (series_tol =
    0: exact everywhere; before P21c the P14 series for z < z_c). See
    _occult_hybrid_samples() for the model; P12: _occult_hybrid_stretches()
    for inputs whose classes come in long stretches (e.g. the window samples
    of a time-sorted light curve, _long_stretches()), else (and for z == k
    or full occultations) the per-sample version.

    Parameters
    ----------
    zs: 1D array
        Squared normalized distances z^2 (P26; z^2 < 0: off the star)
    k: float
        Planet-star radius ratio
    u1, u2: float
        Limb darkening coefficients
    Returns
    -------
    Transit model evaluated at `z`.
    """
    # P21c interior interpolant (z < 1 - k) and P21b limb interpolant: with
    # series_tol > 0 only (occult_exact stays exact)
    interior = _use_interior(k, series_tol)
    limb = series_tol > 0 and k > 0 and k <= LIMB_K_MAX
    if len(zs) >= STRETCH_MIN_N and _long_stretches(zs, k, (1 - k) ** 2 if interior else 0.0):
        ok, flux = _occult_hybrid_stretches(zs, k, u1, u2, interior, limb)
        if ok:
            return flux
    return _occult_hybrid_samples(zs, k, u1, u2, interior, limb)


# P12: stretch scan. Samples of one class (off the star, interior series,
# exact) come in long contiguous stretches when z varies smoothly. A
# stretch is extended by blocks of STRETCH_BLOCK samples with a vectorised
# count of the class condition (0.08 ns per sample; blocks of <= 32 compile
# to a slow unrolled form), then sample by sample.
STRETCH_BLOCK = 64
STRETCH_MIN_AVG = 64    # stretch path only if stretches are this long on average
STRETCH_PROBES = 128    # _long_stretches() looks at this many samples (P23)
STRETCH_MIN_N = 2048    # shorter inputs: per-sample path. The stretch pass
                        # gains ~1 ns per sample on PLATO windows (~5400
                        # samples); on ~1200-sample windows (Kepler, 5x
                        # supersampled) it was 1-2 % slower end to end


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _long_stretches(zs, k, z2_c):
    """True if, on the squared distances zs (P26; z2_c = z_c^2), the class
    (off star / interior / exact) changes rarely enough
    that the stretches average >= STRETCH_MIN_AVG samples: cheap estimate
    from STRETCH_PROBES evenly spaced samples (P23; before: every 8th
    sample, ~0.3 us per call on PLATO windows), branch-free classes. A
    change between two probes is seen at most once, so smooth inputs are
    estimated well; on shuffled input about half the probe pairs differ,
    hence also changes <= STRETCH_PROBES / 4. A wrong guess only costs
    speed."""
    n_z = len(zs)
    n_p = min(STRETCH_PROBES, n_z)
    if n_p < 2:
        return False
    one_k2 = (1 + k) ** 2
    step = (n_z - 1) // (n_p - 1)
    changes = 0
    prev = -1
    for p in range(n_p):
        z2 = zs[p * step]
        c = (1 if (z2 >= 0) & (z2 < one_k2) else 0) * (2 if z2 >= z2_c else 1)
        changes += 1 if c != prev else 0
        prev = c
    changes -= 1                       # the first probe is not a change
    return changes * STRETCH_MIN_AVG <= n_z and changes * 4 <= n_p


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _end_off(zs, j, n_z, one_k):
    """First index >= j of a sample with 0 <= z < one_k (end of an
    off-star stretch); since P26 called with z^2 and one_k^2"""
    while j + STRETCH_BLOCK <= n_z:
        blk = zs[j:j + STRETCH_BLOCK]
        c = 0
        for q in range(STRETCH_BLOCK):
            c += 1 if (blk[q] >= 0) & (blk[q] < one_k) else 0
        if c:
            break
        j += STRETCH_BLOCK
    while j < n_z and (zs[j] >= one_k or zs[j] < 0):
        j += 1
    return j


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _end_interior(zs, j, n_z, z_c):
    """First index >= j of a sample that is not an interior sample
    (0 <= z < z_c; P21c: z_c = 1 - k, z == k needs no care there); since
    P26 called with z^2 and z_c^2"""
    while j + STRETCH_BLOCK <= n_z:
        blk = zs[j:j + STRETCH_BLOCK]
        c = 0
        for q in range(STRETCH_BLOCK):
            c += 1 if (blk[q] >= 0) & (blk[q] < z_c) else 0
        if c < STRETCH_BLOCK:
            break
        j += STRETCH_BLOCK
    while j < n_z and zs[j] < z_c and zs[j] >= 0:
        j += 1
    return j


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _end_exact(zs, j, n_z, z_c, one_k, full, km1):
    """First index >= j of a sample that is not an exact-model sample
    (z_c <= z < one_k, not fully occulted: z > km1 if full); with
    z_c = 1 - k, one_k = 1 + k: of the limb band (P21b). Since P26 called
    with z^2 and squared bounds (z == k is moved by _sqrt_lanes)."""
    while j + STRETCH_BLOCK <= n_z:
        blk = zs[j:j + STRETCH_BLOCK]
        c = 0
        for q in range(STRETCH_BLOCK):
            c += 1 if (blk[q] >= z_c) & (blk[q] < one_k) & ((not full) | (blk[q] > km1)) else 0
        if c < STRETCH_BLOCK:
            break
        j += STRETCH_BLOCK
    while j < n_z and zs[j] >= z_c and zs[j] < one_k and not (full and zs[j] <= km1):
        j += 1
    return j


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_hybrid_stretches(zs, k, u1, u2, interior, limb):
    """_occult_hybrid_samples() by stretches (P12): (ok, flux). Off-star
    stretches are filled with 1 and, with interior (P21c), interior
    stretches (z < 1 - k) evaluated in place by the interpolant (no
    gather, no index scatter); exact-model stretches are copied into one
    buffer for the vectorised batch, as before; with limb (P21b) limb-band
    stretches are evaluated in place by the interpolant if they hold
    >= LIMB_MIN_LANES samples together (else they join the batch). zs are
    squared distances (P26), compared with squared bounds; the batch takes
    z = sqrt(z^2) (_sqrt_lanes, also the z == k nudge). ok = False (the
    caller then uses the per-sample version) at a fully occulted sample
    (k >= 1). Same values as the per-sample version up to rounding (in
    place vs per block)."""
    n_z = len(zs)
    flux = np.empty(n_z)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    one_k = 1 + k
    omk = 1 - k
    one_k2 = one_k * one_k
    z2_c = omk * omk if interior else 0.0
    full = k >= 1
    km1_2 = (k - 1) ** 2 if full else 0.0
    idx = np.empty(n_z, dtype=np.int64)
    zb = np.empty(n_z + 3)                 # + 3: padding of the exact batch (P25)
    # P21b: limb stretches 1 - k <= z < 1 + k, kept as ranges [rl[2r], rl[2r+1])
    one_mk2 = omk * omk if limb else one_k2
    rl = np.empty(n_z + 2 if limb else 0, dtype=np.int64)
    n_r = 0
    n_l = 0
    m = 0
    ic = _int_coeffs(k, u1, u2) if interior else np.zeros(INT_W)
    i = 0
    while i < n_z:
        z2 = zs[i]
        j = i + 1
        if z2 >= one_k2 or z2 < 0:                 # off the star: flux 1
            j = _end_off(zs, j, n_z, one_k2)
            fr = flux[i:j]
            for q in range(j - i):
                fr[q] = 1.0
        elif z2 < z2_c:                            # interior (P21c), in place
            j = _end_interior(zs, j, n_z, z2_c)
            _occult_int_batch(zs, i, j, ic, omk, flux)
        elif full and z2 <= km1_2:
            return False, flux
        elif z2 >= one_mk2:                        # limb band (P21b): range
            j = _end_exact(zs, j, n_z, one_mk2, one_k2, full, km1_2)
            rl[2 * n_r] = i
            rl[2 * n_r + 1] = j
            n_r += 1
            n_l += j - i
        else:                                      # exact (z_c <= z < one_mk): gather
            j = _end_exact(zs, j, n_z, z2_c, one_mk2, full, km1_2)
            cnt = j - i
            zr = zs[i:j]
            zbr = zb[m:m + cnt]
            ir = idx[m:m + cnt]
            for q in range(cnt):
                zbr[q] = zr[q]
                ir[q] = i + q
            m += cnt
        i = j
    use_limb = n_l >= LIMB_MIN_LANES
    if use_limb:                                   # interpolant, in place
        coef = _limb_coeffs(k, u1, u2)
        for r in range(n_r):
            lo = rl[2 * r]
            hi = rl[2 * r + 1]
            _occult_limb_batch(zs[lo:hi], hi - lo, k, coef, flux[lo:hi])
    else:                                          # few limb samples: exact batch
        for r in range(n_r):
            lo = rl[2 * r]
            cnt = rl[2 * r + 1] - lo
            zr = zs[lo:lo + cnt]
            zbr = zb[m:m + cnt]
            ir = idx[m:m + cnt]
            for q in range(cnt):
                zbr[q] = zr[q]
                ir[q] = lo + q
            m += cnt
    if m > 0:
        # P25: pad to a multiple of 4 lanes when that replaces >= 2 scalar
        # remainder lanes (~25 ns each)
        mp = m + (4 - m % 4) % 4 if m % 4 >= 2 else m
        for q in range(m, mp):
            zb[q] = zb[m - 1]
        _sqrt_lanes(zb, 0, mp, k)                  # P26: z^2 -> z
        fb = np.empty(mp)
        dev = np.empty(mp)
        _occult_exact_batch(zb, mp, k, c1, c2, u2, omega, fb, dev)
        # non-converged lanes (scalar), clamp, scatter
        for j in range(m):
            f = fb[j]
            if not dev[j] <= 0:
                f = _occult_point(zb[j], k, u1, u2)
            if f > 1:
                f = 1
            flux[idx[j]] = f
    return True, flux


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _occult_hybrid_samples(zs, k, u1, u2, interior, limb):
    """_occult_hybrid() for inputs without long stretches (before P12 this
    was _occult_hybrid; used for windows below STRETCH_MIN_N samples, e.g.
    Kepler, and unsorted time); interior: use the P21c interior
    interpolant for 0 <= z < 1 - k (from _use_interior()); limb: use the
    P21b limb interpolant if >= LIMB_MIN_LANES samples lie on
    1 - k <= z < 1 + k (only for 0 < k <= LIMB_K_MAX). Since P23 the
    interior approximation is evaluated in place for every sample of a
    block with any on-star sample (select instead of classify / gather /
    scatter), and only the exact samples are collected, from the blocks
    that hold any. The model:

    Until P3 this switched to a "linear interpolation" correction of the
    small-body approximation for k <= 0.05 (z <= 0.65 ... 0.98); that zone
    existed only for speed and was off by up to ~1.1 ppm (bug #28). It is
    gone: the batched exact model below is cheaper than it was.

    Batched (P3b): pass 1 classifies the samples (off star, fully occulted)
    and collects the remaining on-star z; pass 2 computes their exact flux
    in the vectorised _occult_exact_batch(); pass 3 recomputes non-converged
    lanes with the scalar _occult_point() and clamps (the limb's arccos
    terms are part of the batch since P16, with the vectorisable
    trig.acos_poly). Case III is simply "not Case IV", which closes a 1-ulp
    gap at z = 1 - k where the old conditions matched neither case and
    reused the previous sample's terms (bug #29).
    Since P15 the elliptic operands come in closed form from (z, k), exact
    at the contact points: the z -> k + 1e-6 and q -> 1 - 1e-8 nudges and
    the closed-form special case for |k + z - 1| < 1e-8 (up to ~2.6e-8 off
    in the limb-darkening term) are gone.
    From P14 to P21c, interior samples with z < z_c used the P14 series
    (rigorous bound, z_c per call) and z_c <= z < 1 - k the exact kernel;
    since P21c the whole interior z < 1 - k uses the interpolant of
    _int_coeffs() / _occult_int_select() (measured <= ~4e-9 in flux, see
    the comment block above INT_N). occult_exact disables it. All other
    on-star samples take the exact path above (or the P21b interpolant).
    """
    # k is used as given: the old snap k -> 0.5 for |k - 0.5| < 1e-4 was up
    # to 150 ppm off (bug #30); the closed-form operands (P15) are exact at
    # k = 0.5 and z -> k (non-converged lanes take the scalar fallback)
    # P26: zs are squared distances z^2, compared with squared bounds; z
    # itself only for the exact lanes (_sqrt_lanes) and the limb lanes
    # (_occult_limb_batch)
    n_z = len(zs)
    flux = np.empty(n_z)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2
    one_k = 1 + k
    omk = 1 - k
    one_k2 = one_k * one_k
    z2_c = omk * omk if interior else 0.0  # exact model from z_c on

    # Pass 1 (P23), per block of BLOCK_N samples: a vectorised count of the
    # on-star samples; blocks without any are filled with 1, the others get
    # the interior interpolant (or 1 off the star) for every sample,
    # branch-free, in place, with a count of the samples that need the
    # exact model (z_c <= z < 1 + k; z < 0 counts as off the star, as
    # before). No per-sample classification, gather or scatter for the
    # interior; the exact samples are collected in pass 2 from the blocks
    # that hold any. The interpolant is built up front (building it lazily
    # at the first on-star block made the whole pass ~0.25 us slower).
    ic = _int_coeffs(k, u1, u2) if interior else np.zeros(INT_W)
    n_b = (n_z + BLOCK_N - 1) // BLOCK_N
    cnt = np.empty(n_b, dtype=np.int64)
    tot = 0
    for b in range(n_b):
        lo = b * BLOCK_N
        hi = min(lo + BLOCK_N, n_z)
        zr = zs[lo:hi]
        fr = flux[lo:hi]
        on = 0
        for q in range(hi - lo):
            on += 1 if (zr[q] >= 0) & (zr[q] < one_k2) else 0
        if on == 0:
            for q in range(hi - lo):
                fr[q] = 1.0
            c = 0
        elif interior:
            c = _occult_int_select(zs, lo, hi, ic, omk, one_k, flux)
        else:
            c = on
            for q in range(hi - lo):
                fr[q] = 1.0
        cnt[b] = c
        tot += c
    if tot == 0:
        return flux

    # Pass 2: collect the exact samples of the blocks that hold any: band
    # (z_c <= z < 1 - k) from the front of zb / idx, with limb (P21b) the
    # limb band 1 - k <= z < 1 + k from zb[n_z + 4:] (if short, it joins
    # the band's batch); fully occulted samples (k >= 1) get 0
    one_mk2 = omk * omk if limb else one_k2   # one_k2: no sample is a limb sample
    full = k >= 1
    km1_2 = (k - 1) ** 2 if full else 0.0
    # (buffers sized by n_z, not tot: safe even if a scalar comparison
    # below disagreed with the vectorised count)
    idx = np.empty(2 * n_z + 4, dtype=np.int64)
    zb = np.empty(2 * n_z + 4)
    m = 0
    l0 = n_z + 4
    ml = l0
    for b in range(n_b):
        if cnt[b] == 0:
            continue
        lo = b * BLOCK_N
        hi = min(lo + BLOCK_N, n_z)
        for i in range(lo, hi):
            z2 = zs[i]
            if not (z2 >= z2_c and z2 < one_k2):
                continue
            if full and z2 <= km1_2:       # the source is completely occulted
                flux[i] = 0.0
                continue
            # (z == k: one-ulp nudge in _sqrt_lanes)
            if z2 >= one_mk2:
                zb[ml] = z2
                idx[ml] = i
                ml += 1
            else:
                zb[m] = z2
                idx[m] = i
                m += 1
    n_l = ml - l0
    use_limb = n_l >= LIMB_MIN_LANES
    if not use_limb:
        for q in range(n_l):
            zb[m + q] = zb[l0 + q]
            idx[m + q] = idx[l0 + q]
        m += n_l

    # Pass 3: exact flux (vectorised; P25: padded to a multiple of 4 lanes
    # when that replaces >= 2 scalar remainder lanes, ~25 ns each), then
    # non-converged lanes (scalar), clamp, scatter. The limb's arccos
    # terms are in the batch since P16.
    if m > 0:
        mp = m + (4 - m % 4) % 4 if m % 4 >= 2 else m
        for q in range(m, mp):
            zb[q] = zb[m - 1]
        _sqrt_lanes(zb, 0, mp, k)          # P26: z^2 -> z
        fb = np.empty(mp)
        dev = np.empty(mp)
        _occult_exact_batch(zb, mp, k, c1, c2, u2, omega, fb, dev)
        for j in range(m):
            f = fb[j]
            if not dev[j] <= 0:
                f = _occult_point(zb[j], k, u1, u2)
            # Some combinations of (u1,u2) cause flux>1 which is unphysical
            if f > 1:
                f = 1
            flux[idx[j]] = f

    # Pass 4 (P21b): limb band by the per-call interpolant (clamped)
    if use_limb:
        coef = _limb_coeffs(k, u1, u2)
        fl = np.empty(n_l)
        _occult_limb_batch(zb[l0:ml], n_l, k, coef, fl)
        for j in range(n_l):
            flux[idx[l0 + j]] = fl[j]
    return flux
