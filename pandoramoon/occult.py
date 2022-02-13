import numpy as np
from numpy import sqrt, pi, arccos, abs, log, ceil
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def create_occult_cache(u1, u2, dim):
    """2D Cache of size dim*dim with Mandel-Agol occultation values
    as a function of quadratic limb darkening values (u1, u2), k (=radius ratio)"""
    z_max = 1.1
    k_min = 0.001
    k_max = 0.1
    zs = np.linspace(0, z_max, dim)
    ks = np.linspace(0.001, k_max, dim)
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
    return f


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_small_single_value(z, k, u1, u2):
    """Small body approximation by Mandel-Agol. Adequate for k<~0.01
    This version does not calculate an entire array, but just one value.
    It is used by occult_hybrid with its linear interpolation between exact values
    and small-planet approximation."""
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    m = sqrt(1 - min(z**2, 1))
    limb_darkening = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
    area = cci(1, k, z)
    return (s - limb_darkening * area) * (1 / s)


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
    return 0


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
    if abs(k - 0.5) < 1e-4:
        k = 0.5

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
            z += 1e-6

        # The source is unocculted
        if z > 1 + k or z < 0:
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
            kap1 = arccos(min((1 - k2 + z2) / (2 * z + epsilon), 1))
            kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z + epsilon), 1))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
        # Occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            le[i] = k2

        # Edge of occulting body lies at the origin: special expressions in this case:
        if abs(z - k) <= 1e-4 * (z + k):
            # ! Table 3, Case V.:
            if k == 0.5:
                ld[i] = 1 / 3 - 4 * INV_PI / 9
                ed[i] = 3 / 32
            elif z > 0.5:
                ld[i] = (
                    1 / 3
                    + 16 * k / 9 * INV_PI * (2 * k2 - 1) * ellec(0.5 / k)
                    - (32 * k**4 - 20 * k2 + 3) / 9 * INV_PI / k * ellk(0.5 / k)
                )
                dist = sqrt((1 - x1) * (x2 - 1))
                ed[i] = (
                    1
                    / 2
                    * INV_PI
                    * (kap1 + k2 * (k2 + 2 * z2) * kap0 - (1 + 5 * k2 + z2) / 4 * dist)
                )
            elif z < 0.5:
                # ! Table 3, Case VI.:
                ld[i] = 1 / 3 + 2 / 9 * INV_PI * (
                    4 * (2 * k2 - 1) * ellec(2 * k) + (1 - 4 * k2) * ellk(2 * k)
                )
                ed[i] = k2 / 2 * (k2 + 2 * z2)

        # Occulting body partly occults the source and crosses the limb:
        # Table 3, Case III:
        if (z >= 0.5 + abs(k - 0.5) and z < 1 + k) or (
            k > 0.5 and z > abs(1 - k) and z < k
        ):
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)

            # Numerical stability: if k=0.06 then q is close to 1, ellpicb goes to hell
            if (q - 1) < 1e-8:
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
                    - (1 + 5 * k2 + z2) / 4 * sqrt((1 - x1) * (x2 - 1))
                )
            )

        # Occulting body transits the source:
        if k <= 1 and z < (1 - k):
            q = sqrt((x2 - x1) / (1 - x1) + 1e-8)  # re-calc because different condition
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
            if abs(k + z - 1) < 1e-4:
                ld[i] = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            ed[i] = k2 / 2 * (k2 + 2 * z2)
        flux[i] = 1 - (c1 * le[i] + c2 * ld[i] + u2 * ed[i]) * omega
    return flux


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_hybrid(zs, k, u1, u2):
    """Evaluates the transit model for an array of normalized distances.
    This version performs linear interpolation between exact values and small-planet.

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
    if abs(k - 0.5) < 1e-4:
        k = 0.5

    interpol_flux_1 = 0
    interpol_flux_2 = 0

    INV_PI = 1 / pi
    k2 = k**2
    flux = np.empty(len(zs))
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2

    for i in range(-2, len(zs)):

        # Linear interpolation method: First two values are to interpolate
        if i == -2:
            z = 0
        elif i == -1:
            z = 0.65
        else:
            z = zs[i]

        if abs(z - k) < 1e-6:
            z += 1e-6

        # Source is unocculted
        if z > 1 + k or z < 0:
            flux[i] = 1
            continue

        z2 = z**2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k**2 - z2

        # Star partially occulted and the occulting object crosses the limb
        if z >= abs(1 - k) and z <= 1 + k:
            kap1 = arccos(min((1 - k2 + z2) / (2 * z), 1))
            kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z), 1))
            lex = k2 * kap0 + kap1
            lex = (lex - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI

        # Occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            lex = k2

        # Occulting body partly occults source and crosses the limb: Case III:
        if (z > 0.5 + abs(k - 0.5) and z < 1 + k) or (
            k > 0.5 and z > abs(1 - k) and z < k
        ):
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)
            ldx = (
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
                ldx = ldx + 2 / 3
            edx = (
                1
                / 2
                * INV_PI
                * (
                    kap1
                    + k2 * (k2 + 2 * z2) * kap0
                    - (1 + 5 * k2 + z2) / 4 * sqrt((1 - x1) * (x2 - 1))
                )
            )

        # Use interpolation method
        if (
            (i >= 0 and k <= 0.05 and z <= 0.65)
            or (i >= 0 and k <= 0.04 and z <= 0.70)
            or (i >= 0 and k <= 0.03 and z <= 0.80)
            or (i >= 0 and k <= 0.02 and z <= 0.95)
            or (i >= 0 and k <= 0.01 and z <= 0.98)
        ):
            # Perform linear interpolation correction
            flux[i] = (
                occult_small_single_value(z, k, u1, u2)
                + interpol_flux_1
                + interpol_flux_2 * z
            )
            continue

        # Occulting body transits the source: Table 3, Case IV:
        if z < (1 - k):
            # print(i, k, z)
            q = sqrt((x2 - x1) / (1 - x1))
            ldx = (
                2
                / 9
                * INV_PI
                / sqrt(1 - x1)
                * (
                    (1 - 5 * z2 + k2 + x3 * x3) * ellk(q)
                    + (1 - x1) * (z2 + 7 * k2 - 4) * ellec(q)
                    - 3 * x3 / x1 * ellpicb((x2 / x1 - 1), q)
                )
            )
            if z < k:
                ldx = ldx + 2 / 3
            if abs(k + z - 1) < 1e-4:
                ldx = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            edx = k2 / 2 * (k2 + 2 * z2)

        current_flux = 1 - (c1 * lex + c2 * ldx + u2 * edx) * omega

        # Linear interpolation method: First two values are to interpolate
        if i == -2:
            interpol_flux_1 = current_flux - occult_small_single_value(z, k, u1, u2)
        elif i == -1:
            interpol_flux_2 = current_flux - occult_small_single_value(z, k, u1, u2)
        else:
            flux[i] = current_flux

    return flux
