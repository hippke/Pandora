# Modified by Michael Hippke 2021, based on a GPL3 licensed code from:
# PyTransit: fast and easy exoplanet transit modelling in Python.
# Copyright (C) 2010-2020  Hannu Parviainen

# Modifications: 
# - Restricted to quadratic limb-darkening
# - Caching a few expensive values from sqrt calcs
# - Replaced some array generations with floats or copies
# - Overall speed improvements: ~10%

from numba import jit, prange
from numpy import pi, sqrt, arccos, abs, log, ones, empty

HALF_PI = 0.5 * pi
INV_PI = 1 / pi


@jit(cache=False, nopython=True, fastmath=True)
def circle_circle_intersection_area(r1, r2, b):
    """Area of the intersection of two circles."""
    if r1 < b - r2:
        return 0
    elif r1 >= b + r2:
        return pi * r2 ** 2
    elif b - r2 <= -r1:
        return pi * r1 ** 2
    else:
        return (r2 ** 2 * arccos((b ** 2 + r2 ** 2 - r1 ** 2) / (2 * b * r2)) +
                r1 ** 2 * arccos((b ** 2 + r1 ** 2 - r2 ** 2) / (2 * b * r1)) -
                0.5 * sqrt((-b + r2 + r1) * (b + r2 - r1) * (b - r2 + r1) * (b + r2 + r1)))


@jit(cache=False, nopython=True, fastmath=True)
def occult_small(zs, k, u1, u2):
    i = zs.size
    f = ones(i)
    m = empty(i)
    b = abs(zs)
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    for j in range(i):
        m[j] = sqrt(1 - min(b[j]**2, 1))
        if b[j] < 1 + k:
            l = (1 - u1 * (1 - m[j]) - u2 * (1 - m[j]) ** 2)
            a = circle_circle_intersection_area(1, k, b[j])
            f[j] = (s - l * a) / s
    return f


@jit(cache=False, nopython=True, fastmath=True)
def ellpicb(n, k):

    """The complete elliptical integral of the third kind
    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353
    Adapted from L. Kreidbergs C version in BATMAN
    (Kreidberg, L. 2015, PASP 957, 127)
    (https://github.com/lkreidberg/batman)
    which is translated from J. Eastman's IDL routine
    in EXOFAST (Eastman et al. 2013, PASP 125, 83)"""

    e = kc = sqrt(1 - k ** 2)
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


@jit(cache=False, nopython=True, fastmath=True)
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
    return (1 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) + (
        m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1 / m1)
    )


# A version with cached log(m1) (from main func) is slower
@jit(cache=False, nopython=True, fastmath=True)
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
    return (a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) - (
        (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * log(m1)
    )


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
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

    k2 = k ** 2
    lzs = len(zs)
    flux = empty(lzs)
    le = empty(lzs)
    ld = empty(lzs)
    ed = empty(lzs)
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = (1 - u1 - 2 * u2)
    c2 = (u1 + 2 * u2)

    for i in prange(lzs):
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

        z2 = z ** 2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k ** 2 - z2

        # Eq. 26: Star partially occulted and the occulting object crosses the limb
        if z >= abs(1 - k) and z <= 1 + k:
            kap1 = arccos(min((1 - k2 + z2) / (2 * z), 1))
            kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z), 1))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI

        # Occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            le[i] = k2

        # Edge of occulting body lies at the origin: special expressions in this case:
        if abs(z - k) < 1e-4 * (z + k):
            # ! Table 3, Case V.:
            if k == 0.5:
                ld[i] = 1 / 3 - 4 * INV_PI / 9
                ed[i] = 3 / 32
            elif z > 0.5:
                ld[i] = (
                    1 / 3
                    + 16 * k / 9 * INV_PI * (2 * k2 - 1) * ellec(0.5 / k)
                    - (32 * k ** 4 - 20 * k2 + 3) / 9 * INV_PI / k * ellk(0.5 / k)
                )
                dist = sqrt((1 - x1) * (x2 - 1))
                ed[i] = (
                    1
                    / 2
                    * INV_PI
                    * (
                        kap1
                        + k2 * (k2 + 2 * z2) * kap0
                        - (1 + 5 * k2 + z2) / 4 * dist
                    )
                )
            elif z < 0.5:
                # ! Table 3, Case VI.:
                ld[i] = 1 / 3 + 2 / 9 * INV_PI * (
                    4 * (2 * k2 - 1) * ellec(2 * k) + (1 - 4 * k2) * ellk(2 * k)
                )
                ed[i] = k2 / 2 * (k2 + 2 * z2)

        # Occulting body partly occults the source and crosses the limb:
        # Table 3, Case III:
        if (z > 0.5 + abs(k - 0.5) and z < 1 + k) or (
            k > 0.5 and z > abs(1 - k) and z < k
        ):
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)
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
        # Table 3, Case IV.:
        if k <= 1 and z < (1 - k):
            q = sqrt((x2 - x1) / (1 - x1))  # recalculate because different condition
            ld[i] = (
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
                ld[i] = ld[i] + 2 / 3
            if abs(k + z - 1) < 1e-4:
                ld[i] = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            ed[i] = k2 / 2 * (k2 + 2 * z2)

        flux[i] = 1 - (c1 * le[i] + c2 * ld[i] + u2 * ed[i]) * omega
    return flux
