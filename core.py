import numpy as np
from numpy import sqrt, pi, sin, cos, arcsin, arccos, abs, log, tan, arctan, ceil, fliplr, flipud
from numba import jit

# IDEAS:
# check if eclipses possible at all (func of incl, ecc, a, omega). skip if not (6600/sec)
# only one cci: merge funcs
# cache trigo funcs
# cache ellec, ellek
# create all arrays in advance and re-use them
# check if correct: if unphysical: z_moon = xm.copy() <-- xm? rather .large value?
# Bresenham algorithm to paint fast circle
# bilinear interpolation for mandel agol cache. pre-fetch 2 curves from array
# re-introduce special case into occult_hybrid:
#   "Edge of occulting body lies at the origin: special expressions in this case"

# DONE:
# ALensKite and ALens are faster (25%) than CCI: swap? 
# ==> faster if benched separately, but not when inlined

# in occult_small: move line to inner loop? m = sqrt(1 - min(b[j] ** 2, 1))



@jit(nopython=True, fastmath=True)
def create_occult_cache(u1, u2, dim):
    z_max = 1.1
    k_min = 0.001
    k_max = 0.1
    zs = np.linspace(0, z_max, dim)
    ks = np.linspace(0.001, k_max, dim)
    fs = np.empty((dim, dim), dtype="float32")
    for count, k in enumerate(ks):
        fs[count] = occult(zs, k, u1, u2)
    return (fs, ks, zs)


@jit(nopython=True, fastmath=True)
def read_occult_cache(zs_target, k, cache):
    fs, ks, zs = cache
    flux = np.ones(len(zs_target))
    idx_k = int(np.ceil((k - ks[0]) / (ks[1] - ks[0])))
    ratio_k = (ks[idx_k] - k) / (ks[idx_k] - ks[idx_k-1])
    curve = fs[idx_k-1] * ratio_k + fs[idx_k] * (1 - ratio_k)
    for idx in range(len(zs_target)):
        if zs_target[idx] == 0:
            idx_z = 1
        else:
            idx_z = int(np.ceil(zs_target[idx] / zs[1]))
        if zs_target[idx] >= 1.1:
            continue

        ratio_z = (zs[idx_z] - zs_target[idx]) / (zs[idx_z] - zs[idx_z-1])
        res = curve[idx_z-1] * ratio_z + curve[idx_z] * (1 - ratio_z)
        if res > 1:
            res = 1
        if res < 0:
            res = 0
        flux[idx] = res
    return flux




@jit(nopython=True, fastmath=True)
def cci_kite(r, b):
    if (1 + r <= b):
        return 0
    elif abs(1 - r) < b and b <= 1 + r:
        Akite = 0.5 * sqrt((b + (r + 1)) * (1 - (b - r)) * (1 + (b - r)) * (b + (r - 1)))
        #Akite = 0.5 * sqrt((b - r + 1)*(-b + r + 1) * (b + r - 1) * (b + r + 1))
        kappa0 = np.arctan2(2 * Akite, (r - 1) * (r + 1) + b * b)
        kappa1 = np.arctan2(2 * Akite, (1 - r) * (1 + r) + b * b)
        return kappa1 + r * r * kappa0 - Akite
    elif b <= 1 - r:
        return pi * r * r
    elif b <= r - 1:
        return pi


@jit(nopython=True, fastmath=True)
def cci(r1, r2, d):
    """Calculates area of asymmetric "lens" in which two circles intersect
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    if r1 < d - r2:
        return 0
    elif r1 >= d + r2:
        return pi * r2 ** 2
    elif d - r2 <= -r1:
        return pi * r1 ** 2
    else:
        return (
            r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            + r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            - 0.5 * sqrt((-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1))
        )


@jit(nopython=True, fastmath=True)
def occult_small(zs, k, u1, u2):
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


@jit(nopython=True, fastmath=True)
def occult_small_kite(zs, k, u1, u2):
    i = zs.size
    f = np.ones(i)
    b = abs(zs)
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    s_inv = 1 / s
    for j in range(i):
        if b[j] < 1 + k:
            m = sqrt(1 - min(b[j] ** 2, 1))
            limb_darkening = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
            #area = cci(1, k, b[j])
            area = cci_kite(k, b[j])
            f[j] = (s - limb_darkening * area) * s_inv
    return f


@jit(nopython=True, fastmath=True)
def occult_small_single_value(z, k, u1, u2):
    s = 2 * pi * 1 / 12 * (-2 * u1 - u2 + 6)
    m = sqrt(1 - min(z ** 2, 1))
    limb_darkening = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
    area = cci(1, k, z)
    return (s - limb_darkening * area) * (1 / s)



@jit(nopython=True, fastmath=True)
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
    kc = sqrt(1 - k ** 2)
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


@jit(nopython=True, fastmath=True)
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

    # Error if 1/m == 0 in log(1/m1)
    #print("m1", m1)
    #if abs(1 / m1) < 1e-6:
    #    m1 += 1e-6
    #print("m1 fixed", m1)
    return (1 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) + (
        m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1 / (m1 + epsilon))
    )


@jit(nopython=True, fastmath=True)
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
@jit(nopython=True, fastmath=True)
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
    k2 = k ** 2
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

        z2 = z ** 2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k ** 2 - z2

        # Eq. 26: Star partially occulted and the occulting object crosses the limb
        if z >= abs(1 - k) and z <= 1 + k:
            kap1 = arccos(min((1 - k2 + z2) / (2 * z + epsilon), 1))
            kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z + epsilon), 1))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI
            #print("eq26", z, kap1, kap0, le[i])
        # Occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            le[i] = k2

        # Edge of occulting body lies at the origin: special expressions in this case:
        if abs(z - k) <= 1e-4 * (z + k):
            #print("edge case", z, k, abs(z-k), 1e-4 * (z + k))
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
                    * (kap1 + k2 * (k2 + 2 * z2) * kap0 - (1 + 5 * k2 + z2) / 4 * dist)
                )
            elif z < 0.5:
                # ! Table 3, Case VI.:
                ld[i] = 1 / 3 + 2 / 9 * INV_PI * (
                    4 * (2 * k2 - 1) * ellec(2 * k) + (1 - 4 * k2) * ellk(2 * k)
                )
                ed[i] = k2 / 2 * (k2 + 2 * z2)
                #print(z, ld[i], ed[i])

        # Occulting body partly occults the source and crosses the limb:
        # Table 3, Case III:
        if (z >= 0.5 + abs(k - 0.5) and z < 1 + k) or (
            k > 0.5 and z > abs(1 - k) and z < k
        ):
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)

            # Numerical stability: if k=0.06 then q is close to 1, 
            # and then ellpicb goes to hell
            if (q - 1) < 1e-8:
                q -= 1e-8
            #print("q", q)
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
            #ld[i] = 0.000899506134059702

            #print("ld[i]", ld[i], (1 / x1 - 1), q)
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
            #print("ed here", ed[i])

        # Occulting body transits the source:
        # Table 3, Case IV.:
        if k <= 1 and z < (1 - k):
            
            #if x1 == 0:
            #    x1 += 
            #print("x1, x2", x1, x2)
            q = sqrt((x2 - x1) / (1 - x1) + 1e-8) # re-calc because different condition
            #x3 += 1e-8
            #x1 += 1e-9
            #print("vals", x1, x2, z2, k2, x3, k, q)
            #print("(x2 / x1 - 1)", (x2 / x1 - 1))
            #x3 += epsilon
            # x1 = 0
            # x3 = 0
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
        #print(z, c1, le[i], c2, ld[i], u2, ed[i], omega)
        #print(z, ld[i], ed[i])
        flux[i] = 1 - (c1 * le[i] + c2 * ld[i] + u2 * ed[i]) * omega
    return flux


@jit(nopython=True, fastmath=True)
def occult_hybrid(zs, k, u1, u2):
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

    interpolation_flux_1 = 0
    interpolation_flux_2 = 0

    INV_PI = 1 / pi
    k2 = k ** 2
    flux = np.empty(len(zs))
    omega = 1 / (1 - u1 / 3 - u2 / 6)
    c1 = 1 - u1 - 2 * u2
    c2 = u1 + 2 * u2

    for i in range(-2, len(zs)):

        # Linear interpolation method: First two values are to interpolate
        if (i == -2):
            z = 0
        elif (i == -1):
            z = 0.65
        else:
            z = zs[i]

        if abs(z - k) < 1e-6:
            z += 1e-6

        # Source is unocculted
        if z > 1 + k or z < 0:
            flux[i] = 1
            continue

        z2 = z ** 2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k ** 2 - z2

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
            #print(i, interpolation_flux_1, interpolation_flux_2)
            flux[i] = occult_small_single_value(z, k, u1, u2) + interpolation_flux_1 + z * interpolation_flux_2


            continue

        # Occulting body transits the source: Table 3, Case IV:
        if z < (1 - k):
            #print(i, k, z)
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
        if (i == -2):
            interpolation_flux_1 = current_flux - occult_small_single_value(z, k, u1, u2)
        elif (i == -1):
            interpolation_flux_2 = current_flux - occult_small_single_value(z, k, u1, u2)
        else:
            flux[i] = current_flux

    return flux


@jit(nopython=True, fastmath=True)
def ellipse(
    a, per, tau, Omega, i, time, transit_threshold_x, x_bary, mass_ratio, b_bary
):
    """2D x-y Kepler solver without eccentricity"""

    # Debugging, remove later:
    #transit_threshold_x = 1000.0

    bignum = 1.0e8  # a float for px = out of transit (no np.inf in numba)
    l = len(time)
    O = Omega / 180 * pi
    i = i / 180 * pi
    xm = np.full(l, bignum)
    ym = xm.copy()
    xp = xm.copy()
    yp = xm.copy()
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    for idx in range(l):
        if abs(x_bary[idx]) < transit_threshold_x:
            k = pi * (time[idx] - tau * per) / per

            # Standard Version #1
            cos_Q = cos(2 * k)
            sin_Q = sin(2 * k)

            # Alternative Version #2: Swap sin and cos for one tan:
            #h = tan(k)
            #cos_Q = (1 - h ** 2) / (1 + h ** 2)
            #sin_Q = (2 * h) / ((1 + h ** 2))

            vector_x = cos(O) * cos_Q - sin(O) * sin_Q * cos(i)
            vector_y = sin(O) * cos_Q + cos(O) * sin_Q * cos(i)
            xm[idx] = +vector_x * a_moon + x_bary[idx]
            ym[idx] = +vector_y * a_moon + b_bary
            xp[idx] = -vector_x * a_planet + x_bary[idx]
            yp[idx] = -vector_y * a_planet + b_bary
    return xm, ym, xp, yp


@jit(nopython=True, fastmath=True)
def ellipse_vector(
    a, per, tau, Omega, i, time, x_bary, mass_ratio, b_bary
):
    """2D x-y Kepler solver without eccentricity"""

    O = Omega / 180 * pi
    i = i / 180 * pi
    xm = np.empty(len(time))
    ym = xm.copy()
    xp = xm.copy()
    yp = xm.copy()
    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    k = pi * (time - tau * per) / per
    cos_Q = cos(2 * k)
    sin_Q = sin(2 * k)
    vector_x = cos(O) * cos_Q - sin(O) * sin_Q * cos(i)
    vector_y = sin(O) * cos_Q + cos(O) * sin_Q * cos(i)
    xm = +vector_x * a_moon + x_bary
    ym = +vector_y * a_moon + b_bary
    xp = -vector_x * a_planet + x_bary
    yp = -vector_y * a_planet + b_bary
    return xm, ym, xp, yp


@jit(nopython=True, fastmath=True)
def ellipse_ecc(a, per, e, tau, Omega, w, i, time, mass_ratio, x_bary, b_bary):
    """2D x-y Kepler solver WITH eccentricity"""

    M = (2 * pi / per) * (time - (tau * per))
    flip = False
    M = M - (np.floor(M / (2 * pi)) * 2 * pi)
    if M.any() > pi:
        M = 2 * pi - M
        flip = True
    alpha = (3 * pi ** 2 + 1.6 * pi * (pi - abs(M)) / (1 + e)) / (pi ** 2 - 6)
    d = 3 * (1 - e) + alpha * e
    r1 = 3 * alpha * d * (d - 1 + e) * M + M ** 3
    q = 2 * alpha * d * (1 - e) - M ** 2
    w1 = (abs(r1) + sqrt(q ** 3 + r1 ** 2)) ** (2 / 3)
    n = (2 * r1 * w1 / (w1 ** 2 + w1 * q + q ** 2) + M) / d
    f0 = n - e * sin(n) - M
    f1 = 1 - e * cos(n)
    f2 = e * sin(n)
    g = -f0 / (f1 - 0.5 * f0 * f2 / f1)
    h = -f0 / (f1 + 0.5 * g * f2 + (g ** 2) * (1 - f1) / 6)
    k = n - f0 / (f1 + 0.5 * h * f2 + h ** 2 * (1 - f1) / 6 + h ** 3 * (-f2) / 24)
    if flip:
        k = 2 * pi - k
    r = -(1 - e * cos(k))

    wf = (w / 180 * pi) + (arctan(sqrt((1 + e) / (1 - e)) * tan(k / 2)) * 2)
    v = sin(wf) * cos((i / 180 * pi))
    O = Omega / 180 * pi
    vector_x = (cos(O) * cos(wf) - sin(O) * v) * r
    vector_y = (sin(O) * cos(wf) + cos(O) * v) * r

    a_planet = (a * mass_ratio) / (1 + mass_ratio)
    a_moon = a - a_planet
    xm = +vector_x * a_moon + x_bary
    ym = +vector_y * a_moon + b_bary
    xp = -vector_x * a_planet + x_bary
    yp = -vector_y * a_planet + b_bary
    return xm, ym, xp, yp


@jit(nopython=True, fastmath=True)
def eclipse_ratio(distance_planet_moon, r_planet, r_moon):
    """Returns eclipsed ratio [0..1] using circle_circle_intersect"""
    eclipsing = False
    eclipse_ratio = 0
    if abs(distance_planet_moon) < (r_planet + r_moon):
        eclipsing = True
        if (r_planet - r_moon) > abs(distance_planet_moon):
            eclipse_ratio = 1
            return eclipse_ratio
    # For partial eclipses, get the fraction of moon eclipse using transit...
    if eclipsing:
        if distance_planet_moon == 0:
            distance_planet_moon = 1e-10
        eclipse_ratio = cci(r_planet, r_moon, distance_planet_moon)
        # ...and transform this value into how much AREA is really eclipsed
        eclipse_ratio = eclipse_ratio / (pi * r_moon ** 2)
    return eclipse_ratio


@jit(nopython=True, fastmath=True)
def pixelart(xp, yp, xm, ym, r_planet, r_moon, numerical_grid):
    if numerical_grid % 2 == 0:  # assure pixel number is odd for perfect circle
        numerical_grid += 1
    r_star = (1 / r_moon) * numerical_grid
    image = np.zeros((numerical_grid + 1, numerical_grid + 1), dtype="int8")
    
    color_star = 5    # arbitrary values, but useful for visualization
    color_moon = 3    # the sum of the values must be unique to identify the
    color_planet = 2  # total overlap: area of moon on star occulted by planet
    all_colors = color_star + color_moon + color_planet

    # Paint moon circle by painting one quarter, flipping it over, and rolling it down
    # Faster than painting naively the full circle, because it saves 3/4 of dist calcs
    # Caching this part would be a good idea, but numba memory mgmt caused crashes
    # The gain is very small anyways and the complexity not worth it
    # The version here, which replaces the usual sqrt with **2, is ~5% faster

    # Paint upper left corner
    anti_aliasing = np.sqrt(((numerical_grid + 1) ** 2 - (numerical_grid ** 2))) / 2
    mid = int(ceil(numerical_grid / 2))
    for x in range(mid):
        for y in range(mid):
            d_moon = (numerical_grid - 2 * x) ** 2 + (numerical_grid - 2 * y) ** 2
            if d_moon < (numerical_grid**2 + anti_aliasing):
                image[x, y] = color_moon
    image[mid:,:mid] = flipud(image[:mid:,:mid]) # Copy upper left to upper right
    image[:,mid:] = fliplr(image[:,:mid])  # Copy upper half to lower half

    # Now add planet and star
    anti_aliasing = -0.5 / numerical_grid  # Working with sqrt again
    for x in range(numerical_grid + 1):
        for y in range(numerical_grid + 1):
            d_star = sqrt(
                (xm * r_star + 2 * x - numerical_grid) ** 2
                + (ym * r_star + 2 * y - numerical_grid) ** 2
            )
            if d_star < r_star - anti_aliasing:
                image[x, y] += color_star

            d_planet = sqrt(
                ((-(xp - xm) * r_star) + 2 * x - numerical_grid) ** 2
                + ((-(yp - ym) * r_star) + 2 * y - numerical_grid) ** 2
            )
            if d_planet < (r_planet / r_moon) * numerical_grid - anti_aliasing:
                image[x, y] += color_planet

    moon_sum_analytical = pi * ((numerical_grid) / 2) ** 2
    moon_occult_frac = np.sum(image == all_colors) / moon_sum_analytical
    cci = eclipse_ratio(sqrt(xm ** 2 + ym ** 2), 1, r_moon)
    if cci > 0:
        return min(1, (1 - ((cci - moon_occult_frac) / cci)))
    else:
        return 1


@jit(nopython=True, fastmath=True)
def eclipse(xp, yp, xm, ym, r_planet, r_moon, flux_moon, numerical_grid):
    """Checks if planet-moon occultation present. If yes, returns adjusted moon flux.
    Parameters
    ----------
    xp, yp, x_m, ym : float
        Planet and moon coordinates. Normalized so that R_star = 1 at (0,0)
    r_planet, r_moon : float
        Planet and moon radii. Normalized so that R_star = 1
    flux_moon : float
        Un-occulted moon flux (from Mandel-Agol model) in [0,1]

    Returns
    -------
    occulted_flux_moon : float
        Occulted moon flux <= flux_moon. Assumes planet occults moon.
    """

    # Planet-Moon occultation
    # Case 1: No occultation
    # Case 2: Occultation, both bodies on star or off star --> 2-circle intersect
    # Case 3: Occultation, any body on limb --> Numerical solution
    for idx in range(len(xp)):
        planet_moon_occultation = False
        on_limb = False

        # Check if moon or planet are on stellar limb
        if abs(1 - (sqrt(xm[idx] ** 2 + ym[idx] ** 2))) < (r_moon):
            on_limb = True
        if abs(1 - (sqrt(xp[idx] ** 2 + yp[idx] ** 2))) < (r_planet):
            on_limb = True

        # Check if planet-moon occultation
        distance_p_m = sqrt((xm[idx] - xp[idx]) ** 2 + (ym[idx] - yp[idx]) ** 2)
        if abs(distance_p_m) < (r_planet + r_moon):
            planet_moon_occultation = True

        # Case 1: No occultation
        else:
            continue

        # Case 2: Occultation, both bodies on star or off star --> 2 circle intersect
        if planet_moon_occultation and not on_limb:
            er = eclipse_ratio(distance_p_m, r_planet, r_moon)

        # Case 3: Occultation, any body on limb --> numerical estimate with pixel-art
        if planet_moon_occultation and on_limb:
            er = pixelart(
                xp[idx],
                yp[idx],
                xm[idx],
                ym[idx],
                r_planet,
                r_moon,
                numerical_grid
            )

        # For Cases 2+3: Calculate reduced moon flux
        if er > 0:
            flux_moon[idx] = -(1 - flux_moon[idx]) * 10 ** 6
            flux_moon[idx] = flux_moon[idx] * (1 - er)
            flux_moon[idx] = 1 - (-flux_moon[idx] * 10 ** -6)
    return flux_moon


@jit(nopython=True, fastmath=True)
def resample(arr, factor):
    out_samples = int(len(arr) / factor)
    out_arr = np.ones(out_samples)
    for idx in range(out_samples):
        start = idx * factor
        end = start + factor - 1
        out_arr[idx] = np.mean(arr[start:end])
    return out_arr


@jit(nopython=True, fastmath=True)
def ld_convert(q1, q2):
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return u1, u2


@jit(nopython=True, fastmath=True)
def timegrid(t0_bary, epochs, epoch_duration, cadences_per_day, epoch_distance, supersampling_factor):
    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_bary`, however, is a free parameter
    ti_epoch_midtimes = np.arange(
        start=t0_bary,
        stop=t0_bary + epoch_distance * epochs,
        step=epoch_distance,
    )

    # arrays of epoch start and end dates [day]
    t_starts = ti_epoch_midtimes - epoch_duration / 2
    t_ends = ti_epoch_midtimes + epoch_duration / 2

    # "Morphological light-curve distortions due to finite integration time"
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract
    # Data gets smeared over long integration. Relevant for e.g., 30min cadences
    # To counter the effect: Set supersampling_factor = 5 (recommended value)
    # Then, 5x denser in time sampling, and averaging after, approximates effect
    if supersampling_factor < 1:
        print("supersampling_factor must be positive integer")
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)

    cadences = int(supersampled_cadences_per_day * epoch_duration)
    time = np.empty(shape=(epochs, cadences))
    for epoch in range(epochs):
        time[epoch] = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
    return time.ravel()


@jit(nopython=True, fastmath=True)
def x_bary_grid(time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary, w_bary):
    # Planetary transit duration at b=0 equals the width of the star
    # Formally correct would be: (R_star+r_planet) for the transit duration T1-T4
    # Here, however, we need points where center of planet is on stellar limb
    # https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    tdur = per_bary / pi * arcsin(1 / a_bary)

    # Correct transit duration based on relative orbital velocity
    # of circular versus eccentric case
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))

    # t0_bary_offset in [days] ==> convert to x scale (i.e. 0.5 transit dur radius)
    t0_shift_planet = t0_bary_offset / (tdur / 2)

    epochs = int((time[-1] - t0_bary) / per_bary + 0.5) + 1  # maximum, may contain empty epochs
    ti_epoch_midtimes = np.arange(
        start=t0_bary,
        stop=t0_bary + epoch_distance * epochs,
        step=epoch_distance,
    )
    #print("ti_epoch_midtimes original", ti_epoch_midtimes)
    x_bary = np.empty(len(time))
    for idx in range(len(time)):
        epoch = int((time[idx] - t0_bary) / per_bary + 0.5)

        # Push planet following per_bary, which is a free parameter in [days]
        # For reference: Distance between epoch segments is fixed as segment_distance
        # Have to convert to x scale == 0.5 transit duration for stellar radii
        per_shift_planet = ((per_bary - epoch_distance) * epoch) / (tdur / 2)
        
        time_from_nearest_midtransit = 2 * (time[idx] - ti_epoch_midtimes[epoch])
        x_bary[idx] = time_from_nearest_midtransit / tdur - t0_shift_planet - per_shift_planet
    return x_bary



@jit(nopython=True, fastmath=True)
# Identical output to x_bary_grid but without np.arange (=slightly faster, but uglier)
def x_bary_grid_plain(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary, w_bary
):
    tdur = per_bary / pi * arcsin(1 / a_bary)
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))
    t0_shift_planet = t0_bary_offset / (tdur / 2)
    x_bary = np.empty(len(time))
    for idx in range(len(time)):
        epoch = int((time[idx] - t0_bary) / per_bary + 0.5)
        x_bary[idx] = (
            (2 * (time[idx] - (t0_bary + epoch_distance * epoch))) / tdur
            - t0_shift_planet
            - (((per_bary - epoch_distance) * epoch) / (tdur / 2))
        )
    return x_bary


@jit(nopython=True, fastmath=True)
def x_bary_grid_array(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary, w_bary
):
    # epoch_distance: float (fixed for all models)
    # t0_bary:        float (fixed for all models)

    # time: 1d array (same for all models)

    # a_bary, per_bary, t0_bary_offset, ecc_bary, w_bary : 1d arrays

    # x_bary: 2d array (models, x_bary)
    x_bary = np.empty(shape=(len(a_bary), len(time)))

    for model_idx in range(len(a_bary)):

        tdur = per_bary[model_idx] / pi * arcsin(1 / a_bary[model_idx])
        if ecc_bary[model_idx] > 0:
            tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))
        t0_shift_planet = t0_bary_offset[model_idx] / (tdur / 2)
        
        for time_idx in range(len(time)):
            epoch = int((time[time_idx] - t0_bary) / per_bary[model_idx] + 0.5)
            x_bary[model_idx,time_idx] = (
                (2 * (time[time_idx] - (t0_bary + epoch_distance * epoch))) / tdur
                - t0_shift_planet
                - (((per_bary[model_idx] - epoch_distance) * epoch) / (tdur / 2))
            )
    return x_bary

