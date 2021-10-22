from numba import jit, prange
from numpy import pi, sqrt, arccos, abs, log, zeros, array, ndarray, copysign, ravel

HALF_PI = 0.5 * pi
INV_PI = 1 / pi


@jit(cache=True, nopython=True, fastmath=True)
def ellpicb(n, k):
    """The complete elliptical integral of the third kind
    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353
    Adapted from L. Kreidbergs C version in BATMAN
    (Kreidberg, L. 2015, PASP 957, 127)
    (https://github.com/lkreidberg/batman)
    which is translated from J. Eastman's IDL routine
    in EXOFAST (Eastman et al. 2013, PASP 125, 83)"""

    kc = sqrt(1 - k ** 2)
    e = kc
    p = sqrt(n + 1)
    ip = 1 / p
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


@jit(cache=True, nopython=True, fastmath=True)
def ellec(k):
    a1 = 0.443251414630
    a2 = 0.062606012200
    a3 = 0.047573835460
    a4 = 0.017365064510
    b1 = 0.249983683100
    b2 = 0.092001800370
    b3 = 0.040696975260
    b4 = 0.005264496390
    m1 = 1 - k ** 2
    return (1 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))) + (
        m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * log(1 / m1)
    )


@jit(cache=True, nopython=True, fastmath=True)
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


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def occult_array(zs, k, u: ndarray):
    """Evaluates the transit model for an array of normalized distances.
    Parameters
    ----------
    z: 1D array
        Normalized distances
    k: float
        Planet-star radius ratio
    u: 2D array
        Limb darkening coefficients
    Returns
    -------
    Transit model evaluated at `z`.
    """
    if abs(k - 0.5) < 1e-4:
        k = 0.5

    npt = len(zs)
    npb = u.shape[0]

    k2 = k ** 2
    omega = zeros(npb)
    flux = zeros((npt, npb))
    le = zeros(npt)
    ld = zeros(npt)
    ed = zeros(npt)

    for i in range(npb):
        omega[i] = 1 - u[i, 0] / 3 - u[i, 1] / 6

    for i in prange(npt):
        z = zs[i]

        if abs(z - k) < 1e-6:
            z += 1e-6

        # The source is unocculted
        if z > 1 + k or z < 0:
            flux[i, :] = 1
            le[i] = 0
            ld[i] = 0
            ed[i] = 0
            continue

        # The source is completely occulted
        elif k >= 1 and z <= k - 1:
            flux[i, :] = 0
            le[i] = 1
            ld[i] = 1
            ed[i] = 1
            continue

        z2 = z ** 2
        x1 = (k - z) ** 2
        x2 = (k + z) ** 2
        x3 = k ** 2 - z2

        # The source is partially occulted and the occulting object crosses the limb
        # Equation (26):
        if z >= abs(1 - k) and z <= 1 + k:
            kap1 = arccos(min((1 - k2 + z2) / (2 * z), 1))
            kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z), 1))
            le[i] = k2 * kap0 + kap1
            le[i] = (le[i] - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI

        # The occulting object transits the source star (but doesn't completely cover it):
        if z <= 1 - k:
            le[i] = k2

        # The edge of the occulting star lies at the origin- special expressions in this case:
        if abs(z - k) < 1e-4 * (z + k):
            # ! Table 3, Case V.:
            if k == 0.5:
                ld[i] = 1 / 3 - 4 * INV_PI / 9
                ed[i] = 3 / 32
            elif z > 0.5:
                q = 0.5 / k
                Kk = ellk(q)
                Ek = ellec(q)
                ld[i] = (
                    1 / 3
                    + 16 * k / 9 * INV_PI * (2 * k2 - 1) * Ek
                    - (32 * k ** 4 - 20 * k2 + 3) / 9 * INV_PI / k * Kk
                )
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
            elif z < 0.5:
                # ! Table 3, Case VI.:
                q = 2 * k
                Kk = ellk(q)
                Ek = ellec(q)
                ld[i] = 1 / 3 + 2 / 9 * INV_PI * (
                    4 * (2 * k2 - 1) * Ek + (1 - 4 * k2) * Kk
                )
                ed[i] = k2 / 2 * (k2 + 2 * z2)

        # The occulting star partly occults the source and crosses the limb:
        # Table 3, Case III:
        if (z > 0.5 + abs(k - 0.5) and z < 1 + k) or (
            k > 0.5 and z > abs(1 - k) and z < k
        ):
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)
            Kk = ellk(q)
            Ek = ellec(q)
            n = 1 / x1 - 1
            Pk = ellpicb(n, q)
            ld[i] = (
                1
                / 9
                * INV_PI
                / sqrt(k * z)
                * (
                    ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)) * Kk
                    + 4 * k * z * (z2 + 7 * k2 - 4) * Ek
                    - 3 * x3 / x1 * Pk
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

        # The occulting star transits the source:
        # Table 3, Case IV.:
        if k <= 1 and z < (1 - k):
            q = sqrt((x2 - x1) / (1 - x1))
            Kk = ellk(q)
            Ek = ellec(q)
            n = x2 / x1 - 1
            Pk = ellpicb(n, q)
            ld[i] = (
                2
                / 9
                * INV_PI
                / sqrt(1 - x1)
                * (
                    (1 - 5 * z2 + k2 + x3 * x3) * Kk
                    + (1 - x1) * (z2 + 7 * k2 - 4) * Ek
                    - 3 * x3 / x1 * Pk
                )
            )
            if z < k:
                ld[i] = ld[i] + 2 / 3
            if abs(k + z - 1) < 1e-4:
                ld[i] = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            ed[i] = k2 / 2 * (k2 + 2 * z2)

        for j in range(npb):
            flux[i, j] = (
                1
                - (
                    (1 - u[j, 0] - 2 * u[j, 1]) * le[i]
                    + (u[j, 0] + 2 * u[j, 1]) * ld[i]
                    + u[j, 1] * ed[i]
                )
                / omega[j]
            )

    return ravel(flux)


@jit(cache=True, nopython=True, fastmath=True)
def occult_single(z: float, k: float, u: ndarray):
    """Evaluates the transit model for scalar normalized distance.
    Parameters
    ----------
    z: float
        Normalized distance
    k: float
        Planet-star radius ratio
    u: 1D array
        Limb darkening coefficients
    Returns
    -------
    Transit model evaluated at `z`.
    """
    if abs(k - 0.5) < 1e-4:
        k = 0.5

    k2 = k ** 2
    omega = 1 - u[0] / 3 - u[1] / 6

    if abs(z - k) < 1e-6:
        z += 1e-6

    # The source is unocculted
    if z > 1 + k or (copysign(1, z) < 0):
        return 1

    # The source is completely occulted
    elif k >= 1 and z <= k - 1:
        return 0

    z2 = z ** 2
    x1 = (k - z) ** 2
    x2 = (k + z) ** 2
    x3 = k ** 2 - z ** 2

    # LE
    # --
    # Case I: The occulting object fully inside the disk of the source
    if z <= 1 - k:
        le = k2

    # Case II: ingress and egress
    elif z >= abs(1 - k) and z <= 1 + k:
        kap1 = arccos(min((1 - k2 + z2) / (2 * z), 1))
        kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z), 1))
        le = k2 * kap0 + kap1
        le = (le - 0.5 * sqrt(max(4 * z2 - (1 + z2 - k2) ** 2, 0))) * INV_PI

    # LD and ED
    # ---------
    is_edge_at_origin = abs(z - k) < 1e-4 * (z + k)
    is_full_transit = k <= 1 and z < (1 - k)

    # Case 0: The edge of the occulting object lies at the origin
    if is_edge_at_origin:
        if k == 0.5:
            ld = 1 / 3 - 4 * INV_PI / 9
            ed = 3 / 32

        elif z > 0.5:
            q = 0.5 / k
            Kk = ellk(q)
            Ek = ellec(q)
            ld = (
                1 / 3
                + 16 * k / 9 * INV_PI * (2 * k2 - 1) * Ek
                - (32 * k ** 4 - 20 * k2 + 3) / 9 * INV_PI / k * Kk
            )
            ed = (
                1
                / 2
                * INV_PI
                * (
                    kap1
                    + k2 * (k2 + 2 * z2) * kap0
                    - (1 + 5 * k2 + z2) / 4 * sqrt((1 - x1) * (x2 - 1))
                )
            )

        elif z < 0.5:
            q = 2 * k
            Kk = ellk(q)
            Ek = ellec(q)
            ld = 1 / 3 + 2 / 9 * INV_PI * (4 * (2 * k2 - 1) * Ek + (1 - 4 * k2) * Kk)
            ed = k2 / 2 * (k2 + 2 * z2)
    else:
        # Case I: The occulting object fully inside the disk of the source
        if is_full_transit:
            q = sqrt((x2 - x1) / (1 - x1))
            Kk = ellk(q)
            Ek = ellec(q)
            n = x2 / x1 - 1
            Pk = ellpicb(n, q)
            ld = (
                2
                / 9
                * INV_PI
                / sqrt(1 - x1)
                * (
                    (1 - 5 * z2 + k2 + x3 * x3) * Kk
                    + (1 - x1) * (z2 + 7 * k2 - 4) * Ek
                    - 3 * x3 / x1 * Pk
                )
            )
            if z < k:
                ld = ld + 2 / 3
            if abs(k + z - 1) < 1e-4:
                ld = 2 / 3 * INV_PI * arccos(1 - 2 * k) - 4 / 9 * INV_PI * sqrt(
                    k * (1 - k)
                ) * (3 + 2 * k - 8 * k2)
            ed = k2 / 2 * (k2 + 2 * z2)

        # Case II: ingress and egress
        else:
            q = sqrt((1 - (k - z) ** 2) / 4 / z / k)
            Kk = ellk(q)
            Ek = ellec(q)
            n = 1 / x1 - 1
            Pk = ellpicb(n, q)
            ld = (
                1
                / 9
                * INV_PI
                / sqrt(k * z)
                * (
                    ((1 - x2) * (2 * x2 + x1 - 3) - 3 * x3 * (x2 - 2)) * Kk
                    + 4 * k * z * (z2 + 7 * k2 - 4) * Ek
                    - 3 * x3 / x1 * Pk
                )
            )
            if z < k:
                ld = ld + 2 / 3
            ed = (
                1
                / 2
                * INV_PI
                * (
                    kap1
                    + k2 * (k2 + 2 * z2) * kap0
                    - (1 + 5 * k2 + z2) / 4 * sqrt((1 - x1) * (x2 - 1))
                )
            )

    flux = 1 - ((1 - u[0] - 2 * u[1]) * le + (u[0] + 2 * u[1]) * ld + u[1] * ed) / omega
    return flux
