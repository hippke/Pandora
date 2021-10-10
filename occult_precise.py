import numpy as np
from numba import jit

# @jit(nopython=True)
def occult_precise(z, u1, u2, p0, return_components=False):
    """
    #### Mandel-Agol code:
    #   Python translation of IDL code.
    #   This routine computes the lightcurve for occultation of a
    #   quadratically limb-darkened source without microlensing.  Please
    #   cite Mandel & Agol (2002) and Eastman & Agol (2008) if you make use
    #   of this routine in your research.  Please report errors or bugs to
    #   jdeast@astronomy.ohio-state.edu

    """
    z = np.atleast_1d(z)
    nz = np.size(z)
    lambdad = np.zeros(nz)
    etad = np.zeros(nz)
    lambdae = np.zeros(nz)
    omega = 1.0 - u1 / 3.0 - u2 / 6.0

    ## tolerance for double precision equalities
    ## special case integrations
    tol = 1e-14

    p = np.absolute(p0)

    z = np.where(np.absolute(p - z) < tol, p, z)
    z = np.where(np.absolute((p - 1) - z) < tol, p - 1.0, z)
    z = np.where(np.absolute((1 - p) - z) < tol, 1.0 - p, z)
    z = np.where(z < tol, 0.0, z)

    x1 = (p - z) ** 2.0
    x2 = (p + z) ** 2.0
    x3 = p ** 2.0 - z ** 2.0

    def finish(p, z, u1, u2, lambdae, lambdad, etad):
        omega = 1.0 - u1 / 3.0 - u2 / 6.0
        # avoid Lutz-Kelker bias
        if p0 > 0:
            # limb darkened flux
            muo1 = (
                1
                - (
                    (1 - u1 - 2 * u2) * lambdae
                    + (u1 + 2 * u2) * (lambdad + 2.0 / 3 * (p > z))
                    + u2 * etad
                )
                / omega
            )
            # uniform disk
            mu0 = 1 - lambdae
        else:
            # limb darkened flux
            muo1 = (
                1
                + (
                    (1 - u1 - 2 * u2) * lambdae
                    + (u1 + 2 * u2) * (lambdad + 2.0 / 3 * (p > z))
                    + u2 * etad
                )
                / omega
            )
            # uniform disk
            mu0 = 1 + lambdae
        if return_components:
            return muo1, (mu0, lambdad, etad)
        else:
            return muo1

    ## trivial case of no planet
    if p <= 0.0:
        return finish(p, z, u1, u2, lambdae, lambdad, etad)

    ## Case 1 - the star is unocculted:
    ## only consider points with z lt 1+p
    nuy = np.where(z < (1.0 + p))[0]
    if np.size(nuy) == 0:
        return finish(p, z, u1, u2, lambdae, lambdad, etad)

    # Case 11 - the  source is completely occulted:
    if p >= 1.0:
        cond = z[nuy] <= p - 1.0
        occulted = np.where(cond)  # ,complement=nu2)
        nu2 = np.where(~cond)
        # occulted = where(z[nuy] <= p-1.)#,complement=nu2)
        if np.size(occulted) != 0:
            ndxuse = nuy[occulted]
            etad[ndxuse] = 0.5  # corrected typo in paper
            lambdae[ndxuse] = 1.0
            # lambdad = 0 already
            # nu2 = where(z[nuy] > p-1)
            if np.size(nu2) == 0:
                return finish(p, z, u1, u2, lambdae, lambdad, etad)
            nuy = nuy[nu2]

    # Case 2, 7, 8 - ingress/egress (uniform disk only)
    inegressuni = np.where(
        (z[nuy] >= np.absolute(1.0 - p)) & (z[nuy] < 1.0 + p)
    )
    if np.size(inegressuni) != 0:
        ndxuse = nuy[inegressuni]
        tmp = (1.0 - p ** 2.0 + z[ndxuse] ** 2.0) / 2.0 / z[ndxuse]
        tmp = np.where(tmp > 1.0, 1.0, tmp)
        tmp = np.where(tmp < -1.0, -1.0, tmp)
        kap1 = np.arccos(tmp)
        tmp = (p ** 2.0 + z[ndxuse] ** 2 - 1.0) / 2.0 / p / z[ndxuse]
        tmp = np.where(tmp > 1.0, 1.0, tmp)
        tmp = np.where(tmp < -1.0, -1.0, tmp)
        kap0 = np.arccos(tmp)
        tmp = 4.0 * z[ndxuse] ** 2 - (1.0 + z[ndxuse] ** 2 - p ** 2) ** 2
        tmp = np.where(tmp < 0, 0, tmp)
        lambdae[ndxuse] = (p ** 2 * kap0 + kap1 - 0.5 * np.sqrt(tmp)) / np.pi
        # eta_1
        etad[ndxuse] = (
            1.0
            / 2.0
            / np.pi
            * (
                kap1
                + p ** 2 * (p ** 2 + 2.0 * z[ndxuse] ** 2) * kap0
                - (1.0 + 5.0 * p ** 2 + z[ndxuse] ** 2)
                / 4.0
                * np.sqrt((1.0 - x1[ndxuse]) * (x2[ndxuse] - 1.0))
            )
        )

    # Case 5, 6, 7 - the edge of planet lies at origin of star
    cond = z[nuy] == p
    ocltor = np.where(cond)
    notused3 = np.where(~cond)
    t = np.where(z[nuy] == p)
    if np.size(ocltor) != 0:
        ndxuse = nuy[ocltor]
        if p < 0.5:
            # Case 5
            q = 2.0 * p  # corrected typo in paper (2k -> 2p)
            Ek, Kk = ellke(q)
            # lambda_4
            lambdad[ndxuse] = 1.0 / 3.0 + 2.0 / 9.0 / np.pi * (
                4.0 * (2.0 * p ** 2 - 1.0) * Ek + (1.0 - 4.0 * p ** 2) * Kk
            )
            # eta_2
            etad[ndxuse] = p ** 2 / 2.0 * (p ** 2 + 2.0 * z[ndxuse] ** 2)
            lambdae[ndxuse] = p ** 2  # uniform disk
        elif p > 0.5:
            # Case 7
            q = 0.5 / p  # corrected typo in paper (1/2k -> 1/2p)
            Ek, Kk = ellke(q)
            # lambda_3
            lambdad[ndxuse] = (
                1.0 / 3.0
                + 16.0 * p / 9.0 / np.pi * (2.0 * p ** 2 - 1.0) * Ek
                - (32.0 * p ** 4 - 20.0 * p ** 2 + 3.0) / 9.0 / np.pi / p * Kk
            )
        else:
            # Case 6
            lambdad[ndxuse] = 1.0 / 3.0 - 4.0 / np.pi / 9.0
            etad[ndxuse] = 3.0 / 32.0
        # notused3 = where(z[nuy] != p)
        if np.size(notused3) == 0:
            return finish(p, z, u1, u2, lambdae, lambdad, etad)
        nuy = nuy[notused3]

    # Case 2, Case 8 - ingress/egress (with limb darkening)
    cond = (
        (z[nuy] > 0.5 + np.absolute(p - 0.5)) & (z[nuy] < 1.0 + p)
    ) | ((p > 0.5) & (z[nuy] > np.absolute(1.0 - p)) & (z[nuy] < p))
    inegress = np.where(cond)
    nu4 = np.where(~cond)
    if np.size(inegress) != 0:

        ndxuse = nuy[inegress]
        q = np.sqrt((1.0 - x1[ndxuse]) / (x2[ndxuse] - x1[ndxuse]))
        Ek, Kk = ellke(q)
        n = 1.0 / x1[ndxuse] - 1.0

        # lambda_1:
        lambdad[ndxuse] = (
            2.0
            / 9.0
            / np.pi
            / np.sqrt(x2[ndxuse] - x1[ndxuse])
            * (
                (
                    (1.0 - x2[ndxuse]) * (2.0 * x2[ndxuse] + x1[ndxuse] - 3.0)
                    - 3.0 * x3[ndxuse] * (x2[ndxuse] - 2.0)
                )
                * Kk
                + (x2[ndxuse] - x1[ndxuse]) * (z[ndxuse] ** 2 + 7.0 * p ** 2 - 4.0) * Ek
                - 3.0 * x3[ndxuse] / x1[ndxuse] * ellpic_bulirsch(n, q)
            )
        )
        if np.size(nu4) == 0:
            return finish(p, z, u1, u2, lambdae, lambdad, etad)
        nuy = nuy[nu4]

    # Case 3, 4, 9, 10 - planet completely inside star
    if p < 1.0:
        cond = z[nuy] <= (1.0 - p)
        inside = np.where(cond)
        if np.size(inside) != 0:
            ndxuse = nuy[inside]

            ## eta_2
            etad[ndxuse] = p ** 2 / 2.0 * (p ** 2 + 2.0 * z[ndxuse] ** 2)

            ## uniform disk
            lambdae[ndxuse] = p ** 2

            ## Case 4 - edge of planet hits edge of star
            edge = np.where(z[ndxuse] == 1.0 - p)  # , complement=notused6)
            if np.size(edge[0]) != 0:
                ## lambda_5
                lambdad[ndxuse[edge]] = 2.0 / 3.0 / np.pi * np.arccos(
                    1.0 - 2.0 * p
                ) - 4.0 / 9.0 / np.pi * np.sqrt(p * (1.0 - p)) * (
                    3.0 + 2.0 * p - 8.0 * p ** 2
                )
                if p > 0.5:
                    lambdad[ndxuse[edge]] -= 2.0 / 3.0
                notused6 = np.where(z[ndxuse] != 1.0 - p)
                if np.size(notused6) == 0:
                    return finish(p, z, u1, u2, lambdae, lambdad, etad)
                ndxuse = ndxuse[notused6[0]]

            ## Case 10 - origin of planet hits origin of star
            origin = np.where(z[ndxuse] == 0)  # , complement=notused7)
            if np.size(origin) != 0:
                ## lambda_6
                lambdad[ndxuse[origin]] = -2.0 / 3.0 * (1.0 - p ** 2) ** 1.5
                notused7 = np.where(z[ndxuse] != 0)
                if np.size(notused7) == 0:
                    return finish(p, z, u1, u2, lambdae, lambdad, etad)
                ndxuse = ndxuse[notused7[0]]

            q = np.sqrt((x2[ndxuse] - x1[ndxuse]) / (1.0 - x1[ndxuse]))
            n = x2[ndxuse] / x1[ndxuse] - 1.0
            Ek, Kk = ellke(q)

            ## Case 3, Case 9 - anywhere in between
            ## lambda_2
            lambdad[ndxuse] = (
                2.0
                / 9.0
                / np.pi
                / np.sqrt(1.0 - x1[ndxuse])
                * (
                    (1.0 - 5.0 * z[ndxuse] ** 2 + p ** 2 + x3[ndxuse] ** 2) * Kk
                    + (1.0 - x1[ndxuse]) * (z[ndxuse] ** 2 + 7.0 * p ** 2 - 4.0) * Ek
                    - 3.0 * x3[ndxuse] / x1[ndxuse] * ellpic_bulirsch(n, q)
                )
            )

        return finish(p, z, u1, u2, lambdae, lambdad, etad)


# Computes Hasting's polynomial approximation for the complete
# elliptic integral of the first (ek) and second (kk) kind
# @jit(nopython=True)
def ellke(k):
    m1 = 1.0 - k ** 2
    logm1 = np.log(m1)

    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * (-logm1)
    ek = ee1 + ee2

    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012
    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * logm1
    kk = ek1 - ek2

    return [ek, kk]


# Computes the complete elliptical integral of the third kind using
# the algorithm of Bulirsch (1965):
# @jit(nopython=True)
def ellpic_bulirsch(n, k):
    kc = np.sqrt(1.0 - k ** 2)
    p = n + 1.0
    m0 = 1.0
    c = 1.0
    p = np.sqrt(p)
    d = 1.0 / p
    e = kc
    while 1:
        f = c
        c = d / p + c
        g = e / p
        d = 2.0 * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0
        if (np.absolute(1.0 - kc / g)).max() > 1.0e-8:
            kc = 2 * np.sqrt(e)
            e = kc * m0
        else:
            return 0.5 * np.pi * (c * m0 + d) / (m0 * (m0 + p))

"""
import time

t1 = time.time()
for i in range(1000):
    occultquad(z=0.1, u1=0.1, u2=0.1, p0=0.1, return_components=False)
t2 = time.time()
print(t2 - t1)


series = np.linspace(0, 1, 1000)
t1 = time.time()
result = occultquad(z=series, u1=0.1, u2=0.1, p0=0.1, return_components=False)
t2 = time.time()
print(t2 - t1)
# print(result)
# assert result == 0.989481713365983
print(result[0])
"""