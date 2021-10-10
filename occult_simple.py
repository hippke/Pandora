import numpy as np
from numpy import sqrt, pi, arcsin, arccos, sum
import matplotlib.pyplot as plt


def eval_int_at_limit(limit, ld):
    total = (
        1
        - ld[0] * (1 - 0.8 * np.sqrt(limit))
        - ld[1] * (1 - (2 / 3) * limit)
        - ld[2] * (1 - (4 / 7) * limit * np.sqrt(limit))
        - ld[3] * (1 - 0.5 * limit ** 2)
    )
    return -(limit ** 2) * total


def integral(p, ld, lower, upper):
    lower = np.array(lower, copy=True)
    upper = np.array(upper, copy=True)
    return eval_int_at_limit(upper, ld) - eval_int_at_limit(lower, ld)


def occult_simple(z, p, ld):
    """Nonlinear limb-darkening light curve in the small-planet
    approximation (section 5 of Mandel & Agol 2002).
    :INPUTS:
        z -- sequence of positional offset values
        p -- planet/star radius ratio
        ld -- four-sequence nonlinear limb darkening coefficients
    """
    ld = np.array([ld]).ravel()
    z = np.array(z)
    F = np.ones(z.shape, float)
    z[z == 0] = 1e-10
    a = (z - p) ** 2
    b = (z + p) ** 2
    Omega = 0.25 * (1 - sum(ld)) + sum(ld / np.arange(5, 9))
    intrans = ((1 - p) < z) * ((1 + p) > z)
    pretrans = z <= (1 - p)
    aind1 = 1 - a[intrans]
    zind1m1 = z[intrans] - 1
    edge = integral(p, ld, sqrt(aind1), np.array([0.0])) / aind1
    inside = integral(p, ld, sqrt(1 - a[pretrans]), sqrt(1 - b[pretrans])) / z[pretrans]
    F[intrans] = 1 - (0.25 * edge / (pi * Omega)) * (
        (p * p * arccos((zind1m1) / p))
        - ((zind1m1) * sqrt(p * p - (zind1m1 * zind1m1)))
    )
    F[pretrans] = 1 - 0.0625 * p * inside / Omega
    return F





import time
zgrid = np.linspace(0, 1, 1000)
p = 0.1#1800/690000
ld = (0.1, 0.1, 0.1, 0.1)
for z in zgrid:
    flux = occult_flux(z, p, ld)
    print(flux)


