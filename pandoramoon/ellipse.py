import numpy as np
from numpy import sqrt, pi, sin, cos, abs, tan, arctan
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ellipse(
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


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
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