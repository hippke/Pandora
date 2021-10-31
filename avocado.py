import numpy as np
from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin, floor
from numba import jit, prange





@jit(cache=False, nopython=True, fastmath=True)
def arctan_cordic(y, l1, l2, x=1.0):
    r = 0.0
    for idx in range(len(l1)):
        if y < 0:
            r, x, y = r - l2[idx], x - l1[idx]*y, y + l1[idx]*x
        else:
            r, x, y = r + l2[idx], x + l1[idx]*y, y - l1[idx]*x
    return r


@jit(cache=False, nopython=True, fastmath=True)
def ellipse_pos(a, per, tau, Omega, i, time):
    """2D x-y Kepler solver WITHOUT eccentricity, WITHOUT mass"""
    
    # Scale tau to period
    # tau_moon is the position of the moon on its orbit, given as [0..1]
    # for the first timestamp of the first epoch
    # Cannot be given in units of days in prior, because moon orbit period varies
    # Example: Prior has tau in [5, 100] but model tests orbit with per_moon = 10
    #          It would physically still be OK, because it is circular and wraps
    #          around. However, the sampler would not converge when testing models.
    # So, we use tau in [0..1] and propagate to following epochs manually

    Q = 2 * arctan(tan((pi * (time - tau * per) / per)))
    V = sin(Q) * cos((i / 180 * pi))
    O = Omega / 180 * pi
    cos_Omega = cos(O)
    sin_Omega = sin(O)
    cos_Q = cos(Q)
    x = (cos_Omega * cos_Q - sin_Omega * V) * a
    y = (sin_Omega * cos_Q + cos_Omega * V) * a
    return x, y


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def ellipse_pos_iter(a, per, tau, Omega, i, time, transit_threshold_x, xp):
    """2D x-y Kepler solver WITHOUT eccentricity, WITHOUT mass"""
    
    # Scale tau to period
    # tau_moon is the position of the moon on its orbit, given as [0..1]
    # for the first timestamp of the first epoch
    # Cannot be given in units of days in prior, because moon orbit period varies
    # Example: Prior has tau in [5, 100] but model tests orbit with per_moon = 10
    #          It would physically still be OK, because it is circular and wraps
    #          around. However, the sampler would not converge when testing models.
    # So, we use tau in [0..1] and propagate to following epochs manually
    tau_per = tau * per

    """
    cordic_table_size = 60
    l1 = np.empty(cordic_table_size)
    l2 = l1.copy()
    for idx in range(cordic_table_size):
        exp = np.exp2(-idx)
        l1[idx] = exp
        l2[idx] = np.arctan(exp)
    """
    O = Omega / 180 * pi
    cos_Omega = cos(O)
    sin_Omega = sin(O)
    k = cos(i / 180 * pi)
    x = np.zeros(len(time))
    y = x.copy()
    for idx in prange(len(time)):
        if abs(xp[idx]) > transit_threshold_x:
            x[idx] = np.inf
        else:
            Q = 2 * arctan(tan((pi * (time[idx] - tau_per) / per)))
            #Q = 2 * arctan_cordic(y=val, l1=l1, l2=l2)  # slower
            x[idx] = (cos_Omega * cos(Q) - sin_Omega * sin(Q) * k) * a
            y[idx] = (sin_Omega * cos(Q) + cos_Omega * sin(Q) * k) * a
    return x, y


#@jit(cache=False, nopython=True, fastmath=True)
def ellipse_pos_eccentricity(a, per, e, tau, Omega, w, i, time):
    """2D x-y Kepler solver WITH eccentricity, WITHOUT mass"""

    # Scale tau to period
    tau = tau * (per)

    M = (2 * pi / per) * (time - tau)
    flip = False
    M = M - (floor(M / (2 * pi)) * 2 * pi)
    if M > pi:
        M = 2 * pi - M
        flip = True
    alpha = (3 * pi**2 + 1.6 * pi * (pi - abs(M)) / (1 + e)) / (pi**2 - 6)
    d = 3 * (1 - e) + alpha * e
    r1 = 3 * alpha * d * (d - 1 + e) * M + M**3
    q = 2 * alpha * d * (1 - e) - M**2
    w1 = (abs(r1) + sqrt(q**3 + r1**2))**(2 / 3)
    n = (2 * r1 * w1 / (w1**2 + w1 * q + q**2) + M) / d
    f0 = n - e * sin(n) - M
    f1 = 1 - e * cos(n)
    f2 = e * sin(n)
    g = -f0 / (f1 - 0.5 * f0 * f2 / f1)
    h = -f0 / (f1 + 0.5 * g * f2 + (g**2) * (1 - f1) / 6)
    k = n -f0 / (f1 + 0.5 * h * f2 + h**2 * (1 - f1) / 6 + h**3 * (-f2) / 24)
    if flip:
        k = 2 * pi - k
    r = a * (1 - e * cos(k))
    wf = (w / 180 * pi) + (arctan(sqrt((1 + e) / (1 - e)) * tan(k / 2)) * 2)
    v1 = sin(wf) * cos((i / 180 * pi))
    x = (cos((Omega / 180 * pi)) * cos(wf) - sin((Omega / 180 * pi)) * v1) * r
    y = (sin((Omega / 180 * pi)) * cos(wf) + cos((Omega / 180 * pi)) * v1) * r 
    #z = (sin(wf) * sin((i / 180 * pi))) * r
    return x, y


"""
val1, val2 = ellipse_pos_eccentricity(a=1, per=2, e=0.123, tau=0.534, Omega=54.2, w=12.12, i=55.4, time=4.5)
assert val1 == 0.32562991217909504
assert val2 == -0.5306948667069533
print(val1, val2)
"""

