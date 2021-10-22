from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
from numba import jit

@jit(cache=False, nopython=True, fastmath=True)
def ellipse_pos(a, per, tau, Omega, w, i, time):
    """2D x-y Kepler solver without eccentricity and mass"""
#    Q = 2 * arctan(tan((pi * (time - tau) / per))) + (pi * w) / 180
    
    # Scale tau to period
    tau = tau * per

    #print(tau)
    Q = 2 * arctan(tan((pi * (time - tau) / per))) + (pi * w) / 180
    V = sin(Q) * cos((i / 180 * pi))
    O = Omega / 180 * pi
    cos_Omega = cos(O)
    sin_Omega = sin(O)
    cos_Q = cos(Q)
    x = (cos_Omega * cos_Q - sin_Omega * V) * a
    y = (sin_Omega * cos_Q + cos_Omega * V) * a
    return x, y

@jit(cache=True, nopython=True, fastmath=True)
def bary_pos(xm, ym, xp, b_planet, mass_ratio):
    xm_bary = xm + xp + xm * mass_ratio
    ym_bary = ym + b_planet + ym * mass_ratio
    xp_bary = xp - xm * mass_ratio
    py_bary = b_planet - ym * mass_ratio
    return xm_bary, ym_bary, xp_bary, py_bary

"""
# coordinate transformations negligible in time
# 13 million avocados per second
# on 1 core (1 threads) of i7-1185G7
# ==> 52m on 4C 8T

import time as ttime
import numpy as np
xypos(1, 1, 1, 1, 1, 1, 1)
t1 = ttime.time()
t = np.linspace(0, 1, 13_000_000)
for i in range(1):
    a = xypos(1, 1, 1, 1, 1, 1, t)
t2 = ttime.time()
print("Time", t2-t1)

#print(a)
"""