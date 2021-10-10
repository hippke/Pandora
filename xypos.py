from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin

def xypos_moon(a_moon, per_moon, tau_moon, Omega_moon, w_moon, i_moon, time):
    """2D x-y Kepler solver without eccentricity"""
    Q = 2 * arctan(tan((pi * (time - tau_moon) / per_moon))) + (pi * w_moon) / 180
    V = sin(Q) * cos((i_moon / 180 * pi))
    O = Omega_moon / 180 * pi
    cos_Omega = cos(O)
    sin_Omega = sin(O)
    cos_Q = cos(Q)
    x = (cos_Omega * cos_Q - sin_Omega * V) * a_moon
    y = (sin_Omega * cos_Q + cos_Omega * V) * a_moon
    return x, y
