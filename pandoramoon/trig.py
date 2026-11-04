"""Vectorisable sine and cosine of a phase given in turns.

numba cannot vectorise loops that call libm sin/cos (no SVML in the PyPI
llvmlite), and scalar libm sin + cos cost ~8 ns per sample. sincos_turns()
is pure arithmetic (floor, multiply, add, select), so LLVM auto-vectorises
loops that call it (AVX2 / NEON) under fastmath.

The phase is reduced in turns before converting to radians (P24c):
    q4 = round(4 u)           quarter turns (exact for |u| < 2^50)
    d = u - q4/4              in [-1/8, 1/8]   (exact: Sterbenz)
    r = 2 pi d                in [-pi/4, pi/4] (one rounding)
then fdlibm/Cephes minimax polynomials on [-pi/4, pi/4] and a quadrant
select on q4 mod 4. The error of sin(2 pi u), cos(2 pi u) is <= 2e-16 for
|u| < 2^50 and independent of |u| (libm on 2 pi u in radians inherits the rounding of
2 pi u, i.e. the argument's ulp, which grows with |u|).
NaN or inf input gives NaN.

sincos_small(phi) (P24d) is sin / cos of a small angle |phi| <= 0.3 rad
(two short polynomials, no reduction): the orbit loops on the runs take
sincos_turns() once per block anchor and sincos_small() of the angle
from the anchor per sample.

acos_poly(x) (P16) is a branch-free arccos for |x| <= 1 (fdlibm's e_acos
with selects instead of branches), so loops that call it vectorise too;
<= 1 ulp from libm, ~0.9 instead of ~8.4 ns per element.
"""
import numpy as np
from numba import jit

TWO_PI = 6.283185307179586

# fdlibm __kernel_sin / __kernel_cos coefficients on [-pi/4, pi/4]
S1 = -1.66666666666666324348e-01
S2 = 8.33333333332248946124e-03
S3 = -1.98412698298579493134e-04
S4 = 2.75573137070700676789e-06
S5 = -2.50507602534068634195e-08
S6 = 1.58969099521155010221e-10
C1 = 4.16666666666666019037e-02
C2 = -1.38888888888741095749e-03
C3 = 2.48015872894767294178e-05
C4 = -2.75573143513906633035e-07
C5 = 2.08757232129817482790e-09
C6 = -1.13596475577881948265e-11


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def sincos_turns(u):
    """(sin(2 pi u), cos(2 pi u)) for a phase u in turns, |u| < 2^50."""
    # P24c: the quarter-turn count directly from u, q4 = round(4 u) (4 u and
    # + 0.5 exact for |u| < 2^50), d = u - q4/4 in [-1/8, 1/8] exact
    # (Sterbenz): one floor less on the dependency chain than reducing
    # f = u - round(u) first (the orbit loop is latency-bound), and the
    # quadrant select runs in parallel with the polynomials
    q4 = np.floor(4.0 * u + 0.5)
    r = TWO_PI * (u - 0.25 * q4)
    z = r * r
    s = r + r * z * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))))
    c = 1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))))
    # quadrant k = q4 mod 4 in {0, 1, 2, 3} (exact), selected in float lanes:
    # no float -> int conversion, which AVX2 has no vector instruction for
    # (P24a: it cost 4 scalar vcvttsd2si per vector)
    k = q4 - 4.0 * np.floor(0.25 * q4)
    odd = (k == 1.0) | (k == 3.0)
    s2 = c if odd else s
    c2 = s if odd else c
    c2 = -c2 if (k == 1.0) | (k == 2.0) else c2
    s2 = -s2 if k >= 2.0 else s2
    return s2, c2


# P24d: sin / cos of a small angle |phi| <= SMALL_PHI_MAX [rad]; near-minimax
# odd / even polynomials (Lawson-weighted fit in mpmath, scratch_p24d/coef.py):
# sin = phi + phi z S(z), cos = 1 + z C(z), z = phi^2; fit error 6e-17 /
# 6e-19, in double <= 1.1e-16 (the rounding of 1 + ...)
SMALL_PHI_MAX = 0.3
SS1 = -0.16666666666648974
SS2 = 0.008333333317368548
SS3 = -0.00019841222081918382
SS4 = 2.749903241858877e-06
SC1 = -0.4999999999999997
SC2 = 0.04166666666661884
SC3 = -0.0013888888864428722
SC4 = 2.480153269904455e-05
SC5 = -2.75019806552096e-07


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def sincos_small(ph):
    """(sin ph, cos ph) for |ph| <= SMALL_PHI_MAX = 0.3 rad, error <= 1.2e-16
    (P24d). No reduction and no quadrant select: about half the work of
    sincos_turns(); the orbit loop uses it for the angle from a block
    anchor. Outside the range the error grows quickly (ph^11 / 11!)."""
    z = ph * ph
    s = ph + ph * z * (SS1 + z * (SS2 + z * (SS3 + z * SS4)))
    c = 1.0 + z * (SC1 + z * (SC2 + z * (SC3 + z * (SC4 + z * SC5))))
    return s, c


# fdlibm e_asin / e_acos rational kernel R(z) ~ (asin(sqrt z) / sqrt z - 1) / z
PS0 = 1.66666666666666657415e-01
PS1 = -3.25565818622400915405e-01
PS2 = 2.01212532134862925881e-01
PS3 = -4.00555345006794114027e-02
PS4 = 7.91534994289814532176e-04
PS5 = 3.47933107596021167570e-05
QS1 = -2.40339491173441421878e+00
QS2 = 2.02094576023350569471e+00
QS3 = -6.88283971605453293030e-01
QS4 = 7.70381505559019352791e-02
HALF_PI = 1.5707963267948966
PI = 3.141592653589793


@jit(cache=True, nopython=True, fastmath=True, parallel=False, error_model="numpy")
def acos_poly(x):
    """arccos(x) for -1 <= x <= 1, branch-free (P16).

    fdlibm: |x| <= 1/2: pi/2 - (x + x R(x^2)); |x| > 1/2: with
    z = (1 - |x|)/2 (exact), s = sqrt(z): acos(|x|) = 2 (s + s R(z)) and
    acos(-|x|) = pi - acos(|x|). Both branches are evaluated and selected,
    so LLVM vectorises loops that call it. <= 1 ulp from libm on [-1, 1],
    incl. +-1, +-1/2, 0 (test_acos_poly). The caller must clamp the
    argument to [-1, 1]; outside it the result is meaningless (NaN or
    garbage under fastmath)."""
    ax = abs(x)
    small = ax <= 0.5
    z = x * x if small else 0.5 * (1.0 - ax)
    s = np.sqrt(z)
    p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))))
    q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)))
    r = p / q
    a_small = HALF_PI - (x + x * r)
    t = 2.0 * (s + s * r)
    a_big = t if x > 0 else PI - t
    return a_small if small else a_big
