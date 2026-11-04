"""Phase-at-transit reparametrisation of the moon orbit (experimental; derivation and tests:
REPARAMETRIZATION.md in the project directory).

New moon coordinates (phi_c, delta, u_m) instead of (per_moon, tau):
  N        = m + delta,  m = M(u_m) (inverse CDF of q(m) ~ 1/m - 1/(m+1), m = m_min..m_max)
  per_moon = per_bary / N
  tau      = (t_c / per_moon - phi_c) mod 1,  t_c = t0_bary + t0_bary_offset + e_c per_bary
with the exact importance weight (Sect. 5.3) that restores the user's prior
per_moon ~ U(0, p_max], tau ~ U[0, 1):
  w = (per_bary / p_max) * Q * m (m + 1) / N^2 * [per_bary / N <= p_max]

The integer range follows the user's priors (Sect. 5.4):
  m_min = max(1, floor(pb_min / p_max))     pb_min = lower bound of the planet-period prior
  m_max = ceil(pb_max / p_lo)               p_lo   = a period below which every moon in the
                                                     prior box is unphysical (a_m < r_p + r_m)
Planet periods are usually known precisely (a narrow per_bary prior), so the range is set by
pb / p_max and pb / p_lo.

    rp = Reparam(pb_min=199.99, pb_max=200.01, p_max=50.0, p_lo=0.05)
    per_moon, tau, logw = rp.to_native(per_bary, t0_bary, t0_off, phi_c, delta, u_m, e_c)   # numba
    rp.to_native_vec(...), rp.from_native_vec(...), rp.new_to_native_samples(...)            # numpy

Module-level names (to_native, M_MIN, CDF, ...) are the notebook's default instance
(P_b prior 360.25-370.25 d, p_max 50 d, p_lo 0.05 d), kept for the existing scripts.
"""
import math

import numpy as np
from numba import njit

LOGZERO = -1e300      # log L' outside the original prior (P_m > p_max); finite for UltraNest, exp() = 0
from pandoramoon.constants import G, DAY  # noqa: E402  (physics.cfg)


def min_physical_period(R_star_max, rsum_min, M_total_max):
    """shortest moon period [d] that is physical somewhere in the prior box (Pandora:
    a_m >= r_p + r_m in R_star units): a = rsum_min * R_star_max [m], M = M_total_max [kg]"""
    a = rsum_min * R_star_max
    return 2 * math.pi * math.sqrt(a ** 3 / (G * M_total_max)) / DAY


def fold_native(per_moon, tau, Omega):
    """canonical native representative of the double cover: Omega in [0, 180),
    (Omega + 180, tau - 1/2) -> (Omega, tau) (phase + 1/2 <=> tau - 1/2)"""
    Om = np.mod(Omega, 360.0)
    hi = Om >= 180.0
    tau2 = np.where(hi, np.mod(tau - 0.5, 1.0), tau)
    return per_moon, tau2, np.where(hi, Om - 180.0, Om)


class Reparam:
    def __init__(self, pb_min, pb_max, p_max=50.0, p_lo=0.05, e_c=1):
        assert 0 < pb_min <= pb_max and 0 < p_lo < p_max
        self.pb_min, self.pb_max, self.p_max, self.p_lo, self.e_c = (float(pb_min), float(pb_max), float(p_max),
                                                                      float(p_lo), int(e_c))
        # m >= 1: a moon period >= the planet period is far outside the Hill sphere anyway
        M_MIN = max(1, int(math.floor(pb_min / p_max)))
        M_MAX = int(math.ceil(pb_max / p_lo))
        ms = np.arange(M_MIN, M_MAX + 1, dtype=np.float64)
        qm = 1.0 / ms - 1.0 / (ms + 1.0)
        Q = float(qm.sum())                       # = 1/M_MIN - 1/(M_MAX+1)
        CDF = np.cumsum(qm) / Q
        CDF[-1] = 1.0
        P_MAX = float(p_max)
        self.M_MIN, self.M_MAX, self.Q, self.CDF, self.P_MAX, self.P_LO = M_MIN, M_MAX, Q, CDF, P_MAX, float(p_lo)

        @njit(cache=False)
        def m_of_u(u):
            """m = smallest m with CDF(m) > u (monotone step map)"""
            lo, hi = 0, CDF.shape[0] - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if CDF[mid] > u:
                    hi = mid
                else:
                    lo = mid + 1
            return M_MIN + lo

        @njit(cache=False)
        def to_native(per_bary, t0_bary, t0_off, phi_c, delta, u_m, e_c):
            """-> (per_moon, tau, log_w); log_w = LOGZERO outside the original prior"""
            m = m_of_u(u_m)
            N = m + delta
            per_moon = per_bary / N
            t_c = t0_bary + t0_off + e_c * per_bary
            x = t_c * N / per_bary - phi_c
            tau = x - math.floor(x)
            if per_moon > P_MAX:
                return per_moon, tau, LOGZERO
            logw = math.log(per_bary / P_MAX * Q * m * (m + 1.0) / (N * N))
            return per_moon, tau, logw

        self.m_of_u = m_of_u
        self.to_native = to_native

    def m_of_u_vec(self, u_m):
        return np.minimum(self.M_MIN + np.searchsorted(self.CDF, u_m, side="right"), self.M_MAX)

    def to_native_vec(self, per_bary, t0_bary, t0_off, phi_c, delta, u_m, e_c=None):
        e_c = self.e_c if e_c is None else e_c
        m = self.m_of_u_vec(u_m)
        N = m + delta
        per_moon = per_bary / N
        t_c = t0_bary + t0_off + e_c * per_bary
        x = t_c * N / per_bary - phi_c
        tau = x - np.floor(x)
        with np.errstate(divide="ignore"):
            logw = np.log(per_bary / self.P_MAX * self.Q * m * (m + 1.0) / (N * N))
        logw = np.where(per_moon > self.P_MAX, LOGZERO, logw)
        return per_moon, tau, logw, m

    def from_native_vec(self, per_bary, t0_bary, t0_off, per_moon, tau, e_c=None):
        """inverse map -> (phi_c, delta, m, u_m interval [CDF(m-1), CDF(m)))"""
        e_c = self.e_c if e_c is None else e_c
        N = per_bary / per_moon
        m = np.floor(N).astype(np.int64)
        delta = N - m
        t_c = t0_bary + t0_off + e_c * per_bary
        x = t_c / per_moon - tau
        phi_c = x - np.floor(x)
        k = m - self.M_MIN
        ulo = np.where(k > 0, self.CDF[np.clip(k - 1, 0, len(self.CDF) - 1)], 0.0)
        uhi = self.CDF[np.clip(k, 0, len(self.CDF) - 1)]
        return phi_c, delta, m, ulo, uhi

    def new_to_native_samples(self, samples, t0_bary=11.0, e_c=None):
        """(n, 16) moon-phase samples -> (n, 15) native samples (folded), names as Problem('moon')"""
        s = np.asarray(samples)
        pm, tau, logw, m = self.to_native_vec(s[:, 1], t0_bary, s[:, 5], s[:, 8], s[:, 9], s[:, 10], e_c)
        pm, tau, Om = fold_native(pm, tau, s[:, 11])
        q = np.empty((len(s), 15))
        q[:, :8] = s[:, :8]
        q[:, 8] = pm
        q[:, 9] = tau
        q[:, 10] = Om
        q[:, 11:] = s[:, 12:]
        return q

    def __repr__(self):
        return (f"Reparam(pb {self.pb_min}..{self.pb_max} d, p_max {self.P_MAX} d, p_lo {self.P_LO} d: "
                f"m {self.M_MIN}..{self.M_MAX})")
