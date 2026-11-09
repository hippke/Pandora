"""Simultaneous GP detrending (opt-in): ``Likelihood(time, data, yerr, ..., gp=GP(amp, tau, err_scale))``.

The light curve is not detrended first. Its noise is a Gaussian process plus white noise, fitted together with the
transit model:

    data = model + g + white,   g ~ GP(0, k),   white ~ N(0, (err_scale yerr)^2)

with the kernel of a stochastically driven, damped harmonic oscillator with Q = 1 / sqrt(2) (SHO; Foreman-Mackey et
al. 2017; the kernel of pizzetti's GP):

    k(dt) = amp^2 exp(-x) (cos x + sin x),   x = |dt| / (sqrt(2) tau)

(smooth, not periodic; the correlation falls to 1/2 at about 1.4 tau). The hyperparameters amp, tau and err_scale are
FIXED during the fit (estimate them first, e.g. with ``GP.fit`` on the data outside the transit windows).

The GP acts on the output cadences (after supersampling). With Sigma = K + diag((s yerr)^2) = L L^T, the residual
r = data - f = (data - 1) + D (deficit D = 1 - f, nonzero only near the transits) gives

    chi^2 = r^T Sigma^-1 r = |L^-1 (data - 1)|^2 + sum_k (Dw_k^2 + 2 e_k Dw_k),   Dw = L^-1 D,  e = L^-1 (data - 1)

The SHO is a two-state linear stochastic differential equation, so L^-1 is its Kalman filter: with fixed
hyperparameters the gains, the innovation variances and e are computed once (``gp_setup``, float64), and each
likelihood call only filters D (a two-number state recursion from the first to the last cadence of the model's
window; the remaining cadences, where D = 0, are one closed-form term: ``gp_quad``). The log-likelihood is

    log L = -chi^2 / 2 - (sum_k ln L_kk - sum_k ln yerr_k)

i.e. relative to the normalisation of the white-noise Likelihood: amp -> 0 with err_scale = 1 gives the white-noise
value. The same GP enters the planet-only model (Bayes factors compare the models under the same noise model).
"""
import math

import numpy as np
from numba import jit

from pandoramoon.kernel import (
    _window_model_runs, _window_cadences, x_bary_runs_sorted, occult_hybrid_z2, params_valid, PLANET_ONLY_MOON,
    NEG_INF,
)

_SQRT2 = math.sqrt(2.0)


class GP(object):
    """Fixed hyperparameters of the simultaneous GP (see the module docstring).

    :param amp: rms of the GP, in relative flux (e.g. 3e-4 for 300 ppm)
    :param tau: time scale 1 / omega_0 in days
    :param err_scale: factor on yerr (the white noise is err_scale * yerr)
    """

    kernel = "sho"

    def __init__(self, amp, tau, err_scale=1.0):
        amp, tau, err_scale = float(amp), float(tau), float(err_scale)
        if not (amp > 0 and tau > 0 and err_scale > 0 and math.isfinite(amp + tau + err_scale)):
            raise ValueError(f"GP: amp, tau and err_scale must be positive and finite, got {amp}, {tau}, {err_scale}")
        self.amp, self.tau, self.err_scale = amp, tau, err_scale

    def __repr__(self):
        return f"GP(amp={self.amp:.6g}, tau={self.tau:.6g}, err_scale={self.err_scale:.6g})"

    def key(self):
        """the hyperparameters as a tuple (data hashes, resume)"""
        return (self.kernel, self.amp, self.tau, self.err_scale)

    def loglike_noise(self, time, flux, yerr, mask=None):
        """log likelihood of the light curve under the GP alone (model flux 1 everywhere), relative to the white-noise
        normalisation as in Likelihood; points with mask True (or non-finite flux / yerr) are left out"""
        t, y1, yerr_, valid = _prepare(time, flux, yerr, mask)
        var = (self.err_scale * yerr_) ** 2
        tab = gp_setup(t, y1, var, valid, yerr_, self.amp, self.tau)
        return -0.5 * tab[6] - tab[7]

    @classmethod
    def fit(cls, time, flux, yerr, mask=None, err_scale=None, amp_bounds=None, tau_bounds=None, bin_width=10.0 / 1440,
            n_starts=3):
        """Maximum-likelihood (type II) hyperparameters of the GP alone on the light curve (model flux 1), e.g. with
        the transit windows masked (mask True = left out). err_scale: a number to hold it fixed, or None to fit it
        (bounds 1/3 - 3). Bounds (defaults): amp from 0.02 to 50 x the median yerr, tau from the median time step to
        the time span. bin_width [d]: the fit uses data binned to this width if the median time step is shorter than
        half of it (None: never bin); as TLS's gp fit (fit_bin_width, 10 min).
        As TLS 2.2's gradient fit of its baselines: the exact gradient of the Kalman likelihood (here forward mode in
        (ln amp, ln tau, ln err_scale), about 4 likelihood evaluations), a bounded quasi-Newton (projected BFGS,
        _minimize_bounded), transitions once per distinct time step (time_table), from n_starts values of tau.
        Returns a GP with fit_nll (the minimum), fit_nfev and fit_points (after binning)."""
        t, y1, yerr_, valid = _prepare(time, flux, yerr, mask)
        t, y1, yerr_ = t[valid], y1[valid], yerr_[valid]
        if len(t) < 10:
            raise ValueError("GP.fit: fewer than 10 usable points")
        dt_med = float(np.median(np.diff(t)))
        if bin_width is not None and dt_med < 0.5 * bin_width:
            t, y1, yerr_ = _bin(t, y1, yerr_, float(bin_width))
            dt_med = float(np.median(np.diff(t)))
        ye = float(np.median(yerr_))
        span = float(t[-1] - t[0])
        alo, ahi = amp_bounds or (0.02 * ye, 50.0 * ye)
        tlo, thi = tau_bounds or (dt_med, span)
        fit_s = err_scale is None
        lo = np.array([math.log(alo), math.log(tlo)] + ([math.log(1.0 / 3.0)] if fit_s else []))
        hi = np.array([math.log(ahi), math.log(thi)] + ([math.log(3.0)] if fit_s else []))
        idx, steps = time_table(t, 1e-6)
        var0 = np.ascontiguousarray(yerr_ * yerr_)
        s_fix = 1.0 if fit_s else float(err_scale)
        nfev = [0]

        def fg(z):
            nfev[0] += 1
            s = math.exp(z[2]) if fit_s else s_fix
            f, g = _nll_grad(idx, steps, y1, var0 * s * s, yerr_, math.exp(z[0]), math.exp(z[1]))
            return f, (g if fit_s else g[:2])

        rms = float(np.std(y1))
        a0 = math.log(min(max(math.sqrt(max(rms * rms - ye * ye, (0.1 * ye) ** 2)), alo * 1.01), ahi / 1.01))
        best = None
        for j in range(n_starts):
            t0 = math.log(min(max(dt_med * 10.0 ** (1 + j), tlo * 1.01), thi / 1.01))
            z0 = np.array([a0, t0] + ([0.0] if fit_s else []))
            z, f = _minimize_bounded(fg, z0, lo, hi)
            if best is None or f < best[1]:
                best = (z, f)
        z = best[0]
        g = cls(math.exp(z[0]), math.exp(z[1]), math.exp(z[2]) if fit_s else s_fix)
        g.fit_nll, g.fit_nfev, g.fit_points = float(best[1]), int(nfev[0]), int(len(t))
        return g

    def predict(self, time, residual, yerr, mask=None):
        """posterior mean and sd of the GP at the given (ascending) times, given residual = data - model there
        (Kalman filter + Rauch-Tung-Striebel smoother; mask True = left out): the trend to subtract for plots"""
        t, r, yerr_, valid = _prepare(time, np.asarray(residual, dtype=np.float64) + 1.0, yerr, mask)
        mean, var = _rts(t, r, (self.err_scale * yerr_) ** 2, valid, self.amp, self.tau)
        return mean, np.sqrt(np.maximum(var, 0.0))


def _prepare(time, flux, yerr, mask):
    t = np.ascontiguousarray(time, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    e = np.broadcast_to(np.asarray(yerr, dtype=np.float64), y.shape)
    if t.shape != y.shape:
        raise ValueError("time and flux must have the same length")
    if len(t) > 1 and not np.all(np.diff(t) >= 0):
        raise ValueError("the GP needs ascending times")
    valid = np.isfinite(y) & np.isfinite(e) & (e > 0)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)
    y1 = np.ascontiguousarray(np.where(valid, y - 1.0, 0.0))
    e1 = np.ascontiguousarray(np.where(valid, e, 1.0))
    return t, y1, e1, np.ascontiguousarray(valid)


def _bin(t, y1, yerr, width):
    """bins of the given width [d] (on the time axis from t[0]): mean time, mean flux, error / sqrt(n)"""
    k = np.floor((t - t[0]) / width).astype(np.int64)
    u, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, weights=t) / cnt
    yb = np.bincount(inv, weights=y1) / cnt
    eb = np.sqrt(np.bincount(inv, weights=yerr * yerr)) / cnt
    return np.ascontiguousarray(tb), np.ascontiguousarray(yb), np.ascontiguousarray(eb)


def _minimize_bounded(fg, x0, lower, upper, maxiter=500, gtol=1e-4, ftol=2.2e-9, max_ls=30):
    """Bounded quasi-Newton minimisation (projected BFGS) of f within lower <= x <= upper, fg(x) -> (f, gradient);
    from TLS 2.2 (transitleastsquares/gp/quasi_newton.py, MIT licence, M. Hippke): the BFGS direction in the free
    variables, a projected backtracking Armijo line search, the inverse Hessian updated off the bounds. Returns
    (x, f)."""
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    x = np.clip(np.asarray(x0, float), lo, hi)
    n = len(x)
    f, g = fg(x)
    if not np.isfinite(f):
        return x, f
    H = np.eye(n)
    scaled = False
    for _ in range(maxiter):
        free = ~(((x <= lo) & (g > 0)) | ((x >= hi) & (g < 0)))
        if not free.any() or np.max(np.abs(g[free])) <= gtol:
            break
        d = np.zeros(n)
        d[free] = -H[np.ix_(free, free)] @ g[free]
        if g @ d >= 0:
            H = np.eye(n)
            scaled = False
            d[free] = -g[free]
        step = 1.0 if scaled else min(1.0, 1.0 / np.linalg.norm(d))
        for _ls in range(max_ls):
            xn = np.clip(x + step * d, lo, hi)
            fn, gn = fg(xn)
            slope = g @ (xn - x)
            if np.isfinite(fn) and fn <= f + 1e-4 * slope:
                break
            if np.isfinite(fn) and fn - f - slope > 0:
                step = min(max(-0.5 * slope * step / (fn - f - slope), 0.1 * step), 0.5 * step)
            else:
                step *= 0.1
        else:
            break
        s, yv = xn - x, gn - g
        held = ~free | (xn <= lo) | (xn >= hi)
        s, yv = np.where(held, 0.0, s), np.where(held, 0.0, yv)
        f_old = f
        x, f, g = xn, fn, gn
        if f_old - f <= ftol * max(abs(f_old), abs(f), 1.0):
            break
        sy = s @ yv
        if np.any(s) and sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(yv):
            if not scaled:
                H = (sy / (yv @ yv)) * np.eye(n)
                scaled = True
            rho = 1.0 / sy
            V = np.eye(n) - rho * np.outer(s, yv)
            H = V @ H @ V.T + rho * np.outer(s, s)
    return x, float(f)


@jit(cache=True, nopython=True)
def time_table(t, tol):
    """(entry of each time step, distinct steps): steps within a relative tol of a group's smallest step share its
    mean (pizzetti's _gp.time_table); idx[i] is the step from t[i - 1] to t[i], idx[0] unused"""
    n = t.shape[0]
    dt = np.diff(t)
    o = np.argsort(dt)
    idx = np.zeros(n, dtype=np.int64)
    steps = np.empty(max(n - 1, 1))
    u = -1
    ref = 0.0
    acc = 0.0
    cnt = 0
    for r in range(n - 1):
        v = dt[o[r]]
        if u < 0 or v > ref * (1.0 + tol):
            if u >= 0:
                steps[u] = acc / cnt
            u += 1
            ref, acc, cnt = v, 0.0, 0
        acc += v
        cnt += 1
        idx[o[r] + 1] = u
    if u >= 0:
        steps[u] = acc / cnt
    return idx, steps[:u + 1]


@jit(cache=True, nopython=True)
def _sho_step_dx(x):
    """d/dx of _sho_step's transition (f00, f01, f10, f11) and of its process noise per unit variance (q00, q01, q11)"""
    e = math.exp(-x)
    c, sn = math.cos(x), math.sin(x)
    e2 = e * e
    r2 = 1.4142135623730951
    h0 = 2.0 * sn * (c + sn)
    dh0 = 2.0 * (c * c - sn * sn) + 4.0 * sn * c
    h1 = 2.0 * sn * (sn - c)
    dh1 = 4.0 * sn * c - 2.0 * (c * c - sn * sn)
    return (-2.0 * e * sn, r2 * e * (c - sn), -r2 * e * (c - sn), -2.0 * e * c,
            2.0 * e2 + 2.0 * e2 * h0 - e2 * dh0, 4.0 * r2 * e2 * sn * (c - sn), 2.0 * e2 + 2.0 * e2 * h1 - e2 * dh1)


@jit(cache=True, nopython=True)
def _nll_grad(idx, steps, y1, var, yerr, amp, tau):
    """negative log likelihood of y1 (= data - 1, all points used) under the GP, relative to the white-noise
    normalisation (0.5 |L^-1 y1|^2 + sum ln L_kk - sum ln yerr), and its exact gradient in (ln amp, ln tau, ln s)
    (forward mode: the state and covariance derivatives are propagated with the filter; var = (s yerr)^2)"""
    n = y1.shape[0]
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    nu = steps.shape[0]
    T = np.empty((nu, 7))
    dT = np.empty((nu, 7))                         # d/d ln tau of the transition and of the process noise
    for u in range(nu):
        x = lam * steps[u]
        f00, f01, f10, f11, q00, q01, q11 = _sho_step(x, s2)
        T[u, 0], T[u, 1], T[u, 2], T[u, 3], T[u, 4], T[u, 5], T[u, 6] = f00, f01, f10, f11, q00, q01, q11
        g00, g01, g10, g11, r00, r01, r11 = _sho_step_dx(x)
        dT[u, 0], dT[u, 1], dT[u, 2], dT[u, 3] = -x * g00, -x * g01, -x * g10, -x * g11
        dT[u, 4], dT[u, 5], dT[u, 6] = -x * s2 * r00, -x * s2 * r01, -x * s2 * r11
    m0 = 0.0
    m1 = 0.0
    p00, p01, p11 = s2, 0.0, s2
    dm0 = np.zeros(3)
    dm1 = np.zeros(3)
    dp00 = np.zeros(3)
    dp01 = np.zeros(3)
    dp11 = np.zeros(3)
    dp00[0] = 2.0 * s2
    dp11[0] = 2.0 * s2
    nll = 0.0
    grad = np.zeros(3)
    for i in range(n):
        if i > 0:
            u = idx[i]
            f00, f01, f10, f11 = T[u, 0], T[u, 1], T[u, 2], T[u, 3]
            a00 = f00 * p00 + f01 * p01
            a01 = f00 * p01 + f01 * p11
            a10 = f10 * p00 + f11 * p01
            a11 = f10 * p01 + f11 * p11
            n00 = a00 * f00 + a01 * f01 + T[u, 4]
            n01 = a00 * f10 + a01 * f11 + T[u, 5]
            n11 = a10 * f10 + a11 * f11 + T[u, 6]
            for k in range(3):
                # F dP F'
                b00 = f00 * dp00[k] + f01 * dp01[k]
                b01 = f00 * dp01[k] + f01 * dp11[k]
                b10 = f10 * dp00[k] + f11 * dp01[k]
                b11 = f10 * dp01[k] + f11 * dp11[k]
                e00 = b00 * f00 + b01 * f01
                e01 = b00 * f10 + b01 * f11
                e11 = b10 * f10 + b11 * f11
                dm0n = f00 * dm0[k] + f01 * dm1[k]
                dm1n = f10 * dm0[k] + f11 * dm1[k]
                if k == 0:                         # ln amp: Q scales with amp^2
                    e00 += 2.0 * T[u, 4]
                    e01 += 2.0 * T[u, 5]
                    e11 += 2.0 * T[u, 6]
                elif k == 1:                       # ln tau: dF P F' + F P dF' + dQ, dF m
                    d00, d01, d10, d11 = dT[u, 0], dT[u, 1], dT[u, 2], dT[u, 3]
                    c00 = d00 * p00 + d01 * p01
                    c01 = d00 * p01 + d01 * p11
                    c10 = d10 * p00 + d11 * p01
                    c11 = d10 * p01 + d11 * p11
                    x00 = c00 * f00 + c01 * f01
                    x01 = c00 * f10 + c01 * f11
                    x10 = c10 * f00 + c11 * f01
                    x11 = c10 * f10 + c11 * f11
                    e00 += 2.0 * x00 + dT[u, 4]
                    e01 += x01 + x10 + dT[u, 5]
                    e11 += 2.0 * x11 + dT[u, 6]
                    dm0n += d00 * m0 + d01 * m1
                    dm1n += d10 * m0 + d11 * m1
                dp00[k], dp01[k], dp11[k] = e00, e01, e11
                dm0[k], dm1[k] = dm0n, dm1n
            p00, p01, p11 = n00, n01, n11
            nm0 = f00 * m0 + f01 * m1
            m1 = f10 * m0 + f11 * m1
            m0 = nm0
        S = p00 + var[i]
        inn = y1[i] - m0
        iS = 1.0 / S
        nll += 0.5 * (inn * inn * iS + math.log(S)) - math.log(yerr[i])
        k0 = p00 * iS
        k1 = p01 * iS
        for k in range(3):
            dS = dp00[k] + (2.0 * var[i] if k == 2 else 0.0)
            dinn = -dm0[k]
            grad[k] += inn * dinn * iS - 0.5 * inn * inn * dS * iS * iS + 0.5 * dS * iS
            dk0 = dp00[k] * iS - p00 * dS * iS * iS
            dk1 = dp01[k] * iS - p01 * dS * iS * iS
            dm0[k] += dk0 * inn + k0 * dinn
            dm1[k] += dk1 * inn + k1 * dinn
            # P - P[:, 0] P[0, :] / S
            q00 = (2.0 * dp00[k] * p00) * iS - p00 * p00 * dS * iS * iS
            q01 = (dp01[k] * p00 + p01 * dp00[k]) * iS - p01 * p00 * dS * iS * iS
            q11 = (2.0 * dp01[k] * p01) * iS - p01 * p01 * dS * iS * iS
            dp00[k] -= q00
            dp01[k] -= q01
            dp11[k] -= q11
        m0 += k0 * inn
        m1 += k1 * inn
        p11 -= k1 * p01
        p01 -= k1 * p00
        p00 -= k0 * p00
    return nll, grad


@jit(cache=True, nopython=True)
def _sho_step(x, s2):
    """transition matrix (f00, f01, f10, f11) and process noise (q00, q01, q11) of the state (g, g' / omega_0) over a
    step of x = dt / (sqrt(2) tau), stationary variance s2 (as pizzetti's _gp._step)"""
    e = math.exp(-x)
    c, sn = math.cos(x), math.sin(x)
    f01 = 1.4142135623730951 * e * sn
    e2 = e * e
    em = -math.expm1(-2.0 * x)
    return (e * (c + sn), f01, -f01, e * (c - sn),
            s2 * (em - e2 * 2.0 * sn * (c + sn)), s2 * (2.8284271247461903 * e2 * sn * sn),
            s2 * (em - e2 * 2.0 * sn * (sn - c)))


@jit(cache=True, nopython=True)
def gp_setup(t, y1, var, valid, yerr, amp, tau):
    """Everything that does not depend on the transit model, for fixed hyperparameters (float64), at the output
    cadences t (ascending): y1 = data - 1, var = (err_scale yerr)^2, valid (False: left out; no update).
    Returns (F (n, 4): transition from cadence k - 1 to k (row 0: identity), K (n, 2): Kalman gains, irs (n):
    1 / sqrt(innovation variance), e (n): whitened y1, W (n, 3) and v (n, 2): the zero-input tail after cadence k
    (sum over j > k of Dw_j^2 + 2 e_j Dw_j = m' W m + 2 v' m for the filtered D-state m after k), c0 = |e|^2,
    ld = sum ln sqrt(S_k) - sum ln yerr_k over the valid cadences)."""
    n = t.shape[0]
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    F = np.zeros((n, 4))
    K = np.zeros((n, 2))
    irs = np.zeros(n)
    e = np.zeros(n)
    W = np.zeros((n, 3))
    v = np.zeros((n, 2))
    m0 = 0.0
    m1 = 0.0
    p00, p01, p11 = s2, 0.0, s2
    c0 = 0.0
    ld = 0.0
    tidx, steps = time_table(t, 1e-6)
    T = np.empty((steps.shape[0], 7))
    for u in range(steps.shape[0]):
        T[u, 0], T[u, 1], T[u, 2], T[u, 3], T[u, 4], T[u, 5], T[u, 6] = _sho_step(lam * steps[u], s2)
    for i in range(n):
        if i == 0:
            F[i, 0] = 1.0
            F[i, 3] = 1.0
        else:
            u = tidx[i]
            f00, f01, f10, f11, q00, q01, q11 = T[u, 0], T[u, 1], T[u, 2], T[u, 3], T[u, 4], T[u, 5], T[u, 6]
            F[i, 0], F[i, 1], F[i, 2], F[i, 3] = f00, f01, f10, f11
            a00 = f00 * p00 + f01 * p01
            a01 = f00 * p01 + f01 * p11
            a10 = f10 * p00 + f11 * p01
            a11 = f10 * p01 + f11 * p11
            p00 = a00 * f00 + a01 * f01 + q00
            p01 = a00 * f10 + a01 * f11 + q01
            p11 = a10 * f10 + a11 * f11 + q11
            nm0 = f00 * m0 + f01 * m1
            m1 = f10 * m0 + f11 * m1
            m0 = nm0
        if not valid[i]:
            continue
        S = p00 + var[i]
        rs = math.sqrt(S)
        irs[i] = 1.0 / rs
        ld += math.log(rs) - math.log(yerr[i])
        k0 = p00 / S
        k1 = p01 / S
        K[i, 0] = k0
        K[i, 1] = k1
        inn = y1[i] - m0
        e[i] = inn / rs
        c0 += e[i] * e[i]
        m0 += k0 * inn
        m1 += k1 * inn
        p11 -= k1 * p01
        p01 -= k1 * p00
        p00 -= k0 * p00
    # zero-input tails, backwards: W_k = g g' + A' W_(k+1) A, v_k = e_(k+1) g + A' v_(k+1), with
    # g = -irs_(k+1) (F00, F01)_(k+1) and A = (I - K e0') F of cadence k + 1
    w00, w01, w11, v0, v1 = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(n - 2, -1, -1):
        j = i + 1
        f00, f01, f10, f11 = F[j, 0], F[j, 1], F[j, 2], F[j, 3]
        k0, k1 = K[j, 0], K[j, 1]
        g0 = -irs[j] * f00
        g1 = -irs[j] * f01
        A00 = (1.0 - k0) * f00
        A01 = (1.0 - k0) * f01
        A10 = f10 - k1 * f00
        A11 = f11 - k1 * f01
        h00 = w00 * A00 + w01 * A10
        h01 = w00 * A01 + w01 * A11
        h10 = w01 * A00 + w11 * A10
        h11 = w01 * A01 + w11 * A11
        n00 = A00 * h00 + A10 * h10 + g0 * g0
        n01 = A00 * h01 + A10 * h11 + g0 * g1
        n11 = A01 * h01 + A11 * h11 + g1 * g1
        nv0 = A00 * v0 + A10 * v1 + e[j] * g0
        nv1 = A01 * v0 + A11 * v1 + e[j] * g1
        w00, w01, w11, v0, v1 = n00, n01, n11, nv0, nv1
        W[i, 0], W[i, 1], W[i, 2] = w00, w01, w11
        v[i, 0], v[i, 1] = v0, v1
    return F, K, irs, e, W, v, c0, ld


@jit(cache=True, nopython=True, fastmath=True)
def gp_quad(out_idx, Dv, F, K, irs, e, W, v):
    """sum_k Dw_k^2 + 2 e_k Dw_k (Dw = L^-1 D) for the deficit D = Dv at the ascending output cadences out_idx
    (0 elsewhere): the D-filter from the first to the last of them, plus the closed-form tail"""
    nk = out_idx.shape[0]
    if nk == 0:
        return 0.0
    acc = 0.0
    m0 = 0.0
    m1 = 0.0
    i0 = out_idx[0]
    i1 = out_idx[nk - 1]
    k = 0
    for c in range(i0, i1 + 1):
        if c > i0:
            nm0 = F[c, 0] * m0 + F[c, 1] * m1
            m1 = F[c, 2] * m0 + F[c, 3] * m1
            m0 = nm0
        d = 0.0
        if k < nk and out_idx[k] == c:
            d = Dv[k]
            k += 1
        inn = d - m0
        dw = inn * irs[c]
        acc += dw * (dw + 2.0 * e[c])
        m0 += K[c, 0] * inn
        m1 += K[c, 1] * inn
    acc += W[i1, 0] * m0 * m0 + 2.0 * W[i1, 1] * m0 * m1 + W[i1, 2] * m1 * m1 + 2.0 * (v[i1, 0] * m0 + v[i1, 1] * m1)
    return acc


@jit(cache=True, nopython=True, fastmath=True)
def _runs_idx(los, his):
    tot = 0
    for k in range(len(los)):
        tot += his[k] - los[k]
    idx = np.empty(tot, dtype=np.int64)
    m = 0
    for k in range(len(los)):
        for j in range(los[k], his[k]):
            idx[m] = j
            m += 1
    return idx


@jit(cache=True, nopython=True, fastmath=True)
def pandora_loglike_gp(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance, supersampling_factor,
    occult_small_threshold, hill_sphere_threshold, time,
    F, K, irs, e, W, v, c0, ld, cache, time_sorted
):
    """pandora_loglike() with the GP noise model (module docstring): the same window model, the deficits of the
    output cadences it touches, and the GP chi^2 of gp_quad"""
    if not params_valid(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        supersampling_factor
    ):
        return NEG_INF
    los, his, flux_planet_w, flux_moon_w = _window_model_runs(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, time,
        cache, time_sorted
    )
    idx = _runs_idx(los, his)
    out_idx, _fp, _fm, ft = _window_cadences(len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor)
    Dv = np.empty(len(ft))
    for k in range(len(ft)):
        Dv[k] = 1.0 - ft[k]
    return -0.5 * (c0 + gp_quad(out_idx, Dv, F, K, irs, e, W, v)) - ld


@jit(cache=True, nopython=True, fastmath=True)
def pandora_loglike_planet_gp(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, epoch_distance, supersampling_factor,
    occult_small_threshold, hill_sphere_threshold, time,
    F, K, irs, e, W, v, c0, ld, cache, time_sorted
):
    """pandora_loglike_planet() with the GP noise model: the planet's z^2 and flux on its transit windows (the fast
    path of pandora_loglike_planet), else pandora_loglike_gp with the planet-only moon parameters"""
    r_m, p_m, tau_m, om_m, i_m, e_m, w_m, m_m = PLANET_ONLY_MOON
    fast = (ecc_bary == 0 and time_sorted and r_planet >= 1e-3)
    if cache is not None:
        fast = False
    ok = False
    if fast:
        if not params_valid(
            u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
            t0_bary, t0_bary_offset, M_planet, r_m, p_m, tau_m, om_m, i_m, e_m, w_m,
            m_m, epoch_distance, supersampling_factor
        ):
            return NEG_INF
        reach = 1 + r_planet + 1e-6
        ok, los, his, c_t, c_d, tdur, t0_shift = x_bary_runs_sorted(
            time, float(a_bary), float(per_bary), float(t0_bary), float(t0_bary_offset),
            epoch_distance, 0.0, float(w_bary), reach
        )
    if not ok:
        return pandora_loglike_gp(
            u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
            t0_bary, t0_bary_offset, M_planet, r_m, p_m, tau_m, om_m, i_m, e_m,
            w_m, m_m, epoch_distance, supersampling_factor, occult_small_threshold,
            hill_sphere_threshold, time, F, K, irs, e, W, v, c0, ld, cache, time_sorted
        )
    idx = _runs_idx(los, his)
    b2 = float(b_bary) * float(b_bary)
    z2 = np.empty(len(idx))
    m = 0
    for k in range(len(los)):
        ct = c_t[k]
        cd = c_d[k]
        for j in range(los[k], his[k]):
            x = (2 * (time[j] - ct)) / tdur - t0_shift - cd
            z2[m] = x * x + b2
            m += 1
    flux = occult_hybrid_z2(z2, float(r_planet), u1, u2)
    ones = np.ones(len(flux))
    out_idx, _fp, _fm, ft = _window_cadences(len(time), idx, flux, ones, supersampling_factor)
    Dv = np.empty(len(ft))
    for k in range(len(ft)):
        Dv[k] = 1.0 - ft[k]
    return -0.5 * (c0 + gp_quad(out_idx, Dv, F, K, irs, e, W, v)) - ld


@jit(cache=True, nopython=True)
def _rts(t, r, var, valid, amp, tau):
    """Kalman filter + RTS smoother of the SHO state given the residual r = data - model (times ascending): posterior
    mean and variance of g"""
    n = t.shape[0]
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    mf = np.zeros((n, 2))
    Pf = np.zeros((n, 3))
    mp = np.zeros((n, 2))
    Pp = np.zeros((n, 3))
    Fs = np.zeros((n, 4))
    m0, m1 = 0.0, 0.0
    p00, p01, p11 = s2, 0.0, s2
    for i in range(n):
        if i > 0:
            f00, f01, f10, f11, q00, q01, q11 = _sho_step(lam * (t[i] - t[i - 1]), s2)
            Fs[i] = (f00, f01, f10, f11)
            a00 = f00 * p00 + f01 * p01
            a01 = f00 * p01 + f01 * p11
            a10 = f10 * p00 + f11 * p01
            a11 = f10 * p01 + f11 * p11
            p00 = a00 * f00 + a01 * f01 + q00
            p01 = a00 * f10 + a01 * f11 + q01
            p11 = a10 * f10 + a11 * f11 + q11
            m0, m1 = f00 * m0 + f01 * m1, f10 * m0 + f11 * m1
        mp[i] = (m0, m1)
        Pp[i] = (p00, p01, p11)
        if valid[i]:
            S = p00 + var[i]
            k0, k1 = p00 / S, p01 / S
            inn = r[i] - m0
            m0 += k0 * inn
            m1 += k1 * inn
            p11 -= k1 * p01
            p01 -= k1 * p00
            p00 -= k0 * p00
        mf[i] = (m0, m1)
        Pf[i] = (p00, p01, p11)
    mean = np.zeros(n)
    vv = np.zeros(n)
    sm0, sm1 = mf[n - 1, 0], mf[n - 1, 1]
    s00, s01, s11 = Pf[n - 1, 0], Pf[n - 1, 1], Pf[n - 1, 2]
    mean[n - 1], vv[n - 1] = sm0, s00
    for i in range(n - 2, -1, -1):
        f00, f01, f10, f11 = Fs[i + 1, 0], Fs[i + 1, 1], Fs[i + 1, 2], Fs[i + 1, 3]
        a00, a01, a11 = Pf[i, 0], Pf[i, 1], Pf[i, 2]
        b00, b01, b11 = Pp[i + 1, 0], Pp[i + 1, 1], Pp[i + 1, 2]
        # C = Pf F' Pp^-1
        c00 = a00 * f00 + a01 * f01
        c01 = a00 * f10 + a01 * f11
        c10 = a01 * f00 + a11 * f01
        c11 = a01 * f10 + a11 * f11
        det = b00 * b11 - b01 * b01
        i00, i01, i11 = b11 / det, -b01 / det, b00 / det
        g00 = c00 * i00 + c01 * i01
        g01 = c00 * i01 + c01 * i11
        g10 = c10 * i00 + c11 * i01
        g11 = c10 * i01 + c11 * i11
        d0 = sm0 - mp[i + 1, 0]
        d1 = sm1 - mp[i + 1, 1]
        n0 = mf[i, 0] + g00 * d0 + g01 * d1
        n1 = mf[i, 1] + g10 * d0 + g11 * d1
        # P = Pf + G (Ps - Pp) G'
        e00, e01, e11 = s00 - b00, s01 - b01, s11 - b11
        h00 = g00 * e00 + g01 * e01
        h01 = g00 * e01 + g01 * e11
        h10 = g10 * e00 + g11 * e01
        h11 = g10 * e01 + g11 * e11
        s00 = a00 + h00 * g00 + h01 * g01
        s01 = a01 + h00 * g10 + h01 * g11
        s11 = a11 + h10 * g10 + h11 * g11
        sm0, sm1 = n0, n1
        mean[i], vv[i] = sm0, s00
    return mean, vv


def output_times(time, supersampling_factor):
    """the output cadence times (mean of each block of supersampled times)"""
    t = np.asarray(time, dtype=np.float64)
    f = int(supersampling_factor)
    if f <= 1:
        return t.copy()
    n_out = -(-len(t) // f)
    out = np.empty(n_out)
    for k in range(n_out):
        out[k] = t[k * f:(k + 1) * f].mean()
    return out


@jit(cache=True, nopython=True)
def whiten_deficit(Dfull, F, K, irs):
    """L^-1 D for the deficit D at all output cadences (full length): the whitened model part of the residual
    (the whitened residual is e + whiten_deficit(D), e = gp_setup's e)"""
    n = Dfull.shape[0]
    out = np.zeros(n)
    m0 = 0.0
    m1 = 0.0
    for c in range(n):
        if c > 0:
            nm0 = F[c, 0] * m0 + F[c, 1] * m1
            m1 = F[c, 2] * m0 + F[c, 3] * m1
            m0 = nm0
        inn = Dfull[c] - m0
        out[c] = inn * irs[c]
        m0 += K[c, 0] * inn
        m1 += K[c, 1] * inn
    return out
