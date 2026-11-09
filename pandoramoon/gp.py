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
likelihood call only filters D (a two-number state recursion over the cadences of the model's windows; the
remaining cadences, where D = 0, are closed-form terms: ``gp_quad``). Between windows the D-filter's state decays
without input; after a gap longer than its horizon H (``gp_horizon``: the zero-input steps after which the state's
influence is below 1e-12) the gap is not stepped: the decaying state's terms are the closed-form tail, and the next
window starts from state 0 (dropped: cross terms below ~1e-8 nats). The log-likelihood is

    log L = -chi^2 / 2 - (sum_k ln L_kk - sum_k ln yerr_k)

i.e. relative to the normalisation of the white-noise Likelihood: amp -> 0 with err_scale = 1 gives the white-noise
value. The same GP enters the planet-only model (Bayes factors compare the models under the same noise model).

General kernels (``GP(terms=[...], baseline=..., gap=...)``): a sum of SHO terms of any quality factor Q (``SHO``;
``Rotation`` = celerite2's RotationTerm, two underdamped SHO terms at the rotation period and its half), optionally
with a baseline per segment (``baseline="offset"`` or ``"linear"``, marginalised analytically with a wide Gaussian
prior, ``baseline_sd``) and independent segments (``gap``: a new segment after a gap longer than ``gap`` days; the GP
and the baseline restart there). All of it is one linear state-space model of d <= 8 states (2 per SHO term, 1 for
an offset, 2 for a linear baseline), written in coordinates in which the observed sum is state 0; the per-call work
is the same D-filter with d states (d = 2 uses the code of the single SHO, on the CPU and the GPU).
"""
import math

import numpy as np
from numba import jit

from pandoramoon.kernel import (
    _window_model_runs, _window_cadences, x_bary_runs_sorted, occult_hybrid_z2, params_valid, PLANET_ONLY_MOON,
    NEG_INF,
)

_SQRT2 = math.sqrt(2.0)
_Q0 = 1.0 / math.sqrt(2.0)              # the SHO quality of GP(amp, tau) (pizzetti's kernel)
D_MAX = 8                               # states of the general model (the GPU kernel's limit)
_BASELINES = (None, "offset", "linear")
KERNELS = ("sho", "sho_q", "sho2", "rotation")


def _pos(name, x):
    x = float(x)
    if not (x > 0 and math.isfinite(x)):
        raise ValueError(f"{name} must be positive and finite, got {x}")
    return x


class SHO(object):
    """One term of a GP kernel: the stochastically driven, damped harmonic oscillator (celerite's SHOTerm)

        k(dt) = amp^2 exp(-s / (2 Q)) (cos(eta s) + sin(eta s) / (2 Q eta)),  s = |dt| / tau,  eta = sqrt(1 - 1/(4 Q^2))

    (Q < 1/2: overdamped, cosh / sinh; Q = 1/2: critically damped). amp: rms (relative flux), tau = 1 / omega_0 [d],
    Q: quality factor (default 1/sqrt(2): not oscillating, the kernel of GP(amp, tau)). Q > 1/2 oscillates with the
    period 2 pi tau / eta (a quasi-periodic signal, e.g. spots), damped on the time scale 2 Q tau."""

    def __init__(self, amp, tau, Q=_Q0):
        self.amp, self.tau, self.Q = _pos("SHO amp", amp), _pos("SHO tau", tau), _pos("SHO Q", Q)

    def __repr__(self):
        return f"SHO(amp={self.amp:.6g}, tau={self.tau:.6g}, Q={self.Q:.6g})"

    def key(self):
        return ("sho", self.amp, self.tau, self.Q)

    def sho_terms(self):
        return [self]

    def kernel(self, dt):
        """k(dt) (numpy)"""
        s = np.abs(np.asarray(dt, dtype=np.float64)) / self.tau
        z = 0.5 / self.Q
        if z < 1:
            eta = math.sqrt(1 - z * z)
            c = np.cos(eta * s) + z / eta * np.sin(eta * s)
            return self.amp ** 2 * np.exp(-z * s) * c
        if z == 1:
            return self.amp ** 2 * np.exp(-s) * (1 + s)
        eta = math.sqrt(z * z - 1)
        return self.amp ** 2 * 0.5 * ((1 + z / eta) * np.exp(-(z - eta) * s) + (1 - z / eta) * np.exp(-(z + eta) * s))


class Rotation(object):
    """A rotation kernel (celerite2's RotationTerm): two underdamped SHO terms at the rotation period and at half of
    it, for a spotted rotating star. amp: total rms (relative flux), period [d], Q0 > 0: quality of the secondary
    (half-period) term minus 1/2, dQ >= 0: the primary term's quality above the secondary's, f in [0, 1]: the
    secondary's variance relative to the primary's. The terms (Q1 = 1/2 + Q0 + dQ, Q2 = 1/2 + Q0) oscillate exactly
    at period and period / 2. Needs data covering a few rotations to be constrained."""

    def __init__(self, amp, period, Q0=1.0, dQ=1.0, f=0.5):
        self.amp, self.period, self.Q0 = _pos("Rotation amp", amp), _pos("Rotation period", period), _pos("Q0", Q0)
        self.dQ, self.f = float(dQ), float(f)
        if not (self.dQ >= 0 and math.isfinite(self.dQ)):
            raise ValueError(f"Rotation dQ must be >= 0, got {dQ}")
        if not (0 <= self.f <= 1):
            raise ValueError(f"Rotation f must be in [0, 1], got {f}")

    def __repr__(self):
        return (f"Rotation(amp={self.amp:.6g}, period={self.period:.6g}, Q0={self.Q0:.6g}, dQ={self.dQ:.6g}, "
                f"f={self.f:.6g})")

    def key(self):
        return ("rotation", self.amp, self.period, self.Q0, self.dQ, self.f)

    def sho_terms(self):
        q1 = 0.5 + self.Q0 + self.dQ
        w1 = 4 * math.pi * q1 / (self.period * math.sqrt(4 * q1 * q1 - 1))
        out = [SHO(self.amp / math.sqrt(1 + self.f), 1 / w1, q1)]
        if self.f > 0:
            q2 = 0.5 + self.Q0
            w2 = 8 * math.pi * q2 / (self.period * math.sqrt(4 * q2 * q2 - 1))
            out.append(SHO(self.amp * math.sqrt(self.f / (1 + self.f)), 1 / w2, q2))
        return out

    def kernel(self, dt):
        return sum(t.kernel(dt) for t in self.sho_terms())


class GP(object):
    """Fixed hyperparameters of the simultaneous GP (see the module docstring).

    GP(amp, tau, err_scale=1.0): one SHO term with Q = 1/sqrt(2) (the default; as pizzetti's GP);
    GP(amp, tau, Q=...): one SHO term of quality Q;
    GP(terms=[SHO(...), Rotation(...), ...]): a sum of terms (at most 8 states with the baseline: 2 per SHO, a
    Rotation is 2 SHO terms).

    :param amp: rms of the GP, in relative flux (e.g. 3e-4 for 300 ppm)
    :param tau: time scale 1 / omega_0 in days
    :param err_scale: factor on yerr (the white noise is err_scale * yerr)
    :param baseline: None, "offset" (a constant per segment) or "linear" (offset + slope per segment), each
        marginalised analytically (a Gaussian prior with the sd baseline_sd = (offset, slope per day); wide by
        default, the Bayes factor does not depend on it as long as it is wide)
    :param gap: None (one segment) or a time in days: a gap longer than this starts a new, independent segment (the
        GP and the baseline restart; e.g. Kepler quarters, or the data windows around each transit)
    """

    def __init__(self, amp=None, tau=None, err_scale=1.0, Q=None, terms=None, baseline=None, gap=None,
                 baseline_sd=(0.01, 0.01)):
        if terms is None:
            if amp is None or tau is None:
                raise ValueError("GP: give amp and tau (one SHO term) or terms")
            terms = [SHO(amp, tau, _Q0 if Q is None else Q)]
        else:
            if amp is not None or tau is not None or Q is not None:
                raise ValueError("GP: give either amp, tau (and Q) or terms, not both")
            terms = list(terms)
            if not terms or not all(hasattr(x, "sho_terms") for x in terms):
                raise ValueError("GP: terms must be a non-empty list of SHO / Rotation terms")
        self.terms = terms
        self.err_scale = _pos("GP err_scale", err_scale)
        if baseline not in _BASELINES:
            raise ValueError(f"GP: baseline must be one of {_BASELINES}, got {baseline!r}")
        self.baseline = baseline
        self.gap = None if gap is None else _pos("GP gap", gap)
        sd = tuple(float(x) for x in np.broadcast_to(np.asarray(baseline_sd, dtype=np.float64), (2,)))
        self.baseline_sd = (_pos("baseline_sd", sd[0]), _pos("baseline_sd", sd[1]))
        self.sho = [s for x in terms for s in x.sho_terms()]
        self.dim = 2 * len(self.sho) + {None: 0, "offset": 1, "linear": 2}[baseline]
        if self.dim > D_MAX:
            raise ValueError(f"GP: {self.dim} states, at most {D_MAX} (2 per SHO term, 2 per Rotation, 1-2 baseline)")
        one = len(terms) == 1 and isinstance(terms[0], SHO)
        self.amp = terms[0].amp if one else None
        self.tau = terms[0].tau if one else None
        self.Q = terms[0].Q if one else None
        self.kernel = "sho" if self.is_simple() else "ss"

    def is_simple(self):
        """one SHO term of Q = 1/sqrt(2), no baseline, one segment: the original model (its own compiled path)"""
        return (len(self.terms) == 1 and isinstance(self.terms[0], SHO) and self.terms[0].Q == _Q0
                and self.baseline is None and self.gap is None)

    def __repr__(self):
        if self.is_simple():
            return f"GP(amp={self.amp:.6g}, tau={self.tau:.6g}, err_scale={self.err_scale:.6g})"
        if len(self.terms) == 1 and isinstance(self.terms[0], SHO):
            s = f"amp={self.amp:.6g}, tau={self.tau:.6g}, Q={self.Q:.6g}"
        else:
            s = "terms=[" + ", ".join(repr(x) for x in self.terms) + "]"
        s += f", err_scale={self.err_scale:.6g}"
        if self.baseline:
            s += f", baseline={self.baseline!r}"
        if self.gap is not None:
            s += f", gap={self.gap:.6g}"
        return f"GP({s})"

    def key(self):
        """the hyperparameters as a tuple (data hashes, resume)"""
        if self.is_simple():
            return ("sho", self.amp, self.tau, self.err_scale)
        return ("ss", tuple(x.key() for x in self.terms), self.err_scale, self.baseline,
                self.baseline_sd if self.baseline else None, self.gap)

    def refit_options(self):
        """GP.fit keyword arguments that refit a GP of this form (kernel, baseline, gap, baseline_sd, the rotation
        period as the start), or None if no GP.fit kernel has this form (a hand-made sum of terms)"""
        t = self.terms
        o = dict(baseline=self.baseline, gap=self.gap, baseline_sd=self.baseline_sd)
        if len(t) == 1 and isinstance(t[0], SHO):
            return dict(o, kernel="sho" if t[0].Q == _Q0 else "sho_q")
        if len(t) == 2 and all(isinstance(x, SHO) and x.Q == _Q0 for x in t):
            return dict(o, kernel="sho2")
        if len(t) == 2 and isinstance(t[0], Rotation) and isinstance(t[1], SHO) and t[1].Q == _Q0:
            return dict(o, kernel="rotation", period=t[0].period)
        return None

    def blocks(self):
        """the state-space blocks as arrays (kind 0 = SHO (amp, tau, Q), 1 = offset (sd), 2 = linear (sd, sd'))"""
        kind, pa, pb, pc = [], [], [], []
        for s in self.sho:
            kind.append(0), pa.append(s.amp), pb.append(s.tau), pc.append(s.Q)
        if self.baseline == "offset":
            kind.append(1), pa.append(self.baseline_sd[0]), pb.append(0.0), pc.append(0.0)
        elif self.baseline == "linear":
            kind.append(2), pa.append(self.baseline_sd[0]), pb.append(self.baseline_sd[1]), pc.append(0.0)
        return (np.array(kind, dtype=np.int64), np.array(pa, dtype=np.float64), np.array(pb, dtype=np.float64),
                np.array(pc, dtype=np.float64))

    def segment_starts(self, time):
        """True at the first cadence of each segment (the first cadence, and after every gap longer than gap)"""
        t = np.asarray(time, dtype=np.float64)
        seg = np.zeros(len(t), dtype=np.bool_)
        if len(t):
            seg[0] = True
        if self.gap is not None and len(t) > 1:
            seg[1:] = np.diff(t) > self.gap
        return seg

    def setup(self, t, y1, valid, yerr):
        """the tables of the likelihood at the ascending cadence times t, y1 = data - 1: (F, K, irs, e, W, v, H, c0,
        ld) of gp_setup / gp_setup_ss with the horizons H (gp_horizon) after v"""
        var = np.ascontiguousarray((self.err_scale * yerr) ** 2)
        if self.is_simple():
            F, K, irs, e, W, v, c0, ld = gp_setup(t, y1, var, valid, yerr, self.amp, self.tau)
        else:
            kind, pa, pb, pc = self.blocks()
            F, K, irs, e, W, v, c0, ld = gp_setup_ss(t, y1, var, valid, yerr, kind, pa, pb, pc,
                                                     self.segment_starts(t))
        return F, K, irs, e, W, v, gp_horizon(F, K), c0, ld

    def loglike_noise(self, time, flux, yerr, mask=None):
        """log likelihood of the light curve under the GP alone (model flux 1 everywhere), relative to the white-noise
        normalisation as in Likelihood; points with mask True (or non-finite flux / yerr) are left out"""
        t, y1, yerr_, valid = _prepare(time, flux, yerr, mask)
        tab = self.setup(t, y1, valid, yerr_)
        return -0.5 * tab[-2] - tab[-1]

    @classmethod
    def fit(cls, time, flux, yerr, mask=None, err_scale=None, amp_bounds=None, tau_bounds=None, bin_width=10.0 / 1440,
            n_starts=3, kernel="sho", baseline=None, gap=None, baseline_sd=(0.01, 0.01), period=None, clip=None):
        """Maximum-likelihood (type II) hyperparameters of the GP alone on the light curve (model flux 1), e.g. with
        the transit windows masked (mask True = left out). err_scale: a number to hold it fixed, or None to fit it
        (bounds 1/3 - 3). Bounds (defaults): amp from 0.02 to 50 x the median yerr, tau from the median time step to
        the time span. bin_width [d]: the fit uses data binned to this width if the median time step is shorter than
        half of it (None: never bin); as TLS's gp fit (fit_bin_width, 10 min).

        kernel: "sho" (one SHO term, Q = 1/sqrt(2); the default), "sho_q" (one SHO term with its Q fitted, 0.2 - 50),
        "sho2" (two SHO terms of Q = 1/sqrt(2): e.g. a slow trend plus granulation), "rotation" (Rotation + one SHO
        term for granulation; period: a starting rotation period [d], or None: the three highest peaks of a
        Lomb-Scargle periodogram of the data). baseline, gap, baseline_sd: as in GP (held, not fitted).
        clip: None, or (upper, lower) [sigma] (True: (4, 5)): iterate the fit with flag_outliers (flares above
        upper sigma everywhere, outliers below lower sigma outside mask) masked; the flagged points are returned as
        the GP's outliers (a boolean array over the input points; set those fluxes to NaN for the Likelihood).

        "sho" without baseline and gap: as TLS 2.2's gradient fit of its baselines: the exact gradient of the Kalman
        likelihood (forward mode in (ln amp, ln tau, ln err_scale)), a bounded quasi-Newton (projected BFGS,
        _minimize_bounded), transitions once per distinct time step (time_table), from n_starts values of tau. The
        other kernels: the same optimiser on the general state-space likelihood with central-difference gradients.
        Returns a GP with fit_nll (the minimum), fit_nfev and fit_points (after binning)."""
        if kernel not in KERNELS:
            raise ValueError(f"GP.fit: kernel must be one of {KERNELS}, got {kernel!r}")
        opts = dict(err_scale=err_scale, amp_bounds=amp_bounds, tau_bounds=tau_bounds, bin_width=bin_width,
                    n_starts=n_starts, kernel=kernel, baseline=baseline, gap=gap, baseline_sd=baseline_sd,
                    period=period)
        if clip is None or clip is False:
            return cls._fit(time, flux, yerr, mask, **opts)
        upper, lower = (4.0, 5.0) if clip is True else (float(clip[0]), float(clip[1]))
        m0 = np.zeros(len(np.asarray(time)), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        out = np.zeros_like(m0)
        for _ in range(5):
            g = cls._fit(time, flux, yerr, m0 | out, **opts)
            new = out | g.flag_outliers(time, flux, yerr, mask=m0 | out, upper=upper, lower=lower, inside=m0)
            if np.array_equal(new, out):
                break
            out = new
        g.outliers = out
        return g

    @classmethod
    def _fit(cls, time, flux, yerr, mask, err_scale, amp_bounds, tau_bounds, bin_width, n_starts, kernel, baseline,
             gap, baseline_sd, period):
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
        var0 = np.ascontiguousarray(yerr_ * yerr_)
        s_fix = 1.0 if fit_s else float(err_scale)
        rms = float(np.std(y1))
        a0 = math.log(min(max(math.sqrt(max(rms * rms - ye * ye, (0.1 * ye) ** 2)), alo * 1.01), ahi / 1.01))
        if kernel == "sho" and baseline is None and gap is None:
            idx, steps = time_table(t, 1e-6)
            lo = np.array([math.log(alo), math.log(tlo)] + ([math.log(1.0 / 3.0)] if fit_s else []))
            hi = np.array([math.log(ahi), math.log(thi)] + ([math.log(3.0)] if fit_s else []))
            nfev = [0]

            def fg(z):
                nfev[0] += 1
                s = math.exp(z[2]) if fit_s else s_fix
                f, g = _nll_grad(idx, steps, y1, var0 * s * s, yerr_, math.exp(z[0]), math.exp(z[1]))
                return f, (g if fit_s else g[:2])

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
        return _fit_general(cls, t, y1, yerr_, var0, kernel, fit_s, s_fix, baseline, gap, baseline_sd, n_starts,
                            period, a0, (alo, ahi), (tlo, thi), dt_med, span)

    def sample(self, time, rng=None):
        """a random draw of the GP (the correlated part only: the SHO terms, without the baseline and the white
        noise) at the ascending times, e.g. to simulate stellar variability or for injection tests (exact
        state-space simulation; independent segments after gaps longer than gap)"""
        rng = np.random.default_rng(rng)
        t = np.asarray(time, dtype=np.float64)
        if len(t) > 1 and not np.all(np.diff(t) >= 0):
            raise ValueError("GP.sample needs ascending times")
        if self.is_simple():
            z = rng.standard_normal((len(t), 2))
            return _sho_sample(t, self.amp, self.tau, z)
        seg = self.segment_starts(t)
        out = np.zeros(len(t))
        for s in self.sho:
            out += _sho_sample_q(t, s.amp, s.tau, s.Q, rng.standard_normal((len(t), 2)), seg)
        return out

    def predict(self, time, residual, yerr, mask=None):
        """posterior mean and sd of the GP (with the baseline, if any) at the given (ascending) times, given
        residual = data - model there (Kalman filter + Rauch-Tung-Striebel smoother; mask True = left out): the
        trend to subtract for plots"""
        t, r, yerr_, valid = _prepare(time, np.asarray(residual, dtype=np.float64) + 1.0, yerr, mask)
        var_w = (self.err_scale * yerr_) ** 2
        if self.is_simple():
            mean, var = _rts(t, r, var_w, valid, self.amp, self.tau)
        else:
            kind, pa, pb, pc = self.blocks()
            mean, var = _rts_ss(t, r, var_w, valid, kind, pa, pb, pc, self.segment_starts(t))
        return mean, np.sqrt(np.maximum(var, 0.0))

    def flag_outliers(self, time, flux, yerr, mask=None, upper=4.0, lower=5.0, inside=None, max_iter=100, sep=3):
        """Flares and outliers under this GP: points whose leave-one-out residual (the residual from the GP's
        prediction given all other points; (r - mean) / sqrt(s^2 yerr^2 - var) for a point in the fit, the exact LOO
        identity of the smoother) is above upper sigma (flares: everywhere, also inside the transit windows: above
        the trend, which a transit cannot produce) or below -lower sigma (outside `inside`, e.g. the transit
        windows, where dips are the signal). Iterated, peak first: each pass flags the candidates whose |z| is the
        largest within sep cadences (the neighbours of a flare's peak look low while the peak pulls the prediction
        up), leaves them out of the prediction, and repeats until no point is added (flares of many cadences).
        mask: points left out of the prediction from the start (e.g. the transit windows); inside: where only the
        upper test applies (default: mask). Returns a boolean array (True = flagged)."""
        t, y1, yerr_, ok = _prepare(time, flux, yerr, None)
        n = len(t)
        m0 = np.zeros(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        ins = m0 if inside is None else np.asarray(inside, dtype=bool)
        var_w = (self.err_scale * yerr_) ** 2
        out = np.zeros(n, dtype=bool)
        for _ in range(max_iter):
            use = ok & ~m0 & ~out
            mean, sd = self.predict(t, y1, yerr_, mask=~use)
            r = y1 - mean
            v = sd * sd
            z = np.where(use, r / np.sqrt(np.maximum(var_w - v, 1e-6 * var_w)), r / np.sqrt(var_w + v))
            cand = ok & ~out & ((z > upper) | (~ins & (z < -lower)))
            if not cand.any():
                break
            a = np.where(cand, np.abs(z), 0.0)
            peak = a.copy()
            for k in range(1, int(sep) + 1):
                peak[k:] = np.maximum(peak[k:], a[:-k])
                peak[:-k] = np.maximum(peak[:-k], a[k:])
            out = out | (cand & (a >= peak))
        return out


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
def _sho_sample(t, amp, tau, z):
    """GP.sample: the SHO state propagated with its exact transition and process noise (z: standard normals)"""
    n = t.shape[0]
    out = np.empty(n)
    s2 = amp * amp
    lam = 1.0 / (tau * 1.4142135623730951)
    x0 = amp * z[0, 0]
    x1 = amp * z[0, 1]
    out[0] = x0
    for i in range(1, n):
        f00, f01, f10, f11, q00, q01, q11 = _sho_step(lam * (t[i] - t[i - 1]), s2)
        y0 = f00 * x0 + f01 * x1
        y1 = f10 * x0 + f11 * x1
        # Cholesky factor of the process noise (q00 > 0 for a positive step)
        l00 = math.sqrt(max(q00, 0.0))
        l10 = q01 / l00 if l00 > 0 else 0.0
        l11 = math.sqrt(max(q11 - l10 * l10, 0.0))
        x0 = y0 + l00 * z[i, 0]
        x1 = y1 + l10 * z[i, 0] + l11 * z[i, 1]
        out[i] = x0
    return out


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


# --------------------------------------------------------------------------------------------------------------------
# General state-space model (GP(terms=..., baseline=..., gap=...)): blocks of SHO terms (any Q) and a baseline, in
# coordinates z = T m with z_0 = h . m the observed sum (T = I with its row 0 replaced by h; h_0 = 1, the first block
# is always an SHO term). The tables have the layout of gp_setup's for any d: F (n, d*d) row-major, K (n, d), irs, e,
# W (n, d (d+1) / 2) the upper triangle row-major, v (n, d); d = 2 is the single-SHO layout.
# --------------------------------------------------------------------------------------------------------------------

@jit(cache=True, nopython=True)
def _sho_trans(dt, tau, Q, s2):
    """transition (f00, f01, f10, f11) and process noise (q00, q01, q11) of an SHO block of quality Q over the step
    dt (state (g, g' tau), stationary covariance s2 I): F = exp(A dt) in closed form, Q = s2 (I - F F'), written
    without the cancellation of 1 - e^-2zs; Q = 1/sqrt(2): _sho_step (the same closed form as the single SHO)"""
    if Q == _Q0:
        return _sho_step(dt / (1.4142135623730951 * tau), s2)
    s = dt / tau
    z = 0.5 / Q
    if z < 1.0:
        eta = math.sqrt(1.0 - z * z)
        E = math.exp(-z * s)
        EC = E * math.cos(eta * s)
        ES = E * math.sin(eta * s) / eta
    elif z == 1.0:
        E = math.exp(-s)
        EC = E
        ES = E * s
    else:
        eta = math.sqrt(z * z - 1.0)
        a = math.exp(-(z - eta) * s)
        b = math.exp(-(z + eta) * s)
        EC = 0.5 * (a + b)
        ES = -0.5 * a * math.expm1(-2.0 * eta * s) / eta
    em = -math.expm1(-2.0 * z * s)
    return (EC + z * ES, ES, -ES, EC - z * ES,
            s2 * (em - 2.0 * z * ES * (z * ES + EC)), s2 * (2.0 * z * ES * ES), s2 * (em - 2.0 * z * ES * (z * ES - EC)))


@jit(cache=True, nopython=True)
def _ss_dim(kind):
    d = 0
    for k in kind:
        d += 1 if k == 1 else 2
    return d


@jit(cache=True, nopython=True)
def _ss_mats(kind, pa, pb, pc, dt, F, Qm):
    """transition F and process noise Qm (d x d, original coordinates) of the blocks over the step dt"""
    F[:, :] = 0.0
    Qm[:, :] = 0.0
    o = 0
    for b in range(kind.shape[0]):
        if kind[b] == 0:
            f00, f01, f10, f11, q00, q01, q11 = _sho_trans(dt, pb[b], pc[b], pa[b] * pa[b])
            F[o, o] = f00
            F[o, o + 1] = f01
            F[o + 1, o] = f10
            F[o + 1, o + 1] = f11
            Qm[o, o] = q00
            Qm[o, o + 1] = q01
            Qm[o + 1, o] = q01
            Qm[o + 1, o + 1] = q11
            o += 2
        elif kind[b] == 1:
            F[o, o] = 1.0
            o += 1
        else:
            F[o, o] = 1.0
            F[o, o + 1] = dt
            F[o + 1, o + 1] = 1.0
            o += 2


@jit(cache=True, nopython=True)
def _ss_prior(kind, pa, pb, d):
    """the state's prior covariance at a segment start (the SHO terms stationary) and the observation vector h"""
    P0 = np.zeros((d, d))
    h = np.zeros(d)
    o = 0
    for b in range(kind.shape[0]):
        if kind[b] == 0:
            P0[o, o] = pa[b] * pa[b]
            P0[o + 1, o + 1] = pa[b] * pa[b]
            h[o] = 1.0
            o += 2
        elif kind[b] == 1:
            P0[o, o] = pa[b] * pa[b]
            h[o] = 1.0
            o += 1
        else:
            P0[o, o] = pa[b] * pa[b]
            P0[o + 1, o + 1] = pb[b] * pb[b]
            h[o] = 1.0
            o += 2
    return P0, h


@jit(cache=True, nopython=True)
def _ss_zmats(kind, pa, pb, pc, steps, d):
    """per distinct step: the transition and process noise in the z coordinates (Fz (nu, d, d), Qz), and the prior
    P0z: z = T m, T = I with row 0 = h, T^-1 = I with row 0 = 2 e_0 - h"""
    P0, h = _ss_prior(kind, pa, pb, d)
    T = np.eye(d)
    Ti = np.eye(d)
    for j in range(d):
        T[0, j] = h[j]
        Ti[0, j] = -h[j]
    Ti[0, 0] = 1.0
    nu = steps.shape[0]
    Fz = np.empty((nu, d, d))
    Qz = np.empty((nu, d, d))
    F = np.empty((d, d))
    Qm = np.empty((d, d))
    for u in range(nu):
        _ss_mats(kind, pa, pb, pc, steps[u], F, Qm)
        Fz[u] = T @ F @ Ti
        Qz[u] = T @ Qm @ T.T
    return Fz, Qz, T @ P0 @ T.T


@jit(cache=True, nopython=True)
def _ss_filter(t, y1, var, valid, yerr, kind, pa, pb, pc, seg, tables):
    """the Kalman filter of the data under the general model (z coordinates, z_0 observed): c0 = |e|^2 and ld as
    in gp_setup, and (tables) F (n, d*d), K (n, d), irs, e (F row 0 and every segment start: 0)"""
    n = t.shape[0]
    d = _ss_dim(kind)
    tidx, steps = time_table(t, 1e-6)
    Fz, Qz, P0z = _ss_zmats(kind, pa, pb, pc, steps, d)
    nt = n if tables else 0
    F = np.zeros((nt, d * d))
    K = np.zeros((nt, d))
    irs = np.zeros(nt)
    e = np.zeros(nt)
    m = np.zeros(d)
    P = np.zeros((d, d))
    row = np.zeros(d)
    c0 = 0.0
    ld = 0.0
    for i in range(n):
        if seg[i]:
            m[:] = 0.0
            P[:, :] = P0z
        else:
            Fi = Fz[tidx[i]]
            m = Fi @ m
            P = Fi @ P @ Fi.T + Qz[tidx[i]]
            if tables:
                for a in range(d):
                    for b in range(d):
                        F[i, a * d + b] = Fi[a, b]
        if not valid[i]:
            continue
        S = P[0, 0] + var[i]
        rs = math.sqrt(S)
        inn = y1[i] - m[0]
        ei = inn / rs
        c0 += ei * ei
        ld += math.log(rs) - math.log(yerr[i])
        for a in range(d):
            row[a] = P[0, a]
        for a in range(d):
            k = row[a] / S
            m[a] += k * inn
            for b in range(d):
                P[a, b] -= k * row[b]
            if tables:
                K[i, a] = k
        for a in range(d):
            for b in range(a + 1, d):
                x = 0.5 * (P[a, b] + P[b, a])
                P[a, b] = x
                P[b, a] = x
        if tables:
            irs[i] = 1.0 / rs
            e[i] = ei
    return F, K, irs, e, c0, ld


@jit(cache=True, nopython=True)
def _ss_nll(t, y1, var, yerr, kind, pa, pb, pc, seg):
    """negative log likelihood of y1 (all points used) under the general model, normalised as _nll_grad"""
    valid = np.ones(t.shape[0], dtype=np.bool_)
    _F, _K, _i, _e, c0, ld = _ss_filter(t, y1, var, valid, yerr, kind, pa, pb, pc, seg, False)
    return 0.5 * c0 + ld


@jit(cache=True, nopython=True)
def gp_setup_ss(t, y1, var, valid, yerr, kind, pa, pb, pc, seg):
    """gp_setup for the general model (blocks kind, pa, pb, pc; seg: True at segment starts): the same tables, d
    states (module comment above), and the zero-input tails W_k = g g' + A' W_(k+1) A, v_k = e_(k+1) g + A' v_(k+1)
    with g = -irs_(k+1) F_(k+1)[0, :] and A = F_(k+1) - K_(k+1) F_(k+1)[0, :]"""
    F, K, irs, e, c0, ld = _ss_filter(t, y1, var, valid, yerr, kind, pa, pb, pc, seg, True)
    n = t.shape[0]
    d = K.shape[1]
    nw = d * (d + 1) // 2
    W = np.zeros((n, nw))
    v = np.zeros((n, d))
    Wm = np.zeros((d, d))
    vv = np.zeros(d)
    A = np.empty((d, d))
    g = np.empty(d)
    for i in range(n - 2, -1, -1):
        j = i + 1
        for a in range(d):
            g[a] = -irs[j] * F[j, a]
        for a in range(d):
            for b in range(d):
                A[a, b] = F[j, a * d + b] - K[j, a] * F[j, b]
        Wm = A.T @ Wm @ A
        vv = A.T @ vv
        for a in range(d):
            vv[a] += e[j] * g[a]
            for b in range(d):
                Wm[a, b] += g[a] * g[b]
        q = 0
        for a in range(d):
            for b in range(a, d):
                W[i, q] = 0.5 * (Wm[a, b] + Wm[b, a])
                q += 1
            v[i, a] = vv[a]
    return F, K, irs, e, W, v, c0, ld


@jit(cache=True, nopython=True, fastmath=True)
def _gp_quad_ss(out_idx, Dv, F, K, irs, e, W, v, H):
    """gp_quad with d states (K.shape[1])"""
    nk = out_idx.shape[0]
    if nk == 0:
        return 0.0
    d = K.shape[1]
    m = np.zeros(d)
    mn = np.zeros(d)
    acc = 0.0
    c = out_idx[0]
    start = c
    k = 0
    while True:
        if c > start:
            for a in range(d):
                x = 0.0
                for b in range(d):
                    x += F[c, a * d + b] * m[b]
                mn[a] = x
            for a in range(d):
                m[a] = mn[a]
        dd = 0.0
        if out_idx[k] == c:
            dd = Dv[k]
            k += 1
        inn = dd - m[0]
        dw = inn * irs[c]
        acc += dw * (dw + 2.0 * e[c])
        for a in range(d):
            m[a] += K[c, a] * inn
        if k == nk:
            break
        s = out_idx[k]
        if s - c > H[c]:
            # the gap to the next window cadence is longer than the horizon: the tail of the decaying state, then
            # the next window from state 0
            acc += _tail_ss(W, v, c, m)
            for a in range(d):
                m[a] = 0.0
            c = s
            start = s
        else:
            c += 1
    return acc + _tail_ss(W, v, c, m)


@jit(cache=True, nopython=True, fastmath=True)
def _tail_ss(W, v, c, m):
    """the zero-input tail after cadence c of the filtered D-state m: m' W_c m + 2 v_c' m"""
    d = m.shape[0]
    acc = 0.0
    q = 0
    for a in range(d):
        acc += 2.0 * v[c, a] * m[a]
        for b in range(a, d):
            acc += (1.0 if a == b else 2.0) * W[c, q] * m[a] * m[b]
            q += 1
    return acc


@jit(cache=True, nopython=True)
def _whiten_deficit_ss(Dfull, F, K, irs):
    n = Dfull.shape[0]
    d = K.shape[1]
    out = np.zeros(n)
    m = np.zeros(d)
    mn = np.zeros(d)
    for c in range(n):
        if c > 0:
            for a in range(d):
                x = 0.0
                for b in range(d):
                    x += F[c, a * d + b] * m[b]
                mn[a] = x
            for a in range(d):
                m[a] = mn[a]
        inn = Dfull[c] - m[0]
        out[c] = inn * irs[c]
        for a in range(d):
            m[a] += K[c, a] * inn
    return out


@jit(cache=True, nopython=True)
def _rts_ss(t, r, var, valid, kind, pa, pb, pc, seg):
    """_rts for the general model: posterior mean and variance of z_0 (the GP terms plus the baseline)"""
    n = t.shape[0]
    d = _ss_dim(kind)
    tidx, steps = time_table(t, 1e-6)
    Fz, Qz, P0z = _ss_zmats(kind, pa, pb, pc, steps, d)
    mf = np.zeros((n, d))
    mp = np.zeros((n, d))
    Pf = np.zeros((n, d, d))
    Pp = np.zeros((n, d, d))
    m = np.zeros(d)
    P = np.zeros((d, d))
    for i in range(n):
        if seg[i]:
            m = np.zeros(d)
            P = P0z.copy()
        else:
            Fi = Fz[tidx[i]]
            m = Fi @ m
            P = Fi @ P @ Fi.T + Qz[tidx[i]]
        mp[i] = m
        Pp[i] = P
        if valid[i]:
            S = P[0, 0] + var[i]
            k = P[:, 0] / S
            inn = r[i] - m[0]
            m = m + k * inn
            P = P - np.outer(k, P[0, :])
            P = 0.5 * (P + P.T)
        mf[i] = m
        Pf[i] = P
    mean = np.zeros(n)
    vv = np.zeros(n)
    ms = mf[n - 1].copy()
    Ps = Pf[n - 1].copy()
    mean[n - 1], vv[n - 1] = ms[0], Ps[0, 0]
    for i in range(n - 2, -1, -1):
        if seg[i + 1]:
            ms = mf[i].copy()
            Ps = Pf[i].copy()
        else:
            Fn = Fz[tidx[i + 1]]
            G = np.linalg.solve(Pp[i + 1], Fn @ Pf[i]).T
            ms = mf[i] + G @ (ms - mp[i + 1])
            Ps = Pf[i] + G @ (Ps - Pp[i + 1]) @ G.T
        mean[i], vv[i] = ms[0], Ps[0, 0]
    return mean, vv


@jit(cache=True, nopython=True)
def _sho_sample_q(t, amp, tau, Q, z, seg):
    """GP.sample for one SHO term of quality Q: the exact transition and process noise per step, restarting from
    the stationary distribution at the segment starts"""
    n = t.shape[0]
    out = np.empty(n)
    s2 = amp * amp
    x0 = 0.0
    x1 = 0.0
    for i in range(n):
        if seg[i]:
            x0 = amp * z[i, 0]
            x1 = amp * z[i, 1]
        else:
            f00, f01, f10, f11, q00, q01, q11 = _sho_trans(t[i] - t[i - 1], tau, Q, s2)
            y0 = f00 * x0 + f01 * x1
            y1 = f10 * x0 + f11 * x1
            l00 = math.sqrt(max(q00, 0.0))
            l10 = q01 / l00 if l00 > 0 else 0.0
            l11 = math.sqrt(max(q11 - l10 * l10, 0.0))
            x0 = y0 + l00 * z[i, 0]
            x1 = y1 + l10 * z[i, 0] + l11 * z[i, 1]
        out[i] = x0
    return out


@jit(cache=True, nopython=True)
def chunk_maps(Fm, Km, h, C):
    """products of the D-filter's maps A_c = (I - K_c h') F_c (blocks' coordinates, Fm (n, d, d), Km (n, d)) over the
    cadences j C ... (j + 1) C - 1 (the GPU's pass 1 for d > 2 states)"""
    n, d = Km.shape
    nch = (n + C - 1) // C
    out = np.zeros((nch, d, d))
    A = np.empty((d, d))
    hf = np.empty(d)
    for j in range(nch):
        M = np.eye(d)
        for c in range(j * C, min(n, (j + 1) * C)):
            for b in range(d):
                x = 0.0
                for k in range(d):
                    x += h[k] * Fm[c, k, b]
                hf[b] = x
            for a in range(d):
                for b in range(d):
                    A[a, b] = Fm[c, a, b] - Km[c, a] * hf[b]
            M = A @ M
        out[j] = M
    return out


def _lomb_scargle_peaks(t, y, pmin, pmax, npk=3):
    """periods of the highest peaks of the classical Lomb-Scargle periodogram (floating mean) of y(t), pmin - pmax
    [d] (local maxima, at least 20 % apart in period)"""
    span = float(t[-1] - t[0])
    fmin, fmax = 1.0 / pmax, 1.0 / pmin
    nf = int(min(20000, max(200, 5 * span * (fmax - fmin))))
    f = np.linspace(fmin, fmax, nf)
    y = y - y.mean()
    pw = np.empty(nf)
    for a in range(0, nf, 256):
        w = 2 * np.pi * f[a:a + 256, None]
        tau = np.arctan2(np.sum(np.sin(2 * w * t), 1), np.sum(np.cos(2 * w * t), 1)) / (2 * w[:, 0])
        arg = w * (t - tau[:, None])
        c, s = np.cos(arg), np.sin(arg)
        pw[a:a + 256] = (c @ y) ** 2 / np.sum(c * c, 1) + (s @ y) ** 2 / np.sum(s * s, 1)
    loc = np.where((pw[1:-1] >= pw[:-2]) & (pw[1:-1] >= pw[2:]))[0] + 1
    out = []
    for i in loc[np.argsort(pw[loc])[::-1]]:
        p = 1.0 / f[i]
        if all(abs(p / q - 1) > 0.2 for q in out):
            out.append(p)
        if len(out) == npk:
            break
    return out


def _fit_general(cls, t, y1, yerr_, var0, kernel, fit_s, s_fix, baseline, gap, baseline_sd, n_starts, period, a0,
                 ab, tb, dt_med, span):
    """GP.fit for the general kernels: ML hyperparameters (log space, bounds) with central-difference gradients
    and _minimize_bounded, from several starts"""
    seg = cls(1.0, 1.0, gap=gap).segment_starts(t)
    lA = (math.log(ab[0]), math.log(ab[1]))
    lT = (math.log(tb[0]), math.log(tb[1]))

    def clipz(x, b):
        return min(max(x, b[0] + 1e-3), b[1] - 1e-3)

    if kernel == "sho":
        bounds = [lA, lT]
        build = lambda z: [SHO(math.exp(z[0]), math.exp(z[1]))]                                    # noqa: E731
        starts = [[a0, clipz(math.log(dt_med * 10.0 ** (1 + j)), lT)] for j in range(n_starts)]
    elif kernel == "sho_q":
        bounds = [lA, lT, (math.log(0.2), math.log(50.0))]
        build = lambda z: [SHO(math.exp(z[0]), math.exp(z[1]), math.exp(z[2]))]                   # noqa: E731
        starts = [[a0, clipz(math.log(dt_med * 10.0 ** (1 + j)), lT), math.log(_Q0)] for j in range(n_starts)]
    elif kernel == "sho2":
        bounds = [lA, lT, lA, lT]
        build = lambda z: [SHO(math.exp(z[0]), math.exp(z[1])), SHO(math.exp(z[2]), math.exp(z[3]))]  # noqa: E731
        starts = []
        for j in range(n_starts):
            tl = clipz(math.log(span / (5.0 * 3.0 ** j)), lT)
            ts = clipz(math.log(dt_med * 5.0 * 3.0 ** j), lT)
            starts.append([clipz(a0 - 0.35, lA), tl, clipz(a0 - 1.0, lA), min(ts, tl - 1.0)])
    else:
        pmin, pmax = max(5.0 * dt_med, 0.05), span / 1.2
        if pmax <= pmin:
            raise ValueError("GP.fit(kernel='rotation'): the data span too short a time")
        bounds = [lA, (math.log(pmin), math.log(pmax)), (math.log(0.01), math.log(50.0)),
                  (math.log(1e-3), math.log(100.0)), (math.log(0.01), 0.0), lA, lT]

        def build(z):
            return [Rotation(math.exp(z[0]), math.exp(z[1]), math.exp(z[2]), math.exp(z[3]), math.exp(z[4])),
                    SHO(math.exp(z[5]), math.exp(z[6]))]
        if period is not None:
            cands = [float(period)]
        else:
            pk = _lomb_scargle_peaks(t, y1, pmin, pmax, 2)
            cands = pk + ([2 * pk[0]] if pk and 2 * pk[0] < pmax else [])
        starts = [[a0, clipz(math.log(p), bounds[1]), 0.0, 0.0, math.log(0.5), clipz(a0 - 1.0, lA),
                   clipz(math.log(dt_med * 10.0), lT)] for p in cands]
    if fit_s:
        bounds.append((math.log(1.0 / 3.0), math.log(3.0)))
        starts = [s + [0.0] for s in starts]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    nfev = [0]

    def make(z):
        return cls(terms=build(z), err_scale=math.exp(z[-1]) if fit_s else s_fix, baseline=baseline, gap=gap,
                   baseline_sd=baseline_sd)

    def nll(z):
        nfev[0] += 1
        g = make(z)
        kind, pa, pb, pc = g.blocks()
        return _ss_nll(t, y1, var0 * g.err_scale ** 2, yerr_, kind, pa, pb, pc, seg)

    def fg(z):
        f = nll(z)
        g = np.empty(len(z))
        for i in range(len(z)):
            h = 1e-4
            zp, zm = z.copy(), z.copy()
            zp[i] = min(z[i] + h, hi[i])
            zm[i] = max(z[i] - h, lo[i])
            g[i] = (nll(zp) - nll(zm)) / (zp[i] - zm[i])
        return f, g

    best = None
    for z0 in starts:
        z, f = _minimize_bounded(fg, np.array(z0, dtype=np.float64), lo, hi)
        if best is None or f < best[1]:
            best = (z, f)
    g = make(best[0])
    g.fit_nll, g.fit_nfev, g.fit_points = float(best[1]), int(nfev[0]), int(len(t))
    return g


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
def gp_quad(out_idx, Dv, F, K, irs, e, W, v, H):
    """sum_k Dw_k^2 + 2 e_k Dw_k (Dw = L^-1 D) for the deficit D = Dv at the ascending output cadences out_idx
    (0 elsewhere): the D-filter over the window cadences and the gaps between them shorter than the horizon H
    (gp_horizon), the closed-form tail after each stretch (the next one starts from state 0) (d = K.shape[1]
    states; d > 2: _gp_quad_ss)"""
    if K.shape[1] != 2:
        return _gp_quad_ss(out_idx, Dv, F, K, irs, e, W, v, H)
    nk = out_idx.shape[0]
    if nk == 0:
        return 0.0
    acc = 0.0
    m0 = 0.0
    m1 = 0.0
    c = out_idx[0]
    start = c
    k = 0
    while True:
        if c > start:
            nm0 = F[c, 0] * m0 + F[c, 1] * m1
            m1 = F[c, 2] * m0 + F[c, 3] * m1
            m0 = nm0
        d = 0.0
        if out_idx[k] == c:
            d = Dv[k]
            k += 1
        inn = d - m0
        dw = inn * irs[c]
        acc += dw * (dw + 2.0 * e[c])
        m0 += K[c, 0] * inn
        m1 += K[c, 1] * inn
        if k == nk:
            break
        s = out_idx[k]
        if s - c > H[c]:
            acc += (W[c, 0] * m0 * m0 + 2.0 * W[c, 1] * m0 * m1 + W[c, 2] * m1 * m1
                    + 2.0 * (v[c, 0] * m0 + v[c, 1] * m1))
            m0 = 0.0
            m1 = 0.0
            c = s
            start = s
        else:
            c += 1
    acc += W[c, 0] * m0 * m0 + 2.0 * W[c, 1] * m0 * m1 + W[c, 2] * m1 * m1 + 2.0 * (v[c, 0] * m0 + v[c, 1] * m1)
    return acc


@jit(cache=True, nopython=True)
def gp_horizon(F, K, eps=1e-12, block=16):
    """H[c]: the zero-input steps of the D-filter after cadence c until its transition Phi(c -> c + H) =
    A_(c+H) ... A_(c+1), A_j = (I - K_j e0') F_j, is below eps in the infinity norm (n - c: never within the data).
    gp_quad steps a gap between window cadences c and s only if s - c <= H[c]; otherwise the dropped terms (the
    decaying state's cross terms with the next windows' deficits) are ~eps x the chi^2 terms involved. A segment
    start (F = 0) ends the horizon. Blocked (n block d^3 + n / block H d^3 instead of n H d^3): explicit steps from c
    to the next block start b; beyond it |Phi(c -> j)| <= |Phi(b -> j)| r with r = |Phi(c -> b)|, so H[c] =
    (b - c) + the horizon of b for the tolerance eps / r, from a ladder of tolerances eps 2^k (k = -7 ... 40) kept
    at the block starts: at most a factor 2 (a few steps) beyond the exact horizon; explicit search when r > 128.
    block >= n: the exact horizon everywhere (tests)."""
    n, d = K.shape
    H = np.empty(n, np.int64)
    KLO, KHI = -7, 40
    nb = (n + block - 1) // block
    Hb = np.empty((nb, KHI - KLO + 1), np.int64)
    P = np.empty((d, d))
    Q = np.empty((d, d))

    def step(j, P, Q):
        # P <- A_j P; returns |P| (infinity norm)
        for i in range(d):
            for k in range(d):
                x = 0.0
                for l in range(d):
                    x += F[j, i * d + l] * P[l, k]
                Q[i, k] = x
        big = 0.0
        for i in range(d):
            r = 0.0
            for k in range(d):
                P[i, k] = Q[i, k] - K[j, i] * Q[0, k]
                r += abs(P[i, k])
            big = max(big, r)
        return big

    def eye(P):
        for i in range(d):
            for k in range(d):
                P[i, k] = 1.0 if i == k else 0.0

    def search(c, lim, tol, P, Q):
        # the first j in (c, lim] with |Phi(c -> j)| < tol (-1 if none); P = Phi(c -> min(j, lim))
        eye(P)
        for j in range(c + 1, lim + 1):
            if step(j, P, Q) < tol:
                return j
        return -1

    if block < n:
        for ib in range(nb):
            b = ib * block
            for kk in range(KHI - KLO + 1):
                Hb[ib, kk] = n - b                   # not reached within the data
            eye(P)
            kc = KHI                                 # the loosest tolerance not reached yet
            for j in range(b + 1, n):
                nrm = step(j, P, Q)
                while kc >= KLO and nrm < eps * 2.0 ** kc:
                    Hb[ib, kc - KLO] = j - b
                    kc -= 1
                if kc < KLO:
                    break
    for c in range(n):
        b2 = (c // block + 1) * block
        if c % block == 0 or b2 >= n:
            j = search(c, n - 1, eps, P, Q)
            H[c] = (j - c) if j >= 0 else n - c
            continue
        j = search(c, b2, eps, P, Q)
        if j >= 0:
            H[c] = j - c
            continue
        r = 0.0
        for i in range(d):
            x = 0.0
            for k in range(d):
                x += abs(P[i, k])
            r = max(r, x)
        if r > 2.0 ** -KLO:
            j = search(c, n - 1, eps, P, Q)
            H[c] = (j - c) if j >= 0 else n - c
            continue
        k = min(KHI, int(math.floor(-math.log2(max(r, 1e-300)))))
        hb = Hb[b2 // block, k - KLO]
        H[c] = (b2 - c) + hb if b2 + hb < n else n - c
    return H


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
    F, K, irs, e, W, v, H, c0, ld, cache, time_sorted
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
    return -0.5 * (c0 + gp_quad(out_idx, Dv, F, K, irs, e, W, v, H)) - ld


@jit(cache=True, nopython=True, fastmath=True)
def pandora_loglike_planet_gp(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, epoch_distance, supersampling_factor,
    occult_small_threshold, hill_sphere_threshold, time,
    F, K, irs, e, W, v, H, c0, ld, cache, time_sorted
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
            hill_sphere_threshold, time, F, K, irs, e, W, v, H, c0, ld, cache, time_sorted
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
    return -0.5 * (c0 + gp_quad(out_idx, Dv, F, K, irs, e, W, v, H)) - ld


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
    if K.shape[1] != 2:
        return _whiten_deficit_ss(Dfull, F, K, irs)
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
