"""First-look exomoon screening with Pandora (example; not part of the package core).

Idea: a full planet+moon retrieval with self-checks costs ~1e9 likelihood calls. Screening many light
curves needs something ~100x cheaper that still flags nearly all well-detectable moons. A first look
runs ONE short nested-sampling run per model (planet only, planet+moon) with the compiled sampler
`pandoramoon.nss` in the phase-at-transit moon coordinates (`pandoramoon.reparam`), and reports

    dlnZ = lnZ(planet+moon) - lnZ(planet only)

with a verdict "interesting" if dlnZ exceeds a threshold calibrated on injections (bench_firstlook.py),
plus a rough summary of the moon parameters from the same run. Interesting light curves then go to a
full fit (nss.fit). The numbers of the calibration are in README.md.

    import firstlook
    fl = firstlook.FirstLook(time, flux, flux_err, t0_bary=11.0, per_bary=(360.25, 370.25), a_bary=(210, 220))
    r = fl.run()                      # one light curve, one core, a few seconds
    r["dlnz"], r["interesting"], r["moon"]

    results = firstlook.run_many([fl1, fl2, ...], threads=16)   # many light curves, one per core

All compiled functions take the data as an argument, so numba compiles once per process (~20-40 s) and
not per light curve.

Model and priors (the notebook's injection-retrieval example; override with `priors=`):
  planet only : per_bary, a_bary [R_star], r_planet [R_star], b_bary, t0_bary_offset [d], q1, q2
  planet+moon : R_star [m], per_bary, a_bary, r_planet, b_bary, t0_bary_offset, M_planet [kg], r_moon [R_star],
                (phi_c, delta, u_m) <-> (per_moon [d], tau), Omega_moon [deg, 0-360 double cover], i_moon [deg],
                M_moon [kg], q1, q2
Both models use the same priors for the shared parameters, so dlnZ is a proper Bayes factor.
"""
import math
import time as _time

import numpy as np
from numba import njit

import pandoramoon as pandora
from pandoramoon import nss
from pandoramoon.pandora import pandora_loglike, pandora_loglike_planet
from pandoramoon.reparam import LOGZERO, Reparam, min_physical_period

R_SUN = 696_342_000.0

DEFAULT_PRIORS = dict(
    R_star=(0.4 * R_SUN, 1.2 * R_SUN), r_planet=(0.07, 0.11), b_bary=(0.0, 1.0), t0_bary_offset=(-0.05, 0.05),
    M_planet=(1.5e27, 2.5e27), r_moon=(0.01, 0.07), per_moon_max=50.0, per_moon_lo=0.05, i_moon=(0.0, 180.0),
    M_moon=(1.0, 1e27 + 1.0), q1=(0.0, 1.0), q2=(0.0, 1.0))

# first-look settings ("A" in README.md: recall 100 % of 166 moons with reference dlnZ >= 8, 0 of 160 no-moon
# light curves above 0; white-noise injections, bench_firstlook.py). threshold 2: a 3-nat margin above the
# largest no-moon value (-1.1)
DEFAULT_SETTINGS = dict(nlive_planet=200, nsteps_planet=28, nlive_moon=200, nsteps_moon=32, repeats=1, threshold=2.0)

# recall presets: target recall of well-detectable moons (reference dlnZ >= 8) -> settings. Repeated small moon
# runs are cheaper than one large run for the same recall (recall_study.py, README.md "Choosing the recall").
RETRY_BELOW, RETRY_MAX = -15.0, 2
DEFAULT_RECALL = 0.95
# Round 12 (study2: smaller runs): planet run 50 live / nsteps 12; moon runs 50 live / nsteps 8 (0.90, 0.95) or
# 50 live / nsteps 16 (0.99, 0.999)
_P = dict(nlive_planet=50, nsteps_planet=12, threshold=3.0)
# (validate3: 70 live / nsteps 8 x 3 / x 6 reached only 99.35 / 99.87 %: misses of short chains are correlated
# within a light curve, so 0.99 / 0.999 keep nsteps 16)
RECALL_PRESETS = {0.90: dict(_P, nlive_moon=50, nsteps_moon=8, repeats=1),
                  0.95: dict(_P, nlive_moon=50, nsteps_moon=8, repeats=2),
                  0.99: dict(_P, nlive_moon=50, nsteps_moon=16, repeats=3),
                  0.999: dict(_P, nlive_moon=50, nsteps_moon=16, repeats=6)}

MOON_NAMES = ['R_star', 'per_bary', 'a_bary', 'r_planet', 'b_bary', 't0_bary_offset', 'M_planet', 'r_moon',
              'phi_c', 'delta', 'u_m', 'Omega_moon', 'i_moon', 'M_moon', 'q1', 'q2']
PLANET_NAMES = ['per_bary', 'a_bary', 'r_planet', 'b_bary', 't0_bary_offset', 'q1', 'q2']


# ---------------------------------------------------------------------------------------------
# compiled likelihoods: ll(u, D) with u in the unit cube and D a tuple of data and prior constants

@njit(cache=True)
def _m_of_u(u, CDF, M_MIN):
    lo, hi = 0, CDF.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if CDF[mid] > u:
            hi = mid
        else:
            lo = mid + 1
    return M_MIN + lo


@njit(cache=True)
def _to_native(per_bary, t0_bary, t0_off, phi_c, delta, u_m, e_c, CDF, M_MIN, Q, P_MAX):
    """pandoramoon.reparam.Reparam.to_native with the tables as arguments"""
    m = _m_of_u(u_m, CDF, M_MIN)
    N = m + delta
    per_moon = per_bary / N
    t_c = t0_bary + t0_off + e_c * per_bary
    x = t_c * N / per_bary - phi_c
    tau = x - math.floor(x)
    if per_moon > P_MAX:
        return per_moon, tau, LOGZERO
    return per_moon, tau, math.log(per_bary / P_MAX * Q * m * (m + 1.0) / (N * N))


@njit(fastmath=False)
def ll_moon(u, D):
    lo, span, T0, E_C, CDF, M_MIN, Q, P_MAX, ed, ssf, ost, hst, tgrid, inv_var, lin, chi2f, tsorted = D
    th = u * span + lo
    pm, tau, logw = _to_native(th[1], T0, th[5], th[8], th[9], th[10], E_C, CDF, M_MIN, Q, P_MAX)
    if not logw > -1e300:
        return -1e300
    sq = math.sqrt(th[14])
    return logw + pandora_loglike(2.0 * sq * th[15], sq * (1.0 - 2.0 * th[15]), th[0], th[1], th[2], th[3], th[4],
                                  0.0, 0.0, T0, th[5], th[6], th[7], pm, tau, th[11], th[12], 0.0, 0.0, th[13],
                                  ed, ssf, ost, hst, tgrid, inv_var, lin, chi2f, None, tsorted)


@njit(fastmath=False)
def ll_planet(u, D):
    lo, span, T0, ed, ssf, ost, hst, tgrid, inv_var, lin, chi2f, tsorted = D
    th = u * span + lo
    sq = math.sqrt(th[5])
    # pandora_loglike_planet: Pandora's planet-only path (moon orbit inside the planet) as one fused kernel; same
    # logL as a dummy physical moon up to rounding, ~2.4x faster per call (SAMPLING.md Round 12)
    return pandora_loglike_planet(2.0 * sq * th[6], sq * (1.0 - 2.0 * th[6]), R_SUN, th[0], th[1], th[2], th[3],
                                  0.0, 0.0, T0, th[4], 1e27, ed, ssf, ost, hst, tgrid, inv_var, lin, chi2f,
                                  None, tsorted)


# ---------------------------------------------------------------------------------------------

class FirstLook:
    def __init__(self, time, flux, flux_err, t0_bary, per_bary, a_bary, epoch_distance=None, priors=None,
                 supersampling_factor=1):
        """time [d], flux (normalised), flux_err: the light curve (windows around the transits).
        t0_bary [d]: reference transit time (the fit's t0_bary_offset is relative to it).
        per_bary = (lo, hi) [d], a_bary = (lo, hi) [R_star]: uniform priors of the planet's period and
        semi-major axis (usually narrow: the planet is known). priors: overrides of DEFAULT_PRIORS."""
        pr = dict(DEFAULT_PRIORS)
        pr.update(priors or {})
        pr["per_bary"], pr["a_bary"] = tuple(map(float, per_bary)), tuple(map(float, a_bary))
        self.priors = pr
        self.t0_bary = float(t0_bary)
        ed = float(epoch_distance) if epoch_distance is not None else 0.5 * sum(pr["per_bary"]) + 0.01
        self.like = like = pandora.Likelihood(np.asarray(time, float), np.asarray(flux, float),
                                              np.asarray(flux_err, float), epoch_distance=ed,
                                              supersampling_factor=supersampling_factor, occult_small_threshold=0.0,
                                              hill_sphere_threshold=1.0)
        tail = like._tail
        assert tail[8] is None                                # the occultation cache (not used)
        self._tail = tail[:8] + tail[9:]
        ep = np.unique(np.floor((np.asarray(time) - self.t0_bary) / ed + 0.5).astype(int))
        self.e_c = int(round(float(np.mean(ep))))
        self.ndata = len(time)
        # planet+moon: unit cube -> parameters
        b = lambda k: pr[k]                                   # noqa: E731
        lo = np.array([b("R_star")[0], b("per_bary")[0], b("a_bary")[0], b("r_planet")[0], b("b_bary")[0],
                       b("t0_bary_offset")[0], b("M_planet")[0], b("r_moon")[0], 0, 0, 0, 0, b("i_moon")[0],
                       b("M_moon")[0], b("q1")[0], b("q2")[0]], dtype=float)
        hi = np.array([b("R_star")[1], b("per_bary")[1], b("a_bary")[1], b("r_planet")[1], b("b_bary")[1],
                       b("t0_bary_offset")[1], b("M_planet")[1], b("r_moon")[1], 1, 1, 1, 360, b("i_moon")[1],
                       b("M_moon")[1], b("q1")[1], b("q2")[1]], dtype=float)
        self.lo_m, self.span_m = lo, hi - lo
        self.wrapped_m = np.zeros(16, dtype=np.bool_)
        self.wrapped_m[[8, 9, 11]] = True                     # phi_c, delta, Omega (double cover)
        self.rp = rp = Reparam(pr["per_bary"][0], pr["per_bary"][1], p_max=pr["per_moon_max"], p_lo=pr["per_moon_lo"],
                               e_c=self.e_c)
        p_phys = min_physical_period(R_star_max=hi[0], rsum_min=lo[3] + lo[7], M_total_max=hi[6] + hi[13])
        if not pr["per_moon_lo"] < p_phys:
            raise ValueError(f"per_moon_lo must be below {p_phys:.4f} d (shortest physical moon period)")
        self.D_m = (lo, self.span_m, self.t0_bary, float(self.e_c), rp.CDF, int(rp.M_MIN), float(rp.Q),
                    float(rp.P_MAX)) + self._tail
        # planet only
        idx = [1, 2, 3, 4, 5, 14, 15]
        self.lo_p, self.span_p = lo[idx].copy(), self.span_m[idx].copy()
        self.D_p = (self.lo_p, self.span_p, self.t0_bary) + self._tail

    # -- helpers --
    def moon_native(self, U):
        """unit-cube moon-model samples -> dict of physical arrays (per_moon, tau folded with Omega)"""
        th = self.lo_m + np.atleast_2d(U) * self.span_m
        q = self.rp.new_to_native_samples(th, t0_bary=self.t0_bary, e_c=self.e_c)
        names = ['R_star', 'per_bary', 'a_bary', 'r_planet', 'b_bary', 't0_bary_offset', 'M_planet', 'r_moon',
                 'per_moon', 'tau', 'Omega_moon', 'i_moon', 'M_moon', 'q1', 'q2']
        return {n: q[:, i] for i, n in enumerate(names)}

    def run_planet(self, nlive, nsteps, seed=1):
        c0 = _time.thread_time()
        r = nss.run(ll_planet, 7, nlive=nlive, nsteps=nsteps, seed=seed, dirs="pairs", kpair=4, data=self.D_p)
        r["cpu"] = _time.thread_time() - c0
        return r

    def run_moon(self, nlive, nsteps, seed=1):
        c0 = _time.thread_time()
        r = nss.run(ll_moon, 16, wrapped=self.wrapped_m, nlive=nlive, nsteps=nsteps, seed=seed, dirs="pairs", kpair=4,
                    data=self.D_m)
        r["cpu"] = _time.thread_time() - c0
        return r

    def run(self, recall=DEFAULT_RECALL, nlive_planet=None, nsteps_planet=None, nlive_moon=None, nsteps_moon=None, repeats=None,
            seed=1, threshold=None, keep_samples=False):
        """first look of this light curve (serial, one core).
        recall: target recall of well-detectable moons (0.9, 0.95 = default, 0.99, 0.999): picks the settings from
        RECALL_PRESETS (README.md); explicit settings override it; recall=None: DEFAULT_SETTINGS (200 live points,
        one moon run: the first calibration's "A").
        One planet-only run and `repeats` independent planet+moon runs; lnZ(moon) = log mean Z of the moon runs
        (a run that missed the moon's mode then costs only ln(repeats)).
        Returns dict: dlnz, logz_planet, logz_moon, interesting (dlnz > threshold), moon (median and 16/84 % of
        the main moon parameters, from the best moon run), maxL point, calls, cpu seconds."""
        st = dict(DEFAULT_SETTINGS)
        if recall is not None:
            if recall not in RECALL_PRESETS:
                raise ValueError(f"recall must be one of {sorted(RECALL_PRESETS)}")
            st.update(RECALL_PRESETS[recall])
        for k, v in (("nlive_planet", nlive_planet), ("nsteps_planet", nsteps_planet), ("nlive_moon", nlive_moon),
                     ("nsteps_moon", nsteps_moon), ("repeats", repeats), ("threshold", threshold)):
            if v is not None:
                st[k] = v
        t0 = _time.perf_counter()
        rp_ = self.run_planet(st["nlive_planet"], st["nsteps_planet"], seed=seed)
        rms = [self.run_moon(st["nlive_moon"], st["nsteps_moon"], seed=seed + 7 + 101 * j) for j in range(st["repeats"])]
        lzm = float(np.logaddexp.reduce([r["logz"] for r in rms]) - math.log(len(rms)))
        # safeguard: the moon model (almost) contains the planet-only model, so a correct moon run is never far below
        # it (no-moon light curves: dlnZ >= -10.4 in 99.9 % of the cases). dlnZ < RETRY_BELOW means a collapsed run
        # (~0.7 % of the runs with 50 live points): add moon runs (at most RETRY_MAX)
        nretry = 0
        while lzm - rp_["logz"] < RETRY_BELOW and nretry < RETRY_MAX:
            nretry += 1
            rms.append(self.run_moon(st["nlive_moon"], st["nsteps_moon"], seed=seed + 7 + 101 * (st["repeats"] + nretry)))
            # the collapsed runs carry no information: keep the best ones
            good = [r for r in rms if r["logz"] - rp_["logz"] >= RETRY_BELOW] or rms
            lzm = float(np.logaddexp.reduce([r["logz"] for r in good]) - math.log(len(good)))
        wall = _time.perf_counter() - t0
        rm = max(rms, key=lambda r: r["logz"])
        dl = lzm - rp_["logz"]
        nat = self.moon_native(rm["samples_u"])
        summ = {}
        for n in ("per_moon", "r_moon", "i_moon", "Omega_moon"):
            summ[n] = tuple(float(v) for v in np.percentile(nat[n], [50, 16, 84]))
        mr = nat["M_moon"] / nat["M_planet"]
        summ["M_moon/M_planet"] = tuple(float(v) for v in np.percentile(mr, [50, 16, 84]))
        best = self.moon_native(rm["dead_u"][np.argmax(rm["dead_L"])])
        cpu_m = sum(r["cpu"] for r in rms)
        out = dict(dlnz=float(dl), dlnz_err=float(math.hypot(rm["logzerr"], rp_["logzerr"])),
                   logz_planet=rp_["logz"], logz_moon=lzm, logz_moon_runs=[r["logz"] for r in rms],
                   maxL_planet=rp_["maxL"], maxL_moon=rm["maxL"], dmaxL=float(rm["maxL"] - rp_["maxL"]),
                   interesting=(None if st["threshold"] is None else bool(dl > st["threshold"])),
                   moon=summ, best={k: float(v[0]) for k, v in best.items()},
                   ncall_planet=rp_["ncall"], ncall_moon=sum(r["ncall"] for r in rms),
                   ncall=rp_["ncall"] + sum(r["ncall"] for r in rms),
                   wall=wall, cpu_planet=rp_["cpu"], cpu_moon=cpu_m, cpu=rp_["cpu"] + cpu_m, settings=st,
                   ndata=self.ndata, insertion_p_moon=rm["insertion"].get("p"), retries=nretry)
        if keep_samples:
            out["samples_moon"] = nat
        return out


def warmup():
    """compile everything once (call before using threads)"""
    t = np.linspace(10.0, 12.0, 96)
    fl = FirstLook(t, np.ones_like(t), np.full_like(t, 1e-3), t0_bary=11.0, per_bary=(360.0, 370.0),
                   a_bary=(210.0, 220.0))
    nss.run(ll_planet, 7, nlive=40, nsteps=2, max_iter=3, dirs="pairs", data=fl.D_p)
    nss.run(ll_moon, 16, wrapped=fl.wrapped_m, nlive=40, nsteps=2, max_iter=3, dirs="pairs", data=fl.D_m)


def run_many(firstlooks, threads=16, seeds=None, **kw):
    """first looks of many light curves in parallel threads (one light curve per thread; the compiled
    sampler releases the GIL). Returns the list of results in input order."""
    from concurrent.futures import ThreadPoolExecutor
    warmup()
    seeds = seeds or [1 + 1000 * i for i in range(len(firstlooks))]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(f.run, seed=s, **kw) for f, s in zip(firstlooks, seeds)]
        return [f.result() for f in futs]
