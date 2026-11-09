import numpy as np
from numba import float64

from pandoramoon.pandora import pandora_loglike


def _compiled_loglike(tail):
    """P8: the compiled pandora_loglike for 20 float64 model parameters and
    this dataset's fixed arguments (tail), i.e. its entry point. Calling it
    skips numba's per-call type dispatch over all 32 arguments (~0.4 us per
    call); the parameters are converted to float like float(x). Falls back
    to the dispatcher if the entry point cannot be obtained."""
    try:
        sig = (float64,) * 20 + tuple(pandora_loglike.typeof_pyval(x) for x in tail)
        return pandora_loglike.compile(sig)
    except Exception:  # pragma: no cover - e.g. a numba without compile()
        return pandora_loglike


class Likelihood(object):
    """Gaussian log-likelihood of a light curve for samplers (e.g. UltraNest).

    Precomputes the data terms once; each call evaluates the Pandora model
    only near transit (no full-length arrays are allocated per call):

        like = Likelihood(time, data, yerr, epoch_distance=..., ...)
        logL = like(u1, u2, R_star, per_bary, ..., M_moon)   # same order as pandora()

    time is the (supersampled) time grid as for pandora(); data and yerr are
    at output cadence (len(time) / supersampling_factor, rounded up). The
    result equals -0.5 * nansum(((flux_total - data) / yerr) ** 2): data or
    yerr that are NaN/inf get weight 0. All arithmetic is float64.

    For plots and videos use pandora_flux() (fluxes) or pandora()
    (fluxes and coordinates).

    The 20 model parameters are taken as float64 (any value that float()
    accepts); the call goes straight to the compiled model (P8).

    gp: None (default: white noise, as above) or a pandoramoon.gp.GP with fixed
    hyperparameters (amp, tau, err_scale): simultaneous GP detrending, the
    light curve is fitted as model + GP + white noise instead of being
    detrended first (pandoramoon/gp.py). The data must then be the raw
    (normalised, not detrended) flux, in ascending time. The same GP is used
    by the planet-only model of NestedSampler.
    """

    def __init__(
        self,
        time,
        data,
        yerr,
        epoch_distance,
        supersampling_factor=1,
        occult_small_threshold=None,
        hill_sphere_threshold=None,
        numerical_grid=None,
        cache=None,
        eclipse_method="analytic",
        gp=None,
    ):
        # numerical_grid and eclipse_method="pixelmap" (the legacy pixel raster for limb eclipses) were removed in
        # 2026-10; the names stay so that positional calls keep working
        from pandoramoon._config import settings
        if occult_small_threshold is None:
            occult_small_threshold = settings()["model"]["occult_small_threshold"]
        if hill_sphere_threshold is None:
            hill_sphere_threshold = settings()["model"]["hill_sphere_threshold"]
        if eclipse_method != "analytic":
            raise ValueError('eclipse_method: only "analytic" (the pixel raster "pixelmap" was removed in 2026-10)')
        if numerical_grid is not None:
            import warnings
            warnings.warn("Likelihood(numerical_grid=...) is ignored: the pixel raster was removed in 2026-10",
                          DeprecationWarning, stacklevel=2)
        supersampling_factor = int(supersampling_factor)
        if supersampling_factor < 1:
            raise ValueError("supersampling_factor must be a positive integer")
        self.time = np.ascontiguousarray(time, dtype=np.float64)
        n = len(self.time)
        # Ascending time (no NaN): the transit windows are found by binary
        # search instead of a pass over all samples
        self.time_sorted = bool(n < 2 or np.all(self.time[1:] >= self.time[:-1]))
        n_out = n if supersampling_factor == 1 else -(-n // supersampling_factor)
        data = np.asarray(data, dtype=np.float64)
        yerr = np.broadcast_to(np.asarray(yerr, dtype=np.float64), data.shape)
        if data.shape != (n_out,):
            raise ValueError(
                "data has %d values, expected %d (time %d / supersampling_factor %d)"
                % (data.size, n_out, n, supersampling_factor)
            )
        valid = np.isfinite(data) & np.isfinite(yerr)
        if np.any(yerr[valid] <= 0):
            raise ValueError("yerr must be positive")
        self.data = np.ascontiguousarray(np.where(valid, data, 1.0))
        self.inv_err = np.ascontiguousarray(np.where(valid, 1.0 / np.where(valid, yerr, 1.0), 0.0))
        flat_terms = ((1.0 - self.data) * self.inv_err) ** 2
        self.chi2_flat = float(np.sum(flat_terms))
        # P13b: per cadence chi^2 - flat term = D (inv_var D - lin_terms), D = 1 - model
        self.inv_var = self.inv_err ** 2
        self.lin_terms = 2.0 * (1.0 - self.data) * self.inv_var
        self.epoch_distance = float(epoch_distance)
        self.supersampling_factor = supersampling_factor
        self.occult_small_threshold = float(occult_small_threshold)
        self.hill_sphere_threshold = float(hill_sphere_threshold)
        self.cache = cache
        # P8: fixed trailing arguments and the compiled entry point
        self._tail = (
            self.epoch_distance, self.supersampling_factor, self.occult_small_threshold,
            self.hill_sphere_threshold, self.time,
            self.inv_var, self.lin_terms, self.chi2_flat, self.cache,
            self.time_sorted,
        )
        self._loglike = _compiled_loglike(self._tail)
        self.gp = gp
        self.gp_tables = None
        if gp is not None:
            from pandoramoon.gp import GP, gp_setup, output_times
            if not isinstance(gp, GP):
                raise TypeError("gp must be a pandoramoon.gp.GP (or None)")
            if not self.time_sorted:
                raise ValueError("Likelihood(gp=...): the time must be ascending")
            t_out = output_times(self.time, supersampling_factor)
            if len(t_out) != n_out:
                raise ValueError("Likelihood(gp=...): cadence times do not match the data")
            yerr_v = np.ascontiguousarray(np.where(valid, yerr, 1.0))
            y1 = np.ascontiguousarray(np.where(valid, data - 1.0, 0.0))
            self.time_out = t_out
            self.gp_tables = gp_setup(t_out, y1, (gp.err_scale * yerr_v) ** 2, np.ascontiguousarray(valid),
                                      yerr_v, gp.amp, gp.tau)
            self._gp_tail = (
                self.epoch_distance, self.supersampling_factor, self.occult_small_threshold,
                self.hill_sphere_threshold, self.time) + tuple(self.gp_tables) + (self.cache, self.time_sorted)

    def __call__(
        self, u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
        i_moon, ecc_moon, w_moon, M_moon
    ):
        if self.gp is not None:
            from pandoramoon.gp import pandora_loglike_gp
            return pandora_loglike_gp(
                float(u1), float(u2), float(R_star), float(per_bary), float(a_bary), float(r_planet),
                float(b_bary), float(w_bary), float(ecc_bary), float(t0_bary), float(t0_bary_offset),
                float(M_planet), float(r_moon), float(per_moon), float(tau_moon), float(Omega_moon),
                float(i_moon), float(ecc_moon), float(w_moon), float(M_moon), *self._gp_tail)
        return self._loglike(
            u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
            t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
            Omega_moon, i_moon, ecc_moon, w_moon, M_moon, *self._tail
        )

    loglike = __call__
