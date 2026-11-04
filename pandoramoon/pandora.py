"""Pandora's public model functions: pandora() (fluxes and sky positions), pandora_flux(), pandora_deficit(),
pandora_window(), and the class-based interface moon_model. The parameter container and range check are in
params.py, the compiled model core and the log-likelihoods in kernel.py (re-exported here, so that
pandoramoon.pandora.pandora_loglike etc. keep working)."""
from numba import jit

from pandoramoon.params import PARAMS_ERROR, params_valid, check_params, model_params, time  # noqa: F401
from pandoramoon.kernel import (  # noqa: F401
    NEG_INF,
    pandora_loglike,
    PLANET_ONLY_MOON,
    pandora_loglike_planet,
    _moon_setup,
    _orbit,
    _orbit_z,
    _cull_reach,
    _cull_window,
    _window_fluxes,
    _window_fluxes_z,
    _scatter,
    _total_and_resample,
    _window_model,
    _idx_runs,
    _window_model_runs,
    _n_out,
    _window_cadences,
    _window_chi2_runs,
    _window_chi2_runs_planet,
    _densify,
    _densify_deficit32,
)
from pandoramoon.grids import x_bary_grid  # noqa: F401


class moon_model(object):
    def __init__(self, params):

        # Star parameters
        self.u1 = params.u1
        self.u2 = params.u2
        self.R_star = params.R_star

        # Planet parameters
        self.per_bary = params.per_bary
        self.a_bary = params.a_bary
        self.r_planet = params.r_planet
        self.b_bary = params.b_bary
        self.w_bary = params.w_bary
        self.ecc_bary = params.ecc_bary
        self.t0_bary = params.t0_bary
        self.t0_bary_offset = params.t0_bary_offset
        self.M_planet = params.M_planet

        # Moon parameters
        self.r_moon = params.r_moon
        self.per_moon = params.per_moon
        self.tau_moon = params.tau_moon
        self.Omega_moon = params.Omega_moon
        self.i_moon = params.i_moon
        self.ecc_moon = params.ecc_moon
        self.w_moon = params.w_moon
        self.M_moon = params.M_moon

        # Other model parameters
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor
        self.occult_small_threshold = params.occult_small_threshold
        self.hill_sphere_threshold = params.hill_sphere_threshold
        if getattr(params, "eclipse_method", "analytic") != "analytic":
            raise ValueError('eclipse_method: only "analytic" (the pixel raster "pixelmap" was removed in 2026-10)')

    def _evaluate(self, time, cache=None):
        """pandora() for this model at the given (supersampled) time grid"""
        return pandora(
            self.u1, self.u2, self.R_star,
            self.per_bary, self.a_bary, self.r_planet, self.b_bary, self.w_bary, self.ecc_bary, self.t0_bary,
            self.t0_bary_offset, self.M_planet,
            self.r_moon, self.per_moon, self.tau_moon, self.Omega_moon, self.i_moon, self.ecc_moon, self.w_moon,
            self.M_moon,
            self.epoch_distance, self.supersampling_factor, self.occult_small_threshold, self.hill_sphere_threshold,
            time, cache)

    def video(self, time, limb_darkening=True, teff=6000, planet_color="black", moon_color="black", ld_circles=100):
        """matplotlib animation of the transit (pandoramoon.video.transit_video); save it with
        video.save("video.mp4", fps=25, dpi=200). Also sets self.flux_planet, flux_moon, flux_total, px, py, mx, my."""
        from pandoramoon.video import transit_video
        (self.flux_planet, self.flux_moon, self.flux_total,
         self.px, self.py, self.mx, self.my) = self._evaluate(time)
        return transit_video(self.px, self.py, self.mx, self.my, self.u1, self.u2, self.r_planet, self.r_moon,
                             limb_darkening=limb_darkening, teff=teff, planet_color=planet_color,
                             moon_color=moon_color, ld_circles=ld_circles)

    def light_curve(self, time):
        """total, planet and moon flux at the output cadences"""
        flux_planet, flux_moon, flux_total, px, py, mx, my = self._evaluate(time)
        return flux_total, flux_planet, flux_moon

    def coordinates(self, time):
        """sky positions of the planet (px, py) and the moon (mx, my) [R_star] at every (supersampled) time"""
        flux_planet, flux_moon, flux_total, px, py, mx, my = self._evaluate(time)
        return px, py, mx, my


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    time,
    cache=None,
):
    # limb mutual eclipses via the exact analytic three-circle overlap (eclipse.py, overlap_analytic.py)
    if not params_valid(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        supersampling_factor
    ):
        raise ValueError(PARAMS_ERROR)

    # Make sure to work with floats. Large values as ints would overflow.
    R_star = float(R_star)
    per_bary = float(per_bary)
    a_bary = float(a_bary)
    r_planet = float(r_planet)
    b_bary = float(b_bary)
    t0_bary = float(t0_bary)
    t0_bary_offset = float(t0_bary_offset)
    M_planet = float(M_planet)
    r_moon = float(r_moon)
    per_moon = float(per_moon)
    tau_moon = float(tau_moon)
    Omega_moon = float(Omega_moon)
    i_moon = float(i_moon)
    M_moon = float(M_moon)

    a_moon, unphysical = _moon_setup(
        R_star, per_bary, a_bary, M_planet, M_moon, per_moon, ecc_moon,
        r_planet, r_moon, hill_sphere_threshold
    )
    x_bary = x_bary_grid(
        time, 
        a_bary, 
        per_bary, 
        t0_bary, 
        t0_bary_offset, 
        epoch_distance, 
        ecc_bary, 
        w_bary
    )
    xm, ym, xp, yp = _orbit(
        time, x_bary, unphysical, a_moon, per_moon, tau_moon, Omega_moon,
        w_moon, i_moon, ecc_moon, M_moon / M_planet, b_bary
    )

    # Occultations and mutual eclipses only for samples that can be in
    # transit; all others have flux exactly 1 (see _cull_window)
    idx = _cull_window(
        x_bary, _cull_reach(a_moon, ecc_moon, r_planet, r_moon, unphysical, cache)
    )
    if len(idx) == len(time):
        flux_planet, flux_moon = _window_fluxes(
            xp, yp, xm, ym, unphysical, r_planet, r_moon, u1, u2,
            occult_small_threshold, cache
        )
    else:
        flux_planet_w, flux_moon_w = _window_fluxes(
            xp[idx], yp[idx], xm[idx], ym[idx], unphysical, r_planet, r_moon,
            u1, u2, occult_small_threshold, cache
        )
        flux_planet, flux_moon = _scatter(len(time), idx, flux_planet_w, flux_moon_w)
    flux_planet, flux_moon, flux_total = _total_and_resample(
        flux_planet, flux_moon, supersampling_factor
    )
    return flux_planet, flux_moon, flux_total, xp, yp, xm, ym


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_flux(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    time,
    cache=None,
):
    """Same arguments and fluxes as pandora(), but returns only
    (flux_planet, flux_moon, flux_total) and computes the orbit only for
    samples that can be in transit. float64, full length (output cadence).
    For likelihoods, Likelihood (no full-length arrays) or pandora_deficit()
    (float32) are faster."""
    if not params_valid(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        supersampling_factor
    ):
        raise ValueError(PARAMS_ERROR)
    idx, flux_planet_w, flux_moon_w = _window_model(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, time,
        cache, False
    )
    out_idx, fp, fm, ft = _window_cadences(
        len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor
    )
    n_out = _n_out(len(time), supersampling_factor)
    return (
        _densify(n_out, out_idx, fp, 1.0),
        _densify(n_out, out_idx, fm, 1.0),
        _densify(n_out, out_idx, ft, 1.0),
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_deficit(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    time,
    cache=None,
):
    """Same arguments as pandora_flux(), but returns the flux deficits
    (1 - flux_planet, 1 - flux_moon, 1 - flux_total) as float32 arrays
    (full length, output cadence; exactly 0 out of transit).

    Half the memory of float64 flux arrays. The deficit, unlike the flux,
    keeps float32's full relative precision (2^-24 ~ 6e-8 of the deficit
    itself, i.e. ~1e-9 absolute for a 1 % transit); float32 flux would be
    quantised to 6e-8 absolute near 1. All computations are float64; only
    the stored result is rounded. In a chi^2 use (1 - data) - deficit."""
    if not params_valid(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        supersampling_factor
    ):
        raise ValueError(PARAMS_ERROR)
    idx, flux_planet_w, flux_moon_w = _window_model(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, time,
        cache, False
    )
    out_idx, fp, fm, ft = _window_cadences(
        len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor
    )
    n_out = _n_out(len(time), supersampling_factor)
    return (
        _densify_deficit32(n_out, out_idx, fp),
        _densify_deficit32(n_out, out_idx, fm),
        _densify_deficit32(n_out, out_idx, ft),
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_window(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    time,
    cache=None,
    time_sorted=False
):
    """Sparse model output: same arguments as pandora_flux(), returns
    (idx, flux_planet, flux_moon, flux_total), where idx (int64, ascending)
    are the output cadences near transit and the float64 fluxes are given
    there only. All other output cadences have flux exactly 1 (no full-length
    array is created). pandora_flux() is this, densified.
    time_sorted=True: time is ascending (not checked); the window is then
    found by binary search per epoch (same result, no pass over all
    samples). Wrong results if time is not sorted."""
    if not params_valid(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        supersampling_factor
    ):
        raise ValueError(PARAMS_ERROR)
    idx, flux_planet_w, flux_moon_w = _window_model(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon,
        Omega_moon, i_moon, ecc_moon, w_moon, M_moon, epoch_distance,
        occult_small_threshold, hill_sphere_threshold, time,
        cache, time_sorted
    )
    return _window_cadences(
        len(time), idx, flux_planet_w, flux_moon_w, supersampling_factor
    )
