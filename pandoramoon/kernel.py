"""Pandora's compiled model core: moon orbit set-up, sky positions, culling to the samples near transit,
occultation and mutual-eclipse fluxes, supersampling, and the log-likelihoods (pandora_loglike, the fused
planet-only pandora_loglike_planet). The public model functions are in pandora.py; Likelihood (likelihood.py) and
the samplers call pandora_loglike / pandora_loglike_planet."""
import numpy as np
from numpy import sqrt, pi
from numba import jit

from pandoramoon.eclipse import eclipse, eclipse_possible, eclipse_impossible_circular
from pandoramoon.ellipse import (
    ellipse, ellipse_ecc, ellipse_z, ellipse_ecc_z, ellipse_z_runs, ellipse_z_only_runs,
    ellipse_subruns,
    subrun_cell_dt, subruns_pay, ellipse_ecc_z_runs
)
from pandoramoon.occult import occult_small, occult_hybrid_z2, read_occult_cache
from pandoramoon.helpers import resample
from pandoramoon.constants import G as G_GRAV, DAY
from pandoramoon.grids import (
    x_bary_window, x_bary_window_sorted_t,
    x_bary_runs_sorted, runs_x_t
)
from pandoramoon.params import params_valid

NEG_INF = -np.inf


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_loglike(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance, supersampling_factor,
    occult_small_threshold, hill_sphere_threshold, time,
    inv_var, lin_terms, chi2_flat, cache, time_sorted
):
    """Gaussian log-likelihood -chi^2/2 of the total flux, without
    full-length arrays: chi^2 = chi2_flat + sum over window cadences of
    ((f - data) inv_err)^2 - ((1 - data) inv_err)^2 = D (inv_var D -
    lin_terms) with the deficit D = 1 - f (P13b), where inv_var = inv_err^2,
    lin_terms = 2 (1 - data) inv_err^2 and chi2_flat = sum(((1 - data)
    inv_err)^2) (model = 1 everywhere) are precomputed per dataset. Use it
    via Likelihood.
    P18: the window is kept as per-epoch sample runs (no idx/x/time window
    arrays, see _window_model_runs) and the chi^2 is summed per run.
    Parameters outside their physical range (params_valid) give -inf, the
    usual "outside the prior" value for samplers."""
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
    chi2 = chi2_flat + _window_chi2_runs(
        len(time), los, his, flux_planet_w, flux_moon_w, supersampling_factor,
        inv_var, lin_terms
    )
    return -0.5 * chi2


# Planet-only model: the moon parameters that select Pandora's planet-only path
# (moon orbit inside the planet -> _moon_setup "unphysical": planet at the
# barycentre, window = the planet's transit only)
PLANET_ONLY_MOON = (0.0, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # r_moon, per_moon, tau, Omega, i, ecc, w, M_moon


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora_loglike_planet(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, epoch_distance, supersampling_factor,
    occult_small_threshold, hill_sphere_threshold, time,
    inv_var, lin_terms, chi2_flat, cache, time_sorted
):
    """Log-likelihood of the planet-only model (no moon): the same value as
    pandora_loglike() with the moon parameters PLANET_ONLY_MOON (r_moon =
    M_moon = 0, per_moon = 1e-6 d), up to rounding, but faster (no moon
    setup, no moon arrays and occultation pass; the planet's z^2, flux and
    the chi^2 in three passes over the planet's transit windows). The
    trailing arguments are those of Likelihood._tail. Supersampled cadences
    are averaged over their sub-exposures as in pandora_loglike()
    (_window_chi2_runs_planet). Falls back to pandora_loglike() for eccentric
    planets, the occultation cache, unsorted time, a planet smaller than the
    dummy orbit (r_planet < 1e-3) or when the sorted window search does not
    apply."""
    r_m, p_m, tau_m, om_m, i_m, e_m, w_m, m_m = PLANET_ONLY_MOON
    fast = (ecc_bary == 0 and time_sorted and r_planet >= 1e-3)
    if cache is not None:
        fast = False
    if not fast:
        return pandora_loglike(
            u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
            t0_bary, t0_bary_offset, M_planet, r_m, p_m, tau_m, om_m, i_m, e_m,
            w_m, m_m, epoch_distance, supersampling_factor, occult_small_threshold,
            hill_sphere_threshold, time, inv_var, lin_terms,
            chi2_flat, cache, time_sorted
        )
    if not params_valid(
        u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
        t0_bary, t0_bary_offset, M_planet, r_m, p_m, tau_m, om_m, i_m, e_m, w_m,
        m_m, epoch_distance, supersampling_factor
    ):
        return NEG_INF
    reach = 1 + r_planet + 1e-6          # _cull_reach() of the planet-only path
    ok, los, his, c_t, c_d, tdur, t0_shift = x_bary_runs_sorted(
        time, float(a_bary), float(per_bary), float(t0_bary), float(t0_bary_offset),
        epoch_distance, 0.0, float(w_bary), reach
    )
    if not ok:
        return pandora_loglike(
            u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
            t0_bary, t0_bary_offset, M_planet, r_m, p_m, tau_m, om_m, i_m, e_m,
            w_m, m_m, epoch_distance, supersampling_factor, occult_small_threshold,
            hill_sphere_threshold, time, inv_var, lin_terms,
            chi2_flat, cache, time_sorted
        )
    total = 0
    for k in range(len(los)):
        total += his[k] - los[k]
    b = float(b_bary)
    b2 = b * b
    z2 = np.empty(total)
    m = 0
    for k in range(len(los)):
        ct = c_t[k]
        cd = c_d[k]
        lo = los[k]
        for j in range(his[k] - lo):
            x = (2 * (time[lo + j] - ct)) / tdur - t0_shift - cd
            z2[m + j] = x * x + b2
        m += his[k] - lo
    flux = occult_hybrid_z2(z2, float(r_planet), u1, u2)
    if supersampling_factor > 1:
        return -0.5 * (chi2_flat + _window_chi2_runs_planet(
            len(time), los, his, flux, supersampling_factor, inv_var, lin_terms))
    chi2 = 0.0
    m = 0
    for k in range(len(los)):
        lo = los[k]
        cnt = his[k] - lo
        wr = inv_var[lo:lo + cnt]
        cr = lin_terms[lo:lo + cnt]
        fr = flux[m:m + cnt]
        for j in range(cnt):
            d = 1.0 - fr[j]
            chi2 += d * (wr[j] * d - cr[j])
        m += cnt
    return -0.5 * (chi2_flat + chi2)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _moon_setup(
    R_star, per_bary, a_bary, M_planet, M_moon, per_moon, ecc_moon, r_planet,
    r_moon, hill_sphere_threshold
):
    """Planet-moon semi-major axis [R_star] and the plausibility flag
    (unphysical moon orbits are moved out of transit)"""
    # Calculate moon period around planet
    G = G_GRAV
    day = DAY
    a_moon = (
        G * (M_planet + M_moon) / (2 * pi / (per_moon * day)) ** 2
    ) ** (1 / 3)
    a_moon /= R_star

    # Check physical plausibility of a_moon
    # Should be inside [Roche lobe, Hill sphere] plus/minus some user-set margin
    M_star = ((4 * pi ** 2 / G) * ((a_bary * R_star) ** 3)) / (per_bary * day) ** 2
    r_hill = a_bary * (M_planet / (3 * M_star)) ** (1 / 3)
    r_hill_fraction = a_moon * (1 + ecc_moon) / r_hill
    if r_hill_fraction > hill_sphere_threshold:
        unphysical = True
    else:
        unphysical = False

    # Roche lobe: Not used here, because we don't know densities of planet and moon
    # Instead of taking density guesses, we just demand a_moon > (R_planet + R_moon)
    if a_moon * (1 - ecc_moon) < (r_planet + r_moon):
        unphysical = True
    return a_moon, unphysical


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _orbit(
    time, x_bary, unphysical, a_moon, per_moon, tau_moon, Omega_moon, w_moon,
    i_moon, ecc_moon, mass_ratio, b_bary
):
    """Sky-plane coordinates xm, ym, xp, yp [R_star] of moon and planet"""
    # Unphysical moon orbit: Keep planet, but put moon at far out of transit position
    if unphysical:
        bignum = 1e8
        xp = x_bary
        yp = np.full(len(x_bary), b_bary)
        xm = np.full(len(x_bary), bignum)
        ym = xm.copy()
    else:  # valid, physical system
        if ecc_moon == 0:
            xm, ym, xp, yp = ellipse(
                a=a_moon,
                per=per_moon,
                tau=tau_moon,
                Omega=Omega_moon,
                i=i_moon,
                time=time,
                x_bary=x_bary,
                mass_ratio=mass_ratio,
                b_bary=b_bary,
            )
        else:
            xm, ym, xp, yp = ellipse_ecc(
                a=a_moon,
                per=per_moon,
                e=ecc_moon,
                tau=tau_moon,
                Omega=Omega_moon,
                w=w_moon,
                i=i_moon,
                time=time,
                mass_ratio=mass_ratio,
                x_bary=x_bary,
                b_bary=b_bary,
            )
    return xm, ym, xp, yp


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _orbit_z(
    time, x_bary, unphysical, a_moon, per_moon, tau_moon, Omega_moon, w_moon,
    i_moon, ecc_moon, mass_ratio, b_bary
):
    """_orbit() plus the squared distances from the stellar centre, in one
    fused loop per sample: (xm, ym, xp, yp, zm2, zp2) (P26: z^2, the
    occultations take it). For the culled path (_window_model); pandora()
    keeps _orbit() and _window_fluxes()."""
    if unphysical:
        n = len(x_bary)
        xp = x_bary
        yp = np.full(n, b_bary)
        xm = np.full(n, 1e8)
        ym = xm.copy()
        zp2 = np.empty(n)
        for j in range(n):
            zp2[j] = xp[j] * xp[j] + b_bary * b_bary
        zm2 = np.full(n, 1e16)
        return xm, ym, xp, yp, zm2, zp2
    if ecc_moon == 0:
        return ellipse_z(
            a_moon, per_moon, tau_moon, Omega_moon, i_moon, time, x_bary,
            mass_ratio, b_bary
        )
    return ellipse_ecc_z(
        a_moon, per_moon, ecc_moon, tau_moon, Omega_moon, w_moon, i_moon, time,
        mass_ratio, x_bary, b_bary
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _cull_reach(a_moon, ecc_moon, r_planet, r_moon, unphysical, cache):
    """Conservative bound on |x_bary| [R_star] beyond which neither body can
    touch the star, so that all fluxes are exactly 1.

    Both bodies are offset from the barycentre by at most a_moon * (1 + e)
    on the sky (|projected unit vector| <= 1, orbital radius <= a (1 + e);
    the planet's share is a q/(1+q), the moon's a/(1+q)). A body has
    flux exactly 1 for z >= 1 + r (occult_small/occult_hybrid) or, with the
    occultation cache, for z >= 1.1 (bilinear interpolation reaches past
    1 + r up to the grid end). Mutual eclipses only modify a moon flux < 1.
    The margin 1e-6 covers float round-off."""
    r_limit = 1 + r_planet
    if not unphysical:  # moon at 1e8 otherwise
        r_limit = 1 + max(r_planet, r_moon)
    if cache is not None:
        r_limit = max(r_limit, 1.1)
    if unphysical:  # planet sits at the barycentre
        return r_limit + 1e-6
    return r_limit + a_moon * (1 + abs(ecc_moon)) + 1e-6


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _cull_window(x_bary, reach):
    """Indices of samples with |x_bary| < reach (NaN is kept: not culled)"""
    n = len(x_bary)
    idx = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        if not (abs(x_bary[i]) >= reach):
            idx[m] = i
            m += 1
    return idx[:m]


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_fluxes(
    xp, yp, xm, ym, unphysical, r_planet, r_moon, u1, u2,
    occult_small_threshold, cache
):
    """Planet and moon flux (incl. mutual eclipses) for the given samples"""
    # Squared distances of planet and moon from (0,0) = center of star (P26:
    # the occultations take z^2). Not sufficient to only calculate z in
    # ellipse func: we also need full coordinates for mutual eclipses below
    z2_planet = xp * xp + yp * yp
    if unphysical:
        z2_moon = xm * xm      # moon far off the star (1e8)
    else:
        z2_moon = xm * xm + ym * ym
    return _window_fluxes_z(
        xp, yp, xm, ym, z2_planet, z2_moon, unphysical, r_planet, r_moon, u1, u2,
        occult_small_threshold, cache
    )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_fluxes_z(
    xp, yp, xm, ym, z2_planet, z2_moon, unphysical, r_planet, r_moon, u1, u2,
    occult_small_threshold, cache
):
    """_window_fluxes() with precomputed squared distances z2_planet,
    z2_moon (P26: occult_hybrid_z2 takes z^2; only the cache and
    occult_small take z = sqrt(z^2)). unphysical=True also means "no
    eclipse()": the positions xp, yp, xm, ym are then not read (P24b passes
    empty arrays)."""
    # Cached Mandel-Agol occultation model for planet < 0.1, else hybrid
    if cache is not None and r_planet < 0.1:
        if r_planet < cache[1][0]:
            # k below the cache grid: use exact small-body approximation
            flux_planet = occult_small(zs=sqrt(z2_planet), k=r_planet, u1=u1, u2=u2)
        else:
            flux_planet = read_occult_cache(
                zs_target=sqrt(z2_planet),
                k=r_planet,
                cache=cache
            )
    else:
        flux_planet = occult_hybrid_z2(z2_planet, r_planet, u1, u2)

    # For moon transit: small-body approximation only if the user sets
    # occult_small_threshold > 0 (default 0: exact, bug #9)
    if r_moon < occult_small_threshold:
        flux_moon = occult_small(zs=sqrt(z2_moon), k=r_moon, u1=u1, u2=u2)
    elif cache is not None and r_moon < 0.1:
        if r_moon < cache[1][0]:
            # k below the cache grid: use exact small-body approximation
            flux_moon = occult_small(zs=sqrt(z2_moon), k=r_moon, u1=u1, u2=u2)
        else:
            flux_moon = read_occult_cache(zs_target=sqrt(z2_moon), k=r_moon, cache=cache)
    else:
        flux_moon = occult_hybrid_z2(z2_moon, r_moon, u1, u2)

    # Mutual planet-moon occultations; skipped when no sample can overlap
    # (P5: one vectorised min-distance pass instead of eclipse()'s per-sample
    # limb checks; ~90 % of the draws of a fit have no mutual eclipse)
    if not unphysical and eclipse_possible(xp, yp, xm, ym, r_planet, r_moon):
        flux_moon = eclipse(
            xp, yp, xm, ym, r_planet, r_moon, flux_moon,
            u1, u2
        )
    return flux_planet, flux_moon


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _scatter(n, idx, flux_planet_w, flux_moon_w):
    """Full-length flux arrays: 1 everywhere except at the window indices"""
    flux_planet = np.ones(n)
    flux_moon = np.ones(n)
    for j in range(len(idx)):
        flux_planet[idx[j]] = flux_planet_w[j]
        flux_moon[idx[j]] = flux_moon_w[j]
    return flux_planet, flux_moon


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _total_and_resample(flux_planet, flux_moon, supersampling_factor):
    flux_total = flux_moon + flux_planet - 1

    # Supersampling downconversion
    if supersampling_factor > 1:
        flux_planet = resample(flux_planet, supersampling_factor)
        flux_moon = resample(flux_moon, supersampling_factor)
        flux_total = resample(flux_total, supersampling_factor)
    return flux_planet, flux_moon, flux_total


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_model(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance, occult_small_threshold,
    hill_sphere_threshold, time, cache,
    time_sorted
):
    """Culled model core: (ascending sample indices of the window, planet
    flux, moon flux there). Samples outside the window have flux exactly 1.
    time_sorted=True (time ascending, caller's guarantee): the window is
    found by binary search per epoch instead of a pass over all samples."""
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
    # Barycentre positions only for samples that can be in transit
    reach = _cull_reach(a_moon, ecc_moon, r_planet, r_moon, unphysical, cache)
    if time_sorted:  # P11: window times come from the search itself
        idx, x_bary, time_w = x_bary_window_sorted_t(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
    else:
        idx, x_bary = x_bary_window(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
        time_w = time[idx]
    xm, ym, xp, yp, zm2, zp2 = _orbit_z(
        time_w, x_bary, unphysical, a_moon, per_moon, tau_moon,
        Omega_moon, w_moon, i_moon, ecc_moon, M_moon / M_planet, b_bary
    )
    flux_planet_w, flux_moon_w = _window_fluxes_z(
        xp, yp, xm, ym, zp2, zm2, unphysical, r_planet, r_moon, u1, u2,
        occult_small_threshold, cache
    )
    return idx, flux_planet_w, flux_moon_w


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _idx_runs(idx):
    """Maximal contiguous runs of ascending indices: (los, his) with
    idx = concatenation of arange(los[k], his[k])"""
    m = len(idx)
    los = np.empty(m, dtype=np.int64)
    his = np.empty(m, dtype=np.int64)
    r = 0
    j = 0
    while j < m:
        lo = idx[j]
        j += 1
        while j < m and idx[j] == idx[j - 1] + 1:
            j += 1
        los[r] = lo
        his[r] = idx[j - 1] + 1
        r += 1
    return los[:r], his[:r]


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_model_runs(
    u1, u2, R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary,
    t0_bary, t0_bary_offset, M_planet, r_moon, per_moon, tau_moon, Omega_moon,
    i_moon, ecc_moon, w_moon, M_moon, epoch_distance, occult_small_threshold,
    hill_sphere_threshold, time, cache,
    time_sorted
):
    """_window_model() with the window as sample runs (P18): (los, his,
    planet flux, moon flux), where the window samples are
    time[los[k]:his[k]] for all k in this order (ascending, disjoint) and
    the fluxes are given there. For sorted time and a physical moon
    (circular, or since P7c eccentric with 0 < e < 1), the runs come from
    the binary search and the fused orbit loop computes x_bary inline
    (ellipse_z_runs / ellipse_ecc_z_runs): no idx, x_bary or time window
    arrays. Otherwise the window arrays are built as in _window_model()
    and converted to runs. Same fluxes as _window_model().
    P10: for a physical moon (P7c: also eccentric) without occultation cache the runs are
    then narrowed to the sub-runs where a body can be on the star
    (ellipse_subruns); the other window samples have flux exactly 1 and are
    not part of the returned runs.
    P24b: on that path, when the circular orbit's geometry rules out a planet-moon
    eclipse (eclipse_impossible_circular), only z is computed
    (ellipse_z_only_runs) and eclipse() is skipped (same fluxes)."""
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
    reach = _cull_reach(a_moon, ecc_moon, r_planet, r_moon, unphysical, cache)
    if time_sorted:
        ok, los, his, c_t, c_d, tdur, t0_shift = x_bary_runs_sorted(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
    else:
        ok = False
        los = np.empty(0, dtype=np.int64)
        his = los
        c_t = np.empty(0)
        c_d = c_t
        tdur = 0.0
        t0_shift = 0.0
    if ok and not unphysical and ecc_moon >= 0 and ecc_moon < 1:
        # P10: only the parts of the runs where a body can be on the star
        # (the occultation cache's flux reaches past 1 + r, so not with it;
        # and only where the moon cells are long enough to pay); P7c: also
        # for eccentric moons
        if cache is None and subruns_pay(
            time, los, his, subrun_cell_dt(a_moon, per_moon, M_moon / M_planet, ecc_moon)[0]
        ):
            los, his, c_t, c_d = ellipse_subruns(
                a_moon, per_moon, tau_moon, Omega_moon, i_moon, time, los, his,
                c_t, c_d, tdur, t0_shift, M_moon / M_planet, b_bary, r_planet,
                r_moon, ecc_moon, w_moon
            )
        # P24b: if the orbit's geometry rules out a planet-moon eclipse
        # (~90 % of the draws), only z is needed: no positions, no
        # eclipse_possible() pass. P7e: an eccentric orbit's separation is
        # >= a (1 - e), so the same bound with a (1 - e) (positions are still
        # computed, but the eclipse_possible() pass is skipped)
        no_eclipse = eclipse_impossible_circular(a_moon * (1 - ecc_moon), i_moon, r_planet, r_moon)
        if ecc_moon != 0:
            # P7c: eccentric moon on the runs (x_bary inline)
            xm, ym, xp, yp, zm2, zp2 = ellipse_ecc_z_runs(
                a_moon, per_moon, ecc_moon, tau_moon, Omega_moon, w_moon, i_moon,
                time, los, his, c_t, c_d, tdur, t0_shift, M_moon / M_planet, b_bary
            )
        elif no_eclipse:
            zm2, zp2 = ellipse_z_only_runs(
                a_moon, per_moon, tau_moon, Omega_moon, i_moon, time, los, his,
                c_t, c_d, tdur, t0_shift, M_moon / M_planet, b_bary
            )
            xm = np.empty(0)
            ym = xm
            xp = xm
            yp = xm
        else:
            xm, ym, xp, yp, zm2, zp2 = ellipse_z_runs(
                a_moon, per_moon, tau_moon, Omega_moon, i_moon, time, los, his,
                c_t, c_d, tdur, t0_shift, M_moon / M_planet, b_bary
            )
    else:
        no_eclipse = unphysical
        if ok:
            _, x_bary, time_w = runs_x_t(time, los, his, c_t, c_d, tdur, t0_shift)
        else:
            idx, x_bary = x_bary_window(
                time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
                ecc_bary, w_bary, reach
            )
            time_w = time[idx]
            los, his = _idx_runs(idx)
        xm, ym, xp, yp, zm2, zp2 = _orbit_z(
            time_w, x_bary, unphysical, a_moon, per_moon, tau_moon,
            Omega_moon, w_moon, i_moon, ecc_moon, M_moon / M_planet, b_bary
        )
    flux_planet_w, flux_moon_w = _window_fluxes_z(
        xp, yp, xm, ym, zp2, zm2, no_eclipse, r_planet, r_moon, u1, u2,
        occult_small_threshold, cache
    )
    return los, his, flux_planet_w, flux_moon_w


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _n_out(n, supersampling_factor):
    """Number of output cadences for n samples (as resample())"""
    if supersampling_factor <= 1:
        return n
    return int(np.ceil(n / supersampling_factor))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_cadences(n, idx, flux_planet_w, flux_moon_w, supersampling_factor):
    """Output cadences touched by the (ascending) window sample indices, and
    planet/moon/total flux there; all other output cadences have flux 1.
    With supersampling, a touched cadence is the mean over its block with
    flux 1 at non-window samples (as resample() of the full array).
    Same values as _scatter() followed by _total_and_resample() up to
    rounding (~1e-16), without full-length temporaries: the block mean is
    accumulated as a deficit, 1 - sum(1 - f)/length (non-window samples
    contribute 0, so a block of exact 1s stays exactly 1), and the total is
    1 - (deficit_planet + deficit_moon)."""
    m = len(idx)
    if supersampling_factor <= 1:
        flux_total_w = np.empty(m)
        for j in range(m):
            flux_total_w[j] = flux_moon_w[j] + flux_planet_w[j] - 1
        return idx, flux_planet_w, flux_moon_w, flux_total_w

    factor = supersampling_factor
    out_idx = np.empty(m, dtype=np.int64)
    flux_planet = np.empty(m)
    flux_moon = np.empty(m)
    flux_total = np.empty(m)
    j = 0
    k = 0
    while j < m:
        out = idx[j] // factor
        start = out * factor
        end = start + factor
        if end > n:
            end = n
        length = end - start
        dp = 0.0
        dm = 0.0
        while j < m and idx[j] < end:
            dp += 1.0 - flux_planet_w[j]
            dm += 1.0 - flux_moon_w[j]
            j += 1
        dp /= length
        dm /= length
        out_idx[k] = out
        flux_planet[k] = 1.0 - dp
        flux_moon[k] = 1.0 - dm
        flux_total[k] = 1.0 - (dp + dm)
        k += 1
    return out_idx[:k], flux_planet[:k], flux_moon[:k], flux_total[:k]


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_chi2_runs(n, los, his, flux_planet_w, flux_moon_w,
                      supersampling_factor, inv_var, lin_terms):
    """chi^2 part of the window cadences, sum of ((f - data) inv_err)^2 -
    ((1 - data) inv_err)^2 = D (inv_var D - lin_terms) over them (D = 1 - f,
    see pandora_loglike), for the total flux f of the
    output cadences touched by the window samples, given as ascending,
    disjoint runs [los[k], his[k]) (fluxes in run order): the same cadences
    and block means as _window_cadences() on the concatenated indices, to
    ~2e-16 (the per-sample total deficit (1 - f_planet) + (1 - f_moon) is
    summed per block, not planet and moon separately).

    P13: no cadence arrays (fused with the chi^2), and full blocks of a run
    as fixed-length sums without integer division or per-sample end tests.
    P18: the run boundaries are given (no idx array, no contiguity tests).
    Per run: a partial first block (or the continuation of a block left
    open by the previous run, if the runs touch), full blocks, a partial
    last block that stays open for the next run. Without supersampling the
    loop runs over inv_var[lo:hi] directly.
    P13b: the difference to the flat model in the form D (inv_var D -
    lin_terms) instead of r^2 - flat_terms: two data arrays instead of three
    (the loop is load-bound: PLATO 1-min windows 1.15 -> 0.93 us in numba),
    and no cancellation of r^2 against flat_terms."""
    chi2 = 0.0
    m = 0
    nr = len(los)
    if supersampling_factor <= 1:
        for k in range(nr):
            lo = los[k]
            cnt = his[k] - lo
            # slices: no wraparound selects for data[lo + j] (cf. ellipse_z_runs)
            wr = inv_var[lo:lo + cnt]
            cr = lin_terms[lo:lo + cnt]
            fpr = flux_planet_w[m:m + cnt]
            fmr = flux_moon_w[m:m + cnt]
            for j in range(cnt):
                d = (1.0 - fpr[j]) + (1.0 - fmr[j])
                chi2 += d * (wr[j] * d - cr[j])
            m += cnt
        return chi2

    factor = supersampling_factor
    out = -1     # open (not yet added) block: output cadence, end, deficit sum
    end = 0
    s = 0.0
    for k in range(nr):
        p = los[k]
        hi = his[k]
        while p < hi:
            if out >= 0:
                if p < end:  # the run continues the open block
                    q = min(hi, end)
                    for j in range(m, m + q - p):
                        s += (1.0 - flux_planet_w[j]) + (1.0 - flux_moon_w[j])
                    m += q - p
                    p = q
                    continue
                d = s / (end - out * factor)
                chi2 += d * (inv_var[out] * d - lin_terms[out])
                out = -1
            o = p // factor
            start = o * factor
            if p == start:
                # full blocks inside the run (start + factor <= hi <= n)
                nb = (hi - p) // factor
                for b in range(nb):
                    s = 0.0
                    for j in range(m, m + factor):
                        s += (1.0 - flux_planet_w[j]) + (1.0 - flux_moon_w[j])
                    d = s / factor
                    chi2 += d * (inv_var[o + b] * d - lin_terms[o + b])
                    m += factor
                p += nb * factor
                if nb > 0:
                    continue
            # partial block: open it
            out = o
            end = min(start + factor, n)
            s = 0.0
            q = min(hi, end)
            for j in range(m, m + q - p):
                s += (1.0 - flux_planet_w[j]) + (1.0 - flux_moon_w[j])
            m += q - p
            p = q
    if out >= 0:
        d = s / (end - out * factor)
        chi2 += d * (inv_var[out] * d - lin_terms[out])
    return chi2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _window_chi2_runs_planet(n, los, his, flux_w, supersampling_factor, inv_var, lin_terms):
    """_window_chi2_runs() for supersampling_factor > 1 with the planet flux only (no moon term): the same
    cadences, the same block means (deficit sums over the sub-exposures divided by the full block length; samples
    outside the window have deficit 0) and the same chi^2 form D (inv_var D - lin_terms)"""
    factor = supersampling_factor
    chi2 = 0.0
    m = 0
    out = -1
    end = 0
    s = 0.0
    for k in range(len(los)):
        p = los[k]
        hi = his[k]
        while p < hi:
            if out >= 0:
                if p < end:
                    q = min(hi, end)
                    for j in range(m, m + q - p):
                        s += 1.0 - flux_w[j]
                    m += q - p
                    p = q
                    continue
                d = s / (end - out * factor)
                chi2 += d * (inv_var[out] * d - lin_terms[out])
                out = -1
            o = p // factor
            start = o * factor
            if p == start:
                nb = (hi - p) // factor
                for b in range(nb):
                    s = 0.0
                    for j in range(m, m + factor):
                        s += 1.0 - flux_w[j]
                    d = s / factor
                    chi2 += d * (inv_var[o + b] * d - lin_terms[o + b])
                    m += factor
                p += nb * factor
                if nb > 0:
                    continue
            out = o
            end = min(start + factor, n)
            s = 0.0
            q = min(hi, end)
            for j in range(m, m + q - p):
                s += 1.0 - flux_w[j]
            m += q - p
            p = q
    if out >= 0:
        d = s / (end - out * factor)
        chi2 += d * (inv_var[out] * d - lin_terms[out])
    return chi2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _densify(n_out, out_idx, values, fill):
    """Full-length float64 array: fill, with values at out_idx"""
    out = np.full(n_out, fill)
    for k in range(len(out_idx)):
        out[out_idx[k]] = values[k]
    return out


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _densify_deficit32(n_out, out_idx, flux):
    """Full-length float32 deficit 1 - flux (0 elsewhere)"""
    out = np.zeros(n_out, dtype=np.float32)
    for k in range(len(out_idx)):
        out[out_idx[k]] = 1.0 - flux[k]
    return out
