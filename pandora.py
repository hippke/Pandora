from numba import jit, prange
from numpy import sqrt, pi, arcsin
import numpy as np

from avocado import ellipse_pos, bary_pos
from mangold import occult_single, occult_array
# from pumpkin import eclipse_ratio

G = 6.67408 * 10 ** -11

@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def pandora_epoch(
    r_moon,
    a_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    M_moon,
    per_planet,
    a_planet,
    r_planet,
    b_planet,
    t0_planet_offset,
    M_planet,
    R_star,
    u,
    time,
    epoch_distance,
    epoch
):
    # Calculate moon period around planet.
    # Keep "1000.0"**3 as float: numba can't handle long ints
    # Check if correct. Add Moon mass to equation?
    per_moon = (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * (M_planet + M_moon)))) / 60 / 60 / 24
    mass_ratio = M_moon / M_planet
    t_start = np.min(time)
    t_end = np.max(time)
    t_dur = t_end - t_start
    cadences = len(time)

    # Planetary transit duration at b=0 equals the width of the star
    # Check: Not fully matching Batman model -- too wide
    tdur_p = per_planet / pi * arcsin(sqrt((r_planet / 2 + R_star) ** 2) / a_planet)

    # Get Kepler Ellipse
    xm, ym = ellipse_pos(
        a_moon / R_star, per_moon, tau_moon, Omega_moon, i_moon, time
    )

    # (x,y) grid in units of stellar radii; star at (0,0)
    # Get x pos for start of time series
    start = -t_dur / tdur_p
    end = t_dur / tdur_p
    xpos_array = np.linspace(start, end, cadences)

    # user gives t0_planet_offset in units of days
    # have to convert to x scale == 0.5 transit duration for stellar radii
    t0_shift_planet = - t0_planet_offset / (tdur_p / 2)

    # Push planet following per_planet, which is a free parameter
    # For reference: Distance between epoch segments is fixed as segment_distance
    # Again, free parameter per_planet in units of days
    # have to convert to x scale == 0.5 transit duration for stellar radii
    per_shift_planet = - ((per_planet - epoch_distance) * epoch) / (tdur_p / 2)

    # Adjust position of planet and moon due to mass: barycentric wobble
    xm_bary, ym_bary, xp_bary, yp_bary = bary_pos(
        xm, ym, xpos_array + t0_shift_planet + per_shift_planet, b_planet, mass_ratio
    )

    # Distances of planet and moon from (0,0) = center of star
    z_planet = sqrt(xp_bary ** 2 + yp_bary ** 2)
    z_moon = sqrt(xm_bary ** 2 + ym_bary ** 2)
    flux_planet = occult_array(zs=z_planet, u=u, k=r_planet / R_star)
    flux_moon = occult_array(zs=z_moon, u=u, k=r_moon / R_star)

    # Here: Mutual planet-moon occultations
    flux_total = 1 - ((1 - flux_planet) + (1 - flux_moon))
    return flux_planet, flux_moon, flux_total, xp_bary, yp_bary, xm_bary, ym_bary


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def pandora(
    r_moon,
    a_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    M_moon,
    per_planet,
    a_planet,
    r_planet,
    b_planet,
    t0_planet_offset,
    M_planet,
    R_star,
    u,
    t0_planet,
    epochs,
    epoch_duration,
    cadences_per_day,
    epoch_distance
):
    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_planet`, however, is a free parameter


    t0_planet_transit_times = np.arange(
        start=t0_planet,
        stop=t0_planet + epoch_distance * epochs,
        step=epoch_distance,
    )

    # tau_moon is the position of the moon on its orbit, given as [0..1]
    # for the first timestamp of the first epoch
    # Cannot be given in units of days in prior, because moon orbit period varies
    # Example: Prior has tau in [5, 100] but model tests orbit with per_moon = 10
    #          It would physically still be OK, because it is circular and wraps
    #          around. However, the sampler would not converge when testing models.
    # So, we use tau in [0..1] and propagate to following epochs manually

    per_moon = (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * (M_planet + M_moon)))) / 60 / 60 / 24

    # Stroboscopic effect
    # Example: per_planet = 100d, per_moon = 20d ==> strobo_factor = 5
    #          Moon has made 5 orbits between planet transits
    # However, no need to convert anything here because ellipse_pos from avocado
    # will subtract the (constant) tau from the period to determine orbit position
    # strobo_factor = per_planet / per_moon
    # print("strobo_factor", strobo_factor)

    # Each epoch must contain a segment of data, centered at the planetary transit
    # Each epoch must be the same time duration
    # cadences_per_day = 48  # switch this to automatic calculation? What about gaps?

    t_starts = (
        t0_planet_transit_times - epoch_duration / 2
    )  # array of epoch start dates [day]
    t_ends = (
        t0_planet_transit_times + epoch_duration / 2
    )  # array of epoch end dates [day]

    cadences = int(cadences_per_day * epoch_duration)

    time_arrays = np.ones(shape=(epochs, cadences))

    # Loop over epochs and call pandora_segment for each. Then, stitch together:
    flux_planet_array = np.ones(shape=(epochs, cadences))
    flux_moon_array = np.ones(shape=(epochs, cadences))
    flux_total_array = np.ones(shape=(epochs, cadences))
    px_bary_array = np.ones(shape=(epochs, cadences))
    py_bary_array = np.ones(shape=(epochs, cadences))
    mx_bary_array = np.ones(shape=(epochs, cadences))
    my_bary_array = np.ones(shape=(epochs, cadences))

    for epoch in prange(epochs):
        time_arrays[epoch] = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
        (
            flux_planet_array[epoch],
            flux_moon_array[epoch],
            flux_total_array[epoch],
            px_bary_array[epoch],
            py_bary_array[epoch],
            mx_bary_array[epoch],
            my_bary_array[epoch]
        ) = pandora_epoch(
            r_moon,
            a_moon,
            tau_moon,
            Omega_moon,
            i_moon,
            M_moon,
            per_planet,
            a_planet,
            r_planet,
            b_planet,
            t0_planet_offset,
            M_planet,
            R_star,
            u,
            time_arrays[epoch],
            epoch_distance,
            epoch
        )
    return (
        flux_planet_array.ravel(),
        flux_moon_array.ravel(),
        flux_total_array.ravel(),
        px_bary_array.ravel(),
        py_bary_array.ravel(),
        mx_bary_array.ravel(),
        my_bary_array.ravel(),
        time_arrays.ravel()
    )
