from numba import jit, prange
from numpy import sqrt, pi, arcsin
import numpy as np
import numba as nb

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
    w_moon,
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
):
    # Calculate moon period around planet.
    # Keep "1000.0"**3 as float: numba can't handle long ints

    per_moon = (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * M_planet))) / 60 / 60 / 24
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
        a_moon / R_star, per_moon, tau_moon, Omega_moon, w_moon, i_moon, time
    )

    # (x,y) grid in units of stellar radii; star at (0,0)
    # Get x pos for start of time series
    start = -t_dur / tdur_p
    end = t_dur / tdur_p
    xpos_array = np.linspace(start, end, cadences)
    # print(xpos_array)

    # user gives t0_planet_offset in units of days
    # have to convert to x scale == 0.5 transit duration for stellar radii
    t0_shift_planet = -t0_planet_offset / (tdur_p / 2)

    # Adjust position of planet and moon due to mass: barycentric wobble
    xm_bary, ym_bary, xp_bary, yp_bary = bary_pos(
        xm, ym, xpos_array + t0_shift_planet, b_planet, mass_ratio
    )

    # Distances of planet and moon from (0,0) = center of star
    z_planet = sqrt(xp_bary ** 2 + yp_bary ** 2)
    z_moon = sqrt(xm_bary ** 2 + ym_bary ** 2)
    flux_planet = occult_array(zs=z_planet, u=u, k=r_planet / R_star)
    flux_moon = occult_array(zs=z_moon, u=u, k=r_moon / R_star)

    # Here: Mutual planet-moon occultations
    flux_total = 1 - ((1 - flux_planet) + (1 - flux_moon))
    return flux_planet, flux_moon, flux_total, xp_bary, yp_bary, xm_bary, ym_bary


@jit(cache=True, nopython=True, fastmath=True)
def pandora(
    r_moon,
    a_moon,
    tau_moon,
    Omega_moon,
    w_moon,
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
):

    t0_planet_transit_times = np.arange(
        start=t0_planet,
        stop=t0_planet + per_planet * epochs,
        step=per_planet,
    )
    # print(t0_planet_transit_times)

    # tau_moon is the position of the moon on its orbit, given as [0..1]
    # for the first timestamp of the first epoch
    # Cannot be given in units of days in prior, because moon orbit period varies
    # Example: Prior has tau in [5, 100] but model tests orbit with per_moon = 10
    #          It would physically still be OK, because it is circular and wraps
    #          around. However, the sampler would not converge when testing models.
    # So, we use tau in [0..1] and propagate to following epochs manually

    per_moon = (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * M_planet))) / 60 / 60 / 24
    # print("per_moon", per_moon)
    # print("per_planet", per_planet)
    # print("tau_moon first epoch", tau_moon)

    # Stroboscopic effect
    # Example: per_planet = 100d, per_moon = 20d ==> strobo_factor = 5
    #          Moon has made 5 orbits between planet transits
    # However, no need to convert anything here because ellipse_pos from avocado
    # will subtract the (constant) tau from the period to determine orbit position
    # strobo_factor = per_planet / per_moon
    # print("strobo_factor", strobo_factor)

    # Each epoch must contain a segment of data, centered at the planetary transit
    # Each epoch must be the same time duration
    # epoch_duration = 4  # days
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

    for epoch in range(epochs):
        time_array = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
        (
            flux_planet,
            flux_moon,
            flux_total,
            px_bary,
            py_bary,
            mx_bary,
            my_bary,
        ) = pandora_epoch(
            r_moon,
            a_moon,
            tau_moon,
            Omega_moon,
            w_moon,
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
            time_array,
        )
        flux_planet_array[epoch, :] = flux_planet
        flux_moon_array[epoch] = flux_moon
        flux_total_array[epoch] = flux_total
        px_bary_array[epoch] = px_bary
        py_bary_array[epoch] = py_bary
        mx_bary_array[epoch] = mx_bary
        my_bary_array[epoch] = my_bary
        time_arrays[epoch] = time_array


    # print(flux_planet_array.ravel())
    return (
        flux_planet_array.ravel(),
        flux_moon_array.ravel(),
        flux_total_array.ravel(),
        px_bary_array.ravel(),
        py_bary_array.ravel(),
        mx_bary_array.ravel(),
        my_bary_array.ravel(),
        time_arrays.ravel(),
    )
