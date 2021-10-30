from numba import jit, prange
from numpy import sqrt, pi, arcsin
import numpy as np

from avocado import ellipse_pos, bary_pos
from mangold import occult_single, occult_array

# from pumpkin import eclipse_ratio


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def resample(arr, factor):
    out_samples = int(len(arr) / factor)
    out_arr = np.ones(out_samples)
    for idx in range(out_samples):
        start = idx * factor
        end = start + factor - 1
        out_arr[idx] = np.mean(arr[start:end])
    return out_arr


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
    epoch_distance,
    supersampling_factor=1,
):
    # "Morphological light-curve distortions due to finite integration time"
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract
    # Data gets smeared over long integration. Relevant for e.g., 30min cadences
    # To counter the effect: Set supersampling_factor = 5 (recommended value)
    # Then, 5x denser in time sampling, and averaging after, approximates effect
    if supersampling_factor < 1:
        print("supersampling_factor must be positive integer")
        # return False
    # print("supersampling_factor", supersampling_factor)
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)

    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_planet`, however, is a free parameter
    t0_planet_transit_times = np.arange(
        start=t0_planet,
        stop=t0_planet + epoch_distance * epochs,
        step=epoch_distance,
    )

    # Planetary transit duration at b=0 equals the width of the star
    # Check: Not fully matching Batman model -- too wide
    tdur_p = per_planet / pi * arcsin(sqrt((r_planet / 2 + R_star) ** 2) / a_planet)

    # Calculate moon period around planet.
    # Keep "1000.0"**3 as float: numba can't handle long ints
    G = 6.67408e-11
    per_moon = (
        (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * (M_planet + M_moon))))
        / 60
        / 60
        / 24
    )

    # t0_planet_offset in [days] ==> convert to x scale (i.e. 0.5 transit dur radius)
    t0_shift_planet = -t0_planet_offset / (tdur_p / 2)

    # arrays of epoch start and end dates [day]
    t_starts = t0_planet_transit_times - epoch_duration / 2
    t_ends = t0_planet_transit_times + epoch_duration / 2
    cadences = int(supersampled_cadences_per_day * epoch_duration)

    # Loop over epochs and stitch together:
    time_arrays = np.ones(shape=(epochs, cadences))
    px_bary = time_arrays.copy()
    py_bary = time_arrays.copy()
    mx_bary = time_arrays.copy()
    my_bary = time_arrays.copy()
    xps = time_arrays.copy()
    xpos_array = np.linspace(
        -epoch_duration / tdur_p, epoch_duration / tdur_p, cadences
    )

    for epoch in prange(epochs):
        time_array_this_epoch = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
        time_arrays[epoch] = time_array_this_epoch

        xm, ym = ellipse_pos(
            a=a_moon / R_star,
            per=per_moon,
            tau=tau_moon,
            Omega=Omega_moon,
            i=i_moon,
            time=time_array_this_epoch,
        )

        # Push planet following per_planet, which is a free parameter
        # For reference: Distance between epoch segments is fixed as segment_distance
        # Again, free parameter per_planet in units of days
        # have to convert to x scale == 0.5 transit duration for stellar radii
        per_shift_planet = ((per_planet - epoch_distance) * epoch) / (tdur_p / 2)

        # Adjust position of planet and moon due to mass: barycentric wobble
        xps[epoch] = xpos_array + t0_shift_planet - per_shift_planet
        mx_bary[epoch], my_bary[epoch], px_bary[epoch], py_bary[epoch] = bary_pos(
            xm=xm,
            ym=ym,
            xp=xpos_array + t0_shift_planet - per_shift_planet,
            b_planet=b_planet,
            mass_ratio=M_moon / M_planet,
        )

    px_bary = px_bary.ravel()
    py_bary = py_bary.ravel()
    mx_bary = mx_bary.ravel()
    my_bary = my_bary.ravel()
    time_arrays = time_arrays.ravel()

    # Distances of planet and moon from (0,0) = center of star
    z_planet = sqrt(px_bary ** 2 + py_bary ** 2)
    z_moon = sqrt(mx_bary ** 2 + my_bary ** 2)
    flux_planet = occult_array(zs=z_planet, u=u, k=r_planet / R_star)
    flux_moon = occult_array(zs=z_moon, u=u, k=r_moon / R_star)

    # Here: Mutual planet-moon occultations
    flux_total = 1 - ((1 - flux_planet) + (1 - flux_moon))

    # Supersampling downconversion

    if supersampling_factor > 1:
        # print("supersampling_factor", supersampling_factor)
        flux_planet = resample(flux_planet, supersampling_factor)
        flux_moon = resample(flux_moon, supersampling_factor)
        flux_total = resample(flux_total, supersampling_factor)
        time_arrays = resample(time_arrays, supersampling_factor)

    return (
        flux_planet,
        flux_moon,
        flux_total,
        px_bary,
        py_bary,
        mx_bary,
        my_bary,
        time_arrays,
    )
