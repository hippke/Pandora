import numpy as np
from numpy import sqrt, pi, cos, arcsin, empty
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def timegrid(t0_bary, epochs, epoch_duration, cadences_per_day, epoch_distance, supersampling_factor):
    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_bary`, however, is a free parameter
    ti_epoch_midtimes = np.arange(
        start=t0_bary,
        stop=t0_bary + epoch_distance * epochs,
        step=epoch_distance,
    )

    # arrays of epoch start and end dates [day]
    t_starts = ti_epoch_midtimes - epoch_duration / 2
    t_ends = ti_epoch_midtimes + epoch_duration / 2

    # "Morphological light-curve distortions due to finite integration time"
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract
    # Data gets smeared over long integration. Relevant for e.g., 30min cadences
    # To counter the effect: Set supersampling_factor = 5 (recommended value)
    # Then, 5x denser in time sampling, and averaging after, approximates effect
    if supersampling_factor < 1:
        print("supersampling_factor must be positive integer")
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)

    cadences = int(supersampled_cadences_per_day * epoch_duration)
    time = np.empty(shape=(epochs, cadences))
    for epoch in range(epochs):
        time[epoch] = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
    return time.ravel()


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_grid(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary, w_bary
):
    # Planetary transit duration at b=0 equals the width of the star
    # Formally correct would be: (R_star+r_planet) for the transit duration T1-T4
    # Here, however, we need points where center of planet is on stellar limb
    # https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    tdur = per_bary / pi * arcsin(1 / a_bary)

    # Correct transit duration based on relative orbital velocity
    # of circular versus eccentric case
    # Subtract (w_bary - 90) to match batman and PyTransit coordinate system
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))

    # t0_bary_offset in [days] ==> convert to x scale (i.e. 0.5 transit dur radius)
    t0_shift_planet = t0_bary_offset / (tdur / 2)
    x_bary = np.empty(len(time))
    for idx in range(len(time)):
        epoch = int((time[idx] - t0_bary) / per_bary + 0.5)
        x_bary[idx] = (
            (2 * (time[idx] - (t0_bary + epoch_distance * epoch))) / tdur
            - t0_shift_planet
            - (((per_bary - epoch_distance) * epoch) / (tdur / 2))
        )
    return x_bary
