import numpy as np
from numpy import sqrt, pi, cos, arcsin
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def timegrid(t0_bary, epochs, epoch_duration, cadences_per_day, epoch_distance, supersampling_factor):
    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_bary`, however, is a free parameter
    start = np.float64(t0_bary)
    step = np.float64(epoch_distance)
    stop = np.float64(t0_bary + epoch_distance * epochs)
    ti_epoch_midtimes = np.arange(start, stop, step)

    # "Morphological light-curve distortions due to finite integration time"
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract
    # Data gets smeared over long integration. Relevant for e.g., 30min cadences
    # To counter the effect: Set supersampling_factor = 5 (recommended value)
    # Then, 5x denser in time sampling, and averaging after, approximates effect
    if supersampling_factor < 1:
        raise ValueError("supersampling_factor must be a positive integer")
    # Cadences are spaced by exactly 1 / cadences_per_day. Each cadence is an
    # exposure of that length, split into supersampling_factor sub-exposures
    # whose mid-times are returned (their mean is the cadence mid-time). The
    # int(...) cadences fill at most epoch_duration; the block is centred on
    # the epoch mid-time, so for integer cadences_per_day * epoch_duration
    # the exposures tile [t_start, t_end] exactly.
    # The tolerance keeps float round-off (e.g. 720 * 0.35 = 251.99999999999997)
    # from dropping a cadence; double round-off is ~1e-16 relative.
    n_float = np.float64(cadences_per_day) * np.float64(epoch_duration)
    cadences = int(n_float + 1e-9 * max(1.0, n_float)) * int(supersampling_factor)
    substep = 1.0 / (np.float64(cadences_per_day) * int(supersampling_factor))
    offsets = (np.arange(cadences) + 0.5 - cadences / 2) * substep
    time = np.empty(shape=(epochs, cadences))
    for epoch in range(epochs):
        time[epoch] = ti_epoch_midtimes[epoch] + offsets
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
        epoch = int(np.floor((time[idx] - t0_bary) / per_bary + 0.5))
        x_bary[idx] = (
            (2 * (time[idx] - (t0_bary + epoch_distance * epoch))) / tdur
            - t0_shift_planet
            - (((per_bary - epoch_distance) * epoch) / (tdur / 2))
        )
    return x_bary


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_window(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary,
    w_bary, reach
):
    """Like x_bary_grid(), but returns only the samples with |x_bary| < reach:
    (indices into time, x_bary at those indices). NaN is kept. No full-length
    array is written, which matters for long time series (memory traffic).
    Keep the formulas identical to x_bary_grid()."""
    tdur = per_bary / pi * arcsin(1 / a_bary)
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))
    t0_shift_planet = t0_bary_offset / (tdur / 2)
    n = len(time)
    idx = np.empty(n, dtype=np.int64)
    x_win = np.empty(n)
    m = 0
    for i in range(n):
        epoch = int(np.floor((time[i] - t0_bary) / per_bary + 0.5))
        x = (
            (2 * (time[i] - (t0_bary + epoch_distance * epoch))) / tdur
            - t0_shift_planet
            - (((per_bary - epoch_distance) * epoch) / (tdur / 2))
        )
        if not (abs(x) >= reach):
            idx[m] = i
            x_win[m] = x
            m += 1
    return idx[:m].copy(), x_win[:m].copy()


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _x_bary_window_t(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary,
    w_bary, reach
):
    """x_bary_window() plus the window times time[idx]"""
    idx, x_win = x_bary_window(
        time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
        ecc_bary, w_bary, reach
    )
    return idx, x_win, time[idx]


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_runs_sorted(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary,
    w_bary, reach
):
    """The search of x_bary_window_sorted_t() without filling the window
    arrays (P18): (ok, los, his, c_t, c_d, tdur, t0_shift_planet). Per
    epoch k the window samples are the run time[los[k]:his[k]] (possibly
    empty; runs are ascending and disjoint), and x_bary of a sample t in it
    is (2 * (t - c_t[k])) / tdur - t0_shift_planet - c_d[k]. ok is False
    if the search does not apply (see x_bary_window_sorted_t(); the caller
    must then use the scan). time MUST be ascending."""
    tdur = per_bary / pi * arcsin(1 / a_bary)
    if ecc_bary > 0:
        tdur /= 1 / sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * cos((w_bary - 90) / 180 * pi))
    t0_shift_planet = t0_bary_offset / (tdur / 2)
    n = len(time)
    half = reach * tdur / 2  # [days]
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0)
    if n == 0:
        return True, empty_i, empty_i, empty_f, empty_f, tdur, t0_shift_planet
    # Margin covers round-off of t_mid and of x (|x| errors ~1e-12)
    pad = 1e-6 * tdur + 1e-9 * (abs(time[0]) + abs(time[n - 1]))
    if not (half + abs(t0_bary_offset) + pad < 0.45 * per_bary):
        return False, empty_i, empty_i, empty_f, empty_f, tdur, t0_shift_planet
    e_first = int(np.floor((time[0] - t0_bary) / per_bary + 0.5))
    e_last = int(np.floor((time[n - 1] - t0_bary) / per_bary + 0.5))
    n_ep = e_last - e_first + 1
    if n_ep > n:  # more epochs than samples: scan instead
        return False, empty_i, empty_i, empty_f, empty_f, tdur, t0_shift_planet
    # Per epoch: exact run [lo, hi) of window samples (the windows are small)
    los = np.empty(n_ep, dtype=np.int64)
    his = np.empty(n_ep, dtype=np.int64)
    c_ts = np.empty(n_ep)
    c_ds = np.empty(n_ep)
    for k in range(n_ep):
        e = e_first + k
        t_mid = t0_bary + per_bary * e + t0_bary_offset
        lo = np.searchsorted(time, t_mid - half - pad, side="left")
        hi = np.searchsorted(time, t_mid + half + pad, side="right")
        c_t = t0_bary + epoch_distance * e
        c_d = ((per_bary - epoch_distance) * e) / (tdur / 2)
        while lo < hi and abs((2 * (time[lo] - c_t)) / tdur - t0_shift_planet - c_d) >= reach:
            lo += 1
        while hi > lo and abs((2 * (time[hi - 1] - c_t)) / tdur - t0_shift_planet - c_d) >= reach:
            hi -= 1
        los[k] = lo
        his[k] = hi
        c_ts[k] = c_t
        c_ds[k] = c_d
    return True, los, his, c_ts, c_ds, tdur, t0_shift_planet


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def runs_x_t(time, los, his, c_t, c_d, tdur, t0_shift_planet):
    """Window arrays (idx, x_bary, time[idx]) of the runs of
    x_bary_runs_sorted(), filled branch-free per run"""
    total = 0
    for k in range(len(los)):
        total += his[k] - los[k]
    idx = np.empty(total, dtype=np.int64)
    x_win = np.empty(total)
    t_win = np.empty(total)
    m = 0
    for k in range(len(los)):
        ct = c_t[k]
        cd = c_d[k]
        lo = los[k]
        cnt = his[k] - lo
        for j in range(cnt):
            t = time[lo + j]
            idx[m + j] = lo + j
            t_win[m + j] = t
            x_win[m + j] = (2 * (t - ct)) / tdur - t0_shift_planet - cd
        m += cnt
    return idx, x_win, t_win


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_window_sorted_t(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary,
    w_bary, reach
):
    """Window search for ascending time: (idx, x_bary, time[idx]) as
    x_bary_window(), but only visits samples near each transit. Per epoch
    e, x_bary is linear in time, x = 2 (t - t_mid) / tdur with t_mid = t0_bary + per_bary * e +
    t0_bary_offset, so |x| < reach only for |t - t_mid| < reach * tdur / 2.
    That interval (plus a margin) is found by binary search.

    P11: the guard below puts every candidate sample within 0.45 per_bary
    of t0_bary + per_bary * e, so x_bary_window()'s per-sample epoch
    floor((t - t0_bary) / per_bary + 0.5) equals e: the epoch terms are
    hoisted out of the per-sample loop (this also avoids the fastmath
    re-association of t - (t0_bary + epoch_distance * e), so the returned x
    is within ~1 ulp of exact; the scan's x is off by up to ~4e-13). Inside
    one epoch x is non-decreasing in t (each rounded operation is
    monotonic), so the samples with |x| < reach form one contiguous run per
    epoch: only the margin samples at both ends are tested, and the run is
    filled branch-free (vectorised) with idx, x and t. The index set equals
    x_bary_window()'s except possibly for samples with |x| within ~1e-12
    of reach, where the different x round-off decides; such samples have
    flux exactly 1 either way (reach carries a 1e-6 margin).

    Falls back to x_bary_window() if the window is not well inside one
    epoch (huge moon orbits, very short periods) or for non-finite
    constants. time MUST be ascending (not checked here; Likelihood checks
    it once). P18: the search is x_bary_runs_sorted(), the fill
    runs_x_t()."""
    ok, los, his, c_t, c_d, tdur, t0_shift_planet = x_bary_runs_sorted(
        time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
        ecc_bary, w_bary, reach
    )
    if not ok:
        return _x_bary_window_t(
            time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
            ecc_bary, w_bary, reach
        )
    return runs_x_t(time, los, his, c_t, c_d, tdur, t0_shift_planet)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def x_bary_window_sorted(
    time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance, ecc_bary,
    w_bary, reach
):
    """x_bary_window() for ascending time, found by binary search per
    epoch: (idx, x_bary). See x_bary_window_sorted_t(), which also returns
    the window times; x agrees with x_bary_window() to ~1e-12 (the index
    set too, except at |x| = reach within round-off). time MUST be
    ascending."""
    idx, x_win, _ = x_bary_window_sorted_t(
        time, a_bary, per_bary, t0_bary, t0_bary_offset, epoch_distance,
        ecc_bary, w_bary, reach
    )
    return idx, x_win
