"""Test case for the simultaneous GP detrending (pandoramoon/gp.py): the HH22 system (the Pandora paper's Fig. 4: a
Neptune-size moon, P_moon = 1.28 d, four 2-day transit windows at 10-min cadence, 100 ppm white noise) with stellar
variability that has to be detrended:

  - rotation: a quasi-periodic signal, P_rot = 6 d, amplitudes 1500 and 600 ppm (fundamental and first harmonic):
    strong curvature across each 2-day window, larger than the moon's 1250 ppm dip;
  - granulation-like correlated noise: an SHO draw with rms 200 ppm and tau = 0.15 d (3.6 h), on the time scale
    of the moon's dips;
  - white noise 100 ppm (as HH22).

The GP model (one SHO term with fixed hyperparameters) does not match this trend exactly (realistic). A biweight
filter needs a window longer than the masked transit plus the moon's reach and leaves the 0.15 d component and
part of the curvature in the window.

    from gp_trend_case import make
    c = make(moon=True, seed=0)      # dict: time, flux, yerr, flux_model, trend, truth, mask, t_transits, ...
"""
import math

import numpy as np

import pandoramoon as pandora

R_SUN = 696342e3
HH22 = dict(t0_bary=20.0, epochs=4, epoch_duration=2.0, cadences_per_day=144, noise_ppm=100.0,
            r_moon=24622e3 / 696342e3, per_moon=1.28, tau_moon=0.3, Omega_moon=0.0, i_moon=80.0, M_moon_ratio=0.05395)
TREND = dict(p_rot=6.0, a1_ppm=1500.0, a2_ppm=600.0, sho_ppm=200.0, sho_tau=0.15)
MASK_HALFWIDTH = 0.5          # days around each planet transit: T14 / 2 = 0.29 d plus the moon's reach (~0.13 d)


def sho_draw(t, amp, tau, rng):
    """an exact draw of the SHO (Q = 1/sqrt(2)) GP at ascending times t (state-space simulation)"""
    from pandoramoon.gp import _sho_step
    s2 = amp * amp
    lam = 1.0 / (math.sqrt(2.0) * tau)
    x = rng.standard_normal(2) * amp
    out = np.empty(len(t))
    out[0] = x[0]
    for i in range(1, len(t)):
        f00, f01, f10, f11, q00, q01, q11 = _sho_step(lam * (t[i] - t[i - 1]), s2)
        x = np.array([f00 * x[0] + f01 * x[1], f10 * x[0] + f11 * x[1]])
        Q = np.array([[q00, q01], [q01, q11]])
        w, V = np.linalg.eigh(Q)
        x = x + V @ (np.sqrt(np.maximum(w, 0.0)) * rng.standard_normal(2))
        out[i] = x[0]
    return out


def make(moon=True, seed=0, trend=None, noise_ppm=None):
    """the light curve (output cadence = model time grid, no supersampling): dict with time, flux (model + trend +
    noise), yerr, flux_model, trend, truth (Pandora parameters), mask (True within MASK_HALFWIDTH of a transit),
    t_transits, epoch_distance, per_bary, r_planet, t0_bary"""
    tr = dict(TREND, **(trend or {}))
    c = dict(HH22)
    params = pandora.model_params()
    params.R_star = R_SUN
    params.u1, params.u2 = 0.4089, 0.2556
    params.per_bary = 365.25
    params.a_bary = 215.0
    params.r_planet = 0.1
    params.b_bary = 0.3
    params.t0_bary = c["t0_bary"]
    params.t0_bary_offset = 0.01
    params.M_planet = 1.8986e+27
    params.w_bary = 0
    params.ecc_bary = 0.0
    if moon:
        params.r_moon = c["r_moon"]
        params.per_moon = c["per_moon"]
        params.tau_moon = c["tau_moon"]
        params.Omega_moon = c["Omega_moon"]
        params.i_moon = c["i_moon"]
        params.M_moon = c["M_moon_ratio"] * params.M_planet
    else:
        params.r_moon, params.per_moon, params.tau_moon = 1e-8, 30.0, 0.0
        params.Omega_moon, params.i_moon, params.M_moon = 0.0, 0.0, 1e-8
    params.ecc_moon = 0
    params.w_moon = 0
    params.epochs = c["epochs"]
    params.epoch_duration = c["epoch_duration"]
    params.cadences_per_day = c["cadences_per_day"]
    params.epoch_distance = params.per_bary + 0.01
    params.supersampling_factor = 1
    params.occult_small_threshold = 0
    params.hill_sphere_threshold = 1.2
    t = pandora.time(params).grid()
    flux_model, _, _ = pandora.moon_model(params).light_curve(t)
    rng = np.random.default_rng(1000 + seed)
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    rot = (tr["a1_ppm"] * np.sin(2 * np.pi * t / tr["p_rot"] + ph1)
           + tr["a2_ppm"] * np.sin(4 * np.pi * t / tr["p_rot"] + ph2)) * 1e-6
    gran = sho_draw(t, tr["sho_ppm"] * 1e-6, tr["sho_tau"], rng) if tr["sho_ppm"] > 0 else 0.0 * t
    trend_ = rot + gran
    sig = (noise_ppm or c["noise_ppm"]) * 1e-6
    flux = np.asarray(flux_model) + trend_ + rng.normal(0, sig, len(t))
    yerr = np.full(len(t), sig)
    t_tr = params.t0_bary + params.t0_bary_offset + params.per_bary * np.arange(params.epochs)
    mask = np.min(np.abs(t[:, None] - t_tr[None, :]), axis=1) < MASK_HALFWIDTH
    truth = dict(u1=params.u1, u2=params.u2, R_star=params.R_star, per_bary=params.per_bary, a_bary=params.a_bary,
                 r_planet=params.r_planet, b_bary=params.b_bary, t0_bary=params.t0_bary,
                 t0_bary_offset=params.t0_bary_offset, M_planet=params.M_planet, r_moon=params.r_moon,
                 per_moon=params.per_moon, tau_moon=params.tau_moon, Omega_moon=params.Omega_moon,
                 i_moon=params.i_moon, M_moon=params.M_moon)
    return dict(time=np.asarray(t, dtype=np.float64), flux=flux, yerr=yerr, flux_model=np.asarray(flux_model),
                trend=trend_, truth=truth, mask=mask, t_transits=t_tr, epoch_distance=params.epoch_distance,
                per_bary=params.per_bary, r_planet=params.r_planet, t0_bary=params.t0_bary, moon=bool(moon),
                seed=seed, trend_spec=tr)


def biweight(c, window_length=1.5):
    """the classical route: wotan's biweight with the transit windows masked (the window must exceed the mask);
    returns the detrended flux and the trend"""
    from wotan import flatten
    fl, trend = flatten(c["time"], c["flux"], method="biweight", window_length=window_length, return_trend=True,
                        mask=c["mask"])
    return fl, trend
