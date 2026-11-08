"""The system and data of examples/tutorial_characterisation.ipynb (cells 3, 5, 8, 10), for the GPU tests"""
import numpy as np
import pandoramoon as pandora

R_sun = 696_342_000


def system():
    p = pandora.model_params()
    p.R_star = R_sun; p.u1 = 0.4089; p.u2 = 0.2556
    p.per_bary = 365.25; p.a_bary = 215; p.r_planet = 0.1; p.b_bary = 0.3; p.t0_bary = 11; p.t0_bary_offset = 0.01
    p.M_planet = 1.8986e27; p.w_bary = 0; p.ecc_bary = 0
    p.r_moon = 0.03; p.per_moon = 5; p.tau_moon = 0.07; p.Omega_moon = 0; p.i_moon = 90; p.ecc_moon = 0; p.w_moon = 0
    p.M_moon = 0.054 * p.M_planet
    p.epochs = 3; p.epoch_duration = 10; p.cadences_per_day = 48; p.epoch_distance = 365.25
    p.occult_small_threshold = 0; p.hill_sphere_threshold = 1.0
    return p


def make(seed=1):
    """(params, Likelihood, default priors) as in the notebook"""
    p = system()
    rng = np.random.default_rng(seed)
    p.supersampling_factor = 1
    time = pandora.time(p).grid()
    p.supersampling_factor = 30
    flux_true, _, _ = pandora.moon_model(p).light_curve(pandora.time(p).grid())
    p.supersampling_factor = 1
    star = 1 + 400e-6 * np.sin(2 * np.pi * time / 6.3 + 1.0) + 600e-6 * np.sin(2 * np.pi * time / 13.7 + 2.0)
    a = np.exp(-1 / 6.0)
    kick = rng.normal(0, 1, len(time))
    red = np.zeros(len(time))
    for i in range(1, len(time)):
        red[i] = a * red[i - 1] + np.sqrt(1 - a * a) * kick[i]
    flux = flux_true * (star + 120e-6 * red) + rng.normal(0, 400e-6, len(time))
    T14 = p.per_bary / np.pi * np.arcsin(np.sqrt((1 + p.r_planet) ** 2 - p.b_bary ** 2) / p.a_bary)
    lc = pandora.prepare_lightcurve(time, flux, t0=p.t0_bary + p.t0_bary_offset, period=p.per_bary, duration=T14)
    like = pandora.Likelihood(lc["time"], lc["flux"], lc["flux_err"], epoch_distance=p.per_bary,
                              hill_sphere_threshold=1.0)
    priors = pandora.default_priors(t0_bary=p.t0_bary, per_bary=p.per_bary, r_planet=0.1)
    return p, like, priors


def truth_theta(p):
    return np.array([p.u1, p.u2, p.R_star, p.per_bary, p.a_bary, p.r_planet, p.b_bary, p.w_bary, p.ecc_bary,
                     p.t0_bary, p.t0_bary_offset, p.M_planet, p.r_moon, p.per_moon, p.tau_moon, p.Omega_moon,
                     p.i_moon, p.ecc_moon, p.w_moon, p.M_moon], dtype=np.float64)
