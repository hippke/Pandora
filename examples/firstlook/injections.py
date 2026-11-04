"""Injected light curves for calibrating and benchmarking the first look (white noise).

Regimes (equally important):
  kepler : 30-min cadence, 4-yr baseline: P_b 150-450 d -> 3-9 transits; noise 100-1000 ppm per 30 min
  plato  : 10-min cadence, 2 or 3 transits; P_b 200-400 d; noise 100-1000 ppm per 10 min
Planet: Jupiter-like (r_p 0.075-0.105 R_star, M_p 1.6-2.4e27 kg) around a Sun (a_bary from Kepler's law),
b 0-0.7. Moon (if any): M_m / M_p log-uniform 0.003-0.1, r_m log-uniform 0.012-0.05 R_star, a_m 0.08-0.45
Hill radii (P_m <= 40 d), tau, Omega uniform, i_moon uniform 0-180 deg.

    import injections
    sysd = injections.draw(17, "kepler", moon=True)       # deterministic in (index, regime, moon)
    t, flux, err = injections.simulate(sysd)
    sysd["priors"]   # per_bary, a_bary prior boxes for FirstLook
    # long-cadence data with exposure smearing (truth averaged over 30 sub-exposures per cadence), fitted with
    # supersampling factor 5 (t = the fit's sub-exposure grid, flux / err at the cadences; SAMPLING.md backlog 7):
    t, flux, err = injections.simulate(sysd, true_ssf=30, fit_ssf=5)
"""
import math

import numpy as np

import pandoramoon as pandora

R_SUN = 696_342_000.0
M_SUN = 1.989e30
G = 6.67408e-11
DAY = 86400.0
LD = (0.4089, 0.2556)


REGIMES = {"kepler": 1, "plato": 2, "kepler-faint": 3, "plato-faint": 4}


def draw(i, regime, moon=True, seed=2026):
    """regime: kepler | plato, or kepler-faint | plato-faint (noise 600-4000 ppm: near the detection limit)"""
    rng = np.random.default_rng([seed, i, REGIMES[regime], int(moon)])
    nlo, nhi = (600.0, 4000.0) if regime.endswith("-faint") else (100.0, 1000.0)
    for _ in range(1000):
        if regime.startswith("kepler"):
            P = rng.uniform(150.0, 450.0)
            epochs = int(min(1460.0 // P, 9))
            cad, noise = 48, math.exp(rng.uniform(math.log(nlo), math.log(nhi)))
        else:
            P = rng.uniform(200.0, 400.0)
            epochs = int(rng.choice([2, 3]))
            cad, noise = 144, math.exp(rng.uniform(math.log(nlo), math.log(nhi)))
        Mp = rng.uniform(1.6e27, 2.4e27)
        a_b = (G * (M_SUN + Mp) * (P * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / R_SUN
        s = dict(index=int(i), regime=regime, moon=bool(moon), per_bary=P, epochs=epochs, cadences_per_day=cad,
                 noise_ppm=noise, a_bary=a_b, r_planet=rng.uniform(0.075, 0.105), b_bary=rng.uniform(0.0, 0.7),
                 t0_bary=11.0, t0_bary_offset=rng.uniform(-0.02, 0.02), M_planet=Mp)
        v = 2 * math.pi * a_b * R_SUN / P                      # orbital speed [m/d]
        if moon:
            q = math.exp(rng.uniform(math.log(0.003), math.log(0.1)))
            Mm = q * Mp
            r_m = math.exp(rng.uniform(math.log(0.012), math.log(0.05)))
            R_H = a_b * R_SUN * (Mp / (3 * M_SUN)) ** (1 / 3)
            a_m = rng.uniform(0.08, 0.45) * R_H
            Pm = 2 * math.pi * math.sqrt(a_m ** 3 / (G * (Mp + Mm))) / DAY
            if Pm > 40.0 or a_m / R_SUN < 1.5 * (s["r_planet"] + r_m):
                continue
            s.update(r_moon=r_m, per_moon=Pm, tau_moon=rng.uniform(), Omega_moon=rng.uniform(0, 180),
                     i_moon=rng.uniform(0, 180), M_moon=Mm, a_moon_m=a_m)
            reach = a_m                                        # the moon's maximal sky offset from the planet
        else:
            s.update(r_moon=1e-8, per_moon=30.0, tau_moon=0.0, Omega_moon=0.0, i_moon=0.0, M_moon=1e-8, a_moon_m=0.0)
            reach = 0.0
        # window per transit: transit duration + the moon's maximal offset (+ margin), at least 1.5 d
        dur = P / math.pi * (1 + s["r_planet"]) / a_b
        s["epoch_duration"] = max(1.5, 2 * (1.15 * reach / v + dur))
        s["epoch_distance"] = P + 0.01
        dP = 0.0137 * P                                        # +-1.37 % (the notebook's 360.25-370.25 for 365.25)
        s["priors"] = dict(per_bary=(P - dP, P + dP), a_bary=(0.977 * a_b, 1.023 * a_b))
        return s
    raise RuntimeError("no valid system")


def simulate(s, noise_seed=None, true_ssf=1, fit_ssf=1):
    """light curve of system s: (time, flux, err). true_ssf: supersampling factor of the injected truth (> 1:
    each cadence is the average over true_ssf sub-exposures, i.e. real exposure smearing; 1 = instantaneous flux at
    the cadence mid-times, as in Round 11). fit_ssf: the returned time grid is the sub-exposure grid of that factor
    (for Likelihood / FirstLook(supersampling_factor=fit_ssf)); flux and err are at the cadences either way. The
    noise is the same for every (true_ssf, fit_ssf) of a system."""
    p = pandora.model_params()
    p.R_star = R_SUN
    p.u1, p.u2 = LD
    for k in ("per_bary", "a_bary", "r_planet", "b_bary", "t0_bary", "t0_bary_offset", "M_planet", "r_moon",
              "per_moon", "tau_moon", "Omega_moon", "i_moon", "M_moon", "epochs", "epoch_duration",
              "cadences_per_day", "epoch_distance"):
        setattr(p, k, s[k])
    p.w_bary = 0
    p.ecc_bary = 0.0
    p.ecc_moon = 0
    p.w_moon = 0
    p.supersampling_factor = int(true_ssf)
    p.occult_small_threshold = 0
    # the fits use 1.0 (FirstLook); was 1.2 until Round 17 (no injection lies above 0.45 Hill radii: data unchanged)
    p.hill_sphere_threshold = 1.0
    flux, _, _ = pandora.moon_model(p).light_curve(pandora.time(p).grid())     # at the cadences
    p.supersampling_factor = int(fit_ssf)
    t = pandora.time(p).grid()
    rng = np.random.default_rng(noise_seed if noise_seed is not None else
                                [7, s["index"], REGIMES[s["regime"]], int(s["moon"])])
    err = np.full(len(flux), s["noise_ppm"] * 1e-6)
    return t, flux + rng.normal(0.0, 1.0, len(flux)) * err, err


def planet_snr(s):
    """photometric S/N of the planet's transits: depth r_p^2 x sqrt(in-transit cadences) / noise per cadence"""
    dur = s["per_bary"] / math.pi * (1 + s["r_planet"]) / s["a_bary"]
    return float(s["r_planet"] ** 2 * math.sqrt(s["epochs"] * dur * s["cadences_per_day"]) / (s["noise_ppm"] * 1e-6))


def moon_snr(s, t, err):
    """photometric S/N of the moon's own transits: depth r_m^2 times sqrt(points in the moon's transits) / noise,
    with the moon's transit duration ~ the planet's (a proxy label; the reference fit decides detectability)"""
    if not s["moon"]:
        return 0.0
    dur = s["per_bary"] / math.pi * (1 + s["r_moon"]) / s["a_bary"]
    n_in = s["epochs"] * dur * s["cadences_per_day"]
    return float(s["r_moon"] ** 2 * math.sqrt(n_in) / (s["noise_ppm"] * 1e-6))
