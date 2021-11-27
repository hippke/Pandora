import numpy as np
from core import cci, eclipse, ellipse, ellipse_ecc, occult, occult_small, resample, timegrid, x_bary_grid, pixelart
import time
from numba import jit
import pandora


@jit(nopython=True, fastmath=True)
def batch_cci(r1, r2, d, n):
    for index in range(n):
        cci(r1, r2, d)
    return 0


print("Pandora benchmark 27 Nov 2021")


k = 0.01
u1 = 0.4
u2 = 0.6
print("Warm-up: occult_small")
occult_small(np.array([1]), k, u1, u2)

print("Warm-up: occult")
occult(np.array([1]), k, u1, u2)

print("Warm-up: Kepler ellipse (e=0)")
ellipse(5, 5, 1, 10, 10, np.array([1]), 0, np.array([1]), 0.5, 0.5)

print("Warm-up: Kepler ellipse (e=0.2)")
ellipse_ecc(5, 5, 0.2, 1, 10, 10, 30, np.array([1]), 0, np.array([1]), 0.5)

print("Warm-up: Numerical occultations")
pixelart(1, 0, 1, 0.01, 0.1, 0.01, 25)

print("Warm-up: Analytical occultations")
batch_cci(1, 0.1, 1.05, 1)

print("Warm-up: Full model")
# Call Pandora and get model with these parameters
params = pandora.model_params()
R_sun = 696342000
params.R_star = 1 * R_sun  # [m]
params.u1 = 0.4089  # [0..1] as per Claret & Bloemen 2011, logg=4.50; Teff=5750; log[M/H]=0.0; micro_turb_vel=0.0
params.u2 = 0.2556  # [0..1] as per Claret & Bloemen 2011, logg=4.50; Teff=5750; log[M/H]=0.0; micro_turb_vel=0.0

# Planet parameters
params.per_bary = 365.25  # [days]
params.a_bary = 215  # [R_star]
params.r_planet = 0.1  # [R_star]
params.b_bary = 0.3   # [0..1]
params.t0_bary = 11  # [days]
params.t0_bary_offset = 0  # [days]
params.M_planet = 1.8986e+27  # [kg]
params.w_bary = 20  # [deg]
params.ecc_bary = 0.2  # [0..1]  

# Moon parameters
params.r_moon = 0.03526 # [R_star],  R_ear: 0.00916, R_nep: 0.03526
params.per_moon = 0.3 # [days]
params.tau_moon = 0.07  # [0..1]
params.Omega_moon = 0  # [0..180]
params.i_moon = 80  # [0..180]
params.e_moon = 0.9  # [0..1]
params.w_moon = 20  # [deg]
params.mass_ratio = 0.05395   # [0..1]

# Other model parameters
params.epochs = 10  # [int]
params.epoch_duration = 1  # 5  # [days]
params.cadences_per_day = 48  # [int]
params.epoch_distance = 365.26   # [days] value close to per_planet, but not identical
params.supersampling_factor = 1  # [int]
params.occult_small_threshold = 0.01  # [0..1]
params.hill_sphere_threshold = 1.2

t = pandora.time(params).grid()
model = pandora.moon_model(params)
flux_total, flux_planet, flux_moon = model.light_curve(t)

print("Starting benchmark:")
n = 10_000_000
zs = np.linspace(0, 1, n)
t1 = time.perf_counter()
occult(zs, k, u1, u2)
t2 = time.perf_counter()
print('{:.2e}'.format(n/(t2-t1)), "occult per second")

n = 100_000_000
zs = np.linspace(0, 1, n)
t1 = time.perf_counter()
occult_small(zs, k, u1, u2)
t2 = time.perf_counter()
print('{:.2e}'.format(n/(t2-t1)), "occult_small per second")

n = 10_000_000
zs = np.linspace(0, 1, n)
t1 = time.perf_counter()
ellipse(5, 5, 1, 10, 10, zs, 0, zs, 0.5, 0.5)
t2 = time.perf_counter()
print('{:.2e}'.format(n/(t2-t1)), "Kepler ellipse (e=0) per second")

n = 10_000_000
zs = np.linspace(0, 1, n)
t1 = time.perf_counter()
ellipse_ecc(5, 5, 0.2, 1, 10, 10, 30, zs, 0, zs, 0.5)
t2 = time.perf_counter()
print('{:.2e}'.format(n/(t2-t1)), "Kepler ellipse (e=0.2) per second")


n = 100_000
t1 = time.perf_counter()
for idx in range(n):
    pixelart(1, 0, 1, 0.01, 0.1, 0.01, 25)
t2 = time.perf_counter()
print('{:.2e}'.format(n/(t2-t1)), "Numerical occultations (n=25) per second")

n = 10_000
t1 = time.perf_counter()
for idx in range(n):
    pixelart(1, 0, 1, 0.01, 0.1, 0.01, 100)
t2 = time.perf_counter()
print('{:.2e}'.format(n/(t2-t1)), "Numerical occultations (n=100) per second")

n = 10_000_000
t1 = time.perf_counter()
batch_cci(1, 0.1, 1.05, n)
t2 = time.perf_counter()
print('{:.2e}'.format(n/(t2-t1)), "Analytical occultations per second")

n = 10_000
t1 = time.perf_counter()
for idx in range(n):
    flux_total, flux_planet, flux_moon = model.light_curve(t)
t2 = time.perf_counter()
print(np.abs(np.sum(flux_total)))
assert np.abs(np.sum(flux_total)) - 477.71053987639004 < 1e-10
print('{:.0f}'.format(n/(t2-t1)), "Full models per second")