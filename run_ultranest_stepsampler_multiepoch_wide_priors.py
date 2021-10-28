import numpy as np
from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
from numpy import sin, pi
import matplotlib.pyplot as plt
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot
from pandora import pandora
from numba import jit


@jit(cache=True, nopython=True, fastmath=True)
def prior_transform(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    # 0  r_moon
    # 1  a_moon
    # 2  tau_moon
    # 3  Omega_moon
    # 4  w_moon
    # 5  i_moon
    # 6  M_moon
    # 7  per_planet
    # 8  a_planet
    # 9  r_planet
    # 10 b_planet
    # 11 t0_planet
    # 12 M_planet

    params     = cube.copy()
    params[0]  = cube[0]  * 50000 + 1 # r_moon (0, 25000) [km]
    params[1]  = cube[1]  * 3000000 + 1000 # a_moon (20000, 2_020_000) [km]
    params[2]  = cube[2]  / 2 # tau_moon (normalized 0..1)
    params[3]  = cube[3]  * 90 # Omega_moon
    params[4]  = cube[4]  * 90 # i_moon
    params[5]  = cube[5]  * 1e27 + 5e24 # M_moon
    params[6]  = cube[6]  * 10 + 360 # per_planet
    params[7]  = cube[7]  * 1.5e8 + 5e7 # a_planet
    params[8]  = cube[8]  * 100000 # r_planet
    params[9] = cube[9] # b_planet
    params[10] = cube[10] - 0.5 # t0_planet_offset
    params[11] = cube[11] * 1e27 + 5e24 # M_planet
    return params


@jit(cache=True, nopython=True, fastmath=True)
def log_likelihood(params):

    r_moon,a_moon,tau_moon,Omega_moon,i_moon,M_moon,per_planet,a_planet,r_planet,b_planet,t0_planet_offset,M_planet = params
    flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary, time_arrays = pandora(
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
        R_star=1 * 696342,  # km
        u=u,
        t0_planet=t0_planet,
        epochs=epochs,
        epoch_duration=epoch_duration,
        cadences_per_day=cadences_per_day,
        epoch_distance = epoch_distance
    )
    loglike = -0.5 * (((flux_total - testdata) / yerr)**2).sum()
    return loglike





# Set stellar parameters
# Please leave these as is right now. It's all in units of km. Will change units later.
R_star = 696342  # km
u1 = 0.5  # Currently not a free parameter, please leave as is!
u2 = 0.5  # We assume that we have u1, u1 from spectroscopy and stellar model
u = np.array([[u1, u2]])


# Here the fun begins. Choose these parameters:

# Planet parameters
r_planet = 63710  # radius [km] My prior: 1..100,000 km
a_planet = 149597870.700  # semimajor axis [km] My prior: 5e7..2e8 km
b_planet = 0.4  # impact parameter [0..1.x] central transit is 0. My prior: 0..1
per_planet = 365.25  # period [days] My prior: 360..370 days
M_planet = 5e27  # mass [kg] My prior: 5e24..5e27
t0_planet_offset = 0.1  # true offset from assumed t0; constant for all periods [days]. My prior: -0.5..+0.5

# Set moon parameters
r_moon = 18000  # radius [km] My prior: 1..50,000 km
a_moon = 2e6  # semimajor axis [km] My prior: 10,000...3,000,000 km
Omega_moon = 5  # degrees [0..90] My prior: 0..90 deg
i_moon = 85    # degrees [0..90]. 90 is edge-on. My prior: 0..90 deg
tau_moon = 0.25  # moon orbital position [0..0.5] My prior: 0..0.5
M_moon = 5e25  # mass [kg] My prior: 5e24..5e27


# The following parameters are not part of the physical mode,
# but used to generate synthetic data
t0_planet = 111  # [days] first planetary mid-transit time
epochs = 10  # Centered approximately at each planetary transit
epoch_duration = 5  # [days] data worth around each transit. Identical for all epochs.
cadences_per_day = 48  # Kepler: 30min LC
epoch_distance = 365.25  # [days] Constant time distance between each epoch.
# Should be from a planet-only model. Must cover planet+moon transit.
# For this experiment, please choose a value close (but not identical) to the true period.

noise_level = 1e-4  # Gaussian noise to be added to the generated data
# Let's not add TOOOO much noise for now. We can test "fishing the noise" limits later



flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary, time_arrays = pandora(
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
)


# Create noise and merge with flux
stdev = 1e-4
noise = np.random.normal(0, stdev, len(time_arrays))
testdata = noise + flux_total
yerr = np.full(len(testdata), stdev)

plt.plot(time_arrays, flux_planet, color="blue")
plt.plot(time_arrays, flux_moon, color="red")
plt.plot(time_arrays, flux_total, color="black")
plt.scatter(time_arrays, testdata, color="black", s=5)
plt.show()


# Recover the test data

parameters = [
    'r_moon', 
    'a_moon', 
    'tau_moon', 
    'Omega_moon',
    'i_moon',
    'M_moon',
    'per_planet',
    'a_planet',
    'r_planet',
    'b_planet',
    't0_planet',
    'M_planet'
    ]

sampler2 = ReactiveNestedSampler(parameters, log_likelihood, prior_transform,
    wrapped_params=[
    False,
    False,
    True, 
    False, 
    False, 
    False, 
    False, 
    False, 
    False, 
    False, 
    False, 
    False, 
    ],
)

import ultranest.stepsampler
import ultranest

nsteps = 2 * len(parameters)
sampler2.stepsampler = ultranest.stepsampler.CubeSliceSampler(nsteps=nsteps)
result2 = sampler2.run()#min_num_live_points=400, update_interval_ncall=1000)
sampler2.print_results()


cornerplot(result2)
plt.show()
"""
plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.errorbar(x=t, y=y, yerr=yerr,
             marker='o', ls=' ', color='orange')

plt.show()
"""

#cornerplot(result)
"""
plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.errorbar(x=t, y=y, yerr=yerr,
             marker='o', ls=' ', color='orange')
"""
#plt.show()
