import numpy as np
from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
from numpy import sin, pi
import matplotlib.pyplot as plt
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot
from pandora import pandora_model
from numba import jit

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
    params[0]  = cube[0]  * 25000 + 1 # r_moon (0, 25000) [km]
    params[1]  = cube[1]  * 2000000 + 20000 # a_moon (20000, 2_020_000) [km]
    params[2]  = cube[2]  # tau_moon (normalized 0..1)
    params[3]  = cube[3]  * 200 + 1 # Omega_moon
    params[4]  = cube[4]  * 90 # w_moon
    params[5]  = cube[5]  * 90 # i_moon
    params[6]  = cube[6]  * 2e22 + 5e22  # M_moon
    params[7]  = cube[7]  * 10 + 360 # per_planet
    params[8]  = cube[8]  * 14959787 + 142117976.5 # a_planet
    params[9]  = cube[9]  * 100000 # r_planet
    params[10] = cube[10] # b_planet
    params[11] = cube[11] - 0.5 # t0_planet
    params[12] = cube[12] * 2e24 + 5e24 # M_planet
    return params


@jit(cache=True, nopython=True, fastmath=True)
def log_likelihood(params):

    r_moon,a_moon,tau_moon,Omega_moon,w_moon,i_moon,M_moon,per_planet,a_planet,r_planet,b_planet,t0_planet,M_planet = params
    _, _, flux_total, _, _, _, _ = pandora_model(
        r_moon, a_moon,per_moon,tau_moon,Omega_moon,w_moon,i_moon,per_planet,a_planet,r_planet,b_planet,t0_planet,M_planet,
        R_star=1 * 696342,  # km
        u=u,
        time=time_array
    )
    loglike = -0.5 * (((flux_total - testdata) / yerr)**2).sum()
    return loglike



# Create model to be recovered later
np.random.seed(seed=42)  # reproducibility
t_start = 101
t_end = 104
t_dur = t_end - t_start
cadences_per_day = 48
cadences = int(cadences_per_day * t_dur)
time_array = np.linspace(t_start, t_end, cadences)
#print("cadences", cadences)

G = 6.67408 * 10 ** -11

# Set stellar parameters
R_star = 1 * 696342  # km
u1 = 0.5
u2 = 0.5

# Set planet parameters
r_planet = 63710  # km
a_planet = 1 * 149597870.700  # [km]
b_planet = 0.0  # [0..1.x]; central transit is 0.
per_planet = 365.25  # [days]
M_planet = 5.972 * 10 ** 24
#M_sun = 2e30

transit_duration_planet = per_planet / pi * arcsin(sqrt(((r_planet/2) + R_star) ** 2) / a_planet)

# Set moon parameters
r_moon_fake = 18000  # [km]
a_moon = 384000 * 3  # [km]
Omega_moon = 10#20  # degrees
w_moon = 20#50.0  # degrees
i_moon = 80 #60.0  # 0..90 in degrees. 90 is edge on
tau_moon = 0.25
mass_ratio = 0.1
u = np.array([[u1, u2]])
per_moon = (2 * pi * sqrt((a_moon * 1000) ** 3 / (G * M_planet))) / 60 / 60 / 24

M_moon = 6e22

#print("moon period (days)", per_moon)
# OK to keep planetary transit duration at b=0 as the star is always R=1
#print("transit_duration (days, b=0)", transit_duration_planet)


t0_planet = 0.1

# Set planet parameters
r_planet = 63710  # km
a_planet = 1 * 149597870.700  # [km]
b_planet = 0.0  # [0..1.x]; central transit is 0.
per_planet = 365.25  # [days]


# Array form:
flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary = pandora_model(
    r_moon_fake,
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
    t0_planet,
    M_planet,
    R_star,
    u,
    time_array
    )


# Create noise and merge with flux
stdev = 1e-5
noise = np.random.normal(0, stdev, len(time_array))
testdata = flux_total + noise
yerr = np.full(len(time_array), stdev)
#print(yerr)

plt.plot(time_array, flux_planet, color="blue")
plt.plot(time_array, flux_moon, color="red")
plt.plot(time_array, flux_total, color="green", linestyle="dashed")
plt.scatter(time_array, testdata, s=20, marker="x", color="black")
plt.show()



# Recover the test data

parameters = [
    'r_moon', 
    'a_moon', 
    'tau_moon', 
    'Omega_moon',
    'w_moon',
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
    False
    ],
)

import ultranest.stepsampler
import ultranest

nsteps = 2 * len(parameters)
sampler2.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=nsteps)
result2 = sampler2.run(min_num_live_points=400)
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
