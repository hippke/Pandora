from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import batman

from pandora import pandora_model

# Create synthetic batman data, like Kepler LCs
np.random.seed(seed=42)  # reproducibility
t_start = 100
t_end = 1100
t_dur = t_end - t_start
cadences_per_day = 48
cadences = int(cadences_per_day * t_dur)
time_array = np.linspace(t_start, t_end, cadences)
print("cadences", cadences)
"""
t_start = 101+365.25
t_end = 104+365.25
t_dur = t_end - t_start
cadences_per_day = 48
cadences = int(cadences_per_day * t_dur)
time_array2 = np.linspace(t_start, t_end, cadences)
print("cadences", cadences)

t_start = 101+365.25*2
t_end = 104+365.25*2
t_dur = t_end - t_start
cadences_per_day = 48
cadences = int(cadences_per_day * t_dur)
time_array3 = np.linspace(t_start, t_end, cadences)
print("cadences", cadences)

time_array = np.concatenate((time_array1, time_array2, time_array3))
"""

# Use batman to create transits
ma = batman.TransitParams()
ma.t0 = 102.5  # time of inferior conjunction; first transit is X days after start
ma.per = 365.25  # orbital period
ma.rp = 63710 / 696342  # 6371 planet radius (in units of stellar radii)
ma.a = 217  # semi-major axis (in units of stellar radii)
ma.inc = 90  # orbital inclination (in degrees)
ma.ecc = 0  # eccentricity
ma.w = 90  # longitude of periastron (in degrees)
ma.u = [0.5, 0.5]  # limb darkening coefficients
ma.limb_dark = "quadratic"  # limb darkening model
m = batman.TransitModel(ma, time_array)  # initializes model
y = m.light_curve(ma)  # calculates light curve

# Create noise and merge with flux
stdev = 1e-16
noise = np.random.normal(0, stdev, len(time_array))
y = y + noise

plt.plot(time_array, y, color="black")
#plt.show()



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
M_moon = M_planet / 1000000
#M_sun = 2e30

transit_duration_planet = per_planet / pi * arcsin(sqrt(((r_planet/2) + R_star) ** 2) / a_planet)

# Set moon parameters
r_moon = 18000  # [km]
a_moon = 384000 * 3  # [km]
Omega_moon = 10#20  # degrees
w_moon = 20#50.0  # degrees
i_moon = 80 #60.0  # 0..90 in degrees. 90 is edge on
tau_moon = 0.75
#mass_ratio = 0.1

per_moon = (2 * pi * sqrt((a_moon * 1000) ** 3 / (G * M_planet))) / 60 / 60 / 24

print("moon period (days)", per_moon)
# OK to keep planetary transit duration at b=0 as the star is always R=1
print("transit_duration (days, b=0)", transit_duration_planet)


t0_planet = 365

# Set planet parameters
r_planet = 63710  # km
a_planet = 1 * 149597870.700  # [km]
b_planet = 0.0  # [0..1.x]; central transit is 0.
per_planet = 365.25  # [days]

u = np.array([[u1, u2]])

# Array form:
flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary = pandora_model(
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
    t0_planet,
    M_planet,
    R_star,
    u,
    time_array
	)

import time as ttime
t1 = ttime.time()
for counter in range(1):
	flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary = pandora_model(
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
	    t0_planet,
	    M_planet,
	    R_star,
	    u,
	    time_array
		)
t2 = ttime.time()
print("Runtime", t2-t1)

print(np.sum(flux_total))
#assert np.sum(flux_total) == 143.72694875312

plt.plot(time_array, flux_planet, color="blue")
plt.plot(time_array, flux_moon, color="red")
plt.plot(time_array, flux_total, color="green", linestyle="dashed")
plt.show()
