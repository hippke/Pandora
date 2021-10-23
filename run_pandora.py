from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import batman

from pandora import pandora_model, pandora_model_array

# Create synthetic batman data, like Kepler LCs
np.random.seed(seed=42)  # reproducibility
t_start = 101
t_end = 104
t_dur = t_end - t_start
cadences_per_day = 48
cadences = int(cadences_per_day * t_dur)
time_array = np.linspace(t_start, t_end, cadences)
print("cadences", cadences)
#print(t)

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
stdev = 1e-7
noise = np.random.normal(0, stdev, cadences)
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
#M_sun = 2e30

transit_duration_planet = per_planet / pi * arcsin(sqrt(((r_planet/2) + R_star) ** 2) / a_planet)

# Set moon parameters
r_moon = 18000  # [km]
a_moon = 384000 * 3  # [km] <<<<------
e_moon = 0.0  # 0..1
Omega_moon = 0#20  # degrees
w_moon = 0#50.0  # degrees
i_moon = 90 #60.0  # 0..90 in degrees. 90 is edge on
tau_moon = -30
mass_ratio = 0.00001

per_moon = (2 * pi * sqrt((a_moon * 1000) ** 3 / (G * M_planet))) / 60 / 60 / 24
"""
print("moon period (days)", per_moon)
# OK to keep planetary transit duration at b=0 as the star is always R=1
#transit_duration_planet = 0.54
print("transit_duration (days, b=0)", transit_duration_planet)

# (x,y) grid in units of stellar radii; star at (0,0)
# Get x pos for start of time series
xpos_start = -t_dur / transit_duration_planet
xpos_end = t_dur / transit_duration_planet
print("xpos_start", xpos_start)
x_length = abs(2 * xpos_start)
print("x_length", x_length)
x_step = x_length / cadences
print("x_step", x_step)
print("total x_steps", int(x_length / x_step))


xpos_array = np.linspace(xpos_start, xpos_end, cadences)
flux_planet_array = np.ones(cadences)
flux_moon_array = np.ones(cadences)
x_offset = 0.0#1

import time as ttime
t1 = ttime.time()

# x_planet can use offset (for t0) 
# and other min/max-scale (for other tdur = semimajor axis)
for idx in range(len(xpos_array)):
	x_planet= xpos_array[idx] + x_offset
	time = time_array[idx]
	
	flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary = pandora_model(
	    r_moon,
	    a_moon,
	    per_moon,
	    tau_moon,
	    Omega_moon,
	    w_moon,
	    i_moon,
	    r_planet,
	    b_planet,
	    x_planet,
	    mass_ratio,
	    R_star,
	    u1,
	    u2,
	    time
	)
	flux_planet_array[idx] = flux_planet
	flux_moon_array[idx] = flux_moon
	
	#print(x_planet, flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary)

t2 = ttime.time()
print("Time", t2-t1)


"""
t0_planet = 0.1

# Set planet parameters
r_planet = 63710  # km
a_planet = 1 * 149597870.700  # [km]
b_planet = 0.0  # [0..1.x]; central transit is 0.
per_planet = 365.25  # [days]


# Array form:
flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary = pandora_model_array(
    r_moon,
    a_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    w_moon,
    i_moon,
    per_planet,
    a_planet,
    r_planet,
    b_planet,
    t0_planet,
    mass_ratio,
    R_star,
    u1,
    u2,
    time_array
	)

plt.plot(time_array, flux_planet, color="blue")
plt.plot(time_array, flux_moon, color="red")
plt.show()

"""

datapoints = 200  # in total time grid
factor_durations = 7  # Duration of time grid in units of b=0 planetary transit durations
t_start = -factor_durations * transit_duration_planet
t_end = factor_durations * transit_duration_planet
timegrid = np.linspace(t_start, t_end, datapoints)


    
t2 = ttime.time()
print("Time", t2-t1)


#plt.show()

import mplcyberpunk
plt.style.use("cyberpunk")

plt.rc('font',  family='serif', serif='Computer Modern Roman')
plt.rc('text', usetex=True)
size = 10
aspect_ratio = 0.5
plt.figure(figsize=(size, size*aspect_ratio))

ppm = 100
stdev = 10 ** -6 * ppm
noise = np.random.normal(0, stdev, int(datapoints))

#plt.plot(timegrid, planet_flux_array, color="blue", marker="s", linestyle="None")
plt.plot(timegrid[:180], moon_flux_array[:180], color = "white", linestyle="dashed")
plt.plot(timegrid, total_flux_array, color = "white")
plt.scatter(timegrid, total_flux_array+noise, color = "yellow", s=15, alpha=0.5, zorder=10)
#plt.scatter(timegrid, planet_flux_array+noise, color='orange', s=5, alpha=0.5, zorder=10)
#plt.scatter(timegrid, moon_flux_array+noise, color='red', s=5, alpha=0.5, zorder=10)

mplcyberpunk.add_glow_effects()
plt.xlim(-3, 2)
plt.show()
#plt.style.use("dark_background")

#plt.savefig('logo.pdf', bbox_inches='tight')#, dpi=600)

#plt.plot(x_planet_array, y_planet_array, color="blue")
#plt.plot(x_moon_array, y_moon_array, color = "black")
#plt.plot(timegrid, total_flux_array, color = "red")
#plt.show()

# convert *.png movie_bary.gif

"""