from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import batman

from pandora import pandora_epoch


np.random.seed(seed=42)  # reproducibility
t_start = 100
t_end = 114
t_dur = t_end - t_start
cadences_per_day = 48
cadences = int(cadences_per_day * t_dur)
time_array = np.linspace(t_start, t_end, cadences)
print("cadences", cadences)

G = 6.67408 * 10 ** -11

# Set stellar parameters
R_star = 1 * 696342  # km
u1 = 0.5
u2 = 0.5

# Set moon parameters
r_moon = 18000  # [km]
a_moon = 384000 * 2   # [km]
Omega_moon = 10#20  # degrees
w_moon = 20#50.0  # degrees
i_moon = 80 #60.0  # 0..90 in degrees. 90 is edge on
tau_moon = 0
#mass_ratio = 0.1

M_moon = 6e22
M_planet = 6e24

t0_planet_offset = 1

# Set planet parameters
r_planet = 63710 / 2  # km
a_planet = 0.1 * 149597870.700  # [km]
b_planet = 0.0  # [0..1.x]; central transit is 0.
per_planet = 36.525  # [days]

u = np.array([[u1, u2]])

# Array form:
flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary = pandora_epoch(
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
    t0_planet_offset,
    M_planet,
    R_star,
    u,
    time_array
	)




"""
plt.plot(time_array, flux_planet, color="blue")
plt.plot(time_array, flux_moon, color="red")
plt.plot(time_array, flux_total, color="black")
#plt.scatter(time_array, y, color="black", s=5)
plt.show()
"""


# Model with 5 epochs:
# t0_planet_offset must be constant for all
# new variable: t0_planet: Time (in days) of first planet mid-transit in time series

# Example: First planet mid-transit at time t0_planet = 100  # days
t0_planet = 100  # days
epochs = 5
"""
t0_planet_transit_times = np.arange(
	start=t0_planet,
	stop=per_planet * epochs,
	step=per_planet
	)
"""
t0_planet_transit_times = np.arange(
	start=t0_planet,
	stop=t0_planet + per_planet * epochs,
	step=per_planet,
	#endpoint=True
	)
print(t0_planet_transit_times)

# tau_moon is the position of the moon on its orbit, given as [0..1]
# for the first timestamp of the first epoch
# Cannot be given in units of days in prior, because moon orbit period varies
# Example: Prior has tau in [5, 100] but model tests orbit with per_moon = 10
#          It would physically still be OK, because it is circular and wraps
#          around. However, the sampler would not converge when testing models.
# So, we use tau in [0..1] and propagate to following epochs manually

per_moon = (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * M_planet))) / 60 / 60 / 24
print("per_moon", per_moon)
print("per_planet", per_planet)
print("tau_moon first epoch", tau_moon)

# Stroboscopic effect
# Example: per_planet = 100d, per_moon = 20d ==> strobo_factor = 5
#          Moon has made 5 orbits between planet transits
# However, no need to convert anything here because ellipse_pos from avocado
# will subtract the (constant) tau from the period to determine orbit position
strobo_factor = per_planet / per_moon
print("strobo_factor", strobo_factor)


# Each epoch must contain a segment of data, centered at the planetary transit
# Each epoch must be the same time duration
epoch_duration = 4  # days
cadences_per_day = 48  # switch this to automatic calculation? What about gaps?

t_starts = t0_planet_transit_times - epoch_duration / 2  # array of epoch start dates [day]
t_ends = t0_planet_transit_times + epoch_duration / 2   # array of epoch end dates [day]

cadences = int(cadences_per_day * epoch_duration)

time_arrays = np.ones(shape=(epochs,cadences))
#print("time_array", time_array)

# Loop over epochs and call pandora_segment for each. Then, stitch together:
flux_planet_array = np.ones(shape=(epochs,cadences))
flux_moon_array = np.ones(shape=(epochs,cadences))
flux_total_array = np.ones(shape=(epochs,cadences))
px_bary_array = np.ones(shape=(epochs,cadences))
py_bary_array = np.ones(shape=(epochs,cadences))
mx_bary_array = np.ones(shape=(epochs,cadences))
my_bary_array = np.ones(shape=(epochs,cadences))

for epoch in range(epochs):
	#print("epoch", epoch)
	#print("taus[epoch]", taus[epoch])
	#print("time_array", np.linspace(t_starts[epoch], t_ends[epoch], cadences))
	time_array = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
	flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary = pandora_epoch(
	    r_moon,
	    a_moon,
	    tau_moon,#taus[epoch],
	    Omega_moon,
	    w_moon,
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
	    time_array
	)
	#print("flux_planet", flux_planet)
	flux_planet_array[epoch,:] = flux_planet
	flux_moon_array[epoch] = flux_moon
	flux_total_array[epoch] = flux_total
	px_bary_array[epoch] = px_bary
	py_bary_array[epoch] = py_bary
	mx_bary_array[epoch] = mx_bary
	my_bary_array[epoch] = my_bary
	time_arrays[epoch] = time_array

#print(flux_planet_array)
flux_planet_array = flux_planet_array.ravel()
time_arrays = time_arrays.ravel()
flux_moon_array = flux_moon_array.ravel()
flux_total_array = flux_total_array.ravel()

# Create noise and merge with flux
stdev = 1e-5
noise = np.random.normal(0, stdev, len(time_arrays))
y = noise + flux_total_array

plt.plot(time_arrays, flux_planet_array, color="blue")
plt.plot(time_arrays, flux_moon_array, color="red")
plt.plot(time_arrays, flux_total_array, color="black")
plt.scatter(time_arrays, y, color="black", s=5)
plt.show()