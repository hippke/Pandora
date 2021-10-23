from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import batman

from pandora import pandora


np.random.seed(seed=42)  # reproducibility
t_start = 100
t_end = 114
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

# Set moon parameters
r_moon = 18000  # [km]
a_moon = 384000 * 1   # [km]
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
a_planet = 1 * 149597870.700  # [km]
b_planet = 0.0  # [0..1.x]; central transit is 0.
per_planet = 365.25  # [days]

u = np.array([[u1, u2]])


# Model with 5 epochs:
# t0_planet_offset must be constant for all
# new variable: t0_planet: Time (in days) of first planet mid-transit in time series

# Example: First planet mid-transit at time t0_planet = 100  # days
t0_planet = 100  # days
epochs = 5

# Each epoch must contain a segment of data, centered at the planetary transit
# Each epoch must be the same time duration
epoch_duration = 3  # days
cadences_per_day = 48  # switch this to automatic calculation? What about gaps?


flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary, time_arrays = pandora(
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
    t0_planet,
    epochs,
    epoch_duration,
    cadences_per_day
)


# Create noise and merge with flux
stdev = 1e-5
noise = np.random.normal(0, stdev, len(time_arrays))
y = noise + flux_total

plt.plot(time_arrays, flux_planet, color="blue")
plt.plot(time_arrays, flux_moon, color="red")
plt.plot(time_arrays, flux_total, color="black")
plt.scatter(time_arrays, y, color="black", s=5)
plt.show()