import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
from pandora import pandora
import time as ttime

# Set stellar parameters
# Please leave these as is right now. It's all in units of km. Will change units later.
R_star = 696342  # km
u1 = 0.5  # Currently not a free parameter, please leave as is!
u2 = 0.5  # We assume that we have u1, u1 from spectroscopy and stellar model
u = np.array([[u1, u2]])


# Here the fun begins. Choose these parameters:

# Planet parameters
r_planet = 6371  # radius [km] My prior: 1..100,000 km
a_planet = 149597870.700  # semimajor axis [km] My prior: 5e7..2e8 km
b_planet = 0.4  # impact parameter [0..1.x] central transit is 0. My prior: 0..1
per_planet = 365.25  # period [days] My prior: 360..370 days
M_planet = 5e27  # mass [kg] My prior: 5e24..5e27
t0_planet_offset = 0.1  # true offset from assumed t0; constant for all periods [days]. My prior: -0.5..+0.5

# Set moon parameters
r_moon = 1800  # radius [km] My prior: 1..50,000 km
a_moon = 2e6  # semimajor axis [km] My prior: 10,000...3,000,000 km
Omega_moon = 5  # degrees [0..90] My prior: 0..90 deg
i_moon = 85  # degrees [0..90]. 90 is edge-on. My prior: 0..90 deg
tau_moon = 0.25  # moon orbital position [0..0.5] My prior: 0..0.5
M_moon = 5e25  # mass [kg] My prior: 5e24..5e27


# The following parameters are not part of the physical mode,
# but used to generate synthetic data
t0_planet = 111  # [days] first planetary mid-transit time
epochs = 8  # Centered approximately at each planetary transit
# Can in principle be any value. Let's choose something realistic, i.e. 3--15.
epoch_duration = 2.5  # [days] data worth around each transit. Identical for all epochs.
cadences_per_day = 48  # Kepler: 30min LC. Let's keep that fixed for now (untested)
epoch_distance = 365.25  # [days] Constant time distance between each epoch.
# Should be from a planet-only model. Must cover planet+moon transit.
# For this experiment, please choose a value close (but not identical) to the true period.

noise_level = 3e-4  # Gaussian noise to be added to the generated data
# Let's not add TOOOO much noise for now. We can test "fishing the noise" limits later

# Call Pandora and get model with these parameters
flux_planet_original1, flux_moon_original1, flux_total_original1, px_bary, py_bary, mx_bary, my_bary, time_arrays_original = pandora(
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
    epoch_distance,
    supersampling_factor=1
)



flux_planet5, flux_moon5, flux_total5, px_bary, py_bary, mx_bary, my_bary, time_arrays = pandora(
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
    epoch_distance,
    supersampling_factor=5
)


plt.rc('font',  family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.plot(time_arrays_original, -(1-flux_total_original1*1e6)-1e6, color="black")
plt.plot(time_arrays, -(1-flux_total5*1e6)-1e6, color="red", linestyle="dashed")
plt.xlim(840.96, 842)
#plt.ylim()
plt.xlabel("Time (days)")
plt.ylabel("Flux (ppm)")
#plt.show()
plt.savefig('fig_supersampling.pdf', bbox_inches='tight')
