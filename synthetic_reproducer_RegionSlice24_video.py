import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
from pandora import pandora


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
flux_planet_original, flux_moon_original, flux_total_original, px_bary, py_bary, mx_bary, my_bary, time_arrays_original = pandora(
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
noise = np.random.normal(0, noise_level, len(time_arrays_original))
testdata = noise + flux_total_original
yerr = np.full(len(testdata), noise_level)

# Plot synthetic data
#plt.plot(time_arrays, flux_planet, color="blue")
#plt.plot(time_arrays, flux_moon, color="red")


### Retrieved model:
u1 = 0.5  # Currently not a free parameter, please leave as is!
u2 = 0.5  # We assume that we have u1, u1 from spectroscopy and stellar model
u = np.array([[u1, u2]])


# Here the fun begins. Choose these parameters:
# Planet parameters
r_planet = 64612  # radius [km] My prior: 1..100,000 km
a_planet = 143506581  # semimajor axis [km] My prior: 5e7..2e8 km
b_planet = 0.4750  # impact parameter [0..1.x] central transit is 0. My prior: 0..1
per_planet = 365.249945  # period [days] My prior: 360..370 days
M_planet = 3.67925271721349E+027  # mass [kg] My prior: 5e24..5e27
t0_planet_offset = 0.10013  # true offset from assumed t0; constant for all periods [days]. My prior: -0.5..+0.5

# Set moon parameters
r_moon = 17904  # radius [km] My prior: 1..50,000 km
a_moon = 1883167  # semimajor axis [km] My prior: 10,000...3,000,000 km
Omega_moon = 179.958  # degrees [0..90] My prior: 0..90 deg
i_moon = 87.80   # degrees [0..90]. 90 is edge-on. My prior: 0..90 deg
tau_moon = 0.14595  # moon orbital position [0..0.5] My prior: 0..0.5
M_moon = 3.50789809356188E+025  # mass [kg] My prior: 5e24..5e27


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


noise_level = 2e-4  # Gaussian noise to be added to the generated data
# Let's not add TOOOO much noise for now. We can test "fishing the noise" limits later

# Call Pandora and get model with these parameters
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
"""
plt.plot(time_arrays, flux_planet, color="blue")
plt.plot(time_arrays, flux_moon, color="red")
plt.plot(time_arrays, flux_total, color="black")

plt.show()
"""
t0_planet_transit_times = np.arange(
        start=t0_planet,
        stop=t0_planet + epoch_distance * epochs,
        step=epoch_distance,
    )

x = 4
y = 2
counter = 0

plt.rc('font',  family='serif', serif='Times')
plt.rc('text', usetex=True)
fig, axs = plt.subplots(x, y, figsize=(10, 16))
for a in range(x):
    for b in range(y):
        print(a,b)
        segment_start = t0_planet_transit_times[counter] - epoch_duration / 2
        segment_end = t0_planet_transit_times[counter] + epoch_duration / 2
        axs[a,b].set_xlim(segment_start, segment_end)

        # Injected data
        axs[a,b].scatter(
            time_arrays_original, 
            testdata, 
            color="gray", 
            s=10, 
            #alpha=0.5, 
            facecolors='gray', 
            edgecolors='none'
        )
        axs[a,b].plot(time_arrays_original, flux_total_original, color="gray", alpha=0.5)
        axs[a,b].plot(time_arrays_original, flux_planet_original, color="blue", alpha=0.5)
        axs[a,b].plot(time_arrays_original, flux_moon_original, color="red", alpha=0.5)

        # Retrieved Model
        axs[a,b].plot(time_arrays, flux_total, color="black", linestyle="dashed")
        
        axs[a,b].set_xlabel("Time (days)")
        axs[a,b].set_ylabel("Flux")
        counter += 1
        axs[a,b].set_ylim(0.988, 1.001)
#plt.show()
plt.savefig('fig_retrieval_normal.pdf', bbox_inches='tight')


#flux_total_original  # injected model
#testdata  # injected model + noise
#flux_total  # retrieved model



loglike1 = -0.5 * (((flux_total_original - testdata) / yerr)**2).sum()
print("loglike original", loglike1)
plt.close()
plt.scatter(time_arrays, flux_total_original - testdata)
#plt.show()

loglike2 = -0.5 * (((flux_total - testdata) / yerr)**2).sum()
print("loglike retrieved", loglike2)


import matplotlib.pyplot as plt
digits_filename = 4
R_star = 696342  # km
for idx in range(len(px_bary)):
    print(idx)
    figure, axes = plt.subplots()
    plt.gcf().gca().add_artist(plt.Circle((0, 0), R_star/R_star, color="yellow"))
    plt.gcf().gca().add_artist(plt.Circle((px_bary[idx], py_bary[idx]), r_planet/R_star, color="blue", fill=True))
    plt.gcf().gca().add_artist(plt.Circle((mx_bary[idx], my_bary[idx]), r_moon/R_star, color="red", fill=True))
    #plt.text(0,1.2,"{:10.4f}".format(er), horizontalalignment="center")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    axes.set_aspect(1)
    filename = str(idx).zfill(digits_filename) + ".png"
    plt.savefig(filename, bbox_inches='tight')
    #plt.show()
    plt.close()

# convert *.png movie_bary.gif