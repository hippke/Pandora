import pandoramoon as pandora
#from pandoramoon import pandora
#import pandoramoon
#import pandoramoon
import numpy as np
import matplotlib.pyplot as plt

# Call Pandora and get model with these parameters
params = pandora.model_params()
R_sun = 696342000
params.R_star = 1 * R_sun  # [m]
params.u1 = 0.4089  # [0..1] as per Claret & Bloemen 2011, logg=4.50; Teff=5750; log[M/H]=0.0; micro_turb_vel=0.0
params.u2 = 0.2556  # [0..1] as per Claret & Bloemen 2011, logg=4.50; Teff=5750; log[M/H]=0.0; micro_turb_vel=0.0

# Planet parameters
params.per_bary = 365.25  # [days]
params.a_bary = 215  # [R_star]
params.r_planet = 0.1 # [R_star]
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
params.epochs = 3  # [int]
params.epoch_duration = 0.6  # 5  # [days]
params.cadences_per_day = 250  # [int]
params.epoch_distance = 365.26   # [days] value close to per_planet, but not identical
params.supersampling_factor = 1  # [int]
params.occult_small_threshold = 0.1  # [0..1]
params.hill_sphere_threshold = 1.2

time = pandora.time(params).grid()
model = pandora.moon_model(params)

flux_total, flux_planet, flux_moon = model.light_curve(time)
xp, yp, xm, ym = model.coordinates(time)
print(np.sum(flux_total))
assert np.abs(np.sum(flux_total) - 445.96052326469663) < 1e-4
print(np.sum(time))
assert np.abs(np.sum(time)-  169317.0) < 1e-10
print(np.sum(xp* yp- xm+ ym))
assert np.abs((np.sum(xp* yp- xm+ ym)) - 122.36217451693565) < 1e-4

# Create noise and merge with flux
noise_level = 100e-6  # Gaussian noise to be added to the generated data
noise = np.random.normal(0, noise_level, len(time))
testdata = noise + flux_total
yerr = np.full(len(testdata), noise_level)
#print("std", np.std(noise))
#np.savetxt("output_v4.csv", np.transpose(np.array((time, testdata))), fmt='%8f')


# Plot synthetic data
plt.plot(time, flux_planet, color="blue")
plt.plot(time, flux_moon, color="red")
plt.plot(time, flux_total, color="black")
plt.scatter(time, testdata, color="black", s=0.5)
plt.xlabel("Time (days)")
plt.ylabel("Relative flux")
plt.show()




video = model.video(
    time=time,
    limb_darkening=True, 
    teff=3200,
    planet_color="black",
    moon_color="black",
    ld_circles=100
)
video.save(filename="video.mp4", fps=25, dpi=200)


