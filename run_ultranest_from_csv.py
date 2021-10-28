import numpy as np
from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
from numpy import sin, pi
import matplotlib.pyplot as plt
from ultranest import ReactiveNestedSampler
import ultranest.stepsampler
import ultranest
from ultranest.plot import cornerplot
from pandora import pandora
from numba import jit


#@jit(cache=True, nopython=True, fastmath=True)
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
    params[2]  = cube[2]   # tau_moon (normalized 0..1)
    params[3]  = cube[3]  * 180 # Omega_moon
    params[4]  = cube[4]  * 180 # i_moon
    params[5]  = cube[5]  * 1e27 + 5e24 # M_moon
    params[6]  = cube[6]  * 10 + 360 # per_planet
    params[7]  = cube[7]  * 1.5e8 + 5e7 # a_planet
    params[8]  = cube[8]  * 100000 # r_planet
    params[9] = cube[9] # b_planet
    params[10] = cube[10] - 0.5 # t0_planet_offset
    params[11] = cube[11] * 1e28 + 5e24 # M_planet
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
u1 = 0.5  # Currently not a free parameter
u2 = 0.5  # We assume that we have u1, u1 from spectroscopy and stellar model
u = np.array([[u1, u2]])

t0_planet = 111.1  # [days] first planetary mid-transit time
epochs = 8  # Centered approximately at each planetary transit
epoch_duration = 2.5  # [days] data worth around each transit. Identical for all epochs.
cadences_per_day = 48  # Kepler: 30min LC. Let's keep that fixed for now (untested)
epoch_distance = 365.25  # [days] Constant time distance between each epoch.

time_arrays, testdata = np.loadtxt("output.csv", unpack=True)

stdev = 1e-4
noise = np.random.normal(0, stdev, len(time_arrays))
yerr = np.full(len(testdata), stdev)
"""
plt.scatter(time_arrays, testdata, color="black", s=5)
plt.show()
"""

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
    False, 
    ],
)
nsteps = 2 * len(parameters)

# Versuch 1: Korrektur per_moon mit jetzt incl. Planetenmasse; nsteps=2param
# ==> offset tau 0.30
# Versuch 2: nsteps=100 -- not better
# Versuch 3: nsteps=50;  prange in pandora; speed-test <4x ? ==> 1.4x at 4 cores vs 1
# ==> good result, but still slightly different (35min runtime)
# Versuch 4: Mehr Rauschen (3e-4); 2.5 days duration, 2n params ==> OK, slightly off
# Versuch 5: 5n params ==> OK, slightly off
# Versuch 6 : Zeus MCMC (ensemble slice) ==> prior blöd gemacht
# Versuch 7: dynesty

#sampler2.stepsampler = ultranest.stepsampler.AHARMSampler(nsteps=nsteps)
#sampler2.stepsampler = ultranest.stepsampler.RegionBallSliceSampler(nsteps=nsteps)
#sampler2.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=nsteps)
#sampler2.stepsampler = ultranest.stepsampler.RegionMHSampler(nsteps=nsteps)
# sampler2.stepsampler = ultranest.stepsampler.CubeMHSampler(nsteps=nsteps)  
# ultranest.stepsampler.SliceSampler -- crashes, worked earlier
# ReactiveNestedSampler -- very slow (days)
# sampler2.stepsampler = ultranest.stepsampler.BallSliceSampler(nsteps=nsteps)  

#sampler2.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=50)  

# 10min, slightly wrong
#sampler2.stepsampler = ultranest.stepsampler.CubeSliceSampler(2 * len(parameters))#nsteps=50)  
#sampler2.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=500)#nsteps=50)  
sampler2.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=nsteps)#nsteps=50)  



result2 = sampler2.run()#min_num_live_points=400, update_interval_ncall=1000)
sampler2.print_results()


#cornerplot(result2)
#plt.show()
