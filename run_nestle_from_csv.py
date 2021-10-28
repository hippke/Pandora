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
import matplotlib as mpl
import corner

# plot the resulting posteriors
mpl.rcParams.update({'font.size': 16})

def plotposts(samples, **kwargs):
    """
    Function to plot posteriors using corner.py and scipy's gaussian KDE function.
    """
    if "truths" not in kwargs:
        kwargs["truths"] = [m, c]

    fig = corner.corner(samples, labels=[r'$m$', r'$c$'], hist_kwargs={'density': True}, **kwargs)

    # plot KDE smoothed version of distributions
    for axidx, samps in zip([0, 3], samples.T):
        kde = gaussian_kde(samps)
        xvals = fig.axes[axidx].get_xlim()
        xvals = np.linspace(xvals[0], xvals[1], 100)
        fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')


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
    params[2]  = cube[2]   # tau_moon (normalized 0..1)
    params[3]  = cube[3]  * 180 # Omega_moon
    params[4]  = cube[4]  * 180 # i_moon
    params[5]  = cube[5]  * 1e27 + 5e24 # M_moon
    params[6]  = cube[6]  * 10 + 360 # per_planet
    params[7]  = cube[7]  * 1.5e8 + 5e7 # a_planet
    params[8]  = cube[8]  * 100000 # r_planet
    params[9] = cube[9] # b_planet
    params[10] = cube[10] - 0.5 # t0_planet_offset
    params[11] = cube[11] * 1e27 + 5e24 # M_planet
    return params


#@jit(cache=True, nopython=True, fastmath=True)
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
u1 = 0.52  # Currently not a free parameter
u2 = 0.48  # We assume that we have u1, u1 from spectroscopy and stellar model
u = np.array([[u1, u2]])

t0_planet = 111.1  # [days] first planetary mid-transit time
epochs = 10  # Centered approximately at each planetary transit
epoch_duration = 3  # [days] data worth around each transit. Identical for all epochs.
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

"""
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
sampler2.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=nsteps)
#sampler2.stepsampler = ultranest.stepsampler.CubeSliceSampler(nsteps=nsteps)
result2 = sampler2.run()#min_num_live_points=400, update_interval_ncall=1000)
sampler2.print_results()


cornerplot(result2)
plt.show()"""


import nestle

print('Nestle version: {}'.format(nestle.__version__))

nlive = 1024     # number of live points
method = 'multi' # use MutliNest algorithm
ndims = 12        # two parameters
tol = 0.1        # the stopping criterion


res = nestle.sample(
    log_likelihood,
    prior_transform,
    ndims,
    method=method,
    npoints=nlive,
    dlogz=tol,
    callback=nestle.print_progress
)
logZnestle = res.logz                         # value of logZ
infogainnestle = res.h                        # value of the information gain in nats
logZerrnestle = np.sqrt(infogainnestle/nlive) # estimate of the statistcal uncertainty on logZ

print("log(Z) = {} ± {}".format(logZnestle, logZerrnestle))

print(res.summary())
# re-scale weights to have a maximum of one
nweights = res.weights/np.max(res.weights)

# get the probability of keeping a sample from the weights
keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

# get the posterior samples
samples_nestle = res.samples[keepidx,:]

resdict['mnestle_mu'] = np.mean(samples_nestle[:,0])      # mean of m samples
resdict['mnestle_sig'] = np.std(samples_nestle[:,0])      # standard deviation of m samples
resdict['cnestle_mu'] = np.mean(samples_nestle[:,1])      # mean of c samples
resdict['cnestle_sig'] = np.std(samples_nestle[:,1])      # standard deviation of c samples
resdict['ccnestle'] = np.corrcoef(samples_nestle.T)[0,1]  # correlation coefficient between parameters
resdict['nestle_npos'] = len(samples_nestle)              # number of posterior samples
resdict['nestle_time'] = timenestle                       # run time
resdict['nestle_logZ'] = logZnestle                       # log marginalised likelihood
resdict['nestle_logZerr'] = logZerrnestle                 # uncertainty on log(Z)

print('Number of posterior samples is {}'.format(len(samples_nestle)))

plotposts(samples_nestle)
