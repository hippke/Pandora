from numba import jit, prange
from numpy import sqrt, pi, arcsin
import numpy as np
import numba as nb

from avocado import ellipse_pos, bary_pos
from mangold import occult_single, occult_array
#from pumpkin import eclipse_ratio


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def pandora_model(
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
    time
):
	# Calculate moon period around planet. 
	# Keep "1000.0"**3 as float: numba can't handle long ints
	G = 6.67408 * 10 ** -11
	per_moon = (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * M_planet))) / 60 / 60 / 24
	mass_ratio = M_moon / M_planet
	t_start = np.min(time)
	t_end = np.max(time)
	t_dur = t_end - t_start

	cadences = len(time)
	
	# Planetary transit duration at b=0 equals the width of the star
	tdur_p = per_planet / pi * arcsin(sqrt((r_planet/2 + R_star) ** 2) / a_planet)

	# Get Kepler Ellipse
	xm, ym = ellipse_pos(
	    a_moon / R_star, per_moon, tau_moon, Omega_moon, w_moon, i_moon, time
	)


	# (x,y) grid in units of stellar radii; star at (0,0)
	# Get x pos for start of time series
	# Check: Not fully matching Batman model -- too wide
	xpos_array = np.linspace(-t_dur / tdur_p, t_dur / tdur_p, cadences)

	# Adjust position of planet and moon due to mass: barycentric wobble
	xm_bary, ym_bary, xp_bary, yp_bary = bary_pos(
		xm, ym, xpos_array + t0_planet, b_planet, mass_ratio)

	# Distances of planet and moon from (0,0) = center of star
	z_planet = sqrt(xp_bary ** 2 + yp_bary ** 2)
	z_moon = sqrt(xm_bary ** 2 + ym_bary ** 2)
	flux_planet = occult_array(zs=z_planet, u=u, k=r_planet / R_star)
	flux_moon = occult_array(zs=z_moon, u=u, k=r_moon / R_star)
	
	# Here: Mutual planet-moon occultations
	flux_total = 1 - ((1 - flux_planet) + (1 - flux_moon))
	return flux_planet, flux_moon, flux_total, xp_bary, yp_bary, xm_bary, ym_bary


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def pandora_vectorized(    
	r_moon_array,
    a_moon_array,
    tau_moon_array,
    Omega_moon_array,
    w_moon_array,
    i_moon_array,
    M_moon_array,
    per_planet_array,
    a_planet_array,
    r_planet_array,
    b_planet_array,
    t0_planet_array,
    M_planet_array,
    R_star,
    u,
    time):

	# array of arrays [[u1, u2]]
	flux_total_array = np.ones(shape=(len(r_moon_array),len(time)))
	
	for idx in range(len(r_moon_array)):
		#print("Omega_moon_array", Omega_moon_array)
		flux_planet, flux_moon, fnow, xp_bary, yp_bary, xm_bary, ym_bary = pandora_model(
			r_moon_array[idx],
		    a_moon_array[idx],
		    tau_moon_array[idx],
		    Omega_moon_array[idx],
		    w_moon_array[idx],
		    i_moon_array[idx],
		    M_moon_array[idx],
		    per_planet_array[idx],
		    a_planet_array[idx],
		    r_planet_array[idx],
		    b_planet_array[idx],
		    t0_planet_array[idx],
		    M_planet_array[idx],
		    R_star,
		    u,
		    time
			)
		flux_total_array[idx,:] = fnow
	return flux_total_array
