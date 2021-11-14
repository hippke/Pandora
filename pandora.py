from numba import jit, prange
from numpy import sqrt, pi, arcsin
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from avocado import ellipse_pos, ellipse_pos_iter, ellipse_pos_iter_bary
from mangold import occult, occult_small
from helpers import resample
# from pumpkin import eclipse_ratio


class model_params(object):
    def __init__(self):

        # Star parameters
        self.u1 = None
        self.u2 = None
        self.R_star = None

        # Planet parameters
        self.per_planet = None
        self.a_planet = None
        self.r_planet = None
        self.b_planet = None
        self.t0_planet = None
        self.t0_planet_offset = None        
        self.M_planet = None

        # Moon parameters
        self.r_moon = None
        self.per_moon = None
        self.tau_moon = None
        self.Omega_moon = None
        self.i_moon = None
        self.mass_ratio = None

        # Other model parameters
        self.epochs = None
        self.epoch_duration = None
        self.cadences_per_day = None
        self.epoch_distance = None
        self.supersampling_factor = None
        self.occult_small_threshold = None


class moon_model(object):
    def __init__(self, params):

        
        # Star parameters
        self.u1 = params.u1
        self.u2 = params.u2
        self.R_star = params.R_star

        # Planet parameters
        self.per_planet = params.per_planet
        self.a_planet = params.a_planet
        self.r_planet = params.r_planet
        self.b_planet = params.b_planet
        self.t0_planet = params.t0_planet
        self.t0_planet_offset = params.t0_planet_offset        
        self.M_planet = params.M_planet

        # Moon parameters
        self.r_moon = params.r_moon
        self.per_moon = params.per_moon
        self.tau_moon = params.tau_moon
        self.Omega_moon = params.Omega_moon
        self.i_moon = params.i_moon
        self.mass_ratio = params.mass_ratio

        # Other model parameters
        self.epochs = params.epochs
        self.epoch_duration = params.epoch_duration
        self.cadences_per_day = params.cadences_per_day
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor
        self.occult_small_threshold = params.occult_small_threshold

    def video(self, limb_darkening=True, teff=6000, planet_color="black", moon_color="black"):
        self.flux_planet, self.flux_moon, self.flux_total, self.px_bary, self.py_bary, self.mx_bary, self.my_bary, self.time_arrays = pandora(        
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_planet,
            self.a_planet,
            self.r_planet,
            self.b_planet,
            self.t0_planet,
            self.t0_planet_offset,   
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.mass_ratio,

            # Other model parameters
            self.epochs,
            self.epoch_duration,
            self.cadences_per_day,
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold
        )
        # Build video with matplotlib
        fig = plt.figure(figsize = (5,5))
        axes = fig.add_subplot(111)
        plt.axis('off')
        plt.style.use('dark_background')
        plt.gcf().gca().add_artist(plt.Circle((0, 0), 5, color="black"))
        if limb_darkening:
            s = 100
            if teff > 12000:
                teff = 12000
            if teff < 2300:
                teff = 2300
            star_colors = np.genfromtxt('star_colors.csv', delimiter=',')
            row = np.argmax(star_colors[:,0] >= teff)
            r_star = star_colors[row,1]
            g_star = star_colors[row,2]
            b_star = star_colors[row,3]
            for i in reversed(range(s)):
                impact = (i / s)
                m = sqrt(1 - min(impact**2, 1))
                ld = (1 - self.u1 * (1 - m) - self.u2 * (1 - m) ** 2)
                r = r_star * ld
                g = g_star * ld
                b = b_star * ld
                Sun = plt.Circle((0, 0), impact, color=(r, g, b))
                plt.gcf().gca().add_artist(Sun)
        else:
            plt.gcf().gca().add_artist(plt.Circle((0, 0), 1, color="yellow"))

        axes.set_xlim(-1.05, 1.05)
        axes.set_ylim(-1.05, 1.05)
        moon, = axes.plot(
            self.mx_bary[0],
            self.my_bary[0],
            'o',
            color=planet_color,
            markerfacecolor=moon_color,
            markeredgecolor=moon_color,
            markersize=175 * self.r_moon
        )
        planet, = axes.plot(
            self.px_bary[0],
            self.py_bary[0], 
            'o', 
            color=planet_color,
            markeredgecolor=planet_color,
            markerfacecolor=planet_color,
            markersize=175 * self.r_planet
        )

        def ani(coords):
            moon.set_data(coords[0],coords[1])
            planet.set_data(coords[2],coords[3])
            pbar.update(1)
            return moon, planet

        def frames():
            for mx, my, px, py in zip(self.mx_bary, self.my_bary, self.px_bary, self.py_bary):
                yield mx, my, px, py

        pbar = tqdm(total=len(self.mx_bary))
        ani = FuncAnimation(fig, ani, frames=frames, save_count=1e15, blit=True)
        return ani

    def light_curve(self):
        flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary, time_arrays = pandora(        
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_planet,
            self.a_planet,
            self.r_planet,
            self.b_planet,
            self.t0_planet,
            self.t0_planet_offset,   
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.mass_ratio,

            # Other model parameters
            self.epochs,
            self.epoch_duration,
            self.cadences_per_day,
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold
        )
        return time_arrays, flux_total, flux_planet, flux_moon

    def coordinates(self):
        flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary, time_arrays = pandora(        
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_planet,
            self.a_planet,
            self.r_planet,
            self.b_planet,
            self.t0_planet,
            self.t0_planet_offset,   
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.mass_ratio,

            # Other model parameters
            self.epochs,
            self.epoch_duration,
            self.cadences_per_day,
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold
        )
        return time_arrays, px_bary, py_bary, mx_bary, my_bary


@jit(cache=False, nopython=True, fastmath=True, parallel=False)
def pandora(
    u1,
    u2,
    R_star,

    # Planet parameters
    per_planet,
    a_planet,
    r_planet,
    b_planet,
    t0_planet,
    t0_planet_offset,   
    M_planet,

    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    mass_ratio,

    # Other model parameters
    epochs,
    epoch_duration,
    cadences_per_day,
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
):

    # "Morphological light-curve distortions due to finite integration time"
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract
    # Data gets smeared over long integration. Relevant for e.g., 30min cadences
    # To counter the effect: Set supersampling_factor = 5 (recommended value)
    # Then, 5x denser in time sampling, and averaging after, approximates effect
    if supersampling_factor < 1:
        print("supersampling_factor must be positive integer")
    supersampled_cadences_per_day = cadences_per_day * int(supersampling_factor)

    # epoch_distance is the fixed constant distance between subsequent data epochs
    # Should be identical to the initial guess of the planetary period
    # The planetary period `per_planet`, however, is a free parameter
    t0_planet_transit_times = np.arange(
        start=t0_planet,
        stop=t0_planet + epoch_distance * epochs,
        step=epoch_distance,
    )

    # Planetary transit duration at b=0 equals the width of the star
    # Formally correct would be: (R_star+r_planet) for the transit duration T1-T4
    # Here, however, we need points where center of planet is on stellar limb
    # https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    tdur_p = per_planet / pi * arcsin(1 / a_planet)

    # Calculate moon period around planet
    G = 6.67408e-11
    day = 60 * 60 * 24
    a_moon = (G * (M_planet + mass_ratio * M_planet) / (2 * pi / (per_moon * day)) ** 2) ** (1/3)
    # t0_planet_offset in [days] ==> convert to x scale (i.e. 0.5 transit dur radius)
    t0_shift_planet = t0_planet_offset / (tdur_p / 2)

    # arrays of epoch start and end dates [day]
    t_starts = t0_planet_transit_times - epoch_duration / 2
    t_ends = t0_planet_transit_times + epoch_duration / 2
    cadences = int(supersampled_cadences_per_day * epoch_duration)

    # Loop over epochs and stitch together:
    # np.empty much faster than np.ones
    time = np.empty(shape=(epochs, cadences))
    xp = np.empty(shape=(epochs, cadences))
    
    xpos = epoch_duration / tdur_p
    xpos_array = np.linspace(-xpos, xpos, cadences)

    for epoch in range(epochs):  
        time[epoch] = np.linspace(t_starts[epoch], t_ends[epoch], cadences)

        # Push planet following per_planet, which is a free parameter in [days]
        # For reference: Distance between epoch segments is fixed as segment_distance
        # Have to convert to x scale == 0.5 transit duration for stellar radii
        per_shift_planet = ((per_planet - epoch_distance) * epoch) / (tdur_p / 2)

        xp[epoch] = xpos_array - t0_shift_planet - per_shift_planet

    time = time.ravel()
    xp = xp.ravel()
    
    # Select segment in xp array close enough to star so that ellipse CAN transit
    # If the threshold is too generous, it only costs compute.
    # it was too tight with semimajor + 3rp + tdur
    # If the threshold is too tight, it will make the result totally wrong
    # one semimajor axis + half one for bary wobble + transit dur + 2r planet
    # Maximum: A binary system mass_ratio = 1; from numerical experiments 3*a is OK
    transit_threshold_x = 3 * (a_moon / R_star) + 2 * r_planet + 2 * r_moon
    if transit_threshold_x < 2:
        transit_threshold_x = 2
    #print("transit_threshold_x", transit_threshold_x)

    xm_bary, ym_bary, xp_bary, yp_bary = ellipse_pos_iter_bary(
            a=a_moon / R_star,
            per=per_moon,
            tau=tau_moon,
            Omega=Omega_moon,
            i=i_moon,
            time=time,
            transit_threshold_x=transit_threshold_x,
            xp=xp,
            mass_ratio=mass_ratio,
            b_planet=b_planet
        )

    # Distances of planet and moon from (0,0) = center of star
    z_planet = sqrt(xp_bary ** 2 + yp_bary ** 2)
    z_moon = sqrt(xm_bary ** 2 + ym_bary ** 2)

    # Always use precise Mandel-Agol occultation model for planet 
    flux_planet = occult(zs=z_planet, u1=u1, u2=u2, k=r_planet)

    # For moon transit: User can "set occult_small_threshold > 0"
    if r_moon < occult_small_threshold:
        flux_moon = occult_small(zs=z_moon, k=r_moon, u1=u1, u2=u2)
    else:
        flux_moon = occult(zs=z_moon, k=r_moon, u1=u1, u2=u2)

    # Here: Pumpkin: Mutual planet-moon occultations

    flux_total = 1 - ((1 - flux_planet) + (1 - flux_moon))

    # Supersampling downconversion
    if supersampling_factor > 1:
        flux_planet = resample(flux_planet, supersampling_factor)
        flux_moon = resample(flux_moon, supersampling_factor)
        flux_total = resample(flux_total, supersampling_factor)
        time = resample(time, supersampling_factor)

    return (
        flux_planet,
        flux_moon,
        flux_total,
        xp_bary,
        yp_bary,
        xm_bary,
        ym_bary,
        time,
    )
