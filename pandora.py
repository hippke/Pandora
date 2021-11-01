from numba import jit, prange
from numpy import sqrt, pi, arcsin
import numpy as np

from avocado import ellipse_pos, ellipse_pos_iter
from mangold import occult, occult_small
from helpers import resample
# from pumpkin import eclipse_ratio


class model_params(object):
    def __init__(self):

        # Star parameters
        self.R_star = None
        self.u1 = None
        self.u2 = None

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
        self.a_moon = None
        self.tau_moon = None
        self.Omega_moon = None
        self.i_moon = None
        self.M_moon = None

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
        self.R_star = params.R_star
        self.u1 = params.u1
        self.u2 = params.u2

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
        self.a_moon = params.a_moon
        self.tau_moon = params.tau_moon
        self.Omega_moon = params.Omega_moon
        self.i_moon = params.i_moon
        self.M_moon = params.M_moon

        # Other model parameters
        self.epochs = params.epochs
        self.epoch_duration = params.epoch_duration
        self.cadences_per_day = params.cadences_per_day
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor
        self.occult_small_threshold = params.occult_small_threshold

    def evaluate(self):
        self.flux_planet, self.flux_moon, self.flux_total, self.px_bary, self.py_bary, self.x_bary, self.my_bary, self.time_arrays = pandora(        
            self.R_star,
            self.u1,
            self.u2,

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
            self.a_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.M_moon,

            # Other model parameters
            self.epochs,
            self.epoch_duration,
            self.cadences_per_day,
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold
        )
        return time_arrays, flux_total, flux_planet, flux_moon

    def light_curve(self):
        flux_planet, flux_moon, flux_total, px_bary, py_bary, mx_bary, my_bary, time_arrays = pandora(        
            self.R_star,
            self.u1,
            self.u2,

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
            self.a_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.M_moon,

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
            self.R_star,
            self.u1,
            self.u2,

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
            self.a_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.M_moon,

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
    R_star,
    u1,
    u2,

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
    a_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    M_moon,

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
        # return False
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
    # Here, however, we need T2-T3 (points where center of planet is on stellar limb)
    # https://www.paulanthonywilson.com/exoplanets/exoplanet-detection-techniques/the-exoplanet-transit-method/
    tdur_p = per_planet / pi * arcsin(sqrt(R_star ** 2) / a_planet)

    # Calculate moon period around planet.
    # Keep "1000.0"**3 as float: numba can't handle long ints
    G = 6.67408e-11
    per_moon = (
        (2 * pi * sqrt((a_moon * 1000.0) ** 3 / (G * (M_planet + M_moon))))
        / 60
        / 60
        / 24
    )

    # t0_planet_offset in [days] ==> convert to x scale (i.e. 0.5 transit dur radius)
    t0_shift_planet = t0_planet_offset / (tdur_p / 2)

    # arrays of epoch start and end dates [day]
    t_starts = t0_planet_transit_times - epoch_duration / 2
    t_ends = t0_planet_transit_times + epoch_duration / 2
    cadences = int(supersampled_cadences_per_day * epoch_duration)

    # Loop over epochs and stitch together:
    time = np.ones(shape=(epochs, cadences))
    xp = time.copy()
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
    # 3x r_planet: 2*r_planet because bary wobble up to one planet diameter
    # Should this rather be: (tdur_p/2) ?
    transit_threshold_x = a_moon / R_star + (3 * r_planet / R_star) + tdur_p

    """
    xm, ym = ellipse_pos(
            a=a_moon / R_star,
            per=per_moon,
            tau=tau_moon,
            Omega=Omega_moon,
            i=i_moon,
            time=time,
        )
    """

    # "ellipse_pos_iter" is ~10% slower than array version "ellipse_pos"
    # But: If >10% are out of transit (which is almost always the case), 
    #      then linear speed increase (typically 2x)
    xm, ym = ellipse_pos_iter(
            a=a_moon / R_star,
            per=per_moon,
            tau=tau_moon,
            Omega=Omega_moon,
            i=i_moon,
            time=time,
            transit_threshold_x=transit_threshold_x,
            xp=xp
        )
    
    
    # Add barycentric correction to planet and moon as function of mass ratio
    # Negligible runtime (<1%): No benefit from skipping for out of transit parts
    mass_ratio = M_moon / M_planet
    if mass_ratio > 1:
        mass_ratio = 1
    xm_bary = xm + xp + xm * mass_ratio
    ym_bary = ym + b_planet + ym * mass_ratio
    xp_bary = xp - xm * mass_ratio
    yp_bary = b_planet - ym * mass_ratio

    # Distances of planet and moon from (0,0) = center of star
    z_planet = sqrt(xp_bary ** 2 + yp_bary ** 2)
    z_moon = sqrt(xm_bary ** 2 + ym_bary ** 2)

    # Always use precise Mandel-Agol occultationmodel for planet 
    flux_planet = occult(zs=z_planet, u1=u1, u2=u2, k=r_planet / R_star)

    # For moon transit: User can "set occult_small_threshold > 0"
    if (r_moon / R_star) < occult_small_threshold:
        flux_moon = occult_small(zs=z_moon, k=r_moon / R_star, u1=u1, u2=u2)
    else:
        flux_moon = occult(zs=z_moon, k=r_moon / R_star, u1=u1, u2=u2)

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
