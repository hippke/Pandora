import numpy as np
from numpy import sqrt, pi, arcsin, cos, degrees
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from os import path

# Pandora
from pandoramoon.eclipse import eclipse
from pandoramoon.ellipse import ellipse, ellipse_ecc
from pandoramoon.occult import occult, occult_small, occult_hybrid, create_occult_cache, read_occult_cache
from pandoramoon.helpers import resample
from pandoramoon.grids import timegrid, x_bary_grid


class model_params(object):
    def __init__(self):

        # Star parameters
        self.u1 = None
        self.u2 = None
        self.R_star = None

        # Planet parameters
        self.per_bary = None
        self.a_bary = None
        self.r_planet = None
        self.b_bary = None
        self.w_bary = 0
        self.ecc_bary = 0
        self.t0_bary = None
        self.t0_bary_offset = None
        self.M_planet = None

        # Moon parameters
        self.r_moon = None
        self.per_moon = None
        self.tau_moon = None
        self.Omega_moon = None
        self.i_moon = None
        self.ecc_moon = 0
        self.w_moon = 0
        self.M_moon = None

        # Other model parameters
        self.epochs = None
        self.epoch_duration = None
        self.cadences_per_day = None
        self.epoch_distance = None
        self.supersampling_factor = 1
        self.occult_small_threshold = 0.01
        self.hill_sphere_threshold = 1.1
        self.numerical_grid = 25
        self.time = None


class time(object):
    def __init__(self, params):
        self.t0_bary = params.t0_bary
        self.epochs = params.epochs
        self.epoch_duration = params.epoch_duration
        self.cadences_per_day = params.cadences_per_day
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor

    def grid(self):
        return timegrid(
            self.t0_bary, 
            self.epochs, 
            self.epoch_duration, 
            self.cadences_per_day, 
            self.epoch_distance, 
            self.supersampling_factor
        )


class moon_model(object):
    def __init__(self, params):

        # Star parameters
        self.u1 = params.u1
        self.u2 = params.u2
        self.R_star = params.R_star

        # Planet parameters
        self.per_bary = params.per_bary
        self.a_bary = params.a_bary
        self.r_planet = params.r_planet
        self.b_bary = params.b_bary
        self.w_bary = params.w_bary
        self.ecc_bary = params.ecc_bary
        self.t0_bary = params.t0_bary
        self.t0_bary_offset = params.t0_bary_offset
        self.M_planet = params.M_planet

        # Moon parameters
        self.r_moon = params.r_moon
        self.per_moon = params.per_moon
        self.tau_moon = params.tau_moon
        self.Omega_moon = params.Omega_moon
        self.i_moon = params.i_moon
        self.ecc_moon = params.ecc_moon
        self.w_moon = params.w_moon
        self.M_moon = params.M_moon

        # Other model parameters
        self.epoch_distance = params.epoch_distance
        self.supersampling_factor = params.supersampling_factor
        self.occult_small_threshold = params.occult_small_threshold
        self.hill_sphere_threshold = params.hill_sphere_threshold
        self.numerical_grid = params.numerical_grid
        self.time = params.time
        self.cache = create_occult_cache(self.u1, self.u2, dim=300)

    def video(
        self,
        time,
        limb_darkening=True,
        teff=6000,
        planet_color="black",
        moon_color="black",
        ld_circles=100,
    ):
        (
            self.flux_planet,
            self.flux_moon,
            self.flux_total,
            self.px,
            self.py,
            self.mx,
            self.my
        ) = pandora(
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_bary,
            self.a_bary,
            self.r_planet,
            self.b_bary,
            self.w_bary,
            self.ecc_bary,
            self.t0_bary,
            self.t0_bary_offset,
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.ecc_moon,
            self.w_moon,
            self.M_moon,

            # Other model parameters
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold,
            self.hill_sphere_threshold,
            self.numerical_grid,
            time
        )
        # Build video with matplotlib
        fig = plt.figure(figsize=(5, 5))
        axes = fig.add_subplot(111)
        plt.axis("off")
        plt.style.use("dark_background")
        plt.gcf().gca().add_artist(plt.Circle((0, 0), 5, color="black"))
        if limb_darkening:
            if teff > 12000:
                teff = 12000
            if teff < 2300:
                teff = 2300
            data_dir = path.join(path.join(path.dirname(__file__)), "star_colors.csv")
            star_colors = np.genfromtxt(data_dir, delimiter=",")
            row = np.argmax(star_colors[:, 0] >= teff)
            r_star = star_colors[row, 1]
            g_star = star_colors[row, 2]
            b_star = star_colors[row, 3]
            for i in reversed(range(ld_circles)):
                impact = i / ld_circles
                m = sqrt(1 - min(impact ** 2, 1))
                ld = 1 - self.u1 * (1 - m) - self.u2 * (1 - m) ** 2
                r = r_star * ld
                g = g_star * ld
                b = b_star * ld
                sun = plt.Circle((0, 0), impact, color=(r, g, b))
                plt.gcf().gca().add_artist(sun)
        else:
            plt.gcf().gca().add_artist(plt.Circle((0, 0), 1, color="yellow"))

        axes.set_xlim(-1.05, 1.05)
        axes.set_ylim(-1.05, 1.05)
        (moon,) = axes.plot(
            self.mx[0],
            self.my[0],
            "o",
            color=planet_color,
            markerfacecolor=moon_color,
            markeredgecolor=moon_color,
            markersize=260 * self.r_moon,
        )
        (planet,) = axes.plot(
            self.px[0],
            self.py[0],
            "o",
            color=planet_color,
            markeredgecolor=planet_color,
            markerfacecolor=planet_color,
            markersize=260 * self.r_planet,
        )

        def ani(coords):
            moon.set_data(coords[0], coords[1])
            planet.set_data(coords[2], coords[3])
            pbar.update(1)
            return moon, planet

        def frames():
            for mx, my, px, py in zip(self.mx, self.my, self.px, self.py):
                yield mx, my, px, py

        pbar = tqdm(total=len(self.mx))
        ani = FuncAnimation(fig, ani, frames=frames, save_count=1e15, blit=True)
        return ani


    def light_curve(self, time):
        flux_planet, flux_moon, flux_total, px, py, mx, my = pandora(
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_bary,
            self.a_bary,
            self.r_planet,
            self.b_bary,
            self.w_bary,
            self.ecc_bary,
            self.t0_bary,
            self.t0_bary_offset,
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.ecc_moon,
            self.w_moon,
            self.M_moon,

            # Other model parameters
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold,
            self.hill_sphere_threshold,
            self.numerical_grid,
            time,
            self.cache
        )
        return flux_total, flux_planet, flux_moon

    def coordinates(self, time):
        flux_planet, flux_moon, flux_total, px, py, mx, my = pandora(
            self.u1,
            self.u2,
            self.R_star,

            # Planet parameters
            self.per_bary,
            self.a_bary,
            self.r_planet,
            self.b_bary,
            self.w_bary,
            self.ecc_bary,
            self.t0_bary,
            self.t0_bary_offset,
            self.M_planet,

            # Moon parameters
            self.r_moon,
            self.per_moon,
            self.tau_moon,
            self.Omega_moon,
            self.i_moon,
            self.ecc_moon,
            self.w_moon,
            self.M_moon,

            # Other model parameters
            self.epoch_distance,
            self.supersampling_factor,
            self.occult_small_threshold,
            self.hill_sphere_threshold,
            self.numerical_grid,
            time
        )
        return px, py, mx, my


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pandora(
    u1,
    u2,
    R_star,
    # Planet parameters
    per_bary,
    a_bary,
    r_planet,
    b_bary,
    w_bary,
    ecc_bary,
    t0_bary,
    t0_bary_offset,
    M_planet,
    # Moon parameters
    r_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    i_moon,
    ecc_moon,
    w_moon,
    M_moon,
    # Other model parameters
    epoch_distance,
    supersampling_factor,
    occult_small_threshold,
    hill_sphere_threshold,
    numerical_grid,
    time,
    cache=None
):

    # Make sure to work with floats. Large values as ints would overflow.
    R_star = float(R_star)
    per_bary = float(per_bary)
    a_bary = float(a_bary)
    r_planet = float(r_planet)
    b_bary = float(b_bary)
    t0_bary = float(t0_bary)
    t0_bary_offset = float(t0_bary_offset)
    M_planet = float(M_planet)
    r_moon = float(r_moon)
    per_moon = float(per_moon)
    tau_moon = float(tau_moon)
    Omega_moon = float(Omega_moon)
    i_moon = float(i_moon)
    M_moon = float(M_moon)

    # Calculate moon period around planet
    G = 6.67408e-11
    day = 60 * 60 * 24
    a_moon = (
        G * (M_planet + M_moon) / (2 * pi / (per_moon * day)) ** 2
    ) ** (1 / 3)
    a_moon /= R_star

    x_bary = x_bary_grid(
        time, 
        a_bary, 
        per_bary, 
        t0_bary, 
        t0_bary_offset, 
        epoch_distance, 
        ecc_bary, 
        w_bary
    )

    # Check physical plausibility of a_moon
    # Should be inside [Roche lobe, Hill sphere] plus/minus some user-set margin
    M_star = ((4 * pi ** 2 / G) * ((a_bary * R_star) ** 3)) / (per_bary * day) ** 2
    r_hill = a_bary * (M_planet / (3 * M_star)) ** (1 / 3)
    r_hill_fraction = a_moon / r_hill
    if r_hill_fraction > hill_sphere_threshold:
        unphysical = True
    else:
        unphysical = False

    # Roche lobe: Not used here, because we don't know densities of planet and moon
    # Instead of taking density guesses, we just demand a_moon > (R_planet + R_moon)
    if a_moon < (r_planet + r_moon):
        unphysical = True

    # Unphysical moon orbit: Keep planet, but put moon at far out of transit position
    if unphysical:
        bignum = 1e8
        xp = x_bary
        yp = np.full(len(x_bary), b_bary)
        xm = np.full(len(x_bary), bignum)
        ym = xm.copy()
    else:  # valid, physical system
        if ecc_moon == 0:
            xm, ym, xp, yp = ellipse(
                a=a_moon,
                per=per_moon,
                tau=tau_moon,
                Omega=Omega_moon,
                i=i_moon,
                time=time,
                x_bary=x_bary,
                mass_ratio=M_moon / M_planet,
                b_bary=b_bary,
            )
        else:
            xm, ym, xp, yp = ellipse_ecc(
                a=a_moon,
                per=per_moon,
                e=ecc_moon,
                tau=tau_moon,
                Omega=Omega_moon,
                w=w_moon,
                i=i_moon,
                time=time,
                mass_ratio=M_moon / M_planet,
                x_bary=x_bary,
                b_bary=b_bary,
            )

    # Distances of planet and moon from (0,0) = center of star
    # Not sufficient to only calculate z in ellipse func: 
    # We also need full coordinates to determine mutual eclipses below
    z_planet = sqrt(xp ** 2 + yp ** 2)
    if unphysical:
        z_moon = xm.copy()
    else:
        z_moon = sqrt(xm ** 2 + ym ** 2)

    # Cached Mandel-Agol occultation model for planet < 0.1, else hybrid
    if cache is not None and r_planet < 0.1:
        flux_planet = read_occult_cache(
            zs_target=z_planet, 
            k=r_planet, 
            cache=cache
        )
    else:
        flux_planet = occult_hybrid(zs=z_planet, u1=u1, u2=u2, k=r_planet)
        #flux_planet = occult(zs=z_planet, u1=u1, u2=u2, k=r_planet)
    
    # For moon transit: User can "set occult_small_threshold > 0"
    if r_moon < occult_small_threshold:
        flux_moon = occult_small(zs=z_moon, k=r_moon, u1=u1, u2=u2)
    elif cache is not None and r_moon < 0.1:
        flux_moon = read_occult_cache(zs_target=z_moon, k=r_moon, cache=cache)
    else:
        flux_moon = occult_hybrid(zs=z_moon, k=r_moon, u1=u1, u2=u2)

    # Mutual planet-moon occultations
    if not unphysical:# and eclipses_occur:
        flux_moon = eclipse(xp, yp, xm, ym, r_planet, r_moon, flux_moon, numerical_grid)
    flux_total = flux_moon + flux_planet - 1

    # Supersampling downconversion
    if supersampling_factor > 1:
        flux_planet = resample(flux_planet, supersampling_factor)
        flux_moon = resample(flux_moon, supersampling_factor)
        flux_total = resample(flux_total, supersampling_factor)

    return flux_planet, flux_moon, flux_total, xp, yp, xm, ym
