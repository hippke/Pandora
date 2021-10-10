from numpy import pi, sin, cos, tan, arctan, sqrt, arcsin
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from matplotlib import rc
from xypos import xypos_moon
from occult_precise import occult_precise
from circle_circle_intersect import eclipse_ratio
from occult_precise_pytransit import eval_quad_z_s

G = 6.67408 * 10 ** -11

# Set stellar parameters
R_star = 1 * 696342  # km
u1 = 0.5971
u2 = 0.1172

# Set planet parameters
r_planet = 63710  # km
a_planet = 1 * 149597870.700  # [km]
b_planet = 0.25  # [0..1.x]; central transit is 0.
per_planet = 365.25  # [days]
M_planet = 5.972 * 10 ** 24
transit_duration_planet = per_planet / pi * arcsin(sqrt((r_planet + R_star) ** 2) / a_planet)

# Set moon parameters
r_moon = 18000  # [km]
a_moon = 384000  # [km] <<<<------
e_moon = 0.0  # 0..1
Omega_moon = 20  # degrees
w_moon = 50.0  # degrees
i_moon = 60.0  # 0..90 in degrees. 0 is the reference plain (no incl).
tau_moon = 0
mass_ratio = 0.1
per_moon = (2 * pi * sqrt((a_moon * 1000) ** 3 / (G * M_planet))) / 60 / 60 / 24

print("moon period (days)", per_moon)
# OK to keep planetary transit duration at b=0 as the star is always R=1

print("transit_duration (days, b=0)", transit_duration_planet)


def flux(
    r_moon,
    a_moon,
    per_moon,
    tau_moon,
    Omega_moon,
    w_moon,
    i_moon,
    r_planet,
    x_planet,
    y_planet,
    mass_ratio,
    R_star,
    u1,
    u2,
    time,
):
    x_moon, y_moon = xypos_moon(
        a_moon / R_star, per_moon, tau_moon, Omega_moon, w_moon, i_moon, time
    )
    moon_x_bary = x_moon + x_planet + x_moon * mass_ratio
    moon_y_bary = y_moon + y_planet + y_moon * mass_ratio
    planet_x_bary = x_planet - x_moon * mass_ratio
    planet_y_bary = y_planet - y_moon * mass_ratio
    #print("planet_y_bary", planet_y_bary)
    radial_distance_moon = sqrt(moon_x_bary ** 2 + moon_y_bary ** 2)
    radial_distance_planet = sqrt(planet_x_bary ** 2 + planet_y_bary ** 2)
    flux_planet = occult_precise(
        z=radial_distance_planet, u1=u1, u2=u2, p0=r_planet / R_star
    )
    """ Version from PyTransit based on numba
    flux_planet = eval_quad_z_s(
        z = radial_distance_planet,
        k = r_planet / R_star,
        u = (u1, u2)
        )
    print(flux_planet, f123)
    """
    flux_moon = occult_precise(z=radial_distance_moon, u1=u1, u2=u2, p0=r_moon / R_star)

    # Planet-Moon occultation
    # Case 1: None
    # Case 2: Occultation, both bodies on star or off star --> 2 circle intersect
    # Case 3: Occultation, any body on limb --> 3 circle intersect

    planet_moon_occultation = False
    on_limb = False

    distance_planet_moon = (
        sqrt((planet_x_bary - moon_x_bary) ** 2 + (planet_y_bary - moon_y_bary) ** 2)
        * R_star
    )

    if (abs(1-(radial_distance_moon)) < (r_moon / R_star)):
        on_limb = True
        #print("Moon on limb", x_planet, abs(1-(radial_distance_moon)), r_moon / R_star)
    if (abs(1-(radial_distance_planet)) < (r_planet / R_star)):
        on_limb = True
        #print("Planet on limb", x_planet, abs(1-(radial_distance_planet)), r_planet / R_star)

    if abs(distance_planet_moon) < (r_planet + r_moon):
        planet_moon_occultation = True
        print("Planet-moon eclipse", er)

    # Case 2: Occultation, both bodies on star or off star --> 2 circle intersect
    if planet_moon_occultation and not on_limb:
        er = eclipse_ratio(distance_planet_moon, r_planet, r_moon)
        if er > 0:
            print("2 body eclipse", er)
            flux_moon = -(1 - flux_moon) * 10 ** 6
            flux_moon = flux_moon * (1 - er)
            flux_moon = 1 - (-flux_moon * 10 ** -6)
        flux_total = 1 - ((1 - flux_planet[0]) + (1 - flux_moon[0]))

    # Case 3: Occultation, any body on limb --> 3 circle intersect
    # HERE: ADD FEWELL WHEN READY
    if planet_moon_occultation and on_limb:
        er = eclipse_ratio(distance_planet_moon, r_planet, r_moon)
        if er > 0:
            print("2 body eclipse", er)
            flux_moon = -(1 - flux_moon) * 10 ** 6
            flux_moon = flux_moon * (1 - er)
            flux_moon = 1 - (-flux_moon * 10 ** -6)
    
    flux_total = 1 - ((1 - flux_planet[0]) + (1 - flux_moon[0]))

    # Return flux values:
    # 1) As if just the planet would transit
    # 2) As if just the moon would transit
    # 3) Both transit (real case)

    return flux_planet[0], flux_moon[0], flux_total



datapoints = 1000  # in total time grid
factor_durations = 5  # Duration of time grid in units of b=0 planetary transit durations
t_start = -factor_durations * transit_duration_planet
t_end = factor_durations * transit_duration_planet
timegrid = np.linspace(t_start, t_end, datapoints)
moon_flux_array = np.ones(datapoints)
planet_flux_array = np.ones(datapoints)
total_flux_array = np.ones(datapoints)

#print(timegrid)
#x_planet = 0.2
y_planet = 0.5
#time = 1.1
time_offset = 10

for idx, x_planet in enumerate(timegrid):

    flux_planet, flux_moon, flux_total = flux(
        r_moon,
        a_moon,
        per_moon,
        tau_moon,
        Omega_moon,
        w_moon,
        i_moon,
        r_planet,
        x_planet,
        y_planet,
        mass_ratio,
        R_star,
        u1,
        u2,
        time=time_offset + x_planet,
    )
    #print(flux_planet, flux_moon, flux_total)
    
    planet_flux_array[idx] = flux_planet
    moon_flux_array[idx] = flux_moon
    total_flux_array[idx] = flux_total


plt.plot(timegrid, planet_flux_array, color="blue")
plt.plot(timegrid, moon_flux_array, color = "black")
plt.plot(timegrid, total_flux_array, color = "red")
plt.show()

"""

phases = np.linspace(0, 1, 100)
for index, phase in enumerate(phases):
    figure, axes = plt.subplots()
    Sun = plt.Circle((0, 0), 1, color="yellow")
    plt.gcf().gca().add_artist(Sun)

    # Planet
    x_planet = 0.2
    y_planet = 0.5
    PlanetCircle = plt.Circle((x_planet, y_planet), r_planet/R_star, color = 'black', zorder = 4)
    plt.gcf().gca().add_artist(PlanetCircle)

    # Moon single position
    x_moon, y_moon = xypos_moon(
        a_moon / R_star,
        per_moon,
        tau_moon,
        Omega_moon,
        w_moon,
        i_moon,
        phase * per_moon
        )

    # Moon whole ellipse (array)
    xvals, yvals = xypos_moon(
        a_moon / R_star,
        per_moon,
        tau_moon,
        Omega_moon,
        w_moon,
        i_moon,
        np.linspace(0, 1 * per_moon, datapoints)
        )

    moon_x_static = x_moon + x_planet
    moon_y_static = y_moon + y_planet

    planet_x_bary = x_planet - x_moon * mass_ratio
    planet_y_bary = y_planet - y_moon * mass_ratio

    moon_x_bary = x_moon + x_planet + x_moon * mass_ratio
    moon_y_bary = y_moon + y_planet + y_moon * mass_ratio

    pc = plt.Circle((planet_x_bary, planet_y_bary), r_planet/R_star, color = 'red', zorder = 4)
    plt.gcf().gca().add_artist(pc)

    pc = plt.Circle((moon_x_bary, moon_y_bary), r_moon/R_star, color = 'red', zorder = 4)
    plt.gcf().gca().add_artist(pc)

    MoonCircle = plt.Circle((moon_x_static, moon_y_static), r_moon/R_star, color = 'black', zorder = 4)
    plt.gcf().gca().add_artist(MoonCircle)

    # Planet orbit
    plt.plot(xvals+x_planet, yvals+y_planet, color="black")

    # Bary curve
    x_moon_with_bary = xvals+x_planet + xvals * mass_ratio
    y_moon_with_bary = yvals+y_planet + yvals * mass_ratio
    plt.plot(x_moon_with_bary, y_moon_with_bary, color="red")





    plt.axis([-1.1, +1.1, -1.1, +1.1])  
    axes.set_aspect(1)
    digits_filename = 4
    filename = str(index).zfill(digits_filename) + ".png"
    plt.show()
    #plt.savefig(filename, bbox_inches='tight')
    #plt.close()
# convert *.png movie_bary.gif
"""
