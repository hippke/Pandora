from numpy import sqrt, pi, arccos
from numba import jit


@jit(cache=True, nopython=True, fastmath=True)
def circle_circle_intersect(r1, r2, d):
    """Calculates area of asymmetric "lens" in which two circles intersect
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    return (
        r1 ** 2 * (arccos(((d ** 2) + (r1 ** 2) - (r2 ** 2)) / (2 * d * r1)))
        + ((r2 ** 2) * (arccos((((d ** 2) + (r2 ** 2) - (r1 ** 2)) / (2 * d * r2)))))
        - (0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    )


@jit(cache=True, nopython=True, fastmath=True)
def eclipse_ratio(distance_planet_moon, r_planet, r_moon):
    """Returns eclipsed ratio [0..1] using circle_circle_intersect"""
    eclipsing = False
    eclipse_ratio = 0
    if abs(distance_planet_moon) < (r_planet + r_moon):
        eclipsing = True
        if ((r_planet - r_moon) > abs(distance_planet_moon)):
            eclipse_ratio = 1
            return eclipse_ratio
    # For partial eclipses, get the fraction of moon eclipse using transit
    if eclipsing:
        eclipse_ratio = circle_circle_intersect(
            r_planet, r_moon, distance_planet_moon)
        # ...and transform this value into how much AREA is really eclipsed


        # CHECK THIS MIGHT BE WRONG???

        eclipse_ratio = eclipse_ratio / (pi * r_moon ** 2)
    return eclipse_ratio



"""
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
    print("Planet-moon eclipse")

# Case 2: Occultation, both bodies on star or off star --> 2 circle intersect
if planet_moon_occultation and not on_limb:
    er = eclipse_ratio(distance_planet_moon, r_planet, r_moon)
    if er > 0:
        print("2 body eclipse", er)
        flux_moon = -(1 - flux_moon) * 10 ** 6
        flux_moon = flux_moon * (1 - er)
        flux_moon = 1 - (-flux_moon * 10 ** -6)
    flux_total = 1 - ((1 - flux_planet) + (1 - flux_moon))

# Case 3: Occultation, any body on limb --> 3 circle intersect
# HERE: ADD FEWELL WHEN READY
if planet_moon_occultation and on_limb:
    er = eclipse_ratio(distance_planet_moon, r_planet, r_moon)
    if er > 0:
        print("2 body eclipse", er)
        flux_moon = -(1 - flux_moon) * 10 ** 6
        flux_moon = flux_moon * (1 - er)
        flux_moon = 1 - (-flux_moon * 10 ** -6)
"""