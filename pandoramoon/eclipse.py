import numpy as np
from numpy import sqrt, pi, arccos, abs, ceil, fliplr, flipud
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def cci(r1, r2, d):
    """Calculates area of asymmetric "lens" in which two circles intersect
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    if r1 < d - r2:
        return 0
    elif r1 >= d + r2:
        return pi * r2 ** 2
    elif d - r2 <= -r1:
        return pi * r1 ** 2
    else:
        return (
            r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            + r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            - 0.5 * sqrt((-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1))
        )


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def eclipse_ratio(distance_planet_moon, r_planet, r_moon):
    """Returns eclipsed ratio [0..1] using circle_circle_intersect"""
    eclipsing = False
    eclipse_ratio = 0
    if abs(distance_planet_moon) < (r_planet + r_moon):
        eclipsing = True
        if (r_planet - r_moon) > abs(distance_planet_moon):
            eclipse_ratio = 1
            return eclipse_ratio
    # For partial eclipses, get the fraction of moon eclipse using transit...
    if eclipsing:
        if distance_planet_moon == 0:
            distance_planet_moon = 1e-10
        eclipse_ratio = cci(r_planet, r_moon, distance_planet_moon)
        # ...and transform this value into how much AREA is really eclipsed
        eclipse_ratio = eclipse_ratio / (pi * r_moon ** 2)
    return eclipse_ratio


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def pixelart(xp, yp, xm, ym, r_planet, r_moon, numerical_grid):
    if numerical_grid % 2 == 0:  # assure pixel number is odd for perfect circle
        numerical_grid += 1
    r_star = (1 / r_moon) * numerical_grid
    image = np.zeros((numerical_grid + 1, numerical_grid + 1), dtype="int8")
    
    color_star = 5    # arbitrary values, but useful for visualization
    color_moon = 3    # the sum of the values must be unique to identify the
    color_planet = 2  # total overlap: area of moon on star occulted by planet
    all_colors = color_star + color_moon + color_planet

    # Paint moon circle by painting one quarter, flipping it over, and rolling it down
    # Faster than painting naively the full circle, because it saves 3/4 of dist calcs
    # Caching this part would be a good idea, but numba memory mgmt caused crashes
    # The gain is very small anyways and the complexity not worth it
    # The version here, which replaces the usual sqrt with **2, is ~5% faster

    # Paint upper left corner
    anti_aliasing = np.sqrt(((numerical_grid + 1) ** 2 - (numerical_grid ** 2))) / 2
    mid = int(ceil(numerical_grid / 2))
    for x in range(mid):
        for y in range(mid):
            d_moon = (numerical_grid - 2 * x) ** 2 + (numerical_grid - 2 * y) ** 2
            if d_moon < (numerical_grid**2 + anti_aliasing):
                image[x, y] = color_moon
    image[mid:,:mid] = flipud(image[:mid:,:mid]) # Copy upper left to upper right
    image[:,mid:] = fliplr(image[:,:mid])  # Copy upper half to lower half

    # Now add planet and star
    anti_aliasing = -0.5 / numerical_grid  # Working with sqrt again
    for x in range(numerical_grid + 1):
        for y in range(numerical_grid + 1):
            d_star = sqrt(
                (xm * r_star + 2 * x - numerical_grid) ** 2
                + (ym * r_star + 2 * y - numerical_grid) ** 2
            )
            if d_star < r_star - anti_aliasing:
                image[x, y] += color_star

            d_planet = sqrt(
                ((-(xp - xm) * r_star) + 2 * x - numerical_grid) ** 2
                + ((-(yp - ym) * r_star) + 2 * y - numerical_grid) ** 2
            )
            if d_planet < (r_planet / r_moon) * numerical_grid - anti_aliasing:
                image[x, y] += color_planet

    moon_sum_analytical = pi * ((numerical_grid) / 2) ** 2
    moon_occult_frac = np.sum(image == all_colors) / moon_sum_analytical
    cci = eclipse_ratio(sqrt(xm ** 2 + ym ** 2), 1, r_moon)
    if cci > 0:
        return min(1, (1 - ((cci - moon_occult_frac) / cci)))
    else:
        return 1


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def eclipse(xp, yp, xm, ym, r_planet, r_moon, flux_moon, numerical_grid):
    """Checks if planet-moon occultation present. If yes, returns adjusted moon flux.
    Parameters
    ----------
    xp, yp, x_m, ym : float
        Planet and moon coordinates. Normalized so that R_star = 1 at (0,0)
    r_planet, r_moon : float
        Planet and moon radii. Normalized so that R_star = 1
    flux_moon : float
        Un-occulted moon flux (from Mandel-Agol model) in [0,1]

    Returns
    -------
    occulted_flux_moon : float
        Occulted moon flux <= flux_moon. Assumes planet occults moon.
    """

    # Planet-Moon occultation
    # Case 1: No occultation
    # Case 2: Occultation, both bodies on star or off star --> 2-circle intersect
    # Case 3: Occultation, any body on limb --> Numerical solution
    for idx in range(len(xp)):
        planet_moon_occultation = False
        on_limb = False

        # Check if moon or planet are on stellar limb
        if abs(1 - (sqrt(xm[idx] ** 2 + ym[idx] ** 2))) < (r_moon):
            on_limb = True
        if abs(1 - (sqrt(xp[idx] ** 2 + yp[idx] ** 2))) < (r_planet):
            on_limb = True

        # Check if planet-moon occultation
        distance_p_m = sqrt((xm[idx] - xp[idx]) ** 2 + (ym[idx] - yp[idx]) ** 2)
        if abs(distance_p_m) < (r_planet + r_moon):
            planet_moon_occultation = True

        # Case 1: No occultation
        else:
            continue

        # Case 2: Occultation, both bodies on star or off star --> 2 circle intersect
        if planet_moon_occultation and not on_limb:
            er = eclipse_ratio(distance_p_m, r_planet, r_moon)

        # Case 3: Occultation, any body on limb --> numerical estimate with pixel-art
        if planet_moon_occultation and on_limb:
            er = pixelart(
                xp[idx],
                yp[idx],
                xm[idx],
                ym[idx],
                r_planet,
                r_moon,
                numerical_grid
            )

        # For Cases 2+3: Calculate reduced moon flux
        if er > 0:
            flux_moon[idx] = -(1 - flux_moon[idx]) * 10 ** 6
            flux_moon[idx] = flux_moon[idx] * (1 - er)
            flux_moon[idx] = 1 - (-flux_moon[idx] * 10 ** -6)
    return flux_moon
