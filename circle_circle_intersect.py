from numpy import sqrt, pi, arccos


def circle_circle_intersect(r1, r2, d):
    """Calculates area of asymmetric "lens" in which two circles intersect
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    return (
        r1 ** 2 * (arccos(((d ** 2) + (r1 ** 2) - (r2 ** 2)) / (2 * d * r1)))
        + ((r2 ** 2) * (arccos((((d ** 2) + (r2 ** 2) - (r1 ** 2)) / (2 * d * r2)))))
        - (0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    )



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