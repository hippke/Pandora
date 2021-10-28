from numpy import sqrt, pi, sin, cos, arcsin, arccos
from numba import jit


#@jit(cache=True, nopython=True, fastmath=True)
def circle_circle_intersect(r1, r2, d):
    """Calculates area of asymmetric "lens" in which two circles intersect
    Source: http://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    return (
        r1 ** 2 * (arccos(((d ** 2) + (r1 ** 2) - (r2 ** 2)) / (2 * d * r1)))
        + ((r2 ** 2) * (arccos((((d ** 2) + (r2 ** 2) - (r1 ** 2)) / (2 * d * r2)))))
        - (0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    )


#@jit(cache=True, nopython=True, fastmath=True)
def eclipse_ratio(distance_planet_moon, r_planet, r_moon):
    """Returns eclipsed ratio [0..1] using circle_circle_intersect"""
    eclipsing = False
    eclipse_ratio = 0
    if abs(distance_planet_moon) < (r_planet + r_moon):
        eclipsing = True
        if ((r_planet - r_moon) > abs(distance_planet_moon)):
            eclipse_ratio = 1
            return eclipse_ratio
    # For partial eclipses, get the fraction of moon eclipse using transit...
    if eclipsing:
        eclipse_ratio = circle_circle_intersect(
            r_planet, r_moon, distance_planet_moon)
        # ...and transform this value into how much AREA is really eclipsed
        eclipse_ratio = eclipse_ratio / (pi * r_moon ** 2)
    return eclipse_ratio


def fewell_coordinate_transform(x_p, y_p, x_m, y_m):
    """
    Coordinate system transformation required for Fewell (2006) 3-circle intersect
    1st circle (not treated here) is at coordinate origin [0,0]
    2nd (x_p, y_p) and 3rd (x_m, y_m) circle locations are input parameters in [-inf, +inf]
    Algorithm turns the coordinate system so that: y_p == 0 and y_m > 0

    Parameters
    ----------
    x_p, y_p, x_m, y_m : float
        Original coordinates in [-inf, +inf]
    Returns
    -------
    x_p_F, y_p_F, x_m_F, y_m_F, theta_p_F : float
        Turned coordinates with y_p_F == 0 and y_m_F > 0
    """
    
    # Determine angle between positive x axis and planet (theta_p_F)
    if y_p >= 0:
        theta_p_F = arccos(x_p / sqrt(x_p ** 2 + y_p ** 2))
    elif y_p < 0:
        theta_p_F = 2 * pi - arccos(x_p / sqrt(x_p ** 2 + y_p ** 2))
    
    # Cartesian rotation of coordinate system by angle theta_p_F
    x_p_F =  x_p * cos(theta_p_F) + y_p * sin(theta_p_F)
    y_p_F = -x_p * sin(theta_p_F) + y_p * cos(theta_p_F)

    x_m_F = x_m * cos(theta_p_F) + y_m * sin(theta_p_F)
    y_m_F = -x_m * sin(theta_p_F) + y_m * cos(theta_p_F)

    if y_m_F < 0:
        y_m_F = -y_m_F

    assert y_m_F > 0
    assert y_p_F < 1e-8

    return x_p_F, y_p_F, x_m_F, y_m_F  #, theta_p_F



#@jit(nopython=True, cache=True)
def fewell(x1, y1, x2, y2, x3, y3, r1, r2, r3):
    """Returns area of intersection of 3 circles with different locations and radii
    Algorithm following description of M.P. Fewell (2006)
    "Area of Common Overlap of Three Circles" Section 5.1
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.989.1088&rep=rep1&type=pdf
    """

    # The input parameters are the three radii, ordered so that r1 ≥ r2 ≥ r3
    # x1, y1: star; x2, y2: planet; x3, y3: moon
    x_p_F, y_p_F, x_m_F, y_m_F = fewell_coordinate_transform(x2, y2, x3, y3)
    print("x_p_F, y_p_F, x_m_F, y_m_F", x_p_F, y_p_F, x_m_F, y_m_F)

    # Separations of circle centres d12, d13, d23
    d12 = sqrt((x_p_F - x1) ** 2 + (y_p_F - y1) ** 2)
    d13 = sqrt((x_m_F - x1) ** 2 + (y_m_F - y1) ** 2)
    d23 = sqrt((x_m_F - x_p_F) ** 2 + (y_m_F - y_p_F) ** 2)

    # The steps required to compute the area of the circular triangle are:
    # Step 1. Check whether circles 1 and 2 intersect by testing d12.
    # If not satisfied, then there is no circular triangle and the algorithm terminates.

    # Equation 4:
    if not (r1 - r2) < d12 < (r1 + r2):
        print("Fewell stopped at condition 1, moving to circle_circle_intersect")
        d = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) * r1
        return eclipse_ratio(d, r2, r3)

    # Step 2. Calculate the coordinates of the relevant intersection point of 
    # circles 1 and 2 (Equation 6):
    x12 = (r1 ** 2 - r2 ** 2 + d12 ** 2) / (2 * d12)
    y12 = (1 / (2 * d12)) * sqrt(
        2 * d12 ** 2 * (r1 ** 2 + r2 ** 2) - (r1 ** 2 - r2 ** 2) ** 2 - d12 ** 4
    )

    # Step 3. Calculate the values of the sines and cosines of the angles θ′ and θ″:
    # Equation 9
    # Numerical stability:
    divisor =  (2 * d12 * d13)
    if divisor == 0:
        divisor = 1e-8
    cos_theta1 = (d12 ** 2 + d13 ** 2 - d23 ** 2) / divisor

    # Numerical stability:
    root = 1 - cos_theta1 ** 2
    if root < 0:
        root = 0
    sin_theta1 = sqrt(root)

    # Equation 12
    # Numerical stability:
    divisor =  (2 * d12 * d23)
    if divisor == 0:
        divisor = 1e-8
    cos_theta2 = -(d12 ** 2 + d23 ** 2 - d13 ** 2) / divisor

    # Numerical stability:
    root = 1 - cos_theta2 ** 2
    if root < 0:
        root = 0
    sin_theta2 = sqrt(root)

    # Step 4. Check that circle 3 is placed so as to form a circular triangle. 
    # The conditions must both be satisfied. Otherwise, there is no circular triangle 
    # and the algorithm terminates. (Equation 14):
    condition2 = (x12 - d13 * cos_theta1) ** 2 + (y12 - d13 * sin_theta1) ** 2 < r3 ** 2
    condition3 = (x12 - d13 * cos_theta1) ** 2 + (y12 + d13 * sin_theta1) ** 2 > r3 ** 2
    
    if not (condition2 and condition3):
        print("condition2", condition2)
        print("left part of cond2", (x12 - d13 * cos_theta1) ** 2 + (y12 - d13 * sin_theta1) ** 2)
        print("right part of cond2", r3 ** 2)
        print("condition3", condition3)
        print("Fewell stopped at condition 2, moving to circle_circle_intersect")
        d = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) * r1
        return eclipse_ratio(d, r2, r3)    
    
    # Step 5. Calculate the values of the coordinates of the relevant intersection 
    # points involving circle 3:

    # Equation 7:
    x13i = (r1 ** 2 - r3 ** 2 + d13 ** 2) / (2 * d13)
    y13i = (-1 / (2 * d13)) * sqrt(
        2 * d13 ** 2 * (r1 ** 2 + r3 ** 2) - (r1 ** 2 - r3 ** 2) ** 2 - d13 ** 4
    )

    # Equation 8:
    x13 = x13i * cos_theta1 - y13i * sin_theta1
    y13 = x13i * sin_theta1 + y13i * cos_theta1

    # Equation 10:
    x23ii = (r2 ** 2 - r3 ** 2 + d23 ** 2) / ((2 * d23))
    y23ii = (1 / (2 * d23)) * sqrt(
        2 * d23 ** 2 * (r2 ** 2 + r3 ** 2) - (r2 ** 2 - r3 ** 2) ** 2 - d23 ** 4
    )

    # Equation 11:
    x23 = x23ii * cos_theta2 - y23ii * sin_theta2 + d12
    y23 = x23ii * sin_theta2 + y23ii * cos_theta2

    # Step 6. Use the coordinates of the intersection points to calculate 
    # the chord lengths c1, c2, c3 (Equation 3):
    c1 = sqrt((x12 - x13) ** 2 + (y12 - y13) ** 2)
    c2 = sqrt((x12 - x23) ** 2 + (y12 - y23) ** 2)
    c3 = sqrt((x13 - x23) ** 2 + (y13 - y23) ** 2)

    # Step 7. Check whether more than half of circle 3 is included in the circular 
    # triangle, so as to choose the correct expression for the area.
    # That is, determine whether condition4 is true or false (Equation 15):
    condition4 = (d13 * sin_theta1) < (y13 + ((y23 - y13) / (x23 - x13))) * (
        d13 * cos_theta1 - x13
    )

    # Equation 16:
    variant = 0.25 * c3 * sqrt(4 * r3 ** 2 - c3 ** 2)
    if not condition4:
        variant = -variant

    # The area is given by (Equation 1):
    segment1 = (
        0.25 * sqrt((c1 + c2 + c3) * (c2 + c3 - c1) * (c1 + c3 - c2) * (c1 + c2 - c3))
    )

    s1 = r1 ** 2 * arcsin(c1 / (2 * r1))
    s2 = r2 ** 2 * arcsin(c2 / (2 * r2))
    s3 = r3 ** 2 * arcsin(c3 / (2 * r3))
    segment2 = s1 + s2 + s3

    p1 = 0.25 * c1 * sqrt(4 * r1 ** 2 - c1 ** 2)
    p2 = 0.25 * c2 * sqrt(4 * r2 ** 2 - c2 ** 2)
    segment3 = p1 + p2

    A = segment1 + segment2 - segment3 + variant
    return A


def pumpkin(x_p, y_p, x_m, y_m, r_planet, r_moon, flux_moon):
    """Checks if planet-moon occultation present. If yes, returns adjusted moon flux.
    Parameters
    ----------
    x_p, y_p, x_m, y_m : float
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
    # Case 1: None
    # Case 2: Occultation, both bodies on star or off star --> 2-circle intersect
    # Case 3: Occultation, any body on limb --> Fewell 3-circle intersect
    #         This case has sub-cases where Fewell calls 2-circle intersect
    #         Example: Planet on limb, moon fully on star: Occulted area fully on star

    R_star = 1

    planet_moon_occultation = False
    on_limb = False

    distance_planet_moon = (
        sqrt((x_m - x_p) ** 2 + (y_m - y_p) ** 2) * R_star
    )
    radial_distance_moon = sqrt(x_m ** 2 + y_m ** 2)
    radial_distance_planet = sqrt(x_p ** 2 + y_p ** 2)
    if (abs(1-(radial_distance_moon)) < (r_moon / R_star)):
        on_limb = True
    if (abs(1-(radial_distance_planet)) < (r_planet / R_star)):
        on_limb = True

    if abs(distance_planet_moon) < (r_planet + r_moon):
        planet_moon_occultation = True
        print("Planet-moon eclipse YES")
    else:
        print("Planet-moon eclipse NO")
        return flux_moon, 0

    # Case 2: Occultation, both bodies on star or off star --> 2 circle intersect
    if planet_moon_occultation and not on_limb:
        er = eclipse_ratio(r_planet, r_moon, distance_planet_moon)
        if er > 0:
            print("Planet-moon eclipse, no body on limb", er)
            flux_moon = -(1 - flux_moon) * 10 ** 6
            flux_moon = flux_moon * (1 - er)
            flux_moon = 1 - (-flux_moon * 10 ** -6)
        flux_total = 1 - ((1 - flux_planet) + (1 - flux_moon))
        return flux_total, er

    # Case 3: Occultation, any body on limb --> 3 circle intersect
    # HERE: ADD FEWELL WHEN READY
    if planet_moon_occultation and on_limb:
        print("Planet-moon eclipse, at least one body on limb")
        er = fewell(0, 0, x_p, y_p, x_m, y_m, R_star, r_planet, r_moon)
        print("Eclipsed ratio:", er)
        if er > 0:
            flux_moon = -(1 - flux_moon) * 10 ** 6
            flux_moon = flux_moon * (1 - er)
            flux_moon = 1 - (-flux_moon * 10 ** -6)
    return flux_moon, er
