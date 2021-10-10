from math import pi, cos, acos, sqrt
import matplotlib.pyplot as plt

from pylab import *

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
    
    # determine angle between positive x axis and planet (theta_p_F)
    if y_p >= 0:
        theta_p_F = acos(x_p / sqrt(x_p ** 2 + y_p ** 2))
    elif y_p < 0:
        theta_p_F = 2*pi - acos(x_p / sqrt(x_p ** 2 + y_p ** 2))
    
    # Cartesian rotation of coordinate system by angle theta_p_F
    x_p_F =  x_p * cos(theta_p_F) + y_p * sin(theta_p_F)
    y_p_F = -x_p * sin(theta_p_F) + y_p * cos(theta_p_F)

    x_m_F =  x_m * cos(theta_p_F) + y_m * sin(theta_p_F)
    y_m_F = -x_m * sin(theta_p_F) + y_m * cos(theta_p_F)

    return x_p_F, y_p_F, x_m_F, y_m_F, theta_p_F


# We start with locations in a xy grid and convert to the Fewell notation later
# Star:
x_s = 0
y_s = 0

# Planet:
x_p = -0.9
y_p = -0.0

# Moon:
x_m = 0.45
y_m = 0.2

# The input parameters are the three radii, ordered so that r1 ≥ r2 ≥ r3,
r1 = 1  # star
r2 = 0.5  # planet
r3 = 0.3  # moon

print("Distances before coordinate transformation")
d1_F = sqrt((x_s - x_p)**2 + (y_s - y_p)**2)
d2_F = sqrt((x_s - x_m)**2 + (y_s - y_m)**2)
d3_F = sqrt((x_p - x_m)**2 + (y_p - y_m)**2)
print("Distance star-planet", d1_F)
print("Distance star-moon", d2_F)
print("Distance planet-moon", d3_F)

# Get transformed coordinates
x_p_F, y_p_F, x_m_F, y_m_F, theta_p_F = fewell_coordinate_transform(x_p, y_p, x_m, y_m)
print("x_p_F, y_p_F, x_m_F, y_m_F, theta_p_F", x_p_F, y_p_F, x_m_F, y_m_F, theta_p_F/pi*180)

print("Distances after coordinate transformation")
d1_F = sqrt((x_s - x_p_F)**2 + (y_s - y_p_F)**2)
d2_F = sqrt((x_s - x_m_F)**2 + (y_s - y_m_F)**2)
d3_F = sqrt((x_p_F - x_m_F)**2 + (y_p_F - y_m_F)**2)
print("Distance star-planet", d1_F)
print("Distance star-moon", d2_F)
print("Distance planet-moon", d3_F)


# Make visualization to visually verify original circle radii and locations
figure, axes = plt.subplots()
plt.gcf().gca().add_artist(plt.Circle((x_s, y_s), r1, color="yellow"))
plt.gcf().gca().add_artist(plt.Circle((x_p, y_p), r2, color="blue", fill=False))
plt.gcf().gca().add_artist(plt.Circle((x_m, y_m), r3, color="black", fill=False))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
axes.set_aspect(1)
plt.show()

# Make visualization to visually verify NEW circle radii and locations
figure, axes = plt.subplots()
plt.gcf().gca().add_artist(plt.Circle((x_s, y_s), r1, color="yellow"))
plt.gcf().gca().add_artist(plt.Circle((x_p_F, y_p_F), r2, color="blue", fill=False))
plt.gcf().gca().add_artist(plt.Circle((x_m_F, y_m_F), r3, color="black", fill=False))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
axes.set_aspect(1)
plt.show()

