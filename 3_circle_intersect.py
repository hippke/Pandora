#from numba import jit
from math import acos, degrees, asin, sqrt, radians, cos, pi


#@jit(nopython=True, cache=True)
def circle_circle_circle_intersect(x1, y1, x2, y2, x3, y3, r1, r2, r3):
    """Returns area of intersection of 3 circles with different locations and radii
    Algorithm following description of M.P. Fewell (2006)
    "Area of Common Overlap of Three Circles" Section 5.1
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.989.1088&rep=rep1&type=pdf
    """

    # Separations of circle centres d12, d13, d23
    d12 = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    d13 = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    d23 = sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    print("d12, d13, d23", d12, d13, d23)

    # The steps required to compute the area of the circular triangle are:
    # Step 1. Check whether circles 1 and 2 intersect by testing d12.
    # If not satisfied, then there is no circular triangle and the algorithm terminates.

    # Equation 4:
    print("r1 - r2", r1 - r2)
    print("d12", d12)
    print("r1 + r2", r1 + r2)
    if not (r1 - r2) < d12 < (r1 + r2):
        print("stop at cond1")
        return 0
    else:
        print("continue at cond1")

    # Step 2. Calculate the coordinates of the relevant intersection point of 
    # circles 1 and 2 (Equation 6):
    x12 = (r1 ** 2 - r2 ** 2 + d12 ** 2) / (2 * d12)
    print("x12", x12)
    y12 = (1 / (2 * d12)) * sqrt(
        2 * d12 ** 2 * (r1 ** 2 + r2 ** 2) - (r1 ** 2 - r2 ** 2) ** 2 - d12 ** 4
    )
    print("y12", y12)

    # Step 3. Calculate the values of the sines and cosines of the angles θ′ and θ″:
    # Equation 9
    cos_theta1 = (d12 ** 2 + d13 ** 2 - d23 ** 2) / (2 * d12 * d13)
    print("cos_theta1", cos_theta1)
    sin_theta1 = sqrt(1 - cos_theta1 ** 2)
    print("sin_theta1", sin_theta1)

    # Equation 12
    cos_theta2 = -(d12 ** 2 + d23 ** 2 - d13 ** 2) / (2 * d12 * d23)
    print("cos_theta2", cos_theta2)
    sin_theta2 = sqrt(1 - cos_theta2 ** 2)
    print("sin_theta2", sin_theta2)

    # Step 4. Check that circle 3 is placed so as to form a circular triangle. 
    # The conditions must both be satisfied. Otherwise, there is no circular triangle 
    # and the algorithm terminates. (Equation 14):
    
    print((x12 - d13 * cos_theta1) ** 2, (y12 - d13 * sin_theta1) ** 2, r3 ** 2)

    condition2 = (x12 - d13 * cos_theta1) ** 2 + (y12 - d13 * sin_theta1) ** 2 < r3 ** 2
    condition3 = (x12 - d13 * cos_theta1) ** 2 + (y12 + d13 * sin_theta1) ** 2 > r3 ** 2
    if not (condition2 and condition3):
        print(condition2, condition3)
        print("stop at cond2 cond3")
        #return 0
    

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
        #print("stop at cond4")
        variant = -variant

    # The area is given by (Equation 1):
    segment1 = (
        0.25 * sqrt((c1 + c2 + c3) * (c2 + c3 - c1) * (c1 + c3 - c2) * (c1 + c2 - c3))
    )

    s1 = r1 ** 2 * asin(c1 / (2 * r1))
    s2 = r2 ** 2 * asin(c2 / (2 * r2))
    s3 = r3 ** 2 * asin(c3 / (2 * r3))
    segment2 = s1 + s2 + s3

    p1 = 0.25 * c1 * sqrt(4 * r1 ** 2 - c1 ** 2)
    p2 = 0.25 * c2 * sqrt(4 * r2 ** 2 - c2 ** 2)
    segment3 = p1 + p2

    A = segment1 + segment2 - segment3 + variant
    return A


"""
3 circle intersect Python Code

"""
import time
import matplotlib.pyplot as plt

# We start with locations in a xy grid and convert to the separation notation later
# Sun:
x1 = 0
y1 = 0

# Planet:
x2 = 1
y2 = 0

# Moon:
x3 = 0.45
y3 = 0.2

# Here the fun from Fewell begins:
# The input parameters are the three radii, ordered so that r1 ≥ r2 ≥ r3,
r1 = 1  # star
r2 = 0.5  # planet
r3 = 0.3  # moon

# Make visualization to visually verify circle radii and locations
figure, axes = plt.subplots()
plt.gcf().gca().add_artist(plt.Circle((x1, y1), r1, color="yellow"))
plt.gcf().gca().add_artist(plt.Circle((x2, y2), r2, color="blue", fill=False))
plt.gcf().gca().add_artist(plt.Circle((x3, y3), r3, color="black", fill=False))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
axes.set_aspect(1)
plt.show()

t1 = time.time()
for i in range(1):
    a = circle_circle_circle_intersect(x1, y1, x2, y2, x3, y3, r1, r2, r3)
t2 = time.time()
print("Area a, a/a(r1)", a, a/(pi*r1**2))
