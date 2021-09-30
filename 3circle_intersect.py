"""
3 circle intersect Python Code
Algorithm following description of M.P. Fewell (2006) "Area of Common Overlap of Three Circles " 
Section 5.1
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.989.1088&rep=rep1&type=pdf
"""

from numpy import sqrt, cos, arcsin

# The input parameters are the three radii, ordered so that r1 ≥ r2 ≥ r3,
r1 = 1  # star
r2 = 1  # planet
r3 = 1  # moon

# and the three separations of circle centres d12, d13, d23
d12 = 0.2
d13 = 0.3
d23 = 0.4

# The steps required to compute the area of the circular triangle are:
# Step 1. Check whether circles 1 and 2 intersect by testing d12.
# If not satisfied, then there is no circular triangle and the algorithm terminates.

# Equation 4:
condition1 = (r1 - r2) < d12 < (r1 + r2)
if condition1:
    print("Condition 1 true, continue")
else:
    print("Condition 1 false; stop here")

# Step 2. Calculate the coordinates of the relevant intersection point of circles 1 and 2:

# Equation 6:
x12 = (r1 ** 2 - r2 ** 2 + d12 ** 2) / (2 * d12)
y12 = (1 / (2 * d12)) * sqrt(
    2 * d12 ** 2 * (r1 ** 2 + r2 ** 2) - (r1 ** 2 - r2 ** 2) ** 2 - d12 ** 4
)

# Step 3. Calculate the values of the sines and cosines of the angles θ′ and θ″:
# Equation 9
cos_theta1 = (d12 ** 2 + d13 ** 2 - d23 ** 2) / (2 * d12 * d13)
sin_theta1 = sqrt(1 - cos_theta1 ** 2)
# Equation 12
cos_theta2 = -(d12 ** 2 + d23 ** 2 - d13 ** 2) / (2 * d12 * d23)
sin_theta2 = sqrt(1 - cos_theta2 ** 2)

# Step 4. Check that circle 3 is placed so as to form a circular triangle. The conditions
# must both be satisfied. Otherwise, there is no circular triangle and the algorithm terminates.
# Equation 14:
condition2 = (x12 - d13 * cos_theta1) ** 2 + (y12 - d13 * sin_theta1) ** 2 < r3 ** 2
condition3 = (x12 - d13 * cos_theta1) ** 2 + (y12 + d13 * sin_theta1) ** 2 > r3 ** 2

print(condition2, condition3)
if not condition2 and condition3:
    print("end")
else:
    print("continue")
# Step 5. Calculate the values of the coordinates of the relevant intersection points involving circle 3:

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
x23 = x23ii * cos_theta2 - y23ii * sin_theta1 + d12
y23 = x23ii * sin_theta2 + y23ii * cos_theta2

# Step 6. Use the coordinates of the intersection points to calculate the chord lengths c1, c2, c3
# i=1 j=2 k=3
# Equation 3:
c1 = sqrt((x12 - x23) ** 2 + (y12 - y23) ** 2)
c2 = sqrt((x12 - x13) ** 2 + (y12 - y13) ** 2)
c3 = sqrt((x23 - x13) ** 2 + (x23 - y13) ** 2)

# Step 7. Check whether more than half of circle 3 is included in the circular triangle,
# so as to choose the correct expression for the area.
# That is, determine whether condition5 is true or false.
# Equation 15:
condition4 = (d13 * sin_theta1) < (y13 + (y23 - y13) / (x23 - x13)) * (
    d13 * cos_theta1 - x13
)
print(condition4)

if condition4:
    variant = (c3 / 4) * sqrt(4 * r3 ** 2 - c3 ** 2)
else:
    variant = (-c3 / 4) * sqrt(4 * r3 ** 2 - c3 ** 2)
print(variant)

# The area is given by
# Equation 1,16:

segment1 = (
    1 / 4 * sqrt((c1 + c2 + c3) * (c2 + c2 - c1) * (c1 + c3 - c2) * (c1 + c2 - c3))
)
print(segment1)

s1 = r1 ** 2 * arcsin(c1 / (2 * r1))
s2 = r2 ** 2 * arcsin(c2 / (2 * r2))
s3 = r3 ** 2 * arcsin(c3 / (2 * r3))
segment2 = s1 + s2 + s3

p1 = c1 / 4 * sqrt(4 * r1 ** 2 - c1 ** 2)
p2 = c2 / 4 * sqrt(4 * r2 ** 2 - c2 ** 2)
segment3 = p1 + p2
A = segment1 + segment2 - segment3 + variant
print(A)
