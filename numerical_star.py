import numpy as np
import matplotlib.pyplot as plt

# Planet:
x_p = -0.97
y_p = -0.04

# Moon:
x_m = -0.99
y_m = -0.1

# The input parameters are the three radii, ordered so that r1 ≥ r2 ≥ r3,
r_planet = 0.1  # planet
r_moon = 0.05  # moon

axis_px = 100

r_star = (1/r_moon)*axis_px

image = np.zeros((axis_px + 1, axis_px + 1))
moon_sum_analytical = np.pi * ((axis_px)/2)**2

# Make radius half a pixel smaller to distribute error equally
anti_aliasing = -0.5*1/axis_px

for x in range(axis_px + 1):
    for y in range(axis_px + 1):
        d_star = np.sqrt((x_m * r_star+2*x-axis_px)**2 + (y_m * r_star+2*y-axis_px)**2)
        if d_star < r_star-anti_aliasing:
            image[x,y] = 0.5

        d_moon = np.sqrt((axis_px-2*x)**2 + (axis_px-2*y)**2)
        if d_moon < (axis_px-anti_aliasing):
            image[x,y] += 0.3

        d_planet = np.sqrt(((-(x_p - x_m) * r_star)+2*x-axis_px)**2 + ((-(y_p - y_m) * r_star)+2*y-axis_px)**2)
        if d_planet < (r_planet/r_moon)*axis_px-anti_aliasing:
            image[x,y] += 0.2

moon_occult_frac = np.sum(image==1) / moon_sum_analytical
print("occult fraction", moon_occult_frac)

c = plt.imshow(np.rot90(image), cmap ='gray', interpolation ='none')
plt.show()


