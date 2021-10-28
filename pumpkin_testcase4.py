from pumpkin import pumpkin
import matplotlib.pyplot as plt

# We start with locations in a xy grid and convert to the Fewell notation later
R_star = 1

# Planet:
x_p = 0.6
y_p = -0.3

# Moon:
x_m = 0.7
y_m = -0.4

# The input parameters are the three radii, ordered so that r1 ≥ r2 ≥ r3,
r_planet = 0.5  # planet
r_moon = 0.3  # moon

flux_moon = 0.99

occulted_flux_moon, er = pumpkin(x_p, y_p, x_m, y_m, r_planet, r_moon, flux_moon)
print("occulted_flux_moon", occulted_flux_moon)

# Make visualization to visually verify original circle radii and locations

figure, axes = plt.subplots()
plt.gcf().gca().add_artist(plt.Circle((0, 0), R_star, color="yellow"))
plt.gcf().gca().add_artist(plt.Circle((x_p, y_p), r_planet, color="blue", fill=False))
plt.gcf().gca().add_artist(plt.Circle((x_m, y_m), r_moon, color="black", fill=False))
plt.text(0,1.2,"{:10.4f}".format(er), horizontalalignment="center")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
axes.set_aspect(1)
plt.show()