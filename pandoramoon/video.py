"""Transit videos of a planet with a moon (matplotlib; imported only when a video is made)."""
import os

import numpy as np


def transit_video(px, py, mx, my, u1, u2, r_planet, r_moon, limb_darkening=True, teff=6000, planet_color="black",
                  moon_color="black", ld_circles=100, progress=True):
    """matplotlib FuncAnimation of the sky positions (px, py) of the planet and (mx, my) of the moon [R_star]
    in front of the star (limb-darkened with u1, u2 and coloured by teff [K] if limb_darkening).
    Save it with video.save("video.mp4", fps=25, dpi=200)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    with plt.style.context("dark_background"):          # does not change the user's global matplotlib style
        fig = plt.figure(figsize=(5, 5))
        axes = fig.add_subplot(111)
        axes.axis("off")
        axes.add_artist(plt.Circle((0, 0), 5, color="black"))
        if limb_darkening:
            teff = min(max(teff, 2300), 12000)           # the range of star_colors.csv
            star_colors = np.genfromtxt(os.path.join(os.path.dirname(__file__), "star_colors.csv"), delimiter=",")
            row = np.argmax(star_colors[:, 0] >= teff)
            r_star, g_star, b_star = star_colors[row, 1:4]
            for i in reversed(range(ld_circles)):
                impact = (i + 1) / ld_circles
                r_mid = (i + 0.5) / ld_circles
                m = np.sqrt(1 - min(r_mid ** 2, 1))
                ld = 1 - u1 * (1 - m) - u2 * (1 - m) ** 2
                axes.add_artist(plt.Circle((0, 0), impact, color=(r_star * ld, g_star * ld, b_star * ld)))
        else:
            axes.add_artist(plt.Circle((0, 0), 1, color="yellow"))
        axes.set_xlim(-1.05, 1.05)
        axes.set_ylim(-1.05, 1.05)
        # marker sizes: 260 points per stellar radius at figsize 5 and axis limits +-1.05
        (moon,) = axes.plot(mx[0], my[0], "o", color=moon_color, markerfacecolor=moon_color,
                            markeredgecolor=moon_color, markersize=260 * r_moon)
        (planet,) = axes.plot(px[0], py[0], "o", color=planet_color, markeredgecolor=planet_color,
                              markerfacecolor=planet_color, markersize=260 * r_planet)
    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(mx))
    else:
        pbar = None

    def update(coords):
        mx_, my_, px_, py_ = coords
        moon.set_data([mx_], [my_])
        planet.set_data([px_], [py_])
        if pbar is not None:
            pbar.update(1)
        return moon, planet

    def frames():
        yield from zip(mx, my, px, py)

    return FuncAnimation(fig, update, frames=frames, save_count=len(mx), blit=True)
