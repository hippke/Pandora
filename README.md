![Logo](https://github.com/hippke/Pandora/blob/main/docs/source/logo.png?raw=true)
## Fast open-source exomoon transit detection algorithm

[![pip](https://img.shields.io/badge/pip-install%20pandora--moon-blue.svg)](https://pypi.org/project/wotan/)
[![Documentation](https://img.shields.io/badge/documentation-%E2%9C%93-blue.svg)](https://wotan.readthedocs.io/en/latest/index.html)
[![Image](https://img.shields.io/badge/tutorials-%E2%9C%93-blue.svg)](https://github.com/hippke/wotan/tree/master/tutorials)
[![Image](https://img.shields.io/badge/arXiv-1906.00966-blue.svg)](https://arxiv.org/abs/1906.00966)

> Named after the imaginary 5th moon of the gas giant Polyphemus, orbiting Alpha Centauri A, in the film [Avatar](https://james-camerons-avatar.fandom.com/wiki/Pandora). 

Pandora employs an analytical photodynamical model, including:
- Stellar limb darkening
- Full and partial planet-moon eclipses
- Barycentric motion of planet and moon
- Find moons with nested sampling using [UltraNest](https://github.com/JohannesBuchner/UltraNest), [dynesty](https://github.com/joshspeagle/dynesty), and other tools
- Bayesian evidence calculation and model comparison


### Create transit lightcurve
```
import pandora
params = pandora.model_params()
params.per_bary = 365.25  # [days] 169.7
# (...) See tutorials for list of parameters
model = pandora.moon_model(params)
time, flux_total, flux_planet, flux_moon = model.light_curve()

plt.plot(time, flux_planet, color="blue")
plt.plot(time, flux_moon, color="red")
plt.plot(time, flux_total, color="black")
plt.show()
```
![lc](https://github.com/hippke/Pandora/blob/main/docs/source/lc_image.png?raw=true)

### Create video
With Pandora, you can create transit videos to understand, teach, and explore exomoon transits. Try it out:

```
video = model.video(
    limb_darkening=True, 
    teff=3200,
    planet_color="black",
    moon_color="black",
    ld_circles=100
)
video.save(filename="video.mp4", fps=25, dpi=200)
```
Videos should approximate the true situation as calculated by Pandora very well. They are, however, not pixel-perfect due to the underlying Matplotlib rendering.


![Video](https://github.com/hippke/Pandora/blob/main/docs/source/video_image.png?raw=true)

### Installation
Install with `pip install pandora-moon`.

Attribution
----------------
Please cite [Hippke et al. (2019, AJ, 158, 143)](https://ui.adsabs.harvard.edu/abs/2019AJ....158..143H/abstract) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@ARTICLE{2019AJ....158..143H,
       author = {{Hippke}, Michael and {David}, Trevor J. and {Mulders}, Gijs D. and
         {Heller}, Ren{\'e}},
        title = "{W{\={o}}tan: Comprehensive Time-series Detrending in Python}",
      journal = {\aj},
     keywords = {eclipses, methods: data analysis, methods: statistical, planetary systems, planets and satellites: detection, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = "2019",
        month = "Oct",
       volume = {158},
       number = {4},
          eid = {143},
        pages = {143},
          doi = {10.3847/1538-3881/ab3984},
archivePrefix = {arXiv},
       eprint = {1906.00966},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019AJ....158..143H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```
