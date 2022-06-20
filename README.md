![Logo](https://github.com/hippke/Pandora/blob/main/docs/source/logo_v6.png?raw=true)
## Fast open-source exomoon transit detection algorithm

[![pip](https://img.shields.io/badge/pip-install%20pandoramoon-blue.svg)](https://pypi.org/project/pandoramoon/)
[![Documentation](https://img.shields.io/badge/Documentation-%E2%9C%93-blue.svg)](https://pandora-moon.readthedocs.io/en/latest/index.html)
[![Image](https://img.shields.io/badge/Examples-%E2%9C%93-blue.svg)](https://github.com/hippke/Pandora/tree/main/examples)
[![Image](https://img.shields.io/badge/arXiv-2205.09410-blue.svg)](https://arxiv.org/abs/2205.09410))

> Named after the imaginary 5th moon of the gas giant Polyphemus, orbiting Alpha Centauri A, in the film [Avatar](https://james-camerons-avatar.fandom.com/wiki/Pandora). 

Pandora employs an analytical photodynamical model, including:
- Stellar limb darkening
- Full and partial planet-moon eclipses
- Barycentric motion of planet and moon

To search for moons, Pandora can be used with nested samplers, e.g. [UltraNest](https://github.com/JohannesBuchner/UltraNest) or [dynesty](https://github.com/joshspeagle/dynesty). We provide an [example workflow](https://github.com/hippke/Pandora/blob/main/examples/injection_retrieval_simple_ultranest.ipynb) for a planet+moon injection into Kepler-like data with Gaussian noise, UltraNest+Pandora recovery, cornerplots, and posterior exploration. Pandora is fast, calculating 10,000 models and log-likelihood evaluation per second (give or take an order of magnitude, depending on parameters and data). This means that a retrieval with 250 Mio. evaluations until convergence takes about 5 hours on a single core. Scaling with cores is worse than linear. For searches in large amounts of data, it is most efficient to assign one core per light curve.


### Create transit light curve

The following Python code snippet illustrates how to work with Pandora. To keep it short, it omits the long list of required parameters. A complete and working example can be found [here](https://github.com/hippke/Pandora/blob/main/examples/example.py).

```
import pandoramoon as pandora
params = pandora.model_params()
params.per_bary = 365.25  # [days]
# (...) See tutorials for list of parameters
time = pandora.time(params).grid()
model = pandora.moon_model(params)
flux_total, flux_planet, flux_moon = model.light_curve(time)

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
    time=time,
    limb_darkening=True, 
    teff=3200,
    planet_color="black",
    moon_color="black",
    ld_circles=100
)
video.save(filename="video.mp4", fps=25, dpi=200)
```
Videos approximate the true light curve as calculated by Pandora very well. They are, however, not pixel-perfect due to the underlying Matplotlib render engine. Klick the image to view the video:

[![Video](https://github.com/hippke/Pandora/blob/main/docs/source/Screenshot%202022-05-19%20at%2023.39.03.png?raw=true)](https://youtu.be/89lEuPgrl8s)

### Installation
Install with `pip install pandoramoon`. If you end up in dependency hell, set up a fresh environment:

```
conda create -n pandora_env python=3.9
conda activate pandora_env
conda install numpy matplotlib numba 
pip install pandoramoon
```

For sampling, the following packages will be useful:
```
conda install cython scipy
pip install ultranest dynesty h5py
```



Attribution
----------------
Please cite [Hippke & Heller (2022, A&A)](https://ui.adsabs.harvard.edu/abs/2022A%26A...662A..37H/abstract) if you find this code useful in your research. The BibTeX entry for the paper is:

```
@ARTICLE{2022A&A...662A..37H,
       author = {{Hippke}, Michael and {Heller}, Ren{\'e}},
        title = "{Pandora: A fast open-source exomoon transit detection algorithm}",
      journal = {\aap},
     keywords = {methods: data analysis, planets and satellites: detection, techniques: photometric},
         year = 2022,
        month = jun,
       volume = {662},
          eid = {A37},
        pages = {A37},
          doi = {10.1051/0004-6361/202243129},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022A&A...662A..37H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
