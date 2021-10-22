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
- Nested sampling with [UltraNest](https://github.com/JohannesBuchner/UltraNest)
- Bayesian evidence calculation
