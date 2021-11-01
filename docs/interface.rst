Python Interface
================

This describes the Python interface of Pandora. There are actually two ways to obtain a model: A class-based and a function-based approach.

- The class-based version is more "pythonic" and in line with what users know from modules like batman, wotan, TLS, and others. It offers various gimicks such as video animations of transits.
- The function-based version is simpler, procedural, but "ugly" due to passing along many parameters. On the other hand, it is about 10% faster due to the lack of class instantiation. Its main use cases is for model evaluation with MCMC and nested samplers. 

Both versions must yield perfectly identical results. If you find any differences, please open a bug ticket on Github.


Define data for a model
-----------------------

.. class:: model_params(params)

Star parameters:

:R_star: *(float)* Stellar radius [km]
:u1: *(float)* Quadratic limb-darkening [0..1]
:u2: *(float)* Quadratic limb-darkening [0..1]

Planet parameters:

:per_planet: *(float)* Period [days]. Should be a free parameter.
:a_planet: *(float)* Semimajor axis [km]. Should be a free parameter.
:r_planet: *(float)* Radius [km]. Should be a free parameter.
:b_planet: *(float)* Impact parameter [0..1.x]. Should be a free parameter.
:t0_planet: *(float)* Time of inferior conjunction [days]. Should NOT be a free parameter.
:t0_planet_offset: *(float)* Time difference to inferior conjunction [days]. Should be a free parameter.
:M_planet: *(float)* Mass [kg]. Should be a free parameter.

Moon parameters:

:r_moon: *(float)*  Radius [km]. Should be a free parameter.
:a_moon: *(float)*  Semimajor axis [km]. Should be a free parameter.
:tau_moon: *(float)*  Time of periapsis passage [0..1] normalized by period. Should be a free parameter.
:Omega_moon: *(float)* Longitude of the ascending node [0..360 deg]. Should be a free parameter.
:i_moon: *(float)* Orbit inclination [0..360 deg]. Should be a free parameter.
:M_moon: *(float)* Mass [kg]. Should be a free parameter.

Other model parameters:

:epochs: (*int*) Number of transit epochs in time series.
:epoch_duration: (*float*) Duration of each epoch, centered at planetary transit [days]
:cadences_per_day: (*int*) Number of exposured (cadences) per day. Should be constant.
:epoch_distance: (*float*) Time distance between each epoch [days]. Should NOT be a free parameter.
:supersampling_factor: (*int*) Default 1 = no supersampling. Higher values compensate for morphological deformation at the cost of computational expense.
:occult_small_threshold: (*float*) If the moon radius (R_S/R_star) is smaller than this value, its occultation is approximated with constant limb darkening under its area. To obtain a precise estimate even for very small moons, set `occult_small_threshold` to a very small value (e.g., 1e-8)

.. note::

   Some text

Example:

::

   import pandora
   params = pandora.model_params()
   params.R_star = 696342
   params.u1 = 0.4
   params.u2 = 0.6

   # Planet parameters
   params.per_planet = 365.25
   params.a_planet = 217 * 696342
   params.r_planet = 63710
   params.b_planet = 0.0
   params.t0_planet = 100
   params.t0_planet_offset = 0.0
   params.M_planet = 5e27

   # Moon parameters
   params.r_moon = 18000
   params.a_moon = 2e6
   params.tau_moon = 0.25
   params.Omega_moon = 5
   params.i_moon = 85
   params.M_moon = 5e21

   # Other model parameters
   params.epochs = 1
   params.epoch_duration = 2.5
   params.cadences_per_day = 4800
   params.epoch_distance = 365.25
   params.supersampling_factor = 1


Evaluate model and obtain lightcurve
------------------------------------

.. class:: model.light_curve()

Parameters: None

Return parameters:

.. _returnvalues:

:time: (*array*) Timestamps of the model
:flux_total: (*array*) Lightcurve of planet and moon model
:flux_planet: (*array*) Only contributions by the planet
:flux_moon: (*array*)  Only contributions by the moon

Example:

::

   model = pandora.moon_model(params)
   time, flux_total, flux_planet, flux_moon = model.light_curve()


Evaluate model and obtain positions
-----------------------------------

.. class:: model.coordinates()

Parameters: None

Return parameters:

.. _returnvalues:

:time: (*array*) Timestamps of the model
:px_bary: (*array*) Planet X position at each timestamp
:py_bary: (*array*)  Planet Y position at each timestamp
:mx_bary: (*array*) Moon X position at each timestamp
:my_bary: (*array*) Moon Y position at each timestamp

Example:

::

   model = pandora.moon_model(params)
   time, px_bary, py_bary, mx_bary, my_bary = model.coordinates()


Evaluate model and obtain transit video
---------------------------------------

.. class:: model.transit_video()


Parameters:

:filename: (*string*) Path and filename of video to be written to disk
:codec: (*string*) Codec supported by matplotlib, such as `h264` or `gif`

.. _returnvalues:

(None)
The video is saved to disk directly to the given path and filename.

Example:

::

   model = pandora.moon_model(params)
   video = model.transit_video()
