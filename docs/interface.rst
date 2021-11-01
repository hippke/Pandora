Python Interface
================

This describes the Python interface of Pandora.


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
:b_planet: *(0..1.x)* Impact parameter [km]. Should be a free parameter.
:t0_planet: *(float)* Time of inferior conjunction [days]. Should not be a free parameter.
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

:epochs:
:epoch_duration:
:cadences_per_day:
:epoch_distance:
:supersampling_factor:

.. note::

   Some text
