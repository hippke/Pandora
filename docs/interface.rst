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
:per_planet: *(float)* Period [days]
:a_planet: *(float)* Semimajor axis [km]
:r_planet: *(float)* Radius [km]
:b_planet: *(0..1.x)* Impact parameter [km]
:t0_planet: *(float)* Time of inferior conjunction [days]
:t0_planet_offset: *(float)* Time difference to inferior conjunction [days] 
:M_planet: *(float)* Mass [kg]

Moon parameters
:r_moon: *(float)*  Radius [km]
:a_moon: *(float)*  Semimajor axis [km]
:tau_moon: *(float)*  Time of periapsis passage [0..1] normalized by period
:Omega_moon: *(float)* Longitude of the ascending node [0..360 deg]
:i_moon: *(float)* Orbit inclination [0..360 deg]
:M_moon: *(float)* Mass [kg]

Other model parameters
:epochs:
:epoch_duration:
:cadences_per_day:
:epoch_distance:
:supersampling_factor:

.. note::

   Some text
