Fitting: evidence and posterior
===============================

Pandora contains its own nested sampler, ``NestedSampler``. It fits a light curve with a planet+moon
model and with a planet-only model and returns both evidences and their ratio (the Bayes factor). It
also returns the posterior of every parameter and checks its own convergence. This page describes the
interface and the background needed to use it well.

.. contents::
   :local:
   :depth: 2


Quick start
-----------

::

   import numpy as np
   import pandoramoon as pandora
   from pandoramoon import NestedSampler, Uniform, LogUniform, Normal

   R_sun = 696_342_000  # [m]
   # time, flux, flux_err: your light curve (e.g. segments of +-1 d around each transit)
   like = pandora.Likelihood(time, flux, flux_err, epoch_distance=365.26, hill_sphere_threshold=1.0)

   priors = dict(
       q1=Uniform(0, 1), q2=Uniform(0, 1),                 # limb darkening (Kipping 2013)
       R_star=Uniform(0.4 * R_sun, 1.2 * R_sun),
       per_bary=Uniform(360.25, 370.25), a_bary=Uniform(210, 220), r_planet=Uniform(0.07, 0.11),
       b_bary=Uniform(0, 1), t0_bary=11.0, t0_bary_offset=Uniform(-0.05, 0.05),
       M_planet=Uniform(1.5e27, 2.5e27),
       r_moon=Uniform(0.01, 0.07), per_moon=Uniform(0, 50), tau_moon=Uniform(0, 1),
       Omega_moon=Uniform(0, 180), i_moon=Uniform(0, 180), M_moon=Uniform(1, 1e27),
   )
   sampler = NestedSampler(like, priors, log_dir="fit_out")
   results = sampler.run()
   sampler.print_results()
   sampler.plot_corner()

   print(results["logz"], results["logzerr"])        # planet+moon evidence ln Z
   print(results["planet"]["logz"])                   # planet-only evidence
   print(results["dlogz"], results["dlogzerr"])      # Bayes factor ln Z(moon) - ln Z(planet)
   samples = results["samples"]                      # equal-weight posterior, columns results["paramnames"]

Complete worked examples are the tutorials ``examples/tutorial_detection.ipynb`` and
``examples/tutorial_characterisation.ipynb``. On data like the tutorials' (3 transits, 30-min cadence, 500 ppm) the
planet+moon fit takes a few minutes on 16 cores (accuracy medium) and the planet-only fit seconds.

Typical run times for Kepler long-cadence targets (3–40 transits, ~400–1700 points in the fit windows; accuracy
medium; AMD Ryzen 9 7950X, 16 cores):

=====================================  ==========================================================
50 % of full planet+moon fits finish   within ~8–10 min
90 %                                   within ~30 min
longest (rare)                         ~1 h
compilation, once per process          ~20 s
planet-only fit                        ~1–5 s
``detect`` (first look)                1–5 s per light curve
=====================================  ==========================================================


Searching for moons: detection
------------------------------

A search has two stages: a fast **detection** test for every light curve, and the full **fit** (this page's
``NestedSampler``) for the candidates. Tutorials: ``examples/tutorial_detection.ipynb`` and
``examples/tutorial_characterisation.ipynb``.

::

   lc = pandora.prepare_lightcurve(time, flux, t0=t0, period=P, duration=T14)   # needs wotan
   like = pandora.Likelihood(lc["time"], lc["flux"], lc["flux_err"], epoch_distance=P)
   priors = pandora.default_priors(t0_bary=t0, per_bary=P, r_planet=0.1)
   det = pandora.detect(like, priors)            # seconds
   if det["candidate"]:
       results = pandora.NestedSampler(like, priors).run()

.. function:: prepare_lightcurve(time, flux, t0, period, duration, mask=1.0, window=4.0, method="biweight", fit_halfwidth=None, inflate_beta=True)

   Detrends the light curve with ``wotan`` (``method``, window ``window`` × T14) with the transits masked over
   ``mask`` × T14 (the defaults are the best setting of a study of 300 injected planet+moon systems in real Kepler
   light curves, robust to T14 errors of −20 … +25 %). Points without a trend are dropped. Keeps the fit
   windows (±(mask/2 + 1) × T14 around each transit, or ``fit_halfwidth`` days) and measures the noise on the
   detrended data 1–11 T14 beyond them: ``sigma`` (robust scatter) and the red-noise factor ``beta`` (on bins of
   T14/4; Pont, Zucker & Queloz 2006). Returns ``time, flux, flux_err`` (= σ × β) for the ``Likelihood``, and
   ``sigma, beta, epochs``, ``warnings`` and arrays for plots. If β > 2 it warns: the inflated errors lower a
   moon's Δln Z by about β², so weak moons around active or noisy stars may be missed (in our Kepler tests β = 3.8
   turned a clear moon, Δln Z 129 with white errors, into a non-detection).

.. function:: default_priors(t0_bary, per_bary, r_planet, per_bary_err=None, a_bary=None, R_star=R_sun, M_star=M_sun, M_planet=M_Jup, t0_halfwidth=0.05, per_moon_max=None)

   Priors for a moon search around a known planet: planet priors around the given values and wide moon priors
   (r_moon 0.1–0.7 × r_planet, per_moon up to the period at the Hill radius
   ≈ 0.19 × per_bary, all orientations, M_moon up to half the planet's mass). The Bayes factor depends on the moon
   priors; ``detect``'s threshold was calibrated with priors of this kind.

   **a_bary** (a/R⋆): by default log-uniform over 0.4–2.5 × the value from Kepler's law for the given star and a
   circular orbit. That allows for errors of the catalogued stellar density (a ∝ ρ^⅓) and for eccentric orbits,
   which change the transit duration by up to ~1.7×. A narrow prior around Kepler's law (± 2.3 %, the default before
   Pandora 2.0) is dangerous: if the transit's duration lies outside it, the planet-only model cannot fit the
   transit, and the moon model "repairs" it with a large or heavy moon. On Kepler-747 b (a/R⋆ 32.5 from the transit,
   54.6 from Kepler's law) this gave a false Bayes factor of ~1800.
   ``a_bary=(lo, hi)`` sets a uniform range directly; keep it wide. Do not centre it on a planet-only fit: a moon
   biases the planet-only a/R⋆ low (its dips lengthen the apparent transit), and the moon fit then runs into the
   prior's edge (tested: fewer moons recovered, prior-edge warnings in 15 % of moon systems).

   **Prior-edge warnings:** ``NestedSampler`` and ``detect`` warn when the posterior of a_bary, r_planet, per_bary or
   t0_bary_offset piles up at an edge of its prior (> 10 % of the mass in the outer 2 % of the range), and when the
   moon's r_moon or M_moon sits at the upper edge of its prior. A moon at its size or mass limit typically repairs a
   misfit of the planet-only model (a wrong prior, or changes of the transit's timing or shape between epochs), not a
   real moon.

.. function:: detect(likelihood, priors, recall=0.95, threshold=3.0, seed=1, jack_min=18, ld_correlation=0.0, verbose=False, threads=None)

   A first look: one short nested-sampling run of the planet-only model and a few of the planet+moon model
   (seconds). ``recall`` (0.90, 0.95, 0.99, 0.999) is the fraction of clearly detectable moons the
   settings are calibrated to find (0.35 / 0.6 / 1.6 / 3.1 CPU-s per light curve in the calibration). The runs are
   independent and run in parallel threads (``threads=None``: one per run, at most the number of cores; the result
   does not depend on it); use ``threads=1`` when you already run one ``detect`` per core.
   Returns a dict with ``dlnz`` (noisy, ±0.5), ``interesting`` (``dlnz > threshold``), ``jackknife``,
   ``candidate`` (interesting and the jackknife passed), ``moon`` (rough median and 68 % range of the moon's
   period, radius, mass ratio, inclination), ``best`` (best-fit parameters of both models), ``warnings``.

   The **epoch jackknife** (``pandora.epoch_jackknife``) splits the χ² improvement of the moon model over the
   planet-only model by transit. A moon must be seen in at least two transits: the improvement without the
   strongest transit must be ≥ ``jack_min`` (18 ≈ 2 × (threshold 3 + Occam penalty ~6)). Single-transit
   artefacts were the cause of all false moon signals in our real-data tests. On real Kepler light curves
   (injected moons and no-moon controls, with β-inflated errors from ``prepare_lightcurve``): moons reached
   ≥ 32 (24 of 25), controls and planet-only systems ≤ 26 (70); "Δln Z > 3 and jackknife ≥ 18" gave 0 false
   candidates in 50 controls, Δln Z alone 2 (both on one star).

Interface
---------

.. class:: NestedSampler(likelihood, priors, model="moon", compare_planet=True, moon_phase="auto", ld_correlation=0.0, log_dir=None, resume="subfolder", threads=None, seed=1, verbose=True, supersampling="auto", planet_evidence="is")

   :likelihood: A :class:`pandoramoon.Likelihood`. It holds the data, the errors and Pandora's model
      settings (``supersampling_factor``, ``hill_sphere_threshold``, ``occult_small_threshold``,
      ``epoch_distance``). The occultation ``cache`` is not supported (use ``cache=None``).
   :priors: *(dict)* For each model parameter, either a prior (:class:`Uniform`, :class:`LogUniform`,
      :class:`Normal`) or a number, which fixes the parameter. The names and units are those of
      Pandora's model (see :doc:`Interface`):
      ``R_star, per_bary, a_bary, r_planet, b_bary, w_bary, ecc_bary, t0_bary, t0_bary_offset,
      M_planet, r_moon, per_moon, tau_moon, Omega_moon, i_moon, ecc_moon, w_moon, M_moon``, plus the
      limb darkening as either ``q1, q2`` (Kipping 2013) or ``u1, u2``.
      ``w_bary``, ``ecc_bary``, ``ecc_moon`` and ``w_moon`` default to ``0`` (circular orbits).
      For ``model="planet"`` the moon parameters are not needed. Neither are ``R_star`` and
      ``M_planet``, which act only through the moon's orbit.
   :model: *(str)* ``"moon"`` (planet + moon, default) or ``"planet"``.
   :compare_planet: *(bool)* For ``model="moon"``: also fit the planet-only model with the same priors
      for the shared parameters, and return the Bayes factor. Default ``True``.
   :moon_phase: ``"auto"`` (default), ``True`` or ``False``. Sample the moon's period and phase in the
      phase-at-transit coordinates (see `Moon orbit: phase-at-transit coordinates`_). With ``"auto"``
      they are used whenever the priors allow it. ``True`` raises an error if they cannot be used.
      ``False`` samples ``per_moon`` and ``tau_moon`` directly (not recommended for moons; see below).
   :ld_correlation: *(float)* Correlation coefficient between :class:`Normal` priors on ``u1`` and
      ``u2`` (e.g. limb darkening from stellar models). Default ``0``.
   :log_dir: *(str)* Output folder (see `Files`_). ``None``: no files.
   :resume: ``"subfolder"`` (default, as in UltraNest: a new folder ``run1``, ``run2``, … inside
      *log_dir* for every new sampler), ``"resume"`` (continue or reuse the fit stored in *log_dir*)
      or ``"overwrite"``.
   :threads: *(int)* Number of runs in parallel. Default: all cores available to the process.
   :seed: *(int)* Random seed. A fit is reproducible for a given seed and number of threads, with the same
      compiled code: numba's code compiled fresh (first use after an update) and loaded from its cache can differ in
      the last bit of a likelihood value (fastmath), and nested sampling then takes another path to a statistically
      equivalent result. Model light curves *simulated* in two processes can differ in the same way; data read
      from files are identical (resuming a fit of real data is not affected).
   :verbose: *(bool)* Print the progress of the checks.
   :planet_evidence: How the planet-only model is fitted. ``"is"`` (default): one short nested-sampling
      pilot (4 runs of 100 live points, in parallel) gives the posterior's shape, and importance sampling
      (Student-t kernels around pilot points plus a wide Student-t; 50,000 or more likelihood calls,
      evaluated in parallel) gives ln Z and the posterior. The planet-only posterior is single-peaked, which
      this needs. It checks itself (effective sample size, the largest weight, agreement with the pilot's ln Z)
      and falls back to the full nested-sampling fit if a check fails. On a 16-core Ryzen 7950X it takes
      ~1 s instead of 4–15 s, with ln Z errors of 0.01–0.02 (20 seeds: scatter equal to the quoted error;
      agrees with the full high-accuracy fit). ``"nested"``: the full nested-sampling fit as for the moon
      model (needed to test the run store / resume of a planet-only fit).
   :supersampling: ``"auto"`` (default), an integer, or ``None``. ``"auto"``: if the likelihood is not
      supersampled and the cadence is ≥ 10 min, the fit computes each data point as the average of 5
      sub-exposures (exposure = cadence). Without it, 30-min data bias the Bayes factor and the moon's
      parameters. An integer: that many sub-exposures. ``None``: use the likelihood as given. The likelihood the fit
      actually uses is ``sampler.likelihood`` (e.g. ``sampler.likelihood(**sampler.parameters())``
      gives the ``logl`` of ``results["maxlikelihood"]``).

   .. method:: run(accuracy=None, max_weighted_samples=None, progress=None, **kwargs)

      Runs the fit (for ``model="moon"``: first the planet-only fit, then the planet+moon fit) and
      returns the results dictionary (also stored as ``sampler.results``).

      :accuracy: ``"medium"`` (default since 2.0; ``None`` reads ``settings.cfg`` ``[fit] default_accuracy``):
         statistical error of ln Z ≤ 0.2, posterior reproducible to a KS distance ≤ 0.08, 500 live points
         per run, at least 8 runs. ``"high"``: ≤ 0.1 and ≤ 0.05, 1000 live points per run, at least 16
         runs; about 2–5× slower. Use ``"medium"`` to follow up
         detections; for evidence comparisons at the 0.1–0.3 nat level use ``"high"`` (on our test cases medium
         fits deviated by up to 0.5 nats, 2.5 × their quoted error, from the high-accuracy values).
      :max_weighted_samples: Thin ``weighted_samples`` (all points of the merged runs, up to ~10⁶) to about
         this many points. Unbiased Horvitz–Thompson thinning: heavy points are kept, light ones kept at
         random with their weight scaled up. ``None`` (default): keep all.
      :progress: A status line while the fit runs: the model, runs done (and the minimum / maximum),
         runs in progress, live points and chain length, the current ln Z and its error, likelihood calls
         so far and per second, and the elapsed time, e.g. ``moon model: 8 runs done (min 8, max 96),
         4 running | n_live 500, nsteps 64 | lnZ -194.77 +- 0.18 (target 0.2) | 107,153,206 calls,
         683,412 calls/s | 0:02:34``. In a terminal or Jupyter the line is updated in place every 2 s; in a
         log file a new line is written every 30 s. ``None`` (default): on if ``verbose``; ``True`` /
         ``False``; or the interval in seconds. It does not change the result.
      :kwargs: Passed to :func:`pandoramoon.nss.fit`, e.g. ``nlive``, ``lz_tol``, ``ks_tol``,
         ``min_runs``, ``max_runs`` (default 96), ``nsteps`` (initial chain length, default 4 × the
         number of sampled coordinates).

   .. method:: print_results()

      Prints ln Z ± error, whether the checks passed, and median, 68 % interval, mean and standard
      deviation of every parameter (for ``model="moon"`` also the planet-only fit and the Bayes factor).

   .. method:: plot_run(model=None, filename=None)

      Diagnostic plots of the fit (saved as ``plots/run.pdf``): ln Z of every run with the merged value, the
      history of the convergence checks (ln Z ± error and chain length vs the number of runs), and the
      posterior weights of the merged points.

   .. method:: parameters(point=None, model=None)

      All 20 Pandora model parameters (a dict with Pandora's names; limb darkening as ``u1, u2``) for a
      point of the free parameters, by default the maximum-likelihood point of the fit. With
      ``model="planet"``: the planet-only fit, with Pandora's planet-only moon values. Example, the
      best-fit light curve::

         for k, v in sampler.parameters().items():
             setattr(params, k, v)          # params: a pandora.model_params() with the time-grid settings
         flux_total, flux_planet, flux_moon = pandora.moon_model(params).light_curve(time)

   .. method:: plot_corner(model=None, filename=None, **kwargs)

      Corner plot of the posterior (needs the ``corner`` package). With a *log_dir*, it is saved as
      ``plots/corner.pdf``. ``model="planet"`` plots the planet-only fit.


Priors
~~~~~~

.. class:: Uniform(lo, hi)

   Flat on [*lo*, *hi*].

.. class:: LogUniform(lo, hi)

   Flat in ln *x* on [*lo*, *hi*], 0 < *lo* < *hi*. Typical for scale parameters spanning orders of
   magnitude (e.g. moon mass).

.. class:: Normal(mean, sigma, lo=-inf, hi=inf)

   Gaussian, optionally truncated to [*lo*, *hi*] and renormalised. Typical for parameters measured
   elsewhere (stellar radius, limb darkening, planet period).

Every prior is applied exactly. The sampler draws in a unit cube, and the inverse cumulative
distribution function of each prior maps the cube to the parameter. The evidence is therefore that of
your prior.

**Priors reaching outside Pandora's valid range** (e.g. a ``Normal`` on ``r_planet`` that extends below 0,
on ``q1`` beyond [0, 1], or on an eccentricity outside [0, 1)) are truncated to the valid range and
renormalised, with a warning. Their mass outside would otherwise be lost silently (the likelihood is
−∞ there), which would lower the evidence. The truncated fractions are reported in
``results["priors_truncated"]``.

**Limb darkening.**

- ``q1, q2``: Kipping's parametrisation. ``q1, q2 ~ Uniform(0, 1)`` is the same as a flat prior over all
  physical quadratic laws.
- ``u1, u2``: any of the priors above, e.g. ``u1=Normal(0.41, 0.05), u2=Normal(0.26, 0.05)``. The prior
  is truncated to Pandora's physical region (u1 ≥ 0, u1 + u2 ≤ 1, u1 + 2 u2 ≥ 0) and renormalised, so
  the evidence stays correct. ``ld_correlation`` correlates two untruncated Normal priors.

**Bayes factors depend on the priors.** The evidence is the likelihood averaged over the prior, so wider
priors lower it. Comparing planet+moon with planet-only is meaningful only with the same priors on the
shared parameters. ``NestedSampler`` guarantees this by building both models from one prior dictionary.
For example, on the notebook's data the Bayes factor is 131.6 with the planet box a_bary ∈ [210, 220],
r_planet ∈ [0.07, 0.11]. A planet-only fit with the notebook's wide planet priors (a_bary ∈ [1, 301],
r_planet ∈ [0.001, 0.201]) has an evidence 1.2 lower.


Results
~~~~~~~

The results dictionary uses UltraNest's keys:

:logz, logzerr: ln Z and its statistical error (the scatter of ln Z between the independent runs,
   divided by √(number of runs)).
:logzerr_bs: Bootstrap error over the runs (agrees with ``logzerr``).
:logz_single: Mean ln Z of the single runs.
:paramnames: Names of the free parameters (the columns of the samples).
:samples: Equal-weight posterior samples, shape (n, number of free parameters).
:weighted_samples: Dictionary of all points of the merged runs: ``points`` (parameters),
   ``upoints`` (sampled unit-cube coordinates), ``weights`` (posterior weights, sum 1), ``logl``
   (log-likelihood).
:posterior: Dictionary of lists per parameter: ``mean``, ``stdev``, ``median``, ``errlo``, ``errup``
   (15.9 % and 84.1 % quantiles).
:maxlikelihood: The point with the highest likelihood found: ``logl``, ``point``,
   ``point_untransformed``.
:ess: Effective sample size of the posterior.
:H: Information gain from the prior to the posterior [nats].
:ncall, niter: Likelihood calls (including the check runs) and nested-sampling iterations.
:insertion_order_test: Insertion-index test (Fowlie, Handley & Su 2020), pooled over runs.
:passed: ``True`` if all convergence checks passed (see `Self-checks`_). If ``False``, the result is
   usable but its error is not guaranteed; ``status`` and ``reason`` say why.
:status, reason: ``"passed"``; ``"stalled"``: a check asked for longer chains or more live points but both
   were at their limit (or longer chains were ruled out by ``multimodal``), and the fit stopped once the ln Z
   error was within the tolerance; ``"run_cap"``: ``max_runs`` reached. ``reason`` lists the checks that were not
   met and by how much (e.g. ``split-half KS 0.062 > ks_tol 0.05``).
:nsteps, runs, nlive, n_escalations, history: Final chain length, number of merged runs, live points
   per run, number of chain-length doublings, and the history of the checks.
:multimodal, n_nlive_escalations: ``multimodal`` is ``True`` if the runs disagreed the way runs do that
   miss a mode of the posterior (a few runs with a much higher ln Z, or a scatter that longer chains do
   not reduce). The fit then doubles the live points per run (``n_nlive_escalations`` times, up to
   ``max_nlive``, default 4 × ``nlive``). If it is still ``True`` at the end, ln Z may be too low:
   check with a larger ``nlive`` and look at ``run_logz`` (ln Z of every run).

For ``model="moon"`` with the planet comparison, the dictionary also has:

:jackknife: The epoch jackknife (see `Searching for moons: detection`_): the fit improvement of the moon
   model per transit and whether the improvement without the strongest transit reaches 18 (``passed``).
:warnings: Notes on the fit (supersampling used, checks not passed, one transit dominating, runs that
   disagree). ``print_results()`` prints them.

:planet: The planet-only results (the same keys).
:dlogz, dlogzerr: The Bayes factor ln Z(moon) − ln Z(planet) and its error.
:bayes_factor: ``text`` (e.g. ``e^3.87 = 47.7``), ``favoured`` (``"moon"`` or ``"planet-only"``), ``jeffreys``
   (the category on Jeffreys' scale), ``ln_b`` (\|ΔlnZ\|). ``print_results()`` prints it as a sentence. Read
   `Interpreting the Bayes factor`_ before quoting the category.
:logz_excluded_correction: The (tiny) exact correction for the short-period part of the moon-period
   prior (see `Moon orbit: phase-at-transit coordinates`_).

A rough guide to Bayes factors (Kass & Raftery 1995): ΔlnZ > 5 is strong evidence for a moon.
Calibration on injected white-noise light curves: light curves without a moon gave ΔlnZ between about −6
and +2. The moon model's Occam penalty makes negative values the norm.


Files
~~~~~

With *log_dir*, the layout follows UltraNest:

====================================  ======================================================================
``info/results.json``                 all results except the samples (plus the priors)
``info/post_summary.csv``             mean, stdev, median, errlo, errup of each parameter (one row)
``chains/equal_weighted_post.txt``    equal-weight posterior samples, header = parameter names
``chains/weighted_post.npz``          weighted samples (``points``, ``upoints``, ``weights``, ``logl``;
                                      npz instead of text, because merged fits have up to ~10⁶ points)
``plots/corner.pdf``                  from ``plot_corner()``
``runs/``                             every finished nested-sampling run (for resume; ~3 MB per run)
``info/fit_state.json``               current chain length and a hash of data, priors and settings
``planet/``                           the planet-only fit, same layout
====================================  ======================================================================

**Resume.** A new ``NestedSampler`` with ``resume="resume"`` and the same *log_dir* reloads the stored
runs, evaluates the checks, and continues with new runs if needed. A finished fit is returned at once.
Runs that were in flight at an interruption are redone with their original seeds, so a resumed fit
usually gives the same result as an uninterrupted one. It is not guaranteed: which check arm a new run
joins depends on the timing.
If the data, priors or settings differ from the stored fit, an error is raised.


Background
----------

Nested sampling
~~~~~~~~~~~~~~~

The evidence Z = ∫ L(θ) π(θ) dθ is the likelihood averaged over the prior. Nested sampling (Skilling
2006) computes it, and the posterior along the way, as follows:

- It keeps *n* "live" points drawn from the prior.
- It repeatedly removes the point with the lowest likelihood L* and replaces it by a new prior point
  with L > L*.
- After *i* replacements the prior volume with L > L* has shrunk to about X_i = exp(−i/n).
- Z = Σ L_i (X_{i−1} − X_i).
- The removed points, weighted by L_i (X_{i−1} − X_i) / Z, are the posterior samples.

The hard part is drawing the new point from the prior restricted to L > L*. For Pandora's 15–17
parameters this is done with Markov chains.

The sampler: ``pandoramoon.nss``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nss`` (nested slice sampler) is written for very cheap likelihoods. A Pandora likelihood call takes
2–10 µs, so a sampler with Python code per call (tens of µs) would spend most of its time on itself.
Everything in one run is compiled with numba: there is no Python per likelihood call and none per
iteration. The sampler's own work costs about 0.25 µs per call, against 2–3 µs for the likelihood.

**One run.**

- n live points (1000 for ``accuracy="high"``, 500 for ``"medium"``) in the unit cube, drawn only where
  the prior is nonzero (rejection); the run starts at the volume fraction of that region (exact; regions
  outside the prior, e.g. limb darkening outside the physical triangle, otherwise bias ln Z).
- A new point comes from a slice-sampling chain of ``nsteps`` moves at the fixed threshold L*, started
  at a random surviving live point (Neal 2003):

  - pick a direction;
  - place an interval around the point at random;
  - step it out until both ends are below L*;
  - shrink it until a point above L* is found.

  The interval width adapts so that it needs about one step-out per move.
- **Directions ("pairs"):** the difference of a random live point and one of its four nearest live
  neighbours (in coordinates scaled by the live points' spread). This gives the local scale and
  orientation of the constrained region, also for strongly correlated and curved posteriors. The
  direction never depends on the chain's own position, so every move leaves the constrained prior
  invariant (an exact sampler). The neighbour lists are refreshed every n/4 replacements.
- The run stops when the remaining live points can add less than 1 % to Z.

**Many runs, merged.** ``nss.fit``:

- runs many such runs independently, one per core, without synchronisation (near 100 % parallel
  efficiency);
- merges the runs as one run with the summed number of live points (Higson et al. 2018). The merged
  ln Z is about the mean of the runs' ln Z and unbiased in known-evidence tests;
- takes the error of ln Z from the scatter between the runs. On Pandora posteriors this is 1.2–2.3× the
  textbook error sqrt(H/n) of a single run, because of incomplete mixing; the empirical error covers
  this;
- excludes a run that never found the best mode (much lower maximum likelihood and ln Z) and counts it.
  More than 10 % such runs fail the fit.

Self-checks
~~~~~~~~~~~

Slice chains that are too short leave new points correlated with their start points, and this biases
ln Z upwards. Neither the scatter between runs nor the insertion-index test detects it. ``nss.fit``
therefore runs two arms:

- the result arm with chains of ``nsteps`` moves;
- a check arm with ``nsteps / 2``, at about one third of the cost.

It adds runs until all of these hold:

- the error of ln Z ≤ ``lz_tol`` (0.1 for ``"high"``, 0.2 for ``"medium"``);
- the posterior is reproducible: the largest KS distance over all parameters between the merged
  posteriors of the even- and odd-numbered runs is ≤ ``ks_tol`` (0.05 / 0.08);
- the chain length is sufficient: ln Z of the two arms agrees within 2 ``lz_tol`` (including its
  error), and their posteriors within ``ks_tol``;
- at least ``min_runs`` runs (16 / 8) and at most 10 % failed runs.

The two arms always differ a little (by noise, and by small real differences well below the
tolerances), so the check judges the *size* of their difference against the tolerances: the KS distance
between the arms is corrected for its noise (measured by randomly re-splitting the runs of both arms), and
the ln Z difference must exceed the tolerance by a clear margin.

Four rules decide (all constants in ``pandoramoon/settings.cfg``, section ``[fit_rules]``, available as ``pandoramoon.nss.FIT_RULES``):

1. **Done?** The conditions above hold: stop.
2. **Chains long enough?** If the two arms disagree by more than the tolerances, ``nsteps`` is doubled: the
   old result arm becomes the check arm and sampling continues.
3. **Excess scatter?** If ln Z scatters between the runs by more than 2.75 times the single-run error, the
   runs do not sample alike. If the shorter arm scatters clearly more, longer chains help (``nsteps``
   doubled); otherwise the number of live points per run is doubled.
4. **Did some runs find a better solution?** If a few runs have a clearly higher ln Z and likelihood than the
   others, the others missed part of the posterior: the number of live points is doubled (up to
   ``max_nlive``), and the result is flagged ``multimodal``. Chain lengths are then not increased any further;
   rule 3 doubles the live points instead.

The checks are evaluated after every finished run, as soon as a decision is possible. If a rule asks for an
escalation that is no longer possible (live points at ``max_nlive``, chains at ``max_nsteps``, or longer chains
ruled out by ``multimodal``), more runs would only shrink the error bar: the fit stops as soon as the ln Z error
is within the tolerance (``passed=False``, ``status="stalled"``). Otherwise a fit that does not pass within
``max_runs`` runs is returned with ``passed=False``, ``status="run_cap"``.

**What the error means.** ``logzerr`` is the statistical error of this fit; ten C0 fits with different
seeds scattered by 0.06, against a quoted 0.07. Different sampler settings (live points, chain length)
can shift ln Z by up to ~0.1. Add this as a systematic error when comparing evidences at the 0.1-nat
level.

Moon orbit: phase-at-transit coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The problem.** A moon is observed only during the planet's transits, once per planetary orbit. Many
moon periods P_m therefore put the moon at nearly the same phases in every transit. In the native
parameters (per_moon, tau_moon) the posterior is a comb of 50–80 very narrow, separated modes, spaced
by about P_m² / P_b in period. No general-purpose sampler mixes between them reliably: evidences
scattered by 1–3 nats between runs and posteriors were not reproducible.

**The coordinates.** ``NestedSampler`` samples instead:

- φ_c: the moon's orbital phase at a reference transit e_c (the mean of the transits in the data);
- δ: the fractional phase advance per planetary orbit;
- m: the integer number of moon orbits per planetary orbit (sampled through a smooth step map u_m → m).

These satisfy P_m = P_b / (m + δ) and φ_e = φ_c + (e − e_c) δ for transit e.

- The moon's phase at every transit depends only on (φ_c, δ). The comb collapses into one broad
  solution in (φ_c, δ) and a smooth sequence in m.
- The map from (φ_c, δ, m) to (per_moon, tau_moon) is a bijection.
- An exact importance weight makes the prior, the evidence and the posterior identical to those of the
  native parameters with your prior.
- Any per_moon prior (Uniform, LogUniform, Normal) is applied as an exact density ratio.
- Eccentric moons work the same way (tau_moon is a mean-anomaly offset in Pandora).
- Results are always reported in the native parameters.

Conditions (``moon_phase="auto"`` checks them):

- the per_moon prior has a finite upper bound ≤ the shortest planet period in the per_bary prior;
- tau_moon ~ Uniform(0, 1).

**Ω double cover.** With Omega_moon ~ Uniform(lo, lo + 180), the node is sampled on [lo, lo + 360). The
orbit at Ω + 180° with the phase shifted by half a turn is the same orbit, so the prior is unchanged.
The sampler no longer sees a boundary at lo and lo + 180. Samples are reported folded back to
[lo, lo + 180).

**Short periods.** A per_moon prior that starts at 0 includes periods so short that the moon would be
inside the planet. Pandora treats those moons as unphysical and returns the planet-only light curve.
Periods below 0.6 × the shortest physical period of the prior box are not sampled. Their prior mass F
(10⁻³ for the notebook's priors) contributes exactly F · Z_planet, which is added back after the
planet-only fit.

Compilation
~~~~~~~~~~~

The likelihood that the sampler sees is a compiled function of the unit-cube point and a tuple holding
the data and all prior constants. It is compiled once per Python process (about 15 s) and reused for
every model, data set and prior, so later fits in the same process start immediately.


Validation
----------

The sampler and the interface were validated on the server used for the development (16 cores),
see ``SAMPLING.md`` and ``NSS_SAMPLER.md`` in the project directory. In summary:

- **Known evidences:** Gaussian mixtures and a 16-dimensional problem with bananas, prior-edge
  truncation and strong correlations. Unbiased within ±0.05 once the chains are long enough. The
  chain-length check bounds the remaining bias to ~0.1 nats.
- **A suite of 12 injected data sets:** more or fewer transits, noise from 100 to 2000 ppm, 2-min
  cadence, no moon, a long-period inclined moon, a missing transit, a Kepler-like planet, and the
  Pandora paper's system. Every case passes its checks by itself, and the best fit lies above the
  injected truth in every case. All 12 together take 55 minutes on 16 cores.
- **Simulation-based calibration:** 263 planet+moon fits and 256 planet-only fits of injections drawn
  from the prior (``accuracy="medium"``). The rank of the true value in its posterior is uniform for
  all 15 resp. 7 parameters, so the posteriors have the right width and location.
- **The interface itself:**

  - the compiled likelihood equals ``Likelihood`` to 10⁻¹⁴;
  - prior transforms, the moon-phase weights and the limb-darkening prior reproduce their target
    distributions;
  - a fit resumed after a crash continues; in the test it gives the same result as an uninterrupted fit;
  - evidences agree with the earlier scripts (notebook data: −185.975 ± 0.018 over five seeds vs
    −185.95 ± 0.02) and with an independent prediction for an informative limb-darkening prior.
  - Tests: ``test_cases/test_sampling.py``.


Interpreting the Bayes factor
-----------------------------

``print_results()`` gives the Bayes factor B = exp(ΔlnZ), ΔlnZ = ln Z(moon) − ln Z(planet only), and its category on
Jeffreys' scale, e.g.::

   Bayes factor: dlogZ = logZ(moon) - logZ(planet) = 3.865 +- 0.134
   Bayes factor: e^3.87 = 47.7 in favor of the moon model; Jeffreys' scale: "very strong" evidence (...)

The common verbal scales, converted to ΔlnZ (Pandora uses the first):

==============================  ==========================  ==============================  ==============
ΔlnZ (Bayes factor)             Jeffreys (1961)             Kass & Raftery (1995)           Trotta (2008)
==============================  ==========================  ==============================  ==============
0 – 1.15 (1 – 3.2)              barely worth mentioning     bare mention (to 1), positive   inconclusive (to 1), weak
1.15 – 2.3 (3.2 – 10)           substantial                 positive                        weak (to 2.5)
2.3 – 3.45 (10 – 32)            strong                      positive (to 3), strong         moderate (from 2.5)
3.45 – 4.6 (32 – 100)           very strong                 strong                          moderate
> 4.6 (> 100)                   decisive                    strong (to 5), very strong      moderate (to 5), strong
==============================  ==========================  ==============================  ==============

Negative ΔlnZ favours the planet-only model on the same scale. **These words describe the evidence only if the
models describe the data apart from white Gaussian noise with the given errors.** For moon searches in real light
curves, use them with these caveats:

- **Unmodelled effects inflate ΔlnZ.** The moon model is flexible: through the planet's motion around the
  barycentre and the moon's own dips it can absorb anything the planet-only model cannot fit. Examples from our
  tests on Kepler light curves:

  - a slow drift of the impact parameter between transits (orbital precession): ΔlnZ ≈ +35 with no moon;
  - a prior on a/R⋆ that excludes the transit's duration: ΔlnZ ≈ +1800 (fixed in ``default_priors``);
  - stellar variability left after detrending on an active star: up to ΔlnZ ≈ +190 in no-moon controls.

  Signs of this: a moon at the edge of its prior (Pandora warns), a fit improvement that does not survive the
  epoch jackknife, a result that changes with the detrending, timing or duration changes of the planet's transits.

- **On real data the relevant yardstick is empirical**: how often the same star produces such a ΔlnZ without a moon.
  Inject planet-only transits at other phases of the same light curve (after removing the real transits), and moon
  signals, process them exactly like the real data (detrending, priors, ``detect`` or the full fit), and compare.
  A ΔlnZ that the star's own controls also reach is no evidence, whatever the scale says.

- **ΔlnZ depends on the moon priors.** The Bayes factor contains an Occam penalty set by the volume of the moon priors
  (radius, period, mass, orientation). Wider priors lower it, narrower ones raise it. Quote it together with the
  priors (e.g. ``default_priors``).

- **The real uncertainty is larger than ``dlogzerr``.** ``dlogzerr`` is the statistical error. Add the sampler's
  systematic (up to +0.13 nats from the default chain length, on the high side; see `Limitations`_); at
  ``accuracy="medium"`` fits deviated by up to ~0.5 nats from high-accuracy fits on our test cases. Detrending choices
  can move ΔlnZ by much more.

- **The Bayes factor updates prior odds.** It multiplies the prior odds that this planet has a detectable moon, which
  are small. A Bayes factor of 50 on small prior odds can still leave modest posterior odds for a moon.

- ``detect``'s threshold of 3 was calibrated on white-noise injections and real Kepler controls (with the epoch
  jackknife); it is a screening threshold, not a statement of evidence.

Limitations
-----------

- **Evidence systematic of the sampler:** against long-chain reference fits, the default chain length (64
  steps) biases ln Z by −0.02 … +0.13 nats on our test cases (high side). Treat ±0.13 as a systematic when
  comparing evidences at the 0.1-nat level; it does not matter for detection (threshold 3).
- The noise model is white Gaussian noise with the given errors. There is no jitter term and no
  correlated noise. ``prepare_lightcurve`` inflates the errors by the red-noise factor β, which corrects
  the Bayes factor to first order; artefacts confined to one transit are caught by the epoch jackknife.
- **Long-cadence data need supersampling.** ``NestedSampler(supersampling="auto")`` (default) uses 5
  sub-exposures per data point when the cadence is ≥ 10 min (exposure = cadence); pass an already
  supersampled ``Likelihood`` (``supersampling_factor`` and a supersampled time grid, as for ``pandora()``)
  for other exposure times. On injected Kepler-like 30-min data (exposure smearing
  simulated at a factor of 30), fits without supersampling biased the limb darkening and transit shape
  by up to 1.5–2.5 standard deviations and overestimated the Bayes factor by 1.7 (3 transits, 500 ppm)
  to 12.6 nats (7 transits, 300 ppm). Use ``supersampling_factor=5`` for 30-min cadence (posterior
  shifts ≤ 0.1 sigma, Bayes factor within 0.4) and 7 for evidence comparisons at high signal-to-noise. Each
  call then costs about 2× (5) or 2.4× (7) as much. The Bayes-factor inflation without supersampling grows
  with the planet's signal-to-noise, not the moon's. For light curves without a moon it may therefore
  produce false detections near the usual thresholds (not yet tested).
- The chain-length check compares ``nsteps`` with ``nsteps / 2``. It detects a chain-length bias that
  changes by more than about 2 ``lz_tol`` per doubling, but not one that stays nearly constant between
  ``nsteps / 2`` and ``nsteps`` before it falls. Such a plateau was found in a 16-dimensional test problem
  with two strongly curved (banana) degeneracies (bias +0.26 nats at 64 steps, undetected). On Pandora's
  test data, ln Z does not depend on the chain length between 32 and 512 steps, and simulation-based
  calibration is passed. For other, strongly curved posteriors, start with longer chains (``nsteps=``).
- Very narrow per_moon priors are handled through a density ratio. They are correct but sample less
  efficiently.
- One process at a time should use the cores (``threads``): fits are already parallel.


References
----------

- Skilling 2006, Bayesian Analysis 1, 833 (nested sampling)
- Neal 2003, Annals of Statistics 31, 705 (slice sampling)
- Higson, Handley, Hobson & Lasenby 2019, Statistics and Computing, arXiv:1704.03459 (merging runs, errors)
- Fowlie, Handley & Su 2020, MNRAS 497, 5256 (insertion-index test)
- Karamanis & Beutler 2021, Statistics and Computing, arXiv:2002.06212 (ensemble slice sampling)
- Kipping 2013, MNRAS 435, 2152 (limb-darkening parametrisation)
- Kass & Raftery 1995, JASA 90, 773 (Bayes factors)
- Talts et al. 2018, arXiv:1804.06788 (simulation-based calibration)
- Buchner 2021, JOSS 6, 3001 (UltraNest; result conventions)
- Hippke & Heller 2022, A&A, arXiv:2205.09410 (Pandora)
