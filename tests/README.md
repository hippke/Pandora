# Pandora tests

Regression and unit tests for Pandora 2.0 (`pandoramoon`). They run against the **installed** package, so they can
test a PyPI release (`pip install pandoramoon`) as well as a checkout (`pip install -e .`, or set
`PANDORA_PATH=/path/to/Pandora`).

```
python run_tests.py                # 12 light-curve cases + 44 unit tests, about 1 min, exit code 0 = pass
python run_tests.py -v             # with timings
python run_tests.py --lc           # light-curve cases only
python run_tests.py --unit         # unit tests only (same as python unit_tests.py)
python run_tests.py --sampling     # in addition the fit API and sampler tests (test_sampling.py, several minutes)
python test_sampling.py --quick    # sampler tests without the end-to-end fits
```

Needed: `numpy`, `numba`, `matplotlib` (video test), `tqdm`, `wotan` (only for one check in T12). `pytest` can collect
`unit_tests.py`, but three unit tests (`test_interior_interpolant`, `test_limb_interpolant`,
`test_occult_stretches_match_samples`) depend on the order in which the tests run. Use `python unit_tests.py`
(alphabetical order), which is what `run_tests.py` does.

## Files

| file | what it does |
|---|---|
| `cases.py` | the 12 light-curve cases (star, planet, moon, cadence) and their fingerprints |
| `reference.json` | stored fingerprints of the 12 cases (relative tolerance 1e-7) |
| `run_tests.py` | runs the cases against `reference.json`, the unit tests and optionally the sampler tests |
| `unit_tests.py` | bug-pinning unit tests against independent references (radial integration, disk intersection by integration, mpmath values) |
| `test_sampling.py` | tests of the fit API (`NestedSampler`, priors, moon-phase coordinates, `detect`) |
| `generate_reference.py` | regenerates `reference.json`. Run it only after an intended model change and after verifying it, never to make a test pass. |

## How the light-curve cases work

Each case defines a realistic system and observing set-up (Kepler long and short cadence, TESS, PLATO, M dwarf host,
eccentric moon, grazing planet, unphysical moon, supersampling). A case passes if (1) its definition is unchanged,
(2) the physical invariants hold (flux between 0 and 1, moon dip only during the transit, mutual eclipse present
where required) and (3) the fingerprint of the light curve (sums, moments, binned values, projections) agrees with
`reference.json` to 1e-7 relative. The fingerprints are unchanged since Pandora commit `430f8fc`. Pandora 2.0 passes
all 12.

## Changes for Pandora 2.0

* The pixel-raster eclipse (`eclipse_method="pixelmap"`, `numerical_grid`) was removed. The two pixelmap cases
  and the tests of the switch are gone. `test_pixelmap_removed` checks that the option is rejected.
* `moon_model.light_curve()` uses the exact path and no longer the float32 occultation cache (it differs from the
  cached path by at most a few ppm). The cached cases pass the cache to `pandora()` explicitly, so their
  fingerprints stay valid. `test_light_curve_exact` covers the new behaviour.
* `pandora()`, `pandora_flux()`, `pandora_deficit()`, `pandora_window()`, `pandora_loglike()` and `params_valid()` lost
  the `numerical_grid` argument. Callers that still pass it get an error and never a silently misaligned call.
* `test_interior_series` was removed with the interior series it tested (the interpolant tests cover the current code).
* The planet-only evidence of `NestedSampler` is computed by importance sampling by default
  (`planet_evidence="is"`). `test_sampling.py` runs the file and resume tests (T5, T6) with
  `planet_evidence="nested"` and compares the default with the nested-sampling result. The tests of posterior
  repartitioning (T11) were removed with the feature.
