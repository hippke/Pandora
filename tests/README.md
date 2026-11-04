# Pandora tests

Regression, unit and sampler tests of Pandora 2.0 (`pandoramoon`). They run against the **installed** package, so
they test a PyPI release (`pip install pandoramoon`) as well as a checkout (`pip install -e ..`, or
`PANDORA_PATH=/path/to/Pandora`).

```
python run_tests.py                # 12 light-curve cases + 45 unit tests, about 1 min, exit code 0 = pass
python run_tests.py -v             # with timings
python run_tests.py --lc           # light-curve cases only
python run_tests.py --unit         # unit tests only (same as python unit_tests.py)
python run_tests.py --sampling     # in addition the fit API and sampler tests (test_sampling.py, several minutes)
python test_sampling.py --quick    # sampler tests without the end-to-end fits
python check_sensitivity.py        # do the tests catch deliberate code mutations? (needs a git checkout, ~3 min)
```

Needed: `numpy`, `numba`, `matplotlib` (video test), `tqdm`, `wotan` (one check in T12). Use `python unit_tests.py` or
`run_tests.py` rather than `pytest`: three unit tests (`test_interior_interpolant`, `test_limb_interpolant`,
`test_occult_stretches_match_samples`) depend on the order in which the tests run, and pytest uses another order.

## Files

| file | what it does |
|---|---|
| `cases.py` | the 12 light-curve cases (star, planet, moon, cadence) and their fingerprints |
| `reference.json` | stored fingerprints of the 12 cases (relative tolerance 1e-7) |
| `run_tests.py` | runs the cases against `reference.json`, the unit tests and optionally the sampler tests |
| `unit_tests.py` | 45 bug-pinning unit tests against independent references (radial integration, disk intersection by integration, mpmath values) |
| `test_sampling.py` | tests of the fit API (`NestedSampler`, priors, moon-phase coordinates, importance-sampled planet evidence, `detect`) |
| `check_sensitivity.py` | meta test: applies code mutations (time shift, wrong moon phase, eclipses off, ...) and reports which cases and unit tests catch them |
| `generate_reference.py` | regenerates `reference.json`. Run it only after an intended, verified model change, never to make a test pass. |
| `HISTORY.md` | the development log of these tests (written before 2.0) |

## How the light-curve cases work

Each case defines a realistic system and observing set-up (Kepler long and short cadence, TESS, PLATO, M dwarf host,
eccentric moon, grazing planet, unphysical moon, supersampling). A case passes if (1) its definition is unchanged,
(2) the physical invariants hold (flux between 0 and 1, moon dip only during the transit, mutual eclipse present
where required) and (3) the fingerprint of the light curve (sums, moments, binned values, projections) agrees with
`reference.json` to 1e-7 relative. The fingerprints are unchanged since Pandora commit `430f8fc`.

## Status

Verified on 2026-11-04 against `pandoramoon` 2.0 from PyPI (numpy 2.5.4, numba 0.68.1, Python 3.12): 12 of 12 cases,
45 of 45 unit tests and all sampler tests pass. `check_sensitivity.py --mutations`: all 7 mutations are detected.

## Notes for 2.0

* The pixel-raster eclipse and `numerical_grid` were removed. Positional callers that still pass the grid get an
  error and never a silently misaligned call (`test_pixelmap_removed`).
* `moon_model.light_curve()` uses the exact path, not the float32 occultation cache (differs by at most a few ppm,
  `test_light_curve_exact`). The cached cases pass the cache to `pandora()` explicitly.
* `check_sensitivity.py --commits` (old commits with known bugs) is no longer meaningful: those commits have the
  old call signatures, so every case fails there with a call error. Only the mutations are informative.
