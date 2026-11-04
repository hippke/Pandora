> **Historical development log.** This is the README of the development workspace, written before Pandora 2.0
> (paths such as `../Pandora`, the 14 cases and the pixelmap options refer to that state). It documents how the
> reference was generated and regenerated. The current instructions are in `README.md`.

# Pandora regression test cases

Regression tests for the bug-fixed Pandora (`pandoramoon`) in `../Pandora`:
14 light-curve cases plus 37 unit tests. The reference was first generated from
Pandora commit `d80058c`, after bugs #1–#27 in `../Pandora/BUGS.md` were
fixed (bug #9 is deliberately still open). It was regenerated several times
(newest first):

- **`430f8fc` (P21d):** the limb interpolant's coefficients come from an
  offline table in √k (≤ 8.1e-12 from the per-call build), and the
  interpolant is used from 16 band samples on (was 64, plus a
  len(zs) k ≥ 32 pre-check), so moons and short calls use it too (rule 9).
  All 14 cases still passed at 1e-7; their fluxes moved by ≤ 7.2e-11 vs
  `fe51c33`. Regenerated so that later work compares against this model.
  (Not regenerated for `b8d245c` P24d and `fe51c33` P26: ≤ 2.2e-16.)
- **`54be54e` (P21c):** interior interpolant (≤ 6.4e-9 on the cases).
- **`aab1740` (P21b):** `occult_hybrid` evaluates the limb band
  1 - k <= z < 1 + k with a per-call polynomial interpolant (k <= 0.25,
  >= 64 band samples per call), measured <= 5.6e-9 off mpmath (0.006 ppm;
  rule 9). All 14 cases still passed at 1e-7; their fluxes moved by
  <= 9.3e-10 vs `8fc945e`. Regenerated so that later speed work compares
  against the interpolant model.
- *Not regenerated* for `0150aa4` (P16, polynomial limb arccos): fluxes
  change by <= 2.6e-15 (dense grid incl. contact points), all cases pass.
- **`8647e96` (P14):** `occult_hybrid` uses an interior series for bodies
  fully on the star, with a rigorous error bound of 1e-8 (0.01 ppm; rule 9
  of START_HERE.md). All 14 cases still passed at 1e-7; their fluxes moved
  by <= 6.3e-9 (vs the exact `occult_exact`: <= 8.1e-9, vs mpmath
  <= 8.0e-9). Regenerated so that later exact-speed work compares against
  the series model.
- *Not regenerated* for `b011e7a` (bug #29 in `occult()`), `f19a3de`
  (bug #9: default `occult_small_threshold` 0) and `cb75a54` (P5): all 14
  cases are unchanged, because the cases pass their thresholds explicitly
  (several deliberately keep covering the `occult_small` path).
- **`50cb806` (P3, bug #28):** `occult_hybrid` is exact for every k; the
  small-body "linear interpolation" zone (k <= 0.05, wrong formula, up to
  2.6 ppm off mpmath) is gone. Toward exact: the 5 cases with a k <= 0.05
  moon on the `occult_hybrid` path moved by <= 7.0e-8 (flagged by the 1e-7
  relative fingerprint tolerance on sums); perf draws <= 9.5e-8. The P3b
  commit `ac9c5e8` before it changed no case (<= 4.2e-14). Old reference
  kept as `../scratch_p3/reference_055fbb6.json`.
- **`055fbb6` (P3a):** `occult_hybrid` evaluates K, E and Π exactly in one
  Bulirsch loop (`ell_combo`) instead of the ~2e-8 polynomial K and E.
  Model change toward exact (vs mpmath): regression cases ≤ 2.6e-8 moon /
  6.5e-9 planet flux (0.026 ppm); perf and 300 random draws ≤ 4.2e-8.
  The 1e-7 relative fingerprint tolerance flagged 12 cases (e.g.
  `deficit_sumsq` 2e-7 relative). Old reference kept as
  `../scratch_p3/reference_d80058c.json`.
- **`d80058c`:** case 14 added for the float32 `pandora_deficit()`. Cases
  1–13 are bit-identical to the `c535cca` reference (float32 deficits, P9
  `Likelihood` and the binary-search window changed no existing output).

- **`c535cca`:** case 13 added for the new flux-only entry point
  `pandora_flux()` (P1 sample culling). No model change: the fingerprints
  of cases 1–12 agree with the `5edf35f` reference to < 1e-12 relative
  (two are bit-identical); the rest is fastmath compile-context noise.

- **`5edf35f`:** the exact analytic limb mutual-eclipse overlap became the
  default (`eclipse_method="analytic"`). Moon flux changed by up to 4.4 ppm
  (case 6), 0.8 ppm (case 10), 0.22 ppm (case 5) and 0.07 ppm (case 7), each
  time toward an independent integration (see "Analytic eclipse
  spot-check" below). Cases 11–12 were added for the legacy `"pixelmap"`.
- **`33e9850`:** for bug #26, `timegrid()` spaces cadences by exactly
  `1/cadences_per_day`, which changed every time grid.

Earlier references came from `01b61c6`.

## Quick start

```bash
cd /home/michael/Downloads/pandora/test_cases
python run_tests.py          # light-curve cases + unit tests; exit code 0 = pass
python run_tests.py -v       # with per-case/per-test timing
python run_tests.py --lc     # light-curve cases only
python run_tests.py --unit   # unit tests only (same as: python unit_tests.py)
python run_tests.py NAME ... # selected light-curve cases only
pytest run_tests.py unit_tests.py   # if pytest is installed (it currently is not)
```

Runtime: **~0.9 s** wall time for everything in a fresh process (about
0.2 s for the light-curve cases and 0.25 s for the unit tests). The first run
after Pandora's source changes takes **~12 s**, which is numba recompiling
`pandoramoon` (its `cache=True` cache); this cost is the same for any test
that imports Pandora.

Pandora is imported from `../Pandora` (falls back to the installed package);
set `PANDORA_PATH=/path/to/Pandora` to test another checkout.

## Files

| File | Purpose |
|---|---|
| `cases.py` | Light-curve case definitions, `run_case()`, fingerprint, comparison, invariants |
| `run_tests.py` | Runs light-curve cases and unit tests (standalone and pytest entry point) |
| `unit_tests.py` | Unit tests pinning individual bug fixes; no stored reference needed |
| `reference.json` | Stored reference: case parameters + fingerprints + version info (~65 kB) |
| `generate_reference.py` | Regenerates `reference.json` and `plots/` (`--plots`: plots only) |
| `plots/*.pdf` | One light-curve PDF per case for human inspection |
| `check_sensitivity.py` | Developer tool: proves the tests detect real/synthetic regressions (~3 min) |
| `performance/` | Separate single-core A/B performance benchmark (see `performance/README.md`) |

## The cases

All systems use realistic radii/masses (Sun R = 696,342 km; Jupiter, Neptune,
Earth, Luna, Ganymede from standard values), quadratic limb darkening, and
moon eccentricity 0 except in case 5. Every epoch segment contains only the
transit plus slack (≈25 % beyond the planet T14 + moon excursion), and the
test verifies that every segment starts and ends at flux exactly 1.

| # | Case | k_planet | k_moon | route planet/moon | cadence [1/d] | ssf | epochs | points | eclipse pts |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `sun_jupiter_ganymede_kepler_lc_ssf5` | 0.1027 | 0.00378 | hybrid/small | 48.9 (Kepler LC) | 5 | 3 | 285 | 39 |
| 2 | `sun_jupiter_ganymede_kepler_sc_nocache_limb` | 0.1027 | 0.00378 | hybrid/hybrid | 1468 (Kepler SC) | 1 | 1 | 2582 | 118 |
| 3 | `sun_earth_luna_plato_1min` | 0.0091 | 0.00250 | cache/small | 1440 (PLATO 1 min) | 1 | 2 | 3034 | 263 |
| 4 | `sun_jupiter_earthmoon_tess_2min` | 0.1027 | 0.00915 | hybrid/cache | 720 (TESS 2 min) | 1 | 2 | 1826 | 89 |
| 5 | `sun_jupiter_eccentric_moon_tess_200s` | 0.1027 | 0.00757 | hybrid/cache | 432 (TESS 200 s) | 1 | 2 | 1550 | 72 |
| 6 | `mdwarf_neptune_earthmoon_tess_20s` | 0.1179 | 0.0305 | hybrid/cache | 4320 (TESS 20 s) | 1 | 3 | 1992 | 296 |
| 7 | `sun_jupiter_grazing_kepler_sc` | 0.1027 | 0.00378 | hybrid/small | 1468 (Kepler SC) | 1 | 1 | 1990 | 108 |
| 8 | `giant_earth_luna_tess_10min` | 0.0009 | 0.00025 | small/small | 144 (TESS 10 min) | 1 | 1 | 1008 | 26 |
| 9 | `sun_jupiter_unphysical_moon_plato_1min` | 0.1027 | 0.00378 | hybrid/small | 1440 (PLATO 1 min) | 1 | 1 | 956 | 0 |
| 10 | `sun_jupiter_neptunemoon_kepler_lc_ssf3_nocache` | 0.1027 | 0.0354 | hybrid/hybrid | 48.9 (Kepler LC) | 3 | 3 | 195 | 16 |
| 11 | `mdwarf_neptune_earthmoon_tess_20s_pixelmap` | 0.1179 | 0.0305 | hybrid/cache | 4320 (TESS 20 s) | 1 | 3 | 1992 | 296 |
| 12 | `sun_jupiter_neptunemoon_kepler_lc_ssf3_nocache_pixelmap` | 0.1027 | 0.0354 | hybrid/hybrid | 48.9 (Kepler LC) | 3 | 3 | 195 | 16 |
| 13 | `sun_jupiter_neptunemoon_kepler_lc_ssf3_pandora_flux` | 0.1027 | 0.0354 | hybrid/hybrid | 48.9 (Kepler LC) | 3 | 3 | 423 | 16 |
| 14 | `sun_jupiter_eccentric_moon_tess_200s_pandora_deficit` | 0.1027 | 0.00757 | hybrid/hybrid | 432 (TESS 200 s) | 1 | 2 | 3102 | 72 |

"route" = occultation code path in `pandora()`: `hybrid` = `occult_hybrid`,
`cache` = bilinear occultation cache, `small` = small-body approximation.
"eclipse pts" = supersampled time stamps with planet–moon overlap on the star.
The full description of each case is in `cases.py` / `reference.json`.

What the cases cover:

- **Bodies:** Jupiter/Ganymede, Earth/Luna, Jupiter + Earth-sized moon,
  Jupiter + large icy moon, M dwarf (0.3 R☉)/Neptune/Earth-sized moon,
  red giant (10 R☉)/Earth/Luna, Jupiter + Neptune-sized "binary planet".
- **Cadences:** Kepler LC (29.4 min, with 3× and 5× supersampling), Kepler SC
  (58.8 s), TESS 20 s / 2 min / 200 s / 10 min, PLATO 1 min (as specified;
  PLATO's nominal cadence is 25 s).
- **Mutual eclipses** in 13 of 14 cases. Some have both bodies on the stellar
  disk (two-circle lens path). Others have a body on the stellar limb: the
  analytic three-circle path in cases 2, 6, 7 and 10, the legacy `pixelart`
  raster in cases 11 and 12 (`eclipse_method="pixelmap"`, same systems as 6
  and 10). In case 7, the on-star part of the moon is always fully inside
  the planet disk, so its moon flux is exactly 1. The old pixel raster gave
  a spurious 0.07 ppm there. There is also a total overlap
  (moon disk entirely inside the planet disk; case 6, epoch 0). In the circular cases the moon
  phase `tau_moon` is computed (`tau_for_conjunction`) so that a conjunction
  falls at mid-transit or at ingress. The eclipse raises the moon flux by
  0.08 ppm (case 8) up to 1040 ppm (case 6).
- **Code paths and edge cases:** occult_hybrid, the cache, occult_small; k
  below the cache grid (case 8, bug #3); grazing transit b = 1.0 (case 7);
  eccentric planet orbit (case 6); the one eccentric moon e = 0.5 (case 5);
  supersampling 3× and 5× (cases 1, 10; bug #22); `per_bary ≠ epoch_distance`
  drift and `t0_bary_offset ≠ 0` (cases 1, 10); direct `pandora.pandora()`
  without a cache, as in the retrieval notebook (cases 2, 10); moon outside
  the Hill sphere removed by the plausibility check (case 9);
  `numerical_grid = 51` (cases 10, 12); the `eclipse_method` switch via
  `moon_model` (case 11) and via `pandora(..., eclipse_pixelmap=True)`
  (case 12); the flux-only `pandora_flux()` (case 13: as case 10 with 3×
  wider epoch windows, so ~80 % of the output cadences lie beyond the
  culling bound and supersampled cadences straddle it; coordinates for the
  fingerprint come from `pandora()`); the float32 `pandora_deficit()`
  (case 14: the eccentric moon of case 5 without cache, flux = 1 −
  deficit; the invariant `total = planet + moon − 1` allows 2^-23 × max
  deficit there, since each component is rounded separately); multiple epochs with varying moon
  phase (non-integer `per_bary/per_moon`).
- **Case 5 tuning:** its `tau_moon = 0.73` was chosen so that the late orbital
  phase (approach to periapsis) is in transit, where the eccentric Kepler
  solver is most error-prone. Reverting the bug #2 fix changes its moon flux
  by 1.3 % of the moon depth; at e = 0.25 the best phase gave only 0.01 %.

## What is checked

For every case, `run_tests.py` checks three things:

1. **Case definition unchanged.** The parameters in `cases.py` must equal the
   ones stored in `reference.json`, so an accidental edit of a case cannot
   silently pass as a model change.
2. **Physics invariants** (independent of the reference): all fluxes finite
   and within [0, 1]; `total = planet + moon − 1`; every epoch segment starts
   and ends at flux 1 (slack present, transit fully contained); every segment
   shows a transit; a mutual eclipse is present where one is required; the
   unphysical-moon case has moon flux identically 1.
3. **Fingerprint vs reference.** For each component (planet, moon, total) of
   the deficit `d = 1 − flux` at output cadence:
   - `sum(d)`, `sum(d²)`, and the time moment `sum(d·(t − t₀))`, which reacts
     to timing shifts;
   - minimum flux;
   - 64 equal-length index bins of `sum(d)`, which localise *where* a change
     happens;
   - 8 random-weight projections `Σ wᵢ dᵢ` (fixed seed), which are sensitive
     to a change at any single point;
   - plus array lengths, the time grid (first/last/sum), and the number of
     eclipse points (±1 allowed).

   The tolerance is relative: `1e-7 × sum(d)` per bin and projection, and
   `1e-7` relative for the scalars. This is far above floating-point noise
   (repeated warm runs are bit-identical; fastmath reassociation across
   numba/LLVM versions is expected around 1e-15), and far below any physical
   change. **Cold vs cached code is not bit-identical.** Measured on
   `fa77161` and `5edf35f`: code freshly compiled by numba and the same code
   loaded from the `cache=True` cache differ by up to 3.4e-14 (fastmath).
   Explicit vs omitted optional arguments, which give different numba
   specialisations, differ at the 1e-16 level. For bit-identity checks
   between two versions, compare cold with cold or warm with warm. Verified: perturbing a single in-transit point by 1 ppm, or a
   single moon point by 0.01 ppm, is caught in 10 of 10 cases (measured
   before cases 11-12 were added).

A failure prints the first mismatching quantity per check, e.g.
`fingerprint: moon.bins[37]: ref=... new=...`. The bin index tells you which
part of which epoch changed.

## On the cumulative-sum idea

The idea was to store `cumsum(flux)` and compare against it.

- **The full cumsum array is valid.** It is lossless (the light curve is its
  first difference), so it catches everything. But it costs as much to store
  as the light curve itself.
- **Only its final value, `sum(flux)`, is cheap but incomplete.** It is blind
  to anything that conserves the integrated deficit:
  - time shifts and mirror-image timing errors of a fully contained transit;
  - a moon placed on the wrong side of the planet (bug #8 is exactly this);
  - compensating errors, e.g. ingress too deep and egress too shallow;
  - errors in one component cancelling in another.

  It also says nothing about *where* a difference is.

The stored fingerprint keeps the cheapness (≈65 kB of JSON for all cases, no
array dumps) and closes those gaps. The binned sums are a coarse cumsum, the
projections give point sensitivity, and the time moment gives timing
sensitivity.

`check_sensitivity.py` measures the difference. It ran the tests against old
Pandora commits that still contain known bugs and against synthetic code
mutations (before case 13 and the P1 unit test existed). SUM and SUITE count light-curve cases flagged (out of 12); UNIT
counts failing unit tests (out of 16).

| Variant | SUM only | Full LC suite | UNIT |
|---|---|---|---|
| original upstream code (bugs #1–#27) | 12 | 12 | 14 |
| bug #2 (eccentric solver) + #19–#27 open | 12 | 12 | 10 |
| bugs #20–#27 open | 12 | 12 | 9 |
| bugs #21–#27 open | 12 | 12 | 8 |
| bugs #22–#27 open | 12 | 12 | 7 |
| bugs #23–#27 open | 12 | 12 | 6 |
| bugs #24–#27 open | 12 | 12 | 5 |
| bugs #25–#27 open | 12 | 12 | 4 |
| bugs #26–#27 open (`timegrid` spacing, count) | 12 | 12 | 3 |
| bug #27 open (`timegrid` count round-off) | 5 | 5 | 2 |
| `fa77161`: pre-analytic (pixelart on the limb) | 5 | 5 | 1 |
| barycentre shifted ~2 s in time | 10 | **12** | 0 |
| bug #8 re-introduced (eccentric moon mirrored) | 1 | 1 | 1 |
| circular moon on opposite side | 10 | 10 | 1 |
| mutual eclipses switched off | 11 | 11 | 1 |
| occult_small normalisation +1 ppm | 3 | 3 | 0 |
| cache resolution 300 → 299 | 5 | 5 | 0 |
| limb raster `numerical_grid` + 2 (pixelmap only) | 2 | 2 | 0 |
| analytic overlap fraction × (1 + 1e-4) | 3 | 3 | 1 |
| analytic default replaced by pixelart | 4 | 4 | 1 |

Notes on the newest rows:
- The 5 cases flagged for `33e9850`/`fa77161` are the 4 cases whose limb
  eclipses changed with the analytic method (5, 6, 7, 10), plus case 12,
  which fails because `pandora()` had no `eclipse_pixelmap` argument yet.
- The analytic ×(1 + 1e-4) mutation is a 0.01 % change of the overlap. It
  is caught in the cases with large limb overlaps and by
  `test_analytic_eclipse_matches_reference`.

Reading the results:

- **SUM alone does better than theory suggests here.** At finite cadence, a
  shift or a mirror also changes how ingress and egress are sampled, so the
  sum moves slightly. This is incidental, not guaranteed.
- **The light-curve suite flagged every case SUM flagged, plus more.** It
  catches the 2-second time shift in all cases. With the eccentric case tuned
  at e = 0.25 (before re-tuning), bug #2 was caught *only* by the suite (via
  `sum(d²)`).
- **Every commit before `33e9850` fails every light-curve case.** All of them still
  contain bug #26, so their time grids differ from the reference. The LC
  columns can no longer separate the earlier fix steps; the UNIT column
  still does. Against the pre-#26 reference (`01b61c6`), the LC suite
  flagged 8, 4, 3, 3, 2, 0, 0, 0 cases for the first eight rows, and the
  `numerical_grid` + 2 mutation flagged 5 cases instead of 4. The other
  mutations give the same counts as before.
- **Every fix step is caught by at least one test.** Each successive commit
  in the table fixes one more bug, and at every step the number of failing
  tests drops. The UNIT column catches the bugs the light curves cannot see.

Light-curve non-detections (covered by unit tests instead):

- **Bugs #20 and #25** (NaN / unset terms at exact boundaries z = 1 − k and
  z = 1 + k): these exact points essentially never occur in real time grids.
- **Bug #23** (plausibility check ignored moon eccentricity): needs an
  eccentric moon whose periapsis/apoapsis violates the limits. That conflicts
  with "only one eccentric case", and case 5 is physical.
- **Bug #24** (video): `video()` is not part of the light-curve tests.
- **Bug #27** (cadence count round-off): no case has an integer
  `cadences_per_day * epoch_duration`.
- **Moon on opposite side:** this mutation edits only the circular solver
  `ellipse()`. The 2 unflagged cases don't run it: the eccentric case (uses
  `ellipse_ecc()`) and the unphysical-moon case (moon removed).

## Analytic eclipse spot-check (reference regeneration for `5edf35f`)

Before regenerating, every limb-eclipse sample of every case was checked
against an "ideal" moon flux. The ideal uses the same model assumption
(moon deficit × (1 − er)), with er computed by an independent midpoint
integration of the exact disk intersections (~1e-9). Values are the max
deviation of the moon flux in ppm:

| Case | Limb pts | Analytic | Pixelmap (n=25/51) |
|---|---|---|---|
| 2 `..._kepler_sc_nocache_limb` | 85 | 0 | 0 |
| 5 `..._eccentric_moon_tess_200s` | 39 | 7e-8 | 0.22 |
| 6 `mdwarf_neptune_earthmoon_tess_20s` | 82 | 1.4e-6 | 4.41 |
| 7 `..._grazing_kepler_sc` | 232 | 0 | 0.07 |
| 10 `..._neptunemoon_kepler_lc_ssf3_nocache` | 14 | 1.7e-6 | 1.46 (supersampled) |

The pixelmap mode was bit-identical to `fa77161` on all cases (cold vs cold
and warm vs warm). Outside limb eclipses the two methods agree bit-for-bit.

## Unit tests (`unit_tests.py`)

Each test pins one or more fixed bugs with a direct check; none of them needs
a stored reference.

| Test | Bugs | What it checks |
|---|---|---|
| `test_occult_finite_at_boundaries` | #4, #5, #6, #20 | No NaN/inf/exception in `occult`/`occult_hybrid` at z = 1−k, k, k−1 and float neighbours, for k = 0.005…0.995 and k > 1 |
| `test_occult_first_contact_is_unity` | #25 | Flux is exactly 1 at z = 1+k, for varied preceding elements and a dirtied heap |
| `test_occult_accuracy_near_limb_contact` | #21 | Within 2e-4 of z = 1−k both functions match an independent radial-integration reference to < 1 ppm |
| `test_hybrid_matches_exact` | #6, #7, #11, #12 | `occult_hybrid` == `occult` on random z (< 1e-6), boundary points vs the reference, interpolated flux ≤ 1 |
| `test_complete_occultation` | #16 | k > 1, z ≤ k−1 gives flux 0 |
| `test_cache_below_grid_keeps_transit` | #3 | Cache accuracy inside its grid; tiny planets (k < 0.001) keep the correct depth through `pandora()` |
| `test_eccentric_orbit_matches_kepler` | #2, #8 | `ellipse_ecc` matches a bisection Kepler reference (1e-12) for e = 0.1/0.5/0.9; periapsis at Ω + w; e→0 equals `ellipse()` |
| `test_plausibility_uses_periapsis_and_apoapsis` | #23 | e = 0.9 moon inside the planet is rejected; apoapsis outside the Hill limit is rejected; valid orbits are kept |
| `test_video_disk_full_size` | #24 | Stellar disk in `video()` has radius 1 with the requested number of circles |
| `test_resample` | #1, #17 | Supersampling means, partial tail |
| `test_grid_helpers` | #10, #14, #15, #22 | Pre-t0 epochs, `supersampling_factor` validation, epoch-aligned supersample groups, `ld_invert(0,0)` |
| `test_timegrid_cadence_spacing` | #26 | `timegrid()` matches an independent exposure layout (1e-9 d): cadences 1/cpd apart, sub-exposures centred per cadence, block centred on the epoch, exact tiling for integer cpd·dur |
| `test_analytic_eclipse_matches_reference` | analytic eclipse | `er_fast` vs an independent chord integration on 80 random limb geometries (< 1e-6 of the moon area; measured 2e-8), exact symmetric/tangent inputs; pixelart within its raster bound (< 2 %) |
| `test_eclipse_method_switch` | analytic eclipse | Default is `"analytic"`; via `moon_model` and `pandora(eclipse_pixelmap=...)` both methods agree bit-for-bit outside limb eclipses and differ at the pixel-raster level on the limb; invalid names raise `ValueError` |
| `test_timegrid_cadence_count_roundoff` | #27 | Cadence count equals the floor of the exact decimal product cpd·dur (e.g. 720 · 0.35 → 252, not 251) |
| `test_ellpicb_converges` | #18 | Extreme valid arguments converge to finite positive values |
| `test_pandora_deficit_float32` | float32 step | `pandora_deficit()` is float32 and equals the round-to-nearest of `1 − pandora_flux()` (≤ 2^-24 relative), 0 exactly where flux is 1; logL change < 0.01 (40 random draws) |
| `test_pandora_window_and_likelihood` | P9 | `pandora_window()` densified == `pandora_flux()` (≤ 1e-12); `Likelihood` == `-0.5 nansum(((f − d)/σ)²)` (≤ 1e-10 relative) incl. NaN data/yerr, ssf with partial cadence, cache, pixelmap; invalid inputs raise `ValueError` |
| `test_sorted_time_window_search` | binary search | `x_bary_window_sorted()` == full scan (identical indices, x ≤ 1e-12) on 300 random geometries incl. samples exactly at/one ulp around the window edges, duplicates, drift, t0 offsets, eccentric planet, fallbacks; `pandora_window(time_sorted=True)` == scan; `Likelihood` sorted == shuffled time. Kills "half-width × 0.9999" |
| `test_pandora_flux_matches_pandora` | P1 culling | `pandora_flux()` and `pandora()` vs an unculled reimplementation of the pre-P1 pipeline (all samples through z, occultation, `eclipse()`, `resample`), ≤ 1e-12, same flux == 1 pattern; 80 random draws (unphysical, eccentric moon/planet, ssf 1/3/5 with partial cadence, cache, pixelmap, unsorted time with gaps, coplanar eclipsing moons, edge-on low-mass moons sampled at the culling bound) with coverage asserts; the cache contact bin beyond z = 1+k; a keyword-argument call. Kills the mutations "bound × 0.99 on a_moon" and "no 1.1 cache limit" |
| `test_acos_poly` | P16 | `trig.acos_poly` ≤ 1 ulp vs libm on [−1, 1] (uniform, near ±1 down to 1 ulp, ±1/2, 0), ≤ 1.5 ulp vs long double, exact at ±1; a loop over it is vectorised (uncached copy) |
| `test_sincos_turns_and_fused_orbit` | P7 | `sincos_turns` ≤ 4e-16 vs a long-double reference (exact turn reduction) for \|u\| up to 1e8 turns, quadrant edges, NaN/inf; fused `ellipse_z` == `ellipse()` + sqrt (≤ 1e-11); the loop is vectorised (uncached copy) |
| `test_ell_combo_and_occultation_exact` | P3, #28 | `ell_combo` K, E, Π vs mpmath (≤ 2e-15 relative); `occult_hybrid` vs mpmath radial-quadrature fluxes (1e-10; 2e-9 within 2e-6 of z = k, 1 − k since P15, 5e-7 before with the nudges) for k = 0.003…0.7, incl. the former small-body zone (fails on `ac9c5e8`) |
| `test_hybrid_second_contact_gap` | #29 | At the 1-ulp gap z = 1 − k (where the old Case III/IV conditions matched neither case) `occult_hybrid` matches the radial-integration reference, whatever sample came before (fails on `055fbb6`); `occult` is continuous with the next float after dirtying the heap (fails on `50cb806`) |
| `test_moon_exact_by_default` | #9 | `model_params` and `Likelihood` default `occult_small_threshold` 0; no example/notebook sets another value; the default moon path equals the exact model (≤ 1e-12) and differs from `occult_small` (fails on `50cb806`) |
| `test_eclipse_skip_is_exact` | P5 | `eclipse_possible()` False only where `eclipse()` leaves the flux unchanged: tangency ± ulp, on/limb/off star, both eclipse methods; the loop is vectorised. Kills "bound × (1 − 1e-6)" |
| `test_occult_batch_matches_scalar` | P3b | The vectorised exact pass equals the scalar per-sample model (≤ 2e-11; 5e-9 within 1e-6 of first contact) on dense z grids incl. ulp boundaries, k = 0.001…1.2; non-converged lanes take the scalar fallback; the batch loop is vectorised. Kills "no fallback" |
| `test_occult_contact_points_exact` | P15 | `occult_hybrid` vs embedded mpmath fluxes (`CONTACT_REF`) at z = k ± 1e-15…1e-3, 1 − k ± 1e-15…1e-3, 1 + k − 1e-6/1e-3, k = 0.001…0.2: ≤ 1e-12 (5e-10 on the limb side within 1e-6 of 1 − k: limb-area arccos); z == k exactly finite and continuous for k = 0…1.2. Kills the old z → k + 1e-6 and q → 1 − 1e-8 nudges (4.3e-9 / 2.9e-8) |
| `test_no_k_snap_near_one_half` | #30 | `occult_hybrid` (1e-10; 5e-9 at z = 1 − k) and `occult` (1 ppm) vs embedded mpmath fluxes (`K05_REF`) at k = 0.5 ∓ 9e-5, ∓ 1e-6 and 0.5, u = (0.4, 0.25) and (2, −1). Fails on `945b2d5` by 103 ppm (the k → 0.5 snap) |

**Bug #25 was found by these tests.** The cache-accuracy check failed only
on cold runs. The cause was the exact contact point z = 1+k: there `occult`
returned uninitialised memory (observed −0.43 and −5e98) and `occult_hybrid`
reused the previous element's terms. It is fixed in Pandora commit `01b61c6`
and documented in `BUGS.md`; the light-curve reference is bit-identical
before and after the fix. The unit tests were run 3× with cold numba caches
and passed every time.

`test_video_disk_full_size` creates `video()` animations without saving
them. It closes all figures afterwards, because unsaved animations plus tqdm
segfault at interpreter exit (third-party issue, see BUGS.md known
limitations). Keep that `finally: plt.close("all")` if you edit the test.

## Rules for future agents

- **A failing test means the model output changed.** Find out why (the
  failing bin and component point to the place). Only regenerate the
  reference after confirming the change is an intended, verified correction,
  and record why in `../Pandora/BUGS.md` or the commit message:
  ```bash
  python generate_reference.py   # rewrites reference.json and plots/*.pdf
  ```
  It refuses to write a reference if any invariant fails.
- **Do not edit cases in place to make tests pass.** The definition check
  will flag it. Add new cases with new names instead; then regenerate (new
  cases appear as `NEW` until then).
- After a model change, look at `plots/` (`python generate_reference.py
  --plots` regenerates the PDFs without touching the reference).
- **A failing unit test points at a specific bug number** (see the table
  above and `../Pandora/BUGS.md`): a fixed bug has come back. Unit tests
  carry no stored reference; never relax their thresholds to make them pass.
  When you fix a new bug, add a unit test with a `Bug #N` docstring.
- **Also run the unit tests on a cold numba cache** after changing Pandora:
  `rm -f ../Pandora/pandoramoon/__pycache__/*.nb*`. Uninitialised-memory
  bugs (like #25) can pass by luck on warm runs.
- **Plot layout:** the top row shows planet (blue), moon (red) and total
  (black) on a common scale from min to max flux with 10 % margins. The
  bottom row repeats the moon alone on its own scale in ppm, because
  Luna/Ganymede-sized moons are invisible at the planet's scale.
- `check_sensitivity.py` needs the git history of `../Pandora` (commit hashes
  above) and takes ~3 minutes. Run it if you change the fingerprint or the
  tolerances.
- Platform note: references were generated with Python 3.13.11,
  numpy 2.5.3, numba 0.68.0 (recorded in `reference.json`). A failure that
  appears only after a library upgrade, at the ~1e-7 relative level, should
  be investigated before loosening `rtol` in `compare_fingerprints()`.
