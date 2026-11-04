# Pandora Bug Audit

Date: 2026-10-06
Base commit: `baa7788` ("Update README.md")
Scope: full review of `pandoramoon/` (`pandora.py`, `occult.py`, `eclipse.py`, `ellipse.py`, `grids.py`, `helpers.py`), examples, and notebooks. Every bug below was verified numerically against reference implementations before fixing; short repro snippets are given. Line numbers refer to the base commit (they shift after fixes).

Status legend:
- **FIXED** - fixed in this patch series (commit subject given)
- **OPEN** - known, deliberately not fixed yet (details below)
- **IGNORED** - accepted for now per project decision

---

## 1. `resample()` drops the last supersampled point of every cadence - FIXED

Severity: critical (wrong results whenever `supersampling_factor > 1`)
Location: `pandoramoon/helpers.py:8-13`
Fixed in: "Fix off-by-one in resample() supersampling average (bug #1)"

```python
end = start + factor - 1
out_arr[idx] = np.mean(arr[start:end])
```

`np.mean(arr[start:end])` excludes the end index, so only `factor - 1` of the
`factor` sub-cadences are averaged; the last sample of every cadence is
silently discarded. With `supersampling_factor = 2` no averaging happens at
all (plain decimation). Verified: `resample([0..9], 5)` returned
`[1.5, 6.5]` instead of `[2.0, 7.0]`.

The fix averages `[start, start + factor)` and additionally includes a
partial trailing group when `len(arr) % factor != 0` (previously the tail was
dropped - see also #17) so no data is silently lost.

## 2. `ellipse_ecc()`: `if M.any() > pi:` never True - eccentric moon positions wrong near the periastron recurrence - FIXED

Severity: critical for eccentric moon orbits (`e_moon > 0`)
Location: `pandoramoon/ellipse.py:36-41` and `54-55` (at base commit)
Fixed in: "Fix element-wise M>pi reflection in ellipse_ecc (bug #2)"

The code:

```python
M = (2 * pi / per) * (time - (tau * per))
flip = False
M = M - (np.floor(M / (2 * pi)) * 2 * pi)
if M.any() > pi:
    M = 2 * pi - M
    flip = True
...
if flip:
    k = 2 * pi - k
```

**Intent.** The Kepler starter + high-order corrector used here (lines 42-53)
is accurate for mean anomaly `M` in `[0, pi]`. For `M` in `(pi, 2*pi)` the
standard technique is to solve with the reflected anomaly `2*pi - M` and
reflect the result back, using the symmetry
`E(2*pi - M) = 2*pi - E(M)`. The `flip` flag implements this reflection: it is
set when any part of the orbit is in the second half, the solver then runs on
the reflected anomaly, and the eccentric anomaly `k` is un-reflected at the
end.

**What actually happens.** `M` is a 1-D array (one mean anomaly per time
stamp). `M.any()` reduces the whole array to a *single* boolean ("is any
element nonzero") - it is not an element-wise comparison. The condition
therefore evaluates `True > pi` (or `False > pi`), i.e. `1 > 3.14159...`,
which is always `False`. The reflection branch is dead code: `flip` is always
`False`, `M` is never reflected, and `k` is never un-reflected. This was
confirmed by static analysis and by matching the numba function's output
against a pure-Python replication of the same (unreflected) code path.

**Measured impact.** Comparing the code's eccentric anomaly `k` against a
bisection reference solver (converged to machine precision) over a full
orbit (20,000 mean anomalies; numba replica with the same `fastmath=True`
flags verified identical to the shipped function to 7e-16):

| e    | max error for M in [0, pi) | max error for M in [pi, 2*pi) |
|------|----------------------------|-------------------------------|
| 0.2  | 8.9e-16                    | 4.6e-07                       |
| 0.5  | 8.9e-16                    | 5.6e-04                       |
| 0.9  | 8.9e-16                    | 1.9e-01                       |

The first half of the orbit is machine-precision accurate. On the second
half the error is not uniform: it stays near machine precision until
roughly M ~ 4.8 (~60% into the second half) and then grows rapidly toward
M -> 2*pi, i.e. as the moon approaches the *next* periastron passage
(the starter formula `alpha ~ (pi - abs(M))` is progressively further
outside its design range as |M| exceeds pi, and the fixed-order g/h/k
corrector amplifies rather than fully corrects a poor starter). At e = 0.9
the anomaly error reaches ~0.18 rad near M -> 2*pi. The corresponding sky
position error is muted near periastron (the orbit is small and slow-moving
in radius there): measured against a correctly scaled reference, the worst
sky-position error at e = 0.9 is ~1.7% of a_moon (mean ~0.1% over the
second half), concentrated in the final ~15% of the orbital period.

What breaks is the *time parameterization*, not the orbit geometry: the
code's radius `1 - e*cos(k)` and true anomaly `nu(k)` both derive from the
same (wrong) `k`, so the moon always lies on the correct Keplerian ellipse
- but at the wrong point of it for the given time stamp. The anomaly offset
delta(M) varies from ~0 at M ~ pi to ~0.18 rad at M -> 2*pi (e = 0.9), so
it cannot be absorbed by any fixed redefinition of `tau_moon` or
`w_moon`; the model family cannot represent the true Keplerian ephemeris
for that part of the orbit.

**Second, structural problem.** Even if the comparison were fixed to
`(M > pi).any()`, a single scalar `flip` cannot be correct for a *time array*:
real grids (e.g. `per_moon = 0.3 d` over a `0.6 d` transit window) contain time
stamps in both halves of the orbit. The reflection must be applied per
element: for each time stamp either solve for `M` and use `E`, or solve for
`2*pi - M` and use `2*pi - E`.

**Consequences.**
- Any model with `e_moon > 0` places the moon at the wrong point of its
  (correctly shaped) orbit for the time stamps in the affected zone near the
  periastron recurrence; the anomaly error grows rapidly with eccentricity
  and with proximity to `M -> 2*pi`.
- Injection-retrieval loops that use the same code for injection and
  retrieval partially absorb the error into other parameters, but because
  delta(M) varies across the orbit it cannot be fully absorbed: the
  `e_moon` / `w_moon` / `tau_moon` posteriors are biased, and the model
  cannot represent the true Keplerian ephemeris near the periastron
  recurrence.
- Circular moon orbits (`e_moon = 0`) use `ellipse()` instead and are
  unaffected.
- Interplay with bug #8 (fixed): the former `r = -(1 - e*cos(k))` negation
  mirrored the whole orbit through the barycenter, equivalent to redefining
  `w_moon -> w_moon + 180 deg`. With #8 fixed, the `M in [0, pi)` half
  followed the standard convention while the `M >= pi` half remained
  incorrect until this bug was fixed.

**Fix.** The reflection is now applied element-wise and vectorized:

```python
M = M - (np.floor(M / (2 * pi)) * 2 * pi)
flip = M > pi                                   # per element
M = np.where(flip, 2 * pi - M, M)               # solve in [0, pi] only
... starter + g/h/k corrector -> k ...
k = np.where(flip, 2 * pi - k, k)               # un-reflect
```

The `arctan(tan(k/2))*2` true-anomaly branch already returns values in
(-pi, pi] for k in (pi, 2*pi), which is equivalent modulo 2*pi, so no
further branch handling is needed. Verified against a machine-precision
bisection reference over dense (M, e) grids: positions now agree to
~4e-16 over the *full* orbit for e in [0.1, 0.9] (previously up to
~1.9e-1 rad anomaly error); the e -> 0 limit matches the circular
`ellipse()`; periastron direction, M = pi, and M -> 2*pi edges are
correct. Runtime cost of the two `np.where` passes: ~25 ns per time
stamp (~10 ms per 100k stamps), negligible relative to the full model.
The injection-retrieval notebook (circular injection) is unaffected by
construction; eccentric injection-retrieval should be spot-checked
against an independent ephemeris when next exercised.

## 3. `read_occult_cache()`: index wraparound for k <= 0.001 erases transits - FIXED

Severity: critical (silent signal loss for very small bodies)
Location: `pandoramoon/occult.py:26-27`; call sites `pandoramoon/pandora.py:438-443, 451-452`
Fixed in: "Fix occult-cache wraparound for k below grid; route to occult_small (bug #3)"

```python
idx_k = int(np.ceil((k - ks[0]) / (ks[1] - ks[0])))
```

For `k <= ks[0] = 0.001` this yields `idx_k = 0`, and the subsequent
`fs[idx_k - 1]` / `ks[idx_k - 1]` wrap around to the *last* cache row
(k = 0.1) with a negative interpolation weight. Combined with the
`flux > 1` clamp, the light curve of any body with `k < 0.001` is clipped to
1: the transit disappears entirely. Verified end-to-end via
`moon_model.light_curve`: `r_planet = 0.0005` and `0.0009` produced depth
exactly 0.0 (expected ~3.0e-7 and ~9.8e-7, i.e. k^2 / (1 - u1/3 - u2/6));
`k >= 0.001` is accurate.

Trigger paths: any `r_planet < 0.001` (cache is always used for
`r_planet < 0.1`), or `r_moon < 0.001` when `occult_small_threshold < 0.001`.

The fix clamps `idx_k` into `[1, len(ks) - 1]` in `read_occult_cache()` and,
in `pandora()`, routes bodies with `k` below the cache grid (`cache[1][0]`)
to `occult_small()`, which is exact to < 0.1 ppm there.

## 4. `z += 1e-6` guard can round *back onto* k -> ZeroDivisionError - FIXED

Severity: high (latent crash of the whole sampler)
Location: `pandoramoon/occult.py:220-221` (`occult`), `383-384` (`occult_hybrid`)
Fixed in: "Fix z-k perturbation rounding onto k (crash) (bug #4)"

```python
if abs(z - k) < 1e-6:
    z += 1e-6
```

For example `z = 0.299999`, `k = 0.3`: the float64 difference is
`9.999999999e-07 < 1e-6`, so the guard fires - and `0.299999 + 1e-6` rounds to
exactly `k`. Then `x1 = (k - z)**2 = 0`, and the case-III/IV formulas divide
by `x1` (`3*x3/x1`, `x2/x1`) -> `ZeroDivisionError` (numba raises for float
division by zero). Verified: `occult([0.299999], 0.3, ...)` and
`occult_hybrid([0.299999], 0.3, ...)` both crash; the crash set includes
k = 0.26...0.5 (float-rounding dependent) at `z = k - 1e-6`.

Per-call probability is tiny, but a long nested-sampling run evaluates
billions of (z, k) combinations and would eventually die. The fix assigns
`z = k + 1e-6` instead of incrementing, which can never round onto `k`.

## 5. Case-IV `+1e-8` regularizer pushes q above 1 -> ZeroDivisionError just below z = 1 - k - FIXED

Severity: high (latent crash in a narrow z-band)
Location: `pandoramoon/occult.py:316`
Fixed in: "Remove case-IV q regularizer that pushed q above 1 (crash band) (bug #5)"

```python
q = sqrt((x2 - x1) / (1 - x1) + 1e-8)  # re-calc because different condition
```

In case IV (`z < 1 - k`, k < 0.5), the ratio `(x2 - x1)/(1 - x1)` approaches 1
as `z -> 1 - k` from below. The `+ 1e-8` pushes `q` slightly above 1 for z
within ~1e-9 of `1 - k`, so `kc = sqrt(1 - q**2)` receives a negative
argument, NaN-poisons `ellpicb`, and numba raises `ZeroDivisionError`.
Verified: `occult([0.9 - 1e-9], 0.1, ...)` crashes while `0.9 - 1e-8` is fine;
band width grows with k (~1e-9 at k = 0.05, ~5e-9 at k = 0.3). The same
mechanism crashes `occult(0.3, 0.7)` where float rounding routes the
boundary point into case IV. In pure Python the failure mode is a log domain
error instead. Any point in the affected band lies deep inside the
`abs(k + z - 1) < 1e-4` special-case band, where the general (q-based) value
is immediately overwritten - the regularizer serves no purpose.

The fix removes the `+ 1e-8` (q is mathematically < 1 inside case IV, and
q = 0 at z = 0 is handled fine by `ellk`/`ellec`/`ellpicb`) and adds a
defensive `q >= 1 -> q = 1 - 1e-8` clamp for float-rounding edge cases.

A follow-up commit applies the same clamp to `occult_hybrid`'s case IV,
which had no guard at all: at the `z = 1 - k` boundary float rounding can
make `q` *exactly* 1.0 (e.g. z = 0.3, k = 0.7), so `kc = sqrt(1 - q**2) = 0`
and the `ellpicb` iteration can never converge. Before bug #18's fix this
silently returned 0 after 1000 dead iterations (harmless only because the
`|k + z - 1| < 1e-4` special case overwrites `ldx` there); with #18 it
raised. `ellpicb` itself converges for any q < 1 (verified down to
1 - q ~ 3e-16), so clamping q below 1 is sufficient. Verified with a dense
boundary sweep (k = 0.02...0.98, z within 1e-9...1e-3 of 1 - k and of k):
no exceptions, both functions agree with the reference.

## 6. Boundary gaps at z = 1 - k / z = k leave `ld`/`ed` uninitialized -> garbage flux or crash - FIXED

Severity: high (silently wrong flux at exact boundary values)
Location: `pandoramoon/occult.py:282-284` (`occult` case III, second clause),
`408-409` (`occult_hybrid` case III, both clauses)
Fixed in: "Close z=1-k boundary gaps in occult/occult_hybrid; add missing q guard (bug #6)"

`occult_hybrid` uses strict inequalities in both case-III clauses
(`z > 0.5 + abs(k - 0.5)` and `z > abs(1 - k)`), while case IV uses strict
`z < 1 - k`. The single points `z = 1 - k` (k <= 0.5) and `z in {1 - k, k}`
(k > 0.5) are therefore covered by *no* branch: `ldx`/`edx` keep their
`np.empty` contents and the returned flux is arbitrary. Verified:
`occult_hybrid([0.9], 0.1)` returned 0.989231 where the correct value is
0.991853; `occult_hybrid([0.4], 0.6)` returned 0.6957 vs 0.6080. The file's
own header comment notes the `>=` fix ("must be >= in z >= 0.5 + abs(k - 0.5)")
- but it was applied only to `occult`, not to `occult_hybrid`.

`occult` has the same gap at `z = abs(1 - k)` for k > 0.5 (strict `z >` in the
second clause): the boundary point either falls through to uninitialized
values or, due to float rounding of `1 - k`, enters case IV and crashes via
bug #5. Verified: `occult([0.3], 0.7)` -> ZeroDivisionError; `occult([0.4], 0.6)`
uses stale memory.

The fix makes the case-III lower bounds inclusive in both functions
(matching the Mandel & Agol 2002 validity range `|1 - k| <= z <= 1 + k`), adds
the missing q-regularization guard (`q -> 1 - 1e-8` when q is within 1e-8 of
1, as already present in `occult`) to `occult_hybrid` case III, and clamps the
`sqrt((1 - x1) * (x2 - 1))` argument in case III with `max(..., 0)` to guard
against float-rounding negatives at the boundary.

## 7. `occult_hybrid` writes to `flux[-2]`/`flux[-1]` and reads uninitialized memory - FIXED

Severity: medium (undefined behavior, harmless only by accident)
Location: `pandoramoon/occult.py:386-389, 485-487`
Fixed in: "Guard negative-index flux writes in occult_hybrid (bug #7)"

The loop `for i in range(-2, len(zs))` uses `i = -2, -1` for the two
interpolation support points (z = 0 and z = 0.65) which do not correspond to
array elements. The code nonetheless executes `flux[i] = 1` (unocculted
branch) and reads/writes `flux[i]` in the final clamp. With Python/numba
negative indexing these writes hit the *last two elements* (later overwritten
- benign), and for `len(zs) < 2` they are out-of-bounds (undefined behavior
under numba, which does not bounds-check by default). The fix guards all
`flux[i]` accesses with `i >= 0`.

## 8. `ellipse_ecc()` negates the radius: `r = -(1 - e*cos(k))` - w_moon convention off by 180 deg - FIXED

Severity: medium (wrong parameter interpretation; inconsistent conventions)
Location: `pandoramoon/ellipse.py:56`
Fixed in: "Fix w_moon convention: remove radius negation in ellipse_ecc (bug #8)"

The negative radius mirrors both the moon and the planet through the
barycenter, which is equivalent to redefining `w_moon -> w_moon + 180 deg`
(the traced orbit is still a valid Kepler orbit with that flipped periastron,
for the M < pi half - see bug #2 for the other half). Verified: at
perisastron passage time `t = tau_moon * per_moon` the code placed the moon at
position angle 180 deg instead of the given `w_moon = 20 deg`. The convention
was also inconsistent with the circular model `ellipse()`, where the same
`tau_moon` puts the moon on the opposite side.

Because `w_moon` (and `tau_moon`) are free parameters in retrievals, fits are
self-consistent either way, but the parameter meaning differs from the
documented `[deg]` convention and from the e = 0 model. The fix uses the
positive radius. Note: this changes model output for all `e_moon > 0`
injections (the moon now appears where the parameters say it should).

## 9. Examples use `occult_small` far beyond its validity range - FIXED

Severity: medium (model error larger than injected noise in the examples)
Location: `examples/*.py` (`params.occult_small_threshold = 0.1`),
`pandoramoon/occult.py:70-71` (docstring: "Adequate for k<~0.01")
Fixed in: "Exact moon occultation by default: occult_small_threshold = 0 (bug #9)"

The small-planet approximation assumes the occulted stellar patch is
uniformly lit. Measured error against the exact Mandel-Agol model
(u1 = 0.4089, u2 = 0.2556):

| k       | max error        |
|---------|------------------|
| 0.01    | 4.3 ppm          |
| 0.02    | 25.9 ppm         |
| 0.03526 | 105.4 ppm        |
| 0.05    | 248.1 ppm        |
| 0.1     | 1335.5 ppm       |

All examples set `occult_small_threshold = 0.1` with `r_moon = 0.03526`, so
the moon light curve carries ~105 ppm model error - larger than the 100 ppm
injected noise. The default in `model_params` (0.01, ~4.3 ppm at the
boundary) is consistent with the docstring; the examples' 0.1 is not.
Accepted at first (project decision). **Fixed** once the exact model was
cheap (P3: `occult_hybrid` exact for every k and vectorised, ~10–15 ns per
on-star sample vs ~2.7 ns for `occult_small`): the user accepted the speed
cost and the model change.

- The default `occult_small_threshold` is now 0 in `model_params` and
  `Likelihood` (exact Mandel–Agol for every moon); examples, notebooks,
  README and `docs/Interface.rst` use 0. A positive value still selects
  the approximation (opt-in).
- Even the old default 0.01 was not safe: moons just below it were
  1.9 / 3.0 / 4.4 ppm off mpmath at k = 0.007 / 0.0085 / 0.0099
  (LD 0.4, 0.25), up to 8.0 ppm with the perf workload's free LD.
- Perf workload (sampler likelihood, r_moon 0.007–0.012): moon flux change
  ≤ 3.6 ppm (Kepler) / 8.0 ppm (PLATO), |ΔlogL| ≤ 1.9; cost ~4–6 % per
  likelihood call (interleaved, same code).
- `examples/example.py` (r_moon = 0.03526, `light_curve()`): the moon flux
  error vs the exact model drops from 62 ppm to 1.0 ppm; what is left is the
  float32 occultation cache that `light_curve()` uses for fixed LD.
- The regression cases set their thresholds explicitly and are unchanged
  (they still cover the `occult_small` path). Pinned by
  `test_moon_exact_by_default` (defaults, examples/notebooks, and the
  default moon path equals the exact model; fails on `50cb806`).

## 10. `x_bary_grid()` epoch rounding truncates toward zero for data before t0 - FIXED

Severity: low/medium (only when data exist before `t0_bary`)
Location: `pandoramoon/grids.py:57`
Fixed in: "Fix epoch rounding for data before t0 (bug #10)"

```python
epoch = int((time[idx] - t0_bary) / per_bary + 0.5)
```

Python/numba `int()` truncates toward zero, so negative values in
`(-1, -0.5]` round to 0 instead of -1. Any epoch more than half a period
*before* `t0_bary` gets the wrong epoch index and a wildly wrong position
(verified: `x_bary ~ -1400` instead of ~+1 for a time one period before
t0). The fix uses `np.floor`.

## 11. `occult_hybrid` interpolation branch: dead `testv`, duplicate expensive call, missing clamp - FIXED

Severity: low (performance + code quality; missing clamp is cosmetic)
Location: `pandoramoon/occult.py:445-451`
Fixed in: "Remove dead testv and duplicate call; clamp interpolated flux (bug #11)"

```python
testv = occult_small_single_value(z, k, u1, u2)   # never used
flux[i] = (
    occult_small_single_value(z, k, u1, u2)        # computed a second time
    + interpol_flux_1 + interpol_flux_2 * z)
continue                                          # skips the flux > 1 clamp
```

The interpolation branch calls the (non-trivial) small-planet approximation
twice per point and discards the first result, and its result bypasses the
`flux > 1` clamp that all other branches apply. The fix computes it once and
clamps.

## 12. `occult` case-III q guard fires for every q - FIXED

Severity: low (numerically negligible, but wrong logic)
Location: `pandoramoon/occult.py:287-289`
Fixed in: "Fix case-III q guard condition in occult (bug #12)"

```python
if (q - 1) < 1e-8:
    q -= 1e-8
```

The intent is to push q away from the ellk singularity at q = 1, i.e. the
condition should be `1 - q < 1e-8`. As written, `(q - 1) < 1e-8` is true for
*every* q < 1 + 1e-8, so all case-III evaluations get their q shifted by
-1e-8. The induced flux error is O(1e-8) except within ~1e-8 of q = 1, where
the shift actually saves the evaluation - which is why the bug went
unnoticed. The fix corrects the condition; the affected q values still get
shifted.

## 13. `create_occult_cache()` ignores its own `k_min` - FIXED

Severity: low (dead constant, hardcoded duplicate)
Location: `pandoramoon/occult.py:11, 14`
Fixed in: "Use k_min constant in create_occult_cache grid (bug #13)"

`k_min = 0.001` is defined but the grid uses a hardcoded
`np.linspace(0.001, k_max, dim)`. The fix uses the constant (no behavior
change). Note the cache itself stores float32, quantizing fluxes to ~6e-8
(~0.06 ppm) - adequate for Kepler-class photometry, marginal for anything
much more precise.

## 14. `timegrid()`: duplicated line; invalid `supersampling_factor` only prints - FIXED

Severity: low
Location: `pandoramoon/grids.py:25-28`
Fixed in: "Validate supersampling_factor; remove duplicated line (bug #14)"

`supersampled_cadences_per_day` is assigned twice verbatim (refactoring
leftover), and a `supersampling_factor < 1` only prints a message before
silently building an empty/degenerate time grid. The fix removes the
duplicate and raises `ValueError`.

## 15. `ld_invert()` divides by zero for u1 + u2 == 0 - FIXED

Severity: low (edge case of a helper)
Location: `pandoramoon/helpers.py:24-26`
Fixed in: "Guard ld_invert against u1+u2 == 0 (bug #15)"

`q2 = u1 / (2 * (u1 + u2))` raises `ZeroDivisionError` for the degenerate
Kipping case (u1, u2) = (0, 0). The fix returns `(0.0, 0.5)`, which maps back
to (0, 0) under `ld_convert` (any q2 does once q1 = 0).

## 16. `occult_hybrid` leaves `le/ld/ed` uninitialized for k > 1, z < k - 1 - FIXED

Severity: low (unphysical k > 1, but pandora() has no guard against it)
Location: `pandoramoon/occult.py:373-489` (missing complete-occultation branch)
Fixed in: "Handle complete occultation (k >= 1) in occult_hybrid (bug #16)"

`occult` returns flux 0 for `k >= 1 and z <= k - 1` (star fully behind the
occulter); `occult_hybrid` has no such branch, so for k > 1 none of the
lex/ldx/edx cases fire and the flux is uninitialized memory. Verified:
`occult_hybrid([0.0, 0.05, 0.1, 0.19], 1.2)` returned 0.2217 for all points
(stale memory) where the correct value is 0. `pandora()` does not currently
guard `r_planet >= 1` or `r_moon >= 1`, so this path is reachable with
nonsense parameters. The fix adds the same complete-occultation branch as
`occult`.

## 17. `resample()` drops the trailing partial group - FIXED (with #1)

Severity: low
Location: `pandoramoon/helpers.py:8`
Fixed in: "Fix off-by-one in resample() supersampling average (bug #1)"

`int(len(arr) / factor)` truncates, so a trailing group shorter than
`factor` was silently discarded (e.g. cadences = 98 with factor 5 loses the
last 3 supersamples). The fix includes the partial group in the output.

## 18. `ellpicb()` silently returns 0 on non-convergence - FIXED (eclipse limitation documented)

Severity: low
Location: `pandoramoon/occult.py:125-138`
Fixed in: "Raise on ellpicb non-convergence instead of silent 0 (bug #18)"

If the Bulirsch iteration does not converge within 1000 rounds the function
returns 0, which flows into the limb-darkening terms as a silently wrong
value. In practice convergence takes < 10 iterations, but the failure mode
should be loud. The fix raises `ValueError`.

## 19. Examples set `params.e_moon`, but the model reads `params.ecc_moon` - FIXED

Severity: medium (examples silently ignored the moon eccentricity)
Location: `examples/example.py:29`, `examples/example_figure.py:29`,
`examples/example_video.py:29`, `examples/example.ipynb`,
`examples/injection_retrieval_simple_ultranest.ipynb`; attribute defined as
`ecc_moon` in `pandoramoon/pandora.py:42` and read in `moon_model.__init__`.
Fixed in: "Fix e_moon/ecc_moon attribute name in examples; use physical e=0.1 (bug #19)"

Discovered during final verification of this patch series: the standard
example output was bit-identical before and after fixing bug #8, although it
claims `e_moon = 0.9` and should have exercised `ellipse_ecc()`. The reason:
all examples assign `params.e_moon = 0.9`, but `model_params` defines the
attribute as `ecc_moon` (default 0) and `moon_model` reads `params.ecc_moon`.
The misspelled attribute is silently ignored, so the examples actually model a
*circular* moon orbit via `ellipse()`; the intended eccentric case never runs.
(The injection-retrieval notebook uses `e_moon = 0`, which coincides with the
effective default, so its injection is unaffected.)

Fixed together with bug #2 as planned: the attribute is renamed to
`ecc_moon` in all three example scripts and both notebooks. The tutorial
examples' value was changed from 0.9 to 0.1: the 0.9 was never actually
exercised (this bug) and is physically invalid for this configuration - at
e = 0.9 the moon's periastron (a*(1-e) ~ 0.019 R_star) lies deep inside the
planet (r_planet = 0.1 R_star), so the moon would pass through the planet
once per orbit. At e = 0.1 the periastron (0.170 R_star) stays comfortably
outside r_planet + r_moon = 0.135 R_star. The injection-retrieval notebook
keeps its (circular) e = 0 under the corrected name.

---

Bugs #20-#24 were found in a second audit pass (2026-10-06). Unlike the
first pass, which cross-checked `occult` against `occult_hybrid` (both share
the same Mandel-Agol formulas and can be wrong in the same way), this pass
compared against an independent reference: direct radial integration of the
quadratic limb-darkened disk (400k-point midpoint rule, ~0.01 ppm accurate).

## 20. `occult_hybrid` returns NaN at z = |1 - k| (unclamped arccos argument) - FIXED

Severity: high (silent NaN in the light curve)
Location: `pandoramoon/occult.py:411-412` (also `250-251` in `occult`)
Fixed in: "Clamp arccos arguments in occult/occult_hybrid partial case (bug #20)"

```python
kap0 = arccos(min((k2 + z2 - 1) / (2 * k * z), 1))
```

`min()` clamps the argument only from above. At z = 1 - k the argument is
mathematically exactly -1; float rounding gives e.g. `-1.000000000000002`,
so `arccos` returns NaN and the flux point is NaN. Verified: NaN for 69 of
199 k values in (0, 1) at z = 1 - k, and 1 NaN in a 2001-point ingress grid
at k = 0.01. `kap1` has the same issue at z = k - 1 for k > 1. `occult`
survives only because its `+ epsilon` in the denominator shrinks the
argument just inside [-1, 1].

Impact: the injection-retrieval notebook calls `pandora.pandora()` without a
cache, so every planet goes through `occult_hybrid`, and its likelihood uses
`np.nansum`, which silently drops NaN points and biases the fit. The
previous boundary sweep checked only for exceptions, not NaN.

The fix clamps both arccos arguments to [-1, 1] in both functions.

## 21. Case-IV special-case band too wide: up to 21 ppm error and a flux discontinuity - FIXED

Severity: medium-high (systematic model error; discontinuous likelihood)
Location: `pandoramoon/occult.py:336` (`occult`), `490` (`occult_hybrid`)
Fixed in: "Narrow case-IV k+z=1 special-case band to 1e-8 (bug #21)"

```python
if abs(k + z - 1) < 1e-4:
    ld = <value exactly at k + z = 1>
```

Within 1e-4 of z = 1 - k the limb-darkening term is replaced by its
boundary value, so the flux is constant across the band and jumps at its
edge. Measured against the independent reference just inside the band
edge (jump at the edge is the same size):

| k    | error / jump |
|------|--------------|
| 0.05 | 1.0 ppm      |
| 0.1  | 2.8 ppm      |
| 0.2  | 7.5 ppm      |
| 0.3  | 13 ppm       |
| 0.45 | 21 ppm       |

The general case-IV formula stays accurate to < 0.03 ppm down to 1e-8 from
the boundary (degrading only at ~1e-10). The fix narrows the tolerance to
1e-8 in both functions, where the special value's own error is ~1e-3 ppm.

## 22. Supersampling groups straddle epoch boundaries - FIXED

Severity: medium (wrong supersampled light curves for common settings)
Location: `pandoramoon/grids.py:29` with `pandoramoon/helpers.py:resample`
Fixed in: "Make supersamples per epoch a multiple of supersampling_factor (bug #22)"

```python
cadences = int(supersampled_cadences_per_day * epoch_duration)
```

The number of supersamples per epoch is generally not a multiple of
`supersampling_factor`, but `resample()` groups the flattened array in
blocks of `factor` straight across epochs. Verified: cpd = 48,
dur = 0.6 d, ssf = 5 gives 144 supersamples per epoch, and group 28
averages a sample at t = 11.29 d with samples at t = 375.96 d; every later
group is misaligned too. The fix uses
`cadences = int(cadences_per_day * epoch_duration) * int(supersampling_factor)`,
so each epoch holds exactly `int(cpd * dur)` complete groups.

## 23. Physical-plausibility check ignores moon eccentricity - FIXED

Severity: medium (unphysical orbits accepted)
Location: `pandoramoon/pandora.py:382, 390`
Fixed in: "Use periapsis/apoapsis in moon plausibility checks (bug #23)"

The Hill-sphere check (`a_moon / r_hill`) and the planet-contact check
(`a_moon < r_planet + r_moon`) both use the semi-major axis. For an
eccentric moon the relevant distances are the periapsis a(1 - e) (contact)
and apoapsis a(1 + e) (Hill sphere). Verified: e_moon = 0.9 (the original
example value) is accepted as physical, yet the minimum planet-moon
separation is 0.019 R_star vs r_planet + r_moon = 0.135 R_star - the moon
passes through the planet every orbit. The fix uses a(1 + e) / r_hill and
a(1 - e) < r_planet + r_moon (identical to before for e = 0).

## 24. `video()` draws the stellar disk too small - FIXED

Severity: low (visualization only)
Location: `pandoramoon/pandora.py:184-185`
Fixed in: "Draw full-size stellar disk in video() (bug #24)"

`for i in reversed(range(ld_circles)): impact = i / ld_circles` makes the
largest limb-darkening circle radius (n - 1)/n: 0.99 R_star for the
default 100 circles, 0.9 for 10, so planets near the limb appear off-star
while the model has them on-star. Each annulus is also coloured with the
intensity of its outer edge. The fix draws radii (i + 1)/n, each coloured
with the intensity at its annulus mid-radius.

## 25. Exact first/last contact z = 1 + k left the limb-darkening terms unset - FIXED

Severity: medium (garbage flux at a single, exactly-hit point)
Location: `pandoramoon/occult.py:228` (`occult`), `393` (`occult_hybrid`)
Fixed in: "Treat z = 1+k (contact point) as unocculted (bug #25)"

Found while adding unit tests (a cache-accuracy test failed only on cold
runs). The unocculted branch used `z > 1 + k` while case III requires
`z < 1 + k`, so at exactly z = 1 + k no branch sets `ld`/`ed`:
`occult` then returns uninitialised `np.empty` memory (non-deterministic;
observed -0.43 and -5e98), `occult_hybrid` reuses the previous element's
`ldx`/`edx` (e.g. 0.960 instead of 1 for k = 0.3). The exact value at the
contact point is 1 (zero overlap). Real time grids essentially never hit the
point exactly, but synthetic grids do (e.g. `np.linspace(0, 1 + k, n)`), as
does any caller evaluating the contact point. The fix uses `z >= 1 + k` in
both functions. Verified by `test_occult_first_contact_is_unity` in
`../test_cases/unit_tests.py` (fails on the old code every run); the
light-curve regression reference is unchanged.

## 26. `timegrid()` cadence spacing is not `1 / cadences_per_day` - FIXED

Severity: medium (every grid built by `pandora.time(...).grid()` is off;
supersampled exposures have the wrong length)
Location: `pandoramoon/grids.py:30` (`timegrid`)
Fixed in: "Space timegrid() cadences by 1/cadences_per_day, centre sub-exposures (bug #26)"

```python
cadences = int(cadences_per_day * epoch_duration) * int(supersampling_factor)
time[epoch] = np.linspace(t_starts[epoch], t_ends[epoch], cadences)
```

`linspace` includes both window edges, so the sample spacing is
`epoch_duration / (cadences - 1)` instead of
`1 / (cadences_per_day * supersampling_factor)`. Because `int()` also
truncates the cadence count, the remaining fraction of a cadence is spread
over all samples too. With supersampling the sub-samples are spread evenly
over the whole window rather than centred inside each exposure: the first
group starts exactly at `t_start`, and the group span (the modelled exposure
time) is wrong.

Measured:
- 250/day over 0.6 d: 5.7987 min instead of 5.76 min (+0.67 %).
- With ssf = 5, the cadence spacing is 5.7677 min, the first cadence
  centre is 2.307 min after `t_start` (should be 2.88), and the sub-step
  is 1.1535 min (should be 1.152).
- Kepler LC (29.4244 min) over 1 d: 30.13 min (+2.4 %).
- The 10 regression cases have cadence errors between +0.06 % (Kepler SC)
  and +1.5 % (Kepler LC, ssf = 3).
- For the two supersampled Kepler LC cases, the flux at the same cadence
  mid-times differs from correctly placed sub-exposures by up to 2.9 ppm
  (ssf = 5) and 4.0 ppm (ssf = 3). This is the exposure-smearing error alone.

Impact: model-vs-data fits that pass the user's own time stamps are
unaffected, apart from the supersampling convention, which `pandora()`
leaves to the caller. Anything that relies on `grid()` reproducing an
instrument cadence is affected: injection–retrieval tests, simulated
data, and supersampled exposure smearing.

Fix: the cadences are spaced exactly `1 / cadences_per_day`. Each cadence
holds `supersampling_factor` sub-exposure mid-times, spaced
`1 / (cpd * ssf)`, whose mean is the cadence mid-time. The block of
`int(cpd * dur)` cadences is centred on the epoch mid-time, like the old
grid. When `cpd * dur` is an integer, the exposures tile
`[t_start, t_end]` exactly. The number of points is unchanged, so bug #22's
epoch-aligned groups still hold. Verified by
`test_timegrid_cadence_spacing` in `../test_cases/unit_tests.py`, which
fails on the old code. Feeding the old `linspace` grid into the fixed
code reproduces the old regression reference bit-for-bit, so the time grid
is the only change. The reference was regenerated deliberately for the new
grid.

Not changed here: `int(cpd * dur)` truncates float round-off (bug #27).

## 27. `timegrid()` loses a cadence when `cadences_per_day * epoch_duration` rounds down - FIXED

Severity: low (one cadence missing per epoch for common settings)
Location: `pandoramoon/grids.py:29` (`timegrid`)
Fixed in: "Tolerate float round-off in timegrid() cadence count (bug #27)"

```python
cadences = int(cadences_per_day * epoch_duration) * int(supersampling_factor)
```

`int()` truncates. When the product is meant to be an integer but the
float result lands just below it, one cadence per epoch is lost. The grid
then fills only `(n - 1) / n` of the requested window, though it stays
centred. Verified:
- `int(100 * 0.29)` = 28 instead of 29;
- `int(720 * 0.35)` = 251 instead of 252 (TESS 2 min over 0.35 d);
- 34 of the 3588 pairs of common cadences (24 ... 4320 per day, Kepler LC)
  and durations of 0.01 ... 2.99 d in steps of 0.01 d are affected.

None of the regression cases are affected: their products have fractional
parts of 0.33-0.99. Because of bug #22, `n_super` is a multiple of
`supersampling_factor`, so the missing cadence never splits a supersample
group.

Fix: `int(x + 1e-9 * max(1, x))` with `x = cadences_per_day * epoch_duration`.
Double-precision round-off in `x` is ~1e-16 relative, far below this
tolerance. A product that truly falls short of the next integer by less than
1e-9 cadences (relative) now gets that cadence. The exposures then overrun
the window by < 1e-9 cadence, which is negligible. Verified by
`test_timegrid_cadence_count_roundoff` in `../test_cases/unit_tests.py`
(fails on the old code). The light-curve regression reference is unchanged.

## 28. `occult_hybrid` small-body "linear interpolation" zone: wrong formula, up to 2.6 ppm - FIXED

Severity: low (≤ 2.6 ppm; ≤ 0.4 ppm for k ≈ 0.01 moons)
Location: `pandoramoon/occult.py`, `occult_hybrid` (k ≤ 0.05)
Fixed in: "Exact occult_hybrid for all k: drop the small-body interpolation zone (bug #28)"

For k ≤ 0.05 and z ≤ 0.65 ... 0.98 (the limit depends on k), `occult_hybrid`
did not evaluate the exact model. It returned the small-body approximation
plus a correction meant to interpolate linearly between the exact values at
z = 0 and z = 0.65:

```python
flux = occult_small_single_value(z, ...) + interpol_flux_1 + interpol_flux_2 * z
# interpol_flux_1 = exact(0) - small(0), interpol_flux_2 = exact(0.65) - small(0.65)
```

That is `c0 + c065 z`, not the interpolation `c0 + (c065 - c0) z / 0.65`.
Verified against mpmath (radial quadrature, 50 digits), 2394 points over
k ∈ [0.001, 0.05] and 7 limb-darkening pairs incl. u2 < 0:
- old code: up to 2.56 ppm (k = 0.05, u = (1.6, -0.6)), 1.49 ppm at
  k = 0.02, 0.38 ppm at k = 0.01;
- corrected formula: still up to 1.47 ppm, so fixing the formula is not
  enough;
- exact model: ≤ 0.037 ppm, only at the existing z = 1 - k nudge.

The zone existed only for speed. After P3b (vectorised exact elliptic
integrals) the exact model is cheap, so the zone is removed and
`occult_hybrid` is exact for every k. `occult_small()` (used below
`occult_small_threshold`) is a separate approximation and unchanged
(bug #9). Model change: regression cases ≤ 7.0e-8, perf draws ≤ 9.5e-8,
random wide draws ≤ 1.15e-6 (small planets/moons without cache), all toward
the exact value. Pinned by `test_ell_combo_and_occultation_exact` (mpmath
fluxes at k = 0.003, 0.01, 0.03 inside the former zone; fails on the old
code: 0.74 ppm at k = 0.01, z = 0.297).

## 29. Case III / Case IV conditions leave a 1-ulp gap at z = 1 - k - FIXED

Severity: low for `occult_hybrid` (stale values), medium for `occult`
(uninitialised memory, but only at isolated float values)
Location: `pandoramoon/occult.py`, `occult` and (before P3b) `occult_hybrid`
Fixed in: "P3b: vectorised exact Mandel-Agol pass in occult_hybrid" (`occult_hybrid`),
"Fix the 1-ulp Case III/IV gap at z = 1 - k in occult() (bug #29)" (`occult`)

Case III runs for `z >= 0.5 + abs(k - 0.5)` and Case IV for `z < 1 - k`.
For k < 0.5 both bounds are 1 - k mathematically, but in floating point
`0.5 + (0.5 - k)` can round one ulp above `1 - k`. A sample with z equal to
the lower value then matches neither case. Verified: this happens for 388 of
20000 random k ∈ [0.001, 0.5], e.g. k = 0.051, z = 0.949.
- `occult_hybrid` (≤ `055fbb6`): the limb-darkening terms of the
  *previous* sample were reused. Wrong by an arbitrary amount when that
  sample was elsewhere (within ~0.04 ppm only when it was the adjacent grid
  point).
- `occult`: `ld[i]` and `ed[i]` come from `np.empty` and stay
  uninitialised: 150 ppm ... 17 % errors in a test (`scratch_p3/gap_check.py`).
  `occult` builds the occultation cache, and the cache grid only hits such
  a z by coincidence (the default caches are bit-identical before and
  after the fix). Fixed like `occult_hybrid`: Case III is `1 - k <= z < 1 + k`,
  the exact complement of Case IV. On a dense (z, k, LD) grid
  (846k values, `scratch_p3/occult_dump.py`) only the 14 gap points
  changed. No speed impact (`occult` is not on the likelihood path;
  `create_occult_cache(dim=300)` 3.4 ms before and after).

P3b rewrote `occult_hybrid` as "Case IV if z < 1 - k, else Case III", which
has no gap. Pinned by `test_hybrid_second_contact_gap` (both functions; for
`occult`: continuity with the next float, since it still uses the ellk/ellec
polynomials, ~2 ppm near z = 1 - k for k ~ 0.001).

## 30. `occult_hybrid` / `occult` replace k by 0.5 for |k - 0.5| < 1e-4: up to 150 ppm - FIXED

Severity: medium for k within 1e-4 of 0.5 (giant planets around small
stars), none otherwise
Location: `pandoramoon/occult.py`, first lines of `occult_hybrid` and `occult`
Fixed in: "Do not snap k to 0.5 in occult_hybrid / occult (bug #30)"

Both functions started with `if abs(k - 0.5) < 1e-4: k = 0.5`, a
"numerical stability" guard of the original PyTransit-derived code (the
Table 3 Case V/VI closed forms at z = k need the modulus 2k or 1/(2k) < 1).
For k in (0.4999, 0.5001) the model therefore described a body of radius
0.5 instead of k. Verified against mpmath (radial quadrature, dps 30;
`scratch_k05/scan.py`), at k = 0.5 -+ 1e-4:
- the error is largest at mid-transit of a central transit (z = 0), where
  the occulted disk is fully on the star: 2 k |dk| I(r = k) / <I>, i.e.
  -/+114 ppm for u = (0.4, 0.25), 100 ppm for uniform LD, 130 ppm for
  u = (1, 0), **150 ppm for u = (2, -1)** (the edge of the physical LD
  range); about 4.6e-4 of the transit depth;
- fully on the star (z <= 1 - k) the error depends only on I along the
  body's rim; on the limb it falls to ~42 ppm at z = 1 and to 0 at first
  contact; it scales linearly with |k - 0.5| (half at 5e-5).

Since P15 `occult_hybrid` evaluates k = 0.5 and z -> k exactly (lanes the
fixed-step batch does not converge go to the scalar fallback), so the snap
is simply removed: now exact (<= 1e-10; <= 3e-9 at z = 1 - k = k = 0.5
exactly, the limb area's arccos). In `occult` the Case V/VI values were
always overwritten by Case III/IV, so the snap is removed there too
(Case VI is skipped for 2k >= 1); `occult` is then within its ellk/ellec
polynomial accuracy (<= 0.5 ppm near z = 0.5). No regression case or perf
draw has k near 0.5: outputs there are bit-identical, speed unchanged.
Pinned by `test_no_k_snap_near_one_half` (embedded mpmath fluxes at
k = 0.5 -+ 9e-5, -+ 1e-6 and 0.5; fails on the old code by 103 ppm).

## 31. Parameters outside their physical range were not rejected (per_moon < 0: Likelihood 175 ppm off) - FIXED

Severity: low (only for out-of-range input; sampler priors in the
examples and the benchmark stay inside), but silently wrong results
Location: `pandoramoon/pandora.py`, the public model functions and
`pandora_loglike`
Fixed in: "Reject parameters outside their physical range (bug #31)"

No entry point checked its parameters. Found in R6 (2026-10-08): for
per_moon < 0 the block anchors of the run-based orbit loops (P24d,
`ellipse_z_runs` / `ellipse_z_only_runs`) advance the phase by
abs(2 pi / per) while the phase itself is (t - tau per) / per, so the
anchored positions run the wrong way (0.5 R_star off `ellipse_z`), and
`Likelihood` differed from `pandora_flux()` by up to 175 ppm (40 Kepler
draws with per_moon negated; positive periods: 2.3e-14). A negative period
has no meaning in Pandora (the orbit size uses per^2; a retrograde moon is
i_moon in (90, 180] deg). Other unchecked inputs gave silent nonsense too,
e.g. tau_moon = NaN returned a finite log-likelihood.

Fix: `params_valid()` (jitted, compiled without fastmath so the NaN / inf
tests survive) runs before any computation in `pandora()`,
`pandora_flux()`, `pandora_deficit()`, `pandora_window()` (raise
ValueError) and `pandora_loglike()` / `Likelihood` (return -inf, the usual
"outside the prior" value for samplers). Ranges: all parameters finite;
R_star, per_bary, a_bary, M_planet, per_moon, epoch_distance > 0;
r_planet, r_moon, M_moon >= 0; 0 <= ecc_bary, ecc_moon < 1; physical
quadratic LD (Kipping 2013) u1 >= 0, u1 + u2 <= 1, u1 + 2 u2 >= 0 (1e-12
slack for converted q1, q2: `ld_convert` stays inside on 1.2 M (q1, q2)
pairs incl. the edges); supersampling_factor, numerical_grid >= 1.
`pandoramoon.check_params(...)` lists the violated rules. Moon orbits that
are physically implausible (Hill sphere, contact) keep the planet-only
model as before. No output change for valid parameters (regression suite,
logL bit-identical); speed unchanged (same-process A/B 0.998x / 1.000x).
Pinned by `test_params_valid` (every rule just inside and just outside its
boundary, NaN / +-inf in every parameter, all five entry points; the
jitted check equals `check_params` on 3000 random draws; mutations caught:
a dropped rule, fastmath on the check, a dropped finiteness test).

## Known limitations (documented, not bugs)

- `eclipse()` jumps by T4's gradient term at er -> 1 on the stellar limb:
  the term is skipped only for er == 1 exactly, and a limb moon almost
  fully behind the planet gets `er_fast` = 1 - O(1e-15) or exactly 1, so
  positions that differ by an ulp can change the moon flux by the term
  (1.4e-7 in one of 3000 random test draws, R6c in PERFORMANCE_LOG.md
  section 45). Near planet-moon tangency the lens area also has ~1e-13
  rounding noise. Both are far below the 1 ppm rule and the T4 accuracy
  (<= 0.6 ppm on the limb).
- Mutual eclipses are independent of which body is in front: this is
  correct, not a limitation. (An earlier version of this list said the
  overlap was double-counted when the moon is in front. That was wrong.)
  Both bodies are opaque disks, so the star loses the light under the
  *union* of their shadows, and the union does not depend on the order.
  `eclipse()` adds the light under the overlap back to `flux_moon`, so
  `flux_total = flux_planet + flux_moon - 1` subtracts the overlap exactly
  once. It uses only sky-plane positions, so the result is the same for
  both orders by construction. The only order-dependent choice is
  bookkeeping: the overlap is always charged to the moon component. Read
  `flux_moon` alone as "the moon's signal" only with that in mind. The
  order would matter only for non-opaque bodies or the bodies' own light
  (atmospheres, rings, reflected or thermal emission), which Pandora does
  not model.
- The mutual-eclipse correction scales the moon's limb-darkened deficit by
  the *geometric* fraction of its on-star area covered by the planet, plus
  (since T4, "Eclipses: first-order limb-darkening gradient across the
  overlap") a first-order term `lens_gradient_term()`: the intensity
  gradient at the moon centre times the offset of the overlap's centroid,
  tapered to zero across the limb band. It is not an exact LD-weighted
  overlap. Measured against an independent radial integration of the union
  of both disks (gefera-style exact result), Jupiter-sized planet: max error
  Earth-sized moon (k = 0.0092) 1.36 -> 0.20 ppm (moon fully on the star),
  1.37 -> 0.58 ppm (limb band); k = 0.035: 34 -> 5.9 / 36 -> 16 ppm;
  rms 4-30x smaller; for all LD laws tested. `eclipse_method="pixelmap"`
  keeps the old treatment.
- Overlap *area* on the stellar limb: since the analytic method became the
  default (`eclipse_method="analytic"`, `pandoramoon/overlap_analytic.py`),
  it is exact to < 1e-6 of the moon area against an independent
  integration. The legacy `eclipse_method="pixelmap"` (`pixelart()` on a
  `numerical_grid` raster, default 25) is off by up to ~1 % of the moon area
  (er itself is off by up to 1.0 when only a sliver of the moon is on the
  star). In the regression cases that is up to 4.4 ppm (case 6) and
  0.07 ppm spurious moon signal in case 7, where the moon is fully hidden.
- The occultation cache is float32 and bilinear in (z, k): ~0.06 ppm
  quantization plus interpolation error of a few ppm near ingress (measured
  worst case 3.7 ppm over k in [0.002, 0.099]).
- `pandora()` has no guard against `r_planet >= 1` / `r_moon >= 1`
  (now handled inside `occult_hybrid` via bug #16's fix) or negative radii.
- `ellipse_ecc()` has no guard against `e_moon >= 1`: at e = 1 the true
  anomaly factor `sqrt((1+e)/(1-e))` divides by zero (ZeroDivisionError in
  numba), and e > 1 is a hyperbolic orbit outside the model's scope. The
  starter/corrector has also only been validated for e < ~0.99.
- `video()` returns an unsaved `FuncAnimation` whose callback updates a
  `tqdm` progress bar. If the animation is never saved (or the figure
  closed) before the interpreter exits, Python segfaults at shutdown
  (exit 139). Reproduced without any Pandora code (plain matplotlib
  `FuncAnimation` + `tqdm` in the callback), so this is a third-party
  teardown interaction, not a model bug; saving the video (as all examples
  do) or calling `plt.close('all')` avoids it. Cosmetic: the progress bar
  also overcounts (e.g. 453/450) because matplotlib draws an extra init
  frame.

## Verification

All fixes were verified with targeted numerical tests (repros above plus
consistency checks `occult` vs `occult_hybrid` vs `occult_small` across
k in [0.001, 0.95] and full z ranges, and a reference Newton Kepler solver),
followed by a full `examples/example.py` run (light curve + 450-frame video).

The model output changes relative to the base commit only for:
- supersampled light curves (bug #1 - previously wrong),
- bodies with k < 0.001 used through the cache (bug #3 - previously erased),
- eccentric moon orbits (bug #2 - previously wrong near the periastron
  recurrence; bug #8 - previously mirrored by 180 deg),
- the exact boundary/crash points (bugs #4-#7, #16 - previously garbage or
  exceptions).

Through the first fix round the standard `examples/example.py` run was
bit-identical to the base commit (its planet uses `occult_hybrid`, its moon
ran the default circular orbit because of bug #19, and none of its (z, k)
values hit the fixed boundary points), confirming those fixes were
non-regressive on the normal path.

After fixing bugs #2 and #19, the example exercises the eccentric solver
for real (e_moon = 0.1), so its moon light curve now differs from the base
commit - by design: it is now the correctly computed Keplerian orbit. The
planet light curve is unchanged to numerical precision. Everything else
(including the standard example planet light curve) is
unchanged to numerical precision.

Second pass (#20-#24): after the fixes, `occult` and `occult_hybrid` agree
with the independent radial-integration reference to <= 0.5 ppm (largely
reference-limited) over k = 0.01-0.7, for random z plus the boundary points
z in {1 - k, |1 - k|, k, 1 + k}, with no NaN; the cache stays within ~3 ppm.
The 18-check regression suite and the bug #2 orbit tests still pass, and
`examples/example.py` (light curve + 450-frame video) runs and exits
cleanly. Bug #22 changes the time grid only for `supersampling_factor > 1`;
bug #23 only rejects orbits whose periapsis/apoapsis violate the existing
limits; bugs #20/#21 change flux only at the former NaN points and within
1e-4 of z = 1 - k.

Bug #26 changes every time grid built with `pandora.time(...).grid()`, and
with it the light-curve regression reference. The model functions
themselves are unchanged: the old grid gives bit-identical old output.
