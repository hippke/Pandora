"""Regression test cases for Pandora (pandoramoon).

Each case defines a realistic star/planet/moon system and an observing setup.
`run_case(case)` builds the light curve with the installed (bug-fixed)
Pandora and returns arrays; `fingerprint(...)` reduces them to a compact,
tolerance-comparable summary that is stored in `reference.json`.

See README.md for the design rationale.
"""
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
# The tests run against the installed pandoramoon. Set PANDORA_PATH to a source checkout to test that instead.
_repo = os.environ.get("PANDORA_PATH", "")
if _repo and os.path.isdir(_repo) and _repo not in sys.path:
    sys.path.insert(0, _repo)

import pandoramoon as pandora  # noqa: E402
from pandoramoon.helpers import resample  # noqa: E402

# ----------------------------------------------------------------------------
# Physical constants and bodies
# ----------------------------------------------------------------------------
G = 6.67408e-11          # same value as pandora.py
DAY = 86400.0
R_SUN_M = 696_342_000.0  # [m]
KM = 1000.0

R_JUPITER, M_JUPITER = 71492 * KM, 1.8986e27
R_NEPTUNE, M_NEPTUNE = 24622 * KM, 1.0243e26
R_EARTH, M_EARTH = 6371 * KM, 5.9722e24
R_LUNA, M_LUNA = 1737.4 * KM, 7.342e22
R_GANYMEDE, M_GANYMEDE = 2634.1 * KM, 1.4819e23

# Limb darkening (quadratic): Sun-like and M-dwarf/giant-ish values
LD_SUN = (0.4089, 0.2556)
LD_MDWARF = (0.25, 0.35)
LD_GIANT = (0.60, 0.15)

# Cadences [per day]
KEPLER_LC = 1440.0 / 29.4244   # 29.4 min long cadence
KEPLER_SC = 1440.0 / 0.98074   # 58.8 s short cadence
TESS_FFI10 = 144.0             # 10 min FFI (sectors 27-55)
TESS_FFI200 = 432.0            # 200 s FFI (sector 56+)
TESS_2MIN = 720.0              # 2 min postage stamps
TESS_20S = 4320.0              # 20 s fast cadence
PLATO_1MIN = 1440.0            # 1 min (PLATO cadence as specified)

# Mutual-eclipse geometry flags used by the test runner
ECLIPSE_REQUIRED = "required"
ECLIPSE_NONE = "none"


def moon_semimajor_rstar(M_planet, M_moon, per_moon, R_star):
    """Planet-moon semi-major axis in stellar radii (Pandora's formula)."""
    return (G * (M_planet + M_moon) / (2 * np.pi / (per_moon * DAY)) ** 2) ** (1 / 3) / R_star


def tdur_b0(per_bary, a_bary, ecc_bary=0.0, w_bary=0.0):
    """Pandora's internal crossing time of one stellar diameter (x_bary_grid)."""
    t = per_bary / np.pi * np.arcsin(1 / a_bary)
    if ecc_bary > 0:
        t /= 1 / np.sqrt(1 - ecc_bary ** 2) * (1 + ecc_bary * np.cos((w_bary - 90) / 180 * np.pi))
    return t


def tau_for_conjunction(t_conj, per_moon):
    """Circular orbit, Omega=0: moon passes behind/in front of the planet
    (sky separation minimal) when theta = 2pi(t - tau*P)/P = pi/2."""
    return ((t_conj - per_moon / 4) / per_moon) % 1.0


# ----------------------------------------------------------------------------
# Case definitions
# ----------------------------------------------------------------------------
def _base(R_star, ld, per_bary, a_bary, R_p, M_p, b, R_m, M_m, per_moon,
          i_moon=90.0, Omega_moon=0.0, ecc_moon=0.0, w_moon=0.0, tau_moon=0.0,
          ecc_bary=0.0, w_bary=0.0, t0_bary=10.0, t0_bary_offset=0.0,
          epochs=1, epoch_distance=None, cadence=KEPLER_LC, ssf=1,
          occult_small_threshold=0.01, hill_sphere_threshold=1.2,
          use_cache=True):
    return dict(
        R_star=float(R_star), u1=float(ld[0]), u2=float(ld[1]),
        per_bary=float(per_bary), a_bary=float(a_bary),
        r_planet=float(R_p / R_star), M_planet=float(M_p), b_bary=float(b),
        w_bary=float(w_bary), ecc_bary=float(ecc_bary),
        t0_bary=float(t0_bary), t0_bary_offset=float(t0_bary_offset),
        r_moon=float(R_m / R_star), M_moon=float(M_m), per_moon=float(per_moon),
        tau_moon=float(tau_moon), Omega_moon=float(Omega_moon), i_moon=float(i_moon),
        ecc_moon=float(ecc_moon), w_moon=float(w_moon),
        epochs=int(epochs),
        epoch_distance=float(per_bary if epoch_distance is None else epoch_distance),
        cadences_per_day=float(cadence), supersampling_factor=int(ssf),
        occult_small_threshold=float(occult_small_threshold),
        hill_sphere_threshold=float(hill_sphere_threshold),
        use_cache=bool(use_cache),
    )


def _window(c, slack=1.25):
    """Epoch window [days]: planet T14 + moon excursion, plus slack and drift."""
    td = tdur_b0(c["per_bary"], c["a_bary"], c["ecc_bary"], c["w_bary"])
    a_m = moon_semimajor_rstar(c["M_planet"], c["M_moon"], c["per_moon"], c["R_star"])
    a_m *= (1 + c["ecc_moon"])
    half_chord = np.sqrt(max((1 + c["r_planet"]) ** 2 - c["b_bary"] ** 2, 0.0))
    half_extent = half_chord + a_m + c["r_moon"]          # [R_star] along x
    half = half_extent * td / 2                            # [days]
    drift = abs(c["per_bary"] - c["epoch_distance"]) * (c["epochs"] - 1)
    return 2 * (half * slack + drift + abs(c["t0_bary_offset"]))


def _finish(name, description, c, eclipse=ECLIPSE_REQUIRED, slack=1.25):
    c["epoch_duration"] = float(_window(c, slack))
    c["name"] = name
    c["description"] = description
    c["eclipse"] = eclipse
    return c


def _ingress_time(c):
    td = tdur_b0(c["per_bary"], c["a_bary"], c["ecc_bary"], c["w_bary"])
    return c["t0_bary"] + c["t0_bary_offset"] - td / 2 * np.sqrt(max(1 - c["b_bary"] ** 2, 0.0))


def build_cases():
    cases = []

    # 1. Jupiter + Ganymede, Kepler long cadence with 5x supersampling, 3 epochs,
    #    epoch_distance != per_bary (drift). Mutual eclipse at mid-transit (ep. 0).
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.3,
              R_GANYMEDE, M_GANYMEDE, 7.1546, epochs=3, epoch_distance=365.20,
              cadence=KEPLER_LC, ssf=5)
    c["tau_moon"] = tau_for_conjunction(c["t0_bary"], c["per_moon"])
    cases.append(_finish("sun_jupiter_ganymede_kepler_lc_ssf5",
        "Sun/Jupiter/Ganymede, Kepler LC (29.4 min) with 5x supersampling, 3 epochs, "
        "per_bary != epoch_distance (drift); mutual eclipse at mid-transit of epoch 0; "
        "planet via occult_hybrid (k=0.103), moon via occult_small.", c))

    # 2. Same system, Kepler short cadence, direct pandora() call WITHOUT cache
    #    (notebook path), moon forced through occult_hybrid; eclipse at ingress (limb).
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.3,
              R_GANYMEDE, M_GANYMEDE, 7.1546, cadence=KEPLER_SC,
              occult_small_threshold=0.0, use_cache=False)
    c["tau_moon"] = tau_for_conjunction(_ingress_time(c), c["per_moon"])
    cases.append(_finish("sun_jupiter_ganymede_kepler_sc_nocache_limb",
        "Sun/Jupiter/Ganymede, Kepler SC (58.8 s), pandora() without cache (all bodies "
        "via occult_hybrid); mutual eclipse while the planet crosses the stellar limb "
        "(analytic limb path).", c))

    # 3. Earth + Luna, PLATO 1 min, 2 epochs. Planet via cache, moon via occult_small.
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_EARTH, M_EARTH, 0.1,
              R_LUNA, M_LUNA, 27.3217, epochs=2, cadence=PLATO_1MIN)
    c["tau_moon"] = tau_for_conjunction(c["t0_bary"], c["per_moon"])
    cases.append(_finish("sun_earth_luna_plato_1min",
        "Sun/Earth/Luna, PLATO 1 min cadence, 2 epochs; mutual eclipse at mid-transit "
        "of epoch 0; planet via the occultation cache (k=0.0091), moon via occult_small.", c))

    # 4. Jupiter + Earth-sized moon, TESS 2 min, 2 epochs, moon via the cache.
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.45,
              R_EARTH, M_EARTH, 3.0, epochs=2, cadence=TESS_2MIN,
              occult_small_threshold=0.001)
    c["tau_moon"] = tau_for_conjunction(c["t0_bary"], c["per_moon"])
    cases.append(_finish("sun_jupiter_earthmoon_tess_2min",
        "Sun/Jupiter/Earth-sized exomoon (P=3 d), TESS 2 min, 2 epochs; mutual eclipse "
        "at mid-transit of epoch 0; moon via the occultation cache (k=0.0091).", c))

    # 5. The ONLY eccentric moon: Jupiter + large icy moon, TESS 200 s FFI.
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.2,
              2.0 * R_GANYMEDE, 3.0 * M_GANYMEDE, 4.0, i_moon=88.5, Omega_moon=5.0,
              ecc_moon=0.5, w_moon=60.0, epochs=2, cadence=TESS_FFI200,
              occult_small_threshold=0.001)
    c["tau_moon"] = ECC_CASE_TAU
    cases.append(_finish("sun_jupiter_eccentric_moon_tess_200s",
        "Sun/Jupiter/large icy moon (2 R_Ganymede), the only eccentric moon "
        "(e=0.5, w=60 deg, i=88.5 deg, Omega=5 deg), TESS 200 s FFI, 2 epochs; "
        "tau puts the late-orbit phase (approach to periapsis) and a mutual eclipse "
        "in transit; moon via cache.", c))

    # 6. M dwarf + Neptune + Earth-sized moon, eccentric PLANET orbit, TESS 20 s.
    R_md = 0.3 * R_SUN_M
    c = _base(R_md, LD_MDWARF, 10.0, 43.6, R_NEPTUNE, M_NEPTUNE, 0.5,
              R_EARTH, M_EARTH, 1.137, i_moon=89.5, ecc_bary=0.1, w_bary=40.0,
              epochs=3, cadence=TESS_20S)
    c["tau_moon"] = tau_for_conjunction(c["t0_bary"], c["per_moon"])
    cases.append(_finish("mdwarf_neptune_earthmoon_tess_20s",
        "M dwarf (0.3 R_sun)/Neptune/Earth-sized moon (P=1.137 d), eccentric planet orbit "
        "(e=0.1, w=40 deg), TESS 20 s, 3 epochs; k_planet=0.118 (occult_hybrid), "
        "k_moon=0.030 (cache); mutual eclipse at mid-transit of epoch 0.", c, slack=1.4))

    # 7. Grazing Jupiter (b=1.0): planet always on the limb; eclipse on the limb.
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 1.0,
              R_GANYMEDE, M_GANYMEDE, 7.1546, cadence=KEPLER_SC)
    c["tau_moon"] = tau_for_conjunction(c["t0_bary"], c["per_moon"])
    cases.append(_finish("sun_jupiter_grazing_kepler_sc",
        "Grazing Sun/Jupiter transit (b=1.0, planet centre on the limb), Ganymede, "
        "Kepler SC; mutual eclipse on the limb (analytic limb path; the on-star part of the "
        "moon is always fully behind the planet disk, so the moon flux is exactly 1), "
        "case-III boundary geometry throughout.", c))

    # 8. Red giant (10 R_sun) + Earth + Luna: k_planet < 0.001 (below cache grid).
    R_rg = 10.0 * R_SUN_M
    c = _base(R_rg, LD_GIANT, 365.25, 21.5, R_EARTH, M_EARTH, 0.2,
              R_LUNA, M_LUNA, 27.3217, cadence=TESS_FFI10)
    c["tau_moon"] = tau_for_conjunction(c["t0_bary"], c["per_moon"])
    cases.append(_finish("giant_earth_luna_tess_10min",
        "Red giant (10 R_sun)/Earth/Luna, TESS 10 min FFI; k_planet=0.0009 is below "
        "the cache grid (routed to occult_small, bug #3), k_moon=0.00025; ~6 d transit; "
        "mutual eclipse at mid-transit.", c))

    # 9. Moon outside the Hill sphere -> treated as unphysical (moon removed).
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.5,
              R_GANYMEDE, M_GANYMEDE, 400.0, cadence=PLATO_1MIN)
    c["tau_moon"] = 0.3
    c = _finish("sun_jupiter_unphysical_moon_plato_1min",
        "Sun/Jupiter with a moon on a 400 d orbit (outside the Hill sphere): Pandora's "
        "plausibility check removes the moon; planet-only transit, PLATO 1 min.",
        c, eclipse=ECLIPSE_NONE)
    # the window would be dominated by the (absent) huge moon orbit: use planet only
    td = tdur_b0(c["per_bary"], c["a_bary"])
    c["epoch_duration"] = float(2 * 1.25 * np.sqrt((1 + c["r_planet"]) ** 2 - c["b_bary"] ** 2) * td / 2)
    cases.append(c)

    # 10. "Binary planet": Jupiter + Neptune-mass/size moon, no cache, 3x
    #     supersampling, t0 offset, drift; eclipse at ingress.
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.6,
              R_NEPTUNE, M_NEPTUNE, 2.0, i_moon=89.0, epochs=3,
              epoch_distance=365.30, t0_bary_offset=0.02, cadence=KEPLER_LC, ssf=3,
              use_cache=False)
    c["tau_moon"] = tau_for_conjunction(_ingress_time(c), c["per_moon"])
    cases.append(_finish("sun_jupiter_neptunemoon_kepler_lc_ssf3_nocache",
        "Sun/Jupiter/Neptune-sized moon ('binary planet', P=2 d), Kepler LC with 3x "
        "supersampling, no cache (both via occult_hybrid), t0_bary_offset=0.02 d, "
        "drift, numerical_grid=51; mutual eclipse during ingress (limb).", c))

    # (cases 11 and 12 covered the legacy pixel-raster eclipse_method='pixelmap', removed in 2026-10)

    # 13. Flux-only entry point pandora_flux() (P1 sample culling), as case 10
    #     but with 3x wider epoch windows, so that ~2/3 of the samples lie
    #     beyond the culling bound and supersampled cadences straddle it.
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.6,
              R_NEPTUNE, M_NEPTUNE, 2.0, i_moon=89.0, epochs=3,
              epoch_distance=365.30, t0_bary_offset=0.02, cadence=KEPLER_LC, ssf=3,
              use_cache=False)
    c["tau_moon"] = tau_for_conjunction(_ingress_time(c), c["per_moon"])
    c["entry"] = "pandora_flux"
    cases.append(_finish("sun_jupiter_neptunemoon_kepler_lc_ssf3_pandora_flux",
        "As case 10 (Jupiter/Neptune-sized moon, Kepler LC, 3x supersampling, no cache), "
        "fluxes from pandora_flux() (culls samples that cannot be in transit; coordinates "
        "for the fingerprint from pandora()), epoch windows 3x wider (slack 3.0).", c,
        slack=3.0))

    # 14. float32 deficit entry point pandora_deficit(): the eccentric moon of
    #     case 5 (late-orbit phase + mutual eclipse in transit), no cache (both
    #     via occult_hybrid), 2x wider epoch windows (culled samples).
    c = _base(R_SUN_M, LD_SUN, 365.25, 215.0, R_JUPITER, M_JUPITER, 0.2,
              2.0 * R_GANYMEDE, 3.0 * M_GANYMEDE, 4.0, i_moon=88.5, Omega_moon=5.0,
              ecc_moon=0.5, w_moon=60.0, epochs=2, cadence=TESS_FFI200,
              occult_small_threshold=0.001, use_cache=False)
    c["tau_moon"] = ECC_CASE_TAU
    c["entry"] = "pandora_deficit"
    cases.append(_finish("sun_jupiter_eccentric_moon_tess_200s_pandora_deficit",
        "As case 5 (eccentric moon e=0.5, TESS 200 s, 2 epochs) without cache, fluxes "
        "from the float32 deficits of pandora_deficit() (flux = 1 - deficit; float32 "
        "rounding <= 6e-8 of the deficit), epoch windows 2x wider (slack 2.5).", c,
        slack=2.5))

    return cases


# tau for the eccentric case: chosen once (scan in steps of 0.01) as the value
# with an in-transit mutual eclipse that is most sensitive to errors in the
# eccentric Kepler solver (bug #2 regression changes the moon flux by 1.3% of
# its depth; at e=0.25 the best tau gave only 0.01%). Frozen here.
ECC_CASE_TAU = 0.73


# ----------------------------------------------------------------------------
# Running a case
# ----------------------------------------------------------------------------
def _params(c):
    p = pandora.model_params()
    for key in ("R_star", "u1", "u2", "per_bary", "a_bary", "r_planet", "b_bary",
                "w_bary", "ecc_bary", "t0_bary", "t0_bary_offset", "M_planet",
                "r_moon", "per_moon", "tau_moon", "Omega_moon", "i_moon",
                "ecc_moon", "w_moon", "M_moon", "epochs", "epoch_duration",
                "cadences_per_day", "epoch_distance", "supersampling_factor",
                "occult_small_threshold", "hill_sphere_threshold"):
        setattr(p, key, c[key])
    return p


def run_case(c):
    """Return dict with time (output cadence), fluxes, supersampled time and
    coordinates of planet and moon."""
    p = _params(c)
    t_super = pandora.time(p).grid()
    # use_cache: the float32 occultation cache (create_occult_cache, dim 300) passed to pandora(), as
    # moon_model.light_curve() did until 2026-10 (light_curve() now uses the exact path: unit test
    # test_light_curve_exact)
    if True:
        kw = {}
        args = (
            c["u1"], c["u2"], c["R_star"],
            c["per_bary"], c["a_bary"], c["r_planet"], c["b_bary"], c["w_bary"],
            c["ecc_bary"], c["t0_bary"], c["t0_bary_offset"], c["M_planet"],
            c["r_moon"], c["per_moon"], c["tau_moon"], c["Omega_moon"], c["i_moon"],
            c["ecc_moon"], c["w_moon"], c["M_moon"],
            c["epoch_distance"], c["supersampling_factor"],
            c["occult_small_threshold"], c["hill_sphere_threshold"], t_super)
        if c["use_cache"]:
            fp, fm, ft, xp, yp, xm, ym = pandora.pandora(
                *args, pandora.create_occult_cache(c["u1"], c["u2"], dim=300))
        else:
            fp, fm, ft, xp, yp, xm, ym = pandora.pandora(*args, **kw)
        if c.get("entry") == "pandora_flux":
            # fluxes from the flux-only entry point; coordinates from pandora()
            fp, fm, ft = pandora.pandora_flux(*args, **kw)
        elif c.get("entry") == "pandora_deficit":
            # float32 deficits -> flux (float64); coordinates from pandora()
            fp, fm, ft = (1.0 - d.astype(np.float64) for d in pandora.pandora_deficit(*args, **kw))
    ssf = c["supersampling_factor"]
    t_out = resample(t_super, ssf) if ssf > 1 else t_super
    return dict(time=t_out, flux_planet=np.asarray(fp), flux_moon=np.asarray(fm),
                flux_total=np.asarray(ft), time_super=t_super,
                xp=np.asarray(xp), yp=np.asarray(yp), xm=np.asarray(xm), ym=np.asarray(ym))


def mutual_eclipse_points(c, r):
    """Supersampled time stamps where planet and moon overlap on the sky while
    the moon overlaps the stellar disk (i.e. eclipse() modifies the moon flux)."""
    sep = np.hypot(r["xm"] - r["xp"], r["ym"] - r["yp"])
    z_m = np.hypot(r["xm"], r["ym"])
    return int(np.sum((sep < c["r_planet"] + c["r_moon"]) & (z_m < 1 + c["r_moon"])))


def find_tau_for_eclipse(c, n=2000):
    """Scan tau_moon and return the first value that produces a mutual
    eclipse in epoch 0 (used to freeze the eccentric case's tau)."""
    for tau in np.linspace(0, 1, n, endpoint=False):
        c2 = dict(c, tau_moon=float(tau), epochs=1)
        r = run_case(c2)
        if mutual_eclipse_points(c2, r) > 0:
            return float(tau)
    raise RuntimeError("no tau gives a mutual eclipse")


# ----------------------------------------------------------------------------
# Fingerprint
# ----------------------------------------------------------------------------
N_BINS = 64
N_PROJ = 8


def _projection_weights(n):
    rng = np.random.default_rng(20261006)
    return rng.uniform(-1.0, 1.0, size=(N_PROJ, n))


def fingerprint(c, r):
    fp = dict(n=int(len(r["time"])),
              n_super=int(len(r["time_super"])),
              time_sum=float(np.sum(r["time"])),
              time_first=float(r["time"][0]),
              time_last=float(r["time"][-1]),
              mutual_eclipse_points=mutual_eclipse_points(c, r))
    W = _projection_weights(len(r["time"]))
    for comp in ("planet", "moon", "total"):
        f = r["flux_" + comp]
        d = 1.0 - f
        bins = [float(np.sum(b)) for b in np.array_split(d, N_BINS)]
        fp[comp] = dict(
            deficit_sum=float(np.sum(d)),
            deficit_sumsq=float(np.sum(d * d)),
            time_moment=float(np.sum(d * (r["time"] - r["time"][0]))),
            min_flux=float(np.min(f)),
            bins=bins,
            projections=[float(x) for x in W @ d],
        )
    return fp


def compare_fingerprints(ref, new, rtol=1e-7):
    """Return a list of human-readable mismatch descriptions (empty = pass)."""
    errors = []
    for key in ("n", "n_super"):
        if ref[key] != new[key]:
            errors.append(f"{key}: ref={ref[key]} new={new[key]}")
    if errors:
        return errors  # array lengths differ -> everything else is meaningless
    tscale = max(abs(ref["time_last"]), 1.0)
    for key in ("time_first", "time_last"):
        if abs(ref[key] - new[key]) > 1e-9 * tscale:
            errors.append(f"{key}: ref={ref[key]!r} new={new[key]!r}")
    if abs(ref["time_sum"] - new["time_sum"]) > 1e-9 * tscale * ref["n"]:
        errors.append(f"time_sum: ref={ref['time_sum']!r} new={new['time_sum']!r}")
    if abs(ref["mutual_eclipse_points"] - new["mutual_eclipse_points"]) > 1:
        errors.append(f"mutual_eclipse_points: ref={ref['mutual_eclipse_points']} "
                      f"new={new['mutual_eclipse_points']}")
    for comp in ("planet", "moon", "total"):
        a, b = ref[comp], new[comp]
        scale = max(abs(a["deficit_sum"]), 1e-300)
        tol = rtol * scale + 1e-15 * ref["n"]
        for key in ("deficit_sum", "min_flux"):
            if not abs(a[key] - b[key]) <= (tol if key == "deficit_sum" else rtol * 1e-3 + 1e-14):
                errors.append(f"{comp}.{key}: ref={a[key]!r} new={b[key]!r}")
        if not abs(a["deficit_sumsq"] - b["deficit_sumsq"]) <= rtol * max(a["deficit_sumsq"], 1e-300) + 1e-20:
            errors.append(f"{comp}.deficit_sumsq: ref={a['deficit_sumsq']!r} new={b['deficit_sumsq']!r}")
        tm_tol = rtol * max(abs(a["time_moment"]), 1e-300) + 1e-15 * ref["n"]
        if not abs(a["time_moment"] - b["time_moment"]) <= tm_tol:
            errors.append(f"{comp}.time_moment: ref={a['time_moment']!r} new={b['time_moment']!r}")
        for name in ("bins", "projections"):
            for j, (x, y) in enumerate(zip(a[name], b[name])):
                if not abs(x - y) <= tol:
                    errors.append(f"{comp}.{name}[{j}]: ref={x!r} new={y!r} (tol {tol:.2e})")
                    break
    return errors


def invariants(c, r):
    """Physics/consistency checks that must hold independent of the reference."""
    errors = []
    fp, fm, ft = r["flux_planet"], r["flux_moon"], r["flux_total"]
    for name, f in (("planet", fp), ("moon", fm), ("total", ft)):
        if not np.all(np.isfinite(f)):
            errors.append(f"{name}: {np.sum(~np.isfinite(f))} non-finite values")
        if np.any(f > 1 + 1e-12) or np.any(f < 0):
            errors.append(f"{name}: flux outside [0, 1]")
    tol = 1e-12
    if c.get("entry") == "pandora_deficit":
        # each float32 deficit is rounded separately (<= 2^-24 relative)
        tol += 2.0 ** -23 * float(np.max(np.abs(1.0 - ft)))
    if np.max(np.abs(ft - (fp + fm - 1))) > tol:
        errors.append("total != planet + moon - 1")
    # slack: first and last cadence of every epoch must be out of transit
    n_ep = c["epochs"]
    per_ep = len(ft) // n_ep
    for e in range(n_ep):
        for idx in (e * per_ep, (e + 1) * per_ep - 1):
            for name, f in (("planet", fp), ("moon", fm)):
                if abs(f[idx] - 1.0) > 1e-12:
                    errors.append(f"epoch {e}: {name} flux at window edge (idx {idx}) "
                                  f"= {f[idx]!r} != 1 (no slack)")
        seg = ft[e * per_ep:(e + 1) * per_ep]
        if np.min(seg) > 1 - 1e-9:
            errors.append(f"epoch {e}: no transit signal in segment")
    n_ecl = mutual_eclipse_points(c, r)
    if c["eclipse"] == ECLIPSE_REQUIRED and n_ecl == 0:
        errors.append("expected a mutual planet-moon eclipse, found none")
    if c["eclipse"] == ECLIPSE_NONE:
        if np.any(np.abs(fm - 1.0) > 1e-12):
            errors.append("unphysical-moon case: moon flux should be identically 1")
    return errors
