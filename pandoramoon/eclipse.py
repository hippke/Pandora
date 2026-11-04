from numpy import sqrt, pi, arccos, abs, cos
from numba import jit
from pandoramoon.overlap_analytic import er_fast
from pandoramoon.occult import cci  # circle-circle lens area (one copy for the package)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def eclipse_ratio(distance_planet_moon, r_planet, r_moon):
    """Returns eclipsed ratio [0..1] using circle_circle_intersect"""
    eclipsing = False
    eclipse_ratio = 0
    if abs(distance_planet_moon) < (r_planet + r_moon):
        eclipsing = True
        if (r_planet - r_moon) > abs(distance_planet_moon):
            eclipse_ratio = 1
            return eclipse_ratio
    # For partial eclipses, get the fraction of moon eclipse using transit...
    if eclipsing:
        if distance_planet_moon == 0:
            distance_planet_moon = 1e-10
        eclipse_ratio = cci(r_planet, r_moon, distance_planet_moon)
        # ...and transform this value into how much AREA is really eclipsed
        eclipse_ratio = eclipse_ratio / (pi * r_moon ** 2)
    return eclipse_ratio


@jit(cache=True, nopython=True, fastmath=False)
def _is_nan(x):
    """NaN test compiled without fastmath (under fastmath the compiler may assume no NaN and drop x != x)"""
    return x != x


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def eclipse_possible(xp, yp, xm, ym, r_planet, r_moon):
    """False if no sample can have a planet-moon overlap, i.e. eclipse()
    would return flux_moon unchanged (P5). eclipse() modifies a sample only
    if sqrt(dx^2 + dy^2) < r_planet + r_moon; here the samples with
    dx^2 + dy^2 < (r_planet + r_moon)^2 (1 + 1e-9) are counted (branch-free,
    vectorised; a float min reduction is not vectorised by numba/LLVM). The
    relative margin 1e-9 makes rounding differences (sqrt vs square, FMA
    contraction) cause at most an unnecessary eclipse() call, never a
    skipped overlap. NaN positions are never modified by eclipse() either."""
    r = r_planet + r_moon
    r2 = r * r * (1 + 1e-9)
    n_close = 0
    for i in range(len(xp)):
        dx = xm[i] - xp[i]
        dy = ym[i] - yp[i]
        n_close += (dx * dx + dy * dy) < r2
    return n_close > 0


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def eclipse_impossible_circular(a, i, r_planet, r_moon):
    """True if a circular moon orbit (planet-moon separation a [R_star],
    inclination i [deg], as in ellipse_z_runs()) can never bring planet and
    moon closer than r_planet + r_moon on the sky, so eclipse() would
    return flux_moon unchanged at every sample (P24b).

    The sky-plane separation is a |v| with v the rotated unit vector of
    ellipse_z_runs(): |v|^2 = cos^2 Q + sin^2 Q cos^2 i >= cos^2 i, so it is
    at least a |cos i| at every phase. The margins (relative 1e-9 on a,
    absolute 1e-12 R_star) cover the rounding of the positions (a few ulp of
    |x_bary| + a) and of sincos_turns (|v|^2 = 1 +- 1e-15); with them this
    never claims "impossible" where eclipse_possible() could count a sample.
    NaN gives False. P7e: for an eccentric moon pass a (1 - e): the
    separation is a (1 - e cos E) >= a (1 - e) times the same projection
    factor, and the solver's (cos E, sin E) are within ~1e-15 (inside the
    margins)."""
    cos_i = cos(i / 180 * pi)
    return abs(a * cos_i) * (1 - 1e-9) - 1e-12 > r_planet + r_moon


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def lens_gradient_term(xp, yp, xm, ym, d, r_planet, r_moon, u1, u2):
    """T4: first-order limb-darkening correction of the light under a
    partial planet-moon overlap (in units of the unocculted stellar flux).

    The constant-intensity treatment gives the overlap the moon disk's mean
    intensity I_bar (exact, from the moon's Mandel-Agol deficit). To first
    order the overlap's mean intensity is I_bar + grad I(m) . (c - m), with
    m the moon centre and c the overlap (lens) centroid. Returned:
    w * A * dI/dr(z_m) * ((c - m) . r_hat_m) / F0, with A the lens area,
    I(r) = 1 - u1 (1 - mu) - u2 (1 - mu)^2, mu = sqrt(1 - r^2),
    F0 = pi (1 - u1/3 - u2/6), and w = min(1, (1 - z_m) / r_moon) >= 0
    a taper that switches the term off continuously across the limb band
    (dI/dr diverges at the limb; on the limb the overlap area is the
    three-circle area, its centroid is not computed).
    Lens: the segments of the planet beyond the radical line (distance a
    from the planet centre) and of the moon; their first moments about the
    planet centre add up to A_moon_segment * d, so the centroid lies at
    s = d A_m / A from the planet centre towards the moon.
    Accuracy vs an exact LD-weighted union of both disks (radial
    integration; Gordon & Agol 2022 for the exact method): Earth-sized moon
    next to Jupiter, LD (0.4, 0.26): max error 1.36 -> 0.20 ppm (moon fully
    on the star), 1.37 -> 0.58 ppm (limb band); see BUGS.md, known
    limitations."""
    zm = sqrt(xm * xm + ym * ym)
    if not (zm < 1):
        return 0.0
    w = min(1.0, (1 - zm) / r_moon)
    if d <= r_planet - r_moon:         # moon fully behind the planet: exact (er = 1)
        return 0.0
    if d <= r_moon - r_planet:         # planet fully inside the moon disk
        A = pi * r_planet * r_planet
        s = 0.0
    else:
        a = (d * d + r_planet * r_planet - r_moon * r_moon) / (2 * d)
        h = sqrt(max(r_planet * r_planet - a * a, 0.0))
        b = d - a
        A_p = r_planet * r_planet * arccos(max(min(a / r_planet, 1.0), -1.0)) - a * h
        A_m = r_moon * r_moon * arccos(max(min(b / r_moon, 1.0), -1.0)) - b * h
        A = A_p + A_m
        if not (A > 0):
            return 0.0
        s = d * A_m / A
    # (c - m) . r_hat_m * dI/dr; c - m = (s - d) e_pm with e_pm the unit
    # vector planet -> moon; z_m cancels: dI/dr = -(u1 + 2 u2 (1 - mu)) z_m / mu
    mu = sqrt(1 - zm * zm)
    e_dot_m = ((xm - xp) * xm + (ym - yp) * ym) / d
    grad = -(u1 + 2 * u2 * (1 - mu)) / mu * (s - d) * e_dot_m
    return w * A * grad / (pi * (1 - u1 / 3 - u2 / 6))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def eclipse(xp, yp, xm, ym, r_planet, r_moon, flux_moon, u1=None, u2=None):
    """Checks if planet-moon occultation present. If yes, returns adjusted moon flux.
    Parameters
    ----------
    xp, yp, x_m, ym : float
        Planet and moon coordinates. Normalized so that R_star = 1 at (0,0)
    r_planet, r_moon : float
        Planet and moon radii. Normalized so that R_star = 1
    flux_moon : float
        Un-occulted moon flux (from Mandel-Agol model) in [0,1]
    u1, u2 : float or None
        Quadratic limb darkening. If given (Pandora passes them; T4), partial
        overlaps get the first-order limb-darkening correction
        lens_gradient_term() instead of the moon disk's mean intensity alone.
        None: the constant-intensity treatment of Pandora before T4.

    Returns
    -------
    occulted_flux_moon : float
        Moon flux >= flux_moon: the light under the planet-moon overlap is
        added back to the moon component, so that
        flux_total = flux_planet + flux_moon - 1 is the flux blocked by the
        union of both opaque disks. This is independent of which body is in
        front (only sky-plane positions are used); assigning the overlap to
        the moon component is a bookkeeping convention.
    """

    # Planet-Moon occultation
    # Case 1: No occultation
    # Case 2: Occultation, both bodies on star or off star --> 2-circle intersect
    # Case 3: Occultation, any body on limb --> analytic 3-circle overlap (overlap_analytic.er_fast)
    # R6a: reject by the squared separation first (no sqrt for the samples
    # without overlap, most of them); the margin 1e-9 keeps every sample the
    # sqrt test below accepts (as eclipse_possible)
    r_sum2 = (r_planet + r_moon) * (r_planet + r_moon) * (1 + 1e-9)
    for idx in range(len(xp)):
        dx = xm[idx] - xp[idx]
        dy = ym[idx] - yp[idx]
        # Case 1: No occultation
        if not (dx * dx + dy * dy < r_sum2):
            continue
        distance_p_m = sqrt(dx ** 2 + dy ** 2)
        if not (abs(distance_p_m) < (r_planet + r_moon)):
            continue

        # Check if moon or planet are on stellar limb
        on_limb = False
        if abs(1 - (sqrt(xm[idx] ** 2 + ym[idx] ** 2))) < (r_moon):
            on_limb = True
        if abs(1 - (sqrt(xp[idx] ** 2 + yp[idx] ** 2))) < (r_planet):
            on_limb = True

        # Case 2: Occultation, both bodies on star or off star --> 2 circle intersect
        if not on_limb:
            er = eclipse_ratio(distance_p_m, r_planet, r_moon)

        # Case 3: Occultation, any body on limb
        else:
            er = er_fast(xp[idx], yp[idx], r_planet, xm[idx], ym[idx], r_moon)
            if _is_nan(er):
                # measure-zero degenerate geometry (exact tangencies): the two-circle ratio, which ignores the
                # stellar limb (the pixel raster that served as this fallback was removed in 2026-10)
                er = eclipse_ratio(distance_p_m, r_planet, r_moon)

        # For Cases 2+3: Calculate reduced moon flux
        if er > 0:
            f = 1 - (1 - flux_moon[idx]) * (1 - er)
            # T4: limb-darkening gradient across the overlap (partial overlaps)
            if u1 is not None and u2 is not None and er < 1:
                f = f + lens_gradient_term(
                    xp[idx], yp[idx], xm[idx], ym[idx], distance_p_m, r_planet, r_moon, u1, u2
                )
                if f > 1:
                    f = 1.0
            flux_moon[idx] = f
    return flux_moon
