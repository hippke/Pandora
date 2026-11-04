"""Analytic area of the common overlap of three disks (star, planet, moon).

Used by eclipse() for mutual planet-moon eclipses with a body on the stellar
limb; it replaced the pixel raster pixelart() (removed in 2026-10).

Vendored from the standalone development project ../overlap3
(overlap3/analytic_fast.py, "v3"), which contains the pure-Python reference
implementation, the curated 20-example taxonomy suite, 171 tests, pixel-map
validation and the literature notes (Fewell 2006; Kipping 2011; Gordon &
Agol 2022 App. B reflex correction). Only er_fast_batch was left out.

Original module docstring follows.

Optimized numba implementation of the analytic 3-circle overlap (v3).

Same algorithm, eps windows, dispatch order and case codes as analytic.py /
analytic_numba.py (v1), optimized for single-core throughput. Improvements
over v2 (which introduced the vector-form walk):

  * vertices are collected into scalars (no scratch array at all before the
    walk dispatch); the single scratch array is allocated only for the
    generic walk path (V = 4 or exotic V = 3),
  * **V = 3 fast path, fully scalar and scratch-free**: with one vertex per
    pair (the only non-degenerate V = 3 structure), each circle hosts
    exactly two vertices, so the inside-zone on every circle is a SINGLE
    arc. Therefore the midpoint test alone selects the correct arc
    direction (the open-arc no-crossing test is provably redundant for
    V = 3), and the three adjacencies are known a priori:
    (v01,v02) on circle 0, (v02,v12) on circle 2, (v12,v01) on circle 1.
    A V = 3 walk now costs 3 midpoint tests (3 sqrt) + 3 atan2 + ~60 flops,
    with zero allocations and no candidate scan,
  * er_fast: A(star & moon) is computed first; if it is zero (moon off the
    star) the function returns 1.0 without computing the triple overlap,
    mirroring pixelart()'s cci == 0 branch,
  * er_fast_batch: array API for Pandora-style light-curve loops (er for
    every point; callable from jitted code without per-point Python
    boundary costs).

The Python->numba call boundary costs ~150 ns per scalar call (measured);
inside jitted code (njit->njit, Pandora's production pattern) calls cost a
few ns. benchmark_internal.py measures the boundary-free costs.

Not bit-identical to v1 by design (fp differences at the ~1e-16 relative
level from sqrt-vs-hypot, the vector reformulation and the V = 3 summation
order, plus measure-zero classification differences inside the 1e-12
angular/collision windows); equivalence is asserted in tests at
1e-12 * disk-area level over the full example suite and thousands of fuzz
configurations.
"""

import numpy as np
from numba import njit

EPS_TAN = 1e-12
EPS_IN = 1e-9
EPS_COIN = 1e-9
TWO_PI = 2.0 * np.pi

CASE_EMPTY = 0
CASE_FULL_DISK = 1
CASE_LENS = 2
CASE_WALK = 3
CASE_DEGENERATE = 4


@njit(cache=True)
def _lens_area(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2) or d == 0.0:
        return np.pi * min(r1, r2) ** 2
    t1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)
    t2 = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)
    a1 = r1 * r1 * np.arccos(min(1.0, max(-1.0, t1)))
    a2 = r2 * r2 * np.arccos(min(1.0, max(-1.0, t2)))
    tri = (-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1)
    return a1 + a2 - 0.5 * np.sqrt(max(tri, 0.0))


@njit(cache=True)
def _crossings(x1, y1, r1, x2, y2, r2, d):
    """Crossing points of two circle boundaries (pair distance d precomputed).
    Mirrors v1's _pair_points incl. the tangency window; returns
    (n, ax, ay, bx, by) with n in {0, 1, 2} (1 = tangent)."""
    scale = max(d + r1 + r2, r1 + r2, 1.0)
    if d <= EPS_TAN * scale:
        return 0, 0.0, 0.0, 0.0, 0.0
    a = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    h2 = r1 * r1 - a * a
    ux = (x2 - x1) / d
    uy = (y2 - y1) / d
    bx = x1 + a * ux
    by = y1 + a * uy
    if h2 < -EPS_TAN * scale * scale:
        return 0, 0.0, 0.0, 0.0, 0.0
    if h2 <= EPS_TAN * scale * scale:
        return 1, bx, by, 0.0, 0.0
    h = np.sqrt(h2)
    return 2, bx - h * uy, by + h * ux, bx + h * uy, by - h * ux


@njit(cache=True)
def _inside(px, py, cx, cy, mm):
    """Strictly-inside test against the precomputed margin square."""
    return (px - cx) ** 2 + (py - cy) ** 2 < mm


@njit(cache=True)
def area3_fast_core(x1, y1, r1, x2, y2, r2, x3, y3, r3):
    """Area of the common overlap of three disks. Returns
    (area, case_code, n_vertices, reflex_flag). Single-threaded."""
    # ---- containment reduction (drop the larger of any contained pair) ----
    # mirrors v1's scan order (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) with restart
    al0 = al1 = al2 = True
    changed = True
    while changed:
        changed = False
        if al0 and al1 and r1 <= r2 + EPS_TAN * max(1.0, r2):
            dd = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dd + r1 <= r2 + EPS_TAN * max(1.0, r2):
                al1 = False
                changed = True
        if not changed and al0 and al2 and r1 <= r3 + EPS_TAN * max(1.0, r3):
            dd = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
            if dd + r1 <= r3 + EPS_TAN * max(1.0, r3):
                al2 = False
                changed = True
        if not changed and al1 and al0 and r2 <= r1 + EPS_TAN * max(1.0, r1):
            dd = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dd + r2 <= r1 + EPS_TAN * max(1.0, r1):
                al0 = False
                changed = True
        if not changed and al1 and al2 and r2 <= r3 + EPS_TAN * max(1.0, r3):
            dd = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
            if dd + r2 <= r3 + EPS_TAN * max(1.0, r3):
                al2 = False
                changed = True
        if not changed and al2 and al0 and r3 <= r1 + EPS_TAN * max(1.0, r1):
            dd = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
            if dd + r3 <= r1 + EPS_TAN * max(1.0, r1):
                al0 = False
                changed = True
        if not changed and al2 and al1 and r3 <= r2 + EPS_TAN * max(1.0, r2):
            dd = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            if dd + r3 <= r2 + EPS_TAN * max(1.0, r2):
                al1 = False
                changed = True

    # ---- pack survivors into slots 0..n_alive-1 (ascending original order) ---
    s0x = s0y = s0r = 0.0
    s1x = s1y = s1r = 0.0
    s2x = s2y = s2r = 0.0
    n_alive = 0
    if al0:
        s0x = x1; s0y = y1; s0r = r1
        n_alive = 1
    if al1:
        if n_alive == 0:
            s0x = x2; s0y = y2; s0r = r2
        else:
            s1x = x2; s1y = y2; s1r = r2
        n_alive += 1
    if al2:
        if n_alive == 0:
            s0x = x3; s0y = y3; s0r = r3
        elif n_alive == 1:
            s1x = x3; s1y = y3; s1r = r3
        else:
            s2x = x3; s2y = y3; s2r = r3
        n_alive += 1
    if n_alive == 0:
        return np.nan, CASE_DEGENERATE, 0, False  # unreachable

    scale = max(max(r1, r2), max(r3, 1.0))
    a_min_disk = np.pi * min(min(r1, r2), r3) ** 2
    margin = EPS_IN * scale

    # ---- 1 circle left: full disk -------------------------------------------
    if n_alive == 1:
        return np.pi * s0r ** 2, CASE_FULL_DISK, 0, False

    # ---- 2 circles left: lens (or empty if disjoint / externally tangent) ----
    if n_alive == 2:
        dd = np.sqrt((s0x - s1x) ** 2 + (s0y - s1y) ** 2)
        if dd >= s0r + s1r - EPS_TAN * scale:
            return 0.0, CASE_EMPTY, 0, False
        return _lens_area(s0r, s1r, dd), CASE_LENS, 0, False

    # ---- 3 circles: pair distances (computed once, reused everywhere) -------
    d01 = np.sqrt((s0x - s1x) ** 2 + (s0y - s1y) ** 2)
    d02 = np.sqrt((s0x - s2x) ** 2 + (s0y - s2y) ** 2)
    d12 = np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2)

    if d01 >= s0r + s1r - EPS_TAN * scale:
        return 0.0, CASE_EMPTY, 0, False
    if d02 >= s0r + s2r - EPS_TAN * scale:
        return 0.0, CASE_EMPTY, 0, False
    if d12 >= s1r + s2r - EPS_TAN * scale:
        return 0.0, CASE_EMPTY, 0, False

    # ---- vertices: pairwise crossings strictly inside the third disk ---------
    # collected into SCALAR slots (no scratch array on non-generic paths)
    v0x = v0y = v0a = v0b = 0.0
    v1x = v1y = v1a = v1b = 0.0
    v2x = v2y = v2a = v2b = 0.0
    v3x = v3y = v3a = v3b = 0.0
    nv = 0
    # inside-third margins (squared, precomputed)
    m0 = s0r - margin
    m1 = s1r - margin
    m2 = s2r - margin
    mm0 = m0 * m0
    mm1 = m1 * m1
    mm2 = m2 * m2

    # pair (0, 1), third = 2
    n, ax, ay, bx, by = _crossings(s0x, s0y, s0r, s1x, s1y, s1r, d01)
    if n >= 1 and _inside(ax, ay, s2x, s2y, mm2):
        if nv == 0:
            v0x = ax; v0y = ay; v0a = 0.0; v0b = 1.0; nv = 1
        elif nv == 1:
            v1x = ax; v1y = ay; v1a = 0.0; v1b = 1.0; nv = 2
        elif nv == 2:
            v2x = ax; v2y = ay; v2a = 0.0; v2b = 1.0; nv = 3
        elif nv == 3:
            v3x = ax; v3y = ay; v3a = 0.0; v3b = 1.0; nv = 4
        else:
            return np.nan, CASE_DEGENERATE, 5, False
    if n == 2 and _inside(bx, by, s2x, s2y, mm2):
        if nv == 0:
            v0x = bx; v0y = by; v0a = 0.0; v0b = 1.0; nv = 1
        elif nv == 1:
            v1x = bx; v1y = by; v1a = 0.0; v1b = 1.0; nv = 2
        elif nv == 2:
            v2x = bx; v2y = by; v2a = 0.0; v2b = 1.0; nv = 3
        elif nv == 3:
            v3x = bx; v3y = by; v3a = 0.0; v3b = 1.0; nv = 4
        else:
            return np.nan, CASE_DEGENERATE, 5, False
    # pair (0, 2), third = 1
    n, ax, ay, bx, by = _crossings(s0x, s0y, s0r, s2x, s2y, s2r, d02)
    if n >= 1 and _inside(ax, ay, s1x, s1y, mm1):
        if nv == 0:
            v0x = ax; v0y = ay; v0a = 0.0; v0b = 2.0; nv = 1
        elif nv == 1:
            v1x = ax; v1y = ay; v1a = 0.0; v1b = 2.0; nv = 2
        elif nv == 2:
            v2x = ax; v2y = ay; v2a = 0.0; v2b = 2.0; nv = 3
        elif nv == 3:
            v3x = ax; v3y = ay; v3a = 0.0; v3b = 2.0; nv = 4
        else:
            return np.nan, CASE_DEGENERATE, 5, False
    if n == 2 and _inside(bx, by, s1x, s1y, mm1):
        if nv == 0:
            v0x = bx; v0y = by; v0a = 0.0; v0b = 2.0; nv = 1
        elif nv == 1:
            v1x = bx; v1y = by; v1a = 0.0; v1b = 2.0; nv = 2
        elif nv == 2:
            v2x = bx; v2y = by; v2a = 0.0; v2b = 2.0; nv = 3
        elif nv == 3:
            v3x = bx; v3y = by; v3a = 0.0; v3b = 2.0; nv = 4
        else:
            return np.nan, CASE_DEGENERATE, 5, False
    # pair (1, 2), third = 0
    n, ax, ay, bx, by = _crossings(s1x, s1y, s1r, s2x, s2y, s2r, d12)
    if n >= 1 and _inside(ax, ay, s0x, s0y, mm0):
        if nv == 0:
            v0x = ax; v0y = ay; v0a = 1.0; v0b = 2.0; nv = 1
        elif nv == 1:
            v1x = ax; v1y = ay; v1a = 1.0; v1b = 2.0; nv = 2
        elif nv == 2:
            v2x = ax; v2y = ay; v2a = 1.0; v2b = 2.0; nv = 3
        elif nv == 3:
            v3x = ax; v3y = ay; v3a = 1.0; v3b = 2.0; nv = 4
        else:
            return np.nan, CASE_DEGENERATE, 5, False
    if n == 2 and _inside(bx, by, s0x, s0y, mm0):
        if nv == 0:
            v0x = bx; v0y = by; v0a = 1.0; v0b = 2.0; nv = 1
        elif nv == 1:
            v1x = bx; v1y = by; v1a = 1.0; v1b = 2.0; nv = 2
        elif nv == 2:
            v2x = bx; v2y = by; v2a = 1.0; v2b = 2.0; nv = 3
        elif nv == 3:
            v3x = bx; v3y = by; v3a = 1.0; v3b = 2.0; nv = 4
        else:
            return np.nan, CASE_DEGENERATE, 5, False

    # ---- dispatch on vertex count ---------------------------------------------
    if nv == 0:
        return 0.0, CASE_EMPTY, 0, False
    if nv == 1:
        return 0.0, CASE_EMPTY, 1, False
    if nv == 2:
        if v0a == v1a and v0b == v1b:
            # lens of one pair, fully inside the third disk
            if v0a == 0.0:
                if v0b == 1.0:
                    return _lens_area(s0r, s1r, d01), CASE_LENS, 2, False
                return _lens_area(s0r, s2r, d02), CASE_LENS, 2, False
            return _lens_area(s1r, s2r, d12), CASE_LENS, 2, False
        return 0.0, CASE_EMPTY, 2, False

    # ---- coincident vertices -> degenerate (squared-distance form) -----------
    vsq = (EPS_COIN * scale) ** 2
    if nv >= 2:
        if (v0x - v1x) ** 2 + (v0y - v1y) ** 2 < vsq:
            return np.nan, CASE_DEGENERATE, nv, False
    if nv >= 3:
        if (v0x - v2x) ** 2 + (v0y - v2y) ** 2 < vsq:
            return np.nan, CASE_DEGENERATE, nv, False
        if (v1x - v2x) ** 2 + (v1y - v2y) ** 2 < vsq:
            return np.nan, CASE_DEGENERATE, nv, False
    if nv >= 4:
        if (v0x - v3x) ** 2 + (v0y - v3y) ** 2 < vsq:
            return np.nan, CASE_DEGENERATE, nv, False
        if (v1x - v3x) ** 2 + (v1y - v3y) ** 2 < vsq:
            return np.nan, CASE_DEGENERATE, nv, False
        if (v2x - v3x) ** 2 + (v2y - v3y) ** 2 < vsq:
            return np.nan, CASE_DEGENERATE, nv, False

    # ---- V = 3 fast path (one vertex per pair, fully scalar) ----------------
    # With one vertex per pair (the only non-degenerate V = 3 structure),
    # each circle hosts exactly two vertices, so the inside-zone on every
    # circle is a single arc and the midpoint test alone selects the arc
    # direction. Adjacencies are fixed: (v01,v02) on circle 0,
    # (v02,v12) on circle 2, (v12,v01) on circle 1.
    if nv == 3:
        # one-per-pair <=> no two slots share the same pair id
        one3 = not ((v0a == v1a and v0b == v1b)
                    or (v0a == v2a and v0b == v2b)
                    or (v1a == v2a and v1b == v2b))
        if one3:
            # slot lookup: xa,ya = pair(0,1) vertex; xb,yb = pair(0,2); xc,yc = pair(1,2)
            xa = ya = xb = yb = xc = yc = 0.0
            if v0a == 0.0 and v0b == 1.0:
                xa = v0x; ya = v0y
            elif v0a == 0.0 and v0b == 2.0:
                xb = v0x; yb = v0y
            else:
                xc = v0x; yc = v0y
            if v1a == 0.0 and v1b == 1.0:
                xa = v1x; ya = v1y
            elif v1a == 0.0 and v1b == 2.0:
                xb = v1x; yb = v1y
            else:
                xc = v1x; yc = v1y
            if v2a == 0.0 and v2b == 1.0:
                xa = v2x; ya = v2y
            elif v2a == 0.0 and v2b == 2.0:
                xb = v2x; yb = v2y
            else:
                xc = v2x; yc = v2y
            a3, refl3 = _v3_fast(xa, ya, xb, yb, xc, yc,
                                 s0x, s0y, s0r, s1x, s1y, s1r, s2x, s2y, s2r,
                                 mm0, mm1, mm2, scale, a_min_disk)
            return a3, CASE_WALK, 3, refl3
        # else: exotic pair structure -> generic walk below

    # ---- V = 4 fast path (Fewell case d, fully scalar) ------------------------
    # A non-degenerate V = 4 is always the quadrilateral: two vertices from
    # pair (i,k) and two from pair (j,k), all four on circle k. On circle k
    # the boundary consists of exactly TWO ISOLATED SINGLE-GAP arcs (the
    # two components of the in-i and in-j zone intersection); traversing
    # the boundary keeps the region inside disk k, so those arcs run CCW
    # from the angularly earlier to the later endpoint (no direction test
    # needed). The arcs on circles i (between the two W vertices) and j
    # (between the two X vertices) are selected by the midpoint test as in
    # the V = 3 path. Any structural surprise falls back to the generic
    # walk, which reproduces v1 exactly.
    if nv == 4:
        ok4, a4, refl4 = _v4_fast(
            v0x, v0y, v0a, v0b, v1x, v1y, v1a, v1b,
            v2x, v2y, v2a, v2b, v3x, v3y, v3a, v3b,
            s0x, s0y, s0r, s1x, s1y, s1r, s2x, s2y, s2r,
            mm0, mm1, mm2, scale, a_min_disk)
        if ok4:
            return a4, CASE_WALK, 4, refl4
        # else: fall through to the generic walk

    # ---- generic walk path (V = 4, exotic V = 3) ------------------------------
    # scratch layout:
    #   [u*4 + 0..3]     vertex u: x, y, ca, cb              (u = 0..3)
    #   [16 + c*8 + 2t]  crossing point t (x, y) of circle c (t = 0..3)
    #   [40 + c]         crossing point count of circle c
    #   [44 + u*2]       out-arc of vertex u: next vertex
    #   [45 + u*2]       ... on circle s
    #   [52 + u]         visited flag (cycle extraction)
    scr = np.empty(56)
    scr[0] = v0x; scr[1] = v0y; scr[2] = v0a; scr[3] = v0b
    scr[4] = v1x; scr[5] = v1y; scr[6] = v1a; scr[7] = v1b
    scr[8] = v2x; scr[9] = v2y; scr[10] = v2a; scr[11] = v2b
    scr[12] = v3x; scr[13] = v3y; scr[14] = v3a; scr[15] = v3b
    for k in range(52, 56):
        scr[k] = 0.0

    # store ALL crossing points per circle (for the open-arc test)
    c0 = c1 = c2 = 0
    n, ax, ay, bx, by = _crossings(s0x, s0y, s0r, s1x, s1y, s1r, d01)
    if n >= 1:
        scr[16 + 2 * c0] = ax; scr[17 + 2 * c0] = ay; c0 += 1
        scr[24 + 2 * c1] = ax; scr[25 + 2 * c1] = ay; c1 += 1
    if n == 2:
        scr[16 + 2 * c0] = bx; scr[17 + 2 * c0] = by; c0 += 1
        scr[24 + 2 * c1] = bx; scr[25 + 2 * c1] = by; c1 += 1
    n, ax, ay, bx, by = _crossings(s0x, s0y, s0r, s2x, s2y, s2r, d02)
    if n >= 1:
        scr[16 + 2 * c0] = ax; scr[17 + 2 * c0] = ay; c0 += 1
        scr[32 + 2 * c2] = ax; scr[33 + 2 * c2] = ay; c2 += 1
    if n == 2:
        scr[16 + 2 * c0] = bx; scr[17 + 2 * c0] = by; c0 += 1
        scr[32 + 2 * c2] = bx; scr[33 + 2 * c2] = by; c2 += 1
    n, ax, ay, bx, by = _crossings(s1x, s1y, s1r, s2x, s2y, s2r, d12)
    if n >= 1:
        scr[24 + 2 * c1] = ax; scr[25 + 2 * c1] = ay; c1 += 1
        scr[32 + 2 * c2] = ax; scr[33 + 2 * c2] = ay; c2 += 1
    if n == 2:
        scr[24 + 2 * c1] = bx; scr[25 + 2 * c1] = by; c1 += 1
        scr[32 + 2 * c2] = bx; scr[33 + 2 * c2] = by; c2 += 1
    scr[40] = c0
    scr[41] = c1
    scr[42] = c2

    # ---- directed arc walk (vector form) --------------------------------------
    for u in range(nv):
        n_out = 0
        ua = scr[u * 4 + 2]
        ub = scr[u * 4 + 3]
        for v in range(nv):
            if u == v:
                continue
            va = scr[v * 4 + 2]
            vb = scr[v * 4 + 3]
            sh0 = -1.0
            sh1 = -1.0
            if ua == va or ua == vb:
                sh0 = ua
            if (ub == va or ub == vb) and ub != sh0:
                sh1 = ub
            for s_idx in range(2):
                if s_idx == 0:
                    s = sh0
                else:
                    s = sh1
                if s < 0.0:
                    continue
                si = int(s)
                if si == 0:
                    ox = s0x; oy = s0y; r = s0r
                elif si == 1:
                    ox = s1x; oy = s1y; r = s1r
                else:
                    ox = s2x; oy = s2y; r = s2r
                ux_ = scr[u * 4] - ox
                uy_ = scr[u * 4 + 1] - oy
                vx_ = scr[v * 4] - ox
                vy_ = scr[v * 4 + 1] - oy
                cr_ = ux_ * vy_ - uy_ * vx_
                dt_ = ux_ * vx_ + uy_ * vy_
                # near-zero CCW span (mirrors v1's delta < 1e-12 skip)
                if dt_ > 0.0 and abs(cr_) < 1e-12 * r * r:
                    continue
                short = cr_ > 0.0   # CCW span u->v is < pi
                # test 1: no crossing point strictly inside the open CCW span
                # (skip the arc endpoints u, v themselves)
                bad = False
                base = 16 + si * 8
                cnt = int(scr[40 + si])
                for tt in range(cnt):
                    qx0 = scr[base + 2 * tt]
                    qy0 = scr[base + 2 * tt + 1]
                    if (qx0 == scr[u * 4] and qy0 == scr[u * 4 + 1]) or \
                       (qx0 == scr[v * 4] and qy0 == scr[v * 4 + 1]):
                        continue
                    qx = qx0 - ox
                    qy = qy0 - oy
                    if short:
                        if (ux_ * qy - uy_ * qx) > 0.0 and (qx * vy_ - qy * vx_) > 0.0:
                            bad = True
                            break
                    else:
                        if not ((vx_ * qy - vy_ * qx) > 0.0 and (qx * uy_ - qy * ux_) > 0.0):
                            bad = True
                            break
                if bad:
                    continue
                # test 2: arc midpoint strictly inside the other two disks
                wx = ux_ + vx_
                wy = uy_ + vy_
                ww = wx * wx + wy * wy
                if ww < 1e-16 * r * r:
                    pmx = ox - uy_
                    pmy = oy + ux_
                else:
                    inv = r / np.sqrt(ww)
                    if short:
                        pmx = ox + wx * inv
                        pmy = oy + wy * inv
                    else:
                        pmx = ox - wx * inv
                        pmy = oy - wy * inv
                ok = True
                if si == 0:
                    if not (_inside(pmx, pmy, s1x, s1y, mm1)
                            and _inside(pmx, pmy, s2x, s2y, mm2)):
                        ok = False
                elif si == 1:
                    if not (_inside(pmx, pmy, s0x, s0y, mm0)
                            and _inside(pmx, pmy, s2x, s2y, mm2)):
                        ok = False
                else:
                    if not (_inside(pmx, pmy, s0x, s0y, mm0)
                            and _inside(pmx, pmy, s1x, s1y, mm1)):
                        ok = False
                if not ok:
                    continue
                if n_out > 0:
                    return np.nan, CASE_DEGENERATE, nv, False
                scr[44 + u * 2] = v
                scr[45 + u * 2] = s
                n_out += 1
                break   # first valid shared circle for this (u, v) suffices
        if n_out != 1:
            return np.nan, CASE_DEGENERATE, nv, False

    # ---- extract cycle, accumulate the Green's-theorem area -------------------
    cur = 0
    steps = 0
    reflex = False
    area = 0.0
    while True:
        if cur < 0 or scr[52 + cur] == 1.0:
            break
        scr[52 + cur] = 1.0
        u = cur
        v = int(scr[44 + cur * 2])
        si = int(scr[45 + cur * 2])
        if si == 0:
            ox = s0x; oy = s0y; r = s0r
        elif si == 1:
            ox = s1x; oy = s1y; r = s1r
        else:
            ox = s2x; oy = s2y; r = s2r
        ux_ = scr[u * 4] - ox
        uy_ = scr[u * 4 + 1] - oy
        vx_ = scr[v * 4] - ox
        vy_ = scr[v * 4 + 1] - oy
        delta = np.arctan2(ux_ * vy_ - uy_ * vx_, ux_ * vx_ + uy_ * vy_)
        if delta < 0.0:
            delta += TWO_PI
        if delta > np.pi:
            reflex = True
        # 0.5*[r^2*dtheta + ox*(v_y - u_y) - oy*(v_x - u_x)]  (trig cancels)
        area = area + 0.5 * (
            r * r * delta
            + ox * (scr[v * 4 + 1] - scr[u * 4 + 1])
            - oy * (scr[v * 4] - scr[u * 4])
        )
        cur = v
        steps += 1
        if steps > nv:
            return np.nan, CASE_DEGENERATE, nv, False
    if steps != nv or cur != 0:
        return np.nan, CASE_DEGENERATE, nv, False

    # ---- plausibility ----------------------------------------------------------
    if area < -1e-9 * scale * scale or area > a_min_disk * (1.0 + 1e-9) + 1e-12:
        return np.nan, CASE_DEGENERATE, nv, reflex
    return max(area, 0.0), CASE_WALK, nv, reflex


@njit(cache=True)
def _v3_fast(xa, ya, xb, yb, xc, yc,
              s0x, s0y, s0r, s1x, s1y, s1r, s2x, s2y, s2r,
              mm0, mm1, mm2, scale, a_min_disk):
    """Scalar V = 3 walk: xa = pair(0,1) vertex, xb = pair(0,2), xc = pair(1,2).
    Adjacencies: (a,b) on circle 0, (b,c) on circle 2, (c,a) on circle 1.
    The midpoint test alone selects each arc's direction (the zone on each
    circle is a single arc for the one-vertex-per-pair structure).
    Returns (area, reflex)."""
    area = 0.0
    reflex = False
    # starts of the three directed arcs as vertex codes (a=0, b=1, c=2);
    # a valid cycle requires all three distinct (exactly one out-edge each)
    s1_ = 2
    s2_ = 2
    s3_ = 2

    # adjacency (a, b) on circle 0
    ok, d_ar, ar, sw = _v3_arc(xa, ya, xb, yb, s0x, s0y, s0r,
                               s1x, s1y, mm1, s2x, s2y, mm2)
    if not ok:
        return np.nan, False
    area += d_ar
    if ar:
        reflex = True
    if sw:
        s1_ = 1
    else:
        s1_ = 0
    # adjacency (b, c) on circle 2
    ok, d_ar, ar, sw = _v3_arc(xb, yb, xc, yc, s2x, s2y, s2r,
                               s0x, s0y, mm0, s1x, s1y, mm1)
    if not ok:
        return np.nan, False
    area += d_ar
    if ar:
        reflex = True
    if sw:
        s2_ = 2
    else:
        s2_ = 1
    # adjacency (c, a) on circle 1
    ok, d_ar, ar, sw = _v3_arc(xc, yc, xa, ya, s1x, s1y, s1r,
                               s0x, s0y, mm0, s2x, s2y, mm2)
    if not ok:
        return np.nan, False
    area += d_ar
    if ar:
        reflex = True
    if sw:
        s3_ = 0
    else:
        s3_ = 2
    # cycle consistency: three distinct starts
    if s1_ == s2_ or s1_ == s3_ or s2_ == s3_:
        return np.nan, False

    if area < -1e-9 * scale * scale or area > a_min_disk * (1.0 + 1e-9) + 1e-12:
        return np.nan, reflex
    return max(area, 0.0), reflex


@njit(cache=True)
def _v3_arc(px, py, qx, qy, ox, oy, r, c1x, c1y, mm1, c2x, c2y, mm2):
    """One V = 3 adjacency: select the arc direction on the circle
    (ox, oy, r) between vertices p and q via the midpoint test against the
    two other disks (margin squares mm1, mm2), then return the
    Green's-theorem contribution of the chosen arc.
    Returns (ok, contribution, reflex, swapped): swapped=False means the
    valid arc runs p -> q, True means q -> p."""
    ux_ = px - ox
    uy_ = py - oy
    vx_ = qx - ox
    vy_ = qy - oy
    cr_ = ux_ * vy_ - uy_ * vx_
    dt_ = ux_ * vx_ + uy_ * vy_
    if dt_ > 0.0 and abs(cr_) < 1e-12 * r * r:
        return False, 0.0, False, False   # near-zero span: not a valid adjacency
    wx = ux_ + vx_
    wy = uy_ + vy_
    ww = wx * wx + wy * wy
    # midpoint of the CCW arc p->q:  +w/|w| if span < pi (cr_ > 0) else -w/|w|
    # midpoint of the CCW arc q->p:  the other one
    if ww < 1e-16 * r * r:
        # antipodal: the two midpoints are +-perp(u'); the unnormalized
        # perp is fine (error ~ 1e-16 r is far below the margin)
        p1x = ox - uy_
        p1y = oy + ux_
        p2x = ox + uy_
        p2y = oy - ux_
    else:
        inv = r / np.sqrt(ww)
        if cr_ > 0.0:
            p1x = ox + wx * inv; p1y = oy + wy * inv   # p -> q (short)
            p2x = ox - wx * inv; p2y = oy - wy * inv   # q -> p (long)
        else:
            p1x = ox - wx * inv; p1y = oy - wy * inv   # p -> q (long)
            p2x = ox + wx * inv; p2y = oy + wy * inv   # q -> p (short)
    m1 = _inside(p1x, p1y, c1x, c1y, mm1) and _inside(p1x, p1y, c2x, c2y, mm2)
    m2 = _inside(p2x, p2y, c1x, c1y, mm1) and _inside(p2x, p2y, c2x, c2y, mm2)
    if m1 == m2:
        # both or neither: ambiguous (degenerate) or invalid
        return False, 0.0, False, False
    swapped = m2
    if m2:
        # valid arc is q -> p: swap arc start and end
        tx = ux_; ux_ = vx_; vx_ = tx
        ty = uy_; uy_ = vy_; vy_ = ty
    # else: valid arc is p -> q (u = p, v = q as set)
    delta = np.arctan2(ux_ * vy_ - uy_ * vx_, ux_ * vx_ + uy_ * vy_)
    if delta < 0.0:
        delta += TWO_PI
    # 0.5*[r^2*delta + ox*(v_y - u_y) - oy*(v_x - u_x)]; the global-coordinate
    # differences reduce to local ones (the oy/ox offsets cancel)
    contrib = 0.5 * (r * r * delta + ox * (vy_ - uy_) - oy * (vx_ - ux_))
    return True, contrib, delta > np.pi, swapped


@njit(cache=True)
def _v4_fast(v0x, v0y, v0a, v0b, v1x, v1y, v1a, v1b,
             v2x, v2y, v2a, v2b, v3x, v3y, v3a, v3b,
             s0x, s0y, s0r, s1x, s1y, s1r, s2x, s2y, s2r,
             mm0, mm1, mm2, scale, a_min_disk):
    """V = 4 fast path (Fewell case d): two vertices from pair (i,k) ("W")
    and two from pair (j,k) ("X"), all on circle k. The two boundary arcs on
    circle k are isolated single gaps between angularly adjacent vertices,
    traversed CCW (no direction test); the arcs on circles i and j are
    selected by the midpoint test (_v3_arc). Any structural surprise
    returns ok=False and the caller falls back to the generic walk, which
    reproduces v1 exactly (including its degenerate outcomes).
    Returns (ok, area, reflex)."""
    # ---- pair census ---------------------------------------------------------
    c01 = c02 = c12 = 0.0
    if v0a == 0.0 and v0b == 1.0:
        c01 += 1.0
    elif v0a == 0.0 and v0b == 2.0:
        c02 += 1.0
    elif v0a == 1.0 and v0b == 2.0:
        c12 += 1.0
    if v1a == 0.0 and v1b == 1.0:
        c01 += 1.0
    elif v1a == 0.0 and v1b == 2.0:
        c02 += 1.0
    elif v1a == 1.0 and v1b == 2.0:
        c12 += 1.0
    if v2a == 0.0 and v2b == 1.0:
        c01 += 1.0
    elif v2a == 0.0 and v2b == 2.0:
        c02 += 1.0
    elif v2a == 1.0 and v2b == 2.0:
        c12 += 1.0
    if v3a == 0.0 and v3b == 1.0:
        c01 += 1.0
    elif v3a == 0.0 and v3b == 2.0:
        c02 += 1.0
    elif v3a == 1.0 and v3b == 2.0:
        c12 += 1.0
    if c01 == 2.0 and c02 == 2.0:
        k = 0; iw = 1; jw = 2
    elif c01 == 2.0 and c12 == 2.0:
        k = 1; iw = 0; jw = 2
    elif c02 == 2.0 and c12 == 2.0:
        k = 2; iw = 0; jw = 1
    else:
        return False, 0.0, False

    # ---- circle data by role --------------------------------------------------
    if k == 0:
        okx = s0x; oky = s0y; okr = s0r; mmk = mm0
    elif k == 1:
        okx = s1x; oky = s1y; okr = s1r; mmk = mm1
    else:
        okx = s2x; oky = s2y; okr = s2r; mmk = mm2
    if iw == 0:
        oix = s0x; oiy = s0y; oir = s0r; mmi = mm0
    elif iw == 1:
        oix = s1x; oiy = s1y; oir = s1r; mmi = mm1
    else:
        oix = s2x; oiy = s2y; oir = s2r; mmi = mm2
    if jw == 0:
        ojx = s0x; ojy = s0y; ojr = s0r; mmj = mm0
    elif jw == 1:
        ojx = s1x; ojy = s1y; ojr = s1r; mmj = mm1
    else:
        ojx = s2x; ojy = s2y; ojr = s2r; mmj = mm2

    # ---- W / X vertex collection ------------------------------------------------
    # W-pair = (iw, k) sorted; X-pair = (jw, k) sorted
    wpa = min(float(iw), float(k)); wpb = max(float(iw), float(k))
    xpa = min(float(jw), float(k)); xpb = max(float(jw), float(k))
    w1x = w1y = w2x = w2y = x1x = x1y = x2x = x2y = 0.0
    nw = nx = 0
    if v0a == wpa and v0b == wpb:
        if nw == 0:
            w1x = v0x; w1y = v0y; nw = 1
        else:
            w2x = v0x; w2y = v0y
    else:
        if nx == 0:
            x1x = v0x; x1y = v0y; nx = 1
        else:
            x2x = v0x; x2y = v0y
    if v1a == wpa and v1b == wpb:
        if nw == 0:
            w1x = v1x; w1y = v1y; nw = 1
        else:
            w2x = v1x; w2y = v1y
    else:
        if nx == 0:
            x1x = v1x; x1y = v1y; nx = 1
        else:
            x2x = v1x; x2y = v1y
    if v2a == wpa and v2b == wpb:
        if nw == 0:
            w1x = v2x; w1y = v2y; nw = 1
        else:
            w2x = v2x; w2y = v2y
    else:
        if nx == 0:
            x1x = v2x; x1y = v2y; nx = 1
        else:
            x2x = v2x; x2y = v2y
    if v3a == wpa and v3b == wpb:
        if nw == 0:
            w1x = v3x; w1y = v3y; nw = 1
        else:
            w2x = v3x; w2y = v3y
    else:
        if nx == 0:
            x1x = v3x; x1y = v3y; nx = 1
        else:
            x2x = v3x; x2y = v3y

    # ---- angles on circle k + insertion sort of the 4 records -------------------
    r0a = np.arctan2(w1y - oky, w1x - okx); r0x = w1x; r0y = w1y; r0w = 1.0
    r1a = np.arctan2(w2y - oky, w2x - okx); r1x = w2x; r1y = w2y; r1w = 1.0
    r2a = np.arctan2(x1y - oky, x1x - okx); r2x = x1x; r2y = x1y; r2w = 0.0
    r3a = np.arctan2(x2y - oky, x2x - okx); r3x = x2x; r3y = x2y; r3w = 0.0
    # insertion sort (ascending angles), records swapped field by field
    if r1a < r0a:
        ta = r1a; r1a = r0a; r0a = ta
        tx = r1x; r1x = r0x; r0x = tx
        ty = r1y; r1y = r0y; r0y = ty
        tw = r1w; r1w = r0w; r0w = tw
    if r2a < r1a:
        ta = r2a; r2a = r1a; r1a = ta
        tx = r2x; r2x = r1x; r1x = tx
        ty = r2y; r2y = r1y; r1y = ty
        tw = r2w; r2w = r1w; r1w = tw
    if r1a < r0a:
        ta = r1a; r1a = r0a; r0a = ta
        tx = r1x; r1x = r0x; r0x = tx
        ty = r1y; r1y = r0y; r0y = ty
        tw = r1w; r1w = r0w; r0w = tw
    if r3a < r2a:
        ta = r3a; r3a = r2a; r2a = ta
        tx = r3x; r3x = r2x; r2x = tx
        ty = r3y; r3y = r2y; r2y = ty
        tw = r3w; r3w = r2w; r2w = tw
    if r2a < r1a:
        ta = r2a; r2a = r1a; r1a = ta
        tx = r2x; r2x = r1x; r1x = tx
        ty = r2y; r2y = r1y; r1y = ty
        tw = r2w; r2w = r1w; r1w = tw
    if r1a < r0a:
        ta = r1a; r1a = r0a; r0a = ta
        tx = r1x; r1x = r0x; r0x = tx
        ty = r1y; r1y = r0y; r0y = ty
        tw = r1w; r1w = r0w; r0w = tw

    # ---- gap midpoint tests (gap g: record g -> record g+1, wrap for g = 3) ------
    # midpoint inside the two non-k disks; antipodal gaps fall back
    p0 = _gap_pass(r0a, r0x, r0y, r1a, r1x, r1y, okx, oky, okr,
                   oix, oiy, mmi, ojx, ojy, mmj)
    p1 = _gap_pass(r1a, r1x, r1y, r2a, r2x, r2y, okx, oky, okr,
                   oix, oiy, mmi, ojx, ojy, mmj)
    p2 = _gap_pass(r2a, r2x, r2y, r3a, r3x, r3y, okx, oky, okr,
                   oix, oiy, mmi, ojx, ojy, mmj)
    p3 = _gap_pass(r3a, r3x, r3y, r0a + TWO_PI, r0x, r0y, okx, oky, okr,
                   oix, oiy, mmi, ojx, ojy, mmj)

    # ---- structure: exactly two passing, opposite gaps, each W--X -------------
    # The undirected boundary is the 4-cycle
    #   (gaps {0,2}): r0-k-r1-?-r2-k-r3-?-r0     (gaps {1,3}): shift by one
    # where ? is the i-arc (W-W) on one side and the j-arc (X-X) on the other.
    # k-gap directions are forced (CCW around O_k). Only the ALTERNATING
    # arrangements W X X W / X W W X admit a consistent directed cycle; the
    # other alternations (W X W X, X W X W) have no consistent direction
    # assignment (v1 would report degenerate), so they fall back. For the
    # consistent arrangements the required start vertices of the i- and
    # j-arcs are fixed; mismatches fall back to the generic walk.
    area = 0.0
    reflex = False
    req_sw_i = -1.0   # required _v3_arc swapped flag; -1 = no valid arrangement
    req_sw_j = -1.0

    if p0 and p2:
        if r0w + r1w != 1.0 or r2w + r3w != 1.0:
            return False, 0.0, False
        d0 = r1a - r0a
        d2 = r3a - r2a
        area += 0.5 * (okr * okr * d0 + okx * (r1y - r0y) - oky * (r1x - r0x))
        area += 0.5 * (okr * okr * d2 + okx * (r3y - r2y) - oky * (r3x - r2x))
        if d0 > np.pi:
            reflex = True
        if d2 > np.pi:
            reflex = True
        if r0w == 1.0 and r3w == 1.0:
            # W X X W: r0->r1 (k), r1->r2 (j), r2->r3 (k), r3->r0 (i)
            # i-arc start = the W at r3; j-arc start = the X at r1
            if r3x == w1x and r3y == w1y:
                req_sw_i = 0.0
            else:
                req_sw_i = 1.0
            if r1x == x1x and r1y == x1y:
                req_sw_j = 0.0
            else:
                req_sw_j = 1.0
        elif r0w == 0.0 and r3w == 0.0:
            # X W W X: r0->r1 (k), r1->r2 (i), r2->r3 (k), r3->r0 (j)
            # i-arc start = the W at r1; j-arc start = the X at r3
            if r1x == w1x and r1y == w1y:
                req_sw_i = 0.0
            else:
                req_sw_i = 1.0
            if r3x == x1x and r3y == x1y:
                req_sw_j = 0.0
            else:
                req_sw_j = 1.0
    elif p1 and p3:
        if r1w + r2w != 1.0 or r3w + r0w != 1.0:
            return False, 0.0, False
        d1 = r2a - r1a
        d3 = r0a + TWO_PI - r3a
        area += 0.5 * (okr * okr * d1 + okx * (r2y - r1y) - oky * (r2x - r1x))
        area += 0.5 * (okr * okr * d3 + okx * (r0y - r3y) - oky * (r0x - r3x))
        if d1 > np.pi:
            reflex = True
        if d3 > np.pi:
            reflex = True
        if r1w == 1.0 and r0w == 1.0:
            # (r1,r2,r3,r0) = W X X W: r1->r2 (k), r2->r3 (j), r3->r0 (k), r0->r1 (i)
            # i-arc start = the W at r0; j-arc start = the X at r2
            if r0x == w1x and r0y == w1y:
                req_sw_i = 0.0
            else:
                req_sw_i = 1.0
            if r2x == x1x and r2y == x1y:
                req_sw_j = 0.0
            else:
                req_sw_j = 1.0
        elif r1w == 0.0 and r0w == 0.0:
            # (r1,r2,r3,r0) = X W W X: r1->r2 (k), r2->r3 (i), r3->r0 (k), r0->r1 (j)
            # i-arc start = the W at r2; j-arc start = the X at r0
            if r2x == w1x and r2y == w1y:
                req_sw_i = 0.0
            else:
                req_sw_i = 1.0
            if r0x == x1x and r0y == x1y:
                req_sw_j = 0.0
            else:
                req_sw_j = 1.0

    if req_sw_i < 0.0:
        return False, 0.0, False   # no consistent arrangement -> generic walk

    # ---- arcs on circles i (between the W's) and j (between the X's) -------------
    ok, contrib, refl, sw = _v3_arc(w1x, w1y, w2x, w2y, oix, oiy, oir,
                                    ojx, ojy, mmj, okx, oky, mmk)
    if not ok or sw != req_sw_i:
        return False, 0.0, False   # direction mismatch -> generic walk
    area += contrib
    if refl:
        reflex = True
    ok, contrib, refl, sw = _v3_arc(x1x, x1y, x2x, x2y, ojx, ojy, ojr,
                                    oix, oiy, mmi, okx, oky, mmk)
    if not ok or sw != req_sw_j:
        return False, 0.0, False   # direction mismatch -> generic walk
    area += contrib
    if refl:
        reflex = True

    # ---- plausibility (surprises fall back to the generic walk) -------------------
    if area < -1e-9 * scale * scale or area > a_min_disk * (1.0 + 1e-9) + 1e-12:
        return False, 0.0, False
    return True, max(area, 0.0), reflex


@njit(cache=True)
def _gap_pass(a1, x1, y1, a2, x2, y2, ox, oy, r, c1x, c1y, mm1, c2x, c2y, mm2):
    """Midpoint of the CCW gap arc (a1 -> a2, a2 may exceed 2pi) on circle
    (ox, oy, r) strictly inside the two other disks? Antipodal endpoints
    (gap ~ pi, w ~ 0) return False (caller falls back)."""
    if a2 - a1 >= np.pi:
        short = False
    else:
        short = True
    ux_ = (x1 - ox) / r
    uy_ = (y1 - oy) / r
    vx_ = (x2 - ox) / r
    vy_ = (y2 - oy) / r
    wx = ux_ + vx_
    wy = uy_ + vy_
    ww = wx * wx + wy * wy
    if ww < 1e-16:
        return False   # antipodal: ambiguous, fall back
    inv = r / np.sqrt(ww)
    if short:
        pmx = ox + wx * inv
        pmy = oy + wy * inv
    else:
        pmx = ox - wx * inv
        pmy = oy - wy * inv
    return (_inside(pmx, pmy, c1x, c1y, mm1)
            and _inside(pmx, pmy, c2x, c2y, mm2))


@njit(cache=True)
def er_fast(xp, yp, rp, xm, ym, rm):
    """Analytic replacement for Pandora's pixelart() (optimized):
        er = min(1, A(star & planet & moon) / A(star & moon)),
    with the A_SM = 0 -> 1 branch (computed first, so off-star moons
    return immediately without evaluating the triple overlap)."""
    a_sm = _lens_area(1.0, rm, np.sqrt(xm * xm + ym * ym))
    if a_sm <= 0.0:
        return 1.0
    a_smp = area3_fast_core(0.0, 0.0, 1.0, xp, yp, rp, xm, ym, rm)[0]
    return min(1.0, a_smp / a_sm)
