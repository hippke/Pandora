"""Phase 1, component 4 (GPU_PLAN.md): the area of the common overlap of disks, formulated for FP32.

Boundary of the intersection of disks = for every circle i, the arcs of circle i that lie inside all other disks.
Each other disk j restricts circle i to one angular interval (or none / all of it); two intervals on a circle
intersect in at most two arcs. The area is, by Green's theorem, the sum over the arcs of
    0.5 r_i^2 (L - sin L) + 0.5 P_s x P_e          (circular segment + shoelace term of the chord)
with P_s, P_e the arc's end points. All coordinates are relative to the moon centre, so the polygon terms are
small; the half-angles come from the Heron form of the half-chord with d - r = (d^2 - r^2) / (d + r) (no
cancellation for the star circle r = 1 at d ~ 1); L - sin L by its series for small L. There is no vertex walk and
no tangency classification: tangencies give intervals of length ~ 0 and contribute ~ 0 (continuous).

area_disks(circles, dtype) is the reference (FP64, or FP32 emulated with numpy float32 scalars, no FMA).
er(xp, yp, rp, xm, ym, rm) = min(1, A(star & planet & moon) / A(star & moon)), the quantity of
pandoramoon.overlap_analytic.er_fast.

    python area3.py [N]      # fuzz test against er_fast (FP64 and emulated FP32)
"""
import math
import sys

import numpy as np

TWO_PI = 2 * math.pi


def _seg(L, f):
    """L - sin L (series for small L)"""
    if L < f(0.5):
        L2 = L * L
        return L * L2 * (f(1 / 6) - L2 * (f(1 / 120) - L2 * (f(1 / 5040) - L2 * f(1 / 362880))))
    return L - f(math.sin(L))


def _interval(ci, ri, cj, rj, f):
    """circle i restricted to disk j: ('none' | 'full' | 'disjoint' | (start, length))"""
    dx = cj[0] - ci[0]
    dy = cj[1] - ci[1]
    d2 = dx * dx + dy * dy
    d = f(math.sqrt(d2))
    if d >= ri + rj:
        return "disjoint"
    if d <= rj - ri:
        return "full"
    if d <= ri - rj:
        return "none"
    dmi = (d2 - ri * ri) / (d + ri)          # d - r_i, stable
    dmj = (d2 - rj * rj) / (d + rj)          # d - r_j
    f1 = ri + rj + d
    f2 = rj - dmi                             # r_i + r_j - d
    f3 = dmj + ri                             # d + r_i - r_j
    f4 = dmi + rj                             # d - r_i + r_j
    p = f1 * f2 * f3 * f4
    h = f(math.sqrt(max(p, f(0)))) / (f(2) * d)
    a = (d2 + (ri - rj) * (ri + rj)) / (f(2) * d)
    alpha = f(math.atan2(h, a))
    phi = f(math.atan2(dy, dx))
    return (phi - alpha, f(2) * alpha)


def _wrap(x, f):
    x = x % f(TWO_PI)
    return x if x >= 0 else x + f(TWO_PI)


def _arc_terms(c, r, s, L, f):
    e = s + L
    psx = c[0] + r * f(math.cos(s))
    psy = c[1] + r * f(math.sin(s))
    pex = c[0] + r * f(math.cos(e))
    pey = c[1] + r * f(math.sin(e))
    return f(0.5) * r * r * _seg(L, f) + f(0.5) * (psx * pey - psy * pex)


def area_disks(circles, dtype=np.float64):
    """area of the common overlap of 2 or 3 disks [(cx, cy, r), ...] (coordinates as given)"""
    f = dtype
    C = [(f(c[0]), f(c[1])) for c in circles]
    R = [f(c[2]) for c in circles]
    n = len(circles)
    area = f(0)
    for i in range(n):
        ivs = []
        empty = False
        for j in range(n):
            if j == i:
                continue
            iv = _interval(C[i], R[i], C[j], R[j], f)
            if iv == "disjoint":
                return f(0)
            if iv == "none":
                empty = True
            elif iv != "full":
                ivs.append(iv)
        if empty:
            continue
        if not ivs:                               # circle i inside all other disks: the overlap is disk i
            return f(math.pi) * R[i] * R[i]
        if len(ivs) == 1:
            arcs = [ivs[0]]
        else:
            (s1, L1), (s2, L2) = ivs
            dl = _wrap(s2 - s1, f)
            arcs = []
            if dl < L1:
                arcs.append((s1 + dl, min(L1, dl + L2) - dl))
            if dl + L2 > f(TWO_PI):
                arcs.append((s1, min(L1, dl + L2 - f(TWO_PI))))
        for s, L in arcs:
            if L > 0:
                area += _arc_terms(C[i], R[i], s, L, f)
    return area if area > 0 else f(0)


def er(xp, yp, rp, xm, ym, rm, dtype=np.float64):
    """min(1, A(star & planet & moon) / A(star & moon)), moon-centred coordinates"""
    f = dtype
    star = (-f(xm), -f(ym), f(1))
    pl = (f(xp) - f(xm), f(yp) - f(ym), f(rp))
    mo = (f(0), f(0), f(rm))
    a_sm = area_disks([star, mo], dtype)
    if a_sm <= 0:
        return f(1)
    return min(f(1), area_disks([star, pl, mo], dtype) / a_sm)


def fuzz(n, seed=1):
    """random limb-eclipse configurations as eclipse() sees them: an overlap and at least one body on the limb"""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        rp = rng.uniform(0.02, 0.25)
        rm = rng.uniform(0.005, 0.12) if rng.uniform() < 0.9 else rng.uniform(0.005, 0.3)
        # moon near the limb, planet within rp + rm of the moon
        zm = 1 + rng.uniform(-1.2, 1.2) * rm
        th = rng.uniform(0, 2 * np.pi)
        xm, ym = zm * np.cos(th), zm * np.sin(th)
        dd = rng.uniform(0, 1) ** 0.7 * (rp + rm)
        ph = rng.uniform(0, 2 * np.pi)
        xp, yp = xm + dd * np.cos(ph), ym + dd * np.sin(ph)
        zp = math.hypot(xp, yp)
        if not (abs(1 - zm) < rm or abs(1 - zp) < rp):
            continue
        if not (math.hypot(xp - xm, yp - ym) < rp + rm):
            continue
        out.append((xp, yp, rp, xm, ym, rm))
    return np.array(out)


if __name__ == "__main__":
    from pandoramoon.overlap_analytic import er_fast
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    X = fuzz(N)
    ref = np.array([er_fast(*x) for x in X])
    nan = ~np.isfinite(ref)
    e64 = np.array([er(*x) for x in X])
    e32 = np.array([float(er(*x, dtype=np.float32)) for x in X])
    ok = ~nan
    # the moon deficit error is D_moon |d er| with D_moon <~ A(star & moon) / pi (limb darkening <= 1 / (1 - u1/3 - u2/6)
    # ~ 1.25 above the mean): the deficit scale A_sm / pi
    from pandoramoon.overlap_analytic import _lens_area
    a_sm = np.array([_lens_area(1.0, x[5], math.hypot(x[3], x[4])) for x in X])
    for lab, e in (("FP64", e64), ("FP32 (emulated, no FMA)", e32)):
        d = np.abs(e[ok] - ref[ok])
        dd = d * a_sm[ok] / np.pi
        print(f"{lab}: |er - er_fast| median {np.median(d):.2e}, 99.9% {np.quantile(d, 0.999):.2e}, max {d.max():.2e}; "
              f"deficit scale |d er| A_sm / pi: median {np.median(dd):.1e}, 99.9% {np.quantile(dd, 0.999):.1e}, max {dd.max():.2e}")
    print(f"er_fast NaN (degenerate) cases: {nan.sum()} of {N}")
    w = np.argsort(-np.abs(e32 - ref) * a_sm * ok)[:3]
    for i in w:
        print("  worst FP32 (deficit scale):", np.round(X[i], 6), "ref", ref[i], "fp32", e32[i], "A_sm", a_sm[i])
