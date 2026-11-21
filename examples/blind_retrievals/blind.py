"""Helpers for the blind-retrieval notebooks (six synthetic PLATO-like light curves, lc_a ... lc_f).

Each notebook runs the analysis step by step with Pandora's own functions (prepare_lightcurve, GP.fit, Likelihood,
default_priors, detect, NestedSampler). This module holds what is the same for all six light curves, so that the
notebooks stay readable:

- loading the data (feather files: time [s], flux normalised to 1),
- data checks: gaps, offsets, ramps at the segment edges, outliers, noise that changes with time,
- a model-free first look at the transits (period, epoch, depth, duration),
- the noise model (a running local white-noise level, a red-noise factor),
- the two light-curve preparations (biweight detrending; raw flux for the simultaneous GP),
- plots (light curve, transits with models, corner plots) and the physics of a planet+moon solution.
"""
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandoramoon as pandora
from pandoramoon.gp import GP
from pandoramoon.constants import G, DAY, R_SUN, M_SUN, M_JUP
from pandoramoon.search import model_flux, cadence_times

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R_JUP = 71_492_000.0      # [m] equatorial
R_EARTH = 6_371_000.0     # [m]
M_EARTH = 5.9722e24       # [kg]
AU = 1.495978707e11       # [m]

C_DATA, C_BIN, C_PLANET, C_MOON, C_TREND = "0.65", "k", "C0", "C3", "C1"


# --------------------------------------------------------------------------------------------- data

def load(name):
    """time [d] and flux of data/lc_<name>.ftr (the file has time in seconds)"""
    df = pd.read_feather(os.path.join(DATA, f"lc_{name}.ftr"))
    return df["time"].to_numpy(float) / DAY, df["flux"].to_numpy(float)


def segments(t, gap=0.02):
    """index ranges [i0, i1) of the continuous stretches (separated by gaps longer than `gap` days)"""
    g = np.where(np.diff(t) > gap)[0]
    edges = np.r_[0, g + 1, len(t)]
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def binned(t, y, width):
    """mean of y in bins of `width` (same unit as t); empty bins dropped"""
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    k = np.floor((t - t[0]) / width).astype(np.int64)
    n = np.bincount(k)
    s = np.bincount(k, y)
    st = np.bincount(k, t)
    m = n > 0
    return st[m] / n[m], s[m] / n[m], n[m]


def local_sigma(t, f, width=2.0, mask=None):
    """White-noise level at each point: the robust point-to-point scatter (1.4826 MAD of the differences / sqrt 2)
    in a running window of `width` days. Differences of neighbouring points do not see slow signals (a moon's dip,
    stellar trends). They do see a transit's ingress and egress: with a deep transit these steep steps raise the
    running median by 15-20 % within a day of each transit, exactly where the errors matter. mask: points to leave
    out (e.g. the transits, |t - t_c| < T14 / 2 + a cadence); the estimate is interpolated across them."""
    d = np.diff(f) / math.sqrt(2.0)
    td = 0.5 * (t[1:] + t[:-1])
    ok = np.diff(t) < 1.5 * np.median(np.diff(t))
    if mask is not None:
        ok &= ~(mask[1:] | mask[:-1])
    n = max(50, int(width / np.median(np.diff(t))))
    s = pd.Series(np.abs(d[ok])).rolling(n, center=True, min_periods=n // 4).median().to_numpy() * 1.4826
    s = pd.Series(s).bfill().ffill().to_numpy()
    return np.interp(t, td[ok], s)


def data_checks(t, f, edge=1.0, verbose=True):
    """Checks of a raw light curve. Returns a dict with: segments (start, end, n, mean, sigma, slope), gap jumps,
    the mean flux in the first / last `edge` days of each segment against the middle, and outlier counts."""
    seg = segments(t)
    rows = []
    for i0, i1 in seg:
        tt, ff = t[i0:i1], f[i0:i1]
        m = ff > 1 - 5 * 1.4826 * np.median(np.abs(np.diff(ff)))     # leave out deep transit points
        s = 1.4826 * np.median(np.abs(np.diff(ff[m]))) / math.sqrt(2)
        slope = np.polyfit(tt[m] - tt[m].mean(), ff[m], 1)[0]
        rows.append(dict(start=tt[0], end=tt[-1], n=i1 - i0, mean_ppm=1e6 * (ff[m].mean() - 1), sigma_ppm=1e6 * s,
                         slope_ppm_per_d=1e6 * slope))
    jumps = []
    for (a0, a1), (b0, b1) in zip(seg[:-1], seg[1:]):
        pre = slice(max(a0, a1 - 72), a1)
        post = slice(b0, min(b1, b0 + 72))
        e = math.sqrt(np.var(f[pre]) / 72 + np.var(f[post]) / 72)
        jumps.append(dict(t=t[a1 - 1], gap=t[b0] - t[a1 - 1], jump_ppm=1e6 * (f[post].mean() - f[pre].mean()),
                          err_ppm=1e6 * e))
    head, tail = [], []
    for i0, i1 in seg:
        tt, ff = t[i0:i1], f[i0:i1]
        mid = (tt > tt[0] + 2 * edge) & (tt < tt[-1] - 2 * edge) & (ff > 1 - 3e-4)
        h = (tt < tt[0] + edge) & (ff > 1 - 3e-4)
        e = (tt > tt[-1] - edge) & (ff > 1 - 3e-4)
        if mid.sum() > 100:
            head.append(1e6 * (ff[h].mean() - ff[mid].mean()))
            tail.append(1e6 * (ff[e].mean() - ff[mid].mean()))
    sig = local_sigma(t, f)
    r = f - pd.Series(f).rolling(13, center=True, min_periods=3).median().to_numpy()
    z = r / sig
    # autocorrelation of the noise: residuals from a 1-day running median, divided by sigma(t), transits left out
    med = pd.Series(f).rolling(145, center=True, min_periods=20).median().to_numpy()
    zz = (f - med) / sig
    zz[(f < 1 - 3e-4) | ~np.isfinite(zz)] = np.nan
    acf = {}
    for lag in (1, 2, 3, 6, 12, 36):
        a, b = zz[:-lag], zz[lag:]
        ok = np.isfinite(a) & np.isfinite(b) & (t[lag:] - t[:-lag] < 1.01 * lag * np.median(np.diff(t)))
        acf[lag] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    zf = zz[np.isfinite(zz)]
    kurt = float(np.mean(zf ** 4) / np.var(zf) ** 2 - 3)
    out = dict(segments=rows, jumps=jumps, head_ppm=np.array(head), tail_ppm=np.array(tail),
               n_high5=int((z > 5).sum()), n_low5=int((z < -5).sum()), sigma=sig, acf=acf, kurtosis=kurt,
               cadence_s=float(np.median(np.diff(t)) * DAY), n=len(t), span=float(t[-1] - t[0]))
    if verbose:
        print(f"{out['n']} points, cadence {out['cadence_s']:.3f} s, span {out['span']:.2f} d, {len(seg)} segments")
        print("segment      start      end       N   mean [ppm]  sigma [ppm/cadence]  slope [ppm/d]")
        for i, r_ in enumerate(rows):
            print(f"  {i:2d}  {r_['start']:9.2f} {r_['end']:9.2f} {r_['n']:7d}   {r_['mean_ppm']:+8.1f}   "
                  f"{r_['sigma_ppm']:10.1f}           {r_['slope_ppm_per_d']:+7.3f}")
        print("jumps across the gaps (12 h after minus 12 h before) [ppm]: " +
              ", ".join(f"{j['jump_ppm']:+.0f}±{j['err_ppm']:.0f}" for j in jumps))
        print(f"first {edge:g} d of each segment vs its middle: mean {np.mean(head):+.1f} ppm "
              f"(each: {', '.join(f'{x:+.0f}' for x in head)})")
        print(f"last  {edge:g} d of each segment vs its middle: mean {np.mean(tail):+.1f} ppm "
              f"(each: {', '.join(f'{x:+.0f}' for x in tail)})")
        print(f"white noise (running 2-d estimate): {1e6 * sig.min():.0f} ... {1e6 * sig.max():.0f} ppm per cadence")
        print(f"single points > 5 sigma above / below a running median: {out['n_high5']} / {out['n_low5']} "
              f"(white noise expects {len(t) * 2.87e-7:.2f} each)")
        print("noise autocorrelation at lags of 1, 2, 3, 6, 12, 36 cadences: " +
              ", ".join(f"{v:+.3f}" for v in acf.values()) + f"; excess kurtosis {kurt:+.2f} (Gaussian: 0)")
    return out


def plot_checks(t, f, checks, ephem=None, edge=1.0):
    """Figure: (1) the raw light curve with 6-h bins, (2) the running white-noise level, (3) the flux in the first
    and last 3 days of every segment (ramps at the segment edges)."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1.2])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, 1e3 * (f - 1), ".", ms=0.5, color=C_DATA, rasterized=True, label="10-min data")
    tb, fb, _ = binned(t, f, 0.25)
    ax.plot(tb, 1e3 * (fb - 1), ".", ms=2, color=C_BIN, label="6-h bins")
    for j in checks["jumps"]:
        ax.axvspan(j["t"], j["t"] + j["gap"], color="C1", alpha=0.3, lw=0)
    ax.set_ylabel("flux - 1 [ppt]")
    ax.set_xlabel("time [d]")
    ax.legend(loc="lower right", fontsize=8, markerscale=4)
    ax.set_title("raw light curve (orange: gaps)", loc="left")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, 1e6 * checks["sigma"], color="C0", lw=1)
    ax.set_ylabel("white noise [ppm / cadence]")
    ax.set_xlabel("time [d]")
    ax.set_title("noise level (running 2-day estimate from point-to-point differences)", loc="left")
    axh, axt = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1], sharey=None)
    seg = segments(t)
    for k, (i0, i1) in enumerate(seg):
        tt, ff = t[i0:i1], f[i0:i1]
        ok = ff > 1 - 3e-4
        x, y, _ = binned(tt[ok] - tt[0], ff[ok], 2 / 24)
        axh.plot(x[x < 3], 1e6 * (y[x < 3] - 1), lw=0.8, alpha=0.7)
        x, y, _ = binned(tt[ok] - tt[-1], ff[ok], 2 / 24)
        axt.plot(x[x > -3], 1e6 * (y[x > -3] - 1), lw=0.8, alpha=0.7)
    for a in (axh, axt):
        a.axhline(0, color="k", lw=0.5)
        a.set_ylim(-120, 120)
        a.set_ylabel("flux - 1 [ppm], 2-h bins")
    axh.axvspan(0, edge, color="C3", alpha=0.08)
    axt.axvspan(-edge, 0, color="C3", alpha=0.08)
    axh.set_xlabel("days after a segment start")
    axt.set_xlabel("days before a segment end")
    axh.set_title("segment starts (one line per segment)", loc="left")
    axt.set_title("segment ends", loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------- first look at transits

def find_transits(t, f, sigma, nsig=5.0, min_points=4):
    """Model-free transit finder: runs of >= min_points cadences more than nsig sigma below 1. Returns the
    transit mid-times (half-depth crossings), depths, half-depth durations, and a linear ephemeris (P, T0)."""
    low = np.where(f < 1 - nsig * sigma)[0]
    if len(low) == 0:
        return None
    br = np.where(np.diff(t[low]) > 0.3)[0]
    runs = [(low[a], low[b]) for a, b in zip(np.r_[0, br + 1], np.r_[br, len(low) - 1]) if low[b] - low[a] >= min_points]
    depth = float(np.median([1 - np.percentile(f[a:b + 1], 10) for a, b in runs]))
    mids, durs = [], []
    for a, b in runs:
        i0, i1 = max(a - 150, 0), min(b + 150, len(t) - 1)
        tt, ff = t[i0:i1 + 1], f[i0:i1 + 1]
        fs = np.convolve(ff, np.ones(3) / 3, mode="same")
        fs[0], fs[-1] = ff[0], ff[-1]
        lvl = 1 - depth / 2
        j = np.where(fs < lvl)[0]
        j0, j1 = j[0], j[-1]
        c0 = tt[j0 - 1] + (lvl - fs[j0 - 1]) * (tt[j0] - tt[j0 - 1]) / (fs[j0] - fs[j0 - 1])
        c1 = tt[j1] + (lvl - fs[j1]) * (tt[j1 + 1] - tt[j1]) / (fs[j1 + 1] - fs[j1])
        mids.append(0.5 * (c0 + c1))
        durs.append(c1 - c0)
    mids, durs = np.array(mids), np.array(durs)
    if len(mids) > 1:
        P0 = np.min(np.diff(mids))
        n = np.round((mids - mids[0]) / P0).astype(int)
        P, T0 = np.polyfit(n, mids, 1)
    else:
        n, P, T0 = np.zeros(1, int), np.nan, mids[0]
    return dict(tmid=mids, epoch=n, depth=depth, dur_half=durs, P=float(P), T0=float(T0))


def transit_duration(t, f, ft, sigma):
    """T14 (first to last contact) from the stack of all transits: the span where the 30-min binned stacked flux is
    more than max(3 sigma, 5 % of the depth) below 1 (the 5 % keeps shallow moon dips next to the transit out)"""
    x = np.concatenate([t[np.abs(t - tc) < 2] - tc for tc in ft["tmid"]])
    y = np.concatenate([f[np.abs(t - tc) < 2] for tc in ft["tmid"]])
    o = np.argsort(x)
    xb, yb, nb = binned(x[o], y[o], 0.5 / 24)
    sb = np.median(sigma) / np.sqrt(nb)
    inside = xb[yb < 1 - np.maximum(3 * sb, 0.05 * ft["depth"])]
    inside = inside[np.abs(inside) < 1.5 * np.median(ft["dur_half"])]
    return float(inside.max() - inside.min() + 0.5 / 24)


def moon_reach(per_bary, per_moon_max, M_planet_max, M_star=M_SUN, R_star=R_SUN):
    """How far (in days of the planet's motion) a moon can transit from its planet under the moon prior: the
    semi-major axis at the longest prior period and the largest prior planet mass, over the planet's orbital speed.
    Also returns a_bary [R_star] and the planet's Hill radius [R_star] for these masses."""
    a = (G * (M_star + M_planet_max) * (per_bary * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3)
    a_moon = (G * M_planet_max * (per_moon_max * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3)
    v = 2 * math.pi * a / per_bary                          # [m / d]
    r_hill = a * (M_planet_max / (3 * M_star)) ** (1 / 3)
    return dict(reach_days=a_moon / v, a_moon_max=a_moon / R_star, a_bary=a / R_star, r_hill=r_hill / R_star,
                hill_days=r_hill / v)


# ------------------------------------------------------------------------------------ noise and windows

def phase_distance(t, T0, P):
    """|time from the nearest transit| [d]"""
    return np.abs(((t - T0 + 0.5 * P) % P) - 0.5 * P)


def edge_mask(t, edge=1.0):
    """True within `edge` days of the start or end of a segment (the ramps)"""
    m = np.zeros(len(t), bool)
    for i0, i1 in segments(t):
        m[i0:i1] = (t[i0:i1] < t[i0] + edge) | (t[i0:i1] > t[i1 - 1] - edge)
    return m


def red_noise_beta(t, resid, sigma, use, bin_width):
    """Red-noise factor beta (Pont, Zucker & Queloz 2006) of residuals normalised by their own sigma: the rms of
    the weighted bin means (bins of `bin_width` days) relative to white noise. >= 1."""
    tt, z, s = t[use], resid[use], sigma[use]
    k = np.floor((tt - tt[0]) / bin_width).astype(np.int64)
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    w = 1 / s ** 2
    num = np.bincount(inv, z * w)
    den = np.bincount(inv, w)
    ok = cnt >= max(3, int(0.5 * np.median(cnt)))
    zb = num[ok] / den[ok] * np.sqrt(den[ok])
    return max(1.0, float(np.sqrt(np.mean(zb ** 2))))


def prepare_biweight(t, f, sigma, T0, P, T14, halfwidth, bad=None):
    """Biweight route: pandora.prepare_lightcurve (wotan biweight, transits masked over 1.0 T14, window 4.0 T14:
    Pandora's defaults), with fit windows of +-halfwidth around each transit. Then the error bars are replaced by
    the local white noise sigma(t) times the red-noise factor beta of the detrended data outside the windows
    (normalised by sigma(t), bins of T14/4). bad: points to leave out (NaN)."""
    ff = f.copy()
    if bad is not None:
        ff[bad] = np.nan
    lc = pandora.prepare_lightcurve(t, ff, t0=T0, period=P, duration=T14, fit_halfwidth=halfwidth)
    ta = lc["time_all"]
    s_all = np.interp(ta, t, sigma)
    far = np.isfinite(lc["flat"]) & (phase_distance(ta, T0, P) > halfwidth + T14)
    beta = red_noise_beta(ta, lc["flat"] - 1, s_all, far, T14 / 4)
    lc["sigma_t"] = np.interp(lc["time"], t, sigma)
    lc["beta_local"] = beta
    lc["flux_err"] = lc["sigma_t"] * beta
    return lc


def prepare_gp(t, f, sigma, T0, P, halfwidth, bad=None, kernel="sho", baseline="offset", gap=0.5, clip=True,
               cache=None):
    """GP route: hyperparameters by maximum likelihood on the raw flux outside the fit windows (GP.fit with
    the windows masked; flares and outliers clipped), then the raw flux inside the windows for the Likelihood.
    Returns (gp, dict(time, flux, flux_err) of the windows, outlier flags over all points).
    cache: an .npz file. GP.fit's result differs in the last digits between CPUs (here 9.8441e-06 vs 9.84414e-06 on
    an Intel and an AMD machine), and a stored fit is reused only for bit-identical data and GP. So the first run
    stores the hyperparameters and the outlier flags there, and later runs (on any machine) read them back."""
    if cache is not None and os.path.exists(cache):
        z = np.load(cache)
        gp = GP(float(z["amp"]), float(z["tau"]), float(z["err_scale"]), Q=float(z["Q"]),
                baseline=str(z["baseline"]) if str(z["baseline"]) != "None" else None,
                gap=float(z["gap"]) if np.isfinite(z["gap"]) else None, baseline_sd=tuple(z["baseline_sd"]))
        out = z["outliers"].astype(bool)
        if len(out) != len(t):
            raise ValueError(f"{cache}: stored outlier flags do not match the light curve")
        use = np.ones(len(t), bool) if bad is None else ~bad
        keep = (phase_distance(t, T0, P) < halfwidth) & use & ~out
        return gp, dict(time=t[keep], flux=f[keep], flux_err=sigma[keep]), out
    use = np.ones(len(t), bool) if bad is None else ~bad
    win = phase_distance(t, T0, P) < halfwidth
    gp = GP.fit(t[use], f[use], sigma[use], mask=win[use], kernel=kernel, baseline=baseline, gap=gap, clip=clip)
    out = np.zeros(len(t), bool)
    if clip:
        out[np.where(use)[0][gp.outliers]] = True
    if cache is not None:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, amp=gp.amp, tau=gp.tau, err_scale=gp.err_scale, Q=gp.Q, baseline=str(gp.baseline),
                 gap=np.nan if gp.gap is None else gp.gap, baseline_sd=np.array(gp.baseline_sd), outliers=out)
    keep = win & use & ~out
    return gp, dict(time=t[keep], flux=f[keep], flux_err=sigma[keep]), out


# ------------------------------------------------------------------------------------------ results

def evidence_words(dlnz):
    """Jeffreys' scale (as in Pandora's print_results): B = exp(dlnz)"""
    if dlnz < 0:
        return "favours the planet-only model"
    if dlnz < 1.15:
        return "barely worth mentioning"
    if dlnz < 2.3:
        return "substantial"
    if dlnz < 3.45:
        return "strong"
    if dlnz < 4.6:
        return "very strong"
    return "decisive"


def samples_dict(results):
    """equal-weight posterior samples as a dict, with derived columns"""
    S = {n: results["samples"][:, i] for i, n in enumerate(results["paramnames"])}
    if "M_moon" in S:
        S["M_moon/M_planet"] = S["M_moon"] / S["M_planet"]
        S["Omega_moon"] = S["Omega_moon"] % 180.0
    return S


def corner_plot(S, keys, labels=None, title=None, truths=None, **kw):
    import corner
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    X = np.column_stack([S[k] for k in keys])
    fig = corner.corner(X, labels=labels or keys, show_titles=True, title_fmt=".3g", quantiles=[0.16, 0.5, 0.84],
                        truths=truths, **kw)
    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
    return fig


def physics(S, n_max=20000):
    """Physical quantities of the planet+moon posterior samples (dict of arrays). The star's mass follows from
    a_bary, per_bary and R_star (Kepler's law, as Pandora does for the Hill-sphere check)."""
    i = slice(0, min(n_max, len(S["R_star"])))
    Rs, P, a = S["R_star"][i], S["per_bary"][i], S["a_bary"][i]
    Mp, Mm = S["M_planet"][i], S["M_moon"][i]
    rp, rm, Pm = S["r_planet"][i], S["r_moon"][i], S["per_moon"][i]
    M_star = 4 * math.pi ** 2 * (a * Rs) ** 3 / (G * (P * DAY) ** 2)
    a_moon = (G * (Mp + Mm) * (Pm * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3)        # [m]
    r_hill = a * Rs * (Mp / (3 * M_star)) ** (1 / 3)
    Rp, Rm = rp * Rs, rm * Rs
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_p = Mp / (4 / 3 * math.pi * Rp ** 3) / 1000           # [g / cm^3]
        rho_m = Mm / (4 / 3 * math.pi * Rm ** 3) / 1000
        roche = 2.44 * Rp * (rho_p / rho_m) ** (1 / 3)             # fluid Roche limit of the moon [m]
    v = 2 * math.pi * a * Rs / (P * DAY)                          # planet's orbital speed [m / s]
    # the planet's barycentric wobble: a_moon Mm / (Mp + Mm); as a timing offset at transit (maximum):
    ttv = a_moon * Mm / (Mp + Mm) / v / 60                         # [min]
    return {
        "M_star [M_sun]": M_star / M_SUN, "R_star [R_sun]": Rs / R_SUN,
        "rho_star [g/cm^3]": M_star / (4 / 3 * math.pi * Rs ** 3) / 1000,
        "a_bary [AU]": a * Rs / AU,
        "R_planet [R_jup]": Rp / R_JUP, "R_planet [R_earth]": Rp / R_EARTH,
        "M_planet [M_jup]": Mp / M_JUP, "rho_planet [g/cm^3]": rho_p,
        "R_moon [R_earth]": Rm / R_EARTH, "M_moon [M_earth]": Mm / M_EARTH, "rho_moon [g/cm^3]": rho_m,
        "M_moon/M_planet": Mm / Mp,
        "per_moon [d]": Pm, "a_moon [R_planet]": a_moon / Rp, "a_moon [1000 km]": a_moon / 1e6,
        "a_moon / R_Hill": a_moon / r_hill, "R_Hill [R_planet]": r_hill / Rp, "R_Hill [1e6 km]": r_hill / 1e9,
        "a_moon / Roche limit": a_moon / roche,
        "moon dip [ppm]": 1e6 * rm ** 2,
        "max. TTV of the planet [min]": ttv,
        "moon's max. distance from the planet on the sky [h of transit time]": a_moon / v / 3600,
    }


def print_physics(Q, keys=None):
    for k in (keys or Q):
        x = np.asarray(Q[k])
        x = x[np.isfinite(x)]
        lo, med, hi = np.percentile(x, [15.87, 50, 84.13])
        print(f"  {k:62s} {med:10.4g}   (68 %: {lo:.4g} ... {hi:.4g})")


# ------------------------------------------------------------------------------------ light-curve plots

def plot_lightcurve(t, f, T0, P, halfwidth, bw=None, gp_trend=None, bad=None, title=None, edges=None):
    """Total light curve: raw with the biweight trend, the biweight-detrended flux, and the raw flux minus the
    GP's trend (its posterior mean given the best planet+moon model; only inside the fit windows)."""
    rows = 1 + (bw is not None) + (gp_trend is not None)
    fig, axs = plt.subplots(rows, 1, figsize=(14, 2.8 * rows), sharex=True, squeeze=False)
    axs = axs[:, 0]
    ax = axs[0]
    ax.plot(t, 1e3 * (f - 1), ".", ms=0.5, color=C_DATA, rasterized=True)
    if bad is not None and bad.any():
        ax.plot(t[bad], 1e3 * (f[bad] - 1), "x", ms=4, color="C3", label=f"outliers left out ({bad.sum()})")
    if edges is not None and edges.any():
        for i0, i1 in segments(t):
            e = np.where(edges[i0:i1])[0] + i0
            for part in np.split(e, np.where(np.diff(e) > 1)[0] + 1):
                if len(part):
                    for a in axs:
                        a.axvspan(t[part[0]], t[part[-1]], color="C3", alpha=0.12, lw=0)
    if bw is not None:
        ax.plot(bw["time_all"], 1e3 * (bw["trend"] - 1), color=C_TREND, lw=1, label="biweight trend")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylabel("raw flux - 1 [ppt]")
    ax.set_title(title or "raw light curve", loc="left")
    k = 1
    if bw is not None:
        ax = axs[k]
        k += 1
        ax.plot(bw["time_all"], 1e3 * (bw["flat"] - 1), ".", ms=0.5, color=C_DATA, rasterized=True)
        ax.plot(bw["time"], 1e3 * (bw["flux"] - 1), ".", ms=0.8, color="C0", rasterized=True)
        ax.set_ylabel("detrended - 1 [ppt]")
        ax.set_title("biweight-detrended (blue: the fit windows)", loc="left")
    if gp_trend is not None:
        ax = axs[k]
        tg, fg, mu = gp_trend
        ax.plot(tg, 1e3 * (fg - mu - 1), ".", ms=0.8, color="C2", rasterized=True)
        ax.set_ylabel("raw - GP trend - 1 [ppt]")
        ax.set_title("raw flux minus the GP's trend (fit windows only)", loc="left")
    axs[-1].set_xlabel("time [d]")
    fig.tight_layout()
    return fig


def model_components(like, params):
    """Pandora's planet and total flux for a parameter dict at the likelihood's output cadence (exposure-averaged as
    in the fit): (planet component, total). total - planet component is the moon's contribution, <= 0 (with mutual
    eclipses taken into account)."""
    from pandoramoon.pandora import pandora_flux
    p = params
    fp, fm, ft = pandora_flux(p["u1"], p["u2"], p["R_star"], p["per_bary"], p["a_bary"], p["r_planet"], p["b_bary"],
                              p["w_bary"], p["ecc_bary"], p["t0_bary"], p["t0_bary_offset"], p["M_planet"], p["r_moon"],
                              p["per_moon"], p["tau_moon"], p["Omega_moon"], p["i_moon"], p["ecc_moon"], p["w_moon"],
                              p["M_moon"], like.epoch_distance, like.supersampling_factor, like.occult_small_threshold,
                              like.hill_sphere_threshold, like.time)
    return np.asarray(fp), np.asarray(ft)


def best_models(sampler):
    """cadence times, data, and the best-fit planet+moon and planet-only model fluxes of a fitted NestedSampler"""
    L = sampler.likelihood
    tc = cadence_times(L)
    fm = model_flux(L, sampler.parameters())
    fp = model_flux(L, sampler.parameters(model="planet"))
    return tc, np.asarray(L.data), fm, fp


def best_planet_component(sampler):
    """the planet's own flux in the best planet+moon fit, at the likelihood's cadence"""
    fpc, _ = model_components(sampler.likelihood, sampler.parameters())
    return fpc


def gp_trends(sampler, gp, time, flux, flux_err):
    """the GP's trend (posterior mean, Kalman smoother) given the best planet+moon and the best planet-only
    model, at the data times"""
    tc, _, fm, fp = best_models(sampler)
    fpc = best_planet_component(sampler)
    fm_t = np.interp(time, tc, fm)
    fp_t = np.interp(time, tc, fp)
    fpc_t = np.interp(time, tc, fpc)
    mu_m, sd_m = gp.predict(time, flux - fm_t, flux_err)
    mu_p, sd_p = gp.predict(time, flux - fp_t, flux_err)
    return dict(mu_moon=mu_m, sd_moon=sd_m, mu_planet=mu_p, sd_planet=sd_p, f_moon=fm_t, f_planet=fp_t,
                f_planet_comp=fpc_t)


def plot_transits(time, flux, f_moon, f_planet, tmids, labels=None, halfwidth=1.0, bin_h=1.0, title=None,
                  ylim_res=None, model_time=None, model_moon=None, model_planet=None, f_planet_comp=None,
                  model_planet_comp=None):
    """One column per transit. Top: data (grey; 1-h bins black) with the best planet-only model (blue) and the best
    planet+moon model (red). Bottom: the data minus the planet in the planet+moon fit (its planet component), with
    the moon's own transit in that fit (red; total minus planet component, never above zero): what the moon model
    attributes to the moon. For the GP route pass the data minus the GP trend. model_*: finer model curves."""
    n = len(tmids)
    fig, axs = plt.subplots(2, n, figsize=(max(3.2 * n, 7), 6), sharex="col", sharey="row", squeeze=False,
                            gridspec_kw=dict(height_ratios=[2, 1.3]))
    for k, tc in enumerate(tmids):
        m = np.abs(time - tc) < halfwidth
        x = (time[m] - tc) * 24
        a0, a1 = axs[0, k], axs[1, k]
        a0.plot(x, 1e3 * (flux[m] - 1), ".", ms=1.5, color=C_DATA, rasterized=True)
        xb, yb, _ = binned(x, flux[m], bin_h)
        a0.plot(xb, 1e3 * (yb - 1), "o", ms=3, color=C_BIN)
        if model_time is not None:
            mm = np.abs(model_time - tc) < halfwidth
            xm = (model_time[mm] - tc) * 24
            a0.plot(xm, 1e3 * (model_planet[mm] - 1), color=C_PLANET, lw=1.2, label="planet only")
            a0.plot(xm, 1e3 * (model_moon[mm] - 1), color=C_MOON, lw=1.2, label="planet + moon")
            a1.plot(xm, 1e6 * (model_moon[mm] - model_planet_comp[mm]), color=C_MOON, lw=1.5,
                    label="moon's own transit (planet+moon fit)")
        else:
            a0.plot(x, 1e3 * (f_planet[m] - 1), color=C_PLANET, lw=1.2, label="planet only")
            a0.plot(x, 1e3 * (f_moon[m] - 1), color=C_MOON, lw=1.2, label="planet + moon")
            a1.plot(x, 1e6 * (f_moon[m] - f_planet_comp[m]), color=C_MOON, lw=1.5,
                    label="moon's own transit (planet+moon fit)")
        r = (flux - f_planet_comp)[m]
        a1.plot(x, 1e6 * r, ".", ms=1.5, color=C_DATA, rasterized=True)
        xb, rb, _ = binned(x, r, bin_h)
        a1.plot(xb, 1e6 * rb, "o", ms=3, color=C_BIN, label=f"data - planet of that fit ({bin_h:g}-h bins)")
        a1.axhline(0, color="k", lw=0.5)
        a0.set_title(labels[k] if labels else f"t = {tc:.2f} d", fontsize=9)
        a1.set_xlabel("hours from planet mid-transit")
    axs[0, 0].set_ylabel("flux - 1 [ppt]")
    axs[1, 0].set_ylabel("data - planet [ppm]")
    if ylim_res:
        axs[1, 0].set_ylim(*ylim_res)
    axs[0, 0].legend(fontsize=8, loc="lower left")
    axs[1, 0].legend(fontsize=7, loc="lower left")
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def model_on_grid(sampler, tmids, halfwidth, step=2.0 / 1440):
    """best-fit planet+moon and planet-only fluxes on a fine time grid around the given transits (exposure
    averaged as in the fit), for smooth model curves in the plots"""
    L = sampler.likelihood
    tg = np.concatenate([np.arange(tc - halfwidth, tc + halfwidth, step) for tc in tmids])
    p_m, p_p = sampler.parameters(), sampler.parameters(model="planet")
    ss = L.supersampling_factor
    exp = getattr(L, "exposure", None)
    Lg = pandora.Likelihood(tg, np.ones_like(tg), np.ones_like(tg), epoch_distance=L.epoch_distance,
                            supersampling_factor=1, hill_sphere_threshold=L.hill_sphere_threshold)
    if ss > 1:
        from pandoramoon.search import resample_likelihood, cadence_of
        Lg = resample_likelihood(Lg, ss, exposure=cadence_of(L.time) * ss if exp is None else exp)
    fpc, ftot = model_components(Lg, p_m)
    return cadence_times(Lg), ftot, model_flux(Lg, p_p), fpc


# ------------------------------------------------------------------------- GP hyperparameter posterior

def gp_posterior(gp, time, resid1, yerr, n_steps=6000, n_burn=1500, seed=1, step=0.05):
    """Posterior of the GP's hyperparameters (amp, tau, err_scale) given the residuals of the best transit model
    (resid1 = flux - model + 1) at the fit windows: a simple Metropolis sampler in log space with log-uniform
    priors over the GP.fit bounds. The transit fit itself holds the hyperparameters fixed at their GP.fit values;
    this shows how well the data constrain them."""
    rng = np.random.default_rng(seed)
    med = float(np.median(yerr))
    dt = float(np.median(np.diff(time)))
    lo = np.log([0.02 * med, dt, 1 / 3])
    hi = np.log([50 * med, time[-1] - time[0], 3.0])

    def make(x):
        return GP(math.exp(x[0]), math.exp(x[1]), math.exp(x[2]), baseline=gp.baseline, gap=gp.gap,
                  baseline_sd=gp.baseline_sd)

    def lp(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        return make(x).loglike_noise(time, resid1, yerr)

    x = np.log([gp.amp, gp.tau, gp.err_scale])
    x = np.clip(x, lo + 1e-6, hi - 1e-6)
    l0 = lp(x)
    cov = np.diag([0.3, 0.3, 0.02]) ** 2
    chain = []
    acc = 0
    for i in range(n_steps):
        if i == n_burn // 2 and len(chain) > 50:
            cov = np.cov(np.array(chain).T) * 2.38 ** 2 / 3 + 1e-10 * np.eye(3)
        y = rng.multivariate_normal(x, cov)
        l1 = lp(y)
        if np.log(rng.uniform()) < l1 - l0:
            x, l0 = y, l1
            acc += 1
        chain.append(x.copy())
    c = np.exp(np.array(chain[n_burn:]))
    return dict(amp=c[:, 0], tau=c[:, 1], err_scale=c[:, 2], acceptance=acc / n_steps)


def run_record(log_dir, results):
    """Run time, likelihood calls and sampler settings of a full fit, kept in log_dir/run_record.json: a fit
    reused with resume="resume" reports wall = 0 and ncall = 0, so we store the numbers of the run that did the
    work and read them back later."""
    import json
    import datetime
    path = os.path.join(log_dir, "run_record.json")
    if results.get("wall", 0) > 0 and results.get("ncall", 0) > 0 and not results.get("loaded"):
        # a resumed fit that sampled more (Pandora may continue a stored fit, ISSUES item 20) adds to the record
        old = {}
        if os.path.exists(path):
            with open(path) as fh:
                old = json.load(fh)
        rec = dict(wall=float(results["wall"]) + float(old.get("wall", 0.0)),
                   ncall=int(results["ncall"]) + int(old.get("ncall", 0)), nlive=int(results["nlive"]),
                   nsteps=int(results["nsteps"]), status=str(results["status"]),
                   date=datetime.datetime.now().isoformat(timespec="seconds"),
                   sessions=int(old.get("sessions", 1)) + 1 if old else 1)
        os.makedirs(log_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        return rec
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return dict(wall=np.nan, ncall=0, nlive=results.get("nlive"), nsteps=results.get("nsteps"),
                status=results.get("status"), date="?")


def residual_check(time, resid, err, tmids, T14, halfwidth, label="", ax=None, bin_h=0.5):
    """Residuals of the best planet+moon model around each transit: chi2 / N inside the transit (|dt| < T14 / 2) and
    outside, the mean in-transit residual per transit, and a plot of the binned residuals of all transits against
    the time from mid-transit. Spot crossings would show as bumps inside the transit; a wrong transit shape as a
    pattern repeated at ingress / egress."""
    rows = []
    for k, tc in enumerate(tmids):
        dt = time - tc
        m = np.abs(dt) < halfwidth
        inn = m & (np.abs(dt) < T14 / 2)
        out = m & ~inn
        z = resid / err
        rows.append(dict(t=tc, chi2_in=float(np.mean(z[inn] ** 2)) if inn.any() else np.nan,
                         chi2_out=float(np.mean(z[out] ** 2)) if out.any() else np.nan,
                         mean_in_ppm=1e6 * float(np.mean(resid[inn])) if inn.any() else np.nan,
                         err_in_ppm=1e6 * float(np.sqrt(np.sum(err[inn] ** 2)) / max(inn.sum(), 1)), n_in=int(inn.sum())))
    print(f"{label}: residuals of the best planet+moon model (chi2/N: 1 = as expected from the error bars)")
    print("   transit time   chi2/N in transit   chi2/N outside   mean in-transit residual [ppm]")
    for r in rows:
        print(f"   {r['t']:10.3f}      {r['chi2_in']:8.2f}          {r['chi2_out']:8.2f}        "
              f"{r['mean_in_ppm']:+7.1f} +- {r['err_in_ppm']:.1f}")
    if ax is not None:
        for k, tc in enumerate(tmids):
            m = np.abs(time - tc) < halfwidth
            xb, rb, nb = binned((time[m] - tc) * 24, resid[m], bin_h)
            ax.plot(xb, 1e6 * rb, "-", lw=0.8, alpha=0.7)
        ax.axvspan(-12 * T14, 12 * T14, color="0.9", zorder=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("hours from planet mid-transit")
        ax.set_ylabel(f"residual [ppm], {60 * bin_h:.0f}-min bins")
        ax.set_title(f"{label}: data - best planet+moon model (one line per transit; grey: in transit)", loc="left",
                     fontsize=9)
    return rows


def gp_profile_cached(sampler, results):
    """The conservative Bayes factor of a GP fit (NestedSampler.gp_profile), computed once and read back from
    log_dir/gp_profile.json afterwards. Pandora's gp_profile refits with resume="overwrite", so calling it again
    repeats its fits (2.8 hours for lc_b, where the planet+moon model had to be refitted). Sets
    results["gp_profile"] and results["dlogz_conservative"] as gp_profile() does."""
    import json
    path = os.path.join(sampler.log_dir, "gp_profile.json")
    if os.path.exists(path):
        with open(path) as fh:
            out = json.load(fh)
        results["gp_profile"] = out
        results["dlogz_conservative"] = float(min(results["dlogz"], out["dlogz"]))
        return out
    return sampler.gp_profile()


def plot_detrending(t, f, bw, gp, sigma, T0, P, T14, n_transits=3, halfwidth=2.0, bad=None, title=None):
    """The raw flux around the first n transits, zoomed to the out-of-transit level, with the two trend models: the
    biweight trend of prepare_lightcurve (transits masked over 1.0 T14) and the GP's trend, its posterior mean
    predicted from the data outside the transits (|t - t_c| > T14 / 2 + 30 min), so that it shows the trend alone.
    Below: the raw flux minus each trend (1-h bins). Shows how well each detrender follows the systematics, and
    what it leaves near the transits."""
    tc_all = T0 + P * np.arange(int(np.floor((t[0] - T0) / P)), int(np.ceil((t[-1] - T0) / P)) + 1)
    tcs = [tc for tc in tc_all if np.any(np.abs(t - tc) < 0.5 * T14)][:n_transits]
    n = len(tcs)
    use = np.isfinite(f) if bad is None else (np.isfinite(f) & ~bad)
    intr = phase_distance(t, T0, P) < 0.5 * T14 + 0.5 / 24
    fig, axs = plt.subplots(2, n, figsize=(4.6 * n + 1, 6.2), sharey="row", squeeze=False,
                            gridspec_kw=dict(height_ratios=[2, 1]))
    for k, tc in enumerate(tcs):
        m = (np.abs(t - tc) < halfwidth + 1.0) & use
        tt, ff, ss = t[m], f[m], sigma[m]
        mu, sd = gp.predict(tt, ff - 1.0, ss, mask=intr[m])
        bwt = np.interp(tt, bw["time_all"], bw["trend"], left=np.nan, right=np.nan)
        show = np.abs(tt - tc) < halfwidth
        x = (tt - tc) * 24
        a0, a1 = axs[0, k], axs[1, k]
        a0.plot(x[show], 1e6 * (ff[show] - 1), ".", ms=1.5, color=C_DATA, rasterized=True, label="raw flux (10 min)")
        xb, yb, _ = binned(x[show & ~intr[m]], ff[show & ~intr[m]], 1.0)
        a0.plot(xb, 1e6 * (yb - 1), "o", ms=2.5, color=C_BIN, label="1-h bins (out of transit)")
        a0.plot(x[show], 1e6 * (bwt[show] - 1), color="C0", lw=1.8, label="biweight trend")
        a0.plot(x[show], 1e6 * mu[show], color="C3", lw=1.5, label="GP trend (out-of-transit data)")
        a0.fill_between(x[show], 1e6 * (mu - sd)[show], 1e6 * (mu + sd)[show], color="C3", alpha=0.15, lw=0)
        a0.axvspan(-12 * T14, 12 * T14, color="0.85", alpha=0.6, lw=0, zorder=0)
        a0.set_ylim(-250, 200)
        a0.set_title(f"transit at t = {tc:.2f} d", fontsize=9)
        for lab, tr, c in (("biweight", bwt - 1, "C0"), ("GP", mu, "C3")):
            for side in (x < 0, x > 0):                       # before and after the transit: no line across it
                sel = show & ~intr[m] & np.isfinite(tr) & side
                if sel.sum() > 3:
                    xb, rb, _ = binned(x[sel], (ff - 1 - tr)[sel], 1.0)
                    a1.plot(xb, 1e6 * rb, "o-", ms=2, lw=0.6, color=c,
                            label=f"raw - {lab} trend" if side is not None and side[0] else None)
        a1.axhline(0, color="k", lw=0.5)
        a1.axvspan(-12 * T14, 12 * T14, color="0.85", alpha=0.6, lw=0, zorder=0)
        a1.set_ylim(-80, 80)
        a1.set_xlabel("hours from mid-transit")
    axs[0, 0].set_ylabel("flux - 1 [ppm] (transit off-scale)")
    axs[1, 0].set_ylabel("residual [ppm], 1-h bins")
    axs[0, 0].legend(fontsize=7, loc="lower left", markerscale=3)
    axs[1, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(title or "trend models around the first transits (grey band: the transit)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_post_transit(t, f, sigma, T0, P, T14, tmids, halfwidth=3.0, bad=None, skip=None, title=None):
    """All transits stacked: the out-of-transit flux within +-halfwidth days of each transit, minus the median level
    of its segment (data more than 3 d from any transit), in 2-h bins. A real systematic tied to the transits (here:
    a brightening of 5-20 ppm during 1-2 days after each transit, in all six light curves) adds up; noise does not.
    skip: list of (start, end) offsets [d] from mid-transit to leave out (e.g. a known moon dip)."""
    ph = phase_distance(t, T0, P)
    ok = np.ones(len(t), bool) if bad is None else ~bad
    X, Y = [], []
    for tc in tmids:
        seg = [(a, b) for a, b in segments(t) if t[a] <= tc <= t[b - 1]]
        if not seg:
            continue
        a, b = seg[0]
        inseg = np.zeros(len(t), bool)
        inseg[a:b] = True
        ref = np.median(f[ok & inseg & (ph > 3.0)])
        m = ok & (np.abs(t - tc) < halfwidth) & (np.abs(t - tc) > 0.5 * T14 + 0.02)
        for lo, hi in (skip or []):
            m &= ~((t - tc > lo) & (t - tc < hi))
        X.append(t[m] - tc)
        Y.append(f[m] - ref)
    x, y = np.concatenate(X), np.concatenate(Y)
    o = np.argsort(x)
    xb, yb, nb = binned(x[o], y[o], 2 / 24)
    eb = np.median(sigma) / np.sqrt(nb)
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.errorbar(xb * 24, yb * 1e6, eb * 1e6, fmt="o", ms=3, color="k", elinewidth=0.6)
    ax.axhline(0, color="0.5")
    ax.axvspan(-12 * T14, 12 * T14, color="C0", alpha=0.1)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("flux - segment level [ppm], 2-h bins")
    ax.set_title(title or f"{len(X)} transits stacked: out-of-transit flux around the transits (blue: the transit, left out)",
                 loc="left", fontsize=10)
    h = 0.5 * T14 + 1 / 24
    pre = (xb < -h) & (xb > -h - 1.0)
    post = (xb > h) & (xb < h + 1.0)
    w = nb.astype(float)
    print(f"mean level in the day before the transits: {1e6 * np.average(yb[pre], weights=w[pre]):+.1f} ppm; "
          f"in the day after: {1e6 * np.average(yb[post], weights=w[post]):+.1f} ppm "
          f"(+- {1e6 * np.median(sigma) / np.sqrt(w[post].sum()):.1f} ppm)")
    fig.tight_layout()
    return fig


def fit_stored(log_dir):
    """True if log_dir holds a finished planet+moon fit with its planet-only comparison (results files written)"""
    import json
    try:
        st = json.load(open(os.path.join(log_dir, "info", "fit_state.json")))
    except (OSError, ValueError):
        return False
    return bool(st.get("finished")) and all(os.path.exists(os.path.join(log_dir, *p)) for p in
                                            (("info", "results.json"), ("chains", "equal_weighted_post.txt"),
                                             ("planet", "info", "results.json"),
                                             ("planet", "chains", "equal_weighted_post.txt")))


def load_fit(likelihood, priors, log_dir):
    """Read a finished NestedSampler fit from its log_dir, without sampling and without a GPU: the results Pandora
    stored (info/results.json, chains/equal_weighted_post.txt, and the same for planet/), checked against the
    current data and priors (the hashes in info/fit_state.json). Returns (sampler, results) as NestedSampler.run()
    would: the sampler (on the CPU, never run) gives parameters(), print_results() and the likelihood for the plots;
    the epoch jackknife and the notes are recomputed as run() does; wall and ncall are the stored fit's
    (run_record.json). Pandora's own resume="resume" re-merges all stored runs and may sample more (ISSUES item 20);
    this reads exactly what was stored."""
    import json
    s = pandora.NestedSampler(likelihood, priors, log_dir=log_dir, resume="resume", verbose=False, device="cpu")
    st = json.load(open(os.path.join(log_dir, "info", "fit_state.json")))
    stored = st.get("hash_parts") or {}
    now = s._hash_parts("moon", {})
    for part in ("data", "priors"):
        if stored.get(part) and stored[part] != now[part]:
            raise ValueError(f"{log_dir}: the stored fit was made with other {part}; refit (REFIT = True)")

    def read(path):
        r = json.load(open(os.path.join(path, "info", "results.json")))
        with open(os.path.join(path, "chains", "equal_weighted_post.txt")) as fh:
            names = fh.readline().split()
        r["samples"] = np.loadtxt(os.path.join(path, "chains", "equal_weighted_post.txt"), skiprows=1, ndmin=2)
        if list(names) != list(r["paramnames"]):
            raise ValueError(f"{path}: sample columns {names} != paramnames {r['paramnames']}")
        for k in ("mean", "stdev", "median", "errlo", "errup"):
            r["posterior"][k] = np.asarray(r["posterior"][k], dtype=float)
        for k, v in list(r.items()):
            if isinstance(v, str) and v in ("nan", "inf", "-inf"):
                r[k] = float(v)
        return r
    res = read(log_dir)
    res["planet"] = read(os.path.join(log_dir, "planet"))
    rec_path = os.path.join(log_dir, "run_record.json")
    if os.path.exists(rec_path):
        rec = json.load(open(rec_path))
        res["wall"], res["ncall"] = float(rec["wall"]), int(rec["ncall"])
    res.setdefault("warnings", [])
    res["loaded"] = True                    # run_record must not count this as another session
    s.results = res
    s._add_checks(res)
    return s, res
