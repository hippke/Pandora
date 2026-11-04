"""nss: a compiled nested slice sampler for Pandora (experimental).

Description: NSS_SAMPLER.md; development history and validation: SAMPLING.md (Rounds 4-14) in the
project directory. `fit()` is the entry point (independent runs merged, self-checking); `run()` is one
nested-sampling run. Both take a numba-compiled function ll_u(u) of a point u in the unit cube
(prior transform inside). A builder of ll_u from a `pandoramoon.Likelihood` and user priors is not
part of the package yet (the scripts use scratch_r8/un_numba.make_ll_u).

Goal (SAMPLING.md, user 2026-10-08): sampler overhead -> 0, i.e. wall time
~ calls x kernel time. Everything in one numba function: no Python per
call and none per iteration. Python only builds the likelihood (as
un_numba.make_ll_u) and post-processes the dead points.

Algorithm (standard nested sampling; Skilling 2006; slice moves as in
PolyChord / UltraNest's slice samplers, Neal 2003):

- n live points in the unit cube; per iteration the k lowest are removed
  (k = 1: classic; k > 1: parallel nested sampling, Henderson & Goggans
  2014: shrinkage of the j-th removed point 1/(n - j)), and k new points
  are drawn above Lmin = the highest removed logL, each by a slice chain
  of `nsteps` accepted moves started at a random surviving live point.
  The k chains run in parallel threads (numba prange) when k > 1.
- Directions, dirs="pairs" (Round 7; the only one since Round 23): v = +-(U_a - U_b) with a a random
  live point and b one of its `kpair` nearest neighbours (neighbour lists recomputed every `recluster` x n
  replacements by pair_lists(); new points inherit their start point's list). Local orientation and scale, but
  independent of the chain's position (exact).
- Removed: in Round 14 dirs="knn" (biased), "knnmh", "localcov" and the differential-evolution mode jumps `pjump`
  (archived in scratch_r8/archive/nss_legacy.py of the development repository, git tag nss-all-dirs); in Round 23
  (not adopted) dirs="cluster" (K-nearest-neighbour clusters, run()'s old default) and "lcov" (local covariance),
  and the extra step types / adaptation options paxis, pglob, ppca, exp_target, scale_gain, freeze_after, minimg (git).
- Slice (Neal 2003): an interval of width w = `scale` (in units of v)
  placed at random around the point, linear stepping-out by w per side
  (exact; doubling would need Neal's extra acceptance test), then
  shrinking. `scale` is adapted between iterations: x1.1 if there were
  more than ~1 expansion per step, /1.1 otherwise. Wrapped (periodic) dimensions are taken modulo 1.
- Evidence: logX after removing the j-th point of a batch decreases by
  1/(n - j); dead-point weights X_{i-1} - X_i; at the end the live points
  get X_final / n each. Stop when the live points' remaining evidence is
  < frac_remain of logZ (UltraNest default 0.01).

    from pandoramoon import nss
    import un_numba, problem                                  # project scripts (scratch_r8)
    P = problem.Problem("moon-phase"); ll = un_numba.make_ll_u(P)
    r = nss.fit(ll, P.ndim, wrapped=P.wrapped, nlive=1000, threads=16)
    r["passed"], r["logz"], r["logzerr"], r["samples_u"]      # merged equal-weight posterior (unit cube)
"""
import math
import time

import numpy as np
from numba import njit, prange

_SAMPLERS = {}


MAXEXP = 100     # max linear step-outs per side and slice
INIT_VERSION = 2   # 2: rejection init above the LOGZERO plateau by default (Round 20); stored runs carry init_*
PLATEAU = -1e299  # logL at or below this is the LOGZERO plateau (outside the prior; sampling.LOGZERO = -1e300)


def make_sampler(ll_u, parallel, with_data=False):
    """compiled sampler for ll_u. with_data=False: ll_u(u); with_data=True: ll_u(u, D) with a data tuple D passed
    through run(data=D), so ONE compiled sampler serves many data sets (no recompilation per light curve)."""
    key = (id(ll_u), parallel, with_data)
    if key in _SAMPLERS:
        return _SAMPLERS[key]
    if with_data:
        LL = ll_u
    else:
        @njit(inline="always")
        def LL(x, D):
            return ll_u(x)

    @njit(inline="always")
    def wrap_inside(x, wrapped):
        for i in range(x.shape[0]):
            if wrapped[i]:
                x[i] -= math.floor(x[i])
            elif not (x[i] > 0.0 and x[i] < 1.0):
                return False
        return True

    @njit
    def k_smallest(vals, k, mask):
        """indices of the k smallest vals (ascending) among mask[i] == True
        (insertion into a sorted buffer: O(n k), no full sort)"""
        bd = np.full(k, np.inf)
        bi = np.full(k, -1, dtype=np.int64)
        m = 0
        for q in range(vals.shape[0]):
            if not mask[q]:
                continue
            d = vals[q]
            if m < k:
                p = m
                m += 1
            elif d < bd[k - 1]:
                p = k - 1
            else:
                continue
            while p > 0 and bd[p - 1] > d:
                bd[p] = bd[p - 1]
                bi[p] = bi[p - 1]
                p -= 1
            bd[p] = d
            bi[p] = q
        return bi[:m]

    @njit
    def pair_lists(U, alive, sd, kk, M, nbr, ref):
        """dirs="pairs" (Round 13): nbr[q, :kk] = the kk nearest alive points to U[q] (metric scaled by sd) among
        the reference set (all alive points if n <= M, else M distinct random ones), ascending. The same lists as
        the first kk columns of the removed cluster()'s K-nearest graph (checked: test_knnfast.py), faster: scaled
        reference coordinates stored transposed, so the
        squared distances of one point to all references are summed dimension by dimension over contiguous
        arrays (SIMD), then one selection pass keeps the kk smallest; no allocation per point."""
        n, ndim = U.shape
        nref = refset(alive, -1, M, ref)
        inv = np.empty(ndim)
        for i in range(ndim):
            inv[i] = 1.0 / sd[i]
        VT = np.empty((ndim, nref))
        for a in range(nref):
            r = ref[a]
            for i in range(ndim):
                VT[i, a] = U[r, i] * inv[i]
        d = np.empty(nref)
        bd = np.empty(kk)
        bi = np.empty(kk, dtype=np.int64)
        for q in range(n):
            if not alive[q]:
                continue
            for a in range(nref):
                d[a] = 0.0
            for i in range(ndim):
                xi = U[q, i] * inv[i]
                row = VT[i]
                for a in range(nref):
                    t = row[a] - xi
                    d[a] += t * t
            m = 0
            worst = np.inf
            for a in range(nref):
                s = d[a]
                if s >= worst or ref[a] == q:
                    continue
                if m < kk:
                    p = m
                    m += 1
                else:
                    p = kk - 1
                while p > 0 and bd[p - 1] > s:
                    bd[p] = bd[p - 1]
                    bi[p] = bi[p - 1]
                    p -= 1
                bd[p] = s
                bi[p] = ref[a]
                if m == kk:
                    worst = bd[kk - 1]
            for t in range(kk):
                nbr[q, t] = bi[t] if t < m else q

    @njit
    def refset(alive, j, M, R):
        """R[:nR] = all alive points (n <= M) or M distinct random alive ones"""
        n = alive.shape[0]
        nR = 0
        if n <= M:
            for q in range(n):
                if alive[q]:
                    R[nR] = q
                    nR += 1
            return nR
        while True:
            stride = 1 + np.random.randint(n - 1)
            a_, b_ = stride, n
            while b_:
                a_, b_ = b_, a_ % b_
            if a_ == 1:
                break
        q = np.random.randint(n)
        for _t in range(n):
            if alive[q]:
                R[nR] = q
                nR += 1
                if nR == M:
                    break
            q = (q + stride) % n
        return nR

    @njit
    def chain(U, j, L0, Lmin, nsteps, scale, wrapped, x, xn, alive, nbr, kpair, D):
        """slice chain at fixed Lmin from U[j] (logL L0); directions: local pairs.
        Returns (L, ncalls, nexpand, nstuck, nmaxexp) with the end point in x; nstuck = steps whose shrink collapsed
        without an accepted point (the chain did not move), nmaxexp = step-outs that hit MAXEXP (B7)"""
        ndim = U.shape[1]
        n = U.shape[0]
        for i in range(ndim):
            x[i] = U[j, i]
        L = L0
        nc = 0
        nexp = 0
        nstuck = 0
        nmaxexp = 0
        v = np.empty(ndim)
        for _ in range(nsteps):
            # two draws left from the removed "cluster" directions (Round 23): they keep the random streams, and so
            # all results, identical to the earlier versions
            np.random.randint(2)
            np.random.randint(1)
            vv = 0.0
            # local pair difference: a random alive point and one of its kpair nearest neighbours
            # (neighbour lists from the last refresh; independent of the chain's position)
            while True:
                pa = np.random.randint(n)
                if alive[pa]:
                    break
            pb = nbr[pa, np.random.randint(kpair)]
            sg = 1.0 if np.random.uniform(0.0, 1.0) < 0.5 else -1.0
            for i in range(ndim):
                v[i] = sg * (U[pa, i] - U[pb, i])
                vv += v[i] * v[i]
            if vv == 0.0:          # duplicate points (should not happen): no direction
                continue
            left = -scale * np.random.uniform(0.0, 1.0)   # random placement of the initial interval
            right = left + scale
            # step out (linear)
            for _e in range(MAXEXP):
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * left
                if not wrap_inside(xn, wrapped):
                    break
                nc += 1
                if LL(xn, D) > Lmin:
                    left -= scale
                    nexp += 1
                    if _e == MAXEXP - 1:
                        nmaxexp += 1
                else:
                    break
            for _e in range(MAXEXP):
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * right
                if not wrap_inside(xn, wrapped):
                    break
                nc += 1
                if LL(xn, D) > Lmin:
                    right += scale
                    nexp += 1
                    if _e == MAXEXP - 1:
                        nmaxexp += 1
                else:
                    break
            # shrink
            while True:
                t = np.random.uniform(left, right)
                for i in range(ndim):
                    xn[i] = x[i] + v[i] * t
                if wrap_inside(xn, wrapped):
                    nc += 1
                    Lx = LL(xn, D)
                    if Lx > Lmin:
                        for i in range(ndim):
                            x[i] = xn[i]
                        L = Lx
                        break
                if t < 0.0:
                    left = t
                else:
                    right = t
                if right - left < 1e-14:
                    nstuck += 1
                    break
        return L, nc, nexp, nstuck, nmaxexp

    @njit
    def grow(dU, dL, dlogw, nd):
        m = 2 * dU.shape[0]
        a = np.empty((m, dU.shape[1]))
        b = np.empty(m)
        c = np.empty(m)
        for q in range(nd):
            for i in range(dU.shape[1]):
                a[q, i] = dU[q, i]
            b[q] = dL[q]
            c[q] = dlogw[q]
        return a, b, c

    @njit(parallel=parallel, nogil=True)
    def ns(U, L, nsteps, K, k, wrapped, frac_remain, max_iter, seed, scale0, M, recluster, kpair,
           stop, D, logx0):
        # Round 19 options (SAMPLING_REVIEW.md; defaults = the behaviour before): logx0 = ln X at the start (rejection
        # init, B1)
        n, ndim = U.shape
        # seeds (per thread in parallel mode: numba seeds each thread's state)
        np.random.seed(seed)
        cap = 1 << 16
        dU = np.empty((cap, ndim))
        dL = np.empty(cap)
        dlogw = np.empty(cap)
        nd = 0
        logX = logx0
        logZ = -np.inf
        ncall = 0
        nstuck_tot = 0
        nmaxexp_tot = 0
        scale = scale0
        it = 0
        nexp_tot = 0
        nstep_tot = 0
        sd = np.empty(ndim)
        newU = np.empty((k, ndim))
        newL = np.empty(k)
        ncs = np.zeros(k, dtype=np.int64)
        nexps = np.zeros(k, dtype=np.int64)
        nstucks = np.zeros(k, dtype=np.int64)
        nmaxexps = np.zeros(k, dtype=np.int64)
        starts = np.empty(k, dtype=np.int64)
        xs = np.empty((k, ndim))
        xns = np.empty((k, ndim))
        alive = np.ones(n, dtype=np.bool_)
        sd_every = max(1, n // (10 * k))
        nbr = np.empty((n, K), dtype=np.int64)
        for q0 in range(n):
            for t0 in range(K):
                nbr[q0, t0] = q0
        ref = np.empty(n, dtype=np.int64)
        nrecl = 0
        repl_since = n * 10           # force the neighbour lists at the first iteration
        ranks = np.zeros(k, dtype=np.int64)
        mix = np.zeros(ndim)
        nmix = 0
        ins = np.empty(1 << 16, dtype=np.int64)
        nins = 0
        while it < max_iter:
            if stop[0] != 0:                       # abort requested by fit() (the run is discarded)
                break
            sel = k_smallest(L, k, alive)          # ascending
            Lmin = L[sel[k - 1]]
            # record the k removed points
            for jj in range(k):
                q = sel[jj]
                logX_new = logX - 1.0 / (n - jj)
                logw = logX + math.log(-math.expm1(-1.0 / (n - jj)))    # X_old - X_new
                if nd >= dU.shape[0]:
                    dU, dL, dlogw = grow(dU, dL, dlogw, nd)
                dU[nd] = U[q]
                dL[nd] = L[q]
                dlogw[nd] = logw
                nd += 1
                logZ = np.logaddexp(logZ, logw + L[q])
                logX = logX_new
            # stopping: remaining live evidence vs accumulated
            Lmax = L.max()
            if np.isfinite(logZ) and logX + Lmax - logZ < math.log(frac_remain):
                break
            # spread of the live points (for the neighbour metric), every n/10 replacements
            if it % sd_every == 0:
                for i in range(ndim):
                    mu = 0.0
                    for a in range(n):
                        mu += U[a, i]
                    mu /= n
                    s = 0.0
                    for a in range(n):
                        s += (U[a, i] - mu) ** 2
                    sd[i] = math.sqrt(s / n) + 1e-300
            for jj in range(k):
                alive[sel[jj]] = False
            if repl_since >= recluster * n:
                pair_lists(U, alive, sd, kpair, M, nbr, ref)
                nrecl += 1
                repl_since = 0
            repl_since += k
            if k > 1:
                # reproducible parallel runs (Round 17): numba keeps one random state per thread, and which thread
                # runs which chain is not fixed. Each chain therefore reseeds its thread's state from (seed, iteration,
                # chain), and the main thread is reseeded after the parallel loop (below).
                np.random.seed((seed * 1000003 + 2 * it + 1) % 4294967296)
            for c in range(k):
                while True:
                    j = np.random.randint(n)
                    if alive[j]:
                        break
                starts[c] = j
            for c in prange(k):
                if k > 1:
                    np.random.seed((seed * 2654435761 + it * k + c) % 4294967296)
                Lc, nc, ne, nst, nmx = chain(U, starts[c], L[starts[c]], Lmin, nsteps, scale, wrapped, xs[c], xns[c],
                                             alive, nbr, kpair, D)
                newL[c] = Lc
                ncs[c] = nc
                nexps[c] = ne
                nstucks[c] = nst
                nmaxexps[c] = nmx
                for i in range(ndim):
                    newU[c, i] = xs[c, i]
                # insertion index (Fowlie, Handley & Su 2020): rank of the new logL among the n - k
                # surviving live points; uniform on {0..n-k} if the new point is a fair draw above Lmin
                r = 0
                for q2 in range(n):
                    if alive[q2] and L[q2] < Lc:
                        r += 1
                ranks[c] = r
            if nins + k > ins.shape[0]:
                ins2 = np.empty(2 * ins.shape[0] + k, dtype=np.int64)
                for q2 in range(nins):
                    ins2[q2] = ins[q2]
                ins = ins2
            for c in range(k):
                ins[nins] = ranks[c]
                nins += 1
            for c in range(k):                     # mixing diagnostic: (end - start)^2 / (2 var_live) per dim
                for i in range(ndim):
                    dlt = newU[c, i] - U[starts[c], i]
                    if wrapped[i]:
                        dlt -= math.floor(dlt + 0.5)
                    mix[i] += dlt * dlt / (2.0 * sd[i] * sd[i])
                nmix += 1
            for c in range(k):
                q = sel[c]
                for i in range(ndim):
                    U[q, i] = newU[c, i]
                L[q] = newL[c]
                for t0 in range(K):                # new point inherits its start's neighbour list
                    nbr[q, t0] = nbr[starts[c], t0]
                alive[q] = True
                ncall += ncs[c]
                nexp_tot += nexps[c]
                nstuck_tot += nstucks[c]
                nmaxexp_tot += nmaxexps[c]
            nstep_tot += k * nsteps
            # scale adaptation: aim at ~1 expansion per step (interval comparable to the slice)
            if nexps.sum() > k * nsteps:
                scale *= 1.1
            else:
                scale /= 1.1
            scale = min(max(scale, 1e-3), 1e3)
            it += 1
            if stop.shape[0] > 2:                  # progress for fit()'s status line (read by the main thread)
                stop[1] = ncall
                stop[2] = it
        # live points
        logw_live = logX - math.log(n)
        for a in range(n):
            if nd >= dU.shape[0]:
                dU, dL, dlogw = grow(dU, dL, dlogw, nd)
            dU[nd] = U[a]
            dL[nd] = L[a]
            dlogw[nd] = logw_live
            nd += 1
        return (dU[:nd], dL[:nd], dlogw[:nd], ncall, it, scale, nexp_tot / max(nstep_tot, 1),
                nrecl, ins[:nins], mix / max(nmix, 1), nstuck_tot, nmaxexp_tot)

    @njit(parallel=parallel, nogil=True)
    def init(n, ndim, seed, D):
        np.random.seed(seed)
        U = np.random.uniform(0.0, 1.0, (n, ndim))
        L = np.empty(n)
        for a in prange(n):
            L[a] = LL(U[a], D)
        return U, L

    _SAMPLERS[key] = (ns, init)
    return ns, init


def _ks_p(D, N):
    """asymptotic Kolmogorov p-value (Stephens 1970 correction)"""
    if N <= 0:
        return 1.0
    lam = (math.sqrt(N) + 0.12 + 0.11 / math.sqrt(N)) * D
    p = 0.0
    for j in range(1, 101):
        p += 2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
    return float(min(max(p, 0.0), 1.0))


def insertion_test(ranks, m, nlive, seed=0):
    """Insertion-index cross-check (Fowlie, Handley & Su 2020, MNRAS 497, 5256): the rank of
    each new point among the m = n - k surviving live points is uniform on {0..m}. Returns the KS
    distance and p-value of all ranks (jittered to a continuous uniform), and per chunk of
    nlive insertions: the minimum p and the fraction of chunks with p < 0.01 (expected 0.01)."""
    if len(ranks) == 0:
        return dict(n=0)
    rng = np.random.default_rng(seed + 7)
    u = (ranks + rng.uniform(size=len(ranks))) / (m + 1)

    def ks(x):
        x = np.sort(x)
        n = len(x)
        i = np.arange(1, n + 1)
        return float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    D = ks(u)
    ps = []
    for a in range(0, len(u) - nlive + 1, nlive):
        ps.append(_ks_p(ks(u[a:a + nlive]), nlive))
    ps = np.array(ps) if ps else np.array([1.0])
    # signed U test (Buchner, "Nested sampling methods", arXiv:2101.09675; NEW_LEADS L6): z = (2 sum u - n) / sqrt(n / 3)
    # ~ N(0, 1) for exact NS; z < 0: new points rank too low (too close to the threshold), z > 0: too high
    z = float((2.0 * u.sum() - len(u)) / math.sqrt(len(u) / 3.0))
    return dict(n=int(len(u)), D=D, p=_ks_p(D, len(u)), chunk=int(nlive), nchunks=int(len(ps)),
                chunk_p_min=float(ps.min()), chunk_frac_p01=float(np.mean(ps < 0.01)),
                mean=float(u.mean()), z=z)


def run(ll_u, ndim, wrapped=None, nlive=1000, nsteps=None, K=None, k=1, frac_remain=0.01,
        max_iter=10_000_000, seed=1, scale0=1.0, threads=None, nbsub=2000, dirs="pairs",
        recluster=0.25, kpair=4, stop=None, data=None, init_reject=True,
        init_min_draws=None, logx0=None):
    """returns dict(logz, logzerr, ncall, niter, wall, samples_u (equal weight), dead_u, dead_L, logw).
    Round 19 options (SAMPLING_REVIEW.md):
    init_reject (default True since Round 20; False = the initial live set of Rounds 1-19): draw the initial live points only where logL > PLATEAU (the LOGZERO plateau, B1) and start at
      ln X0 = ln(accepted / drawn) over >= init_min_draws draws (or the given logx0, e.g. an analytic 1 - f);
      the draws are counted in ncall. Without it, plateau points enter the live set (n_plateau in the result).
    data: None -> ll_u(u); a tuple D -> ll_u(u, D) (one compiled sampler for many data sets).
    dirs: "pairs" (the only direction scheme; the others were removed in Rounds 14 and 23, see the module docstring)."""
    import numba
    parallel = k > 1
    assert 1 <= k <= nlive // 4, "k (points replaced per iteration) must be <= nlive / 4"
    if threads is not None:
        numba.set_num_threads(threads)
    ns, init = make_sampler(ll_u, parallel, data is not None)
    D = data if data is not None else (0,)
    wrapped = np.array(wrapped if wrapped is not None else [False] * ndim, dtype=np.bool_)
    nsteps = nsteps or _FIT["nsteps_per_dim"] * ndim
    K = K or 2 * ndim
    t0 = time.perf_counter()
    ninit = nlive
    init_min_draws = init_min_draws or _FIT["init_min_draws"]
    if init_reject:
        Us, Ls, nacc, ndraw, j = [], [], 0, 0, 0
        while nacc < nlive or ndraw < init_min_draws:
            m = max(nlive, min(init_min_draws, 100_000))
            U_, L_ = init(m, ndim, (seed * 7919 + 1_000_003 * (j + 1)) % 4294967296, D)
            ok = L_ > PLATEAU
            nacc += int(ok.sum())
            ndraw += m
            if sum(len(q) for q in Ls) < nlive:
                Us.append(U_[ok])
                Ls.append(L_[ok])
            j += 1
            if j > 10_000:
                raise RuntimeError("init_reject: almost no prior volume above the LOGZERO plateau")
        U = np.ascontiguousarray(np.concatenate(Us)[:nlive])
        L = np.ascontiguousarray(np.concatenate(Ls)[:nlive])
        lx0 = float(logx0) if logx0 is not None else math.log(nacc / ndraw)
        ninit = ndraw
    else:
        U, L = init(nlive, ndim, seed, D)
        nacc, ndraw = 0, 0
        lx0 = float(logx0) if logx0 is not None else 0.0
    n_plateau = int(np.sum(L <= PLATEAU))
    t1 = time.perf_counter()
    if dirs != "pairs":
        raise ValueError(f"dirs={dirs!r}: only \"pairs\" (cluster / lcov removed in Round 23, knn* in Round 14; git)")
    (dU, dL, dlogw, ncall, it, scale, exp_rate, nrecl,
     ins, mix, nstuck, nmaxexp) = ns(U, L, nsteps, K, k, wrapped, frac_remain, max_iter, seed + 1, scale0,
                                              nbsub, recluster, kpair,
                                              stop if stop is not None else np.zeros(3, dtype=np.int64), D,
                                              lx0)
    t2 = time.perf_counter()
    lw = dlogw + dL
    logz = float(np.logaddexp.reduce(lw))
    w = np.exp(lw - logz)
    # information and error (Skilling): sqrt(H / n)
    fin = w > 0
    H = float(np.sum(w[fin] * (dL[fin] - logz)))
    # equal-weight posterior (systematic resampling)
    rng = np.random.default_rng(seed)
    cw = np.cumsum(w)
    cw /= cw[-1]
    nsamp = int(np.exp(-np.sum(w[fin] * np.log(w[fin]))))   # ~ESS-ish (Kish would be 1/sum w^2)
    ess = float(1.0 / np.sum(w ** 2))
    pos = (rng.uniform() + np.arange(nsamp)) / nsamp
    samples = dU[np.searchsorted(cw, pos)]
    insd = insertion_test(ins, nlive - k, nlive, seed)
    return dict(seed=int(seed), nlive=int(nlive), ins_u=((ins + np.random.default_rng(seed + 11).uniform(size=len(ins))) / (nlive - k + 1)
                                       ).astype(np.float32), mix=mix, insertion=insd, logz=logz, logzerr=math.sqrt(max(H, 0.0) / nlive), H=H, ncall=int(ncall) + ninit, niter=int(it),
                wall=t2 - t0, wall_init=t1 - t0, samples_u=samples, ess=ess, dead_u=dU, dead_L=dL, logw=dlogw,
                maxL=float(dL.max()), final_scale=scale, expand_rate=exp_rate,
                nrecluster=int(nrecl), dirs=dirs, logx0=lx0, init_acc=int(nacc), init_draws=int(ndraw),
                n_plateau=n_plateau, nstuck=int(nstuck), nmaxexp=int(nmaxexp))


# ---------------------------------------------------------------------------------------------
# fit(): M independent runs in parallel (one per thread, each serial), merged, with self-checks
# (SAMPLING.md Round 6: merged runs ~ one run with more live points; reproducibility is checked
# between runs, not inside one).

def _ks(a, b):
    a = np.sort(a)
    b = np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a)
                               - np.searchsorted(b, x, side="right") / len(b))))


# ---------------------------------------------------------------------------------------------
# evidence by importance sampling (2026-10-14): for unimodal, near-Gaussian posteriors (the planet-only model). One
# nested-sampling run gives the posterior shape; a Student-t proposal fitted to it gives lnZ = ln mean(L / q) over
# draws in the unit cube (draws outside the cube count with weight 0: unbiased without normalising the truncation).

_BATCHES = {}


def make_batch(ll_u, with_data=False):
    """compiled parallel evaluation of ll_u at the rows of U (-inf outside the open unit cube)"""
    key = (id(ll_u), with_data)
    if key in _BATCHES:
        return _BATCHES[key]
    if with_data:
        LL = ll_u
    else:
        @njit(inline="always")
        def LL(x, D):
            return ll_u(x)

    @njit(parallel=True, nogil=True)
    def batch(U, D):
        n = U.shape[0]
        out = np.empty(n)
        for i in prange(n):
            ok = True
            for j in range(U.shape[1]):
                if not (U[i, j] > 0.0 and U[i, j] < 1.0):
                    ok = False
            out[i] = LL(U[i], D) if ok else -np.inf
        return out
    _BATCHES[key] = batch
    return batch


def _t_draw(rng, mu, C, df, n):
    """n draws of a multivariate Student-t (location mu, shape C) and their log densities"""
    d = len(mu)
    Lc = np.linalg.cholesky(C)
    z = rng.standard_normal((n, d))
    g = rng.chisquare(df, n) / df
    x = mu + (z @ Lc.T) / np.sqrt(g)[:, None]
    y = np.linalg.solve(Lc, (x - mu).T).T
    m = np.sum(y * y, axis=1)
    logdet = 2.0 * np.sum(np.log(np.diag(Lc)))
    lq = (math.lgamma((df + d) / 2) - math.lgamma(df / 2) - 0.5 * d * math.log(df * math.pi) - 0.5 * logdet
          - 0.5 * (df + d) * np.log1p(m / df))
    return x, lq


def _weighted_moments(U, w):
    w = w / w.sum()
    mu = w @ U
    X = U - mu
    C = (X * w[:, None]).T @ X
    C += np.eye(len(mu)) * 1e-12 * max(np.trace(C), 1e-300)
    return mu, C


@njit(parallel=True, cache=True)
def _kde_logq(Yx, Yc, base, df):
    """log of the mean over centres of Student-t kernels, in whitened coordinates (rows of Yx: points, of Yc:
    centres; base: the kernel's log normalisation)"""
    n, d = Yx.shape
    m = Yc.shape[0]
    out = np.empty(n)
    for i in prange(n):
        t = np.empty(m)
        mx = -np.inf
        for c in range(m):
            s = 0.0
            for j in range(d):
                r = Yx[i, j] - Yc[c, j]
                s += r * r
            t[c] = -0.5 * (df + d) * math.log1p(s / df)
            mx = max(mx, t[c])
        acc = 0.0
        for c in range(m):
            acc += math.exp(t[c] - mx)
        out[i] = base + mx + math.log(acc / m)
    return out


def _mix_draw(rng, Uc, mu, C, S, n):
    """draws of the proposal (KDE of t kernels around the centres Uc + defensive global t) and their log density"""
    d = len(mu)
    df, h, sc, fd = float(S["df"]), float(S["kde_bandwidth"]), float(S["scale"]), float(S["defensive"])
    nk = int(rng.binomial(n, 1.0 - fd))
    Lk = np.linalg.cholesky(h * h * C)
    z = rng.standard_normal((nk, d))
    g = rng.chisquare(df, nk) / df
    xk = Uc[rng.integers(0, len(Uc), nk)] + (z @ Lk.T) / np.sqrt(g)[:, None]
    xg, _ = _t_draw(rng, mu, C * sc * sc, df, n - nk)
    x = np.vstack([xk, xg])
    _, lg = _t_logpdf(x, mu, C * sc * sc, df)
    Li = np.linalg.inv(Lk)
    base = (math.lgamma((df + d) / 2) - math.lgamma(df / 2) - 0.5 * d * math.log(df * math.pi)
            - float(np.sum(np.log(np.diag(Lk)))))
    lk = _kde_logq(np.ascontiguousarray(x @ Li.T), np.ascontiguousarray(Uc @ Li.T), base, df)
    return x, np.logaddexp(math.log(1.0 - fd) + lk, math.log(fd) + lg)


def _t_logpdf(x, mu, C, df):
    d = len(mu)
    Lc = np.linalg.cholesky(C)
    y = np.linalg.solve(Lc, (x - mu).T).T
    m = np.sum(y * y, axis=1)
    return x, (math.lgamma((df + d) / 2) - math.lgamma(df / 2) - 0.5 * d * math.log(df * math.pi)
               - float(np.sum(np.log(np.diag(Lc)))) - 0.5 * (df + d) * np.log1p(m / df))


def fit_is(ll_u, ndim, wrapped=None, data=None, threads=12, seed=1, lz_tol=0.2, verbose=True):
    """lnZ and posterior by importance sampling (settings.cfg [planet_is]): a pilot nss.run gives the posterior; the
    proposal is a mixture of Student-t kernels around points drawn from it plus a defensive wide Student-t; lnZ =
    ln mean(L / q) over draws in the unit cube (draws outside it count with weight 0: unbiased without normalising the
    truncation), more draws until the error <= error_frac x lz_tol. Returns dict(ok, logz, logzerr, U (draws with
    w > 0), logL, w (normalised), ess, max_share, ncall, wall, pilot, reason). ok=False: a check failed (the caller
    falls back to fit()). Not for wrapped (periodic) dimensions or multimodal posteriors."""
    import numba
    S = _settings()["planet_is"]
    t0 = time.perf_counter()
    if wrapped is not None and any(wrapped):
        return dict(ok=False, reason="wrapped dimensions", ncall=0, wall=0.0)
    numba.set_num_threads(max(1, int(threads)))
    nl = int(S["pilot_nlive"])
    npil = max(1, int(S["pilot_runs"]))
    # pilot: npil runs in parallel threads (compiled once outside the threads, as in fit), merged as one run
    run(ll_u, ndim, wrapped=wrapped, nlive=max(40, 2 * ndim + 8), nsteps=2, k=1, max_iter=3, seed=seed + 999,
        data=data, init_reject=False)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(npil, max(1, int(threads)))) as ex:
        runs = list(ex.map(lambda i: run(ll_u, ndim, wrapped=wrapped, nlive=nl, k=1, seed=seed + 1000 * i,
                                         dirs="pairs", data=data), range(npil)))
    lzp, lw = merge_weights(runs)
    Up = np.concatenate([q["dead_u"] for q in runs])
    Hm = float(np.mean([q["H"] for q in runs]))
    p = dict(logz=float(lzp), logzerr=math.sqrt(max(Hm, 0.0) / (nl * npil)), ncall=int(sum(q["ncall"] for q in runs)),
             niter=int(sum(q["niter"] for q in runs)))
    batch = make_batch(ll_u, data is not None)
    D = data if data is not None else (0,)
    w = np.exp(lw - lw.max())
    mu, C = _weighted_moments(Up, w)
    rng = np.random.default_rng(seed + 54321)
    Uc = Up[rng.choice(len(w), size=int(S["kde_centres"]), p=w / w.sum())]
    ncall = int(p["ncall"])
    n = int(S["n_draws"])
    xs, Ls, lws = [], [], []
    while True:
        x, lq = _mix_draw(rng, Uc, mu, C, S, n if not xs else sum(len(q) for q in xs))
        L = batch(np.ascontiguousarray(x), D)
        ncall += len(x)
        xs.append(x)
        Ls.append(L)
        lws.append(L - lq)
        lwa = np.concatenate(lws)
        m = lwa.max()
        wa = np.exp(lwa - m) if np.isfinite(m) else np.zeros(len(lwa))
        N = len(wa)
        lz = float(m + math.log(wa.mean())) if wa.sum() > 0 else -np.inf
        err = float(wa.std() / wa.mean() / math.sqrt(N)) if wa.sum() > 0 else np.inf
        if err <= float(S["error_frac"]) * lz_tol or N >= int(S["n_max"]):
            break
    ess = float(wa.sum() ** 2 / (wa ** 2).sum()) if wa.sum() > 0 else 0.0
    share = float(wa.max() / wa.sum()) if wa.sum() > 0 else 1.0
    keep = wa > 0
    U = np.concatenate(xs)[keep]
    LL = np.concatenate(Ls)[keep]
    wn = wa[keep] / wa[keep].sum() if keep.any() else wa[keep]
    dz = abs(lz - p["logz"])
    lim = float(S["consistency_k"]) * math.hypot(err, p["logzerr"]) + 0.1
    fails = []
    if ess < float(S["ess_min"]):
        fails.append(f"IS effective sample size {ess:.0f} < {S['ess_min']}")
    if share > float(S["max_share"]):
        fails.append(f"largest weight {share:.3f} of the total > {S['max_share']}")
    if not dz <= lim:
        fails.append(f"IS lnZ {lz:.3f} and pilot lnZ {p['logz']:.3f} differ by {dz:.3f} > {lim:.3f}")
    if not err <= lz_tol:
        fails.append(f"IS error {err:.3f} > lz_tol {lz_tol}")
    out = dict(ok=not fails, reason="; ".join(fails), logz=lz, logzerr=err, U=U, logL=LL, w=wn, ess=ess,
               max_share=share, ncall=ncall, n_draws=N, wall=time.perf_counter() - t0,
               pilot=dict(logz=float(p["logz"]), logzerr=float(p["logzerr"]), nlive=nl, runs=npil,
                          ncall=int(p["ncall"]), niter=int(p["niter"])))
    if verbose:
        print(f"  importance sampling: lnZ {lz:.3f} +- {err:.3f} (pilot run {p['logz']:.3f} +- {p['logzerr']:.3f}), "
              f"ESS {ess:.0f} of {N}, {ncall:,} calls, {out['wall']:.1f} s" + ("" if out["ok"] else
              f"; checks failed ({out['reason']}): full nested-sampling fit"), flush=True)
    return out


def merge_weights(rs):
    """Merge independent runs into ONE nested-sampling run (Skilling 2006 Sect. 9; Higson et al. 2018,
    dynesty merge_runs): all points sorted by logL; the number of live points just before point j is
    N_live = sum n_i minus the final live points (of any run) already passed; the shrinkage per point
    is exp(-1/N_live) as in run(). The live count only drops at final live points (a dead point of a
    run is replaced by a new one of the same run). Returns (logZ, ln w + ln L per point, order)."""
    Ls, fl = [], []
    for r in rs:
        n = r.get("nlive")
        L = r["dead_L"]
        f = np.zeros(len(L), dtype=bool)
        f[len(L) - n:] = True                   # run() appends its n final live points last
        Ls.append(L)
        fl.append(f)
    L = np.concatenate(Ls)
    f = np.concatenate(fl)
    o = np.argsort(L, kind="stable")
    Lo = L[o]
    fo = f[o]
    ntot = int(f.sum())
    nl = ntot - (np.cumsum(fo) - fo)            # live points just before removing point j
    dlx = 1.0 / nl
    # start volume (Round 19, init_reject): pooled acceptance fraction of the runs' rejection draws, else the mean logx0
    if all(r.get("init_draws", 0) > 0 for r in rs):
        lx0 = math.log(sum(r["init_acc"] for r in rs) / sum(r["init_draws"] for r in rs))
    else:
        lx0 = float(np.mean([r.get("logx0", 0.0) for r in rs]))
    lx = lx0 - np.cumsum(dlx)
    lxprev = lx + dlx
    lw = lxprev + np.log(-np.expm1(-dlx))       # X_{j-1} - X_j
    lwl = np.empty(len(L))
    lwl[o] = lw + Lo
    return float(np.logaddexp.reduce(lwl)), lwl


def merge(rs, nsamp=None, seed=0):
    """Merge of independent runs as ONE run with sum n_i live points (merge_weights; SAMPLING.md Round 8); lnZ ~
    the mean of the runs' lnZ, centred on the truth in the known-Z tests. Not robust to runs that never found the
    best mode: fit() excludes those first (failed_runs). (The log-mean-Z merge of Rounds 6b-7 was biased high by
    ~var(lnZ_i) / 2 and was removed in the 2026-10-13 cleanup.) Returns (logZ, equal-weight samples_u, ESS)."""
    lzm, lw = merge_weights(rs)
    w = np.exp(lw - np.logaddexp.reduce(lw))
    ess = float(1.0 / np.sum(w ** 2))
    n = nsamp or int(min(ess, 200_000))
    rng = np.random.default_rng(seed)
    cw = np.cumsum(w)
    cw /= cw[-1]
    pos = (rng.uniform() + np.arange(n)) / n
    idx = np.searchsorted(cw, pos)
    # gather the selected points from the runs without stacking all of them (a full vstack copied
    # ~0.4 GB per merge at ~50 runs; fit() merges 5x per check)
    off = np.cumsum([0] + [len(r["dead_L"]) for r in rs])
    ir = np.searchsorted(off, idx, side="right") - 1
    out = np.empty((n, rs[0]["dead_u"].shape[1]))
    for q in np.unique(ir):
        m = ir == q
        out[m] = rs[q]["dead_u"][idx[m] - off[q]]
    return float(lzm), out, ess


def split_half_ks(rs, sub=40_000, seed=0, post_map=None):
    """max over parameters of the KS distance between the merged posteriors of the even- and
    odd-numbered runs (reproducibility of a merge of M/2 runs; the full merge is better by ~sqrt 2).
    post_map: unit-cube samples -> the (physical) coordinates in which to compare, e.g. folding
    exact symmetry copies whose relative weight is meaningless."""
    if len(rs) < 2:
        return np.nan, -1
    _, A, _ = merge(rs[0::2], nsamp=sub, seed=seed)
    _, B, _ = merge(rs[1::2], nsamp=sub, seed=seed + 1)
    if post_map is not None:
        A, B = post_map(A), post_map(B)
    d = [_ks(A[:, i], B[:, i]) for i in range(A.shape[1])]
    j = int(np.argmax(d))
    return float(d[j]), j


# ---------------------------------------------------------------------------------------------
# The decision rules of fit() (Round 23: consolidated from Rounds 8-21). Four rules, one per question:
#   1 done?            the result arm's lnZ error <= lz_tol, split-half KS <= ks_tol, >= min_runs runs, and at most
#                      failed_max of the runs failed (failed_runs) -- plus rule 2 passed
#   2 chains long?     the arms at N and N / 2 agree (dlnZ, KS with a permutation null); else nsteps x 2
#   3 excess scatter?  per-run lnZ sd > scatter_ratio x quoted: nsteps x 2 if the N / 2 arm scatters more, else n_live x 2
#   4 better solution? a gap in the runs' lnZ with higher maxL above it: n_live x 2
# Once rule 3 / 4 sent a fit to more live points, nsteps is never doubled again. The constants are in
# settings.cfg [fit_rules] (each with its source), read once at import.
from pandoramoon._config import settings as _settings  # noqa: E402

FIT_RULES = dict(_settings()["fit_rules"])          # settings.cfg [fit_rules] (each value with its source there)
_FIT = _settings()["fit"]


def _better_solution_gap(runs, q):
    """rule 4: the largest gap in the sorted lnZ of the runs exceeds max(gap_quoted x q, gap_nats) and the runs above
    it reach a higher maximum likelihood (median maxL > gap_maxl higher)"""
    FR = FIT_RULES
    o = np.argsort([r["logz"] for r in runs])
    zs = np.array([runs[i]["logz"] for i in o])
    ms = np.array([runs[i]["maxL"] for i in o])
    if len(zs) < 4:
        return False
    jg = int(np.argmax(np.diff(zs)))
    gap = float(zs[jg + 1] - zs[jg])
    higher = float(np.median(ms[jg + 1:]) - np.median(ms[:jg + 1])) > FR["gap_maxl"]
    return bool(higher and gap > max(FR["gap_quoted"] * q, FR["gap_nats"]))


def failed_runs(rs, dl_max=None, dz_min=None):
    """indices of runs that never found the best mode: maxL more than dl_max below the best run's AND lnZ
    more than max(5 MAD-sd, dz_min) below the median (C0 at 150 live: 2 of 10 runs at lnZ -204 vs -186)"""
    dl_max = FIT_RULES["failed_dl"] if dl_max is None else dl_max
    dz_min = FIT_RULES["failed_dz"] if dz_min is None else dz_min
    if len(rs) < 3:
        return []
    z = np.array([r["logz"] for r in rs])
    mL = np.array([r["maxL"] for r in rs])
    med = np.median(z)
    mad = 1.4826 * np.median(np.abs(z - med))
    return [i for i in range(len(rs)) if mL[i] < mL.max() - dl_max and z[i] < med - max(FIT_RULES["failed_mad"] * mad, dz_min)]


def _lnz_stats(rs):
    """per-arm statistics over the good runs: mean / sd / MAD-sd of lnZ, err = sd / sqrt(M); also the
    number of failed runs"""
    bad = set(failed_runs(rs))
    z = np.array([r["logz"] for i, r in enumerate(rs) if i not in bad])
    M = len(z)
    sd = float(z.std(ddof=1)) if M > 1 else np.nan
    mad = float(1.4826 * np.median(np.abs(z - np.median(z)))) if M > 1 else np.nan
    return dict(M=M, mean=float(z.mean()), sd=sd, err=sd / math.sqrt(M) if M > 1 else np.inf, mad_sd=mad,
                n_failed=len(bad), M_all=len(rs),
                good=[r for i, r in enumerate(rs) if i not in bad])


def _ks_sets(ra, rb, post_map, sub=40_000, seed=0):
    _, A, _ = merge(ra, nsamp=sub, seed=seed)
    _, B, _ = merge(rb, nsamp=sub, seed=seed + 1)
    if post_map is not None:
        A, B = post_map(A), post_map(B)
    d = [_ks(A[:, i], B[:, i]) for i in range(A.shape[1])]
    j = int(np.argmax(d))
    return float(d[j]), j


def ks_null(ra, rb, post_map, nperm=19, sub=20_000, seed=0):
    """noise distribution of the KS distance between the merged arms ra and rb (Round 16): KS distances between
    random re-splits of the pooled runs into two sets of the arms' sizes (if both arms sample the same posterior,
    their runs are exchangeable). Returns an array of nperm values."""
    rng = np.random.default_rng(seed)
    allr = list(ra) + list(rb)
    na = len(ra)
    out = []
    for k in range(nperm):
        idx = rng.permutation(len(allr))
        A = [allr[i] for i in sorted(idx[:na])]
        B = [allr[i] for i in sorted(idx[na:])]
        out.append(_ks_sets(A, B, post_map, sub=sub, seed=seed + 31 * k)[0])
    return np.array(out)


def pooled_insertion(rs):
    """insertion-index test pooled over runs: KS of all jittered insertion ranks (uniform for exact NS) and
    the Fisher combination of the per-run p-values. Returns (D, p_pooled, p_fisher)."""
    u = np.concatenate([r["ins_u"] for r in rs if "ins_u" in r]).astype(np.float64)
    if len(u) == 0:
        return np.nan, np.nan, np.nan, np.nan
    u.sort()
    n = len(u)
    i = np.arange(1, n + 1)
    D = float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))
    z_u = float((2.0 * u.sum() - n) / math.sqrt(n / 3.0))           # signed U test, pooled (reported only)
    ps = np.array([max(r["insertion"]["p"], 1e-300) for r in rs])
    chi2 = -2.0 * np.sum(np.log(ps))
    # chi2 with 2M dof: Wilson-Hilferty normal approximation of the upper tail
    k = 2 * len(ps)
    z = ((chi2 / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    p_f = float(0.5 * math.erfc(z / math.sqrt(2)))
    return D, _ks_p(D, n), p_f, z_u


# accuracy presets of fit() (SAMPLING.md Round 12)
ACCURACY = {a: dict(nlive=_FIT[f"{a}_nlive"], lz_tol=_FIT[f"{a}_lz_tol"], ks_tol=_FIT[f"{a}_ks_tol"],
                    min_runs=_FIT[f"{a}_min_runs"]) for a in ("high", "medium")}    # settings.cfg [fit]


def fit(ll_u, ndim, wrapped=None, nlive=2000, nsteps=None, threads=12, min_runs=None,
        max_runs=None, ks_tol=0.05, lz_tol=0.1, seed=1, dirs="pairs", frac_remain=0.01, verbose=True,
        post_map=None, kpair=4, check=True, max_nsteps=None, recluster=0.25, data=None,
        accuracy=None, resume=None, callback=None, max_nlive=None, progress=False, progress_label="fit"):
    """Nested sampling with built-in convergence checks.

    Independent serial nss runs (n_live = nlive each) run in parallel threads and are merged as one run
    (merge_weights). Two arms: the result arm at `nsteps` (default 4 ndim) and, if check=True, a check arm at
    nsteps / 2 (a new run goes to the arm with fewer runs). After every `threads` finished runs (or earlier, when a
    cheap scalar precheck says a decision is possible) the four rules of FIT_RULES decide:

      1 done?            the result arm's lnZ error (sd / sqrt(M)) <= lz_tol, its split-half KS <= ks_tol, >= min_runs
                         runs, rule 2 passed, and at most 10 % failed runs  ->  stop, passed=True
      2 chains long?     pass if |dlnZ(N/2, N)| + sigma <= 2 lz_tol and the KS between the arms (noise-corrected by a
                         permutation null) <= ks_tol; nsteps x 2 if |dlnZ| - 2 sigma > 2 lz_tol (3 sigma with few
                         runs), or -- once the result arm meets rule 1 and sigma <= 0.6 lz_tol -- |dlnZ| - sigma > lz_tol;
                         or if the KS clearly exceeds ks_tol
      3 excess scatter?  per-run lnZ sd > 2.75 x the quoted sqrt(H/n): nsteps x 2 if the N / 2 arm scatters >= 1.25 x
                         more (longer chains help), else n_live x 2
      4 better solution? a gap in the runs' lnZ (> max(6 x quoted, 2 nats)) with a higher maxL above it: n_live x 2

    On nsteps x 2 the old result arm becomes the check arm; on n_live x 2 both arms restart. Once rule 3 or 4 sent the
    fit to more live points (result "multimodal": True), nsteps is not doubled again (the arms then differ by which
    got the runs that found the better solution); rule 3 then doubles n_live instead. When a rule asks for an escalation
    that cannot be done (n_live / nsteps at its cap max_nlive (4 nlive) / max_nsteps (64 ndim), or an nsteps request
    blocked by multimodal), the fit stops as soon as the lnZ error is within lz_tol: passed=False, status "stalled"
    (Round 23; more runs would only shrink the error). Otherwise a fit that does not pass stops at max_runs: status
    "run_cap". The result's status ("passed" / "stalled" / "run_cap") and reason (which checks were not met, by how
    much) say why.

    progress: False (default), True (a status line every 2 s in a terminal / Jupyter, every 30 s otherwise: runs done,
    n_live, nsteps, the current lnZ, calls, calls per second, elapsed time) or the interval in seconds;
    progress_label: the line's prefix. Does not change the result.
    accuracy: None (use the arguments) or a preset of ACCURACY (overrides nlive, lz_tol, ks_tol, min_runs):
    "high" (lnZ +-0.1, KS 0.05, 1000 live, >= 16 runs), "medium" (+-0.2, KS 0.08, 500 live, >= 8 runs).
    Returns dict(logz, logzerr (empirical), logzerr_boot, samples_u, runs, history, passed, nsteps, nlive,
    multimodal, n_escalations, n_nlive_escalations, ...). Known blind spots: a chain-length bias that barely changes
    between N / 2 and N (C7: +0.12 at 64 steps), and a mode that every run misses.

    Resume (Round 15, used by pandoramoon.sampling): callback(event, **kw) is called in the main thread with
    event "run" (r=run dict, nsteps=arm, idx=run index) for every finished run that enters an arm, and "state"
    (N=current result nsteps, nesc, dropped=nsteps of a dropped arm, dropped_idx=the run indices of that arm and of
    the runs aborted with it) after every escalation. resume = dict(..., skip=[indices not to reuse]) +
    dict(runs=[(nsteps, idx, run dict), ...], N=..., nesc=...) restarts from stored runs: the runs of the arms N
    and N/2 are reloaded, new runs take the smallest unused indices (run seed = seed + 1000 idx; the runs lost in
    flight at a crash are redone with their original seeds), and
    the checks are evaluated before any new run is started (a fit that had passed passes again at once)."""
    from concurrent.futures import ThreadPoolExecutor
    import sys
    # progress status (Round 23): a status line every few seconds while the runs execute (as UltraNest's). The compiled
    # runs write their calls / iterations into their stop array; the main thread, otherwise idle in wait(), reads them.
    # Interactive (terminal, Jupyter): one line rewritten in place every 2 s; otherwise (logs) a new line every 30 s.
    interactive = bool(getattr(sys.stdout, "isatty", lambda: False)() or "ipykernel" in sys.modules)
    p_every = (2.0 if interactive else 30.0) if progress is True else (float(progress) if progress else 0.0)
    p_state = dict(len=0, hist=[])

    def clear_status():
        if p_state["len"]:
            sys.stdout.write("\r" + " " * p_state["len"] + "\r")
            p_state["len"] = 0

    def say(msg):
        clear_status()
        print(msg, flush=True)

    def status():
        now = time.perf_counter()
        calls = ncall_tot[0] + sum(int(v[1][1]) for v in inflight.values())
        ph = p_state["hist"]
        ph.append((now, calls))
        while len(ph) > 2 and now - ph[0][0] > 15.0:
            ph.pop(0)
        rate = (ph[-1][1] - ph[0][1]) / (ph[-1][0] - ph[0][0]) if len(ph) > 1 and ph[-1][0] > ph[0][0] else 0.0
        el = now - t0
        R = arms.get(N, [])
        lz = f"lnZ {hist[-1]['logz']:.2f} +- {hist[-1]['logz_err']:.2f} (target {lz_tol:g})" if hist else "lnZ -"
        s = (f"{progress_label}: {len(R)} runs done (min {min_runs}, max {max_runs}), {len(inflight)} running | "
             f"n_live {nlive}, nsteps {N} | {lz} | {calls:,} calls, {int(round(rate)):,} calls/s | "
             f"{int(el // 3600)}:{int(el % 3600 // 60):02d}:{int(el % 60):02d}")
        if interactive:
            sys.stdout.write("\r" + s.ljust(p_state["len"]))
            sys.stdout.flush()
            p_state["len"] = len(s)
        else:
            print(s, flush=True)

    if accuracy is not None:
        a = ACCURACY[accuracy]
        nlive, lz_tol, ks_tol = a["nlive"], a["lz_tol"], a["ks_tol"]
        min_runs = min_runs or a["min_runs"]
    max_runs = max_runs or _FIT["max_runs"]
    nsteps = nsteps or _FIT["nsteps_per_dim"] * ndim
    max_nsteps = max_nsteps or _FIT["max_nsteps_per_dim"] * ndim
    min_runs = min_runs or max(threads, 4)
    nlive0 = nlive
    max_nlive = max_nlive or _FIT["max_nlive_factor"] * nlive
    nlesc = 0
    multimodal = [False]
    tc = time.perf_counter()
    make_sampler(ll_u, False, data is not None)  # compile once, outside the threads
    run(ll_u, ndim, wrapped=wrapped, nlive=max(40, 2 * ndim + 8), nsteps=2, k=1, max_iter=3, seed=seed + 999, data=data,
        dirs=dirs, kpair=min(kpair, 4), init_reject=False)
    t_compile = time.perf_counter() - tc         # numba compile (first call per process); not in `wall`
    arms = {}                                    # nsteps -> list of runs
    hist = []
    t0 = time.perf_counter()
    nxt = 0
    N = nsteps
    nesc = 0
    ncall_tot = [0]
    used = set()                                 # run indices consumed (seed = seed + 1000 idx)
    if resume is not None:
        N = int(resume.get("N", nsteps))
        nesc = int(resume.get("nesc", 0))
        nlive = int(resume.get("nlive") or nlive)
        nlesc = int(resume.get("nlesc", 0))
        # indices of runs of dropped arms (and runs aborted with them) are not reused; indices of runs that were in
        # flight at a crash are (the same seeds as in the interrupted fit, Round 17)
        used |= set(int(i) for i in resume.get("skip", []))
        for ns_, idx_, r_ in resume.get("runs", []):
            used.add(int(idx_))
            r_["idx"] = int(idx_)
            if int(r_.get("nlive", nlive)) != nlive:
                continue
            if ns_ == N or (check and ns_ == max(1, N // 2)):
                arms.setdefault(ns_, []).append(r_)
        for v_ in arms.values():
            v_.sort(key=lambda q: q["seed"])

    def precheck(N):
        """cheap necessary condition (scalars only: lnZ statistics of both arms) for evaluate() to pass or to
        escalate by the dlnZ rule; the KS-based escalation is covered by the regular evaluation every `threads`
        finished runs (Round 13)"""
        Nc = max(1, N // 2)
        R = arms.get(N, [])
        if len(R) < 2:
            return False
        sR = _lnz_stats(R)
        if sR["M"] >= min_runs and sR["err"] <= lz_tol:
            return True                          # pass possible
        if not check:
            return False
        C = arms.get(Nc, [])
        if len(C) < 2:
            return False
        sC = _lnz_stats(C)
        d = sC["mean"] - sR["mean"]
        sig = math.hypot(sR["err"], sC["err"])
        gate = max(FIT_RULES["runs_gate"], min_runs // 2)
        enough = sC["M"] >= gate and sR["M"] >= gate
        return bool(enough and N < max_nsteps and not multimodal[0]
                    and abs(d) - FIT_RULES["k_sig"] * sig > FIT_RULES["bias_esc"] * lz_tol)   # imprecise tier

    def evaluate(N, final=False):
        Nc = max(1, N // 2)
        R = arms.setdefault(N, [])
        C = arms.setdefault(Nc, []) if check else []
        sR = _lnz_stats(R)
        Rg = sR["good"]
        lzR, _ = merge_weights(Rg)
        ks, j = split_half_ks(Rg, seed=len(R), post_map=post_map)
        # the pooled insertion test is reported only (no power, Round 9): computed for the final evaluation only
        insp = pooled_insertion(Rg) if final else (np.nan, np.nan, np.nan, np.nan)
        h = dict(nsteps=N, runs=sR["M_all"], runs_good=sR["M"], n_failed=sR["n_failed"], wall=time.perf_counter() - t0,
                 logz=lzR, logz_meanlnz=sR["mean"], logz_err=sR["err"], logz_sd=sR["sd"], logz_mad_sd=sR["mad_sd"],
                 quoted=float(np.mean([r["logzerr"] for r in R]) / math.sqrt(sR["M"])), ks_split=ks, ks_param=j,
                 insertion_p_min=min(r["insertion"]["p"] for r in R), final=final,
                 **dict(zip(("ins_pooled_D", "ins_pooled_p", "ins_fisher_p", "ins_pooled_z"), insp)),
                 ncall=int(ncall_tot[0]), ncall_result=int(sum(r["ncall"] for r in R)))
        FR = FIT_RULES
        gate = max(FR["runs_gate"], min_runs // 2)            # runs per arm before an arm statistic is trusted
        q_run = float(np.mean([r["logzerr"] for r in Rg]))
        h.update(scatter_ratio=float(sR["sd"] / q_run) if q_run > 0 and np.isfinite(sR["sd"]) else np.nan,
                 nlive=nlive)
        ok_bias, esc_nsteps, esc_nlive, why = True, False, False, ""
        sC = None
        if check:
            sC = _lnz_stats(C) if len(C) >= 2 else dict(M=len(C), mean=np.nan, err=np.inf, sd=np.nan, good=C, n_failed=0)
            d = sC["mean"] - sR["mean"]
            sig = math.hypot(sR["err"], sC["err"])
            ksa, ja = _ks_sets(sC["good"], Rg, post_map, seed=len(R) + 7) if len(sC["good"]) >= 2 else (np.nan, -1)
            h.update(check_nsteps=Nc, check_runs=sC["M"], dlnz=d, dlnz_err=sig, ks_arms=ksa, ks_arms_param=ja)
            # rule 2 (chain length): the arms at N and N / 2 must agree. lnZ: the bias at N is bounded by
            # dlnZ(N / 2, N) if it falls at least like 1 / nsteps (known-Z tests); KS: noise-corrected by a permutation
            # null of the runs (Round 16)
            need = bool(np.isfinite(ksa) and ksa > ks_tol)
            if need:
                nl = ks_null(sC["good"], Rg, post_map, seed=len(R) + 101)
                med, hi = float(np.median(nl)), float(nl.max())
            else:
                med = hi = 0.0
            h.update(ks_null_med=med, ks_null_hi=hi)
            ks_ok = (not need) or (ksa * ksa - med * med <= ks_tol * ks_tol)
            ok_bias = bool(abs(d) + sig <= FR["bias_pass"] * lz_tol and ks_ok)
            enough = sC["M"] >= gate and sR["M"] >= gate
            k_sig = FR["k_sig_few"] if min(sR["M"], sC["M"]) < min_runs else FR["k_sig"]
            own_ok_ = bool(sR["M"] >= min_runs and ks <= ks_tol and sR["err"] <= lz_tol)
            ks_esc = need and (ksa * ksa - hi * hi > ks_tol * ks_tol or
                               (own_ok_ and ksa * ksa - med * med > ks_tol * ks_tol
                                and min(sR["M"], sC["M"]) >= FR["ks_runs"] * min_runs))
            dl_esc = (abs(d) - k_sig * sig > FR["bias_esc"] * lz_tol or
                      (own_ok_ and sig <= FR["precise_sigma"] * lz_tol and abs(d) - sig > FR["precise_esc"] * lz_tol))
            if enough and (dl_esc or ks_esc):
                esc_nsteps, why = True, ("dlnZ" if dl_esc else "KS") + " between the arms"
        # rule 3 (excess scatter): the per-run lnZ scatter far above the quoted sqrt(H/n) -> the runs do not sample
        # alike. If the check arm (N / 2) scatters clearly more, longer chains help (nsteps); else more live points
        if (sR["M"] >= max(FR["scatter_runs_min"], FR["scatter_runs_x"] * min_runs) and q_run > 0
                and sR["sd"] > FR["scatter_ratio"] * q_run):
            falls = bool(sC is not None and sC["M"] >= gate and np.isfinite(sC.get("sd", np.nan))
                         and sC["sd"] >= FR["scatter_falls"] * sR["sd"])
            if falls:
                esc_nsteps, why = True, "excess scatter that falls with nsteps"
            else:
                esc_nlive, why = True, "excess scatter that does not fall with nsteps"
        # rule 4 (runs that found a better solution): a gap in the sorted lnZ of all runs of an arm with the runs
        # above it reaching a higher maximum likelihood (far-too-short chains scatter too, but reach the same maxL)
        for runs_ in ((R, C) if check else (R,)):
            if len(runs_) >= gate and _better_solution_gap(runs_, q_run):
                esc_nlive, why = True, "some runs found a better solution"
        if esc_nlive:
            multimodal[0] = True
        blocked = False
        if multimodal[0] and esc_nsteps:
            # once runs disagree like runs that miss modes, the comparison of the arms only shows which arm got the
            # lucky runs (Kepler artefact fits escalated nsteps to 1024 on that): no nsteps escalation any more.
            # Rule 3's excess scatter goes to n_live instead (Round 23: n_live lowers the chain-length bias about like
            # nsteps, 1 / (nsteps n_live), and the mode loss); a blocked rule-2 request has no substitute.
            esc_nsteps = False
            if why.startswith("excess scatter"):
                esc_nlive, why = True, why + " (multimodal: n_live instead of nsteps)"
            else:
                blocked = True
        h.update(mode_problem=bool(esc_nlive), escalation_reason=why)
        # no remedy left (Round 23): the requested escalation cannot be done (n_live or nsteps at its cap, or an nsteps
        # request blocked by multimodal). More runs only shrink the error, so the fit stops once the lnZ error is
        # within lz_tol (passed=False, status "stalled")
        can_act = (esc_nlive and nlive < max_nlive) or (esc_nsteps and N < max_nsteps)
        stalled = bool((esc_nlive or esc_nsteps or blocked) and not can_act)
        h.update(stalled=stalled)
        hist.append(h)
        if verbose:
            s = (f"  fit: nsteps {N:4d}: {sR['M']:3d} runs x {nlive} live: logZ {lzR:.3f} +- {sR['err']:.3f} "
                 f"(per-run sd {sR['sd']:.3f}, MAD-sd {sR['mad_sd']:.3f}, quoted {h['quoted']:.3f}; mean lnZ {sR['mean']:.3f}, "
                 f"failed runs {sR['n_failed']}), split KS {ks:.3f} (p{j})")
            if check:
                s += (f" | check nsteps {Nc}: {h['check_runs']} runs, dlnZ {h['dlnz']:+.3f} +- {h['dlnz_err']:.3f}, "
                      f"KS arms {h['ks_arms']:.3f} (p{h['ks_arms_param']})")
            say(s + f", {h['wall']:.1f} s" + (" [final]" if final else ""))
        own_ok = bool(sR["M"] >= min_runs and ks <= ks_tol and sR["err"] <= lz_tol)
        passed = bool(own_ok and ok_bias and sR["n_failed"] <= FR["failed_max"] * sR["M_all"])
        # which checks are not met (reported in the result: status / reason)
        fails = []
        if sR["M"] < min_runs:
            fails.append(f"{sR['M']} runs < min_runs {min_runs}")
        if sR["err"] > lz_tol:
            fails.append(f"lnZ error {sR['err']:.3f} > lz_tol {lz_tol}")
        if ks > ks_tol:
            fails.append(f"split-half KS {ks:.3f} > ks_tol {ks_tol}")
        if not ok_bias:
            fails.append(f"chain-length check: arms N/2, N differ (dlnZ {h.get('dlnz', np.nan):+.3f} +- "
                         f"{h.get('dlnz_err', np.nan):.3f}, KS {h.get('ks_arms', np.nan):.3f})")
        if sR["n_failed"] > FR["failed_max"] * sR["M_all"]:
            fails.append(f"{sR['n_failed']} failed runs")
        h.update(fails=fails)
        if (stalled and not passed and not final and sR["M"] >= FR["stall_runs_x"] * min_runs
                and sR["err"] <= lz_tol):
            if verbose:
                say(f"  fit: {why}, no further escalation possible; lnZ error {sR['err']:.3f} <= lz_tol: stop")
            return False, "stop", sR["M_all"]
        if esc_nlive and nlive < max_nlive and not final:
            if verbose:
                say(f"  fit: {why} -> n_live {nlive} -> {2 * nlive}")
            return False, "nlive", sR["M_all"]
        escalate = bool(esc_nsteps and N < max_nsteps and not passed)
        if escalate and verbose:
            say(f"  fit: {why}")
        return passed, escalate, sR["M_all"]

    # continuous scheduling: the pool is kept full (no batch barrier); a new run goes to the arm with fewer
    # runs (done + in flight); the checks are evaluated after every `threads` completed runs (the workers
    # keep running meanwhile: the compiled runs release the GIL)
    from concurrent.futures import wait, FIRST_COMPLETED
    inflight = {}                                # future -> (nsteps, its own stop flag)
    passed = False
    n_aborted = [0]
    n_obsolete = [0]                             # runs aborted at an escalation (their arm was dropped)
    n_eval = [0, 0]                              # full evaluations: regular cadence / triggered by precheck
    stop_cause = [""]                            # "stalled": no remedy left and the lnZ error within lz_tol
    t_eval = [0.0]                               # wall time of the Python checks (main thread)
    # free unreachable objects in the main thread before the worker threads start: closed matplotlib figures of a Tk
    # backend must not be garbage-collected in a worker thread (Tcl aborts the process: "Tcl_AsyncDelete: async
    # handler deleted by the wrong thread"; seen 10-14 in test_sampling after plot_run())
    import gc
    gc.collect()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        def top_up():
            nonlocal nxt
            while len(inflight) < threads:
                Nc = max(1, N // 2)
                nR = len(arms.get(N, [])) + sum(v[0] == N for v in inflight.values())
                nC = len(arms.get(Nc, [])) + sum(v[0] == Nc for v in inflight.values())
                ns = Nc if (check and nC < nR) else N
                while nxt in used:                # smallest free index (fills the gaps left by a crash)
                    nxt += 1
                idx = nxt
                used.add(idx)
                sd = seed + 1000 * idx
                flag = np.zeros(3, dtype=np.int64)      # [stop request, calls, iterations] (progress)
                inflight[ex.submit(run, ll_u, ndim, wrapped=wrapped, nlive=nlive, nsteps=ns, k=1, seed=sd, dirs=dirs,
                                   frac_remain=frac_remain, kpair=kpair, stop=flag,
                                   recluster=recluster, data=data)] = (ns, flag, idx)

        def escalate_nlive():
            nonlocal nlive, nlesc
            dropped_idx = []
            for ns_ in list(arms):
                dropped_idx += [int(r.get("idx", -1)) for r in arms.pop(ns_)]
            for f, (ns, flag, i_) in inflight.items():
                flag[0] = 1
                n_obsolete[0] += 1
                dropped_idx.append(int(i_))
            nlive *= 2
            nlesc += 1
            if callback is not None:
                callback("state", N=N, nesc=nesc, nlive=nlive, nlesc=nlesc, dropped=None, dropped_idx=dropped_idx)

        def escalate_now():
            nonlocal N, nesc
            if verbose:
                say(f"  fit: nsteps {N} vs {max(1, N // 2)} disagree -> nsteps {2 * N}")
            dropped = max(1, N // 2)
            dropped_idx = [int(r.get("idx", -1)) for r in arms.pop(dropped, [])]
            N *= 2
            nesc += 1
            # runs of the dropped arm (old N / 2) still in flight are useless: abort them now
            for f, (ns, flag, i_) in inflight.items():
                if ns != N and ns != max(1, N // 2):
                    flag[0] = 1
                    n_obsolete[0] += 1
                    dropped_idx.append(int(i_))
            if callback is not None:
                callback("state", N=N, nesc=nesc, nlive=nlive, nlesc=nlesc, dropped=dropped, dropped_idx=dropped_idx)

        # When to evaluate (Round 13): the full evaluation (merges, KS) costs ~0.5-3 s, so up to Round 12 it ran
        # after every `threads` finished runs, and a fit passed on average ~half a run time after its criteria
        # were met. Now the cheap scalar precheck runs after every finished run and triggers the full evaluation
        # as soon as a pass (or a dlnZ escalation) is possible; while the precheck stays true but the KS checks
        # fail, the full evaluation is repeated every max(1, threads // 4) runs. The regular cadence stays.
        stop_now = False
        if resume is not None and len(arms.get(N, [])) >= 2:
            # resumed: evaluate the stored runs first (a fit that had passed passes again without new runs)
            while True:
                passed, escalate, M = evaluate(N)
                if passed or M >= max_runs or escalate == "stop":
                    stop_now = True
                    stop_cause[0] = "stalled" if escalate == "stop" else stop_cause[0]
                    break
                if not escalate:
                    break
                if escalate == "nlive":
                    escalate_nlive()
                else:
                    escalate_now()
                if len(arms.get(N, [])) < 2:
                    break
        if not stop_now:
            top_up()
        ndone = 0
        pre_last = False
        while not stop_now:
            done, _ = wait(list(inflight), timeout=p_every or None, return_when=FIRST_COMPLETED)
            if p_every and (not done or time.perf_counter() - p_state.get("t", 0.0) >= p_every):
                p_state["t"] = time.perf_counter()
                status()
            if not done:
                continue
            for f in done:
                ns, _flag, idx = inflight.pop(f)
                r = f.result()
                r.pop("samples_u", None)
                ncall_tot[0] += r["ncall"]
                if _flag[0] != 0:                # aborted (dropped arm): incomplete, never used
                    ndone += 1
                    continue
                if (ns == N or (check and ns == max(1, N // 2))) and r["nlive"] == nlive:
                    r["idx"] = int(idx)
                    if callback is not None:
                        callback("run", r=r, nsteps=ns, idx=idx)
                    # kept sorted by seed (Round 14): merges, split halves and KS depend only on WHICH runs are
                    # done, not on the order in which they finished (Round 13: KS arms 0.049 vs 0.053 for the same
                    # runs in a different order flipped an escalation at ks_tol 0.05)
                    arms.setdefault(ns, []).append(r)
                    arms[ns].sort(key=lambda q: q["seed"])
                ndone += 1
            top_up()                             # refill before the (slow, Python) evaluation
            if len(arms.get(N, [])) < 2:
                continue
            regular = ndone >= threads
            pre = precheck(N)
            if regular or (pre and (not pre_last or ndone >= max(1, threads // 4))):
                ndone = 0
                n_eval[0 if regular else 1] += 1
                te = time.perf_counter()
                passed, escalate, M = evaluate(N)
                t_eval[0] += time.perf_counter() - te
                pre_last = pre
                if escalate == "stop":
                    stop_cause[0] = "stalled"
                    break
                if passed or (M >= max_runs and escalate != "nlive"):
                    break
                if escalate == "nlive":
                    escalate_nlive()
                    pre_last = False
                elif escalate:
                    escalate_now()
                    pre_last = False
            top_up()
        # abort the runs still in flight (the compiled loop polls its stop flag every iteration) and discard them:
        # waiting for them cost up to one run time (C7 at nsteps 256: 140 s of 790)
        for ns, flag, _i in inflight.values():
            flag[0] = 1
        for f in list(inflight):
            inflight.pop(f)
            r = f.result()
            ncall_tot[0] += r["ncall"]
            n_aborted[0] += 1
    clear_status()
    passed, _, _ = evaluate(N, final=True)
    fails = hist[-1]["fails"]
    if passed:
        status, reason = "passed", ""
    elif stop_cause[0] == "stalled":
        status = "stalled"
        reason = (f"no further escalation possible ({hist[-2]['escalation_reason'] if len(hist) > 1 else ''}; n_live "
                  f"{nlive}, nsteps {N}); stopped with the lnZ error within lz_tol. Not met: " + "; ".join(fails))
    else:
        status = "run_cap"
        reason = f"max_runs ({max_runs}) reached. Not met: " + "; ".join(fails)
    bad = set(failed_runs(arms[N]))
    R = [r for i, r in enumerate(arms[N]) if i not in bad]
    lzm, samples, ess = merge(R, seed=seed)
    zg = np.array([r["logz"] for r in R])        # bootstrap of mean lnZ over runs (scalars; the full re-merge
    _rng = np.random.default_rng(seed)            # bootstrap_logz cost ~1 s per resample at 100 runs)
    lzb = float(np.std(zg[_rng.integers(0, len(zg), (1000, len(zg)))].mean(axis=1), ddof=1)) if len(zg) > 1 else np.nan
    mixm = np.mean([r["mix"] for r in R], axis=0)
    return dict(mix=mixm, logz=lzm, logzerr=hist[-1]["logz_err"], logzerr_boot=lzb, samples_u=samples, ess=ess,
                runs=R, n_failed=len(bad), n_aborted=n_aborted[0], n_obsolete=n_obsolete[0], n_eval=n_eval[0],
                n_eval_pre=n_eval[1], t_eval=t_eval[0], t_compile=t_compile, runs_check=arms.get(max(1, N // 2), []) if check else [], history=hist, passed=passed,
                wall=time.perf_counter() - t0, ncall=hist[-1]["ncall"], nlive=nlive, nlive0=nlive0,
                n_nlive_escalations=nlesc, multimodal=multimodal[0], nsteps=N, nsteps0=nsteps,
                n_escalations=nesc, dlnz_check=hist[-1].get("dlnz"), dlnz_check_err=hist[-1].get("dlnz_err"),
                ks_arms=hist[-1].get("ks_arms"), maxL=float(max(r["maxL"] for r in R)), status=status, reason=reason)
