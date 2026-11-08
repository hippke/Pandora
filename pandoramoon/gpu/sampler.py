"""GPU nested slice sampler (GPU_PLAN.md phase 3; kernels in sampler.cu).

    from pandoramoon.gpu.sampler import gpu_runs, PandoraEvaluator
    ev = PandoraEvaluator(sampler.models["moon"], sampler.likelihood, t0_ref)
    runs = gpu_runs(ev, ndim, wrapped, R=32, nlive=500, k=64, seeds=range(1, 33))

Each element of `runs` is a dict in the format of pandoramoon.nss.run() (dead points, weights, lnZ, insertion
ranks, ...), so nss.merge_weights and the checks of nss.fit apply unchanged. R runs execute at once; every run
replaces its k lowest live points per iteration by k slice chains (parallel nested sampling, as nss.run(k > 1)).
"""
import math
import time

import numpy as np

from pandoramoon.gpu import _cuda

LOGZERO = -1e300
PLATEAU = -1e299

RUN_DTYPE = np.dtype({
    "names": ["logX", "logZ", "Lmin", "scale", "ncall", "nexp_tot", "nstep_tot", "nstuck", "nmaxexp",
              "it", "nd", "nins", "nrecl", "repl_since", "status", "nmix", "overflow", "rng", "phase", "nsel"],
    "formats": ["f8"] * 4 + ["i8"] * 5 + ["i4"] * 8 + ["u8"] + ["i4"] * 2,
    "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 76, 80, 84, 88, 92, 96, 100, 104, 112, 116],
    "itemsize": 120})
CHAIN_DTYPE = np.dtype({
    "names": ["XL", "left", "right", "t", "nc", "nexp", "nstuck", "nmaxexp", "ctr", "state", "step", "e", "start"],
    "formats": ["f8"] * 4 + ["i8"] * 4 + ["u8"] + ["i4"] * 4,
    "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 76, 80, 84],
    "itemsize": 88})


# ------------------------------------------------------------------------------------------------ evaluators
class PandoraEvaluator:
    """L(u) = log w(u) + logL(theta(u)) of a NestedSampler model on the GPU (prior transform + likelihood)"""

    def __init__(self, model, like, t0_ref, **kw):
        from pandoramoon.gpu.transform import GPUTransform
        from pandoramoon.gpu.likelihood import GPULikelihood
        if int(model.D[9]) != 1:
            raise ValueError("PandoraEvaluator: the planet+moon model only (the planet-only model stays on the CPU)")
        self.tr = GPUTransform(model)
        self.like = GPULikelihood(like, t0_ref=t0_ref, **kw)
        self._th = None

    def evaluate(self, dU, dACT, n, dLW, dLLK):
        from numba import cuda
        if self._th is None or self._th.shape[0] < n:
            self._th = cuda.device_array((n, 20), np.float64)
        self.tr.device(dU, n, self._th, dLW, dACT)
        self.like.loglike_device(self._th, n, dLLK, dACT=dACT)


class GaussEvaluator:
    """synthetic targets with known evidence (lnZ = 0 up to the mass outside the cube): one Gaussian at 0.5, or two
    (weights 1/2) at 0.3 and 0.7, with sd sigma in every dimension"""

    def __init__(self, ndim, sigma, twomode=False):
        self.ndim = int(ndim)
        self.sigma = float(sigma)
        self.kernel = _cuda.sampler_module().get_kernel("t_twomode" if twomode else "t_gauss")
        # exact lnZ: the mass of the normalised target inside the unit cube
        from math import erf, sqrt
        if twomode:
            m = 0.5 * (erf(0.7 / (sqrt(2) * sigma)) - erf(-0.3 / (sqrt(2) * sigma)))
        else:
            m = erf(0.5 / (sqrt(2) * sigma))
        self.logz_true = self.ndim * math.log(m)

    def evaluate(self, dU, dACT, n, dLW, dLLK):
        _cuda.launch(self.kernel, (n + 127) // 128, 128, _cuda.ptr(dU), _cuda.ptr(dACT), np.int32(n),
                     np.int32(self.ndim), np.float64(self.sigma), _cuda.ptr(dLW), _cuda.ptr(dLLK))


def _eval_host(ev, U):
    """L at the rows of U (host arrays), all evaluated"""
    from numba import cuda
    n = U.shape[0]
    dU = cuda.to_device(np.ascontiguousarray(U, dtype=np.float64))
    dLW = cuda.device_array(n, np.float64)
    dLLK = cuda.device_array(n, np.float64)
    ev.evaluate(dU, None, n, dLW, dLLK)
    cuda.synchronize()
    lw = dLW.copy_to_host()
    llk = dLLK.copy_to_host()
    return np.where(lw > LOGZERO, lw + llk, LOGZERO)


def _init_live(ev, ndim, R, nlive, seeds, init_min_draws, chunk=100_000):
    """rejection init above the LOGZERO plateau (nss.run(init_reject=True)): per run, draws until >= nlive accepted
    and >= init_min_draws drawn; ln X0 = ln(accepted / drawn)"""
    LU = np.empty((R, nlive, ndim))
    LL = np.empty((R, nlive))
    logx0 = np.empty(R)
    ndraw_all = np.empty(R, dtype=np.int64)
    nacc_all = np.empty(R, dtype=np.int64)
    for r, seed in enumerate(seeds):
        rng = np.random.default_rng((int(seed) * 7919 + 1_000_003) % 2 ** 32)
        Us, Ls, nacc, ndraw = [], [], 0, 0
        while nacc < nlive or ndraw < init_min_draws:
            m = max(nlive, min(init_min_draws, chunk))
            U = rng.uniform(size=(m, ndim))
            L = _eval_host(ev, U)
            ok = L > PLATEAU
            nacc += int(ok.sum())
            ndraw += m
            if sum(len(q) for q in Ls) < nlive:
                Us.append(U[ok])
                Ls.append(L[ok])
            if ndraw > 1000 * max(init_min_draws, nlive):
                raise RuntimeError("init: almost no prior volume above the LOGZERO plateau")
        LU[r] = np.concatenate(Us)[:nlive]
        LL[r] = np.concatenate(Ls)[:nlive]
        logx0[r] = math.log(nacc / ndraw)
        ndraw_all[r] = ndraw
        nacc_all[r] = nacc
    return LU, LL, logx0, ndraw_all, nacc_all


# ------------------------------------------------------------------------------------------------ the sampler
def gpu_runs(ev, ndim, wrapped=None, R=32, nlive=500, nsteps=None, k=64, seeds=None, frac_remain=0.01, kpair=4,
             recluster=0.25, init_min_draws=100_000, scale0=1.0, cap_factor=150, check_every=256,
             max_ticks=50_000_000, verbose=False, profile=False, overlap=True):
    """R independent nested-sampling runs on the GPU; returns a list of nss.run()-style dicts"""
    from numba import cuda
    from pandoramoon.nss import insertion_test, _FIT
    if nlive > 2048:
        raise ValueError("gpu_runs: nlive <= 2048 (NMAX of sampler.cu)")
    if not 1 <= k <= nlive // 4:
        raise ValueError("gpu_runs: k (points replaced per iteration) must be <= nlive / 4")
    D = int(ndim)
    n = int(nlive)
    seeds = list(range(1, R + 1)) if seeds is None else [int(s) for s in seeds]
    assert len(seeds) == R
    nsteps = int(nsteps or _FIT["nsteps_per_dim"] * D)
    wrapped = np.array(wrapped if wrapped is not None else [False] * D, dtype=np.int32)
    cap = int(cap_factor * n)
    mod = _cuda.sampler_module(D)
    k_chain = mod.get_kernel("chain_tick")
    k_run = mod.get_kernel("run_tick")
    t0 = time.perf_counter()
    LU, LL, logx0, ndraw, nacc = _init_live(ev, D, R, n, seeds, init_min_draws)
    t1 = time.perf_counter()
    C = R * k
    rs = np.zeros(R, dtype=RUN_DTYPE)
    rs["logX"] = logx0
    rs["logZ"] = -np.inf
    rs["scale"] = scale0
    rs["status"] = 1
    rs["phase"] = 0
    rs["repl_since"] = 10 * n                 # neighbour lists at the first iteration (as nss.ns)
    ch = np.zeros(C, dtype=CHAIN_DTYPE)
    nbr0 = np.repeat(np.arange(n, dtype=np.int32)[None, :, None], R, axis=0).repeat(kpair, axis=2)
    d = dict(
        RS=cuda.to_device(rs.view(np.uint8)), CH=cuda.to_device(ch.view(np.uint8)),
        X=cuda.device_array((C, D), np.float64), V=cuda.device_array((C, D), np.float64),
        PU=cuda.to_device(np.full((C, D), 0.5)), LW=cuda.device_array(C, np.float64),
        LLK=cuda.device_array(C, np.float64), ACT=cuda.to_device(np.zeros(C, np.int32)),
        LU=cuda.to_device(np.ascontiguousarray(LU)), LL=cuda.to_device(np.ascontiguousarray(LL)),
        ALIVE=cuda.to_device(np.ones((R, n), np.uint8)), NBR=cuda.to_device(np.ascontiguousarray(nbr0)),
        SD=cuda.to_device(np.full((R, D), 0.1)), MIX=cuda.to_device(np.zeros((R, D))),
        SEL=cuda.device_array((R, k), np.int32), STARTS=cuda.device_array((R, k), np.int32),
        DU=cuda.device_array((R, cap, D), np.float64), DL=cuda.device_array((R, cap), np.float64),
        DLW=cuda.device_array((R, cap), np.float64), INS=cuda.device_array((R, cap), np.int32),
        WR=cuda.to_device(wrapped), PROF=cuda.to_device(np.zeros((R, 16), np.int64)))
    P = {q: _cuda.ptr(a) for q, a in d.items()}
    seed32 = np.uint32((seeds[0] * 2654435761) % 2 ** 32)
    sd_every = max(1, n // (10 * k))
    chain_args = (np.int32(R), np.int32(k), np.int32(n), np.int32(D), np.int32(kpair), np.int32(nsteps), seed32,
                  P["CH"], P["RS"], P["X"], P["V"], P["PU"], P["LW"], P["LLK"], P["ACT"], P["LU"], P["ALIVE"],
                  P["NBR"], P["WR"])
    run_args = (np.int32(R), np.int32(k), np.int32(n), np.int32(D), np.int32(kpair), np.int32(nsteps), np.int32(cap),
                np.float64(math.log(frac_remain)), np.float64(recluster), seed32, np.int32(sd_every),
                P["RS"], P["CH"], P["X"], P["LU"], P["LL"], P["ALIVE"], P["NBR"], P["SD"], P["MIX"], P["SEL"],
                P["STARTS"], P["DU"], P["DL"], P["DLW"], P["INS"], P["WR"], P["PROF"] if profile else 0)
    gc = (32 * C + 127) // 128              # chain_tick: one warp per chain
    import cuda.core as cc
    main = cc.LEGACY_DEFAULT_STREAM
    side = cc.Device().create_stream()
    e_run = None
    tick = 0
    prof = None
    while tick < max_ticks:
        if profile and prof is None and tick >= 20 * check_every:
            # device time of each part of a tick (events between the launches), the active chains, the host loop
            ev_t = {q: [] for q in ("chain_tick", "run_tick", "evaluate")}
            nact = []
            for _ in range(400):
                e0, e1, e2, e3 = cuda.event(), cuda.event(), cuda.event(), cuda.event()
                e0.record()
                _cuda.launch(k_chain, gc, 128, *chain_args)
                e1.record()
                _cuda.launch(k_run, R, 512, *run_args)
                e2.record()
                ev.evaluate(d["PU"], d["ACT"], C, d["LW"], d["LLK"])
                e3.record()
                e3.synchronize()
                ev_t["chain_tick"].append(cuda.event_elapsed_time(e0, e1))
                ev_t["run_tick"].append(cuda.event_elapsed_time(e1, e2))
                ev_t["evaluate"].append(cuda.event_elapsed_time(e2, e3))
                nact.append(int(d["ACT"].copy_to_host().sum()))
            cuda.synchronize()
            th0 = time.perf_counter()
            for _ in range(200):
                _cuda.launch(k_chain, gc, 128, *chain_args)
                _cuda.launch(k_run, R, 512, *run_args)
                ev.evaluate(d["PU"], d["ACT"], C, d["LW"], d["LLK"])
            th1 = time.perf_counter()
            cuda.synchronize()
            th2 = time.perf_counter()
            tick += 600
            prof = {q: 1e3 * float(np.median(v)) for q, v in ev_t.items()}
            prof.update({q + "_mean": 1e3 * float(np.mean(v)) for q, v in ev_t.items()})
            prof.update({q + "_max": 1e3 * float(np.max(v)) for q, v in ev_t.items()})
            prof.update(active_mean=float(np.mean(nact)), chains=C, host_us_per_tick=1e6 * (th1 - th0) / 200,
                        wall_us_per_tick=1e6 * (th2 - th0) / 200)
            print("  gpu_runs profile [us per tick]:", {q: round(v, 1) for q, v in prof.items()}, flush=True)
            pr = d["PROF"].copy_to_host().sum(axis=0)[:10]
            its = int(d["RS"].copy_to_host().view(RUN_DTYPE)["it"].sum())
            names = ["done-check+ranks", "mixing", "replace+scale", "sort", "dead copy", "weights", "stop", "spread",
                     "neighbours", "starts+chains"]
            print("  run_tick sections [us per iteration end, clock 2.5 GHz assumed]:",
                  {nm: round(v / 2.5e3 / max(its, 1), 1) for nm, v in zip(names, pr)}, flush=True)
        for _ in range(check_every):
            if overlap:
                # run_tick on a second stream, concurrent with the evaluation: it touches only runs whose chains are
                # all done (ACT = 0, not evaluated); the next chain_tick waits for it
                main.wait(e_run) if tick or _ else None
                _cuda.launch(k_chain, gc, 128, *chain_args, stream=main)
                e_chain = main.record()
                side.wait(e_chain)
                _cuda.launch(k_run, R, 512, *run_args, stream=side)
                e_run = side.record()
                ev.evaluate(d["PU"], d["ACT"], C, d["LW"], d["LLK"])
            else:
                _cuda.launch(k_chain, gc, 128, *chain_args)
                _cuda.launch(k_run, R, 512, *run_args)
                ev.evaluate(d["PU"], d["ACT"], C, d["LW"], d["LLK"])
        tick += check_every
        if overlap:
            main.wait(e_run)
        cuda.synchronize()
        st = d["RS"].copy_to_host().view(RUN_DTYPE)
        if verbose and tick % (check_every * 40) == 0:
            print(f"  gpu_runs: tick {tick}, running {np.sum(st['status'] == 1)}, it {st['it'].min()}-{st['it'].max()}, "
                  f"calls {st['ncall'].sum():,}", flush=True)
        if np.all(st["status"] != 1):
            break
    t2 = time.perf_counter()
    st = d["RS"].copy_to_host().view(RUN_DTYPE)
    if np.any(st["status"] == 3):
        raise RuntimeError(f"gpu_runs: dead-point buffer overflow in {np.sum(st['status'] == 3)} run(s) "
                           f"(raise cap_factor, now {cap_factor})")
    if np.any(st["status"] == 1):
        raise RuntimeError("gpu_runs: max_ticks reached")
    DU = d["DU"].copy_to_host()
    DL = d["DL"].copy_to_host()
    DLW = d["DLW"].copy_to_host()
    INS = d["INS"].copy_to_host()
    MIX = d["MIX"].copy_to_host()
    out = []
    for r in range(R):
        nd = int(st["nd"][r])
        dU, dL, dlogw = DU[r, :nd], DL[r, :nd], DLW[r, :nd]
        ins = INS[r, :int(st["nins"][r])].astype(np.int64)
        lw = dlogw + dL
        logz = float(np.logaddexp.reduce(lw))
        w = np.exp(lw - logz)
        fin = w > 0
        H = float(np.sum(w[fin] * (dL[fin] - logz)))
        rng = np.random.default_rng(seeds[r])
        cw = np.cumsum(w)
        cw /= cw[-1]
        nsamp = int(np.exp(-np.sum(w[fin] * np.log(w[fin]))))
        pos = (rng.uniform() + np.arange(nsamp)) / nsamp
        samples = dU[np.searchsorted(cw, pos)]
        insd = insertion_test(ins, n - k, n, seeds[r])
        out.append(dict(
            seed=seeds[r], nlive=n, k=k, nsteps=nsteps, device="gpu",
            ins_u=((ins + np.random.default_rng(seeds[r] + 11).uniform(size=len(ins))) / (n - k + 1)).astype(np.float32),
            mix=MIX[r] / max(int(st["nmix"][r]), 1), insertion=insd, logz=logz,
            logzerr=math.sqrt(max(H, 0.0) / n), H=H, ncall=int(st["ncall"][r]) + int(ndraw[r]),
            niter=int(st["it"][r]), wall=t2 - t0, wall_init=t1 - t0, samples_u=samples,
            ess=float(1.0 / np.sum(w ** 2)), dead_u=dU, dead_L=dL, logw=dlogw, maxL=float(dL.max()),
            final_scale=float(st["scale"][r]), expand_rate=float(st["nexp_tot"][r] / max(st["nstep_tot"][r], 1)),
            nrecluster=int(st["nrecl"][r]), dirs="pairs", logx0=float(logx0[r]), init_acc=int(nacc[r]),
            init_draws=int(ndraw[r]), n_plateau=0, nstuck=int(st["nstuck"][r]), nmaxexp=int(st["nmaxexp"][r]),
            ticks=tick))
    return out
