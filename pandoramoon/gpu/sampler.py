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
              "it", "nd", "nins", "nrecl", "repl_since", "status", "nmix", "overflow", "seed", "phase", "nsteps"],
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
        # eccentric moon only if ecc_moon is sampled or fixed above 0 (else the kernel without the Kepler loop)
        D = model.D
        ecc = bool(17 in set(np.asarray(D[11]).tolist()) or float(np.asarray(D[10])[17]) != 0.0)
        kw.setdefault("eccentric", ecc)
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


# ------------------------------------------------------------------------------------------------ the run pool
def _seed64(seed):
    """splitmix64 of the run's seed: the key of all its random streams on the device"""
    z = (int(seed) + 0x9E3779B97F4A7C15) % 2 ** 64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % 2 ** 64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % 2 ** 64
    return z ^ (z >> 31)


class _Req:
    __slots__ = ("nlive", "nsteps", "seed", "flag", "future", "t_start", "ndraw", "nacc", "logx0", "t_init")

    def __init__(self, nlive, nsteps, seed, flag, future):
        self.nlive, self.nsteps, self.seed, self.flag, self.future = nlive, nsteps, seed, flag, future


class GPURunPool:
    """An executor for nss.fit: submit(run, ..., nlive=, nsteps=, seed=, stop=flag) returns a Future of the run's
    nss.run()-style dict. A background thread drives the GPU (chain_tick / run_tick / evaluation per tick) and starts
    queued runs in the slots of finished or aborted ones (continuous scheduling). stop[0] != 0 aborts a run (its
    future gets a partial dict); stop[1], stop[2] are kept at its calls and iterations (progress).

    Every run replaces k = min(64, nlive // 8) points per iteration (parallel nested sampling, validated up to
    k / n_live = 1/8, GPU_LOG Round 2.6); slots = 4096 // k runs at once (about 4096 chains in flight), fewer if the
    dead-point buffers would exceed mem_budget bytes. A change of nlive between runs waits for the slots to drain.
    The random streams of a run derive from its seed, not from its slot."""

    def __init__(self, ev, ndim, wrapped=None, slots=None, k=None, frac_remain=0.01, kpair=4, recluster=0.25,
                 init_min_draws=100_000, scale0=1.0, cap_factor=60, ticks_per_check=128, mem_budget=3.0e9,
                 overlap=True):
        import threading
        from collections import deque
        self.ev = ev
        self.D = int(ndim)
        self.wrapped = np.array(wrapped if wrapped is not None else [False] * self.D, dtype=np.int32)
        self.slots_req, self.k_req = slots, k
        self.frac_remain, self.kpair, self.recluster = float(frac_remain), int(kpair), float(recluster)
        self.init_min_draws, self.scale0, self.cap_factor = int(init_min_draws), float(scale0), int(cap_factor)
        self.ticks_per_check, self.mem_budget, self.overlap = int(ticks_per_check), float(mem_budget), bool(overlap)
        self._queue = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closing = False
        self.error = None
        self.n = None
        self.stats = dict(ticks=0, runs_done=0, runs_aborted=0, t_init=0.0, t_ticks=0.0)
        self._thread = threading.Thread(target=self._main, name="GPURunPool", daemon=True)
        self._thread.start()

    # -------------------------------------------------------------------------------------------- configuration
    def layout(self, nlive):
        """(k, slots, cap) for a given n_live"""
        n = int(nlive)
        k = int(self.k_req or min(64, max(1, n // 8)))
        if not 1 <= k <= n // 4:
            raise ValueError("GPURunPool: k must be <= nlive / 4")
        if n > 8192:
            raise ValueError("GPURunPool: nlive <= 8192")
        cap = self.cap_factor * n
        per_slot = (cap * (self.D * 8 + 8 + 8 + 4) + n * (self.D * 8 + 8 + 1 + 4 * self.kpair) + k * (3 * self.D * 8)
                    + (12 * n * 2 if n > 2048 else 0))
        slots = int(self.slots_req or max(4, min(256, 4096 // k)))
        slots = max(1, min(slots, int(self.mem_budget // per_slot)))
        return k, slots, cap

    def slots_for(self, nlive):
        return self.layout(nlive)[1]

    # -------------------------------------------------------------------------------------------- executor API
    def submit(self, fn=None, *args, nlive, nsteps, seed, stop=None, **kw):
        from concurrent.futures import Future
        fut = Future()
        flag = stop if stop is not None else np.zeros(3, dtype=np.int64)
        with self._lock:
            if self.error is not None:
                fut.set_exception(self.error)
                return fut
            self._queue.append(_Req(int(nlive), int(nsteps), int(seed), flag, fut))
        self._wake.set()
        return fut

    def shutdown(self, wait=True):
        self._closing = True
        self._wake.set()
        if wait:
            self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown(wait=True)
        return False

    # -------------------------------------------------------------------------------------------- the device
    def _alloc(self, nlive):
        from numba import cuda
        import cuda.core as cc
        n, D = int(nlive), self.D
        k, R, cap = self.layout(n)
        self.n, self.k, self.R, self.cap = n, k, R, cap
        C = R * k
        self.C = C
        nbr0 = np.repeat(np.arange(n, dtype=np.int32)[None, :, None], R, axis=0).repeat(self.kpair, axis=2)
        self.d = d = dict(
            RS=cuda.to_device(np.zeros(R, dtype=RUN_DTYPE).view(np.uint8)),
            CH=cuda.to_device(np.zeros(C, dtype=CHAIN_DTYPE).view(np.uint8)),
            X=cuda.device_array((C, D), np.float64), V=cuda.device_array((C, D), np.float64),
            PU=cuda.to_device(np.full((C, D), 0.5)), LW=cuda.device_array(C, np.float64),
            LLK=cuda.device_array(C, np.float64), ACT=cuda.to_device(np.zeros(C, np.int32)),
            LU=cuda.to_device(np.full((R, n, D), 0.5)), LL=cuda.to_device(np.zeros((R, n))),
            ALIVE=cuda.to_device(np.ones((R, n), np.uint8)), NBR=cuda.to_device(np.ascontiguousarray(nbr0)),
            SD=cuda.to_device(np.full((R, D), 0.1)), MIX=cuda.to_device(np.zeros((R, D))),
            SEL=cuda.device_array((R, k), np.int32), STARTS=cuda.device_array((R, k), np.int32),
            DU=cuda.device_array((R, cap, D), np.float64), DL=cuda.device_array((R, cap), np.float64),
            DLW=cuda.device_array((R, cap), np.float64), INS=cuda.device_array((R, cap), np.int32),
            WR=cuda.to_device(self.wrapped))
        ng = 1
        while ng < n:
            ng <<= 1
        if n > 2048:                     # selection in global scratch (sampler.cu NMAX)
            d["GK"] = cuda.device_array((R, ng), np.float64)
            d["GI"] = cuda.device_array((R, ng), np.int32)
        mod = _cuda.sampler_module(D)
        self.k_chain = mod.get_kernel("chain_tick")
        self.k_run = mod.get_kernel("run_tick")
        self.ng = ng
        self._args()
        self.gc = (32 * C + 127) // 128
        self.main_stream = cc.LEGACY_DEFAULT_STREAM
        self.side = cc.Device().create_stream()
        self.slot = [None] * R           # slot -> _Req
        self.nbr0_slot = np.ascontiguousarray(nbr0[0])

    def _args(self):
        """the kernel argument tuples (after an allocation or a buffer growth)"""
        d, n, k, R, D, cap, ng = self.d, self.n, self.k, self.R, self.D, self.cap, self.ng
        P = {q: _cuda.ptr(a) for q, a in d.items()}
        sd_every = max(1, n // (10 * k))
        self.chain_args = (np.int32(R), np.int32(k), np.int32(n), np.int32(D), np.int32(self.kpair), np.int32(0),
                           np.uint32(0), P["CH"], P["RS"], P["X"], P["V"], P["PU"], P["LW"], P["LLK"], P["ACT"],
                           P["LU"], P["ALIVE"], P["NBR"], P["WR"])
        self.run_args = (np.int32(R), np.int32(k), np.int32(n), np.int32(D), np.int32(self.kpair), np.int32(0),
                         np.int32(cap), np.float64(math.log(self.frac_remain)), np.float64(self.recluster),
                         np.uint32(0), np.int32(sd_every), P["RS"], P["CH"], P["X"], P["LU"], P["LL"], P["ALIVE"],
                         P["NBR"], P["SD"], P["MIX"], P["SEL"], P["STARTS"], P["DU"], P["DL"], P["DLW"], P["INS"],
                         P["WR"], 0, P.get("GK", 0), P.get("GI", 0), np.int32(ng))

    def _grow(self):
        """double the dead-point buffers (DU, DL, DLW, INS), keeping their contents; False if the memory budget
        does not allow it (a run then aborts with a buffer overflow)"""
        from numba import cuda
        R, D, cap = self.R, self.D, self.cap
        new = 2 * cap
        if R * new * (D * 8 + 8 + 8 + 4) > 2.0 * self.mem_budget:
            return False
        for name, dt, tail in (("DU", np.float64, (D,)), ("DL", np.float64, ()), ("DLW", np.float64, ()),
                               ("INS", np.int32, ())):
            old = self.d[name]
            arr = cuda.device_array((R, new) + tail, dt)
            for r in range(R):
                arr[r, :cap].copy_to_device(old[r])
            self.d[name] = arr
            del old
        self.cap = new
        self._args()
        self.stats["grow"] = self.stats.get("grow", 0) + 1
        return True

    def _start(self, r, req):
        """initialise slot r for request req (the device is idle: called between tick bursts)"""
        t = time.perf_counter()
        n, D, k = self.n, self.D, self.k
        LU, LL, logx0, ndraw, nacc = _init_live(self.ev, D, 1, n, [req.seed], self.init_min_draws)
        d = self.d
        rec = np.zeros(1, dtype=RUN_DTYPE)
        rec["logX"] = logx0[0]
        rec["logZ"] = -np.inf
        rec["scale"] = self.scale0
        rec["status"] = 1
        rec["phase"] = 0
        rec["repl_since"] = 10 * n          # neighbour lists at the first iteration (as nss.ns)
        rec["seed"] = _seed64(req.seed)
        rec["nsteps"] = req.nsteps
        isz, csz = RUN_DTYPE.itemsize, CHAIN_DTYPE.itemsize
        d["RS"][r * isz:(r + 1) * isz].copy_to_device(rec.view(np.uint8))
        d["CH"][r * k * csz:(r + 1) * k * csz].copy_to_device(np.zeros(k, dtype=CHAIN_DTYPE).view(np.uint8))
        d["ACT"][r * k:(r + 1) * k].copy_to_device(np.zeros(k, np.int32))
        d["LU"][r].copy_to_device(np.ascontiguousarray(LU[0]))
        d["LL"][r].copy_to_device(np.ascontiguousarray(LL[0]))
        d["ALIVE"][r].copy_to_device(np.ones(n, np.uint8))
        d["NBR"][r].copy_to_device(self.nbr0_slot)
        d["MIX"][r].copy_to_device(np.zeros(D))
        d["SD"][r].copy_to_device(np.full(D, 0.1))
        req.ndraw, req.nacc, req.logx0 = int(ndraw[0]), int(nacc[0]), float(logx0[0])
        req.t_start = time.perf_counter()
        req.t_init = req.t_start - t
        self.stats["t_init"] += req.t_init
        self.slot[r] = req

    def _free(self, r):
        rec = np.zeros(1, dtype=RUN_DTYPE)
        isz = RUN_DTYPE.itemsize
        self.d["RS"][r * isz:(r + 1) * isz].copy_to_device(rec.view(np.uint8))
        self.slot[r] = None

    def _result(self, r, st):
        """the nss.run()-style dict of the finished run in slot r"""
        from pandoramoon.nss import insertion_test
        req, d, n, k = self.slot[r], self.d, self.n, self.k
        nd, nins = int(st["nd"]), int(st["nins"])
        dU = d["DU"][r, :nd].copy_to_host()
        dL = d["DL"][r, :nd].copy_to_host()
        dlogw = d["DLW"][r, :nd].copy_to_host()
        ins = d["INS"][r, :nins].copy_to_host().astype(np.int64)
        mix = d["MIX"][r].copy_to_host()
        lw = dlogw + dL
        logz = float(np.logaddexp.reduce(lw))
        w = np.exp(lw - logz)
        fin = w > 0
        H = float(np.sum(w[fin] * (dL[fin] - logz)))
        rng = np.random.default_rng(req.seed)
        cw = np.cumsum(w)
        cw /= cw[-1]
        nsamp = int(np.exp(-np.sum(w[fin] * np.log(w[fin]))))
        pos = (rng.uniform() + np.arange(nsamp)) / nsamp
        samples = dU[np.searchsorted(cw, pos)]
        insd = insertion_test(ins, n - k, n, req.seed)
        wall = time.perf_counter() - req.t_start
        return dict(
            seed=req.seed, nlive=n, k=k, nsteps=req.nsteps, device="gpu",
            ins_u=((ins + np.random.default_rng(req.seed + 11).uniform(size=len(ins))) / (n - k + 1)).astype(np.float32),
            mix=mix / max(int(st["nmix"]), 1), insertion=insd, logz=logz, logzerr=math.sqrt(max(H, 0.0) / n), H=H,
            ncall=int(st["ncall"]) + req.ndraw, niter=int(st["it"]), wall=wall + req.t_init, wall_init=req.t_init,
            samples_u=samples, ess=float(1.0 / np.sum(w ** 2)), dead_u=dU, dead_L=dL, logw=dlogw,
            maxL=float(dL.max()), final_scale=float(st["scale"]),
            expand_rate=float(st["nexp_tot"] / max(st["nstep_tot"], 1)), nrecluster=int(st["nrecl"]), dirs="pairs",
            logx0=req.logx0, init_acc=req.nacc, init_draws=req.ndraw, n_plateau=0, nstuck=int(st["nstuck"]),
            nmaxexp=int(st["nmaxexp"]))

    @staticmethod
    def _aborted(req, ncall=0):
        return dict(seed=req.seed, nlive=req.nlive, nsteps=req.nsteps, ncall=int(ncall), aborted=True,
                    logz=float("nan"), samples_u=None, device="gpu")

    def _service(self):
        """between tick bursts (device idle): finished / aborted runs out, queued runs in. Returns #active slots."""
        from numba import cuda
        if self.n is not None:
            st = self.d["RS"].copy_to_host().view(RUN_DTYPE)
            run_ = st["status"] == 1
            if run_.any() and int(st["nd"][run_].max()) > 0.8 * self.cap:
                if self._grow():
                    st = self.d["RS"].copy_to_host().view(RUN_DTYPE)
            for r, req in enumerate(self.slot):
                if req is None:
                    continue
                s = int(st["status"][r])
                req.flag[1] = int(st["ncall"][r]) + req.ndraw
                req.flag[2] = int(st["it"][r])
                if s == 2:
                    res = self._result(r, st[r])
                    self._free(r)
                    self.stats["runs_done"] += 1
                    req.future.set_result(res)
                elif s == 3:
                    self._free(r)
                    req.future.set_exception(RuntimeError(
                        f"GPURunPool: dead-point buffer overflow (cap_factor {self.cap_factor} x n_live); raise cap_factor"))
                elif req.flag[0] != 0:
                    ncall = int(st["ncall"][r]) + req.ndraw
                    self._free(r)
                    self.stats["runs_aborted"] += 1
                    req.future.set_result(self._aborted(req, ncall))
        # queued requests: aborted before their start, or started in free slots
        while True:
            with self._lock:
                if not self._queue:
                    break
                req = self._queue[0]
                if req.flag[0] != 0:
                    self._queue.popleft()
                    req.future.set_result(self._aborted(req))
                    continue
                active = 0 if self.n is None else sum(q is not None for q in self.slot)
                if self.n != req.nlive:
                    if active:
                        break                    # an n_live change: wait until the slots drain
                    self._queue.popleft()
                    self._alloc(req.nlive)
                else:
                    if all(q is not None for q in self.slot):
                        break
                    self._queue.popleft()
            r = self.slot.index(None)
            self._start(r, req)
        cuda.synchronize()
        return 0 if self.n is None else sum(q is not None for q in self.slot)

    def _burst(self):
        t = time.perf_counter()
        d, ev, C = self.d, self.ev, self.C
        e_run = None
        for i in range(self.ticks_per_check):
            if self.overlap:
                if e_run is not None:
                    self.main_stream.wait(e_run)
                _cuda.launch(self.k_chain, self.gc, 128, *self.chain_args, stream=self.main_stream)
                e_chain = self.main_stream.record()
                self.side.wait(e_chain)
                _cuda.launch(self.k_run, self.R, 512, *self.run_args, stream=self.side)
                e_run = self.side.record()
                ev.evaluate(d["PU"], d["ACT"], C, d["LW"], d["LLK"])
            else:
                _cuda.launch(self.k_chain, self.gc, 128, *self.chain_args)
                _cuda.launch(self.k_run, self.R, 512, *self.run_args)
                ev.evaluate(d["PU"], d["ACT"], C, d["LW"], d["LLK"])
        if e_run is not None:
            self.main_stream.wait(e_run)
        from numba import cuda
        cuda.synchronize()
        self.stats["ticks"] += self.ticks_per_check
        self.stats["t_ticks"] += time.perf_counter() - t

    def _main(self):
        try:
            import cuda.core as cc
            from numba import cuda
            cc.Device().set_current()
            cuda.current_context()
            while True:
                active = self._service()
                if active:
                    self._burst()
                    continue
                if self._closing:
                    with self._lock:
                        empty = not self._queue
                    if empty:
                        break
                self._wake.wait(0.05)
                self._wake.clear()
        except BaseException as ex:      # fail every pending future (the fit raises in the main thread)
            self.error = ex
            with self._lock:
                pend = list(self._queue)
                self._queue.clear()
            for req in pend + [q for q in (self.slot if self.n is not None else []) if q is not None]:
                if not req.future.done():
                    req.future.set_exception(ex)


def gpu_runs(ev, ndim, wrapped=None, R=32, nlive=500, nsteps=None, k=64, seeds=None, frac_remain=0.01, kpair=4,
             recluster=0.25, init_min_draws=100_000, scale0=1.0, cap_factor=60, overlap=True, **kw):
    """R independent nested-sampling runs on the GPU at once (a GPURunPool with R slots); returns their
    nss.run()-style dicts in the order of seeds"""
    from pandoramoon.nss import _FIT
    D = int(ndim)
    if nlive > 8192:
        raise ValueError("gpu_runs: nlive <= 8192")
    seeds = list(range(1, R + 1)) if seeds is None else [int(s) for s in seeds]
    nsteps = int(nsteps or _FIT["nsteps_per_dim"] * D)
    with GPURunPool(ev, D, wrapped=wrapped, slots=len(seeds), k=k, frac_remain=frac_remain, kpair=kpair,
                    recluster=recluster, init_min_draws=init_min_draws, scale0=scale0, cap_factor=cap_factor,
                    overlap=overlap) as pool:
        futs = [pool.submit(nlive=nlive, nsteps=nsteps, seed=s) for s in seeds]
        out = [f.result() for f in futs]
        for o in out:
            o["ticks"] = pool.stats["ticks"]
    return out
