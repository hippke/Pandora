"""Batched FP32 planet+moon log-likelihood on the GPU (GPU_PLAN.md phase 2; model: model.cu, GPU_MODEL.md).

    from pandoramoon.gpu import GPULikelihood
    G = GPULikelihood(like, t0_ref=t0_bary)       # like: a pandoramoon.Likelihood (as the sampler uses it)
    logl = G(theta)                                # theta: (n, 20) Pandora parameters (pandora_loglike's order)

Scope (anything else raises ValueError; the CPU path handles it): a circular planet orbit (ecc_bary = 0), a
moon orbit with ecc_moon <= 0.9, sorted time, no occultation cache, occult_small_threshold = 0 (exact moon transit),
supersampled time as Likelihood stores it (len(time) = supersampling_factor x cadences), r_planet and r_moon
inside the occultation tables' k-range.
"""
import numpy as np

from pandoramoon.gpu import _cuda

# k-ranges of the interpolant tables (occult.LIMB_TAB_*, INT_TAB_*): the GPU has no per-call exact build
K_MIN = 5e-4
K_MAX = 0.25
ECC_MOON_MAX = 0.9


def _two_float(x):
    hi = x.astype(np.float32)
    lo = (x - hi.astype(np.float64)).astype(np.float32)
    return hi, lo


def _block_tables(gp, F, K, irs, e, W, v):
    """the tables of gp_pass for d > 2 states (model.cu, GP_D > 2), from pandoramoon.gp's tables (z coordinates,
    z_0 = h . m): back to the blocks' coordinates m = T^-1 z (F block-diagonal: per distinct step the SHO blocks'
    2 x 2 transitions and the linear baseline's dt; -1 at segment starts), the gains, 1 / sqrt(S), e, the tails
    (T' W T, T' v), and the products of the maps A_c = (I - K_c h') F_c over a grid of C cadences (C ~ n / 64: two
    chunks per lane when a call spans all cadences)"""
    from pandoramoon.gp import chunk_maps
    kind = gp.blocks()[0]
    nsho = int(np.sum(kind == 0))
    base = int(np.sum(kind == 1) + 2 * np.sum(kind == 2))
    n, d = K.shape
    h = np.zeros(d)
    h[0:2 * nsho:2] = 1.0
    if base:
        h[2 * nsho] = 1.0
    T = np.eye(d)
    T[0] = h
    Ti = np.eye(d)
    Ti[0] = -h
    Ti[0, 0] = 1.0
    Fm = np.einsum("ij,njk,kl->nil", Ti, F.reshape(n, d, d), T)
    Km = K @ Ti.T
    rows = [Fm[:, 2 * j:2 * j + 2, 2 * j:2 * j + 2].reshape(n, 4) for j in range(nsho)]
    rows.append(Fm[:, 2 * nsho, 2 * nsho + 1][:, None] if base == 2 else np.zeros((n, 1)))
    R = np.concatenate(rows, axis=1)
    reset = ~np.any(Fm.reshape(n, -1) != 0, axis=1)
    # distinct steps (the CPU's time_table groups the steps, so equal rows are exactly equal)
    _u, first, inv = np.unique(R, axis=0, return_index=True, return_inverse=True)
    Tt = R[first]
    step = np.where(reset, -1, inv.ravel()).astype(np.int32)
    iu = np.triu_indices(d)
    Wf = np.zeros((n, d, d))
    Wf[:, iu[0], iu[1]] = W
    Wf[:, iu[1], iu[0]] = W
    Wm = np.einsum("ji,njk,kl->nil", T, Wf, T)[:, iu[0], iu[1]]
    vm = v @ T
    C = max(2, n // 64)
    return dict(nsho=nsho, base=base, C=C, step=np.ascontiguousarray(step), T=Tt,
                kie=np.concatenate([Km, irs[:, None], e[:, None]], axis=1),
                w=np.concatenate([Wm, vm], axis=1), M=chunk_maps(np.ascontiguousarray(Fm), np.ascontiguousarray(Km),
                                                                h, C).reshape(-1, d * d))


class GPULikelihood:
    def __init__(self, like, t0_ref, warps=4, minb=8, extra=(), eccentric=True):
        from numba import cuda
        if like.cache is not None:
            raise ValueError("GPULikelihood: the occultation cache is not supported (pass cache=None)")
        if not like.time_sorted:
            raise ValueError("GPULikelihood: time must be ascending")
        if like.occult_small_threshold != 0:
            raise ValueError("GPULikelihood: occult_small_threshold must be 0 (exact moon transit)")
        ss = int(like.supersampling_factor)
        t = np.asarray(like.time, dtype=np.float64)
        nc = len(like.inv_var)
        if len(t) != ss * nc:
            raise ValueError("GPULikelihood: len(time) must be supersampling_factor x the number of cadences")
        self.like = like
        self.ss = ss
        self.nc = nc
        self.warps = int(warps)
        self.t0_ref = float(t0_ref)
        ed = float(like.epoch_distance)
        self.ed = ed
        tc = t.reshape(nc, ss).mean(axis=1)
        n_abs = np.floor((tc - self.t0_ref) / ed + 0.5).astype(np.int64)
        ue = np.unique(n_abs)
        ep = np.searchsorted(ue, n_abs).astype(np.int32)
        te = np.array([0.5 * (tc[ep == i].min() + tc[ep == i].max()) for i in range(len(ue))])
        self.te = te
        self.n_epoch = ue
        dt = t - np.repeat(te[ep], ss)
        dtr = dt.reshape(nc, ss)
        self.hspan = np.float32(np.abs(dtr - dtr[:, ss // 2:ss // 2 + 1]).max() * (1 + 1e-6) + 1e-12)
        ehi, elo = _two_float(te - te[0])
        self.d = dict(
            DT=cuda.to_device(dt.astype(np.float32)),
            EP=cuda.to_device(ep),
            EDT=cuda.to_device((te - self.t0_ref - ed * ue).astype(np.float32)),
            EN=cuda.to_device(ue.astype(np.float32)),
            EHI=cuda.to_device(ehi), ELO=cuda.to_device(elo),
            IV=cuda.to_device(np.asarray(like.inv_var, dtype=np.float32)),
            LN=cuda.to_device((1.0 - np.asarray(like.data, dtype=np.float64)).astype(np.float32)))   # E = 1 - data
        # per-epoch windows (model.cu main_loop, GPU_LOG Round 8): the cadence range of each epoch (time ascending:
        # contiguous) and the FP64 prefix sums of the FP32 flat terms IV E^2 (the chi^2 of a cadence with d = 0)
        ec = np.searchsorted(ep, np.arange(len(ue) + 1)).astype(np.int32)
        iv32 = np.asarray(like.inv_var, dtype=np.float32)
        e32 = (1.0 - np.asarray(like.data, dtype=np.float64)).astype(np.float32)
        r32 = np.float32(0) - e32
        flat = ((iv32 * r32) * r32).astype(np.float64)            # as the kernel: (IV r) r in FP32
        pf = np.concatenate([[0.0], np.cumsum(flat)])
        self.windows = not any(x[0] == "NO_WINDOW" for x in extra)
        self.pft = float(pf[-1]) if self.windows else 0.0
        self.d.update(EC=cuda.to_device(ec), PF=cuda.to_device(pf))
        # eccentric=False: compiled without the eccentric-moon loop (about 5 % faster); ecc_moon must then be 0
        self.eccentric = bool(eccentric)
        # simultaneous GP (Likelihood(gp=...), opt-in): the GP build of model.cu (deficits per cadence, then gp_pass
        # with the FP32 tables of pandoramoon.gp.gp_setup)
        self.gp = getattr(like, "gp", None) is not None
        if self.gp:
            if not self.windows:
                raise ValueError("GPULikelihood: the GP needs the windowed cadence loop (no NO_WINDOW)")
            F, K, irs, e, W, v, H, c0, ld = like.gp_tables
            if len(irs) != nc:
                raise ValueError("GPULikelihood: GP tables do not match the cadences")
            self.gp_d = int(K.shape[1])            # states (2: one SHO term; general model: up to 8)
            f32 = lambda x: cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))   # noqa: E731
            if self.gp_d == 2:
                gw = np.zeros((nc, 2, 4), np.float32)
                gw[:, 0, :3] = W
                gw[:, 1, :2] = v
                ke = np.stack([K[:, 0], K[:, 1], irs, e], axis=1)     # one float4 load per cadence
                self.d.update(GF=f32(F), GKE=f32(ke), GW=f32(gw.reshape(nc * 2, 4)))
            else:
                tb = _block_tables(like.gp, F, K, irs, e, W, v)
                self.gp_chunk = tb["C"]
                self.d.update(GSTEP=cuda.to_device(tb["step"]), GT=f32(tb["T"]), GKIE=f32(tb["kie"]),
                              GW=f32(tb["w"]), GM=f32(tb["M"]))
                # GP_SCAN: the warp scan's cost per run in cadence steps (d x d maps; model.cu gp_use_span)
                gp_defs = (("GP_D", str(self.gp_d)), ("GP_NSHO", str(tb["nsho"])), ("GP_BASE", str(tb["base"])),
                           ("GP_SCAN", str(2 * (self.gp_d ** 2 + self.gp_d))))
            # gap skipping (gp.gp_horizon): the horizon per cadence, and per call the runs of windows (gp_runs)
            self.d["GH"] = cuda.to_device(np.minimum(np.asarray(H), nc).astype(np.int32))
            self.gp_c0, self.gp_ld = float(c0), float(ld)
            self._gdb = self._glh = self._grun = None
            extra = tuple(extra) + (("GP", "1"),) + (gp_defs if self.gp_d != 2 else ())
        self.mod = _cuda.module(ss, warps, minb, tuple(extra) + (() if self.eccentric else (("NO_ECC", "1"),)))
        a = self.mod.get_kernel("loglike").attributes
        self.regs, self.local_bytes = a.num_regs, a.local_size_bytes
        self.kernel = self.mod.get_kernel("loglike")
        self.kernel_fix = self.mod.get_kernel("limb_fix")
        self.kernel_gp = self.mod.get_kernel("gp_pass") if self.gp else None
        self.fix_blocks = 4 * cuda.get_current_device().MULTIPROCESSOR_COUNT
        self._q = None             # limb-eclipse queue (capacity qpercall() x the largest batch so far)
        self._pc = self._co = None  # per-call setup and coefficients for limb_fix
        self.dQN = cuda.to_device(np.zeros(1, np.int32))
        self.dDONE = cuda.to_device(np.zeros(1, np.uint32))
        self.LT, self.IT = _cuda.tables()

    def check_theta(self, theta):
        """the v1 scope for a batch of parameter vectors (raises ValueError)"""
        th = np.asarray(theta)
        if np.any(th[:, 8] != 0):
            raise ValueError("GPULikelihood: circular planet orbits only (ecc_bary = 0)")
        if not self.eccentric and np.any(th[:, 17] != 0):
            raise ValueError("GPULikelihood(eccentric=False): ecc_moon must be 0")
        if np.any(th[:, 17] > ECC_MOON_MAX):
            raise ValueError(f"GPULikelihood: ecc_moon <= {ECC_MOON_MAX} (validated range of the FP32 Kepler solver)")
        for col, name in ((5, "r_planet"), (12, "r_moon")):
            v = th[:, col]
            on = v > 0
            if np.any((v[on] < K_MIN) | (v[on] > K_MAX)):
                raise ValueError(f"GPULikelihood v1: {name} outside [{K_MIN}, {K_MAX}] (occultation tables)")

    def qpercall(self):
        """queue entries per call: an upper bound on the limb-eclipse cadences of one call"""
        return int(min(self.nc, 64 + 8 * len(self.te)))

    gp_budget = 512 * 2 ** 20        # bytes of the GP build's per-call buffers (gp_chunk_calls)

    def gp_chunk_calls(self):
        """GP build: the largest batch whose per-call buffers (deficits per cadence, epoch windows and runs, the
        limb-eclipse queue, set-up) stay within gp_budget bytes; larger batches are launched in parts"""
        ne = len(self.te)
        per = 4 * self.nc + 8 * ne + 16 * ne + 8 * self.qpercall() + 4 * 4 * 36 + 128
        return max(1024, int(self.gp_budget // per))

    def loglike_device(self, dTH, n, dOUT, dNLIMB=None, fix=True, main=True, dACT=None):
        """launch on device arrays: dTH (n, 20) float64 -> dOUT (n,) float64 (no sync). Two kernels: loglike
        (every cadence without a limb eclipse; those are queued) and limb_fix (the queued cadences). A call whose
        queue entries did not fit gets NaN (never observed: QPERCALL entries per call). The GP build launches
        large batches (e.g. the rejection init's 100 000 draws) in parts of gp_chunk_calls()."""
        if self.gp and main and fix and n > self.gp_chunk_calls():
            m = self.gp_chunk_calls()
            for i0 in range(0, n, m):
                i1 = min(n, i0 + m)
                self._loglike_device(dTH[i0:i1], i1 - i0, dOUT[i0:i1],
                                     None if dNLIMB is None else dNLIMB[i0:i1], True, True,
                                     None if dACT is None else dACT[i0:i1])
            return
        self._loglike_device(dTH, n, dOUT, dNLIMB, fix, main, dACT)

    def _loglike_device(self, dTH, n, dOUT, dNLIMB, fix, main, dACT):
        from numba import cuda
        d = self.d
        L = self.like
        if self._q is None or self._q.shape[0] < self.qpercall() * n:
            self._q = cuda.device_array((self.qpercall() * n, 2), np.int32)
        if self._pc is None or self._pc.shape[0] < n:
            self._pc = cuda.device_array((n, 32), np.float32)        # CallP (<= 128 bytes) per call
            self._co = cuda.device_array((n, 4, 36), np.float32)     # coefficient rows per call
        qcap = np.int32(self._q.shape[0])
        grid = (n + self.warps - 1) // self.warps
        gpa = ()
        if self.gp:
            ne = len(self.te)
            if self._gdb is None or self._gdb.shape[0] < n:
                self._gdb = cuda.device_array((n, self.nc), np.float32)
                self._glh = cuda.device_array((n, ne, 2), np.int32)
                self._grun = cuda.device_array((n, ne, 4), np.int32)
            gpa = (_cuda.ptr(self._gdb), _cuda.ptr(self._glh))
        if main:
            _cuda.launch(self.kernel, grid, 32 * self.warps,
                         _cuda.ptr(dTH), np.int32(n),
                         _cuda.ptr(d["DT"]), _cuda.ptr(d["EP"]), _cuda.ptr(d["EDT"]), _cuda.ptr(d["EN"]),
                         _cuda.ptr(d["EHI"]), _cuda.ptr(d["ELO"]), _cuda.ptr(d["IV"]), _cuda.ptr(d["LN"]),
                         np.int32(self.nc), np.float64(self.t0_ref), np.float64(self.te[0]), np.float64(self.ed),
                         np.float32(L.hill_sphere_threshold), np.float32(self.hspan),
                         _cuda.ptr(self.LT), _cuda.ptr(self.IT), _cuda.ptr(dOUT), _cuda.ptr(dNLIMB),
                         _cuda.ptr(self._q), _cuda.ptr(self.dQN), qcap, _cuda.ptr(self._pc), _cuda.ptr(self._co),
                         _cuda.ptr(dACT), np.int32(len(self.te)), _cuda.ptr(d["EC"]), _cuda.ptr(d["PF"]),
                         np.float64(self.pft), *gpa)
        if not fix:
            return
        _cuda.launch(self.kernel_fix, self.fix_blocks, 128,
                     _cuda.ptr(d["DT"]), _cuda.ptr(d["EP"]), _cuda.ptr(d["EDT"]), _cuda.ptr(d["EN"]),
                     _cuda.ptr(d["EHI"]), _cuda.ptr(d["ELO"]), _cuda.ptr(d["IV"]), _cuda.ptr(d["LN"]),
                     _cuda.ptr(dOUT), _cuda.ptr(self._q), _cuda.ptr(self.dQN), qcap, _cuda.ptr(self._pc),
                     _cuda.ptr(self._co), _cuda.ptr(self.dDONE),
                     *((gpa[0], np.int32(self.nc)) if self.gp else ()))
        if self.gp:
            tabs = ((_cuda.ptr(d["GF"]), _cuda.ptr(d["GKE"]), _cuda.ptr(d["GW"])) if self.gp_d == 2 else
                    (_cuda.ptr(d["GSTEP"]), _cuda.ptr(d["GT"]), _cuda.ptr(d["GKIE"]), _cuda.ptr(d["GW"]),
                     _cuda.ptr(d["GM"]), np.int32(self.gp_chunk)))
            _cuda.launch(self.kernel_gp, (n + 3) // 4, 128, np.int32(n), np.int32(self.nc), np.int32(len(self.te)),
                         gpa[1], gpa[0], _cuda.ptr(d["GH"]), _cuda.ptr(self._grun), *tabs,
                         np.float64(self.gp_c0), np.float64(self.gp_ld),
                         _cuda.ptr(dOUT), _cuda.ptr(dACT))

    def __call__(self, theta, check=True, return_nlimb=False):
        from numba import cuda
        th = np.ascontiguousarray(theta, dtype=np.float64)
        if th.ndim == 1:
            th = th[None, :]
        if check:
            self.check_theta(th)
        n = th.shape[0]
        dTH = cuda.to_device(th)
        dOUT = cuda.device_array(n, np.float64)
        dNL = cuda.device_array(n, np.int32) if return_nlimb else None
        self.loglike_device(dTH, n, dOUT, dNL)
        cuda.synchronize()
        out = dOUT.copy_to_host()
        if np.isnan(out).any():
            raise RuntimeError("GPULikelihood: limb-eclipse queue overflow (qpercall)")
        if return_nlimb:
            return out, dNL.copy_to_host()
        return out


def deficit(k, u1, u2, z2):
    """component test: the GPU deficit of one body at squared distances z2 (float32)"""
    from numba import cuda
    mod = _cuda.module(1)
    LT, IT = _cuda.tables()
    z2 = np.ascontiguousarray(z2, dtype=np.float32)
    dZ = cuda.to_device(z2)
    dO = cuda.device_array(len(z2), np.float32)
    _cuda.launch(mod.get_kernel("t_deficit"), 1, 32, np.float32(k), np.float32(u1), np.float32(u2), _cuda.ptr(dZ),
                 np.int32(len(z2)), _cuda.ptr(LT), _cuda.ptr(IT), _cuda.ptr(dO))
    cuda.synchronize()
    return dO.copy_to_host()


def er3(X):
    """component test: er = min(1, A(star & planet & moon) / A(star & moon)) at rows (xp, yp, rp, xm, ym, rm)"""
    from numba import cuda
    mod = _cuda.module(1)
    X = np.ascontiguousarray(X, dtype=np.float32)
    m = X.shape[0]
    dX = cuda.to_device(X.ravel())
    dO = cuda.device_array(m, np.float32)
    _cuda.launch(mod.get_kernel("t_er3"), (m + 127) // 128, 128, _cuda.ptr(dX), np.int32(m), _cuda.ptr(dO))
    cuda.synchronize()
    return dO.copy_to_host()


def eclipse_dm(X):
    """component test: the moon deficit after the planet-moon overlap at rows (xp, yp, xm, ym, kp, km, dm, u1, u2)"""
    from numba import cuda
    mod = _cuda.module(1)
    X = np.ascontiguousarray(X, dtype=np.float32)
    m = X.shape[0]
    dX = cuda.to_device(X.ravel())
    dO = cuda.device_array(m, np.float32)
    _cuda.launch(mod.get_kernel("t_eclipse_dm"), (m + 127) // 128, 128, _cuda.ptr(dX), np.int32(m), _cuda.ptr(dO))
    cuda.synchronize()
    return dO.copy_to_host()
