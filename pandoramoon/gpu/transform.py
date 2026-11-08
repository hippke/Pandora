"""The prior transform of NestedSampler (sampling._theta) on the GPU (kernel `transform` in model.cu, FP64)."""
import numpy as np

from pandoramoon.gpu import _cuda


def pack(D):
    """(IS int32, DS float64) from a sampling._Model data tuple D (see the comment above sampling._theta)"""
    nat_p, nat_c, nat_k, nat_par = (np.asarray(D[i]) for i in (11, 12, 13, 14))
    nk = len(nat_p)
    ld_c = np.asarray(D[16], dtype=np.int64)
    mp_c = np.asarray(D[19], dtype=np.int64)
    CDF = np.asarray(D[20], dtype=np.float64)
    ld_c2 = list(ld_c[:2]) + [0] * (2 - min(2, len(ld_c)))
    mp_c3 = list(mp_c[:3]) + [0] * (3 - min(3, len(mp_c)))
    IS = np.array([nk, int(D[15]), int(D[18]), int(D[22]), *ld_c2, *mp_c3, len(CDF),
                   *nat_p, *nat_c, *nat_k], dtype=np.int32)
    ld_par = np.zeros(6)
    ld_par[:len(D[17])] = D[17]
    mp_f = np.zeros(6)
    mp_f[:len(D[21])] = D[21]
    pm_par = np.zeros(4)
    pm_par[:len(D[23])] = D[23]
    DS = np.concatenate([np.asarray(D[10], dtype=np.float64), np.asarray(nat_par, dtype=np.float64).ravel(), ld_par,
                         mp_f, pm_par, CDF if len(CDF) else np.zeros(1)])
    return IS, DS


class GPUTransform:
    """unit cube -> (theta (n, 20), log weight (n,)) for a NestedSampler model (sampler.models["moon"])"""

    def __init__(self, model):
        from numba import cuda
        self.ndim = int(model.ndim)
        IS, DS = pack(model.D)
        self.dIS = cuda.to_device(IS)
        self.dDS = cuda.to_device(DS)
        self.kernel = _cuda.module(1).get_kernel("transform")

    def device(self, dU, n, dTH, dLW, dACT=None):
        _cuda.launch(self.kernel, (n + 127) // 128, 128, _cuda.ptr(dU), np.int32(n), np.int32(self.ndim),
                     _cuda.ptr(self.dIS), _cuda.ptr(self.dDS), _cuda.ptr(dTH), _cuda.ptr(dLW), _cuda.ptr(dACT))

    def __call__(self, U):
        from numba import cuda
        U = np.ascontiguousarray(U, dtype=np.float64)
        n = U.shape[0]
        dU = cuda.to_device(U)
        dTH = cuda.device_array((n, 20), np.float64)
        dLW = cuda.device_array(n, np.float64)
        self.device(dU, n, dTH, dLW)
        cuda.synchronize()
        return dTH.copy_to_host(), dLW.copy_to_host()
