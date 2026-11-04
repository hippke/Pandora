import numpy as np
from numba import jit


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def resample(arr, factor):
    out_samples = int(np.ceil(len(arr) / factor))
    out_arr = np.ones(out_samples)
    for idx in range(out_samples):
        start = idx * factor
        end = start + factor
        if end > len(arr):
            end = len(arr)
        out_arr[idx] = np.mean(arr[start:end])
    return out_arr


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ld_convert(q1, q2):
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return u1, u2

@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def ld_invert(u1, u2):
    if u1 + u2 == 0:
        return 0.0, 0.5
    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2