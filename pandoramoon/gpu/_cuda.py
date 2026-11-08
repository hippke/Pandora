"""Run-time compilation (NVRTC through cuda.core) and launching of the kernels in model.cu.

Optional dependencies (the `gpu` extra): cuda-core, cuda-bindings, nvidia-cuda-nvrtc and numba-cuda (device arrays).
"""
import math
import os

import numpy as np

_MODULES = {}
_SRC = None


def available():
    """True if an NVIDIA GPU and the CUDA Python packages are usable"""
    try:
        import cuda.core  # noqa: F401
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def _source():
    global _SRC
    if _SRC is None:
        with open(os.path.join(os.path.dirname(__file__), "model.cu")) as f:
            _SRC = f.read()
    return _SRC


def module(ss, warps=4, minb=8, extra=()):
    """compiled model.cu for supersampling factor ss (cached per process)"""
    key = (int(ss), int(warps), int(minb), tuple(extra))
    if key in _MODULES:
        return _MODULES[key]
    import cuda.core as cc
    from pandoramoon import occult as O
    from pandoramoon.constants import G, DAY
    dev = cc.Device()
    dev.set_current()
    cap = dev.compute_capability
    defs = [("SS", str(int(ss))), ("WARPS", str(int(warps))), ("MINB", str(int(minb))),
            ("LY0", f"{float(np.float32(O.LIMB_TAB_Y0))!r}f"), ("LSC", f"{float(np.float32(O.LIMB_TAB_SCALE))!r}f"),
            ("IY0", f"{float(np.float32(O.INT_TAB_Y0))!r}f"), ("ISC", f"{float(np.float32(O.INT_TAB_SCALE))!r}f"),
            ("GDAY", f"{G * DAY ** 2 / (4 * math.pi ** 2)!r}f")] + list(extra)
    opts = cc.ProgramOptions(arch=f"sm_{cap.major}{cap.minor}", use_fast_math=True, define_macro=defs, std="c++17")
    mod = cc.Program(_source(), code_type="c++", options=opts).compile("cubin")
    _MODULES[key] = mod
    return mod


def ptr(a):
    """device pointer of a numba-cuda device array (or 0 for None)"""
    if a is None:
        return 0
    return a.__cuda_array_interface__["data"][0]


def launch(kernel, grid, block, *args):
    import cuda.core as cc
    cc.launch(cc.LEGACY_DEFAULT_STREAM, cc.LaunchConfig(grid=grid, block=block), kernel, *args)


_TABLES = None


def tables():
    """(limb table, interior table) on the device, FP32"""
    global _TABLES
    if _TABLES is None:
        from numba import cuda
        from pandoramoon import occult as O
        _TABLES = (cuda.to_device(np.ascontiguousarray(O.LIMB_TAB, dtype=np.float32)),
                   cuda.to_device(np.ascontiguousarray(O.INT_TAB, dtype=np.float32)))
    return _TABLES
