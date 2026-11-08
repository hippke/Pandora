"""Optional GPU backend (FP32, NVIDIA): GPU_PLAN.md / GPU_MODEL.md in the development repository.

Imported only on request (it needs the `gpu` extra: cuda-core, cuda-bindings, nvidia-cuda-nvrtc, numba-cuda).
"""
from pandoramoon.gpu._cuda import available
from pandoramoon.gpu.likelihood import GPULikelihood

__all__ = ["available", "GPULikelihood"]
