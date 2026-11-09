"""Tests of the GPU nested sampler (pandoramoon/gpu/sampler.cu, sampler.py) on targets with known evidence.

    python tests/gpu/test_sampler.py        # skipped (exit 0) without a GPU; about 1 min on an RTX 5060
"""
import sys

import numpy as np


def _check(D, R, n, k, two, sigma, seeds):
    from pandoramoon.gpu.sampler import gpu_runs, GaussEvaluator
    ev = GaussEvaluator(D, sigma, twomode=two)
    rs = gpu_runs(ev, D, R=R, nlive=n, k=k, seeds=seeds)
    lz = np.array([r["logz"] for r in rs]) - ev.logz_true
    sem = lz.std(ddof=1) / np.sqrt(R)
    q = np.mean([r["logzerr"] for r in rs])
    assert abs(lz.mean()) <= 3.5 * sem, ("bias", lz.mean(), sem)
    assert 0.6 <= lz.std(ddof=1) / q <= 1.6, ("scatter vs quoted", lz.std(ddof=1), q)
    assert abs(np.mean([r["insertion"]["z"] for r in rs])) < 1.0
    # nested-sampling invariant: dead logL non-decreasing (the final live block excluded)
    for r in rs[:4]:
        assert np.all(np.diff(r["dead_L"][:-n]) >= 0)
    extra = ""
    if two:
        fr = np.mean([np.mean(r["samples_u"][:, 0] < 0.5) for r in rs])
        assert abs(fr - 0.5) < 0.05, ("mode fraction", fr)
        extra = f", mode fraction {fr:.3f}"
    return f"lnZ - true {lz.mean():+.3f} +- {sem:.3f}, sd {lz.std(ddof=1):.3f} / quoted {q:.3f}{extra}"


def test_gauss_6d():
    return _check(6, 64, 500, 16, False, 0.05, range(201, 265))


def test_gauss_16d_k64():
    return _check(16, 32, 500, 64, False, 0.05, range(101, 133))


def test_twomode_4d():
    return _check(4, 64, 500, 32, True, 0.04, range(101, 165))


def test_reproducible():
    from pandoramoon.gpu.sampler import gpu_runs, GaussEvaluator
    ev = GaussEvaluator(6, 0.05)
    a = gpu_runs(ev, 6, R=4, nlive=200, k=8, seeds=[5, 6, 7, 8])
    b = gpu_runs(ev, 6, R=4, nlive=200, k=8, seeds=[5, 6, 7, 8])
    c = gpu_runs(ev, 6, R=4, nlive=200, k=8, seeds=[5, 6, 7, 8], overlap=False)
    assert all(x["logz"] == y["logz"] == z["logz"] for x, y, z in zip(a, b, c))
    assert all(np.array_equal(x["dead_u"], z["dead_u"]) for x, z in zip(a, c))
    return "identical runs (twice, and without the stream overlap)"


TESTS = [test_gauss_6d, test_gauss_16d_k64, test_twomode_4d, test_reproducible]

if __name__ == "__main__":
    from pandoramoon import gpu
    if not gpu.available():
        print("SKIP: no usable GPU")
        sys.exit(0)
    bad = 0
    for t in TESTS:
        try:
            print(f"PASS  {t.__name__}: {t()}", flush=True)
        except AssertionError as ex:
            bad += 1
            print(f"FAIL  {t.__name__}: {ex}", flush=True)
    sys.exit(1 if bad else 0)
