"""Pandora regression tests.

    python run_tests.py              # light-curve cases + unit tests; exit 0 = pass
    python run_tests.py -v           # also print per-case timing
    python run_tests.py --lc         # light-curve cases only
    python run_tests.py --unit       # unit tests only (unit_tests.py)
    python run_tests.py NAME [NAME]  # selected light-curve cases only
    python run_tests.py --sampling   # in addition: the fit API / sampler tests (test_sampling.py, ~3-5 min)
    pytest run_tests.py unit_tests.py   # same checks under pytest (if installed)

Each light-curve case is checked for (1) unchanged case definition, (2)
physics invariants, and (3) agreement of the light-curve fingerprint with
reference.json. See README.md.
"""
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")  # silence video() progress bars (read at import)

import cases as C

HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = os.path.join(HERE, "reference.json")


def _param_diff(ref, new):
    diffs = []
    for k in sorted(set(ref) | set(new)):
        a, b = ref.get(k), new.get(k)
        if isinstance(a, float) and isinstance(b, float):
            same = a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b))
        else:
            same = a == b
        if not same:
            diffs.append(f"param {k}: ref={a!r} now={b!r}")
    return diffs


def check_case(c, ref_entry):
    """Return (errors, runtime_seconds)."""
    errors = []
    params = {k: v for k, v in c.items() if k != "description"}
    errors += _param_diff(ref_entry["params"], params)
    if errors:
        return ["case definition differs from reference (edit cases.py back, or "
                "regenerate deliberately):"] + errors, 0.0
    t0 = time.perf_counter()
    r = C.run_case(c)
    dt = time.perf_counter() - t0
    errors += ["invariant: " + e for e in C.invariants(c, r)]
    errors += ["fingerprint: " + e for e in
               C.compare_fingerprints(ref_entry["fingerprint"], C.fingerprint(c, r))]
    return errors, dt


def run(selected=None, verbose=False):
    with open(REF_FILE) as fh:
        ref = json.load(fh)
    cases = C.build_cases()
    names = {c["name"] for c in cases}
    missing = set(ref["cases"]) - names
    failures = 0
    t_start = time.perf_counter()
    for c in cases:
        if selected and c["name"] not in selected:
            continue
        if c["name"] not in ref["cases"]:
            print(f"NEW   {c['name']}: not in reference.json (run generate_reference.py)")
            failures += 1
            continue
        errors, dt = check_case(c, ref["cases"][c["name"]])
        status = "PASS" if not errors else "FAIL"
        timing = f"  ({dt * 1e3:.1f} ms)" if verbose else ""
        print(f"{status}  {c['name']}{timing}")
        for e in errors[:12]:
            print(f"        {e}")
        if len(errors) > 12:
            print(f"        ... {len(errors) - 12} more")
        failures += bool(errors)
    for m in sorted(missing):
        print(f"GONE  {m}: in reference.json but no longer defined in cases.py")
        failures += 1
    total = time.perf_counter() - t_start
    print(f"\n{'OK' if failures == 0 else 'FAILED'}: {failures} failing case(s); "
          f"total {total:.2f} s (reference from Pandora {ref['pandora_commit']}, "
          f"generated {ref['generated']})")
    return failures


def test_regression():  # pytest entry point
    assert run() == 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    failures = 0
    if "--unit" not in sys.argv:
        failures += run(selected=set(args) or None, verbose=verbose)
    if "--lc" not in sys.argv and not args:
        import unit_tests
        print()
        failures += unit_tests.run(verbose=verbose)
    if "--sampling" in sys.argv:
        import subprocess
        print("\nfit API / sampler tests (test_sampling.py):", flush=True)
        failures += subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "test_sampling.py")]) != 0
    sys.exit(1 if failures else 0)
