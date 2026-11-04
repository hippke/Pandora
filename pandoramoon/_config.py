"""Configuration files of Pandora and two import-time helpers.

- physics.cfg: physical constants (pandoramoon.constants);
- settings.cfg: default settings and tuning values of the model, the sampler and the search (settings());
- the numba cache guard: numba's on-disk cache (cache=True) notices only changes of a function's own source file,
  not of constants or data tables it reads from other files. Pandora hashes all of its source, configuration and
  table files at import and clears its numba cache once if any of them changed (one recompilation, then cached
  again).
- the compile notice: a one-line message when numba starts compiling Pandora code (first use after installation
  or an update takes up to a few minutes; later runs load the cache). Silence it with PANDORA_QUIET=1.
"""
import configparser
import glob
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def read_cfg(name):
    """{section: {key: value}} of a .cfg file next to this module; numbers are converted (int if integral text)"""
    cp = configparser.ConfigParser()
    cp.optionxform = str                                  # keep the case of the keys (R_sun, M_jup)
    path = os.path.join(HERE, name)
    if not cp.read(path):
        raise FileNotFoundError(f"pandoramoon: {path} is missing")
    out = {}
    for sec in cp.sections():
        d = {}
        for k, v in cp[sec].items():
            t = v.strip()
            try:
                d[k] = int(t) if t.lstrip("-").isdigit() else float(t)
            except ValueError:
                d[k] = {"true": True, "false": False}.get(t.lower(), t)
        out[sec] = d
    return out


def _source_hash():
    h = hashlib.sha1()
    for pattern in ("*.py", "*.cfg", "*.csv"):
        for f in sorted(glob.glob(os.path.join(HERE, pattern))):
            h.update(os.path.basename(f).encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def cache_guard():
    """clear Pandora's numba cache (pandoramoon/__pycache__/*.nbi, *.nbc and NUMBA_CACHE_DIR's pandoramoon entries)
    once if any source, configuration or table file changed since it was written"""
    try:
        digest = _source_hash()
        dirs = [os.path.join(HERE, "__pycache__")]
        env = os.environ.get("NUMBA_CACHE_DIR")
        if env:
            dirs += glob.glob(os.path.join(env, "*pandoramoon*"))
        marker = os.path.join(dirs[0], "pandora_source.sha1")
        old = open(marker).read().strip() if os.path.exists(marker) else None
        if old == digest:
            return
        for d in dirs:
            for f in glob.glob(os.path.join(d, "*.nbi")) + glob.glob(os.path.join(d, "*.nbc")):
                try:
                    os.remove(f)
                except OSError:
                    pass
        os.makedirs(dirs[0], exist_ok=True)
        with open(marker, "w") as fh:
            fh.write(digest)
    except OSError:
        pass                                              # read-only installation: numba uses its own cache dir


_SETTINGS = {}


def settings():
    """settings.cfg as {section: {key: value}} (read once)"""
    if not _SETTINGS:
        _SETTINGS.update(read_cfg("settings.cfg"))
    return _SETTINGS


_last_notice = [0.0]


def install_compile_notice():
    if os.environ.get("PANDORA_QUIET"):
        return
    try:
        from numba.core import event
    except ImportError:
        return

    class _Notice(event.Listener):
        def on_start(self, ev):
            try:
                mod = ev.data["dispatcher"].py_func.__module__
            except Exception:
                return
            now = time.monotonic()
            if mod.startswith("pandoramoon") and now - _last_notice[0] > 60.0:
                _last_notice[0] = now
                print("Pandora: compiling numerical code (first use after installation or an update can take a few "
                      "minutes; the result is cached) ...", file=sys.stderr, flush=True)

        def on_end(self, ev):
            pass

    event.register("numba:compile", _Notice())
