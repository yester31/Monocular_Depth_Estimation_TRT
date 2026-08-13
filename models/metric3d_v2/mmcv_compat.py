"""Make `import mmcv` work well enough to load Metric3D.

Metric3D's `mono/utils/comm.py` does this at module scope, so it runs before
any model can be built:

    from mmcv.utils import collect_env as collect_base_env
    from mmcv.utils import get_git_hash

That is the **mmcv 1.x** layout. mmcv 2.x moved both functions into mmengine
and `mmcv.utils` no longer exists, while mmcv 1.x has no Windows wheel and
building it from source needs a CUDA toolchain. Metric3D's own
`requirements_v2.txt` cannot be installed either: it pins `torch==2.0.1`,
which resolves against nothing usable here.

Neither function does any work that matters. `collect_env` gathers version
strings for a log line and `get_git_hash` stamps a run; nothing downstream
reads either. So this provides them, preferring mmengine's real
implementations when mmengine happens to be installed and falling back to
stubs when it is not.

The other three `mmcv` imports in Metric3D are all `from mmcv.utils import
Config` inside `if __name__ == '__main__'` blocks that already fall back to
mmengine, so they never run from here.

Import this **before** `Metric3D.hubconf`:

    import mmcv_compat  # noqa: F401  (must precede the hubconf import)
    from Metric3D.hubconf import *
"""

import sys
import types


def install():
    """Register a minimal `mmcv` / `mmcv.utils` if the real one is unusable."""
    try:
        import mmcv.utils  # noqa: F401
        if hasattr(mmcv.utils, "collect_env"):
            return "real mmcv"
    except ImportError:
        pass

    collect_env, get_git_hash = None, None
    try:
        from mmengine.utils.dl_utils import collect_env as _ce
        from mmengine.utils import get_git_hash as _gh
        collect_env, get_git_hash = _ce, _gh
        source = "mmengine"
    except ImportError:
        collect_env = lambda: {}                       # noqa: E731
        get_git_hash = lambda *a, **k: "unknown"       # noqa: E731
        source = "stub"

    util = types.ModuleType("mmcv.utils")
    util.collect_env = collect_env
    util.get_git_hash = get_git_hash

    mmcv = sys.modules.get("mmcv") or types.ModuleType("mmcv")
    mmcv.utils = util
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.utils"] = util
    return source


_SOURCE = install()
print(f"[MDET] mmcv compatibility: {_SOURCE}")
