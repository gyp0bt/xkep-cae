"""暫定 re-export（未移行モジュール）."""

import importlib as _il

_m = _il.import_module("xkep_cae_deprecated.materials")
for _k in dir(_m):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_m, _k)
