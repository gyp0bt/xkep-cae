"""output パッケージ — 新プロセス + deprecated 互換.

新パッケージのプロセス:
  - ExportProcess: CSV/JSON エクスポート
  - BeamRenderProcess: 2D投影レンダリング

deprecated 互換: tests/test_output.py 等から使われる旧 API は
__getattr__ 経由で xkep_cae_deprecated.output から遅延ロードする。
"""

from xkep_cae.output.export import ExportConfig, ExportProcess, ExportResult
from xkep_cae.output.render import BeamRenderProcess, RenderConfig, RenderResult

__all__ = [
    "ExportConfig",
    "ExportProcess",
    "ExportResult",
    "BeamRenderProcess",
    "RenderConfig",
    "RenderResult",
]


def __getattr__(name: str):
    """deprecated 互換: 旧 API を遅延ロード."""
    import importlib

    _m = importlib.import_module("xkep_cae_deprecated.output")
    if hasattr(_m, name):
        return getattr(_m, name)
    raise AttributeError(f"module 'xkep_cae.output' has no attribute {name!r}")
