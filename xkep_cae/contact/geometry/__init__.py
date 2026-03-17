"""ContactGeometry Strategy サブパッケージ.

接触幾何の検出・ギャップ計算・制約ヤコビアン構築。

Note: _contact_pair → geometry._compute → geometry/__init__ → geometry.strategy
      → _contact_pair の循環参照を避けるため、遅延インポートを使用。
"""

__all__ = [
    "PointToPointProcess",
    "LineToLineGaussProcess",
    "MortarSegmentProcess",
    "ContactGeometryInput",
    "ContactGeometryOutput",
]


def __getattr__(name: str):
    if name in __all__:
        from xkep_cae.contact.geometry.strategy import (
            ContactGeometryInput,
            ContactGeometryOutput,
            LineToLineGaussProcess,
            MortarSegmentProcess,
            PointToPointProcess,
        )

        _exports = {
            "PointToPointProcess": PointToPointProcess,
            "LineToLineGaussProcess": LineToLineGaussProcess,
            "MortarSegmentProcess": MortarSegmentProcess,
            "ContactGeometryInput": ContactGeometryInput,
            "ContactGeometryOutput": ContactGeometryOutput,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
