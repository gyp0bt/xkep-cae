"""ContactGeometry Strategy サブパッケージ.

接触幾何の検出・ギャップ計算・制約ヤコビアン構築。
"""

from xkep_cae.contact.geometry.strategy import (
    ContactGeometryInput,
    ContactGeometryOutput,
    LineToLineGaussProcess,
    MortarSegmentProcess,
    PointToPointProcess,
)

__all__ = [
    "PointToPointProcess",
    "LineToLineGaussProcess",
    "MortarSegmentProcess",
    "ContactGeometryInput",
    "ContactGeometryOutput",
]
