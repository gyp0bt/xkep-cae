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
from xkep_cae.contact.geometry.strategy import (
    _create_contact_geometry_strategy as create_contact_geometry_strategy,
)

__all__ = [
    "PointToPointProcess",
    "LineToLineGaussProcess",
    "MortarSegmentProcess",
    "ContactGeometryInput",
    "ContactGeometryOutput",
    "create_contact_geometry_strategy",
]
