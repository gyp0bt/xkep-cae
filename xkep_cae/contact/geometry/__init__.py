"""ContactGeometry Strategy サブパッケージ.

接触幾何の検出・ギャップ計算・制約ヤコビアン構築。
"""

from xkep_cae.contact.geometry._legacy import (
    ClosestPointResult,
    build_contact_frame,
    build_contact_frame_batch,
    closest_point_segments,
    closest_point_segments_batch,
    compute_gap,
    compute_st_jacobian,
)
from xkep_cae.contact.geometry.strategy import (
    ContactGeometryInput,
    ContactGeometryOutput,
    LineToLineGaussProcess,
    MortarSegmentProcess,
    PointToPointProcess,
)

__all__ = [
    # Legacy geometry functions
    "ClosestPointResult",
    "closest_point_segments",
    "closest_point_segments_batch",
    "compute_gap",
    "compute_st_jacobian",
    "build_contact_frame",
    "build_contact_frame_batch",
    # Strategy processes
    "PointToPointProcess",
    "LineToLineGaussProcess",
    "MortarSegmentProcess",
    "ContactGeometryInput",
    "ContactGeometryOutput",
]
