"""初期貫入検出・座標調整（プライベート）.

xkep_cae_deprecated/contact/initial_penetration.py からの移植。
geometry 関数も同梱し、deprecated 依存を完全除去。
ContactPair は duck typing（.nodes_a, .nodes_b, .radius_a, .radius_b,
.core_radius_a, .core_radius_b 属性を持つ任意のオブジェクト）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _ClosestPointResult:
    """最近接点計算の結果."""

    s: float
    t: float
    point_a: np.ndarray
    point_b: np.ndarray
    distance: float
    normal: np.ndarray
    parallel: bool = False


def _closest_point_segments(
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    *,
    tol_parallel: float = 1e-10,
) -> _ClosestPointResult:
    """2線分間の最近接点を計算する."""
    dA = xA1 - xA0
    dB = xB1 - xB0
    w0 = xA0 - xB0

    a = float(dA @ dA)
    b = float(dA @ dB)
    c = float(dB @ dB)
    d = float(dA @ w0)
    e = float(dB @ w0)

    det = a * c - b * b
    is_parallel = det < tol_parallel * max(a * c, 1e-30)

    if is_parallel:
        s = 0.0
        t = np.clip(e / c, 0.0, 1.0) if c > tol_parallel else 0.0
    else:
        s_unc = (b * e - c * d) / det
        t_unc = (a * e - b * d) / det

        s = np.clip(s_unc, 0.0, 1.0)
        t = np.clip(t_unc, 0.0, 1.0)

        if s_unc < 0.0 or s_unc > 1.0:
            t = np.clip((b * s + e) / c, 0.0, 1.0) if c > tol_parallel else 0.0
        if t_unc < 0.0 or t_unc > 1.0:
            s = np.clip((b * t - d) / a, 0.0, 1.0) if a > tol_parallel else 0.0
            t = np.clip((b * s + e) / c, 0.0, 1.0) if c > tol_parallel else 0.0

    point_a = xA0 + s * dA
    point_b = xB0 + t * dB
    diff = point_a - point_b
    distance = float(np.linalg.norm(diff))

    if distance > 1e-30:
        normal = diff / distance
    else:
        normal = np.array([0.0, 0.0, 1.0])

    return _ClosestPointResult(
        s=s,
        t=t,
        point_a=point_a,
        point_b=point_b,
        distance=distance,
        normal=normal,
        parallel=is_parallel,
    )


def _compute_gap(distance: float, radius_a: float, radius_b: float) -> float:
    """ギャップを計算する: g = distance - (r_a + r_b)."""
    return distance - (radius_a + radius_b)


def _check_initial_penetration(
    pairs: Sequence[object],
    node_coords: np.ndarray,
    coating_stiffness: float = 0.0,
) -> int:
    """初期貫入をチェックする.

    pairs は duck typing: .nodes_a, .nodes_b, .radius_a, .radius_b,
    .core_radius_a, .core_radius_b 属性を持つオブジェクト。
    """
    coords = np.asarray(node_coords, dtype=float)
    n_pen = 0
    use_coating = coating_stiffness > 0.0

    for pair in pairs:
        xA0 = coords[pair.nodes_a[0]]
        xA1 = coords[pair.nodes_a[1]]
        xB0 = coords[pair.nodes_b[0]]
        xB1 = coords[pair.nodes_b[1]]

        result = _closest_point_segments(xA0, xA1, xB0, xB1)
        if use_coating:
            gap = _compute_gap(result.distance, pair.core_radius_a, pair.core_radius_b)
        else:
            gap = _compute_gap(result.distance, pair.radius_a, pair.radius_b)

        if gap < 0.0:
            n_pen += 1

    return n_pen


def _adjust_initial_positions(
    pairs: Sequence[object],
    node_coords: np.ndarray,
    position_tolerance: float = 0.0,
    *,
    max_iterations: int = 10,
) -> tuple[np.ndarray, int, int]:
    """初期節点座標を調整して貫入解消・ギャップ閉鎖を行う."""
    coords = np.asarray(node_coords, dtype=float)
    n_pen_fixed = 0
    n_gap_closed = 0

    for _iteration in range(max_iterations):
        corrections = np.zeros_like(coords)
        correction_count = np.zeros(len(coords), dtype=int)
        any_adjusted = False

        for pair in pairs:
            nA0, nA1 = int(pair.nodes_a[0]), int(pair.nodes_a[1])
            nB0, nB1 = int(pair.nodes_b[0]), int(pair.nodes_b[1])
            xA0, xA1 = coords[nA0], coords[nA1]
            xB0, xB1 = coords[nB0], coords[nB1]

            result = _closest_point_segments(xA0, xA1, xB0, xB1)
            gap = _compute_gap(result.distance, pair.radius_a, pair.radius_b)

            if gap >= position_tolerance:
                continue

            adjust_dist = -gap
            half_adjust = adjust_dist / 2.0

            normal = result.normal
            if float(np.linalg.norm(normal)) < 1e-15:
                continue

            for nid in [nA0, nA1]:
                corrections[nid] -= half_adjust * normal
                correction_count[nid] += 1
            for nid in [nB0, nB1]:
                corrections[nid] += half_adjust * normal
                correction_count[nid] += 1

            any_adjusted = True
            if gap < 0:
                n_pen_fixed += 1
            else:
                n_gap_closed += 1

        if not any_adjusted:
            break

        mask = correction_count > 0
        coords[mask] += corrections[mask] / correction_count[mask, np.newaxis]

    return coords, n_pen_fixed, n_gap_closed
