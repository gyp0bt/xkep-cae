"""初期貫入検出・座標調整 — 純関数ユーティリティ.

ContactManager.check_initial_penetration / adjust_initial_positions を
純関数化したもの。pair.py の責務軽量化（status-170）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from __xkep_cae_deprecated.contact.geometry import closest_point_segments, compute_gap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from __xkep_cae_deprecated.contact.pair import ContactPair


def check_initial_penetration(
    pairs: Sequence[ContactPair],
    node_coords: np.ndarray,
    coating_stiffness: float = 0.0,
) -> int:
    """初期貫入をチェックする.

    被膜モデル有効時: 芯線半径ベースでギャップを計算。
    被膜モデル無効時: 被膜込み半径（total radius）でギャップを計算。

    Args:
        pairs: 接触ペアリスト
        node_coords: 初期節点座標 (n_nodes, 3)
        coating_stiffness: 被膜剛性（>0 で被膜モデル有効）

    Returns:
        初期貫入ペア数
    """
    coords = np.asarray(node_coords, dtype=float)
    n_pen = 0
    use_coating = coating_stiffness > 0.0

    for pair in pairs:
        xA0 = coords[pair.nodes_a[0]]
        xA1 = coords[pair.nodes_a[1]]
        xB0 = coords[pair.nodes_b[0]]
        xB1 = coords[pair.nodes_b[1]]

        result = closest_point_segments(xA0, xA1, xB0, xB1)
        if use_coating:
            gap = compute_gap(result.distance, pair.core_radius_a, pair.core_radius_b)
        else:
            gap = compute_gap(result.distance, pair.radius_a, pair.radius_b)

        if gap < 0.0:
            n_pen += 1

    return n_pen


def adjust_initial_positions(
    pairs: Sequence[ContactPair],
    node_coords: np.ndarray,
    position_tolerance: float = 0.0,
    *,
    max_iterations: int = 10,
) -> tuple[np.ndarray, int, int]:
    """初期節点座標を調整して貫入解消・ギャップ閉鎖を行う.

    各接触ペアのギャップを検査し:
      - gap < 0（貫入）: 節点を法線方向に離してgap = 0にする
      - 0 <= gap < position_tolerance: 節点を法線方向に近づけてgap = 0にする

    Args:
        pairs: 接触ペアリスト
        node_coords: 初期節点座標 (n_nodes, 3)。**破壊的に変更される**。
        position_tolerance: ギャップ閉鎖の閾値 [m]。
        max_iterations: 調整反復の上限

    Returns:
        (adjusted_coords, n_penetration_fixed, n_gap_closed)
    """
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

            result = closest_point_segments(xA0, xA1, xB0, xB1)
            gap = compute_gap(result.distance, pair.radius_a, pair.radius_b)

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
