"""Broadphase 接触候補探索（KD-tree + AABB フィルタ）.

プライベートモジュール（C16 準拠）。

status-308: 空間ハッシュグリッドから scipy.spatial.cKDTree に置換。
内部ループのC実装化により大規模メッシュでの高速化を実現。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _broadphase_aabb(
    segments: list[tuple[np.ndarray, np.ndarray]],
    radii: np.ndarray | float = 0.0,
    *,
    margin: float = 0.0,
    cell_size: float | None = None,
) -> list[tuple[int, int]]:
    """KD-tree + AABB フィルタによる候補ペア探索.

    各セグメントの AABB 中心で KD-tree を構築し、
    query_pairs で近接ペアを高速に列挙した後、
    AABB 重複判定でフィルタリングする。

    Args:
        segments: [(x0, x1), ...] セグメント端点のリスト。各端点は (3,)
        radii: セグメントごとの断面半径。スカラーなら全セグメント共通
        margin: 追加探索マージン
        cell_size: 未使用（後方互換のため残置）

    Returns:
        候補ペア (i, j) のリスト（i < j）
    """
    n = len(segments)
    if n < 2:
        return []

    # radii をベクトル化
    if np.isscalar(radii):
        r_arr = np.full(n, float(radii))
    else:
        r_arr = np.asarray(radii, dtype=float)

    # AABB 計算（ベクトル化）
    x0_arr = np.array([s[0] for s in segments])  # (n, 3)
    x1_arr = np.array([s[1] for s in segments])  # (n, 3)
    expand = r_arr[:, None] + margin  # (n, 1)
    lo_all = np.minimum(x0_arr, x1_arr) - expand  # (n, 3)
    hi_all = np.maximum(x0_arr, x1_arr) + expand  # (n, 3)

    # AABB 中心と半対角長
    mid_all = 0.5 * (lo_all + hi_all)  # (n, 3)
    half_all = 0.5 * (hi_all - lo_all)  # (n, 3)
    half_diag = np.linalg.norm(half_all, axis=1)  # (n,)

    # KD-tree 構築 + 近接ペア探索
    # ペア (i,j) が AABB 重複する可能性があるのは
    # ||mid_i - mid_j|| <= half_diag_i + half_diag_j <= 2 * max(half_diag)
    max_half_diag = float(np.max(half_diag))
    tree = cKDTree(mid_all)
    kd_pairs = tree.query_pairs(r=2.0 * max_half_diag, output_type="ndarray")

    if len(kd_pairs) == 0:
        return []

    # バッチ AABB 重複判定（ベクトル化）
    pi, pj = kd_pairs[:, 0], kd_pairs[:, 1]
    overlap = np.all(lo_all[pi] <= hi_all[pj], axis=1) & np.all(lo_all[pj] <= hi_all[pi], axis=1)
    return [(int(kd_pairs[k, 0]), int(kd_pairs[k, 1])) for k in np.nonzero(overlap)[0]]
