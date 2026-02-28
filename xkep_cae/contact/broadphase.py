"""Broadphase 接触候補探索（AABB格子）.

セグメント群のAABB (Axis-Aligned Bounding Box) を計算し、
空間ハッシュ格子による O(n) の候補ペア抽出を行う。

Phase C1 で実装。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def compute_segment_aabb(
    x0: np.ndarray,
    x1: np.ndarray,
    radius: float = 0.0,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """セグメントの AABB を計算する.

    Args:
        x0: 始点 (3,)
        x1: 終点 (3,)
        radius: 断面半径（AABB を膨張）
        margin: 追加マージン（探索余裕）

    Returns:
        (min_corner, max_corner): 各 (3,)
    """
    expand = radius + margin
    lo = np.minimum(x0, x1) - expand
    hi = np.maximum(x0, x1) + expand
    return lo, hi


def _aabb_overlap(
    lo_a: np.ndarray,
    hi_a: np.ndarray,
    lo_b: np.ndarray,
    hi_b: np.ndarray,
) -> bool:
    """2つの AABB が重なるか判定."""
    return bool(np.all(lo_a <= hi_b) and np.all(lo_b <= hi_a))


def broadphase_aabb(
    segments: list[tuple[np.ndarray, np.ndarray]],
    radii: np.ndarray | float = 0.0,
    *,
    margin: float = 0.0,
    cell_size: float | None = None,
) -> list[tuple[int, int]]:
    """AABB 空間ハッシュによる候補ペア探索.

    各セグメントの AABB を均一格子にビニングし、
    同一セルまたは隣接セル内のペアを候補として返す。

    Args:
        segments: [(x0, x1), ...] セグメント端点のリスト。各端点は (3,)
        radii: セグメントごとの断面半径。スカラーなら全セグメント共通
        margin: 追加探索マージン
        cell_size: 格子セルサイズ。None なら自動推定

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

    # AABB 計算（ベクトル化: numpy 一括処理）
    x0_arr = np.array([s[0] for s in segments])  # (n, 3)
    x1_arr = np.array([s[1] for s in segments])  # (n, 3)
    expand = r_arr[:, None] + margin  # (n, 1)
    lo_all = np.minimum(x0_arr, x1_arr) - expand  # (n, 3)
    hi_all = np.maximum(x0_arr, x1_arr) + expand  # (n, 3)
    aabbs: list[tuple[np.ndarray, np.ndarray]] = [(lo_all[i], hi_all[i]) for i in range(n)]

    # セルサイズ自動推定: 各 AABB の最大辺長の平均
    if cell_size is None:
        sizes = []
        for lo, hi in aabbs:
            sizes.append(float(np.max(hi - lo)))
        cell_size = max(np.mean(sizes) * 1.5, 1e-30)

    inv_cell = 1.0 / cell_size

    # 空間ハッシュ: 各セグメントが占めるセルにビニング
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    segment_cells: list[list[tuple[int, int, int]]] = []

    for i, (lo, hi) in enumerate(aabbs):
        ilo = np.floor(lo * inv_cell).astype(int)
        ihi = np.floor(hi * inv_cell).astype(int)
        cells = []
        for ix in range(ilo[0], ihi[0] + 1):
            for iy in range(ilo[1], ihi[1] + 1):
                for iz in range(ilo[2], ihi[2] + 1):
                    key = (ix, iy, iz)
                    grid[key].append(i)
                    cells.append(key)
        segment_cells.append(cells)

    # 候補ペア抽出: 同一セル内のペアで AABB 重複を確認
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []

    for cell_indices in grid.values():
        for a_idx in range(len(cell_indices)):
            for b_idx in range(a_idx + 1, len(cell_indices)):
                i = cell_indices[a_idx]
                j = cell_indices[b_idx]
                pair_key = (min(i, j), max(i, j))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                # AABB の精密重複判定
                if _aabb_overlap(aabbs[i][0], aabbs[i][1], aabbs[j][0], aabbs[j][1]):
                    candidates.append(pair_key)

    return candidates
