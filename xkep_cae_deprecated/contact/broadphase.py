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

    # セルサイズ自動推定（ベクトル化: np.max(axis=1) 一括計算）
    if cell_size is None:
        sizes = np.max(hi_all - lo_all, axis=1)  # (n,)
        cell_size = max(float(np.mean(sizes)) * 1.5, 1e-30)

    inv_cell = 1.0 / cell_size

    # セルインデックスを一括計算（per-segment np.floor を排除）
    ilo_all = np.floor(lo_all * inv_cell).astype(np.intp)  # (n, 3)
    ihi_all = np.floor(hi_all * inv_cell).astype(np.intp)  # (n, 3)

    # 空間ハッシュ: 各セグメントが占めるセルにビニング
    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for i in range(n):
        ix0 = int(ilo_all[i, 0])
        iy0 = int(ilo_all[i, 1])
        iz0 = int(ilo_all[i, 2])
        ix1 = int(ihi_all[i, 0])
        iy1 = int(ihi_all[i, 1])
        iz1 = int(ihi_all[i, 2])
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    grid[(ix, iy, iz)].append(i)

    # 候補ペア収集（重複除去のみ、AABB チェックは後でバッチ処理）
    seen: set[tuple[int, int]] = set()
    for cell_indices in grid.values():
        nc = len(cell_indices)
        if nc < 2:
            continue
        for a_idx in range(nc):
            for b_idx in range(a_idx + 1, nc):
                i = cell_indices[a_idx]
                j = cell_indices[b_idx]
                seen.add((i, j) if i < j else (j, i))

    if not seen:
        return []

    # バッチ AABB 重複判定（ベクトル化: 全ペア一括で numpy 演算）
    pairs_arr = np.array(list(seen), dtype=np.intp)  # (m, 2)
    pi, pj = pairs_arr[:, 0], pairs_arr[:, 1]
    overlap = np.all(lo_all[pi] <= hi_all[pj], axis=1) & np.all(lo_all[pj] <= hi_all[pi], axis=1)
    return [(int(r[0]), int(r[1])) for r in pairs_arr[overlap]]
