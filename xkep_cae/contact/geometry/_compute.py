"""接触幾何バッチ計算の純粋関数.

numpy のみに依存する純粋関数。
"""

from __future__ import annotations

import numpy as np


def _smooth_clip_01(
    s: np.ndarray,
    epsilon: float = 0.02,
) -> np.ndarray:
    """Huber 風 C1 スムースクランプ [0, 1].

    セグメント端点での np.clip 勾配不連続を除去する。

    領域分割:
        s < -ε       → 0                        (ハードクランプ)
        -ε ≤ s < ε   → (s + ε)² / (4ε)          (C1 二次遷移, 下端)
        ε ≤ s ≤ 1-ε  → s                        (線形通過)
        1-ε < s ≤ 1+ε → 1 - (1+ε - s)² / (4ε)  (C1 二次遷移, 上端)
        s > 1+ε      → 1                        (ハードクランプ)

    連続性:
        値:  全区間 C0
        勾配: 全境界 C1 (ds_smooth/ds が連続)

    Args:
        s: パラメトリック座標 (N,)
        epsilon: 遷移幅 (デフォルト 0.02)

    Returns:
        スムースクランプされた s (N,)
    """
    out = np.empty_like(s)

    # 区間マスク
    hard_lo = s < -epsilon
    trans_lo = (~hard_lo) & (s < epsilon)
    hard_hi = s > 1.0 + epsilon
    trans_hi = (~hard_hi) & (s > 1.0 - epsilon)
    linear = ~(hard_lo | trans_lo | hard_hi | trans_hi)

    out[hard_lo] = 0.0
    out[trans_lo] = (s[trans_lo] + epsilon) ** 2 / (4.0 * epsilon)
    out[linear] = s[linear]
    out[trans_hi] = 1.0 - (1.0 + epsilon - s[trans_hi]) ** 2 / (4.0 * epsilon)
    out[hard_hi] = 1.0

    return out


def _closest_point_segments_batch(
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    *,
    tol_parallel: float = 1e-10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """複数ペアの最近接点を一括計算する（ベクトル化版）.

    Args:
        xA0: (N, 3) セグメントA始点
        xA1: (N, 3) セグメントA終点
        xB0: (N, 3) セグメントB始点
        xB1: (N, 3) セグメントB終点
        tol_parallel: 平行判定閾値

    Returns:
        s, t, point_a, point_b, distance, normal, parallel, s_unclamped, t_unclamped
    """
    N = len(xA0)
    if N == 0:
        empty3 = np.empty((0, 3))
        empty1 = np.empty(0)
        return (
            empty1,
            empty1,
            empty3,
            empty3,
            empty1,
            empty3,
            np.empty(0, dtype=bool),
            empty1,
            empty1,
        )

    dA = xA1 - xA0
    dB = xB1 - xB0
    w0 = xA0 - xB0

    a = np.einsum("ij,ij->i", dA, dA)
    b = np.einsum("ij,ij->i", dA, dB)
    c = np.einsum("ij,ij->i", dB, dB)
    d = np.einsum("ij,ij->i", dA, w0)
    e = np.einsum("ij,ij->i", dB, w0)

    det = a * c - b * b
    ac_product = np.maximum(a * c, 1e-30)
    is_parallel = det < tol_parallel * ac_product

    safe_det = np.where(is_parallel, 1.0, det)
    s_unc = np.where(is_parallel, 0.0, (b * e - c * d) / safe_det)
    t_unc = np.where(is_parallel, 0.0, (a * e - b * d) / safe_det)

    safe_c = np.where(c > tol_parallel, c, 1.0)
    # 平行セグメントでは端点遷移不連続は起きないため np.clip のまま
    t_parallel = np.clip(e / safe_c, 0.0, 1.0)
    t_parallel = np.where(c > tol_parallel, t_parallel, 0.0)

    s = np.where(is_parallel, 0.0, _smooth_clip_01(s_unc))
    t = np.where(is_parallel, t_parallel, _smooth_clip_01(t_unc))

    s_clamped = (s_unc < 0.0) | (s_unc > 1.0)
    t_clamped = (t_unc < 0.0) | (t_unc > 1.0)

    t_recalc = _smooth_clip_01((b * s + e) / safe_c)
    t_recalc = np.where(c > tol_parallel, t_recalc, 0.0)
    need_t_recalc = s_clamped & ~is_parallel
    t = np.where(need_t_recalc, t_recalc, t)

    safe_a = np.where(a > tol_parallel, a, 1.0)
    s_recalc = _smooth_clip_01((b * t - d) / safe_a)
    s_recalc = np.where(a > tol_parallel, s_recalc, 0.0)
    need_s_recalc = t_clamped & ~is_parallel
    s = np.where(need_s_recalc, s_recalc, s)

    t_recalc2 = _smooth_clip_01((b * s + e) / safe_c)
    t_recalc2 = np.where(c > tol_parallel, t_recalc2, 0.0)
    t = np.where(need_s_recalc, t_recalc2, t)

    point_a = xA0 + s[:, None] * dA
    point_b = xB0 + t[:, None] * dB
    diff = point_a - point_b
    distance = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    safe_dist = np.maximum(distance, 1e-30)
    normal = diff / safe_dist[:, None]
    zero_mask = distance < 1e-30
    normal[zero_mask] = [0.0, 0.0, 1.0]

    return s, t, point_a, point_b, distance, normal, is_parallel, s_unc, t_unc


def _build_contact_frame_batch(
    normals: np.ndarray,
    prev_tangent1s: np.ndarray | None = None,
    prev_normals: np.ndarray | None = None,
    has_prev_mask: np.ndarray | None = None,
    has_prev_n_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """バッチ版接触フレーム構築.

    Rodrigues 平行輸送 + Gram-Schmidt フォールバック。

    Args:
        normals: (N, 3) 法線ベクトル
        prev_tangent1s: (N, 3) 前ステップの接線基底1
        prev_normals: (N, 3) 前ステップの法線
        has_prev_mask: (N,) bool 前ステップの接線が有効
        has_prev_n_mask: (N,) bool 前ステップの法線が有効

    Returns:
        n, t1, t2: (N, 3) 正規直交基底
    """
    N = len(normals)
    if N == 0:
        empty = np.empty((0, 3))
        return empty, empty, empty

    n_norms = np.sqrt(np.einsum("ij,ij->i", normals, normals))
    safe_norms = np.maximum(n_norms, 1e-30)
    n = normals / safe_norms[:, None]

    t1 = np.zeros((N, 3))
    t2 = np.zeros((N, 3))
    processed = np.zeros(N, dtype=bool)

    # --- 平行輸送ルート ---
    if prev_tangent1s is not None and prev_normals is not None:
        if has_prev_mask is None:
            has_prev_mask = np.ones(N, dtype=bool)
        if has_prev_n_mask is None:
            has_prev_n_mask = np.ones(N, dtype=bool)

        pt_mask = has_prev_mask & has_prev_n_mask & ~processed
        pt_idx = np.where(pt_mask)[0]

        if len(pt_idx) > 0:
            n_old_raw = prev_normals[pt_idx]
            n_old_norms = np.sqrt(np.einsum("ij,ij->i", n_old_raw, n_old_raw))
            safe_old = np.maximum(n_old_norms, 1e-30)
            n_old = n_old_raw / safe_old[:, None]
            n_new = n[pt_idx]
            t1_old = prev_tangent1s[pt_idx]

            v = np.cross(n_old, n_new)
            s_val = np.sqrt(np.einsum("ij,ij->i", v, v))
            c_val = np.einsum("ij,ij->i", n_old, n_new)

            t1_transported = t1_old.copy()

            big_rot = s_val >= 1e-12
            big_idx = np.where(big_rot)[0]
            if len(big_idx) > 0:
                vb = v[big_idx]
                cb = c_val[big_idx]
                sb = s_val[big_idx]
                vxt1 = np.cross(vb, t1_old[big_idx])
                vxvxt1 = np.cross(vb, vxt1)
                factor = (1.0 - cb) / (sb * sb)
                t1_transported[big_idx] = t1_old[big_idx] + vxt1 + factor[:, None] * vxvxt1

            t1_raw = t1_transported - np.einsum("ij,ij->i", t1_transported, n_new)[:, None] * n_new
            t1_raw_norms = np.sqrt(np.einsum("ij,ij->i", t1_raw, t1_raw))
            valid = t1_raw_norms > 1e-10
            valid_idx = pt_idx[valid]
            if len(valid_idx) > 0:
                t1[valid_idx] = t1_raw[valid] / t1_raw_norms[valid, None]
                t2[valid_idx] = np.cross(n[valid_idx], t1[valid_idx])
                processed[valid_idx] = True

        gs_mask = has_prev_mask & ~processed
        gs_idx = np.where(gs_mask)[0]
        if len(gs_idx) > 0:
            pt1 = prev_tangent1s[gs_idx]
            ni = n[gs_idx]
            t1_raw = pt1 - np.einsum("ij,ij->i", pt1, ni)[:, None] * ni
            t1_raw_norms = np.sqrt(np.einsum("ij,ij->i", t1_raw, t1_raw))
            valid = t1_raw_norms > 1e-10
            valid_idx = gs_idx[valid]
            if len(valid_idx) > 0:
                t1[valid_idx] = t1_raw[valid] / t1_raw_norms[valid, None]
                t2[valid_idx] = np.cross(n[valid_idx], t1[valid_idx])
                processed[valid_idx] = True

    # --- 初回フォールバック ---
    rem_idx = np.where(~processed)[0]
    if len(rem_idx) > 0:
        ni = n[rem_idx]
        abs_n = np.abs(ni)
        min_axis = np.argmin(abs_n, axis=1)
        ref = np.zeros_like(ni)
        ref[np.arange(len(min_axis)), min_axis] = 1.0

        t1_raw = ref - np.einsum("ij,ij->i", ref, ni)[:, None] * ni
        t1_raw_norms = np.sqrt(np.einsum("ij,ij->i", t1_raw, t1_raw))
        safe = np.maximum(t1_raw_norms, 1e-30)
        t1[rem_idx] = t1_raw / safe[:, None]
        t2[rem_idx] = np.cross(n[rem_idx], t1[rem_idx])

    return n, t1, t2


def _auto_select_n_gauss(
    dA: np.ndarray,
    dB: np.ndarray,
    *,
    default: int = 3,
) -> int:
    """セグメント間角度に基づいて Gauss 点数を自動選択する.

    | 角度 θ        | n_gauss | 理由                          |
    |---------------|---------|-------------------------------|
    | θ > 30°       | 2       | PtP と同等精度                |
    | 10° < θ < 30° | 3       | 中間領域、標準                |
    | θ < 10°       | 5       | 長い接触帯、高精度必須        |
    """
    lenA = float(np.linalg.norm(dA))
    lenB = float(np.linalg.norm(dB))

    if lenA < 1e-30 or lenB < 1e-30:
        return default

    cos_angle = abs(float(dA @ dB / (lenA * lenB)))

    if cos_angle < 0.866:  # θ > 30°
        return 2
    elif cos_angle < 0.985:  # 10° < θ < 30°
        return 3
    else:  # θ < 10°
        return 5


# ── Hermite 中心線補間 ──────────────────────────────────────


def _compute_node_tangents(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """各節点の接線ベクトルを隣接要素から Catmull-Rom 式で計算.

    節点 i の接線 = Σ(隣接要素の方向ベクトル) を正規化し、
    要素長でスケール（Hermite の m₀, m₁ に直接使える形）。

    内部節点: 両隣の要素方向の平均 × 要素長
    端点: 片方の要素方向 × 要素長

    Returns:
        (n_nodes, 3) 節点接線ベクトル（要素長スケール）
    """
    n_nodes = len(node_coords)
    tangents = np.zeros((n_nodes, 3))
    counts = np.zeros(n_nodes)

    for elem_idx in range(len(connectivity)):
        n0, n1 = connectivity[elem_idx]
        d = node_coords[n1] - node_coords[n0]
        tangents[n0] += d
        tangents[n1] += d
        counts[n0] += 1.0
        counts[n1] += 1.0

    safe_counts = np.maximum(counts, 1.0)
    tangents /= safe_counts[:, None]

    return tangents


def _hermite_eval(
    s: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    """Hermite 補間: s ∈ [0,1] → 位置 (N, 3).

    H00(s) = 2s³ - 3s² + 1
    H10(s) = s³ - 2s² + s
    H01(s) = -2s³ + 3s²
    H11(s) = s³ - s²

    point(s) = H00·x0 + H10·m0 + H01·x1 + H11·m1
    """
    s2 = s * s
    s3 = s2 * s

    h00 = (2.0 * s3 - 3.0 * s2 + 1.0)[:, None]
    h10 = (s3 - 2.0 * s2 + s)[:, None]
    h01 = (-2.0 * s3 + 3.0 * s2)[:, None]
    h11 = (s3 - s2)[:, None]

    return h00 * x0 + h10 * m0 + h01 * x1 + h11 * m1


def _hermite_deriv(
    s: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    """Hermite 補間の s 微分: dp/ds (N, 3).

    H00' = 6s² - 6s
    H10' = 3s² - 4s + 1
    H01' = -6s² + 6s
    H11' = 3s² - 2s
    """
    s2 = s * s

    dh00 = (6.0 * s2 - 6.0 * s)[:, None]
    dh10 = (3.0 * s2 - 4.0 * s + 1.0)[:, None]
    dh01 = (-6.0 * s2 + 6.0 * s)[:, None]
    dh11 = (3.0 * s2 - 2.0 * s)[:, None]

    return dh00 * x0 + dh10 * m0 + dh01 * x1 + dh11 * m1


def _closest_point_hermite_refine(
    s0: np.ndarray,
    t0: np.ndarray,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    mA0: np.ndarray,
    mA1: np.ndarray,
    mB0: np.ndarray,
    mB1: np.ndarray,
    *,
    max_iter: int = 5,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """線形最近接点を初期値として Hermite 曲線上の最近接点を Newton 精密化.

    最近接点条件:
        F₁ = δ · dpA/ds = 0
        F₂ = -δ · dpB/dt = 0
    ただし δ = pA(s) - pB(t)（Hermite 補間上の点）

    Returns:
        s, t, point_a, point_b, distance, normal
    """
    s = s0.copy()
    t = t0.copy()
    eps = 0.02  # smooth_clip の遷移幅と同じ

    for _iter in range(max_iter):
        # Hermite 評価
        pA = _hermite_eval(s, xA0, xA1, mA0, mA1)
        pB = _hermite_eval(t, xB0, xB1, mB0, mB1)
        dpA = _hermite_deriv(s, xA0, xA1, mA0, mA1)
        dpB = _hermite_deriv(t, xB0, xB1, mB0, mB1)

        delta = pA - pB

        # 残差
        F1 = np.einsum("ij,ij->i", delta, dpA)
        F2 = -np.einsum("ij,ij->i", delta, dpB)

        # 収束チェック
        res = np.sqrt(F1 * F1 + F2 * F2)
        if np.all(res < tol):
            break

        # Jacobian 要素
        a = np.einsum("ij,ij->i", dpA, dpA)
        b = np.einsum("ij,ij->i", dpA, dpB)
        c = np.einsum("ij,ij->i", dpB, dpB)

        # d²p/ds² も含めた完全 Jacobian
        # ∂F₁/∂s = dpA·dpA + δ·d²pA/ds² = a + (高次項)
        # 簡略化: Gauss-Newton 近似（高次項省略）
        det = a * c - b * b
        safe_det = np.where(np.abs(det) < 1e-30, 1.0, det)
        inv_det = 1.0 / safe_det

        ds = -(c * F1 + b * F2) * inv_det
        dt = -(b * F1 + a * F2) * inv_det

        # 特異ペアは更新しない
        singular = np.abs(det) < 1e-30
        ds = np.where(singular, 0.0, ds)
        dt = np.where(singular, 0.0, dt)

        s += ds
        t += dt

        # smooth clamp
        s = _smooth_clip_01(s, eps)
        t = _smooth_clip_01(t, eps)

    # 最終評価
    pA = _hermite_eval(s, xA0, xA1, mA0, mA1)
    pB = _hermite_eval(t, xB0, xB1, mB0, mB1)
    diff = pA - pB
    distance = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    safe_dist = np.maximum(distance, 1e-30)
    normal = diff / safe_dist[:, None]
    zero_mask = distance < 1e-30
    normal[zero_mask] = [0.0, 0.0, 1.0]

    return s, t, pA, pB, distance, normal
