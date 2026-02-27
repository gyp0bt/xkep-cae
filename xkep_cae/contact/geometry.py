"""接触幾何: segment-to-segment 最近接点計算.

2線分間の最近接点をパラメトリックに求める。

Phase C5: build_contact_frame に平行輸送（parallel transport）ベースの
フレーム連続更新を追加。法線回転が大きい場合のフレーム不連続を防止する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClosestPointResult:
    """最近接点計算の結果.

    Attributes:
        s: セグメントA上のパラメータ ∈ [0,1]
        t: セグメントB上のパラメータ ∈ [0,1]
        point_a: A上の最近接点 (3,)
        point_b: B上の最近接点 (3,)
        distance: 最近接距離
        normal: 法線ベクトル (3,)  d / ||d||
        parallel: 平行判定フラグ
    """

    s: float
    t: float
    point_a: np.ndarray
    point_b: np.ndarray
    distance: float
    normal: np.ndarray
    parallel: bool = False


def closest_point_segments(
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    *,
    tol_parallel: float = 1e-10,
) -> ClosestPointResult:
    """2線分間の最近接点を計算する.

    セグメントA: p(s) = xA0 + s*(xA1 - xA0),  s ∈ [0,1]
    セグメントB: q(t) = xB0 + t*(xB1 - xB0),  t ∈ [0,1]

    最近接条件:
        d(s,t) = p(s) - q(t) を最小化
        → (dA · dA) s - (dA · dB) t = -(w0 · dA)
          -(dA · dB) s + (dB · dB) t = (w0 · dB)
        ただし dA = xA1 - xA0, dB = xB1 - xB0, w0 = xA0 - xB0

    境界処理: s, t を [0,1] にクランプ後、相手パラメータを再計算。

    Args:
        xA0: セグメントA の始点 (3,)
        xA1: セグメントA の終点 (3,)
        xB0: セグメントB の始点 (3,)
        xB1: セグメントB の終点 (3,)
        tol_parallel: 平行判定の閾値

    Returns:
        ClosestPointResult
    """
    dA = xA1 - xA0  # (3,)
    dB = xB1 - xB0  # (3,)
    w0 = xA0 - xB0  # (3,)

    a = float(dA @ dA)  # |dA|²
    b = float(dA @ dB)  # dA·dB
    c = float(dB @ dB)  # |dB|²
    d = float(dA @ w0)  # dA·w0
    e = float(dB @ w0)  # dB·w0

    det = a * c - b * b

    is_parallel = det < tol_parallel * max(a * c, 1e-30)

    if is_parallel:
        # 平行/縮退ケース: sを0に固定し、tを最適化
        s = 0.0
        t = np.clip(e / c, 0.0, 1.0) if c > tol_parallel else 0.0
    else:
        # 無拘束の最適解
        s_unc = (b * e - c * d) / det
        t_unc = (a * e - b * d) / det

        # [0,1] にクランプ
        s = np.clip(s_unc, 0.0, 1.0)
        t = np.clip(t_unc, 0.0, 1.0)

        # クランプ後の再計算
        if s_unc < 0.0 or s_unc > 1.0:
            # s がクランプされた → t を再計算
            t = np.clip((b * s + e) / c, 0.0, 1.0) if c > tol_parallel else 0.0
        if t_unc < 0.0 or t_unc > 1.0:
            # t がクランプされた → s を再計算
            s = np.clip((b * t - d) / a, 0.0, 1.0) if a > tol_parallel else 0.0
            # s が再クランプされたら t も再計算
            t = np.clip((b * s + e) / c, 0.0, 1.0) if c > tol_parallel else 0.0

    point_a = xA0 + s * dA
    point_b = xB0 + t * dB
    diff = point_a - point_b
    distance = float(np.linalg.norm(diff))

    if distance > 1e-30:
        normal = diff / distance
    else:
        # 距離ゼロ: 法線不定（デフォルトで z 軸）
        normal = np.array([0.0, 0.0, 1.0])

    return ClosestPointResult(
        s=s,
        t=t,
        point_a=point_a,
        point_b=point_b,
        distance=distance,
        normal=normal,
        parallel=is_parallel,
    )


def compute_gap(
    distance: float,
    radius_a: float,
    radius_b: float,
) -> float:
    """ギャップを計算する.

    g = distance - (r_a + r_b)
    g >= 0: 離間
    g <  0: 貫通

    Args:
        distance: 中心線間距離
        radius_a: 断面半径A
        radius_b: 断面半径B

    Returns:
        gap: ギャップ値
    """
    return distance - (radius_a + radius_b)


def _parallel_transport(
    t1_old: np.ndarray,
    n_old: np.ndarray,
    n_new: np.ndarray,
) -> np.ndarray:
    """平行輸送で接線ベクトルを新しい法線面に輸送する.

    n_old → n_new の最短回転（Rodrigues 公式）を t1_old に適用する。
    Gram-Schmidt よりも大きな法線回転に対して連続性が高い。

    Args:
        t1_old: 旧接線基底1 (3,)
        n_old: 旧法線ベクトル (3,)
        n_new: 新法線ベクトル (3,)

    Returns:
        t1_new: 輸送された接線ベクトル (3,)（未正規化）
    """
    v = np.cross(n_old, n_new)
    s = float(np.linalg.norm(v))
    c = float(n_old @ n_new)

    if s < 1e-12:
        if c > 0:
            # n_old ≈ n_new: 回転なし
            return t1_old.copy()
        else:
            # n_old ≈ -n_new: 180° 回転（t1 はそのまま反転不要、n だけ反転）
            return t1_old.copy()

    # Rodrigues 公式: R(v, θ) * t1_old
    # R = I + [v]_x + [v]_x^2 * (1 - c) / s^2
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R @ t1_old


def compute_st_jacobian(
    s: float,
    t: float,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    *,
    tol_boundary: float = 1e-10,
    tol_parallel: float = 1e-20,
) -> tuple[np.ndarray, np.ndarray] | None:
    """∂(s,t)/∂u を陰関数の定理で計算する.

    最近接点条件:
      F₁ = δ · dA = 0
      F₂ = -δ · dB = 0
    ただし δ = pA(s) - pB(t), dA = xA1 - xA0, dB = xB1 - xB0

    暗関数の定理:
      J · d(s,t)/du = -∂F/∂u

    J = [[dA·dA, -(dA·dB)],
         [-(dA·dB), dB·dB]]

    ∂F/∂u は δ の ∂dA/∂u, ∂dB/∂u 項も含む完全版。

    境界処理:
    - s がクランプ（0 or 1）→ ds/du = 0, dt のみ 1×1 系で計算
    - t がクランプ → dt/du = 0, ds のみ 1×1 系で計算
    - 両方クランプ → ds/du = dt/du = 0

    Phase C6-L2: 一貫接線の完全化のコア関数。

    Args:
        s, t: 現在の最近接パラメータ
        xA0, xA1: セグメント A の変形後端点座標 (3,)
        xB0, xB1: セグメント B の変形後端点座標 (3,)
        tol_boundary: 境界判定の閾値
        tol_parallel: 平行特異判定の閾値

    Returns:
        (ds_du, dt_du): 各 (12,) ベクトル。平行特異の場合は None。
    """
    dA = xA1 - xA0
    dB = xB1 - xB0
    delta = (1.0 - s) * xA0 + s * xA1 - ((1.0 - t) * xB0 + t * xB1)

    a = float(dA @ dA)  # |dA|²
    b = float(dA @ dB)  # dA·dB
    c = float(dB @ dB)  # |dB|²

    s_clamped = (s < tol_boundary) or (s > 1.0 - tol_boundary)
    t_clamped = (t < tol_boundary) or (t > 1.0 - tol_boundary)

    if s_clamped and t_clamped:
        return np.zeros(12), np.zeros(12)

    # ∂F/∂u (2×12) — δ·∂dA/∂u, δ·∂dB/∂u 項を含む完全版
    # F₁ = δ · dA,  F₂ = -δ · dB
    dF_du = np.zeros((2, 12))
    # ∂F₁/∂u = ∂δ/∂u · dA + δ · ∂dA/∂u
    dF_du[0, 0:3] = (1.0 - s) * dA - delta
    dF_du[0, 3:6] = s * dA + delta
    dF_du[0, 6:9] = -(1.0 - t) * dA
    dF_du[0, 9:12] = -t * dA
    # ∂F₂/∂u = -∂δ/∂u · dB - δ · ∂dB/∂u
    dF_du[1, 0:3] = -(1.0 - s) * dB
    dF_du[1, 3:6] = -s * dB
    dF_du[1, 6:9] = (1.0 - t) * dB + delta
    dF_du[1, 9:12] = t * dB - delta

    ds_du = np.zeros(12)
    dt_du = np.zeros(12)

    if s_clamped:
        # s 固定, t のみ自由: c * dt/du = -dF_du[1]
        if abs(c) < tol_parallel:
            return None
        dt_du = -dF_du[1, :] / c
    elif t_clamped:
        # t 固定, s のみ自由: a * ds/du = -dF_du[0]
        if abs(a) < tol_parallel:
            return None
        ds_du = -dF_du[0, :] / a
    else:
        # 両方自由: 2×2 系
        det_J = a * c - b * b
        if abs(det_J) < tol_parallel * max(a * c, 1e-30):
            return None  # 平行特異 → フォールバック
        J_inv = np.array([[c, b], [b, a]]) / det_J
        dst_du = -J_inv @ dF_du  # (2, 12)
        ds_du = dst_du[0]
        dt_du = dst_du[1]

    return ds_du, dt_du


def build_contact_frame(
    normal: np.ndarray,
    prev_tangent1: np.ndarray | None = None,
    prev_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """接触ローカルフレーム (n, t1, t2) を構築する.

    法線 n に対して接線基底 t1, t2 を生成する。
    前ステップの法線と接線が与えられた場合、平行輸送で連続性を保つ。
    前ステップの法線がない場合は Gram-Schmidt で直交化する。

    Args:
        normal: 法線ベクトル (3,)
        prev_tangent1: 前ステップの接線基底1（連続更新用）
        prev_normal: 前ステップの法線ベクトル（平行輸送用、Phase C5）

    Returns:
        n, t1, t2: 正規直交基底 (3,) × 3
    """
    n = normal / max(float(np.linalg.norm(normal)), 1e-30)

    if prev_tangent1 is not None:
        if prev_normal is not None:
            # Phase C5: 平行輸送による連続フレーム更新
            n_old = prev_normal / max(float(np.linalg.norm(prev_normal)), 1e-30)
            t1_transported = _parallel_transport(prev_tangent1, n_old, n)
            # 正規化（数値誤差で n に完全直交でない可能性）
            t1_raw = t1_transported - (t1_transported @ n) * n
            t1_norm = float(np.linalg.norm(t1_raw))
            if t1_norm > 1e-10:
                t1 = t1_raw / t1_norm
                t2 = np.cross(n, t1)
                return n, t1, t2

        # Gram-Schmidt フォールバック: 前ステップの t1 を新法線に直交化
        t1_raw = prev_tangent1 - (prev_tangent1 @ n) * n
        t1_norm = float(np.linalg.norm(t1_raw))
        if t1_norm > 1e-10:
            t1 = t1_raw / t1_norm
            t2 = np.cross(n, t1)
            return n, t1, t2

    # 初回または退化: 最も直交的な座標軸を選択
    abs_n = np.abs(n)
    if abs_n[0] <= abs_n[1] and abs_n[0] <= abs_n[2]:
        ref = np.array([1.0, 0.0, 0.0])
    elif abs_n[1] <= abs_n[2]:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    t1 = ref - (ref @ n) * n
    t1 = t1 / float(np.linalg.norm(t1))
    t2 = np.cross(n, t1)

    return n, t1, t2
