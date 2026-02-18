"""接触幾何: segment-to-segment 最近接点計算.

2線分間の最近接点をパラメトリックに求める。
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


def build_contact_frame(
    normal: np.ndarray,
    prev_tangent1: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """接触ローカルフレーム (n, t1, t2) を構築する.

    法線 n に対して接線基底 t1, t2 を生成する。
    前ステップの t1 が与えられた場合、連続性を保つ。

    Args:
        normal: 法線ベクトル (3,)
        prev_tangent1: 前ステップの接線基底1（連続更新用）

    Returns:
        n, t1, t2: 正規直交基底 (3,) × 3
    """
    n = normal / max(float(np.linalg.norm(normal)), 1e-30)

    if prev_tangent1 is not None:
        # 前ステップの t1 を法線に直交化
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
