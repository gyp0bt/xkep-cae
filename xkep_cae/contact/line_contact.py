"""Line-to-line 接触（Gauss 積分）.

Phase C6-L1: Point-to-point (PtP) 接触の拡張として、セグメントに沿った
Gauss 積分により接触力・接線剛性を評価する。

PtP では各セグメントペアに対して1点の最近接点 (s*, t*) のみで接触力を
評価するが、準平行な梁（撚線の同層梁など）では1点評価が不正確となる。
Line-to-line 接触は Gauss 積分で接触力の空間分布を捉え、精度を向上する。

参考文献:
- Meier, Popp, Wall (2016): "A unified approach for beam-to-beam contact"
- Meier, Popp, Wall (2016): "A finite element approach for the line-to-line
  contact interaction of thin beams"

設計仕様: docs/contact/contact-algorithm-overhaul-c6.md §3
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.pair import ContactPair


def gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """[0,1] 区間の Gauss-Legendre 積分点と重みを返す.

    numpy の leggauss は [-1,1] 区間を返すため、
    s = (xi + 1) / 2, w_01 = w / 2 で [0,1] に変換する。

    Args:
        n: 積分点数（1-10）

    Returns:
        points: (n,) 積分点 s ∈ [0,1]
        weights: (n,) 重み（合計 1.0）
    """
    xi, w = np.polynomial.legendre.leggauss(n)
    points = (xi + 1.0) / 2.0
    weights = w / 2.0
    return points, weights


def project_point_to_segment(
    p: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> float:
    """点 p からセグメント [x0, x1] への最近接パラメータ t ∈ [0,1] を返す.

    t = clamp((p - x0) · d / |d|², 0, 1)

    Args:
        p: 投影する点 (3,)
        x0: セグメント始点 (3,)
        x1: セグメント終点 (3,)

    Returns:
        t: 最近接パラメータ ∈ [0,1]
    """
    d = x1 - x0
    len_sq = float(d @ d)
    if len_sq < 1e-30:
        return 0.0
    t = float((p - x0) @ d / len_sq)
    return max(0.0, min(1.0, t))


def auto_select_n_gauss(
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    *,
    default: int = 3,
) -> int:
    """セグメント間角度に基づいて Gauss 点数を自動選択する.

    角度 θ が小さい（準平行）ほど接触帯が長くなるため、
    多くの Gauss 点が必要。

    | 角度 θ       | n_gauss | 理由                              |
    |--------------|---------|-----------------------------------|
    | θ > 30°      | 2       | PtP と同等精度、追加コスト最小     |
    | 10° < θ < 30° | 3      | 中間領域、標準                    |
    | θ < 10°      | 5       | 長い接触帯、高精度必須            |

    Args:
        xA0, xA1: セグメント A の端点 (3,)
        xB0, xB1: セグメント B の端点 (3,)
        default: 退化ケースのデフォルト値

    Returns:
        n_gauss: 推奨 Gauss 点数
    """
    dA = xA1 - xA0
    dB = xB1 - xB0
    lenA = float(np.linalg.norm(dA))
    lenB = float(np.linalg.norm(dB))

    if lenA < 1e-30 or lenB < 1e-30:
        return default

    cos_angle = abs(float(dA @ dB / (lenA * lenB)))

    if cos_angle < 0.866:  # θ > 30°
        return 2
    elif cos_angle < 0.985:  # 10° < θ < 30°
        return 3
    else:  # θ < 10°, 準平行
        return 5


def _build_shape_vector_at_gp(
    s_gp: float,
    t_closest: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Gauss 点での形状ベクトルを構築する.

    4節点（A0, A1, B0, B1）× 3DOF = 12 成分。

    Args:
        s_gp: セグメント A 上のパラメータ
        t_closest: セグメント B 上の最近接パラメータ
        normal: 法線方向 (3,)

    Returns:
        g_n: (12,) 形状ベクトル
    """
    g_n = np.zeros(12)
    g_n[0:3] = -(1.0 - s_gp) * normal
    g_n[3:6] = -s_gp * normal
    g_n[6:9] = (1.0 - t_closest) * normal
    g_n[9:12] = t_closest * normal
    return g_n


def _geometric_stiffness_at_gp(
    s_gp: float,
    t_closest: float,
    normal: np.ndarray,
    p_n_gp: float,
    dist: float,
) -> np.ndarray:
    """Gauss 点での幾何剛性行列を計算する.

    K_geo = -(p_n / dist) * G^T * (I₃ - n⊗n) * G

    Args:
        s_gp: セグメント A 上のパラメータ
        t_closest: セグメント B 上の最近接パラメータ
        normal: 法線方向 (3,)
        p_n_gp: この Gauss 点での法線力
        dist: 中心線間距離

    Returns:
        K_geo: (12, 12) 幾何剛性行列
    """
    G = np.zeros((3, 12))
    G[:, 0:3] = (1.0 - s_gp) * np.eye(3)
    G[:, 3:6] = s_gp * np.eye(3)
    G[:, 6:9] = -(1.0 - t_closest) * np.eye(3)
    G[:, 9:12] = -t_closest * np.eye(3)

    P_proj = np.eye(3) - np.outer(normal, normal)
    PG = P_proj @ G
    return -(p_n_gp / dist) * (G.T @ PG)


def compute_line_contact_force_local(
    pair: ContactPair,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    n_gauss: int = 3,
) -> tuple[np.ndarray, float]:
    """Line-to-line 接触力を Gauss 積分で評価する.

    セグメント A に沿って Gauss 点を配置し、各点での
    接触力を計算して重み付き加算する。

    各 Gauss 点 s_i での処理:
    1. A 上の点 pA = (1-s_i)*xA0 + s_i*xA1
    2. pA から B への最近接パラメータ t_i を計算
    3. ギャップ g_i = |pA - pB(t_i)| - (r_A + r_B)
    4. 法線力 p_n_i = max(0, λ_n + k_pen * (-g_i))
    5. 形状ベクトル g_n_i を (s_i, t_i) で構築
    6. f_local += w_i * p_n_i * g_n_i

    Args:
        pair: 接触ペア
        xA0, xA1: セグメント A の端点座標 (3,)
        xB0, xB1: セグメント B の端点座標 (3,)
        n_gauss: Gauss 積分点数 (2-5)

    Returns:
        f_local: (12,) 局所接触力ベクトル
        total_p_n: 合計法線力（重み付き和、診断用）
    """
    gp, gw = gauss_legendre_01(n_gauss)
    f_local = np.zeros(12)
    total_p_n = 0.0

    lambda_n = pair.state.lambda_n
    k_pen = pair.state.k_pen
    r_sum = pair.radius_a + pair.radius_b

    for s_gp, w in zip(gp, gw, strict=True):
        # A 側の Gauss 点位置
        pA = (1.0 - s_gp) * xA0 + s_gp * xA1

        # pA から B セグメントへの最近接パラメータ
        t_closest = project_point_to_segment(pA, xB0, xB1)
        pB = (1.0 - t_closest) * xB0 + t_closest * xB1

        # ギャップと法線
        diff = pA - pB
        dist = float(np.linalg.norm(diff))
        gap = dist - r_sum

        if dist > 1e-30:
            normal = diff / dist
        else:
            # 距離ゼロ: ペアの法線を使用
            normal = pair.state.normal.copy()

        # 法線力 (AL)
        p_n_gp = max(0.0, lambda_n + k_pen * (-gap))

        if p_n_gp > 0.0:
            g_n = _build_shape_vector_at_gp(s_gp, t_closest, normal)
            f_local += w * p_n_gp * g_n
            total_p_n += w * p_n_gp

    return f_local, total_p_n


def compute_line_contact_stiffness_local(
    pair: ContactPair,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    n_gauss: int = 3,
    *,
    use_geometric_stiffness: bool = True,
) -> np.ndarray:
    """Line-to-line 接触剛性を Gauss 積分で評価する.

    K_c = ∫₀¹ k_eff(s) · g_n(s) · g_n(s)^T ds
        + ∫₀¹ K_geo(s) ds  （use_geometric_stiffness=True の場合）

    Args:
        pair: 接触ペア
        xA0, xA1: セグメント A の端点座標 (3,)
        xB0, xB1: セグメント B の端点座標 (3,)
        n_gauss: Gauss 積分点数 (2-5)
        use_geometric_stiffness: 幾何剛性を含めるか

    Returns:
        K_local: (12, 12) 局所剛性行列
    """
    gp, gw = gauss_legendre_01(n_gauss)
    K_local = np.zeros((12, 12))

    lambda_n = pair.state.lambda_n
    k_pen = pair.state.k_pen
    r_sum = pair.radius_a + pair.radius_b

    for s_gp, w in zip(gp, gw, strict=True):
        # A 側の Gauss 点位置
        pA = (1.0 - s_gp) * xA0 + s_gp * xA1

        # pA から B セグメントへの最近接パラメータ
        t_closest = project_point_to_segment(pA, xB0, xB1)
        pB = (1.0 - t_closest) * xB0 + t_closest * xB1

        # ギャップと法線
        diff = pA - pB
        dist = float(np.linalg.norm(diff))
        gap = dist - r_sum

        if dist > 1e-30:
            normal = diff / dist
        else:
            normal = pair.state.normal.copy()

        # 法線力 (AL)
        p_n_gp = max(0.0, lambda_n + k_pen * (-gap))

        if p_n_gp <= 0.0:
            continue

        # 形状ベクトル
        g_n = _build_shape_vector_at_gp(s_gp, t_closest, normal)

        # 主項: k_eff * g_n ⊗ g_n
        K_local += w * k_pen * np.outer(g_n, g_n)

        # 幾何剛性
        if use_geometric_stiffness and dist > 1e-30:
            K_geo_gp = _geometric_stiffness_at_gp(s_gp, t_closest, normal, p_n_gp, dist)
            K_local += w * K_geo_gp

    return K_local


def compute_line_contact_gap_at_gp(
    pair: ContactPair,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    n_gauss: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """各 Gauss 点でのギャップ値を計算する（診断用）.

    Args:
        pair: 接触ペア
        xA0, xA1: セグメント A の端点座標 (3,)
        xB0, xB1: セグメント B の端点座標 (3,)
        n_gauss: Gauss 積分点数

    Returns:
        gaps: (n_gauss,) 各 Gauss 点でのギャップ
        s_points: (n_gauss,) Gauss 点パラメータ
        min_gap: 最小ギャップ
    """
    gp, _ = gauss_legendre_01(n_gauss)
    r_sum = pair.radius_a + pair.radius_b
    gaps = np.empty(n_gauss)

    for i, s_gp in enumerate(gp):
        pA = (1.0 - s_gp) * xA0 + s_gp * xA1
        t_closest = project_point_to_segment(pA, xB0, xB1)
        pB = (1.0 - t_closest) * xB0 + t_closest * xB1
        dist = float(np.linalg.norm(pA - pB))
        gaps[i] = dist - r_sum

    return gaps, gp.copy(), float(np.min(gaps))
