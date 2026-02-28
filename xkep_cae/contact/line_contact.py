"""Line-to-line 接触（Gauss 積分）.

Phase C6-L1: Point-to-point (PtP) 接触の拡張として、セグメントに沿った
Gauss 積分により接触力・接線剛性を評価する。

Phase C6-L1b: 摩擦力の line contact 拡張。各 Gauss 点で独立に Coulomb
return mapping を実行し、分布摩擦力・接線剛性を Gauss 積分で評価する。

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


def compute_t_jacobian_at_gp(
    s_gp: float,
    t_closest: float,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    *,
    tol_boundary: float = 1e-10,
    tol_singular: float = 1e-20,
) -> np.ndarray | None:
    """Line contact Gauss 点での ∂t/∂u を計算する.

    Gauss 点 s_gp は固定（積分点パラメータ）なので ds/du = 0。
    t_closest のみが u に依存する。

    射影条件:
      G(t, u) = (pA(s_gp) - pB(t)) · dB = 0

    暗関数の定理:
      ∂G/∂t · dt/du = -∂G/∂u|_{t fixed}

    ∂G/∂t = -|dB|²

    Phase C6-L2: Line contact での一貫接線。

    Args:
        s_gp: Gauss 点パラメータ（固定）
        t_closest: 最近接パラメータ ∈ [0,1]
        xA0, xA1: セグメント A の変形後端点 (3,)
        xB0, xB1: セグメント B の変形後端点 (3,)
        tol_boundary: 境界判定の閾値
        tol_singular: 特異判定の閾値

    Returns:
        dt_du: (12,) ∂t/∂u ベクトル。t がクランプまたは特異の場合は None。
    """
    t = t_closest

    # t がクランプされている場合は dt/du = 0
    if t < tol_boundary or t > 1.0 - tol_boundary:
        return np.zeros(12)

    dB = xB1 - xB0
    c = float(dB @ dB)  # |dB|²
    if abs(c) < tol_singular:
        return None  # 縮退セグメント

    # delta_gp = pA(s_gp) - pB(t)
    pA = (1.0 - s_gp) * xA0 + s_gp * xA1
    pB = (1.0 - t) * xB0 + t * xB1
    delta_gp = pA - pB

    # ∂G/∂u|_{t fixed}  (12,)
    # G = delta_gp · dB
    # ∂G/∂u_k = ∂delta_gp/∂u_k · dB + delta_gp · ∂dB/∂u_k
    dG_du = np.zeros(12)
    # ∂delta/∂uA0 = (1-s_gp)*I₃ → ·dB = (1-s_gp)*dB
    dG_du[0:3] = (1.0 - s_gp) * dB
    # ∂delta/∂uA1 = s_gp*I₃ → ·dB = s_gp*dB
    dG_du[3:6] = s_gp * dB
    # ∂delta/∂uB0 = -(1-t)*I₃, ∂dB/∂uB0 = -I₃
    dG_du[6:9] = -(1.0 - t) * dB - delta_gp
    # ∂delta/∂uB1 = -t*I₃, ∂dB/∂uB1 = I₃
    dG_du[9:12] = -t * dB + delta_gp

    # ∂G/∂t = -|dB|² (∂delta/∂t · dB = -dB · dB)
    dG_dt = -c

    # dt/du = -(1/∂G/∂t) * ∂G/∂u = (1/|dB|²) * ∂G/∂u
    dt_du = -dG_du / dG_dt  # = dG_du / c

    return dt_du


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


# ============================================================
# Phase C6-L1b: 摩擦力の line contact 拡張
# ============================================================


def _build_tangent_shape_vector_at_gp(
    s_gp: float,
    t_closest: float,
    tangent: np.ndarray,
) -> np.ndarray:
    """Gauss 点での接線方向形状ベクトルを構築する.

    法線形状ベクトルと同じ配分方式だが、方向が tangent ベクトル。
    摩擦力は A を滑り方向に引きずり、B を反対方向に引く。

    Args:
        s_gp: セグメント A 上のパラメータ
        t_closest: セグメント B 上の最近接パラメータ
        tangent: 接線方向ベクトル (3,)

    Returns:
        g_t: (12,) 接線方向形状ベクトル
    """
    g_t = np.zeros(12)
    g_t[0:3] = (1.0 - s_gp) * tangent
    g_t[3:6] = s_gp * tangent
    g_t[6:9] = -(1.0 - t_closest) * tangent
    g_t[9:12] = -t_closest * tangent
    return g_t


def _compute_tangent_frame_at_gp(
    normal: np.ndarray,
    ref_tangent1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss 点での接線フレームを法線から構築する.

    ペアの参照接線 t1 を法線に直交化して接線フレームを作る。

    Args:
        normal: 法線ベクトル (3,)
        ref_tangent1: 参照接線ベクトル (3,)

    Returns:
        t1: 接線基底1 (3,)
        t2: 接線基底2 (3,)
    """
    # ref_tangent1 を法線に直交化
    t1_raw = ref_tangent1 - float(np.dot(ref_tangent1, normal)) * normal
    t1_norm = float(np.linalg.norm(t1_raw))

    if t1_norm < 1e-30:
        # ref_tangent1 が法線と平行 → 別の軸を試す
        candidates = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        for c in candidates:
            t1_raw = c - float(np.dot(c, normal)) * normal
            t1_norm = float(np.linalg.norm(t1_raw))
            if t1_norm > 1e-10:
                break
        if t1_norm < 1e-30:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    t1 = t1_raw / t1_norm
    t2 = np.cross(normal, t1)
    t2_norm = float(np.linalg.norm(t2))
    if t2_norm > 1e-30:
        t2 = t2 / t2_norm
    return t1, t2


def _init_gp_friction_states(pair: ContactPair, n_gauss: int) -> None:
    """Gauss 点摩擦状態を初期化する（必要に応じて）.

    既存の状態の Gauss 点数が一致しない場合はリセットする。

    Args:
        pair: 接触ペア
        n_gauss: Gauss 積分点数
    """
    if pair.state.gp_z_t is None or len(pair.state.gp_z_t) != n_gauss:
        pair.state.gp_z_t = [np.zeros(2) for _ in range(n_gauss)]
        pair.state.gp_stick = [True] * n_gauss
        pair.state.gp_q_trial_norm = [0.0] * n_gauss


def compute_line_friction_force_local(
    pair: ContactPair,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    u_cur: np.ndarray,
    u_ref: np.ndarray,
    ndof_per_node: int,
    mu: float,
    n_gauss: int = 3,
) -> tuple[np.ndarray, float]:
    """Line-to-line 摩擦力を Gauss 積分で評価する.

    各 Gauss 点で独立に Coulomb return mapping を実行し、
    摩擦力を Gauss 積分で統合する。

    各 Gauss 点 s_i での処理:
    1. A 上の点 pA = (1-s_i)*xA0 + s_i*xA1
    2. pA から B への最近接パラメータ t_i を計算
    3. 法線力 p_n_i = max(0, λ_n + k_pen*(-g_i))
    4. 接線フレーム (t1, t2) を法線から構築
    5. 接線相対変位 Δu_t を計算
    6. Coulomb return mapping（GP 独立の z_t_gp）
    7. 摩擦力を形状ベクトル経由で 12成分に変換
    8. f_local += w_i * (q_t1*g_t1 + q_t2*g_t2)

    pair.state.gp_z_t, gp_stick, gp_q_trial_norm を更新する（副作用）。

    Args:
        pair: 接触ペア
        xA0, xA1: セグメント A の変形後端点 (3,)
        xB0, xB1: セグメント B の変形後端点 (3,)
        u_cur: (ndof_total,) 現在の変位ベクトル
        u_ref: (ndof_total,) 参照変位ベクトル（ステップ開始時）
        ndof_per_node: 1節点あたりの DOF 数
        mu: 有効摩擦係数
        n_gauss: Gauss 積分点数

    Returns:
        f_local: (12,) 局所摩擦力ベクトル
        total_friction: 合計摩擦力（重み付きノルム和、診断用）
    """
    gp, gw = gauss_legendre_01(n_gauss)
    f_local = np.zeros(12)
    total_friction = 0.0

    _init_gp_friction_states(pair, n_gauss)

    lambda_n = pair.state.lambda_n
    k_pen = pair.state.k_pen
    k_t = pair.state.k_t
    r_sum = pair.radius_a + pair.radius_b

    if mu <= 0.0 or k_t <= 0.0:
        return f_local, total_friction

    # 変位増分
    du = u_cur - u_ref
    nA0, nA1 = int(pair.nodes_a[0]), int(pair.nodes_a[1])
    nB0, nB1 = int(pair.nodes_b[0]), int(pair.nodes_b[1])
    du_A0 = du[nA0 * ndof_per_node : nA0 * ndof_per_node + 3]
    du_A1 = du[nA1 * ndof_per_node : nA1 * ndof_per_node + 3]
    du_B0 = du[nB0 * ndof_per_node : nB0 * ndof_per_node + 3]
    du_B1 = du[nB1 * ndof_per_node : nB1 * ndof_per_node + 3]

    for idx, (s_gp, w) in enumerate(zip(gp, gw, strict=True)):
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

        # 法線力
        p_n_gp = max(0.0, lambda_n + k_pen * (-gap))
        if p_n_gp <= 0.0:
            continue

        # 接線フレーム
        t1, t2 = _compute_tangent_frame_at_gp(normal, pair.state.tangent1)

        # 接線相対変位（s_gp, t_closest で補間）
        du_A = (1.0 - s_gp) * du_A0 + s_gp * du_A1
        du_B = (1.0 - t_closest) * du_B0 + t_closest * du_B1
        du_rel = du_B - du_A
        delta_ut_gp = np.array([float(du_rel @ t1), float(du_rel @ t2)])

        # Return mapping at Gauss point
        z_t_old = pair.state.gp_z_t[idx].copy()
        q_trial = z_t_old + k_t * delta_ut_gp
        q_trial_norm = float(np.linalg.norm(q_trial))
        pair.state.gp_q_trial_norm[idx] = q_trial_norm

        f_yield = mu * p_n_gp
        if q_trial_norm <= f_yield:
            # stick
            q_gp = q_trial.copy()
            pair.state.gp_z_t[idx] = q_gp.copy()
            pair.state.gp_stick[idx] = True
        else:
            # slip → radial return
            if q_trial_norm > 1e-30:
                q_gp = f_yield * q_trial / q_trial_norm
            else:
                q_gp = np.zeros(2)
            pair.state.gp_z_t[idx] = q_gp.copy()
            pair.state.gp_stick[idx] = False

        # 摩擦力を形状ベクトル経由で 12成分に変換
        for axis in range(2):
            if abs(q_gp[axis]) < 1e-30:
                continue
            tang = t1 if axis == 0 else t2
            g_t = _build_tangent_shape_vector_at_gp(s_gp, t_closest, tang)
            f_local += w * q_gp[axis] * g_t

        total_friction += w * float(np.linalg.norm(q_gp))

    return f_local, total_friction


def compute_line_friction_stiffness_local(
    pair: ContactPair,
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    mu: float,
    n_gauss: int = 3,
) -> np.ndarray:
    """Line-to-line 摩擦接線剛性を Gauss 積分で評価する.

    各 Gauss 点での摩擦接線剛性 D_t (2×2) を
    形状ベクトル経由で 12×12 行列に変換し、Gauss 積分で統合する。

    gp_z_t, gp_stick, gp_q_trial_norm が
    compute_line_friction_force_local() で設定済みであること。

    Args:
        pair: 接触ペア（GP 摩擦状態が設定済み）
        xA0, xA1: セグメント A の変形後端点 (3,)
        xB0, xB1: セグメント B の変形後端点 (3,)
        mu: 有効摩擦係数
        n_gauss: Gauss 積分点数

    Returns:
        K_local: (12, 12) 摩擦接線剛性行列
    """
    gp, gw = gauss_legendre_01(n_gauss)
    K_local = np.zeros((12, 12))

    if pair.state.gp_z_t is None or mu <= 0.0:
        return K_local

    lambda_n = pair.state.lambda_n
    k_pen = pair.state.k_pen
    k_t = pair.state.k_t
    r_sum = pair.radius_a + pair.radius_b

    if k_t <= 0.0:
        return K_local

    for idx, (s_gp, w) in enumerate(zip(gp, gw, strict=True)):
        # A 側の Gauss 点位置
        pA = (1.0 - s_gp) * xA0 + s_gp * xA1
        t_closest = project_point_to_segment(pA, xB0, xB1)
        pB = (1.0 - t_closest) * xB0 + t_closest * xB1

        diff = pA - pB
        dist = float(np.linalg.norm(diff))
        gap = dist - r_sum

        if dist > 1e-30:
            normal = diff / dist
        else:
            normal = pair.state.normal.copy()

        p_n_gp = max(0.0, lambda_n + k_pen * (-gap))
        if p_n_gp <= 0.0:
            continue

        if idx >= len(pair.state.gp_z_t):
            continue

        # 接線フレーム
        t1, t2 = _compute_tangent_frame_at_gp(normal, pair.state.tangent1)

        # D_t (2×2) at GP
        gp_stick = pair.state.gp_stick[idx]
        if gp_stick:
            D_t = k_t * np.eye(2)
        else:
            z_gp = pair.state.gp_z_t[idx]
            z_norm = float(np.linalg.norm(z_gp))
            q_tn = pair.state.gp_q_trial_norm[idx]
            if z_norm > 1e-30 and q_tn > 1e-30:
                q_hat = z_gp / z_norm
                ratio = (mu * p_n_gp) / q_tn
                D_t = ratio * k_t * (np.eye(2) - np.outer(q_hat, q_hat))
            else:
                D_t = k_t * np.eye(2)

        # 接線形状ベクトル
        g_t = [
            _build_tangent_shape_vector_at_gp(s_gp, t_closest, t1),
            _build_tangent_shape_vector_at_gp(s_gp, t_closest, t2),
        ]

        # K_f_gp = Σ D_t[a1,a2] * g_t_a1 ⊗ g_t_a2
        K_f_gp = np.zeros((12, 12))
        for a1 in range(2):
            for a2 in range(2):
                d_val = D_t[a1, a2]
                if abs(d_val) > 1e-30:
                    K_f_gp += d_val * np.outer(g_t[a1], g_t[a2])

        K_local += w * K_f_gp

    return K_local
