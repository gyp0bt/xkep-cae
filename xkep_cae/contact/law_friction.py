"""Coulomb 摩擦則（return mapping）.

Phase C3: 接線方向の摩擦力計算（Coulomb return mapping）。
Phase C5: slip consistent tangent の実装 + q_trial_norm 記録。

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §5

摩擦の return mapping:
    1. q_trial = z_t_old + k_t * Δu_t       (弾性予測)
    2. Coulomb 条件: ||q_trial|| <= μ * p_n
    3. stick: ||q_trial|| <= μ * p_n → q = q_trial
    4. slip:  ||q_trial|| >  μ * p_n → q = μ * p_n * q_trial / ||q_trial||
    5. 散逸: D_inc = q · Δu_t  (非負)

μランプ:
    μ_eff = μ_target * min(1, ramp_counter / mu_ramp_steps)
    Outer loop ごとに ramp_counter を漸増し、初期の収束を安定化する。

slip consistent tangent (Phase C5):
    D_t = (μ*p_n / ||q_trial||) * k_t * (I₂ - q̂⊗q̂)
    v0.1 では stick と同じ k_t*I₂ を使用していたが、v0.2 では正確な式を使用。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.pair import ContactPair, ContactStatus


def compute_tangential_displacement(
    pair: ContactPair,
    u_cur: np.ndarray,
    u_ref: np.ndarray,
    node_coords_ref: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """接触点における接線相対変位増分 Δu_t を計算する.

    最近接パラメータ (s, t) を固定し、現在の変位と参照変位の差から
    接触点での接線方向の相対変位増分を求める。

    Δu_rel = [(1-t)(u_B0 - u_B0_ref) + t(u_B1 - u_B1_ref)]
           - [(1-s)(u_A0 - u_A0_ref) + s(u_A1 - u_A1_ref)]
    Δu_t = [Δu_rel · t1, Δu_rel · t2]

    Args:
        pair: 接触ペア（state.s, state.t, state.tangent1/2 が設定済み）
        u_cur: (ndof_total,) 現在の変位ベクトル
        u_ref: (ndof_total,) 参照変位ベクトル（前ステップの収束解）
        node_coords_ref: (n_nodes, 3) 参照節点座標（未使用だが整合性のため保持）
        ndof_per_node: 1節点あたりの DOF 数

    Returns:
        delta_ut: (2,) 接線相対変位増分 [Δu·t1, Δu·t2]
    """
    s = pair.state.s
    t = pair.state.t
    t1 = pair.state.tangent1
    t2 = pair.state.tangent2

    du = u_cur - u_ref

    # 各節点の変位増分（並進成分のみ）
    nA0, nA1 = pair.nodes_a
    nB0, nB1 = pair.nodes_b
    du_A0 = du[nA0 * ndof_per_node : nA0 * ndof_per_node + 3]
    du_A1 = du[nA1 * ndof_per_node : nA1 * ndof_per_node + 3]
    du_B0 = du[nB0 * ndof_per_node : nB0 * ndof_per_node + 3]
    du_B1 = du[nB1 * ndof_per_node : nB1 * ndof_per_node + 3]

    # 接触点での相対変位増分
    du_A = (1.0 - s) * du_A0 + s * du_A1
    du_B = (1.0 - t) * du_B0 + t * du_B1
    du_rel = du_B - du_A  # B-A 方向

    # 接線面への投影
    delta_ut = np.array([float(np.dot(du_rel, t1)), float(np.dot(du_rel, t2))])
    return delta_ut


def friction_return_mapping(
    pair: ContactPair,
    delta_ut: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Coulomb 摩擦の return mapping を実行する.

    弾性予測 → Coulomb 判定 → stick/slip 分岐 → 散逸計算。
    ContactState の z_t, stick, dissipation, status を更新する。

    Args:
        pair: 接触ペア（state.z_t, state.k_t, state.p_n が設定済み）
        delta_ut: (2,) 接線相対変位増分
        mu: 有効摩擦係数（μランプ適用後）

    Returns:
        q: (2,) 接線摩擦力（接触局所座標系）
    """
    if pair.state.status == ContactStatus.INACTIVE:
        return np.zeros(2)

    z_t_old = pair.state.z_t.copy()
    k_t = pair.state.k_t
    p_n = pair.state.p_n

    if p_n <= 0.0 or mu <= 0.0:
        # 法線力ゼロ or 摩擦なし → 摩擦力なし
        pair.state.stick = True
        pair.state.dissipation = 0.0
        return np.zeros(2)

    # 弾性予測（trial）
    q_trial = z_t_old + k_t * delta_ut
    q_trial_norm = float(np.linalg.norm(q_trial))

    # Coulomb 条件
    f_yield = mu * p_n

    # q_trial_norm を記録（slip consistent tangent 用、Phase C5）
    pair.state.q_trial_norm = q_trial_norm

    if q_trial_norm <= f_yield:
        # stick
        q = q_trial.copy()
        pair.state.z_t = q.copy()
        pair.state.stick = True
        pair.state.status = ContactStatus.ACTIVE
    else:
        # slip → radial return
        q = f_yield * q_trial / q_trial_norm
        pair.state.z_t = q.copy()
        pair.state.stick = False
        pair.state.status = ContactStatus.SLIDING

    # 散逸増分（非負であるべき）
    d_inc = float(np.dot(q, delta_ut))
    pair.state.dissipation = d_inc

    return q


def friction_tangent_2x2(
    pair: ContactPair,
    mu: float,
) -> np.ndarray:
    """摩擦の接線剛性（2×2 局所座標系）を返す.

    - stick: D_t = k_t * I₂
    - slip:  D_t = (μ * p_n / ||q_trial||) * k_t * (I₂ - q̂ ⊗ q̂)

    Phase C5 で slip consistent tangent を実装。
    v0.1 では slip 時も k_t * I₂ を使用していたが、
    v0.2 では正確な式で Newton 収束性を改善する。

    Args:
        pair: 接触ペア
        mu: 有効摩擦係数

    Returns:
        D_t: (2, 2) 摩擦接線剛性
    """
    k_t = pair.state.k_t
    p_n = pair.state.p_n

    if pair.state.status == ContactStatus.INACTIVE or p_n <= 0.0 or mu <= 0.0:
        return np.zeros((2, 2))

    if pair.state.stick:
        # stick: 完全弾性接線
        return k_t * np.eye(2)
    else:
        # slip: consistent tangent (Phase C5)
        # D_t = (μ*p_n / ||q_trial||) * k_t * (I₂ - q̂⊗q̂)
        z_t = pair.state.z_t
        z_norm = float(np.linalg.norm(z_t))
        q_tn = pair.state.q_trial_norm

        if z_norm < 1e-30 or q_tn < 1e-30:
            return k_t * np.eye(2)

        # q̂ = z_t / ||z_t|| = q_trial / ||q_trial||（同じ方向）
        q_hat = z_t / z_norm
        ratio = (mu * p_n) / q_tn
        return ratio * k_t * (np.eye(2) - np.outer(q_hat, q_hat))


def rotate_friction_history(
    z_t: np.ndarray,
    t1_old: np.ndarray,
    t2_old: np.ndarray,
    t1_new: np.ndarray,
    t2_new: np.ndarray,
) -> np.ndarray:
    """摩擦履歴ベクトルを旧フレームから新フレームに回転する.

    z_t は局所接線座標系 (t1, t2) での2成分ベクトル。
    接触フレームが回転した場合、物理的な摩擦力ベクトルは同じでも
    局所座標成分が変わるため、フレーム変換が必要。

    変換行列 R_{2×2}:
        R[i,j] = t_i_new · t_j_old  (i,j = 1,2)

    Args:
        z_t: (2,) 旧フレームでの摩擦履歴
        t1_old, t2_old: 旧接線基底 (3,)
        t1_new, t2_new: 新接線基底 (3,)

    Returns:
        z_t_new: (2,) 新フレームでの摩擦履歴
    """
    if float(np.linalg.norm(z_t)) < 1e-30:
        return z_t.copy()

    # 2×2 回転行列: 旧フレーム → 新フレーム
    R = np.array(
        [
            [float(t1_new @ t1_old), float(t1_new @ t2_old)],
            [float(t2_new @ t1_old), float(t2_new @ t2_old)],
        ]
    )
    return R @ z_t


def compute_mu_effective(
    mu_target: float,
    ramp_counter: int,
    mu_ramp_steps: int,
) -> float:
    """μランプを適用した有効摩擦係数を返す.

    μ_eff = μ_target * min(1, ramp_counter / mu_ramp_steps)

    Args:
        mu_target: 目標摩擦係数
        ramp_counter: 現在のランプカウンタ（Outerループ回数）
        mu_ramp_steps: ランプの総ステップ数（0 ならランプなし）

    Returns:
        mu_eff: 有効摩擦係数
    """
    if mu_ramp_steps <= 0:
        return mu_target
    ratio = min(1.0, ramp_counter / mu_ramp_steps)
    return mu_target * ratio
