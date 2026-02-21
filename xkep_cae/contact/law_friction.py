"""Coulomb 摩擦則（return mapping）.

Phase C3: 接線方向の摩擦力計算（Coulomb return mapping）。

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

    - stick: D_t = k_t * I_2
    - slip:  D_t = (μ * p_n / ||q_trial||) * (I_2 - q̂ ⊗ q̂) * k_t
             ただし v0.1 では近似として k_t * I_2 を使用

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
        # slip: consistent tangent
        z_t = pair.state.z_t
        z_norm = float(np.linalg.norm(z_t))
        if z_norm < 1e-30:
            return k_t * np.eye(2)

        # slip の consistent tangent: (μ*p_n) * k_t / ||q_trial|| * (I - q̂⊗q̂)
        # v0.1 では近似として stick 同等の k_t*I を使用（過大推定だが安定）
        return k_t * np.eye(2)


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
