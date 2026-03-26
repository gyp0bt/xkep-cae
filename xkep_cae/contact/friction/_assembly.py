"""摩擦力アセンブリ — ヘルパー関数群.

プライベートモジュール（C16 準拠）。

主要関数:
- _compute_tangential_displacement: 接線相対変位増分
- _friction_return_mapping_loop: return mapping + 力アセンブリの統合ループ
- _assemble_friction_force: 局所摩擦力 → グローバルベクトル
- _assemble_friction_tangent_stiffness: 摩擦接線剛性行列
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact._types import ContactStatus
from xkep_cae.contact.friction.law_friction import (
    _return_mapping_core,
    _tangent_2x2_core,
)


def _contact_dofs(pair: object, ndof_per_node: int = 6) -> np.ndarray:
    """接触ペアの全体 DOF インデックス (4節点 × ndof_per_node)."""
    nodes = np.array(
        [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]],
        dtype=int,
    )
    offsets = np.arange(ndof_per_node, dtype=int)
    return (nodes[:, None] * ndof_per_node + offsets).ravel()


def _contact_tangent_shape_vector(
    pair: object, axis: int, *, use_hermite: bool = False
) -> np.ndarray:
    """接線方向形状ベクトル (12,).

    axis=0 → t1, axis=1 → t2.
    線形: A 側 (1-s)*ti, s*ti / B 側 -(1-t)*ti, -t*ti
    Hermite: A 側 H00(s)*ti, H01(s)*ti / B 側 -H00(t)*ti, -H01(t)*ti
    """
    s = pair.state.s
    t = pair.state.t
    ti = pair.state.tangent1 if axis == 0 else pair.state.tangent2

    if use_hermite:
        from xkep_cae.contact.contact_force.strategy import _hermite_shape_coeffs

        coeffs = _hermite_shape_coeffs(s, t)
    else:
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]

    g_t = np.zeros(12)
    for k in range(4):
        g_t[k * 3 : k * 3 + 3] = coeffs[k] * ti
    return g_t


def _compute_tangential_displacement(
    pair: object,
    u_cur: np.ndarray,
    u_ref: np.ndarray,
    ndof_per_node: int = 6,
    *,
    use_hermite: bool = False,
) -> np.ndarray:
    """接線相対変位増分 Δu_t (2,) を計算.

    線形: Δu_rel = [(1-t)(du_B0) + t(du_B1)] - [(1-s)(du_A0) + s(du_A1)]
    Hermite: Δu_rel = [H00(t)du_B0 + H01(t)du_B1] - [H00(s)du_A0 + H01(s)du_A1]
    Δu_t = [Δu_rel · t1, Δu_rel · t2]
    """
    s = pair.state.s
    t = pair.state.t
    t1 = pair.state.tangent1
    t2 = pair.state.tangent2

    du = u_cur - u_ref

    nA0, nA1 = pair.nodes_a
    nB0, nB1 = pair.nodes_b
    du_A0 = du[nA0 * ndof_per_node : nA0 * ndof_per_node + 3]
    du_A1 = du[nA1 * ndof_per_node : nA1 * ndof_per_node + 3]
    du_B0 = du[nB0 * ndof_per_node : nB0 * ndof_per_node + 3]
    du_B1 = du[nB1 * ndof_per_node : nB1 * ndof_per_node + 3]

    if use_hermite:
        from xkep_cae.contact.contact_force.strategy import _hermite_shape_coeffs

        coeffs = _hermite_shape_coeffs(s, t)
        du_A = coeffs[0] * du_A0 + coeffs[1] * du_A1
        du_B = (-coeffs[2]) * du_B0 + (-coeffs[3]) * du_B1
    else:
        du_A = (1.0 - s) * du_A0 + s * du_A1
        du_B = (1.0 - t) * du_B0 + t * du_B1
    du_rel = du_B - du_A

    return np.array([float(np.dot(du_rel, t1)), float(np.dot(du_rel, t2))])


def _assemble_friction_force(
    contact_pairs: list,
    friction_forces_local: dict[int, np.ndarray],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """局所摩擦力 → グローバル力ベクトルに組み立て（バッチ化版: status-246）."""
    f_friction = np.zeros(ndof_total)
    if not friction_forces_local:
        return f_friction

    # アクティブペアのデータをバッチ抽出
    idx_list = sorted(friction_forces_local.keys())
    n = len(idx_list)
    q_arr = np.zeros((n, 2))  # 摩擦力 (q1, q2)
    s_arr = np.zeros(n)
    t_arr = np.zeros(n)
    t1_arr = np.zeros((n, 3))
    t2_arr = np.zeros((n, 3))
    nodes_arr = np.zeros((n, 4), dtype=int)
    for j, idx in enumerate(idx_list):
        q_arr[j] = friction_forces_local[idx]
        pair = contact_pairs[idx]
        s_arr[j] = pair.state.s
        t_arr[j] = pair.state.t
        t1_arr[j] = pair.state.tangent1
        t2_arr[j] = pair.state.tangent2
        nodes_arr[j] = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]

    # 係数 (N, 4)
    coeffs = np.column_stack([1.0 - s_arr, s_arr, -(1.0 - t_arr), -t_arr])

    # f_local: (N, 12) = q1 * coeffs * t1 + q2 * coeffs * t2
    f_local = np.zeros((n, 12))
    for k in range(4):
        ck = coeffs[:, k : k + 1]  # (N, 1)
        f_local[:, k * 3 : k * 3 + 3] = ck * (q_arr[:, 0:1] * t1_arr + q_arr[:, 1:2] * t2_arr)

    # scatter
    for k in range(4):
        for d in range(3):
            gdofs = nodes_arr[:, k] * ndof_per_node + d
            np.add.at(f_friction, gdofs, f_local[:, k * 3 + d])

    return f_friction


def _assemble_friction_tangent_stiffness(
    contact_pairs: list,
    friction_tangents: dict[int, np.ndarray],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """摩擦接線剛性行列（材料項）バッチ化版（status-246）.

    K_fric = Σ D_t[a1,a2] * g_t[a1] ⊗ g_t[a2]
    """
    if not friction_tangents:
        return sp.csr_matrix((ndof_total, ndof_total))

    idx_list = sorted(friction_tangents.keys())
    n = len(idx_list)
    D_arr = np.zeros((n, 2, 2))
    s_arr = np.zeros(n)
    t_arr = np.zeros(n)
    t1_arr = np.zeros((n, 3))
    t2_arr = np.zeros((n, 3))
    nodes_arr = np.zeros((n, 4), dtype=int)
    for j, idx in enumerate(idx_list):
        D_arr[j] = friction_tangents[idx]
        pair = contact_pairs[idx]
        s_arr[j] = pair.state.s
        t_arr[j] = pair.state.t
        t1_arr[j] = pair.state.tangent1
        t2_arr[j] = pair.state.tangent2
        nodes_arr[j] = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]

    # 係数 (N, 4)
    coeffs = np.column_stack([1.0 - s_arr, s_arr, -(1.0 - t_arr), -t_arr])

    # g_t1, g_t2: (N, 12)
    g_t1 = np.zeros((n, 12))
    g_t2 = np.zeros((n, 12))
    for k in range(4):
        g_t1[:, k * 3 : k * 3 + 3] = coeffs[:, k : k + 1] * t1_arr
        g_t2[:, k * 3 : k * 3 + 3] = coeffs[:, k : k + 1] * t2_arr

    # K_local (N, 12, 12) = Σ_{a1,a2} D[a1,a2] * g[a1] ⊗ g[a2]
    g_list = [g_t1, g_t2]
    K_local = np.zeros((n, 12, 12))
    for a1 in range(2):
        for a2 in range(2):
            d_val = D_arr[:, a1, a2]  # (N,)
            mask = np.abs(d_val) > 1e-30
            if not mask.any():
                continue
            K_local[mask] += d_val[mask, None, None] * (
                g_list[a1][mask, :, None] * g_list[a2][mask, None, :]
            )

    # DOF インデックス (N, 12)
    gdofs = np.zeros((n, 12), dtype=int)
    for k in range(4):
        for d in range(3):
            gdofs[:, k * 3 + d] = nodes_arr[:, k] * ndof_per_node + d

    # COO 構築
    row_idx = np.broadcast_to(gdofs[:, :, None], (n, 12, 12)).ravel()
    col_idx = np.broadcast_to(gdofs[:, None, :], (n, 12, 12)).ravel()
    val_arr = K_local.ravel()
    mask = np.abs(val_arr) > 1e-30
    if not mask.any():
        return sp.csr_matrix((ndof_total, ndof_total))
    return sp.coo_matrix(
        (val_arr[mask], (row_idx[mask], col_idx[mask])),
        shape=(ndof_total, ndof_total),
    ).tocsr()


def _assemble_friction_geometric_stiffness(
    contact_pairs: list,
    friction_forces_local: dict[int, np.ndarray],
    ndof_total: int,
    ndof_per_node: int = 6,
    *,
    use_hermite: bool = False,
) -> sp.csr_matrix:
    """摩擦接線幾何剛性行列（バッチ化版: status-246）.

    M_{ij} = -q₁·n_i·t1_j + q₂·ε_{ijk}·t1_k - q₂·t2_i·n_j
    K_geo_fric = Σ_{ki,kj} c_ki·c_kj/dist · M
    """
    if not friction_forces_local:
        return sp.csr_matrix((ndof_total, ndof_total))

    # アクティブペアのデータ抽出
    idx_list = sorted(friction_forces_local.keys())
    n = len(idx_list)
    q1_arr = np.zeros(n)
    q2_arr = np.zeros(n)
    n_arr = np.zeros((n, 3))
    t1_arr = np.zeros((n, 3))
    t2_arr = np.zeros((n, 3))
    s_arr = np.zeros(n)
    t_arr = np.zeros(n)
    inv_dist_arr = np.zeros(n)
    nodes_arr = np.zeros((n, 4), dtype=int)
    valid = np.ones(n, dtype=bool)

    for j, idx in enumerate(idx_list):
        q = friction_forces_local[idx]
        q1_arr[j] = q[0]
        q2_arr[j] = q[1]
        if abs(q[0]) < 1e-30 and abs(q[1]) < 1e-30:
            valid[j] = False
            continue
        pair = contact_pairs[idx]
        dist = pair.state.gap + pair.radius_a + pair.radius_b
        if dist < 1e-15:
            valid[j] = False
            continue
        inv_dist_arr[j] = 1.0 / dist
        n_arr[j] = pair.state.normal
        t1_arr[j] = pair.state.tangent1
        t2_arr[j] = pair.state.tangent2
        s_arr[j] = pair.state.s
        t_arr[j] = pair.state.t
        nodes_arr[j] = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]

    if not valid.any():
        return sp.csr_matrix((ndof_total, ndof_total))

    # フィルタ
    q1 = q1_arr[valid]
    q2 = q2_arr[valid]
    nv = n_arr[valid]
    t1v = t1_arr[valid]
    t2v = t2_arr[valid]
    sv = s_arr[valid]
    tv = t_arr[valid]
    inv_d = inv_dist_arr[valid]
    nodes_v = nodes_arr[valid]
    nv_count = int(valid.sum())

    # M (N, 3, 3): -q1*n⊗t1 - q2*t2⊗n + q2*skew(t1)
    M = -(q1[:, None, None] * (nv[:, :, None] * t1v[:, None, :])) - (
        q2[:, None, None] * (t2v[:, :, None] * nv[:, None, :])
    )
    # skew(t1) の加算: ε_{ijk} t1_k
    M[:, 0, 1] += q2 * t1v[:, 2]
    M[:, 0, 2] -= q2 * t1v[:, 1]
    M[:, 1, 0] -= q2 * t1v[:, 2]
    M[:, 1, 2] += q2 * t1v[:, 0]
    M[:, 2, 0] += q2 * t1v[:, 1]
    M[:, 2, 1] -= q2 * t1v[:, 0]

    # 係数 (N, 4)
    if use_hermite:
        from xkep_cae.contact.contact_force.strategy import HuberContactForceProcess

        coeffs = HuberContactForceProcess._batch_hermite_coeffs(sv, tv)
    else:
        coeffs = np.column_stack([1.0 - sv, sv, -(1.0 - tv), -tv])

    # cc * inv_dist: (N, 4, 4)
    cc = coeffs[:, :, None] * coeffs[:, None, :]  # (N, 4, 4)
    w = cc * inv_d[:, None, None]  # (N, 4, 4)

    # K_local (N, 12, 12)
    K_local = np.zeros((nv_count, 12, 12))
    for ki in range(4):
        for kj in range(4):
            K_local[:, ki * 3 : (ki + 1) * 3, kj * 3 : (kj + 1) * 3] = (
                w[:, ki, kj][:, None, None] * M
            )

    # DOF + COO
    gdofs = np.zeros((nv_count, 12), dtype=int)
    for k in range(4):
        for d in range(3):
            gdofs[:, k * 3 + d] = nodes_v[:, k] * ndof_per_node + d

    row_idx = np.broadcast_to(gdofs[:, :, None], (nv_count, 12, 12)).ravel()
    col_idx = np.broadcast_to(gdofs[:, None, :], (nv_count, 12, 12)).ravel()
    val_arr = K_local.ravel()
    mask = np.abs(val_arr) > 1e-30
    if not mask.any():
        return sp.csr_matrix((ndof_total, ndof_total))
    return sp.coo_matrix(
        (val_arr[mask], (row_idx[mask], col_idx[mask])),
        shape=(ndof_total, ndof_total),
    ).tocsr()


def _assemble_friction_st_stiffness(
    contact_pairs: list,
    friction_forces_local: dict[int, np.ndarray],
    ndof_total: int,
    node_coords: np.ndarray,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """摩擦の K_st（接触点滑り剛性）を組み立て.

    f_fric = Σ_α q_α · G_tα の s,t 依存の連鎖微分:
        ∂f_fric/∂s = Σ_α q_α · ∂G_tα/∂s
        ∂f_fric/∂t = Σ_α q_α · ∂G_tα/∂t

    ∂G_tα/∂s の係数変化項: [-tα, tα, 0, 0]
    ∂G_tα/∂t の係数変化項: [0, 0, tα, -tα]
    """
    from xkep_cae.contact.geometry._st_jacobian import (
        ComputeStJacobianProcess,
        StJacobianInput,
    )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    st_proc = ComputeStJacobianProcess()

    for pair_idx, pair in enumerate(contact_pairs):
        if not hasattr(pair, "state"):
            continue
        # SDI 排除: INACTIVE skip を除去（status-233）。
        if pair_idx not in friction_forces_local:
            continue

        q = friction_forces_local[pair_idx]
        q1, q2 = float(q[0]), float(q[1])
        if abs(q1) < 1e-30 and abs(q2) < 1e-30:
            continue

        st = pair.state
        xA0 = node_coords[pair.nodes_a[0]]
        xA1 = node_coords[pair.nodes_a[1]]
        xB0 = node_coords[pair.nodes_b[0]]
        xB1 = node_coords[pair.nodes_b[1]]

        out = st_proc.process(StJacobianInput(xA0=xA0, xA1=xA1, xB0=xB0, xB1=xB1, s=st.s, t=st.t))
        if not out.valid:
            continue

        t1 = st.tangent1
        t2 = st.tangent2

        # ∂f_fric/∂s = Σ_α q_α · ∂G_tα/∂s
        # ∂G_tα/∂s の係数変化: dc_k/ds * tα_i
        dc_ds = [-1.0, 1.0, 0.0, 0.0]
        dc_dt = [0.0, 0.0, 1.0, -1.0]

        df_ds = np.zeros(12)
        df_dt = np.zeros(12)
        for _alpha, (qa, ta) in enumerate([(q1, t1), (q2, t2)]):
            if abs(qa) < 1e-30:
                continue
            for k in range(4):
                for i in range(3):
                    li = k * 3 + i
                    df_ds[li] += qa * dc_ds[k] * ta[i]
                    df_dt[li] += qa * dc_dt[k] * ta[i]

        # K_st_fric = outer(df_ds, ds_du) + outer(df_dt, dt_du)
        # 摩擦剛性は TangentAssembly で K_T - K_fric（符号反転）されるので
        # ここでは +df/d(s,t) · d(s,t)/du を返す
        K_local = np.outer(df_ds, out.ds_du) + np.outer(df_dt, out.dt_du)

        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        gdofs = np.empty(12, dtype=int)
        for k, node_id in enumerate(nodes):
            for d in range(3):
                gdofs[k * 3 + d] = node_id * ndof_per_node + d

        for li in range(12):
            gi = gdofs[li]
            for lj in range(12):
                gj = gdofs[lj]
                val = K_local[li, lj]
                if abs(val) > 1e-30:
                    rows.append(gi)
                    cols.append(gj)
                    data.append(val)

    if len(data) == 0:
        return sp.csr_matrix((ndof_total, ndof_total))

    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(ndof_total, ndof_total),
    ).tocsr()


def _friction_return_mapping_loop(
    contact_pairs: list,
    u: np.ndarray,
    u_ref: np.ndarray,
    ndof: int,
    ndof_per_node: int,
    k_pen: float,
    k_t_ratio: float,
    mu_eff: float,
    compute_p_n: callable,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """摩擦 return mapping ループの統合実装.

    各ペアで:
    1. compute_p_n() で法線力を取得
    2. 接線変位増分を計算
    3. Coulomb return mapping（純粋関数版）
    4. pair.state を更新
    5. 接線剛性を計算

    最後に assemble_friction_force() でグローバル力ベクトルを構築。

    Returns:
        (f_friction, friction_residual, friction_tangents, friction_forces_local)
    """
    friction_forces_local: dict[int, np.ndarray] = {}
    friction_tangents: dict[int, np.ndarray] = {}
    residuals: list[float] = []

    for i, pair in enumerate(contact_pairs):
        if not hasattr(pair, "state"):
            continue

        p_n = compute_p_n(i, pair)
        if p_n <= 0.0 or mu_eff <= 0.0:
            continue

        # ペナルティ剛性の初期化（未設定時）
        cur_state = pair.state
        if cur_state.k_pen <= 0.0:
            cur_state = _evolve_state(cur_state, k_pen=k_pen, k_t=k_pen * k_t_ratio)
            contact_pairs[i] = _evolve_pair(pair, state=cur_state)
            pair = contact_pairs[i]

        # 接線変位
        delta_ut = _compute_tangential_displacement(pair, u, u_ref, ndof_per_node)

        # Coulomb return mapping（純粋関数）
        q, is_stick, q_trial_norm, dissipation = _return_mapping_core(
            cur_state.z_t.copy(), delta_ut, cur_state.k_t, cur_state.p_n, mu_eff
        )

        # pair.state を更新
        contact_pairs[i] = _evolve_pair(
            pair,
            state=_evolve_state(
                cur_state,
                z_t=q.copy(),
                stick=is_stick,
                q_trial_norm=q_trial_norm,
                dissipation=dissipation,
                status=ContactStatus.ACTIVE if is_stick else ContactStatus.SLIDING,
            ),
        )
        pair = contact_pairs[i]

        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-30:
            continue

        residuals.append(max(0.0, q_norm - mu_eff * p_n))
        friction_forces_local[i] = q

        # 摩擦接線剛性（純粋関数）
        D_t = _tangent_2x2_core(
            pair.state.k_t, pair.state.p_n, mu_eff, pair.state.z_t, q_trial_norm, is_stick
        )
        friction_tangents[i] = D_t

    # グローバル力ベクトル組み立て
    f_friction = _assemble_friction_force(contact_pairs, friction_forces_local, ndof, ndof_per_node)
    friction_residual = np.array(residuals) if residuals else np.zeros(0)
    return f_friction, friction_residual, friction_tangents, friction_forces_local
