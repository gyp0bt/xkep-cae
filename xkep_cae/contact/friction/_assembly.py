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


def _contact_tangent_shape_vector(pair: object, axis: int) -> np.ndarray:
    """接線方向形状ベクトル (12,).

    axis=0 → t1, axis=1 → t2.
    A 側: (1-s)*ti, s*ti  /  B 側: -(1-t)*ti, -t*ti
    """
    s = pair.state.s
    t = pair.state.t
    ti = pair.state.tangent1 if axis == 0 else pair.state.tangent2

    g_t = np.zeros(12)
    g_t[0:3] = (1.0 - s) * ti
    g_t[3:6] = s * ti
    g_t[6:9] = -(1.0 - t) * ti
    g_t[9:12] = -t * ti
    return g_t


def _compute_tangential_displacement(
    pair: object,
    u_cur: np.ndarray,
    u_ref: np.ndarray,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """接線相対変位増分 Δu_t (2,) を計算.

    Δu_rel = [(1-t)(du_B0) + t(du_B1)] - [(1-s)(du_A0) + s(du_A1)]
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
    """局所摩擦力 → グローバル力ベクトルに組み立て."""
    f_friction = np.zeros(ndof_total)

    for pair_idx, q in friction_forces_local.items():
        pair = contact_pairs[pair_idx]
        dofs = _contact_dofs(pair, ndof_per_node)
        for axis in range(2):
            if abs(q[axis]) < 1e-30:
                continue
            g_t = _contact_tangent_shape_vector(pair, axis)
            for k in range(4):
                for d in range(3):
                    f_friction[dofs[k * ndof_per_node + d]] += q[axis] * g_t[k * 3 + d]

    return f_friction


def _assemble_friction_tangent_stiffness(
    contact_pairs: list,
    friction_tangents: dict[int, np.ndarray],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """摩擦接線剛性行列（材料項）を COO 形式で組み立て → CSR 変換."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pair_idx, pair in enumerate(contact_pairs):
        if not hasattr(pair, "state"):
            continue
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        if pair_idx not in friction_tangents:
            continue

        D_t = friction_tangents[pair_idx]
        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        gdofs = np.empty(12, dtype=int)
        for k, node in enumerate(nodes):
            for d in range(3):
                gdofs[k * 3 + d] = node * ndof_per_node + d

        g_t = [
            _contact_tangent_shape_vector(pair, 0),
            _contact_tangent_shape_vector(pair, 1),
        ]
        for a1 in range(2):
            for a2 in range(2):
                d_val = D_t[a1, a2]
                if abs(d_val) < 1e-30:
                    continue
                for i in range(12):
                    for j in range(12):
                        val = d_val * g_t[a1][i] * g_t[a2][j]
                        if abs(val) > 1e-30:
                            rows.append(gdofs[i])
                            cols.append(gdofs[j])
                            data.append(val)

    if len(data) == 0:
        return sp.csr_matrix((ndof_total, ndof_total))

    return sp.coo_matrix(
        (data, (rows, cols)),
        shape=(ndof_total, ndof_total),
    ).tocsr()


def _assemble_friction_geometric_stiffness(
    contact_pairs: list,
    friction_forces_local: dict[int, np.ndarray],
    ndof_total: int,
    ndof_per_node: int = 6,
) -> sp.csr_matrix:
    """摩擦接線幾何剛性行列（接線方向回転項）を COO 形式で組み立て.

    摩擦力 f_fric = q₁·G_t1 + q₂·G_t2 に対して、
    G_tα = c_k·tα（接線方向形状ベクトル）が変形に伴い回転する項。

    df_fric_{k,i}/du_{l,j} = c_k·c_l/dist · M_{ij}

    M_{ij} = -q₁·n_i·t1_j + q₂·ε_{ijk}·t1_k - q₂·t2_i·n_j

    ここで:
    - dt1/dn = -n⊗t1（法線変化に対する t1 の追従）
    - dt2/dn·P_perp = ε_{ijk}·t1_k - t2_i·n_j（t2 = n×t1 の連鎖微分 × 法線射影）
    - dn/du = c_l/dist · (I - n⊗n)
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pair_idx, pair in enumerate(contact_pairs):
        if not hasattr(pair, "state"):
            continue
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        if pair_idx not in friction_forces_local:
            continue

        q = friction_forces_local[pair_idx]
        q1, q2 = float(q[0]), float(q[1])
        if abs(q1) < 1e-30 and abs(q2) < 1e-30:
            continue

        n = pair.state.normal
        t1 = pair.state.tangent1
        t2 = pair.state.tangent2
        s = pair.state.s
        t = pair.state.t

        # dist = gap + r_A + r_B（接触間距離）
        dist = pair.state.gap + pair.radius_a + pair.radius_b
        if dist < 1e-15:
            continue

        inv_dist = 1.0 / dist
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]

        # M_{ij} = -q₁·n_i·t1_j + q₂·ε_{ijk}·t1_k - q₂·t2_i·n_j
        # ε_{ijk}·t1_k: skew-symmetric part (δn × t1 の連鎖微分)
        # -t2_i·n_j: P_perp 射影による補正項
        M = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                M[i, j] = -q1 * n[i] * t1[j] - q2 * t2[i] * n[j]
        # skew part: ε_{ijk}·t1_k のエントリ
        M[0, 1] += q2 * t1[2]
        M[0, 2] += q2 * (-t1[1])
        M[1, 0] += q2 * (-t1[2])
        M[1, 2] += q2 * t1[0]
        M[2, 0] += q2 * t1[1]
        M[2, 1] += q2 * (-t1[0])

        nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
        gdofs = np.empty(12, dtype=int)
        for k, node in enumerate(nodes):
            for d in range(3):
                gdofs[k * 3 + d] = node * ndof_per_node + d

        for ki in range(4):
            ci = coeffs[ki]
            if abs(ci) < 1e-30:
                continue
            for kj in range(4):
                cj = coeffs[kj]
                if abs(cj) < 1e-30:
                    continue
                w = ci * cj * inv_dist
                for di in range(3):
                    gi = gdofs[ki * 3 + di]
                    for dj in range(3):
                        gj = gdofs[kj * 3 + dj]
                        val = w * M[di, dj]
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
        if pair.state.status == ContactStatus.INACTIVE:
            continue
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
