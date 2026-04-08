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
    *,
    use_hermite: bool = False,
    node_tangents: np.ndarray | None = None,
    node_counts: np.ndarray | None = None,
    adj_node_map: dict | None = None,
) -> sp.csr_matrix:
    """摩擦の K_st（接触点滑り剛性）をバッチ計算で組み立て.

    f_fric = Σ_α q_α · G_tα の s,t 依存の連鎖微分:
        ∂f_fric/∂s = Σ_α q_α · ∂G_tα/∂s
        ∂f_fric/∂t = Σ_α q_α · ∂G_tα/∂t

    status-274: Hermite隣接ノードDOF拡張。
    status-310: ペアforループを排除しNumPyバッチ化。
    """
    zero = sp.csr_matrix((ndof_total, ndof_total))

    # ── アクティブペア抽出 ──
    n_pairs = len(contact_pairs)
    if n_pairs == 0:
        return zero

    # friction_forces_localに存在し、stateを持つペアを抽出
    act_indices = []
    for pair_idx, pair in enumerate(contact_pairs):
        if not hasattr(pair, "state"):
            continue
        if pair_idx not in friction_forces_local:
            continue
        q = friction_forces_local[pair_idx]
        if abs(q[0]) < 1e-30 and abs(q[1]) < 1e-30:
            continue
        act_indices.append(pair_idx)

    n_act = len(act_indices)
    if n_act == 0:
        return zero

    # ── ペアデータをNumPy配列に一括抽出 ──
    s_arr = np.zeros(n_act)
    t_arr = np.zeros(n_act)
    s_unc_arr = np.zeros(n_act)
    t_unc_arr = np.zeros(n_act)
    nodes = np.zeros((n_act, 4), dtype=int)
    elem_a_arr = np.full(n_act, -1, dtype=int)
    elem_b_arr = np.full(n_act, -1, dtype=int)
    q1_arr = np.zeros(n_act)
    q2_arr = np.zeros(n_act)
    t1_arr = np.zeros((n_act, 3))
    t2_arr = np.zeros((n_act, 3))

    for i, pair_idx in enumerate(act_indices):
        pair = contact_pairs[pair_idx]
        st = pair.state
        s_arr[i] = st.s
        t_arr[i] = st.t
        s_unc_arr[i] = getattr(st, "s_unclamped", st.s) or st.s
        t_unc_arr[i] = getattr(st, "t_unclamped", st.t) or st.t
        nodes[i, 0] = pair.nodes_a[0]
        nodes[i, 1] = pair.nodes_a[1]
        nodes[i, 2] = pair.nodes_b[0]
        nodes[i, 3] = pair.nodes_b[1]
        elem_a_arr[i] = getattr(pair, "elem_a", -1)
        elem_b_arr[i] = getattr(pair, "elem_b", -1)
        q = friction_forces_local[pair_idx]
        q1_arr[i] = q[0]
        q2_arr[i] = q[1]
        t1_arr[i] = st.tangent1
        t2_arr[i] = st.tangent2

    # ── 座標抽出 ──
    nc = node_coords
    xA0 = nc[nodes[:, 0]]
    xA1 = nc[nodes[:, 1]]
    xB0 = nc[nodes[:, 2]]
    xB1 = nc[nodes[:, 3]]

    # ── バッチ StJacobian ──
    ds_du_adj = None
    dt_du_adj = None
    if use_hermite and node_tangents is not None:
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_hermite,
        )

        mA0 = node_tangents[nodes[:, 0]]
        mA1 = node_tangents[nodes[:, 1]]
        mB0 = node_tangents[nodes[:, 2]]
        mB1 = node_tangents[nodes[:, 3]]
        dm_A_batch = None
        dm_B_batch = None
        dm_ext_A_batch = None
        dm_ext_B_batch = None
        if node_counts is not None:
            from xkep_cae.contact.contact_force.strategy import (
                HuberContactForceProcess,
            )

            dm_A_batch, dm_B_batch = HuberContactForceProcess._batch_dm_coeffs(node_counts, nodes)
            # 隣接ノード係数のバッチ計算（status-311）
            if adj_node_map is not None:
                c_a0 = np.maximum(node_counts[nodes[:, 0]], 1.0)
                c_a1 = np.maximum(node_counts[nodes[:, 1]], 1.0)
                c_b0 = np.maximum(node_counts[nodes[:, 2]], 1.0)
                c_b1 = np.maximum(node_counts[nodes[:, 3]], 1.0)
                dm_ext_A_batch = np.zeros((n_act, 2))
                dm_ext_B_batch = np.zeros((n_act, 2))
                dm_ext_A_batch[:, 0] = np.where(c_a0 >= 1.5, -1.0 / c_a0, 0.0)
                dm_ext_A_batch[:, 1] = np.where(c_a1 >= 1.5, 1.0 / c_a1, 0.0)
                dm_ext_B_batch[:, 0] = np.where(c_b0 >= 1.5, -1.0 / c_b0, 0.0)
                dm_ext_B_batch[:, 1] = np.where(c_b1 >= 1.5, 1.0 / c_b1, 0.0)
        ds_du, dt_du, valid, ds_du_adj, dt_du_adj = _batch_st_jacobian_hermite(
            xA0,
            xA1,
            xB0,
            xB1,
            s_arr,
            t_arr,
            s_unc_arr,
            t_unc_arr,
            mA0,
            mA1,
            mB0,
            mB1,
            dm_A_batch,
            dm_B_batch,
            dm_ext_A_batch,
            dm_ext_B_batch,
        )
    else:
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_linear,
        )

        ds_du, dt_du, valid = _batch_st_jacobian_linear(
            xA0,
            xA1,
            xB0,
            xB1,
            s_arr,
            t_arr,
            s_unc_arr,
            t_unc_arr,
        )

    # invalidペアを除外
    if not np.all(valid):
        keep = valid
        nodes = nodes[keep]
        elem_a_arr = elem_a_arr[keep]
        elem_b_arr = elem_b_arr[keep]
        q1_arr = q1_arr[keep]
        q2_arr = q2_arr[keep]
        t1_arr = t1_arr[keep]
        t2_arr = t2_arr[keep]
        ds_du = ds_du[keep]
        dt_du = dt_du[keep]
        if ds_du_adj is not None:
            ds_du_adj = ds_du_adj[keep]
            dt_du_adj = dt_du_adj[keep]
        n_act = int(np.sum(keep))

    if n_act == 0:
        return zero

    # ── df_ds, df_dt (N, 12) のバッチ計算 ──
    # dc_ds = [-1, 1, 0, 0], dc_dt = [0, 0, 1, -1]
    # df_ds[k*3+i] = Σ_α q_α * dc_ds[k] * tα[i]
    # q1*t1 + q2*t2 の各成分: (N, 3)
    qt_ds = q1_arr[:, None] * t1_arr + q2_arr[:, None] * t2_arr  # (N, 3)

    df_ds = np.zeros((n_act, 12))
    df_ds[:, 0:3] = -qt_ds  # k=0: dc_ds=-1
    df_ds[:, 3:6] = qt_ds  # k=1: dc_ds=+1
    # k=2,3: dc_ds=0 → ゼロのまま

    df_dt = np.zeros((n_act, 12))
    # k=0,1: dc_dt=0 → ゼロのまま
    df_dt[:, 6:9] = qt_ds  # k=2: dc_dt=+1
    df_dt[:, 9:12] = -qt_ds  # k=3: dc_dt=-1

    # ── K_st_local = outer(df_ds, ds_du) + outer(df_dt, dt_du): (N, 12, 12) ──
    K_st_local = np.einsum("ni,nj->nij", df_ds, ds_du) + np.einsum("ni,nj->nij", df_dt, dt_du)

    # ── DOF インデックス (N, 12) ──
    ndpn = ndof_per_node
    gdofs = np.zeros((n_act, 12), dtype=int)
    for k in range(4):
        for d in range(3):
            gdofs[:, k * 3 + d] = nodes[:, k] * ndpn + d

    # ── COO 構築 ──
    row_idx = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
    col_idx = np.broadcast_to(gdofs[:, None, :], (n_act, 12, 12)).ravel()
    val_arr = K_st_local.ravel()
    mask = np.abs(val_arr) > 1e-30
    rows_np = row_idx[mask]
    cols_np = col_idx[mask]
    vals_np = val_arr[mask]

    # ── 隣接ノードDOF拡張（status-311: バッチ化） ──
    adj_rows_np = np.array([], dtype=int)
    adj_cols_np = np.array([], dtype=int)
    adj_vals_np = np.array([], dtype=float)
    if ds_du_adj is not None and adj_node_map is not None and n_act > 0:
        # K_fric_adj = outer(df_ds, ds_du_adj) + outer(df_dt, dt_du_adj): (N, 12, 12)
        K_adj_local = np.einsum("ni,nj->nij", df_ds, ds_du_adj) + np.einsum(
            "ni,nj->nij", df_dt, dt_du_adj
        )

        # 隣接ノードグローバルインデックス (N, 4): [A-1, A+2, B-1, B+2]
        n_elem_max = max(adj_node_map.keys()) + 1 if adj_node_map else 0
        _adj_arr = np.full((n_elem_max, 2), -1, dtype=int)
        for elem_idx, (n0, n1) in adj_node_map.items():
            _adj_arr[elem_idx, 0] = n0
            _adj_arr[elem_idx, 1] = n1

        adj_nodes = np.full((n_act, 4), -1, dtype=int)
        valid_ea = (elem_a_arr >= 0) & (elem_a_arr < n_elem_max)
        valid_eb = (elem_b_arr >= 0) & (elem_b_arr < n_elem_max)
        if np.any(valid_ea):
            adj_nodes[valid_ea, 0] = _adj_arr[elem_a_arr[valid_ea], 0]
            adj_nodes[valid_ea, 1] = _adj_arr[elem_a_arr[valid_ea], 1]
        if np.any(valid_eb):
            adj_nodes[valid_eb, 2] = _adj_arr[elem_b_arr[valid_eb], 0]
            adj_nodes[valid_eb, 3] = _adj_arr[elem_b_arr[valid_eb], 1]

        # 隣接ノードが存在するペアのみ処理
        has_adj = np.any(adj_nodes >= 0, axis=1)
        if np.any(has_adj):
            adj_gdofs = np.zeros((n_act, 12), dtype=int)
            for adj_k in range(4):
                safe_node = np.maximum(adj_nodes[:, adj_k], 0)
                for d in range(3):
                    adj_gdofs[:, adj_k * 3 + d] = safe_node * ndpn + d

            # COO構築: rows=primary(12), cols=adj(12)
            adj_row_idx = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
            adj_col_idx = np.broadcast_to(adj_gdofs[:, None, :], (n_act, 12, 12)).ravel()
            adj_val_arr = K_adj_local.ravel()

            # adj_node < 0 のエントリをマスク
            adj_valid_mask = np.ones((n_act, 12, 12), dtype=bool)
            for adj_k in range(4):
                node_invalid = adj_nodes[:, adj_k] < 0  # (N,)
                adj_valid_mask[node_invalid, :, adj_k * 3 : adj_k * 3 + 3] = False

            combined_mask = adj_valid_mask.ravel() & (np.abs(adj_val_arr) > 1e-30)
            adj_rows_np = adj_row_idx[combined_mask]
            adj_cols_np = adj_col_idx[combined_mask]
            adj_vals_np = adj_val_arr[combined_mask]

    # 全COOを統合
    all_rows = np.concatenate([rows_np, adj_rows_np]) if len(adj_rows_np) > 0 else rows_np
    all_cols = np.concatenate([cols_np, adj_cols_np]) if len(adj_cols_np) > 0 else cols_np
    all_vals = np.concatenate([vals_np, adj_vals_np]) if len(adj_vals_np) > 0 else vals_np

    if len(all_vals) == 0:
        return zero
    return sp.coo_matrix(
        (all_vals, (all_rows, all_cols)),
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
