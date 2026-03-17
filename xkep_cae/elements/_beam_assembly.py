"""Corotational Timoshenko 3D 梁のグローバルアセンブリ.

assemble_cr_beam3d: COO/CSR 高速アセンブリ関数
_assemble_cr_beam3d_batch: ベクトル化バッチアセンブリ（解析的接線剛性専用）
"""

from __future__ import annotations

import numpy as np

from xkep_cae.elements._beam_cr import (
    _batch_build_local_axes,
    _batch_rodrigues_rotation,
    _batch_rotmat_to_rotvec,
    _batch_rotvec_to_rotmat,
    _batch_skew,
    _batch_tangent_operator,
    _batch_tangent_operator_inv,
    _batch_timo_ke_local,
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent,
    timo_beam3d_cr_tangent_analytical,
)


def assemble_cr_beam3d(
    nodes_init: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
    *,
    v_ref: np.ndarray | None = None,
    scf: float | None = None,
    stiffness: bool = True,
    internal_force: bool = True,
    sparse: bool = True,
    analytical_tangent: bool = True,
) -> tuple:
    """Corotational Timoshenko 3D 梁のグローバルアセンブリ（COO/CSR高速版）.

    Args:
        nodes_init: (n_nodes, 3) 初期節点座標
        connectivity: (n_elems, 2) 要素接続
        u: (ndof,) 全体変位ベクトル
        E, G: ヤング率、せん断弾性率
        A, Iy, Iz, J: 断面定数
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        scf: スレンダネス補償係数
        stiffness: 接線剛性行列を計算するか
        internal_force: 内力ベクトルを計算するか
        sparse: True → CSR行列、False → 密行列
        analytical_tangent: True → 解析的接線剛性

    Returns:
        (K_T, f_int): 接線剛性行列と内力ベクトル
    """
    n_nodes = len(nodes_init)
    n_elems = len(connectivity)
    ndof = 6 * n_nodes
    m = 12

    f_int_global = np.zeros(ndof, dtype=float) if internal_force else None

    conn_int = connectivity.astype(np.int64)
    dof_offsets = np.arange(6, dtype=np.int64)
    all_edofs = (conn_int[:, :, None] * 6 + dof_offsets[None, None, :]).reshape(n_elems, m)

    # --- sparse=False: 密行列 ---
    if stiffness and not sparse:
        K_T_dense = np.zeros((ndof, ndof), dtype=float)
        for i in range(n_elems):
            n1, n2 = int(conn_int[i, 0]), int(conn_int[i, 1])
            coords = nodes_init[np.array([n1, n2])]
            edofs = all_edofs[i]
            u_elem = u[edofs]
            if internal_force:
                f_e = timo_beam3d_cr_internal_force(
                    coords,
                    u_elem,
                    E,
                    G,
                    A,
                    Iy,
                    Iz,
                    J,
                    kappa_y,
                    kappa_z,
                    v_ref=v_ref,
                    scf=scf,
                )
                f_int_global[edofs] += f_e
            _tangent_fn = (
                timo_beam3d_cr_tangent_analytical if analytical_tangent else timo_beam3d_cr_tangent
            )
            K_e = _tangent_fn(
                coords,
                u_elem,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                v_ref=v_ref,
                scf=scf,
            )
            K_T_dense[np.ix_(edofs, edofs)] += K_e
        return K_T_dense, f_int_global

    # --- ベクトル化パス（解析的接線 + sparse） ---
    if analytical_tangent and n_elems > 0:
        batch_coo, batch_fint = _assemble_cr_beam3d_batch_impl(
            nodes_init,
            connectivity,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            v_ref=v_ref,
            scf=scf,
            stiffness=stiffness,
            internal_force=internal_force,
        )

        if internal_force and batch_fint is not None:
            f_int_global = batch_fint

        if stiffness and batch_coo is not None:
            import scipy.sparse as sp

            coo_rows = np.repeat(all_edofs, m, axis=1).ravel()
            coo_cols = np.tile(all_edofs, (1, m)).ravel()
            K_T_csr = sp.csr_matrix((batch_coo, (coo_rows, coo_cols)), shape=(ndof, ndof))
            K_T_csr.sum_duplicates()
            return K_T_csr, f_int_global

        return None, f_int_global

    # --- フォールバック: 数値微分用の逐次ループ ---
    if stiffness:
        import scipy.sparse as sp

        block_nnz = m * m
        total_nnz = n_elems * block_nnz
        coo_rows = np.repeat(all_edofs, m, axis=1).ravel()
        coo_cols = np.tile(all_edofs, (1, m)).ravel()
        coo_data = np.empty(total_nnz, dtype=float)

    for i in range(n_elems):
        n1, n2 = int(conn_int[i, 0]), int(conn_int[i, 1])
        coords = nodes_init[np.array([n1, n2])]
        edofs = all_edofs[i]
        u_elem = u[edofs]
        if internal_force:
            f_e = timo_beam3d_cr_internal_force(
                coords,
                u_elem,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                v_ref=v_ref,
                scf=scf,
            )
            f_int_global[edofs] += f_e
        if stiffness:
            K_e = timo_beam3d_cr_tangent(
                coords,
                u_elem,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                v_ref=v_ref,
                scf=scf,
            )
            offset = i * block_nnz
            coo_data[offset : offset + block_nnz] = K_e.ravel()

    if stiffness:
        K_T_csr = sp.csr_matrix((coo_data, (coo_rows, coo_cols)), shape=(ndof, ndof))
        K_T_csr.sum_duplicates()
        return K_T_csr, f_int_global

    return None, f_int_global


def _assemble_cr_beam3d_batch_impl(
    nodes_init: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
    *,
    v_ref: np.ndarray | None = None,
    scf: float | None = None,
    stiffness: bool = True,
    internal_force: bool = True,
) -> tuple:
    """ベクトル化された CR Timoshenko 3D 梁アセンブリ."""
    conn_int = connectivity.astype(np.int64)
    n_elems = len(conn_int)
    n_nodes = len(nodes_init)
    ndof = 6 * n_nodes

    coords_all = nodes_init[conn_int]
    m = 12
    dof_offsets = np.arange(6, dtype=np.int64)
    all_edofs = (conn_int[:, :, None] * 6 + dof_offsets[None, None, :]).reshape(n_elems, m)
    u_all = u[all_edofs]

    dx0 = coords_all[:, 1] - coords_all[:, 0]
    L_0 = np.linalg.norm(dx0, axis=1)
    e_x_0 = dx0 / L_0[:, None]

    R_0 = _batch_build_local_axes(e_x_0, v_ref)

    x1_def = coords_all[:, 0] + u_all[:, 0:3]
    x2_def = coords_all[:, 1] + u_all[:, 6:9]
    dx_def = x2_def - x1_def
    L_def = np.linalg.norm(dx_def, axis=1)
    e_x_def = dx_def / L_def[:, None]

    R_rod = _batch_rodrigues_rotation(e_x_0, e_x_def)
    R_cr = np.einsum("nij,nkj->nik", R_0, R_rod)

    R_node1 = _batch_rotvec_to_rotmat(u_all[:, 3:6])
    R_node2 = _batch_rotvec_to_rotmat(u_all[:, 9:12])

    R_0_T = np.einsum("nji->nij", R_0)
    R_def1 = np.einsum("nij,njk,nkl->nil", R_cr, R_node1, R_0_T)
    R_def2 = np.einsum("nij,njk,nkl->nil", R_cr, R_node2, R_0_T)
    theta_def1 = _batch_rotmat_to_rotvec(R_def1)
    theta_def2 = _batch_rotmat_to_rotvec(R_def2)

    d_cr = np.zeros((n_elems, 12), dtype=float)
    d_cr[:, 3:6] = theta_def1
    d_cr[:, 6] = L_def - L_0
    d_cr[:, 9:12] = theta_def2

    Ke_local = _batch_timo_ke_local(E, G, A, Iy, Iz, J, L_0, kappa_y, kappa_z, scf=scf)
    f_cr = np.einsum("nij,nj->ni", Ke_local, d_cr)

    R_cr_T = np.einsum("nji->nij", R_cr)
    f_int_global = None
    if internal_force:
        f_global = np.empty((n_elems, 12), dtype=float)
        for blk in range(4):
            s = 3 * blk
            f_global[:, s : s + 3] = np.einsum("nij,nj->ni", R_cr_T, f_cr[:, s : s + 3])
        f_int_global = np.zeros(ndof, dtype=float)
        np.add.at(f_int_global, all_edofs.ravel(), f_global.ravel())

    coo_data = None
    if stiffness:
        S_ex = _batch_skew(e_x_def)
        R_cr_S_ex = np.einsum("nij,njk->nik", R_cr, S_ex)
        inv_L = 1.0 / L_def
        dpsi_du1 = inv_L[:, None, None] * R_cr_S_ex
        dpsi_du2 = -dpsi_du1

        T_inv1 = _batch_tangent_operator_inv(theta_def1)
        T_inv2 = _batch_tangent_operator_inv(theta_def2)
        T_s1 = _batch_tangent_operator(u_all[:, 3:6])
        T_s2 = _batch_tangent_operator(u_all[:, 9:12])

        B = np.zeros((n_elems, 12, 12), dtype=float)
        B[:, 6, 0:3] = -e_x_def
        B[:, 6, 6:9] = e_x_def
        B[:, 3:6, 0:3] = np.einsum("nij,njk->nik", T_inv1, dpsi_du1)
        B[:, 3:6, 6:9] = np.einsum("nij,njk->nik", T_inv1, dpsi_du2)
        B[:, 3:6, 3:6] = np.einsum("nij,njk,nkl->nil", T_inv1, R_cr, T_s1)
        B[:, 9:12, 0:3] = np.einsum("nij,njk->nik", T_inv2, dpsi_du1)
        B[:, 9:12, 6:9] = np.einsum("nij,njk->nik", T_inv2, dpsi_du2)
        B[:, 9:12, 9:12] = np.einsum("nij,njk,nkl->nil", T_inv2, R_cr, T_s2)

        KB = np.einsum("nij,njk->nik", Ke_local, B)
        K_mat = np.empty((n_elems, 12, 12), dtype=float)
        for blk in range(4):
            s = 3 * blk
            K_mat[:, s : s + 3, :] = np.einsum("nij,njk->nik", R_cr_T, KB[:, s : s + 3, :])

        K_geo = np.zeros((n_elems, 12, 12), dtype=float)
        for blk in range(4):
            s = 3 * blk
            f_blk = f_cr[:, s : s + 3]
            Sf = _batch_skew(f_blk)
            RtSf = np.einsum("nij,njk->nik", R_cr_T, Sf)
            K_geo[:, s : s + 3, 0:3] += np.einsum("nij,njk->nik", RtSf, dpsi_du1)
            K_geo[:, s : s + 3, 6:9] += np.einsum("nij,njk->nik", RtSf, dpsi_du2)

        K_T = K_mat + K_geo
        K_T = 0.5 * (K_T + np.einsum("nij->nji", K_T))
        coo_data = K_T.reshape(-1)

    return coo_data, f_int_global
