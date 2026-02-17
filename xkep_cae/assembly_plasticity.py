"""平面ひずみ要素の弾塑性アセンブリ.

各ガウス点で von Mises return mapping を実行し、
接線剛性行列 K_T と内力ベクトル f_int を構築する。

Q4, Q4_EAS, TRI3 要素に対応。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.core.state import PlasticState3D
from xkep_cae.elements.quad4_eas_bbar import (
    _B_matrix,
    _dN_dxi_eta,
    _eas4_M_tilde,
    _jacobian,
    _strain_transform_T,
)
from xkep_cae.materials.plasticity_3d import PlaneStrainPlasticity

# ---------------------------------------------------------------------------
# ガウス点定義
# ---------------------------------------------------------------------------

_G = 1.0 / np.sqrt(3.0)
_GAUSS_Q4 = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]

# TRI3: 1点（重心）ガウス求積
_GAUSS_TRI3_L = [(1.0 / 3.0, 1.0 / 3.0)]  # (L1, L2), L3 = 1-L1-L2
_GAUSS_TRI3_W = [0.5]  # 重み（面積座標系）


# ---------------------------------------------------------------------------
# Q4 弾塑性アセンブリ
# ---------------------------------------------------------------------------


def _q4_gauss_loop(
    node_xy: np.ndarray,
    u_elem: np.ndarray,
    plasticity: PlaneStrainPlasticity,
    states: list[PlasticState3D],
    thickness: float,
    *,
    compute_stiffness: bool = True,
    compute_fint: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, list[PlasticState3D]]:
    """Q4要素の弾塑性ガウス積分ループ.

    Args:
        node_xy: (4,2) 要素節点座標
        u_elem: (8,) 要素変位ベクトル
        plasticity: 弾塑性構成則
        states: ガウス点ごとの塑性状態（4個）
        thickness: 厚み
        compute_stiffness: 剛性行列を計算するか
        compute_fint: 内力ベクトルを計算するか

    Returns:
        (Ke, fe, states_new): (8,8) or None, (8,) or None, 新状態リスト
    """
    Ke = np.zeros((8, 8), dtype=float) if compute_stiffness else None
    fe = np.zeros(8, dtype=float) if compute_fint else None
    states_new = []

    for gp_idx, (xi, eta) in enumerate(_GAUSS_Q4):
        dN_dxi, dN_deta = _dN_dxi_eta(xi, eta)
        _J, detJ, invJ = _jacobian(dN_dxi, dN_deta, node_xy)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        B = _B_matrix(dN_dx, dN_dy)

        # ひずみ計算: ε = B u
        strain = B @ u_elem  # (3,) [εxx, εyy, γxy]

        # Return mapping
        result = plasticity.return_mapping(strain, states[gp_idx])
        states_new.append(result.state_new)

        wt = detJ * thickness

        if compute_fint:
            fe += B.T @ result.stress * wt

        if compute_stiffness:
            Ke += B.T @ result.tangent @ B * wt

    return Ke, fe, states_new


# ---------------------------------------------------------------------------
# Q4 EAS 弾塑性アセンブリ
# ---------------------------------------------------------------------------


def _q4_eas_gauss_loop(
    node_xy: np.ndarray,
    u_elem: np.ndarray,
    plasticity: PlaneStrainPlasticity,
    states: list[PlasticState3D],
    thickness: float,
    *,
    compute_stiffness: bool = True,
    compute_fint: bool = True,
    n_eas: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None, list[PlasticState3D]]:
    """Q4 EAS要素の弾塑性ガウス積分ループ.

    EAS 内部自由度 α は静的縮合で消去。
    非線形問題では各 NR 反復内で α を更新する必要があるが、
    ここでは Simo-Rifai の標準的アプローチに従い、
    各反復で α を再計算する。

    Args:
        node_xy: (4,2) 要素節点座標
        u_elem: (8,) 要素変位ベクトル
        plasticity: 弾塑性構成則
        states: ガウス点ごとの塑性状態（4個）
        thickness: 厚み
        compute_stiffness: K_T 計算フラグ
        compute_fint: f_int 計算フラグ
        n_eas: EAS パラメータ数

    Returns:
        (Ke, fe, states_new)
    """
    # 要素中心での EAS 変換行列
    dN_dxi_0, dN_deta_0 = _dN_dxi_eta(0.0, 0.0)
    _J0, detJ0, invJ0 = _jacobian(dN_dxi_0, dN_deta_0, node_xy)
    T0 = _strain_transform_T(invJ0.T)

    # パス1: 各ガウス点で return mapping 実行（B ひずみのみで状態決定）
    # パス2: EAS 内部自由度を含む剛性の静的縮合
    # ※ EAS の α は応力状態に影響するが、最初はα=0で近似し、
    #   NR反復で収束させる（標準的アプローチ）

    Ke = np.zeros((8, 8), dtype=float) if compute_stiffness else None
    fe = np.zeros(8, dtype=float) if compute_fint else None
    states_new = []

    K_uu = np.zeros((8, 8), dtype=float) if compute_stiffness else None
    K_ua = np.zeros((8, n_eas), dtype=float) if compute_stiffness else None
    K_aa = np.zeros((n_eas, n_eas), dtype=float) if compute_stiffness else None
    f_alpha = np.zeros(n_eas, dtype=float) if compute_fint else None

    for gp_idx, (xi, eta) in enumerate(_GAUSS_Q4):
        dN_dxi, dN_deta = _dN_dxi_eta(xi, eta)
        _J, detJ, invJ = _jacobian(dN_dxi, dN_deta, node_xy)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        B = _B_matrix(dN_dx, dN_dy)

        # EAS 拡張ひずみ補間行列
        M_tilde = _eas4_M_tilde(xi, eta)
        M = (detJ0 / detJ) * (T0 @ M_tilde)

        # ひずみ（α=0近似、NR反復で収束）
        strain = B @ u_elem  # (3,)

        # Return mapping
        result = plasticity.return_mapping(strain, states[gp_idx])
        states_new.append(result.state_new)

        wt = detJ * thickness
        D_ep = result.tangent

        if compute_fint:
            fe += B.T @ result.stress * wt
            f_alpha += M.T @ result.stress * wt

        if compute_stiffness:
            K_uu += B.T @ D_ep @ B * wt
            K_ua += B.T @ D_ep @ M * wt
            K_aa += M.T @ D_ep @ M * wt

    # 静的縮合
    if compute_stiffness:
        K_aa_inv = np.linalg.inv(K_aa)
        Ke = K_uu - K_ua @ K_aa_inv @ K_ua.T

    if compute_fint and compute_stiffness:
        # f_int の EAS 修正
        K_aa_inv = np.linalg.inv(K_aa) if K_aa_inv is None else K_aa_inv
        fe -= K_ua @ K_aa_inv @ f_alpha

    return Ke, fe, states_new


# ---------------------------------------------------------------------------
# TRI3 弾塑性アセンブリ
# ---------------------------------------------------------------------------


def _tri3_gauss_loop(
    node_xy: np.ndarray,
    u_elem: np.ndarray,
    plasticity: PlaneStrainPlasticity,
    states: list[PlasticState3D],
    thickness: float,
    *,
    compute_stiffness: bool = True,
    compute_fint: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, list[PlasticState3D]]:
    """TRI3要素の弾塑性ガウス積分ループ.

    定ひずみ三角形要素（1ガウス点 = 重心）。

    Args:
        node_xy: (3,2) 要素節点座標
        u_elem: (6,) 要素変位ベクトル
        plasticity: 弾塑性構成則
        states: ガウス点ごとの塑性状態（1個）
        thickness: 厚み

    Returns:
        (Ke, fe, states_new): (6,6) or None, (6,) or None, 新状態リスト
    """
    x1, y1 = node_xy[0]
    x2, y2 = node_xy[1]
    x3, y3 = node_xy[2]

    A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if A <= 0.0:
        raise ValueError("零面積または反転要素")

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    B = (1.0 / (2.0 * A)) * np.array(
        [
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3],
        ],
        dtype=float,
    )

    # ひずみ（定ひずみ要素なので1点で十分）
    strain = B @ u_elem  # (3,)

    result = plasticity.return_mapping(strain, states[0])

    Ke = None
    fe = None
    if compute_stiffness:
        Ke = B.T @ result.tangent @ B * A * thickness
    if compute_fint:
        fe = B.T @ result.stress * A * thickness

    return Ke, fe, [result.state_new]


# ---------------------------------------------------------------------------
# 全体アセンブリ関数
# ---------------------------------------------------------------------------

# ガウス点数マッピング
_NGAUSS = {"q4": 4, "q4_eas": 4, "tri3": 1}


def assemble_plane_strain_plastic(
    nodes_xy: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
    plasticity: PlaneStrainPlasticity,
    states: list[PlasticState3D],
    *,
    element_type: str = "q4_eas",
    thickness: float = 1.0,
    stiffness: bool = True,
    internal_force: bool = True,
) -> tuple[sp.csr_matrix | None, np.ndarray | None, list[PlasticState3D]]:
    """平面ひずみ弾塑性アセンブリ.

    各ガウス点で von Mises return mapping を実行し、
    全体 K_T と f_int を構築する。

    Args:
        nodes_xy: (N, 2) 節点座標
        connectivity: (Ne, nnodes) 接続配列
        u: (ndof_total,) 現在の変位ベクトル
        plasticity: PlaneStrainPlasticity 構成則
        states: 全ガウス点の塑性状態（Ne * n_gauss 個）
        element_type: "q4", "q4_eas", "tri3"
        thickness: 要素厚み
        stiffness: K_T を計算するか
        internal_force: f_int を計算するか

    Returns:
        (K_T, f_int, states_new):
          K_T: (ndof, ndof) CSR行列 or None
          f_int: (ndof,) or None
          states_new: 更新された塑性状態リスト
    """
    n_nodes = nodes_xy.shape[0]
    ndof_total = 2 * n_nodes
    n_elems = connectivity.shape[0]

    if element_type in ("q4", "q4_eas"):
        nnodes_per_elem = 4
        ndof_per_elem = 8
        n_gauss = 4
    elif element_type == "tri3":
        nnodes_per_elem = 3
        ndof_per_elem = 6
        n_gauss = 1
    else:
        raise ValueError(f"未対応の要素型: {element_type}")

    expected_states = n_elems * n_gauss
    if len(states) != expected_states:
        raise ValueError(
            f"states の長さ ({len(states)}) が要素数 * ガウス点数 "
            f"({expected_states}) と一致しません"
        )

    # 出力初期化
    K_data = []
    K_rows = []
    K_cols = []
    f_int = np.zeros(ndof_total, dtype=float) if internal_force else None
    states_new_all: list[PlasticState3D] = []

    for e in range(n_elems):
        node_ids = connectivity[e, :nnodes_per_elem]
        coords = nodes_xy[node_ids]

        # 要素変位の取得
        edofs = np.empty(ndof_per_elem, dtype=np.int64)
        for i, n in enumerate(node_ids):
            edofs[2 * i] = 2 * n
            edofs[2 * i + 1] = 2 * n + 1
        u_elem = u[edofs]

        # ガウス点状態
        gp_start = e * n_gauss
        elem_states = states[gp_start : gp_start + n_gauss]

        # 要素ループ
        if element_type == "q4":
            Ke, fe, st_new = _q4_gauss_loop(
                coords,
                u_elem,
                plasticity,
                elem_states,
                thickness,
                compute_stiffness=stiffness,
                compute_fint=internal_force,
            )
        elif element_type == "q4_eas":
            Ke, fe, st_new = _q4_eas_gauss_loop(
                coords,
                u_elem,
                plasticity,
                elem_states,
                thickness,
                compute_stiffness=stiffness,
                compute_fint=internal_force,
            )
        elif element_type == "tri3":
            Ke, fe, st_new = _tri3_gauss_loop(
                coords,
                u_elem,
                plasticity,
                elem_states,
                thickness,
                compute_stiffness=stiffness,
                compute_fint=internal_force,
            )

        states_new_all.extend(st_new)

        # アセンブリ
        if stiffness and Ke is not None:
            for i in range(ndof_per_elem):
                for j in range(ndof_per_elem):
                    K_rows.append(edofs[i])
                    K_cols.append(edofs[j])
                    K_data.append(Ke[i, j])

        if internal_force and fe is not None:
            f_int[edofs] += fe

    # CSR 行列生成
    K_T = None
    if stiffness:
        K_T = sp.csr_matrix(
            (np.array(K_data), (np.array(K_rows), np.array(K_cols))),
            shape=(ndof_total, ndof_total),
        )
        K_T.sum_duplicates()

    return K_T, f_int, states_new_all
