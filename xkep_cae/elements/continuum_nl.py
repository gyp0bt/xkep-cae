"""幾何学的非線形定式化（連続体要素用）.

Total Lagrangian (TL) 定式化:
- Green-Lagrange ひずみ: E = 0.5*(F^T F - I)
- 第二 Piola-Kirchhoff 応力: S = D:E (Saint-Venant Kirchhoff)
- 内力: f_int = ∫ B_L^T S dV₀
- 接線剛性: K_T = K_mat + K_geo

Updated Lagrangian (UL) は参照配置を更新することで実現する。
各収束ステップ後に参照座標を現配置座標に更新し、変位をリセットする。

対応要素:
- Q4 双線形四角形要素（平面ひずみ）
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Gauss 積分点（2×2 フル積分）
# ---------------------------------------------------------------------------
_GP = 1.0 / np.sqrt(3.0)
_GAUSS_POINTS_2x2 = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]


# ---------------------------------------------------------------------------
# 形状関数微分（Q4）
# ---------------------------------------------------------------------------
def _q4_shape_deriv(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Q4 形状関数の自然座標微分 dN/dξ, dN/dη を返す."""
    dN_dxi = 0.25 * np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    )
    dN_deta = 0.25 * np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    )
    return dN_dxi, dN_deta


# ---------------------------------------------------------------------------
# Jacobian（参照配置）
# ---------------------------------------------------------------------------
def _jacobian_2d(
    dN_dxi: np.ndarray,
    dN_deta: np.ndarray,
    node_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """2D Jacobian を計算し、物理座標微分 dN/dX, dN/dY と detJ を返す.

    Jacobian の定義（quad4.py 準拠）:
      J = [[∂X/∂ξ, ∂X/∂η],
           [∂Y/∂ξ, ∂Y/∂η]]
    """
    J = np.empty((2, 2), dtype=float)
    J[0, 0] = dN_dxi @ node_xy[:, 0]
    J[0, 1] = dN_deta @ node_xy[:, 0]
    J[1, 0] = dN_dxi @ node_xy[:, 1]
    J[1, 1] = dN_deta @ node_xy[:, 1]

    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=float) / detJ

    dN_dX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    return dN_dX, dN_dY, detJ


# ---------------------------------------------------------------------------
# 変形勾配テンソル
# ---------------------------------------------------------------------------
def deformation_gradient_2d(
    dN_dX: np.ndarray,
    dN_dY: np.ndarray,
    u_elem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss 点での変形勾配 F = I + H を計算する.

    Args:
        dN_dX: (4,) 参照配置での dN_I/dX
        dN_dY: (4,) 参照配置での dN_I/dY
        u_elem: (8,) 要素変位 [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y]

    Returns:
        F: (2,2) 変形勾配テンソル
        H: (2,2) 変位勾配テンソル H_ij = ∂u_i/∂X_j
    """
    ux = u_elem[0::2]  # (4,)
    uy = u_elem[1::2]  # (4,)

    H = np.array(
        [
            [dN_dX @ ux, dN_dY @ ux],
            [dN_dX @ uy, dN_dY @ uy],
        ],
        dtype=float,
    )
    F = np.eye(2) + H
    return F, H


# ---------------------------------------------------------------------------
# Green-Lagrange ひずみ
# ---------------------------------------------------------------------------
def green_lagrange_strain_2d(F: np.ndarray) -> np.ndarray:
    """Green-Lagrange ひずみを Voigt 表記で返す.

    E = 0.5*(F^T F - I)

    Returns:
        E_voigt: (3,) [E₁₁, E₂₂, 2E₁₂] (engineering shear)
    """
    C = F.T @ F
    E11 = 0.5 * (C[0, 0] - 1.0)
    E22 = 0.5 * (C[1, 1] - 1.0)
    E12 = 0.5 * C[0, 1]  # C is symmetric
    return np.array([E11, E22, 2.0 * E12], dtype=float)


# ---------------------------------------------------------------------------
# 線形化 B 行列（GL ひずみ用）
# ---------------------------------------------------------------------------
def b_matrix_nl_2d(
    dN_dX: np.ndarray,
    dN_dY: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """GL ひずみの線形化 B 行列 B_L = B_0 + B_NL を構築する.

    B_L は変位の変分 δu を GL ひずみの変分 δE (Voigt) に写す:
      δE_voigt = B_L · δu_e    [δE₁₁, δE₂₂, 2δE₁₂]^T = B_L · δu_e

    Args:
        dN_dX: (4,) dN_I/dX
        dN_dY: (4,) dN_I/dY
        H: (2,2) 変位勾配テンソル

    Returns:
        B_L: (3, 8)
    """
    B = np.zeros((3, 8), dtype=float)
    for ni in range(4):
        c = 2 * ni  # column offset

        # --- 線形部分 B_0 ---
        B[0, c] = dN_dX[ni]  # δE₁₁ / δux
        B[1, c + 1] = dN_dY[ni]  # δE₂₂ / δuy
        B[2, c] = dN_dY[ni]  # 2δE₁₂ / δux
        B[2, c + 1] = dN_dX[ni]  # 2δE₁₂ / δuy

        # --- 非線形部分 B_NL ---
        # H[0,0] = ∂ux/∂X, H[0,1] = ∂ux/∂Y
        # H[1,0] = ∂uy/∂X, H[1,1] = ∂uy/∂Y
        B[0, c] += H[0, 0] * dN_dX[ni]
        B[0, c + 1] += H[1, 0] * dN_dX[ni]

        B[1, c] += H[0, 1] * dN_dY[ni]
        B[1, c + 1] += H[1, 1] * dN_dY[ni]

        B[2, c] += H[0, 0] * dN_dY[ni] + H[0, 1] * dN_dX[ni]
        B[2, c + 1] += H[1, 0] * dN_dY[ni] + H[1, 1] * dN_dX[ni]

    return B


# ---------------------------------------------------------------------------
# 幾何剛性行列（初期応力剛性）
# ---------------------------------------------------------------------------
def _geometric_stiffness_gp(
    dN_dX: np.ndarray,
    dN_dY: np.ndarray,
    S_voigt: np.ndarray,
) -> np.ndarray:
    """Gauss 点 1 点分の幾何剛性行列（重み・detJ は未乗算）.

    K_geo = G^T Σ G  （1 Gauss 点）

    2D の場合:
      K_geo(2I+p, 2J+q) = δ_pq · ∇N_I^T · S · ∇N_J

    Args:
        dN_dX: (4,) dN_I/dX
        dN_dY: (4,) dN_I/dY
        S_voigt: (3,) [S₁₁, S₂₂, S₁₂]

    Returns:
        K_geo: (8, 8)
    """
    S_mat = np.array(
        [[S_voigt[0], S_voigt[2]], [S_voigt[2], S_voigt[1]]],
        dtype=float,
    )

    K_geo = np.zeros((8, 8), dtype=float)
    for ni in range(4):
        grad_ni = np.array([dN_dX[ni], dN_dY[ni]])
        for nj in range(4):
            grad_nj = np.array([dN_dX[nj], dN_dY[nj]])
            scalar = grad_ni @ S_mat @ grad_nj
            K_geo[2 * ni, 2 * nj] += scalar
            K_geo[2 * ni + 1, 2 * nj + 1] += scalar

    return K_geo


# ===================================================================
# Q4 要素レベル関数
# ===================================================================


def quad4_nl_internal_force(
    node_xy_ref: np.ndarray,
    u_elem: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
) -> np.ndarray:
    """Q4 要素の幾何学的非線形内力ベクトル（TL 定式化）.

    f_int = ∫_V₀ B_L^T · S dV₀

    Args:
        node_xy_ref: (4,2) 参照配置の節点座標
        u_elem: (8,) 要素変位ベクトル
        D: (3,3) 弾性マトリクス（平面ひずみ, engineering shear）
        t: 厚み

    Returns:
        f_int: (8,) 内力ベクトル
    """
    f_int = np.zeros(8, dtype=float)

    for xi, eta in _GAUSS_POINTS_2x2:
        dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
        dN_dX, dN_dY, detJ = _jacobian_2d(dN_dxi, dN_deta, node_xy_ref)

        F, H = deformation_gradient_2d(dN_dX, dN_dY, u_elem)
        E_voigt = green_lagrange_strain_2d(F)
        S_voigt = D @ E_voigt

        B_L = b_matrix_nl_2d(dN_dX, dN_dY, H)
        f_int += B_L.T @ S_voigt * detJ * t

    return f_int


def quad4_nl_tangent_stiffness(
    node_xy_ref: np.ndarray,
    u_elem: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
) -> np.ndarray:
    """Q4 要素の幾何学的非線形接線剛性行列（TL 定式化）.

    K_T = K_mat + K_geo
    K_mat = ∫ B_L^T D B_L dV₀
    K_geo = ∫ G^T Σ G dV₀

    Args:
        node_xy_ref: (4,2) 参照配置の節点座標
        u_elem: (8,) 要素変位ベクトル
        D: (3,3) 弾性マトリクス（平面ひずみ, engineering shear）
        t: 厚み

    Returns:
        K_T: (8,8) 接線剛性行列
    """
    K_T = np.zeros((8, 8), dtype=float)

    for xi, eta in _GAUSS_POINTS_2x2:
        dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
        dN_dX, dN_dY, detJ = _jacobian_2d(dN_dxi, dN_deta, node_xy_ref)

        F, H = deformation_gradient_2d(dN_dX, dN_dY, u_elem)
        E_voigt = green_lagrange_strain_2d(F)
        S_voigt = D @ E_voigt

        B_L = b_matrix_nl_2d(dN_dX, dN_dY, H)

        # 材料剛性
        K_T += B_L.T @ D @ B_L * detJ * t
        # 幾何剛性
        K_T += _geometric_stiffness_gp(dN_dX, dN_dY, S_voigt) * detJ * t

    return K_T


def quad4_nl_force_and_stiffness(
    node_xy_ref: np.ndarray,
    u_elem: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Q4 要素の非線形内力と接線剛性を同時に計算する（NR 反復用）.

    Gauss 点ループを共有するため、個別に呼ぶよりも効率的。

    Args:
        node_xy_ref: (4,2) 参照配置の節点座標
        u_elem: (8,) 要素変位ベクトル
        D: (3,3) 弾性マトリクス（平面ひずみ, engineering shear）
        t: 厚み

    Returns:
        f_int: (8,) 内力ベクトル
        K_T: (8,8) 接線剛性行列
    """
    f_int = np.zeros(8, dtype=float)
    K_T = np.zeros((8, 8), dtype=float)

    for xi, eta in _GAUSS_POINTS_2x2:
        dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
        dN_dX, dN_dY, detJ = _jacobian_2d(dN_dxi, dN_deta, node_xy_ref)

        F, H = deformation_gradient_2d(dN_dX, dN_dY, u_elem)
        E_voigt = green_lagrange_strain_2d(F)
        S_voigt = D @ E_voigt

        B_L = b_matrix_nl_2d(dN_dX, dN_dY, H)

        # 内力
        f_int += B_L.T @ S_voigt * detJ * t
        # 材料剛性
        K_T += B_L.T @ D @ B_L * detJ * t
        # 幾何剛性
        K_T += _geometric_stiffness_gp(dN_dX, dN_dY, S_voigt) * detJ * t

    return f_int, K_T


# ===================================================================
# 全体アセンブリ（NR ソルバー用コールバック生成）
# ===================================================================


def make_nl_assembler_q4(
    nodes_ref: np.ndarray,
    connectivity: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
) -> tuple[
    # assemble_internal_force(u) -> f_int
    # assemble_tangent(u) -> K_T (CSR)
    object,
    object,
]:
    """Q4 要素群の非線形アセンブラを生成する.

    newton_raphson() ソルバーに渡すコールバック関数を返す。

    Args:
        nodes_ref: (n_nodes, 2) 参照配置の節点座標
        connectivity: (n_elems, 2 or more) 接続配列 [n1, n2, n3, n4]
        D: (3,3) 弾性マトリクス
        t: 厚み

    Returns:
        assemble_internal_force: u → f_int (ndof,)
        assemble_tangent: u → K_T (CSR)

    Usage::

        f_int_fn, K_T_fn = make_nl_assembler_q4(nodes, conn, D, t)
        result = newton_raphson(f_ext, fixed_dofs, K_T_fn, f_int_fn)
    """
    import scipy.sparse as sp

    n_nodes = len(nodes_ref)
    ndof = 2 * n_nodes

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        f_int = np.zeros(ndof, dtype=float)
        for elem_nodes in connectivity:
            nids = elem_nodes[:4].astype(int)
            coords = nodes_ref[nids]
            edofs = np.empty(8, dtype=int)
            for i, n in enumerate(nids):
                edofs[2 * i] = 2 * n
                edofs[2 * i + 1] = 2 * n + 1
            u_e = u[edofs]
            fe = quad4_nl_internal_force(coords, u_e, D, t)
            for ii in range(8):
                f_int[edofs[ii]] += fe[ii]
        return f_int

    def assemble_tangent(u: np.ndarray) -> sp.csr_matrix:
        rows_list = []
        cols_list = []
        data_list = []
        for elem_nodes in connectivity:
            nids = elem_nodes[:4].astype(int)
            coords = nodes_ref[nids]
            edofs = np.empty(8, dtype=int)
            for i, n in enumerate(nids):
                edofs[2 * i] = 2 * n
                edofs[2 * i + 1] = 2 * n + 1
            u_e = u[edofs]
            Ke = quad4_nl_tangent_stiffness(coords, u_e, D, t)
            rows_list.append(np.repeat(edofs, 8))
            cols_list.append(np.tile(edofs, 8))
            data_list.append(Ke.ravel())

        rows_arr = np.concatenate(rows_list)
        cols_arr = np.concatenate(cols_list)
        data_arr = np.concatenate(data_list)
        K = sp.csr_matrix((data_arr, (rows_arr, cols_arr)), shape=(ndof, ndof))
        K.sum_duplicates()
        return K

    return assemble_internal_force, assemble_tangent


def make_nl_assembler_q4_combined(
    nodes_ref: np.ndarray,
    connectivity: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
) -> tuple[object, object]:
    """効率的な統合アセンブラ（内力と剛性を同時に計算）.

    内部で quad4_nl_force_and_stiffness を使用し、
    Gauss 点計算を共有する。

    Returns:
        assemble_internal_force: u → f_int
        assemble_tangent: u → K_T (CSR)

    Note:
        同一変位ベクトルに対して assemble_internal_force と
        assemble_tangent が連続呼出しされる場合のみキャッシュが有効。
    """
    import scipy.sparse as sp

    n_nodes = len(nodes_ref)
    ndof = 2 * n_nodes

    # 最新の計算結果をキャッシュ
    _cache: dict = {"u_hash": None, "f_int": None, "K_T": None}

    def _assemble_both(u: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
        f_int = np.zeros(ndof, dtype=float)
        rows_list = []
        cols_list = []
        data_list = []

        for elem_nodes in connectivity:
            nids = elem_nodes[:4].astype(int)
            coords = nodes_ref[nids]
            edofs = np.empty(8, dtype=int)
            for i, n in enumerate(nids):
                edofs[2 * i] = 2 * n
                edofs[2 * i + 1] = 2 * n + 1
            u_e = u[edofs]
            fe, Ke = quad4_nl_force_and_stiffness(coords, u_e, D, t)

            for ii in range(8):
                f_int[edofs[ii]] += fe[ii]
            rows_list.append(np.repeat(edofs, 8))
            cols_list.append(np.tile(edofs, 8))
            data_list.append(Ke.ravel())

        rows_arr = np.concatenate(rows_list)
        cols_arr = np.concatenate(cols_list)
        data_arr = np.concatenate(data_list)
        K = sp.csr_matrix((data_arr, (rows_arr, cols_arr)), shape=(ndof, ndof))
        K.sum_duplicates()
        return f_int, K

    def _ensure_cache(u: np.ndarray) -> None:
        u_hash = u.data.tobytes()
        if _cache["u_hash"] != u_hash:
            f_int, K_T = _assemble_both(u)
            _cache["u_hash"] = u_hash
            _cache["f_int"] = f_int
            _cache["K_T"] = K_T

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        _ensure_cache(u)
        return _cache["f_int"]

    def assemble_tangent(u: np.ndarray) -> sp.csr_matrix:
        _ensure_cache(u)
        return _cache["K_T"]

    return assemble_internal_force, assemble_tangent
