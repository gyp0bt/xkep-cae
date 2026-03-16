"""幾何学的非線形定式化（連続体要素用）.

Total Lagrangian (TL) 定式化:
- Green-Lagrange ひずみ: E = 0.5*(F^T F - I)
- 第二 Piola-Kirchhoff 応力: S = D:E (Saint-Venant Kirchhoff)
- 内力: f_int = ∫ B_L^T S dV₀
- 接線剛性: K_T = K_mat + K_geo

Updated Lagrangian (UL) 定式化:
- TL定式化と同じ要素計算を再利用
- 各収束ステップ後に参照配置を現配置に更新し、増分変位をリセット
- 全体変位は増分変位の累積で管理
- 各ステップの増分が小さいため数値精度が向上する

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


# ===================================================================
# Updated Lagrangian (UL) アセンブラ
# ===================================================================


class ULAssemblerQ4:
    """Q4 要素群の Updated Lagrangian アセンブラ.

    各収束ステップ後に参照配置を現配置に更新し、Gauss点に蓄積応力を
    保持することで正確な UL を実現する。

    応力追跡:
    - 各 Gauss 点に蓄積 Cauchy 応力 σ_stored を保持
    - 内力: f_int = ∫ B_L^T (σ_stored + D:ΔE) dV
    - 参照配置更新時: σ_stored ← Cauchy stress from converged state

    Usage::

        ul = ULAssemblerQ4(nodes_init, conn, D, t)
        result = newton_raphson_ul(f_ext_total, fixed_dofs, ul, ...)
    """

    def __init__(
        self,
        nodes_init: np.ndarray,
        connectivity: np.ndarray,
        D: np.ndarray,
        t: float = 1.0,
    ) -> None:
        import scipy.sparse as sp

        self._sp = sp
        self._nodes_ref = nodes_init.copy()
        self._nodes_init = nodes_init.copy()
        self._connectivity = connectivity.copy()
        self._D = D.copy()
        self._t = t
        self._n_nodes = len(nodes_init)
        self._ndof = 2 * self._n_nodes
        self._n_elems = len(connectivity)
        self._n_gp = 4  # Q4: 2×2 Gauss 積分

        # 累積変位（初期配置からの全変位）
        self._u_total = np.zeros(self._ndof, dtype=float)

        # Gauss 点蓄積応力 [n_elems, n_gp, 3] → [σ₁₁, σ₂₂, σ₁₂] (Voigt)
        self._stress_stored = np.zeros((self._n_elems, self._n_gp, 3), dtype=float)

    @property
    def nodes_ref(self) -> np.ndarray:
        """現在の参照配置の節点座標."""
        return self._nodes_ref.copy()

    @property
    def u_total(self) -> np.ndarray:
        """初期配置からの累積変位."""
        return self._u_total.copy()

    @property
    def ndof(self) -> int:
        """全自由度数."""
        return self._ndof

    def _element_internal_force(
        self, coords_ref: np.ndarray, u_e: np.ndarray, stress_stored: np.ndarray
    ) -> np.ndarray:
        """要素内力ベクトル（蓄積応力込み）.

        f_int = ∫ B_L^T · S_total · dV
        S_total = σ_stored + D:E_inc (線形化近似)
        """
        f_int = np.zeros(8, dtype=float)
        for gp_idx, (xi, eta) in enumerate(_GAUSS_POINTS_2x2):
            dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
            dN_dX, dN_dY, detJ = _jacobian_2d(dN_dxi, dN_deta, coords_ref)
            F, H = deformation_gradient_2d(dN_dX, dN_dY, u_e)
            E_voigt = green_lagrange_strain_2d(F)
            S_inc = self._D @ E_voigt
            S_total = stress_stored[gp_idx] + S_inc
            B_L = b_matrix_nl_2d(dN_dX, dN_dY, H)
            f_int += B_L.T @ S_total * detJ * self._t
        return f_int

    def _element_tangent(
        self, coords_ref: np.ndarray, u_e: np.ndarray, stress_stored: np.ndarray
    ) -> np.ndarray:
        """要素接線剛性行列（蓄積応力込み）.

        K_T = K_mat + K_geo
        K_mat = ∫ B_L^T D B_L dV  (増分の材料剛性)
        K_geo = ∫ G^T Σ_total G dV  (全応力の幾何剛性)
        """
        K_T = np.zeros((8, 8), dtype=float)
        for gp_idx, (xi, eta) in enumerate(_GAUSS_POINTS_2x2):
            dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
            dN_dX, dN_dY, detJ = _jacobian_2d(dN_dxi, dN_deta, coords_ref)
            F, H = deformation_gradient_2d(dN_dX, dN_dY, u_e)
            E_voigt = green_lagrange_strain_2d(F)
            S_inc = self._D @ E_voigt
            S_total = stress_stored[gp_idx] + S_inc
            B_L = b_matrix_nl_2d(dN_dX, dN_dY, H)
            K_T += B_L.T @ self._D @ B_L * detJ * self._t
            K_T += _geometric_stiffness_gp(dN_dX, dN_dY, S_total) * detJ * self._t
        return K_T

    def assemble_internal_force(self, u_inc: np.ndarray) -> np.ndarray:
        """増分変位 u_inc に対する内力ベクトル（蓄積応力込み）."""
        f_int = np.zeros(self._ndof, dtype=float)
        for e_idx, elem_nodes in enumerate(self._connectivity):
            nids = elem_nodes[:4].astype(int)
            coords = self._nodes_ref[nids]
            edofs = np.empty(8, dtype=int)
            for i, n in enumerate(nids):
                edofs[2 * i] = 2 * n
                edofs[2 * i + 1] = 2 * n + 1
            u_e = u_inc[edofs]
            fe = self._element_internal_force(coords, u_e, self._stress_stored[e_idx])
            for ii in range(8):
                f_int[edofs[ii]] += fe[ii]
        return f_int

    def assemble_tangent(self, u_inc: np.ndarray):
        """増分変位 u_inc に対する接線剛性行列（蓄積応力込み）."""
        sp = self._sp
        rows_list = []
        cols_list = []
        data_list = []
        for e_idx, elem_nodes in enumerate(self._connectivity):
            nids = elem_nodes[:4].astype(int)
            coords = self._nodes_ref[nids]
            edofs = np.empty(8, dtype=int)
            for i, n in enumerate(nids):
                edofs[2 * i] = 2 * n
                edofs[2 * i + 1] = 2 * n + 1
            u_e = u_inc[edofs]
            Ke = self._element_tangent(coords, u_e, self._stress_stored[e_idx])
            rows_list.append(np.repeat(edofs, 8))
            cols_list.append(np.tile(edofs, 8))
            data_list.append(Ke.ravel())

        rows_arr = np.concatenate(rows_list)
        cols_arr = np.concatenate(cols_list)
        data_arr = np.concatenate(data_list)
        K = sp.csr_matrix((data_arr, (rows_arr, cols_arr)), shape=(self._ndof, self._ndof))
        K.sum_duplicates()
        return K

    def update_reference(self, u_inc: np.ndarray) -> None:
        """参照配置を更新し、蓄積応力を保存する.

        1. 収束した増分変位から Cauchy 応力を計算
        2. 蓄積応力を更新: σ_stored ← Cauchy stress
        3. 参照座標を現配置に更新: X_ref ← X_ref + Δu
        4. 累積変位を更新: u_total += Δu

        Args:
            u_inc: 収束した増分変位ベクトル
        """
        # 1. 各 Gauss 点の Cauchy 応力を計算・保存
        for e_idx, elem_nodes in enumerate(self._connectivity):
            nids = elem_nodes[:4].astype(int)
            coords = self._nodes_ref[nids]
            edofs = np.empty(8, dtype=int)
            for i, n in enumerate(nids):
                edofs[2 * i] = 2 * n
                edofs[2 * i + 1] = 2 * n + 1
            u_e = u_inc[edofs]

            for gp_idx, (xi, eta) in enumerate(_GAUSS_POINTS_2x2):
                dN_dxi, dN_deta = _q4_shape_deriv(xi, eta)
                dN_dX, dN_dY, _ = _jacobian_2d(dN_dxi, dN_deta, coords)
                F, _ = deformation_gradient_2d(dN_dX, dN_dY, u_e)
                E_voigt = green_lagrange_strain_2d(F)
                S_inc = self._D @ E_voigt
                S_total = self._stress_stored[e_idx, gp_idx] + S_inc

                # S (2nd PK) → σ (Cauchy): σ = (1/J) F·S·F^T
                J = np.linalg.det(F)
                S_mat = np.array(
                    [
                        [S_total[0], S_total[2]],
                        [S_total[2], S_total[1]],
                    ]
                )
                sigma_mat = (1.0 / J) * F @ S_mat @ F.T

                # 新しい参照配置での蓄積応力 = Cauchy 応力
                # (新しい参照 = 現配置なので F_new = I, S_new = σ)
                self._stress_stored[e_idx, gp_idx] = np.array(
                    [sigma_mat[0, 0], sigma_mat[1, 1], sigma_mat[0, 1]]
                )

        # 2. 参照座標を更新
        for i in range(self._n_nodes):
            self._nodes_ref[i, 0] += u_inc[2 * i]
            self._nodes_ref[i, 1] += u_inc[2 * i + 1]

        # 3. 累積変位を更新
        self._u_total += u_inc

    def reset(self) -> None:
        """参照配置を初期配置にリセットする."""
        self._nodes_ref[:] = self._nodes_init
        self._u_total[:] = 0.0
        self._stress_stored[:] = 0.0


class ULResult:
    """Updated Lagrangian 解析結果."""

    def __init__(
        self,
        u: np.ndarray,
        converged: bool,
        n_load_steps: int,
        total_iterations: int,
        load_history: list,
        displacement_history: list,
        residual_history: list,
    ) -> None:
        self.u = u
        self.converged = converged
        self.n_load_steps = n_load_steps
        self.total_iterations = total_iterations
        self.load_history = load_history
        self.displacement_history = displacement_history
        self.residual_history = residual_history


def newton_raphson_ul(
    f_ext_total: np.ndarray,
    fixed_dofs: np.ndarray,
    ul_assembler: ULAssemblerQ4,
    *,
    n_load_steps: int = 10,
    max_iter: int = 30,
    tol_force: float = 1e-8,
    tol_disp: float = 1e-10,
    show_progress: bool = True,
) -> ULResult:
    """Updated Lagrangian 定式化の Newton-Raphson 荷重増分法.

    各荷重ステップで:
    1. 荷重増分 Δf = f_ext_total * (step/n) - f_int(u=0 on current ref)
    2. NR反復で増分変位 Δu を求解（参照は現ステップ開始時の配置）
    3. 収束後、参照配置を更新: X_ref ← X_ref + Δu, u_total += Δu

    TL版 newton_raphson() との主な違い:
    - 各ステップで増分変位のみを扱う（全変位ではない）
    - 収束後に参照配置を更新

    Args:
        f_ext_total: 最終外力ベクトル
        fixed_dofs: 拘束DOFインデックス
        ul_assembler: ULAssemblerQ4 インスタンス
        n_load_steps: 荷重ステップ数
        max_iter: NR最大反復回数
        tol_force: 力残差収束判定値
        tol_disp: 変位増分収束判定値
        show_progress: 進捗表示

    Returns:
        ULResult: 解析結果
    """
    ndof = len(f_ext_total)
    fixed = np.asarray(fixed_dofs, dtype=int)
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed] = False

    load_history = []
    disp_history = []
    residual_history = []
    total_iter = 0

    for step in range(1, n_load_steps + 1):
        lam = step / n_load_steps
        f_ext_step = lam * f_ext_total

        # 増分変位（各ステップの開始は 0）
        u_inc = np.zeros(ndof, dtype=float)

        converged = False
        res_norm = 0.0

        for k in range(max_iter):
            f_int = ul_assembler.assemble_internal_force(u_inc)
            R = f_ext_step - f_int
            R[fixed] = 0.0

            f_ref = max(float(np.linalg.norm(f_ext_step)), 1.0)
            res_norm = float(np.linalg.norm(R))

            if res_norm / f_ref < tol_force:
                converged = True
                total_iter += k + 1
                if show_progress:
                    print(
                        f"  Step {step}/{n_load_steps}: "
                        f"converged in {k + 1} iter, ||R||/ref = {res_norm / f_ref:.3e}"
                    )
                break

            K_T = ul_assembler.assemble_tangent(u_inc)
            K_T_d = K_T.toarray() if hasattr(K_T, "toarray") else np.array(K_T)

            for dof in fixed:
                K_T_d[dof, :] = 0.0
                K_T_d[:, dof] = 0.0
                K_T_d[dof, dof] = 1.0

            du = np.linalg.solve(K_T_d, R)
            u_inc += du
            u_inc[fixed] = 0.0

            u_ref = max(float(np.linalg.norm(u_inc)), 1e-30)
            if float(np.linalg.norm(du)) / u_ref < tol_disp:
                converged = True
                total_iter += k + 1
                if show_progress:
                    print(f"  Step {step}/{n_load_steps}: converged (disp) in {k + 1} iter")
                break

        if not converged:
            if show_progress:
                print(f"  Step {step}: not converged in {max_iter} iterations")
            return ULResult(
                u=ul_assembler.u_total + u_inc,
                converged=False,
                n_load_steps=step,
                total_iterations=total_iter + max_iter,
                load_history=load_history,
                displacement_history=disp_history,
                residual_history=residual_history,
            )

        # 参照配置を更新
        ul_assembler.update_reference(u_inc)

        load_history.append(lam)
        disp_history.append(ul_assembler.u_total.copy())
        residual_history.append(res_norm)

    return ULResult(
        u=ul_assembler.u_total.copy(),
        converged=True,
        n_load_steps=n_load_steps,
        total_iterations=total_iter,
        load_history=load_history,
        displacement_history=disp_history,
        residual_history=residual_history,
    )
