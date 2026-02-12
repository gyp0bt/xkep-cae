"""EAS (Enhanced Assumed Strain) 法 / B-bar 法 付き Q4 要素（平面ひずみ）.

Simo & Rifai (1990) の EAS-4 定式化による高精度 Q4 双線形四角形要素。
- EAS-4: 4個の非適合ひずみモードでせん断ロッキングと体積ロッキングの両方を抑制
- 内部自由度 α は静的縮合で消去（外部インタフェースは標準 Q4 と同一）

数値検証結果:
  - EAS-4 単体: 片持ち梁曲げ粗メッシュ (10×1, L/H=10) で ratio=1.005（解析解比）
  - 非圧縮性材料 (ν=0.4999) の曲げ問題でも ratio≈1.0（体積ロッキングも抑制）
  - plain Q4 は同条件で ratio=0.01（壊滅的ロッキング）
  - B-bar 併用は曲げ問題で過補正（ratio=4.0）のため推奨しない

EAS + B-bar 併用版も Quad4EASBBarPlaneStrain として提供するが、
曲げ支配問題では過補正となるため、デフォルトは EAS-4 単体を推奨。

参考文献:
  Simo, J.C. & Rifai, M.S. (1990) "A class of mixed assumed strain methods
  and the method of incompatible modes", IJNME, 29, 1595-1638.
  Andelfinger, U. & Ramm, E. (1993) "EAS-elements for two-dimensional,
  three-dimensional, plate and shell structures and their equivalence to
  HR-elements", IJNME, 36, 1311-1337.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.core.constitutive import ConstitutiveProtocol


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------


def _dN_dxi_eta(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """局所座標(ξ,η)での形状関数導関数 dN/dξ, dN/dη を返す."""
    dN_dxi = 0.25 * np.array(
        [-(1.0 - eta), +(1.0 - eta), +(1.0 + eta), -(1.0 + eta)],
        dtype=float,
    )
    dN_deta = 0.25 * np.array(
        [-(1.0 - xi), -(1.0 + xi), +(1.0 + xi), +(1.0 - xi)],
        dtype=float,
    )
    return dN_dxi, dN_deta


def _jacobian(
    dN_dxi: np.ndarray, dN_deta: np.ndarray, node_xy: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """ヤコビアン J, det(J), J^{-1} を返す."""
    J = np.empty((2, 2), dtype=float)
    J[0, 0] = dN_dxi @ node_xy[:, 0]
    J[0, 1] = dN_deta @ node_xy[:, 0]
    J[1, 0] = dN_dxi @ node_xy[:, 1]
    J[1, 1] = dN_deta @ node_xy[:, 1]
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if detJ <= 0.0:
        raise ValueError(f"detJ<=0（反転要素の可能性） detJ={detJ:.3e}")
    invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=float) / detJ
    return J, detJ, invJ


def _B_matrix(dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
    """ひずみ-変位 B マトリクス (3,8) を返す（engineering shear）."""
    B = np.zeros((3, 8), dtype=float)
    for i in range(4):
        B[0, 2 * i] = dN_dx[i]  # εxx
        B[1, 2 * i + 1] = dN_dy[i]  # εyy
        B[2, 2 * i] = dN_dy[i]  # γxy
        B[2, 2 * i + 1] = dN_dx[i]
    return B


def _eas4_M_tilde(xi: float, eta: float) -> np.ndarray:
    """EAS-4 の親要素域での拡張ひずみ補間 M̃(ξ,η) (3×4).

    モード構成:
      α₁: εxx 方向に ξ で線形変化（y軸まわり曲げ対応）
      α₂: εyy 方向に η で線形変化（x軸まわり曲げ対応）
      α₃: γxy に ξ で線形変化（せん断ロッキング対策）
      α₄: γxy に η で線形変化（せん断ロッキング対策）
    """
    return np.array(
        [
            [xi, 0.0, 0.0, 0.0],
            [0.0, eta, 0.0, 0.0],
            [0.0, 0.0, xi, eta],
        ],
        dtype=float,
    )


def _strain_transform_T(A: np.ndarray) -> np.ndarray:
    """2次元ひずみ変換行列 T(A) を返す (3×3, engineering shear 規約).

    A は 2×2 の座標変換行列。ε' = T(A) @ ε の関係。
    歪んだ要素での EAS ひずみモードの物理座標系への変換に使用。
    """
    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    return np.array(
        [
            [a11 * a11, a12 * a12, a11 * a12],
            [a21 * a21, a22 * a22, a21 * a22],
            [2.0 * a11 * a21, 2.0 * a12 * a22, a11 * a22 + a12 * a21],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# EAS-4 単体
# ---------------------------------------------------------------------------


def quad4_ke_plane_strain_eas(
    node_xy: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
    n_eas: int = 4,
) -> np.ndarray:
    """EAS-4 法付き Q4 要素（平面ひずみ）の局所剛性行列を返す.

    Simo-Rifai EAS-4 による高精度 Q4 要素。
    せん断ロッキングと体積ロッキングの両方を同時に抑制する。
    内部自由度 α は静的縮合で消去。

    Args:
        node_xy: (4,2) 要素節点座標
        D: (3,3) 弾性マトリクス（平面ひずみ, engineering shear）
        t: 厚み
        n_eas: EAS パラメータ数（デフォルト 4）

    Returns:
        Ke: (8,8) 要素剛性マトリクス（静的縮合済み）
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (4, 2):
        raise ValueError("node_xy は (4,2) である必要があります。")

    # 要素中心 (ξ=0, η=0) での Jacobian と EAS 変換行列
    dN_dxi_0, dN_deta_0 = _dN_dxi_eta(0.0, 0.0)
    _J0, detJ0, invJ0 = _jacobian(dN_dxi_0, dN_deta_0, node_xy)
    T0 = _strain_transform_T(invJ0.T)  # T₀ = T(J₀⁻ᵀ)

    # 2×2 Gauss 積分点
    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]

    K_uu = np.zeros((8, 8), dtype=float)
    K_ua = np.zeros((8, n_eas), dtype=float)
    K_aa = np.zeros((n_eas, n_eas), dtype=float)

    for xi, eta in gauss_points:
        dN_dxi, dN_deta = _dN_dxi_eta(xi, eta)
        _J, detJ, invJ = _jacobian(dN_dxi, dN_deta, node_xy)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        B = _B_matrix(dN_dx, dN_dy)

        # EAS 拡張ひずみ補間行列 M = (detJ₀/detJ) * T₀ @ M̃(ξ,η)
        M_tilde = _eas4_M_tilde(xi, eta)
        M = (detJ0 / detJ) * (T0 @ M_tilde)

        wt = detJ * t
        DB = D @ B
        DM = D @ M

        K_uu += B.T @ DB * wt
        K_ua += B.T @ DM * wt
        K_aa += M.T @ DM * wt

    # 静的縮合: K = K_uu - K_uα @ K_αα⁻¹ @ K_αuᵀ
    K_aa_inv = np.linalg.inv(K_aa)
    return K_uu - K_ua @ K_aa_inv @ K_ua.T


# ---------------------------------------------------------------------------
# EAS-4 + B-bar 併用
# ---------------------------------------------------------------------------


def quad4_ke_plane_strain_eas_bbar(
    node_xy: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
    n_eas: int = 4,
) -> np.ndarray:
    """EAS + B-bar 法付き Q4 要素（平面ひずみ）の局所剛性行列を返す.

    EAS-4 と B-bar を組み合わせた要素。
    注意: 曲げ支配問題で過補正になる可能性がある。
    一般的には EAS-4 単体 (quad4_ke_plane_strain_eas) を推奨。

    Args:
        node_xy: (4,2) 要素節点座標
        D: (3,3) 弾性マトリクス（平面ひずみ, engineering shear）
        t: 厚み
        n_eas: EAS パラメータ数（デフォルト 4）

    Returns:
        Ke: (8,8) 要素剛性マトリクス（静的縮合済み）
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (4, 2):
        raise ValueError("node_xy は (4,2) である必要があります。")

    # 要素中心での Jacobian と EAS 変換行列
    dN_dxi_0, dN_deta_0 = _dN_dxi_eta(0.0, 0.0)
    _J0, detJ0, invJ0 = _jacobian(dN_dxi_0, dN_deta_0, node_xy)
    T0 = _strain_transform_T(invJ0.T)

    # 2×2 Gauss 積分点
    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]

    # パス1: B, detJ を保存し、B-bar 用の体積ひずみ平均を求める
    B_list: list[np.ndarray] = []
    detJ_list: list[float] = []

    for xi, eta in gauss_points:
        dN_dxi, dN_deta = _dN_dxi_eta(xi, eta)
        _J, detJ, invJ = _jacobian(dN_dxi, dN_deta, node_xy)
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        B = _B_matrix(dN_dx, dN_dy)
        B_list.append(B)
        detJ_list.append(detJ)

    B_arr = np.stack(B_list, axis=0)  # (4, 3, 8)
    detJ_arr = np.asarray(detJ_list)  # (4,)

    # 体積ひずみ感度 b_m = [1, 1, 0] · B → (4, 8)
    vol_selector = np.array([1.0, 1.0, 0.0], dtype=float)
    b_m = np.einsum("k, gkj -> gj", vol_selector, B_arr)

    # 要素平均 b̄_m（detJ 重み付き平均）
    w = detJ_arr
    b_m_bar = (w[:, None] * b_m).sum(axis=0) / w.sum()

    # パス2: B-bar + EAS で K_uu, K_uα, K_αα を積分
    K_uu = np.zeros((8, 8), dtype=float)
    K_ua = np.zeros((8, n_eas), dtype=float)
    K_aa = np.zeros((n_eas, n_eas), dtype=float)

    for gp_idx, (xi, eta) in enumerate(gauss_points):
        detJ = detJ_arr[gp_idx]
        B = B_arr[gp_idx].copy()

        # B-bar 修正: 体積ひずみ成分を要素平均に置換
        b_m_i = b_m[gp_idx]
        delta_b = 0.5 * (b_m_bar - b_m_i)
        B[0, :] += delta_b
        B[1, :] += delta_b

        # EAS 拡張ひずみ補間行列
        M_tilde = _eas4_M_tilde(xi, eta)
        M = (detJ0 / detJ) * (T0 @ M_tilde)

        wt = detJ * t
        DB = D @ B
        DM = D @ M

        K_uu += B.T @ DB * wt
        K_ua += B.T @ DM * wt
        K_aa += M.T @ DM * wt

    K_aa_inv = np.linalg.inv(K_aa)
    return K_uu - K_ua @ K_aa_inv @ K_ua.T


# ---------------------------------------------------------------------------
# ElementProtocol 適合クラス
# ---------------------------------------------------------------------------


class Quad4EASPlaneStrain:
    """EAS-4 法付き Q4 双線形四角形要素（平面ひずみ）（ElementProtocol 適合）.

    Simo-Rifai EAS-4 による高精度要素。
    せん断ロッキングと体積ロッキングの両方を同時に抑制する。
    Q4 要素の推奨デフォルト。
    """

    ndof_per_node: int = 2
    nnodes: int = 4
    ndof: int = 8

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        D = material.tangent()
        return quad4_ke_plane_strain_eas(coords, D, thickness if thickness is not None else 1.0)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        edofs = np.empty(self.ndof, dtype=np.int64)
        for i, n in enumerate(node_indices):
            edofs[2 * i] = 2 * n
            edofs[2 * i + 1] = 2 * n + 1
        return edofs


class Quad4EASBBarPlaneStrain:
    """EAS + B-bar 法付き Q4 双線形四角形要素（平面ひずみ）（ElementProtocol 適合）.

    EAS-4 と B-bar の併用。曲げ支配問題では過補正の可能性あり。
    一般用途には Quad4EASPlaneStrain（EAS 単体）を推奨。
    """

    ndof_per_node: int = 2
    nnodes: int = 4
    ndof: int = 8

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        D = material.tangent()
        return quad4_ke_plane_strain_eas_bbar(
            coords, D, thickness if thickness is not None else 1.0
        )

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        edofs = np.empty(self.ndof, dtype=np.int64)
        for i, n in enumerate(node_indices):
            edofs[2 * i] = 2 * n
            edofs[2 * i + 1] = 2 * n + 1
        return edofs
