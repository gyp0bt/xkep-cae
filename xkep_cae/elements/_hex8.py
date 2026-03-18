"""8節点線形六面体要素 (HEX8).

標準等パラメトリック要素。3自由度/節点（並進のみ）。
6自由度/節点レイアウトへの埋め込み機能付き（梁要素との混合組立用）。

節点番号:
    7----6
   /|   /|
  4----5 |
  | 3--|-2
  |/   |/
  0----1

自然座標: ξ ∈ [-1,1]³

[← README](../../README.md)
"""

from __future__ import annotations

import numpy as np

# 8 節点の自然座標
_NODE_COORDS_NAT = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=float,
)

# 2×2×2 ガウス積分点
_GP = 1.0 / np.sqrt(3.0)
_GAUSS_PTS = np.array(
    [[s1 * _GP, s2 * _GP, s3 * _GP] for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
)
_GAUSS_W = np.ones(8)  # 各点の重み = 1


def _shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """形状関数 N_i (8,)."""
    N = np.empty(8)
    for i in range(8):
        xn, yn, zn = _NODE_COORDS_NAT[i]
        N[i] = 0.125 * (1.0 + xn * xi) * (1.0 + yn * eta) * (1.0 + zn * zeta)
    return N


def _shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
    """形状関数の自然座標微分 dN/dξ (3, 8)."""
    dN = np.empty((3, 8))
    for i in range(8):
        xn, yn, zn = _NODE_COORDS_NAT[i]
        dN[0, i] = 0.125 * xn * (1.0 + yn * eta) * (1.0 + zn * zeta)
        dN[1, i] = 0.125 * (1.0 + xn * xi) * yn * (1.0 + zn * zeta)
        dN[2, i] = 0.125 * (1.0 + xn * xi) * (1.0 + yn * eta) * zn
    return dN


def _isotropic_D(E: float, nu: float) -> np.ndarray:
    """等方弾性 D 行列 (6×6) — Voigt 順序."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            D[i, j] = lam
        D[i, i] += 2.0 * mu
    D[3, 3] = mu
    D[4, 4] = mu
    D[5, 5] = mu
    return D


def hex8_stiffness_3dof(
    node_coords: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """HEX8 要素剛性行列 (24×24, 3 DOF/node).

    Args:
        node_coords: (8, 3) 要素節点の物理座標
        E: ヤング率
        nu: ポアソン比

    Returns:
        Ke: (24, 24) 要素剛性行列
    """
    D = _isotropic_D(E, nu)
    Ke = np.zeros((24, 24))

    for gp_idx in range(8):
        xi, eta, zeta = _GAUSS_PTS[gp_idx]
        w = _GAUSS_W[gp_idx]

        dN_nat = _shape_derivatives(xi, eta, zeta)
        J = dN_nat @ node_coords  # (3, 3) ヤコビ行列
        detJ = np.linalg.det(J)
        dN_phys = np.linalg.solve(J, dN_nat)  # (3, 8) dN/dx

        # B 行列 (6, 24)
        B = np.zeros((6, 24))
        for i in range(8):
            c = 3 * i
            B[0, c] = dN_phys[0, i]  # ε_xx
            B[1, c + 1] = dN_phys[1, i]  # ε_yy
            B[2, c + 2] = dN_phys[2, i]  # ε_zz
            B[3, c] = dN_phys[1, i]  # γ_xy
            B[3, c + 1] = dN_phys[0, i]
            B[4, c + 1] = dN_phys[2, i]  # γ_yz
            B[4, c + 2] = dN_phys[1, i]
            B[5, c] = dN_phys[2, i]  # γ_xz
            B[5, c + 2] = dN_phys[0, i]

        Ke += w * detJ * (B.T @ D @ B)

    return Ke


def hex8_internal_force_3dof(
    node_coords: np.ndarray,
    u_elem: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """HEX8 要素内力ベクトル (24,).

    Args:
        node_coords: (8, 3) 要素節点の物理座標
        u_elem: (24,) 要素変位ベクトル（3 DOF/node）
        E: ヤング率
        nu: ポアソン比

    Returns:
        f_int: (24,) 要素内力ベクトル
    """
    D = _isotropic_D(E, nu)
    f_int = np.zeros(24)

    for gp_idx in range(8):
        xi, eta, zeta = _GAUSS_PTS[gp_idx]
        w = _GAUSS_W[gp_idx]

        dN_nat = _shape_derivatives(xi, eta, zeta)
        J = dN_nat @ node_coords
        detJ = np.linalg.det(J)
        dN_phys = np.linalg.solve(J, dN_nat)

        B = np.zeros((6, 24))
        for i in range(8):
            c = 3 * i
            B[0, c] = dN_phys[0, i]
            B[1, c + 1] = dN_phys[1, i]
            B[2, c + 2] = dN_phys[2, i]
            B[3, c] = dN_phys[1, i]
            B[3, c + 1] = dN_phys[0, i]
            B[4, c + 1] = dN_phys[2, i]
            B[4, c + 2] = dN_phys[1, i]
            B[5, c] = dN_phys[2, i]
            B[5, c + 2] = dN_phys[0, i]

        strain = B @ u_elem
        stress = D @ strain
        f_int += w * detJ * (B.T @ stress)

    return f_int


def hex8_stiffness_6dof(
    node_coords: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """HEX8 要素剛性行列を 6 DOF/node レイアウトに埋め込み (48×48).

    回転DOF (d=3,4,5) はゼロ剛性。梁要素との混合組立用。

    Args:
        node_coords: (8, 3) 要素節点の物理座標
        E: ヤング率
        nu: ポアソン比

    Returns:
        Ke_6: (48, 48) 要素剛性行列（6 DOF/node 埋め込み）
    """
    Ke_3 = hex8_stiffness_3dof(node_coords, E, nu)
    Ke_6 = np.zeros((48, 48))

    for i in range(8):
        for j in range(8):
            for di in range(3):
                for dj in range(3):
                    Ke_6[6 * i + di, 6 * j + dj] = Ke_3[3 * i + di, 3 * j + dj]

    return Ke_6


def hex8_internal_force_6dof(
    node_coords: np.ndarray,
    u_elem_6dof: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """HEX8 要素内力を 6 DOF/node レイアウトで計算 (48,).

    Args:
        node_coords: (8, 3) 要素節点の物理座標
        u_elem_6dof: (48,) 要素変位ベクトル（6 DOF/node）
        E: ヤング率
        nu: ポアソン比

    Returns:
        f_int_6: (48,) 要素内力ベクトル（6 DOF/node）
    """
    # 並進DOFのみ抽出
    u_3dof = np.zeros(24)
    for i in range(8):
        u_3dof[3 * i : 3 * i + 3] = u_elem_6dof[6 * i : 6 * i + 3]

    f_3dof = hex8_internal_force_3dof(node_coords, u_3dof, E, nu)

    # 6 DOF レイアウトに埋め戻し
    f_6dof = np.zeros(48)
    for i in range(8):
        f_6dof[6 * i : 6 * i + 3] = f_3dof[3 * i : 3 * i + 3]

    return f_6dof
