"""HEX8 要素ファミリ — 8 節点 6 面体（レンガ要素）.

3 つのバリエーションを提供:

  C3D8 (Hex8SRI):
    選択低減積分 (SRI)。偏差成分は 1 点低減積分、体積成分は 2×2×2 完全積分。
    せん断ロッキングを回避しつつ体積拘束を維持。

  C3D8R (Hex8Reduced):
    均一低減積分 (1 点積分)。高速だがアワーグラスモード (12 個) を持つ。
    alpha_hg > 0 で Flanagan-Belytschko 型アワーグラス制御を適用可能。
    陽的時間積分向け。

  C3D8I (Hex8Incompatible):
    Wilson-Taylor 非適合モード (incompatible modes)。
    標準形状関数に 3 つの内部バブルモード α_i = 1-ξ_i² を追加し、
    静的縮合 (static condensation) で内部自由度を除去。
    せん断・体積ロッキングの両方を高精度に回避。

== 共通定式化 ==

Voigt 表記:
  σ = [σxx, σyy, σzz, τyz, τxz, τxy]
  ε = [εxx, εyy, εzz, γyz, γxz, γxy]

参考文献:
  - Hughes, T.J.R. "The Finite Element Method" — SRI/B-bar
  - Wilson, Taylor et al. "Incompatible displacement models" (1973)
  - Belytschko et al. "Nonlinear Finite Elements" — HEX8 定式化
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.core.constitutive import ConstitutiveProtocol


# ============================================================
# ガウス積分点
# ============================================================

_G2 = 1.0 / np.sqrt(3.0)
_GAUSS_2x2x2 = [
    (-_G2, -_G2, -_G2),
    (+_G2, -_G2, -_G2),
    (+_G2, +_G2, -_G2),
    (-_G2, +_G2, -_G2),
    (-_G2, -_G2, +_G2),
    (+_G2, -_G2, +_G2),
    (+_G2, +_G2, +_G2),
    (-_G2, +_G2, +_G2),
]
_GAUSS_WEIGHTS_2x2x2 = [1.0] * 8  # w_i = 1 for all 8 points


# ============================================================
# 形状関数
# ============================================================


def _hex8_shape(xi: float, eta: float, zeta: float) -> np.ndarray:
    """HEX8 形状関数 N_i (i=0..7).

    節点順序（自然座標）:
      0: (-1,-1,-1)  1: (+1,-1,-1)  2: (+1,+1,-1)  3: (-1,+1,-1)
      4: (-1,-1,+1)  5: (+1,-1,+1)  6: (+1,+1,+1)  7: (-1,+1,+1)
    """
    xm, xp = 1.0 - xi, 1.0 + xi
    em, ep = 1.0 - eta, 1.0 + eta
    zm, zp = 1.0 - zeta, 1.0 + zeta
    return 0.125 * np.array(
        [
            xm * em * zm,
            xp * em * zm,
            xp * ep * zm,
            xm * ep * zm,
            xm * em * zp,
            xp * em * zp,
            xp * ep * zp,
            xm * ep * zp,
        ]
    )


def _hex8_dNdxi(xi: float, eta: float, zeta: float) -> np.ndarray:
    """HEX8 形状関数の自然座標微分 dN/dξ, dN/dη, dN/dζ.

    Returns:
        dNdxi: (3, 8) — [dN/dξ; dN/dη; dN/dζ]
    """
    xm, xp = 1.0 - xi, 1.0 + xi
    em, ep = 1.0 - eta, 1.0 + eta
    zm, zp = 1.0 - zeta, 1.0 + zeta
    return 0.125 * np.array(
        [
            # dN/dξ
            [
                -em * zm,
                +em * zm,
                +ep * zm,
                -ep * zm,
                -em * zp,
                +em * zp,
                +ep * zp,
                -ep * zp,
            ],
            # dN/dη
            [
                -xm * zm,
                -xp * zm,
                +xp * zm,
                +xm * zm,
                -xm * zp,
                -xp * zp,
                +xp * zp,
                +xm * zp,
            ],
            # dN/dζ
            [
                -xm * em,
                -xp * em,
                -xp * ep,
                -xm * ep,
                +xm * em,
                +xp * em,
                +xp * ep,
                +xm * ep,
            ],
        ]
    )


# ============================================================
# B 行列構築
# ============================================================


def _build_B(dN_dx: np.ndarray) -> np.ndarray:
    """標準 B 行列 (6×24) を構築.

    Args:
        dN_dx: (3, 8) — 物理座標での形状関数微分 [dN/dx; dN/dy; dN/dz]

    Returns:
        B: (6, 24) — ひずみ-変位行列
    """
    B = np.zeros((6, 24), dtype=float)
    for i in range(8):
        c = 3 * i
        B[0, c] = dN_dx[0, i]  # εxx = du/dx
        B[1, c + 1] = dN_dx[1, i]  # εyy = dv/dy
        B[2, c + 2] = dN_dx[2, i]  # εzz = dw/dz
        B[3, c + 1] = dN_dx[2, i]  # γyz = dv/dz + dw/dy
        B[3, c + 2] = dN_dx[1, i]
        B[4, c] = dN_dx[2, i]  # γxz = du/dz + dw/dx
        B[4, c + 2] = dN_dx[0, i]
        B[5, c] = dN_dx[1, i]  # γxy = du/dy + dv/dx
        B[5, c + 1] = dN_dx[0, i]
    return B


def _build_B_cols(dNi_dx: np.ndarray) -> np.ndarray:
    """1 節点分の B サブマトリクス (6×3) を構築.

    Args:
        dNi_dx: (3,) — 1節点の物理座標微分 [dN/dx, dN/dy, dN/dz]
    """
    Bi = np.zeros((6, 3), dtype=float)
    Bi[0, 0] = dNi_dx[0]
    Bi[1, 1] = dNi_dx[1]
    Bi[2, 2] = dNi_dx[2]
    Bi[3, 1] = dNi_dx[2]
    Bi[3, 2] = dNi_dx[1]
    Bi[4, 0] = dNi_dx[2]
    Bi[4, 2] = dNi_dx[0]
    Bi[5, 0] = dNi_dx[1]
    Bi[5, 1] = dNi_dx[0]
    return Bi


# ============================================================
# D 行列の偏差-体積分解
# ============================================================


def _split_D_vol_dev(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """弾性テンソル D (6×6) を体積部と偏差部に分解.

    D = D_vol + D_dev
    D_vol = K * m ⊗ m    (K = 体積弾性率, m = [1,1,1,0,0,0]^T)
    D_dev = D - D_vol

    Returns:
        (D_vol, D_dev): 各 (6, 6)
    """
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    # 体積弾性率: K = (1/9) * m^T D m
    K = m @ D @ m / 9.0
    D_vol = K * np.outer(m, m)
    D_dev = D - D_vol
    return D_vol, D_dev


# ============================================================
# 入力検証ヘルパー
# ============================================================


def _validate_inputs(node_xyz: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """共通入力バリデーション."""
    node_xyz = np.asarray(node_xyz, dtype=float)
    if node_xyz.shape != (8, 3):
        raise ValueError(f"node_xyz は (8,3) が必要。実際: {node_xyz.shape}")
    D = np.asarray(D, dtype=float)
    if D.shape != (6, 6):
        raise ValueError(f"D は (6,6) が必要。実際: {D.shape}")
    return node_xyz, D


def _compute_B_detJ(node_xyz: np.ndarray, xi: float, eta: float, zeta: float):
    """指定ガウス点での B, detJ, invJ を計算."""
    dNdxi = _hex8_dNdxi(xi, eta, zeta)
    J = dNdxi @ node_xyz
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError(f"detJ={detJ:.3e} <= 0（反転要素）")
    invJ = np.linalg.inv(J)
    dN_dx = invJ @ dNdxi
    B = _build_B(dN_dx)
    return B, detJ, invJ, dN_dx


# ============================================================
# C3D8: 選択低減積分 (SRI) 要素剛性行列
# ============================================================


def hex8_ke_sri(
    node_xyz: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """SRI 付き HEX8 要素剛性行列 (C3D8).

    選択低減積分 (Selective Reduced Integration):
      - 偏差成分 (D_dev): 1 点低減積分（要素中心）→ せん断ロッキング回避
      - 体積成分 (D_vol): 2×2×2 完全積分 → 体積拘束の維持

    Args:
        node_xyz: (8, 3) 要素節点座標
        D: (6, 6) 弾性テンソル

    Returns:
        Ke: (24, 24) 要素剛性行列
    """
    node_xyz, D = _validate_inputs(node_xyz, D)
    D_vol, D_dev = _split_D_vol_dev(D)

    Ke = np.zeros((24, 24), dtype=float)

    # ==== 体積成分: 2×2×2 完全積分 ====
    for xi, eta, zeta in _GAUSS_2x2x2:
        B, detJ, _, _ = _compute_B_detJ(node_xyz, xi, eta, zeta)
        Ke += B.T @ D_vol @ B * detJ  # w = 1

    # ==== 偏差成分: 1 点低減積分（要素中心 ξ=η=ζ=0） ====
    B0, detJ0, _, _ = _compute_B_detJ(node_xyz, 0.0, 0.0, 0.0)
    Ke += B0.T @ D_dev @ B0 * detJ0 * 8.0  # w = 8 (=2×2×2)

    return Ke


# 後方互換性のため旧名を維持
hex8_ke_bbar = hex8_ke_sri


# ============================================================
# アワーグラスベクトル（Flanagan-Belytschko 1981）
# ============================================================

# HEX8 のアワーグラスベースベクトル（4本）
# 定数モード [1,1,...] と線形モード [ξ_i, η_i, ζ_i] に直交
_HG_VECTORS = np.array(
    [
        [+1, -1, +1, -1, -1, +1, -1, +1],  # ξη モード
        [+1, -1, -1, +1, -1, +1, +1, -1],  # ξζ モード
        [+1, +1, -1, -1, -1, -1, +1, +1],  # ηζ モード
        [-1, +1, -1, +1, +1, -1, +1, -1],  # ξηζ モード
    ],
    dtype=float,
)


# ============================================================
# C3D8R: 均一低減積分 + アワーグラス制御 要素剛性行列
# ============================================================


def hex8_ke_reduced(
    node_xyz: np.ndarray,
    D: np.ndarray,
    *,
    alpha_hg: float = 0.0,
) -> np.ndarray:
    """均一低減積分 HEX8 要素剛性行列 (C3D8R).

    1 点ガウス積分（要素中心）。12 個のアワーグラスモードを持つ。
    alpha_hg > 0 で Flanagan-Belytschko 型アワーグラス剛性制御を適用。

    Args:
        node_xyz: (8, 3) 要素節点座標
        D: (6, 6) 弾性テンソル
        alpha_hg: アワーグラス制御係数。0.0=無効、推奨 0.03〜0.05。

    Returns:
        Ke: (24, 24) 要素剛性行列
    """
    node_xyz, D = _validate_inputs(node_xyz, D)

    B0, detJ0, _, _ = _compute_B_detJ(node_xyz, 0.0, 0.0, 0.0)
    Ke = B0.T @ D @ B0 * detJ0 * 8.0  # w = 8 (2^3)

    if alpha_hg > 0.0:
        V_elem = detJ0 * 8.0
        L_char = V_elem ** (1.0 / 3.0)
        D_max = np.max(np.diag(D))
        k_hg = alpha_hg * D_max * V_elem / (L_char**2)

        for alpha_idx in range(4):
            h = _HG_VECTORS[alpha_idx]
            hh = h @ h  # = 8
            for d in range(3):
                q = np.zeros(24, dtype=float)
                q[d::3] = h
                Ke += (k_hg / hh) * np.outer(q, q)

    return Ke


# ============================================================
# C3D8I: 非適合モード (Incompatible Modes) 要素剛性行列
# ============================================================


def hex8_ke_incompatible(
    node_xyz: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """非適合モード付き HEX8 要素剛性行列 (C3D8I).

    Wilson-Taylor の非適合モード法:
      3 つの内部バブルモード α_k = 1 - ξ_k² (k=1,2,3) を追加し、
      要素内部の変位場を拡張。静的縮合で内部 DOF (9個) を除去。

    K_condensed = K_uu - K_ua @ inv(K_aa) @ K_au

    ここで:
      K_uu: 標準 DOF 同士の剛性 (24×24)
      K_ua: 標準-内部 DOF 結合   (24×9)
      K_aa: 内部 DOF 同士の剛性  (9×9)

    Args:
        node_xyz: (8, 3) 要素節点座標
        D: (6, 6) 弾性テンソル

    Returns:
        Ke: (24, 24) 要素剛性行列（静的縮合済み）
    """
    node_xyz, D = _validate_inputs(node_xyz, D)

    # 要素中心のヤコビアン（非適合モードの座標変換に使用）
    dNdxi_0 = _hex8_dNdxi(0.0, 0.0, 0.0)
    J0 = dNdxi_0 @ node_xyz
    detJ0 = np.linalg.det(J0)
    if detJ0 <= 0.0:
        raise ValueError(f"detJ0={detJ0:.3e} <= 0（反転要素）")
    invJ0 = np.linalg.inv(J0)

    K_uu = np.zeros((24, 24), dtype=float)
    K_ua = np.zeros((24, 9), dtype=float)
    K_aa = np.zeros((9, 9), dtype=float)

    for xi, eta, zeta in _GAUSS_2x2x2:
        # 標準 B 行列
        B, detJ, _, _ = _compute_B_detJ(node_xyz, xi, eta, zeta)

        # 非適合モード B 行列 (6×9)
        # α_1 = 1-ξ², α_2 = 1-η², α_3 = 1-ζ²
        # dα/dξ = [-2ξ, 0, 0; 0, -2η, 0; 0, 0, -2ζ]
        dAlpha_dxi = np.zeros((3, 3), dtype=float)
        dAlpha_dxi[0, 0] = -2.0 * xi
        dAlpha_dxi[1, 1] = -2.0 * eta
        dAlpha_dxi[2, 2] = -2.0 * zeta

        # 物理座標微分（要素中心ヤコビアンで変換: Taylor 1976）
        dAlpha_dx = invJ0 @ dAlpha_dxi  # (3, 3)

        # B_alpha (6×9): 各非適合モードに 3 変位成分
        B_alpha = np.zeros((6, 9), dtype=float)
        for k in range(3):
            col = 3 * k
            B_alpha[0, col] = dAlpha_dx[0, k]  # εxx
            B_alpha[1, col + 1] = dAlpha_dx[1, k]  # εyy
            B_alpha[2, col + 2] = dAlpha_dx[2, k]  # εzz
            B_alpha[3, col + 1] = dAlpha_dx[2, k]  # γyz
            B_alpha[3, col + 2] = dAlpha_dx[1, k]
            B_alpha[4, col] = dAlpha_dx[2, k]  # γxz
            B_alpha[4, col + 2] = dAlpha_dx[0, k]
            B_alpha[5, col] = dAlpha_dx[1, k]  # γxy
            B_alpha[5, col + 1] = dAlpha_dx[0, k]

        # detJ0/detJ でスケーリング（修正非適合法: Simo & Armero 1992 に準拠）
        scale = detJ0 / detJ
        B_alpha_scaled = B_alpha * scale

        K_uu += B.T @ D @ B * detJ
        K_ua += B.T @ D @ B_alpha_scaled * detJ
        K_aa += B_alpha_scaled.T @ D @ B_alpha_scaled * detJ

    # 静的縮合: K = K_uu - K_ua @ inv(K_aa) @ K_au
    K_aa_inv = np.linalg.inv(K_aa)
    Ke = K_uu - K_ua @ K_aa_inv @ K_ua.T

    return Ke


# ============================================================
# ElementProtocol 適合クラス
# ============================================================


def _dof_indices_3d(node_indices: np.ndarray, ndof: int) -> np.ndarray:
    """3D 要素の DOF インデックスを計算（共通ヘルパー）."""
    edofs = np.empty(ndof, dtype=np.int64)
    for i, n in enumerate(node_indices):
        edofs[3 * i] = 3 * n
        edofs[3 * i + 1] = 3 * n + 1
        edofs[3 * i + 2] = 3 * n + 2
    return edofs


class Hex8SRI:
    """C3D8: SRI 法 HEX8 要素（ElementProtocol 適合）.

    選択低減積分:
      偏差成分 → 1 点低減積分（せん断ロッキング回避）
      体積成分 → 2×2×2 完全積分（体積拘束維持）
    """

    element_type: str = "C3D8"
    ndof_per_node: int = 3
    nnodes: int = 8
    ndof: int = 24

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        D = material.tangent()
        return hex8_ke_sri(coords, D)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        return _dof_indices_3d(node_indices, self.ndof)


class Hex8Incompatible:
    """C3D8I: 非適合モード HEX8 要素（ElementProtocol 適合）.

    Wilson-Taylor 非適合モード + 静的縮合。
    せん断・体積ロッキングの両方を高精度に回避。
    """

    element_type: str = "C3D8I"
    ndof_per_node: int = 3
    nnodes: int = 8
    ndof: int = 24

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        D = material.tangent()
        return hex8_ke_incompatible(coords, D)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        return _dof_indices_3d(node_indices, self.ndof)


class Hex8Reduced:
    """C3D8R: 均一低減積分 HEX8 要素（ElementProtocol 適合）.

    1 点ガウス積分 + オプションでアワーグラス剛性制御。

    Args:
        alpha_hg: アワーグラス制御係数。0.0=無効、推奨 0.03〜0.05。
    """

    element_type: str = "C3D8R"
    ndof_per_node: int = 3
    nnodes: int = 8
    ndof: int = 24

    def __init__(self, alpha_hg: float = 0.0) -> None:
        self.alpha_hg = alpha_hg

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        D = material.tangent()
        return hex8_ke_reduced(coords, D, alpha_hg=self.alpha_hg)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        return _dof_indices_3d(node_indices, self.ndof)


# 後方互換エイリアス
Hex8BBar = Hex8SRI
