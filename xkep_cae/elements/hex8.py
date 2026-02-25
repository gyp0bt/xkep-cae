"""HEX8 要素 — 8 節点 6 面体（レンガ要素）.

B-bar 法（体積ロッキング回避）+ 選択低減積分（せん断ロッキング回避）を
組み合わせた3D固体要素。

== 定式化 ==

Voigt 表記:
  σ = [σxx, σyy, σzz, τyz, τxz, τxy]
  ε = [εxx, εyy, εzz, γyz, γxz, γxy]

B-bar 法:
  体積ひずみ (εxx + εyy + εzz) を要素平均に置き換え、
  非圧縮性付近のロッキングを回避。

選択低減積分:
  偏差成分: 2×2×2 フル積分（8ガウス点）
  体積成分: 1点積分（要素中心）
  → せん断ロッキング回避

参考文献:
  - Hughes, T.J.R. "The Finite Element Method" — B-bar/SRI
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


# ============================================================
# 要素剛性行列
# ============================================================


def hex8_ke_bbar(
    node_xyz: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """B-bar 法 + 選択低減積分付き HEX8 要素剛性行列.

    - 体積ひずみ: 要素平均（1点積分相当）→ 体積ロッキング回避
    - 偏差成分: 2×2×2 フル積分 → せん断ロッキング回避（B-bar 効果）

    Args:
        node_xyz: (8, 3) 要素節点座標
        D: (6, 6) 弾性テンソル

    Returns:
        Ke: (24, 24) 要素剛性行列
    """
    node_xyz = np.asarray(node_xyz, dtype=float)
    if node_xyz.shape != (8, 3):
        raise ValueError(f"node_xyz は (8,3) が必要。実際: {node_xyz.shape}")
    D = np.asarray(D, dtype=float)
    if D.shape != (6, 6):
        raise ValueError(f"D は (6,6) が必要。実際: {D.shape}")

    # 体積ひずみセレクタ: εvol = εxx + εyy + εzz
    # vol_sel @ B = [dN1/dx, dN1/dy, dN1/dz, ...]
    vol_sel = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

    # ==== Pass 1: 各ガウス点の B, detJ を計算、体積ひずみ平均を求める ====
    B_list = []
    detJ_list = []
    dN_dx_list = []

    for xi, eta, zeta in _GAUSS_2x2x2:
        dNdxi = _hex8_dNdxi(xi, eta, zeta)
        J = dNdxi @ node_xyz  # (3, 3) ヤコビアン
        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(f"detJ={detJ:.3e} <= 0（反転要素）")
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ dNdxi  # (3, 8)

        B = _build_B(dN_dx)
        B_list.append(B)
        detJ_list.append(detJ)
        dN_dx_list.append(dN_dx)

    B_arr = np.array(B_list)  # (8, 6, 24)
    detJ_arr = np.array(detJ_list)  # (8,)

    # 各ガウス点の体積ひずみ感度 b_vol = vol_sel @ B → (8, 24)
    b_vol = np.einsum("k, gkj -> gj", vol_sel, B_arr)

    # 要素平均 b̄_vol （detJ 重み付き平均）
    total_vol = detJ_arr.sum()
    b_vol_bar = (detJ_arr[:, None] * b_vol).sum(axis=0) / total_vol  # (24,)

    # ==== Pass 2: B-bar を使って Ke を組み立て ====
    Ke = np.zeros((24, 24), dtype=float)

    for gp_idx in range(8):
        B = B_arr[gp_idx].copy()  # (6, 24)
        b_vol_gp = b_vol[gp_idx]  # (24,)
        detJ = detJ_arr[gp_idx]

        # B̄ の構築: 体積ひずみ成分を要素平均に置換
        # delta_b = (b̄ - b_gp) / 3.0
        delta_b = (b_vol_bar - b_vol_gp) / 3.0  # (24,)

        # εxx, εyy, εzz の各行に delta_b を加算
        B[0, :] += delta_b
        B[1, :] += delta_b
        B[2, :] += delta_b

        Ke += B.T @ D @ B * detJ * 1.0  # w = 1 for 2×2×2

    return Ke


# ============================================================
# ElementProtocol 適合クラス
# ============================================================


class Hex8BBar:
    """HEX8 B-bar 法要素（ElementProtocol 適合）.

    3D 等方弾性 + B-bar（体積ロッキング回避）。
    """

    ndof_per_node: int = 3
    nnodes: int = 8
    ndof: int = 24

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        """局所剛性行列を計算."""
        D = material.tangent()
        return hex8_ke_bbar(coords, D)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """グローバル節点インデックスから要素 DOF インデックスを返す."""
        edofs = np.empty(self.ndof, dtype=np.int64)
        for i, n in enumerate(node_indices):
            edofs[3 * i] = 3 * n
            edofs[3 * i + 1] = 3 * n + 1
            edofs[3 * i + 2] = 3 * n + 2
        return edofs
