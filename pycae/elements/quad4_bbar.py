from __future__ import annotations
from typing import Literal
import numpy as np


def quad4_ke_plane_strain_bbar(
    node_xy: np.ndarray,
    D: np.ndarray,
    t: float = 1.0,
) -> np.ndarray:
    """B̄法付きQ4一次要素（平面ひずみ）の局所剛性マトリクスを返す.

    - 2×2 Gaussフル積分
    - dev成分は通常どおり、体積ひずみ(εxx+εyy)に対応する部分のみ
      B̄法で要素平均に置き換える簡易版B̄実装。

    Args:
        node_xy: (4,2) 要素節点座標 [ [x1,y1], ..., [x4,y4] ]
        D: (3,3) 弾性マトリクス（平面ひずみ, engineering shear）
        t: 厚み

    Returns:
        Ke: (8,8) 要素剛性マトリクス
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (4, 2):
        raise ValueError("node_xy は (4,2) である必要があります。")

    def dN_dxi_eta(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
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

    # 2×2 Gauss点
    g = 1.0 / np.sqrt(3.0)
    gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)]

    # 1回目ループで B と detJ を保存し、b_m の要素平均を求める
    B_list: list[np.ndarray] = []
    detJ_list: list[float] = []

    for xi, eta in gauss_points:
        dN_dxi, dN_deta = dN_dxi_eta(xi, eta)

        # ヤコビアン
        J = np.empty((2, 2), dtype=float)
        J[0, 0] = dN_dxi @ node_xy[:, 0]
        J[0, 1] = dN_deta @ node_xy[:, 0]
        J[1, 0] = dN_dxi @ node_xy[:, 1]
        J[1, 1] = dN_deta @ node_xy[:, 1]

        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 0.0:
            raise ValueError(f"detJ<=0（反転要素の可能性） detJ={detJ:.3e}")
        invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=float) / detJ

        # dN/dx, dN/dy
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        # 通常のBマトリクス（engineering shear）
        B = np.zeros((3, 8), dtype=float)
        for i in range(4):
            B[0, 2 * i] = dN_dx[i]  # εxx
            B[1, 2 * i + 1] = dN_dy[i]  # εyy
            B[2, 2 * i] = dN_dy[i]  # γxy = du/dy + dv/dx
            B[2, 2 * i + 1] = dN_dx[i]

        B_list.append(B)
        detJ_list.append(detJ)

    B_arr = np.stack(B_list, axis=0)  # (4,3,8)
    detJ_arr = np.asarray(detJ_list)  # (4,)

    # 各Gauss点の「体積ひずみ感度」 b_m = [1 1 0] * B  (1×8)
    # → εm = εxx + εyy だけを取り出す
    vol_selector = np.array([1.0, 1.0, 0.0], dtype=float)  # [εxx, εyy, γxy]
    b_m = np.einsum("k, gij -> gi", vol_selector, B_arr)  # (4,8)

    # 要素平均 b̄_m （detJで重み付け）
    w = detJ_arr
    b_m_bar = (w[:, None] * b_m).sum(axis=0) / w.sum()  # (8,)

    # 2回目ループで B̄ を使って Ke を積分
    Ke = np.zeros((8, 8), dtype=float)

    for gp_idx, ((xi, eta), detJ) in enumerate(zip(gauss_points, detJ_arr)):
        B = B_arr[gp_idx]  # (3,8)
        b_m_i = b_m[gp_idx]  # (8,)

        # B̄ を構成：εxx,εyy列に (b̄_m - b_m_i)/2 を足す
        Bbar = B.copy()
        delta_b = 0.5 * (b_m_bar - b_m_i)  # (8,)

        # 各DOF列 j に対して、εxx,εyy成分に同じ delta_b[j] を足す
        for j in range(8):
            Bbar[0, j] += delta_b[j]  # εxx 列
            Bbar[1, j] += delta_b[j]  # εyy 列
            # γxy 行 (2) はそのまま

        Ke += (Bbar.T @ D @ Bbar) * detJ * t

    return Ke
