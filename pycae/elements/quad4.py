from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pycae.core.constitutive import ConstitutiveProtocol


def quad4_ke_plane_strain(
    node_xy: np.ndarray, D: np.ndarray, t: float = 1.0
) -> np.ndarray:
    """Q4一次要素（平面歪み）の局所剛性マトリクスを返す。

    Args:
        node_xy: 要素節点座標 (4,2)。節点順は (-1,-1),(+1,-1),(+1,+1),(-1,+1) を仮定。
        D: 弾性マトリクス (3,3)
        t: 厚み

    Returns:
        Ke: (8,8) 局所剛性
    """
    D_tmp = D.copy()
    D_tmp[0, 1] *= 2
    D_tmp[1, 0] *= 2

    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (4, 2):
        raise ValueError("node_xy は (4,2) である必要があります。")

    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    Ke = np.zeros((8, 8), dtype=float)

    def dN_dxi_eta(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        dN_dxi = 0.25 * np.array(
            [-(1 - eta), +(1 - eta), +(1 + eta), -(1 + eta)], dtype=float
        )
        dN_deta = 0.25 * np.array(
            [-(1 - xi), -(1 + xi), +(1 + xi), +(1 - xi)], dtype=float
        )
        return dN_dxi, dN_deta

    for xi, eta in gauss:
        dN_dxi, dN_deta = dN_dxi_eta(xi, eta)

        J = np.empty((2, 2), dtype=float)
        J[0, 0] = dN_dxi @ node_xy[:, 0]
        J[0, 1] = dN_deta @ node_xy[:, 0]
        J[1, 0] = dN_dxi @ node_xy[:, 1]
        J[1, 1] = dN_deta @ node_xy[:, 1]
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 0.0:
            raise ValueError(f"detJ<=0（反転）の可能性。detJ={detJ:.3e}")
        invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]], dtype=float) / detJ

        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        B = np.zeros((3, 8), dtype=float)
        for i in range(4):
            B[0, 2 * i] = dN_dx[i]  # εxx
            B[1, 2 * i + 1] = dN_dy[i]  # εyy
            B[2, 2 * i] = dN_dy[i]  # γxy
            B[2, 2 * i + 1] = dN_dx[i]  # γxy

        # Ke += (B.T @ D @ B) * detJ * t
        Ke += (B.T @ D_tmp @ B) * detJ * t

    return Ke


class Quad4PlaneStrain:
    """Q4双線形四角形要素（平面ひずみ）（ElementProtocol適合）."""

    ndof_per_node: int = 2
    nnodes: int = 4
    ndof: int = 8

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float,
    ) -> np.ndarray:
        D = material.tangent()
        return quad4_ke_plane_strain(coords, D, thickness)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        edofs = np.empty(self.ndof, dtype=np.int64)
        for i, n in enumerate(node_indices):
            edofs[2 * i] = 2 * n
            edofs[2 * i + 1] = 2 * n + 1
        return edofs
