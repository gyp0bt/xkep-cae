# fem/elements/tri3.py

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pycae.core.constitutive import ConstitutiveProtocol


def tri3_ke_plane_strain(
    node_xy: np.ndarray, D: np.ndarray, t: float = 1.0
) -> np.ndarray:
    """TRI3（一次三角形, 定ひずみ, 平面歪み）の局所剛性 (6x6)。

    Args:
        node_xy: (3,2) 要素節点座標
        D: 弾性マトリクス (3,3)
        t: 厚み

    Returns:
        Ke: (6,6)
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (3, 2):
        raise ValueError("node_xy は (3,2) である必要があります。")

    x1, y1 = node_xy[0]
    x2, y2 = node_xy[1]
    x3, y3 = node_xy[2]

    # 三角形の面積 A
    A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if A <= 0.0:
        raise ValueError("零面積または反転要素（A<=0）")

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    # Bマトリクス（定ひずみ要素）
    B = (1.0 / (2.0 * A)) * np.array(
        [
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3],
        ],
        dtype=float,
    )

    # ★ ここが修正点：2Aではなく A
    Ke = (B.T @ D @ B) * A * t
    return Ke


class Tri3PlaneStrain:
    """TRI3一次三角形要素（平面ひずみ）（ElementProtocol適合）."""

    ndof_per_node: int = 2
    nnodes: int = 3
    ndof: int = 6

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float,
    ) -> np.ndarray:
        D = material.tangent()
        return tri3_ke_plane_strain(coords, D, thickness)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        edofs = np.empty(self.ndof, dtype=np.int64)
        for i, n in enumerate(node_indices):
            edofs[2 * i] = 2 * n
            edofs[2 * i + 1] = 2 * n + 1
        return edofs
