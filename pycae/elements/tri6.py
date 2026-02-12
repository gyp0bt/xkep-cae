# fem/elements/tri6.py

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pycae.core.constitutive import ConstitutiveProtocol

try:
    # numba があれば高速版を有効化
    from numba import njit  # type: ignore

    _USE_NUMBA = True
except Exception:  # ImportError など
    _USE_NUMBA = False

_USE_NUMBA = False


def _tri6_ke_plane_strain_python(
    node_xy: np.ndarray, D: np.ndarray, t: float = 1.0
) -> np.ndarray:
    """TRI6（二次三角形, 平面ひずみ）の局所剛性 (12x12) [pure Python版].

    節点順は Abaqus CPE6 と同じ:
        1,2,3 : 頂点
        4,5,6 : 各辺の中点 (1-2, 2-3, 3-1)

    Args:
        node_xy: (6,2) 要素節点座標 [ [x1,y1], ..., [x6,y6] ]
        D: (3,3) 弾性マトリクス（平面ひずみ）
        t: 厚み

    Returns:
        Ke: (12,12) TRI6 要素剛性マトリクス
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (6, 2):
        raise ValueError("node_xy は (6,2) である必要があります。")

    # --- 面積と面積座標の勾配（頂点3点から計算：TRI3 と同じ） ---
    x1, y1 = node_xy[0]
    x2, y2 = node_xy[1]
    x3, y3 = node_xy[2]

    A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if A <= 0.0:
        raise ValueError("零面積または反転要素（A<=0）")

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    inv2A = 1.0 / (2.0 * A)
    # ∇L_i = [dLi/dx, dLi/dy]
    gradL1 = np.array([b1, c1], dtype=float) * inv2A
    gradL2 = np.array([b2, c2], dtype=float) * inv2A
    gradL3 = np.array([b3, c3], dtype=float) * inv2A

    # --- 3点ガウス積分（2次精度） ---
    gp = [
        (1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0),
        (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0),
        (4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
    ]
    w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    Ke = np.zeros((12, 12), dtype=float)

    for (L1, L2, L3), wi in zip(gp, w):
        # --- dN/dL1, dN/dL2, dN/dL3 を計算 ---
        # N1 = L1(2L1-1)
        dN1_dL1 = 4.0 * L1 - 1.0
        dN1_dL2 = 0.0
        dN1_dL3 = 0.0

        # N2 = L2(2L2-1)
        dN2_dL1 = 0.0
        dN2_dL2 = 4.0 * L2 - 1.0
        dN2_dL3 = 0.0

        # N3 = L3(2L3-1)
        dN3_dL1 = 0.0
        dN3_dL2 = 0.0
        dN3_dL3 = 4.0 * L3 - 1.0

        # N4 = 4 L1 L2
        dN4_dL1 = 4.0 * L2
        dN4_dL2 = 4.0 * L1
        dN4_dL3 = 0.0

        # N5 = 4 L2 L3
        dN5_dL1 = 0.0
        dN5_dL2 = 4.0 * L3
        dN5_dL3 = 4.0 * L2

        # N6 = 4 L3 L1
        dN6_dL1 = 4.0 * L3
        dN6_dL2 = 0.0
        dN6_dL3 = 4.0 * L1

        dN_dL1 = np.array([dN1_dL1, dN2_dL1, dN3_dL1, dN4_dL1, dN5_dL1, dN6_dL1])
        dN_dL2 = np.array([dN1_dL2, dN2_dL2, dN3_dL2, dN4_dL2, dN5_dL2, dN6_dL2])
        dN_dL3 = np.array([dN1_dL3, dN2_dL3, dN3_dL3, dN4_dL3, dN5_dL3, dN6_dL3])

        # --- ∇N_i = dN/dL1 ∇L1 + dN/dL2 ∇L2 + dN/dL3 ∇L3 ---
        gradN = np.zeros((6, 2), dtype=float)
        for i in range(6):
            grad = dN_dL1[i] * gradL1 + dN_dL2[i] * gradL2 + dN_dL3[i] * gradL3
            gradN[i, 0] = grad[0]  # dNi/dx
            gradN[i, 1] = grad[1]  # dNi/dy

        # --- Bマトリクス (3x12): engineering shear γxy 使用 ---
        B = np.zeros((3, 12), dtype=float)
        for i in range(6):
            dNdx = gradN[i, 0]
            dNdy = gradN[i, 1]
            col = 2 * i
            B[0, col] = dNdx  # εxx
            B[1, col + 1] = dNdy  # εyy
            B[2, col] = dNdy  # γxy = du/dy + dv/dx
            B[2, col + 1] = dNdx

        Ke += (B.T @ D @ B) * (A * wi) * t

    return Ke


# ==========================
# Numba 高速版
# ==========================

if _USE_NUMBA:

    @njit(cache=True)
    def _tri6_ke_plane_strain_numba_core(
        node_xy: np.ndarray, D: np.ndarray, t: float
    ) -> np.ndarray:
        """Numba用コア関数（Pythonからは直接呼ばない）."""

        # node_xy: (6,2)
        x1, y1 = node_xy[0, 0], node_xy[0, 1]
        x2, y2 = node_xy[1, 0], node_xy[1, 1]
        x3, y3 = node_xy[2, 0], node_xy[2, 1]

        A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        # A<=0 のチェックは Python 側で済ませる前提でもよいが、
        # ここでは 0 に潰しておく（変な値になったらすぐ気付く）
        if A <= 0.0:
            # numba 内の例外は雑に扱うので、とりあえず非常に小さい値で逃がす
            A = 1e-30

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        inv2A = 1.0 / (2.0 * A)
        gradL1 = np.array([b1, c1]) * inv2A
        gradL2 = np.array([b2, c2]) * inv2A
        gradL3 = np.array([b3, c3]) * inv2A

        # Gauss点3つ（L1,L2,L3, weight）
        L1s = np.array([1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0])
        L2s = np.array([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0])
        L3s = np.array([4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        ws = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        Ke = np.zeros((12, 12))

        for igp in range(3):
            L1 = L1s[igp]
            L2 = L2s[igp]
            L3 = L3s[igp]
            wi = ws[igp]

            # dN/dL
            dN_dL1 = np.zeros(6)
            dN_dL2 = np.zeros(6)
            dN_dL3 = np.zeros(6)

            # N1 = L1(2L1-1)
            dN_dL1[0] = 4.0 * L1 - 1.0

            # N2 = L2(2L2-1)
            dN_dL2[1] = 4.0 * L2 - 1.0

            # N3 = L3(2L3-1)
            dN_dL3[2] = 4.0 * L3 - 1.0

            # N4 = 4 L1 L2
            dN_dL1[3] = 4.0 * L2
            dN_dL2[3] = 4.0 * L1

            # N5 = 4 L2 L3
            dN_dL2[4] = 4.0 * L3
            dN_dL3[4] = 4.0 * L2

            # N6 = 4 L3 L1
            dN_dL1[5] = 4.0 * L3
            dN_dL3[5] = 4.0 * L1

            # gradN (6,2)
            gradN = np.zeros((6, 2))
            for i in range(6):
                grad = dN_dL1[i] * gradL1 + dN_dL2[i] * gradL2 + dN_dL3[i] * gradL3
                gradN[i, 0] = grad[0]
                gradN[i, 1] = grad[1]

            # B (3,12)
            B = np.zeros((3, 12))
            for i in range(6):
                dNdx = gradN[i, 0]
                dNdy = gradN[i, 1]
                col = 2 * i
                B[0, col] = dNdx
                B[1, col + 1] = dNdy
                B[2, col] = dNdy
                B[2, col + 1] = dNdx

            # Ke += (B.T @ D @ B) * (A * wi) * t
            # numba では @ が np.dot に展開される
            tmp = B.T @ D
            Ke += (tmp @ B) * (A * wi) * t

        return Ke

    def tri6_ke_plane_strain(
        node_xy: np.ndarray, D: np.ndarray, t: float = 1.0
    ) -> np.ndarray:
        """TRI6（二次三角形, 平面ひずみ）の局所剛性 (Numba 利用時はJIT)."""
        node_xy = np.asarray(node_xy, dtype=np.float64)
        if node_xy.shape != (6, 2):
            raise ValueError("node_xy は (6,2) である必要があります。")
        D = np.asarray(D, dtype=np.float64)

        # 面積チェックだけは Python 側でやっておくとエラーがわかりやすい
        x1, y1 = node_xy[0]
        x2, y2 = node_xy[1]
        x3, y3 = node_xy[2]
        A = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if A <= 0.0:
            raise ValueError("零面積または反転要素（A<=0）")

        # JIT コアに投げる
        return _tri6_ke_plane_strain_numba_core(node_xy, D, t)

else:
    # numba なし環境では従来のPython版をそのまま使う
    def tri6_ke_plane_strain(
        node_xy: np.ndarray, D: np.ndarray, t: float = 1.0
    ) -> np.ndarray:
        """TRI6（二次三角形, 平面ひずみ）の局所剛性 (numbaなし)."""
        return _tri6_ke_plane_strain_python(node_xy, D, t)


class Tri6PlaneStrain:
    """TRI6二次三角形要素（平面ひずみ）（ElementProtocol適合）."""

    ndof_per_node: int = 2
    nnodes: int = 6
    ndof: int = 12

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float,
    ) -> np.ndarray:
        D = material.tangent()
        return tri6_ke_plane_strain(coords, D, thickness)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        edofs = np.empty(self.ndof, dtype=np.int64)
        for i, n in enumerate(node_indices):
            edofs[2 * i] = 2 * n
            edofs[2 * i + 1] = 2 * n + 1
        return edofs
