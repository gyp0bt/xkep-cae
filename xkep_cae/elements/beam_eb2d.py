"""2D Euler-Bernoulli 梁要素.

各節点の自由度: (ux, uy, θz) → 3 DOF/node, 2 nodes → 6 DOF/element

局所座標系:
  - x軸: 節点1→節点2 方向
  - y軸: x軸に直交（反時計回り90度）

局所剛性行列 Ke_local:
  - 軸方向: EA/L
  - 曲げ: Hermite補間に基づく EI 項

座標変換:
  Ke_global = T^T @ Ke_local @ T
  T は 6x6 のブロック対角回転行列
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.core.constitutive import ConstitutiveProtocol
    from xkep_cae.sections.beam import BeamSection2D


def _beam_length_and_cosines(coords: np.ndarray) -> tuple[float, float, float]:
    """梁要素の長さと方向余弦を計算する.

    Args:
        coords: (2, 2) 節点座標 [[x1,y1],[x2,y2]]

    Returns:
        L: 要素長さ
        c: cos(θ)
        s: sin(θ)
    """
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-15:
        raise ValueError("要素長さがほぼゼロです。2節点が同一座標です。")
    return float(length), float(dx / length), float(dy / length)


def _transformation_matrix_2d(c: float, s: float) -> np.ndarray:
    """2D梁の座標変換行列 T (6x6) を返す.

    局所座標 → 全体座標の変換: u_local = T @ u_global
    Ke_global = T^T @ Ke_local @ T

    Args:
        c: cos(θ)
        s: sin(θ)

    Returns:
        T: (6, 6) 座標変換行列
    """
    T = np.zeros((6, 6), dtype=float)
    # 節点1
    T[0, 0] = c
    T[0, 1] = s
    T[1, 0] = -s
    T[1, 1] = c
    T[2, 2] = 1.0
    # 節点2
    T[3, 3] = c
    T[3, 4] = s
    T[4, 3] = -s
    T[4, 4] = c
    T[5, 5] = 1.0
    return T


def eb_beam2d_ke_local(E: float, A: float, I: float, L: float) -> np.ndarray:  # noqa: E741
    """Euler-Bernoulli梁の局所剛性行列 (6x6) を返す.

    DOF順: [u1, v1, θ1, u2, v2, θ2]（局所座標系）

    Args:
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント
        L: 要素長さ

    Returns:
        Ke: (6, 6) 局所剛性行列
    """
    EA_L = E * A / L
    EI_L3 = E * I / (L * L * L)
    EI_L2 = E * I / (L * L)
    EI_L = E * I / L

    Ke = np.zeros((6, 6), dtype=float)

    # 軸方向 (DOF 0, 3)
    Ke[0, 0] = EA_L
    Ke[0, 3] = -EA_L
    Ke[3, 0] = -EA_L
    Ke[3, 3] = EA_L

    # 曲げ (DOF 1,2, 4,5) — Hermite補間
    Ke[1, 1] = 12.0 * EI_L3
    Ke[1, 2] = 6.0 * EI_L2
    Ke[1, 4] = -12.0 * EI_L3
    Ke[1, 5] = 6.0 * EI_L2

    Ke[2, 1] = 6.0 * EI_L2
    Ke[2, 2] = 4.0 * EI_L
    Ke[2, 4] = -6.0 * EI_L2
    Ke[2, 5] = 2.0 * EI_L

    Ke[4, 1] = -12.0 * EI_L3
    Ke[4, 2] = -6.0 * EI_L2
    Ke[4, 4] = 12.0 * EI_L3
    Ke[4, 5] = -6.0 * EI_L2

    Ke[5, 1] = 6.0 * EI_L2
    Ke[5, 2] = 2.0 * EI_L
    Ke[5, 4] = -6.0 * EI_L2
    Ke[5, 5] = 4.0 * EI_L

    return Ke


def eb_beam2d_ke_global(
    coords: np.ndarray,
    E: float,
    A: float,
    I: float,  # noqa: E741
) -> np.ndarray:
    """全体座標系での Euler-Bernoulli 梁の剛性行列 (6x6) を返す.

    Args:
        coords: (2, 2) 節点座標 [[x1,y1],[x2,y2]]
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント

    Returns:
        Ke_global: (6, 6) 全体座標系の剛性行列
    """
    length, c, s = _beam_length_and_cosines(coords)
    Ke_local = eb_beam2d_ke_local(E, A, I, length)
    T = _transformation_matrix_2d(c, s)
    return T.T @ Ke_local @ T


def eb_beam2d_distributed_load(
    coords: np.ndarray,
    qy_local: float,
) -> np.ndarray:
    """等分布荷重の等価節点力ベクトル（全体座標系）を返す.

    局所y方向に一様分布荷重 qy_local [force/length] が作用する場合の等価節点力。

    Args:
        coords: (2, 2) 節点座標 [[x1,y1],[x2,y2]]
        qy_local: 局所y方向の分布荷重強度

    Returns:
        f_global: (6,) 全体座標系の等価節点力ベクトル
    """
    length, c, s = _beam_length_and_cosines(coords)
    q = qy_local

    # 局所座標系の等価節点力
    f_local = np.array(
        [
            0.0,
            q * length / 2.0,
            q * length**2 / 12.0,
            0.0,
            q * length / 2.0,
            -q * length**2 / 12.0,
        ],
        dtype=float,
    )

    T = _transformation_matrix_2d(c, s)
    return T.T @ f_local


class EulerBernoulliBeam2D:
    """2D Euler-Bernoulli 梁要素（ElementProtocol適合）.

    Args:
        section: 梁断面特性（A, I）
    """

    ndof_per_node: int = 3
    nnodes: int = 2
    ndof: int = 6

    def __init__(self, section: BeamSection2D) -> None:
        self.section = section

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        """全体座標系の剛性行列を返す.

        Args:
            coords: (2, 2) 節点座標
            material: 構成則（tangent() が E を返す）
            thickness: 未使用（梁要素では断面特性で管理）

        Returns:
            Ke: (6, 6) 全体座標系の剛性行列
        """
        D = material.tangent()
        # 1Dの場合: スカラー or (1,) or (1,1) → E値を抽出
        if np.ndim(D) == 0:
            young_e = float(D)
        elif D.shape == (1,):
            young_e = float(D[0])
        elif D.shape == (1, 1):
            young_e = float(D[0, 0])
        else:
            raise ValueError(
                f"梁要素にはスカラーまたは(1,1)の弾性テンソルが必要です。shape={D.shape}"
            )
        return eb_beam2d_ke_global(coords, young_e, self.section.A, self.section.I)

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """グローバル節点インデックスから要素DOFインデックスを返す.

        3 DOF/node: (ux, uy, θz)

        Args:
            node_indices: (2,) グローバル節点インデックス

        Returns:
            edofs: (6,) グローバルDOFインデックス
        """
        edofs = np.empty(self.ndof, dtype=np.int64)
        for idx, n in enumerate(node_indices):
            edofs[3 * idx] = 3 * n
            edofs[3 * idx + 1] = 3 * n + 1
            edofs[3 * idx + 2] = 3 * n + 2
        return edofs

    def distributed_load(
        self,
        coords: np.ndarray,
        qy_local: float,
    ) -> np.ndarray:
        """等分布荷重の等価節点力ベクトル（全体座標系）.

        Args:
            coords: (2, 2) 節点座標
            qy_local: 局所y方向の分布荷重強度

        Returns:
            f_global: (6,) 全体座標系の等価節点力
        """
        return eb_beam2d_distributed_load(coords, qy_local)
