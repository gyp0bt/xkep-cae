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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.core.constitutive import ConstitutiveProtocol
    from xkep_cae.sections.beam import BeamSection2D


@dataclass
class BeamForces2D:
    """2D梁要素の断面力（局所座標系）.

    符号規約:
      - N: 軸力（引張正）
      - V: せん断力（局所y方向正）
      - M: 曲げモーメント（凸下 = sagging で正）

    座標系:
      - x: 梁軸方向（節点1→節点2）
      - y: x軸に直交（反時計回り90度）
    """

    N: float
    V: float
    M: float


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


def eb_beam2d_mass_local(rho: float, A: float, L: float) -> np.ndarray:
    """2D梁の局所整合質量行列 (6x6) を返す.

    DOF順: [u1, v1, θz1, u2, v2, θz2]
    Euler-Bernoulli型の質量行列（Timoshenko梁にも実用的に適用可能）。

    Args:
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        L: 要素長 [m]

    Returns:
        Me: (6, 6) 局所整合質量行列
    """
    m = rho * A * L
    Me = np.zeros((6, 6), dtype=float)

    # 軸方向 (u1, u2) → DOF 0, 3
    Me[0, 0] = m / 3.0
    Me[0, 3] = m / 6.0
    Me[3, 0] = m / 6.0
    Me[3, 3] = m / 3.0

    # 横方向曲げ (v1, θz1, v2, θz2) → DOF 1, 2, 4, 5
    coeff = m / 420.0
    bend_idx = [1, 2, 4, 5]
    M_bend = coeff * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ]
    )
    for i_loc, i_glob in enumerate(bend_idx):
        for j_loc, j_glob in enumerate(bend_idx):
            Me[i_glob, j_glob] = M_bend[i_loc, j_loc]

    return Me


def eb_beam2d_lumped_mass_local(rho: float, A: float, L: float) -> np.ndarray:
    """2D梁の局所集中質量行列 (6x6, 対角) を返す.

    HRZ (Hinton-Rock-Zienkiewicz) 法による集中化。

    DOF順: [u1, v1, θz1, u2, v2, θz2]

    HRZ結果:
        並進: m/2 （各節点・各方向）
        回転: m·L²/78 （各節点）

    Args:
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        L: 要素長 [m]

    Returns:
        Me: (6, 6) 対角集中質量行列
    """
    m = rho * A * L
    diag = np.array([m / 2.0, m / 2.0, m * L**2 / 78.0, m / 2.0, m / 2.0, m * L**2 / 78.0])
    return np.diag(diag)


def eb_beam2d_mass_global(
    coords: np.ndarray,
    rho: float,
    A: float,
    *,
    lumped: bool = False,
) -> np.ndarray:
    """全体座標系での2D梁の質量行列 (6x6) を返す.

    Args:
        coords: (2, 2) 節点座標 [[x1,y1],[x2,y2]]
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        lumped: True の場合は集中質量行列（HRZ法）

    Returns:
        Me_global: (6, 6) 全体座標系の質量行列
    """
    length, c, s = _beam_length_and_cosines(coords)

    if lumped:
        # 集中質量は対角行列 → 座標変換不要（回転不変）
        return eb_beam2d_lumped_mass_local(rho, A, length)

    Me_local = eb_beam2d_mass_local(rho, A, length)
    T = _transformation_matrix_2d(c, s)
    return T.T @ Me_local @ T


class EulerBernoulliBeam2D:
    """2D Euler-Bernoulli 梁要素（ElementProtocol適合）.

    Args:
        section: 梁断面特性（A, I）
    """

    element_type: str = "B21E"
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

    def mass_matrix(
        self,
        coords: np.ndarray,
        rho: float,
        *,
        lumped: bool = False,
    ) -> np.ndarray:
        """全体座標系の質量行列を返す.

        Args:
            coords: (2, 2) 節点座標
            rho: 密度 [kg/m³]
            lumped: True の場合は集中質量行列（HRZ法）

        Returns:
            Me: (6, 6) 全体座標系の質量行列
        """
        return eb_beam2d_mass_global(coords, rho, self.section.A, lumped=lumped)

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

    def section_forces(
        self,
        coords: np.ndarray,
        u_elem_global: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> tuple[BeamForces2D, BeamForces2D]:
        """要素両端の断面力を計算する.

        Args:
            coords: (2, 2) 節点座標
            u_elem_global: (6,) 要素変位ベクトル（全体座標系）
            material: 構成則（E を保持）

        Returns:
            (forces_1, forces_2): 両端の断面力（局所座標系）
        """
        D = material.tangent()
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

        return eb_beam2d_section_forces(
            coords,
            u_elem_global,
            young_e,
            self.section.A,
            self.section.I,
        )


def eb_beam2d_section_forces(
    coords: np.ndarray,
    u_elem_global: np.ndarray,
    E: float,
    A: float,
    I: float,  # noqa: E741
) -> tuple[BeamForces2D, BeamForces2D]:
    """Euler-Bernoulli梁の要素両端の断面力を計算する（局所座標系）.

    全体座標系の変位ベクトルから局所座標系に変換し、
    局所剛性行列を用いて要素端力を求める。

    符号規約:
      - 節点1: 断面力 = -f_local[0:3] （正面の法線が+x方向なので反転）
      - 節点2: 断面力 = f_local[3:6]
      - N 正 = 引張、V 正 = y方向の正のせん断
      - M 正 = 凸下（sagging）

    Args:
        coords: (2, 2) 節点座標
        u_elem_global: (6,) 要素変位ベクトル（全体座標系）
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント

    Returns:
        (forces_1, forces_2): 両端の断面力
    """
    length, c, s = _beam_length_and_cosines(coords)
    T = _transformation_matrix_2d(c, s)
    Ke_local = eb_beam2d_ke_local(E, A, I, length)

    u_local = T @ u_elem_global
    f_local = Ke_local @ u_local

    forces_1 = BeamForces2D(N=-f_local[0], V=-f_local[1], M=-f_local[2])
    forces_2 = BeamForces2D(N=f_local[3], V=f_local[4], M=f_local[5])
    return forces_1, forces_2


def beam2d_max_bending_stress(
    section_forces: BeamForces2D,
    A: float,
    I: float,  # noqa: E741
    y_max: float,
) -> float:
    """断面力から最大曲げ応力（絶対値）を推定する.

    σ_x = N/A ± M·y_max/I

    断面の最外縁での応力の最大絶対値を返す。

    Args:
        section_forces: 断面力
        A: 断面積
        I: 断面二次モーメント
        y_max: 断面の y 方向最外縁距離（中立軸から）

    Returns:
        σ_max: 最大曲げ応力の絶対値
    """
    sigma_axial = section_forces.N / A
    sigma_m = abs(section_forces.M) * y_max / I
    return abs(sigma_axial) + sigma_m


def beam2d_max_shear_stress(
    section_forces: BeamForces2D,
    A: float,
    shape: str = "rectangle",
) -> float:
    """断面力から最大横せん断応力を推定する.

    断面形状に応じた中立軸上の最大せん断応力を返す。

    形状依存の公式:
      - 矩形断面: τ_max = 3V/(2A)
      - 円形断面: τ_max = 4V/(3A)
      - 一般断面: τ_max = V/A（フォールバック）

    Args:
        section_forces: 断面力
        A: 断面積
        shape: 断面形状 ("rectangle", "circle", "general")

    Returns:
        τ_max: 最大横せん断応力の絶対値
    """
    V_abs = abs(section_forces.V)
    if shape == "circle":
        return 4.0 * V_abs / (3.0 * A)
    elif shape == "rectangle":
        return 3.0 * V_abs / (2.0 * A)
    else:
        return V_abs / A
