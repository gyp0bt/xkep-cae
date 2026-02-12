"""2D Timoshenko 梁要素.

Euler-Bernoulli梁に対して、せん断変形を考慮する。
各節点の自由度: (ux, uy, θz) → 3 DOF/node, 2 nodes → 6 DOF/element

Timoshenko梁の剛性行列は、せん断パラメータ Φ = 12EI/(κGAL²) を導入して
Euler-Bernoulli梁の曲げ項を修正した形で表現される。

Φ → 0 のとき Euler-Bernoulli 梁に一致する（細長い梁の極限）。

せん断ロッキング対策:
  Hermite+せん断修正の整合定式化を採用。Φが剛性行列の分母に入る形式で、
  L→0 でもロッキングが発生しない。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xkep_cae.elements.beam_eb2d import _beam_length_and_cosines, _transformation_matrix_2d

if TYPE_CHECKING:
    from xkep_cae.core.constitutive import ConstitutiveProtocol
    from xkep_cae.sections.beam import BeamSection2D


def timo_beam2d_ke_local(
    E: float,
    A: float,
    I: float,  # noqa: E741
    L: float,
    kappa: float,
    G: float,
) -> np.ndarray:
    """Timoshenko梁の局所剛性行列 (6x6) を返す.

    DOF順: [u1, v1, θ1, u2, v2, θ2]（局所座標系）

    せん断パラメータ Φ = 12EI/(κGAL²) を用いた整合定式化。

    Args:
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント
        L: 要素長さ
        kappa: せん断補正係数（矩形: 5/6, 円形: 6/7）
        G: せん断弾性率

    Returns:
        Ke: (6, 6) 局所剛性行列
    """
    EA_L = E * A / L
    EI_L3 = E * I / (L * L * L)
    EI_L2 = E * I / (L * L)
    EI_L = E * I / L

    # せん断パラメータ
    Phi = 12.0 * E * I / (kappa * G * A * L * L)
    denom = 1.0 + Phi

    Ke = np.zeros((6, 6), dtype=float)

    # 軸方向（せん断の影響なし）
    Ke[0, 0] = EA_L
    Ke[0, 3] = -EA_L
    Ke[3, 0] = -EA_L
    Ke[3, 3] = EA_L

    # 曲げ + せん断修正
    Ke[1, 1] = 12.0 * EI_L3 / denom
    Ke[1, 2] = 6.0 * EI_L2 / denom
    Ke[1, 4] = -12.0 * EI_L3 / denom
    Ke[1, 5] = 6.0 * EI_L2 / denom

    Ke[2, 1] = 6.0 * EI_L2 / denom
    Ke[2, 2] = (4.0 + Phi) * EI_L / denom
    Ke[2, 4] = -6.0 * EI_L2 / denom
    Ke[2, 5] = (2.0 - Phi) * EI_L / denom

    Ke[4, 1] = -12.0 * EI_L3 / denom
    Ke[4, 2] = -6.0 * EI_L2 / denom
    Ke[4, 4] = 12.0 * EI_L3 / denom
    Ke[4, 5] = -6.0 * EI_L2 / denom

    Ke[5, 1] = 6.0 * EI_L2 / denom
    Ke[5, 2] = (2.0 - Phi) * EI_L / denom
    Ke[5, 4] = -6.0 * EI_L2 / denom
    Ke[5, 5] = (4.0 + Phi) * EI_L / denom

    return Ke


def timo_beam2d_ke_global(
    coords: np.ndarray,
    E: float,
    A: float,
    I: float,  # noqa: E741
    kappa: float,
    G: float,
) -> np.ndarray:
    """全体座標系での Timoshenko 梁の剛性行列 (6x6) を返す.

    Args:
        coords: (2, 2) 節点座標 [[x1,y1],[x2,y2]]
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント
        kappa: せん断補正係数
        G: せん断弾性率

    Returns:
        Ke_global: (6, 6) 全体座標系の剛性行列
    """
    length, c, s = _beam_length_and_cosines(coords)
    Ke_local = timo_beam2d_ke_local(E, A, I, length, kappa, G)
    T = _transformation_matrix_2d(c, s)
    return T.T @ Ke_local @ T


def timo_beam2d_distributed_load(
    coords: np.ndarray,
    qy_local: float,
) -> np.ndarray:
    """等分布荷重の等価節点力ベクトル（全体座標系）を返す.

    Timoshenko梁でも等分布荷重の等価節点力はEuler-Bernoulli梁と同じ。
    （整合荷重ベクトルを使う場合はΦの影響が出るが、
    通常の等分布荷重については同じ結果になる）

    Args:
        coords: (2, 2) 節点座標 [[x1,y1],[x2,y2]]
        qy_local: 局所y方向の分布荷重強度

    Returns:
        f_global: (6,) 全体座標系の等価節点力ベクトル
    """
    length, c, s = _beam_length_and_cosines(coords)
    q = qy_local

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


class TimoshenkoBeam2D:
    """2D Timoshenko 梁要素（ElementProtocol適合）.

    Args:
        section: 梁断面特性（A, I）
        kappa: せん断補正係数（デフォルト: 5/6、矩形断面用）
    """

    ndof_per_node: int = 3
    nnodes: int = 2
    ndof: int = 6

    def __init__(self, section: BeamSection2D, kappa: float = 5.0 / 6.0) -> None:
        self.section = section
        self.kappa = kappa

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float = 0.0,
    ) -> np.ndarray:
        """全体座標系の剛性行列を返す.

        Args:
            coords: (2, 2) 節点座標
            material: 構成則（E, nu を保持。tangent() が E を返す）
            thickness: 未使用（平面要素との互換性のため保持）

        Returns:
            Ke: (6, 6) 全体座標系の剛性行列
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

        # せん断弾性率はmaterialから取得（BeamElastic1Dの場合はG属性を持つ）
        if hasattr(material, "G"):
            shear_g = float(material.G)
        elif hasattr(material, "nu"):
            shear_g = young_e / (2.0 * (1.0 + float(material.nu)))
        else:
            raise ValueError(
                "材料オブジェクトからせん断弾性率を取得できません。G or nu が必要です。"
            )

        return timo_beam2d_ke_global(
            coords, young_e, self.section.A, self.section.I, self.kappa, shear_g
        )

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """グローバル節点インデックスから要素DOFインデックスを返す."""
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
        """等分布荷重の等価節点力ベクトル（全体座標系）."""
        return timo_beam2d_distributed_load(coords, qy_local)
