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

from xkep_cae.elements.beam_eb2d import (
    BeamForces2D,
    _beam_length_and_cosines,
    _transformation_matrix_2d,
)

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
    scf: float | None = None,
) -> np.ndarray:
    """Timoshenko梁の局所剛性行列 (6x6) を返す.

    DOF順: [u1, v1, θ1, u2, v2, θ2]（局所座標系）

    せん断パラメータ Φ = 12EI/(κGAL²) を用いた整合定式化。

    SCF（スレンダネス補償係数）を指定すると、横せん断パラメータΦが
    スレンダネスに応じて低減され、細長い梁でEB梁に遷移する:
      f_p = 1 / (1 + SCF · L²A/(12I))
      Φ_eff = Φ · f_p

    太い梁 (L²A/(12I) ≈ 1): Φ_eff ≈ Φ（通常のTimoshenko）
    細長い梁 (L²A/(12I) >> 1): Φ_eff → 0（EB梁に遷移）

    Args:
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント
        L: 要素長さ
        kappa: せん断補正係数（矩形: 5/6, 円形: 6/7）
        G: せん断弾性率
        scf: スレンダネス補償係数（None=無効, Abaqusデフォルト=0.25）

    Returns:
        Ke: (6, 6) 局所剛性行列
    """
    EA_L = E * A / L
    EI_L3 = E * I / (L * L * L)
    EI_L2 = E * I / (L * L)
    EI_L = E * I / L

    # せん断パラメータ
    Phi = 12.0 * E * I / (kappa * G * A * L * L)

    # SCF適用: 細長い梁でΦを低減しEB梁に遷移させる
    # f_p = 1 / (1 + SCF · L²A/(12I))
    # Φ_eff = Φ · f_p → Φ_eff → 0 でEB梁に遷移
    if scf is not None and scf > 0.0:
        slenderness = L * L * A / (12.0 * I)
        f_p = 1.0 / (1.0 + scf * slenderness)
        Phi = Phi * f_p

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
    scf: float | None = None,
) -> np.ndarray:
    """全体座標系での Timoshenko 梁の剛性行列 (6x6) を返す.

    Args:
        coords: (2, 2) 節点座標 [[x1,y1],[x2,y2]]
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント
        kappa: せん断補正係数
        G: せん断弾性率
        scf: スレンダネス補償係数（None=無効, Abaqusデフォルト=0.25）

    Returns:
        Ke_global: (6, 6) 全体座標系の剛性行列
    """
    length, c, s = _beam_length_and_cosines(coords)
    Ke_local = timo_beam2d_ke_local(E, A, I, length, kappa, G, scf=scf)
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
        section: 梁断面特性（A, I, shape）
        kappa: せん断補正係数。以下のいずれか:
            - float: 固定値（旧デフォルト: 5/6）
            - "cowper": Cowper (1966) のν依存係数を自動計算（Abaqus準拠）
        scf: スレンダネス補償係数（Slenderness Compensation Factor）。
            None（デフォルト）: SCF無効（純粋なTimoshenko梁）
            0.25: Abaqusデフォルト。細長い梁ではEB梁に自動遷移する。
            0.0: SCF無効（Noneと同等）

    Abaqusとの差異:
        Abaqus B21/B22 はデフォルトで Cowper のν依存κを使用する。
        xkep-cae のデフォルトは κ=5/6（固定値）だが、kappa="cowper" を
        指定することで Abaqus と同等の動作になる。
        ν=0.3・矩形断面の場合: Cowper κ≈0.850 vs 固定 κ=0.833（約2%差）。

    SCF について:
        Abaqus B21/B22 はデフォルトで SCF=0.25 を適用する。
        SCFは横せん断剛性をスレンダネスに応じて低減する補正:
          f_p = 1 / (1 + SCF · L²A/(12I))
        細長い梁ほど f_p → 0 となり、横せん断剛性が実質ゼロ
        （＝Euler-Bernoulli梁の挙動）に遷移する。
        Abaqusとの比較時にSCFを合わせるか、Abaqus側で SCF=0 に
        設定して無効化する必要がある（abaqus-differences.md 参照）。
    """

    ndof_per_node: int = 3
    nnodes: int = 2
    ndof: int = 6

    def __init__(
        self,
        section: BeamSection2D,
        kappa: float | str = 5.0 / 6.0,
        scf: float | None = None,
    ) -> None:
        self.section = section
        self.scf = scf
        if isinstance(kappa, str):
            if kappa != "cowper":
                raise ValueError(f"kappa に指定できる文字列は 'cowper' のみです: '{kappa}'")
            self._kappa_mode = "cowper"
            self._kappa_value: float | None = None
        else:
            self._kappa_mode = "fixed"
            self._kappa_value = float(kappa)

    @property
    def kappa(self) -> float | str:
        """現在のせん断補正係数設定を返す."""
        if self._kappa_mode == "cowper":
            return "cowper"
        assert self._kappa_value is not None
        return self._kappa_value

    def _resolve_kappa(self, nu: float) -> float:
        """材料のνからκを解決する."""
        if self._kappa_mode == "cowper":
            return self.section.cowper_kappa(nu)
        assert self._kappa_value is not None
        return self._kappa_value

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        """全体座標系の剛性行列を返す.

        Args:
            coords: (2, 2) 節点座標
            material: 構成則（E, nu を保持。tangent() が E を返す）
            thickness: 未使用（梁要素では断面特性で管理）

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

        # ポアソン比の取得（Cowperκ計算用）
        nu = float(material.nu) if hasattr(material, "nu") else 0.3

        # せん断弾性率はmaterialから取得（BeamElastic1Dの場合はG属性を持つ）
        if hasattr(material, "G"):
            shear_g = float(material.G)
        elif hasattr(material, "nu"):
            shear_g = young_e / (2.0 * (1.0 + nu))
        else:
            raise ValueError(
                "材料オブジェクトからせん断弾性率を取得できません。G or nu が必要です。"
            )

        kappa_val = self._resolve_kappa(nu)

        return timo_beam2d_ke_global(
            coords, young_e, self.section.A, self.section.I, kappa_val, shear_g,
            scf=self.scf,
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
            material: 構成則（E, nu を保持）

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

        nu = float(material.nu) if hasattr(material, "nu") else 0.3
        if hasattr(material, "G"):
            shear_g = float(material.G)
        elif hasattr(material, "nu"):
            shear_g = young_e / (2.0 * (1.0 + nu))
        else:
            raise ValueError(
                "材料オブジェクトからせん断弾性率を取得できません。G or nu が必要です。"
            )

        kappa_val = self._resolve_kappa(nu)

        return timo_beam2d_section_forces(
            coords, u_elem_global,
            young_e, self.section.A, self.section.I,
            kappa_val, shear_g,
            scf=self.scf,
        )


def timo_beam2d_section_forces(
    coords: np.ndarray,
    u_elem_global: np.ndarray,
    E: float,
    A: float,
    I: float,  # noqa: E741
    kappa: float,
    G: float,
    scf: float | None = None,
) -> tuple[BeamForces2D, BeamForces2D]:
    """Timoshenko梁の要素両端の断面力を計算する（局所座標系）.

    全体座標系の変位ベクトルから局所座標系に変換し、
    局所剛性行列を用いて要素端力を求める。

    符号規約:
      - 節点1: 断面力 = -f_local[0:3]
      - 節点2: 断面力 = f_local[3:6]
      - N 正 = 引張、V 正 = y方向の正のせん断
      - M 正 = 凸下（sagging）

    Args:
        coords: (2, 2) 節点座標
        u_elem_global: (6,) 要素変位ベクトル（全体座標系）
        E: ヤング率
        A: 断面積
        I: 断面二次モーメント
        kappa: せん断補正係数
        G: せん断弾性率
        scf: スレンダネス補償係数

    Returns:
        (forces_1, forces_2): 両端の断面力
    """
    length, c, s = _beam_length_and_cosines(coords)
    T = _transformation_matrix_2d(c, s)
    Ke_local = timo_beam2d_ke_local(E, A, I, length, kappa, G, scf=scf)

    u_local = T @ u_elem_global
    f_local = Ke_local @ u_local

    forces_1 = BeamForces2D(N=-f_local[0], V=-f_local[1], M=-f_local[2])
    forces_2 = BeamForces2D(N=f_local[3], V=f_local[4], M=f_local[5])
    return forces_1, forces_2
