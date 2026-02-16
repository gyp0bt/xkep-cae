"""3D Timoshenko 梁要素.

各節点の自由度: (ux, uy, uz, θx, θy, θz) → 6 DOF/node, 2 nodes → 12 DOF/element

局所座標系:
  - x軸: 節点1→節点2 方向
  - y軸: ユーザー指定 or 自動選択（断面の第1主軸方向）
  - z軸: x × y（右手系）

局所剛性行列 Ke_local:
  - 軸方向: EA/L
  - ねじり: GJ/L
  - xz面曲げ: EIy with Φy = 12EIy/(κy·G·A·L²)
  - xy面曲げ: EIz with Φz = 12EIz/(κz·G·A·L²)

座標変換:
  Ke_global = T^T @ Ke_local @ T
  T は 12x12 のブロック対角回転行列（3x3 回転行列の4ブロック繰り返し）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.core.constitutive import ConstitutiveProtocol
    from xkep_cae.sections.beam import BeamSection


@dataclass
class BeamForces3D:
    """3D梁要素の断面力（局所座標系）.

    符号規約:
      - N: 軸力（引張正）
      - Vy: y方向せん断力
      - Vz: z方向せん断力
      - Mx: ねじりモーメント（トルク）
      - My: y軸まわり曲げモーメント
      - Mz: z軸まわり曲げモーメント

    座標系:
      - x: 梁軸方向（節点1→節点2）
      - y: 断面第1主軸方向
      - z: x × y（右手系）
    """

    N: float
    Vy: float
    Vz: float
    Mx: float
    My: float
    Mz: float


def _beam3d_length_and_direction(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """3D梁要素の長さと方向ベクトルを計算する.

    Args:
        coords: (2, 3) 節点座標 [[x1,y1,z1],[x2,y2,z2]]

    Returns:
        L: 要素長さ
        e_x: 単位方向ベクトル（節点1→節点2）
    """
    dx = coords[1] - coords[0]
    length = float(np.linalg.norm(dx))
    if length < 1e-15:
        raise ValueError("要素長さがほぼゼロです。2節点が同一座標です。")
    return length, dx / length


def _build_local_axes(
    e_x: np.ndarray,
    v_ref: np.ndarray | None = None,
) -> np.ndarray:
    """局所座標系の回転行列 R (3x3) を構築する.

    R の行ベクトルが局所 x, y, z 軸を全体座標系で表したもの:
      R[0,:] = e_x  （梁軸方向）
      R[1,:] = e_y  （断面第1主軸方向）
      R[2,:] = e_z  （e_x × e_y）

    Args:
        e_x: 梁軸方向の単位ベクトル
        v_ref: 参照ベクトル（局所y軸を定義するヒント）。
            None の場合、梁軸に最も直交する座標軸を自動選択。

    Returns:
        R: (3, 3) 回転行列（局所→全体変換の転置、すなわち全体→局所変換）
    """
    if v_ref is None:
        # 梁軸に最も直交する座標軸を選択
        abs_ex = np.abs(e_x)
        if abs_ex[0] <= abs_ex[1] and abs_ex[0] <= abs_ex[2]:
            v_ref = np.array([1.0, 0.0, 0.0])
        elif abs_ex[1] <= abs_ex[2]:
            v_ref = np.array([0.0, 1.0, 0.0])
        else:
            v_ref = np.array([0.0, 0.0, 1.0])

    # Gram-Schmidt: e_z = e_x × v_ref, e_y = e_z × e_x
    e_z = np.cross(e_x, v_ref)
    norm_ez = np.linalg.norm(e_z)
    if norm_ez < 1e-10:
        raise ValueError(f"参照ベクトルが梁軸と平行です。v_ref={v_ref}, e_x={e_x}")
    e_z = e_z / norm_ez
    e_y = np.cross(e_z, e_x)

    R = np.zeros((3, 3), dtype=float)
    R[0, :] = e_x
    R[1, :] = e_y
    R[2, :] = e_z
    return R


def _transformation_matrix_3d(R: np.ndarray) -> np.ndarray:
    """3D梁の座標変換行列 T (12x12) を返す.

    各節点の6 DOFブロック [ux, uy, uz, θx, θy, θz] に対して、
    変位と回転の両方に同じ回転行列 R を適用する。

    u_local = T @ u_global
    Ke_global = T^T @ Ke_local @ T

    Args:
        R: (3, 3) 回転行列

    Returns:
        T: (12, 12) 座標変換行列
    """
    T = np.zeros((12, 12), dtype=float)
    # 4つのブロック: 節点1変位, 節点1回転, 節点2変位, 節点2回転
    for i in range(4):
        T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = R
    return T


def timo_beam3d_ke_local(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    L: float,
    kappa_y: float,
    kappa_z: float,
    scf: float | None = None,
) -> np.ndarray:
    """3D Timoshenko梁の局所剛性行列 (12x12) を返す.

    DOF順: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    局所座標系:
      - u: 軸方向（x）
      - v: y方向（xy面内曲げ）
      - w: z方向（xz面内曲げ）
      - θx: ねじり
      - θy: xz面内回転（w のたわみに対応）
      - θz: xy面内回転（v のたわみに対応）

    xy面曲げ（v たわみ, θz 回転）: EIz, Φz = 12EIz/(κz·G·A·L²)
    xz面曲げ（w たわみ, θy 回転）: EIy, Φy = 12EIy/(κy·G·A·L²)

    Args:
        E: ヤング率
        G: せん断弾性率
        A: 断面積
        Iy: y軸まわり断面二次モーメント（xz面曲げ）
        Iz: z軸まわり断面二次モーメント（xy面曲げ）
        J: ねじり定数
        L: 要素長さ
        kappa_y: y方向せん断補正係数
        kappa_z: z方向せん断補正係数
        scf: スレンダネス補償係数（None=無効）

    Returns:
        Ke: (12, 12) 局所剛性行列
    """
    # せん断パラメータ
    Phi_y = 12.0 * E * Iy / (kappa_y * G * A * L * L)
    Phi_z = 12.0 * E * Iz / (kappa_z * G * A * L * L)

    # SCF適用: 細長い梁でΦを低減しEB梁に遷移させる
    if scf is not None and scf > 0.0:
        # xz面曲げのスレンダネス（Iyベース）
        slenderness_y = L * L * A / (12.0 * Iy)
        f_p_y = 1.0 / (1.0 + scf * slenderness_y)
        Phi_y = Phi_y * f_p_y

        # xy面曲げのスレンダネス（Izベース）
        slenderness_z = L * L * A / (12.0 * Iz)
        f_p_z = 1.0 / (1.0 + scf * slenderness_z)
        Phi_z = Phi_z * f_p_z

    denom_y = 1.0 + Phi_y
    denom_z = 1.0 + Phi_z

    Ke = np.zeros((12, 12), dtype=float)

    # --- 軸方向 (DOF 0, 6) ---
    EA_L = E * A / L
    Ke[0, 0] = EA_L
    Ke[0, 6] = -EA_L
    Ke[6, 0] = -EA_L
    Ke[6, 6] = EA_L

    # --- ねじり (DOF 3, 9) ---
    GJ_L = G * J / L
    Ke[3, 3] = GJ_L
    Ke[3, 9] = -GJ_L
    Ke[9, 3] = -GJ_L
    Ke[9, 9] = GJ_L

    # --- xy面曲げ: v たわみ (DOF 1, 7) と θz 回転 (DOF 5, 11) ---
    # EIz ベース
    EIz_L3 = E * Iz / (L**3)
    EIz_L2 = E * Iz / (L**2)
    EIz_L = E * Iz / L

    Ke[1, 1] = 12.0 * EIz_L3 / denom_z
    Ke[1, 5] = 6.0 * EIz_L2 / denom_z
    Ke[1, 7] = -12.0 * EIz_L3 / denom_z
    Ke[1, 11] = 6.0 * EIz_L2 / denom_z

    Ke[5, 1] = 6.0 * EIz_L2 / denom_z
    Ke[5, 5] = (4.0 + Phi_z) * EIz_L / denom_z
    Ke[5, 7] = -6.0 * EIz_L2 / denom_z
    Ke[5, 11] = (2.0 - Phi_z) * EIz_L / denom_z

    Ke[7, 1] = -12.0 * EIz_L3 / denom_z
    Ke[7, 5] = -6.0 * EIz_L2 / denom_z
    Ke[7, 7] = 12.0 * EIz_L3 / denom_z
    Ke[7, 11] = -6.0 * EIz_L2 / denom_z

    Ke[11, 1] = 6.0 * EIz_L2 / denom_z
    Ke[11, 5] = (2.0 - Phi_z) * EIz_L / denom_z
    Ke[11, 7] = -6.0 * EIz_L2 / denom_z
    Ke[11, 11] = (4.0 + Phi_z) * EIz_L / denom_z

    # --- xz面曲げ: w たわみ (DOF 2, 8) と θy 回転 (DOF 4, 10) ---
    # EIy ベース
    # 注意: w-θy の符号規約は v-θz と逆（右手系の整合性）
    # θy > 0 は w が減少する方向 → カップリング項の符号が反転
    EIy_L3 = E * Iy / (L**3)
    EIy_L2 = E * Iy / (L**2)
    EIy_L = E * Iy / L

    Ke[2, 2] = 12.0 * EIy_L3 / denom_y
    Ke[2, 4] = -6.0 * EIy_L2 / denom_y
    Ke[2, 8] = -12.0 * EIy_L3 / denom_y
    Ke[2, 10] = -6.0 * EIy_L2 / denom_y

    Ke[4, 2] = -6.0 * EIy_L2 / denom_y
    Ke[4, 4] = (4.0 + Phi_y) * EIy_L / denom_y
    Ke[4, 8] = 6.0 * EIy_L2 / denom_y
    Ke[4, 10] = (2.0 - Phi_y) * EIy_L / denom_y

    Ke[8, 2] = -12.0 * EIy_L3 / denom_y
    Ke[8, 4] = 6.0 * EIy_L2 / denom_y
    Ke[8, 8] = 12.0 * EIy_L3 / denom_y
    Ke[8, 10] = 6.0 * EIy_L2 / denom_y

    Ke[10, 2] = -6.0 * EIy_L2 / denom_y
    Ke[10, 4] = (2.0 - Phi_y) * EIy_L / denom_y
    Ke[10, 8] = 6.0 * EIy_L2 / denom_y
    Ke[10, 10] = (4.0 + Phi_y) * EIy_L / denom_y

    return Ke


def timo_beam3d_ke_global(
    coords: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
    v_ref: np.ndarray | None = None,
    scf: float | None = None,
) -> np.ndarray:
    """全体座標系での 3D Timoshenko 梁の剛性行列 (12x12) を返す.

    Args:
        coords: (2, 3) 節点座標 [[x1,y1,z1],[x2,y2,z2]]
        E: ヤング率
        G: せん断弾性率
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        J: ねじり定数
        kappa_y: y方向せん断補正係数
        kappa_z: z方向せん断補正係数
        v_ref: 局所y軸の参照ベクトル（オプション）
        scf: スレンダネス補償係数（None=無効）

    Returns:
        Ke_global: (12, 12) 全体座標系の剛性行列
    """
    length, e_x = _beam3d_length_and_direction(coords)
    R = _build_local_axes(e_x, v_ref)
    Ke_local = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, length, kappa_y, kappa_z, scf=scf)
    T = _transformation_matrix_3d(R)
    return T.T @ Ke_local @ T


def timo_beam3d_distributed_load(
    coords: np.ndarray,
    qy_local: float = 0.0,
    qz_local: float = 0.0,
    qx_local: float = 0.0,
    v_ref: np.ndarray | None = None,
) -> np.ndarray:
    """等分布荷重の等価節点力ベクトル（全体座標系）を返す.

    局所座標系の各方向に一様分布荷重が作用する場合の等価節点力。

    Args:
        coords: (2, 3) 節点座標
        qy_local: 局所y方向の分布荷重強度
        qz_local: 局所z方向の分布荷重強度
        qx_local: 局所x方向の分布荷重強度（軸方向）
        v_ref: 局所y軸の参照ベクトル

    Returns:
        f_global: (12,) 全体座標系の等価節点力ベクトル
    """
    length, e_x = _beam3d_length_and_direction(coords)
    R = _build_local_axes(e_x, v_ref)

    f_local = np.zeros(12, dtype=float)

    # 軸方向（qx）
    if qx_local != 0.0:
        f_local[0] = qx_local * length / 2.0
        f_local[6] = qx_local * length / 2.0

    # y方向（qy）→ θz回転と連成
    if qy_local != 0.0:
        f_local[1] = qy_local * length / 2.0
        f_local[5] = qy_local * length**2 / 12.0
        f_local[7] = qy_local * length / 2.0
        f_local[11] = -qy_local * length**2 / 12.0

    # z方向（qz）→ θy回転と連成（符号反転に注意）
    if qz_local != 0.0:
        f_local[2] = qz_local * length / 2.0
        f_local[4] = -qz_local * length**2 / 12.0
        f_local[8] = qz_local * length / 2.0
        f_local[10] = qz_local * length**2 / 12.0

    T = _transformation_matrix_3d(R)
    return T.T @ f_local


def beam3d_section_forces(
    coords: np.ndarray,
    u_elem_global: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
    v_ref: np.ndarray | None = None,
    scf: float | None = None,
) -> tuple[BeamForces3D, BeamForces3D]:
    """要素両端の断面力を計算する（局所座標系）.

    全体座標系の変位ベクトルから局所座標系に変換し、
    局所剛性行列を用いて要素端力を求める。

    符号規約:
      - 節点1: 断面力 = -f_local[0:6] （正面の法線が+x方向なので反転）
      - 節点2: 断面力 = f_local[6:12]
      - N 正 = 引張、Vy/Vz 正 = y/z方向の正のせん断
      - Mz 正 = xy面内で凸下（sagging）、My/Mx 同様

    Args:
        coords: (2, 3) 節点座標
        u_elem_global: (12,) 要素変位ベクトル（全体座標系）
        E: ヤング率
        G: せん断弾性率
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        J: ねじり定数
        kappa_y: y方向せん断補正係数
        kappa_z: z方向せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        scf: スレンダネス補償係数

    Returns:
        (forces_1, forces_2): 両端の断面力
    """
    length, e_x = _beam3d_length_and_direction(coords)
    R = _build_local_axes(e_x, v_ref)
    T = _transformation_matrix_3d(R)
    Ke_local = timo_beam3d_ke_local(
        E,
        G,
        A,
        Iy,
        Iz,
        J,
        length,
        kappa_y,
        kappa_z,
        scf=scf,
    )

    u_local = T @ u_elem_global
    f_local = Ke_local @ u_local

    # 節点1: 断面力 = -f_local[0:6]
    forces_1 = BeamForces3D(
        N=-f_local[0],
        Vy=-f_local[1],
        Vz=-f_local[2],
        Mx=-f_local[3],
        My=-f_local[4],
        Mz=-f_local[5],
    )
    # 節点2: 断面力 = f_local[6:12]
    forces_2 = BeamForces3D(
        N=f_local[6],
        Vy=f_local[7],
        Vz=f_local[8],
        Mx=f_local[9],
        My=f_local[10],
        Mz=f_local[11],
    )
    return forces_1, forces_2


def beam3d_max_bending_stress(
    section_forces: BeamForces3D,
    A: float,
    Iy: float,
    Iz: float,
    y_max: float,
    z_max: float,
) -> float:
    """断面力から最大曲げ応力（絶対値）を推定する.

    σ_x = N/A ± Mz·y_max/Iz ± My·z_max/Iy

    断面の最外縁での応力の最大絶対値を返す。
    ねじりによるせん断応力は含まない。

    Args:
        section_forces: 断面力
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        y_max: 断面の y 方向最外縁距離（中立軸から）
        z_max: 断面の z 方向最外縁距離（中立軸から）

    Returns:
        σ_max: 最大曲げ応力の絶対値
    """
    sigma_axial = section_forces.N / A
    sigma_mz = abs(section_forces.Mz) * y_max / Iz
    sigma_my = abs(section_forces.My) * z_max / Iy
    return abs(sigma_axial) + sigma_mz + sigma_my


def beam3d_max_shear_stress(
    section_forces: BeamForces3D,
    A: float,
    J: float,
    r_max: float,
    shape: str = "circle",
) -> float:
    """断面力から最大せん断応力を推定する.

    ねじりせん断応力と横せん断応力の保守的な和を返す。

    ねじりせん断応力:
      τ_torsion = |Mx| · r_max / J （円形/パイプ断面の厳密解）

    横せん断応力（形状依存の中立軸上最大値）:
      - 円形断面: τ = 4V/(3A)
      - 矩形断面: τ = 3V/(2A)
      - 一般断面: τ = V/A（フォールバック）

    注意: ねじりと横せん断の最大応力点は一般に異なるため、
    保守的に和を取っている（実際の最大値はこれより小さい場合がある）。

    Args:
        section_forces: 断面力
        A: 断面積
        J: ねじり定数
        r_max: 断面の最外縁距離（ねじりせん断応力の計算用）
        shape: 断面形状 ("circle", "rectangle", "general")

    Returns:
        τ_max: 最大せん断応力の絶対値（保守的推定）
    """
    # ねじりせん断応力
    tau_torsion = abs(section_forces.Mx) * r_max / J

    # 横せん断応力（形状依存）
    if shape == "circle":
        tau_vy = 4.0 * abs(section_forces.Vy) / (3.0 * A)
        tau_vz = 4.0 * abs(section_forces.Vz) / (3.0 * A)
    elif shape == "rectangle":
        tau_vy = 3.0 * abs(section_forces.Vy) / (2.0 * A)
        tau_vz = 3.0 * abs(section_forces.Vz) / (2.0 * A)
    else:
        tau_vy = abs(section_forces.Vy) / A
        tau_vz = abs(section_forces.Vz) / A

    return tau_torsion + max(tau_vy, tau_vz)


class TimoshenkoBeam3D:
    """3D Timoshenko 梁要素（ElementProtocol適合）.

    Args:
        section: 梁断面特性（A, Iy, Iz, J, shape）
        kappa_y: y方向せん断補正係数。float or "cowper"。
        kappa_z: z方向せん断補正係数。float or "cowper"。
        v_ref: 局所y軸の参照ベクトル。None の場合は自動選択。
        scf: スレンダネス補償係数。None=無効。

    DOF配置:
        各節点: (ux, uy, uz, θx, θy, θz) → 6 DOF/node
        要素: 2 nodes → 12 DOF/element
    """

    ndof_per_node: int = 6
    nnodes: int = 2
    ndof: int = 12

    def __init__(
        self,
        section: BeamSection,
        kappa_y: float | str = 5.0 / 6.0,
        kappa_z: float | str = 5.0 / 6.0,
        v_ref: np.ndarray | None = None,
        scf: float | None = None,
    ) -> None:
        self.section = section
        self.v_ref = v_ref
        self.scf = scf

        # kappa_y の設定
        if isinstance(kappa_y, str):
            if kappa_y != "cowper":
                raise ValueError(f"kappa_y に指定できる文字列は 'cowper' のみです: '{kappa_y}'")
            self._kappa_y_mode = "cowper"
            self._kappa_y_value: float | None = None
        else:
            self._kappa_y_mode = "fixed"
            self._kappa_y_value = float(kappa_y)

        # kappa_z の設定
        if isinstance(kappa_z, str):
            if kappa_z != "cowper":
                raise ValueError(f"kappa_z に指定できる文字列は 'cowper' のみです: '{kappa_z}'")
            self._kappa_z_mode = "cowper"
            self._kappa_z_value: float | None = None
        else:
            self._kappa_z_mode = "fixed"
            self._kappa_z_value = float(kappa_z)

    def _resolve_kappa_y(self, nu: float) -> float:
        """材料のνからκyを解決する."""
        if self._kappa_y_mode == "cowper":
            return self.section.cowper_kappa_y(nu)
        assert self._kappa_y_value is not None
        return self._kappa_y_value

    def _resolve_kappa_z(self, nu: float) -> float:
        """材料のνからκzを解決する."""
        if self._kappa_z_mode == "cowper":
            return self.section.cowper_kappa_z(nu)
        assert self._kappa_z_value is not None
        return self._kappa_z_value

    def local_stiffness(
        self,
        coords: np.ndarray,
        material: ConstitutiveProtocol,
        thickness: float | None = None,
    ) -> np.ndarray:
        """全体座標系の剛性行列を返す.

        Args:
            coords: (2, 3) 節点座標
            material: 構成則（E, nu を保持）
            thickness: 未使用

        Returns:
            Ke: (12, 12) 全体座標系の剛性行列
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

        kappa_y_val = self._resolve_kappa_y(nu)
        kappa_z_val = self._resolve_kappa_z(nu)

        return timo_beam3d_ke_global(
            coords,
            young_e,
            shear_g,
            self.section.A,
            self.section.Iy,
            self.section.Iz,
            self.section.J,
            kappa_y_val,
            kappa_z_val,
            v_ref=self.v_ref,
            scf=self.scf,
        )

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """グローバル節点インデックスから要素DOFインデックスを返す.

        6 DOF/node: (ux, uy, uz, θx, θy, θz)

        Args:
            node_indices: (2,) グローバル節点インデックス

        Returns:
            edofs: (12,) グローバルDOFインデックス
        """
        edofs = np.empty(self.ndof, dtype=np.int64)
        for idx, n in enumerate(node_indices):
            for d in range(6):
                edofs[6 * idx + d] = 6 * n + d
        return edofs

    def section_forces(
        self,
        coords: np.ndarray,
        u_elem_global: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> tuple[BeamForces3D, BeamForces3D]:
        """要素両端の断面力を計算する.

        Args:
            coords: (2, 3) 節点座標
            u_elem_global: (12,) 要素変位ベクトル（全体座標系）
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

        kappa_y_val = self._resolve_kappa_y(nu)
        kappa_z_val = self._resolve_kappa_z(nu)

        return beam3d_section_forces(
            coords,
            u_elem_global,
            young_e,
            shear_g,
            self.section.A,
            self.section.Iy,
            self.section.Iz,
            self.section.J,
            kappa_y_val,
            kappa_z_val,
            v_ref=self.v_ref,
            scf=self.scf,
        )

    def distributed_load(
        self,
        coords: np.ndarray,
        qy_local: float = 0.0,
        qz_local: float = 0.0,
        qx_local: float = 0.0,
    ) -> np.ndarray:
        """等分布荷重の等価節点力ベクトル（全体座標系）."""
        return timo_beam3d_distributed_load(
            coords,
            qy_local,
            qz_local,
            qx_local,
            v_ref=self.v_ref,
        )
