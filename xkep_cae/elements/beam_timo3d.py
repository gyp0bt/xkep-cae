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


def timo_beam3d_mass_local(
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
    L: float,
) -> np.ndarray:
    """3D梁の局所整合質量行列 (12x12) を返す.

    DOF順: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Args:
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        Iy: y軸まわり断面二次モーメント [m⁴]
        Iz: z軸まわり断面二次モーメント [m⁴]
        L: 要素長 [m]

    Returns:
        Me: (12, 12) 局所整合質量行列
    """
    m = rho * A * L
    Me = np.zeros((12, 12), dtype=float)

    # 軸方向 (u1, u2) → DOF 0, 6
    Me[0, 0] = m / 3.0
    Me[0, 6] = m / 6.0
    Me[6, 0] = m / 6.0
    Me[6, 6] = m / 3.0

    # ねじり (θx1, θx2) → DOF 3, 9
    Ip = Iy + Iz  # 極慣性モーメント
    m_torsion = rho * Ip * L
    Me[3, 3] = m_torsion / 3.0
    Me[3, 9] = m_torsion / 6.0
    Me[9, 3] = m_torsion / 6.0
    Me[9, 9] = m_torsion / 3.0

    # xy面内曲げ (v1, θz1, v2, θz2) → DOF 1, 5, 7, 11
    coeff = m / 420.0
    M_xy = coeff * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ]
    )
    xy_idx = [1, 5, 7, 11]
    for i_loc, i_glob in enumerate(xy_idx):
        for j_loc, j_glob in enumerate(xy_idx):
            Me[i_glob, j_glob] = M_xy[i_loc, j_loc]

    # xz面内曲げ (w1, θy1, w2, θy2) → DOF 2, 4, 8, 10
    # dw/dx = -θy なので符号反転: signs = [+1, -1, +1, -1]
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    M_xz_base = coeff * np.array(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ]
    )
    M_xz = M_xz_base * np.outer(signs, signs)
    xz_idx = [2, 4, 8, 10]
    for i_loc, i_glob in enumerate(xz_idx):
        for j_loc, j_glob in enumerate(xz_idx):
            Me[i_glob, j_glob] = M_xz[i_loc, j_loc]

    return Me


def timo_beam3d_lumped_mass_local(
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
    L: float,
) -> np.ndarray:
    """3D梁の局所集中質量行列 (12x12, 対角) を返す.

    HRZ法による集中化。

    DOF順: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    HRZ結果:
        並進: m/2 （各節点・各方向）
        ねじり: ρ·Ip·L/2 （各節点）
        曲げ回転: m·L²/78 （各節点・各方向）

    Args:
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        Iy: y軸まわり断面二次モーメント [m⁴]
        Iz: z軸まわり断面二次モーメント [m⁴]
        L: 要素長 [m]

    Returns:
        Me: (12, 12) 対角集中質量行列
    """
    m = rho * A * L
    Ip = Iy + Iz
    m_torsion = rho * Ip * L
    rot_inertia = m * L**2 / 78.0

    diag = np.array(
        [
            m / 2.0,
            m / 2.0,
            m / 2.0,
            m_torsion / 2.0,
            rot_inertia,
            rot_inertia,
            m / 2.0,
            m / 2.0,
            m / 2.0,
            m_torsion / 2.0,
            rot_inertia,
            rot_inertia,
        ]
    )
    return np.diag(diag)


def timo_beam3d_mass_global(
    coords: np.ndarray,
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
    *,
    v_ref: np.ndarray | None = None,
    lumped: bool = False,
) -> np.ndarray:
    """全体座標系での3D梁の質量行列 (12x12) を返す.

    Args:
        coords: (2, 3) 節点座標 [[x1,y1,z1],[x2,y2,z2]]
        rho: 密度 [kg/m³]
        A: 断面積 [m²]
        Iy: y軸まわり断面二次モーメント [m⁴]
        Iz: z軸まわり断面二次モーメント [m⁴]
        v_ref: 局所y軸の参照ベクトル（オプション）
        lumped: True の場合は集中質量行列（HRZ法）

    Returns:
        Me_global: (12, 12) 全体座標系の質量行列
    """
    L, e_x = _beam3d_length_and_direction(coords)

    if lumped:
        # 集中質量は対角行列 → 座標変換不要（回転不変）
        return timo_beam3d_lumped_mass_local(rho, A, Iy, Iz, L)

    R = _build_local_axes(e_x, v_ref)
    Me_local = timo_beam3d_mass_local(rho, A, Iy, Iz, L)
    T = _transformation_matrix_3d(R)
    return T.T @ Me_local @ T


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

    element_type: str = "B31"
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

    def mass_matrix(
        self,
        coords: np.ndarray,
        rho: float,
        *,
        lumped: bool = False,
    ) -> np.ndarray:
        """全体座標系の質量行列を返す.

        Args:
            coords: (2, 3) 節点座標
            rho: 密度 [kg/m³]
            lumped: True の場合は集中質量行列（HRZ法）

        Returns:
            Me: (12, 12) 全体座標系の質量行列
        """
        return timo_beam3d_mass_global(
            coords,
            rho,
            self.section.A,
            self.section.Iy,
            self.section.Iz,
            v_ref=self.v_ref,
            lumped=lumped,
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


# =====================================================================
# Corotational (CR) 定式化 — Timoshenko 3D 梁の幾何学的非線形対応
# =====================================================================
#
# 線形 Timoshenko 3D 梁要素に Corotational 定式化を適用し、
# Updated Lagrangian 的な幾何学的非線形解析を実現する。
#
# アルゴリズム:
#   1. 変形後の節点座標から corotated フレームを構築
#   2. 節点回転を Rodrigues 公式で回転行列に変換
#   3. 剛体回転を除去して「自然変形」を抽出
#   4. 線形 Timoshenko 剛性で局所内力を計算
#   5. corotated 変換行列で全体座標系に変換
#
# 接線剛性は中心差分による数値微分（Cosserat rod 非線形と同じ手法）
# =====================================================================


def _rotvec_to_rotmat(theta: np.ndarray) -> np.ndarray:
    """回転ベクトル → 回転行列 (Rodrigues の公式).

    Args:
        theta: (3,) 回転ベクトル

    Returns:
        R: (3, 3) 回転行列
    """
    from xkep_cae.math.quaternion import quat_from_rotvec, quat_to_rotation_matrix

    q = quat_from_rotvec(theta)
    return quat_to_rotation_matrix(q)


def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """回転行列 → 回転ベクトル (対数写像).

    Args:
        R: (3, 3) 回転行列

    Returns:
        theta: (3,) 回転ベクトル
    """
    from xkep_cae.math.quaternion import quat_to_rotvec, rotation_matrix_to_quat

    q = rotation_matrix_to_quat(R)
    return quat_to_rotvec(q)


def _rodrigues_rotation(e_from: np.ndarray, e_to: np.ndarray) -> np.ndarray:
    """最小回転行列を計算: e_from → e_to に回す回転行列を返す.

    Rodrigues の公式を使い、e_from と e_to の間の最小角度回転を計算する。
    Gram-Schmidt と異なり、回転角に対して C∞ 滑らかな依存性を持つ。

    Args:
        e_from: (3,) 回転元の単位ベクトル
        e_to: (3,) 回転先の単位ベクトル

    Returns:
        R: (3, 3) 回転行列（e_from を e_to に回す）
    """
    v = np.cross(e_from, e_to)
    s = np.linalg.norm(v)  # sin(angle)
    c = np.dot(e_from, e_to)  # cos(angle)

    if s < 1e-14:
        # ほぼ同じ方向 or 反対方向
        if c > 0:
            return np.eye(3)
        else:
            # 180° 回転: e_from に直交する任意軸で回転
            abs_ef = np.abs(e_from)
            if abs_ef[0] <= abs_ef[1] and abs_ef[0] <= abs_ef[2]:
                perp = np.array([1.0, 0.0, 0.0])
            elif abs_ef[1] <= abs_ef[2]:
                perp = np.array([0.0, 1.0, 0.0])
            else:
                perp = np.array([0.0, 0.0, 1.0])
            axis = np.cross(e_from, perp)
            axis = axis / np.linalg.norm(axis)
            # R = 2 * axis ⊗ axis - I
            return 2.0 * np.outer(axis, axis) - np.eye(3)

    # Rodrigues 公式: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + vx + (vx @ vx) * (1.0 - c) / (s * s)


def timo_beam3d_cr_internal_force(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
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
    """Corotational 定式化による Timoshenko 3D 梁の非線形内力ベクトル.

    線形梁剛性を corotated フレームで使い、幾何学的非線形性を捕捉する。
    メッシュ細分化（要素あたりの回転が小さい）条件で正確。

    corotated フレーム R_cr の計算:
      Rodrigues 回転を使い、初期フレーム R_0 を変形後の弦方向に最小回転で追従させる。
      これにより Gram-Schmidt の不連続性を回避し、ヘリカル形状要素の
      大回転時にも滑らかな接線剛性を保証する。

    Args:
        coords_init: (2, 3) 初期節点座標
        u_elem: (12,) 全体座標系の変位ベクトル
        E, G: ヤング率、せん断弾性率
        A: 断面積
        Iy, Iz: 断面二次モーメント
        J: ねじり定数
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        scf: スレンダネス補償係数

    Returns:
        f_global: (12,) 全体座標系の内力ベクトル
    """
    # --- 初期形状 ---
    L_0, e_x_0 = _beam3d_length_and_direction(coords_init)
    R_0 = _build_local_axes(e_x_0, v_ref)

    # --- 変形後の形状 ---
    x1_def = coords_init[0] + u_elem[0:3]
    x2_def = coords_init[1] + u_elem[6:9]
    coords_def = np.array([x1_def, x2_def])
    L_def, e_x_def = _beam3d_length_and_direction(coords_def)

    # --- Rodrigues 回転による corotated フレーム ---
    # R_cr = R_rodrigues(e_x_0 → e_x_def) @ R_0
    # 初期弦方向から変形後弦方向への最小回転で初期フレームを追従させる
    # R_0 の行ベクトルが局所軸: R_0[i,:] を R_rod で回転させて R_cr を得る
    # R_cr[i,:] = R_rod @ R_0[i,:] → R_cr = R_0 @ R_rod^T
    R_rod = _rodrigues_rotation(e_x_0, e_x_def)
    R_cr = R_0 @ R_rod.T

    # --- 剛体回転を除去して自然変形回転を抽出 ---
    # R_def_i = R_cr @ R_node_i @ R_0^T
    #   R_cr: global→corotated (変形後ローカルフレーム)
    #   R_node_i: 節点iの全体回転 (global frame)
    #   R_0^T: initial local→global
    # 剛体回転時 (R_node = R_cr^T @ R_0) → R_def = I → θ_def = 0
    R_node1 = _rotvec_to_rotmat(u_elem[3:6])
    R_node2 = _rotvec_to_rotmat(u_elem[9:12])

    R_def1 = R_cr @ R_node1 @ R_0.T
    R_def2 = R_cr @ R_node2 @ R_0.T
    theta_def1 = _rotmat_to_rotvec(R_def1)
    theta_def2 = _rotmat_to_rotvec(R_def2)

    # --- corotated フレームでの自然変形 ---
    d_cr = np.zeros(12, dtype=float)
    # 節点1: 並進 = 0（corotated フレームの原点）
    d_cr[3:6] = theta_def1  # 節点1 回転
    d_cr[6] = L_def - L_0  # 軸伸び
    # 節点2: 横方向並進 = 0（corotated x軸上）
    d_cr[9:12] = theta_def2  # 節点2 回転

    # --- 線形局所剛性（初期長さベース）---
    Ke_local = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, L_0, kappa_y, kappa_z, scf=scf)

    # --- 局所内力 ---
    f_cr = Ke_local @ d_cr

    # --- 全体座標系へ変換 ---
    T_cr = _transformation_matrix_3d(R_cr)
    return T_cr.T @ f_cr


def timo_beam3d_cr_tangent(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
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
    """Corotational Timoshenko 3D 梁の接線剛性行列（数値微分）.

    中心差分で f_int の DOF 微分を計算する。
    Cosserat rod 非線形と同じ手法。

    Args:
        coords_init: (2, 3) 初期節点座標
        u_elem: (12,) 全体座標系の変位ベクトル
        E, G, A, Iy, Iz, J, kappa_y, kappa_z: 材料・断面定数
        v_ref: 局所y軸の参照ベクトル
        scf: スレンダネス補償係数

    Returns:
        K_T: (12, 12) 接線剛性行列（全体座標系、対称化済み）
    """
    eps = 1e-7
    K_T = np.zeros((12, 12), dtype=float)
    u_p = u_elem.copy()
    u_m = u_elem.copy()

    for j in range(12):
        u_p[j] = u_elem[j] + eps
        u_m[j] = u_elem[j] - eps
        f_p = timo_beam3d_cr_internal_force(
            coords_init,
            u_p,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            v_ref=v_ref,
            scf=scf,
        )
        f_m = timo_beam3d_cr_internal_force(
            coords_init,
            u_m,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            v_ref=v_ref,
            scf=scf,
        )
        K_T[:, j] = (f_p - f_m) / (2.0 * eps)
        u_p[j] = u_elem[j]
        u_m[j] = u_elem[j]

    # 対称化（数値誤差の補正）
    return 0.5 * (K_T + K_T.T)


def _skew(v: np.ndarray) -> np.ndarray:
    """3次元ベクトルの歪対称行列 [v]× を返す."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _tangent_operator(theta: np.ndarray) -> np.ndarray:
    """指数写像の接線演算子 T_s(θ) を計算する.

    δR = [T_s(θ) @ δθ]× @ R の関係を満たす。
    空間角速度 ω = T_s(θ) @ θ̇。

    T_s(θ) = I + ((1-cos|θ|)/|θ|²)[θ]× + (1 - sin|θ|/|θ|)/|θ|² [θ]×²

    Args:
        theta: (3,) 回転ベクトル

    Returns:
        T_s: (3, 3) 接線演算子
    """
    angle = float(np.linalg.norm(theta))
    if angle < 1e-10:
        # 小角度展開: T_s ≈ I + (1/2)[θ]× + (1/6)[θ]×²
        S = _skew(theta)
        return np.eye(3) + 0.5 * S + (1.0 / 6.0) * (S @ S)

    S = _skew(theta)
    S2 = S @ S
    c = np.cos(angle)
    s = np.sin(angle)
    coeff1 = (1.0 - c) / (angle * angle)
    coeff2 = (1.0 - s / angle) / (angle * angle)
    return np.eye(3) + coeff1 * S + coeff2 * S2


def _tangent_operator_inv(theta: np.ndarray) -> np.ndarray:
    """指数写像の接線演算子の逆 T_s^{-1}(θ) を計算する.

    T_s(θ) @ δθ_spatial = δR @ R^T のとき、
    T_s^{-1} は空間角速度から回転ベクトル微分への変換を行う。

    T_s^{-1}(θ) = I - (1/2)[θ]× + (1/|θ|² - (1+cos|θ|)/(2|θ|sin|θ|))[θ]×²

    Args:
        theta: (3,) 回転ベクトル

    Returns:
        T_inv: (3, 3) 接線演算子の逆行列
    """
    angle = float(np.linalg.norm(theta))
    if angle < 1e-10:
        # 小角度展開: T_s^{-1} ≈ I - (1/2)[θ]× + (1/12)[θ]×²
        S = _skew(theta)
        return np.eye(3) - 0.5 * S + (1.0 / 12.0) * (S @ S)

    S = _skew(theta)
    S2 = S @ S
    # 係数: 1/|θ|² - (1+cos|θ|)/(2|θ|sin|θ|)
    c = np.cos(angle)
    s = np.sin(angle)
    coeff = 1.0 / (angle * angle) - (1.0 + c) / (2.0 * angle * s)
    return np.eye(3) - 0.5 * S + coeff * S2


def timo_beam3d_cr_tangent_analytical(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
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
    """Corotational Timoshenko 3D 梁の解析的接線剛性行列.

    Battini & Pacoste (2002) に基づく解析的接線剛性:
      K_T = B^T @ K_l @ B + K_σ

    B: 全体DOF → 自然変形のプロジェクター行列 (12×12)
    K_l: corotated 局所剛性行列 (12×12)
    K_σ: 幾何剛性（内力と変形形状に依存）

    数値微分版 (timo_beam3d_cr_tangent) と比較して:
    - 24× 高速（24回の内力評価が不要）
    - 機械精度で正確（eps に依存しない）
    - 軸-曲げ結合の正確な捕捉（ヘリカル梁の二次収束改善）

    Args:
        coords_init: (2, 3) 初期節点座標
        u_elem: (12,) 全体座標系の変位ベクトル
        E, G, A, Iy, Iz, J, kappa_y, kappa_z: 材料・断面定数
        v_ref: 局所y軸の参照ベクトル
        scf: スレンダネス補償係数

    Returns:
        K_T: (12, 12) 接線剛性行列（全体座標系）
    """
    # === 1. CR kinematics（内力計算と共通） ===
    L_0, e_x_0 = _beam3d_length_and_direction(coords_init)
    R_0 = _build_local_axes(e_x_0, v_ref)

    x1_def = coords_init[0] + u_elem[0:3]
    x2_def = coords_init[1] + u_elem[6:9]
    coords_def = np.array([x1_def, x2_def])
    L_def, e_x_def = _beam3d_length_and_direction(coords_def)

    R_rod = _rodrigues_rotation(e_x_0, e_x_def)
    R_cr = R_0 @ R_rod.T

    R_node1 = _rotvec_to_rotmat(u_elem[3:6])
    R_node2 = _rotvec_to_rotmat(u_elem[9:12])

    R_def1 = R_cr @ R_node1 @ R_0.T
    R_def2 = R_cr @ R_node2 @ R_0.T
    theta_def1 = _rotmat_to_rotvec(R_def1)
    theta_def2 = _rotmat_to_rotvec(R_def2)

    d_cr = np.zeros(12, dtype=float)
    d_cr[3:6] = theta_def1
    d_cr[6] = L_def - L_0
    d_cr[9:12] = theta_def2

    # === 2. 局所剛性と局所内力 ===
    Ke_local = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, L_0, kappa_y, kappa_z, scf=scf)
    f_cr = Ke_local @ d_cr  # corotated 内力

    # === 3. プロジェクター行列 B = ∂d_cr/∂u ===
    # B は 12×12 行列。非ゼロの行: 3,4,5 (θ_def1), 6 (軸), 9,10,11 (θ_def2)
    B = np.zeros((12, 12), dtype=float)

    # --- 軸方向: d_cr[6] = L_def - L_0 ---
    # ∂L_def/∂u1_i = -e_x_def[i], ∂L_def/∂u2_i = +e_x_def[i]
    B[6, 0:3] = -e_x_def
    B[6, 6:9] = e_x_def
    # 回転DOFには依存しない: B[6, 3:6] = B[6, 9:12] = 0

    # --- 回転自然変形: θ_def_i = log(R_cr @ R_node_i @ R_0^T) ---
    #
    # R_def_i = R_cr @ R_node_i @ R_0^T
    # δR_def_i = [δψ_def_i]× @ R_def_i  (left spin in corotated frame)
    #
    # δψ_def_i = δψ_cr_local + R_cr @ T_s(θ_node_i) @ δθ_node_i
    #   δψ_cr_local: CR フレームの回転スピン（corotated フレーム内）
    #   T_s: 指数写像の接線演算子（global → global 角速度変換）
    #   R_cr: global → local 変換
    #
    # CR frame spin (corotated/body frame):
    #   R_rod maps e_x_0 → e_x_def, R_cr = R_0 @ R_rod^T
    #   δψ_cr_local = -R_cr @ δψ_rod
    #   δψ_rod = (1/L_def) * [e_x_def]× @ δΔx  (global frame)
    #   δΔx = δu2 - δu1
    #
    # よって:
    #   ∂ψ_cr_local/∂u1 = +(1/L_def) * R_cr @ [e_x_def]×
    #   ∂ψ_cr_local/∂u2 = -(1/L_def) * R_cr @ [e_x_def]×
    #
    # Natural rotation variation:
    #   δθ_def_i = T_s^{-1}(θ_def_i) @ δψ_def_i  (corotated frame)

    S_ex = _skew(e_x_def)  # [e_x_def]× (3×3, global frame)
    R_cr_S_ex = R_cr @ S_ex  # R_cr @ [e_x_def]× (3×3)
    dpsi_du1 = (1.0 / L_def) * R_cr_S_ex  # ∂ψ_cr_local/∂u1 (3×3)
    dpsi_du2 = -(1.0 / L_def) * R_cr_S_ex  # ∂ψ_cr_local/∂u2 (3×3)

    # 接線演算子の逆（corotated frame での log map tangent inverse）
    T_inv1 = _tangent_operator_inv(theta_def1)
    T_inv2 = _tangent_operator_inv(theta_def2)

    # 接線演算子 T_s（指数写像の接線、global frame）
    T_s1 = _tangent_operator(u_elem[3:6])
    T_s2 = _tangent_operator(u_elem[9:12])

    # --- 節点1 回転の B行列 ---
    # ∂θ_def1/∂u1 = T_inv1 @ dpsi_du1
    # ∂θ_def1/∂u2 = T_inv1 @ dpsi_du2
    # ∂θ_def1/∂θ_node1 = T_inv1 @ R_cr @ T_s(θ_node1)
    B[3:6, 0:3] = T_inv1 @ dpsi_du1
    B[3:6, 6:9] = T_inv1 @ dpsi_du2
    B[3:6, 3:6] = T_inv1 @ R_cr @ T_s1

    # --- 節点2 回転の B行列 ---
    B[9:12, 0:3] = T_inv2 @ dpsi_du1
    B[9:12, 6:9] = T_inv2 @ dpsi_du2
    B[9:12, 9:12] = T_inv2 @ R_cr @ T_s2

    # === 4. 変換行列 ===
    T_cr = _transformation_matrix_3d(R_cr)

    # === 5. 材料接線剛性 ===
    # K_mat = T_cr^T @ K_l @ B
    K_mat = T_cr.T @ Ke_local @ B

    # === 6. 幾何剛性 K_σ = ∂(T_cr^T)/∂u @ f_cr ===
    #
    # T_cr は R_cr のブロック対角。δR_cr = [δψ_local]× @ R_cr (body spin) に対し:
    #   δ(R_cr^T) = -R_cr^T @ [δψ_local]×^T = R_cr^T @ [δψ_local]×
    #
    # T_cr^T @ f_cr の変動（T_cr の変動分のみ）:
    #   δ(T_cr^T) @ f_cr の各ブロック:
    #     δ(R_cr^T) @ f_blk = R_cr^T @ [δψ_local]× @ f_blk
    #                       = R_cr^T @ (δψ_local × f_blk)
    #                       = -R_cr^T @ [f_blk]× @ δψ_local
    #
    # δψ_local = dpsi_du1 @ δu1 + dpsi_du2 @ δu2  (corotated frame)
    #
    # K_geo[3*blk:3*blk+3, :] = R_cr^T @ [f_blk]× @ dpsi_du

    K_geo = np.zeros((12, 12), dtype=float)

    for blk in range(4):
        f_blk = f_cr[3 * blk : 3 * blk + 3]
        if np.linalg.norm(f_blk) < 1e-30:
            continue
        # R_cr^T @ [f_blk]× (3×3)
        RtSf = R_cr.T @ _skew(f_blk)  # (3, 3)
        K_geo[3 * blk : 3 * blk + 3, 0:3] += RtSf @ dpsi_du1
        K_geo[3 * blk : 3 * blk + 3, 6:9] += RtSf @ dpsi_du2

    # === 7. 合成して対称化 ===
    K_T = K_mat + K_geo
    # 理論的に対称だが、数値的な微小な非対称性を補正
    return 0.5 * (K_T + K_T.T)


def assemble_cr_beam3d(
    nodes_init: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
    *,
    v_ref: np.ndarray | None = None,
    scf: float | None = None,
    stiffness: bool = True,
    internal_force: bool = True,
    sparse: bool = True,
    analytical_tangent: bool = True,
) -> tuple:
    """Corotational Timoshenko 3D 梁のグローバルアセンブリ（COO/CSR高速版）.

    edofs事前一括計算 + COO蓄積 → CSR変換で高速化。
    密行列の np.ix_ アクセスを排除し、メモリを O(nnz) に削減。

    Args:
        nodes_init: (n_nodes, 3) 初期節点座標
        connectivity: (n_elems, 2) 要素接続（節点インデックス）
        u: (ndof,) 全体変位ベクトル
        E, G: ヤング率、せん断弾性率
        A, Iy, Iz, J: 断面定数
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        scf: スレンダネス補償係数
        stiffness: True の場合、接線剛性行列を計算
        internal_force: True の場合、内力ベクトルを計算
        sparse: True → CSR行列で返す（デフォルト）。False → 密ndarray。
        analytical_tangent: True の場合、解析的接線剛性を使用（デフォルト）。
            False の場合、数値微分による接線剛性を使用。

    Returns:
        (K_T, f_int): 接線剛性行列と内力ベクトル (ndof,)
            sparse=True: K_T は scipy.sparse.csr_matrix
            sparse=False: K_T は (ndof, ndof) ndarray
            要求されなかった場合は None
    """
    n_nodes = len(nodes_init)
    n_elems = len(connectivity)
    ndof = 6 * n_nodes
    m = 12  # DOF per element (2 nodes × 6 DOF)

    f_int_global = np.zeros(ndof, dtype=float) if internal_force else None

    # --- edofs 事前一括計算 ---
    conn_int = connectivity.astype(np.int64)
    dof_offsets = np.arange(6, dtype=np.int64)
    all_edofs = (conn_int[:, :, None] * 6 + dof_offsets[None, None, :]).reshape(n_elems, m)

    # --- sparse=False: 密行列に直接書き込み（小規模問題向け） ---
    if stiffness and not sparse:
        K_T_dense = np.zeros((ndof, ndof), dtype=float)
        for i in range(n_elems):
            n1, n2 = int(conn_int[i, 0]), int(conn_int[i, 1])
            coords = nodes_init[np.array([n1, n2])]
            edofs = all_edofs[i]
            u_elem = u[edofs]
            if internal_force:
                f_e = timo_beam3d_cr_internal_force(
                    coords,
                    u_elem,
                    E,
                    G,
                    A,
                    Iy,
                    Iz,
                    J,
                    kappa_y,
                    kappa_z,
                    v_ref=v_ref,
                    scf=scf,
                )
                f_int_global[edofs] += f_e
            _tangent_fn = (
                timo_beam3d_cr_tangent_analytical if analytical_tangent else timo_beam3d_cr_tangent
            )
            K_e = _tangent_fn(
                coords,
                u_elem,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                v_ref=v_ref,
                scf=scf,
            )
            K_T_dense[np.ix_(edofs, edofs)] += K_e
        return K_T_dense, f_int_global

    # --- ベクトル化パス（解析的接線 + sparse） ---
    if analytical_tangent and n_elems > 0:
        batch_coo, batch_fint = _assemble_cr_beam3d_batch(
            nodes_init,
            connectivity,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            v_ref=v_ref,
            scf=scf,
            stiffness=stiffness,
            internal_force=internal_force,
            analytical_tangent=True,
        )

        if internal_force and batch_fint is not None:
            f_int_global = batch_fint

        if stiffness and batch_coo is not None:
            import scipy.sparse as sp

            block_nnz = m * m  # 144
            coo_rows = np.repeat(all_edofs, m, axis=1).ravel()
            coo_cols = np.tile(all_edofs, (1, m)).ravel()
            K_T_csr = sp.csr_matrix((batch_coo, (coo_rows, coo_cols)), shape=(ndof, ndof))
            K_T_csr.sum_duplicates()
            return K_T_csr, f_int_global

        return None, f_int_global

    # --- フォールバック: 数値微分用の逐次ループ ---
    if stiffness:
        import scipy.sparse as sp

        block_nnz = m * m  # 144
        total_nnz = n_elems * block_nnz
        coo_rows = np.repeat(all_edofs, m, axis=1).ravel()
        coo_cols = np.tile(all_edofs, (1, m)).ravel()
        coo_data = np.empty(total_nnz, dtype=float)

    for i in range(n_elems):
        n1, n2 = int(conn_int[i, 0]), int(conn_int[i, 1])
        coords = nodes_init[np.array([n1, n2])]
        edofs = all_edofs[i]
        u_elem = u[edofs]
        if internal_force:
            f_e = timo_beam3d_cr_internal_force(
                coords,
                u_elem,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                v_ref=v_ref,
                scf=scf,
            )
            f_int_global[edofs] += f_e
        if stiffness:
            K_e = timo_beam3d_cr_tangent(
                coords,
                u_elem,
                E,
                G,
                A,
                Iy,
                Iz,
                J,
                kappa_y,
                kappa_z,
                v_ref=v_ref,
                scf=scf,
            )
            offset = i * block_nnz
            coo_data[offset : offset + block_nnz] = K_e.ravel()

    if stiffness:
        K_T_csr = sp.csr_matrix((coo_data, (coo_rows, coo_cols)), shape=(ndof, ndof))
        K_T_csr.sum_duplicates()
        return K_T_csr, f_int_global

    return None, f_int_global


# ---------------------------------------------------------------------------
# Vectorized (batch) CR beam assembly — 要素ループのベクトル化
# ---------------------------------------------------------------------------


def _batch_skew(v: np.ndarray) -> np.ndarray:
    """バッチ歪対称行列 [v]×.

    Args:
        v: (n, 3) ベクトル配列

    Returns:
        S: (n, 3, 3) 歪対称行列配列
    """
    n = v.shape[0]
    S = np.zeros((n, 3, 3), dtype=float)
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


def _batch_rotvec_to_rotmat(thetas: np.ndarray) -> np.ndarray:
    """バッチ回転ベクトル→回転行列（四元数経由）.

    Args:
        thetas: (n, 3) 回転ベクトル配列

    Returns:
        Rs: (n, 3, 3) 回転行列配列
    """
    n = thetas.shape[0]
    angles = np.linalg.norm(thetas, axis=1)  # (n,)

    # 四元数生成（指数写像）
    small = angles < 1e-12
    half = angles / 2.0
    half_sq = half * half

    # sinc = sin(half)/angle, cos_half = cos(half)
    sinc = np.empty(n, dtype=float)
    cos_half = np.empty(n, dtype=float)
    sinc[small] = 0.5 - half_sq[small] / 48.0
    cos_half[small] = 1.0 - half_sq[small] / 2.0
    big = ~small
    sinc[big] = np.sin(half[big]) / angles[big]
    cos_half[big] = np.cos(half[big])

    # q = [w, x, y, z]
    w = cos_half  # (n,)
    xyz = sinc[:, None] * thetas  # (n, 3)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 回転行列への変換
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    Rs = np.empty((n, 3, 3), dtype=float)
    Rs[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    Rs[:, 0, 1] = 2.0 * (xy - wz)
    Rs[:, 0, 2] = 2.0 * (xz + wy)
    Rs[:, 1, 0] = 2.0 * (xy + wz)
    Rs[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    Rs[:, 1, 2] = 2.0 * (yz - wx)
    Rs[:, 2, 0] = 2.0 * (xz - wy)
    Rs[:, 2, 1] = 2.0 * (yz + wx)
    Rs[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return Rs


def _batch_rotmat_to_rotvec(Rs: np.ndarray) -> np.ndarray:
    """バッチ回転行列→回転ベクトル（四元数経由、Shepperd法ベクトル化）.

    Args:
        Rs: (n, 3, 3) 回転行列配列

    Returns:
        rotvecs: (n, 3) 回転ベクトル配列
    """
    n = Rs.shape[0]
    trace = Rs[:, 0, 0] + Rs[:, 1, 1] + Rs[:, 2, 2]  # (n,)

    # Shepperd法: 4ケースの最大対角成分を選択
    # case0: trace > max(diag) → w が最大
    # case1: R[0,0] が最大
    # case2: R[1,1] が最大
    # case3: R[2,2] が最大
    diag_vals = np.stack([trace, Rs[:, 0, 0], Rs[:, 1, 1], Rs[:, 2, 2]], axis=1)  # (n, 4)
    best = np.argmax(diag_vals, axis=1)  # (n,)

    q = np.empty((n, 4), dtype=float)

    # Case 0: trace > others
    m0 = best == 0
    if np.any(m0):
        s = 2.0 * np.sqrt(trace[m0] + 1.0)
        q[m0, 0] = 0.25 * s
        q[m0, 1] = (Rs[m0, 2, 1] - Rs[m0, 1, 2]) / s
        q[m0, 2] = (Rs[m0, 0, 2] - Rs[m0, 2, 0]) / s
        q[m0, 3] = (Rs[m0, 1, 0] - Rs[m0, 0, 1]) / s

    # Case 1: R[0,0] が最大
    m1 = best == 1
    if np.any(m1):
        s = 2.0 * np.sqrt(1.0 + Rs[m1, 0, 0] - Rs[m1, 1, 1] - Rs[m1, 2, 2])
        q[m1, 0] = (Rs[m1, 2, 1] - Rs[m1, 1, 2]) / s
        q[m1, 1] = 0.25 * s
        q[m1, 2] = (Rs[m1, 0, 1] + Rs[m1, 1, 0]) / s
        q[m1, 3] = (Rs[m1, 0, 2] + Rs[m1, 2, 0]) / s

    # Case 2: R[1,1] が最大
    m2 = best == 2
    if np.any(m2):
        s = 2.0 * np.sqrt(1.0 + Rs[m2, 1, 1] - Rs[m2, 0, 0] - Rs[m2, 2, 2])
        q[m2, 0] = (Rs[m2, 0, 2] - Rs[m2, 2, 0]) / s
        q[m2, 1] = (Rs[m2, 0, 1] + Rs[m2, 1, 0]) / s
        q[m2, 2] = 0.25 * s
        q[m2, 3] = (Rs[m2, 1, 2] + Rs[m2, 2, 1]) / s

    # Case 3: R[2,2] が最大
    m3 = best == 3
    if np.any(m3):
        s = 2.0 * np.sqrt(1.0 + Rs[m3, 2, 2] - Rs[m3, 0, 0] - Rs[m3, 1, 1])
        q[m3, 0] = (Rs[m3, 1, 0] - Rs[m3, 0, 1]) / s
        q[m3, 1] = (Rs[m3, 0, 2] + Rs[m3, 2, 0]) / s
        q[m3, 2] = (Rs[m3, 1, 2] + Rs[m3, 2, 1]) / s
        q[m3, 3] = 0.25 * s

    # w >= 0 に正規化
    neg_w = q[:, 0] < 0
    q[neg_w] = -q[neg_w]

    # 正規化
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    q = q / norms

    # 対数写像: q → rotvec
    vec = q[:, 1:4]  # (n, 3)
    sin_half = np.linalg.norm(vec, axis=1)  # (n,)
    small = sin_half < 1e-12
    big = ~small

    coeff = np.empty(n, dtype=float)
    coeff[small] = 2.0
    coeff[big] = 2.0 * np.arctan2(sin_half[big], q[big, 0]) / sin_half[big]

    return coeff[:, None] * vec


def _batch_rodrigues_rotation(e_from: np.ndarray, e_to: np.ndarray) -> np.ndarray:
    """バッチ最小回転行列（Rodrigues公式）.

    Args:
        e_from: (n, 3) 回転元の単位ベクトル
        e_to: (n, 3) 回転先の単位ベクトル

    Returns:
        Rs: (n, 3, 3) 回転行列配列
    """
    v = np.cross(e_from, e_to)  # (n, 3)
    s = np.linalg.norm(v, axis=1)  # (n,) sin(angle)
    c = np.sum(e_from * e_to, axis=1)  # (n,) cos(angle)

    small = s < 1e-14

    # 通常ケース: Rodrigues公式
    vx = _batch_skew(v)  # (n, 3, 3)
    vx2 = np.einsum("nij,njk->nik", vx, vx)  # (n, 3, 3)

    eye3 = np.eye(3, dtype=float)
    # (1 - c) / s^2 の係数、s=0 のときは 0 にする
    denom = s * s
    denom[small] = 1.0  # ゼロ除算回避
    coeff = (1.0 - c) / denom  # (n,)
    coeff[small] = 0.0

    Rs = eye3[None, :, :] + vx + coeff[:, None, None] * vx2

    # 同方向（s≈0, c>0）→ 恒等行列
    same_dir = small & (c > 0)
    Rs[same_dir] = eye3

    # 反対方向（s≈0, c<0）→ 180°回転
    opposite = small & (c <= 0)
    if np.any(opposite):
        opp_idx = np.where(opposite)[0]
        for idx in opp_idx:
            ef = e_from[idx]
            abs_ef = np.abs(ef)
            if abs_ef[0] <= abs_ef[1] and abs_ef[0] <= abs_ef[2]:
                perp = np.array([1.0, 0.0, 0.0])
            elif abs_ef[1] <= abs_ef[2]:
                perp = np.array([0.0, 1.0, 0.0])
            else:
                perp = np.array([0.0, 0.0, 1.0])
            axis = np.cross(ef, perp)
            axis = axis / np.linalg.norm(axis)
            Rs[idx] = 2.0 * np.outer(axis, axis) - eye3

    return Rs


def _batch_build_local_axes(e_x: np.ndarray, v_ref: np.ndarray | None = None) -> np.ndarray:
    """バッチ局所座標系構築.

    Args:
        e_x: (n, 3) 梁軸方向の単位ベクトル配列
        v_ref: (3,) 参照ベクトル（全要素共通）。None の場合は自動選択。

    Returns:
        Rs: (n, 3, 3) 回転行列配列（行ベクトルが局所軸）
    """
    n = e_x.shape[0]

    if v_ref is not None:
        # 全要素で同じ参照ベクトル
        vr = np.tile(v_ref, (n, 1))  # (n, 3)
    else:
        # 要素ごとに最も直交する座標軸を選択
        abs_ex = np.abs(e_x)  # (n, 3)
        vr = np.zeros((n, 3), dtype=float)
        min_idx = np.argmin(abs_ex, axis=1)  # (n,)
        vr[np.arange(n), min_idx] = 1.0

    # Gram-Schmidt: e_z = e_x × v_ref, e_y = e_z × e_x
    e_z = np.cross(e_x, vr)  # (n, 3)
    norm_ez = np.linalg.norm(e_z, axis=1, keepdims=True)  # (n, 1)
    norm_ez = np.maximum(norm_ez, 1e-10)
    e_z = e_z / norm_ez
    e_y = np.cross(e_z, e_x)  # (n, 3)

    Rs = np.empty((n, 3, 3), dtype=float)
    Rs[:, 0, :] = e_x
    Rs[:, 1, :] = e_y
    Rs[:, 2, :] = e_z
    return Rs


def _batch_tangent_operator(thetas: np.ndarray) -> np.ndarray:
    """バッチ接線演算子 T_s(θ).

    Args:
        thetas: (n, 3) 回転ベクトル配列

    Returns:
        Ts: (n, 3, 3) 接線演算子配列
    """
    n = thetas.shape[0]
    angles = np.linalg.norm(thetas, axis=1)  # (n,)
    S = _batch_skew(thetas)  # (n, 3, 3)
    S2 = np.einsum("nij,njk->nik", S, S)  # (n, 3, 3)

    small = angles < 1e-10
    big = ~small

    eye3 = np.eye(3, dtype=float)
    Ts = np.tile(eye3, (n, 1, 1))  # (n, 3, 3)

    if np.any(small):
        Ts[small] += 0.5 * S[small] + (1.0 / 6.0) * S2[small]

    if np.any(big):
        a = angles[big]
        c = np.cos(a)
        s = np.sin(a)
        a2 = a * a
        coeff1 = (1.0 - c) / a2
        coeff2 = (1.0 - s / a) / a2
        Ts[big] += coeff1[:, None, None] * S[big] + coeff2[:, None, None] * S2[big]

    return Ts


def _batch_tangent_operator_inv(thetas: np.ndarray) -> np.ndarray:
    """バッチ接線演算子の逆 T_s^{-1}(θ).

    Args:
        thetas: (n, 3) 回転ベクトル配列

    Returns:
        T_inv: (n, 3, 3) 接線演算子の逆配列
    """
    n = thetas.shape[0]
    angles = np.linalg.norm(thetas, axis=1)  # (n,)
    S = _batch_skew(thetas)  # (n, 3, 3)
    S2 = np.einsum("nij,njk->nik", S, S)  # (n, 3, 3)

    small = angles < 1e-10
    big = ~small

    eye3 = np.eye(3, dtype=float)
    T_inv = np.tile(eye3, (n, 1, 1))  # (n, 3, 3)

    if np.any(small):
        T_inv[small] += -0.5 * S[small] + (1.0 / 12.0) * S2[small]

    if np.any(big):
        a = angles[big]
        c = np.cos(a)
        s = np.sin(a)
        a2 = a * a
        coeff = 1.0 / a2 - (1.0 + c) / (2.0 * a * s)
        T_inv[big] += -0.5 * S[big] + coeff[:, None, None] * S2[big]

    return T_inv


def _batch_timo_ke_local(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    L: np.ndarray,
    kappa_y: float,
    kappa_z: float,
    scf: float | None = None,
) -> np.ndarray:
    """バッチ局所剛性行列計算（L のみ要素依存）.

    Args:
        E, G, A, Iy, Iz, J, kappa_y, kappa_z: 材料・断面定数（全要素共通）
        L: (n,) 要素長さ配列
        scf: スレンダネス補償係数

    Returns:
        Ke: (n, 12, 12) 局所剛性行列配列
    """
    n = L.shape[0]

    Phi_y = 12.0 * E * Iy / (kappa_y * G * A * L * L)  # (n,)
    Phi_z = 12.0 * E * Iz / (kappa_z * G * A * L * L)  # (n,)

    if scf is not None and scf > 0.0:
        slenderness_y = L * L * A / (12.0 * Iy)
        f_p_y = 1.0 / (1.0 + scf * slenderness_y)
        Phi_y = Phi_y * f_p_y
        slenderness_z = L * L * A / (12.0 * Iz)
        f_p_z = 1.0 / (1.0 + scf * slenderness_z)
        Phi_z = Phi_z * f_p_z

    denom_y = 1.0 + Phi_y  # (n,)
    denom_z = 1.0 + Phi_z  # (n,)

    Ke = np.zeros((n, 12, 12), dtype=float)

    EA_L = E * A / L  # (n,)
    Ke[:, 0, 0] = EA_L
    Ke[:, 0, 6] = -EA_L
    Ke[:, 6, 0] = -EA_L
    Ke[:, 6, 6] = EA_L

    GJ_L = G * J / L
    Ke[:, 3, 3] = GJ_L
    Ke[:, 3, 9] = -GJ_L
    Ke[:, 9, 3] = -GJ_L
    Ke[:, 9, 9] = GJ_L

    # xy面曲げ (EIz)
    L2 = L * L
    L3 = L2 * L
    EIz_L3 = E * Iz / L3
    EIz_L2 = E * Iz / L2
    EIz_L = E * Iz / L

    Ke[:, 1, 1] = 12.0 * EIz_L3 / denom_z
    Ke[:, 1, 5] = 6.0 * EIz_L2 / denom_z
    Ke[:, 1, 7] = -12.0 * EIz_L3 / denom_z
    Ke[:, 1, 11] = 6.0 * EIz_L2 / denom_z
    Ke[:, 5, 1] = 6.0 * EIz_L2 / denom_z
    Ke[:, 5, 5] = (4.0 + Phi_z) * EIz_L / denom_z
    Ke[:, 5, 7] = -6.0 * EIz_L2 / denom_z
    Ke[:, 5, 11] = (2.0 - Phi_z) * EIz_L / denom_z
    Ke[:, 7, 1] = -12.0 * EIz_L3 / denom_z
    Ke[:, 7, 5] = -6.0 * EIz_L2 / denom_z
    Ke[:, 7, 7] = 12.0 * EIz_L3 / denom_z
    Ke[:, 7, 11] = -6.0 * EIz_L2 / denom_z
    Ke[:, 11, 1] = 6.0 * EIz_L2 / denom_z
    Ke[:, 11, 5] = (2.0 - Phi_z) * EIz_L / denom_z
    Ke[:, 11, 7] = -6.0 * EIz_L2 / denom_z
    Ke[:, 11, 11] = (4.0 + Phi_z) * EIz_L / denom_z

    # xz面曲げ (EIy)
    EIy_L3 = E * Iy / L3
    EIy_L2 = E * Iy / L2
    EIy_L = E * Iy / L

    Ke[:, 2, 2] = 12.0 * EIy_L3 / denom_y
    Ke[:, 2, 4] = -6.0 * EIy_L2 / denom_y
    Ke[:, 2, 8] = -12.0 * EIy_L3 / denom_y
    Ke[:, 2, 10] = -6.0 * EIy_L2 / denom_y
    Ke[:, 4, 2] = -6.0 * EIy_L2 / denom_y
    Ke[:, 4, 4] = (4.0 + Phi_y) * EIy_L / denom_y
    Ke[:, 4, 8] = 6.0 * EIy_L2 / denom_y
    Ke[:, 4, 10] = (2.0 - Phi_y) * EIy_L / denom_y
    Ke[:, 8, 2] = -12.0 * EIy_L3 / denom_y
    Ke[:, 8, 4] = 6.0 * EIy_L2 / denom_y
    Ke[:, 8, 8] = 12.0 * EIy_L3 / denom_y
    Ke[:, 8, 10] = 6.0 * EIy_L2 / denom_y
    Ke[:, 10, 2] = -6.0 * EIy_L2 / denom_y
    Ke[:, 10, 4] = (2.0 - Phi_y) * EIy_L / denom_y
    Ke[:, 10, 8] = 6.0 * EIy_L2 / denom_y
    Ke[:, 10, 10] = (4.0 + Phi_y) * EIy_L / denom_y

    return Ke


def _assemble_cr_beam3d_batch(
    nodes_init: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
    *,
    v_ref: np.ndarray | None = None,
    scf: float | None = None,
    stiffness: bool = True,
    internal_force: bool = True,
    analytical_tangent: bool = True,
) -> tuple:
    """ベクトル化された CR Timoshenko 3D 梁アセンブリ.

    全要素をバッチ処理してPythonループを排除。
    解析的接線剛性 (analytical_tangent=True) 専用。

    Returns:
        (coo_data, f_int_global) — coo_data は (n_elems * 144,) の COO値配列。
        呼び出し元で CSR 変換する。
    """
    conn_int = connectivity.astype(np.int64)
    n_elems = len(conn_int)
    n_nodes = len(nodes_init)
    ndof = 6 * n_nodes

    # --- 全要素の座標・変位を一括抽出 ---
    coords_all = nodes_init[conn_int]  # (n_elems, 2, 3)
    m = 12
    dof_offsets = np.arange(6, dtype=np.int64)
    all_edofs = (conn_int[:, :, None] * 6 + dof_offsets[None, None, :]).reshape(n_elems, m)
    u_all = u[all_edofs]  # (n_elems, 12)

    # --- Step 1: 初期形状の長さ・方向 ---
    dx0 = coords_all[:, 1] - coords_all[:, 0]  # (n, 3)
    L_0 = np.linalg.norm(dx0, axis=1)  # (n,)
    e_x_0 = dx0 / L_0[:, None]  # (n, 3)

    # --- Step 2: 初期ローカルフレーム R_0 ---
    R_0 = _batch_build_local_axes(e_x_0, v_ref)  # (n, 3, 3)

    # --- Step 3: 変形後の形状 ---
    x1_def = coords_all[:, 0] + u_all[:, 0:3]  # (n, 3)
    x2_def = coords_all[:, 1] + u_all[:, 6:9]  # (n, 3)
    dx_def = x2_def - x1_def  # (n, 3)
    L_def = np.linalg.norm(dx_def, axis=1)  # (n,)
    e_x_def = dx_def / L_def[:, None]  # (n, 3)

    # --- Step 4: Rodrigues 回転 → corotated フレーム ---
    R_rod = _batch_rodrigues_rotation(e_x_0, e_x_def)  # (n, 3, 3)
    # R_cr = R_0 @ R_rod^T
    R_cr = np.einsum("nij,nkj->nik", R_0, R_rod)  # (n, 3, 3)

    # --- Step 5: 節点回転行列 ---
    R_node1 = _batch_rotvec_to_rotmat(u_all[:, 3:6])  # (n, 3, 3)
    R_node2 = _batch_rotvec_to_rotmat(u_all[:, 9:12])  # (n, 3, 3)

    # --- Step 6: 自然変形回転 ---
    # R_def = R_cr @ R_node @ R_0^T
    R_0_T = np.einsum("nji->nij", R_0)  # (n, 3, 3) transpose
    R_def1 = np.einsum("nij,njk,nkl->nil", R_cr, R_node1, R_0_T)  # (n, 3, 3)
    R_def2 = np.einsum("nij,njk,nkl->nil", R_cr, R_node2, R_0_T)  # (n, 3, 3)
    theta_def1 = _batch_rotmat_to_rotvec(R_def1)  # (n, 3)
    theta_def2 = _batch_rotmat_to_rotvec(R_def2)  # (n, 3)

    # --- Step 7: d_cr 構築 ---
    d_cr = np.zeros((n_elems, 12), dtype=float)
    d_cr[:, 3:6] = theta_def1
    d_cr[:, 6] = L_def - L_0
    d_cr[:, 9:12] = theta_def2

    # --- Step 8: バッチ局所剛性 ---
    Ke_local = _batch_timo_ke_local(
        E, G, A, Iy, Iz, J, L_0, kappa_y, kappa_z, scf=scf
    )  # (n, 12, 12)

    # --- Step 9: 局所内力 f_cr = Ke_local @ d_cr ---
    f_cr = np.einsum("nij,nj->ni", Ke_local, d_cr)  # (n, 12)

    # --- Step 10: 変換行列 T_cr (12x12 ブロック対角) と f_global ---
    R_cr_T = np.einsum("nji->nij", R_cr)  # (n, 3, 3)
    f_int_global = None
    if internal_force:
        # T_cr^T @ f_cr = R_cr^T @ f_cr_blocks (4ブロック)
        f_global = np.empty((n_elems, 12), dtype=float)
        for blk in range(4):
            s = 3 * blk
            f_global[:, s : s + 3] = np.einsum("nij,nj->ni", R_cr_T, f_cr[:, s : s + 3])

        # scatter-add to global
        f_int_global = np.zeros(ndof, dtype=float)
        np.add.at(f_int_global, all_edofs.ravel(), f_global.ravel())

    # --- Step 11: 接線剛性（解析的） ---
    coo_data = None
    if stiffness and analytical_tangent:
        # B行列構築
        S_ex = _batch_skew(e_x_def)  # (n, 3, 3)
        R_cr_S_ex = np.einsum("nij,njk->nik", R_cr, S_ex)  # (n, 3, 3)
        inv_L = 1.0 / L_def  # (n,)
        dpsi_du1 = inv_L[:, None, None] * R_cr_S_ex  # (n, 3, 3)
        dpsi_du2 = -dpsi_du1  # (n, 3, 3)

        T_inv1 = _batch_tangent_operator_inv(theta_def1)  # (n, 3, 3)
        T_inv2 = _batch_tangent_operator_inv(theta_def2)  # (n, 3, 3)
        T_s1 = _batch_tangent_operator(u_all[:, 3:6])  # (n, 3, 3)
        T_s2 = _batch_tangent_operator(u_all[:, 9:12])  # (n, 3, 3)

        B = np.zeros((n_elems, 12, 12), dtype=float)
        # 軸方向
        B[:, 6, 0:3] = -e_x_def
        B[:, 6, 6:9] = e_x_def

        # 節点1回転
        B[:, 3:6, 0:3] = np.einsum("nij,njk->nik", T_inv1, dpsi_du1)
        B[:, 3:6, 6:9] = np.einsum("nij,njk->nik", T_inv1, dpsi_du2)
        B[:, 3:6, 3:6] = np.einsum("nij,njk,nkl->nil", T_inv1, R_cr, T_s1)

        # 節点2回転
        B[:, 9:12, 0:3] = np.einsum("nij,njk->nik", T_inv2, dpsi_du1)
        B[:, 9:12, 6:9] = np.einsum("nij,njk->nik", T_inv2, dpsi_du2)
        B[:, 9:12, 9:12] = np.einsum("nij,njk,nkl->nil", T_inv2, R_cr, T_s2)

        # T_cr^T @ Ke_local @ B (ブロック対角変換)
        # T_cr^T @ M は各3x3ブロックに R_cr^T を適用
        # まず Ke_local @ B
        KB = np.einsum("nij,njk->nik", Ke_local, B)  # (n, 12, 12)

        # T_cr^T @ KB: ブロック対角なので4つの3x12サブブロックを変換
        K_mat = np.empty((n_elems, 12, 12), dtype=float)
        for blk in range(4):
            s = 3 * blk
            K_mat[:, s : s + 3, :] = np.einsum("nij,njk->nik", R_cr_T, KB[:, s : s + 3, :])

        # 幾何剛性 K_geo — 全要素一括計算（activeフィルタなし）
        K_geo = np.zeros((n_elems, 12, 12), dtype=float)
        for blk in range(4):
            s = 3 * blk
            f_blk = f_cr[:, s : s + 3]  # (n, 3)
            Sf = _batch_skew(f_blk)  # (n, 3, 3)
            RtSf = np.einsum("nij,njk->nik", R_cr_T, Sf)  # (n, 3, 3)
            K_geo[:, s : s + 3, 0:3] += np.einsum("nij,njk->nik", RtSf, dpsi_du1)
            K_geo[:, s : s + 3, 6:9] += np.einsum("nij,njk->nik", RtSf, dpsi_du2)

        # K_T = K_mat + K_geo, 対称化
        K_T = K_mat + K_geo  # (n, 12, 12)
        K_T = 0.5 * (K_T + np.einsum("nij->nji", K_T))

        coo_data = K_T.reshape(-1)  # (n * 144,)

    elif stiffness and not analytical_tangent:
        # 数値微分の場合はバッチ化が困難 → フォールバック
        coo_data = None

    return coo_data, f_int_global


# ---------------------------------------------------------------------------
# Corotational + ファイバー弾塑性
# ---------------------------------------------------------------------------


def _cr_extract_deformations(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    v_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CR kinematics: corotated フレームの自然変形を抽出する.

    Args:
        coords_init: (2, 3) 初期節点座標
        u_elem: (12,) 全体座標系の変位ベクトル
        v_ref: 局所y軸の参照ベクトル

    Returns:
        (d_cr, R_cr, L_0): corotated 変形 (12,), 回転行列 (3,3), 初期長
    """
    L_0, e_x_0 = _beam3d_length_and_direction(coords_init)
    R_0 = _build_local_axes(e_x_0, v_ref)

    x1_def = coords_init[0] + u_elem[0:3]
    x2_def = coords_init[1] + u_elem[6:9]
    coords_def = np.array([x1_def, x2_def])
    L_def, e_x_def = _beam3d_length_and_direction(coords_def)

    # Rodrigues 回転による corotated フレーム
    # R_0 の行ベクトルが局所軸: R_cr[i,:] = R_rod @ R_0[i,:] → R_cr = R_0 @ R_rod^T
    R_rod = _rodrigues_rotation(e_x_0, e_x_def)
    R_cr = R_0 @ R_rod.T

    R_node1 = _rotvec_to_rotmat(u_elem[3:6])
    R_node2 = _rotvec_to_rotmat(u_elem[9:12])

    R_def1 = R_cr @ R_node1 @ R_0.T
    R_def2 = R_cr @ R_node2 @ R_0.T
    theta_def1 = _rotmat_to_rotvec(R_def1)
    theta_def2 = _rotmat_to_rotvec(R_def2)

    d_cr = np.zeros(12, dtype=float)
    d_cr[3:6] = theta_def1
    d_cr[9:12] = theta_def2

    d_cr[6] = L_def - L_0

    return d_cr, R_cr, L_0


def cr_beam3d_fiber_internal_force(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    G: float,
    kappa_y: float,
    kappa_z: float,
    fiber_integrator,
    v_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, list]:
    """CR梁＋ファイバー弾塑性の内力ベクトルを計算する.

    CR kinematics で corotated フレームの自然変形を抽出し、
    軸ひずみ・曲率をファイバー断面積分器に渡して断面力を得る。
    せん断・ねじりは弾性のまま。

    変位ベース定式化（1積分点、reduced integration for shear）:
      一般化ひずみ: ε = B @ d_cr
      断面力:       S = [N, Vy, Vz, T, My, Mz]
      内力:         f_cr = L * B^T @ S

    スレンダーな梁（L/d >> 1）では Timoshenko 精解と一致する。

    Args:
        coords_init: (2, 3) 初期節点座標
        u_elem: (12,) 全体座標系の変位ベクトル
        G: せん断弾性率
        kappa_y, kappa_z: せん断補正係数
        fiber_integrator: FiberIntegrator インスタンス
        v_ref: 局所y軸の参照ベクトル

    Returns:
        (f_global, states_new): 全体内力 (12,) と更新された塑性状態
    """
    d_cr, R_cr, L_0 = _cr_extract_deformations(coords_init, u_elem, v_ref)

    # --- 一般化ひずみ（B行列による、要素中央 1点積分）---
    # DOF順: [u1,v1,w1, θx1,θy1,θz1, u2,v2,w2, θx2,θy2,θz2]
    eps_axial = (d_cr[6] - d_cr[0]) / L_0  # 軸ひずみ
    gamma_y = (d_cr[7] - d_cr[1]) / L_0 - (d_cr[5] + d_cr[11]) / 2.0  # せん断 y
    gamma_z = (d_cr[8] - d_cr[2]) / L_0 + (d_cr[4] + d_cr[10]) / 2.0  # せん断 z
    kappa_x = (d_cr[9] - d_cr[3]) / L_0  # ねじり率
    kappa_y_sec = (d_cr[10] - d_cr[4]) / L_0  # 曲率 y（xz面曲げ）
    kappa_z_sec = (d_cr[11] - d_cr[5]) / L_0  # 曲率 z（xy面曲げ）

    # CR フレームでは v1=v2=w1=w2=u1=0 なので簡略化される:
    # eps_axial = d_cr[6] / L_0
    # gamma_y = -(d_cr[5] + d_cr[11]) / 2.0
    # gamma_z = (d_cr[4] + d_cr[10]) / 2.0

    # --- ファイバー断面積分（軸力・曲げモーメント）---
    fiber_result = fiber_integrator.integrate(eps_axial, kappa_y_sec, kappa_z_sec)

    # --- 弾性断面力（せん断・ねじり）---
    sec = fiber_integrator.section
    Vy = G * sec.A * kappa_y * gamma_y
    Vz = G * sec.A * kappa_z * gamma_z
    T = G * sec.J * kappa_x

    # --- 断面力ベクトル ---
    S = np.array(
        [
            fiber_result.N,  # 軸力（ファイバー）
            Vy,  # せん断力 y（弾性）
            Vz,  # せん断力 z（弾性）
            T,  # ねじりモーメント（弾性）
            fiber_result.My,  # 曲げモーメント y（ファイバー）
            fiber_result.Mz,  # 曲げモーメント z（ファイバー）
        ]
    )

    # --- 内力 = L * B^T @ S ---
    # B^T の各行 = B の各列（転置）
    f_cr = np.zeros(12, dtype=float)

    # DOF 0 (u1): B[0,0]=-1/L → f[0] = L*(-1/L)*N = -N
    f_cr[0] = -S[0]
    # DOF 1 (v1): B[1,1]=-1/L → f[1] = L*(-1/L)*Vy = -Vy
    f_cr[1] = -S[1]
    # DOF 2 (w1): B[2,2]=-1/L → f[2] = L*(-1/L)*Vz = -Vz
    f_cr[2] = -S[2]
    # DOF 3 (θx1): B[3,3]=-1/L → f[3] = L*(-1/L)*T = -T
    f_cr[3] = -S[3]
    # DOF 4 (θy1): B[2,4]=1/2, B[4,4]=-1/L → f[4] = L*(Vz/2 - My/L) = L*Vz/2 - My
    f_cr[4] = L_0 * S[2] / 2.0 - S[4]
    # DOF 5 (θz1): B[1,5]=-1/2, B[5,5]=-1/L → f[5] = L*(-Vy/2 - Mz/L) = -L*Vy/2 - Mz
    f_cr[5] = -L_0 * S[1] / 2.0 - S[5]
    # DOF 6 (u2): B[0,6]=1/L → f[6] = L*(1/L)*N = N
    f_cr[6] = S[0]
    # DOF 7 (v2): B[1,7]=1/L → f[7] = L*(1/L)*Vy = Vy
    f_cr[7] = S[1]
    # DOF 8 (w2): B[2,8]=1/L → f[8] = L*(1/L)*Vz = Vz
    f_cr[8] = S[2]
    # DOF 9 (θx2): B[3,9]=1/L → f[9] = L*(1/L)*T = T
    f_cr[9] = S[3]
    # DOF 10 (θy2): B[2,10]=1/2, B[4,10]=1/L → f[10] = L*(Vz/2 + My/L) = L*Vz/2 + My
    f_cr[10] = L_0 * S[2] / 2.0 + S[4]
    # DOF 11 (θz2): B[1,11]=-1/2, B[5,11]=1/L → f[11] = L*(-Vy/2 + Mz/L) = -L*Vy/2 + Mz
    f_cr[11] = -L_0 * S[1] / 2.0 + S[5]

    # --- 全体座標系へ変換 ---
    T_cr = _transformation_matrix_3d(R_cr)
    return T_cr.T @ f_cr, fiber_result.states_new


def _build_fiber_B_matrix(L: float) -> np.ndarray:
    """ファイバー梁要素の B 行列 (6×12) を構築する.

    一般化ひずみ [ε, γy, γz, κx, κy, κz] = B @ d_cr / L 的に定義。
    ただし内力は f = L * B^T @ S なので、B は 1/L のスケーリングを含む。

    DOF 順: [u1,v1,w1, θx1,θy1,θz1, u2,v2,w2, θx2,θy2,θz2]
    ひずみ順: [ε, γy, γz, κx, κy, κz]
    """
    inv_L = 1.0 / L
    B = np.zeros((6, 12), dtype=float)
    # ε = (u2 - u1) / L
    B[0, 0] = -inv_L
    B[0, 6] = inv_L
    # γy = (v2 - v1) / L - (θz1 + θz2) / 2
    B[1, 1] = -inv_L
    B[1, 7] = inv_L
    B[1, 5] = -0.5
    B[1, 11] = -0.5
    # γz = (w2 - w1) / L + (θy1 + θy2) / 2
    B[2, 2] = -inv_L
    B[2, 8] = inv_L
    B[2, 4] = 0.5
    B[2, 10] = 0.5
    # κx = (θx2 - θx1) / L
    B[3, 3] = -inv_L
    B[3, 9] = inv_L
    # κy = (θy2 - θy1) / L
    B[4, 4] = -inv_L
    B[4, 10] = inv_L
    # κz = (θz2 - θz1) / L
    B[5, 5] = -inv_L
    B[5, 11] = inv_L
    return B


def cr_beam3d_fiber_tangent(
    coords_init: np.ndarray,
    u_elem: np.ndarray,
    G: float,
    kappa_y: float,
    kappa_z: float,
    fiber_integrator,
    v_ref: np.ndarray | None = None,
) -> np.ndarray:
    """CR梁＋ファイバー弾塑性の解析的接線剛性行列.

    K = L * B^T @ D @ B

    D は 6×6 断面構成マトリクス:
      - (ε, κy, κz) ブロック: ファイバー consistent tangent C_sec (3×3)
      - γy: G*A*κ_y
      - γz: G*A*κ_z
      - κx: G*J

    Args:
        coords_init, u_elem, G, kappa_y, kappa_z, fiber_integrator, v_ref:
            cr_beam3d_fiber_internal_force と同じ

    Returns:
        K_T: (12, 12) 接線剛性行列（全体座標系）
    """
    d_cr, R_cr, L_0 = _cr_extract_deformations(coords_init, u_elem, v_ref)

    # 一般化ひずみ（内力と同じ計算）
    eps_axial = (d_cr[6] - d_cr[0]) / L_0
    kappa_y_sec = (d_cr[10] - d_cr[4]) / L_0
    kappa_z_sec = (d_cr[11] - d_cr[5]) / L_0

    # ファイバー積分で C_sec を得る
    fiber_result = fiber_integrator.integrate(eps_axial, kappa_y_sec, kappa_z_sec)
    C_sec = fiber_result.C_sec  # (3, 3): (ε, κy, κz)

    # 断面構成マトリクス D (6×6)
    # ひずみ順: [ε(0), γy(1), γz(2), κx(3), κy(4), κz(5)]
    sec = fiber_integrator.section
    D = np.zeros((6, 6), dtype=float)
    # ファイバー tangent (ε, κy, κz) → D の (0,4,5) 行列ブロック
    fiber_idx = [0, 4, 5]
    for i_f, i_d in enumerate(fiber_idx):
        for j_f, j_d in enumerate(fiber_idx):
            D[i_d, j_d] = C_sec[i_f, j_f]
    # 弾性せん断・ねじり
    D[1, 1] = G * sec.A * kappa_y
    D[2, 2] = G * sec.A * kappa_z
    D[3, 3] = G * sec.J

    # B 行列
    B = _build_fiber_B_matrix(L_0)

    # K_cr = L * B^T @ D @ B
    K_cr = L_0 * (B.T @ D @ B)

    # 全体座標系へ変換
    T_cr = _transformation_matrix_3d(R_cr)
    return T_cr.T @ K_cr @ T_cr


class ULCRBeamAssembler:
    """Updated Lagrangian CR 梁アセンブラ.

    標準 CR 定式化はヘリカル梁で ~13° 以上の累積回転に対し収束劣化する
    （Rodrigues 回転のドリル成分による寄生軸力）。
    UL では各収束ステップ後に参照配置を更新し、ステップ内変形を小さく保つ。

    使い方:
        assembler = ULCRBeamAssembler(coords, conn, E, G, A, Iy, Iz, J, kappa)
        # NR 反復で使用
        K = assembler.assemble_tangent(u_incr)
        f = assembler.assemble_internal_force(u_incr)
        # 収束後
        assembler.update_reference(u_converged)
        # 次のステップは u_incr = 0 から開始
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        E: float,
        G: float,
        A: float,
        Iy: float,
        Iz: float,
        J: float,
        kappa_y: float,
        kappa_z: float = 0.0,
        *,
        v_ref: np.ndarray | None = None,
        scf: float | None = None,
    ):
        self.coords_ref = node_coords.copy()
        self.connectivity = connectivity
        self.n_nodes = len(node_coords)
        self.ndof = self.n_nodes * 6
        self.E = E
        self.G = G
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.kappa_y = kappa_y
        self.kappa_z = kappa_z if kappa_z > 0 else kappa_y
        self.v_ref = v_ref
        self.scf = scf
        # 各節点の参照回転行列（累積回転）
        self.R_ref = np.tile(np.eye(3), (self.n_nodes, 1, 1))  # (n_nodes, 3, 3)
        # 累積変位（初期配置からの全変位を追跡、出力用）
        self._u_total_accum = np.zeros(self.ndof)
        # チェックポイント（adaptive Δt ロールバック用）
        self._ckpt_coords_ref: np.ndarray | None = None
        self._ckpt_R_ref: np.ndarray | None = None
        self._ckpt_u_total_accum: np.ndarray | None = None

    def _to_total_u(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位を CR 要素用の変位に変換.

        CR 要素は coords_ref（更新済み参照座標）を基準とした変位を期待する。
        UL では coords_ref が各ステップ後に更新されるため、
        u_incr をそのまま渡すのが正しい（回転合成は不要）。

        R_ref は update_reference 時の整合性チェック用にのみ保持。
        """
        return u_incr.copy()

    def assemble_tangent(self, u_incr: np.ndarray) -> object:
        """増分変位から接線剛性行列を計算.

        UL では u_incr をそのまま CR 要素に渡すため、
        標準の数値微分（timo_beam3d_cr_tangent）がそのまま使える。
        """
        K_T, _ = assemble_cr_beam3d(
            self.coords_ref,
            self.connectivity,
            u_incr,
            self.E,
            self.G,
            self.A,
            self.Iy,
            self.Iz,
            self.J,
            self.kappa_y,
            self.kappa_z,
            v_ref=self.v_ref,
            scf=self.scf,
            stiffness=True,
            internal_force=False,
            analytical_tangent=True,  # 解析的接線剛性（24x高速化）
        )
        return K_T

    def assemble_internal_force(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位から内力ベクトルを計算."""
        u_total = self._to_total_u(u_incr)
        _, f_int = assemble_cr_beam3d(
            self.coords_ref,
            self.connectivity,
            u_total,
            self.E,
            self.G,
            self.A,
            self.Iy,
            self.Iz,
            self.J,
            self.kappa_y,
            self.kappa_z,
            v_ref=self.v_ref,
            scf=self.scf,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    def update_reference(self, u_incr: np.ndarray) -> None:
        """収束後に参照配置を更新.

        参照座標を変形位置に移動し、参照回転を累積回転に更新する。
        次のステップは u_incr = 0 から開始。
        """
        for i in range(self.n_nodes):
            # 並進: 参照座標を更新
            self.coords_ref[i] += u_incr[6 * i : 6 * i + 3]
            # 回転: 参照回転を更新（乗法更新）
            theta_incr = u_incr[6 * i + 3 : 6 * i + 6]
            R_incr = _rotvec_to_rotmat(theta_incr)
            self.R_ref[i] = self.R_ref[i] @ R_incr
        # 累積変位の追跡
        self._u_total_accum += u_incr

    def checkpoint(self) -> None:
        """参照配置のチェックポイントを保存（adaptive Δt ロールバック用）."""
        self._ckpt_coords_ref = self.coords_ref.copy()
        self._ckpt_R_ref = self.R_ref.copy()
        self._ckpt_u_total_accum = self._u_total_accum.copy()

    def rollback(self) -> None:
        """チェックポイントから参照配置を復元."""
        if self._ckpt_coords_ref is not None:
            self.coords_ref = self._ckpt_coords_ref.copy()
            self.R_ref = self._ckpt_R_ref.copy()
            self._u_total_accum = self._ckpt_u_total_accum.copy()

    @property
    def u_total_accum(self) -> np.ndarray:
        """初期配置からの累積変位（出力用）."""
        return self._u_total_accum

    def get_total_displacement(self, u_incr: np.ndarray) -> np.ndarray:
        """増分変位を初期配置からの total 変位に変換.

        累積変位 + 現在の未コミット増分。
        """
        return self._u_total_accum + u_incr


def assemble_cr_beam3d_fiber(
    nodes_init: np.ndarray,
    connectivity: np.ndarray,
    u: np.ndarray,
    G: float,
    kappa_y: float,
    kappa_z: float,
    fiber_integrators: list,
    *,
    v_ref: np.ndarray | None = None,
    stiffness: bool = True,
    internal_force: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, list[list]]:
    """CR梁＋ファイバー弾塑性のグローバルアセンブリ.

    Args:
        nodes_init: (n_nodes, 3) 初期節点座標
        connectivity: (n_elems, 2) 要素接続
        u: (ndof,) 全体変位ベクトル
        G: せん断弾性率
        kappa_y, kappa_z: せん断補正係数
        fiber_integrators: 各要素の FiberIntegrator のリスト
        v_ref: 局所y軸の参照ベクトル
        stiffness: 接線剛性を計算するか
        internal_force: 内力を計算するか

    Returns:
        (K_T, f_int, all_states_new):
            K_T: 接線剛性行列 (ndof, ndof) or None
            f_int: 内力ベクトル (ndof,) or None
            all_states_new: 各要素の更新された塑性状態リスト
    """
    n_nodes = len(nodes_init)
    ndof = 6 * n_nodes

    K_T_global = np.zeros((ndof, ndof), dtype=float) if stiffness else None
    f_int_global = np.zeros(ndof, dtype=float) if internal_force else None
    all_states_new: list[list] = []

    for ie, elem in enumerate(connectivity):
        n1, n2 = int(elem[0]), int(elem[1])
        coords = nodes_init[np.array([n1, n2])]
        edofs = np.array(
            [6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)],
            dtype=int,
        )
        u_elem = u[edofs]
        fi = fiber_integrators[ie]

        if internal_force:
            f_e, states_new = cr_beam3d_fiber_internal_force(
                coords,
                u_elem,
                G,
                kappa_y,
                kappa_z,
                fi,
                v_ref,
            )
            f_int_global[edofs] += f_e
            all_states_new.append(states_new)
        else:
            all_states_new.append([])

        if stiffness:
            K_e = cr_beam3d_fiber_tangent(
                coords,
                u_elem,
                G,
                kappa_y,
                kappa_z,
                fi,
                v_ref,
            )
            K_T_global[np.ix_(edofs, edofs)] += K_e

    return K_T_global, f_int_global, all_states_new
