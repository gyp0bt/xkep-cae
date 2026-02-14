"""Cosserat rod（幾何学的厳密梁）要素.

四元数ベースの回転表現を用いた Cosserat rod の定式化。
本バージョンは線形化版であり、小変形で Timoshenko 3D と同等の物理を扱う。
非線形拡張は Phase 3 で実施する。

定式化:
  配位:     (r(s), q(s))  — 中心線 r ∈ R³, 断面回転 q ∈ S³
  一般化歪み: Γ = R(q)ᵀ r' - e₁  （せん断 + 軸伸び歪み）
              κ = 2·Im(q* ⊗ q')     （曲率 + ねじり歪み）
  構成則:   n = C_Γ · Γ = diag(EA, κy·GA, κz·GA) · Γ
            m = C_κ · κ = diag(GJ, EIy, EIz) · κ

要素離散化:
  - 2節点線形要素
  - 各節点 6 DOF: (ux, uy, uz, θx, θy, θz) — 線形化版
  - 内部状態として各節点に参照四元数 q₀ を保持
  - B行列 + 1点ガウス求積（せん断ロッキング回避）

Phase 3 への拡張方針:
  - DOF は増分回転ベクトル Δθ のまま、内部で四元数更新
  - q_{n+1} = quat_from_rotvec(Δθ) ⊗ q_n
  - 接線剛性 = 材料剛性 + 幾何剛性
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from xkep_cae.math.quaternion import (
    quat_from_rotvec,
    quat_identity,
    quat_multiply,
    quat_rotate_vector,
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
    skew,
)

if TYPE_CHECKING:
    from xkep_cae.core.constitutive import ConstitutiveProtocol
    from xkep_cae.sections.beam import BeamSection


@dataclass
class CosseratStrains:
    """Cosserat rod の一般化歪み.

    Attributes:
        gamma: (3,) 力歪み [Γ₁, Γ₂, Γ₃]
            Γ₁: 軸伸び歪み (= u₁')
            Γ₂: y方向せん断歪み (= u₂' - θ₃)
            Γ₃: z方向せん断歪み (= u₃' + θ₂)
        kappa: (3,) モーメント歪み [κ₁, κ₂, κ₃]
            κ₁: ねじり (= θ₁')
            κ₂: y軸まわり曲率 (= θ₂')
            κ₃: z軸まわり曲率 (= θ₃')
    """

    gamma: np.ndarray  # (3,) 力歪み
    kappa: np.ndarray  # (3,) モーメント歪み


def _cosserat_b_matrix(
    L: float,
    xi: float,
) -> np.ndarray:
    """Cosserat rod の歪み-変位行列 B を構築する.

    一般化歪みベクトル e = [Γ₁, Γ₂, Γ₃, κ₁, κ₂, κ₃] と
    節点変位ベクトル u = [u₁₁,u₂₁,u₃₁,θ₁₁,θ₂₁,θ₃₁, u₁₂,u₂₁,u₃₂,θ₁₂,θ₂₂,θ₃₂]
    の関係: e = B · u

    線形化版（参照配位が直線、R₀ = I）での B 行列:
      Γ₁ = u₁' = N₁'·u₁₁ + N₂'·u₁₂
      Γ₂ = u₂' - θ₃ = N₁'·u₂₁ + N₂'·u₂₂ - N₁·θ₃₁ - N₂·θ₃₂
      Γ₃ = u₃' + θ₂ = N₁'·u₃₁ + N₂'·u₃₂ + N₁·θ₂₁ + N₂·θ₂₂
      κ₁ = θ₁' = N₁'·θ₁₁ + N₂'·θ₁₂
      κ₂ = θ₂' = N₁'·θ₂₁ + N₂'·θ₂₂
      κ₃ = θ₃' = N₁'·θ₃₁ + N₂'·θ₃₂

    Args:
        L: 要素長さ
        xi: 無次元座標 ξ ∈ [0, 1]

    Returns:
        B: (6, 12) 歪み-変位行列
    """
    N1 = 1.0 - xi
    N2 = xi
    dN1 = -1.0 / L
    dN2 = 1.0 / L

    B = np.zeros((6, 12), dtype=float)

    # Γ₁ = u₁' (軸伸び)
    B[0, 0] = dN1   # u₁₁
    B[0, 6] = dN2   # u₁₂

    # Γ₂ = u₂' - θ₃ (y方向せん断)
    B[1, 1] = dN1   # u₂₁
    B[1, 5] = -N1   # -θ₃₁
    B[1, 7] = dN2   # u₂₂
    B[1, 11] = -N2  # -θ₃₂

    # Γ₃ = u₃' + θ₂ (z方向せん断)
    B[2, 2] = dN1   # u₃₁
    B[2, 4] = N1    # θ₂₁
    B[2, 8] = dN2   # u₃₂
    B[2, 10] = N2   # θ₂₂

    # κ₁ = θ₁' (ねじり)
    B[3, 3] = dN1   # θ₁₁
    B[3, 9] = dN2   # θ₁₂

    # κ₂ = θ₂' (y軸曲率)
    B[4, 4] = dN1   # θ₂₁
    B[4, 10] = dN2  # θ₂₂

    # κ₃ = θ₃' (z軸曲率)
    B[5, 5] = dN1   # θ₃₁
    B[5, 11] = dN2  # θ₃₂

    return B


def _cosserat_constitutive_matrix(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
) -> np.ndarray:
    """Cosserat rod の構成行列 C (6x6) を構築する.

    C = diag(EA, κy·GA, κz·GA, GJ, EIy, EIz)

    一般化歪み [Γ₁, Γ₂, Γ₃, κ₁, κ₂, κ₃] に対する
    一般化力   [N,  Vy, Vz, Mx, My, Mz] の関係。

    Args:
        E: ヤング率
        G: せん断弾性率
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        J: ねじり定数
        kappa_y: y方向せん断補正係数
        kappa_z: z方向せん断補正係数

    Returns:
        C: (6, 6) 構成行列
    """
    return np.diag([
        E * A,           # N  = EA · Γ₁
        kappa_y * G * A,  # Vy = κy·GA · Γ₂
        kappa_z * G * A,  # Vz = κz·GA · Γ₃
        G * J,           # Mx = GJ · κ₁
        E * Iy,          # My = EIy · κ₂
        E * Iz,          # Mz = EIz · κ₃
    ])


def cosserat_ke_local(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    L: float,
    kappa_y: float,
    kappa_z: float,
    n_gauss: int = 1,
) -> np.ndarray:
    """Cosserat rod の局所剛性行列 (12x12) を計算する.

    Ke = ∫₀ᴸ B(s)ᵀ · C · B(s) ds

    1点ガウス求積を標準とする（せん断ロッキング回避）。

    Args:
        E: ヤング率
        G: せん断弾性率
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        J: ねじり定数
        L: 要素長さ
        kappa_y: y方向せん断補正係数
        kappa_z: z方向せん断補正係数
        n_gauss: ガウス積分点数（1 or 2）

    Returns:
        Ke: (12, 12) 局所剛性行列（局所座標系）
    """
    C = _cosserat_constitutive_matrix(E, G, A, Iy, Iz, J, kappa_y, kappa_z)

    # ガウス積分点と重み（[0,1]区間）
    if n_gauss == 1:
        gauss_pts = [0.5]
        gauss_wts = [1.0]
    elif n_gauss == 2:
        gauss_pts = [0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)]
        gauss_wts = [0.5, 0.5]
    else:
        raise ValueError(f"n_gauss は 1 または 2 のみサポート: {n_gauss}")

    Ke = np.zeros((12, 12), dtype=float)
    for xi, w in zip(gauss_pts, gauss_wts):
        B = _cosserat_b_matrix(L, xi)
        Ke += w * L * B.T @ C @ B

    return Ke


def cosserat_ke_local_sri(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    L: float,
    kappa_y: float,
    kappa_z: float,
) -> np.ndarray:
    """選択的低減積分 (SRI) による Cosserat rod の局所剛性行列 (12x12).

    せん断成分（Γ₂, Γ₃）のみ1点ガウス求積（低減積分）、
    それ以外（軸伸びΓ₁, ねじりκ₁, 曲率κ₂, κ₃）は2点ガウス求積（完全積分）。

    従来の全成分1点積分との違い:
      - 曲げ・ねじり・軸伸びの積分精度が向上
      - せん断ロッキングを回避（せん断のみ低減積分）
      - 少ない要素数で高精度な曲げ応答を得られる

    定式化:
      Ke = ∫₀ᴸ Bᵀ · C_non_shear · B ds  （2点ガウス）
         + ∫₀ᴸ Bᵀ · C_shear · B ds       （1点ガウス）

    Args:
        E: ヤング率
        G: せん断弾性率
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        J: ねじり定数
        L: 要素長さ
        kappa_y: y方向せん断補正係数
        kappa_z: z方向せん断補正係数

    Returns:
        Ke: (12, 12) 局所剛性行列（局所座標系）
    """
    # せん断成分: Γ₂ (row 1), Γ₃ (row 2)
    C_shear = np.diag([
        0.0,               # Γ₁: 軸伸び → 完全積分
        kappa_y * G * A,   # Γ₂: y方向せん断 → 低減積分
        kappa_z * G * A,   # Γ₃: z方向せん断 → 低減積分
        0.0,               # κ₁: ねじり → 完全積分
        0.0,               # κ₂: y曲率 → 完全積分
        0.0,               # κ₃: z曲率 → 完全積分
    ])
    # 非せん断成分: Γ₁, κ₁, κ₂, κ₃
    C_full = np.diag([
        E * A,     # Γ₁: 軸伸び
        0.0,       # Γ₂: せん断 → 低減積分
        0.0,       # Γ₃: せん断 → 低減積分
        G * J,     # κ₁: ねじり
        E * Iy,    # κ₂: y曲率
        E * Iz,    # κ₃: z曲率
    ])

    # 2点ガウス求積（完全積分: 非せん断成分）
    pts_2, wts_2 = _gauss_points(2)
    Ke = np.zeros((12, 12), dtype=float)
    for xi, w in zip(pts_2, wts_2):
        B = _cosserat_b_matrix(L, xi)
        Ke += w * L * B.T @ C_full @ B

    # 1点ガウス求積（低減積分: せん断成分）
    pts_1, wts_1 = _gauss_points(1)
    for xi, w in zip(pts_1, wts_1):
        B = _cosserat_b_matrix(L, xi)
        Ke += w * L * B.T @ C_shear @ B

    return Ke


def cosserat_ke_global_sri(
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
) -> np.ndarray:
    """SRI版の全体座標系 Cosserat rod 剛性行列 (12x12) を返す.

    Args:
        coords: (2, 3) 節点座標
        E, G, A, Iy, Iz, J: 材料・断面パラメータ
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル

    Returns:
        Ke_global: (12, 12) 全体座標系の剛性行列
    """
    dx = coords[1] - coords[0]
    L = float(np.linalg.norm(dx))
    if L < 1e-15:
        raise ValueError("要素長さがほぼゼロです。")
    e_x = dx / L

    R, _q_ref = _build_local_axes_from_quat(e_x, v_ref)
    Ke_local = cosserat_ke_local_sri(E, G, A, Iy, Iz, J, L, kappa_y, kappa_z)
    T = _transformation_matrix_12(R)
    return T.T @ Ke_local @ T


def cosserat_internal_force_local_sri(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    L: float,
    kappa_y: float,
    kappa_z: float,
    u_local: np.ndarray,
    kappa_0: np.ndarray | None = None,
) -> np.ndarray:
    """SRI版の局所内力ベクトル (12,).

    せん断成分は1点、非せん断成分は2点ガウス求積。

    Args:
        E, G, A, Iy, Iz, J: 材料・断面パラメータ
        L: 要素長さ
        kappa_y, kappa_z: せん断補正係数
        u_local: (12,) 局所座標系の要素変位
        kappa_0: (3,) 初期曲率（None = ゼロ）

    Returns:
        f_int: (12,) 局所内力ベクトル
    """
    C_shear = np.diag([0.0, kappa_y * G * A, kappa_z * G * A, 0.0, 0.0, 0.0])
    C_full = np.diag([E * A, 0.0, 0.0, G * J, E * Iy, E * Iz])

    f_int = np.zeros(12, dtype=float)

    # 2点ガウス（非せん断成分）
    pts_2, wts_2 = _gauss_points(2)
    for xi, w in zip(pts_2, wts_2):
        B = _cosserat_b_matrix(L, xi)
        strain = B @ u_local
        if kappa_0 is not None:
            strain[3:6] = strain[3:6] - kappa_0
        stress = C_full @ strain
        f_int += w * L * B.T @ stress

    # 1点ガウス（せん断成分）
    pts_1, wts_1 = _gauss_points(1)
    for xi, w in zip(pts_1, wts_1):
        B = _cosserat_b_matrix(L, xi)
        strain = B @ u_local
        if kappa_0 is not None:
            strain[3:6] = strain[3:6] - kappa_0
        stress = C_shear @ strain
        f_int += w * L * B.T @ stress

    return f_int


def cosserat_internal_force_global_sri(
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
    kappa_0: np.ndarray | None = None,
) -> np.ndarray:
    """SRI版の全体座標系内力ベクトル (12,) を返す.

    Args:
        coords: (2, 3) 節点座標
        u_elem_global: (12,) 要素変位ベクトル（全体座標系）
        E, G, A, Iy, Iz, J: 材料・断面パラメータ
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        kappa_0: (3,) 初期曲率

    Returns:
        f_int_global: (12,) 全体座標系の内力ベクトル
    """
    dx = coords[1] - coords[0]
    L = float(np.linalg.norm(dx))
    if L < 1e-15:
        raise ValueError("要素長さがほぼゼロです。")
    e_x = dx / L

    R, _q = _build_local_axes_from_quat(e_x, v_ref)
    T = _transformation_matrix_12(R)
    u_local = T @ u_elem_global

    f_int_local = cosserat_internal_force_local_sri(
        E, G, A, Iy, Iz, J, L, kappa_y, kappa_z, u_local,
        kappa_0=kappa_0,
    )
    return T.T @ f_int_local


def _build_local_axes_from_quat(
    e_x: np.ndarray,
    v_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """梁軸方向から局所座標系の回転行列と四元数を構築する.

    Args:
        e_x: 梁軸方向の単位ベクトル
        v_ref: 参照ベクトル（局所y軸を定義するヒント）

    Returns:
        R: (3, 3) 回転行列
        q: (4,) 対応する四元数
    """
    if v_ref is None:
        abs_ex = np.abs(e_x)
        if abs_ex[0] <= abs_ex[1] and abs_ex[0] <= abs_ex[2]:
            v_ref = np.array([1.0, 0.0, 0.0])
        elif abs_ex[1] <= abs_ex[2]:
            v_ref = np.array([0.0, 1.0, 0.0])
        else:
            v_ref = np.array([0.0, 0.0, 1.0])

    e_z = np.cross(e_x, v_ref)
    norm_ez = np.linalg.norm(e_z)
    if norm_ez < 1e-10:
        raise ValueError(
            f"参照ベクトルが梁軸と平行です。v_ref={v_ref}, e_x={e_x}"
        )
    e_z = e_z / norm_ez
    e_y = np.cross(e_z, e_x)

    R = np.zeros((3, 3), dtype=float)
    R[0, :] = e_x
    R[1, :] = e_y
    R[2, :] = e_z

    q = rotation_matrix_to_quat(R)
    return R, q


def _transformation_matrix_12(R: np.ndarray) -> np.ndarray:
    """12x12 座標変換行列を構築する.

    4つの 3x3 回転行列ブロック（変位×2 + 回転×2）。

    Args:
        R: (3, 3) 回転行列

    Returns:
        T: (12, 12) 座標変換行列
    """
    T = np.zeros((12, 12), dtype=float)
    for i in range(4):
        T[3 * i: 3 * i + 3, 3 * i: 3 * i + 3] = R
    return T


def cosserat_ke_global(
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
    n_gauss: int = 1,
) -> np.ndarray:
    """全体座標系での Cosserat rod の剛性行列 (12x12) を返す.

    Ke_global = Tᵀ · Ke_local · T

    Args:
        coords: (2, 3) 節点座標
        E: ヤング率
        G: せん断弾性率
        A: 断面積
        Iy: y軸まわり断面二次モーメント
        Iz: z軸まわり断面二次モーメント
        J: ねじり定数
        kappa_y: y方向せん断補正係数
        kappa_z: z方向せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        n_gauss: ガウス積分点数

    Returns:
        Ke_global: (12, 12) 全体座標系の剛性行列
    """
    dx = coords[1] - coords[0]
    L = float(np.linalg.norm(dx))
    if L < 1e-15:
        raise ValueError("要素長さがほぼゼロです。")
    e_x = dx / L

    R, _q_ref = _build_local_axes_from_quat(e_x, v_ref)
    Ke_local = cosserat_ke_local(E, G, A, Iy, Iz, J, L, kappa_y, kappa_z, n_gauss)
    T = _transformation_matrix_12(R)
    return T.T @ Ke_local @ T


def cosserat_generalized_strains(
    coords: np.ndarray,
    u_elem_local: np.ndarray,
    q_ref: np.ndarray | None = None,
) -> CosseratStrains:
    """一般化歪み (Γ, κ) を計算する.

    線形化版: 参照配位が直線（q_ref = 恒等四元数）の場合。

    Args:
        coords: (2, 3) 局所座標系の節点座標
        u_elem_local: (12,) 局所座標系の要素変位
        q_ref: (4,) 参照四元数（None = 恒等四元数 = 直線参照）

    Returns:
        CosseratStrains: 要素中央での一般化歪み
    """
    dx = coords[1] - coords[0]
    L = float(np.linalg.norm(dx))

    # 要素中央 (ξ=0.5) での B 行列
    B = _cosserat_b_matrix(L, 0.5)
    strain_vec = B @ u_elem_local

    return CosseratStrains(
        gamma=strain_vec[0:3],
        kappa=strain_vec[3:6],
    )


@dataclass
class CosseratForces:
    """Cosserat rod の一般化断面力（body frame）.

    Attributes:
        N: 軸力（引張正）
        Vy: y方向せん断力
        Vz: z方向せん断力
        Mx: ねじりモーメント（トルク）
        My: y軸まわり曲げモーメント
        Mz: z軸まわり曲げモーメント
    """

    N: float
    Vy: float
    Vz: float
    Mx: float
    My: float
    Mz: float


def cosserat_section_forces(
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
    n_gauss: int = 1,
) -> tuple[CosseratForces, CosseratForces]:
    """要素両端の断面力を計算する（局所座標系）.

    節点力 = Ke_local · u_local から断面力を抽出。
    節点1: 断面力 = -f_local[0:6]
    節点2: 断面力 = f_local[6:12]

    Args:
        coords: (2, 3) 節点座標
        u_elem_global: (12,) 要素変位ベクトル（全体座標系）
        E, G, A, Iy, Iz, J: 材料・断面パラメータ
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        n_gauss: ガウス積分点数

    Returns:
        (forces_1, forces_2): 両端の断面力
    """
    dx = coords[1] - coords[0]
    L = float(np.linalg.norm(dx))
    if L < 1e-15:
        raise ValueError("要素長さがほぼゼロです。")
    e_x = dx / L

    R, _q = _build_local_axes_from_quat(e_x, v_ref)
    T = _transformation_matrix_12(R)

    Ke_local = cosserat_ke_local(E, G, A, Iy, Iz, J, L, kappa_y, kappa_z, n_gauss)
    u_local = T @ u_elem_global
    f_local = Ke_local @ u_local

    # 節点1: 断面力 = -f_local[0:6]
    forces_1 = CosseratForces(
        N=-f_local[0],
        Vy=-f_local[1],
        Vz=-f_local[2],
        Mx=-f_local[3],
        My=-f_local[4],
        Mz=-f_local[5],
    )
    # 節点2: 断面力 = f_local[6:12]
    forces_2 = CosseratForces(
        N=f_local[6],
        Vy=f_local[7],
        Vz=f_local[8],
        Mx=f_local[9],
        My=f_local[10],
        Mz=f_local[11],
    )
    return forces_1, forces_2


def _gauss_points(n_gauss: int) -> tuple[list[float], list[float]]:
    """[0,1]区間上のガウス積分点と重みを返す."""
    if n_gauss == 1:
        return [0.5], [1.0]
    elif n_gauss == 2:
        return (
            [0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)],
            [0.5, 0.5],
        )
    else:
        raise ValueError(f"n_gauss は 1 または 2 のみサポート: {n_gauss}")


def cosserat_internal_force_local(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    L: float,
    kappa_y: float,
    kappa_z: float,
    u_local: np.ndarray,
    kappa_0: np.ndarray | None = None,
    n_gauss: int = 1,
) -> np.ndarray:
    """Cosserat rod の局所内力ベクトル (12,) を計算する.

    f_int = ∫₀ᴸ Bᵀ · σ ds,  σ = C · (ε - ε₀)

    線形版では f_int = Ke · u と等価だが、非線形拡張時にはこちらが基準。

    Args:
        E, G, A, Iy, Iz, J: 材料・断面パラメータ
        L: 要素長さ
        kappa_y, kappa_z: せん断補正係数
        u_local: (12,) 局所座標系の要素変位
        kappa_0: (3,) 初期曲率 [κ₁₀, κ₂₀, κ₃₀]（None = ゼロ）
        n_gauss: ガウス積分点数

    Returns:
        f_int: (12,) 局所内力ベクトル
    """
    C = _cosserat_constitutive_matrix(E, G, A, Iy, Iz, J, kappa_y, kappa_z)
    gauss_pts, gauss_wts = _gauss_points(n_gauss)

    f_int = np.zeros(12, dtype=float)
    for xi, w in zip(gauss_pts, gauss_wts):
        B = _cosserat_b_matrix(L, xi)
        strain = B @ u_local  # (6,)
        # 初期曲率を差し引く（初期歪みの減算）
        if kappa_0 is not None:
            strain[3:6] = strain[3:6] - kappa_0
        stress = C @ strain
        f_int += w * L * B.T @ stress

    return f_int


def cosserat_internal_force_global(
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
    kappa_0: np.ndarray | None = None,
    n_gauss: int = 1,
) -> np.ndarray:
    """全体座標系での Cosserat rod の内力ベクトル (12,) を返す.

    f_int_global = Tᵀ · f_int_local

    Args:
        coords: (2, 3) 節点座標
        u_elem_global: (12,) 要素変位ベクトル（全体座標系）
        E, G, A, Iy, Iz, J: 材料・断面パラメータ
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        kappa_0: (3,) 初期曲率（局所座標系）
        n_gauss: ガウス積分点数

    Returns:
        f_int_global: (12,) 全体座標系の内力ベクトル
    """
    dx = coords[1] - coords[0]
    L = float(np.linalg.norm(dx))
    if L < 1e-15:
        raise ValueError("要素長さがほぼゼロです。")
    e_x = dx / L

    R, _q = _build_local_axes_from_quat(e_x, v_ref)
    T = _transformation_matrix_12(R)
    u_local = T @ u_elem_global

    f_int_local = cosserat_internal_force_local(
        E, G, A, Iy, Iz, J, L, kappa_y, kappa_z, u_local,
        kappa_0=kappa_0, n_gauss=n_gauss,
    )
    return T.T @ f_int_local


def cosserat_geometric_stiffness_local(
    L: float,
    stress: np.ndarray,
    n_gauss: int = 1,
) -> np.ndarray:
    """Cosserat rod の局所幾何剛性行列 Kg (12x12) を計算する.

    軸力 N による幾何剛性（初期応力効果）。
    線形座屈解析および Newton-Raphson 法の接線剛性修正に使用。

    定式化:
      δΠ_g = ∫₀ᴸ N · (δu₂'·u₂' + δu₃'·u₃') ds
            + ∫₀ᴸ N · (δθ₂·θ₂ + δθ₃·θ₃) ds  （近似項）

    ここで u₂, u₃ は横方向変位、N は軸力。
    1点ガウス求積で離散化する。

    Args:
        L: 要素長さ
        stress: (6,) 一般化応力ベクトル [N, Vy, Vz, Mx, My, Mz]
        n_gauss: ガウス積分点数

    Returns:
        Kg: (12, 12) 局所幾何剛性行列
    """
    N = stress[0]   # 軸力
    Mx = stress[3]  # ねじりモーメント

    gauss_pts, gauss_wts = _gauss_points(n_gauss)

    Kg = np.zeros((12, 12), dtype=float)

    for xi, w in zip(gauss_pts, gauss_wts):
        dN1 = -1.0 / L
        dN2 = 1.0 / L
        N1 = 1.0 - xi
        N2 = xi

        # 軸力 N による横方向変位微分の二次形式
        # δu₂' · u₂' + δu₃' · u₃' → 形状関数微分の外積
        # DOF: u₂₁=1, u₂₂=7, u₃₁=2, u₃₂=8
        G_trans = np.zeros((2, 12), dtype=float)
        # u₂ の微分
        G_trans[0, 1] = dN1   # u₂₁
        G_trans[0, 7] = dN2   # u₂₂
        # u₃ の微分
        G_trans[1, 2] = dN1   # u₃₁
        G_trans[1, 8] = dN2   # u₃₂

        Kg += w * L * N * G_trans.T @ G_trans

        # ねじりモーメント Mx による回転の連成項
        # Mx による θ₂-θ₃ 連成（Wagner 効果の近似）
        G_rot = np.zeros((2, 12), dtype=float)
        G_rot[0, 4] = dN1   # θ₂₁
        G_rot[0, 10] = dN2  # θ₂₂
        G_rot[1, 5] = dN1   # θ₃₁
        G_rot[1, 11] = dN2  # θ₃₂

        Kg += w * L * N * G_rot.T @ G_rot

        # Mx による θ₂-θ₃ の反対称連成（ねじり-曲げ連成）
        G_twist = np.zeros((2, 12), dtype=float)
        # θ₂ の形状関数
        G_twist[0, 4] = N1
        G_twist[0, 10] = N2
        # θ₃ の形状関数
        G_twist[1, 5] = N1
        G_twist[1, 11] = N2

        # 反対称行列による連成: Mx * (δθ₂·θ₃' - δθ₃·θ₂')
        G_twist_d = np.zeros((2, 12), dtype=float)
        G_twist_d[0, 4] = dN1
        G_twist_d[0, 10] = dN2
        G_twist_d[1, 5] = dN1
        G_twist_d[1, 11] = dN2

        # 反対称寄与: Mx * (G_twist[0]^T * G_twist_d[1] - G_twist[1]^T * G_twist_d[0])
        Kg_twist = Mx * (
            np.outer(G_twist[0], G_twist_d[1])
            - np.outer(G_twist[1], G_twist_d[0])
        )
        Kg += w * L * Kg_twist

    # 対称化（数値誤差補正）
    Kg = 0.5 * (Kg + Kg.T)
    return Kg


def cosserat_geometric_stiffness_global(
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
    kappa_0: np.ndarray | None = None,
    n_gauss: int = 1,
) -> np.ndarray:
    """全体座標系での Cosserat rod の幾何剛性行列 (12x12) を返す.

    Kg_global = Tᵀ · Kg_local · T

    Args:
        coords: (2, 3) 節点座標
        u_elem_global: (12,) 要素変位ベクトル（全体座標系）
        E, G, A, Iy, Iz, J: 材料・断面パラメータ
        kappa_y, kappa_z: せん断補正係数
        v_ref: 局所y軸の参照ベクトル
        kappa_0: (3,) 初期曲率（局所座標系）
        n_gauss: ガウス積分点数

    Returns:
        Kg_global: (12, 12) 全体座標系の幾何剛性行列
    """
    dx = coords[1] - coords[0]
    L = float(np.linalg.norm(dx))
    if L < 1e-15:
        raise ValueError("要素長さがほぼゼロです。")
    e_x = dx / L

    R, _q = _build_local_axes_from_quat(e_x, v_ref)
    T = _transformation_matrix_12(R)
    u_local = T @ u_elem_global

    # 一般化応力を計算（構成則）
    C = _cosserat_constitutive_matrix(E, G, A, Iy, Iz, J, kappa_y, kappa_z)
    B_mid = _cosserat_b_matrix(L, 0.5)
    strain = B_mid @ u_local
    if kappa_0 is not None:
        strain[3:6] = strain[3:6] - kappa_0
    stress = C @ strain

    Kg_local = cosserat_geometric_stiffness_local(L, stress, n_gauss)
    return T.T @ Kg_local @ T


class CosseratRod:
    """Cosserat rod 要素（ElementProtocol 適合）.

    四元数ベースの回転表現を用いた幾何学的厳密梁の線形化版。
    B行列 + ガウス求積でせん断ロッキングを回避する。

    Timoshenko 3D との違い:
      - 内部で四元数状態を保持（Phase 3 での非線形拡張の準備）
      - 一般化歪み (Γ, κ) を明示的に計算
      - B行列ベースの定式化
      - 線形化時は Timoshenko 3D と同じ物理を扱うが、
        要素剛性行列は等価ではない（B-matrix 定式化と解析定式化の違い）

    積分スキーム:
      - "uniform": 全成分を n_gauss 点で一様積分（デフォルト、1点推奨）
      - "sri": せん断(Γ₂,Γ₃)のみ1点低減積分、他は2点完全積分
        SRIは少ない要素数で曲げ精度を向上させる

    収束特性:
      - uniform/1点: 軸力・ねじり厳密、曲げはメッシュ依存
      - SRI: 軸力・ねじり厳密、曲げの収束が高速

    Args:
        section: 梁断面特性
        kappa_y: y方向せん断補正係数（float or "cowper"）
        kappa_z: z方向せん断補正係数（float or "cowper"）
        v_ref: 局所y軸の参照ベクトル
        n_gauss: ガウス積分点数（1 or 2、uniform スキームのみ有効）
        kappa_0: 初期曲率ベクトル
        integration_scheme: 積分スキーム（"uniform" or "sri"）

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
        n_gauss: int = 1,
        kappa_0: np.ndarray | None = None,
        integration_scheme: str = "uniform",
    ) -> None:
        if integration_scheme not in ("uniform", "sri"):
            raise ValueError(
                f"integration_scheme は 'uniform' または 'sri' のみ: '{integration_scheme}'"
            )
        self.section = section
        self.v_ref = v_ref
        self.n_gauss = n_gauss
        self.integration_scheme = integration_scheme
        self._kappa_0 = np.array(kappa_0, dtype=float) if kappa_0 is not None else None

        # 各節点の参照四元数（線形化版では恒等四元数）
        self._q_ref_nodes: list[np.ndarray] = [
            quat_identity(),
            quat_identity(),
        ]

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

    @property
    def q_ref_nodes(self) -> list[np.ndarray]:
        """各節点の参照四元数を返す."""
        return self._q_ref_nodes

    def _resolve_kappa_y(self, nu: float) -> float:
        if self._kappa_y_mode == "cowper":
            return self.section.cowper_kappa_y(nu)
        assert self._kappa_y_value is not None
        return self._kappa_y_value

    def _resolve_kappa_z(self, nu: float) -> float:
        if self._kappa_z_mode == "cowper":
            return self.section.cowper_kappa_z(nu)
        assert self._kappa_z_value is not None
        return self._kappa_z_value

    def _extract_material_props(
        self, material: ConstitutiveProtocol,
    ) -> tuple[float, float, float]:
        """材料オブジェクトから E, G, nu を抽出する."""
        D = material.tangent()
        if np.ndim(D) == 0:
            E = float(D)
        elif D.shape == (1,):
            E = float(D[0])
        elif D.shape == (1, 1):
            E = float(D[0, 0])
        else:
            raise ValueError(
                f"梁要素にはスカラーまたは(1,1)の弾性テンソルが必要です。shape={D.shape}"
            )

        nu = float(material.nu) if hasattr(material, "nu") else 0.3

        if hasattr(material, "G"):
            G = float(material.G)
        elif hasattr(material, "nu"):
            G = E / (2.0 * (1.0 + nu))
        else:
            raise ValueError(
                "材料オブジェクトからせん断弾性率を取得できません。G or nu が必要です。"
            )

        return E, G, nu

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
        E, G, nu = self._extract_material_props(material)
        kappa_y = self._resolve_kappa_y(nu)
        kappa_z = self._resolve_kappa_z(nu)

        if self.integration_scheme == "sri":
            return cosserat_ke_global_sri(
                coords, E, G,
                self.section.A, self.section.Iy, self.section.Iz, self.section.J,
                kappa_y, kappa_z,
                v_ref=self.v_ref,
            )
        return cosserat_ke_global(
            coords, E, G,
            self.section.A, self.section.Iy, self.section.Iz, self.section.J,
            kappa_y, kappa_z,
            v_ref=self.v_ref,
            n_gauss=self.n_gauss,
        )

    def dof_indices(self, node_indices: np.ndarray) -> np.ndarray:
        """グローバル節点インデックスから要素DOFインデックスを返す.

        6 DOF/node: (ux, uy, uz, θx, θy, θz)
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
    ) -> tuple[CosseratForces, CosseratForces]:
        """要素両端の断面力を計算する.

        Args:
            coords: (2, 3) 節点座標
            u_elem_global: (12,) 要素変位ベクトル（全体座標系）
            material: 構成則

        Returns:
            (forces_1, forces_2): 両端の断面力（局所座標系）
        """
        E, G, nu = self._extract_material_props(material)
        kappa_y = self._resolve_kappa_y(nu)
        kappa_z = self._resolve_kappa_z(nu)

        return cosserat_section_forces(
            coords, u_elem_global,
            E, G,
            self.section.A, self.section.Iy, self.section.Iz, self.section.J,
            kappa_y, kappa_z,
            v_ref=self.v_ref,
            n_gauss=self.n_gauss,
        )

    def internal_force(
        self,
        coords: np.ndarray,
        u_elem_global: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> np.ndarray:
        """全体座標系の内力ベクトル (12,) を返す.

        f_int = Tᵀ · ∫₀ᴸ Bᵀ · σ ds

        線形版では f_int = Ke · u と等価。Phase 3 非線形版への基盤。

        Args:
            coords: (2, 3) 節点座標
            u_elem_global: (12,) 要素変位ベクトル（全体座標系）
            material: 構成則

        Returns:
            f_int: (12,) 全体座標系の内力ベクトル
        """
        E, G, nu = self._extract_material_props(material)
        kappa_y = self._resolve_kappa_y(nu)
        kappa_z = self._resolve_kappa_z(nu)

        if self.integration_scheme == "sri":
            return cosserat_internal_force_global_sri(
                coords, u_elem_global,
                E, G,
                self.section.A, self.section.Iy, self.section.Iz, self.section.J,
                kappa_y, kappa_z,
                v_ref=self.v_ref,
                kappa_0=self._kappa_0,
            )
        return cosserat_internal_force_global(
            coords, u_elem_global,
            E, G,
            self.section.A, self.section.Iy, self.section.Iz, self.section.J,
            kappa_y, kappa_z,
            v_ref=self.v_ref,
            kappa_0=self._kappa_0,
            n_gauss=self.n_gauss,
        )

    def geometric_stiffness(
        self,
        coords: np.ndarray,
        u_elem_global: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> np.ndarray:
        """全体座標系の幾何剛性行列 (12x12) を返す.

        初期応力（軸力・ねじり）による幾何学的剛性効果を計算。
        接線剛性 = 材料剛性（local_stiffness） + 幾何剛性

        Args:
            coords: (2, 3) 節点座標
            u_elem_global: (12,) 要素変位ベクトル（全体座標系）
            material: 構成則

        Returns:
            Kg: (12, 12) 全体座標系の幾何剛性行列
        """
        E, G, nu = self._extract_material_props(material)
        kappa_y = self._resolve_kappa_y(nu)
        kappa_z = self._resolve_kappa_z(nu)

        return cosserat_geometric_stiffness_global(
            coords, u_elem_global,
            E, G,
            self.section.A, self.section.Iy, self.section.Iz, self.section.J,
            kappa_y, kappa_z,
            v_ref=self.v_ref,
            kappa_0=self._kappa_0,
            n_gauss=self.n_gauss,
        )

    @property
    def kappa_0(self) -> np.ndarray | None:
        """初期曲率ベクトルを返す."""
        return self._kappa_0

    def compute_strains(
        self,
        coords: np.ndarray,
        u_elem_global: np.ndarray,
    ) -> CosseratStrains:
        """一般化歪み (Γ, κ) を計算する.

        Args:
            coords: (2, 3) 節点座標
            u_elem_global: (12,) 要素変位ベクトル（全体座標系）

        Returns:
            CosseratStrains: 要素中央での一般化歪み（局所座標系）
        """
        dx = coords[1] - coords[0]
        L = float(np.linalg.norm(dx))
        e_x = dx / L
        R, _ = _build_local_axes_from_quat(e_x, self.v_ref)
        T = _transformation_matrix_12(R)
        u_local = T @ u_elem_global

        return cosserat_generalized_strains(
            np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]]),
            u_local,
        )

    def tangent_stiffness(
        self,
        coords: np.ndarray,
        u_elem_global: np.ndarray,
        material: ConstitutiveProtocol,
    ) -> np.ndarray:
        """接線剛性行列 K_T = K_material + K_geometric (12x12) を返す.

        Newton-Raphson法の反復で使用する接線剛性。

        Args:
            coords: (2, 3) 節点座標
            u_elem_global: (12,) 要素変位ベクトル（全体座標系）
            material: 構成則

        Returns:
            K_T: (12, 12) 全体座標系の接線剛性行列
        """
        Km = self.local_stiffness(coords, material)
        Kg = self.geometric_stiffness(coords, u_elem_global, material)
        return Km + Kg


# ===========================================================================
# 非線形解析ヘルパー: Cosserat rod の非線形梁解析
# ===========================================================================

def assemble_cosserat_beam(
    n_elems: int,
    beam_length: float,
    rod: CosseratRod,
    material: ConstitutiveProtocol,
    u: np.ndarray,
    *,
    stiffness: bool = True,
    internal_force: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Cosserat rod 梁のアセンブリ（直線梁、x軸方向）.

    非線形解析のコールバック用にアセンブリ関数を提供する。

    Args:
        n_elems: 要素数
        beam_length: 梁長さ
        rod: CosseratRod 要素
        material: 構成則
        u: (total_dof,) 現在の変位ベクトル
        stiffness: 接線剛性行列を計算するか
        internal_force: 内力ベクトルを計算するか

    Returns:
        (K_T, f_int): 接線剛性行列と内力ベクトル（不要なものはNone）
    """
    n_nodes = n_elems + 1
    total_dof = n_nodes * 6
    elem_len = beam_length / n_elems

    K_T = np.zeros((total_dof, total_dof)) if stiffness else None
    f_int = np.zeros(total_dof) if internal_force else None

    for i in range(n_elems):
        coords = np.array([
            [i * elem_len, 0.0, 0.0],
            [(i + 1) * elem_len, 0.0, 0.0],
        ])
        dof_start = 6 * i
        dof_end = 6 * (i + 2)
        u_elem = u[dof_start:dof_end]

        if stiffness and K_T is not None:
            K_T_e = rod.tangent_stiffness(coords, u_elem, material)
            K_T[dof_start:dof_end, dof_start:dof_end] += K_T_e

        if internal_force and f_int is not None:
            f_int_e = rod.internal_force(coords, u_elem, material)
            f_int[dof_start:dof_end] += f_int_e

    return K_T, f_int
