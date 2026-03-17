"""Corotational (CR) Timoshenko 3D 梁要素の関数群.

CR定式化は線形 Timoshenko 剛性を corotated フレームで使い、
幾何学的非線形性を捕捉する。

主要関数:
  - timo_beam3d_cr_internal_force: CR内力ベクトル
  - timo_beam3d_cr_tangent: 数値微分による接線剛性
  - timo_beam3d_cr_tangent_analytical: Battini & Pacoste (2002) 解析的接線剛性

補助関数:
  - _beam3d_length_and_direction: 要素長さと方向ベクトル
  - _build_local_axes: 局所座標系構築
  - _transformation_matrix_3d: 12x12 座標変換行列
  - timo_beam3d_ke_local: 局所剛性行列
  - timo_beam3d_ke_global: 全体剛性行列
  - timo_beam3d_lumped_mass_local: 集中質量行列
  - timo_beam3d_mass_local: 整合質量行列
  - timo_beam3d_mass_global: 全体質量行列
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =====================================================================
# 断面力データクラス
# =====================================================================


@dataclass(frozen=True)
class BeamForces3D:
    """3D梁要素の断面力（局所座標系）."""

    N: float
    Vy: float
    Vz: float
    Mx: float
    My: float
    Mz: float


# =====================================================================
# 基本幾何関数
# =====================================================================


def _beam3d_length_and_direction(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """3D梁要素の長さと方向ベクトルを計算する."""
    dx = coords[1] - coords[0]
    length = float(np.linalg.norm(dx))
    if length < 1e-15:
        raise ValueError("要素長さがほぼゼロです。2節点が同一座標です。")
    return length, dx / length


def _build_local_axes(
    e_x: np.ndarray,
    v_ref: np.ndarray | None = None,
) -> np.ndarray:
    """局所座標系の回転行列 R (3x3) を構築する."""
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
        raise ValueError(f"参照ベクトルが梁軸と平行です。v_ref={v_ref}, e_x={e_x}")
    e_z = e_z / norm_ez
    e_y = np.cross(e_z, e_x)

    R = np.zeros((3, 3), dtype=float)
    R[0, :] = e_x
    R[1, :] = e_y
    R[2, :] = e_z
    return R


def _transformation_matrix_3d(R: np.ndarray) -> np.ndarray:
    """3D梁の座標変換行列 T (12x12) を返す."""
    T = np.zeros((12, 12), dtype=float)
    for i in range(4):
        T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = R
    return T


# =====================================================================
# 剛性行列
# =====================================================================


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
    """3D Timoshenko梁の局所剛性行列 (12x12) を返す."""
    Phi_y = 12.0 * E * Iy / (kappa_y * G * A * L * L)
    Phi_z = 12.0 * E * Iz / (kappa_z * G * A * L * L)

    if scf is not None and scf > 0.0:
        slenderness_y = L * L * A / (12.0 * Iy)
        f_p_y = 1.0 / (1.0 + scf * slenderness_y)
        Phi_y = Phi_y * f_p_y
        slenderness_z = L * L * A / (12.0 * Iz)
        f_p_z = 1.0 / (1.0 + scf * slenderness_z)
        Phi_z = Phi_z * f_p_z

    denom_y = 1.0 + Phi_y
    denom_z = 1.0 + Phi_z

    Ke = np.zeros((12, 12), dtype=float)

    EA_L = E * A / L
    Ke[0, 0] = EA_L
    Ke[0, 6] = -EA_L
    Ke[6, 0] = -EA_L
    Ke[6, 6] = EA_L

    GJ_L = G * J / L
    Ke[3, 3] = GJ_L
    Ke[3, 9] = -GJ_L
    Ke[9, 3] = -GJ_L
    Ke[9, 9] = GJ_L

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
    """全体座標系での 3D Timoshenko 梁の剛性行列 (12x12) を返す."""
    length, e_x = _beam3d_length_and_direction(coords)
    R = _build_local_axes(e_x, v_ref)
    Ke_local = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, length, kappa_y, kappa_z, scf=scf)
    T = _transformation_matrix_3d(R)
    return T.T @ Ke_local @ T


# =====================================================================
# 質量行列
# =====================================================================


def timo_beam3d_mass_local(
    rho: float,
    A: float,
    Iy: float,
    Iz: float,
    L: float,
) -> np.ndarray:
    """3D梁の局所整合質量行列 (12x12) を返す."""
    m = rho * A * L
    Me = np.zeros((12, 12), dtype=float)

    Me[0, 0] = m / 3.0
    Me[0, 6] = m / 6.0
    Me[6, 0] = m / 6.0
    Me[6, 6] = m / 3.0

    Ip = Iy + Iz
    m_torsion = rho * Ip * L
    Me[3, 3] = m_torsion / 3.0
    Me[3, 9] = m_torsion / 6.0
    Me[9, 3] = m_torsion / 6.0
    Me[9, 9] = m_torsion / 3.0

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
    """3D梁の局所集中質量行列 (12x12, 対角) を返す."""
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
    """全体座標系での3D梁の質量行列 (12x12) を返す."""
    L, e_x = _beam3d_length_and_direction(coords)

    if lumped:
        return timo_beam3d_lumped_mass_local(rho, A, Iy, Iz, L)

    R = _build_local_axes(e_x, v_ref)
    Me_local = timo_beam3d_mass_local(rho, A, Iy, Iz, L)
    T = _transformation_matrix_3d(R)
    return T.T @ Me_local @ T


# =====================================================================
# 四元数ユーティリティ（C14準拠: deprecated import を避けるため自己完結）
# =====================================================================


def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """回転ベクトル → 四元数（指数写像）."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        half_angle_sq = angle * angle / 4.0
        w = 1.0 - half_angle_sq / 2.0
        coeff = 0.5 - half_angle_sq / 48.0
        q = np.array([w, coeff * rotvec[0], coeff * rotvec[1], coeff * rotvec[2]])
        return q / np.linalg.norm(q)
    half = angle / 2.0
    sinc = np.sin(half) / angle
    return np.array([np.cos(half), sinc * rotvec[0], sinc * rotvec[1], sinc * rotvec[2]])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """四元数 → 3x3 回転行列."""
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 回転行列 → 四元数（Shepperd法）."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """四元数 → 回転ベクトル（対数写像）."""
    if q[0] < 0:
        q = -q
    vec = q[1:4]
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1e-12:
        return 2.0 * vec
    angle = 2.0 * np.arctan2(sin_half, q[0])
    return (angle / sin_half) * vec


# =====================================================================
# CR 回転ユーティリティ
# =====================================================================


def _rotvec_to_rotmat(theta: np.ndarray) -> np.ndarray:
    """回転ベクトル → 回転行列（四元数経由）."""
    q = _quat_from_rotvec(theta)
    return _quat_to_rotmat(q)


def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """回転行列 → 回転ベクトル（四元数経由）."""
    q = _rotmat_to_quat(R)
    return _quat_to_rotvec(q)


def _rodrigues_rotation(e_from: np.ndarray, e_to: np.ndarray) -> np.ndarray:
    """最小回転行列（Rodrigues 公式）."""
    v = np.cross(e_from, e_to)
    s = np.linalg.norm(v)
    c = np.dot(e_from, e_to)

    if s < 1e-14:
        if c > 0:
            return np.eye(3)
        else:
            abs_ef = np.abs(e_from)
            if abs_ef[0] <= abs_ef[1] and abs_ef[0] <= abs_ef[2]:
                perp = np.array([1.0, 0.0, 0.0])
            elif abs_ef[1] <= abs_ef[2]:
                perp = np.array([0.0, 1.0, 0.0])
            else:
                perp = np.array([0.0, 0.0, 1.0])
            axis = np.cross(e_from, perp)
            axis = axis / np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + vx + (vx @ vx) * (1.0 - c) / (s * s)


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
    """指数写像の接線演算子 T_s(θ) を計算する."""
    angle = float(np.linalg.norm(theta))
    if angle < 1e-10:
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
    """指数写像の接線演算子の逆 T_s^{-1}(θ) を計算する."""
    angle = float(np.linalg.norm(theta))
    if angle < 1e-10:
        S = _skew(theta)
        return np.eye(3) - 0.5 * S + (1.0 / 12.0) * (S @ S)

    S = _skew(theta)
    S2 = S @ S
    c = np.cos(angle)
    s = np.sin(angle)
    coeff = 1.0 / (angle * angle) - (1.0 + c) / (2.0 * angle * s)
    return np.eye(3) - 0.5 * S + coeff * S2


# =====================================================================
# CR 内力・接線剛性（スカラー版）
# =====================================================================


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
    """Corotational 定式化による Timoshenko 3D 梁の非線形内力ベクトル."""
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

    Ke_local = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, L_0, kappa_y, kappa_z, scf=scf)
    f_cr = Ke_local @ d_cr

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
    """Corotational Timoshenko 3D 梁の接線剛性行列（数値微分）."""
    eps = 1e-7
    K_T = np.zeros((12, 12), dtype=float)
    u_p = u_elem.copy()
    u_m = u_elem.copy()

    for j in range(12):
        u_p[j] = u_elem[j] + eps
        u_m[j] = u_elem[j] - eps
        f_p = timo_beam3d_cr_internal_force(
            coords_init, u_p, E, G, A, Iy, Iz, J, kappa_y, kappa_z, v_ref=v_ref, scf=scf
        )
        f_m = timo_beam3d_cr_internal_force(
            coords_init, u_m, E, G, A, Iy, Iz, J, kappa_y, kappa_z, v_ref=v_ref, scf=scf
        )
        K_T[:, j] = (f_p - f_m) / (2.0 * eps)
        u_p[j] = u_elem[j]
        u_m[j] = u_elem[j]

    return 0.5 * (K_T + K_T.T)


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
    """Corotational Timoshenko 3D 梁の解析的接線剛性行列（Battini & Pacoste 2002）."""
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

    Ke_local = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, L_0, kappa_y, kappa_z, scf=scf)
    f_cr = Ke_local @ d_cr

    B = np.zeros((12, 12), dtype=float)
    B[6, 0:3] = -e_x_def
    B[6, 6:9] = e_x_def

    S_ex = _skew(e_x_def)
    R_cr_S_ex = R_cr @ S_ex
    dpsi_du1 = (1.0 / L_def) * R_cr_S_ex
    dpsi_du2 = -(1.0 / L_def) * R_cr_S_ex

    T_inv1 = _tangent_operator_inv(theta_def1)
    T_inv2 = _tangent_operator_inv(theta_def2)
    T_s1 = _tangent_operator(u_elem[3:6])
    T_s2 = _tangent_operator(u_elem[9:12])

    B[3:6, 0:3] = T_inv1 @ dpsi_du1
    B[3:6, 6:9] = T_inv1 @ dpsi_du2
    B[3:6, 3:6] = T_inv1 @ R_cr @ T_s1
    B[9:12, 0:3] = T_inv2 @ dpsi_du1
    B[9:12, 6:9] = T_inv2 @ dpsi_du2
    B[9:12, 9:12] = T_inv2 @ R_cr @ T_s2

    T_cr = _transformation_matrix_3d(R_cr)
    K_mat = T_cr.T @ Ke_local @ B

    K_geo = np.zeros((12, 12), dtype=float)
    for blk in range(4):
        f_blk = f_cr[3 * blk : 3 * blk + 3]
        if np.linalg.norm(f_blk) < 1e-30:
            continue
        RtSf = R_cr.T @ _skew(f_blk)
        K_geo[3 * blk : 3 * blk + 3, 0:3] += RtSf @ dpsi_du1
        K_geo[3 * blk : 3 * blk + 3, 6:9] += RtSf @ dpsi_du2

    K_T = K_mat + K_geo
    return 0.5 * (K_T + K_T.T)


# =====================================================================
# バッチ（ベクトル化）版ユーティリティ
# =====================================================================


def _batch_skew(v: np.ndarray) -> np.ndarray:
    """バッチ歪対称行列 [v]×."""
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
    """バッチ回転ベクトル→回転行列（四元数経由）."""
    n = thetas.shape[0]
    angles = np.linalg.norm(thetas, axis=1)

    small = angles < 1e-12
    half = angles / 2.0
    half_sq = half * half

    sinc = np.empty(n, dtype=float)
    cos_half = np.empty(n, dtype=float)
    sinc[small] = 0.5 - half_sq[small] / 48.0
    cos_half[small] = 1.0 - half_sq[small] / 2.0
    big = ~small
    sinc[big] = np.sin(half[big]) / angles[big]
    cos_half[big] = np.cos(half[big])

    w = cos_half
    xyz = sinc[:, None] * thetas
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

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
    """バッチ回転行列→回転ベクトル（Shepperd法ベクトル化）."""
    n = Rs.shape[0]
    trace = Rs[:, 0, 0] + Rs[:, 1, 1] + Rs[:, 2, 2]

    diag_vals = np.stack([trace, Rs[:, 0, 0], Rs[:, 1, 1], Rs[:, 2, 2]], axis=1)
    best = np.argmax(diag_vals, axis=1)

    q = np.empty((n, 4), dtype=float)

    m0 = best == 0
    if np.any(m0):
        s = 2.0 * np.sqrt(trace[m0] + 1.0)
        q[m0, 0] = 0.25 * s
        q[m0, 1] = (Rs[m0, 2, 1] - Rs[m0, 1, 2]) / s
        q[m0, 2] = (Rs[m0, 0, 2] - Rs[m0, 2, 0]) / s
        q[m0, 3] = (Rs[m0, 1, 0] - Rs[m0, 0, 1]) / s

    m1 = best == 1
    if np.any(m1):
        s = 2.0 * np.sqrt(1.0 + Rs[m1, 0, 0] - Rs[m1, 1, 1] - Rs[m1, 2, 2])
        q[m1, 0] = (Rs[m1, 2, 1] - Rs[m1, 1, 2]) / s
        q[m1, 1] = 0.25 * s
        q[m1, 2] = (Rs[m1, 0, 1] + Rs[m1, 1, 0]) / s
        q[m1, 3] = (Rs[m1, 0, 2] + Rs[m1, 2, 0]) / s

    m2 = best == 2
    if np.any(m2):
        s = 2.0 * np.sqrt(1.0 + Rs[m2, 1, 1] - Rs[m2, 0, 0] - Rs[m2, 2, 2])
        q[m2, 0] = (Rs[m2, 0, 2] - Rs[m2, 2, 0]) / s
        q[m2, 1] = (Rs[m2, 0, 1] + Rs[m2, 1, 0]) / s
        q[m2, 2] = 0.25 * s
        q[m2, 3] = (Rs[m2, 1, 2] + Rs[m2, 2, 1]) / s

    m3 = best == 3
    if np.any(m3):
        s = 2.0 * np.sqrt(1.0 + Rs[m3, 2, 2] - Rs[m3, 0, 0] - Rs[m3, 1, 1])
        q[m3, 0] = (Rs[m3, 1, 0] - Rs[m3, 0, 1]) / s
        q[m3, 1] = (Rs[m3, 0, 2] + Rs[m3, 2, 0]) / s
        q[m3, 2] = (Rs[m3, 1, 2] + Rs[m3, 2, 1]) / s
        q[m3, 3] = 0.25 * s

    neg_w = q[:, 0] < 0
    q[neg_w] = -q[neg_w]

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    q = q / norms

    vec = q[:, 1:4]
    sin_half = np.linalg.norm(vec, axis=1)
    small = sin_half < 1e-12
    big = ~small

    coeff = np.empty(n, dtype=float)
    coeff[small] = 2.0
    coeff[big] = 2.0 * np.arctan2(sin_half[big], q[big, 0]) / sin_half[big]

    return coeff[:, None] * vec


def _batch_rodrigues_rotation(e_from: np.ndarray, e_to: np.ndarray) -> np.ndarray:
    """バッチ最小回転行列（Rodrigues公式）."""
    v = np.cross(e_from, e_to)
    s = np.linalg.norm(v, axis=1)
    c = np.sum(e_from * e_to, axis=1)

    small = s < 1e-14

    vx = _batch_skew(v)
    vx2 = np.einsum("nij,njk->nik", vx, vx)

    eye3 = np.eye(3, dtype=float)
    denom = s * s
    denom[small] = 1.0
    coeff = (1.0 - c) / denom
    coeff[small] = 0.0

    Rs = eye3[None, :, :] + vx + coeff[:, None, None] * vx2

    same_dir = small & (c > 0)
    Rs[same_dir] = eye3

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
    """バッチ局所座標系構築."""
    n = e_x.shape[0]

    if v_ref is not None:
        vr = np.tile(v_ref, (n, 1))
    else:
        abs_ex = np.abs(e_x)
        vr = np.zeros((n, 3), dtype=float)
        min_idx = np.argmin(abs_ex, axis=1)
        vr[np.arange(n), min_idx] = 1.0

    e_z = np.cross(e_x, vr)
    norm_ez = np.linalg.norm(e_z, axis=1, keepdims=True)
    norm_ez = np.maximum(norm_ez, 1e-10)
    e_z = e_z / norm_ez
    e_y = np.cross(e_z, e_x)

    Rs = np.empty((n, 3, 3), dtype=float)
    Rs[:, 0, :] = e_x
    Rs[:, 1, :] = e_y
    Rs[:, 2, :] = e_z
    return Rs


def _batch_tangent_operator(thetas: np.ndarray) -> np.ndarray:
    """バッチ接線演算子 T_s(θ)."""
    n = thetas.shape[0]
    angles = np.linalg.norm(thetas, axis=1)
    S = _batch_skew(thetas)
    S2 = np.einsum("nij,njk->nik", S, S)

    small = angles < 1e-10
    big = ~small

    eye3 = np.eye(3, dtype=float)
    Ts = np.tile(eye3, (n, 1, 1))

    if np.any(small):
        Ts[small] += 0.5 * S[small] + (1.0 / 6.0) * S2[small]

    if np.any(big):
        a = angles[big]
        c = np.cos(a)
        s_val = np.sin(a)
        a2 = a * a
        coeff1 = (1.0 - c) / a2
        coeff2 = (1.0 - s_val / a) / a2
        Ts[big] += coeff1[:, None, None] * S[big] + coeff2[:, None, None] * S2[big]

    return Ts


def _batch_tangent_operator_inv(thetas: np.ndarray) -> np.ndarray:
    """バッチ接線演算子の逆 T_s^{-1}(θ)."""
    n = thetas.shape[0]
    angles = np.linalg.norm(thetas, axis=1)
    S = _batch_skew(thetas)
    S2 = np.einsum("nij,njk->nik", S, S)

    small = angles < 1e-10
    big = ~small

    eye3 = np.eye(3, dtype=float)
    T_inv = np.tile(eye3, (n, 1, 1))

    if np.any(small):
        T_inv[small] += -0.5 * S[small] + (1.0 / 12.0) * S2[small]

    if np.any(big):
        a = angles[big]
        c = np.cos(a)
        s_val = np.sin(a)
        a2 = a * a
        coeff = 1.0 / a2 - (1.0 + c) / (2.0 * a * s_val)
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
    """バッチ局所剛性行列計算（L のみ要素依存）."""
    n = L.shape[0]

    Phi_y = 12.0 * E * Iy / (kappa_y * G * A * L * L)
    Phi_z = 12.0 * E * Iz / (kappa_z * G * A * L * L)

    if scf is not None and scf > 0.0:
        slenderness_y = L * L * A / (12.0 * Iy)
        f_p_y = 1.0 / (1.0 + scf * slenderness_y)
        Phi_y = Phi_y * f_p_y
        slenderness_z = L * L * A / (12.0 * Iz)
        f_p_z = 1.0 / (1.0 + scf * slenderness_z)
        Phi_z = Phi_z * f_p_z

    denom_y = 1.0 + Phi_y
    denom_z = 1.0 + Phi_z

    Ke = np.zeros((n, 12, 12), dtype=float)

    EA_L = E * A / L
    Ke[:, 0, 0] = EA_L
    Ke[:, 0, 6] = -EA_L
    Ke[:, 6, 0] = -EA_L
    Ke[:, 6, 6] = EA_L

    GJ_L = G * J / L
    Ke[:, 3, 3] = GJ_L
    Ke[:, 3, 9] = -GJ_L
    Ke[:, 9, 3] = -GJ_L
    Ke[:, 9, 9] = GJ_L

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
