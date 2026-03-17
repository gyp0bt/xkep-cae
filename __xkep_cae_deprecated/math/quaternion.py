"""四元数（クォータニオン）演算モジュール.

Cosserat rod の回転表現に使用する四元数演算を提供する。

規約:
  q = [w, x, y, z] = w + x·i + y·j + z·k
  - w: スカラー部（実部）
  - (x, y, z): ベクトル部（虚部）
  - 単位四元数: ||q|| = 1 が回転を表す

回転との対応:
  - 回転軸 n（単位ベクトル）、回転角 θ のとき:
    q = [cos(θ/2), sin(θ/2)·n]
  - 回転行列 R(q) による変換:
    v' = q ⊗ (0,v) ⊗ q* = R(q)·v

Phase 2.5 (Cosserat rod) および Phase 3 (幾何学的非線形) の基盤。
"""

from __future__ import annotations

import numpy as np


def quat_identity() -> np.ndarray:
    """恒等四元数 [1, 0, 0, 0] を返す."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton積 p ⊗ q を計算する.

    Args:
        p: (4,) 四元数 [pw, px, py, pz]
        q: (4,) 四元数 [qw, qx, qy, qz]

    Returns:
        r: (4,) Hamilton積 p ⊗ q
    """
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数の共役 q* = [w, -x, -y, -z] を返す.

    単位四元数では q* = q⁻¹ （逆回転）。

    Args:
        q: (4,) 四元数

    Returns:
        q_conj: (4,) 共役四元数
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_norm(q: np.ndarray) -> float:
    """四元数のノルム ||q|| を返す.

    Args:
        q: (4,) 四元数

    Returns:
        ノルム
    """
    return float(np.linalg.norm(q))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """単位四元数に正規化する.

    Args:
        q: (4,) 四元数

    Returns:
        q_unit: (4,) 単位四元数 ||q|| = 1

    Raises:
        ValueError: ノルムがほぼゼロの場合
    """
    n = np.linalg.norm(q)
    if n < 1e-15:
        raise ValueError("四元数のノルムがほぼゼロです。正規化できません。")
    return q / n


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """四元数 q でベクトル v を回転する.

    v' = q ⊗ (0, v) ⊗ q* = R(q) · v

    効率的な計算（Hamilton積を展開した形式）を使用。

    Args:
        q: (4,) 単位四元数
        v: (3,) 3Dベクトル

    Returns:
        v_rot: (3,) 回転後のベクトル
    """
    w, x, y, z = q
    vx, vy, vz = v

    # t = 2 * (q_vec × v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # v' = v + w * t + q_vec × t
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ]
    )


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """単位四元数を 3x3 回転行列に変換する.

    R(q) の各成分を四元数の成分から直接計算。

    Args:
        q: (4,) 単位四元数

    Returns:
        R: (3, 3) 回転行列（SO(3) の元）
    """
    w, x, y, z = q

    # 対角成分
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )
    return R


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 回転行列を単位四元数に変換する.

    Shepperd法を使用（数値的に安定）。
    対角成分の中で最大の成分を選び、それを基準に計算する。

    Args:
        R: (3, 3) 回転行列

    Returns:
        q: (4,) 単位四元数（w >= 0 に正規化）
    """
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
    # w >= 0 に正規化（q と -q は同じ回転）
    if q[0] < 0:
        q = -q
    return quat_normalize(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """回転軸と回転角から四元数を生成する.

    q = [cos(θ/2), sin(θ/2)·n]

    Args:
        axis: (3,) 回転軸（単位ベクトルでなくてもよい、正規化される）
        angle: 回転角（ラジアン）

    Returns:
        q: (4,) 単位四元数
    """
    n = np.linalg.norm(axis)
    if n < 1e-15:
        return quat_identity()
    axis_unit = axis / n
    half = angle / 2.0
    return np.array(
        [
            np.cos(half),
            np.sin(half) * axis_unit[0],
            np.sin(half) * axis_unit[1],
            np.sin(half) * axis_unit[2],
        ]
    )


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """回転ベクトルから四元数を生成する（指数写像）.

    回転ベクトル θ = angle · axis から:
      q = exp(θ/2) = [cos(|θ|/2), sin(|θ|/2)·θ/|θ|]

    |θ| → 0 の場合はテイラー展開で数値安定性を確保。

    Args:
        rotvec: (3,) 回転ベクトル（方向 = 回転軸、大きさ = 回転角）

    Returns:
        q: (4,) 単位四元数
    """
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        # テイラー展開: cos(θ/2) ≈ 1 - θ²/8, sin(θ/2)/θ ≈ 1/2 - θ²/48
        half_angle_sq = angle * angle / 4.0
        w = 1.0 - half_angle_sq / 2.0
        coeff = 0.5 - half_angle_sq / 48.0
        return quat_normalize(
            np.array(
                [
                    w,
                    coeff * rotvec[0],
                    coeff * rotvec[1],
                    coeff * rotvec[2],
                ]
            )
        )
    half = angle / 2.0
    sinc = np.sin(half) / angle
    return np.array(
        [
            np.cos(half),
            sinc * rotvec[0],
            sinc * rotvec[1],
            sinc * rotvec[2],
        ]
    )


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """四元数から回転ベクトルを抽出する（対数写像）.

    q = [cos(θ/2), sin(θ/2)·n] → θ = 2·arctan2(|q_vec|, w)·n

    |q_vec| → 0 の場合はテイラー展開で数値安定性を確保。

    Args:
        q: (4,) 単位四元数

    Returns:
        rotvec: (3,) 回転ベクトル
    """
    # w >= 0 に正規化
    if q[0] < 0:
        q = -q

    vec = q[1:4]
    sin_half = float(np.linalg.norm(vec))

    if sin_half < 1e-12:
        # テイラー展開: θ/sin(θ/2) ≈ 2 + θ²/6
        return 2.0 * vec

    angle = 2.0 * np.arctan2(sin_half, q[0])
    return (angle / sin_half) * vec


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """球面線形補間 (SLERP).

    q(t) = q0 · (q0⁻¹ · q1)^t, t ∈ [0, 1]

    Args:
        q0: (4,) 始点の単位四元数
        q1: (4,) 終点の単位四元数
        t: 補間パラメータ [0, 1]

    Returns:
        q: (4,) 補間された単位四元数
    """
    dot = float(np.dot(q0, q1))

    # q と -q は同じ回転 → 短弧を選択
    if dot < 0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # ほぼ同一 → 線形補間
        return quat_normalize((1.0 - t) * q0 + t * q1)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def quat_angular_velocity(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """四元数の時間微分から角速度ベクトルを計算する（body frame）.

    body frame の角速度: ω_body = 2 · q* ⊗ q̇ の虚部

    Args:
        q: (4,) 現在の単位四元数
        q_dot: (4,) 四元数の時間微分

    Returns:
        omega: (3,) body frame の角速度ベクトル
    """
    product = quat_multiply(quat_conjugate(q), q_dot)
    return 2.0 * product[1:4]


def quat_material_curvature(q: np.ndarray, q_prime: np.ndarray) -> np.ndarray:
    """四元数の空間微分から物質曲率ベクトルを計算する.

    物質曲率（body frame）: κ = 2 · q* ⊗ q' の虚部
    = axial(Rᵀ R')

    Cosserat rod の曲率-ねじり歪みベクトルに対応。

    Args:
        q: (4,) 現在の単位四元数
        q_prime: (4,) 四元数の弧長パラメータ s に関する微分

    Returns:
        kappa: (3,) 物質曲率ベクトル [κ₁, κ₂, κ₃]
            κ₁: ねじり（twist）
            κ₂: 曲率 about y（xz面曲げ）
            κ₃: 曲率 about z（xy面曲げ）
    """
    return quat_angular_velocity(q, q_prime)


def skew(v: np.ndarray) -> np.ndarray:
    """ベクトルの歪対称行列（hat map）.

    skew(v) · u = v × u

    Args:
        v: (3,) ベクトル

    Returns:
        S: (3, 3) 歪対称行列
    """
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def axial(S: np.ndarray) -> np.ndarray:
    """歪対称行列のaxialベクトル（vee map）.

    skew(v) の逆演算: S → v

    Args:
        S: (3, 3) 歪対称行列

    Returns:
        v: (3,) axialベクトル
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def so3_right_jacobian(rotvec: np.ndarray) -> np.ndarray:
    """SO(3) の右ヤコビアン J_r(θ) を計算する.

    回転ベクトル θ に対する指数写像の右微分。
    Cosserat rod の曲率歪み κ = J_r(θ) · θ' に使用。

    J_r(θ) = I - c₁·S + c₂·S²
      S = skew(θ), φ = |θ|
      c₁ = (1 - cos φ) / φ²
      c₂ = (φ - sin φ) / φ³

    |θ| → 0 のテイラー展開:
      J_r ≈ I - (1/2)·S + (1/6)·S²

    Args:
        rotvec: (3,) 回転ベクトル

    Returns:
        Jr: (3, 3) 右ヤコビアン行列
    """
    S = skew(rotvec)
    phi = float(np.linalg.norm(rotvec))
    if phi < 1e-6:
        # テイラー展開: c1 = 1/2 - phi^2/24, c2 = 1/6 - phi^2/120
        phi2 = phi * phi
        c1 = 0.5 - phi2 / 24.0
        c2 = 1.0 / 6.0 - phi2 / 120.0
    else:
        c1 = (1.0 - np.cos(phi)) / (phi * phi)
        c2 = (phi - np.sin(phi)) / (phi * phi * phi)
    return np.eye(3) - c1 * S + c2 * (S @ S)


def so3_right_jacobian_inverse(rotvec: np.ndarray) -> np.ndarray:
    """SO(3) の右ヤコビアン逆行列 J_r⁻¹(θ) を計算する.

    J_r⁻¹(θ) = I + (1/2)·S + γ·S²
      S = skew(θ), φ = |θ|
      γ = 1/φ² - (1 + cos φ) / (2φ sin φ)

    |θ| → 0 のテイラー展開:
      J_r⁻¹ ≈ I + (1/2)·S + (1/12)·S²

    Args:
        rotvec: (3,) 回転ベクトル

    Returns:
        Jr_inv: (3, 3) 右ヤコビアン逆行列
    """
    S = skew(rotvec)
    phi = float(np.linalg.norm(rotvec))
    if phi < 1e-6:
        # テイラー展開: gamma = 1/12 + phi^2/720
        phi2 = phi * phi
        gamma = 1.0 / 12.0 + phi2 / 720.0
    else:
        gamma = 1.0 / (phi * phi) - (1.0 + np.cos(phi)) / (2.0 * phi * np.sin(phi))
    return np.eye(3) + 0.5 * S + gamma * (S @ S)
