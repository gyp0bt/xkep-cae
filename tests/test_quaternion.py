"""四元数ユーティリティのテスト."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.math.quaternion import (
    axial,
    quat_angular_velocity,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_rotvec,
    quat_identity,
    quat_material_curvature,
    quat_multiply,
    quat_norm,
    quat_normalize,
    quat_rotate_vector,
    quat_slerp,
    quat_to_rotation_matrix,
    quat_to_rotvec,
    rotation_matrix_to_quat,
    skew,
    so3_right_jacobian,
    so3_right_jacobian_inverse,
)


class TestQuatBasics:
    """四元数の基本演算テスト."""

    def test_identity(self):
        q = quat_identity()
        np.testing.assert_array_equal(q, [1.0, 0.0, 0.0, 0.0])

    def test_conjugate(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = quat_conjugate(q)
        np.testing.assert_array_almost_equal(q_conj, [0.5, -0.5, -0.5, -0.5])

    def test_norm(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(quat_norm(q) - np.sqrt(30.0)) < 1e-12

    def test_normalize(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_n = quat_normalize(q)
        assert abs(quat_norm(q_n) - 1.0) < 1e-12

    def test_normalize_zero_raises(self):
        with pytest.raises(ValueError, match="ゼロ"):
            quat_normalize(np.array([0.0, 0.0, 0.0, 0.0]))


class TestQuatMultiply:
    """Hamilton積のテスト."""

    def test_identity_left(self):
        """恒等四元数は左単位元."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = quat_multiply(quat_identity(), q)
        np.testing.assert_array_almost_equal(result, q)

    def test_identity_right(self):
        """恒等四元数は右単位元."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = quat_multiply(q, quat_identity())
        np.testing.assert_array_almost_equal(result, q)

    def test_inverse(self):
        """q ⊗ q* = 1 (単位四元数の場合)."""
        q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = quat_multiply(q, quat_conjugate(q))
        np.testing.assert_array_almost_equal(result, quat_identity(), decimal=12)

    def test_non_commutative(self):
        """Hamilton積は非可換."""
        p = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 4)
        q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 4)
        pq = quat_multiply(p, q)
        qp = quat_multiply(q, p)
        assert not np.allclose(pq, qp)

    def test_associative(self):
        """(p ⊗ q) ⊗ r = p ⊗ (q ⊗ r)."""
        p = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.3)
        q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5)
        r = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.7)
        lhs = quat_multiply(quat_multiply(p, q), r)
        rhs = quat_multiply(p, quat_multiply(q, r))
        np.testing.assert_array_almost_equal(lhs, rhs, decimal=12)


class TestQuatRotation:
    """四元数による回転テスト."""

    def test_identity_rotation(self):
        """恒等四元数はベクトルを変えない."""
        v = np.array([1.0, 2.0, 3.0])
        result = quat_rotate_vector(quat_identity(), v)
        np.testing.assert_array_almost_equal(result, v)

    def test_90deg_around_z(self):
        """z軸まわり90°回転: (1,0,0) → (0,1,0)."""
        q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        result = quat_rotate_vector(q, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0], decimal=12)

    def test_180deg_around_x(self):
        """x軸まわり180°回転: (0,1,0) → (0,-1,0)."""
        q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
        result = quat_rotate_vector(q, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [0.0, -1.0, 0.0], decimal=12)

    def test_arbitrary_rotation(self):
        """任意回転: 回転行列と一致するか確認."""
        q = quat_from_axis_angle(np.array([1.0, 1.0, 0.0]), 1.2)
        R = quat_to_rotation_matrix(q)
        v = np.array([3.0, -1.0, 2.0])
        result_quat = quat_rotate_vector(q, v)
        result_mat = R @ v
        np.testing.assert_array_almost_equal(result_quat, result_mat, decimal=12)


class TestQuatRotationMatrix:
    """四元数 ↔ 回転行列 の変換テスト."""

    def test_identity_to_matrix(self):
        R = quat_to_rotation_matrix(quat_identity())
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=12)

    def test_identity_from_matrix(self):
        q = rotation_matrix_to_quat(np.eye(3))
        np.testing.assert_array_almost_equal(q, quat_identity(), decimal=12)

    def test_roundtrip_random(self):
        """q → R → q のラウンドトリップ（3ケース）."""
        rng = np.random.default_rng(42)
        for _ in range(3):
            axis = rng.standard_normal(3)
            axis = axis / np.linalg.norm(axis)
            angle = rng.uniform(-np.pi, np.pi)
            q = quat_from_axis_angle(axis, angle)
            R = quat_to_rotation_matrix(q)
            q_back = rotation_matrix_to_quat(R)
            # q と -q は同じ回転
            if np.dot(q, q_back) < 0:
                q_back = -q_back
            np.testing.assert_array_almost_equal(q, q_back, decimal=10)

    def test_rotation_matrix_is_orthogonal(self):
        """R^T R = I, det(R) = 1 を確認."""
        q = quat_from_axis_angle(np.array([1.0, 2.0, 3.0]), 0.8)
        R = quat_to_rotation_matrix(q)
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_90deg_x_axis(self):
        """x軸90°回転の回転行列."""
        q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)
        R = quat_to_rotation_matrix(q)
        R_expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_array_almost_equal(R, R_expected, decimal=12)

    def test_shepperd_all_branches(self):
        """Shepperd法の全分岐をテスト.

        回転行列の対角成分の大小関係を変えて
        rotation_matrix_to_quat の全分岐を通る。
        """
        # trace > 0 (小角回転)
        q1 = quat_from_axis_angle(np.array([1.0, 1.0, 1.0]), 0.1)
        R1 = quat_to_rotation_matrix(q1)
        q1_back = rotation_matrix_to_quat(R1)
        np.testing.assert_array_almost_equal(q1, q1_back, decimal=10)

        # R[0,0] 最大 (x軸まわり大回転)
        q2 = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 2.5)
        R2 = quat_to_rotation_matrix(q2)
        q2_back = rotation_matrix_to_quat(R2)
        if np.dot(q2, q2_back) < 0:
            q2_back = -q2_back
        np.testing.assert_array_almost_equal(q2, q2_back, decimal=10)

        # R[1,1] 最大 (y軸まわり大回転)
        q3 = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 2.5)
        R3 = quat_to_rotation_matrix(q3)
        q3_back = rotation_matrix_to_quat(R3)
        if np.dot(q3, q3_back) < 0:
            q3_back = -q3_back
        np.testing.assert_array_almost_equal(q3, q3_back, decimal=10)

        # R[2,2] 最大 (z軸まわり大回転)
        q4 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 2.5)
        R4 = quat_to_rotation_matrix(q4)
        q4_back = rotation_matrix_to_quat(R4)
        if np.dot(q4, q4_back) < 0:
            q4_back = -q4_back
        np.testing.assert_array_almost_equal(q4, q4_back, decimal=10)


class TestQuatAxisAngle:
    """軸-角度 ↔ 四元数 の変換テスト."""

    def test_zero_angle(self):
        q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.0)
        np.testing.assert_array_almost_equal(q, quat_identity())

    def test_zero_axis(self):
        """ゼロ軸 → 恒等四元数."""
        q = quat_from_axis_angle(np.array([0.0, 0.0, 0.0]), 1.0)
        np.testing.assert_array_almost_equal(q, quat_identity())

    def test_90deg_z(self):
        q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_array_almost_equal(q, expected, decimal=12)


class TestQuatRotvec:
    """回転ベクトル ↔ 四元数 の変換テスト."""

    def test_zero_rotvec(self):
        q = quat_from_rotvec(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(q, quat_identity(), decimal=12)

    def test_small_rotvec(self):
        """微小回転: テイラー展開分岐のテスト."""
        rotvec = np.array([1e-14, 0.0, 0.0])
        q = quat_from_rotvec(rotvec)
        assert abs(quat_norm(q) - 1.0) < 1e-12

    def test_roundtrip(self):
        """rotvec → q → rotvec のラウンドトリップ."""
        rotvec = np.array([0.3, -0.5, 0.7])
        q = quat_from_rotvec(rotvec)
        rotvec_back = quat_to_rotvec(q)
        np.testing.assert_array_almost_equal(rotvec, rotvec_back, decimal=12)

    def test_roundtrip_large(self):
        """大きな回転角でのラウンドトリップ."""
        rotvec = np.array([2.0, 0.0, 0.0])
        q = quat_from_rotvec(rotvec)
        rotvec_back = quat_to_rotvec(q)
        np.testing.assert_array_almost_equal(rotvec, rotvec_back, decimal=10)

    def test_pi_rotation(self):
        """π回転: log/exp の特殊ケース."""
        rotvec = np.array([np.pi, 0.0, 0.0])
        q = quat_from_rotvec(rotvec)
        # cos(π/2) = 0, sin(π/2) = 1
        np.testing.assert_array_almost_equal(q, [0.0, 1.0, 0.0, 0.0], decimal=12)

    def test_consistency_with_axis_angle(self):
        """quat_from_rotvec は quat_from_axis_angle と一致する."""
        axis = np.array([0.0, 1.0, 0.0])
        angle = 1.5
        q1 = quat_from_axis_angle(axis, angle)
        q2 = quat_from_rotvec(axis * angle)
        np.testing.assert_array_almost_equal(q1, q2, decimal=12)


class TestQuatSlerp:
    """SLERP テスト."""

    def test_endpoints(self):
        q0 = quat_identity()
        q1 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        np.testing.assert_array_almost_equal(quat_slerp(q0, q1, 0.0), q0, decimal=12)
        np.testing.assert_array_almost_equal(quat_slerp(q0, q1, 1.0), q1, decimal=12)

    def test_midpoint(self):
        """中点は半分の回転."""
        q0 = quat_identity()
        q1 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        q_mid = quat_slerp(q0, q1, 0.5)
        q_expected = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 4)
        np.testing.assert_array_almost_equal(q_mid, q_expected, decimal=12)

    def test_nearly_identical(self):
        """ほぼ同一の四元数 → 線形補間フォールバック."""
        q0 = quat_identity()
        q1 = quat_from_rotvec(np.array([1e-6, 0.0, 0.0]))
        q_mid = quat_slerp(q0, q1, 0.5)
        assert abs(quat_norm(q_mid) - 1.0) < 1e-10


class TestQuatCurvature:
    """曲率・角速度テスト."""

    def test_zero_derivative(self):
        """微分ゼロ → 曲率ゼロ."""
        q = quat_identity()
        q_dot = np.array([0.0, 0.0, 0.0, 0.0])
        omega = quat_angular_velocity(q, q_dot)
        np.testing.assert_array_almost_equal(omega, [0.0, 0.0, 0.0])

    def test_constant_twist(self):
        """一様ねじり: κ₁ = twist rate.

        q(s) = [cos(τs/2), sin(τs/2), 0, 0] (x軸まわり)
        q'(s) = [-τ/2·sin(τs/2), τ/2·cos(τs/2), 0, 0]
        κ = 2·Im(q* ⊗ q') = [τ, 0, 0]
        """
        tau = 0.5  # twist rate
        s = 0.3   # arbitrary position

        q = np.array([np.cos(tau * s / 2), np.sin(tau * s / 2), 0.0, 0.0])
        q_prime = np.array([
            -tau / 2 * np.sin(tau * s / 2),
            tau / 2 * np.cos(tau * s / 2),
            0.0,
            0.0,
        ])
        kappa = quat_material_curvature(q, q_prime)
        np.testing.assert_array_almost_equal(kappa, [tau, 0.0, 0.0], decimal=12)


class TestSkewAxial:
    """skew/axial (hat/vee) 演算テスト."""

    def test_skew_cross_product(self):
        """skew(v) · u = v × u."""
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([4.0, 5.0, 6.0])
        S = skew(v)
        np.testing.assert_array_almost_equal(S @ u, np.cross(v, u))

    def test_axial_inverse(self):
        """axial は skew の逆."""
        v = np.array([1.0, -2.0, 3.0])
        S = skew(v)
        np.testing.assert_array_almost_equal(axial(S), v)

    def test_skew_antisymmetric(self):
        """skew(v) は歪対称."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        np.testing.assert_array_almost_equal(S + S.T, np.zeros((3, 3)))


class TestSO3Jacobian:
    """SO(3) 右ヤコビアン J_r(θ) のテスト."""

    def test_identity_at_zero(self):
        """J_r(0) = I."""
        Jr = so3_right_jacobian(np.zeros(3))
        np.testing.assert_array_almost_equal(Jr, np.eye(3))

    def test_inverse_at_zero(self):
        """J_r^{-1}(0) = I."""
        Jr_inv = so3_right_jacobian_inverse(np.zeros(3))
        np.testing.assert_array_almost_equal(Jr_inv, np.eye(3))

    def test_inverse_roundtrip(self):
        """J_r * J_r^{-1} = I for various rotation vectors."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            theta = rng.standard_normal(3) * 2.0  # up to ~2 rad
            Jr = so3_right_jacobian(theta)
            Jr_inv = so3_right_jacobian_inverse(theta)
            np.testing.assert_array_almost_equal(
                Jr @ Jr_inv, np.eye(3), decimal=10,
            )

    def test_taylor_branch_matches_exact(self):
        """小角度のテイラー展開が正確な公式と一致."""
        # 十分小さい角度でテイラー分岐に入る
        theta_small = np.array([1e-8, 2e-8, -1e-8])
        Jr_taylor = so3_right_jacobian(theta_small)

        # 少し大きい角度で正確な公式
        theta_exact = np.array([0.1, 0.2, -0.1])
        Jr_exact = so3_right_jacobian(theta_exact)
        Jr_inv_exact = so3_right_jacobian_inverse(theta_exact)

        # テイラー版は恒等行列に近い
        np.testing.assert_array_almost_equal(Jr_taylor, np.eye(3), decimal=6)
        # 正確な版は逆行列と整合
        np.testing.assert_array_almost_equal(
            Jr_exact @ Jr_inv_exact, np.eye(3), decimal=10,
        )

    def test_numerical_derivative(self):
        """J_r(θ)·δθ が回転ベクトル微小変化の角速度と一致."""
        theta = np.array([0.5, -0.3, 0.8])
        Jr = so3_right_jacobian(theta)

        eps = 1e-7
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = eps

            # R(theta + eps*e_i) と R(theta) から数値的な δR を計算
            q_plus = quat_from_rotvec(theta + delta)
            q_base = quat_from_rotvec(theta)
            R_plus = quat_to_rotation_matrix(q_plus)
            R_base = quat_to_rotation_matrix(q_base)

            # δR · R^T の軸ベクトル = 空間角速度の近似
            dR = (R_plus - R_base) / eps
            # 物質角速度: Ω = R^T · dR
            Omega = R_base.T @ dR
            omega_numerical = axial(Omega)

            # J_r(θ) · e_i
            omega_analytical = Jr[:, i]
            np.testing.assert_array_almost_equal(
                omega_analytical, omega_numerical, decimal=4,
            )

    def test_known_value_90deg_z(self):
        """z軸90度回転での既知の値."""
        theta = np.array([0.0, 0.0, np.pi / 2.0])
        Jr = so3_right_jacobian(theta)
        phi = np.pi / 2.0
        c1 = (1.0 - np.cos(phi)) / (phi * phi)
        c2 = (phi - np.sin(phi)) / (phi ** 3)
        S = skew(theta)
        Jr_expected = np.eye(3) - c1 * S + c2 * (S @ S)
        np.testing.assert_array_almost_equal(Jr, Jr_expected, decimal=12)
