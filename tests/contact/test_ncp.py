"""NCP (Nonlinear Complementarity Problem) 関数のテスト — Phase C6-L3α.

Fischer-Burmeister NCP 関数とその一般化微分の検証。

テスト構成:
- TestFischerBurmeister: FB 関数の基本プロパティ
- TestNCPMin: min 関数ベースの NCP
- TestNCPResidualJacobian: ベクトル版の残差・ヤコビアン
- TestGapJacobianWrtU: ∂g_n/∂u の解析値 vs 数値微分
- TestBuildAugmentedResidual: 拡大残差ベクトルの構造
"""

import math

import numpy as np

from xkep_cae.contact.ncp import (
    build_augmented_residual,
    compute_gap_jacobian_wrt_u,
    evaluate_ncp_jacobian,
    evaluate_ncp_residual,
    ncp_fischer_burmeister,
    ncp_min,
)


class TestFischerBurmeister:
    """Fischer-Burmeister NCP 関数の単体テスト."""

    def test_complementarity_satisfied(self):
        """g=0, lam=0 のとき FB ≈ 0 であること."""
        fb, dg, dlam = ncp_fischer_burmeister(0.0, 0.0, reg=1e-12)
        # reg が小さければ fb ≈ sqrt(reg) ≈ 1e-6
        assert abs(fb) < 1e-5

    def test_contact_active(self):
        """g=0, lam>0 は接触中（相補性を満たす）."""
        fb, dg, dlam = ncp_fischer_burmeister(0.0, 1.0, reg=1e-12)
        # FB(0, 1) = sqrt(0 + 1 + reg) - 0 - 1 = 1 - 1 = 0
        assert abs(fb) < 1e-6

    def test_contact_inactive(self):
        """g>0, lam=0 は離間（相補性を満たす）."""
        fb, dg, dlam = ncp_fischer_burmeister(1.0, 0.0, reg=1e-12)
        # FB(1, 0) = sqrt(1 + 0 + reg) - 1 - 0 = 1 - 1 = 0
        assert abs(fb) < 1e-6

    def test_penetration_violation(self):
        """g<0 のとき FB != 0（非貫入条件違反）."""
        fb, _, _ = ncp_fischer_burmeister(-0.1, 0.5, reg=1e-12)
        # g < 0 なので相補性は満たされない
        assert abs(fb) > 0.01

    def test_both_positive_violation(self):
        """g>0, lam>0 のとき FB < 0（g*lam > 0 の相補性違反）."""
        fb, _, _ = ncp_fischer_burmeister(1.0, 1.0, reg=1e-12)
        # FB(1,1) = sqrt(2) - 2 ≈ -0.586
        assert fb < 0

    def test_derivatives_numerical(self):
        """一般化微分が数値微分と一致すること."""
        g, lam = 0.3, 0.7
        _, dg, dlam = ncp_fischer_burmeister(g, lam, reg=1e-12)

        eps = 1e-7
        fb0, _, _ = ncp_fischer_burmeister(g, lam, reg=1e-12)

        fb_gp, _, _ = ncp_fischer_burmeister(g + eps, lam, reg=1e-12)
        dg_num = (fb_gp - fb0) / eps
        assert abs(dg - dg_num) < 1e-5, f"dg: {dg} vs {dg_num}"

        fb_lp, _, _ = ncp_fischer_burmeister(g, lam + eps, reg=1e-12)
        dlam_num = (fb_lp - fb0) / eps
        assert abs(dlam - dlam_num) < 1e-5, f"dlam: {dlam} vs {dlam_num}"

    def test_derivatives_at_origin(self):
        """原点付近でも安定した微分が得られること（正則化による）."""
        _, dg, dlam = ncp_fischer_burmeister(0.0, 0.0, reg=1e-12)
        # reg > 0 なので div-by-zero にならない
        assert math.isfinite(dg)
        assert math.isfinite(dlam)
        # dg = 0/sqrt(reg) - 1 ≈ -1, dlam 同様
        assert dg < 0
        assert dlam < 0

    def test_symmetry(self):
        """FB(g, lam) = FB(lam, g) であること（FB 関数は対称）."""
        g, lam = 0.5, 1.2
        fb1, dg1, dlam1 = ncp_fischer_burmeister(g, lam)
        fb2, dg2, dlam2 = ncp_fischer_burmeister(lam, g)
        assert abs(fb1 - fb2) < 1e-12
        assert abs(dg1 - dlam2) < 1e-12
        assert abs(dlam1 - dg2) < 1e-12


class TestNCPMin:
    """min 関数ベースの NCP のテスト."""

    def test_g_smaller(self):
        """g < lam のとき C = g."""
        c, dg, dlam = ncp_min(0.1, 0.5)
        assert abs(c - 0.1) < 1e-12
        assert dg == 1.0
        assert dlam == 0.0

    def test_lam_smaller(self):
        """lam < g のとき C = lam."""
        c, dg, dlam = ncp_min(0.5, 0.1)
        assert abs(c - 0.1) < 1e-12
        assert dg == 0.0
        assert dlam == 1.0

    def test_equal(self):
        """g == lam のとき一般化微分."""
        c, dg, dlam = ncp_min(0.3, 0.3)
        assert abs(c - 0.3) < 1e-12
        assert abs(dg - 0.5) < 1e-12
        assert abs(dlam - 0.5) < 1e-12

    def test_complementarity(self):
        """min(g, lam) = 0 は相補性条件と等価."""
        # g = 0, lam > 0 → min = 0 (接触)
        c1, _, _ = ncp_min(0.0, 1.0)
        assert abs(c1) < 1e-12

        # g > 0, lam = 0 → min = 0 (離間)
        c2, _, _ = ncp_min(1.0, 0.0)
        assert abs(c2) < 1e-12


class TestNCPResidualJacobian:
    """ベクトル版 NCP 残差・ヤコビアンのテスト."""

    def test_residual_shape(self):
        """残差ベクトルの形状."""
        gaps = np.array([0.1, -0.05, 0.0])
        lams = np.array([0.0, 0.3, 0.5])
        C = evaluate_ncp_residual(gaps, lams, ncp_type="fb")
        assert C.shape == (3,)

    def test_jacobian_shape(self):
        """ヤコビアン（対角）の形状."""
        gaps = np.array([0.1, -0.05, 0.0])
        lams = np.array([0.0, 0.3, 0.5])
        dC_dg, dC_dlam = evaluate_ncp_jacobian(gaps, lams, ncp_type="fb")
        assert dC_dg.shape == (3,)
        assert dC_dlam.shape == (3,)

    def test_residual_consistency(self):
        """ベクトル版がスカラー版と一致すること."""
        gaps = np.array([0.1, 0.5, -0.2])
        lams = np.array([0.3, 0.0, 0.8])
        C = evaluate_ncp_residual(gaps, lams, ncp_type="fb")
        for i in range(3):
            fb, _, _ = ncp_fischer_burmeister(gaps[i], lams[i])
            assert abs(C[i] - fb) < 1e-12

    def test_jacobian_consistency(self):
        """ヤコビアンがスカラー版と一致すること."""
        gaps = np.array([0.1, 0.5, -0.2])
        lams = np.array([0.3, 0.0, 0.8])
        dC_dg, dC_dlam = evaluate_ncp_jacobian(gaps, lams, ncp_type="fb")
        for i in range(3):
            _, dg, dl = ncp_fischer_burmeister(gaps[i], lams[i])
            assert abs(dC_dg[i] - dg) < 1e-12
            assert abs(dC_dlam[i] - dl) < 1e-12

    def test_min_type(self):
        """ncp_type='min' でも動作すること."""
        gaps = np.array([0.1, 0.5])
        lams = np.array([0.3, 0.2])
        C = evaluate_ncp_residual(gaps, lams, ncp_type="min")
        assert C.shape == (2,)
        assert abs(C[0] - 0.1) < 1e-12  # min(0.1, 0.3) = 0.1
        assert abs(C[1] - 0.2) < 1e-12  # min(0.5, 0.2) = 0.2


class TestGapJacobianWrtU:
    """∂g_n/∂u の解析値 vs 数値微分."""

    def test_shape(self):
        """出力の形状確認."""
        nodes_a = np.array([0, 1])
        nodes_b = np.array([2, 3])
        normal = np.array([0.0, 1.0, 0.0])
        dg_du = compute_gap_jacobian_wrt_u(nodes_a, nodes_b, 0.5, 0.5, normal)
        assert dg_du.shape == (24,)  # 4 nodes * 6 DOF

    def test_numerical_differentiation(self):
        """数値微分との一致を検証."""
        # 4節点の座標を設定
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # A0
                [1.0, 0.0, 0.0],  # A1
                [0.5, 0.5, 0.0],  # B0
                [0.5, 1.5, 0.0],  # B1
            ]
        )
        s, t = 0.4, 0.3
        ndof_per_node = 6
        ndof_total = 4 * ndof_per_node

        def compute_gap(u):
            """変位 u から gap を計算."""
            coords = node_coords.copy()
            for i in range(4):
                coords[i, 0] += u[i * ndof_per_node + 0]
                coords[i, 1] += u[i * ndof_per_node + 1]
                coords[i, 2] += u[i * ndof_per_node + 2]
            pA = (1 - s) * coords[0] + s * coords[1]
            pB = (1 - t) * coords[2] + t * coords[3]
            d = pA - pB
            dist = np.linalg.norm(d)
            return dist  # gap = dist - (rA + rB)、半径は定数なので微分には無関係

        u0 = np.zeros(ndof_total)
        dist0 = compute_gap(u0)
        pA = (1 - s) * node_coords[0] + s * node_coords[1]
        pB = (1 - t) * node_coords[2] + t * node_coords[3]
        d = pA - pB
        normal = d / np.linalg.norm(d)

        nodes_a = np.array([0, 1])
        nodes_b = np.array([2, 3])
        dg_du = compute_gap_jacobian_wrt_u(nodes_a, nodes_b, s, t, normal, ndof_per_node)

        # 数値微分
        eps = 1e-7
        dg_du_num = np.zeros(ndof_total)
        for j in range(ndof_total):
            u_p = u0.copy()
            u_p[j] += eps
            dg_du_num[j] = (compute_gap(u_p) - dist0) / eps

        np.testing.assert_allclose(dg_du, dg_du_num, atol=1e-5)

    def test_rotation_dofs_zero(self):
        """回転DOFへの微分はゼロであること."""
        nodes_a = np.array([0, 1])
        nodes_b = np.array([2, 3])
        normal = np.array([0.0, 0.0, 1.0])
        dg_du = compute_gap_jacobian_wrt_u(nodes_a, nodes_b, 0.5, 0.5, normal, ndof_per_node=6)

        # 回転DOF（各節点の index 3,4,5）はゼロ
        for node in range(4):
            for dof in range(3, 6):
                assert dg_du[node * 6 + dof] == 0.0


class TestBuildAugmentedResidual:
    """拡大残差ベクトルの構造テスト."""

    def test_shape(self):
        """拡大残差の形状 = ndof + n_contact."""
        R_u = np.zeros(18)
        gaps = np.array([0.1, -0.05])
        lams = np.array([0.0, 0.3])
        res = build_augmented_residual(R_u, gaps, lams)
        assert res.shape == (20,)  # 18 + 2

    def test_first_part_matches_R_u(self):
        """最初の ndof 成分は R_u と一致."""
        R_u = np.array([1.0, 2.0, 3.0])
        gaps = np.array([0.5])
        lams = np.array([0.0])
        res = build_augmented_residual(R_u, gaps, lams)
        np.testing.assert_array_equal(res[:3], R_u)

    def test_second_part_is_ncp(self):
        """後半の n_contact 成分は NCP 残差."""
        R_u = np.array([1.0, 2.0])
        gaps = np.array([0.5, 0.0])
        lams = np.array([0.0, 1.0])
        res = build_augmented_residual(R_u, gaps, lams)
        C = evaluate_ncp_residual(gaps, lams)
        np.testing.assert_allclose(res[2:], C)

    def test_satisfied_ncp_zero(self):
        """相補性を満たすペアは NCP 残差 ≈ 0."""
        R_u = np.zeros(3)
        # g > 0, lam = 0（離間）と g = 0, lam > 0（接触）
        gaps = np.array([1.0, 0.0])
        lams = np.array([0.0, 0.5])
        res = build_augmented_residual(R_u, gaps, lams)
        assert abs(res[3]) < 1e-5  # 離間ペア
        assert abs(res[4]) < 1e-5  # 接触ペア
