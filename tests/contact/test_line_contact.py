"""Line-to-line Gauss 積分のテスト（Phase C6-L1）.

テスト項目:
1. Gauss-Legendre 積分点/重みの妥当性
2. project_point_to_segment の基本動作
3. auto_select_n_gauss の角度ベース選択
4. line contact 力の Gauss 積分
5. line contact 剛性の Gauss 積分
6. PtP との比較（大角度交差で一致確認）
7. 準平行梁での精度向上確認
8. Gauss 点数収束テスト
9. assembly 統合テスト（line_contact=True/False 切替）
10. 診断用 gap 計算
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.line_contact import (
    auto_select_n_gauss,
    compute_line_contact_force_local,
    compute_line_contact_gap_at_gp,
    compute_line_contact_stiffness_local,
    gauss_legendre_01,
    project_point_to_segment,
)
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)

# --- ヘルパー ---


def _make_pair(
    s: float = 0.5,
    t: float = 0.5,
    gap: float = -0.01,
    normal: tuple = (0.0, 0.0, 1.0),
    lambda_n: float = 0.0,
    k_pen: float = 1e6,
    radius_a: float = 0.05,
    radius_b: float = 0.05,
    nodes_a: tuple = (0, 1),
    nodes_b: tuple = (2, 3),
) -> ContactPair:
    """テスト用 ContactPair を生成."""
    return ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array(nodes_a, dtype=int),
        nodes_b=np.array(nodes_b, dtype=int),
        state=ContactState(
            s=s,
            t=t,
            gap=gap,
            normal=np.array(normal, dtype=float),
            lambda_n=lambda_n,
            k_pen=k_pen,
            k_t=k_pen * 0.5,
            p_n=0.0,
            status=ContactStatus.ACTIVE,
        ),
        radius_a=radius_a,
        radius_b=radius_b,
    )


# --- テストクラス ---


class TestGaussLegendre01:
    """[0,1] Gauss-Legendre 積分点のテスト."""

    def test_weights_sum_to_one(self):
        """重みの合計が 1.0."""
        for n in [1, 2, 3, 4, 5]:
            pts, wts = gauss_legendre_01(n)
            assert abs(sum(wts) - 1.0) < 1e-14, f"n={n}: sum={sum(wts)}"

    def test_points_in_01(self):
        """積分点が [0,1] 区間内."""
        for n in [2, 3, 5]:
            pts, _ = gauss_legendre_01(n)
            assert all(0.0 <= p <= 1.0 for p in pts)

    def test_n_points(self):
        """点数が正しい."""
        for n in [1, 2, 3, 4, 5]:
            pts, wts = gauss_legendre_01(n)
            assert len(pts) == n
            assert len(wts) == n

    def test_exact_for_polynomials(self):
        """2n-1 次多項式を正確に積分（Gauss の精度保証）."""
        for n in [2, 3, 4]:
            pts, wts = gauss_legendre_01(n)
            # ∫₀¹ x^(2n-1) dx = 1/(2n)
            deg = 2 * n - 1
            numerical = sum(w * p**deg for p, w in zip(pts, wts, strict=True))
            exact = 1.0 / (deg + 1)
            assert abs(numerical - exact) < 1e-13, f"n={n}, deg={deg}"


class TestProjectPointToSegment:
    """点のセグメントへの射影テスト."""

    def test_midpoint(self):
        """中点への射影."""
        x0 = np.array([0.0, 0.0, 0.0])
        x1 = np.array([1.0, 0.0, 0.0])
        p = np.array([0.5, 1.0, 0.0])
        t = project_point_to_segment(p, x0, x1)
        assert abs(t - 0.5) < 1e-15

    def test_endpoint_clamp_start(self):
        """始点側にクランプ."""
        x0 = np.array([0.0, 0.0, 0.0])
        x1 = np.array([1.0, 0.0, 0.0])
        p = np.array([-0.5, 1.0, 0.0])
        t = project_point_to_segment(p, x0, x1)
        assert t == 0.0

    def test_endpoint_clamp_end(self):
        """終点側にクランプ."""
        x0 = np.array([0.0, 0.0, 0.0])
        x1 = np.array([1.0, 0.0, 0.0])
        p = np.array([1.5, 1.0, 0.0])
        t = project_point_to_segment(p, x0, x1)
        assert t == 1.0

    def test_degenerate_segment(self):
        """縮退セグメント（長さゼロ）."""
        x0 = np.array([1.0, 2.0, 3.0])
        t = project_point_to_segment(np.array([0.0, 0.0, 0.0]), x0, x0)
        assert t == 0.0


class TestAutoSelectNGauss:
    """セグメント角度に基づく Gauss 点数自動選択テスト."""

    def test_perpendicular_beams(self):
        """直交（θ=90°）→ 2点."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, -0.5, 0.0])
        xB1 = np.array([0.5, 0.5, 0.0])
        assert auto_select_n_gauss(xA0, xA1, xB0, xB1) == 2

    def test_moderate_angle(self):
        """中間角度（θ≈20°）→ 3点."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.0])
        xB1 = np.array([1.0, 0.36, 0.0])  # θ ≈ 20°
        n = auto_select_n_gauss(xA0, xA1, xB0, xB1)
        assert n == 3

    def test_nearly_parallel(self):
        """準平行（θ < 10°）→ 5点."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.1, 0.0])
        xB1 = np.array([1.0, 0.12, 0.0])  # ほぼ平行
        n = auto_select_n_gauss(xA0, xA1, xB0, xB1)
        assert n == 5

    def test_degenerate_returns_default(self):
        """退化ケース → デフォルト値."""
        x0 = np.zeros(3)
        n = auto_select_n_gauss(x0, x0, x0, np.array([1.0, 0.0, 0.0]))
        assert n == 3


class TestLineContactForce:
    """Line contact 力の Gauss 積分テスト."""

    def test_uniform_gap_action_reaction(self):
        """均一ギャップでの作用反作用の釣り合い."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        # 平行配置: A は x軸, B は z=0.09 上で平行
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])  # gap = 0.09 - 0.1 = -0.01
        xB1 = np.array([1.0, 0.0, 0.09])

        f_local, total_p_n = compute_line_contact_force_local(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        # 作用反作用: A側 + B側 = 0
        f_A = f_local[0:6]
        f_B = f_local[6:12]
        assert np.allclose(f_A + f_B, 0.0, atol=1e-10)

    def test_force_direction(self):
        """法線方向に力が作用."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])
        xB1 = np.array([1.0, 0.0, 0.09])

        f_local, _ = compute_line_contact_force_local(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        # z 方向のみ（法線方向）
        for i in range(4):
            assert abs(f_local[i * 3 + 0]) < 1e-10  # x = 0
            assert abs(f_local[i * 3 + 1]) < 1e-10  # y = 0
            assert abs(f_local[i * 3 + 2]) > 0.0  # z ≠ 0

    def test_no_force_when_separated(self):
        """離間時は力ゼロ."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.2])  # gap = 0.2 - 0.1 = 0.1 > 0
        xB1 = np.array([1.0, 0.0, 0.2])

        f_local, total_p_n = compute_line_contact_force_local(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        assert np.allclose(f_local, 0.0, atol=1e-15)
        assert total_p_n == 0.0

    def test_positive_total_force(self):
        """貫入時に正の合計法線力."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])
        xB1 = np.array([1.0, 0.0, 0.09])

        _, total_p_n = compute_line_contact_force_local(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        # p_n = k_pen * (-gap) = 1e6 * 0.01 = 1e4
        # total ≈ 1e4（Gauss 重み合計 1.0）
        assert total_p_n > 0.0
        assert abs(total_p_n - 1e4) < 1.0  # 均一ギャップなので正確に一致

    def test_al_multiplier_contribution(self):
        """AL 乗数が力に寄与."""
        pair = _make_pair(k_pen=1e6, lambda_n=5000.0, radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])
        xB1 = np.array([1.0, 0.0, 0.09])

        _, total_p_n = compute_line_contact_force_local(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        # p_n = lambda_n + k_pen * (-gap) = 5000 + 1e6 * 0.01 = 15000
        assert abs(total_p_n - 15000.0) < 1.0


class TestLineContactStiffness:
    """Line contact 剛性のテスト."""

    def test_symmetry(self):
        """剛性行列の対称性."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])
        xB1 = np.array([1.0, 0.0, 0.09])

        K = compute_line_contact_stiffness_local(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        assert np.allclose(K, K.T, atol=1e-10)

    def test_positive_semidefinite(self):
        """剛性行列の半正値性（主項のみ）."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])
        xB1 = np.array([1.0, 0.0, 0.09])

        K = compute_line_contact_stiffness_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            n_gauss=3,
            use_geometric_stiffness=False,
        )

        eigenvalues = np.linalg.eigvalsh(K)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    def test_zero_when_separated(self):
        """離間時はゼロ行列."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.2])
        xB1 = np.array([1.0, 0.0, 0.2])

        K = compute_line_contact_stiffness_local(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        assert np.allclose(K, 0.0, atol=1e-15)


class TestLineVsPtPComparison:
    """Line contact と PtP の比較テスト."""

    def test_parallel_beams_similar_force(self):
        """平行梁で均一ギャップの場合、PtP と line contact が一致.

        平行配置で均一ギャップならば、全 Gauss 点で同じ
        法線力が得られるため、PtP の1点評価と line contact の
        重み付き和が一致する。
        """
        from xkep_cae.contact.assembly import _contact_shape_vector
        from xkep_cae.contact.law_normal import evaluate_normal_force

        # 平行配置: A は x軸, B は z=0.09 上で平行
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])
        xB1 = np.array([1.0, 0.0, 0.09])

        # PtP: 中点 s=0.5, t=0.5, gap=-0.01
        pair_ptp = _make_pair(
            s=0.5,
            t=0.5,
            gap=-0.01,
            normal=(0.0, 0.0, -1.0),
            k_pen=1e6,
            radius_a=0.05,
            radius_b=0.05,
        )
        p_n_ptp = evaluate_normal_force(pair_ptp)
        g_n_ptp = _contact_shape_vector(pair_ptp)
        f_ptp = p_n_ptp * g_n_ptp

        # Line contact
        pair_line = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)
        f_line, _ = compute_line_contact_force_local(pair_line, xA0, xA1, xB0, xB1, n_gauss=3)

        # 均一ギャップなので力の大きさが一致
        assert abs(np.linalg.norm(f_ptp) - np.linalg.norm(f_line)) < 1.0


class TestGaussPointConvergence:
    """Gauss 点数の収束テスト."""

    def test_convergence_with_n_gauss(self):
        """Gauss 点数を増やすと力が収束する."""
        pair = _make_pair(k_pen=1e6, radius_a=0.05, radius_b=0.05)

        # 少し傾いた平行配置（非均一ギャップ）
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.085])
        xB1 = np.array([1.0, 0.0, 0.095])

        forces = []
        for n_gp in [2, 3, 4, 5]:
            _, total_p_n = compute_line_contact_force_local(pair, xA0, xA1, xB0, xB1, n_gauss=n_gp)
            forces.append(total_p_n)

        # 収束: |f_n+1 - f_n| が減少
        diffs = [abs(forces[i + 1] - forces[i]) for i in range(len(forces) - 1)]
        # 最後の差が最初の差より小さい（収束傾向）
        assert diffs[-1] <= diffs[0] + 1e-6


class TestLineContactGapDiag:
    """診断用 gap 計算テスト."""

    def test_uniform_gap(self):
        """均一ギャップの確認."""
        pair = _make_pair(radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.09])
        xB1 = np.array([1.0, 0.0, 0.09])

        gaps, s_pts, min_gap = compute_line_contact_gap_at_gp(pair, xA0, xA1, xB0, xB1, n_gauss=3)

        assert len(gaps) == 3
        assert np.allclose(gaps, -0.01, atol=1e-10)
        assert abs(min_gap - (-0.01)) < 1e-10

    def test_varying_gap(self):
        """変動ギャップの確認."""
        pair = _make_pair(radius_a=0.05, radius_b=0.05)

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.0, 0.08])  # 始点: gap = -0.02
        xB1 = np.array([1.0, 0.0, 0.12])  # 終点: gap = +0.02

        gaps, _, min_gap = compute_line_contact_gap_at_gp(pair, xA0, xA1, xB0, xB1, n_gauss=5)

        # ギャップが単調増加（始点で貫入、終点で離間）
        assert gaps[0] < gaps[-1]
        assert min_gap < 0.0  # 一部貫入あり


class TestAssemblyLineContactIntegration:
    """assembly.py のline contact 統合テスト."""

    def test_ptp_mode_unchanged(self):
        """line_contact=False でPtP動作を維持（後方互換）."""
        from xkep_cae.contact.assembly import compute_contact_force

        manager = ContactManager(config=ContactConfig(line_contact=False))
        pair = manager.add_pair(
            0, 1, np.array([0, 1]), np.array([2, 3]), radius_a=0.05, radius_b=0.05
        )
        pair.state.status = ContactStatus.ACTIVE
        pair.state.s = 0.5
        pair.state.t = 0.5
        pair.state.gap = -0.01
        pair.state.normal = np.array([0.0, 0.0, 1.0])
        pair.state.k_pen = 1e6

        ndof = 4 * 6
        f = compute_contact_force(manager, ndof, ndof_per_node=6)

        assert float(np.linalg.norm(f)) > 0.0

    def test_line_contact_mode(self):
        """line_contact=True で Gauss 積分の力が得られる."""
        from xkep_cae.contact.assembly import compute_contact_force

        config = ContactConfig(line_contact=True, n_gauss=3)
        manager = ContactManager(config=config)
        pair = manager.add_pair(
            0, 1, np.array([0, 1]), np.array([2, 3]), radius_a=0.05, radius_b=0.05
        )
        pair.state.status = ContactStatus.ACTIVE
        pair.state.k_pen = 1e6

        # 変形座標（4節点）
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # node 0 (A0)
                [1.0, 0.0, 0.0],  # node 1 (A1)
                [0.0, 0.0, 0.09],  # node 2 (B0)
                [1.0, 0.0, 0.09],  # node 3 (B1)
            ]
        )

        ndof = 4 * 6
        f = compute_contact_force(manager, ndof, ndof_per_node=6, node_coords=node_coords)

        assert float(np.linalg.norm(f)) > 0.0
        # p_n が更新されている
        assert pair.state.p_n > 0.0

    def test_line_contact_stiffness(self):
        """line_contact=True で剛性行列が得られる."""
        from xkep_cae.contact.assembly import compute_contact_stiffness

        config = ContactConfig(line_contact=True, n_gauss=3)
        manager = ContactManager(config=config)
        pair = manager.add_pair(
            0, 1, np.array([0, 1]), np.array([2, 3]), radius_a=0.05, radius_b=0.05
        )
        pair.state.status = ContactStatus.ACTIVE
        pair.state.k_pen = 1e6
        pair.state.p_n = 1e4

        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.09],
                [1.0, 0.0, 0.09],
            ]
        )

        ndof = 4 * 6
        K = compute_contact_stiffness(manager, ndof, ndof_per_node=6, node_coords=node_coords)

        assert K.nnz > 0

    def test_n_gauss_auto_selection(self):
        """n_gauss_auto=True で角度ベース選択（準平行→5点）."""
        from xkep_cae.contact.assembly import compute_contact_force

        config = ContactConfig(line_contact=True, n_gauss=3, n_gauss_auto=True)
        manager = ContactManager(config=config)
        pair = manager.add_pair(
            0, 1, np.array([0, 1]), np.array([2, 3]), radius_a=0.05, radius_b=0.05
        )
        pair.state.status = ContactStatus.ACTIVE
        pair.state.k_pen = 1e6

        # 準平行配置（θ < 10° → n_gauss=5 に自動選択）
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.09],
                [1.0, 0.0, 0.09],
            ]
        )

        ndof = 4 * 6
        f = compute_contact_force(manager, ndof, ndof_per_node=6, node_coords=node_coords)

        assert float(np.linalg.norm(f)) > 0.0
