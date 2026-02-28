"""Line-to-line 摩擦力の Gauss 積分テスト（Phase C6-L1b）.

摩擦力の line contact 拡張:
- 各 Gauss 点で独立に Coulomb return mapping
- 分布摩擦力を Gauss 積分で評価
- PtP 代表点評価との比較
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.line_contact import (
    _build_tangent_shape_vector_at_gp,
    _compute_tangent_frame_at_gp,
    _init_gp_friction_states,
    compute_line_friction_force_local,
    compute_line_friction_stiffness_local,
)
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)


def _make_pair(
    s: float = 0.5,
    t: float = 0.5,
    gap: float = -0.001,
    k_pen: float = 1000.0,
    k_t: float = 500.0,
    lambda_n: float = 0.0,
    radius: float = 0.01,
) -> ContactPair:
    """テスト用の接触ペアを作成."""
    normal = np.array([0.0, 1.0, 0.0])
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, 0.0, 1.0])
    pair = ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        radius_a=radius,
        radius_b=radius,
        state=ContactState(
            s=s,
            t=t,
            gap=gap,
            normal=normal,
            tangent1=t1,
            tangent2=t2,
            lambda_n=lambda_n,
            k_pen=k_pen,
            k_t=k_t,
            p_n=max(0.0, lambda_n + k_pen * (-gap)),
            status=ContactStatus.ACTIVE,
        ),
    )
    return pair


class TestTangentShapeVector:
    """接線形状ベクトルのテスト."""

    def test_shape_vector_endpoints(self) -> None:
        """s_gp=0, t_closest=0 での形状ベクトル."""
        tangent = np.array([1.0, 0.0, 0.0])
        g_t = _build_tangent_shape_vector_at_gp(0.0, 0.0, tangent)
        # A0 に全荷重: (1-0)*t = t
        np.testing.assert_allclose(g_t[0:3], tangent)
        # A1 に 0
        np.testing.assert_allclose(g_t[3:6], 0.0)
        # B0 に -全荷重: -(1-0)*t = -t
        np.testing.assert_allclose(g_t[6:9], -tangent)
        # B1 に 0
        np.testing.assert_allclose(g_t[9:12], 0.0)

    def test_shape_vector_midpoint(self) -> None:
        """s_gp=0.5, t_closest=0.5 での形状ベクトル."""
        tangent = np.array([0.0, 0.0, 1.0])
        g_t = _build_tangent_shape_vector_at_gp(0.5, 0.5, tangent)
        np.testing.assert_allclose(g_t[0:3], 0.5 * tangent)
        np.testing.assert_allclose(g_t[3:6], 0.5 * tangent)
        np.testing.assert_allclose(g_t[6:9], -0.5 * tangent)
        np.testing.assert_allclose(g_t[9:12], -0.5 * tangent)

    def test_action_reaction(self) -> None:
        """作用反作用: A側の合計 + B側の合計 = 0."""
        tangent = np.array([1.0, 0.0, 0.0])
        for s in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
                g_t = _build_tangent_shape_vector_at_gp(s, t, tangent)
                f_A = g_t[0:3] + g_t[3:6]
                f_B = g_t[6:9] + g_t[9:12]
                np.testing.assert_allclose(f_A + f_B, 0.0, atol=1e-15)


class TestTangentFrame:
    """接線フレームのテスト."""

    def test_orthogonal(self) -> None:
        """接線フレームが法線に直交."""
        normal = np.array([0.0, 1.0, 0.0])
        ref_t1 = np.array([1.0, 0.0, 0.0])
        t1, t2 = _compute_tangent_frame_at_gp(normal, ref_t1)
        assert abs(float(np.dot(t1, normal))) < 1e-14
        assert abs(float(np.dot(t2, normal))) < 1e-14
        assert abs(float(np.dot(t1, t2))) < 1e-14

    def test_unit_vectors(self) -> None:
        """接線ベクトルが単位ベクトル."""
        normal = np.array([0.3, 0.9, 0.1])
        normal = normal / np.linalg.norm(normal)
        ref_t1 = np.array([1.0, 0.0, 0.0])
        t1, t2 = _compute_tangent_frame_at_gp(normal, ref_t1)
        assert abs(float(np.linalg.norm(t1)) - 1.0) < 1e-14
        assert abs(float(np.linalg.norm(t2)) - 1.0) < 1e-14

    def test_parallel_fallback(self) -> None:
        """ref_tangent が法線と平行の場合のフォールバック."""
        normal = np.array([1.0, 0.0, 0.0])
        ref_t1 = np.array([1.0, 0.0, 0.0])  # 平行
        t1, t2 = _compute_tangent_frame_at_gp(normal, ref_t1)
        assert abs(float(np.dot(t1, normal))) < 1e-10
        assert abs(float(np.linalg.norm(t1)) - 1.0) < 1e-10


class TestGPFrictionInit:
    """GP 摩擦状態の初期化テスト."""

    def test_init_from_none(self) -> None:
        """gp_z_t=None からの初期化."""
        pair = _make_pair()
        assert pair.state.gp_z_t is None
        _init_gp_friction_states(pair, 3)
        assert pair.state.gp_z_t is not None
        assert len(pair.state.gp_z_t) == 3
        assert len(pair.state.gp_stick) == 3
        assert all(s is True for s in pair.state.gp_stick)

    def test_reinit_on_size_change(self) -> None:
        """Gauss 点数が変わったら再初期化."""
        pair = _make_pair()
        _init_gp_friction_states(pair, 3)
        pair.state.gp_z_t[0] = np.array([1.0, 2.0])
        _init_gp_friction_states(pair, 5)
        assert len(pair.state.gp_z_t) == 5
        np.testing.assert_allclose(pair.state.gp_z_t[0], 0.0)

    def test_no_reinit_if_same_size(self) -> None:
        """Gauss 点数が同じなら再初期化しない."""
        pair = _make_pair()
        _init_gp_friction_states(pair, 3)
        pair.state.gp_z_t[0] = np.array([1.0, 2.0])
        _init_gp_friction_states(pair, 3)
        np.testing.assert_allclose(pair.state.gp_z_t[0], [1.0, 2.0])


class TestLineFrictionForce:
    """Line contact 摩擦力の Gauss 積分テスト."""

    def _make_crossing_pair(self) -> tuple:
        """交差する2本のセグメントと変位を準備."""
        # セグメント A: x軸方向 (0,0,0)→(1,0,0)
        # セグメント B: z軸方向 (0.5,-0.019,0)→(0.5,-0.019,1)
        # 半径 0.01 → 接触半径 0.02
        # 中心間距離 = 0.019 < 0.02 → 貫入
        pair = _make_pair(radius=0.01, k_pen=1000.0, k_t=500.0)
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, -0.019, 0.0])
        xB1 = np.array([0.5, -0.019, 1.0])

        ndof_per_node = 6
        ndof = 4 * ndof_per_node
        u_ref = np.zeros(ndof)
        u_cur = np.zeros(ndof)
        # B0 を接線方向に 0.001 移動（摩擦を発生させる）
        u_cur[2 * ndof_per_node + 0] = 0.001  # B0 x方向

        return pair, xA0, xA1, xB0, xB1, u_cur, u_ref, ndof_per_node

    def test_zero_friction_without_displacement(self) -> None:
        """相対変位なし → 摩擦力ゼロ."""
        pair = _make_pair(radius=0.01, k_pen=1000.0, k_t=500.0)
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, -0.019, 0.0])
        xB1 = np.array([0.5, -0.019, 1.0])

        ndof_per_node = 6
        ndof = 4 * ndof_per_node
        u = np.zeros(ndof)

        f_fric, total = compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u,
            u,
            ndof_per_node,
            0.3,
            3,
        )
        np.testing.assert_allclose(f_fric, 0.0, atol=1e-15)
        assert total == pytest.approx(0.0, abs=1e-15)

    def test_nonzero_friction_with_displacement(self) -> None:
        """接線方向変位あり → 摩擦力が発生."""
        pair, xA0, xA1, xB0, xB1, u_cur, u_ref, ndof_per_node = self._make_crossing_pair()

        f_fric, total = compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            3,
        )
        assert float(np.linalg.norm(f_fric)) > 1e-10
        assert total > 0.0

    def test_zero_friction_coefficient(self) -> None:
        """μ=0 → 摩擦力ゼロ."""
        pair, xA0, xA1, xB0, xB1, u_cur, u_ref, ndof_per_node = self._make_crossing_pair()

        f_fric, total = compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.0,
            3,
        )
        np.testing.assert_allclose(f_fric, 0.0, atol=1e-15)

    def test_action_reaction_symmetry(self) -> None:
        """摩擦力の作用反作用: ΣF_A + ΣF_B = 0."""
        pair, xA0, xA1, xB0, xB1, u_cur, u_ref, ndof_per_node = self._make_crossing_pair()

        f_fric, _ = compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            3,
        )
        f_A = f_fric[0:3] + f_fric[3:6]
        f_B = f_fric[6:9] + f_fric[9:12]
        np.testing.assert_allclose(f_A + f_B, 0.0, atol=1e-12)

    def test_gp_states_updated(self) -> None:
        """Return mapping 後に GP 状態が更新される."""
        pair, xA0, xA1, xB0, xB1, u_cur, u_ref, ndof_per_node = self._make_crossing_pair()

        compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            3,
        )
        assert pair.state.gp_z_t is not None
        assert len(pair.state.gp_z_t) == 3
        # 少なくとも1つの GP で摩擦が発生しているはず
        has_friction = any(float(np.linalg.norm(z)) > 1e-15 for z in pair.state.gp_z_t)
        assert has_friction

    def test_gauss_points_increase_accuracy(self) -> None:
        """Gauss 点数を増やしても結果が安定."""
        pair, xA0, xA1, xB0, xB1, u_cur, u_ref, ndof_per_node = self._make_crossing_pair()

        f2, _ = compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            2,
        )
        _init_gp_friction_states(pair, 3)  # リセット
        f3, _ = compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            3,
        )
        _init_gp_friction_states(pair, 5)  # リセット
        f5, _ = compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            5,
        )
        # n=3 と n=5 は n=2 より精度が高い（ノルムが近い）
        assert float(np.linalg.norm(f3 - f5)) < float(np.linalg.norm(f2 - f5)) + 1e-12


class TestLineFrictionStiffness:
    """Line contact 摩擦接線剛性のテスト."""

    def test_zero_without_friction(self) -> None:
        """μ=0 → 摩擦剛性ゼロ."""
        pair = _make_pair(radius=0.01, k_pen=1000.0, k_t=500.0)
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, -0.019, 0.0])
        xB1 = np.array([0.5, -0.019, 1.0])

        K = compute_line_friction_stiffness_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            0.0,
            3,
        )
        np.testing.assert_allclose(K, 0.0, atol=1e-15)

    def test_symmetric_for_stick(self) -> None:
        """Stick 状態での摩擦剛性は対称."""
        pair = _make_pair(radius=0.01, k_pen=1000.0, k_t=500.0)
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, -0.019, 0.0])
        xB1 = np.array([0.5, -0.019, 1.0])

        ndof_per_node = 6
        ndof = 4 * ndof_per_node
        u_ref = np.zeros(ndof)
        u_cur = np.zeros(ndof)
        u_cur[2 * ndof_per_node + 0] = 1e-8  # 微小変位 → stick

        # まず return mapping を実行して GP 状態を設定
        compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            3,
        )
        # 全 GP が stick であることを確認
        assert all(pair.state.gp_stick)

        K = compute_line_friction_stiffness_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            0.3,
            3,
        )
        # 対称性チェック
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_nonzero_with_friction(self) -> None:
        """摩擦あり → 非ゼロの剛性."""
        pair = _make_pair(radius=0.01, k_pen=1000.0, k_t=500.0)
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, -0.019, 0.0])
        xB1 = np.array([0.5, -0.019, 1.0])

        ndof_per_node = 6
        ndof = 4 * ndof_per_node
        u_ref = np.zeros(ndof)
        u_cur = np.zeros(ndof)
        u_cur[2 * ndof_per_node + 0] = 0.001

        compute_line_friction_force_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            u_cur,
            u_ref,
            ndof_per_node,
            0.3,
            3,
        )
        K = compute_line_friction_stiffness_local(
            pair,
            xA0,
            xA1,
            xB0,
            xB1,
            0.3,
            3,
        )
        assert float(np.linalg.norm(K)) > 1e-10


class TestAssemblyIntegration:
    """assembly.py との統合テスト."""

    def test_line_friction_in_contact_force(self) -> None:
        """line_friction_forces パラメータが contact force に反映される."""
        from xkep_cae.contact.assembly import compute_contact_force

        pair = _make_pair(radius=0.01, k_pen=1000.0, k_t=500.0)
        manager = ContactManager(pairs=[pair], config=ContactConfig(line_contact=True))

        ndof_per_node = 6
        ndof = 4 * ndof_per_node

        # 法線力のみ
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, -0.019, 0.0],
                [0.5, -0.019, 1.0],
            ]
        )
        f_normal = compute_contact_force(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
            node_coords=node_coords,
        )

        # line friction force を追加
        f_fric_local = np.zeros(12)
        f_fric_local[0:3] = [0.1, 0.0, 0.0]
        f_fric_local[6:9] = [-0.1, 0.0, 0.0]
        f_with_fric = compute_contact_force(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
            node_coords=node_coords,
            line_friction_forces={0: f_fric_local},
        )

        # 摩擦分の差が反映されているはず
        diff = f_with_fric - f_normal
        assert float(np.linalg.norm(diff)) > 0.05

    def test_line_friction_in_stiffness(self) -> None:
        """line_friction_stiffnesses パラメータが stiffness に反映される."""
        from xkep_cae.contact.assembly import compute_contact_stiffness

        pair = _make_pair(radius=0.01, k_pen=1000.0, k_t=500.0)
        manager = ContactManager(pairs=[pair], config=ContactConfig(line_contact=True))

        ndof_per_node = 6
        ndof = 4 * ndof_per_node

        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, -0.019, 0.0],
                [0.5, -0.019, 1.0],
            ]
        )

        K_normal = compute_contact_stiffness(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
            node_coords=node_coords,
        )

        K_fric_local = np.eye(12) * 10.0
        K_with_fric = compute_contact_stiffness(
            manager,
            ndof,
            ndof_per_node=ndof_per_node,
            node_coords=node_coords,
            line_friction_stiffnesses={0: K_fric_local},
        )

        diff = K_with_fric - K_normal
        assert diff.nnz > 0


class TestContactStateCopy:
    """ContactState の GP 摩擦状態コピーテスト."""

    def test_copy_with_gp_states(self) -> None:
        """GP 摩擦状態を含む deep copy."""
        state = ContactState(status=ContactStatus.ACTIVE)
        state.gp_z_t = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        state.gp_stick = [True, False]
        state.gp_q_trial_norm = [1.5, 2.5]

        copy = state.copy()
        assert copy.gp_z_t is not None
        np.testing.assert_allclose(copy.gp_z_t[0], [1.0, 2.0])
        np.testing.assert_allclose(copy.gp_z_t[1], [3.0, 4.0])

        # 独立性確認
        copy.gp_z_t[0][0] = 99.0
        assert state.gp_z_t[0][0] == pytest.approx(1.0)

    def test_copy_without_gp_states(self) -> None:
        """GP 摩擦状態なしの deep copy."""
        state = ContactState(status=ContactStatus.ACTIVE)
        copy = state.copy()
        assert copy.gp_z_t is None
        assert copy.gp_stick is None
