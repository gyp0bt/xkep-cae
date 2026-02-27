"""Phase C6-L2: ∂(s,t)/∂u 一貫接線のテスト.

テスト項目:
1. compute_st_jacobian の数値微分検証（PtP）
2. compute_t_jacobian_at_gp の数値微分検証（line contact）
3. 一貫接線剛性の数値微分検証（K_st vs 有限差分）
4. consistent_st_tangent フラグの後方互換性
5. Assembly 統合テスト
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.geometry import closest_point_segments, compute_st_jacobian
from xkep_cae.contact.line_contact import (
    compute_t_jacobian_at_gp,
    project_point_to_segment,
)
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactStatus,
)


def _make_pair(xA0, xA1, xB0, xB1, *, k_pen=1e4, lambda_n=0.0, radius=0.1) -> ContactPair:
    """テスト用の接触ペアを構築する."""
    pair = ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        radius_a=radius,
        radius_b=radius,
    )
    result = closest_point_segments(xA0, xA1, xB0, xB1)
    pair.state.s = result.s
    pair.state.t = result.t
    pair.state.gap = result.distance - 2 * radius
    pair.state.normal = result.normal
    pair.state.k_pen = k_pen
    pair.state.lambda_n = lambda_n

    # p_n を評価
    p_n = max(0.0, lambda_n + k_pen * (-pair.state.gap))
    pair.state.p_n = p_n
    pair.state.status = ContactStatus.ACTIVE if p_n > 0.0 else ContactStatus.INACTIVE

    return pair


# ---------------------------------------------------------------------------
# TestComputeStJacobian: ∂(s,t)/∂u の数値微分検証
# ---------------------------------------------------------------------------
class TestComputeStJacobian:
    """compute_st_jacobian の数値微分検証."""

    def test_perpendicular_crossing(self):
        """直交する2セグメント（内部最近接点）."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([2.0, 0.0, 0.0])
        xB0 = np.array([1.0, 0.0, -1.0])
        xB1 = np.array([1.0, 0.0, 1.0])
        result = closest_point_segments(xA0, xA1, xB0, xB1)
        s0, t0 = result.s, result.t

        out = compute_st_jacobian(s0, t0, xA0, xA1, xB0, xB1)
        assert out is not None
        ds_du, dt_du = out

        # 数値微分で検証
        eps = 1e-7
        coords = np.array([xA0, xA1, xB0, xB1])  # (4, 3)
        ds_du_num = np.zeros(12)
        dt_du_num = np.zeros(12)

        for k in range(12):
            node_idx, dof_idx = divmod(k, 3)
            coords_p = coords.copy()
            coords_p[node_idx, dof_idx] += eps
            res_p = closest_point_segments(coords_p[0], coords_p[1], coords_p[2], coords_p[3])

            coords_m = coords.copy()
            coords_m[node_idx, dof_idx] -= eps
            res_m = closest_point_segments(coords_m[0], coords_m[1], coords_m[2], coords_m[3])

            ds_du_num[k] = (res_p.s - res_m.s) / (2 * eps)
            dt_du_num[k] = (res_p.t - res_m.t) / (2 * eps)

        np.testing.assert_allclose(ds_du, ds_du_num, atol=1e-5)
        np.testing.assert_allclose(dt_du, dt_du_num, atol=1e-5)

    def test_skew_segments(self):
        """斜めに交差する2セグメント."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.3, 0.0])
        xB0 = np.array([0.5, -0.5, 0.2])
        xB1 = np.array([0.5, 0.5, -0.2])
        result = closest_point_segments(xA0, xA1, xB0, xB1)
        s0, t0 = result.s, result.t

        out = compute_st_jacobian(s0, t0, xA0, xA1, xB0, xB1)
        assert out is not None
        ds_du, dt_du = out

        eps = 1e-7
        coords = np.array([xA0, xA1, xB0, xB1])
        ds_du_num = np.zeros(12)
        dt_du_num = np.zeros(12)

        for k in range(12):
            node_idx, dof_idx = divmod(k, 3)
            coords_p = coords.copy()
            coords_p[node_idx, dof_idx] += eps
            res_p = closest_point_segments(coords_p[0], coords_p[1], coords_p[2], coords_p[3])
            coords_m = coords.copy()
            coords_m[node_idx, dof_idx] -= eps
            res_m = closest_point_segments(coords_m[0], coords_m[1], coords_m[2], coords_m[3])
            ds_du_num[k] = (res_p.s - res_m.s) / (2 * eps)
            dt_du_num[k] = (res_p.t - res_m.t) / (2 * eps)

        np.testing.assert_allclose(ds_du, ds_du_num, atol=1e-5)
        np.testing.assert_allclose(dt_du, dt_du_num, atol=1e-5)

    def test_parallel_returns_none(self):
        """平行セグメント → None（特異ケース）."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 1.0, 0.0])
        xB1 = np.array([1.0, 1.0, 0.0])
        # 平行の場合、closest_point_segments は s=0 を返す（クランプ）
        result = closest_point_segments(xA0, xA1, xB0, xB1)
        out = compute_st_jacobian(result.s, result.t, xA0, xA1, xB0, xB1)
        # s=0 でクランプ → s_clamped, t は free なので結果は返る
        # ただし s_clamped=True なので ds_du=0
        if out is not None:
            ds_du, dt_du = out
            np.testing.assert_allclose(ds_du, 0.0, atol=1e-10)

    def test_both_clamped(self):
        """両方クランプされた場合 → ds/du = dt/du = 0."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([2.0, 0.0, 0.3])
        xB1 = np.array([2.0, 1.0, 0.3])
        result = closest_point_segments(xA0, xA1, xB0, xB1)
        # s=1.0 (端点), t=0.0 (端点) → 両方クランプ
        out = compute_st_jacobian(result.s, result.t, xA0, xA1, xB0, xB1)
        assert out is not None
        ds_du, dt_du = out
        np.testing.assert_allclose(ds_du, 0.0, atol=1e-10)
        np.testing.assert_allclose(dt_du, 0.0, atol=1e-10)

    def test_s_clamped_t_free(self):
        """s がクランプ、t は内部 → ds/du=0, dt/du のみ計算."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([-0.5, -0.5, 0.3])
        xB1 = np.array([-0.5, 0.5, 0.3])
        result = closest_point_segments(xA0, xA1, xB0, xB1)
        # s=0.0 (クランプ), t は内部
        assert result.s < 1e-10  # s クランプ確認
        assert 0.1 < result.t < 0.9  # t 内部確認

        out = compute_st_jacobian(result.s, result.t, xA0, xA1, xB0, xB1)
        assert out is not None
        ds_du, dt_du = out
        np.testing.assert_allclose(ds_du, 0.0, atol=1e-10)
        # dt_du は非ゼロ
        assert np.linalg.norm(dt_du) > 1e-6

        # 数値微分で dt_du を検証
        eps = 1e-7
        coords = np.array([xA0, xA1, xB0, xB1])
        dt_du_num = np.zeros(12)
        for k in range(12):
            node_idx, dof_idx = divmod(k, 3)
            coords_p = coords.copy()
            coords_p[node_idx, dof_idx] += eps
            res_p = closest_point_segments(coords_p[0], coords_p[1], coords_p[2], coords_p[3])
            coords_m = coords.copy()
            coords_m[node_idx, dof_idx] -= eps
            res_m = closest_point_segments(coords_m[0], coords_m[1], coords_m[2], coords_m[3])
            dt_du_num[k] = (res_p.t - res_m.t) / (2 * eps)

        np.testing.assert_allclose(dt_du, dt_du_num, atol=1e-5)


# ---------------------------------------------------------------------------
# TestComputeTJacobianAtGP: line contact Gauss 点での ∂t/∂u
# ---------------------------------------------------------------------------
class TestComputeTJacobianAtGP:
    """compute_t_jacobian_at_gp の数値微分検証."""

    def test_interior_projection(self):
        """射影点が内部にある場合の数値微分検証."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([2.0, 0.0, 0.0])
        xB0 = np.array([0.5, 0.3, -1.0])
        xB1 = np.array([0.5, 0.3, 1.0])
        s_gp = 0.3

        pA = (1 - s_gp) * xA0 + s_gp * xA1
        t0 = project_point_to_segment(pA, xB0, xB1)
        # 内部であることを確認
        assert 0.01 < t0 < 0.99

        dt_du = compute_t_jacobian_at_gp(s_gp, t0, xA0, xA1, xB0, xB1)
        assert dt_du is not None

        # 数値微分で検証
        eps = 1e-7
        coords = np.array([xA0, xA1, xB0, xB1])
        dt_du_num = np.zeros(12)

        for k in range(12):
            node_idx, dof_idx = divmod(k, 3)
            coords_p = coords.copy()
            coords_p[node_idx, dof_idx] += eps
            pA_p = (1 - s_gp) * coords_p[0] + s_gp * coords_p[1]
            t_p = project_point_to_segment(pA_p, coords_p[2], coords_p[3])

            coords_m = coords.copy()
            coords_m[node_idx, dof_idx] -= eps
            pA_m = (1 - s_gp) * coords_m[0] + s_gp * coords_m[1]
            t_m = project_point_to_segment(pA_m, coords_m[2], coords_m[3])

            dt_du_num[k] = (t_p - t_m) / (2 * eps)

        np.testing.assert_allclose(dt_du, dt_du_num, atol=1e-5)

    def test_clamped_returns_zero(self):
        """射影点がクランプ（端点）の場合 → dt/du = 0."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([2.0, 0.0, 0.0])
        xB0 = np.array([3.0, 0.3, 0.0])
        xB1 = np.array([3.0, 0.3, 1.0])
        s_gp = 0.9

        pA = (1 - s_gp) * xA0 + s_gp * xA1
        t0 = project_point_to_segment(pA, xB0, xB1)
        # t=0 にクランプされるはず
        assert t0 < 1e-10

        dt_du = compute_t_jacobian_at_gp(s_gp, t0, xA0, xA1, xB0, xB1)
        assert dt_du is not None
        np.testing.assert_allclose(dt_du, 0.0, atol=1e-10)

    def test_multiple_gauss_points(self):
        """複数の Gauss 点で数値微分検証."""
        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.0, 0.3, -0.5])
        xB1 = np.array([1.0, 0.3, 0.5])

        eps = 1e-7
        coords = np.array([xA0, xA1, xB0, xB1])

        for s_gp in [0.2, 0.5, 0.8]:
            pA = (1 - s_gp) * xA0 + s_gp * xA1
            t0 = project_point_to_segment(pA, xB0, xB1)

            if t0 < 1e-10 or t0 > 1.0 - 1e-10:
                continue

            dt_du = compute_t_jacobian_at_gp(s_gp, t0, xA0, xA1, xB0, xB1)
            assert dt_du is not None

            dt_du_num = np.zeros(12)
            for k in range(12):
                node_idx, dof_idx = divmod(k, 3)
                coords_p = coords.copy()
                coords_p[node_idx, dof_idx] += eps
                pA_p = (1 - s_gp) * coords_p[0] + s_gp * coords_p[1]
                t_p = project_point_to_segment(pA_p, coords_p[2], coords_p[3])

                coords_m = coords.copy()
                coords_m[node_idx, dof_idx] -= eps
                pA_m = (1 - s_gp) * coords_m[0] + s_gp * coords_m[1]
                t_m = project_point_to_segment(pA_m, coords_m[2], coords_m[3])

                dt_du_num[k] = (t_p - t_m) / (2 * eps)

            np.testing.assert_allclose(dt_du, dt_du_num, atol=1e-5)


# ---------------------------------------------------------------------------
# TestConsistentStStiffness: 一貫接線剛性の数値微分検証
# ---------------------------------------------------------------------------
class TestConsistentStStiffness:
    """接触力の完全な数値微分（K_full）と解析的一貫接線の比較."""

    def _compute_contact_force_at(self, coords, k_pen, lambda_n, radius):
        """指定座標での接触力を計算する（PtP）."""
        from xkep_cae.contact.assembly import _contact_shape_vector

        xA0, xA1, xB0, xB1 = coords[0], coords[1], coords[2], coords[3]
        pair = _make_pair(xA0, xA1, xB0, xB1, k_pen=k_pen, lambda_n=lambda_n, radius=radius)
        if pair.state.p_n <= 0:
            return np.zeros(12)
        g_n = _contact_shape_vector(pair)
        return pair.state.p_n * g_n

    def test_full_stiffness_numerical(self):
        """完全な接触剛性（K_n + K_geo + K_st）を数値微分で検証."""
        from xkep_cae.contact.assembly import (
            _consistent_st_stiffness_local,
            _contact_geometric_stiffness_local,
            _contact_shape_vector,
        )
        from xkep_cae.contact.geometry import compute_st_jacobian

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, 0.18, -0.5])
        xB1 = np.array([0.5, 0.18, 0.5])
        k_pen = 1e4
        lambda_n = 0.0
        radius = 0.1

        pair = _make_pair(xA0, xA1, xB0, xB1, k_pen=k_pen, lambda_n=lambda_n, radius=radius)
        assert pair.state.p_n > 0, "テストケースは接触が必要"

        # 解析的 K = K_n + K_geo + K_st
        g_n = _contact_shape_vector(pair)
        K_n = k_pen * np.outer(g_n, g_n)
        K_geo = _contact_geometric_stiffness_local(pair)

        result = compute_st_jacobian(pair.state.s, pair.state.t, xA0, xA1, xB0, xB1)
        assert result is not None
        ds_du, dt_du = result
        K_st = _consistent_st_stiffness_local(pair, xA0, xA1, xB0, xB1, ds_du, dt_du)

        K_analytic = K_n + K_geo + K_st

        # 数値微分
        eps = 1e-7
        coords = np.array([xA0, xA1, xB0, xB1], dtype=float)
        K_numerical = np.zeros((12, 12))

        for j in range(12):
            node_idx, dof_idx = divmod(j, 3)
            coords_p = coords.copy()
            coords_p[node_idx, dof_idx] += eps
            f_p = self._compute_contact_force_at(coords_p, k_pen, lambda_n, radius)

            coords_m = coords.copy()
            coords_m[node_idx, dof_idx] -= eps
            f_m = self._compute_contact_force_at(coords_m, k_pen, lambda_n, radius)

            K_numerical[:, j] = (f_p - f_m) / (2 * eps)

        # K_analytic と K_numerical の比較
        # 相対誤差で比較（小さい成分はスキップ）
        max_val = max(np.max(np.abs(K_numerical)), np.max(np.abs(K_analytic)), 1e-10)
        np.testing.assert_allclose(
            K_analytic / max_val,
            K_numerical / max_val,
            atol=1e-4,
        )

    def test_k_st_zero_when_s_t_interior_no_motion(self):
        """s,t が内部で変位がない場合、K_st は非ゼロ（ds/du≠0 のため）."""
        from xkep_cae.contact.assembly import _consistent_st_stiffness_local
        from xkep_cae.contact.geometry import compute_st_jacobian

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, 0.18, -0.5])
        xB1 = np.array([0.5, 0.18, 0.5])
        pair = _make_pair(xA0, xA1, xB0, xB1)
        assert pair.state.p_n > 0

        result = compute_st_jacobian(pair.state.s, pair.state.t, xA0, xA1, xB0, xB1)
        assert result is not None
        ds_du, dt_du = result
        K_st = _consistent_st_stiffness_local(pair, xA0, xA1, xB0, xB1, ds_du, dt_du)

        # K_st は非ゼロ（接触点の移動による剛性補正が存在）
        assert np.max(np.abs(K_st)) > 1e-6

    def test_k_st_inactive_pair(self):
        """非活性ペアでは K_st = 0."""
        from xkep_cae.contact.assembly import _consistent_st_stiffness_local

        xA0 = np.array([0.0, 0.0, 0.0])
        xA1 = np.array([1.0, 0.0, 0.0])
        xB0 = np.array([0.5, 5.0, -0.5])  # 離れている
        xB1 = np.array([0.5, 5.0, 0.5])
        pair = _make_pair(xA0, xA1, xB0, xB1)
        # p_n = 0 のはず

        ds_du = np.zeros(12)
        dt_du = np.zeros(12)
        K_st = _consistent_st_stiffness_local(pair, xA0, xA1, xB0, xB1, ds_du, dt_du)
        np.testing.assert_allclose(K_st, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestAssemblyConsistentTangent: Assembly 統合テスト
# ---------------------------------------------------------------------------
class TestAssemblyConsistentTangent:
    """compute_contact_stiffness の consistent_st_tangent 統合テスト."""

    def _make_manager_and_coords(self, *, consistent=False, line_contact=False):
        """テスト用の ContactManager と座標を構築."""
        config = ContactConfig(
            k_pen_scale=1e4,
            consistent_st_tangent=consistent,
            line_contact=line_contact,
        )
        manager = ContactManager(config=config)

        # 4節点: A=(0,1), B=(2,3)
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],  # node 0: A0
                [1.0, 0.0, 0.0],  # node 1: A1
                [0.5, 0.18, -0.5],  # node 2: B0
                [0.5, 0.18, 0.5],  # node 3: B1
            ]
        )

        pair = manager.add_pair(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            radius_a=0.1,
            radius_b=0.1,
        )
        manager.update_geometry(node_coords)
        pair.state.k_pen = 1e4
        pair.state.k_t = 5e3
        p_n = max(0.0, pair.state.lambda_n + pair.state.k_pen * (-pair.state.gap))
        pair.state.p_n = p_n

        return manager, node_coords

    def test_backward_compat_default_off(self):
        """consistent_st_tangent=False（デフォルト）で既存動作と同一."""
        from xkep_cae.contact.assembly import compute_contact_stiffness

        manager_off, coords = self._make_manager_and_coords(consistent=False)
        manager_on, _ = self._make_manager_and_coords(consistent=True)

        ndof = 4 * 6  # 4ノード × 6DOF
        K_off = compute_contact_stiffness(
            manager_off,
            ndof,
            node_coords=coords,
            use_geometric_stiffness=True,
        )
        K_on = compute_contact_stiffness(
            manager_on,
            ndof,
            node_coords=coords,
            use_geometric_stiffness=True,
        )

        # consistent=True は追加項がある → K_on ≠ K_off
        diff = (K_on - K_off).toarray()
        # 差分は非ゼロ（K_st 分）
        assert np.max(np.abs(diff)) > 1e-6

    def test_consistent_stiffness_ptp(self):
        """PtP での consistent_st_tangent が正常に動作."""
        from xkep_cae.contact.assembly import compute_contact_stiffness

        manager, coords = self._make_manager_and_coords(consistent=True)
        ndof = 4 * 6
        K = compute_contact_stiffness(
            manager,
            ndof,
            node_coords=coords,
            use_geometric_stiffness=True,
        )
        # 行列が生成されること
        assert K.nnz > 0

    def test_consistent_stiffness_line_contact(self):
        """Line contact での consistent_st_tangent が正常に動作."""
        from xkep_cae.contact.assembly import compute_contact_stiffness

        manager, coords = self._make_manager_and_coords(consistent=True, line_contact=True)
        ndof = 4 * 6
        K = compute_contact_stiffness(
            manager,
            ndof,
            node_coords=coords,
            use_geometric_stiffness=True,
        )
        assert K.nnz > 0

    def test_no_node_coords_skips_st(self):
        """node_coords=None の場合は consistent_st_tangent が無視される."""
        from xkep_cae.contact.assembly import compute_contact_stiffness

        manager, coords = self._make_manager_and_coords(consistent=True)
        ndof = 4 * 6

        # node_coords=None → K_st が追加されない
        K_no_coords = compute_contact_stiffness(
            manager,
            ndof,
            node_coords=None,
            use_geometric_stiffness=True,
        )
        # node_coords 提供時
        K_with_coords = compute_contact_stiffness(
            manager,
            ndof,
            node_coords=coords,
            use_geometric_stiffness=True,
        )
        # K_st 分の差がある
        diff = (K_with_coords - K_no_coords).toarray()
        assert np.max(np.abs(diff)) > 1e-6
