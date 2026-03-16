"""大規模マルチセグメント（16+セグメント）接触性能評価テスト（NCP版）.

NCP移行版: test_large_scale_contact.py から移行。
旧テスト（ペナルティ/AL）→ newton_raphson_contact_ncp（NCP）。

テスト項目:
1. DOFスケーリング: セグメント数に対するDOF数の線形性
2. Broadphase効率: 候補ペアフィルタリングのサブ二乗スケーリング
3. 16セグメント: 収束、接触検出、Active set局在化、法線力正値性
4. スケーラビリティ: 4/8/16セグメントでの一貫した動作
"""

import numpy as np
import pytest
import scipy.sparse as sp
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
)
from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp

pytestmark = pytest.mark.slow

_NDOF_PER_NODE = 6
_K_SPRING_BASE = 1e4


def _make_large_crossing_beams(
    n_seg_a=16,
    n_seg_b=16,
    k_spring_base=_K_SPRING_BASE,
    z_top=0.082,
    radii=0.04,
):
    """大規模マルチセグメント交差梁ばねモデル."""
    n_nodes_a = n_seg_a + 1
    n_nodes_b = n_seg_b + 1
    n_nodes = n_nodes_a + n_nodes_b
    ndof_total = n_nodes * _NDOF_PER_NODE

    coords_a = np.zeros((n_nodes_a, 3))
    for i in range(n_nodes_a):
        coords_a[i, 0] = i / n_seg_a

    coords_b = np.zeros((n_nodes_b, 3))
    for i in range(n_nodes_b):
        coords_b[i, 0] = 0.5
        coords_b[i, 1] = -0.5 + i / n_seg_b
        coords_b[i, 2] = z_top

    node_coords_ref = np.vstack([coords_a, coords_b])

    conn_a = np.array([[i, i + 1] for i in range(n_seg_a)])
    conn_b = np.array([[n_nodes_a + i, n_nodes_a + i + 1] for i in range(n_seg_b)])
    connectivity = np.vstack([conn_a, conn_b])

    k_elem_a = k_spring_base * n_seg_a
    k_elem_b = k_spring_base * n_seg_b

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for idx, (n0, n1) in enumerate(connectivity):
            k = k_elem_a if idx < n_seg_a else k_elem_b
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                K[d0, d0] += k
                K[d0, d1] -= k
                K[d1, d0] -= k
                K[d1, d1] += k
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for idx, (n0, n1) in enumerate(connectivity):
            k = k_elem_a if idx < n_seg_a else k_elem_b
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k * delta
                f_int[d1] += k * delta
        return f_int

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        assemble_tangent,
        assemble_internal_force,
    )


def _fixed_dofs_large(n_nodes_a, n_nodes_b):
    """大規模マルチセグメント版の拘束DOF."""
    fixed = set()
    n_nodes = n_nodes_a + n_nodes_b
    for i in range(n_nodes):
        for d in range(3, _NDOF_PER_NODE):
            fixed.add(i * _NDOF_PER_NODE + d)
    for d in range(3):
        fixed.add(d)
    node_a_last = n_nodes_a - 1
    for d in range(1, 3):
        fixed.add(node_a_last * _NDOF_PER_NODE + d)
    node_b_first = n_nodes_a
    for d in range(3):
        fixed.add(node_b_first * _NDOF_PER_NODE + d)
    node_b_last = n_nodes_a + n_nodes_b - 1
    fixed.add(node_b_last * _NDOF_PER_NODE + 0)
    return np.array(sorted(fixed), dtype=int)


def _solve_large_problem(
    n_seg_a=16,
    n_seg_b=16,
    f_x=10.0,
    f_y=5.0,
    f_z=50.0,
    k_spring_base=_K_SPRING_BASE,
    k_pen=1e5,
    z_top=0.082,
    radii=0.04,
    n_load_steps=20,
    max_iter=50,
):
    """大規模マルチセグメント接触問題をNCP法で解く."""
    (
        node_coords_ref,
        connectivity,
        radii_val,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        assemble_tangent,
        assemble_internal_force,
    ) = _make_large_crossing_beams(
        n_seg_a=n_seg_a,
        n_seg_b=n_seg_b,
        k_spring_base=k_spring_base,
        z_top=z_top,
        radii=radii,
    )

    f_ext = np.zeros(ndof_total)
    node_a_last = n_nodes_a - 1
    node_b_last = n_nodes_a + n_nodes_b - 1
    f_ext[node_a_last * _NDOF_PER_NODE + 0] = f_x
    f_ext[node_b_last * _NDOF_PER_NODE + 1] = f_y
    f_ext[node_b_last * _NDOF_PER_NODE + 2] = -f_z

    fixed_dofs = _fixed_dofs_large(n_nodes_a, n_nodes_b)

    config = ContactConfig(
        k_pen_scale=k_pen,
        k_t_ratio=0.1,
        g_on=0.0,
        g_off=1e-4,
        use_geometric_stiffness=True,
        tol_penetration_ratio=0.02,
        penalty_growth_factor=2.0,
        k_pen_max=1e12,
    )
    mgr = ContactManager(config=config)

    result = newton_raphson_contact_ncp(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        node_coords_ref,
        connectivity,
        radii_val,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        tol_force=1e-6,
        tol_ncp=1e-6,
        show_progress=False,
        broadphase_margin=0.05,
        k_pen=k_pen,
    )
    return result, mgr, ndof_total


# ====================================================================
# テスト: DOF スケーリング
# ====================================================================


class TestDOFScaling:
    """DOF数のスケーリングを検証."""

    def test_dof_count_scales_linearly(self):
        """DOF数がセグメント数に線形にスケールする."""
        for n_seg in [4, 8, 16, 32]:
            n_nodes = 2 * (n_seg + 1)
            expected_ndof = n_nodes * _NDOF_PER_NODE
            (_, _, _, ndof_total, _, _, _, _) = _make_large_crossing_beams(
                n_seg_a=n_seg, n_seg_b=n_seg
            )
            assert ndof_total == expected_ndof

    def test_connectivity_count(self):
        """要素数がセグメント数の和に等しい."""
        for n_seg in [4, 8, 16, 32]:
            (_, conn, _, _, _, _, _, _) = _make_large_crossing_beams(n_seg_a=n_seg, n_seg_b=n_seg)
            assert len(conn) == 2 * n_seg


# ====================================================================
# テスト: Broadphase フィルタリング効率
# ====================================================================


class TestBroadphaseEfficiency:
    """Broadphaseによる候補ペアフィルタリングの効率検証（NCP版）."""

    def test_16seg_broadphase_filters(self):
        """16セグメントで候補ペア数が全組み合わせ未満."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        n_total_possible = 16 * 16
        n_pairs = mgr.n_pairs
        assert n_pairs < n_total_possible, f"候補ペア数 {n_pairs} >= 全ペア数 {n_total_possible}"

    def test_broadphase_sublinear_scaling(self):
        """セグメント数が4倍になっても候補ペア数は16倍にならない."""
        pairs_counts = {}
        for n_seg in [8, 32]:
            (
                node_coords_ref,
                conn,
                radii_val,
                _,
                _,
                _,
                _,
                _,
            ) = _make_large_crossing_beams(n_seg_a=n_seg, n_seg_b=n_seg)
            mgr = ContactManager(config=ContactConfig())
            mgr.detect_candidates(
                node_coords_ref,
                conn,
                radii_val,
                margin=0.05,
            )
            pairs_counts[n_seg] = mgr.n_pairs

        if pairs_counts[8] > 0:
            ratio = pairs_counts[32] / pairs_counts[8]
            assert ratio < 20.0, (
                f"ペア数比 {ratio:.1f} >= 20 (8seg: {pairs_counts[8]}, 32seg: {pairs_counts[32]})"
            )


# ====================================================================
# テスト: 16セグメント接触検出（NCP版）
# ====================================================================


class TestLargeScale16Segment:
    """16セグメント交差梁の接触テスト（NCP版）."""

    def test_16seg_converges(self):
        """16セグメント梁でNCPソルバーが収束する."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        assert result.converged, "16セグメントでNCPソルバーが収束しなかった"

    def test_16seg_contact_detected(self):
        """16セグメント梁で接触が検出される."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        assert result.converged
        assert mgr.n_active > 0, "接触が検出されなかった"

    def test_16seg_active_pairs_localized(self):
        """接触ペアは交差点近傍に局在する."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        assert result.converged
        n_total = mgr.n_pairs
        n_active = mgr.n_active
        if n_total > 0:
            active_ratio = n_active / n_total
            assert active_ratio < 0.5, f"Active比が高すぎる: {n_active}/{n_total}"

    def test_16seg_normal_force_positive(self):
        """接触法線力が非負."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        assert result.converged
        for pair in mgr.pairs:
            if pair.is_active():
                assert pair.state.p_n >= 0, f"p_n={pair.state.p_n} < 0"


# ====================================================================
# テスト: スケーラビリティ（NCP版）
# ====================================================================


class TestScalability:
    """セグメント数の増加に対する動作検証（NCP版）."""

    def test_all_converge_4_8_16(self):
        """4, 8, 16 セグメント全てで収束する."""
        for n_seg in [4, 8, 16]:
            result, mgr, _ = _solve_large_problem(
                n_seg_a=n_seg,
                n_seg_b=n_seg,
            )
            assert result.converged, f"n_seg={n_seg} で収束しなかった"
            assert mgr.n_active > 0, f"n_seg={n_seg} で接触未検出"

    def test_contact_force_positive_all_scales(self):
        """4, 8, 16 全スケールで接触力が正値."""
        for n_seg in [4, 8, 16]:
            result, mgr, _ = _solve_large_problem(
                n_seg_a=n_seg,
                n_seg_b=n_seg,
            )
            assert result.converged, f"n_seg={n_seg} で収束しなかった"
            active_pns = [p.state.p_n for p in mgr.pairs if p.is_active()]
            assert len(active_pns) > 0, f"n_seg={n_seg} で接触力なし"
            total_force = sum(active_pns)
            assert total_force > 0, f"n_seg={n_seg}: 総接触力 {total_force:.3f} <= 0"
