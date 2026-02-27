"""大規模マルチセグメント（16+セグメント）接触性能評価テスト.

status-050 TODO: 大規模マルチセグメント（16+セグメント）での性能評価

16分割以上のセグメントを持つ梁同士の接触問題で、
アルゴリズムのスケーラビリティと動作を検証する。

ばねモデルの制約:
  - ばねモデルは並進DOFのみ（回転剛性なし）であり、セグメント数増加に
    伴う数値的困難がある（構造剛性/接触剛性比の問題）。
  - 大規模問題での貫入精度検証は実梁要素(test_real_beam_contact.py)で行う。
  - 本テストはアルゴリズムのスケーラビリティ（DOF数, broadphase効率,
    接触検出能力）の検証に主眼を置く。
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
)
from xkep_cae.contact.solver_hooks import (
    newton_raphson_with_contact,
)

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
    """大規模マルチセグメント交差梁ばねモデル.

    要素剛性 = k_spring_base * n_seg（全体剛性を一定に保つ）。
    """
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
    k_pen_scale=1e5,
    z_top=0.082,
    radii=0.04,
    n_load_steps=20,
    max_iter=50,
    use_friction=False,
    mu=0.3,
    tol_penetration_ratio=0.01,
):
    """大規模マルチセグメント接触問題を解く."""
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
        k_pen_scale=k_pen_scale,
        k_t_ratio=0.1,
        mu=mu,
        g_on=0.0,
        g_off=1e-4,
        n_outer_max=8,
        use_friction=use_friction,
        mu_ramp_steps=3 if use_friction else 0,
        use_line_search=True,
        line_search_max_steps=5,
        use_geometric_stiffness=True,
        tol_penetration_ratio=tol_penetration_ratio,
        penalty_growth_factor=2.0,
        k_pen_max=1e12,
    )
    mgr = ContactManager(config=config)

    result = newton_raphson_with_contact(
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
        show_progress=False,
        broadphase_margin=0.05,
    )
    return result, mgr, ndof_total


# ====================================================================
# テストクラス: DOF スケーリング
# ====================================================================


class TestDOFScaling:
    """DOF数のスケーリングを検証."""

    def test_dof_count_scales_linearly(self):
        """DOF数がセグメント数に線形にスケールする."""
        for n_seg in [4, 8, 16, 32]:
            n_nodes = 2 * (n_seg + 1)
            expected_ndof = n_nodes * _NDOF_PER_NODE
            (
                _,
                _,
                _,
                ndof_total,
                _,
                _,
                _,
                _,
            ) = _make_large_crossing_beams(n_seg_a=n_seg, n_seg_b=n_seg)
            assert ndof_total == expected_ndof

    def test_connectivity_count(self):
        """要素数がセグメント数の和に等しい."""
        for n_seg in [4, 8, 16, 32]:
            (
                _,
                conn,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = _make_large_crossing_beams(n_seg_a=n_seg, n_seg_b=n_seg)
            assert len(conn) == 2 * n_seg


# ====================================================================
# テストクラス: Broadphase フィルタリング効率
# ====================================================================


class TestBroadphaseEfficiency:
    """Broadphaseによる候補ペアフィルタリングの効率検証.

    交差梁の接触候補は交差点近傍に集中するため、
    broadphaseが遠方ペアを効率的にフィルタリングすることを確認する。
    """

    def test_16seg_broadphase_filters(self):
        """16セグメントで候補ペア数が全組み合わせ未満."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        n_total_possible = 16 * 16
        n_pairs = mgr.n_pairs
        assert n_pairs < n_total_possible, f"候補ペア数 {n_pairs} >= 全ペア数 {n_total_possible}"

    def test_32seg_broadphase_filters(self):
        """32セグメントで候補ペア数が全組み合わせ未満."""
        result, mgr, _ = _solve_large_problem(
            n_seg_a=32,
            n_seg_b=32,
            n_load_steps=20,
        )
        n_total_possible = 32 * 32
        n_pairs = mgr.n_pairs
        assert n_pairs < n_total_possible, f"候補ペア数 {n_pairs} >= 全ペア数 {n_total_possible}"

    def test_broadphase_sublinear_scaling(self):
        """セグメント数が4倍になっても候補ペア数は16倍にならない.

        Broadphaseの空間分割により、候補ペア数は O(n) に近い。
        """
        pairs_counts = {}
        for n_seg in [8, 32]:
            (
                node_coords_ref,
                conn,
                radii_val,
                ndof_total,
                n_nodes_a,
                n_nodes_b,
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

        # 4倍のセグメントで16倍未満のペア数
        if pairs_counts[8] > 0:
            ratio = pairs_counts[32] / pairs_counts[8]
            assert ratio < 16.0, (
                f"ペア数比 {ratio:.1f} >= 16 (8seg: {pairs_counts[8]}, 32seg: {pairs_counts[32]})"
            )


# ====================================================================
# テストクラス: 16セグメント接触検出
# ====================================================================


class TestLargeScale16Segment:
    """16セグメント交差梁の接触テスト."""

    def test_16seg_converges(self):
        """16セグメント梁で接触付きソルバーが収束する."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        assert result.converged, "16セグメントでソルバーが収束しなかった"

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
        """接触法線力が非負（粘着なし）."""
        result, mgr, _ = _solve_large_problem(n_seg_a=16, n_seg_b=16)
        assert result.converged
        for pair in mgr.pairs:
            if pair.is_active():
                assert pair.state.p_n >= 0, f"p_n={pair.state.p_n} < 0"


# ====================================================================
# テストクラス: 32セグメント（broadphaseのみ）
# ====================================================================
# 注: 32セグメントのばねモデルでは構造剛性/接触剛性比の問題で
# 収束が不安定になる。収束テストは16セグメントまでとし、
# 32セグメントではbroadphase効率のみ検証する
# （TestBroadphaseEfficiency で実施済み）。


# ====================================================================
# テストクラス: スケーラビリティ
# ====================================================================


class TestScalability:
    """セグメント数の増加に対する動作検証."""

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
