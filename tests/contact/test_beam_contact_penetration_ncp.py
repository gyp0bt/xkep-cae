"""梁梁接触の貫入テスト（NCP版）.

NCP移行版: test_beam_contact_penetration.py から移行。
旧テスト（ペナルティ/AL）→ newton_raphson_contact_ncp（NCP）。

テスト項目:
1. 接触検出: 梁同士が接触していること
2. 貫入量制限: ギャップが search_radius の 2% 以下
3. 法線力正値性: 接触中の法線力が正
4. 摩擦有無の影響: 法線方向の貫入に大差がない
5. 変位履歴: 荷重ステップごとの単調性
6. マルチセグメント梁: 複数要素分割での貫入制限
7. 横スライド接触: 接触維持しながら横方向にスライド
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp

pytestmark = pytest.mark.slow

# ====================================================================
# ヘルパー: 交差梁モデル（貫入テスト用）
# ====================================================================

_NDOF_PER_NODE = 6

# NCP版では2%許容（鞍点系の特性により旧ペナルティ法と若干異なる）
_TOL_PEN_RATIO = 0.02


def _make_crossing_beams(
    k_spring: float = 1e4,
    z_top: float = 0.082,
    radii: float = 0.04,
):
    """交差梁ばねモデル（貫入テスト用）."""
    n_nodes = 4
    ndof_total = n_nodes * _NDOF_PER_NODE

    node_coords_ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -0.5, z_top],
            [0.5, 0.5, z_top],
        ]
    )
    connectivity = np.array([[0, 1], [2, 3]])

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                K[d0, d0] += k_spring
                K[d0, d1] -= k_spring
                K[d1, d0] -= k_spring
                K[d1, d1] += k_spring
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_spring * delta
                f_int[d1] += k_spring * delta
        return f_int

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    )


def _make_multi_segment_crossing_beams(
    n_seg_a: int = 4,
    n_seg_b: int = 4,
    k_spring: float = 1e4,
    z_top: float = 0.082,
    radii: float = 0.04,
):
    """マルチセグメント交差梁ばねモデル."""
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

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                K[d0, d0] += k_spring
                K[d0, d1] -= k_spring
                K[d1, d0] -= k_spring
                K[d1, d1] += k_spring
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_spring * delta
                f_int[d1] += k_spring * delta
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


def _fixed_dofs_penetration():
    """貫入テスト用の拘束DOF（単一セグメント版）."""
    fixed = []
    for d in range(_NDOF_PER_NODE):
        fixed.append(0 * _NDOF_PER_NODE + d)
    for d in range(_NDOF_PER_NODE):
        if d != 0:
            fixed.append(1 * _NDOF_PER_NODE + d)
    for d in range(_NDOF_PER_NODE):
        fixed.append(2 * _NDOF_PER_NODE + d)
    for d in range(_NDOF_PER_NODE):
        if d not in (1, 2):
            fixed.append(3 * _NDOF_PER_NODE + d)
    return np.array(fixed, dtype=int)


def _fixed_dofs_multi_segment(n_nodes_a, n_nodes_b):
    """マルチセグメント版の拘束DOF."""
    fixed = set()
    n_nodes = n_nodes_a + n_nodes_b

    for i in range(n_nodes):
        for d in range(3, _NDOF_PER_NODE):
            fixed.add(i * _NDOF_PER_NODE + d)

    for d in range(3):
        fixed.add(0 * _NDOF_PER_NODE + d)
    node_a_last = n_nodes_a - 1
    for d in range(1, 3):
        fixed.add(node_a_last * _NDOF_PER_NODE + d)
    node_b_first = n_nodes_a
    for d in range(3):
        fixed.add(node_b_first * _NDOF_PER_NODE + d)
    node_b_last = n_nodes_a + n_nodes_b - 1
    fixed.add(node_b_last * _NDOF_PER_NODE + 0)

    return np.array(sorted(fixed), dtype=int)


def _solve_penetration_problem_ncp(
    f_x: float = 10.0,
    f_y: float = 5.0,
    f_z: float = 50.0,
    k_spring: float = 1e4,
    k_pen: float = 1e5,
    z_top: float = 0.082,
    radii: float = 0.04,
    n_load_steps: int = 20,
    max_iter: int = 50,
    use_friction: bool = False,
    mu: float = 0.3,
):
    """梁梁貫入問題をNCP法で解く."""
    (
        node_coords_ref,
        connectivity,
        radii_val,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    ) = _make_crossing_beams(k_spring=k_spring, z_top=z_top, radii=radii)

    f_ext = np.zeros(ndof_total)
    f_ext[1 * _NDOF_PER_NODE + 0] = f_x
    f_ext[3 * _NDOF_PER_NODE + 1] = f_y
    f_ext[3 * _NDOF_PER_NODE + 2] = -f_z

    fixed_dofs = _fixed_dofs_penetration()

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=k_pen,
            k_t_ratio=0.1,
            mu=mu,
            g_on=0.0,
            g_off=1e-4,
            use_friction=use_friction,
            mu_ramp_steps=3 if use_friction else 0,
            use_geometric_stiffness=True,
            tol_penetration_ratio=_TOL_PEN_RATIO,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
        ),
    )

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
        use_friction=use_friction,
        mu=mu if use_friction else None,
    )

    return result, mgr, ndof_total, node_coords_ref


def _solve_multi_segment_problem_ncp(
    n_seg_a: int = 4,
    n_seg_b: int = 4,
    f_x: float = 10.0,
    f_y: float = 5.0,
    f_z: float = 50.0,
    k_spring: float = 1e4,
    k_pen: float = 1e5,
    z_top: float = 0.082,
    radii: float = 0.04,
    n_load_steps: int = 20,
    max_iter: int = 50,
):
    """マルチセグメント梁の貫入問題をNCP法で解く."""
    (
        node_coords_ref,
        connectivity,
        radii_val,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        assemble_tangent,
        assemble_internal_force,
    ) = _make_multi_segment_crossing_beams(
        n_seg_a=n_seg_a,
        n_seg_b=n_seg_b,
        k_spring=k_spring,
        z_top=z_top,
        radii=radii,
    )

    f_ext = np.zeros(ndof_total)
    node_a_last = n_nodes_a - 1
    node_b_last = n_nodes_a + n_nodes_b - 1
    f_ext[node_a_last * _NDOF_PER_NODE + 0] = f_x
    f_ext[node_b_last * _NDOF_PER_NODE + 1] = f_y
    f_ext[node_b_last * _NDOF_PER_NODE + 2] = -f_z

    fixed_dofs = _fixed_dofs_multi_segment(n_nodes_a, n_nodes_b)

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=k_pen,
            k_t_ratio=0.1,
            g_on=0.0,
            g_off=1e-4,
            use_geometric_stiffness=True,
            tol_penetration_ratio=_TOL_PEN_RATIO,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
        ),
    )

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

    return result, mgr, radii_val


def _max_penetration_ratio(mgr):
    """全ペアの最大貫入比を返す."""
    max_ratio = 0.0
    for pair in mgr.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        if pair.state.gap >= 0.0:
            continue
        sr = pair.search_radius
        if sr < 1e-30:
            continue
        ratio = abs(pair.state.gap) / sr
        max_ratio = max(max_ratio, ratio)
    return max_ratio


# ====================================================================
# テスト: 接触検出
# ====================================================================


class TestBeamContactDetectionNCP:
    """梁梁接触の検出検証（NCP版）."""

    def test_contact_detected_with_push_down(self):
        """z方向押し下げ力で接触が検出される."""
        result, mgr, _, _ = _solve_penetration_problem_ncp(f_z=50.0)
        assert result.converged, "NCPソルバーが収束しなかった"
        assert mgr.n_active > 0, "接触が検出されなかった"

    def test_no_contact_without_push_down(self):
        """押し下げ力がなければ接触は検出されない."""
        result, mgr, _, _ = _solve_penetration_problem_ncp(f_z=0.0)
        assert result.converged
        assert mgr.n_active == 0, "力なしで接触が検出された"


# ====================================================================
# テスト: 貫入量制限
# ====================================================================


class TestPenetrationBoundNCP:
    """貫入量の制限検証（NCP版）."""

    def test_penetration_within_tolerance(self):
        """貫入量が search_radius の 2% 以下."""
        result, mgr, _, _ = _solve_penetration_problem_ncp(
            f_z=50.0,
            k_pen=1e5,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, (
            f"貫入超過: pen_ratio={pen_ratio:.6e}, tol={_TOL_PEN_RATIO}"
        )

    def test_penetration_with_large_force(self):
        """大きな押し下げ力でも貫入量が許容範囲内."""
        result, mgr, _, _ = _solve_penetration_problem_ncp(
            f_z=200.0,
            k_pen=1e5,
            n_load_steps=30,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, (
            f"大荷重で貫入超過: pen_ratio={pen_ratio:.6e}, tol={_TOL_PEN_RATIO}"
        )

    def test_higher_penalty_reduces_penetration(self):
        """ペナルティ剛性が高いほど貫入が小さい."""
        penetrations = {}
        for k in [1e4, 1e5, 1e6]:
            result, mgr, _, _ = _solve_penetration_problem_ncp(
                f_z=50.0,
                k_pen=k,
            )
            assert result.converged, f"k_pen={k:.0e}で収束しなかった"

            for pair in mgr.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    penetrations[k] = abs(min(pair.state.gap, 0.0))
                    break

        assert len(penetrations) >= 2, f"接触ペアが不足: {len(penetrations)}"

        k_values = sorted(penetrations.keys())
        for i in range(1, len(k_values)):
            pen_low = penetrations[k_values[i - 1]]
            pen_high = penetrations[k_values[i]]
            assert pen_high <= pen_low + 1e-10, (
                f"ペナルティ増加で貫入が増加: "
                f"k={k_values[i - 1]:.0e}→{pen_low:.6e}, "
                f"k={k_values[i]:.0e}→{pen_high:.6e}"
            )


# ====================================================================
# テスト: 法線力
# ====================================================================


class TestNormalForceNCP:
    """接触法線力の検証（NCP版）."""

    def test_normal_force_positive(self):
        """接触中の法線力が正値（圧縮のみ）."""
        result, mgr, _, _ = _solve_penetration_problem_ncp(f_z=50.0)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                assert pair.state.p_n >= 0.0, f"引張接触力: p_n={pair.state.p_n:.6f}"

    def test_normal_force_increases_with_push(self):
        """押し下げ力が大きいほど法線力が大きい."""
        pn_values = {}
        for f_z in [30.0, 50.0, 100.0]:
            result, mgr, _, _ = _solve_penetration_problem_ncp(f_z=f_z)
            assert result.converged

            for pair in mgr.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    pn_values[f_z] = pair.state.p_n
                    break

        assert len(pn_values) >= 2, "接触検出が不十分"

        fz_sorted = sorted(pn_values.keys())
        for i in range(1, len(fz_sorted)):
            assert pn_values[fz_sorted[i]] >= pn_values[fz_sorted[i - 1]] - 1e-8, (
                f"法線力が非単調: f_z={fz_sorted[i - 1]:.0f}→"
                f"p_n={pn_values[fz_sorted[i - 1]]:.4f}, "
                f"f_z={fz_sorted[i]:.0f}→p_n={pn_values[fz_sorted[i]]:.4f}"
            )


# ====================================================================
# テスト: 摩擦の影響
# ====================================================================


class TestFrictionPenetrationEffectNCP:
    """摩擦の有無による貫入量への影響（NCP版）."""

    def test_penetration_bounded_with_friction(self):
        """摩擦ありでも貫入量が許容範囲内."""
        result, mgr, _, _ = _solve_penetration_problem_ncp(
            f_z=50.0,
            use_friction=True,
            mu=0.3,
            k_pen=1e5,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, f"摩擦あり貫入超過: pen_ratio={pen_ratio:.6e}"

    def test_friction_does_not_worsen_penetration(self):
        """摩擦の有無で法線方向の貫入に大差がない."""
        result_nf, mgr_nf, _, _ = _solve_penetration_problem_ncp(
            f_z=50.0,
            use_friction=False,
        )
        result_f, mgr_f, _, _ = _solve_penetration_problem_ncp(
            f_z=50.0,
            use_friction=True,
            mu=0.3,
        )
        assert result_nf.converged
        assert result_f.converged

        gap_nf = None
        gap_f = None
        for pair in mgr_nf.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                gap_nf = pair.state.gap
                break
        for pair in mgr_f.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                gap_f = pair.state.gap
                break

        if gap_nf is not None and gap_f is not None:
            assert abs(gap_nf - gap_f) < 1e-3, (
                f"摩擦による貫入変化が過大: gap_nofric={gap_nf:.6e}, gap_fric={gap_f:.6e}"
            )


# ====================================================================
# テスト: 変位履歴
# ====================================================================


class TestDisplacementHistoryNCP:
    """荷重ステップごとの変位履歴検証（NCP版）."""

    def test_z_displacement_progresses_downward(self):
        """梁B端部のz方向変位が下方に進行する."""
        result, _, _, _ = _solve_penetration_problem_ncp(
            f_z=50.0,
            n_load_steps=10,
        )
        assert result.converged

        uz_history = [u_step[3 * _NDOF_PER_NODE + 2] for u_step in result.displacement_history]

        assert uz_history[-1] < 0.0, f"最終z変位が非負: uz={uz_history[-1]:.6e}"
        assert abs(uz_history[-1]) > abs(uz_history[0]), "最終変位が初期変位より小さい"

    def test_x_tension_positive(self):
        """梁A端部のx方向変位が正（張力方向）."""
        result, _, _, _ = _solve_penetration_problem_ncp(f_x=10.0, f_z=50.0)
        assert result.converged

        ux_node1 = result.u[1 * _NDOF_PER_NODE + 0]
        assert ux_node1 > 0.0, f"x変位が非正: ux_node1={ux_node1:.6e}"


# ====================================================================
# テスト: マルチセグメント梁
# ====================================================================


class TestMultiSegmentBeamPenetrationNCP:
    """マルチセグメント梁での貫入テスト（NCP版）."""

    def test_multi_segment_contact_detected(self):
        """マルチセグメント梁で接触が検出される."""
        result, mgr, _ = _solve_multi_segment_problem_ncp(
            n_seg_a=4,
            n_seg_b=4,
            f_z=50.0,
        )
        assert result.converged, "マルチセグメントNCPが収束しなかった"
        assert mgr.n_active > 0, "マルチセグメント梁で接触が検出されなかった"

    def test_multi_segment_penetration_within_tolerance(self):
        """マルチセグメント梁でも貫入量が許容範囲内."""
        result, mgr, _ = _solve_multi_segment_problem_ncp(
            n_seg_a=4,
            n_seg_b=4,
            f_z=50.0,
            k_pen=1e5,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, f"マルチセグメント貫入超過: pen_ratio={pen_ratio:.6e}"

    def test_multi_segment_large_force(self):
        """マルチセグメント梁で大荷重でも貫入が許容範囲内."""
        result, mgr, _ = _solve_multi_segment_problem_ncp(
            n_seg_a=4,
            n_seg_b=4,
            f_z=100.0,
            k_pen=1e5,
            n_load_steps=30,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, (
            f"マルチセグメント大荷重貫入超過: pen_ratio={pen_ratio:.6e}"
        )


# ====================================================================
# テスト: 横スライド接触
# ====================================================================


class TestSlidingContactNCP:
    """横スライド接触テスト（NCP版）."""

    def _solve_sliding_problem_ncp(
        self,
        f_z: float = 50.0,
        f_slide_x: float = 20.0,
        use_friction: bool = True,
        mu: float = 0.3,
        radii: float = 0.04,
        k_pen: float = 1e5,
        n_load_steps: int = 20,
    ):
        """横スライド接触問題をNCP法で解く."""
        k_spring = 1e4
        z_top = 0.082

        (
            node_coords_ref,
            connectivity,
            radii_val,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_crossing_beams(k_spring=k_spring, z_top=z_top, radii=radii)

        f_ext = np.zeros(ndof_total)
        f_ext[3 * _NDOF_PER_NODE + 0] = f_slide_x
        f_ext[3 * _NDOF_PER_NODE + 1] = 5.0
        f_ext[3 * _NDOF_PER_NODE + 2] = -f_z

        fixed = []
        for d in range(_NDOF_PER_NODE):
            fixed.append(0 * _NDOF_PER_NODE + d)
        for d in range(_NDOF_PER_NODE):
            fixed.append(1 * _NDOF_PER_NODE + d)
        for d in range(_NDOF_PER_NODE):
            fixed.append(2 * _NDOF_PER_NODE + d)
        for d in range(3, _NDOF_PER_NODE):
            fixed.append(3 * _NDOF_PER_NODE + d)
        fixed_dofs = np.array(fixed, dtype=int)

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=k_pen,
                k_t_ratio=0.1,
                mu=mu,
                g_on=0.0,
                g_off=1e-4,
                use_friction=use_friction,
                mu_ramp_steps=3 if use_friction else 0,
                use_geometric_stiffness=True,
                tol_penetration_ratio=_TOL_PEN_RATIO,
                penalty_growth_factor=2.0,
                k_pen_max=1e12,
                # S3改良6: 適応時間増分制御
                adaptive_timestepping=True,
            ),
        )

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
            max_iter=80,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            k_pen=k_pen,
            use_friction=use_friction,
            mu=mu if use_friction else None,
            adaptive_timestepping=True,
        )

        return result, mgr, ndof_total

    def test_sliding_contact_detected(self):
        """横スライド時に接触が検出される."""
        result, mgr, _ = self._solve_sliding_problem_ncp()
        assert result.converged, "スライドNCPが収束しなかった"
        assert mgr.n_active > 0, "スライド時に接触が検出されなかった"

    def test_sliding_penetration_within_tolerance(self):
        """横スライド中も貫入量が許容範囲内."""
        result, mgr, _ = self._solve_sliding_problem_ncp()
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, f"スライド時貫入超過: pen_ratio={pen_ratio:.6e}"

    def test_sliding_displacement_has_x_component(self):
        """スライド時にx方向変位が生じる."""
        result, _, ndof_total = self._solve_sliding_problem_ncp(f_slide_x=20.0)
        assert result.converged

        ux_node3 = result.u[3 * _NDOF_PER_NODE + 0]
        assert ux_node3 > 0.0, f"スライド方向のx変位が非正: ux={ux_node3:.6e}"

    def test_sliding_friction_both_converge(self):
        """摩擦なし/ありの両方でスライド解が収束する."""
        result_nf, _, _ = self._solve_sliding_problem_ncp(
            use_friction=False,
            f_slide_x=20.0,
        )
        result_f, _, _ = self._solve_sliding_problem_ncp(
            use_friction=True,
            mu=0.3,
            f_slide_x=20.0,
        )
        assert result_nf.converged, "摩擦なしスライドが収束しなかった"
        assert result_f.converged, "摩擦ありスライドが収束しなかった"
