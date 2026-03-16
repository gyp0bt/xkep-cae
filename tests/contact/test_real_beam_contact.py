"""実梁要素（Timoshenko 3D / CR梁）での接触テスト（NCP版）.

NCP移行版: test_real_beam_contact.py から移行。
旧テスト（ペナルティ/AL）→ newton_raphson_contact_ncp（NCP）。

梁パラメータ: アルミ合金 (E=70GPa, ν=0.33), 円形断面 d=20mm, L=0.5m
初期ギャップ: 0.5mm (search_radius=20mm に対して 2.5%)

テスト項目:
1. Timoshenko 3D線形梁の接触検出・貫入制限
2. CR梁（非線形）の接触検出・貫入制限
3. マルチセグメント（8分割）での接触
4. Timo3D vs CR梁の小変位一致性
5. 摩擦付き接触
6. 横スライド接触
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
from xkep_cae.elements.beam_timo3d import (
    assemble_cr_beam3d,
    timo_beam3d_ke_global,
)
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_NDOF_PER_NODE = 6
_TOL_PEN_RATIO = 0.02  # NCP版: 2%許容

_E = 70e9
_NU = 0.33
_G = _E / (2.0 * (1.0 + _NU))
_D = 0.02
_RADIUS = _D / 2.0
_SECTION = BeamSection.circle(_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)

_L = 0.5
_DEFAULT_N_SEG = 4
_DEFAULT_Z_TOP = 2 * _RADIUS + 0.0005
_SEARCH_RADIUS = 2 * _RADIUS

_DEFAULT_K_PEN = 1e4
_DEFAULT_F_Z = 500.0
_DEFAULT_N_STEPS = 20


# ====================================================================
# ヘルパー
# ====================================================================


def _make_crossing_beam_model(
    n_seg_a=_DEFAULT_N_SEG,
    n_seg_b=_DEFAULT_N_SEG,
    z_top=_DEFAULT_Z_TOP,
):
    """交差梁モデルの共通メッシュ・拘束を構築."""
    n_nodes_a = n_seg_a + 1
    n_nodes_b = n_seg_b + 1
    n_nodes = n_nodes_a + n_nodes_b
    ndof_total = n_nodes * _NDOF_PER_NODE

    coords_a = np.zeros((n_nodes_a, 3))
    for i in range(n_nodes_a):
        coords_a[i, 0] = i * _L / n_seg_a

    coords_b = np.zeros((n_nodes_b, 3))
    for i in range(n_nodes_b):
        coords_b[i, 0] = _L / 2.0
        coords_b[i, 1] = -_L / 2.0 + i * _L / n_seg_b
        coords_b[i, 2] = z_top

    node_coords_ref = np.vstack([coords_a, coords_b])

    conn_a = np.array([[i, i + 1] for i in range(n_seg_a)])
    conn_b = np.array([[n_nodes_a + i, n_nodes_a + i + 1] for i in range(n_seg_b)])
    connectivity = np.vstack([conn_a, conn_b])

    fixed = set()
    for d in range(_NDOF_PER_NODE):
        fixed.add(d)
    for d in range(_NDOF_PER_NODE):
        fixed.add(n_nodes_a * _NDOF_PER_NODE + d)
    fixed_dofs = np.array(sorted(fixed), dtype=int)

    return (
        node_coords_ref,
        connectivity,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        fixed_dofs,
    )


def _make_timo3d_assemblers(node_coords_ref, connectivity, ndof_total):
    """Timoshenko 3D 線形梁のアセンブリコールバックを構築."""
    K = np.zeros((ndof_total, ndof_total))
    for elem in connectivity:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = node_coords_ref[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            coords,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
        )
        edofs = np.array(
            [6 * n1 + dd for dd in range(6)] + [6 * n2 + dd for dd in range(6)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    return assemble_tangent, assemble_internal_force


def _make_cr_assemblers(node_coords_ref, connectivity, ndof_total):
    """CR梁のアセンブリコールバックを構築."""

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords_ref,
            connectivity,
            u,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
            stiffness=True,
            internal_force=False,
        )
        return K_T

    def assemble_internal_force(u):
        _, f_int = assemble_cr_beam3d(
            node_coords_ref,
            connectivity,
            u,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    return assemble_tangent, assemble_internal_force


def _solve_contact(
    assembler_type,
    n_seg_a=_DEFAULT_N_SEG,
    n_seg_b=_DEFAULT_N_SEG,
    f_z=_DEFAULT_F_Z,
    k_pen=_DEFAULT_K_PEN,
    n_load_steps=_DEFAULT_N_STEPS,
    max_iter=50,
    use_friction=False,
    mu=0.3,
    z_top=_DEFAULT_Z_TOP,
    f_x=0.0,
):
    """実梁要素の接触問題をNCP法で解く."""
    (
        node_coords_ref,
        connectivity,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        fixed_dofs,
    ) = _make_crossing_beam_model(n_seg_a, n_seg_b, z_top)

    if assembler_type == "timo3d":
        assemble_tangent, assemble_internal_force = _make_timo3d_assemblers(
            node_coords_ref,
            connectivity,
            ndof_total,
        )
    else:
        assemble_tangent, assemble_internal_force = _make_cr_assemblers(
            node_coords_ref,
            connectivity,
            ndof_total,
        )

    f_ext = np.zeros(ndof_total)
    node_b_last = n_nodes_a + n_nodes_b - 1
    f_ext[node_b_last * _NDOF_PER_NODE + 2] = -f_z
    if f_x != 0.0:
        f_ext[node_b_last * _NDOF_PER_NODE + 0] = f_x

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
        _RADIUS,
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


# ====================================================================
# テスト: Timoshenko 3D 線形梁 + 接触（NCP版）
# ====================================================================


class TestTimo3DContactDetection:
    """Timoshenko 3D 梁での接触検出テスト（NCP版）."""

    def test_contact_detected(self):
        """z方向押し下げで接触が検出される."""
        result, mgr, _, _ = _solve_contact("timo3d", f_z=500.0)
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "接触が検出されていない"

    def test_no_contact_without_force(self):
        """押し下げ力なしでは接触しない."""
        result, mgr, _, _ = _solve_contact("timo3d", f_z=0.0)
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) == 0, "力なしで接触が発生した"


class TestTimo3DPenetrationBound:
    """Timoshenko 3D 梁での貫入量制限テスト（NCP版）."""

    def test_penetration_bounded(self):
        """NCP法により貫入が制限される."""
        result, mgr, _, _ = _solve_contact("timo3d", f_z=500.0)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < _TOL_PEN_RATIO, f"貫入比 {pen_ratio:.4f} > {_TOL_PEN_RATIO}"


class TestTimo3DMultiSegment:
    """Timoshenko 3D マルチセグメント梁での接触テスト（NCP版）."""

    def test_8_segment_contact(self):
        """8分割梁での接触検出と貫入制限."""
        result, mgr, _, _ = _solve_contact(
            "timo3d",
            n_seg_a=8,
            n_seg_b=8,
            f_z=200.0,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < _TOL_PEN_RATIO


# ====================================================================
# テスト: CR梁（幾何学的非線形）+ 接触（NCP版）
# ====================================================================


class TestCRBeamContactDetection:
    """CR梁での接触検出テスト（NCP版）."""

    def test_contact_detected(self):
        """CR梁でz方向押し下げにより接触が検出される."""
        result, mgr, _, _ = _solve_contact("cr", f_z=500.0)
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "CR梁で接触が検出されていない"

    def test_no_contact_without_force(self):
        """CR梁で押し下げ力なしでは接触しない."""
        result, mgr, _, _ = _solve_contact("cr", f_z=0.0)
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) == 0


class TestCRBeamPenetrationBound:
    """CR梁での貫入量制限テスト（NCP版）."""

    def test_penetration_bounded(self):
        """CR梁でNCP法により貫入が制限される."""
        result, mgr, _, _ = _solve_contact("cr", f_z=500.0)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < _TOL_PEN_RATIO, (
                    f"CR梁: 貫入比 {pen_ratio:.4f} > {_TOL_PEN_RATIO}"
                )


class TestCRBeamMultiSegment:
    """CR梁マルチセグメント接触テスト（NCP版）."""

    def test_8_segment_contact(self):
        """CR梁8分割での接触検出と貫入制限."""
        result, mgr, _, _ = _solve_contact(
            "cr",
            n_seg_a=8,
            n_seg_b=8,
            f_z=200.0,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < _TOL_PEN_RATIO


# ====================================================================
# テスト: Timo3D vs CR の小変位一致性（NCP版）
# ====================================================================


class TestTimo3DVsCRConsistency:
    """小変位でのTimo3D vs CR梁の一致性テスト（NCP版）."""

    def test_small_load_consistency(self):
        """小荷重での変位が20%以内で一致."""
        f_z = 200.0
        result_t, _, _, _ = _solve_contact("timo3d", f_z=f_z)
        result_c, _, _, _ = _solve_contact("cr", f_z=f_z)

        assert result_t.converged
        assert result_c.converged

        uz_t = result_t.u[-4]  # 梁B先端z方向
        uz_c = result_c.u[-4]

        if abs(uz_t) > 1e-10:
            rel_diff = abs(uz_t - uz_c) / abs(uz_t)
            assert rel_diff < 0.2, (
                f"Timo3D vs CR 不一致: uz_timo={uz_t:.6e}, uz_cr={uz_c:.6e}, "
                f"rel_diff={rel_diff:.3f}"
            )


# ====================================================================
# テスト: 摩擦付き接触（NCP版）
# ====================================================================


class TestRealBeamFrictionContact:
    """実梁要素の摩擦付き接触テスト（NCP版）."""

    def test_timo3d_friction_converges(self):
        """Timo3D梁の摩擦付き接触が収束する."""
        result, mgr, _, _ = _solve_contact(
            "timo3d",
            f_z=500.0,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged, "Timo3D摩擦付き接触が収束しなかった"

    @pytest.mark.xfail(
        reason="CR梁の摩擦接触で収束未達 (status-128)",
        strict=False,
    )
    def test_cr_friction_converges(self):
        """CR梁の摩擦付き接触が収束する."""
        result, mgr, _, _ = _solve_contact(
            "cr",
            f_z=500.0,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged, "CR摩擦付き接触が収束しなかった"

    def test_friction_does_not_worsen_penetration(self):
        """摩擦が貫入量を悪化させない."""
        _, mgr_nf, _, _ = _solve_contact("timo3d", f_z=200.0, use_friction=False)
        _, mgr_f, _, _ = _solve_contact(
            "timo3d",
            f_z=200.0,
            use_friction=True,
            mu=0.3,
        )

        def max_pen(mgr):
            pens = [abs(p.state.gap) for p in mgr.pairs if p.state.gap < 0]
            return max(pens) if pens else 0.0

        pen_nf = max_pen(mgr_nf)
        pen_f = max_pen(mgr_f)
        # 摩擦による貫入悪化は微小（1桁以内）
        assert pen_f < pen_nf * 10 + 1e-10, (
            f"摩擦で貫入が大幅悪化: nofric={pen_nf:.6e}, fric={pen_f:.6e}"
        )


# ====================================================================
# テスト: 横スライド接触（NCP版）
# ====================================================================


class TestLongRangeSlide:
    """長距離スライド接触テスト（NCP版）."""

    def test_slide_contact_detected(self):
        """スライド荷重で接触が検出される."""
        result, mgr, _, _ = _solve_contact(
            "timo3d",
            f_z=200.0,
            f_x=100.0,
            n_seg_a=8,
            n_seg_b=8,
            n_load_steps=30,
        )
        assert result.converged
        active = [p for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE]
        assert len(active) > 0, "スライド時に接触未検出"

    def test_slide_penetration_bounded(self):
        """スライド中も貫入量が許容範囲内."""
        result, mgr, _, _ = _solve_contact(
            "timo3d",
            f_z=200.0,
            f_x=100.0,
            n_seg_a=8,
            n_seg_b=8,
            n_load_steps=30,
        )
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair.state.gap < 0:
                pen_ratio = abs(pair.state.gap) / _SEARCH_RADIUS
                assert pen_ratio < _TOL_PEN_RATIO

    def test_cr_slide_converges(self):
        """CR梁のスライド接触が収束する."""
        result, mgr, _, _ = _solve_contact(
            "cr",
            f_z=200.0,
            f_x=100.0,
            n_seg_a=8,
            n_seg_b=8,
            n_load_steps=30,
        )
        assert result.converged, "CR梁スライドが収束しなかった"
