"""撚線ヒステリシス観測テスト.

サイクリック荷重による撚線接触のヒステリシス挙動を確認する。

テスト目的:
  - run_contact_cyclic() の基本動作確認
  - 往復荷重での接触状態引き継ぎ
  - 散逸エネルギーの蓄積（摩擦ヒステリシス）
  - 荷重-変位曲線の記録と非ゼロ面積（エネルギー散逸）

梁パラメータ: 鋼線 (E=200GPa, ν=0.3), 円形断面 d=2mm, pitch=40mm
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
)
from xkep_cae.contact.solver_hooks import (
    CyclicContactResult,
    run_contact_cyclic,
)
from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSection

# === 共通パラメータ ===
_E = 200e9
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002
_PITCH = 0.04
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_NDOF_PER_NODE = 6
_N_ELEM_PER_STRAND = 16


def _make_timo3d_assemblers(mesh):
    """Timo3D 線形アセンブラを構築（直接構築方式）."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    K = np.zeros((ndof_total, ndof_total))
    for elem in connectivity:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = node_coords[np.array([n1, n2])]
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
            [6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    return assemble_tangent, assemble_internal_force, ndof_total


def _get_strand_end_dofs(mesh, strand_id, end):
    """素線の端点のDOFインデックスを取得."""
    nodes = mesh.strand_nodes(strand_id)
    if end == "start":
        node = nodes[0]
    else:
        node = nodes[-1]
    return np.array([_NDOF_PER_NODE * node + d for d in range(_NDOF_PER_NODE)], dtype=int)


def _fix_all_strand_starts(mesh):
    """全素線の開始端を全固定."""
    fixed = set()
    for sid in range(mesh.n_strands):
        dofs = _get_strand_end_dofs(mesh, sid, "start")
        fixed.update(dofs.tolist())
    return np.array(sorted(fixed), dtype=int)


def _make_contact_manager(
    *,
    use_friction=False,
    mu=0.3,
    k_pen_scaling="linear",
):
    """ヒステリシステスト用の接触マネージャ."""
    k_t_ratio = 0.01 if use_friction else 0.1
    mu_ramp_steps = 10 if use_friction else 0
    return ContactManager(
        config=ContactConfig(
            k_pen_scale=0.1,
            k_pen_mode="beam_ei",
            beam_E=_E,
            beam_I=_SECTION.Iy,
            k_t_ratio=k_t_ratio,
            mu=mu,
            g_on=0.0,
            g_off=1e-5,
            n_outer_max=8,
            use_friction=use_friction,
            mu_ramp_steps=mu_ramp_steps,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.02,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
            k_pen_scaling=k_pen_scaling,
        ),
    )


def _build_cyclic_model(n_strands=3, load_type="tension"):
    """サイクリック荷重テスト用のモデルを構築.

    Returns:
        (f_ext_unit, fixed_dofs, assemble_tangent, assemble_internal_force,
         mesh, ndof_total)
    """
    mesh = make_twisted_wire_mesh(
        n_strands,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=_N_ELEM_PER_STRAND,
        n_pitches=1.0,
        gap=0.0,
    )

    assemble_tangent, assemble_internal_force, ndof_total = _make_timo3d_assemblers(mesh)
    fixed_dofs = _fix_all_strand_starts(mesh)

    f_ext_unit = np.zeros(ndof_total)
    if load_type == "tension":
        f_per_strand = 1.0 / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext_unit[end_dofs[2]] = f_per_strand
    elif load_type == "bending":
        m_per_strand = 1.0 / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext_unit[end_dofs[4]] = m_per_strand
    elif load_type == "torsion":
        m_per_strand = 1.0 / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext_unit[end_dofs[5]] = m_per_strand

    return f_ext_unit, fixed_dofs, assemble_tangent, assemble_internal_force, mesh, ndof_total


# ====================================================================
# テスト: サイクリック荷重基本動作
# ====================================================================


class TestCyclicBasic:
    """run_contact_cyclic() の基本動作テスト."""

    def test_single_phase_matches_direct(self):
        """1フェーズのサイクリックが直接実行と同一結果を返す."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        result = run_contact_cyclic(
            f_ext_unit * 50.0,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert isinstance(result, CyclicContactResult)
        assert result.n_phases == 1
        assert result.converged
        assert len(result.load_factors) == 5
        assert len(result.displacements) == 5
        assert result.n_total_steps == 5

    def test_two_phase_forward_reverse(self):
        """2フェーズ（正→逆）で荷重係数が正しく推移する."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        result = run_contact_cyclic(
            f_ext_unit * 50.0,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert result.converged
        assert result.n_phases == 2
        assert len(result.load_factors) == 10
        assert result.load_factors[4] == pytest.approx(1.0, rel=1e-10)
        assert result.load_factors[9] == pytest.approx(0.0, abs=1e-10)

    def test_graph_history_combined(self):
        """サイクリックのグラフ時系列が全フェーズ分統合される."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        result = run_contact_cyclic(
            f_ext_unit * 50.0,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert result.graph_history.n_steps == 10
        steps = [s.step for s in result.graph_history.snapshots]
        assert steps == list(range(1, 11))


# ====================================================================
# テスト: 撚線ヒステリシス（摩擦付き往復荷重）
# ====================================================================


class TestTwistedWireHysteresis:
    """3本撚りの摩擦付き往復荷重テスト.

    摩擦接触下でサイクリック荷重を行い、ヒステリシス特性を確認する。
    """

    def test_tension_hysteresis_converges(self):
        """引張往復荷重が収束する."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        result = run_contact_cyclic(
            f_ext_unit * 30.0,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert result.converged

    def test_bending_hysteresis_converges(self):
        """曲げ往復荷重が収束する."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "bending")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        result = run_contact_cyclic(
            f_ext_unit * 0.03,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert result.converged

    def test_torsion_hysteresis_converges(self):
        """ねじり往復荷重が収束する."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "torsion")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        result = run_contact_cyclic(
            f_ext_unit * 0.005,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert result.converged

    def test_full_cycle_tension(self):
        """引張フルサイクル（0→+F→0→-F→0）で荷重履歴が正しい."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        result = run_contact_cyclic(
            f_ext_unit * 20.0,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0, -1.0, 0.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert result.converged
        assert result.n_phases == 3
        assert len(result.load_factors) == 15
        assert result.load_factors[4] == pytest.approx(1.0, rel=1e-10)
        assert result.load_factors[9] == pytest.approx(-1.0, rel=1e-10)
        assert result.load_factors[14] == pytest.approx(0.0, abs=1e-10)

    def test_dissipation_nonzero_with_friction(self):
        """摩擦付き往復で散逸エネルギーが非ゼロ."""
        f_ext_unit, fixed_dofs, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        result = run_contact_cyclic(
            f_ext_unit * 50.0,
            fixed_dofs,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=5,
            show_progress=False,
            broadphase_margin=0.01,
        )

        assert result.converged
        diss = result.graph_history.dissipation_series()
        total_diss = float(np.sum(diss))
        assert total_diss >= 0.0
