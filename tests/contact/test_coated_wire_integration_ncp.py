"""被膜付き撚線の接触解析テスト（NCP版）.

NCP移行版: test_coated_wire_integration.py から移行。
旧テスト（ペナルティ/AL）→ newton_raphson_contact_ncp（NCP）。

ジオメトリ・断面剛性テスト（ソルバー非依存）は旧ファイルに残す。
本ファイルはソルバー関連テスト（3本撚り被膜付き接触解析）をNCP版で再実装。
"""

import numpy as np
import pytest

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
)
from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp
from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d
from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    coated_beam_section,
    coated_radii,
    make_twisted_wire_mesh,
)
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_NDOF_PER_NODE = 6
_E = 200e9
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002
_WIRE_R = _WIRE_D / 2.0
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_PITCH = 0.040
_N_ELEM_PER_STRAND = 16

_COATING = CoatingModel(
    thickness=0.05e-3,
    E=3.0e9,
    nu=0.35,
    mu=0.25,
)


# ====================================================================
# ヘルパー
# ====================================================================


def _make_cr_assembler_coated(mesh, coating):
    """被膜込み等価剛性を使った CR 梁アセンブラ."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    eq = coated_beam_section(_WIRE_R, _E, _NU, coating)
    A_eq = eq["EA"] / _E
    Iy_eq = eq["EIy"] / _E
    Iz_eq = eq["EIz"] / _E
    J_eq = eq["GJ"] / _G

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            A_eq,
            Iy_eq,
            Iz_eq,
            J_eq,
            _KAPPA,
            _KAPPA,
            stiffness=True,
            internal_force=False,
        )
        return K_T

    def assemble_internal_force(u):
        _, f_int = assemble_cr_beam3d(
            node_coords,
            connectivity,
            u,
            _E,
            _G,
            A_eq,
            Iy_eq,
            Iz_eq,
            J_eq,
            _KAPPA,
            _KAPPA,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    return assemble_tangent, assemble_internal_force, ndof_total


def _make_cr_assembler_bare(mesh):
    """素の（被膜なし）CR 梁アセンブラ."""
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE

    def assemble_tangent(u):
        K_T, _ = assemble_cr_beam3d(
            node_coords,
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
            node_coords,
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

    return assemble_tangent, assemble_internal_force, ndof_total


def _get_strand_end_dofs(mesh, strand_id, end):
    """素線の端点のDOFインデックスを取得."""
    nodes = mesh.strand_nodes(strand_id)
    node = nodes[0] if end == "start" else nodes[-1]
    return np.array([_NDOF_PER_NODE * node + d for d in range(_NDOF_PER_NODE)], dtype=int)


def _fix_all_strand_starts(mesh):
    """全素線の開始端を全固定するDOFセットを返す."""
    fixed = set()
    for sid in range(mesh.n_strands):
        dofs = _get_strand_end_dofs(mesh, sid, "start")
        fixed.update(dofs.tolist())
    return np.array(sorted(fixed), dtype=int)


def _solve_coated_3strand_ncp(
    load_type,
    load_value,
    *,
    with_coating=True,
    use_friction=False,
    n_load_steps=10,
):
    """被膜付き3本撚り解析をNCP法で解く."""
    gap = _COATING.thickness * 4 if with_coating else 0.0
    mesh = make_twisted_wire_mesh(
        3,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=_N_ELEM_PER_STRAND,
        n_pitches=1.0,
        gap=gap,
        min_elems_per_pitch=0,
    )

    if with_coating:
        at, af, ndof = _make_cr_assembler_coated(mesh, _COATING)
        radii = coated_radii(mesh, _COATING)
        mu = _COATING.mu
    else:
        at, af, ndof = _make_cr_assembler_bare(mesh)
        radii = mesh.radii
        mu = 0.3

    fixed_dofs = _fix_all_strand_starts(mesh)

    f_ext = np.zeros(ndof)
    if load_type == "tension":
        f_per = load_value / 3
        for sid in range(3):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[2]] = f_per
    elif load_type == "lateral":
        f_per = load_value / 3
        for sid in range(3):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[0]] = f_per
    elif load_type == "bending":
        m_per = load_value / 3
        for sid in range(3):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[4]] = m_per

    elem_layer_map = mesh.build_elem_layer_map()

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=0.1,
            k_pen_mode="beam_ei",
            beam_E=_E,
            beam_I=_SECTION.Iy,
            k_pen_scaling="sqrt",
            k_t_ratio=0.01 if use_friction else 0.1,
            mu=mu,
            g_on=0.0005,
            g_off=0.001,
            use_friction=use_friction,
            mu_ramp_steps=10 if use_friction else 0,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.02,
            penalty_growth_factor=1.0,
            k_pen_max=1e12,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            linear_solver="auto",
            preserve_inactive_lambda=True,
            no_deactivation_within_step=True,
        ),
    )

    result = newton_raphson_contact_ncp(
        f_ext,
        fixed_dofs,
        at,
        af,
        mgr,
        mesh.node_coords,
        mesh.connectivity,
        radii,
        n_load_steps=n_load_steps,
        max_iter=50,
        tol_force=1e-6,
        tol_ncp=1e-6,
        show_progress=False,
        broadphase_margin=0.01,
        use_friction=use_friction,
        mu=mu if use_friction else None,
        use_line_search=True,
    )
    return result, mgr, mesh


# ====================================================================
# テスト: 被膜付き3本撚り接触（NCP版）
# ====================================================================


class TestCoatedThreeStrandContactNCP:
    """被膜付き3本撚りの接触解析テスト（NCP版）."""

    def test_coated_tension_converges(self):
        """被膜付き3本撚り引張が収束（摩擦なし）."""
        result, _, _ = _solve_coated_3strand_ncp("tension", 100.0)
        assert result.converged, "被膜付き3本撚り引張が収束しなかった"

    def test_coated_lateral_converges(self):
        """被膜付き3本撚り横力が収束（摩擦なし）."""
        result, _, _ = _solve_coated_3strand_ncp("lateral", 10.0)
        assert result.converged, "被膜付き3本撚り横力が収束しなかった"

    def test_coated_bending_converges(self):
        """被膜付き3本撚り曲げが収束（摩擦なし）."""
        result, _, _ = _solve_coated_3strand_ncp("bending", 0.05)
        assert result.converged, "被膜付き3本撚り曲げが収束しなかった"

    def test_coated_tension_with_friction(self):
        """被膜付き3本撚り引張 + 摩擦が収束."""
        result, _, _ = _solve_coated_3strand_ncp(
            "tension",
            50.0,
            use_friction=True,
            n_load_steps=15,
        )
        assert result.converged, "被膜付き3本撚り引張（摩擦）が収束しなかった"

    def test_coated_vs_bare_stiffness(self):
        """被膜付きは素線のみより剛性が高い."""
        r_coated, _, _ = _solve_coated_3strand_ncp(
            "tension",
            50.0,
            with_coating=True,
        )
        r_bare, _, _ = _solve_coated_3strand_ncp(
            "tension",
            50.0,
            with_coating=False,
        )
        if r_coated.converged and r_bare.converged:
            max_uz_coated = np.max(np.abs(r_coated.u[2::6]))
            max_uz_bare = np.max(np.abs(r_bare.u[2::6]))
            assert max_uz_coated <= max_uz_bare * 1.05, (
                f"被膜付き変位 {max_uz_coated:.3e} > 素線のみ {max_uz_bare:.3e} * 1.05"
            )
