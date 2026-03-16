"""撚線モデルの多点接触テスト（NCP版）.

NCP移行版: test_twisted_wire_contact.py から移行。
旧テスト（ペナルティ/AL）→ newton_raphson_contact_ncp（NCP）。

3本・7本撚線の主要テストシナリオをNCPソルバーで検証。
梁パラメータ: 鋼線 (E=200GPa, ν=0.3), 円形断面 d=2mm, pitch=40mm
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
from xkep_cae.mesh.twisted_wire import (
    TwistedWireMesh,
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


# ====================================================================
# ヘルパー
# ====================================================================


def _make_cr_assemblers(mesh: TwistedWireMesh):
    """CR梁のアセンブリコールバックを構築."""
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


def _make_timo3d_assemblers(mesh: TwistedWireMesh):
    """Timoshenko 3D線形梁のアセンブリコールバックを構築."""
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


def _get_strand_end_dofs(mesh: TwistedWireMesh, strand_id: int, end: str):
    """素線の端点のDOFインデックスを取得."""
    nodes = mesh.strand_nodes(strand_id)
    node = nodes[0] if end == "start" else nodes[-1]
    return np.array([_NDOF_PER_NODE * node + d for d in range(_NDOF_PER_NODE)], dtype=int)


def _fix_all_strand_starts(mesh: TwistedWireMesh) -> np.ndarray:
    """全素線の開始端を全固定するDOFセットを返す."""
    fixed = set()
    for sid in range(mesh.n_strands):
        dofs = _get_strand_end_dofs(mesh, sid, "start")
        fixed.update(dofs.tolist())
    return np.array(sorted(fixed), dtype=int)


def _solve_twisted_wire(
    n_strands: int,
    load_type: str,
    load_value: float,
    *,
    n_pitches: float = 1.0,
    assembler_type: str = "cr",
    use_friction: bool = False,
    mu: float = 0.3,
    n_load_steps: int = 15,
    max_iter: int = 50,
    gap: float = 0.0,
    n_elems_per_strand: int = _N_ELEM_PER_STRAND,
    exclude_same_layer: bool = True,
    line_contact: bool = False,
    n_gauss: int = 3,
):
    """撚線の接触問題をNCP法で解く."""
    mesh = make_twisted_wire_mesh(
        n_strands,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=n_elems_per_strand,
        n_pitches=n_pitches,
        gap=gap,
    )

    if assembler_type == "cr":
        assemble_tangent, assemble_internal_force, ndof_total = _make_cr_assemblers(mesh)
    else:
        assemble_tangent, assemble_internal_force, ndof_total = _make_timo3d_assemblers(mesh)

    fixed_dofs = _fix_all_strand_starts(mesh)

    f_ext = np.zeros(ndof_total)

    if load_type == "tension":
        f_per = load_value / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[2]] = f_per
    elif load_type == "torsion":
        m_per = load_value / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[5]] = m_per
    elif load_type == "bending":
        m_per = load_value / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[4]] = m_per
    elif load_type == "lateral":
        f_per = load_value / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[0]] = f_per
    else:
        raise ValueError(f"未知の荷重タイプ: {load_type}")

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
            exclude_same_layer=exclude_same_layer,
            midpoint_prescreening=True,
            linear_solver="auto",
            preserve_inactive_lambda=True,
            no_deactivation_within_step=True,
            line_contact=line_contact,
            n_gauss=n_gauss,
        ),
    )

    result = newton_raphson_contact_ncp(
        f_ext,
        fixed_dofs,
        assemble_tangent,
        assemble_internal_force,
        mgr,
        mesh.node_coords,
        mesh.connectivity,
        mesh.radii,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        tol_force=1e-6,
        tol_ncp=1e-6,
        show_progress=False,
        broadphase_margin=0.01,
        use_friction=use_friction,
        mu=mu if use_friction else None,
        line_contact=line_contact,
        n_gauss=n_gauss if line_contact else None,
        use_line_search=True,
    )

    return result, mgr, mesh


def _count_active_pairs(mgr: ContactManager) -> int:
    """有効な接触ペア数をカウント."""
    return sum(1 for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE)


def _max_penetration_ratio(mgr: ContactManager) -> float:
    """最大貫入比を計算."""
    max_pen = 0.0
    for p in mgr.pairs:
        if p.state.status == ContactStatus.INACTIVE:
            continue
        if p.state.gap < 0:
            pen = abs(p.state.gap) / (p.radius_a + p.radius_b)
            if pen > max_pen:
                max_pen = pen
    return max_pen


# ====================================================================
# テスト: 3本撚り基本接触（NCP版）
# ====================================================================


class TestThreeStrandBasicContact:
    """3本撚りの基本接触テスト（NCP版）."""

    def test_tension_converges(self):
        """3本: 引張荷重で収束."""
        result, mgr, _ = _solve_twisted_wire(3, "tension", 100.0)
        assert result.converged, "3本引張が収束しなかった"

    def test_lateral_converges(self):
        """3本: 横力荷重で収束."""
        result, mgr, _ = _solve_twisted_wire(3, "lateral", 10.0)
        assert result.converged, "3本横力が収束しなかった"

    def test_bending_converges(self):
        """3本: 曲げ荷重で収束."""
        result, mgr, _ = _solve_twisted_wire(3, "bending", 0.1)
        assert result.converged, "3本曲げが収束しなかった"


# ====================================================================
# テスト: 3本撚り摩擦付き（NCP版）
# ====================================================================


class TestThreeStrandFriction:
    """3本撚りの摩擦付き接触テスト（NCP版）."""

    def test_friction_tension(self):
        """3本: 摩擦付き引張で収束."""
        result, mgr, _ = _solve_twisted_wire(
            3,
            "tension",
            100.0,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged, "3本摩擦引張が収束しなかった"

    def test_friction_lateral(self):
        """3本: 摩擦付き横力で収束."""
        result, mgr, _ = _solve_twisted_wire(
            3,
            "lateral",
            10.0,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged, "3本摩擦横力が収束しなかった"

    def test_friction_bending(self):
        """3本: 摩擦付き曲げで収束."""
        result, mgr, _ = _solve_twisted_wire(
            3,
            "bending",
            0.1,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged, "3本摩擦曲げが収束しなかった"


# ====================================================================
# テスト: 3本撚り Line contact（NCP版）
# ====================================================================


class TestThreeStrandLineContact:
    """3本撚りのLine-to-line Gauss積分テスト（NCP版）."""

    def test_line_contact_tension(self):
        """3本: Line contact引張で収束."""
        result, mgr, _ = _solve_twisted_wire(
            3,
            "tension",
            100.0,
            line_contact=True,
            n_gauss=3,
        )
        assert result.converged, "3本Line contact引張が収束しなかった"

    def test_line_contact_bending(self):
        """3本: Line contact曲げで収束."""
        result, mgr, _ = _solve_twisted_wire(
            3,
            "bending",
            0.1,
            line_contact=True,
            n_gauss=3,
        )
        assert result.converged, "3本Line contact曲げが収束しなかった"

    def test_line_contact_friction(self):
        """3本: Line contact + 摩擦で収束."""
        result, mgr, _ = _solve_twisted_wire(
            3,
            "tension",
            100.0,
            line_contact=True,
            n_gauss=3,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged, "3本Line contact摩擦が収束しなかった"


# ====================================================================
# テスト: 7本撚り（NCP版）
# ====================================================================


class TestSevenStrand:
    """7本撚りのNCP版テスト.

    旧ソルバーのTestSevenStrandImprovedSolverの移行版。
    NCP推奨パラメータで収束を検証。
    """

    def test_7strand_tension(self):
        """7本: 引張荷重で収束."""
        result, mgr, _ = _solve_twisted_wire(
            7,
            "tension",
            100.0,
            gap=0.0005,
        )
        assert result.converged, "7本引張が収束しなかった"

    def test_7strand_torsion(self):
        """7本: ねじり荷重で収束."""
        result, mgr, _ = _solve_twisted_wire(
            7,
            "torsion",
            0.1,
            gap=0.0005,
        )
        assert result.converged, "7本ねじりが収束しなかった"

    def test_7strand_bending(self):
        """7本: 曲げ荷重で収束."""
        result, mgr, _ = _solve_twisted_wire(
            7,
            "bending",
            0.05,
            gap=0.0005,
        )
        assert result.converged, "7本曲げが収束しなかった"


# ====================================================================
# テスト: 接触力の検証（NCP版）
# ====================================================================


class TestContactForceVerification:
    """接触力の物理的妥当性検証（NCP版）."""

    def test_contact_force_positive(self):
        """3本引張時の法線接触力が正（圧縮のみ）."""
        result, mgr, _ = _solve_twisted_wire(3, "tension", 100.0)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                assert pair.state.p_n >= 0.0, f"引張接触力: p_n={pair.state.p_n:.6f}"

    def test_penetration_bounded(self):
        """3本引張時の貫入量が2%以下."""
        result, mgr, _ = _solve_twisted_wire(3, "tension", 100.0)
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < 0.02, f"貫入超過: pen_ratio={pen_ratio:.6e}"

    def test_7strand_contact_detected(self):
        """7本引張時に接触ペアが検出される."""
        result, mgr, _ = _solve_twisted_wire(
            7,
            "tension",
            100.0,
            gap=0.0005,
        )
        assert result.converged
        n_active = _count_active_pairs(mgr)
        assert n_active > 0, "7本で接触が検出されなかった"


# ====================================================================
# テスト: 7本撚り Line contact（NCP版）
# ====================================================================


class TestSevenStrandLineContact:
    """7本撚りのLine-to-line Gauss積分テスト（NCP版）."""

    def test_line_contact_tension(self):
        """7本: Line contact引張で収束."""
        result, mgr, _ = _solve_twisted_wire(
            7,
            "tension",
            100.0,
            gap=0.0005,
            line_contact=True,
            n_gauss=3,
        )
        assert result.converged, "7本Line contact引張が収束しなかった"


# ====================================================================
# テスト: 摩擦付き7本撚り（NCP版）
# ====================================================================


class TestSevenStrandFriction:
    """7本撚りの摩擦付き接触テスト（NCP版）."""

    def test_friction_tension(self):
        """7本: 摩擦付き引張で収束."""
        result, mgr, _ = _solve_twisted_wire(
            7,
            "tension",
            100.0,
            gap=0.0005,
            use_friction=True,
            mu=0.3,
        )
        assert result.converged, "7本摩擦引張が収束しなかった"
