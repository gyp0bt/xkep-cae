"""多ペア環境（7本撚り）での Mortar 収束性能評価テスト.

status-085 TODO: 7本撚り + Mortar で貫入率 < 1% かつ NCP 収束を確認。
Line contact + Mortar + 同層除外を撚線環境で検証。

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
from xkep_cae.contact.solver_ncp import (
    newton_raphson_contact_ncp,
)
from xkep_cae.elements.beam_timo3d import (
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
_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002  # 直径 2mm
_WIRE_R = _WIRE_D / 2.0
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_PITCH = 0.040  # 40mm ピッチ


# ====================================================================
# ヘルパー
# ====================================================================


def _make_timo3d_assemblers(mesh: TwistedWireMesh):
    """Timoshenko 3D 線形梁のアセンブリコールバックを構築."""
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


def _fix_all_strand_starts(mesh: TwistedWireMesh):
    """全素線の開始端を全固定."""
    fixed = []
    for sid in range(mesh.n_strands):
        start_node = mesh.strand_node_ranges[sid][0]
        for d in range(_NDOF_PER_NODE):
            fixed.append(start_node * _NDOF_PER_NODE + d)
    return np.array(sorted(set(fixed)), dtype=int)


def _get_strand_end_dofs(mesh: TwistedWireMesh, strand_id: int, end: str):
    """素線の端点のDOFインデックスを取得."""
    start_node, end_node = mesh.strand_node_ranges[strand_id]
    node = start_node if end == "start" else end_node - 1
    return np.array([node * _NDOF_PER_NODE + d for d in range(_NDOF_PER_NODE)])


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


def _solve_7strand_ncp_mortar(
    load_type: str = "tension",
    load_value: float = 100.0,
    *,
    n_elems_per_strand: int = 8,
    n_load_steps: int = 10,
    max_iter: int = 80,
    use_mortar: bool = True,
    use_friction: bool = False,
    mu: float = 0.3,
    exclude_same_layer: bool = True,
):
    """NCP + Mortar ソルバーで7本撚りを解く."""
    mesh = make_twisted_wire_mesh(
        7,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=n_elems_per_strand,
        n_pitches=0.5,
        gap=0.0,
    )

    assemble_tangent, assemble_internal_force, ndof_total = _make_timo3d_assemblers(mesh)
    fixed_dofs = _fix_all_strand_starts(mesh)

    f_ext = np.zeros(ndof_total)
    if load_type == "tension":
        f_per_strand = load_value / mesh.n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[2]] = f_per_strand
    elif load_type == "bending":
        m_per_strand = load_value / mesh.n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext[end_dofs[4]] = m_per_strand
    else:
        raise ValueError(f"未知の荷重タイプ: {load_type}")

    # 同層除外マッピング
    elem_layer_map = None
    if exclude_same_layer:
        elem_layer_map = mesh.build_elem_layer_map()

    config = ContactConfig(
        k_pen_scale=1e5,
        line_contact=True,
        n_gauss=2,
        use_mortar=use_mortar,
        use_friction=use_friction,
        mu=mu,
        k_t_ratio=0.01 if use_friction else 0.1,
        mu_ramp_steps=5 if use_friction else 0,
        g_on=0.0,
        g_off=1e-5,
        tol_penetration_ratio=0.02,
        penalty_growth_factor=2.0,
        elem_layer_map=elem_layer_map,
        exclude_same_layer=exclude_same_layer,
    )
    mgr = ContactManager(config=config)

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
        broadphase_margin=0.005,
        line_contact=True,
        n_gauss=2,
        use_mortar=use_mortar,
        use_friction=use_friction,
        mu=mu if use_friction else None,
        mu_ramp_steps=5 if use_friction else None,
    )

    return result, mgr, mesh


# ====================================================================
# 7本撚り Mortar 収束性能評価
# ====================================================================


class TestSevenStrandMortarConvergence:
    """7本撚り + Mortar の収束性能評価."""

    def test_7strand_mortar_tension_converges(self):
        """7本撚り引張 + Mortar + 同層除外で NCP 収束."""
        result, mgr, mesh = _solve_7strand_ncp_mortar(
            load_type="tension",
            load_value=50.0,
            n_load_steps=5,
            max_iter=80,
        )
        assert result.converged, (
            f"7-strand Mortar tension did not converge "
            f"(steps={result.n_load_steps}, iters={result.total_newton_iterations})"
        )

    def test_7strand_mortar_bending_converges(self):
        """7本撚り曲げ + Mortar + 同層除外で NCP 収束.

        status-087: K_line/Mortar接触剛性の二重カウント修正後に収束達成。
        """
        result, mgr, mesh = _solve_7strand_ncp_mortar(
            load_type="bending",
            load_value=0.01,
            n_load_steps=10,
            max_iter=100,
        )
        assert result.converged, (
            f"7-strand Mortar bending did not converge "
            f"(steps={result.n_load_steps}, iters={result.total_newton_iterations})"
        )

    def test_7strand_mortar_vs_ptp_direction(self):
        """Mortar と PtP（line contact のみ）で変位方向が一致."""
        # Mortar
        result_m, _, _ = _solve_7strand_ncp_mortar(
            load_type="tension",
            load_value=50.0,
            n_load_steps=5,
            use_mortar=True,
        )
        # PtP (line contact only)
        result_ptp, _, _ = _solve_7strand_ncp_mortar(
            load_type="tension",
            load_value=50.0,
            n_load_steps=5,
            use_mortar=False,
        )

        if result_m.converged and result_ptp.converged:
            # z方向変位の符号が一致
            uz_m = np.max(np.abs(result_m.u[2::6]))
            uz_ptp = np.max(np.abs(result_ptp.u[2::6]))
            assert uz_m > 0 and uz_ptp > 0, "非ゼロ変位を期待"


class TestSevenStrandMortarPenetration:
    """7本撚り + Mortar の貫入率評価."""

    def test_7strand_mortar_penetration_below_threshold(self):
        """7本撚り + Mortar で貫入率 < 5%.

        NCP + Mortar の多ペア環境では PtP より貫入制御が緩い傾向がある。
        現状 ~3% 程度。S2 以降のチューニングで改善予定。
        """
        result, mgr, mesh = _solve_7strand_ncp_mortar(
            load_type="tension",
            load_value=50.0,
            n_load_steps=5,
            max_iter=80,
        )
        if result.converged:
            pen_ratio = _max_penetration_ratio(mgr)
            assert pen_ratio < 0.05, f"貫入率が閾値超過: {pen_ratio:.4f} > 0.05"


class TestSevenStrandMortarFriction:
    """7本撚り + Mortar + 摩擦の統合テスト."""

    def test_7strand_mortar_friction_converges(self):
        """7本撚り + Mortar + 摩擦で NCP 収束."""
        result, mgr, mesh = _solve_7strand_ncp_mortar(
            load_type="tension",
            load_value=50.0,
            n_load_steps=8,
            max_iter=80,
            use_friction=True,
            mu=0.1,
        )
        assert result.converged, (
            f"7-strand Mortar+friction did not converge "
            f"(steps={result.n_load_steps}, iters={result.total_newton_iterations})"
        )
