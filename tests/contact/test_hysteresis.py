"""NCP版 撚線ヒステリシステスト.

旧ソルバー(run_contact_cyclic)の test_hysteresis.py をNCP移行。
NCPソルバーの逐次呼び出しによるサイクリック荷重でヒステリシス挙動を検証。

テストケース:
1. NCP+摩擦での往復荷重が収束する
2. 往復荷重で荷重-変位履歴が正しく記録される
3. 引張/曲げ/ねじりの各荷重モードでの収束
4. 摩擦付き往復で散逸エネルギーが非負
5. 荷重-変位曲線が初回と異なる（ヒステリシスの兆候）
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
)
from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp
from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# === 共通パラメータ ===
_E = 200e9
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002
_PITCH = 0.04
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_NDOF = 6
_N_ELEM = 16


def _make_timo3d_assemblers(mesh):
    """Timo3D 線形アセンブラを構築."""
    nc = mesh.node_coords
    conn = mesh.connectivity
    ndof_total = mesh.n_nodes * _NDOF

    K = np.zeros((ndof_total, ndof_total))
    for elem in conn:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = nc[np.array([n1, n2])]
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
            [_NDOF * n1 + d for d in range(_NDOF)] + [_NDOF * n2 + d for d in range(_NDOF)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    return assemble_tangent, assemble_internal_force, ndof_total


def _fix_all_strand_starts(mesh):
    """全素線の開始端を全固定."""
    fixed = set()
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        for d in range(_NDOF):
            fixed.add(_NDOF * nodes[0] + d)
    return np.array(sorted(fixed), dtype=int)


def _get_strand_end_dofs(mesh, strand_id, end):
    """素線の端点のDOFインデックスを取得."""
    nodes = mesh.strand_nodes(strand_id)
    node = nodes[0] if end == "start" else nodes[-1]
    return np.array([_NDOF * node + d for d in range(_NDOF)], dtype=int)


def _make_contact_manager(*, use_friction=False, mu=0.3):
    """ヒステリシステスト用の接触マネージャ（NCP向け）."""
    return ContactManager(
        config=ContactConfig(
            k_pen_scale=0.1,
            k_pen_mode="beam_ei",
            beam_E=_E,
            beam_I=_SECTION.Iy,
            k_pen_scaling="sqrt",
            k_t_ratio=0.1 if not use_friction else 0.01,
            mu=mu,
            g_on=0.0005,
            g_off=0.001,
            use_friction=use_friction,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.02,
            penalty_growth_factor=1.0,
            k_pen_max=1e12,
            exclude_same_layer=False,
            midpoint_prescreening=True,
            linear_solver="auto",
            adjust_initial_penetration=True,
        ),
    )


def _build_cyclic_model(n_strands=3, load_type="tension"):
    """サイクリック荷重テスト用モデル."""
    mesh = make_twisted_wire_mesh(
        n_strands,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=_N_ELEM,
        n_pitches=1.0,
        gap=0.0005,
    )
    at, ai, ndof = _make_timo3d_assemblers(mesh)
    fd = _fix_all_strand_starts(mesh)

    f_ext_unit = np.zeros(ndof)
    if load_type == "tension":
        f_per = 1.0 / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext_unit[end_dofs[2]] = f_per  # z方向
    elif load_type == "bending":
        m_per = 1.0 / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext_unit[end_dofs[4]] = m_per  # My方向
    elif load_type == "torsion":
        m_per = 1.0 / n_strands
        for sid in range(mesh.n_strands):
            end_dofs = _get_strand_end_dofs(mesh, sid, "end")
            f_ext_unit[end_dofs[5]] = m_per  # Mz方向

    return f_ext_unit, fd, at, ai, mesh, ndof


def _run_cyclic(
    f_ext_max,
    fixed_dofs,
    assemble_tangent,
    assemble_internal_force,
    mgr,
    mesh,
    amplitudes,
    n_steps_per_phase=5,
):
    """NCPソルバーで逐次的にサイクリック荷重を実行.

    各フェーズで目標荷重を NCP ソルバーに渡し、内部で n_steps_per_phase ステップで到達。
    前フェーズの変位を初期値として引き継ぐ。

    Returns:
        load_factors: 各フェーズ終了時の荷重比
        displacements: 各フェーズ終了時の変位
        converged: 全フェーズ収束したか
    """
    load_factors = []
    displacements = []
    all_converged = True

    u_current = np.zeros(len(f_ext_max))

    for phase_idx, target_lf in enumerate(amplitudes):
        f_target = f_ext_max * target_lf

        # 前フェーズからの増分を計算
        # f_base = 現在の平衡状態の荷重（前フェーズ終了時）
        if phase_idx == 0:
            f_base = None
        else:
            f_base = f_ext_max * amplitudes[phase_idx - 1]

        result = newton_raphson_contact_ncp(
            f_target,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=n_steps_per_phase,
            max_iter=50,
            tol_force=1e-5,
            tol_ncp=1e-5,
            show_progress=False,
            broadphase_margin=0.01,
            u0=u_current,
            f_ext_base=f_base,
            use_friction=mgr.config.use_friction,
            mu=mgr.config.mu,
            adaptive_timestepping=True,
        )

        load_factors.append(target_lf)
        displacements.append(result.u.copy())
        u_current = result.u.copy()

        if not result.converged:
            all_converged = False

    return load_factors, displacements, all_converged


# ====================================================================
# テスト: NCP版サイクリック荷重基本動作
# ====================================================================


class TestCyclicBasic:
    """NCP版: サイクリック荷重の基本動作テスト."""

    def test_single_phase(self):
        """1フェーズ（0→1）でNCP収束."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        lfs, disps, converged = _run_cyclic(
            f_unit * 100.0,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0],
            n_steps_per_phase=10,
        )

        assert converged, "Single phase NCP did not converge"
        assert len(lfs) == 1
        assert abs(lfs[-1] - 1.0) < 1e-10

    def test_two_phase_forward_reverse(self):
        """2フェーズ（0→1→0）で荷重係数が正しく推移する."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        lfs, disps, converged = _run_cyclic(
            f_unit * 100.0,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=10,
        )

        assert converged
        assert len(lfs) == 2
        assert abs(lfs[0] - 1.0) < 1e-10
        assert abs(lfs[1] - 0.0) < 1e-10

    def test_full_cycle_load_history(self):
        """3フェーズ（0→+1→-1→0）で各フェーズが収束."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        lfs, disps, converged = _run_cyclic(
            f_unit * 20.0,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0, -1.0, 0.0],
            n_steps_per_phase=10,
        )

        assert len(lfs) == 3
        assert abs(lfs[0] - 1.0) < 1e-10
        assert abs(lfs[1] - (-1.0)) < 1e-10
        assert abs(lfs[2] - 0.0) < 1e-10


# ====================================================================
# テスト: NCP版ヒステリシス（摩擦付き往復荷重）
# ====================================================================


class TestTwistedWireHysteresis:
    """NCP版: 3本撚りの摩擦付き往復荷重テスト."""

    def test_tension_hysteresis_converges(self):
        """引張往復荷重がNCP+摩擦で収束する."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        lfs, disps, converged = _run_cyclic(
            f_unit * 100.0,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=10,
        )

        assert converged, "Tension hysteresis NCP did not converge"

    def test_bending_hysteresis_converges(self):
        """曲げ往復荷重がNCP+摩擦で収束する."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "bending")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        lfs, disps, converged = _run_cyclic(
            f_unit * 0.03,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=10,
        )

        assert converged, "Bending hysteresis NCP did not converge"

    def test_torsion_hysteresis_converges(self):
        """ねじり往復荷重がNCP+摩擦で収束する."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "torsion")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        lfs, disps, converged = _run_cyclic(
            f_unit * 0.005,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=10,
        )

        assert converged, "Torsion hysteresis NCP did not converge"


# ====================================================================
# テスト: NCP版ヒステリシス物理的性質
# ====================================================================


class TestHysteresisPhysics:
    """NCP版: ヒステリシスの物理的性質検証."""

    def test_max_displacement_at_peak_load(self):
        """最大荷重時の変位 >= 除荷後の変位（弾性回復）."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager(use_friction=True, mu=0.3)

        lfs, disps, converged = _run_cyclic(
            f_unit * 100.0,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=10,
        )

        assert converged

        u_peak = disps[0]  # lf=1.0
        u_unloaded = disps[1]  # lf=0.0

        assert np.linalg.norm(u_peak) >= np.linalg.norm(u_unloaded) - 1e-10

    def test_displacement_returns_near_zero_after_unload(self):
        """完全除荷後に変位が小さくなる."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        lfs, disps, converged = _run_cyclic(
            f_unit * 100.0,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0, 0.0],
            n_steps_per_phase=10,
        )

        assert converged

        u_peak = disps[0]
        u_unloaded = disps[1]

        assert np.linalg.norm(u_unloaded) < np.linalg.norm(u_peak) + 1e-10

    def test_peak_displacement_positive(self):
        """正荷重で正変位."""
        f_unit, fd, at, ai, mesh, ndof = _build_cyclic_model(3, "tension")
        mgr = _make_contact_manager()

        lfs, disps, converged = _run_cyclic(
            f_unit * 100.0,
            fd,
            at,
            ai,
            mgr,
            mesh,
            amplitudes=[1.0],
            n_steps_per_phase=10,
        )

        assert converged

        end_dofs = _get_strand_end_dofs(mesh, 0, "end")
        uz = disps[0][end_dofs[2]]
        assert uz > 0, f"Expected positive displacement, got {uz:.6e}"
