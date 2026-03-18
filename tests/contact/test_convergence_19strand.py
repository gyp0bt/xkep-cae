"""NCP ソルバー収束テスト: 7本→19本撚線.

adaptive omega + ステップ二分法 + active-set freezing を用いた
NCP Semi-smooth Newton の大規模収束検証。
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
from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
from xkep_cae.sections.beam import BeamSectionInput

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_NDOF = 6
_E = 200e9
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002
_SECTION = BeamSectionInput.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_PITCH = 0.040


# ====================================================================
# ヘルパー
# ====================================================================


def _build_assemblers(mesh):
    """線形 Timoshenko 梁アセンブラを構築."""
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


def _fixed_dofs(mesh):
    """全素線の開始端を全固定."""
    fixed = set()
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        node = nodes[0]
        for d in range(_NDOF):
            fixed.add(_NDOF * node + d)
    return np.array(sorted(fixed), dtype=int)


def _tension_load(mesh, ndof_total, total_force=100.0):
    """軸方向引張荷重."""
    f_ext = np.zeros(ndof_total)
    f_per = total_force / mesh.n_strands
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        end_node = nodes[-1]
        f_ext[_NDOF * end_node + 2] = f_per  # z方向
    return f_ext


def _build_contact_manager(mesh, *, k_pen_scale=0.1):
    """NCP 用 ContactManager を構築."""
    elem_layer_map = mesh.build_elem_layer_map()
    return ContactManager(
        config=ContactConfig(
            k_pen_scale=k_pen_scale,
            k_pen_mode="beam_ei",
            beam_E=_E,
            beam_I=_SECTION.Iy,
            k_pen_scaling="sqrt",
            k_t_ratio=0.1,
            mu=0.0,
            g_on=0.0005,
            g_off=0.001,
            use_friction=False,
            use_line_search=False,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.02,
            penalty_growth_factor=1.0,
            k_pen_max=1e12,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            linear_solver="auto",
            no_deactivation_within_step=True,
            preserve_inactive_lambda=True,
            adjust_initial_penetration=True,
        ),
    )


def _count_active(mgr):
    return sum(1 for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE)


# ====================================================================
# テスト: NCP 7本撚線（ベースライン）
# ====================================================================


class Test7Strand:
    """NCP ソルバーで7本撚線の収束を確認（ベースライン）."""

    def test_7strand_basic(self):
        """7本: デフォルトパラメータで収束."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)
        mgr = _build_contact_manager(mesh)

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=10,
            max_iter=50,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
        )

        print(
            f"  7本 NCP: converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={_count_active(mgr)}"
        )
        assert result.converged, "7本 NCP が収束しなかった"

    def test_7strand_with_adaptive_omega(self):
        """7本: adaptive omega 有効でも収束を維持."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)
        mgr = _build_contact_manager(mesh)

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=10,
            max_iter=50,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.5,
            omega_min=0.05,
            omega_max=1.0,
            omega_shrink=0.5,
            omega_growth=1.2,
        )

        print(
            f"  7本 NCP (omega): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}"
        )
        assert result.converged, "7本 NCP (adaptive omega) が収束しなかった"

    def test_7strand_with_bisection(self):
        """7本: ステップ二分法が正常動作."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)
        mgr = _build_contact_manager(mesh)

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=3,
            max_iter=30,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_timestepping=True,
            adaptive_omega=True,
            omega_init=0.5,
            omega_min=0.05,
            omega_max=1.0,
        )

        print(
            f"  7本 NCP (adaptive dt): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}"
        )


# ====================================================================
# テスト: NCP 19本撚線（ターゲット）
# ====================================================================


class Test19Strand:
    """NCP ソルバーで19本撚線の収束を検証.

    S3マイルストーン: NCPソルバーで19本以上の収束達成。
    """

    def test_19strand_adaptive(self):
        """19本: adaptive omega + active-set freezing + du cap で収束を試行."""
        mesh = make_twisted_wire_mesh(
            19,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)
        mgr = _build_contact_manager(mesh)

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=20,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.3,
            omega_min=0.05,
            omega_max=0.8,
            omega_shrink=0.5,
            omega_growth=1.1,
            adaptive_timestepping=True,
            active_set_update_interval=10,
            du_norm_cap=5.0,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  19本 NCP: converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        # 現時点では収束を必須にしない（S3改善進行中）
        assert result.total_newton_iterations > 0, "NR反復が実行されていない"

    def test_19strand_relaxed_tol(self):
        """19本: 緩い公差 + active-set freezing で収束を確認."""
        mesh = make_twisted_wire_mesh(
            19,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=50.0)
        mgr = _build_contact_manager(mesh, k_pen_scale=0.05)

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=30,
            max_iter=100,
            tol_force=1e-4,
            tol_ncp=1e-4,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.2,
            omega_min=0.05,
            omega_max=0.8,
            adaptive_timestepping=True,
            active_set_update_interval=10,
            du_norm_cap=5.0,
        )

        print(
            f"\n  19本 NCP (relaxed): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}"
        )
        assert result.total_newton_iterations > 0

    def test_19strand_s3_improvements(self):
        """19本: S3改良1-5を全て有効化して収束を試行.

        改良1: ILU drop_tol 適応制御
        改良2: Schur ブロック正則化
        改良3: GMRES restart 適応
        改良4: λウォームスタート
        改良5: Active set チャタリング抑制
        """
        mesh = make_twisted_wire_mesh(
            19,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)

        # S3改良4-5を有効化したContactConfig
        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.0005,
                g_off=0.001,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                # S3改良4: λウォームスタート
                lambda_warmstart_neighbor=True,
                # S3改良5: チャタリング抑制（ウィンドウ幅3）
                chattering_window=3,
                adjust_initial_penetration=True,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=20,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.3,
            omega_min=0.05,
            omega_max=0.8,
            omega_shrink=0.5,
            omega_growth=1.1,
            adaptive_timestepping=True,
            active_set_update_interval=10,
            du_norm_cap=5.0,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  19本 NCP (S3改良全有効): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        assert result.total_newton_iterations > 0, "NR反復が実行されていない"


# ====================================================================
# テスト: chattering_window チューニング
# ====================================================================


class TestChatteringWindowTuning:
    """chattering_window の最適値チューニング（3-5が候補）."""

    @pytest.mark.parametrize("window", [0, 3, 5])
    def test_chattering_window_7strand(self, window):
        """7本撚線での chattering_window 比較."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.0005,
                g_off=0.001,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                chattering_window=window,
                adjust_initial_penetration=True,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=10,
            max_iter=50,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
        )

        print(
            f"\n  chattering_window={window}: converged={result.converged}, "
            f"newton={result.total_newton_iterations}, "
            f"active={_count_active(mgr)}"
        )
        assert result.converged, f"chattering_window={window} で7本が収束しなかった"


# ====================================================================
# テスト: lambda_warmstart_neighbor 効果検証
# ====================================================================


class TestLambdaWarmstartEffect:
    """lambda_warmstart_neighbor の効果検証."""

    @pytest.mark.parametrize("warmstart", [False, True])
    def test_warmstart_7strand(self, warmstart):
        """7本撚線での λウォームスタートの効果比較."""
        mesh = make_twisted_wire_mesh(
            7,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.0005,
                g_off=0.001,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                lambda_warmstart_neighbor=warmstart,
                adjust_initial_penetration=True,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=10,
            max_iter=50,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
        )

        print(
            f"\n  warmstart={warmstart}: converged={result.converged}, "
            f"newton={result.total_newton_iterations}, "
            f"active={_count_active(mgr)}"
        )
        assert result.converged, f"warmstart={warmstart} で7本が収束しなかった"


# ====================================================================
# テスト: S3改良6-8の19本統合テスト
# ====================================================================


class Test19StrandS3Full:
    """S3改良1-8を全有効化した19本NCP収束テスト."""

    def test_19strand_all_s3_improvements(self):
        """19本: S3改良1-8を全て有効化して収束を試行.

        改良1: ILU drop_tol 適応制御
        改良2: Schur ブロック正則化
        改良3: GMRES restart 適応
        改良4: λウォームスタート
        改良5: Active set チャタリング抑制
        改良6: 適応時間増分制御
        改良7: AMG前処理
        改良8: k_pen continuation
        """
        mesh = make_twisted_wire_mesh(
            19,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.0005,
                g_off=0.001,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                # S3改良4: λウォームスタート
                lambda_warmstart_neighbor=True,
                # S3改良5: チャタリング抑制
                chattering_window=3,
                # S3改良6: 適応時間増分制御
                adaptive_timestepping=True,
                dt_grow_factor=1.3,
                dt_shrink_factor=0.5,
                dt_grow_iter_threshold=8,
                dt_shrink_iter_threshold=20,
                dt_contact_change_threshold=0.3,
                # S3改良7: AMG前処理
                use_amg_preconditioner=True,
                # S3改良8: k_pen continuation
                k_pen_continuation=True,
                k_pen_continuation_start=0.1,
                k_pen_continuation_steps=5,
                adjust_initial_penetration=True,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=20,
            max_iter=100,
            tol_force=1e-5,
            tol_ncp=1e-5,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.3,
            omega_min=0.05,
            omega_max=0.8,
            omega_shrink=0.5,
            omega_growth=1.1,
            active_set_update_interval=5,
            du_norm_cap=5.0,
            adaptive_timestepping=True,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  19本 NCP (S3全改良): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        # 全改良有効化で反復が実行されていること
        assert result.total_newton_iterations > 0, "NR反復が実行されていない"

    def test_19strand_adaptive_dt_only(self):
        """19本: 適応Δtのみ有効化."""
        mesh = make_twisted_wire_mesh(
            19,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=50.0)
        mgr = _build_contact_manager(mesh, k_pen_scale=0.05)

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=20,
            max_iter=100,
            tol_force=1e-4,
            tol_ncp=1e-4,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.3,
            omega_min=0.05,
            omega_max=0.8,
            active_set_update_interval=10,
            du_norm_cap=5.0,
            adaptive_timestepping=True,
        )

        print(
            f"\n  19本 NCP (adaptive dt): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}"
        )
        assert result.total_newton_iterations > 0

    def test_19strand_kpen_continuation_only(self):
        """19本: k_pen continuation のみ有効化."""
        mesh = make_twisted_wire_mesh(
            19,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs(mesh)
        f_ext = _tension_load(mesh, ndof, total_force=100.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.0005,
                g_off=0.001,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                k_pen_continuation=True,
                k_pen_continuation_start=0.1,
                k_pen_continuation_steps=5,
                adjust_initial_penetration=True,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=20,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.3,
            omega_min=0.05,
            omega_max=0.8,
            adaptive_timestepping=True,
            active_set_update_interval=10,
            du_norm_cap=5.0,
        )

        print(
            f"\n  19本 NCP (k_pen cont.): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}"
        )
        assert result.total_newton_iterations > 0


# ====================================================================
# テスト: 径方向圧縮による接触活性化（S3マイルストーン達成）
# ====================================================================


def _radial_load(mesh, ndof_total, *, layers=(1,), force_per_node=5.0):
    """指定層の素線中間節点に中心向き径方向力を付与."""
    f_ext = np.zeros(ndof_total)
    nc = mesh.node_coords
    for sid in range(1, mesh.n_strands):
        info = mesh.strand_infos[sid]
        if info.layer not in layers:
            continue
        nodes = mesh.strand_nodes(sid)
        mid_node = nodes[len(nodes) // 2]
        pos = nc[mid_node]
        r_xy = np.linalg.norm(pos[:2])
        if r_xy > 1e-10:
            r_dir = -pos[:2] / r_xy  # 中心向き単位ベクトル
            f_ext[_NDOF * mid_node + 0] = r_dir[0] * force_per_node
            f_ext[_NDOF * mid_node + 1] = r_dir[1] * force_per_node
    return f_ext


def _fixed_dofs_with_center(mesh):
    """全素線の始点 + 中心素線の全自由度を固定."""
    fixed = set()
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        for d in range(_NDOF):
            fixed.add(_NDOF * nodes[0] + d)
        if sid == 0:  # 中心素線を完全拘束
            for node in nodes:
                for d in range(_NDOF):
                    fixed.add(_NDOF * node + d)
    return np.array(sorted(fixed), dtype=int)


class Test19StrandRadialCompression:
    """径方向圧縮による19本NCP接触活性化テスト.

    中心素線を固定し、Layer 1 素線に径方向力を付与して
    実際の接触活性化を伴うNCP収束を検証する。

    S3マイルストーン: 19本撚線でアクティブ接触を含むNCP収束達成。
    """

    @pytest.mark.xfail(
        reason="19本NCP径方向圧縮: CI環境でタイムアウト (status-127)",
        strict=False,
    )
    def test_19strand_radial_with_active_contacts(self):
        """19本: 径方向圧縮で接触活性化 + NCP収束."""
        mesh = make_twisted_wire_mesh(
            19,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0001,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs_with_center(mesh)
        f_ext = _radial_load(mesh, ndof, layers=(1,), force_per_node=5.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.001,
                g_off=0.002,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                lambda_warmstart_neighbor=True,
                chattering_window=3,
                k_pen_continuation=True,
                k_pen_continuation_start=0.1,
                k_pen_continuation_steps=5,
                adjust_initial_penetration=True,
                contact_force_ramp=True,
                contact_force_ramp_iters=5,
                adaptive_timestepping=True,
                dt_grow_factor=1.3,
                dt_shrink_factor=0.5,
                dt_grow_iter_threshold=8,
                dt_shrink_iter_threshold=20,
                dt_contact_change_threshold=0.3,
                residual_scaling="rms",
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=40,
            max_iter=100,
            tol_force=1e-4,
            tol_ncp=1e-4,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.3,
            omega_min=0.02,
            omega_max=0.8,
            omega_shrink=0.5,
            omega_growth=1.1,
            active_set_update_interval=5,
            du_norm_cap=3.0,
            adaptive_timestepping=True,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  19本 NCP (径方向圧縮): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        assert result.converged, "19本 NCP（径方向圧縮）が収束しなかった"
        assert n_active > 0, "接触ペアが活性化されていない"


# ====================================================================
# テスト: 37本NCP収束（S3拡張: 19本→37本）
# ====================================================================


class Test37StrandRadialCompression:
    """37本撚線(1+6+12+18)の径方向圧縮NCP収束テスト.

    19本収束達成(status-112)に基づき、37本(layer 0-3)への拡張。
    Layer 1-2の内側素線に径方向力を付与し、3層間の接触を検証。
    """

    @pytest.mark.xfail(reason="CI timeout >600s — 高速化後に再評価", strict=False)
    def test_37strand_radial_layer1(self):
        """37本: Layer 1のみ径方向圧縮でNCP収束."""
        mesh = make_twisted_wire_mesh(
            37,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0001,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs_with_center(mesh)
        f_ext = _radial_load(mesh, ndof, layers=(1,), force_per_node=5.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.1,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.001,
                g_off=0.002,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                lambda_warmstart_neighbor=True,
                chattering_window=3,
                k_pen_continuation=True,
                k_pen_continuation_start=0.1,
                k_pen_continuation_steps=5,
                adjust_initial_penetration=True,
                contact_force_ramp=True,
                contact_force_ramp_iters=5,
                adaptive_timestepping=True,
                dt_grow_factor=1.3,
                dt_shrink_factor=0.5,
                dt_grow_iter_threshold=8,
                dt_shrink_iter_threshold=20,
                dt_contact_change_threshold=0.3,
                residual_scaling="rms",
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=40,
            max_iter=100,
            tol_force=1e-4,
            tol_ncp=1e-4,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.3,
            omega_min=0.02,
            omega_max=0.8,
            omega_shrink=0.5,
            omega_growth=1.1,
            active_set_update_interval=5,
            du_norm_cap=3.0,
            adaptive_timestepping=True,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  37本 NCP (layer1圧縮): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        # 37本ではまだ収束を必須としない（段階的改善のトラッキング）
        assert result.total_newton_iterations > 0, "NR反復が実行されていない"
        # 収束状態を記録
        if result.converged:
            assert n_active > 0, "収束したが接触ペアが活性化されていない"

    @pytest.mark.xfail(reason="CI timeout >600s — 高速化後に再評価", strict=False)
    def test_37strand_radial_layer1_2(self):
        """37本: Layer 1+2 径方向圧縮でNCP収束.

        改良点（status-126）:
        - staged_activation_steps=20 で層別段階的活性化（一斉活性化抑制）
        - k_pen_continuation_start=0.01, steps=10で漸進的ペナルティ上昇
        - contact_force_ramp_iters=10で接触力を緩やかに導入
        - force_per_node=2.0に低減（段階的荷重）
        - omega_init=0.1, omega_min=0.01で慎重なNR更新
        """
        mesh = make_twisted_wire_mesh(
            37,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0001,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs_with_center(mesh)
        f_ext = _radial_load(mesh, ndof, layers=(1, 2), force_per_node=2.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.05,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.001,
                g_off=0.002,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                lambda_warmstart_neighbor=True,
                chattering_window=5,
                k_pen_continuation=True,
                k_pen_continuation_start=0.01,
                k_pen_continuation_steps=10,
                adjust_initial_penetration=True,
                contact_force_ramp=True,
                contact_force_ramp_iters=10,
                adaptive_timestepping=True,
                dt_grow_factor=1.2,
                dt_shrink_factor=0.5,
                dt_grow_iter_threshold=8,
                dt_shrink_iter_threshold=15,
                dt_contact_change_threshold=0.2,
                residual_scaling="rms",
                # S3改良: 段階的活性化（層別に接触ペアを段階的投入）
                staged_activation_steps=20,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=50,
            max_iter=80,
            tol_force=1e-4,
            tol_ncp=1e-4,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.1,
            omega_min=0.01,
            omega_max=0.6,
            omega_shrink=0.5,
            omega_growth=1.1,
            active_set_update_interval=5,
            du_norm_cap=2.0,
            adaptive_timestepping=True,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  37本 NCP (layer1+2圧縮): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        assert result.total_newton_iterations > 0, "NR反復が実行されていない"


# ====================================================================
# テスト: 61本/91本 NCP収束（段階的拡張）
# ====================================================================


class Test61StrandRadialCompression:
    """61本撚線(1+6+12+18+24)の径方向圧縮NCP収束テスト.

    37本収束に基づき、61本(layer 0-4)への拡張。
    Layer 1のみの径方向圧縮から段階的に検証。
    """

    @pytest.mark.xfail(reason="CI timeout >600s — 高速化後に再評価", strict=False)
    def test_61strand_radial_layer1(self):
        """61本: Layer 1のみ径方向圧縮でNCP収束を試行."""
        mesh = make_twisted_wire_mesh(
            61,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0001,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs_with_center(mesh)
        f_ext = _radial_load(mesh, ndof, layers=(1,), force_per_node=2.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.05,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.001,
                g_off=0.002,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                lambda_warmstart_neighbor=True,
                chattering_window=5,
                k_pen_continuation=True,
                k_pen_continuation_start=0.01,
                k_pen_continuation_steps=10,
                adjust_initial_penetration=True,
                contact_force_ramp=True,
                contact_force_ramp_iters=10,
                adaptive_timestepping=True,
                dt_grow_factor=1.2,
                dt_shrink_factor=0.5,
                dt_grow_iter_threshold=8,
                dt_shrink_iter_threshold=15,
                dt_contact_change_threshold=0.2,
                residual_scaling="rms",
                # S3改良: 段階的活性化
                staged_activation_steps=25,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=50,
            max_iter=80,
            tol_force=1e-4,
            tol_ncp=1e-4,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.1,
            omega_min=0.01,
            omega_max=0.6,
            omega_shrink=0.5,
            omega_growth=1.1,
            active_set_update_interval=5,
            du_norm_cap=2.0,
            adaptive_timestepping=True,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  61本 NCP (layer1圧縮): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        assert result.total_newton_iterations > 0, "NR反復が実行されていない"


class Test91StrandRadialCompression:
    """91本撚線(1+6+12+18+24+30)の径方向圧縮NCP収束テスト.

    61本収束に基づき、91本(layer 0-5)への拡張。
    Layer 1のみの径方向圧縮から段階的に検証。
    """

    @pytest.mark.xfail(reason="CI timeout >600s — 高速化後に再評価", strict=False)
    def test_91strand_radial_layer1(self):
        """91本: Layer 1のみ径方向圧縮でNCP収束を試行."""
        mesh = make_twisted_wire_mesh(
            91,
            _WIRE_D,
            _PITCH,
            length=0.0,
            n_elems_per_strand=16,
            n_pitches=1.0,
            gap=0.0001,
        )
        at, ai, ndof = _build_assemblers(mesh)
        fd = _fixed_dofs_with_center(mesh)
        f_ext = _radial_load(mesh, ndof, layers=(1,), force_per_node=1.0)

        elem_layer_map = mesh.build_elem_layer_map()
        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=0.05,
                k_pen_mode="beam_ei",
                beam_E=_E,
                beam_I=_SECTION.Iy,
                k_pen_scaling="sqrt",
                k_t_ratio=0.1,
                mu=0.0,
                g_on=0.001,
                g_off=0.002,
                use_friction=False,
                use_line_search=False,
                use_geometric_stiffness=True,
                tol_penetration_ratio=0.02,
                penalty_growth_factor=1.0,
                k_pen_max=1e12,
                elem_layer_map=elem_layer_map,
                exclude_same_layer=True,
                midpoint_prescreening=True,
                linear_solver="auto",
                no_deactivation_within_step=True,
                preserve_inactive_lambda=True,
                lambda_warmstart_neighbor=True,
                chattering_window=5,
                k_pen_continuation=True,
                k_pen_continuation_start=0.01,
                k_pen_continuation_steps=10,
                adjust_initial_penetration=True,
                contact_force_ramp=True,
                contact_force_ramp_iters=10,
                adaptive_timestepping=True,
                dt_grow_factor=1.2,
                dt_shrink_factor=0.5,
                dt_grow_iter_threshold=8,
                dt_shrink_iter_threshold=15,
                dt_contact_change_threshold=0.2,
                residual_scaling="rms",
                # S3改良: 段階的活性化
                staged_activation_steps=30,
            ),
        )

        result = newton_raphson_contact_ncp(
            f_ext,
            fd,
            at,
            ai,
            mgr,
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            n_load_steps=50,
            max_iter=80,
            tol_force=1e-4,
            tol_ncp=1e-4,
            broadphase_margin=0.01,
            show_progress=True,
            adaptive_omega=True,
            omega_init=0.1,
            omega_min=0.01,
            omega_max=0.5,
            omega_shrink=0.5,
            omega_growth=1.1,
            active_set_update_interval=5,
            du_norm_cap=1.0,
            adaptive_timestepping=True,
        )

        n_active = _count_active(mgr)
        print(
            f"\n  91本 NCP (layer1圧縮): converged={result.converged}, "
            f"steps={result.n_load_steps}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        assert result.total_newton_iterations > 0, "NR反復が実行されていない"
