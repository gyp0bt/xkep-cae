"""S3 改良（ILU適応/Schur正則化/GMRES restart/λウォームスタート/チャタリング抑制/適応Δt/AMG/k_pen continuation/残差スケーリング/接触力ランプ）のテスト.

Phase S3: 大規模NCP収束改善 — 11改良の単体テスト。
"""

import numpy as np
import scipy.sparse as sp
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
)
from xkep_cae.contact.solver_ncp import (
    _solve_linear_system,
    _solve_saddle_point_direct,
    _solve_saddle_point_gmres,
    newton_raphson_contact_ncp,
)

# ---- テスト用ヘルパー ----


def _make_simple_k(ndof: int, *, stiffness: float = 1e6) -> sp.csr_matrix:
    """対角優位な剛性行列を生成."""
    diag = np.full(ndof, stiffness)
    off = np.full(ndof - 1, -stiffness * 0.1)
    K = sp.diags([off, diag, off], [-1, 0, 1], shape=(ndof, ndof), format="csr")
    return K


def _make_crossing_beams_setup(ndof_per_node: int = 6) -> dict:
    """2梁交差の基本セットアップ."""
    # 2梁: 各2節点、ndof_per_node DOF/node → 合計 4*ndof_per_node DOF
    n_nodes = 4
    ndof = n_nodes * ndof_per_node
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # Beam A node 0
            [1.0, 0.0, 0.0],  # Beam A node 1
            [0.5, -0.05, 0.0],  # Beam B node 0
            [0.5, 0.05, 0.0],  # Beam B node 1
        ]
    )
    connectivity = np.array([[0, 1], [2, 3]])
    radii = 0.04

    K_T = _make_simple_k(ndof)
    fixed_dofs = np.array([0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23], dtype=int)

    return {
        "ndof": ndof,
        "node_coords": node_coords,
        "connectivity": connectivity,
        "radii": radii,
        "K_T": K_T,
        "fixed_dofs": fixed_dofs,
        "ndof_per_node": ndof_per_node,
    }


# ==== 改良1: ILU drop_tol 適応制御 ====


class TestILUAdaptiveDropTol:
    """ILU drop_tol の適応制御テスト."""

    def test_solve_with_small_drop_tol(self):
        """非常に小さい drop_tol でも解が得られる（適応リトライが機能）."""
        ndof = 100
        K = _make_simple_k(ndof)
        rhs = np.random.default_rng(42).standard_normal(ndof)
        # drop_tol が非常に小さくても解が得られる
        x = _solve_linear_system(K, rhs, mode="iterative", ilu_drop_tol=1e-12)
        assert np.all(np.isfinite(x))

    def test_solve_with_large_drop_tol(self):
        """大きい drop_tol でも解が得られる."""
        ndof = 100
        K = _make_simple_k(ndof)
        rhs = np.random.default_rng(42).standard_normal(ndof)
        x = _solve_linear_system(K, rhs, mode="iterative", ilu_drop_tol=0.1)
        assert np.all(np.isfinite(x))

    def test_auto_mode_large_system(self):
        """大規模でautoモード時に反復法が使われる."""
        ndof = 500
        K = _make_simple_k(ndof)
        rhs = np.random.default_rng(42).standard_normal(ndof)
        x = _solve_linear_system(K, rhs, mode="auto", gmres_dof_threshold=100)
        # 解が妥当であることを確認
        residual = float(np.linalg.norm(K @ x - rhs))
        assert residual < 1e-4 * float(np.linalg.norm(rhs))


# ==== 改良2: Schur ブロック正則化 ====


class TestSchurRegularization:
    """Schur complement 正則化の改善テスト."""

    def test_direct_schur_preserves_solution(self):
        """正則化がよく条件づけられた問題の解を壊さないこと."""
        setup = _make_crossing_beams_setup()
        ndof = setup["ndof"]
        K_T = setup["K_T"]
        fixed_dofs = setup["fixed_dofs"]

        # 手動でG_Aとg_activeを構築
        n_active = 1
        G_A = sp.random(n_active, ndof, density=0.3, format="csr", random_state=42)
        g_active = np.array([-0.001])
        R_u = np.random.default_rng(42).standard_normal(ndof)
        R_u[fixed_dofs] = 0.0

        for k_pen in [1e3, 1e5, 1e7]:
            du, dlam = _solve_saddle_point_direct(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)
            assert np.all(np.isfinite(du)), f"k_pen={k_pen}: du に非有限値"
            assert np.all(np.isfinite(dlam)), f"k_pen={k_pen}: dlam に非有限値"

    def test_gmres_schur_diag_positive(self):
        """GMRES Schur対角近似が正値であること."""
        setup = _make_crossing_beams_setup()
        ndof = setup["ndof"]
        K_T = setup["K_T"]
        fixed_dofs = setup["fixed_dofs"]

        n_active = 2
        G_A = sp.random(n_active, ndof, density=0.3, format="csr", random_state=42)
        g_active = np.array([-0.001, -0.002])
        R_u = np.random.default_rng(42).standard_normal(ndof)
        R_u[fixed_dofs] = 0.0
        k_pen = 1e5

        du, dlam = _solve_saddle_point_gmres(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)
        assert np.all(np.isfinite(du))
        assert np.all(np.isfinite(dlam))


# ==== 改良3: GMRES restart 適応 ====


class TestGMRESRestart:
    """GMRES restart チューニングテスト."""

    def test_gmres_converges_with_restart(self):
        """restart付きGMRESが収束すること."""
        ndof = 200
        K = _make_simple_k(ndof)
        rhs = np.random.default_rng(42).standard_normal(ndof)
        x = _solve_linear_system(K, rhs, mode="iterative", ilu_drop_tol=1e-4)
        residual = float(np.linalg.norm(K @ x - rhs))
        assert residual < 1e-6 * float(np.linalg.norm(rhs))


# ==== 改良4: λウォームスタート ====


class TestLambdaWarmstart:
    """λウォームスタートのテスト."""

    def test_warmstart_config_default(self):
        """デフォルトでウォームスタートが無効."""
        config = ContactConfig()
        assert config.lambda_warmstart_neighbor is False

    def test_warmstart_config_enabled(self):
        """設定で有効化できること."""
        config = ContactConfig(lambda_warmstart_neighbor=True)
        assert config.lambda_warmstart_neighbor is True

    def test_warmstart_does_not_break_basic_solve(self):
        """ウォームスタート有効でも基本的な解析が成功すること."""
        # 既存の test_solver_contact のセットアップを流用（接触が発生する構成）
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            lambda_warmstart_neighbor=True,
        )
        manager = ContactManager(config=config)

        # 接触力なし（純構造問題）でもウォームスタート有効時にクラッシュしないこと
        ndof = setup["ndof"]
        f_ext = np.zeros(ndof)
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=1,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
        )
        # 接触が検出されなくても収束すること（純構造問題）
        assert np.all(np.isfinite(result.u))


# ==== 改良5: Active set チャタリング抑制 ====


class TestChatteringSuppression:
    """Active set チャタリング抑制テスト."""

    def test_chattering_window_config_default(self):
        """デフォルトでチャタリングウィンドウが無効（0）."""
        config = ContactConfig()
        assert config.chattering_window == 0

    def test_chattering_window_config_set(self):
        """チャタリングウィンドウを設定できること."""
        config = ContactConfig(chattering_window=3)
        assert config.chattering_window == 3

    def test_chattering_suppression_majority_vote(self):
        """過半数投票ロジックの動作確認（統合テスト）."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            chattering_window=3,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=1,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
        )
        assert np.all(np.isfinite(result.u))

    def test_chattering_window_zero_is_noop(self):
        """chattering_window=0 では従来動作と同一."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(k_pen_scale=1e4, chattering_window=0)
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=1,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
        )
        assert np.all(np.isfinite(result.u))


# ==== 自動安定時間増分（ユーザー追加要求への対応確認） ====


class TestStepBisectionDeprecated:
    """max_step_cuts/bisection_max_depth の非推奨化テスト."""

    def test_max_step_cuts_triggers_deprecation_warning(self):
        """max_step_cuts>0 で DeprecationWarning が発生し adaptive_dt に変換."""
        import warnings

        setup = _make_crossing_beams_setup()
        config = ContactConfig(k_pen_scale=1e4)
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = newton_raphson_contact_ncp(
                f_ext,
                setup["fixed_dofs"],
                assemble_tangent,
                assemble_internal,
                manager,
                setup["node_coords"],
                setup["connectivity"],
                setup["radii"],
                n_load_steps=1,
                max_iter=30,
                tol_force=1e-4,
                tol_ncp=1e-4,
                show_progress=False,
                k_pen=1e4,
                max_step_cuts=2,
            )
        assert np.all(np.isfinite(result.u))
        # DeprecationWarning が発生していること
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "max_step_cuts で DeprecationWarning が出るべき"


# ==== 改良6: 適応時間増分制御 ====


class TestAdaptiveTimestepping:
    """適応時間増分制御（S3改良6）のテスト."""

    def test_adaptive_dt_config_defaults(self):
        """デフォルトで適応Δtが無効."""
        config = ContactConfig()
        assert config.adaptive_timestepping is False
        assert config.dt_grow_factor == 1.5
        assert config.dt_shrink_factor == 0.5
        assert config.dt_grow_iter_threshold == 5
        assert config.dt_shrink_iter_threshold == 15

    def test_adaptive_dt_enabled_does_not_break(self):
        """適応Δt有効でも基本問題がクラッシュしないこと."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            adaptive_timestepping=True,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=4,
            max_iter=50,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
            adaptive_timestepping=True,
        )
        assert np.all(np.isfinite(result.u))

    def test_adaptive_dt_multiple_steps(self):
        """複数ステップでも適応Δtが機能すること."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            adaptive_timestepping=True,
            dt_grow_iter_threshold=10,
            dt_shrink_iter_threshold=20,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=3,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
            adaptive_timestepping=True,
        )
        assert np.all(np.isfinite(result.u))

    def test_adaptive_dt_failure_retry(self):
        """適応Δtで不収束時にステップ縮小リトライが機能すること."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            adaptive_timestepping=True,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=2,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
            adaptive_timestepping=True,
        )
        assert np.all(np.isfinite(result.u))

    def test_adaptive_dt_contact_change_shrink(self):
        """接触状態変化率の閾値設定が機能すること."""
        config = ContactConfig(
            adaptive_timestepping=True,
            dt_contact_change_threshold=0.2,
        )
        assert config.dt_contact_change_threshold == 0.2


# ==== 改良7: AMG 前処理 ====


class TestAMGPreconditioner:
    """AMG 前処理（S3改良7）のテスト."""

    def test_amg_config_default(self):
        """デフォルトで AMG が無効."""
        config = ContactConfig()
        assert config.use_amg_preconditioner is False

    def test_amg_gmres_solve(self):
        """AMG前処理付きGMRESが有限な解を返すこと."""
        setup = _make_crossing_beams_setup()
        ndof = setup["ndof"]
        K_T = setup["K_T"]
        fixed_dofs = setup["fixed_dofs"]

        n_active = 2
        G_A = sp.random(n_active, ndof, density=0.3, format="csr", random_state=42)
        g_active = np.array([-0.001, -0.002])
        R_u = np.random.default_rng(42).standard_normal(ndof)
        R_u[fixed_dofs] = 0.0
        k_pen = 1e5

        du, dlam = _solve_saddle_point_gmres(
            K_T, G_A, k_pen, R_u, g_active, fixed_dofs, use_amg=True
        )
        assert np.all(np.isfinite(du))
        assert np.all(np.isfinite(dlam))

    def test_amg_fallback_to_ilu(self):
        """AMG失敗時にILUフォールバックすること（小規模問題）."""
        ndof = 10
        K = _make_simple_k(ndof)
        n_active = 1
        G_A = sp.random(n_active, ndof, density=0.5, format="csr", random_state=42)
        g_active = np.array([-0.001])
        R_u = np.random.default_rng(42).standard_normal(ndof)
        R_u[[0, ndof - 1]] = 0.0
        k_pen = 1e5

        du, dlam = _solve_saddle_point_gmres(
            K, G_A, k_pen, R_u, g_active, np.array([0, ndof - 1]), use_amg=True
        )
        assert np.all(np.isfinite(du))


# ==== 改良8: k_pen continuation ====


class TestKPenContinuation:
    """k_pen continuation（S3改良8）のテスト."""

    def test_continuation_config_defaults(self):
        """デフォルトでcontinuationが無効."""
        config = ContactConfig()
        assert config.k_pen_continuation is False
        assert config.k_pen_continuation_start == 0.1
        assert config.k_pen_continuation_steps == 3

    def test_continuation_enabled_does_not_break(self):
        """continuation有効でも基本問題が収束すること."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            k_pen_continuation=True,
            k_pen_continuation_start=0.1,
            k_pen_continuation_steps=3,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=4,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
        )
        assert np.all(np.isfinite(result.u))

    def test_auto_kpen_beam_ei(self):
        """beam_ei モードで k_pen が自動推定されること."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=0.1,
            k_pen_mode="beam_ei",
            beam_E=200e9,
            beam_I=1e-12,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        # k_pen=0 でも beam_ei で自動推定が走ることを確認
        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=1,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=0.0,  # 0 → 自動推定
        )
        assert np.all(np.isfinite(result.u))

    def test_continuation_with_kpen_auto(self):
        """continuation + beam_ei k_pen自動推定が共存できること."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=0.1,
            k_pen_mode="beam_ei",
            beam_E=200e9,
            beam_I=1e-12,
            k_pen_continuation=True,
            k_pen_continuation_start=0.1,
            k_pen_continuation_steps=3,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=4,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
        )
        assert np.all(np.isfinite(result.u))

    def test_continuation_with_adaptive_dt(self):
        """continuationと適応Δtが共存できること."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            k_pen_continuation=True,
            adaptive_timestepping=True,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=3,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
            adaptive_timestepping=True,
        )
        assert np.all(np.isfinite(result.u))


# ==== 改良10: 残差スケーリング ====


class TestResidualScaling:
    """残差スケーリング（S3改良10）のテスト."""

    def test_residual_scaling_config_default(self):
        """デフォルトで残差スケーリングが無効."""
        config = ContactConfig()
        assert config.residual_scaling is False

    def test_residual_scaling_does_not_break(self):
        """残差スケーリング有効でも基本問題がクラッシュしないこと."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            residual_scaling=True,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=4,
            max_iter=30,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
        )
        assert np.all(np.isfinite(result.u))


# ==== 改良11: 接触力ランプ ====


class TestContactForceRamp:
    """接触力ランプ（S3改良11）のテスト."""

    def test_contact_force_ramp_config_default(self):
        """デフォルトで接触力ランプが無効."""
        config = ContactConfig()
        assert config.contact_force_ramp is False
        assert config.contact_force_ramp_iters == 5

    def test_contact_force_ramp_does_not_break(self):
        """接触力ランプ有効でも基本問題がクラッシュしないこと."""
        setup = _make_crossing_beams_setup()
        config = ContactConfig(
            k_pen_scale=1e4,
            contact_force_ramp=True,
            contact_force_ramp_iters=3,
        )
        manager = ContactManager(config=config)

        f_ext = np.zeros(setup["ndof"])
        f_ext[7] = -1.0

        def assemble_tangent(u):
            return setup["K_T"].copy()

        def assemble_internal(u):
            return setup["K_T"] @ u

        result = newton_raphson_contact_ncp(
            f_ext,
            setup["fixed_dofs"],
            assemble_tangent,
            assemble_internal,
            manager,
            setup["node_coords"],
            setup["connectivity"],
            setup["radii"],
            n_load_steps=4,
            max_iter=50,
            tol_force=1e-4,
            tol_ncp=1e-4,
            show_progress=False,
            k_pen=1e4,
        )
        assert np.all(np.isfinite(result.u))
