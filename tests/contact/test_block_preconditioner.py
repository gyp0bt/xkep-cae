"""ブロック前処理付き GMRES テスト — Phase C6-L4.

NCP 鞍点系のブロック対角前処理（接触 Schur 補集合）を検証する。

テスト構成:
- TestSaddlePointGMRES: GMRES ソルバー単体テスト（直接法との結果比較）
- TestBlockPreconditionerConvergence: ブロック前処理付き NCP ソルバーの収束検証
- TestPreconditionerConfig: ContactConfig フラグの動作確認
"""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.pair import ContactConfig, ContactManager, ContactStatus
from xkep_cae.contact.solver_ncp import (
    _build_constraint_jacobian,
    _solve_saddle_point_contact,
    _solve_saddle_point_direct,
    _solve_saddle_point_gmres,
    newton_raphson_contact_ncp,
)


def _make_spring_system(
    k_spring: float = 1e4,
    ndof_per_node: int = 6,
    z_sep: float = 0.041,
    radii: float = 0.04,
):
    """2本の直交ばね梁モデル（test_solver_ncp.py と同一）."""
    node_coords_ref = np.array(
        [
            [0.0, 0.0, z_sep],
            [1.0, 0.0, z_sep],
            [0.5, -0.5, -z_sep],
            [0.5, 0.5, -z_sep],
        ]
    )

    connectivity = np.array([[0, 1], [2, 3]])

    n_nodes = 4
    ndof_total = n_nodes * ndof_per_node

    k_rot = k_spring * 0.01

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                K[d0, d0] += k_spring
                K[d0, d1] -= k_spring
                K[d1, d0] -= k_spring
                K[d1, d1] += k_spring
            for d in range(3, ndof_per_node):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                K[d0, d0] += k_rot
                K[d0, d1] -= k_rot
                K[d1, d0] -= k_rot
                K[d1, d1] += k_rot
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_spring * delta
                f_int[d1] += k_spring * delta
            for d in range(3, ndof_per_node):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_rot * delta
                f_int[d1] += k_rot * delta
        return f_int

    fixed_dofs = np.array(
        [0 * ndof_per_node + d for d in range(ndof_per_node)]
        + [2 * ndof_per_node + d for d in range(ndof_per_node)],
        dtype=int,
    )

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        ndof_per_node,
        assemble_tangent,
        assemble_internal_force,
        fixed_dofs,
    )


def _setup_saddle_point_problem():
    """鞍点系テスト用の K_T, G_A, R_u, g_active を構築する."""
    from xkep_cae.contact.pair import ContactPair, ContactState

    (
        coords,
        conn,
        radii,
        ndof,
        ndof_per_node,
        K_fn,
        _,
        fixed_dofs,
    ) = _make_spring_system(z_sep=0.035, radii=0.04)

    u = np.zeros(ndof)
    K_T = K_fn(u)

    mgr = ContactManager(
        config=ContactConfig(k_pen_scale=1e5, g_on=0.01, g_off=0.02),
    )

    # 手動で ACTIVE ペアを設定
    state = ContactState(
        s=0.5,
        t=0.5,
        gap=-0.005,
        normal=np.array([0.0, 0.0, 1.0]),
        status=ContactStatus.ACTIVE,
        k_pen=1e5,
    )
    pair = ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        state=state,
        radius_a=0.04,
        radius_b=0.04,
    )
    mgr.pairs = [pair]

    G_A, active_idx = _build_constraint_jacobian(mgr, ndof, ndof_per_node)
    k_pen = 1e5
    R_u = np.random.default_rng(42).standard_normal(ndof) * 10.0
    R_u[fixed_dofs] = 0.0
    g_active = np.array([-0.005])

    return K_T, G_A, k_pen, R_u, g_active, fixed_dofs


class TestSaddlePointGMRES:
    """GMRES ソルバー単体テスト（直接法との結果比較）."""

    def test_gmres_vs_direct_displacement(self):
        """GMRES と直接 Schur complement の変位が一致すること."""
        K_T, G_A, k_pen, R_u, g_active, fixed_dofs = _setup_saddle_point_problem()

        du_direct, dlam_direct = _solve_saddle_point_direct(
            K_T, G_A, k_pen, R_u, g_active, fixed_dofs
        )

        du_gmres, dlam_gmres = _solve_saddle_point_gmres(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)

        np.testing.assert_allclose(du_gmres, du_direct, atol=1e-6, rtol=1e-4)

    def test_gmres_vs_direct_lambda(self):
        """GMRES と直接法のラグランジュ乗数増分が一致すること."""
        K_T, G_A, k_pen, R_u, g_active, fixed_dofs = _setup_saddle_point_problem()

        _, dlam_direct = _solve_saddle_point_direct(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)

        _, dlam_gmres = _solve_saddle_point_gmres(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)

        np.testing.assert_allclose(dlam_gmres, dlam_direct, atol=1e-6, rtol=1e-4)

    def test_gmres_boundary_conditions(self):
        """GMRES 解の拘束 DOF がゼロであること."""
        K_T, G_A, k_pen, R_u, g_active, fixed_dofs = _setup_saddle_point_problem()

        du_gmres, _ = _solve_saddle_point_gmres(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)

        np.testing.assert_allclose(du_gmres[fixed_dofs], 0.0, atol=1e-10)

    def test_dispatcher_direct_mode(self):
        """_solve_saddle_point_contact が直接法にディスパッチすること."""
        K_T, G_A, k_pen, R_u, g_active, fixed_dofs = _setup_saddle_point_problem()

        du_direct, dlam_direct = _solve_saddle_point_contact(
            K_T,
            G_A,
            k_pen,
            R_u,
            g_active,
            fixed_dofs,
            use_block_preconditioner=False,
        )

        du_ref, dlam_ref = _solve_saddle_point_direct(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)

        np.testing.assert_allclose(du_direct, du_ref, atol=1e-12)
        np.testing.assert_allclose(dlam_direct, dlam_ref, atol=1e-12)

    def test_dispatcher_gmres_mode(self):
        """_solve_saddle_point_contact が GMRES にディスパッチすること."""
        K_T, G_A, k_pen, R_u, g_active, fixed_dofs = _setup_saddle_point_problem()

        du_gmres, dlam_gmres = _solve_saddle_point_contact(
            K_T,
            G_A,
            k_pen,
            R_u,
            g_active,
            fixed_dofs,
            use_block_preconditioner=True,
        )

        # 直接法と比較
        du_ref, dlam_ref = _solve_saddle_point_direct(K_T, G_A, k_pen, R_u, g_active, fixed_dofs)

        np.testing.assert_allclose(du_gmres, du_ref, atol=1e-6, rtol=1e-4)


class TestBlockPreconditionerConvergence:
    """ブロック前処理付き NCP ソルバーの収束検証."""

    def test_ncp_with_block_preconditioner_converges(self):
        """ブロック前処理付き NCP ソルバーが収束すること."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                ncp_block_preconditioner=True,
            ),
        )

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -50.0
        f_ext[3 * ndof_per_node + 2] = 50.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result.converged, (
            f"Block preconditioner NCP solver did not converge, "
            f"{result.total_newton_iterations} iters"
        )

    def test_block_vs_direct_displacement_consistency(self):
        """ブロック前処理と直接法で変位が近いこと."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -30.0
        f_ext[3 * ndof_per_node + 2] = 30.0

        # 直接法
        mgr_direct = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                ncp_block_preconditioner=False,
            ),
        )
        result_direct = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_direct,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        # GMRES
        mgr_gmres = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                ncp_block_preconditioner=True,
            ),
        )
        result_gmres = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_gmres,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result_direct.converged, "Direct NCP solver did not converge"
        assert result_gmres.converged, "GMRES NCP solver did not converge"

        # 変位の比較（非線形反復パスが異なるので厳密一致は期待しない）
        # z方向変位の符号と大きさのオーダーが一致すること
        uz_direct = result_direct.u[1 * ndof_per_node + 2]
        uz_gmres = result_gmres.u[1 * ndof_per_node + 2]

        assert abs(uz_direct) > 1e-8, "Direct: nonzero z disp expected"
        assert abs(uz_gmres) > 1e-8, "GMRES: nonzero z disp expected"
        assert np.sign(uz_direct) == np.sign(uz_gmres), (
            f"Sign mismatch: direct={uz_direct:.6f}, gmres={uz_gmres:.6f}"
        )

        # 相対差が 10% 以内
        rel_diff = abs(uz_gmres - uz_direct) / max(abs(uz_direct), 1e-15)
        assert rel_diff < 0.1, f"Relative displacement difference too large: {rel_diff:.3f}"

    def test_no_contact_with_block_preconditioner(self):
        """接触なし（梁が離間）でブロック前処理でも正常終了."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.5)

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.0,
                g_off=1e-4,
                ncp_block_preconditioner=True,
            ),
        )

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -1.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=1,
            max_iter=20,
            show_progress=False,
        )

        assert result.converged

    def test_lambda_nonnegative_with_block_preconditioner(self):
        """ブロック前処理でも λ >= 0 が保証されること."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                ncp_block_preconditioner=True,
            ),
        )

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=3,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result.converged
        assert np.all(result.lambdas >= 0.0), "λ must be non-negative"


class TestPreconditionerConfig:
    """ContactConfig フラグの動作確認."""

    def test_default_is_direct(self):
        """デフォルトはブロック前処理なし."""
        config = ContactConfig()
        assert config.ncp_block_preconditioner is False

    def test_config_enables_block_preconditioner(self):
        """設定でブロック前処理を有効化できること."""
        config = ContactConfig(ncp_block_preconditioner=True)
        assert config.ncp_block_preconditioner is True
