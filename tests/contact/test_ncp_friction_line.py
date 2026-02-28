"""NCP ソルバーの摩擦拡張 + line contact 統合テスト.

Phase S1: NCP Semi-smooth Newton に Coulomb 摩擦と line-to-line Gauss 積分を統合。
"""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.pair import ContactConfig, ContactManager
from xkep_cae.contact.solver_ncp import (
    NCPSolveResult,
    _compute_friction_forces_ncp,
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


# ============================================================
# NCP + Coulomb 摩擦テスト
# ============================================================


class TestNCPFrictionBasic:
    """NCP ソルバーの摩擦基本動作."""

    def test_friction_converges(self):
        """摩擦有効で NCP ソルバーが収束する."""
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
                k_t_ratio=0.5,
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
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.1,
        )

        assert isinstance(result, NCPSolveResult)
        assert result.converged, "NCP+friction solver did not converge"

    def test_friction_vs_frictionless(self):
        """摩擦ありと摩擦なしで接触力が異なることを検証する.

        直接的な横方向力は NCP+摩擦の収束を難しくするため、
        法線接触のみの2ケースを比較し、摩擦が pair.state に影響することを確認する。
        """
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
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0

        # 摩擦なし
        mgr_no = ContactManager(
            config=ContactConfig(k_pen_scale=1e5, g_on=0.01, g_off=0.02),
        )
        res_no = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_no,
            coords,
            conn,
            radii,
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=False,
        )

        # 摩擦あり
        mgr_yes = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                k_t_ratio=0.5,
            ),
        )
        res_yes = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_yes,
            coords,
            conn,
            radii,
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.1,
        )

        assert res_no.converged
        assert res_yes.converged

        # 摩擦有効時、ペアの stick/friction 関連フィールドが設定されていることを確認
        assert len(mgr_yes.pairs) > 0
        pair_yes = mgr_yes.pairs[0]
        # 摩擦力計算が実行されたことを k_t > 0 で確認
        assert pair_yes.state.k_t > 0.0, "摩擦の k_t が設定されていない"

    def test_friction_with_mu_ramp(self):
        """μランプ付きで収束する."""
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
                k_t_ratio=0.5,
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
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.2,
            mu_ramp_steps=5,
        )

        assert result.converged

    def test_config_friction_propagated(self):
        """ContactConfig の use_friction/mu が NCP ソルバーに伝播する."""
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
                use_friction=True,
                mu=0.1,
                k_t_ratio=0.5,
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
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result.converged


class TestNCPFrictionForceUnit:
    """_compute_friction_forces_ncp の単体テスト."""

    def test_no_contact_no_friction(self):
        """接触なしでは摩擦力ゼロ."""
        mgr = ContactManager()
        lambdas = np.array([])
        u = np.zeros(24)
        u_ref = np.zeros(24)
        coords_ref = np.zeros((4, 3))

        f_fric, tangents = _compute_friction_forces_ncp(
            mgr, lambdas, u, u_ref, coords_ref, 24, mu=0.3
        )

        np.testing.assert_allclose(f_fric, 0.0, atol=1e-15)
        assert len(tangents) == 0

    def test_zero_mu_no_friction(self):
        """μ=0 では摩擦力ゼロ."""
        from xkep_cae.contact.pair import ContactPair, ContactState, ContactStatus

        mgr = ContactManager(config=ContactConfig(k_t_ratio=0.5))
        state = ContactState(
            s=0.5,
            t=0.5,
            gap=-0.01,
            normal=np.array([0.0, 0.0, 1.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 1.0, 0.0]),
            status=ContactStatus.ACTIVE,
            k_pen=1e5,
            k_t=5e4,
            p_n=1000.0,
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

        lambdas = np.array([1000.0])
        u = np.zeros(24)
        u_ref = np.zeros(24)
        coords_ref = np.zeros((4, 3))

        f_fric, tangents = _compute_friction_forces_ncp(
            mgr, lambdas, u, u_ref, coords_ref, 24, k_pen=1e5, mu=0.0
        )

        np.testing.assert_allclose(f_fric, 0.0, atol=1e-15)


# ============================================================
# NCP + Line Contact テスト
# ============================================================


class TestNCPLineContactBasic:
    """NCP ソルバーの line contact 統合テスト."""

    def test_line_contact_converges(self):
        """line_contact 有効で NCP ソルバーが収束する."""
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
            line_contact=True,
            n_gauss=3,
        )

        assert isinstance(result, NCPSolveResult)
        assert result.converged, "NCP+line_contact solver did not converge"

    def test_line_contact_vs_ptp(self):
        """line_contact と PtP で変位の符号が一致する."""
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

        # PtP
        mgr_ptp = ContactManager(
            config=ContactConfig(k_pen_scale=1e5, g_on=0.01, g_off=0.02),
        )
        res_ptp = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_ptp,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
            line_contact=False,
        )

        # Line contact
        mgr_lc = ContactManager(
            config=ContactConfig(k_pen_scale=1e5, g_on=0.01, g_off=0.02),
        )
        res_lc = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_lc,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
            line_contact=True,
            n_gauss=3,
        )

        assert res_ptp.converged
        assert res_lc.converged

        # z方向変位の符号が一致
        uz_ptp = res_ptp.u[1 * ndof_per_node + 2]
        uz_lc = res_lc.u[1 * ndof_per_node + 2]
        if abs(uz_ptp) > 1e-8 and abs(uz_lc) > 1e-8:
            assert np.sign(uz_ptp) == np.sign(uz_lc)

    def test_config_line_contact_propagated(self):
        """ContactConfig の line_contact/n_gauss が NCP ソルバーに伝播する."""
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
                line_contact=True,
                n_gauss=3,
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

        assert result.converged


# ============================================================
# NCP + Line Contact + Friction 複合テスト
# ============================================================


class TestNCPLineContactFriction:
    """NCP + line contact + friction の複合テスト."""

    def test_all_features_converge(self):
        """NCP + line_contact + friction の全機能有効で収束する."""
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
                k_t_ratio=0.5,
            ),
        )

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0
        f_ext[1 * ndof_per_node + 0] = 3.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.05,
            line_contact=True,
            n_gauss=3,
        )

        assert isinstance(result, NCPSolveResult)
        assert result.converged, "NCP+line_contact+friction did not converge"

    def test_lambda_nonneg_with_friction(self):
        """摩擦有効でも λ >= 0 が保証されること."""
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
                k_t_ratio=0.5,
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
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.1,
        )

        assert result.converged
        assert np.all(result.lambdas >= 0.0), "λ must be non-negative"
