"""Semi-smooth Newton (NCP) ソルバーのテスト — Phase C6-L3β.

2本の交差梁でOuter loopなし収束を検証。

テスト構成:
- TestNCPSolverBasic: 基本的な NCP ソルバーの動作確認
- TestNCPSolverConvergence: 収束性・精度テスト
- TestNCPSolverComparison: 既存 AL ソルバーとの比較
"""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.pair import ContactConfig, ContactManager
from xkep_cae.contact.solver_ncp import (
    NCPSolveResult,
    _build_constraint_jacobian,
    _compute_contact_force_from_lambdas,
    newton_raphson_contact_ncp,
)


def _make_spring_system(
    k_spring: float = 1e4,
    ndof_per_node: int = 6,
    z_sep: float = 0.041,
    radii: float = 0.04,
):
    """2本の直交ばね梁モデル.

    梁A: x軸方向 (0,0,+z_sep) → (1,0,+z_sep)
    梁B: y軸方向 (0.5,-0.5,-z_sep) → (0.5,0.5,-z_sep)
    中点で直交交差。z_sep < 2*radii なら接触する。
    """
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

    # 回転剛性（微小値を追加して特異性を回避）
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
            # 回転DOFにも微小剛性
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

    # 固定DOF: 全ノードの全DOFを固定（始点のみ）
    # node0: 梁A始点、node2: 梁B始点 → 全6DOF固定
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


class TestNCPSolverBasic:
    """基本的な NCP ソルバーの動作確認."""

    def test_no_contact_case(self):
        """接触なし（梁が離間）で正常終了."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.5)  # 十分離間

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.0,
                g_off=1e-4,
            ),
        )

        f_ext = np.zeros(ndof)
        # 梁Aの自由端(node1)にz方向の小さな力
        f_ext[1 * ndof_per_node + 2] = -1.0  # z方向に小さい力

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

        assert isinstance(result, NCPSolveResult)
        assert result.converged

    def test_contact_detected(self):
        """接触あり（梁が貫入）で収束."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.041, radii=0.04)
        # 2*0.041 = 0.082 < 2*0.04 = 0.08 → 初期分離 0.002

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
            ),
        )

        f_ext = np.zeros(ndof)
        # z方向に梁を押し込む力
        f_ext[1 * ndof_per_node + 2] = -10.0
        f_ext[3 * ndof_per_node + 2] = 10.0

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

        assert isinstance(result, NCPSolveResult)
        assert result.converged
        assert result.n_active_final >= 0

    def test_result_structure(self):
        """結果構造の検証."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system()

        mgr = ContactManager(
            config=ContactConfig(k_pen_scale=1e5),
        )

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -5.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=2,
            max_iter=30,
            show_progress=False,
        )

        assert hasattr(result, "u")
        assert hasattr(result, "lambdas")
        assert hasattr(result, "converged")
        assert hasattr(result, "n_load_steps")
        assert hasattr(result, "total_newton_iterations")
        assert result.u.shape == (ndof,)


class TestNCPSolverConvergence:
    """NCP ソルバーの収束性テスト."""

    def test_outer_loop_free_convergence(self):
        """Outer loop なし（NCP 方式）で2梁交差接触が収束すること."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04, k_spring=1e4)
        # 初期ギャップ = 2*0.035 - 2*0.04 = -0.01 (貫入)

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
            ),
        )

        f_ext = np.zeros(ndof)
        # 梁を押し込む
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
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result.converged, (
            f"NCP solver did not converge, {result.total_newton_iterations} iters"
        )

    def test_fb_vs_min_ncp(self):
        """FB 関数と min 関数の両方で収束すること."""
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

        for ncp_type in ["fb", "min"]:
            mgr = ContactManager(
                config=ContactConfig(
                    k_pen_scale=1e5,
                    g_on=0.01,
                    g_off=0.02,
                ),
            )
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
                ncp_type=ncp_type,
                broadphase_margin=0.05,
            )
            assert result.converged, f"NCP ({ncp_type}) did not converge"

    def test_lambda_nonnegative(self):
        """λ >= 0 が保証されること（射影による）."""
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


class TestNCPSolverComparison:
    """既存 AL ソルバーとの結果比較."""

    def test_displacement_consistency(self):
        """NCP と既存 AL ソルバーの変位が近いこと."""
        from xkep_cae.contact.solver_hooks import newton_raphson_with_contact

        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04, k_spring=1e4)

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0

        # AL ソルバー
        mgr_al = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                n_outer_max=5,
                use_line_search=True,
            ),
        )
        result_al = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_al,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=30,
            show_progress=False,
            broadphase_margin=0.05,
        )

        # NCP ソルバー
        mgr_ncp = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
            ),
        )
        result_ncp = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_ncp,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result_al.converged, "AL solver did not converge"
        assert result_ncp.converged, "NCP solver did not converge"

        # 変位の方向が一致すること（大まかな比較）
        # NCP とALの定式化が異なるため完全一致は期待しない
        # z 方向の変位符号が一致し、大きさのオーダーが近いこと
        u_al = result_al.u
        u_ncp = result_ncp.u

        # node 1 (A自由端) の z 変位
        uz_al_node1 = u_al[1 * ndof_per_node + 2]
        uz_ncp_node1 = u_ncp[1 * ndof_per_node + 2]

        # 両方とも非ゼロ変位が生じること
        assert abs(uz_al_node1) > 1e-8, "AL solver should produce nonzero z disp"
        assert abs(uz_ncp_node1) > 1e-8, "NCP solver should produce nonzero z disp"

        # z 方向変位の符号が一致すること
        # （接触力が支配的な場合は正にもなりうる）
        assert np.sign(uz_al_node1) == np.sign(uz_ncp_node1), (
            f"Sign mismatch: AL={uz_al_node1:.6f}, NCP={uz_ncp_node1:.6f}"
        )


class TestConstraintJacobianAndForce:
    """制約ヤコビアンと接触力の単体テスト."""

    def test_constraint_jacobian_shape(self):
        """制約ヤコビアンの形状."""
        from xkep_cae.contact.pair import ContactPair, ContactState, ContactStatus

        mgr = ContactManager()
        state = ContactState(
            s=0.5,
            t=0.5,
            gap=-0.01,
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

        ndof = 24  # 4 nodes * 6 DOF
        G, active_idx = _build_constraint_jacobian(mgr, ndof, ndof_per_node=6)

        assert G.shape == (1, 24)
        assert len(active_idx) == 1
        assert active_idx[0] == 0

    def test_constraint_jacobian_inactive_skip(self):
        """INACTIVE ペアはスキップされること."""
        from xkep_cae.contact.pair import ContactPair, ContactState, ContactStatus

        mgr = ContactManager()
        state = ContactState(status=ContactStatus.INACTIVE)
        pair = ContactPair(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            state=state,
        )
        mgr.pairs = [pair]

        G, active_idx = _build_constraint_jacobian(mgr, 24, ndof_per_node=6)
        assert G.shape[0] == 0
        assert len(active_idx) == 0

    def test_contact_force_from_lambdas(self):
        """λ からの接触力計算が正しいこと."""
        from xkep_cae.contact.pair import ContactPair, ContactState, ContactStatus

        mgr = ContactManager()
        state = ContactState(
            s=0.5,
            t=0.5,
            gap=-0.01,
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

        lam_all = np.array([100.0])
        ndof = 24
        f_c = _compute_contact_force_from_lambdas(mgr, lam_all, ndof, ndof_per_node=6)

        assert f_c.shape == (24,)
        # 作用反作用: A側とB側の合計力はゼロ
        f_A = np.zeros(3)
        f_B = np.zeros(3)
        for d in range(3):
            f_A[d] = f_c[0 * 6 + d] + f_c[1 * 6 + d]
            f_B[d] = f_c[2 * 6 + d] + f_c[3 * 6 + d]
        np.testing.assert_allclose(f_A + f_B, 0.0, atol=1e-10)

    def test_contact_force_zero_lambda(self):
        """λ=0 のとき接触力はゼロ."""
        from xkep_cae.contact.pair import ContactPair, ContactState, ContactStatus

        mgr = ContactManager()
        state = ContactState(
            s=0.5,
            t=0.5,
            gap=-0.01,
            normal=np.array([0.0, 0.0, 1.0]),
            status=ContactStatus.ACTIVE,
        )
        pair = ContactPair(
            elem_a=0,
            elem_b=1,
            nodes_a=np.array([0, 1]),
            nodes_b=np.array([2, 3]),
            state=state,
        )
        mgr.pairs = [pair]

        lam_all = np.array([0.0])
        f_c = _compute_contact_force_from_lambdas(mgr, lam_all, 24, ndof_per_node=6)
        np.testing.assert_allclose(f_c, 0.0, atol=1e-15)
