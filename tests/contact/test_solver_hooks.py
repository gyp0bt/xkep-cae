"""接触付き Newton-Raphson ソルバーのテスト.

Phase C2: solver_hooks.py の統合テスト。
摩擦なし梁–梁接触が安定収束することを検証する。
"""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_hooks import (
    _deformed_coords,
    newton_raphson_with_contact,
)

# ====================================================================
# ヘルパー: 簡易ばねモデル（接触検証用）
# ====================================================================


def _all_fixed_except_z_free(n_nodes, ndof_per_node=6, z_fixed_nodes=None):
    """全 DOF を固定し、z 並進のみ自由にする fixed_dofs リストを返す.

    z_fixed_nodes: z も固定するノードのリスト
    """
    if z_fixed_nodes is None:
        z_fixed_nodes = []
    fixed = []
    for node in range(n_nodes):
        for d in range(ndof_per_node):
            if d == 2 and node not in z_fixed_nodes:
                continue  # z 並進は自由
            fixed.append(node * ndof_per_node + d)
    return np.array(fixed, dtype=int)


def _make_spring_system(
    k_spring: float = 1e4,
    ndof_per_node: int = 6,
    z_sep: float = 0.041,
    radii: float = 0.04,
):
    """2本の交差するばね要素の組み立て関数を返す.

    梁A: ノード 0–1 (x方向, z=+z_sep)
    梁B: ノード 2–3 (y方向, z=-z_sep, Aの中点で交差)

    交差配置により closest_point_segments が s≈0.5, t≈0.5 を返し、
    接触力が固定端・自由端の両方に分配される。

    各ノード 6 DOF。z 方向のみばね剛性あり。
    BC: node0, node2 の z を固定（各梁に1拘束 → 剛体モード排除）。
    """
    n_nodes = 4
    ndof_total = n_nodes * ndof_per_node

    # 参照座標: AとBが z 方向で対向し、中点で直交交差
    # 距離 = 2*z_sep, gap = 2*z_sep - 2*radii
    node_coords_ref = np.array(
        [
            [0.0, 0.0, z_sep],  # node 0 (A固定端)
            [1.0, 0.0, z_sep],  # node 1 (A自由端)
            [0.5, -0.5, -z_sep],  # node 2 (B固定端)
            [0.5, 0.5, -z_sep],  # node 3 (B自由端)
        ]
    )

    connectivity = np.array(
        [
            [0, 1],  # elem 0 (A)
            [2, 3],  # elem 1 (B)
        ]
    )

    def assemble_tangent(u):
        """簡易剛性: z 方向のみばね."""
        K = sp.lil_matrix((ndof_total, ndof_total))
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            d0 = n0 * ndof_per_node + 2  # z DOF
            d1 = n1 * ndof_per_node + 2
            K[d0, d0] += k_spring
            K[d0, d1] -= k_spring
            K[d1, d0] -= k_spring
            K[d1, d1] += k_spring
        return K.tocsr()

    def assemble_internal_force(u):
        """簡易内力: z 方向のばね力."""
        f_int = np.zeros(ndof_total)
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            d0 = n0 * ndof_per_node + 2
            d1 = n1 * ndof_per_node + 2
            delta = u[d1] - u[d0]
            f_int[d0] -= k_spring * delta
            f_int[d1] += k_spring * delta
        return f_int

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    )


class TestDeformedCoords:
    """_deformed_coords のテスト."""

    def test_zero_displacement(self):
        """変位ゼロで参照座標と一致."""
        ref = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        u = np.zeros(12)
        coords_def = _deformed_coords(ref, u, ndof_per_node=6)
        assert np.allclose(coords_def, ref)

    def test_translation(self):
        """並進変位が正しく加算される."""
        ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        u = np.zeros(12)
        u[0] = 0.1  # node0 ux
        u[1] = 0.2  # node0 uy
        u[2] = 0.3  # node0 uz
        coords_def = _deformed_coords(ref, u, ndof_per_node=6)
        assert np.allclose(coords_def[0], [0.1, 0.2, 0.3])
        assert np.allclose(coords_def[1], [1.0, 0.0, 0.0])


class TestContactSolverNoContact:
    """接触が発生しない場合のソルバーテスト."""

    def test_no_contact_reduces_to_nr(self):
        """接触なしの場合、通常の NR と同じ結果になる.

        梁Aの自由端(node1)に上向き微小荷重 → 接触しない。
        """
        (
            node_coords_ref,
            connectivity,
            radii,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_spring_system()

        # 小さな上向き荷重（梁同士が離れる方向）
        f_ext = np.zeros(ndof_total)
        f_ext[1 * 6 + 2] = 0.001  # node1 z 上向き

        # node0, node2 の z を固定、全ノードの x,y,回転を固定
        fixed_dofs = _all_fixed_except_z_free(4, z_fixed_nodes=[0, 2])

        mgr = ContactManager(
            config=ContactConfig(k_pen_scale=1e5, g_on=0.0, g_off=1e-6),
        )

        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            node_coords_ref,
            connectivity,
            radii,
            n_load_steps=1,
            show_progress=False,
        )
        assert result.converged
        assert result.n_active_final == 0


class TestContactSolverWithContact:
    """接触が発生する場合のソルバーテスト."""

    def test_two_beams_pressed_together(self):
        """2本の梁の自由端を押し付けて接触が安定収束する.

        梁A: node0(固定)–node1(自由, z↓荷重)
        梁B: node2(固定)–node3(自由, z↑荷重)

        初期 gap = 2*0.041 - 2*0.04 = 0.002
        自由端変位 = F/k = 50/1e4 = 0.005 (片側) → 合計 0.01 > gap
        """
        (
            node_coords_ref,
            connectivity,
            radii,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_spring_system(k_spring=1e4, z_sep=0.041, radii=0.04)

        # 自由端を互いに近づける荷重
        f_ext = np.zeros(ndof_total)
        f_ext[1 * 6 + 2] = -50.0  # node1 z↓
        f_ext[3 * 6 + 2] = 50.0  # node3 z↑

        fixed_dofs = _all_fixed_except_z_free(4, z_fixed_nodes=[0, 2])

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.0,
                g_off=1e-4,
                n_outer_max=5,
                tol_geometry=1e-6,
            ),
        )

        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            node_coords_ref,
            connectivity,
            radii,
            n_load_steps=10,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result.converged
        # 接触が発生しているか確認
        assert result.n_active_final > 0

    def test_contact_prevents_penetration(self):
        """接触力が貫通を防止する.

        初期 gap = 0.002, F/k = 100/1e4 = 0.01 >> gap
        接触ペナルティが貫通を防止する。
        """
        (
            node_coords_ref,
            connectivity,
            radii,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_spring_system(k_spring=1e4, z_sep=0.041, radii=0.04)

        # 大きな荷重で押し付け
        f_ext = np.zeros(ndof_total)
        f_ext[1 * 6 + 2] = -100.0  # node1 z↓
        f_ext[3 * 6 + 2] = 100.0  # node3 z↑

        fixed_dofs = _all_fixed_except_z_free(4, z_fixed_nodes=[0, 2])

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.0,
                g_off=1e-4,
                n_outer_max=5,
            ),
        )

        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            node_coords_ref,
            connectivity,
            radii,
            n_load_steps=20,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result.converged

        # 変形後の距離を確認
        u = result.u
        za1 = node_coords_ref[1, 2] + u[1 * 6 + 2]
        zb1 = node_coords_ref[3, 2] + u[3 * 6 + 2]
        dist = abs(za1 - zb1)

        # 断面半径の和 (0.04 + 0.04 = 0.08) に近い距離を保持
        # ペナルティなので微小な貫通は許容（10% まで）
        assert dist >= 0.08 * 0.8, f"Excessive penetration: dist={dist:.4f}"

    def test_contact_force_history(self):
        """接触力履歴が記録される."""
        (
            node_coords_ref,
            connectivity,
            radii,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_spring_system(k_spring=1e4, z_sep=0.041, radii=0.04)

        f_ext = np.zeros(ndof_total)
        f_ext[1 * 6 + 2] = -50.0  # node1 z↓
        f_ext[3 * 6 + 2] = 50.0  # node3 z↑

        fixed_dofs = _all_fixed_except_z_free(4, z_fixed_nodes=[0, 2])

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.0,
                g_off=1e-4,
                n_outer_max=3,
            ),
        )

        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            node_coords_ref,
            connectivity,
            radii,
            n_load_steps=10,
            show_progress=False,
            broadphase_margin=0.05,
        )

        assert result.converged
        assert len(result.contact_force_history) == 10
        assert len(result.load_history) == 10
        assert len(result.displacement_history) == 10


class TestContactManagerPhaseC2:
    """ContactManager の Phase C2 追加メソッドのテスト."""

    def test_evaluate_contact_forces(self):
        """evaluate_contact_forces で全ペアの p_n が更新される."""
        mgr = ContactManager(config=ContactConfig())
        pair = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]), 0.1, 0.1)
        pair.state.status = ContactStatus.ACTIVE
        pair.state.gap = -0.01
        pair.state.k_pen = 1e4
        pair.state.lambda_n = 0.0

        mgr.evaluate_contact_forces()
        assert abs(pair.state.p_n - 100.0) < 1e-10

    def test_update_al_multipliers(self):
        """update_al_multipliers で全ペアの lambda_n が更新される."""
        mgr = ContactManager(config=ContactConfig())
        pair = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]), 0.1, 0.1)
        pair.state.status = ContactStatus.ACTIVE
        pair.state.gap = -0.01
        pair.state.k_pen = 1e4
        pair.state.lambda_n = 0.0
        pair.state.p_n = 100.0

        mgr.update_al_multipliers()
        assert abs(pair.state.lambda_n - 100.0) < 1e-10

    def test_initialize_penalty(self):
        """initialize_penalty で k_pen, k_t が設定される."""
        mgr = ContactManager(config=ContactConfig(k_t_ratio=0.3))
        pair = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]), 0.1, 0.1)
        pair.state.status = ContactStatus.ACTIVE

        mgr.initialize_penalty(k_pen=1e5)
        assert abs(pair.state.k_pen - 1e5) < 1e-10
        assert abs(pair.state.k_t - 3e4) < 1e-10

    def test_initialize_penalty_skips_inactive(self):
        """INACTIVE ペアは初期化されない."""
        mgr = ContactManager(config=ContactConfig())
        pair = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]), 0.1, 0.1)
        pair.state.status = ContactStatus.INACTIVE

        mgr.initialize_penalty(k_pen=1e5)
        assert pair.state.k_pen == 0.0

    def test_initialize_penalty_skips_already_set(self):
        """既に k_pen が設定されたペアはスキップ."""
        mgr = ContactManager(config=ContactConfig())
        pair = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]), 0.1, 0.1)
        pair.state.status = ContactStatus.ACTIVE
        pair.state.k_pen = 5e4  # 既設定

        mgr.initialize_penalty(k_pen=1e5)
        assert abs(pair.state.k_pen - 5e4) < 1e-10  # 変更されない
