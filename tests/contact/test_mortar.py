"""Mortar 離散化テスト — Phase C6-L5.

Mortar 方式でスレーブ節点ベースの λ を使い、セグメント境界の接触圧連続性を検証。
"""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.mortar import (
    build_mortar_system,
    compute_mortar_p_n,
    identify_mortar_nodes,
)
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_ncp import (
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
    n_nodes = 4
    ndof = n_nodes * ndof_per_node
    connectivity = np.array([[0, 1], [2, 3]])

    K_diag = np.ones(ndof) * k_spring
    K_global = sp.diags(K_diag).tocsr()

    fixed_dofs = np.array(
        [0 * ndof_per_node + d for d in range(ndof_per_node)]
        + [1 * ndof_per_node + d for d in range(ndof_per_node)]
        + [2 * ndof_per_node + d for d in range(ndof_per_node)]
        + [3 * ndof_per_node + d for d in range(ndof_per_node)],
        dtype=int,
    )
    # ただし z 方向のみ自由
    fixed_dofs = np.array(
        [i * ndof_per_node + d for i in range(n_nodes) for d in range(ndof_per_node) if d != 2],
        dtype=int,
    )

    f_ext = np.zeros(ndof)
    # A 梁を下方に、B 梁を上方に押す
    f_ext[0 * ndof_per_node + 2] = -5.0
    f_ext[1 * ndof_per_node + 2] = -5.0
    f_ext[2 * ndof_per_node + 2] = 5.0
    f_ext[3 * ndof_per_node + 2] = 5.0

    def assemble_tangent(u):
        return K_global.copy()

    def assemble_internal(u):
        return K_global @ u

    return {
        "f_ext": f_ext,
        "fixed_dofs": fixed_dofs,
        "assemble_tangent": assemble_tangent,
        "assemble_internal": assemble_internal,
        "node_coords_ref": node_coords_ref,
        "connectivity": connectivity,
        "radii": radii,
        "ndof": ndof,
        "ndof_per_node": ndof_per_node,
        "k_spring": k_spring,
    }


# ---- TestMortarBasic ----


class TestMortarBasic:
    """Mortar 基本テスト."""

    def test_mortar_converges(self):
        """Mortar + line_contact で NCP 収束."""
        sys = _make_spring_system()
        config = ContactConfig(
            k_pen_scale=1e5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )
        manager = ContactManager(config=config)
        result = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            manager,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
            use_mortar=True,
            line_contact=True,
            n_gauss=2,
        )
        assert result.converged

    def test_mortar_lambda_nonneg(self):
        """Mortar 乗数は非負."""
        sys = _make_spring_system()
        config = ContactConfig(k_pen_scale=1e5, line_contact=True, use_mortar=True)
        manager = ContactManager(config=config)
        result = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            manager,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
            use_mortar=True,
            line_contact=True,
        )
        assert result.converged
        # λ_all は全て ≥ 0
        assert np.all(result.lambdas >= -1e-12)

    def test_config_mortar_propagated(self):
        """ContactConfig.use_mortar が伝播されるか."""
        config = ContactConfig(use_mortar=True)
        assert config.use_mortar is True
        config2 = ContactConfig()
        assert config2.use_mortar is False


# ---- TestMortarWeightedGap ----


class TestMortarWeightedGap:
    """Mortar 重み付きギャップのユニットテスト."""

    def _make_two_pair_manager(self):
        """共有節点を持つ2ペアのマネージャ."""
        config = ContactConfig(k_pen_scale=1e5, line_contact=True, use_mortar=True)
        manager = ContactManager(config=config)
        # ペア1: A=[0,1], B=[3,4]
        manager.add_pair(0, 1, np.array([0, 1]), np.array([3, 4]), radius_a=0.04, radius_b=0.04)
        # ペア2: A=[1,2], B=[4,5]  — 節点1を共有
        manager.add_pair(1, 2, np.array([1, 2]), np.array([4, 5]), radius_a=0.04, radius_b=0.04)
        return manager

    def test_identify_mortar_nodes(self):
        """共有節点を持つ2ペアから Mortar 節点を抽出."""
        manager = self._make_two_pair_manager()

        # 手動でアクティブ化
        for pair in manager.pairs:
            pair.state.status = ContactStatus.ACTIVE
            pair.state.gap = 0.01
            pair.state.normal = np.array([0.0, 0.0, 1.0])
            pair.state.s = 0.5
            pair.state.t = 0.5

        mortar_nodes, node_to_pairs = identify_mortar_nodes(manager, [0, 1])
        # A 側ノード: 0, 1, 2 の3つ（1は共有）
        assert mortar_nodes == [0, 1, 2]
        # 節点1は2つのペアから参照される
        assert len(node_to_pairs[1]) == 2

    def test_mortar_gap_uniform(self):
        """均一ギャップ → 重み付きギャップが正しく計算される."""
        manager = self._make_two_pair_manager()

        # 座標: A梁が z=+0.1, B梁が z=-0.1 (gap = 0.2 - 2*0.04 = 0.12)
        z_sep = 0.1
        node_coords = np.array(
            [
                [0.0, 0.0, z_sep],
                [1.0, 0.0, z_sep],
                [2.0, 0.0, z_sep],
                [0.0, 0.0, -z_sep],
                [1.0, 0.0, -z_sep],
                [2.0, 0.0, -z_sep],
            ]
        )

        for pair in manager.pairs:
            pair.state.status = ContactStatus.ACTIVE
            pair.state.gap = 0.12
            pair.state.normal = np.array([0.0, 0.0, 1.0])
            pair.state.s = 0.5
            pair.state.t = 0.5

        mortar_nodes, _ = identify_mortar_nodes(manager, [0, 1])
        ndof = 6 * 6
        G_mortar, g_mortar = build_mortar_system(
            manager,
            [0, 1],
            mortar_nodes,
            node_coords,
            ndof,
            ndof_per_node=6,
            n_gauss=2,
            k_pen=1e5,
        )

        # 重み付きギャップは全て正（非貫入）
        assert len(g_mortar) == 3
        assert np.all(g_mortar > 0)

    def test_mortar_p_n_active(self):
        """貫入時に正の p_n."""
        mortar_nodes = [0, 1]
        lam = np.array([100.0, 200.0])
        g = np.array([-0.01, -0.02])  # 貫入（負ギャップ）
        k_pen = 1e5

        p_n = compute_mortar_p_n(mortar_nodes, lam, g, k_pen)
        assert p_n[0] > 0
        assert p_n[1] > 0
        # p_n = max(0, λ + k_pen * (-g)) = max(0, 100 + 1e5*0.01) = 1100
        assert abs(p_n[0] - (100.0 + 1e5 * 0.01)) < 1e-6


# ---- TestMortarMultiSegment ----


class TestMortarMultiSegment:
    """マルチセグメント Mortar テスト."""

    def test_parallel_beams_converge(self):
        """3セグメント平行梁で Mortar 収束."""
        # A: 4 nodes (0,1,2,3), B: 4 nodes (4,5,6,7)
        z_sep = 0.041
        radii = 0.04
        node_coords_ref = np.array(
            [
                [0.0, 0.0, z_sep],
                [1.0, 0.0, z_sep],
                [2.0, 0.0, z_sep],
                [3.0, 0.0, z_sep],
                [0.0, 0.0, -z_sep],
                [1.0, 0.0, -z_sep],
                [2.0, 0.0, -z_sep],
                [3.0, 0.0, -z_sep],
            ]
        )
        n_nodes = 8
        ndof_per_node = 6
        ndof = n_nodes * ndof_per_node

        connectivity = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])

        k_spring = 1e4
        K_global = sp.diags(np.ones(ndof) * k_spring).tocsr()

        # 両端固定、z 方向のみ自由
        fixed_dofs = np.array(
            [i * ndof_per_node + d for i in range(n_nodes) for d in range(ndof_per_node) if d != 2],
            dtype=int,
        )

        f_ext = np.zeros(ndof)
        for i in range(4):
            f_ext[i * ndof_per_node + 2] = -3.0
        for i in range(4, 8):
            f_ext[i * ndof_per_node + 2] = 3.0

        config = ContactConfig(
            k_pen_scale=1e5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )
        manager = ContactManager(config=config)

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            lambda u: K_global.copy(),
            lambda u: K_global @ u,
            manager,
            node_coords_ref,
            connectivity,
            radii,
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
            use_mortar=True,
            line_contact=True,
            n_gauss=2,
        )
        assert result.converged


# ---- TestMortarVsPtP ----


class TestMortarVsPtP:
    """Mortar vs per-pair 比較テスト."""

    def test_mortar_vs_line_contact_same_direction(self):
        """Mortar と per-pair line contact で変位方向が一致."""
        sys = _make_spring_system()

        # Per-pair line contact
        config_lc = ContactConfig(k_pen_scale=1e5, line_contact=True, n_gauss=2)
        manager_lc = ContactManager(config=config_lc)
        result_lc = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            manager_lc,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
            line_contact=True,
            n_gauss=2,
        )

        # Mortar
        config_m = ContactConfig(k_pen_scale=1e5, line_contact=True, n_gauss=2, use_mortar=True)
        manager_m = ContactManager(config=config_m)
        result_m = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            manager_m,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
            use_mortar=True,
            line_contact=True,
            n_gauss=2,
        )

        assert result_lc.converged
        assert result_m.converged
        # z 変位の符号が一致（A梁は下方, B梁は上方に変位）
        ndof_per_node = sys["ndof_per_node"]
        for i in range(2):
            assert result_m.u[i * ndof_per_node + 2] * result_lc.u[i * ndof_per_node + 2] >= 0

    def test_mortar_requires_line_contact(self):
        """use_mortar=True でも line_contact=False なら per-pair にフォールバック."""
        sys = _make_spring_system()
        config = ContactConfig(k_pen_scale=1e5, use_mortar=True)
        manager = ContactManager(config=config)
        result = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            manager,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
            use_mortar=True,
            line_contact=False,
        )
        # フォールバックしても収束する
        assert result.converged
