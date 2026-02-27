"""梁梁接触の貫入テスト.

2本の梁の接触時の貫入量を検証する。
適応的ペナルティ増大（Adaptive Penalty Augmentation）により、
大半のケースで貫入量を search_radius の 1% 以下に制限する。

セットアップ（基本: 単一セグメント交差梁）:
  梁A: x軸方向 (0,0,0)→(1,0,0), 端部にx方向張力
  梁B: y軸方向 (0.5,-0.5,h)→(0.5,0.5,h),
              端部にy方向張力 + z方向押し下げ力

接触は梁の交差点付近 (s≈0.5, t≈0.5) で発生し、
ペナルティ法/Augmented Lagrangian + 適応的ペナルティにより
法線方向の貫入量が制限されることを検証する。

テスト項目:
1. 接触検出: 梁同士が接触していること
2. 貫入量制限: ギャップが search_radius の 1% 以下
3. 法線力正値性: 接触中の法線力が正
4. ペナルティ依存: 高い初期剛性で小さい貫入
5. 摩擦有無の影響: 法線方向の貫入に大差がない
6. 変位履歴: 荷重ステップごとの単調性
7. マルチセグメント梁: 複数要素分割での貫入制限
8. 横スライド接触: 接触維持しながら横方向にスライド
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_hooks import (
    newton_raphson_with_contact,
)

pytestmark = pytest.mark.slow

# ====================================================================
# ヘルパー: 交差梁モデル（貫入テスト用）
# ====================================================================

_NDOF_PER_NODE = 6

# 1% 貫入許容比（search_radius = r_a + r_b 基準）
_TOL_PEN_RATIO = 0.01


def _make_crossing_beams(
    k_spring: float = 1e4,
    z_top: float = 0.082,
    radii: float = 0.04,
):
    """交差梁ばねモデル（貫入テスト用）.

    梁A: node0-node1 (x方向, z=0)
    梁B: node2-node3 (y方向, z=z_top)
    交差点は (0.5, 0.0) 付近（s≈0.5, t≈0.5）。
    初期ギャップ: z_top - 2*radii

    Args:
        k_spring: 構造ばね剛性
        z_top: 梁Bの初期z座標
        radii: 断面半径
    """
    n_nodes = 4
    ndof_total = n_nodes * _NDOF_PER_NODE

    node_coords_ref = np.array(
        [
            [0.0, 0.0, 0.0],  # node 0: 梁A起点
            [1.0, 0.0, 0.0],  # node 1: 梁A終点
            [0.5, -0.5, z_top],  # node 2: 梁B起点
            [0.5, 0.5, z_top],  # node 3: 梁B終点
        ]
    )

    connectivity = np.array([[0, 1], [2, 3]])

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                K[d0, d0] += k_spring
                K[d0, d1] -= k_spring
                K[d1, d0] -= k_spring
                K[d1, d1] += k_spring
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
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


def _make_multi_segment_crossing_beams(
    n_seg_a: int = 4,
    n_seg_b: int = 4,
    k_spring: float = 1e4,
    z_top: float = 0.082,
    radii: float = 0.04,
):
    """マルチセグメント交差梁ばねモデル.

    梁A: x軸方向 (0,0,0)→(1,0,0), n_seg_a 分割
    梁B: y軸方向 (0.5,-0.5,z_top)→(0.5,0.5,z_top), n_seg_b 分割

    Args:
        n_seg_a: 梁Aの分割数
        n_seg_b: 梁Bの分割数
        k_spring: 構造ばね剛性
        z_top: 梁Bの初期z座標
        radii: 断面半径
    """
    n_nodes_a = n_seg_a + 1
    n_nodes_b = n_seg_b + 1
    n_nodes = n_nodes_a + n_nodes_b
    ndof_total = n_nodes * _NDOF_PER_NODE

    # 梁A: node 0..n_nodes_a-1
    coords_a = np.zeros((n_nodes_a, 3))
    for i in range(n_nodes_a):
        coords_a[i, 0] = i / n_seg_a  # x: 0→1

    # 梁B: node n_nodes_a..n_nodes-1
    coords_b = np.zeros((n_nodes_b, 3))
    for i in range(n_nodes_b):
        coords_b[i, 0] = 0.5
        coords_b[i, 1] = -0.5 + i / n_seg_b  # y: -0.5→0.5
        coords_b[i, 2] = z_top

    node_coords_ref = np.vstack([coords_a, coords_b])

    # 接続性
    conn_a = np.array([[i, i + 1] for i in range(n_seg_a)])
    conn_b = np.array([[n_nodes_a + i, n_nodes_a + i + 1] for i in range(n_seg_b)])
    connectivity = np.vstack([conn_a, conn_b])

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                K[d0, d0] += k_spring
                K[d0, d1] -= k_spring
                K[d1, d0] -= k_spring
                K[d1, d1] += k_spring
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * _NDOF_PER_NODE + d
                d1 = n1 * _NDOF_PER_NODE + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_spring * delta
                f_int[d1] += k_spring * delta
        return f_int

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        assemble_tangent,
        assemble_internal_force,
    )


def _fixed_dofs_penetration():
    """貫入テスト用の拘束DOF（単一セグメント版）.

    - Node 0: 全固定（梁Aの固定端）
    - Node 1: x方向のみ自由（梁Aの張力方向）
    - Node 2: 全固定（梁Bの固定端）
    - Node 3: y,z方向のみ自由（y張力 + z押し下げ）
    """
    fixed = []
    # Node 0: 全固定
    for d in range(_NDOF_PER_NODE):
        fixed.append(0 * _NDOF_PER_NODE + d)
    # Node 1: x のみ自由
    for d in range(_NDOF_PER_NODE):
        if d != 0:
            fixed.append(1 * _NDOF_PER_NODE + d)
    # Node 2: 全固定
    for d in range(_NDOF_PER_NODE):
        fixed.append(2 * _NDOF_PER_NODE + d)
    # Node 3: y, z のみ自由
    for d in range(_NDOF_PER_NODE):
        if d not in (1, 2):
            fixed.append(3 * _NDOF_PER_NODE + d)
    return np.array(fixed, dtype=int)


def _fixed_dofs_multi_segment(n_nodes_a, n_nodes_b):
    """マルチセグメント版の拘束DOF.

    ばねモデルのため回転DOFはすべて拘束する（回転自由度なし）。
    - 梁A node 0: 全固定
    - 梁A 内部 node: 回転のみ固定（並進は自由）
    - 梁A 最終 node: x のみ自由、他は固定
    - 梁B node 0: 全固定
    - 梁B 内部 node: 回転のみ固定（並進は自由）
    - 梁B 最終 node: y, z のみ自由、他は固定
    """
    fixed = set()
    n_nodes = n_nodes_a + n_nodes_b

    # 全ノードの回転DOFを拘束（ばねモデルに回転剛性なし）
    for i in range(n_nodes):
        for d in range(3, _NDOF_PER_NODE):
            fixed.add(i * _NDOF_PER_NODE + d)

    # 梁A node 0: 並進も全固定
    for d in range(3):
        fixed.add(0 * _NDOF_PER_NODE + d)
    # 梁A 最終node: y,z 固定（x のみ自由）
    node_a_last = n_nodes_a - 1
    for d in range(1, 3):
        fixed.add(node_a_last * _NDOF_PER_NODE + d)
    # 梁B node 0: 並進も全固定
    node_b_first = n_nodes_a
    for d in range(3):
        fixed.add(node_b_first * _NDOF_PER_NODE + d)
    # 梁B 最終node: x 固定（y, z のみ自由）
    node_b_last = n_nodes_a + n_nodes_b - 1
    fixed.add(node_b_last * _NDOF_PER_NODE + 0)

    return np.array(sorted(fixed), dtype=int)


def _solve_penetration_problem(
    f_x: float = 10.0,
    f_y: float = 5.0,
    f_z: float = 50.0,
    k_spring: float = 1e4,
    k_pen_scale: float = 1e5,
    z_top: float = 0.082,
    radii: float = 0.04,
    n_load_steps: int = 20,
    max_iter: int = 50,
    use_friction: bool = False,
    mu: float = 0.3,
    use_geometric_stiffness: bool = True,
    tol_penetration_ratio: float = _TOL_PEN_RATIO,
    n_outer_max: int = 8,
):
    """梁梁貫入問題を解く.

    Args:
        f_x: 梁A端部のx方向張力
        f_y: 梁B端部のy方向張力
        f_z: 梁B端部のz方向押し下げ力（正値→z負方向に作用）
        k_spring: 構造ばね剛性
        k_pen_scale: ペナルティ剛性スケール（初期値）
        z_top: 梁Bの初期z座標
        radii: 断面半径
        n_load_steps: 荷重ステップ数
        max_iter: 最大NR反復数
        use_friction: 摩擦使用フラグ
        mu: 摩擦係数
        use_geometric_stiffness: 幾何剛性使用フラグ
        tol_penetration_ratio: 貫入許容比
        n_outer_max: Outer loop 最大反復数

    Returns:
        result: ContactSolveResult
        mgr: ContactManager（最終状態）
        ndof_total: 全体DOF数
        node_coords_ref: 参照座標
    """
    (
        node_coords_ref,
        connectivity,
        radii_val,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    ) = _make_crossing_beams(k_spring=k_spring, z_top=z_top, radii=radii)

    f_ext = np.zeros(ndof_total)
    f_ext[1 * _NDOF_PER_NODE + 0] = f_x  # node1 x方向張力
    f_ext[3 * _NDOF_PER_NODE + 1] = f_y  # node3 y方向張力
    f_ext[3 * _NDOF_PER_NODE + 2] = -f_z  # node3 z方向↓

    fixed_dofs = _fixed_dofs_penetration()

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=k_pen_scale,
            k_t_ratio=0.1,
            mu=mu,
            g_on=0.0,
            g_off=1e-4,
            n_outer_max=n_outer_max,
            use_friction=use_friction,
            mu_ramp_steps=3 if use_friction else 0,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=use_geometric_stiffness,
            tol_penetration_ratio=tol_penetration_ratio,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
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
        radii_val,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        show_progress=False,
        broadphase_margin=0.05,
    )

    return result, mgr, ndof_total, node_coords_ref


def _solve_multi_segment_problem(
    n_seg_a: int = 4,
    n_seg_b: int = 4,
    f_x: float = 10.0,
    f_y: float = 5.0,
    f_z: float = 50.0,
    k_spring: float = 1e4,
    k_pen_scale: float = 1e5,
    z_top: float = 0.082,
    radii: float = 0.04,
    n_load_steps: int = 20,
    max_iter: int = 50,
    tol_penetration_ratio: float = _TOL_PEN_RATIO,
):
    """マルチセグメント梁の貫入問題を解く."""
    (
        node_coords_ref,
        connectivity,
        radii_val,
        ndof_total,
        n_nodes_a,
        n_nodes_b,
        assemble_tangent,
        assemble_internal_force,
    ) = _make_multi_segment_crossing_beams(
        n_seg_a=n_seg_a,
        n_seg_b=n_seg_b,
        k_spring=k_spring,
        z_top=z_top,
        radii=radii,
    )

    f_ext = np.zeros(ndof_total)
    node_a_last = n_nodes_a - 1
    node_b_last = n_nodes_a + n_nodes_b - 1
    f_ext[node_a_last * _NDOF_PER_NODE + 0] = f_x  # 梁A端部 x方向張力
    f_ext[node_b_last * _NDOF_PER_NODE + 1] = f_y  # 梁B端部 y方向張力
    f_ext[node_b_last * _NDOF_PER_NODE + 2] = -f_z  # 梁B端部 z方向↓

    fixed_dofs = _fixed_dofs_multi_segment(n_nodes_a, n_nodes_b)

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=k_pen_scale,
            k_t_ratio=0.1,
            g_on=0.0,
            g_off=1e-4,
            n_outer_max=8,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=tol_penetration_ratio,
            penalty_growth_factor=2.0,
            k_pen_max=1e12,
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
        radii_val,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        show_progress=False,
        broadphase_margin=0.05,
    )

    return result, mgr, radii_val


def _max_penetration_ratio(mgr):
    """全ペアの最大貫入比を返す."""
    max_ratio = 0.0
    for pair in mgr.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        if pair.state.gap >= 0.0:
            continue
        sr = pair.search_radius
        if sr < 1e-30:
            continue
        ratio = abs(pair.state.gap) / sr
        max_ratio = max(max_ratio, ratio)
    return max_ratio


# ====================================================================
# テストクラス: 梁梁接触の検出検証
# ====================================================================


class TestBeamContactDetection:
    """梁梁接触の検出検証."""

    def test_contact_detected_with_push_down(self):
        """z方向押し下げ力で接触が検出される."""
        result, mgr, _, _ = _solve_penetration_problem(f_z=50.0)
        assert result.converged, "ソルバーが収束しなかった"
        assert mgr.n_active > 0, "接触が検出されなかった"

    def test_no_contact_without_push_down(self):
        """押し下げ力がなければ接触は検出されない.

        初期ギャップ = 0.082 - 0.08 = 0.002 > 0 なので、
        z方向力がなければ接触しない。
        """
        result, mgr, _, _ = _solve_penetration_problem(f_z=0.0)
        assert result.converged
        assert mgr.n_active == 0, "力なしで接触が検出された"


# ====================================================================
# テストクラス: 貫入量制限（1% 目標）
# ====================================================================


class TestPenetrationBound:
    """貫入量の制限検証（適応的ペナルティ増大による 1% 目標）."""

    def test_penetration_within_1_percent(self):
        """貫入量が search_radius の 1% 以下.

        適応的ペナルティ増大により、k_pen が自動調整され
        貫入量が search_radius の 1% 以下に制限される。
        """
        radii = 0.04
        search_radius = 2 * radii  # r_a + r_b = 0.08
        result, mgr, _, _ = _solve_penetration_problem(
            f_z=50.0,
            radii=radii,
            k_pen_scale=1e5,
        )
        assert result.converged

        active_found = False
        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                active_found = True
                gap = pair.state.gap
                if gap < 0:
                    penetration = abs(gap)
                    tol = search_radius * _TOL_PEN_RATIO
                    assert penetration < tol, (
                        f"貫入超過: |gap|={penetration:.6e}, "
                        f"許容値={tol:.6e} (search_radius*{_TOL_PEN_RATIO})"
                    )
        assert active_found, "接触ペアが見つからなかった"

    def test_penetration_with_large_force(self):
        """大きな押し下げ力でも貫入量が 1% 以下に制限される.

        f_z を大きくしても、適応的ペナルティ増大により
        貫入は許容範囲内に収まる。
        """
        radii = 0.04
        search_radius = 2 * radii
        result, mgr, _, _ = _solve_penetration_problem(
            f_z=200.0,
            k_pen_scale=1e5,
            radii=radii,
        )
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                gap = pair.state.gap
                if gap < 0:
                    penetration = abs(gap)
                    tol = search_radius * _TOL_PEN_RATIO
                    assert penetration < tol, (
                        f"大荷重で貫入超過: |gap|={penetration:.6e}, 許容値={tol:.6e}"
                    )

    def test_higher_penalty_reduces_penetration(self):
        """ペナルティ剛性が高いほど貫入が小さい.

        適応的ペナルティ無効 (tol=0) で初期 k_pen のみに依存するケース。
        """
        penetrations = {}
        for k_pen in [1e4, 1e5, 1e6]:
            result, mgr, _, _ = _solve_penetration_problem(
                f_z=50.0,
                k_pen_scale=k_pen,
                tol_penetration_ratio=0.0,  # 適応無効
            )
            assert result.converged, f"k_pen={k_pen:.0e}で収束しなかった"

            for pair in mgr.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    penetrations[k_pen] = abs(min(pair.state.gap, 0.0))
                    break

        assert len(penetrations) >= 2, f"接触ペアが不足: {len(penetrations)} k_pen 値でのみ検出"

        k_values = sorted(penetrations.keys())
        for i in range(1, len(k_values)):
            pen_low = penetrations[k_values[i - 1]]
            pen_high = penetrations[k_values[i]]
            assert pen_high <= pen_low + 1e-10, (
                f"ペナルティ増加で貫入が増加: "
                f"k={k_values[i - 1]:.0e}→{pen_low:.6e}, "
                f"k={k_values[i]:.0e}→{pen_high:.6e}"
            )

    def test_adaptive_penalty_improves_penetration(self):
        """適応的ペナルティが貫入量を改善する.

        低い初期 k_pen でも適応的ペナルティにより 1% を達成。
        """
        radii = 0.04

        # 適応なし（低 k_pen → 貫入大）
        _, mgr_no_adapt, _, _ = _solve_penetration_problem(
            f_z=50.0,
            k_pen_scale=1e4,
            radii=radii,
            tol_penetration_ratio=0.0,
        )
        pen_no_adapt = _max_penetration_ratio(mgr_no_adapt)

        # 適応あり（低 k_pen → 自動増大 → 貫入小）
        _, mgr_adapt, _, _ = _solve_penetration_problem(
            f_z=50.0,
            k_pen_scale=1e4,
            radii=radii,
            tol_penetration_ratio=_TOL_PEN_RATIO,
        )
        pen_adapt = _max_penetration_ratio(mgr_adapt)

        # 適応的ペナルティが貫入を改善
        assert pen_adapt < pen_no_adapt + 1e-10, (
            f"適応的ペナルティが改善しなかった: no_adapt={pen_no_adapt:.6e}, adapt={pen_adapt:.6e}"
        )

        # 適応後は 1% 以下
        assert pen_adapt < _TOL_PEN_RATIO, (
            f"適応後も 1% 超過: pen_ratio={pen_adapt:.6e}, tol={_TOL_PEN_RATIO}"
        )


# ====================================================================
# テストクラス: 法線力の検証
# ====================================================================


class TestNormalForce:
    """接触法線力の検証."""

    def test_normal_force_positive(self):
        """接触中の法線力が正値（圧縮のみ）."""
        result, mgr, _, _ = _solve_penetration_problem(f_z=50.0)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                assert pair.state.p_n >= 0.0, f"引張接触力: p_n={pair.state.p_n:.6f}"

    def test_normal_force_increases_with_push(self):
        """押し下げ力が大きいほど法線力が大きい."""
        pn_values = {}
        for f_z in [30.0, 50.0, 100.0]:
            result, mgr, _, _ = _solve_penetration_problem(f_z=f_z)
            assert result.converged

            for pair in mgr.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    pn_values[f_z] = pair.state.p_n
                    break

        assert len(pn_values) >= 2, "接触検出が不十分"

        fz_sorted = sorted(pn_values.keys())
        for i in range(1, len(fz_sorted)):
            assert pn_values[fz_sorted[i]] >= pn_values[fz_sorted[i - 1]] - 1e-8, (
                f"法線力が非単調: f_z={fz_sorted[i - 1]:.0f}→"
                f"p_n={pn_values[fz_sorted[i - 1]]:.4f}, "
                f"f_z={fz_sorted[i]:.0f}→p_n={pn_values[fz_sorted[i]]:.4f}"
            )


# ====================================================================
# テストクラス: 摩擦の貫入量への影響
# ====================================================================


class TestFrictionPenetrationEffect:
    """摩擦の有無による貫入量への影響."""

    def test_penetration_bounded_with_friction(self):
        """摩擦ありでも貫入量が 1% 以下."""
        radii = 0.04
        search_radius = 2 * radii
        result, mgr, _, _ = _solve_penetration_problem(
            f_z=50.0,
            f_y=5.0,
            use_friction=True,
            mu=0.3,
            radii=radii,
            k_pen_scale=1e5,
        )
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                gap = pair.state.gap
                if gap < 0:
                    penetration = abs(gap)
                    tol = search_radius * _TOL_PEN_RATIO
                    assert penetration < tol, (
                        f"摩擦あり貫入超過: |gap|={penetration:.6e}, 許容値={tol:.6e}"
                    )

    def test_friction_does_not_worsen_penetration(self):
        """摩擦の有無で法線方向の貫入に大差がない."""
        result_nf, mgr_nf, _, _ = _solve_penetration_problem(
            f_z=50.0,
            use_friction=False,
        )
        result_f, mgr_f, _, _ = _solve_penetration_problem(
            f_z=50.0,
            use_friction=True,
            mu=0.3,
        )
        assert result_nf.converged
        assert result_f.converged

        gap_nf = None
        gap_f = None
        for pair in mgr_nf.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                gap_nf = pair.state.gap
                break
        for pair in mgr_f.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                gap_f = pair.state.gap
                break

        if gap_nf is not None and gap_f is not None:
            assert abs(gap_nf - gap_f) < 1e-3, (
                f"摩擦による貫入変化が過大: "
                f"gap_nofric={gap_nf:.6e}, gap_fric={gap_f:.6e}, "
                f"Δgap={abs(gap_nf - gap_f):.6e}"
            )


# ====================================================================
# テストクラス: 変位履歴
# ====================================================================


class TestDisplacementHistory:
    """荷重ステップごとの変位履歴検証."""

    def test_z_displacement_progresses_downward(self):
        """梁B端部のz方向変位が荷重ステップを通じて下方に進行する.

        接触確立後にAL乗数更新による微小な押し戻しが発生しうるため、
        厳密な単調性ではなく、全体的な下方進行を検証する。
        """
        result, _, _, _ = _solve_penetration_problem(
            f_z=50.0,
            n_load_steps=10,
        )
        assert result.converged

        uz_history = [u_step[3 * _NDOF_PER_NODE + 2] for u_step in result.displacement_history]

        assert uz_history[-1] < 0.0, f"最終z変位が非負: uz={uz_history[-1]:.6e}"

        # 前半（接触前）は概ね単調減少
        n_half = len(uz_history) // 2
        for i in range(1, n_half):
            assert uz_history[i] <= uz_history[i - 1] + 1e-8, (
                f"接触前のz変位が非単調: step {i - 1}→{i}: "
                f"{uz_history[i - 1]:.6e}→{uz_history[i]:.6e}"
            )

        assert abs(uz_history[-1]) > abs(uz_history[0]), (
            f"最終変位が初期変位より小さい: "
            f"|uz_final|={abs(uz_history[-1]):.6e}, "
            f"|uz_init|={abs(uz_history[0]):.6e}"
        )

    def test_x_tension_positive(self):
        """梁A端部のx方向変位が正（張力方向）."""
        result, _, _, _ = _solve_penetration_problem(f_x=10.0, f_z=50.0)
        assert result.converged

        ux_node1 = result.u[1 * _NDOF_PER_NODE + 0]
        assert ux_node1 > 0.0, f"x変位が非正: ux_node1={ux_node1:.6e}"


# ====================================================================
# テストクラス: マルチセグメント梁の貫入テスト
# ====================================================================


class TestMultiSegmentBeamPenetration:
    """マルチセグメント梁（複数要素分割）での貫入テスト."""

    def test_multi_segment_contact_detected(self):
        """マルチセグメント梁で接触が検出される."""
        result, mgr, _ = _solve_multi_segment_problem(
            n_seg_a=4,
            n_seg_b=4,
            f_z=50.0,
        )
        assert result.converged, "マルチセグメントソルバーが収束しなかった"
        assert mgr.n_active > 0, "マルチセグメント梁で接触が検出されなかった"

    def test_multi_segment_penetration_within_1_percent(self):
        """マルチセグメント梁でも貫入量が 1% 以下."""
        radii = 0.04
        result, mgr, _ = _solve_multi_segment_problem(
            n_seg_a=4,
            n_seg_b=4,
            f_z=50.0,
            radii=radii,
            k_pen_scale=1e5,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, (
            f"マルチセグメント貫入超過: pen_ratio={pen_ratio:.6e}, tol={_TOL_PEN_RATIO}"
        )

    def test_multi_segment_large_force_penetration(self):
        """マルチセグメント梁で大荷重でも貫入 1% 以下.

        マルチセグメント梁は直列ばねで構造剛性が低いため、
        荷重を構造剛性に対して適切な範囲に設定する。
        """
        radii = 0.04
        result, mgr, _ = _solve_multi_segment_problem(
            n_seg_a=4,
            n_seg_b=4,
            f_z=100.0,
            radii=radii,
            k_spring=1e4,
            k_pen_scale=1e5,
            n_load_steps=30,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, (
            f"マルチセグメント大荷重貫入超過: pen_ratio={pen_ratio:.6e}"
        )


# ====================================================================
# テストクラス: 横スライド接触テスト
# ====================================================================


class TestSlidingContact:
    """直交する梁の接触を維持しながら横方向にスライドするテスト.

    セットアップ:
      梁A: x軸方向 (0,0,0)→(1,0,0), 固定
      梁B: y軸方向 (0.5,-0.5,h)→(0.5,0.5,h),
           z方向押し下げ + x方向横スライド力

    梁Bが梁Aの上を z方向に接触しながら x方向にスライドする。
    スライド中も貫入量が 1% 以下に維持されることを検証。
    """

    def _solve_sliding_problem(
        self,
        f_z: float = 50.0,
        f_slide_x: float = 20.0,
        use_friction: bool = True,
        mu: float = 0.3,
        radii: float = 0.04,
        k_pen_scale: float = 1e5,
        n_load_steps: int = 20,
    ):
        """横スライド接触問題を解く.

        梁Bの自由端に z方向押し下げ + x方向スライド力を付与。
        梁Bの固定端は全固定のまま、自由端は x,y,z 自由。

        Args:
            f_z: 押し下げ力
            f_slide_x: x方向スライド力
            use_friction: 摩擦使用フラグ
            mu: 摩擦係数
            radii: 断面半径
            k_pen_scale: ペナルティ剛性
            n_load_steps: 荷重ステップ数
        """
        k_spring = 1e4
        z_top = 0.082

        (
            node_coords_ref,
            connectivity,
            radii_val,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_crossing_beams(k_spring=k_spring, z_top=z_top, radii=radii)

        f_ext = np.zeros(ndof_total)
        # 梁A: node0 全固定, node1 全固定（梁Aは完全に固定）
        # 梁B: node2 全固定, node3: x,y,z 自由
        f_ext[3 * _NDOF_PER_NODE + 0] = f_slide_x  # node3 x方向スライド
        f_ext[3 * _NDOF_PER_NODE + 1] = 5.0  # node3 y方向張力
        f_ext[3 * _NDOF_PER_NODE + 2] = -f_z  # node3 z方向↓

        # 梁A: 両端全固定, 梁B: node2全固定, node3: x,y,z自由
        fixed = []
        for d in range(_NDOF_PER_NODE):
            fixed.append(0 * _NDOF_PER_NODE + d)
        for d in range(_NDOF_PER_NODE):
            fixed.append(1 * _NDOF_PER_NODE + d)
        for d in range(_NDOF_PER_NODE):
            fixed.append(2 * _NDOF_PER_NODE + d)
        # node3: 並進自由(0,1,2), 回転固定(3,4,5)
        for d in range(3, _NDOF_PER_NODE):
            fixed.append(3 * _NDOF_PER_NODE + d)
        fixed_dofs = np.array(fixed, dtype=int)

        mgr = ContactManager(
            config=ContactConfig(
                k_pen_scale=k_pen_scale,
                k_t_ratio=0.1,
                mu=mu,
                g_on=0.0,
                g_off=1e-4,
                n_outer_max=8,
                use_friction=use_friction,
                mu_ramp_steps=3 if use_friction else 0,
                use_line_search=True,
                line_search_max_steps=5,
                use_geometric_stiffness=True,
                tol_penetration_ratio=_TOL_PEN_RATIO,
                penalty_growth_factor=2.0,
                k_pen_max=1e12,
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
            radii_val,
            n_load_steps=n_load_steps,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
        )

        return result, mgr, ndof_total

    def test_sliding_contact_detected(self):
        """横スライド時に接触が検出される."""
        result, mgr, _ = self._solve_sliding_problem()
        assert result.converged, "スライドソルバーが収束しなかった"
        assert mgr.n_active > 0, "スライド時に接触が検出されなかった"

    def test_sliding_penetration_within_1_percent(self):
        """横スライド中も貫入量が 1% 以下."""
        result, mgr, _ = self._solve_sliding_problem()
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, (
            f"スライド時貫入超過: pen_ratio={pen_ratio:.6e}, tol={_TOL_PEN_RATIO}"
        )

    def test_sliding_with_friction_penetration(self):
        """摩擦ありスライドでも貫入 1% 以下."""
        result, mgr, _ = self._solve_sliding_problem(
            use_friction=True,
            mu=0.5,
            f_slide_x=30.0,
        )
        assert result.converged

        pen_ratio = _max_penetration_ratio(mgr)
        assert pen_ratio < _TOL_PEN_RATIO, f"摩擦スライド貫入超過: pen_ratio={pen_ratio:.6e}"

    def test_sliding_displacement_has_x_component(self):
        """スライド時にx方向変位が生じる."""
        result, _, ndof_total = self._solve_sliding_problem(
            f_slide_x=20.0,
        )
        assert result.converged

        ux_node3 = result.u[3 * _NDOF_PER_NODE + 0]
        assert ux_node3 > 0.0, f"スライド方向のx変位が非正: ux={ux_node3:.6e}"

    def test_sliding_friction_both_converge(self):
        """摩擦なし/ありの両方でスライド解が収束する."""
        result_nf, _, _ = self._solve_sliding_problem(
            use_friction=False,
            f_slide_x=20.0,
        )
        result_f, _, _ = self._solve_sliding_problem(
            use_friction=True,
            mu=0.3,
            f_slide_x=20.0,
        )
        assert result_nf.converged, "摩擦なしスライドが収束しなかった"
        assert result_f.converged, "摩擦ありスライドが収束しなかった"

        ux_nf = result_nf.u[3 * _NDOF_PER_NODE + 0]
        ux_f = result_f.u[3 * _NDOF_PER_NODE + 0]
        # 両方ともx方向に正の変位が生じる
        assert ux_nf > 0.0, f"摩擦なしx変位が非正: {ux_nf:.6e}"
        assert ux_f > 0.0, f"摩擦ありx変位が非正: {ux_f:.6e}"
