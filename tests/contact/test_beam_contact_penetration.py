"""梁梁接触の貫入テスト.

2本の梁（針）の接触時の貫入量を検証する。

セットアップ:
  梁A（針1）: x軸方向 (0,0,0)→(1,0,0), 端部にx方向張力
  梁B（針2）: y軸方向 (0.5,-0.5,h)→(0.5,0.5,h),
              端部にy方向張力 + z方向押し下げ力

接触は梁の交差点付近 (s≈0.5, t≈0.5) で発生し、
ペナルティ法/Augmented Lagrangian により法線方向の貫入量が
制限されることを検証する。

テスト項目:
1. 接触検出: 梁同士が接触していること
2. 貫入量制限: ギャップが許容範囲内
3. 法線力正値性: 接触中の法線力が正
4. ペナルティ依存: 高い剛性で小さい貫入
5. 摩擦有無の影響: 法線方向の貫入に大差がない
6. 変位履歴: 荷重ステップごとの単調性
"""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactStatus,
)
from xkep_cae.contact.solver_hooks import (
    newton_raphson_with_contact,
)

# ====================================================================
# ヘルパー: 交差梁モデル（貫入テスト用）
# ====================================================================

_NDOF_PER_NODE = 6


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


def _fixed_dofs_penetration():
    """貫入テスト用の拘束DOF.

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
):
    """梁梁貫入問題を解く.

    Args:
        f_x: 梁A端部のx方向張力
        f_y: 梁B端部のy方向張力
        f_z: 梁B端部のz方向押し下げ力（正値→z負方向に作用）
        k_spring: 構造ばね剛性
        k_pen_scale: ペナルティ剛性スケール
        z_top: 梁Bの初期z座標
        radii: 断面半径
        n_load_steps: 荷重ステップ数
        max_iter: 最大NR反復数
        use_friction: 摩擦使用フラグ
        mu: 摩擦係数
        use_geometric_stiffness: 幾何剛性使用フラグ

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
            n_outer_max=5,
            use_friction=use_friction,
            mu_ramp_steps=3 if use_friction else 0,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=use_geometric_stiffness,
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


# ====================================================================
# テストクラス: 梁梁接触の貫入量検証
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


class TestPenetrationBound:
    """貫入量の制限検証."""

    def test_penetration_within_tolerance(self):
        """貫入量が断面半径の10%以下.

        ペナルティ法/ALでは完全に貫入ゼロにならないが、
        十分小さいことを検証。
        """
        radii = 0.04
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
                    assert penetration < radii * 0.1, (
                        f"過大な貫入: |gap|={penetration:.6e}, 許容値={radii * 0.1:.6e} (radii*10%)"
                    )
        assert active_found, "接触ペアが見つからなかった"

    def test_penetration_with_large_force(self):
        """大きな押し下げ力でも貫入量が制限される.

        f_z を大きくしても、ペナルティ剛性が十分なら
        貫入は許容範囲内に収まる。
        """
        radii = 0.04
        result, mgr, _, _ = _solve_penetration_problem(
            f_z=200.0,
            k_pen_scale=1e6,
            radii=radii,
        )
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                gap = pair.state.gap
                if gap < 0:
                    penetration = abs(gap)
                    assert penetration < radii * 0.15, (
                        f"大荷重で過大な貫入: |gap|={penetration:.6e}"
                    )

    def test_higher_penalty_reduces_penetration(self):
        """ペナルティ剛性が高いほど貫入が小さい."""
        penetrations = {}
        for k_pen in [1e4, 1e5, 1e6]:
            result, mgr, _, _ = _solve_penetration_problem(
                f_z=50.0,
                k_pen_scale=k_pen,
            )
            assert result.converged, f"k_pen={k_pen:.0e}で収束しなかった"

            for pair in mgr.pairs:
                if pair.state.status != ContactStatus.INACTIVE:
                    penetrations[k_pen] = abs(min(pair.state.gap, 0.0))
                    break

        # 少なくとも2つの k_pen 値で接触検出
        assert len(penetrations) >= 2, f"接触ペアが不足: {len(penetrations)} k_pen 値でのみ検出"

        # k_pen が大きいほど貫入が小さい
        k_values = sorted(penetrations.keys())
        for i in range(1, len(k_values)):
            pen_low = penetrations[k_values[i - 1]]
            pen_high = penetrations[k_values[i]]
            assert pen_high <= pen_low + 1e-10, (
                f"ペナルティ増加で貫入が増加: "
                f"k={k_values[i - 1]:.0e}→{pen_low:.6e}, "
                f"k={k_values[i]:.0e}→{pen_high:.6e}"
            )


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

        # 力が大きいほど法線力が大きい
        fz_sorted = sorted(pn_values.keys())
        for i in range(1, len(fz_sorted)):
            assert pn_values[fz_sorted[i]] >= pn_values[fz_sorted[i - 1]] - 1e-8, (
                f"法線力が非単調: f_z={fz_sorted[i - 1]:.0f}→"
                f"p_n={pn_values[fz_sorted[i - 1]]:.4f}, "
                f"f_z={fz_sorted[i]:.0f}→p_n={pn_values[fz_sorted[i]]:.4f}"
            )


class TestFrictionPenetrationEffect:
    """摩擦の有無による貫入量への影響."""

    def test_penetration_bounded_with_friction(self):
        """摩擦ありでも貫入量が許容範囲内."""
        radii = 0.04
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
                    assert penetration < radii * 0.1, f"摩擦あり過大貫入: |gap|={penetration:.6e}"

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

        # node3 の z 方向変位
        uz_history = [u_step[3 * _NDOF_PER_NODE + 2] for u_step in result.displacement_history]

        # 最終変位が負（梁Bが下方に移動した）
        assert uz_history[-1] < 0.0, f"最終z変位が非負: uz={uz_history[-1]:.6e}"

        # 前半（接触前）は概ね単調減少
        n_half = len(uz_history) // 2
        for i in range(1, n_half):
            assert uz_history[i] <= uz_history[i - 1] + 1e-8, (
                f"接触前のz変位が非単調: step {i - 1}→{i}: "
                f"{uz_history[i - 1]:.6e}→{uz_history[i]:.6e}"
            )

        # 全体で下方変位の絶対値が増加傾向
        # （最終ステップは初期ステップより大きな下方変位）
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
