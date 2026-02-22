"""摩擦接触のバリデーションテスト.

Phase C5 TODO: 摩擦ありの解析解バリデーション。

Abaqus環境が利用不可のため、解析解・力のバランス・Coulomb条件との
整合性を検証する物理バリデーションテストを構成する。

テストケース:
1. stick条件: 接線荷重 < μ·P_n → 摩擦力 = 接線荷重（弾性応答）
2. slip条件: 接線荷重 > μ·P_n → 摩擦力 = μ·P_n（Coulomb限界）
3. 力のバランス: 平衡状態での残差がゼロ
4. 摩擦角の検証: 接触力ベクトルがCoulomb円錐内に収まる
5. stick-slip遷移: 荷重増加に伴うstick→slip遷移の検証
6. エネルギー散逸: slip時の散逸 > 0, stick時の散逸 ≈ 0
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
# ヘルパー: 3D ばねモデル（摩擦バリデーション用）
# ====================================================================


def _make_spring_system_validation(
    k_spring: float = 1e4,
    ndof_per_node: int = 6,
    z_sep: float = 0.041,
    radii: float = 0.04,
):
    """バリデーション用2交差梁ばねモデル.

    梁A: node0-node1 (x方向, z=+z_sep)
    梁B: node2-node3 (y方向, z=-z_sep)
    交差点は中点 (s≈0.5, t≈0.5)。

    全方向にばね剛性あり。
    """
    n_nodes = 4
    ndof_total = n_nodes * ndof_per_node

    node_coords_ref = np.array(
        [
            [0.0, 0.0, z_sep],
            [1.0, 0.0, z_sep],
            [0.5, -0.5, -z_sep],
            [0.5, 0.5, -z_sep],
        ]
    )

    connectivity = np.array([[0, 1], [2, 3]])

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
        return f_int

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    )


def _fixed_dofs_validation(n_nodes, ndof_per_node=6, free_nodes=None, free_dirs=None):
    """指定ノード/方向以外を全固定する."""
    if free_nodes is None:
        free_nodes = []
    if free_dirs is None:
        free_dirs = [0, 1, 2]
    fixed = []
    for node in range(n_nodes):
        for d in range(ndof_per_node):
            if node in free_nodes and d in free_dirs:
                continue
            fixed.append(node * ndof_per_node + d)
    return np.array(fixed, dtype=int)


def _solve_friction_problem(
    f_z: float = 50.0,
    f_t_x: float = 0.0,
    f_t_y: float = 0.0,
    mu: float = 0.3,
    k_spring: float = 1e4,
    n_load_steps: int = 20,
    max_iter: int = 50,
    mu_ramp_steps: int = 3,
    use_line_search: bool = True,
    use_geometric_stiffness: bool = True,
):
    """標準的な摩擦接触問題を解くヘルパー.

    Args:
        f_z: 法線方向の押し付け力（正値、node1→z↓, node3→z↑）
        f_t_x: node1に加えるx方向接線荷重
        f_t_y: node1に加えるy方向接線荷重
        mu: 摩擦係数
        k_spring: ばね剛性
        n_load_steps: 荷重ステップ数
        max_iter: 最大NR反復
        mu_ramp_steps: μランプステップ数
        use_line_search: line search有効化
        use_geometric_stiffness: 幾何剛性有効化

    Returns:
        result: ContactSolveResult
        mgr: ContactManager（最終状態）
        ndof_total: 全体DOF数
        node_coords_ref: 参照座標
    """
    (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        assemble_tangent,
        assemble_internal_force,
    ) = _make_spring_system_validation(k_spring=k_spring)

    f_ext = np.zeros(ndof_total)
    f_ext[1 * 6 + 2] = -f_z  # node1 z↓
    f_ext[3 * 6 + 2] = f_z  # node3 z↑
    f_ext[1 * 6 + 0] = f_t_x  # node1 x方向接線荷重
    f_ext[1 * 6 + 1] = f_t_y  # node1 y方向接線荷重

    fixed_dofs = _fixed_dofs_validation(4, free_nodes=[1, 3], free_dirs=[0, 1, 2])

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=1e5,
            k_t_ratio=0.1,
            mu=mu,
            g_on=0.0,
            g_off=1e-4,
            n_outer_max=5,
            use_friction=True,
            mu_ramp_steps=mu_ramp_steps,
            use_line_search=use_line_search,
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
        radii,
        n_load_steps=n_load_steps,
        max_iter=max_iter,
        show_progress=False,
        broadphase_margin=0.05,
    )

    return result, mgr, ndof_total, node_coords_ref


# ====================================================================
# テストクラス: 摩擦接触バリデーション
# ====================================================================


class TestFrictionCoulombCondition:
    """Coulomb条件の物理バリデーション."""

    def test_coulomb_limit_satisfied(self):
        """全ペアで Coulomb 条件 ||q_t|| <= μ·p_n が成立する.

        法線押し付け + 大きな接線荷重（slip 誘発）で、
        摩擦力が Coulomb 限界を超えないことを検証。
        """
        result, mgr, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=20.0, mu=0.3)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            p_n = pair.state.p_n
            q_norm = float(np.linalg.norm(pair.state.z_t))
            mu = 0.3
            # Coulomb 条件: ||q|| <= μ·p_n + 許容誤差
            assert q_norm <= mu * p_n + 1e-8, (
                f"Coulomb violation: ||q||={q_norm:.6f} > μ·p_n={mu * p_n:.6f}"
            )

    def test_slip_friction_equals_mu_pn(self):
        """slip 状態で摩擦力 = μ·p_n（Coulomb 限界）.

        大きな接線荷重で slip を誘発し、摩擦力の大きさが
        μ·p_n に一致することを検証。
        接線荷重は接触摩擦限界を十分に超える大きさが必要。
        """
        result, mgr, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=100.0, mu=0.3)
        assert result.converged

        sliding_found = False
        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.SLIDING:
                sliding_found = True
                p_n = pair.state.p_n
                q_norm = float(np.linalg.norm(pair.state.z_t))
                expected = 0.3 * p_n
                # slip 時: ||q|| ≈ μ·p_n（許容誤差 1%）
                assert abs(q_norm - expected) / expected < 0.01, (
                    f"Slip force mismatch: ||q||={q_norm:.6f}, μ·p_n={expected:.6f}"
                )
        assert sliding_found, "No sliding pair found (expected slip)"

    def test_stick_condition_small_tangential_load(self):
        """小さい接線荷重で stick 状態が保持される.

        接線荷重が Coulomb 限界未満の場合、全ペアが stick。
        """
        result, mgr, _, _ = _solve_friction_problem(
            f_z=100.0,
            f_t_x=1.0,  # 小さな接線荷重
            mu=0.3,
        )
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            assert pair.state.stick, (
                "Expected stick but got slip "
                f"(p_n={pair.state.p_n:.4f}, "
                f"||q||={np.linalg.norm(pair.state.z_t):.4f})"
            )

    def test_friction_cone_two_axes(self):
        """2軸接線荷重で Coulomb 円錐内に収まる.

        x,y 両方向の接線荷重を加えても、
        摩擦力ベクトルのノルムが μ·p_n 以下。
        """
        result, mgr, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=15.0, f_t_y=15.0, mu=0.3)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            p_n = pair.state.p_n
            q_norm = float(np.linalg.norm(pair.state.z_t))
            assert q_norm <= 0.3 * p_n + 1e-8, f"Coulomb cone violation: ||q||={q_norm:.6f}"


class TestFrictionForceBalance:
    """平衡状態での力のバランス検証."""

    def test_residual_zero_at_equilibrium(self):
        """収束解での残差がゼロ（力のバランス成立）.

        平衡: f_ext = f_int(u) + f_contact(u)
        ここで f_contact = 法線力 + 摩擦力。
        """
        from xkep_cae.contact.assembly import compute_contact_force
        from xkep_cae.contact.law_normal import evaluate_normal_force

        k_spring = 1e4
        f_z = 50.0
        f_t_x = 10.0
        mu = 0.3

        (
            node_coords_ref,
            connectivity,
            radii,
            ndof_total,
            assemble_tangent,
            assemble_internal_force,
        ) = _make_spring_system_validation(k_spring=k_spring)

        result, mgr, _, _ = _solve_friction_problem(f_z=f_z, f_t_x=f_t_x, mu=mu, k_spring=k_spring)
        assert result.converged

        u = result.u

        # 外力
        f_ext = np.zeros(ndof_total)
        f_ext[1 * 6 + 2] = -f_z
        f_ext[3 * 6 + 2] = f_z
        f_ext[1 * 6 + 0] = f_t_x

        # 内力
        f_int = assemble_internal_force(u)

        # 接触力（摩擦込み）
        # 各ペアの摩擦力を収集
        friction_forces = {}
        for i, pair in enumerate(mgr.pairs):
            if pair.state.status != ContactStatus.INACTIVE:
                evaluate_normal_force(pair)
                friction_forces[i] = pair.state.z_t.copy()

        f_contact = compute_contact_force(mgr, ndof_total, friction_forces=friction_forces)

        # 残差 = 外力 - 内力 - 接触力
        residual = f_ext - f_int - f_contact

        # 自由DOFのみの残差を検証
        fixed_dofs = _fixed_dofs_validation(4, free_nodes=[1, 3], free_dirs=[0, 1, 2])
        free_dofs = np.setdiff1d(np.arange(ndof_total), fixed_dofs)
        residual_free = residual[free_dofs]

        # 残差が小さい（力のバランス成立）
        res_norm = float(np.linalg.norm(residual_free))
        f_norm = float(np.linalg.norm(f_ext[free_dofs]))
        relative_residual = res_norm / max(f_norm, 1e-10)

        assert relative_residual < 0.05, (
            f"Force balance violated: ||R||/||F|| = {relative_residual:.6e}"
        )

    def test_normal_force_positive_at_contact(self):
        """接触中の法線力が正値（引張なし）."""
        result, mgr, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=5.0, mu=0.3)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                assert pair.state.p_n >= 0.0, f"Tensile contact force: p_n={pair.state.p_n:.6f}"


class TestStickSlipTransition:
    """stick → slip 遷移のバリデーション."""

    def test_increasing_tangential_load_causes_slip(self):
        """接線荷重を段階的に増加させると、stick→slip 遷移が発生する.

        小荷重→stick, 大荷重→slip を確認。
        """
        # ケース1: 小さい接線荷重 → stick
        result_small, mgr_small, _, _ = _solve_friction_problem(f_z=100.0, f_t_x=1.0, mu=0.3)
        assert result_small.converged

        # ケース2: 大きい接線荷重 → slip
        result_large, mgr_large, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=100.0, mu=0.3)
        assert result_large.converged

        # 小荷重: 全 active ペアが stick
        for pair in mgr_small.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                assert pair.state.stick, "Expected stick with small load"

        # 大荷重: 少なくとも1ペアが sliding
        has_sliding = any(pair.state.status == ContactStatus.SLIDING for pair in mgr_large.pairs)
        assert has_sliding, "Expected sliding with large tangential load"

    def test_slip_displacement_larger_than_stick(self):
        """slip 時の接線変位は stick 時より大きい.

        接線荷重増加に伴い、接線変位が増加することを検証。
        """
        result_stick, _, ndof, _ = _solve_friction_problem(f_z=100.0, f_t_x=1.0, mu=0.3)
        result_slip, _, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=100.0, mu=0.3)
        assert result_stick.converged
        assert result_slip.converged

        # node1 の x 方向変位
        ux_stick = abs(result_stick.u[1 * 6 + 0])
        ux_slip = abs(result_slip.u[1 * 6 + 0])

        assert ux_slip > ux_stick, (
            f"Expected slip displacement > stick: ux_slip={ux_slip:.6e}, ux_stick={ux_stick:.6e}"
        )


class TestFrictionEnergyDissipation:
    """摩擦エネルギー散逸のバリデーション."""

    def test_slip_has_positive_dissipation(self):
        """slip 状態で散逸が正（エネルギー散逸あり）.

        大きな接線荷重で slip を誘発し、散逸 > 0 を検証。
        """
        result, mgr, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=100.0, mu=0.3)
        assert result.converged

        # 少なくとも1ペアで正の散逸
        has_positive_dissipation = False
        for pair in mgr.pairs:
            if pair.state.status == ContactStatus.SLIDING:
                if pair.state.dissipation > 0.0:
                    has_positive_dissipation = True
        assert has_positive_dissipation, "Expected positive dissipation in slip"

    def test_dissipation_nonnegative(self):
        """全ペアで散逸が非負（熱力学的整合性）."""
        result, mgr, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=20.0, mu=0.3)
        assert result.converged

        for pair in mgr.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                assert pair.state.dissipation >= -1e-12, (
                    f"Negative dissipation: {pair.state.dissipation:.6e}"
                )


class TestFrictionSymmetry:
    """摩擦接触の対称性バリデーション."""

    def test_opposite_tangential_load_gives_opposite_displacement(self):
        """反対方向の接線荷重で反対方向の変位.

        +x と -x の接線荷重で、変位が符号反転（大きさ同等）。
        """
        result_pos, _, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=10.0, mu=0.3)
        result_neg, _, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=-10.0, mu=0.3)
        assert result_pos.converged
        assert result_neg.converged

        # node1 x方向変位
        ux_pos = result_pos.u[1 * 6 + 0]
        ux_neg = result_neg.u[1 * 6 + 0]

        # 符号が反対で大きさが近い
        assert ux_pos * ux_neg < 0 or (abs(ux_pos) < 1e-10 and abs(ux_neg) < 1e-10), (
            f"Expected opposite signs: ux_pos={ux_pos:.6e}, ux_neg={ux_neg:.6e}"
        )
        if abs(ux_pos) > 1e-10:
            assert abs(abs(ux_pos) - abs(ux_neg)) / abs(ux_pos) < 0.1, (
                f"Asymmetry: |ux_pos|={abs(ux_pos):.6e}, |ux_neg|={abs(ux_neg):.6e}"
            )

    def test_no_tangential_load_gives_zero_tangential_displacement(self):
        """接線荷重なしで接線変位がゼロ.

        法線方向のみの押し付けで、接線方向の変位が発生しない。
        """
        result, _, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=0.0, f_t_y=0.0, mu=0.3)
        assert result.converged

        # node1 の接線方向変位
        ux_node1 = abs(result.u[1 * 6 + 0])
        uy_node1 = abs(result.u[1 * 6 + 1])

        # 数値的な微小残差（接触検出の幾何的影響）を許容
        assert ux_node1 < 1e-3, f"Non-zero ux: {ux_node1:.6e}"
        assert uy_node1 < 1e-3, f"Non-zero uy: {uy_node1:.6e}"


class TestFrictionMuDependence:
    """摩擦係数 μ の影響のバリデーション."""

    def test_zero_friction_no_tangential_resistance(self):
        """μ=0 で接線方向の抵抗なし.

        use_friction=True, μ=0 の場合、摩擦力がゼロ。
        接線荷重に対してばね変位のみが応答する。
        """
        result_mu0, mgr_mu0, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=10.0, mu=0.0)
        assert result_mu0.converged

        # 全ペアの摩擦力がゼロ
        for pair in mgr_mu0.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                q_norm = float(np.linalg.norm(pair.state.z_t))
                assert q_norm < 1e-10, f"Non-zero friction with μ=0: ||q||={q_norm:.6e}"

    def test_higher_mu_gives_less_tangential_displacement(self):
        """μ が大きいほど、接線変位が小さい.

        摩擦力が接線方向の滑りを抑制する効果を検証。
        """
        result_low, _, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=10.0, mu=0.1)
        result_high, _, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=10.0, mu=0.5)
        assert result_low.converged
        assert result_high.converged

        ux_low = abs(result_low.u[1 * 6 + 0])
        ux_high = abs(result_high.u[1 * 6 + 0])

        # 高μで接線変位が小さい（or 同等）
        assert ux_high <= ux_low + 1e-10, (
            f"Higher μ should reduce displacement: ux(μ=0.1)={ux_low:.6e}, ux(μ=0.5)={ux_high:.6e}"
        )

    def test_friction_force_scales_with_mu(self):
        """slip 時の摩擦力が μ に比例する.

        同じ法線力で μ を変えたとき、
        slip 状態の摩擦力は μ に比例（p_n が同程度の場合）。
        """
        mu_values = [0.1, 0.2, 0.4]
        friction_forces = []

        for mu in mu_values:
            result, mgr, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=30.0, mu=mu)
            assert result.converged
            for pair in mgr.pairs:
                if pair.state.status == ContactStatus.SLIDING:
                    q_norm = float(np.linalg.norm(pair.state.z_t))
                    p_n = pair.state.p_n
                    friction_forces.append((mu, q_norm, p_n))
                    break

        # 各μでの q/p_n ≈ μ（Coulomb条件）
        for mu, q_norm, p_n in friction_forces:
            if p_n > 1e-10:
                ratio = q_norm / p_n
                assert abs(ratio - mu) / mu < 0.05, (
                    f"Friction ratio mismatch: q/p_n={ratio:.4f}, μ={mu:.4f}"
                )


class TestFrictionContactPenetration:
    """摩擦の有無による貫通量への影響."""

    def test_friction_does_not_increase_penetration(self):
        """摩擦の有無が法線方向の貫通量に大きな影響を与えない.

        法線方向のみの押し付けでは、摩擦の有無で
        ギャップがほぼ同じであることを検証。
        """
        result_no_fric, mgr_nf, _, coords = _solve_friction_problem(f_z=50.0, f_t_x=0.0, mu=0.0)
        result_fric, mgr_f, _, _ = _solve_friction_problem(f_z=50.0, f_t_x=0.0, mu=0.3)
        assert result_no_fric.converged
        assert result_fric.converged

        # z方向の変位比較
        uz1_nf = result_no_fric.u[1 * 6 + 2]
        uz1_f = result_fric.u[1 * 6 + 2]

        # 法線方向のみの載荷なので、大きな差はないはず
        assert abs(uz1_nf - uz1_f) < 1e-4, (
            f"Friction changed normal displacement: Δuz = {abs(uz1_nf - uz1_f):.6e}"
        )
