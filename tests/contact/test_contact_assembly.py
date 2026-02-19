"""接触内力・接線剛性アセンブリのテスト.

Phase C2: assembly.py の単体テスト。
"""

import numpy as np

from xkep_cae.contact.assembly import (
    _contact_shape_vector,
    compute_contact_force,
    compute_contact_stiffness,
)
from xkep_cae.contact.law_normal import evaluate_normal_force
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)


def _make_active_pair(
    nodes_a=(0, 1),
    nodes_b=(2, 3),
    gap=-0.01,
    s=0.5,
    t=0.5,
    normal=(0.0, 0.0, 1.0),
    k_pen=1e4,
    lambda_n=0.0,
) -> ContactPair:
    """テスト用の ACTIVE 接触ペアを作成."""
    state = ContactState(
        s=s,
        t=t,
        gap=gap,
        normal=np.array(normal, dtype=float),
        tangent1=np.array([1.0, 0.0, 0.0]),
        tangent2=np.array([0.0, 1.0, 0.0]),
        lambda_n=lambda_n,
        k_pen=k_pen,
        p_n=0.0,
        status=ContactStatus.ACTIVE,
    )
    return ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array(nodes_a, dtype=int),
        nodes_b=np.array(nodes_b, dtype=int),
        state=state,
        radius_a=0.1,
        radius_b=0.1,
    )


class TestContactShapeVector:
    """_contact_shape_vector のテスト."""

    def test_midpoint_z_normal(self):
        """s=0.5, t=0.5, n=[0,0,1] の形状ベクトル."""
        pair = _make_active_pair(s=0.5, t=0.5, normal=(0.0, 0.0, 1.0))
        g = _contact_shape_vector(pair)

        # A0: -(1-0.5)*[0,0,1] = [0,0,-0.5]
        assert np.allclose(g[0:3], [0.0, 0.0, -0.5])
        # A1: -0.5*[0,0,1] = [0,0,-0.5]
        assert np.allclose(g[3:6], [0.0, 0.0, -0.5])
        # B0: +(1-0.5)*[0,0,1] = [0,0,0.5]
        assert np.allclose(g[6:9], [0.0, 0.0, 0.5])
        # B1: +0.5*[0,0,1] = [0,0,0.5]
        assert np.allclose(g[9:12], [0.0, 0.0, 0.5])

    def test_endpoint_s0_t1(self):
        """s=0, t=1 の形状ベクトル."""
        pair = _make_active_pair(s=0.0, t=1.0, normal=(1.0, 0.0, 0.0))
        g = _contact_shape_vector(pair)

        # A0: -(1-0)*[1,0,0] = [-1,0,0]
        assert np.allclose(g[0:3], [-1.0, 0.0, 0.0])
        # A1: -0*[1,0,0] = [0,0,0]
        assert np.allclose(g[3:6], [0.0, 0.0, 0.0])
        # B0: +(1-1)*[1,0,0] = [0,0,0]
        assert np.allclose(g[6:9], [0.0, 0.0, 0.0])
        # B1: +1*[1,0,0] = [1,0,0]
        assert np.allclose(g[9:12], [1.0, 0.0, 0.0])

    def test_action_reaction_balance(self):
        """形状ベクトルの合力がゼロ（作用・反作用）."""
        pair = _make_active_pair(s=0.3, t=0.7, normal=(0.5, 0.5, np.sqrt(0.5)))
        g = _contact_shape_vector(pair)
        total_force = g[0:3] + g[3:6] + g[6:9] + g[9:12]
        assert np.allclose(total_force, 0.0, atol=1e-14)


class TestComputeContactForce:
    """compute_contact_force のテスト."""

    def test_single_pair_midpoint(self):
        """1ペアの接触力が正しく節点に配分される."""
        pair = _make_active_pair(
            nodes_a=(0, 1),
            nodes_b=(2, 3),
            gap=-0.01,
            s=0.5,
            t=0.5,
            normal=(0.0, 0.0, 1.0),
            k_pen=1e4,
        )
        mgr = ContactManager(pairs=[pair], config=ContactConfig())

        # 4節点 × 6DOF = 24 DOF
        ndof_total = 24
        f_c = compute_contact_force(mgr, ndof_total, ndof_per_node=6)

        # p_n = max(0, 0 + 1e4 * 0.01) = 100

        # A0 (node 0): DOF 0-5, 力は -(1-0.5)*100*[0,0,1] = [0,0,-50]
        assert abs(f_c[0 * 6 + 2] - (-50.0)) < 1e-10
        # A1 (node 1): DOF 6-11, 力は -0.5*100*[0,0,1] = [0,0,-50]
        assert abs(f_c[1 * 6 + 2] - (-50.0)) < 1e-10
        # B0 (node 2): DOF 12-17, 力は +(1-0.5)*100*[0,0,1] = [0,0,50]
        assert abs(f_c[2 * 6 + 2] - 50.0) < 1e-10
        # B1 (node 3): DOF 18-23, 力は +0.5*100*[0,0,1] = [0,0,50]
        assert abs(f_c[3 * 6 + 2] - 50.0) < 1e-10

    def test_inactive_pair_no_force(self):
        """INACTIVE ペアは力に寄与しない."""
        pair = _make_active_pair()
        pair.state.status = ContactStatus.INACTIVE
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        f_c = compute_contact_force(mgr, 24, ndof_per_node=6)
        assert np.allclose(f_c, 0.0)

    def test_separation_no_force(self):
        """gap > 0 で力がゼロ."""
        pair = _make_active_pair(gap=0.1, lambda_n=0.0)
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        f_c = compute_contact_force(mgr, 24, ndof_per_node=6)
        assert np.allclose(f_c, 0.0)

    def test_total_force_balance(self):
        """全体の接触力の合力がゼロ（作用・反作用の法則）."""
        pair = _make_active_pair(
            nodes_a=(0, 1),
            nodes_b=(4, 5),
            gap=-0.02,
            s=0.3,
            t=0.8,
            normal=(0.3, 0.4, np.sqrt(1 - 0.09 - 0.16)),
            k_pen=5e3,
        )
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        ndof_total = 36  # 6 nodes × 6 DOF
        f_c = compute_contact_force(mgr, ndof_total, ndof_per_node=6)

        # 全節点の力の合計がゼロ
        total_fx = sum(f_c[i * 6 + 0] for i in range(6))
        total_fy = sum(f_c[i * 6 + 1] for i in range(6))
        total_fz = sum(f_c[i * 6 + 2] for i in range(6))
        assert abs(total_fx) < 1e-10
        assert abs(total_fy) < 1e-10
        assert abs(total_fz) < 1e-10

    def test_rotation_dofs_zero(self):
        """回転DOFへの寄与がゼロ."""
        pair = _make_active_pair(gap=-0.01)
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        f_c = compute_contact_force(mgr, 24, ndof_per_node=6)

        for node in range(4):
            for d in [3, 4, 5]:  # 回転DOF
                assert abs(f_c[node * 6 + d]) < 1e-15


class TestComputeContactStiffness:
    """compute_contact_stiffness のテスト."""

    def test_symmetry(self):
        """接触剛性行列が対称."""
        pair = _make_active_pair(gap=-0.01, k_pen=1e4)
        evaluate_normal_force(pair)
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        K_c = compute_contact_stiffness(mgr, 24, ndof_per_node=6)

        K_dense = K_c.toarray()
        assert np.allclose(K_dense, K_dense.T, atol=1e-10)

    def test_positive_semidefinite(self):
        """接触剛性行列が半正定値（固有値 >= 0）."""
        pair = _make_active_pair(gap=-0.01, k_pen=1e4)
        evaluate_normal_force(pair)
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        K_c = compute_contact_stiffness(mgr, 24, ndof_per_node=6)

        eigvals = np.linalg.eigvalsh(K_c.toarray())
        assert all(eigvals >= -1e-10)

    def test_rank_one(self):
        """1ペアの接触剛性行列はランク1（K = k*g*g^T）."""
        pair = _make_active_pair(gap=-0.01, k_pen=1e4)
        evaluate_normal_force(pair)
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        K_c = compute_contact_stiffness(mgr, 24, ndof_per_node=6)

        K_dense = K_c.toarray()
        eigvals = np.linalg.eigvalsh(K_dense)
        n_nonzero = np.sum(eigvals > 1e-6)
        assert n_nonzero == 1

    def test_inactive_pair_empty_matrix(self):
        """INACTIVE ペアは空行列."""
        pair = _make_active_pair()
        pair.state.status = ContactStatus.INACTIVE
        mgr = ContactManager(pairs=[pair], config=ContactConfig())
        K_c = compute_contact_stiffness(mgr, 24, ndof_per_node=6)
        assert K_c.nnz == 0

    def test_consistent_with_force_fd(self):
        """有限差分で接触力の接線を検証する.

        K_c ≈ (f_c(g-Δg) - f_c(g+Δg)) / (2Δg) * dg/du
        """
        k_pen = 1e4
        gap0 = -0.01
        dg = 1e-7

        # 参照接触力
        pair_ref = _make_active_pair(gap=gap0, k_pen=k_pen)
        evaluate_normal_force(pair_ref)

        # gap + dg
        pair_plus = _make_active_pair(gap=gap0 + dg, k_pen=k_pen)
        evaluate_normal_force(pair_plus)
        p_n_plus = pair_plus.state.p_n

        # gap - dg
        pair_minus = _make_active_pair(gap=gap0 - dg, k_pen=k_pen)
        evaluate_normal_force(pair_minus)
        p_n_minus = pair_minus.state.p_n

        # 数値微分 dp/dg
        dp_dg_fd = (p_n_minus - p_n_plus) / (2 * dg)  # 注: gap増→力減

        # 解析解
        dp_dg_exact = k_pen  # d(p_n)/d(-g) = k_pen

        assert abs(dp_dg_fd - dp_dg_exact) / dp_dg_exact < 1e-5


class TestMultiplePairs:
    """複数接触ペアのアセンブリテスト."""

    def test_two_pairs_superposition(self):
        """2ペアの接触力が独立に加算される."""
        pair1 = _make_active_pair(
            nodes_a=(0, 1),
            nodes_b=(2, 3),
            gap=-0.01,
            s=0.5,
            t=0.5,
            normal=(0.0, 0.0, 1.0),
            k_pen=1e4,
        )
        pair2 = _make_active_pair(
            nodes_a=(4, 5),
            nodes_b=(6, 7),
            gap=-0.02,
            s=0.5,
            t=0.5,
            normal=(0.0, 0.0, 1.0),
            k_pen=1e4,
        )
        pair2.elem_a = 2
        pair2.elem_b = 3
        mgr = ContactManager(pairs=[pair1, pair2], config=ContactConfig())

        ndof_total = 48  # 8 nodes × 6 DOF
        f_c = compute_contact_force(mgr, ndof_total, ndof_per_node=6)

        # pair1: p_n = 100, pair2: p_n = 200
        # pair1 の B0 (node 2): +0.5*100*[0,0,1] = [0,0,50]
        assert abs(f_c[2 * 6 + 2] - 50.0) < 1e-10
        # pair2 の B0 (node 6): +0.5*200*[0,0,1] = [0,0,100]
        assert abs(f_c[6 * 6 + 2] - 100.0) < 1e-10
