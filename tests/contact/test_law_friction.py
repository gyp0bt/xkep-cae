"""Coulomb 摩擦則（return mapping）のテスト.

Phase C3: law_friction.py のユニットテスト。

テスト構成:
1. stick 状態の再現
2. slip 状態の再現（radial return）
3. stick → slip 遷移
4. 散逸非負性 (D_inc >= 0)
5. μランプ動作確認
6. 接線相対変位の計算
7. 摩擦接線剛性
8. 非接触ペアの摩擦力ゼロ確認
"""

import numpy as np
import pytest

from xkep_cae.contact.law_friction import (
    compute_mu_effective,
    compute_tangential_displacement,
    friction_return_mapping,
    friction_tangent_2x2,
)
from xkep_cae.contact.pair import ContactPair, ContactState, ContactStatus


def _make_active_pair(
    k_t: float = 1000.0,
    p_n: float = 10.0,
    z_t: np.ndarray | None = None,
) -> ContactPair:
    """テスト用のアクティブ接触ペアを生成する."""
    state = ContactState(
        s=0.5,
        t=0.5,
        gap=-0.01,
        normal=np.array([0.0, 0.0, 1.0]),
        tangent1=np.array([1.0, 0.0, 0.0]),
        tangent2=np.array([0.0, 1.0, 0.0]),
        lambda_n=0.0,
        k_pen=10000.0,
        k_t=k_t,
        p_n=p_n,
        z_t=z_t if z_t is not None else np.zeros(2),
        status=ContactStatus.ACTIVE,
        stick=True,
        dissipation=0.0,
    )
    return ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        state=state,
    )


class TestFrictionReturnMapping:
    """friction_return_mapping のテスト."""

    def test_stick_small_displacement(self):
        """微小変位で stick 状態を維持."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        mu = 0.3
        # μ*p_n = 3.0, k_t*Δu_t = 1000*0.001 = 1.0 < 3.0 → stick
        delta_ut = np.array([0.001, 0.0])

        q = friction_return_mapping(pair, delta_ut, mu)

        assert pair.state.stick is True
        assert pair.state.status == ContactStatus.ACTIVE
        np.testing.assert_allclose(q, [1.0, 0.0])
        np.testing.assert_allclose(pair.state.z_t, [1.0, 0.0])

    def test_slip_large_displacement(self):
        """大変位で slip 状態に遷移（radial return）."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        mu = 0.3
        # μ*p_n = 3.0, k_t*Δu_t = 1000*0.01 = 10.0 > 3.0 → slip
        delta_ut = np.array([0.01, 0.0])

        q = friction_return_mapping(pair, delta_ut, mu)

        assert pair.state.stick is False
        assert pair.state.status == ContactStatus.SLIDING
        # ||q|| = μ*p_n = 3.0
        np.testing.assert_allclose(np.linalg.norm(q), 3.0, atol=1e-12)
        # 方向は q_trial 方向 (= [1, 0])
        np.testing.assert_allclose(q, [3.0, 0.0], atol=1e-12)

    def test_slip_2d_direction(self):
        """2次元接線方向での slip 方向保持."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        mu = 0.3
        delta_ut = np.array([0.01, 0.01])  # 45度方向

        q = friction_return_mapping(pair, delta_ut, mu)

        assert pair.state.stick is False
        np.testing.assert_allclose(np.linalg.norm(q), 3.0, atol=1e-12)
        # q_trial = [10, 10], 方向 = [1/√2, 1/√2]
        expected = 3.0 * np.array([1.0, 1.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(q, expected, atol=1e-12)

    def test_stick_to_slip_transition(self):
        """stick → slip 遷移: 累積変位が閾値を超えたとき."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        mu = 0.3
        # Step 1: stick
        delta_ut1 = np.array([0.002, 0.0])
        q1 = friction_return_mapping(pair, delta_ut1, mu)
        assert pair.state.stick is True
        np.testing.assert_allclose(q1, [2.0, 0.0])

        # Step 2: z_t = [2.0, 0], Δu = [0.002, 0]
        # q_trial = [2.0 + 2.0, 0] = [4.0, 0] > μ*p_n=3.0 → slip
        delta_ut2 = np.array([0.002, 0.0])
        q2 = friction_return_mapping(pair, delta_ut2, mu)
        assert pair.state.stick is False
        np.testing.assert_allclose(np.linalg.norm(q2), 3.0, atol=1e-12)

    def test_dissipation_nonnegative_stick(self):
        """stick 時の散逸."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        mu = 0.3
        delta_ut = np.array([0.001, 0.0])

        friction_return_mapping(pair, delta_ut, mu)
        # stick: q = k_t*Δu_t, D = q·Δu_t = k_t*||Δu_t||^2 >= 0
        assert pair.state.dissipation >= -1e-15

    def test_dissipation_nonnegative_slip(self):
        """slip 時の散逸が非負."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        mu = 0.3
        delta_ut = np.array([0.01, 0.005])

        friction_return_mapping(pair, delta_ut, mu)
        # slip: D = q·Δu_t >= 0 (q と Δu_t は同方向)
        assert pair.state.dissipation >= -1e-15

    def test_zero_normal_force(self):
        """法線力ゼロでは摩擦力ゼロ."""
        pair = _make_active_pair(k_t=1000.0, p_n=0.0)
        mu = 0.3
        delta_ut = np.array([0.01, 0.0])

        q = friction_return_mapping(pair, delta_ut, mu)

        np.testing.assert_allclose(q, [0.0, 0.0])
        assert pair.state.stick is True

    def test_inactive_pair(self):
        """非接触ペアでは摩擦力ゼロ."""
        pair = _make_active_pair()
        pair.state.status = ContactStatus.INACTIVE
        delta_ut = np.array([0.01, 0.0])

        q = friction_return_mapping(pair, delta_ut, mu=0.3)

        np.testing.assert_allclose(q, [0.0, 0.0])

    def test_zero_friction_coefficient(self):
        """μ=0 で摩擦力ゼロ."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        delta_ut = np.array([0.01, 0.0])

        q = friction_return_mapping(pair, delta_ut, mu=0.0)

        np.testing.assert_allclose(q, [0.0, 0.0])

    def test_z_t_updated_correctly(self):
        """z_t がリターン後の q に更新される."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        mu = 0.3
        delta_ut = np.array([0.01, 0.0])

        q = friction_return_mapping(pair, delta_ut, mu)

        np.testing.assert_allclose(pair.state.z_t, q)


class TestFrictionTangent:
    """friction_tangent_2x2 のテスト."""

    def test_stick_tangent(self):
        """stick 時は k_t * I_2."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        pair.state.stick = True
        mu = 0.3

        D_t = friction_tangent_2x2(pair, mu)

        expected = 1000.0 * np.eye(2)
        np.testing.assert_allclose(D_t, expected)

    def test_inactive_tangent_zero(self):
        """非接触では零行列."""
        pair = _make_active_pair()
        pair.state.status = ContactStatus.INACTIVE
        mu = 0.3

        D_t = friction_tangent_2x2(pair, mu)

        np.testing.assert_allclose(D_t, np.zeros((2, 2)))

    def test_slip_tangent_symmetric(self):
        """slip 時の接線剛性が対称."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        pair.state.stick = False
        pair.state.z_t = np.array([2.0, 1.0])
        mu = 0.3

        D_t = friction_tangent_2x2(pair, mu)

        np.testing.assert_allclose(D_t, D_t.T)

    def test_slip_tangent_positive_semidefinite(self):
        """slip 時の接線剛性が半正定値."""
        pair = _make_active_pair(k_t=1000.0, p_n=10.0)
        pair.state.stick = False
        pair.state.z_t = np.array([2.0, 1.0])
        mu = 0.3

        D_t = friction_tangent_2x2(pair, mu)

        eigvals = np.linalg.eigvalsh(D_t)
        assert np.all(eigvals >= -1e-12)


class TestMuRamp:
    """compute_mu_effective のテスト."""

    def test_no_ramp(self):
        """ランプなし (mu_ramp_steps=0) でフル μ."""
        mu_eff = compute_mu_effective(0.3, ramp_counter=0, mu_ramp_steps=0)
        assert mu_eff == pytest.approx(0.3)

    def test_ramp_start(self):
        """ランプ開始時 (counter=0) で μ=0."""
        mu_eff = compute_mu_effective(0.3, ramp_counter=0, mu_ramp_steps=5)
        assert mu_eff == pytest.approx(0.0)

    def test_ramp_halfway(self):
        """ランプ途中 (counter=2, steps=4) で μ=0.5*μ_target."""
        mu_eff = compute_mu_effective(0.4, ramp_counter=2, mu_ramp_steps=4)
        assert mu_eff == pytest.approx(0.2)

    def test_ramp_complete(self):
        """ランプ完了で μ_target."""
        mu_eff = compute_mu_effective(0.3, ramp_counter=5, mu_ramp_steps=5)
        assert mu_eff == pytest.approx(0.3)

    def test_ramp_beyond(self):
        """ランプ超過でもクランプ."""
        mu_eff = compute_mu_effective(0.3, ramp_counter=10, mu_ramp_steps=5)
        assert mu_eff == pytest.approx(0.3)


class TestTangentialDisplacement:
    """compute_tangential_displacement のテスト."""

    def test_relative_sliding(self):
        """B が t1 方向にスライド → 正の Δu_t[0]."""
        pair = _make_active_pair()
        pair.state.s = 0.5
        pair.state.t = 0.5
        pair.state.tangent1 = np.array([1.0, 0.0, 0.0])
        pair.state.tangent2 = np.array([0.0, 1.0, 0.0])
        pair.state.normal = np.array([0.0, 0.0, 1.0])
        # 4節点: A0=0, A1=1, B0=2, B1=3
        # 6DOF/node → 24 DOF total
        ndof = 24
        u_ref = np.zeros(ndof)
        u_cur = np.zeros(ndof)
        # B0 (node 2) を x方向に +0.01 移動
        u_cur[2 * 6 + 0] = 0.01
        # B1 (node 3) も同じ
        u_cur[3 * 6 + 0] = 0.01

        node_coords = np.zeros((4, 3))  # 未使用だが引数として必要

        delta_ut = compute_tangential_displacement(pair, u_cur, u_ref, node_coords, ndof_per_node=6)

        # B が x 方向に 0.01 スライド → t1 方向に 0.01
        np.testing.assert_allclose(delta_ut, [0.01, 0.0], atol=1e-14)

    def test_normal_displacement_no_tangential(self):
        """法線方向の変位は接線成分ゼロ."""
        pair = _make_active_pair()
        pair.state.s = 0.5
        pair.state.t = 0.5
        pair.state.tangent1 = np.array([1.0, 0.0, 0.0])
        pair.state.tangent2 = np.array([0.0, 1.0, 0.0])
        pair.state.normal = np.array([0.0, 0.0, 1.0])
        ndof = 24
        u_ref = np.zeros(ndof)
        u_cur = np.zeros(ndof)
        # B0, B1 を z 方向（法線方向）に移動
        u_cur[2 * 6 + 2] = 0.01
        u_cur[3 * 6 + 2] = 0.01

        node_coords = np.zeros((4, 3))
        delta_ut = compute_tangential_displacement(pair, u_cur, u_ref, node_coords, ndof_per_node=6)

        np.testing.assert_allclose(delta_ut, [0.0, 0.0], atol=1e-14)

    def test_shape_function_weighting(self):
        """s,t に応じた形状関数の重み付けが正しい."""
        pair = _make_active_pair()
        pair.state.s = 0.25  # A 側: 0.75*A0 + 0.25*A1
        pair.state.t = 0.75  # B 側: 0.25*B0 + 0.75*B1
        pair.state.tangent1 = np.array([1.0, 0.0, 0.0])
        pair.state.tangent2 = np.array([0.0, 1.0, 0.0])
        ndof = 24
        u_ref = np.zeros(ndof)
        u_cur = np.zeros(ndof)
        # B1 (node 3) のみ x 方向に移動
        u_cur[3 * 6 + 0] = 0.04

        node_coords = np.zeros((4, 3))
        delta_ut = compute_tangential_displacement(pair, u_cur, u_ref, node_coords, ndof_per_node=6)

        # B 側: 0.25*0 + 0.75*0.04 = 0.03
        np.testing.assert_allclose(delta_ut, [0.03, 0.0], atol=1e-14)
