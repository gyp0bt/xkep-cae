"""法線接触力則（Augmented Lagrangian）のテスト.

Phase C2: law_normal.py の単体テスト。
"""

import numpy as np

from xkep_cae.contact.law_normal import (
    auto_penalty_stiffness,
    evaluate_normal_force,
    initialize_penalty_stiffness,
    normal_force_linearization,
    update_al_multiplier,
)
from xkep_cae.contact.pair import ContactPair, ContactState, ContactStatus


def _make_pair(
    gap: float = -0.01,
    lambda_n: float = 0.0,
    k_pen: float = 1e4,
    status: ContactStatus = ContactStatus.ACTIVE,
) -> ContactPair:
    """テスト用の接触ペアを作成する."""
    state = ContactState(
        gap=gap,
        lambda_n=lambda_n,
        k_pen=k_pen,
        status=status,
        normal=np.array([0.0, 0.0, 1.0]),
    )
    return ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        state=state,
        radius_a=0.1,
        radius_b=0.1,
    )


class TestEvaluateNormalForce:
    """evaluate_normal_force のテスト."""

    def test_penetration_gives_positive_force(self):
        """貫通 (gap < 0) で正の反力."""
        pair = _make_pair(gap=-0.01, lambda_n=0.0, k_pen=1e4)
        p_n = evaluate_normal_force(pair)
        assert p_n > 0.0
        # p_n = max(0, 0 + 1e4 * 0.01) = 100
        assert abs(p_n - 100.0) < 1e-10
        assert abs(pair.state.p_n - 100.0) < 1e-10

    def test_separation_gives_zero_force(self):
        """離間 (gap > 0) でゼロ反力."""
        pair = _make_pair(gap=0.1, lambda_n=0.0, k_pen=1e4)
        p_n = evaluate_normal_force(pair)
        # p_n = max(0, 0 + 1e4 * (-0.1)) = max(0, -1000) = 0
        assert p_n == 0.0

    def test_inactive_gives_zero(self):
        """INACTIVE ペアはゼロ."""
        pair = _make_pair(gap=-0.01, status=ContactStatus.INACTIVE)
        p_n = evaluate_normal_force(pair)
        assert p_n == 0.0

    def test_al_multiplier_effect(self):
        """AL 乗数がある場合の反力計算."""
        pair = _make_pair(gap=-0.01, lambda_n=50.0, k_pen=1e4)
        p_n = evaluate_normal_force(pair)
        # p_n = max(0, 50 + 1e4 * 0.01) = max(0, 150) = 150
        assert abs(p_n - 150.0) < 1e-10

    def test_al_with_positive_gap_and_multiplier(self):
        """gap > 0 だが lambda_n > 0 → lambda_n > k_pen*gap なら接触."""
        pair = _make_pair(gap=0.001, lambda_n=100.0, k_pen=1e4)
        p_n = evaluate_normal_force(pair)
        # p_n = max(0, 100 + 1e4 * (-0.001)) = max(0, 100 - 10) = 90
        assert abs(p_n - 90.0) < 1e-10

    def test_zero_gap_with_zero_multiplier(self):
        """gap = 0, lambda_n = 0 → p_n = 0."""
        pair = _make_pair(gap=0.0, lambda_n=0.0, k_pen=1e4)
        p_n = evaluate_normal_force(pair)
        assert p_n == 0.0


class TestUpdateALMultiplier:
    """update_al_multiplier のテスト."""

    def test_active_pair_update(self):
        """ACTIVE ペアは lambda_n <- p_n."""
        pair = _make_pair(gap=-0.01, lambda_n=0.0, k_pen=1e4)
        evaluate_normal_force(pair)  # p_n = 100
        update_al_multiplier(pair)
        assert abs(pair.state.lambda_n - 100.0) < 1e-10

    def test_inactive_pair_reset(self):
        """INACTIVE ペアは lambda_n = 0."""
        pair = _make_pair(status=ContactStatus.INACTIVE)
        pair.state.lambda_n = 50.0
        update_al_multiplier(pair)
        assert pair.state.lambda_n == 0.0

    def test_repeated_update_converges(self):
        """複数回の AL 更新で乗数が安定する方向に動く."""
        pair = _make_pair(gap=-0.01, lambda_n=0.0, k_pen=1e4)
        prev_lambda = 0.0
        for _ in range(5):
            evaluate_normal_force(pair)
            update_al_multiplier(pair)
            # 乗数は増加方向に動く（ギャップ固定の場合）
            assert pair.state.lambda_n >= prev_lambda
            prev_lambda = pair.state.lambda_n


class TestNormalForceLinearization:
    """normal_force_linearization のテスト."""

    def test_active_with_positive_force(self):
        """接触中は k_pen を返す."""
        pair = _make_pair(gap=-0.01, k_pen=1e4)
        evaluate_normal_force(pair)
        dp_dg = normal_force_linearization(pair)
        assert abs(dp_dg - 1e4) < 1e-10

    def test_inactive_returns_zero(self):
        """INACTIVE はゼロ."""
        pair = _make_pair(status=ContactStatus.INACTIVE)
        dp_dg = normal_force_linearization(pair)
        assert dp_dg == 0.0

    def test_zero_force_returns_zero(self):
        """p_n = 0 のときはゼロ."""
        pair = _make_pair(gap=0.1, lambda_n=0.0, k_pen=1e4)
        evaluate_normal_force(pair)
        dp_dg = normal_force_linearization(pair)
        assert dp_dg == 0.0


class TestInitializePenalty:
    """initialize_penalty_stiffness のテスト."""

    def test_basic_initialization(self):
        """基本的なペナルティ初期化."""
        pair = _make_pair()
        pair.state.k_pen = 0.0
        pair.state.k_t = 0.0
        initialize_penalty_stiffness(pair, k_pen=1e5, k_t_ratio=0.3)
        assert abs(pair.state.k_pen - 1e5) < 1e-10
        assert abs(pair.state.k_t - 3e4) < 1e-10

    def test_default_ratio(self):
        """デフォルトの k_t_ratio = 0.5."""
        pair = _make_pair()
        pair.state.k_pen = 0.0
        initialize_penalty_stiffness(pair, k_pen=1e4)
        assert abs(pair.state.k_t - 5e3) < 1e-10


class TestAutoPenaltyStiffness:
    """auto_penalty_stiffness のテスト."""

    def test_ea_over_l(self):
        """EA/L ベースの推定."""
        k = auto_penalty_stiffness(E=2.1e11, A=1e-4, L=0.1)
        # k = 2.1e11 * 1e-4 / 0.1 = 2.1e8
        assert abs(k - 2.1e8) < 1.0

    def test_scale_factor(self):
        """スケール係数の効果."""
        k1 = auto_penalty_stiffness(E=2.1e11, A=1e-4, L=0.1, scale=1.0)
        k2 = auto_penalty_stiffness(E=2.1e11, A=1e-4, L=0.1, scale=10.0)
        assert abs(k2 / k1 - 10.0) < 1e-10
