"""法線接触力則のテスト.

物理検証 + @binds_to による Process 紐付け。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.process.strategies.penalty import (
    ALNormalForceProcess,
    NormalForceInput,
    NormalForceResult,
    SmoothNormalForceInput,
    SmoothNormalForceProcess,
)
from xkep_cae.process.strategies.penalty.law_normal import (
    _auto_beam_penalty_stiffness as auto_beam_penalty_stiffness,
)
from xkep_cae.process.strategies.penalty.law_normal import (
    _evaluate_al_normal_force as evaluate_al_normal_force,
)
from xkep_cae.process.strategies.penalty.law_normal import (
    _evaluate_smooth_normal_force as evaluate_smooth_normal_force,
)
from xkep_cae.process.strategies.penalty.law_normal import (
    _evaluate_smooth_normal_force_vectorized as evaluate_smooth_normal_force_vectorized,
)
from xkep_cae.process.strategies.penalty.law_normal import (
    _softplus as softplus,
)
from xkep_cae.process.testing import binds_to

# ── softplus 関数テスト ───────────────────────────────────


class TestSoftplus:
    """softplus 関数の数学的性質."""

    def test_large_positive(self) -> None:
        sp, sig = softplus(100.0, 1.0)
        assert sp == pytest.approx(100.0, abs=1e-6)
        assert sig == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self) -> None:
        sp, sig = softplus(-100.0, 1.0)
        assert sp == pytest.approx(0.0, abs=1e-10)
        assert sig == pytest.approx(0.0, abs=1e-10)

    def test_zero(self) -> None:
        sp, sig = softplus(0.0, 1.0)
        assert sp == pytest.approx(np.log(2.0), rel=1e-10)
        assert sig == pytest.approx(0.5, rel=1e-10)

    def test_always_non_negative(self) -> None:
        for x in np.linspace(-10, 10, 100):
            sp, _ = softplus(float(x), 0.1)
            assert sp >= 0.0


# ── AL 法線力テスト ───────────────────────────────────────


@binds_to(ALNormalForceProcess)
class TestALNormalForcePhysics:
    """AL 法線力の物理検証."""

    def test_no_penetration_no_force(self) -> None:
        """非接触（gap > 0）なら力ゼロ."""
        p_n, dp = evaluate_al_normal_force(gap=0.01, lambda_n=0.0, k_pen=1e5)
        assert p_n == 0.0
        assert dp == 0.0

    def test_penetration_gives_force(self) -> None:
        """貫入（gap < 0）なら押し返し力."""
        p_n, dp = evaluate_al_normal_force(gap=-0.001, lambda_n=0.0, k_pen=1e5)
        assert p_n == pytest.approx(1e5 * 0.001)
        assert dp == pytest.approx(-1e5)

    def test_force_proportional_to_penetration(self) -> None:
        """力は貫入量に比例."""
        p1, _ = evaluate_al_normal_force(gap=-0.001, lambda_n=0.0, k_pen=1e5)
        p2, _ = evaluate_al_normal_force(gap=-0.002, lambda_n=0.0, k_pen=1e5)
        assert p2 == pytest.approx(2.0 * p1)

    def test_augmented_lagrangian_multiplier(self) -> None:
        """λ > 0 で非接触でも力発生（AL 更新後）."""
        p_n, _ = evaluate_al_normal_force(gap=0.0005, lambda_n=100.0, k_pen=1e5)
        # λ + k*(-g) = 100 + 1e5*(-0.0005) = 100 - 50 = 50 > 0
        assert p_n == pytest.approx(50.0)

    def test_inactive_always_zero(self) -> None:
        p_n, dp = evaluate_al_normal_force(gap=-0.01, lambda_n=100.0, k_pen=1e5, is_active=False)
        assert p_n == 0.0
        assert dp == 0.0

    def test_process_wrapper(self) -> None:
        proc = ALNormalForceProcess()
        result = proc.process(NormalForceInput(gap=-0.001, lambda_n=0.0, k_pen=1e5))
        assert isinstance(result, NormalForceResult)
        assert result.p_n == pytest.approx(100.0)


# ── Smooth Penalty 法線力テスト ────────────────────────────


@binds_to(SmoothNormalForceProcess)
class TestSmoothNormalForcePhysics:
    """Smooth Penalty 法線力の物理検証."""

    def test_deep_penetration_matches_al(self) -> None:
        """深い貫入ではALと一致."""
        p_smooth, _ = evaluate_smooth_normal_force(gap=-0.01, k_pen=1e5, delta=1e-4)
        p_al, _ = evaluate_al_normal_force(gap=-0.01, lambda_n=0.0, k_pen=1e5)
        assert p_smooth == pytest.approx(p_al, rel=1e-3)

    def test_far_from_contact_near_zero(self) -> None:
        """非接触域では力ほぼゼロ."""
        p_n, _ = evaluate_smooth_normal_force(gap=0.01, k_pen=1e5, delta=1e-4)
        assert p_n < 1e-10

    def test_c_infinity_continuity(self) -> None:
        """遷移域で連続的（不連続なジャンプがない）."""
        gaps = np.linspace(-0.001, 0.001, 100)
        forces = [evaluate_smooth_normal_force(float(g), 1e5, delta=1e-4)[0] for g in gaps]
        diffs = np.diff(forces)
        # 単調減少（gap増加 → 力減少）
        assert all(d <= 0 for d in diffs)

    def test_tangent_stiffness_sign(self) -> None:
        """接線剛性は非正."""
        _, dp = evaluate_smooth_normal_force(gap=-0.001, k_pen=1e5, delta=1e-4)
        assert dp <= 0.0

    def test_vectorized_matches_scalar(self) -> None:
        """ベクトル版とスカラー版の出力一致."""
        gaps = np.array([-0.01, -0.001, 0.0, 0.001, 0.01])
        lambdas = np.zeros(5)
        p_vec, dp_vec = evaluate_smooth_normal_force_vectorized(gaps, 1e5, lambdas, delta=1e-4)
        for i, g in enumerate(gaps):
            p_s, dp_s = evaluate_smooth_normal_force(float(g), 1e5, delta=1e-4)
            assert p_vec[i] == pytest.approx(p_s, rel=1e-10)
            assert dp_vec[i] == pytest.approx(dp_s, rel=1e-10)

    def test_process_wrapper(self) -> None:
        proc = SmoothNormalForceProcess()
        result = proc.process(SmoothNormalForceInput(gap=-0.001, k_pen=1e5))
        assert isinstance(result, NormalForceResult)
        assert result.p_n > 0.0


# ── auto_beam_penalty_stiffness テスト ────────────────────


class TestAutoBeamPenaltyStiffness:
    """ペナルティ剛性自動推定の物理検証."""

    def test_basic(self) -> None:
        k = auto_beam_penalty_stiffness(200e9, 1e-12, 0.01, n_contact_pairs=6)
        k_bend = 12.0 * 200e9 * 1e-12 / 0.01**3
        assert k == pytest.approx(0.1 * k_bend / 6)

    def test_sqrt_scaling(self) -> None:
        k = auto_beam_penalty_stiffness(200e9, 1e-12, 0.01, n_contact_pairs=9, scaling="sqrt")
        k_bend = 12.0 * 200e9 * 1e-12 / 0.01**3
        assert k == pytest.approx(0.1 * k_bend / 3.0)

    def test_invalid_L(self) -> None:
        with pytest.raises(ValueError):
            auto_beam_penalty_stiffness(200e9, 1e-12, 0.0)
