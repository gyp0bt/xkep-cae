"""Coulomb 摩擦則の物理検証テスト.

@binds_to による Process 紐付け + 純粋関数の物理的妥当性検証。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.friction import (
    ReturnMappingInput,
    ReturnMappingProcess,
    TangentInput,
)
from xkep_cae.contact.friction.law_friction import (
    FrictionTangentProcess,
    _compute_mu_effective,
    _return_mapping_core,
    _rotate_friction_history,
    _tangent_2x2_core,
)
from xkep_cae.core.testing import binds_to

# ── return_mapping_core ──────────────────────────────────


class TestReturnMappingCore:
    """Coulomb return mapping の物理テスト."""

    def test_no_contact_gives_zero(self):
        """p_n <= 0 の場合は摩擦力ゼロ."""
        q, is_stick, _, _ = _return_mapping_core(np.zeros(2), np.array([0.1, 0.0]), 1e4, 0.0, 0.3)
        np.testing.assert_array_equal(q, np.zeros(2))
        assert is_stick is True

    def test_no_friction_gives_zero(self):
        """μ = 0 の場合は摩擦力ゼロ."""
        q, is_stick, _, _ = _return_mapping_core(np.zeros(2), np.array([0.1, 0.0]), 1e4, 100.0, 0.0)
        np.testing.assert_array_equal(q, np.zeros(2))

    def test_stick_condition(self):
        """小さい接線変位 → stick."""
        z_old = np.zeros(2)
        delta_ut = np.array([1e-6, 0.0])
        k_t = 1e4
        p_n = 100.0
        mu = 0.3
        q, is_stick, _, _ = _return_mapping_core(z_old, delta_ut, k_t, p_n, mu)
        assert is_stick is True
        # q = z_old + k_t * delta_ut (stick)
        np.testing.assert_allclose(q, k_t * delta_ut)

    def test_slip_condition(self):
        """大きい接線変位 → slip."""
        z_old = np.zeros(2)
        delta_ut = np.array([1.0, 0.0])
        k_t = 1e4
        p_n = 100.0
        mu = 0.3
        q, is_stick, _, _ = _return_mapping_core(z_old, delta_ut, k_t, p_n, mu)
        assert is_stick is False
        # ||q|| = μ × p_n
        np.testing.assert_allclose(np.linalg.norm(q), mu * p_n, rtol=1e-10)

    def test_slip_force_direction(self):
        """slip 時の摩擦力方向は q_trial 方向."""
        z_old = np.zeros(2)
        delta_ut = np.array([1.0, 0.5])
        k_t = 1e4
        p_n = 100.0
        mu = 0.3
        q, is_stick, _, _ = _return_mapping_core(z_old, delta_ut, k_t, p_n, mu)
        assert is_stick is False
        # q 方向は delta_ut 方向（z_old=0 のため）
        expected_dir = delta_ut / np.linalg.norm(delta_ut)
        actual_dir = q / np.linalg.norm(q)
        np.testing.assert_allclose(actual_dir, expected_dir, atol=1e-10)

    def test_dissipation_non_negative(self):
        """散逸は非負."""
        z_old = np.zeros(2)
        delta_ut = np.array([0.5, 0.3])
        _, _, _, dissipation = _return_mapping_core(z_old, delta_ut, 1e4, 100.0, 0.3)
        assert dissipation >= 0.0

    def test_process_wrapper(self):
        """ReturnMappingProcess のラップが純関数と一致."""
        inp = ReturnMappingInput(
            z_t_old=np.zeros(2),
            delta_ut=np.array([0.01, 0.0]),
            k_t=1e4,
            p_n=100.0,
            mu=0.3,
        )
        proc = ReturnMappingProcess()
        result = proc.process(inp)
        q_ref, is_stick_ref, _, _ = _return_mapping_core(
            inp.z_t_old, inp.delta_ut, inp.k_t, inp.p_n, inp.mu
        )
        np.testing.assert_allclose(result.q, q_ref)
        assert result.is_stick == is_stick_ref


@binds_to(ReturnMappingProcess)
class TestReturnMappingProcessBinds:
    """ReturnMappingProcess の @binds_to 紐付け."""

    def test_meta(self):
        assert ReturnMappingProcess.meta.name == "ReturnMapping"
        assert not ReturnMappingProcess.meta.deprecated


# ── tangent_2x2_core ──────────────────────────────────────


class TestTangent2x2Core:
    """摩擦接線剛性の物理テスト."""

    def test_no_contact_zero(self):
        """p_n <= 0 → ゼロ行列."""
        D = _tangent_2x2_core(1e4, 0.0, 0.3, np.zeros(2), 0.0, True)
        np.testing.assert_array_equal(D, np.zeros((2, 2)))

    def test_stick_is_k_t_identity(self):
        """stick → D_t = k_t × I₂."""
        k_t = 1e4
        D = _tangent_2x2_core(k_t, 100.0, 0.3, np.array([10.0, 0.0]), 10.0, True)
        np.testing.assert_allclose(D, k_t * np.eye(2))

    def test_slip_eigenvalues(self):
        """slip → 固有値: 0 と ratio×k_t."""
        k_t = 1e4
        p_n = 100.0
        mu = 0.3
        z_t = np.array([30.0, 0.0])
        q_trial_norm = 50.0
        D = _tangent_2x2_core(k_t, p_n, mu, z_t, q_trial_norm, False)
        eigvals = np.sort(np.linalg.eigvalsh(D))
        ratio = mu * p_n / q_trial_norm
        # slip: 固有値は [0, ratio*k_t]
        np.testing.assert_allclose(eigvals[0], 0.0, atol=1e-8)
        np.testing.assert_allclose(eigvals[1], ratio * k_t, rtol=1e-10)

    def test_slip_symmetric(self):
        """slip 接線剛性は対称."""
        z_t = np.array([10.0, 5.0])
        D = _tangent_2x2_core(1e4, 100.0, 0.3, z_t, 50.0, False)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_process_wrapper(self):
        """FrictionTangentProcess が純関数と一致."""
        inp = TangentInput(
            k_t=1e4, p_n=100.0, mu=0.3, z_t=np.array([10.0, 0.0]), q_trial_norm=20.0, is_stick=True
        )
        proc = FrictionTangentProcess()
        result = proc.process(inp)
        D_ref = _tangent_2x2_core(inp.k_t, inp.p_n, inp.mu, inp.z_t, inp.q_trial_norm, inp.is_stick)
        np.testing.assert_allclose(result.D_t, D_ref)


@binds_to(FrictionTangentProcess)
class TestFrictionTangentProcessBinds:
    """FrictionTangentProcess の @binds_to 紐付け."""

    def test_meta(self):
        assert FrictionTangentProcess.meta.name == "FrictionTangent"


# ── rotate_friction_history ───────────────────────────────


class TestRotateFrictionHistory:
    """摩擦履歴回転テスト."""

    def test_zero_history_unchanged(self):
        """ゼロ履歴は回転しても変わらない."""
        z_new = _rotate_friction_history(
            np.zeros(2),
            np.array([1, 0, 0.0]),
            np.array([0, 1, 0.0]),
            np.array([0, 1, 0.0]),
            np.array([-1, 0, 0.0]),
        )
        np.testing.assert_allclose(z_new, np.zeros(2), atol=1e-15)

    def test_identity_rotation(self):
        """同一フレームなら不変."""
        z_t = np.array([5.0, 3.0])
        t1 = np.array([1, 0, 0.0])
        t2 = np.array([0, 1, 0.0])
        z_new = _rotate_friction_history(z_t, t1, t2, t1, t2)
        np.testing.assert_allclose(z_new, z_t, atol=1e-12)

    def test_90_degree_rotation(self):
        """90度回転."""
        z_t = np.array([1.0, 0.0])
        t1_old = np.array([1, 0, 0.0])
        t2_old = np.array([0, 1, 0.0])
        t1_new = np.array([0, 1, 0.0])
        t2_new = np.array([-1, 0, 0.0])
        z_new = _rotate_friction_history(z_t, t1_old, t2_old, t1_new, t2_new)
        np.testing.assert_allclose(z_new, [0.0, -1.0], atol=1e-12)

    def test_norm_preserved(self):
        """回転でノルムが保存."""
        z_t = np.array([3.0, 4.0])
        # 45度回転
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        t1_old = np.array([1, 0, 0.0])
        t2_old = np.array([0, 1, 0.0])
        t1_new = np.array([c, s, 0.0])
        t2_new = np.array([-s, c, 0.0])
        z_new = _rotate_friction_history(z_t, t1_old, t2_old, t1_new, t2_new)
        np.testing.assert_allclose(np.linalg.norm(z_new), np.linalg.norm(z_t), rtol=1e-12)


# ── compute_mu_effective ──────────────────────────────────


class TestComputeMuEffective:
    """μ ランプテスト."""

    def test_no_ramp(self):
        """mu_ramp_steps=0 → μ_target そのまま."""
        assert _compute_mu_effective(0.3, 0, 0) == 0.3

    def test_ramp_zero(self):
        """ramp_counter=0 → μ_eff=0."""
        assert _compute_mu_effective(0.3, 0, 5) == 0.0

    def test_ramp_half(self):
        """半分 → μ_target/2."""
        np.testing.assert_allclose(_compute_mu_effective(0.3, 5, 10), 0.15)

    def test_ramp_full(self):
        """完了 → μ_target."""
        np.testing.assert_allclose(_compute_mu_effective(0.3, 10, 10), 0.3)

    def test_ramp_exceeded(self):
        """超過 → μ_target（クランプ）."""
        np.testing.assert_allclose(_compute_mu_effective(0.3, 20, 10), 0.3)
