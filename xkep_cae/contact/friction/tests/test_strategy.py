"""FrictionStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + FrictionStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.friction import (
    CoulombReturnMappingProcess,
    FrictionInput,
    NoFrictionProcess,
    SmoothPenaltyFrictionProcess,
    create_friction_strategy,
)
from xkep_cae.core.strategies import FrictionStrategy
from xkep_cae.core.testing import binds_to

# ── Protocol 準拠 ─────────────────────────────────────────


class TestFrictionProtocolConformance:
    """全 Friction 具象が FrictionStrategy Protocol を満たすことを検証."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (NoFrictionProcess, {"ndof": 12}),
            (CoulombReturnMappingProcess, {"ndof": 12}),
            (SmoothPenaltyFrictionProcess, {"ndof": 12}),
        ],
    )
    def test_protocol_conformance(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, FrictionStrategy)


# ── NoFriction ────────────────────────────────────────────


@binds_to(NoFrictionProcess)
class TestNoFrictionProcess:
    """NoFrictionProcess の単体テスト."""

    def test_evaluate_zero_force(self):
        proc = NoFrictionProcess(ndof=12)
        f, r = proc.evaluate(np.zeros(12), [], 0.3)
        np.testing.assert_array_equal(f, np.zeros(12))

    def test_evaluate_zero_residual(self):
        proc = NoFrictionProcess(ndof=12)
        f, r = proc.evaluate(np.zeros(12), [], 0.3)
        assert len(r) == 0

    def test_tangent_zero_matrix(self):
        proc = NoFrictionProcess(ndof=12)
        K = proc.tangent(np.zeros(12), [], 0.3)
        assert K.shape == (12, 12)
        assert K.nnz == 0

    def test_ndof_from_u(self):
        proc = NoFrictionProcess(ndof=0)
        f, r = proc.evaluate(np.zeros(6), [], 0.3)
        assert len(f) == 6

    def test_process_method(self):
        proc = NoFrictionProcess(ndof=12)
        inp = FrictionInput(u=np.zeros(12), contact_pairs=[], mu=0.3)
        out = proc.process(inp)
        np.testing.assert_array_equal(out.friction_force, np.zeros(12))

    def test_meta(self):
        assert NoFrictionProcess.meta.name == "NoFriction"
        assert not NoFrictionProcess.meta.deprecated


# ── CoulombReturnMapping ──────────────────────────────────


@binds_to(CoulombReturnMappingProcess)
class TestCoulombReturnMappingProcess:
    """CoulombReturnMappingProcess の単体テスト."""

    def test_evaluate_no_pairs(self):
        proc = CoulombReturnMappingProcess(ndof=12)
        f, r = proc.evaluate(np.zeros(12), [], 0.3)
        np.testing.assert_array_equal(f, np.zeros(12))
        assert len(r) == 0

    def test_tangent_shape(self):
        proc = CoulombReturnMappingProcess(ndof=12)
        K = proc.tangent(np.zeros(12), [], 0.3)
        assert K.shape == (12, 12)

    def test_process_method(self):
        proc = CoulombReturnMappingProcess(ndof=12)
        inp = FrictionInput(u=np.zeros(12), contact_pairs=[], mu=0.3)
        out = proc.process(inp)
        assert out.friction_force.shape == (12,)

    def test_meta(self):
        assert CoulombReturnMappingProcess.meta.name == "CoulombReturnMapping"
        assert not CoulombReturnMappingProcess.meta.deprecated

    def test_k_pen_stored(self):
        proc = CoulombReturnMappingProcess(ndof=12, k_pen=1e4)
        assert proc._k_pen == 1e4

    def test_compute_k_t(self):
        proc = CoulombReturnMappingProcess(ndof=12, k_pen=1e4, k_t_ratio=0.5)
        assert proc.compute_k_t() == 5e3

    def test_mu_effective(self):
        proc = CoulombReturnMappingProcess(ndof=12, mu_ramp_counter=5, mu_ramp_steps=10)
        np.testing.assert_allclose(proc.compute_mu_effective(0.3), 0.15)


# ── SmoothPenaltyFriction ────────────────────────────────


@binds_to(SmoothPenaltyFrictionProcess)
class TestSmoothPenaltyFrictionProcess:
    """SmoothPenaltyFrictionProcess の単体テスト."""

    def test_evaluate_no_pairs(self):
        proc = SmoothPenaltyFrictionProcess(ndof=12)
        f, r = proc.evaluate(np.zeros(12), [], 0.3)
        np.testing.assert_array_equal(f, np.zeros(12))
        assert len(r) == 0

    def test_tangent_shape(self):
        proc = SmoothPenaltyFrictionProcess(ndof=12)
        K = proc.tangent(np.zeros(12), [], 0.3)
        assert K.shape == (12, 12)

    def test_k_t_ratio_stored(self):
        proc = SmoothPenaltyFrictionProcess(ndof=12, k_t_ratio=0.5)
        assert proc._k_t_ratio == 0.5

    def test_process_method(self):
        proc = SmoothPenaltyFrictionProcess(ndof=12)
        inp = FrictionInput(u=np.zeros(12), contact_pairs=[], mu=0.25)
        out = proc.process(inp)
        assert out.friction_force.shape == (12,)

    def test_meta(self):
        assert SmoothPenaltyFrictionProcess.meta.name == "SmoothPenaltyFriction"
        assert not SmoothPenaltyFrictionProcess.meta.deprecated


# ── create_friction_strategy ファクトリ ───────────────────


class TestCreateFrictionStrategy:
    """create_friction_strategy のテスト."""

    def test_no_friction(self):
        strategy = create_friction_strategy(use_friction=False, ndof=12)
        assert isinstance(strategy, NoFrictionProcess)

    def test_no_friction_default(self):
        strategy = create_friction_strategy(ndof=12)
        assert isinstance(strategy, NoFrictionProcess)

    def test_coulomb_ncp(self):
        strategy = create_friction_strategy(use_friction=True, contact_mode="ncp", ndof=12)
        assert isinstance(strategy, CoulombReturnMappingProcess)

    def test_smooth_penalty(self):
        strategy = create_friction_strategy(
            use_friction=True, contact_mode="smooth_penalty", ndof=12
        )
        assert isinstance(strategy, SmoothPenaltyFrictionProcess)

    def test_k_pen_propagation(self):
        strategy = create_friction_strategy(
            use_friction=True, contact_mode="ncp", ndof=12, k_pen=1e4
        )
        assert isinstance(strategy, CoulombReturnMappingProcess)
        assert strategy._k_pen == 1e4

    def test_k_t_ratio_propagation(self):
        strategy = create_friction_strategy(
            use_friction=True, contact_mode="smooth_penalty", ndof=12, k_t_ratio=0.5
        )
        assert isinstance(strategy, SmoothPenaltyFrictionProcess)
        assert strategy._k_t_ratio == 0.5

    def test_no_friction_evaluate_zero(self):
        strategy = create_friction_strategy(ndof=12)
        f, r = strategy.evaluate(np.zeros(12), [], 0.3)
        np.testing.assert_array_equal(f, np.zeros(12))

    def test_coulomb_evaluate_no_pairs(self):
        strategy = create_friction_strategy(use_friction=True, contact_mode="ncp", ndof=12)
        f, r = strategy.evaluate(np.zeros(12), [], 0.3)
        np.testing.assert_array_equal(f, np.zeros(12))
