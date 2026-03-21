"""FrictionStrategy 具象実装のテスト.

status-222 で CoulombReturnMappingProcess に一本化。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.friction import (
    CoulombReturnMappingProcess,
    FrictionInput,
)
from xkep_cae.contact.friction.strategy import (
    _create_friction_strategy,
)
from xkep_cae.core.strategies import FrictionStrategy
from xkep_cae.core.testing import binds_to

# ── Protocol 準拠 ─────────────────────────────────────────


class TestFrictionProtocolConformance:
    """CoulombReturnMappingProcess が FrictionStrategy Protocol を満たすことを検証."""

    def test_protocol_conformance(self):
        instance = CoulombReturnMappingProcess(ndof=12)
        assert isinstance(instance, FrictionStrategy)


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


# ── _create_friction_strategy ファクトリ ───────────────────


class TestCreateFrictionStrategy:
    """_create_friction_strategy のテスト（status-222 で一本化）."""

    def test_returns_coulomb(self):
        strategy = _create_friction_strategy(ndof=12)
        assert isinstance(strategy, CoulombReturnMappingProcess)

    def test_k_pen_propagation(self):
        strategy = _create_friction_strategy(ndof=12, k_pen=1e4)
        assert isinstance(strategy, CoulombReturnMappingProcess)
        assert strategy._k_pen == 1e4

    def test_k_t_ratio_propagation(self):
        strategy = _create_friction_strategy(ndof=12, k_t_ratio=0.5)
        assert isinstance(strategy, CoulombReturnMappingProcess)
        assert strategy._k_t_ratio == 0.5

    def test_evaluate_no_pairs(self):
        strategy = _create_friction_strategy(ndof=12)
        f, r = strategy.evaluate(np.zeros(12), [], 0.3)
        np.testing.assert_array_equal(f, np.zeros(12))
