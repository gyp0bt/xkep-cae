"""Friction Strategy 具象実装の1:1テスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.process.strategies.friction import (
    CoulombReturnMappingProcess,
    FrictionInput,
    NoFrictionProcess,
    SmoothPenaltyFrictionProcess,
)
from xkep_cae.process.strategies.protocols import FrictionStrategy
from xkep_cae.process.testing import binds_to

# --- Protocol 準拠チェック ---


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


# --- NoFriction ---


@binds_to(NoFrictionProcess)
class TestNoFrictionProcess:
    """NoFrictionProcess の単体テスト."""

    def test_evaluate_zero_force(self):
        """摩擦力はゼロ."""
        proc = NoFrictionProcess(ndof=12)
        f, r = proc.evaluate(np.zeros(12), [], 0.3)
        np.testing.assert_array_equal(f, np.zeros(12))

    def test_evaluate_zero_residual(self):
        """摩擦残差は空."""
        proc = NoFrictionProcess(ndof=12)
        f, r = proc.evaluate(np.zeros(12), [], 0.3)
        assert len(r) == 0

    def test_tangent_zero_matrix(self):
        """接線剛性はゼロ行列."""
        proc = NoFrictionProcess(ndof=12)
        K = proc.tangent(np.zeros(12), [], 0.3)
        assert K.shape == (12, 12)
        assert K.nnz == 0

    def test_ndof_from_u(self):
        """ndof=0 のとき u の長さから推定."""
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


# --- CoulombReturnMapping ---


@binds_to(CoulombReturnMappingProcess)
class TestCoulombReturnMappingProcess:
    """CoulombReturnMappingProcess の単体テスト."""

    def test_evaluate_no_pairs(self):
        """ペアなしの場合はゼロ."""
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


# --- SmoothPenaltyFriction ---


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
