"""ContactForceStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + ContactForceStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact._types import ContactStatus as _ContactStatus
from xkep_cae.contact.contact_force import (
    ContactForceInput,
    ContactForceOutput,
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
    create_contact_force_strategy,
)
from xkep_cae.core.strategies import ContactForceStrategy
from xkep_cae.core.testing import binds_to


class _MockState:
    def __init__(
        self,
        *,
        gap: float = -0.01,
        s: float = 0.5,
        t: float = 0.5,
        status: _ContactStatus = _ContactStatus.ACTIVE,
    ):
        self.gap = gap
        self.s = s
        self.t = t
        self.normal = np.array([0.0, 0.0, 1.0])
        self.status = status
        self.p_n = 0.0
        self.coating_compression = 0.0
        self.tangent1 = np.array([1.0, 0.0, 0.0])
        self.tangent2 = np.array([0.0, 1.0, 0.0])


class _MockPair:
    def __init__(
        self,
        *,
        gap: float = -0.01,
        nodes_a: tuple[int, int] = (0, 1),
        nodes_b: tuple[int, int] = (2, 3),
    ):
        self.state = _MockState(gap=gap)
        self.nodes_a = nodes_a
        self.nodes_b = nodes_b
        self.radius_a = 0.1
        self.radius_b = 0.1


class _MockManager:
    def __init__(self, pairs: list):
        self.pairs = pairs


# ── Protocol 準拠 ─────────────────────────────────────────


class TestContactForceProtocolConformance:
    """全 ContactForce 具象が Protocol を満たすことを検証."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (NCPContactForceProcess, {"ndof": 24}),
            (SmoothPenaltyContactForceProcess, {"ndof": 24}),
        ],
    )
    def test_protocol_conformance(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, ContactForceStrategy)


# ── NCPContactForce ────────────────────────────────────────


@binds_to(NCPContactForceProcess)
class TestNCPContactForceProcess:
    """NCPContactForceProcess の単体テスト."""

    def test_evaluate_no_pairs(self):
        proc = NCPContactForceProcess(ndof=24)
        manager = _MockManager([])
        f, r = proc.evaluate(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))
        assert len(r) == 0

    def test_evaluate_with_active_pair(self):
        proc = NCPContactForceProcess(ndof=24)
        pair = _MockPair(gap=-0.01)
        manager = _MockManager([pair])
        lambdas = np.array([100.0])
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen=1e4)
        assert np.any(f != 0.0)
        assert len(r) == 1

    def test_evaluate_inactive_pair_skipped(self):
        proc = NCPContactForceProcess(ndof=24)
        pair = _MockPair(gap=-0.01)
        pair.state.status = _ContactStatus.INACTIVE
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), np.array([100.0]), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))

    def test_evaluate_positive_gap_zero_lambda(self):
        proc = NCPContactForceProcess(ndof=24)
        pair = _MockPair(gap=0.1)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), np.array([0.0]), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))

    def test_tangent_returns_zero(self):
        proc = NCPContactForceProcess(ndof=24)
        manager = _MockManager([])
        K = proc.tangent(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz == 0

    def test_process_returns_output(self):
        proc = NCPContactForceProcess(ndof=24)
        inp = ContactForceInput(
            u=np.zeros(24),
            lambdas=np.zeros(0),
            manager=_MockManager([]),
            k_pen=1e4,
        )
        out = proc.process(inp)
        assert isinstance(out, ContactForceOutput)

    def test_contact_compliance(self):
        proc = NCPContactForceProcess(ndof=24, contact_compliance=1e-6)
        pair = _MockPair(gap=-0.01)
        manager = _MockManager([pair])
        lambdas = np.array([100.0])
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen=1e4)
        assert np.any(f != 0.0)

    def test_no_manager_pairs(self):
        proc = NCPContactForceProcess(ndof=24)
        manager = object()
        f, r = proc.evaluate(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))


# ── SmoothPenaltyContactForce ─────────────────────────────


@binds_to(SmoothPenaltyContactForceProcess)
class TestSmoothPenaltyContactForceProcess:
    """SmoothPenaltyContactForceProcess の単体テスト."""

    def test_evaluate_no_pairs(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        manager = _MockManager([])
        f, r = proc.evaluate(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))
        assert len(r) == 0

    def test_evaluate_with_active_pair(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        pair = _MockPair(gap=-0.01)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), np.array([0.0]), manager, k_pen=1e4)
        assert np.any(f != 0.0)

    def test_softplus_delta_zero(self):
        assert SmoothPenaltyContactForceProcess._softplus(-0.01, 0.0) == pytest.approx(0.01)
        assert SmoothPenaltyContactForceProcess._softplus(0.01, 0.0) == pytest.approx(0.0)

    def test_softplus_large_penetration(self):
        result = SmoothPenaltyContactForceProcess._softplus(-1.0, 10.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_softplus_derivative_delta_zero(self):
        assert SmoothPenaltyContactForceProcess._softplus_derivative(-0.01, 0.0) == -1.0
        assert SmoothPenaltyContactForceProcess._softplus_derivative(0.01, 0.0) == 0.0

    def test_tangent_no_pairs(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        manager = _MockManager([])
        K = proc.tangent(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz == 0

    def test_tangent_with_penetration(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        pair = _MockPair(gap=-0.01)
        manager = _MockManager([pair])
        K = proc.tangent(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz > 0

    def test_process_returns_output(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        inp = ContactForceInput(
            u=np.zeros(24),
            lambdas=np.zeros(0),
            manager=_MockManager([]),
            k_pen=1e4,
        )
        out = proc.process(inp)
        assert isinstance(out, ContactForceOutput)

    def test_smoothing_delta_parameter(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24, smoothing_delta=10.0)
        pair = _MockPair(gap=-0.01)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), np.array([0.0]), manager, k_pen=1e4)
        assert np.any(f != 0.0)


# ── ファクトリ ─────────────────────────────────────────────


class TestCreateContactForceStrategy:
    """ファクトリ関数のテスト."""

    def test_default_returns_ncp(self):
        s = create_contact_force_strategy(ndof=24)
        assert isinstance(s, NCPContactForceProcess)

    def test_ncp_mode(self):
        s = create_contact_force_strategy(contact_mode="ncp", ndof=24)
        assert isinstance(s, NCPContactForceProcess)

    def test_smooth_penalty_mode(self):
        s = create_contact_force_strategy(contact_mode="smooth_penalty", ndof=24)
        assert isinstance(s, SmoothPenaltyContactForceProcess)

    def test_ncp_with_compliance(self):
        s = create_contact_force_strategy(ndof=24, contact_compliance=0.01)
        assert isinstance(s, NCPContactForceProcess)
        assert s._contact_compliance == 0.01

    def test_smooth_penalty_with_delta(self):
        s = create_contact_force_strategy(
            contact_mode="smooth_penalty",
            ndof=24,
            smoothing_delta=10.0,
        )
        assert isinstance(s, SmoothPenaltyContactForceProcess)
        assert s._smoothing_delta == 10.0
