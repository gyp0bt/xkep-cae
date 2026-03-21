"""ContactForceStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + ContactForceStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact._contact_pair import (
    _ContactPairOutput,
    _ContactStateOutput,
)
from xkep_cae.contact._types import ContactStatus as _ContactStatus
from xkep_cae.contact.contact_force import (
    ContactForceInput,
    ContactForceOutput,
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
)
from xkep_cae.contact.contact_force.strategy import (
    _create_contact_force_strategy,
)
from xkep_cae.core.strategies import ContactForceStrategy
from xkep_cae.core.testing import binds_to


def _make_state(*, gap: float = -0.01, status: _ContactStatus = _ContactStatus.ACTIVE, **kw):
    return _ContactStateOutput(
        gap=gap,
        s=kw.get("s", 0.5),
        t=kw.get("t", 0.5),
        normal=np.array([0.0, 0.0, 1.0]),
        tangent1=np.array([1.0, 0.0, 0.0]),
        tangent2=np.array([0.0, 1.0, 0.0]),
        status=status,
        p_n=kw.get("p_n", 0.0),
    )


def _make_pair(*, gap: float = -0.01, **kw):
    return _ContactPairOutput(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array(kw.get("nodes_a", [0, 1])),
        nodes_b=np.array(kw.get("nodes_b", [2, 3])),
        state=_make_state(gap=gap),
        radius_a=0.1,
        radius_b=0.1,
    )


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
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        lambdas = np.array([100.0])
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen=1e4)
        assert np.any(f != 0.0)
        assert len(r) == 1

    def test_evaluate_inactive_pair_skipped(self):
        proc = NCPContactForceProcess(ndof=24)
        from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state

        pair = _make_pair(gap=-0.01)
        pair = _evolve_pair(pair, state=_evolve_state(pair.state, status=_ContactStatus.INACTIVE))
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), np.array([100.0]), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))

    def test_evaluate_positive_gap_zero_lambda(self):
        proc = NCPContactForceProcess(ndof=24)
        pair = _make_pair(gap=0.1)
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
        pair = _make_pair(gap=-0.01)
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
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), np.array([0.0]), manager, k_pen=1e4)
        assert np.any(f != 0.0)

    def test_huber_epsilon_zero(self):
        """ε=0 でmax(0,-g) にフォールバック."""
        assert SmoothPenaltyContactForceProcess._huber_penalty(-0.01, 0.0) == pytest.approx(0.01)
        assert SmoothPenaltyContactForceProcess._huber_penalty(0.01, 0.0) == pytest.approx(0.0)

    def test_huber_large_penetration(self):
        """大貫入で線形ペナルティ: p = -g - ε/2."""
        eps = 0.1
        result = SmoothPenaltyContactForceProcess._huber_penalty(-1.0, eps)
        assert result == pytest.approx(1.0 - eps / 2.0, abs=0.01)

    def test_huber_derivative_epsilon_zero(self):
        """ε=0 で dp/dg = -1 (g<0), 0 (g≥0)."""
        assert SmoothPenaltyContactForceProcess._huber_penalty_derivative(-0.01, 0.0) == -1.0
        assert SmoothPenaltyContactForceProcess._huber_penalty_derivative(0.01, 0.0) == 0.0

    def test_tangent_no_pairs(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        manager = _MockManager([])
        K = proc.tangent(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz == 0

    def test_tangent_with_penetration(self):
        """smooth_penalty は正定値近似接線を返す（v3.0.0）.

        K_contact = k_pen * |dp/dg| * g_shape ⊗ g_shape ≥ 0.
        貫入時（gap < 0）には非ゼロの正半定値行列を返す。
        """
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        K = proc.tangent(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz > 0  # 正定値近似: 貫入時に非ゼロ
        # 対称性チェック
        K_dense = K.toarray()
        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-12)
        # 正半定値チェック
        eigvals = np.linalg.eigvalsh(K_dense)
        assert np.all(eigvals >= -1e-10)

    def test_tangent_no_penetration_is_small(self):
        """gap が大きい場合、接線はほぼゼロ."""
        proc = SmoothPenaltyContactForceProcess(ndof=24, smoothing_delta=100.0)
        pair = _make_pair(gap=1.0)
        manager = _MockManager([pair])
        K = proc.tangent(np.zeros(24), np.zeros(0), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        # sigmoid(-delta*g) ≈ 0 for large positive gap
        assert K.nnz == 0 or np.max(np.abs(K.toarray())) < 1e-10

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
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), np.array([0.0]), manager, k_pen=1e4)
        assert np.any(f != 0.0)


# ── ファクトリ ─────────────────────────────────────────────


class TestCreateContactForceStrategy:
    """ファクトリ関数のテスト."""

    def test_default_returns_ncp(self):
        s = _create_contact_force_strategy(ndof=24)
        assert isinstance(s, NCPContactForceProcess)

    def test_ncp_mode(self):
        s = _create_contact_force_strategy(contact_mode="ncp", ndof=24)
        assert isinstance(s, NCPContactForceProcess)

    def test_smooth_penalty_mode(self):
        s = _create_contact_force_strategy(contact_mode="smooth_penalty", ndof=24)
        assert isinstance(s, SmoothPenaltyContactForceProcess)

    def test_ncp_with_compliance(self):
        s = _create_contact_force_strategy(ndof=24, contact_compliance=0.01)
        assert isinstance(s, NCPContactForceProcess)
        assert s._contact_compliance == 0.01

    def test_smooth_penalty_with_delta(self):
        s = _create_contact_force_strategy(
            contact_mode="smooth_penalty",
            ndof=24,
            smoothing_delta=10.0,
        )
        assert isinstance(s, SmoothPenaltyContactForceProcess)
        assert s._smoothing_delta == 10.0
