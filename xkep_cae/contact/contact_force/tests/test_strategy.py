"""ContactForceStrategy 具象実装のテスト.

status-222 で HuberContactForceProcess に一本化。
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
    HuberContactForceProcess,
)
from xkep_cae.contact.contact_force.strategy import (
    _create_contact_force_strategy,
)
from xkep_cae.core.strategies import ContactForceStrategy
from xkep_cae.core.testing import binds_to


def _make_state(*, gap: float = -0.01, status: _ContactStatus = _ContactStatus.ACTIVE, **kw):
    return _ContactStateOutput(
        gap=gap,
        p_n=0.0,
        normal=np.array([0.0, 1.0, 0.0]),
        s=0.5,
        t=0.5,
        status=status,
        **kw,
    )


def _make_pair(*, gap: float = -0.01, **kw):
    return _ContactPairOutput(
        elem_a=0,
        elem_b=1,
        nodes_a=[0, 1],
        nodes_b=[2, 3],
        state=_make_state(gap=gap, **kw),
    )


class _MockManager:
    def __init__(self, pairs):
        self.pairs = list(pairs)


class TestContactForceProtocolConformance:
    """HuberContactForceProcess が Protocol を満たすことを検証."""

    def test_protocol_conformance(self):
        instance = HuberContactForceProcess(ndof=24)
        assert isinstance(instance, ContactForceStrategy)


# ── HuberContactForce ────────────────────────────────────────


@binds_to(HuberContactForceProcess)
class TestHuberContactForceProcess:
    """HuberContactForceProcess の単体テスト."""

    def test_evaluate_no_pairs(self):
        proc = HuberContactForceProcess(ndof=24)
        manager = _MockManager([])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))
        assert len(r) == 0

    def test_evaluate_with_active_pair(self):
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert np.any(f != 0.0)
        assert len(r) == 1

    def test_evaluate_inactive_pair_still_evaluated(self):
        """INACTIVE ペアでも gap<0 なら Huber 関数で力が出る（status フィルタ廃止）."""
        proc = HuberContactForceProcess(ndof=24)
        from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state

        pair = _make_pair(gap=-0.01)
        pair = _evolve_pair(pair, state=_evolve_state(pair.state, status=_ContactStatus.INACTIVE))
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert np.any(f != 0.0), "gap<0 のペアは INACTIVE でも力が出るべき"

    def test_evaluate_positive_gap(self):
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=0.1)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))

    def test_tangent_returns_zero(self):
        proc = HuberContactForceProcess(ndof=24)
        manager = _MockManager([])
        K = proc.tangent(np.zeros(24), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz == 0

    def test_tangent_with_penetration(self):
        """Huber は正定値接線を返す.

        K_contact = h'(x) * k_pen * g_shape ⊗ g_shape ≥ 0.
        """
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        K = proc.tangent(np.zeros(24), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz > 0
        K_dense = K.toarray()
        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(K_dense)
        assert np.all(eigvals >= -1e-10)

    def test_process_returns_output(self):
        proc = HuberContactForceProcess(ndof=24)
        inp = ContactForceInput(
            u=np.zeros(24),
            manager=_MockManager([]),
            k_pen=1e4,
        )
        out = proc.process(inp)
        assert isinstance(out, ContactForceOutput)

    def test_no_manager_pairs(self):
        proc = HuberContactForceProcess(ndof=24)
        manager = object()
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))

    def test_huber_function(self):
        h = HuberContactForceProcess._huber
        assert h(-1.0, 0.5) == 0.0
        assert h(1.0, 0.5) == pytest.approx(1.0)
        assert h(0.0, 0.5) > 0.0

    def test_huber_deriv(self):
        hd = HuberContactForceProcess._huber_deriv
        assert hd(-1.0, 0.5) == 0.0
        assert hd(1.0, 0.5) == pytest.approx(1.0)
        assert 0.0 < hd(0.0, 0.5) < 1.0

    def test_p_n_updated_on_pair(self):
        """evaluate が pair.state.p_n を更新する."""
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert manager.pairs[0].state.p_n > 0.0


# ── ファクトリ ─────────────────────────────────────────────


class TestCreateContactForceStrategy:
    """ファクトリ関数のテスト."""

    def test_returns_huber(self):
        s = _create_contact_force_strategy(ndof=24)
        assert isinstance(s, HuberContactForceProcess)

    def test_with_smoothing_delta(self):
        s = _create_contact_force_strategy(ndof=24, smoothing_delta=10.0)
        assert isinstance(s, HuberContactForceProcess)
        assert s._smoothing_delta == 10.0
