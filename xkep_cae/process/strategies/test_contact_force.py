"""ContactForce Strategy 具象実装の1:1テスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.process.strategies.contact_force import (
    ContactForceInput,
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
)
from xkep_cae.process.strategies.protocols import ContactForceStrategy
from xkep_cae.process.testing import binds_to

# --- Protocol 準拠チェック ---


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


# --- NCP ContactForce ---


class _MockState:
    def __init__(self, gap=0.0, status="ACTIVE"):
        self.gap = gap
        self.status = status
        self.p_n = 0.0
        self.z_t = np.zeros(2)


class _MockPair:
    def __init__(self, gap=0.0):
        self.state = _MockState(gap=gap)


class _MockManager:
    def __init__(self, pairs=None):
        self.pairs = pairs or []


@binds_to(NCPContactForceProcess)
class TestNCPContactForceProcess:
    """NCPContactForceProcess の単体テスト."""

    def test_evaluate_no_manager_pairs(self):
        """manager に pairs がない場合."""
        proc = NCPContactForceProcess(ndof=12)
        f, r = proc.evaluate(np.zeros(12), np.zeros(0), object(), 1e6)
        np.testing.assert_array_equal(f, np.zeros(12))
        assert len(r) == 0

    def test_evaluate_with_pairs(self):
        """ペアありの場合、NCP残差が計算される."""
        manager = _MockManager(pairs=[_MockPair(gap=-0.01)])
        proc = NCPContactForceProcess(ndof=24)
        lambdas = np.array([100.0])
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen=1e4)
        assert len(r) == 1

    def test_evaluate_active_pair(self):
        """アクティブペア: 残差 = k_pen * gap."""
        manager = _MockManager(pairs=[_MockPair(gap=-0.01)])
        proc = NCPContactForceProcess(ndof=24)
        lambdas = np.array([100.0])
        k_pen = 1e4
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen)
        # p_n = max(0, 100 + 1e4 * 0.01) = max(0, 200) = 200 > 0 → active
        assert r[0] == pytest.approx(k_pen * (-0.01))

    def test_evaluate_inactive_pair(self):
        """非アクティブペア: 残差 = λ."""
        manager = _MockManager(pairs=[_MockPair(gap=1.0)])
        proc = NCPContactForceProcess(ndof=24)
        lambdas = np.array([0.0])
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen=1e4)
        # p_n = max(0, 0 + 1e4 * (-1.0)) = max(0, -10000) = 0 → inactive
        assert r[0] == pytest.approx(0.0)

    def test_tangent_shape(self):
        proc = NCPContactForceProcess(ndof=24)
        K = proc.tangent(np.zeros(24), np.zeros(0), object(), 1e4)
        assert K.shape == (24, 24)

    def test_contact_compliance(self):
        """δ正則化の効果."""
        manager = _MockManager(pairs=[_MockPair(gap=-0.01)])
        proc = NCPContactForceProcess(ndof=24, contact_compliance=1e-4)
        lambdas = np.array([100.0])
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen=1e4)
        assert len(r) == 1

    def test_process_method(self):
        proc = NCPContactForceProcess(ndof=24)
        inp = ContactForceInput(u=np.zeros(24), lambdas=np.zeros(0), manager=object(), k_pen=1e4)
        out = proc.process(inp)
        assert out.contact_force.shape == (24,)

    def test_meta(self):
        assert NCPContactForceProcess.meta.name == "NCPContactForce"
        assert not NCPContactForceProcess.meta.deprecated


# --- SmoothPenalty ContactForce ---


@binds_to(SmoothPenaltyContactForceProcess)
class TestSmoothPenaltyContactForceProcess:
    """SmoothPenaltyContactForceProcess の単体テスト."""

    def test_softplus_negative_gap(self):
        """負のギャップ（貫入）で正の力."""
        p = SmoothPenaltyContactForceProcess._softplus(-0.01, delta=100.0)
        assert p > 0.0

    def test_softplus_positive_gap(self):
        """正のギャップ（非接触）で小さな力."""
        p_near = SmoothPenaltyContactForceProcess._softplus(0.01, delta=100.0)
        p_far = SmoothPenaltyContactForceProcess._softplus(1.0, delta=100.0)
        assert p_near > p_far
        assert p_far < 1e-10

    def test_softplus_zero_delta(self):
        """δ=0 のとき max(0, -g) に退化."""
        p = SmoothPenaltyContactForceProcess._softplus(-0.5, delta=0.0)
        assert p == pytest.approx(0.5)
        p2 = SmoothPenaltyContactForceProcess._softplus(0.5, delta=0.0)
        assert p2 == pytest.approx(0.0)

    def test_softplus_large_penetration(self):
        """大きな貫入でも数値安定."""
        p = SmoothPenaltyContactForceProcess._softplus(-10.0, delta=100.0)
        assert p == pytest.approx(10.0, rel=0.01)

    def test_evaluate_no_pairs(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        f, r = proc.evaluate(np.zeros(24), np.zeros(0), object(), 1e4)
        np.testing.assert_array_equal(f, np.zeros(24))
        assert len(r) == 0

    def test_evaluate_with_pairs(self):
        manager = _MockManager(pairs=[_MockPair(gap=-0.01)])
        proc = SmoothPenaltyContactForceProcess(ndof=24, smoothing_delta=100.0)
        lambdas = np.array([0.0])
        f, r = proc.evaluate(np.zeros(24), lambdas, manager, k_pen=1e4)
        assert len(r) == 1

    def test_tangent_shape(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        K = proc.tangent(np.zeros(24), np.zeros(0), object(), 1e4)
        assert K.shape == (24, 24)

    def test_process_method(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24)
        inp = ContactForceInput(u=np.zeros(24), lambdas=np.zeros(0), manager=object(), k_pen=1e4)
        out = proc.process(inp)
        assert out.contact_force.shape == (24,)

    def test_meta(self):
        assert SmoothPenaltyContactForceProcess.meta.name == "SmoothPenaltyContactForce"
        assert not SmoothPenaltyContactForceProcess.meta.deprecated

    def test_uzawa_params(self):
        proc = SmoothPenaltyContactForceProcess(ndof=24, n_uzawa_max=10, tol_uzawa=1e-4)
        assert proc._n_uzawa_max == 10
        assert proc._tol_uzawa == 1e-4
