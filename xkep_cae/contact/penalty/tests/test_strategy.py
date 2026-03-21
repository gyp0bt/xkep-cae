"""PenaltyStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + PenaltyStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.penalty import (
    AutoBeamEIPenalty,
    AutoEALPenalty,
    AutoSmoothingDeltaProcess,
    ConstantPenalty,
    ContinuationPenalty,
    PenaltyInput,
    PenaltyOutput,
)
from xkep_cae.contact.penalty.strategy import (
    _create_penalty_strategy,
    _estimate_smoothing_delta,
)
from xkep_cae.core.strategies import PenaltyStrategy
from xkep_cae.core.testing import binds_to

# ── AutoBeamEIPenalty ────────────────────────────────────────


@binds_to(AutoBeamEIPenalty)
class TestAutoBeamEIPenalty:
    """AutoBeamEIPenalty の単体テスト."""

    def test_protocol_conformance(self) -> None:
        p = AutoBeamEIPenalty(beam_E=200e9, beam_I=1e-12, L_elem=0.01)
        assert isinstance(p, PenaltyStrategy)

    def test_basic_computation(self) -> None:
        p = AutoBeamEIPenalty(beam_E=200e9, beam_I=1e-12, L_elem=0.01)
        k_bend = 12.0 * 200e9 * 1e-12 / 0.01**3
        expected = 0.1 * k_bend  # scale=0.1, n_pairs=1
        assert p.compute_k_pen(0, 10) == pytest.approx(expected)

    def test_linear_scaling(self) -> None:
        p = AutoBeamEIPenalty(beam_E=200e9, beam_I=1e-12, L_elem=0.01, n_contact_pairs=6)
        k_bend = 12.0 * 200e9 * 1e-12 / 0.01**3
        expected = 0.1 * k_bend / 6
        assert p.compute_k_pen(0, 10) == pytest.approx(expected)

    def test_sqrt_scaling(self) -> None:
        p = AutoBeamEIPenalty(
            beam_E=200e9, beam_I=1e-12, L_elem=0.01, n_contact_pairs=9, scaling="sqrt"
        )
        k_bend = 12.0 * 200e9 * 1e-12 / 0.01**3
        expected = 0.1 * k_bend / 3.0  # sqrt(9) = 3
        assert p.compute_k_pen(0, 10) == pytest.approx(expected)

    def test_process_returns_frozen(self) -> None:
        p = AutoBeamEIPenalty(beam_E=200e9, beam_I=1e-12, L_elem=0.01)
        out = p.process(PenaltyInput(step=0, total_steps=10))
        assert isinstance(out, PenaltyOutput)
        with pytest.raises(AttributeError):
            out.k_pen = 999.0  # type: ignore[misc]

    def test_invalid_L_elem(self) -> None:
        with pytest.raises(ValueError, match="L_elem"):
            AutoBeamEIPenalty(beam_E=200e9, beam_I=1e-12, L_elem=0.0)

    def test_constant_across_steps(self) -> None:
        p = AutoBeamEIPenalty(beam_E=200e9, beam_I=1e-12, L_elem=0.01)
        k0 = p.compute_k_pen(0, 10)
        k5 = p.compute_k_pen(5, 10)
        assert k0 == k5


# ── AutoEALPenalty ────────────────────────────────────────


@binds_to(AutoEALPenalty)
class TestAutoEALPenalty:
    """AutoEALPenalty の単体テスト."""

    def test_protocol_conformance(self) -> None:
        p = AutoEALPenalty(beam_E=200e9, beam_A=1e-6, L_elem=0.01)
        assert isinstance(p, PenaltyStrategy)

    def test_basic_computation(self) -> None:
        p = AutoEALPenalty(beam_E=200e9, beam_A=1e-6, L_elem=0.01, scale=1.0)
        expected = 200e9 * 1e-6 / 0.01
        assert p.compute_k_pen(0, 1) == pytest.approx(expected)

    def test_process_output(self) -> None:
        p = AutoEALPenalty(beam_E=200e9, beam_A=1e-6, L_elem=0.01)
        out = p.process(PenaltyInput(step=0, total_steps=1))
        assert out.k_pen > 0.0


# ── ContinuationPenalty ───────────────────────────────────


@binds_to(ContinuationPenalty)
class TestContinuationPenalty:
    """ContinuationPenalty の単体テスト."""

    def test_protocol_conformance(self) -> None:
        p = ContinuationPenalty(k_pen_target=1e5)
        assert isinstance(p, PenaltyStrategy)

    def test_geometric_ramp(self) -> None:
        p = ContinuationPenalty(
            k_pen_target=1e5, start_fraction=0.01, ramp_steps=5, mode="geometric"
        )
        k0 = p.compute_k_pen(0, 10)
        k5 = p.compute_k_pen(5, 10)
        assert k0 == pytest.approx(0.01 * 1e5)
        assert k5 == pytest.approx(1e5)

    def test_linear_ramp(self) -> None:
        p = ContinuationPenalty(k_pen_target=1e5, start_fraction=0.1, ramp_steps=10, mode="linear")
        k0 = p.compute_k_pen(0, 10)
        k5 = p.compute_k_pen(5, 10)
        k10 = p.compute_k_pen(10, 10)
        assert k0 == pytest.approx(0.1 * 1e5)
        assert k5 == pytest.approx(1e5 * (0.1 + 0.9 * 5 / 10))
        assert k10 == pytest.approx(1e5)

    def test_monotonically_increasing(self) -> None:
        p = ContinuationPenalty(k_pen_target=1e5, ramp_steps=5)
        values = [p.compute_k_pen(i, 10) for i in range(6)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_beyond_ramp_is_target(self) -> None:
        p = ContinuationPenalty(k_pen_target=1e5, ramp_steps=5)
        assert p.compute_k_pen(100, 100) == pytest.approx(1e5)


# ── ConstantPenalty ──────────────────────────────────────


@binds_to(ConstantPenalty)
class TestConstantPenalty:
    """ConstantPenalty の単体テスト."""

    def test_protocol_conformance(self) -> None:
        p = ConstantPenalty(k_pen=1e6)
        assert isinstance(p, PenaltyStrategy)

    def test_returns_constant(self) -> None:
        p = ConstantPenalty(k_pen=42.0)
        assert p.compute_k_pen(0, 10) == pytest.approx(42.0)
        assert p.compute_k_pen(5, 10) == pytest.approx(42.0)

    def test_process_output(self) -> None:
        p = ConstantPenalty(k_pen=1e6)
        out = p.process(PenaltyInput(step=0, total_steps=1))
        assert isinstance(out, PenaltyOutput)
        assert out.k_pen == pytest.approx(1e6)


# ── ファクトリ関数テスト ──────────────────────────────────


class TestCreatePenaltyStrategy:
    """_create_penalty_strategy ファクトリのテスト."""

    def test_no_beam_params_returns_constant(self) -> None:
        s = _create_penalty_strategy(k_pen=1e5)
        assert isinstance(s, ConstantPenalty)
        assert s.compute_k_pen(0, 1) == pytest.approx(1e5)

    def test_beam_params_returns_auto_beam_ei(self) -> None:
        s = _create_penalty_strategy(beam_E=200e9, beam_I=1e-12, beam_L=0.01)
        assert isinstance(s, AutoBeamEIPenalty)
        assert s.compute_k_pen(0, 1) > 0.0

    def test_partial_beam_params_returns_constant(self) -> None:
        """beam パラメータが不完全な場合は ConstantPenalty にフォールバック."""
        s = _create_penalty_strategy(beam_E=200e9, beam_I=0.0, beam_L=0.01)
        assert isinstance(s, ConstantPenalty)


# ── AutoSmoothingDeltaProcess ────────────────────────────


@binds_to(AutoSmoothingDeltaProcess)
class TestAutoSmoothingDeltaProcess:
    """AutoSmoothingDeltaProcess の単体テスト."""

    def test_wire_diameter_2mm(self) -> None:
        """wire_diameter=2.0mm (r=1.0mm) → δ=5000, ε=0.0002mm."""
        p = AutoSmoothingDeltaProcess(r_min=1.0)
        assert p.delta == pytest.approx(5000.0)
        assert p.epsilon == pytest.approx(0.0002)

    def test_fine_wire(self) -> None:
        """r=0.2mm → δ=25000, ε=0.00004mm."""
        p = AutoSmoothingDeltaProcess(r_min=0.2)
        assert p.delta == pytest.approx(25000.0)
        assert p.epsilon == pytest.approx(0.00004)

    def test_custom_alpha(self) -> None:
        p = AutoSmoothingDeltaProcess(r_min=1.0, alpha=1e-3)
        assert p.delta == pytest.approx(1000.0)
        assert p.epsilon == pytest.approx(0.001)

    def test_invalid_r_min(self) -> None:
        with pytest.raises(ValueError, match="r_min"):
            AutoSmoothingDeltaProcess(r_min=0.0)

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            AutoSmoothingDeltaProcess(r_min=1.0, alpha=0.0)

    def test_process_returns_frozen(self) -> None:
        p = AutoSmoothingDeltaProcess(r_min=1.0)
        out = p.process(PenaltyInput(step=0, total_steps=1))
        assert isinstance(out, PenaltyOutput)
        assert out.k_pen == pytest.approx(5000.0)
        with pytest.raises(AttributeError):
            out.k_pen = 999.0  # type: ignore[misc]


class TestEstimateSmoothingDelta:
    """_estimate_smoothing_delta ユーティリティのテスト."""

    def test_scalar_radius(self) -> None:
        delta = _estimate_smoothing_delta(1.0)
        assert delta == pytest.approx(5000.0)

    def test_array_with_zeros(self) -> None:
        """ジグ要素（r=0）が混在しても正の半径のみで推定."""
        radii = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        delta = _estimate_smoothing_delta(radii)
        # r_min = 0.5 → ε = 2e-4 × 0.5 = 1e-4 → δ = 10000
        assert delta == pytest.approx(10000.0)

    def test_all_zero_returns_zero(self) -> None:
        """全要素が r=0 の場合は δ=0（smoothing なし）."""
        delta = _estimate_smoothing_delta(np.array([0.0, 0.0]))
        assert delta == 0.0
