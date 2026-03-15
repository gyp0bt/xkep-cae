"""PenaltyStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + PenaltyStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import pytest

from xkep_cae.contact.penalty import (
    AutoBeamEIPenalty,
    AutoEALPenalty,
    ContinuationPenalty,
    PenaltyInput,
    PenaltyOutput,
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
