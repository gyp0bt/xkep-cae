"""TestFiber1DMaterialAPI — Protocol 準拠・型・frozen 性検査.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md
Phase F1 完了判定テスト（6件の一部）。

[← README](../../../../README.md)
"""

from __future__ import annotations

import pytest

from xkep_cae.core.strategies.protocols import Fiber1DMaterialStrategy
from xkep_cae.elements.fiber.materials import (
    BilinearKinematicHardening1D,
    Elastic1D,
)
from xkep_cae.elements.fiber.state import Fiber1DState, SectionState


class TestFiber1DMaterialAPI:
    """API 準拠テスト — Protocol 検査・型検査・frozen 性."""

    def test_elastic1d_protocol_compliance(self) -> None:
        """Elastic1D は Fiber1DMaterialStrategy Protocol を満たす."""
        mat = Elastic1D(E=100_000.0)
        assert isinstance(mat, Fiber1DMaterialStrategy)

    def test_bilinear_kh_protocol_compliance(self) -> None:
        """BilinearKinematicHardening1D は Fiber1DMaterialStrategy Protocol を満たす."""
        mat = BilinearKinematicHardening1D(E=100_000.0, sigma_y=300.0, H=10_000.0)
        assert isinstance(mat, Fiber1DMaterialStrategy)

    def test_evaluate_return_types(self) -> None:
        """evaluate() は (float, float, Fiber1DState) を返す."""
        mat = BilinearKinematicHardening1D(E=100_000.0, sigma_y=300.0, H=10_000.0)
        state = Fiber1DState()
        sigma, E_t, new_state = mat.evaluate(0.001, state)
        assert isinstance(sigma, float)
        assert isinstance(E_t, float)
        assert isinstance(new_state, Fiber1DState)

    def test_state_frozen(self) -> None:
        """Fiber1DState は frozen dataclass（mutation 不可）."""
        state = Fiber1DState(eps_p=0.1, alpha=50.0)
        with pytest.raises(AttributeError):
            state.eps_p = 0.2  # type: ignore[misc]

    def test_section_state_frozen(self) -> None:
        """SectionState は frozen dataclass."""
        fibers = (Fiber1DState(), Fiber1DState(eps_p=0.01))
        sec = SectionState(fibers=fibers)
        with pytest.raises(AttributeError):
            sec.fibers = ()  # type: ignore[misc]
        assert len(sec.fibers) == 2

    def test_elastic_monotonic_sigma_eq_E_eps(self) -> None:
        """弾性単調載荷で σ = E·ε を再現."""
        E = 130_000.0
        mat = Elastic1D(E=E)
        state = Fiber1DState()
        for eps in [0.0, 0.001, 0.005, 0.01]:
            sigma, E_t, new_state = mat.evaluate(eps, state)
            assert sigma == pytest.approx(E * eps, abs=1e-10)
            assert E_t == pytest.approx(E, abs=1e-10)
            assert new_state is state  # 状態不変
