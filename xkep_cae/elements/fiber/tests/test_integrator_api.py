"""FiberSectionIntegratorProcess API テスト.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F3
[← README](../../../../README.md)
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.core import binds_to
from xkep_cae.elements.fiber.integrator import (
    FiberIntegratorConfig,
    FiberIntegratorResult,
    FiberSectionIntegratorProcess,
)
from xkep_cae.elements.fiber.materials import Elastic1D
from xkep_cae.elements.fiber.section import CircularFiberSection
from xkep_cae.elements.fiber.state import Fiber1DState, SectionState


def _make_elastic_config(
    d: float = 17.0,
    n_fiber: int = 60,
    E: float = 200_000.0,
    eps0: float = 0.0,
    kappa_y: float = 0.001,
    kappa_z: float = 0.0,
) -> FiberIntegratorConfig:
    """弾性材料でのテスト用 Config を生成."""
    sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
    mat = Elastic1D(E=E)
    state = SectionState(
        fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
    )
    return FiberIntegratorConfig(
        eps0=eps0,
        kappa_y=kappa_y,
        kappa_z=kappa_z,
        section=sec,
        material=mat,
        state=state,
    )


@binds_to(FiberSectionIntegratorProcess)
class TestFiberSectionIntegratorAPI:
    """FiberSectionIntegratorProcess の API 仕様テスト."""

    def test_returns_result_type(self) -> None:
        """process() が FiberIntegratorResult を返す."""
        proc = FiberSectionIntegratorProcess()
        cfg = _make_elastic_config()
        result = proc.process(cfg)
        assert isinstance(result, FiberIntegratorResult)

    def test_result_has_expected_fields(self) -> None:
        """結果に N, M_y, M_z, C_section, state_new がある."""
        proc = FiberSectionIntegratorProcess()
        cfg = _make_elastic_config()
        result = proc.process(cfg)
        assert isinstance(result.N, float)
        assert isinstance(result.M_y, float)
        assert isinstance(result.M_z, float)
        assert isinstance(result.C_section, np.ndarray)
        assert isinstance(result.state_new, SectionState)

    def test_c_section_shape(self) -> None:
        """C_section は 3x3 対称行列."""
        proc = FiberSectionIntegratorProcess()
        cfg = _make_elastic_config()
        result = proc.process(cfg)
        assert result.C_section.shape == (3, 3)
        np.testing.assert_allclose(
            result.C_section,
            result.C_section.T,
            atol=1e-10,
        )

    def test_state_new_length(self) -> None:
        """新状態のファイバー数が入力断面と一致."""
        proc = FiberSectionIntegratorProcess()
        n = 40
        cfg = _make_elastic_config(n_fiber=n)
        result = proc.process(cfg)
        assert len(result.state_new.fibers) == n

    def test_frozen_result(self) -> None:
        """FiberIntegratorResult は frozen dataclass."""
        proc = FiberSectionIntegratorProcess()
        cfg = _make_elastic_config()
        result = proc.process(cfg)
        with pytest.raises(AttributeError):
            result.N = 0.0  # type: ignore[misc]

    def test_zero_strain_gives_zero_forces(self) -> None:
        """ひずみ・曲率ゼロで断面力がゼロ."""
        proc = FiberSectionIntegratorProcess()
        cfg = _make_elastic_config(eps0=0.0, kappa_y=0.0, kappa_z=0.0)
        result = proc.process(cfg)
        assert abs(result.N) < 1e-10
        assert abs(result.M_y) < 1e-10
        assert abs(result.M_z) < 1e-10
