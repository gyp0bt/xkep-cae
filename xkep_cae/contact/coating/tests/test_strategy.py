"""CoatingStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + CoatingStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.coating.strategy import (
    KelvinVoigtCoatingProcess,
    NoCoatingProcess,
    _create_coating_strategy,
)
from xkep_cae.core import SolverProcess
from xkep_cae.core.strategies import CoatingStrategy
from xkep_cae.core.testing import binds_to

# ── NoCoatingProcess ────────────────────────────────────────


@binds_to(NoCoatingProcess)
class TestNoCoatingProcess:
    """NoCoatingProcess の単体テスト."""

    def test_protocol_conformance(self) -> None:
        p = NoCoatingProcess()
        assert isinstance(p, CoatingStrategy)

    def test_is_solver_process(self) -> None:
        p = NoCoatingProcess()
        assert isinstance(p, SolverProcess)

    def test_forces_zero(self) -> None:
        p = NoCoatingProcess()
        coords = np.zeros((4, 3))
        f = p.forces([], coords, type("C", (), {"ndof_per_node": 6})(), 0.01)
        assert f.shape == (24,)
        assert np.allclose(f, 0.0)

    def test_stiffness_zero(self) -> None:
        p = NoCoatingProcess()
        K = p.stiffness([], np.zeros((4, 3)), type("C", (), {"ndof_per_node": 6})(), 24, 0.01)
        assert K.shape == (24, 24)
        assert K.nnz == 0

    def test_friction_forces_zero(self) -> None:
        p = NoCoatingProcess()
        u = np.zeros(24)
        f = p.friction_forces([], np.zeros((4, 3)), type("C", (), {"ndof_per_node": 6})(), u, u)
        assert np.allclose(f, 0.0)

    def test_friction_stiffness_zero(self) -> None:
        p = NoCoatingProcess()
        K = p.friction_stiffness([], np.zeros((4, 3)), type("C", (), {"ndof_per_node": 6})(), 24)
        assert K.shape == (24, 24)
        assert K.nnz == 0


# ── KelvinVoigtCoatingProcess ──────────────────────────────


@binds_to(KelvinVoigtCoatingProcess)
class TestKelvinVoigtCoatingProcess:
    """KelvinVoigtCoatingProcess の単体テスト."""

    def test_protocol_conformance(self) -> None:
        p = KelvinVoigtCoatingProcess()
        assert isinstance(p, CoatingStrategy)

    def test_is_solver_process(self) -> None:
        p = KelvinVoigtCoatingProcess()
        assert isinstance(p, SolverProcess)

    def test_meta(self) -> None:
        assert KelvinVoigtCoatingProcess.meta.name == "KelvinVoigtCoating"
        assert KelvinVoigtCoatingProcess.meta.version == "1.0.0"


# ── ファクトリ関数テスト ──────────────────────────────────


class TestCreateCoatingStrategy:
    """_create_coating_strategy ファクトリのテスト."""

    def test_zero_stiffness_returns_no_coating(self) -> None:
        s = _create_coating_strategy(coating_stiffness=0.0)
        assert isinstance(s, NoCoatingProcess)

    def test_negative_stiffness_returns_no_coating(self) -> None:
        s = _create_coating_strategy(coating_stiffness=-1.0)
        assert isinstance(s, NoCoatingProcess)

    def test_positive_stiffness_returns_kelvin_voigt(self) -> None:
        s = _create_coating_strategy(coating_stiffness=1e6)
        assert isinstance(s, KelvinVoigtCoatingProcess)
