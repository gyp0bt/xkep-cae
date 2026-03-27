"""StaticBeamTestProcess / DynamicBeamTestProcess のテスト.

status-251: STA2 防止 H1-H5 のテスト紐付け。

[← README](../../../README.md)
"""

from __future__ import annotations

from xkep_cae.core import binds_to
from xkep_cae.numerical_tests.core import (
    DynamicTestConfig,
    NumericalTestConfig,
)
from xkep_cae.numerical_tests.dynamic_runner import DynamicBeamTestProcess
from xkep_cae.numerical_tests.runner import StaticBeamTestProcess


@binds_to(StaticBeamTestProcess)
class TestStaticBeamTestProcessAPI:
    """StaticBeamTestProcess の API テスト."""

    def test_bend3p_converges(self) -> None:
        """3点曲げが収束する."""
        cfg = NumericalTestConfig(
            name="bend3p",
            beam_type="timo2d",
            section_shape="circle",
            section_params={"d": 10.0},
            E=200e3,
            nu=0.3,
            length=100.0,
            n_elems=16,
            load_value=-1.0,
        )
        result = StaticBeamTestProcess().process(cfg)
        assert result.relative_error is not None
        assert result.relative_error < 0.02

    def test_tensile_converges(self) -> None:
        """引張試験が収束する."""
        cfg = NumericalTestConfig(
            name="tensile",
            beam_type="timo2d",
            section_shape="circle",
            section_params={"d": 10.0},
            E=200e3,
            nu=0.3,
            length=100.0,
            n_elems=16,
            load_value=1000.0,
        )
        result = StaticBeamTestProcess().process(cfg)
        assert result.relative_error is not None
        assert result.relative_error < 1e-10


@binds_to(DynamicBeamTestProcess)
class TestDynamicBeamTestProcessAPI:
    """DynamicBeamTestProcess の API テスト."""

    def test_dynamic_bend3p_runs(self) -> None:
        """動的3点曲げが実行できる."""
        cfg = DynamicTestConfig(
            name="dynamic_bend3p",
            beam_type="timo3d",
            section_shape="circle",
            section_params={"d": 10.0},
            E=200e3,
            nu=0.3,
            rho=7.85e-9,
            length=100.0,
            n_elems=16,
            load_value=-1.0,
            n_steps=5,
            dt=1e-3,
        )
        result = DynamicBeamTestProcess().process(cfg)
        assert result.displacement is not None
        assert len(result.displacement) > 0
