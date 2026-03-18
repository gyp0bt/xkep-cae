"""ThreePointBendJigProcess の @binds_to 紐付けテスト.

C3 契約: 全 concrete Process に対し @binds_to テストが必須。

[← README](../../../README.md)
"""

from __future__ import annotations

import pytest

from xkep_cae.core.testing import binds_to
from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendJigConfig,
    DynamicThreePointBendJigProcess,
    ThreePointBendContactJigConfig,
    ThreePointBendContactJigProcess,
    ThreePointBendJigConfig,
    ThreePointBendJigProcess,
)


@binds_to(ThreePointBendJigProcess)
class TestThreePointBendJigProcessAPI:
    """ThreePointBendJigProcess の基本動作確認."""

    def test_process_runs(self):
        """小変位で ProcessMeta + 入出力契約を満たす."""
        cfg = ThreePointBendJigConfig(jig_push=0.05)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)
        assert result.solver_result.converged
        assert result.wire_midpoint_deflection > 0


@binds_to(ThreePointBendContactJigProcess)
class TestThreePointBendContactJigProcessAPI:
    """ThreePointBendContactJigProcess の基本動作確認."""

    @pytest.mark.xfail(reason="HEX8接触ジグのNR収束問題（status-210 TODO）")
    def test_process_runs(self):
        """小変位で ProcessMeta + 入出力契約を満たす."""
        cfg = ThreePointBendContactJigConfig(jig_push=0.01, n_uzawa_max=10)
        proc = ThreePointBendContactJigProcess()
        result = proc.process(cfg)
        assert result.solver_result.converged
        assert result.wire_midpoint_deflection > 0


@binds_to(DynamicThreePointBendJigProcess)
class TestDynamicThreePointBendJigProcessAPI:
    """DynamicThreePointBendJigProcess の基本動作確認."""

    def test_process_runs(self):
        """動的解析が収束し、変位が記録される."""
        cfg = DynamicThreePointBendJigConfig(
            jig_push=0.05,
            n_steps=50,
            n_periods=1.0,
        )
        proc = DynamicThreePointBendJigProcess()
        result = proc.process(cfg)
        assert result.solver_result.converged
        assert result.wire_midpoint_deflection > 0
        assert len(result.deflection_history) > 0
        assert result.analytical_frequency_hz > 0
