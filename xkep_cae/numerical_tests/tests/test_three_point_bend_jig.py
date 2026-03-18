"""ThreePointBendJigProcess の @binds_to 紐付けテスト.

C3 契約: 全 concrete Process に対し @binds_to テストが必須。

[← README](../../../README.md)
"""

from __future__ import annotations

from xkep_cae.core.testing import binds_to
from xkep_cae.numerical_tests.three_point_bend_jig import (
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
