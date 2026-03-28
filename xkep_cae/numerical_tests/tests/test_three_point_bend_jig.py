"""ThreePointBendJigProcess の @binds_to 紐付けテスト.

C3 契約: 全 concrete Process に対し @binds_to テストが必須。

[← README](../../../README.md)
"""

from __future__ import annotations

import pytest

from xkep_cae.core.testing import binds_to
from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
    ThreePointBendContactJigConfig,
    ThreePointBendContactJigProcess,
    ThreePointBendJigConfig,
    ThreePointBendJigProcess,
)


@binds_to(ThreePointBendJigProcess)
class TestThreePointBendJigProcessAPI:
    """ThreePointBendJigProcess の基本動作確認."""

    @pytest.mark.skip(reason="status-222: 準静的接触ソルバー削除。動的版のみ対応。")
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
        cfg = ThreePointBendContactJigConfig(jig_push=0.01)
        proc = ThreePointBendContactJigProcess()
        result = proc.process(cfg)
        assert result.solver_result.converged
        assert result.wire_midpoint_deflection > 0


@binds_to(DynamicThreePointBendContactJigProcess)
class TestDynamicThreePointBendContactJigProcessAPI:
    """DynamicThreePointBendContactJigProcess の基本動作確認."""

    @pytest.mark.slow
    def test_process_runs(self):
        """動的接触ジグが収束し、変位が記録される."""
        cfg = DynamicThreePointBendContactJigConfig(
            jig_push=0.05,
            n_periods=2.0,
        )
        proc = DynamicThreePointBendContactJigProcess()
        result = proc.process(cfg)
        assert result.solver_result.converged
        assert result.wire_midpoint_deflection > 0


# --- huber_delta_h パイプライン貫通テスト（status-262）---


class TestThreePointBendHuberDeltaH:
    """huber_delta_h パイプライン貫通の単体テスト."""

    def test_contact_jig_config_default_zero(self):
        """ThreePointBendContactJigConfig の huber_delta_h デフォルトは 0."""
        cfg = ThreePointBendContactJigConfig()
        assert cfg.huber_delta_h == 0.0

    def test_dynamic_config_default_zero(self):
        """DynamicThreePointBendContactJigConfig の huber_delta_h デフォルトは 0."""
        cfg = DynamicThreePointBendContactJigConfig()
        assert cfg.huber_delta_h == 0.0

    def test_contact_jig_config_manual(self):
        """ThreePointBendContactJigConfig に huber_delta_h を手動指定できる."""
        cfg = ThreePointBendContactJigConfig(huber_delta_h=0.02)
        assert cfg.huber_delta_h == 0.02

    def test_dynamic_config_manual(self):
        """DynamicThreePointBendContactJigConfig に huber_delta_h を手動指定できる."""
        cfg = DynamicThreePointBendContactJigConfig(huber_delta_h=0.03)
        assert cfg.huber_delta_h == 0.03
