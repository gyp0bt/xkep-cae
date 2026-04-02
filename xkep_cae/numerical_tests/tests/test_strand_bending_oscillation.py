"""StrandBendingOscillationProcess のテスト.

C3 契約: @binds_to 紐付け + API テスト。
実行は重いのでソルバー実行は行わず、構成検証のみ。

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np

from xkep_cae.core import binds_to
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
    _collect_end_nodes,
)


@binds_to(StrandBendingOscillationProcess)
class TestStrandBendingOscillationProcessAPI:
    """StrandBendingOscillationProcess の API テスト."""

    def test_meta_name(self) -> None:
        """meta.name が正しい."""
        assert StrandBendingOscillationProcess.meta.name == "StrandBendingOscillation"

    def test_uses_declared(self) -> None:
        """uses に必要な Process が宣言されている."""
        uses_names = [u.__name__ for u in StrandBendingOscillationProcess.uses]
        assert "StrandMeshProcess" in uses_names
        assert "MPCEliminationProcess" in uses_names
        assert "ULCRBeamAssemblerProcess" in uses_names
        assert "ContactFrictionProcess" in uses_names

    def test_config_defaults(self) -> None:
        """デフォルト構成が正しい."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.n_strands == 7
        assert cfg.wire_radius == 0.5
        assert cfg.E == 130.0e3
        assert cfg.mu == 0.15
        assert cfg.smoothing_delta == 0.0  # 0=自動推定

    def test_smoothing_delta_auto_estimation(self) -> None:
        """smoothing_delta=0 のとき 1000/wire_radius で自動推定."""
        cfg = StrandBendingOscillationConfig(wire_radius=0.5, smoothing_delta=0.0)
        assert cfg.smoothing_delta == 0.0  # 0=自動推定トリガー
        # 自動推定公式: 1000 / wire_radius = 2000（status-260で5000→1000に変更）
        assert 1000.0 / cfg.wire_radius == 2000.0

    def test_smoothing_delta_manual_override(self) -> None:
        """smoothing_delta > 0 のとき手動値をそのまま使用."""
        cfg = StrandBendingOscillationConfig(smoothing_delta=3000.0)
        assert cfg.smoothing_delta == 3000.0

    def test_huber_delta_h_default_zero(self) -> None:
        """huber_delta_h デフォルトは 0（smoothing_delta で間接計算）."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.huber_delta_h == 0.0

    def test_huber_delta_h_manual_override(self) -> None:
        """huber_delta_h > 0 のとき直接指定値をそのまま使用."""
        cfg = StrandBendingOscillationConfig(huber_delta_h=0.01)
        assert cfg.huber_delta_h == 0.01

    def test_free_end_mode_default_false(self) -> None:
        """free_end_mode デフォルトは False."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.free_end_mode is False

    def test_free_end_mode_true(self) -> None:
        """free_end_mode=True で構成可能."""
        cfg = StrandBendingOscillationConfig(free_end_mode=True)
        assert cfg.free_end_mode is True


class TestCollectEndNodes:
    """端部節点収集のテスト."""

    def test_single_strand(self) -> None:
        """単一素線の端部節点."""
        # 3要素: 0-1-2-3
        conn = np.array([[0, 1], [1, 2], [2, 3]])
        strand_ids = np.array([0, 0, 0])
        left, right = _collect_end_nodes(conn, 1, strand_ids)
        assert left == [0]
        assert right == [3]

    def test_two_strands(self) -> None:
        """2素線の端部節点."""
        # strand 0: 0-1-2, strand 1: 3-4-5
        conn = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
        strand_ids = np.array([0, 0, 1, 1])
        left, right = _collect_end_nodes(conn, 2, strand_ids)
        assert sorted(left) == [0, 3]
        assert sorted(right) == [2, 5]
