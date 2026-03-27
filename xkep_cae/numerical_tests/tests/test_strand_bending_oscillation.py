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
