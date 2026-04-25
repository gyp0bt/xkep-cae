"""PairwiseFreezingProcess ユニットテスト (status-374 Phase 1).

[← freeze README](../__init__.py) | [← project README](../../../../README.md)

per-pair 凍結判定ロジックの単体検証。Phase 2 NR ループ配線は別 status で
別途結合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.freeze import (
    PairwiseFreezingInput,
    PairwiseFreezingOutput,
    PairwiseFreezingProcess,
)
from xkep_cae.contact.freeze.strategy import (
    _is_type_d_dominant,
    _update_pair_active_flips,
)
from xkep_cae.core.testing import binds_to


@binds_to(PairwiseFreezingProcess)
class TestPairwiseFreezingProcessAPI:
    """Process API 契約（no-op / shape ガード / Output 型）の検証."""

    def test_zero_pairs_returns_empty(self) -> None:
        """n_pairs=0 で全フィールド空、freeze_flags shape (0,) bool."""
        inp = PairwiseFreezingInput(
            n_pairs=0,
            pair_active_flip_counts=np.zeros(0, dtype=int),
            is_active_now=np.zeros(0, dtype=bool),
        )
        out = PairwiseFreezingProcess().process(inp)
        assert isinstance(out, PairwiseFreezingOutput)
        assert out.pair_freeze_flags.shape == (0,)
        assert out.pair_freeze_flags.dtype == np.bool_
        assert out.n_frozen == 0
        assert out.n_active_pairs == 0
        assert out.skip_freeze_global is False
        assert out.freeze_reasons == ()

    def test_shape_mismatch_raises(self) -> None:
        """flip_counts / is_active が n_pairs と整合しない場合 ValueError."""
        inp = PairwiseFreezingInput(
            n_pairs=3,
            pair_active_flip_counts=np.zeros(2, dtype=int),  # mismatch
            is_active_now=np.zeros(3, dtype=bool),
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            PairwiseFreezingProcess().process(inp)

    def test_output_types_consistent(self) -> None:
        """Output の dtype と len が n_pairs に揃う."""
        inp = PairwiseFreezingInput(
            n_pairs=4,
            pair_active_flip_counts=np.array([0, 1, 5, 0], dtype=int),
            is_active_now=np.array([True, True, True, False], dtype=bool),
        )
        out = PairwiseFreezingProcess().process(inp)
        assert out.pair_freeze_flags.shape == (4,)
        assert out.pair_freeze_flags.dtype == np.bool_
        assert len(out.freeze_reasons) == 4
        assert all(isinstance(r, str) for r in out.freeze_reasons)


class TestPairwiseFreezingProcessLogic:
    """判定ロジック（threshold / inactive / Type D skip）の検証."""

    def test_below_threshold_no_freeze(self) -> None:
        """flip_counts < threshold の active ペアは freeze=False."""
        inp = PairwiseFreezingInput(
            n_pairs=3,
            pair_active_flip_counts=np.array([0, 1, 2], dtype=int),
            is_active_now=np.array([True, True, True], dtype=bool),
            flip_threshold=3,
        )
        out = PairwiseFreezingProcess().process(inp)
        assert out.n_frozen == 0
        assert not out.pair_freeze_flags.any()
        assert all(r == "below_threshold" for r in out.freeze_reasons)
        assert out.skip_freeze_global is False
        assert out.n_active_pairs == 3

    def test_above_threshold_freezes_active_only(self) -> None:
        """flip_counts >= threshold ∧ is_active で freeze=True."""
        inp = PairwiseFreezingInput(
            n_pairs=4,
            pair_active_flip_counts=np.array([5, 4, 0, 10], dtype=int),
            is_active_now=np.array([True, True, True, True], dtype=bool),
            flip_threshold=3,
        )
        out = PairwiseFreezingProcess().process(inp)
        assert out.n_frozen == 3
        assert out.pair_freeze_flags.tolist() == [True, True, False, True]
        assert out.freeze_reasons[0] == "freeze_flips=5"
        assert out.freeze_reasons[1] == "freeze_flips=4"
        assert out.freeze_reasons[2] == "below_threshold"
        assert out.freeze_reasons[3] == "freeze_flips=10"

    def test_inactive_pair_skipped_even_if_high_flip(self) -> None:
        """is_active_now=False のペアは flip_counts に関係なく freeze=False."""
        inp = PairwiseFreezingInput(
            n_pairs=3,
            pair_active_flip_counts=np.array([10, 10, 10], dtype=int),
            is_active_now=np.array([True, False, True], dtype=bool),
            flip_threshold=3,
        )
        out = PairwiseFreezingProcess().process(inp)
        assert out.pair_freeze_flags.tolist() == [True, False, True]
        assert out.freeze_reasons[1] == "inactive"
        assert out.n_frozen == 2
        assert out.n_active_pairs == 2

    def test_threshold_boundary_inclusive(self) -> None:
        """flip_counts == threshold で freeze=True（>= 判定）."""
        inp = PairwiseFreezingInput(
            n_pairs=2,
            pair_active_flip_counts=np.array([3, 2], dtype=int),
            is_active_now=np.array([True, True], dtype=bool),
            flip_threshold=3,
        )
        out = PairwiseFreezingProcess().process(inp)
        assert out.pair_freeze_flags.tolist() == [True, False]
        assert out.freeze_reasons[0] == "freeze_flips=3"
        assert out.freeze_reasons[1] == "below_threshold"

    def test_type_d_dominant_skips_freeze(self) -> None:
        """chattering_type='D' 単独で skip_when_type_d_dominant=True なら全 skip."""
        inp = PairwiseFreezingInput(
            n_pairs=3,
            pair_active_flip_counts=np.array([10, 10, 10], dtype=int),
            is_active_now=np.array([True, True, True], dtype=bool),
            flip_threshold=3,
            chattering_type="D",
            skip_when_type_d_dominant=True,
        )
        out = PairwiseFreezingProcess().process(inp)
        assert out.skip_freeze_global is True
        assert not out.pair_freeze_flags.any()
        assert out.n_frozen == 0
        assert out.n_active_pairs == 3
        assert all(r == "skip_type_d" for r in out.freeze_reasons)

    def test_type_d_plus_e_does_not_skip(self) -> None:
        """chattering_type='D+E' は単独 D ではないので freeze 実行."""
        inp = PairwiseFreezingInput(
            n_pairs=2,
            pair_active_flip_counts=np.array([5, 0], dtype=int),
            is_active_now=np.array([True, True], dtype=bool),
            flip_threshold=3,
            chattering_type="D+E",
            skip_when_type_d_dominant=True,
        )
        out = PairwiseFreezingProcess().process(inp)
        assert out.skip_freeze_global is False
        assert out.pair_freeze_flags.tolist() == [True, False]


class TestPairwiseFreezingHelpers:
    """ヘルパ純関数の検証."""

    def test_update_flip_counts_increments_on_change(self) -> None:
        """active 状態が変わったペアのみ flip_counts が +1."""
        prev = np.array([0, 2, 5, 1], dtype=int)
        is_now = np.array([True, False, True, True], dtype=bool)
        is_prev = np.array([True, True, False, True], dtype=bool)
        new = _update_pair_active_flips(prev, is_now, is_prev)
        # idx 0: same (True→True), idx 1: changed, idx 2: changed, idx 3: same
        assert new.tolist() == [0, 3, 6, 1]

    def test_update_flip_counts_shape_mismatch_safe(self) -> None:
        """shape 不整合時は prev をそのまま返す（NR ループ側の起動時保護）."""
        prev = np.array([0, 1, 2], dtype=int)
        is_now = np.array([True, False], dtype=bool)  # mismatch
        is_prev = np.array([True, False], dtype=bool)
        new = _update_pair_active_flips(prev, is_now, is_prev)
        assert new is prev

    def test_is_type_d_dominant_truth_table(self) -> None:
        """Type D 単独支配判定の真理値表."""
        # D 単独 → True
        assert _is_type_d_dominant("D") is True
        # 空文字 → False
        assert _is_type_d_dominant("") is False
        # D を含むが他 type もある → False
        assert _is_type_d_dominant("D+E") is False
        assert _is_type_d_dominant("A+D") is False
        assert _is_type_d_dominant("A+B+D+E") is False
        # D 含まない → False
        assert _is_type_d_dominant("A") is False
        assert _is_type_d_dominant("E") is False
        assert _is_type_d_dominant("A+B+E") is False
