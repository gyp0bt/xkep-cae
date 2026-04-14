"""ContactPairLayerClassifierProcess の単体テスト.

status-340: 接触ペアの層分類後処理 Process を、合成入力（kappa_cr_per_pair /
per_pair_dissipation + strand_ids + strand_layers）で純粋関数的に検証する。

[← README](../../README.md)
"""

from __future__ import annotations

import math

import pytest

from xkep_cae.core import binds_to
from xkep_cae.numerical_tests.contact_pair_layer_classifier import (
    ContactPairLayerClassifierInput,
    ContactPairLayerClassifierProcess,
    LayerPairStats,
)

# ====================================================================
# 合成データ
# ====================================================================
# 想定モデル: 19本撚線（1+6+12 構造）の縮約版。
#   strand_id 0    → layer 0（core）
#   strand_id 1-2  → layer 1（inner）
#   strand_id 3-4  → layer 2（outer）
#
#   要素は各 strand に 4 elem ずつ割り当て:
#     elem 0-3   : strand 0 (layer 0)
#     elem 4-7   : strand 1 (layer 1)
#     elem 8-11  : strand 2 (layer 1)
#     elem 12-15 : strand 3 (layer 2)
#     elem 16-19 : strand 4 (layer 2)
STRAND_IDS = tuple([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4)
STRAND_LAYERS = (0, 1, 1, 2, 2)


# ====================================================================
# テスト
# ====================================================================


@binds_to(ContactPairLayerClassifierProcess)
class TestContactPairLayerClassifierAPI:
    """合成入力による純粋関数的検証."""

    def test_empty_input(self) -> None:
        """空入力は全ゼロ結果を返す."""
        result = ContactPairLayerClassifierProcess().process(
            ContactPairLayerClassifierInput(
                kappa_cr_per_pair={},
                per_pair_dissipation={},
                strand_ids=STRAND_IDS,
                strand_layers=STRAND_LAYERS,
            )
        )
        assert result.n_unique_layer_pairs == 0
        assert result.pair_layer_keys == {}
        assert result.per_layer_pair_stats == {}

    def test_pair_layer_key_normalization(self) -> None:
        """(elem_a, elem_b) は (l_min, l_max) に正規化される（順序非依存）."""
        # elem 0 → strand 0 → layer 0
        # elem 12 → strand 3 → layer 2
        kappa = {(0, 12): 1.0e-3, (12, 0): 2.0e-3}
        result = ContactPairLayerClassifierProcess().process(
            ContactPairLayerClassifierInput(
                kappa_cr_per_pair=kappa,
                per_pair_dissipation={},
                strand_ids=STRAND_IDS,
                strand_layers=STRAND_LAYERS,
            )
        )
        # 両ペアとも (0, 2) に分類される
        assert result.pair_layer_keys[(0, 12)] == (0, 2)
        assert result.pair_layer_keys[(12, 0)] == (0, 2)
        assert result.n_unique_layer_pairs == 1
        stats = result.per_layer_pair_stats[(0, 2)]
        assert stats.n_pairs == 2
        assert stats.n_slipped == 2
        assert stats.kappa_cr_mean == pytest.approx(1.5e-3)

    def test_inner_inner_vs_outer_outer_separation(self) -> None:
        """内層対 (1,1) と外層対 (2,2) が分離される（バイモーダル仮説の基盤）."""
        # 内層対: elem 4 (layer 1) ↔ elem 8 (layer 1) — 異 strand
        # 外層対: elem 12 (layer 2) ↔ elem 16 (layer 2) — 異 strand
        kappa = {
            (4, 8): 5.0e-3,
            (5, 9): 5.5e-3,
            (12, 16): 3.0e-3,
            (13, 17): 3.2e-3,
        }
        result = ContactPairLayerClassifierProcess().process(
            ContactPairLayerClassifierInput(
                kappa_cr_per_pair=kappa,
                per_pair_dissipation={},
                strand_ids=STRAND_IDS,
                strand_layers=STRAND_LAYERS,
            )
        )
        assert (1, 1) in result.per_layer_pair_stats
        assert (2, 2) in result.per_layer_pair_stats
        inner = result.per_layer_pair_stats[(1, 1)]
        outer = result.per_layer_pair_stats[(2, 2)]
        assert inner.n_pairs == 2
        assert outer.n_pairs == 2
        assert inner.kappa_cr_mean == pytest.approx(5.25e-3)
        assert outer.kappa_cr_mean == pytest.approx(3.1e-3)
        # 内外層で平均値が分離されている（バイモーダル）
        assert inner.kappa_cr_mean > outer.kappa_cr_mean

    def test_mixed_layer_pair(self) -> None:
        """層跨ぎ対 (1,2) が独立カテゴリとして抽出される."""
        # elem 4 (layer 1) ↔ elem 12 (layer 2)
        kappa = {(4, 12): 4.0e-3, (8, 16): 4.5e-3}
        result = ContactPairLayerClassifierProcess().process(
            ContactPairLayerClassifierInput(
                kappa_cr_per_pair=kappa,
                per_pair_dissipation={},
                strand_ids=STRAND_IDS,
                strand_layers=STRAND_LAYERS,
            )
        )
        assert result.pair_layer_keys[(4, 12)] == (1, 2)
        assert result.pair_layer_keys[(8, 16)] == (1, 2)
        stats = result.per_layer_pair_stats[(1, 2)]
        assert stats.n_pairs == 2
        assert stats.kappa_cr_mean == pytest.approx(4.25e-3)

    def test_dissipation_aggregation(self) -> None:
        """per_pair_dissipation が層ペアで合計・平均される."""
        # 3 ペアとも layer (1, 1)
        diss = {(4, 8): 1.0e-7, (4, 9): 2.0e-7, (5, 8): 3.0e-7}
        result = ContactPairLayerClassifierProcess().process(
            ContactPairLayerClassifierInput(
                kappa_cr_per_pair={},
                per_pair_dissipation=diss,
                strand_ids=STRAND_IDS,
                strand_layers=STRAND_LAYERS,
            )
        )
        stats = result.per_layer_pair_stats[(1, 1)]
        assert stats.n_pairs == 3
        assert stats.n_slipped == 0
        assert stats.dissipation_sum == pytest.approx(6.0e-7)
        assert stats.dissipation_mean == pytest.approx(2.0e-7)

    def test_kappa_and_dissipation_union(self) -> None:
        """kappa_cr と dissipation の和集合が n_pairs にカウントされる."""
        # ペア (4, 8): kappa あり、dissipation あり
        # ペア (5, 9): kappa あり、dissipation なし
        # ペア (6, 10): kappa なし、dissipation あり
        kappa = {(4, 8): 5.0e-3, (5, 9): 5.5e-3}
        diss = {(4, 8): 1.0e-7, (6, 10): 2.0e-7}
        result = ContactPairLayerClassifierProcess().process(
            ContactPairLayerClassifierInput(
                kappa_cr_per_pair=kappa,
                per_pair_dissipation=diss,
                strand_ids=STRAND_IDS,
                strand_layers=STRAND_LAYERS,
            )
        )
        stats = result.per_layer_pair_stats[(1, 1)]
        assert stats.n_pairs == 3
        assert stats.n_slipped == 2
        assert stats.dissipation_sum == pytest.approx(3.0e-7)

    def test_std_computation(self) -> None:
        """kappa_cr_std は母集団標準偏差（n で割る）."""
        kappa = {(4, 8): 1.0e-3, (5, 9): 3.0e-3, (6, 10): 5.0e-3}
        result = ContactPairLayerClassifierProcess().process(
            ContactPairLayerClassifierInput(
                kappa_cr_per_pair=kappa,
                per_pair_dissipation={},
                strand_ids=STRAND_IDS,
                strand_layers=STRAND_LAYERS,
            )
        )
        stats = result.per_layer_pair_stats[(1, 1)]
        assert stats.kappa_cr_mean == pytest.approx(3.0e-3)
        # std = sqrt(((1-3)^2 + 0 + (5-3)^2) / 3) * 1e-3 = sqrt(8/3) * 1e-3
        expected_std = math.sqrt(8.0 / 3.0) * 1e-3
        assert stats.kappa_cr_std == pytest.approx(expected_std)
        assert stats.kappa_cr_min == pytest.approx(1.0e-3)
        assert stats.kappa_cr_max == pytest.approx(5.0e-3)

    def test_default_layer_pair_stats(self) -> None:
        """LayerPairStats のデフォルトは全ゼロ."""
        stats = LayerPairStats()
        assert stats.n_pairs == 0
        assert stats.n_slipped == 0
        assert stats.kappa_cr_mean == 0.0
        assert stats.dissipation_sum == 0.0
