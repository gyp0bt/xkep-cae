"""ContactPairLayerClassifierProcess — 接触ペアの層分類後処理 Process.

`ContactPairAnalysisProcess`（status-337）が抽出する `kappa_cr_per_pair` /
`per_pair_dissipation` を、撚線層構造（core / inner / outer …）の組み合わせで
バケット化する後処理 Process。

status-339 で観測された 19本撚線 κ_cr 分布のバイモーダル気配
（3.2e-3 付近と 5.0-5.3e-3 付近のダブルピーク）を、内層対 / 外層対 / 層跨ぎ対の
3 カテゴリに切り分けて検証するために導入する。

物理的背景:
    撚線の層によって接触力・梃子比が異なるため、同一の曲げ曲率に対して
    各層ペアで κ_cr が分岐する仮説が status-339 で提示されている。
    例: 19本撚線（1+6+12）の場合、(layer_min, layer_max) は
        - (0, 1): core ↔ inner（中心 vs 内層）
        - (0, 2): core ↔ outer（中心 vs 外層、通常成立しない）
        - (1, 1): inner ↔ inner（内層対）
        - (1, 2): inner ↔ outer（層跨ぎ対）
        - (2, 2): outer ↔ outer（外層対）
    の 5 種類に分類される（exclude_same_strand=True 前提）。

status-340: 接触ペア層分類 PostProcess 化。

[← README](../../../README.md)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from xkep_cae.core import PostProcess, ProcessMeta

# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class LayerPairStats:
    """1 つの層ペア (l_min, l_max) の統計量.

    Attributes:
        n_pairs: 該当ペア数（kappa_cr または dissipation のうち多い方）
        n_slipped: κ_cr が記録されたペア数
        kappa_cr_mean / std / min / max: κ_cr 分布統計（n_slipped > 0 のときのみ有意）
        dissipation_sum: per_pair_dissipation 合計
        dissipation_mean: per_pair_dissipation 平均（n_pairs > 0 のときのみ有意）
    """

    n_pairs: int = 0
    n_slipped: int = 0
    kappa_cr_mean: float = 0.0
    kappa_cr_std: float = 0.0
    kappa_cr_min: float = 0.0
    kappa_cr_max: float = 0.0
    dissipation_sum: float = 0.0
    dissipation_mean: float = 0.0


@dataclass(frozen=True)
class ContactPairLayerClassifierInput:
    """接触ペア層分類入力.

    Attributes:
        kappa_cr_per_pair: (elem_a, elem_b) → κ_cr。
            `ContactPairAnalysisResult.kappa_cr_per_pair` を直接渡す想定。
        per_pair_dissipation: (elem_a, elem_b) → 最終散逸エネルギー。
            `ContactPairAnalysisResult.per_pair_dissipation` を直接渡す想定。
        strand_ids: elem_id → strand_id 配列。`MeshData.strand_ids` を渡す。
        strand_layers: strand_id → layer 配列。`StrandInfoOutput.layer` から
            生成する（例: `tuple(info.layer for info in mesh.strand_infos)`)。
    """

    kappa_cr_per_pair: Mapping[tuple[int, int], float]
    per_pair_dissipation: Mapping[tuple[int, int], float]
    strand_ids: Sequence[int]
    strand_layers: Sequence[int]


@dataclass(frozen=True)
class ContactPairLayerClassifierResult:
    """接触ペア層分類結果.

    Attributes:
        pair_layer_keys: (elem_a, elem_b) → (l_min, l_max)。kappa_cr_per_pair と
            per_pair_dissipation の和集合に対して計算する。
        per_layer_pair_stats: (l_min, l_max) → LayerPairStats。
        n_unique_layer_pairs: per_layer_pair_stats のキー数。
    """

    pair_layer_keys: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    per_layer_pair_stats: dict[tuple[int, int], LayerPairStats] = field(default_factory=dict)
    n_unique_layer_pairs: int = 0


# ====================================================================
# 純粋ヘルパー関数
# ====================================================================


def _layer_pair_key(
    elem_a: int,
    elem_b: int,
    strand_ids: Sequence[int],
    strand_layers: Sequence[int],
) -> tuple[int, int]:
    """(elem_a, elem_b) を (min_layer, max_layer) に正規化する."""
    s_a = int(strand_ids[elem_a])
    s_b = int(strand_ids[elem_b])
    l_a = int(strand_layers[s_a])
    l_b = int(strand_layers[s_b])
    return (l_a, l_b) if l_a <= l_b else (l_b, l_a)


def _aggregate_stats(
    kappa_vals: list[float],
    dissipation_vals: list[float],
    n_pairs: int,
) -> LayerPairStats:
    """1 つの層ペアの統計量を集約."""
    n_slipped = len(kappa_vals)
    if n_slipped > 0:
        mean_k = sum(kappa_vals) / n_slipped
        var_k = sum((k - mean_k) ** 2 for k in kappa_vals) / n_slipped
        std_k = math.sqrt(var_k)
        min_k = min(kappa_vals)
        max_k = max(kappa_vals)
    else:
        mean_k = std_k = min_k = max_k = 0.0

    diss_sum = float(sum(dissipation_vals))
    diss_mean = diss_sum / n_pairs if n_pairs > 0 else 0.0

    return LayerPairStats(
        n_pairs=n_pairs,
        n_slipped=n_slipped,
        kappa_cr_mean=float(mean_k),
        kappa_cr_std=float(std_k),
        kappa_cr_min=float(min_k),
        kappa_cr_max=float(max_k),
        dissipation_sum=diss_sum,
        dissipation_mean=float(diss_mean),
    )


# ====================================================================
# Process
# ====================================================================


class ContactPairLayerClassifierProcess(
    PostProcess[ContactPairLayerClassifierInput, ContactPairLayerClassifierResult],
):
    """接触ペアを撚線層構造で分類する後処理 Process.

    `ContactPairAnalysisProcess` が抽出した (elem_a, elem_b) ペアを、
    `MeshData.strand_ids` + `StrandInfoOutput.layer` を介して
    層インデックスペア (l_min, l_max) に集約し、層ペア毎の κ_cr 分布統計と
    累積散逸を返す。

    使用例:
        >>> mesh = StrandMeshProcess().process(cfg).mesh
        >>> info_layers = tuple(info.layer for info in mesh_result_strand_infos)
        >>> classifier_result = ContactPairLayerClassifierProcess().process(
        ...     ContactPairLayerClassifierInput(
        ...         kappa_cr_per_pair=analysis.kappa_cr_per_pair,
        ...         per_pair_dissipation=analysis.per_pair_dissipation,
        ...         strand_ids=mesh.strand_ids.tolist(),
        ...         strand_layers=info_layers,
        ...     )
        ... )
        >>> for (l1, l2), stats in classifier_result.per_layer_pair_stats.items():
        ...     print(f"layer ({l1},{l2}): n={stats.n_pairs}, κ_cr={stats.kappa_cr_mean:.3e}")
    """

    meta = ProcessMeta(
        name="ContactPairLayerClassifier",
        module="post",
        version="1.0.0",
        document_path="docs/contact_pair_layer_classifier.md",
    )
    uses = ()

    def process(
        self, input_data: ContactPairLayerClassifierInput
    ) -> ContactPairLayerClassifierResult:
        """接触ペアを層分類し統計量を返す."""
        kappa = input_data.kappa_cr_per_pair
        diss = input_data.per_pair_dissipation
        strand_ids = input_data.strand_ids
        strand_layers = input_data.strand_layers

        # ペア → 層ペア辞書（kappa と diss の和集合で構築）
        pair_layer_keys: dict[tuple[int, int], tuple[int, int]] = {}
        bucket_kappa: dict[tuple[int, int], list[float]] = {}
        bucket_diss: dict[tuple[int, int], list[float]] = {}
        bucket_n_pairs: dict[tuple[int, int], int] = {}

        all_pairs: set[tuple[int, int]] = set(kappa.keys()) | set(diss.keys())
        for elem_a, elem_b in all_pairs:
            lp = _layer_pair_key(elem_a, elem_b, strand_ids, strand_layers)
            pair_layer_keys[(elem_a, elem_b)] = lp
            bucket_n_pairs[lp] = bucket_n_pairs.get(lp, 0) + 1
            if (elem_a, elem_b) in kappa:
                bucket_kappa.setdefault(lp, []).append(float(kappa[(elem_a, elem_b)]))
            if (elem_a, elem_b) in diss:
                bucket_diss.setdefault(lp, []).append(float(diss[(elem_a, elem_b)]))

        per_layer_pair_stats: dict[tuple[int, int], LayerPairStats] = {}
        for lp, n_pairs in bucket_n_pairs.items():
            per_layer_pair_stats[lp] = _aggregate_stats(
                bucket_kappa.get(lp, []),
                bucket_diss.get(lp, []),
                n_pairs,
            )

        return ContactPairLayerClassifierResult(
            pair_layer_keys=pair_layer_keys,
            per_layer_pair_stats=per_layer_pair_stats,
            n_unique_layer_pairs=len(per_layer_pair_stats),
        )
