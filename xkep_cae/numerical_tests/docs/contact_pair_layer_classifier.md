# ContactPairLayerClassifierProcess — 接触ペア層分類後処理

[← README](../../../README.md) | [← roadmap](../../../docs/roadmap.md)

## 概要

`ContactPairAnalysisProcess`（status-337）が抽出する `kappa_cr_per_pair` /
`per_pair_dissipation` を、撚線層構造（core / inner / outer …）の組み合わせで
バケット化する後処理 Process。

status-339 で観測された 19本撚線 κ_cr 分布のバイモーダル気配
（3.2e-3 付近と 5.0-5.3e-3 付近のダブルピーク）を、
内層対 / 外層対 / 層跨ぎ対の 3 カテゴリに切り分けて検証するために導入する。

## 位置付け

| 階層 | 出力 | Process |
|------|------|---------|
| 集約（M-κ） | loop_area, dissipation_ratio, EI_secant/initial | `CableDissipationProcess` + `_compute_mk_metrics` |
| 素線レベル | κ_cr 分布, 各ペア散逸, 活性ペア数推移 | `ContactPairAnalysisProcess` |
| **層レベル** | **層ペア毎の κ_cr 分布 / 散逸合計** | **`ContactPairLayerClassifierProcess`（本 Process）** |

## 入力

| フィールド | 型 | 説明 |
|-----------|---|------|
| kappa_cr_per_pair | Mapping[(int, int), float] | `ContactPairAnalysisResult.kappa_cr_per_pair` を直接渡す |
| per_pair_dissipation | Mapping[(int, int), float] | `ContactPairAnalysisResult.per_pair_dissipation` を直接渡す |
| strand_ids | Sequence[int] | elem_id → strand_id（`MeshData.strand_ids` の list/tuple/np.ndarray） |
| strand_layers | Sequence[int] | strand_id → layer（例: `tuple(info.layer for info in strand_infos)`） |

## 出力

| フィールド | 型 | 説明 |
|-----------|---|------|
| pair_layer_keys | dict[(int, int), (int, int)] | (elem_a, elem_b) → (l_min, l_max) |
| per_layer_pair_stats | dict[(int, int), LayerPairStats] | 層ペア毎の集約統計 |
| n_unique_layer_pairs | int | per_layer_pair_stats のキー数 |

### LayerPairStats

| フィールド | 型 | 説明 |
|-----------|---|------|
| n_pairs | int | 該当ペア数（kappa or dissipation の和集合） |
| n_slipped | int | κ_cr が記録されたペア数 |
| kappa_cr_mean / std / min / max | float | κ_cr 分布統計（n_slipped > 0 のときのみ有意） |
| dissipation_sum / mean | float | per_pair_dissipation 合計 / 平均 |

## 層分類ルール

1. 各 elem_id を `strand_ids[elem_id]` で strand_id に変換
2. 各 strand_id を `strand_layers[strand_id]` で layer に変換
3. ペア (elem_a, elem_b) → (layer_a, layer_b) を昇順に並び替え (l_min, l_max) に正規化
4. 同一の (l_min, l_max) を共有するペアを集約

## 19本撚線（1+6+12 構造）の例

| (l_min, l_max) | 物理的意味 |
|----------------|-----------|
| (0, 1) | core ↔ inner |
| (0, 2) | core ↔ outer（通常成立しない） |
| (1, 1) | inner ↔ inner |
| (1, 2) | inner ↔ outer |
| (2, 2) | outer ↔ outer |

`exclude_same_strand=True` の場合、同 strand 内ペアは入力に含まれない。

## 使用例

```python
from xkep_cae.numerical_tests.contact_pair_analysis import (
    ContactPairAnalysisProcess,
    ContactPairAnalysisInput,
)
from xkep_cae.numerical_tests.contact_pair_layer_classifier import (
    ContactPairLayerClassifierProcess,
    ContactPairLayerClassifierInput,
)

# StrandMeshProcess の結果から strand_layers を作る
strand_layers = tuple(info.layer for info in strand_infos)

analysis = ContactPairAnalysisProcess().process(
    ContactPairAnalysisInput(
        contact_pair_history=solver_result.contact_pair_history,
        moment_curvature_history=solver_result.moment_curvature_history,
    )
)
classifier = ContactPairLayerClassifierProcess().process(
    ContactPairLayerClassifierInput(
        kappa_cr_per_pair=analysis.kappa_cr_per_pair,
        per_pair_dissipation=analysis.per_pair_dissipation,
        strand_ids=mesh.strand_ids.tolist(),
        strand_layers=strand_layers,
    )
)
for (l1, l2), s in classifier.per_layer_pair_stats.items():
    print(f"({l1},{l2}): n={s.n_pairs}, κ_cr={s.kappa_cr_mean:.3e}±{s.kappa_cr_std:.3e}")
```

## 設計判断

- `PostProcess` カテゴリ（`xkep_cae.core.categories`）に配置。
  解析履歴を持たない純粋集約処理なので、`uses = ()` で他 Process に依存しない。
- `ContactPairAnalysisProcess` を拡張する選択肢もあったが、層分類はメッシュ
  情報（strand_ids / strand_layers）を必要とするのに対し、本体は履歴のみで
  完結するため、責務分離して別 Process とした。
- バイモーダル仮説の検証は、本 Process の出力（per_layer_pair_stats）を
  比較することで定量化可能。
