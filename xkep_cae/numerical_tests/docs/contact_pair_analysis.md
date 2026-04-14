# ContactPairAnalysisProcess — 接触ペア履歴後処理

[← README](../../../README.md) | [← roadmap](../../../docs/roadmap.md)

## 概要

`ContactFrictionProcess` が `track_contact_pairs=True` で出力する
`contact_pair_history`（インクリメント毎の活性接触ペアスナップショット列）を
解析し、以下を抽出する後処理 Process。

- 各ペア (elem_a, elem_b) の **κ_cr**（初回スリップ曲率）
- 各ペアの **最終累積散逸エネルギー**
- インクリメント毎の **活性ペア数** 時系列
- κ_cr **分布統計**（平均 / 標準偏差 / min / max）

status-333 / status-335 / status-336 で整備された M-κ ヒステリシス集約量を
補完し、**素線レベル** での接触分布をファイバー梁キャリブレーション用
指標として提供する。

## 位置付け

| 階層 | 出力 | Process |
|------|------|---------|
| 集約（M-κ） | loop_area, dissipation_ratio, EI_secant/initial | `CableDissipationProcess` + `_compute_mk_metrics` |
| 素線レベル | κ_cr 分布, 各ペア散逸, 活性ペア数推移 | **`ContactPairAnalysisProcess`（本 Process）** |

## 入力

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|---------|------|
| contact_pair_history | tuple | — | `SolverResultData.contact_pair_history` |
| moment_curvature_history | tuple | () | `SolverResultData.moment_curvature_history`。空の場合 κ_cr の代わりに load_frac を格納 |
| slip_threshold | float | 1e-6 | `|(slip_s, slip_t)|` > threshold でスリップ検知 |

## 出力

| フィールド | 型 | 説明 |
|-----------|---|------|
| n_steps | int | 履歴インクリメント数 |
| n_unique_pairs | int | 履歴内で1度以上活性化したペア総数 |
| n_active_per_step | tuple[int, ...] | 各インクリメントの活性ペア数 |
| load_frac_per_step | tuple[float, ...] | 各インクリメントの load_frac |
| kappa_cr_per_pair | dict | (elem_a, elem_b) → κ_cr（初回スリップ曲率） |
| per_pair_dissipation | dict | (elem_a, elem_b) → 最終散逸エネルギー |
| total_dissipation | float | 全ペア散逸合計 |
| kappa_cr_mean/std/min/max | float | κ_cr 分布統計 |
| n_slipped_pairs | int | スリップ観測ペア数（≤ n_unique_pairs） |

## κ_cr 判定ルール

最初に以下 3 条件を満たしたインクリメントの κ を記録する：

1. `stick == False`（return mapping で slip 判定）
2. `|(slip_s, slip_t)| > slip_threshold`（数値雑音除外）
3. そのペアの κ_cr が未記録

以降のスナップショットは無視（最初の遷移のみ追跡）。

## 使用例

```python
from xkep_cae.numerical_tests.contact_pair_analysis import (
    ContactPairAnalysisProcess,
    ContactPairAnalysisInput,
)

# ContactFrictionProcess 実行後の SolverResultData から
result = ContactPairAnalysisProcess().process(
    ContactPairAnalysisInput(
        contact_pair_history=solver_result.contact_pair_history,
        moment_curvature_history=solver_result.moment_curvature_history,
    )
)
print(f"n_unique_pairs={result.n_unique_pairs}")
print(f"κ_cr = {result.kappa_cr_mean:.3e} ± {result.kappa_cr_std:.3e}")
print(f"total dissipation = {result.total_dissipation:.3e}")
```

## 設計判断

- `PostProcess` カテゴリ（`xkep_cae.core.categories`）に配置。
  SolverProcess が生成した履歴に対する純粋後処理なので、`uses = ()` で
  他 Process に依存しない。
- 現時点では統計量出力のみ。ヒストグラム binning / CSV 出力 / 可視化は
  後続 PR で追加（まずは数値指標として成立させる）。
- `_compute_mk_metrics`（`cable_dissipation.py`）とは責務が直交：
  こちらは **接触ペア** レベル、あちらは **M-κ ループ** レベル。
