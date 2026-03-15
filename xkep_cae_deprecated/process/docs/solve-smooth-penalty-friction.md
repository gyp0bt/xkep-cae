# Smooth Penalty 摩擦接触ソルバー設計仕様

[← README](../../../README.md) | [← process-architecture](process-architecture.md)

## 概要

`solver_ncp.py` の smooth_penalty + 摩擦パスを専用関数として切り出し、
冗長分岐のない王道ソルバー構成を提供する。

## 固定構成

| 軸 | 固定値 | 根拠 |
|---|---|---|
| `contact_mode` | `"smooth_penalty"` | NCP鞍点系は摩擦接線剛性符号問題で発散 (status-147) |
| `use_friction` | `True` | 摩擦常時有効 |
| `line_contact` | `True` | Line-to-line Gauss積分 |
| `use_mortar` | `False` | Mortar は別パス |
| `adaptive_timestepping` | `True` | 常時有効 |
| `adaptive_omega` | `False` | レガシー不使用 |
| `modified_nr_threshold` | `0` | Full NR |

## プロセスクラス

### ContactFrictionProcess（統一型, status-172）

- 入力: `ContactFrictionInputData`（動的パラメータ Optional）
- 動的/準静的は `is_dynamic` プロパティで自動判定
  - 動的（mass_matrix + dt_physical > 0）: Generalized-α 時間積分
  - 準静的（それ以外）: 荷重制御 or 変位制御
- 出力: `SolverResultData`

## ソルバー関数

### `solve_smooth_penalty_friction()`

`xkep_cae/contact/solver_smooth_penalty.py` に配置。

`newton_raphson_contact_ncp()` の smooth_penalty パス（Uzawa + Newton内部ループ）を
専用関数として切り出す。以下の分岐を排除:

- smooth/NCP 分岐 → 常に smooth_penalty
- mortar 分岐 → 常に無効
- 摩擦有無分岐 → 常に有効
- adaptive_omega → 常に無効
- modified NR → 常に Full NR

動的/準静的は `TimeIntegrationStrategy` で統一的に制御。

## status 参照

- status-168: 本設計の実装記録
- status-147: NCP鞍点系の符号問題特定（smooth_penalty 必須化の根拠）
- status-158: SolverStrategies 統合
- status-173: deprecated プロセス完全削除（NCPQuasiStatic/NCPDynamic → ContactFrictionProcess）
"""
