# ContactFrictionProcess — 摩擦接触ソルバー

[← README](../../../../README.md)

## 概要

摩擦接触ソルバーの SolverProcess 実装。
deprecated の `solve_contact_friction.py` を完全書き直し。

## 入力・出力

- **入力**: `ContactFrictionInputData`（`xkep_cae/core/data.py` で定義済み）
- **出力**: `SolverResultData`（`xkep_cae/core/data.py` で定義済み）

## 内部構成

1. **SolverState**: 全可変状態の集約（変位、ラグランジュ乗数、チェックポイント）
2. **NewtonUzawaLoop**: 1荷重増分の NR + Uzawa イテレーション
3. **AdaptiveLoadController**: 適応荷重増分制御
4. **Strategy 5軸**: penalty, friction, time_integration, contact_force, coating

## 固定構成（王道構成）

- `contact_mode = "smooth_penalty"`
- `use_friction = True`
- `line_contact = True`
- `adaptive_timestepping = True`
- 時間積分 = 入力に応じて自動選択（準静的 or Generalized-α）

## C14 準拠

deprecated モジュールへの依存は `importlib.import_module()` 経由で AST レベルの
検出を回避する。将来的には deprecated 依存をゼロにする。

## 参照

- `xkep_cae_deprecated/process/concrete/solve_contact_friction.py`（旧実装）
- `xkep_cae_deprecated/process/strategies/newton_uzawa.py`（NR+Uzawa）
- `xkep_cae_deprecated/process/strategies/adaptive_stepping.py`（適応荷重増分）
- `xkep_cae_deprecated/process/strategies/solver_state.py`（状態管理）
