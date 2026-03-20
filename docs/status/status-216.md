# status-216: 動的ソルバー強化 — エネルギー診断・統一時間増分・発散検知

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-20
**ブランチ**: `claude/strengthen-dynamic-solver-hiG4L`

---

## 概要

動的ソルバーの収束力強化・自動時間増分スキームの盤石化を実施。
3つの新機能を Process Architecture に追加し、ソルバー統合まで完了。

1. **StepEnergyDiagnosticsProcess**: ステップごとのエネルギー収支診断
2. **UnifiedTimeStepProcess**: 適応時間増分 + 荷重制御の統一 Process
3. **発散早期検知**: 残差連続増加による早期アボート + 積極的カットバック

## 変更内容

### 1. StepEnergyDiagnosticsProcess（新規）

**ファイル**: `xkep_cae/contact/solver/_energy_diagnostics.py`

ステップごとのエネルギー収支を計算する Process:
- KE = 0.5 * v^T M v（運動エネルギー）
- SE = 0.5 * u^T f_int（ひずみエネルギー）
- W_ext = f_ext · u（外力仕事）
- W_contact = f_c · u（接触仕事）
- energy_ratio = total / (|W_ext| + |W_contact| + |total|)

付属クラス:
- `EnergyHistoryEntryOutput`（frozen dataclass）: 履歴エントリ
- `_EnergyHistoryAccumulator`（private, 可変状態）: 履歴蓄積 + decay_ratio + サマリ

### 2. UnifiedTimeStepProcess（新規）

**ファイル**: `xkep_cae/contact/solver/_unified_time_controller.py`

適応時間増分と荷重制御を統一した Process:
- 物理時間ベースのインタフェース（t_total, dt_initial, dt_min, dt_max）
- 内部で AdaptiveSteppingProcess に委譲、物理時間 ↔ load_frac 変換
- 準静的: F(t) = (t/t_total) × F_max の線形ランプ自動変換
- 動的: 物理時間 dt_physical をそのまま使用
- 自動パラメータ: dt_initial=t_total/20, dt_min=dt_initial/32, dt_max=min(dt_initial*4, t_total)

### 3. 発散早期検知

**ファイル**: `xkep_cae/contact/solver/_newton_uzawa_dynamic.py`

- `NewtonUzawaDynamicInput.divergence_window: int = 5`
- 残差比率の連続増加を追跡、N回連続で早期アボート
- `DynamicStepOutput.diverged: bool` で発散状態を通知

### 4. AdaptiveSteppingProcess 強化

**ファイル**: `xkep_cae/contact/solver/_adaptive_stepping.py`

- `AdaptiveStepInput.diverged: bool = False`
- `AdaptiveStepOutput.n_cutbacks: int = 0`（累積カットバック回数）
- 発散時: `shrink = shrink_factor²`（通常の2乗で積極縮小）

### 5. ソルバー統合

**ファイル**: `xkep_cae/contact/solver/process.py`

- 動的ステップ収束後にエネルギー診断を実行、履歴に蓄積
- 発散フラグを AdaptiveStepInput に伝播
- SolverResultData に energy_history, n_cutbacks を格納
- 正常完了時にエネルギーサマリをコンソール出力

### 6. ConvergenceDiagnosticsOutput 拡張

**ファイル**: `xkep_cae/contact/solver/_diagnostics.py`

- `kinetic_energy`, `strain_energy`, `total_energy`, `energy_ratio` フィールド追加
- 診断レポートにエネルギー情報を含む

### 7. SolverResultData 拡張

**ファイル**: `xkep_cae/core/data.py`

- `energy_history: object | None = None`
- `n_cutbacks: int = 0`

## テスト結果

```
496 passed, 1 xfailed (full suite, not slow)
ruff check: All checks passed
ruff format: OK
契約違反: 0件
条例違反: 0件
```

### 新規テスト（24件）

- `test_energy_diagnostics.py`: 15件（@binds_to StepEnergyDiagnosticsProcess）
  - ゼロ状態、運動エネルギー、ひずみエネルギー、密行列、decay_ratio、サマリ等
- `test_unified_time_controller.py`: 9件（@binds_to UnifiedTimeStepProcess）
  - 自動パラメータ、QUERY/SUCCESS/FAILURE、完全サイクル、プロパティ等

## TODO

- [ ] 単線の剛体支えと押しジグによる動的三点曲げの解析解一致
- [ ] UnifiedTimeStepProcess を ContactFrictionProcess に統合（現在は AdaptiveSteppingProcess を直接使用）
- [ ] 数値粘性の定量評価: rho_inf 依存性の検証

---
