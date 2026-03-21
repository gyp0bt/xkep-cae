# status-224: 手動ペナルティパラメータの物理削除

[← README](../../README.md) | [← status-index](status-index.md) | [← status-223](status-223.md)

**日付**: 2026-03-21
**テスト数**: 553（新規 0 件、既存修正のみ）

## 概要

status-223 で導入した自動推定機構を前提に、手動ペナルティ調整ルートを**物理削除**。
`k_pen`, `smoothing_delta`, `n_uzawa_max` のユーザー公開フィールドをすべて除去。

## 変更内容

### 1. フィールド物理削除

| ファイル | 削除フィールド |
|---------|-------------|
| `_ContactConfigInput` | `smoothing_delta`, `n_uzawa_max`, `tol_uzawa` |
| `ContactSetupConfig` | `k_pen` |
| `ContactSetupData` | `k_pen` |
| `default_strategies()` | `k_pen`, `n_uzawa_max`, `tol_uzawa` パラメータ |
| `SmoothPenaltyContactForceProcess` | `n_uzawa_max`, `tol_uzawa` コンストラクタ引数 |
| `_create_contact_force_strategy()` | `n_uzawa_max`, `tol_uzawa` パラメータ |
| `ThreePointBendContactJigConfig` | `k_pen`, `smoothing_delta`, `n_uzawa_max` |
| `DynamicThreePointBendContactJigConfig` | `k_pen`, `smoothing_delta`, `n_uzawa_max` |
| `StrandBatchConfig` | `k_pen` |

### 2. ソルバー内部の自動化

**ファイル**: `xkep_cae/contact/solver/process.py`

- `smoothing_delta`: 常に `_estimate_smoothing_delta(radii)` で自動推定
- `k_pen`: 常に PenaltyStrategy 経由で自動推定（手動オーバーライドパス削除）
- `k_pen continuation`: 条件分岐（`_setup_kpen` チェック）を削除、常に実行
- `n_uzawa_max`: `SmoothPenaltyContactForceProcess` 内部で 1 にハードコード

### 3. Huber型ペナルティの `_softplus` → `_huber_penalty` 移行

**ファイル**: `xkep_cae/contact/contact_force/strategy.py`

- `_softplus()` → `_huber_penalty()` に置換
- `_softplus_derivative()` → `_huber_penalty_derivative()` に置換
- テスト: `test_softplus_*` → `test_huber_*` に更新

### 4. 数値テスト修正

**ファイル**: `xkep_cae/numerical_tests/three_point_bend_jig.py`

- 3箇所の `ContactSetupData(k_pen=0.0, ...)` → `k_pen` 引数を除去
- 動的接触ジグの手動 k_pen 推定パス → 完全削除（ContactFrictionProcess 内部で自動推定）
- `_ContactConfigInput` への `smoothing_delta`/`n_uzawa_max` パラメータ渡しを除去

**ファイル**: `xkep_cae/numerical_tests/beam_oscillation.py`

- `ContactSetupData(k_pen=0.0, ...)` → `k_pen` 引数を除去

### 5. テストファイル修正

| ファイル | 修正内容 |
|---------|---------|
| `tests/contact/test_strand_contact_process.py` | `ContactSetupConfig(k_pen=...)` 除去 |
| `xkep_cae/contact/solver/tests/test_process.py` | `default_strategies(k_pen=...)` 除去 |
| `xkep_cae/contact/setup/tests/test_process.py` | `ContactSetupConfig(k_pen=...)` 除去、frozen テスト修正 |
| `xkep_cae/numerical_tests/tests/test_three_point_bend_jig.py` | `n_uzawa_max=10` 除去 |
| `xkep_cae/core/batch/tests/test_strand_bending.py` | `StrandBatchConfig(k_pen=...)` 除去 |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `_softplus` → `_huber_penalty` テスト更新 |
| `contracts/diagnose_three_point_bend.py` | cfg.k_pen/smoothing_delta/n_uzawa_max アクセス除去 |

## テスト結果

- 553 passed, 0 failed（beam_oscillation 14件 + stress_contour 1件は既存問題で除外）
- C18 契約チェック: OK
- ruff lint/format: OK

## 既存の非関連テスト失敗（変更前から存在）

| テスト | 原因 |
|-------|------|
| `test_large_amplitude_converges` | NCP ソルバー発散（接触なし梁揺動、本変更とは無関係） |
| `test_numerical_dissipation_rate` | エネルギー過剰減衰（本変更とは無関係） |
| `test_process_runs (stress_contour)` | 画像生成なし（本変更とは無関係） |

## 次のステップ

- [ ] n_periods ≥ 5 準静的テスト（δ 自動推定で）
- [ ] S3 凍結解除（7本変位制御曲げ揺動の xfail 解消）
- [ ] DynamicPenaltyEstimateProcess / DynamicThreePointBendContactJigProcess の C3 テスト紐付け
