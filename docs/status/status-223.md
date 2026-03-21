# status-223: ペナルティパラメータ完全自動化 + 手動調整禁止機構

[← README](../../README.md) | [← status-index](status-index.md) | [← status-222](status-222.md)

**日付**: 2026-03-21
**テスト数**: 188（新規テスト 9 件追加）

## 概要

Huber 型ペナルティの全パラメータを自動推定に移行し、手動調整ルートを排除する機構を追加。

1. **AutoSmoothingDeltaProcess**: δ を梁半径から自動推定
2. **ManualPenaltyParameterWarning**: ランタイム検知（手動δ/Uzawa有効化）
3. **C18 契約チェック**: AST走査で手動パラメータ指定をビルド時検出
4. **n_uzawa_max デフォルト変更**: 全箇所で 5→1（凍結）
5. **数値テスト移行**: 手動 smoothing_delta を全て 0.0（自動推定）に

## 変更内容

### 1. AutoSmoothingDeltaProcess（新規）

**ファイル**: `xkep_cae/contact/penalty/strategy.py`

```
ε = α × r_min    （α=2e-4, r_min=正の梁半径の最小値）
δ = 1/ε
```

| 梁径 | r [mm] | ε [mm] | δ | 備考 |
|------|--------|--------|------|------|
| 2.0mm（標準ワイヤ）| 1.0 | 0.0002 | 5,000 | status-222 の手動値と一致 |
| 0.4mm（細線）| 0.2 | 0.00004 | 25,000 | スケール追従 |

### 2. ManualPenaltyParameterWarning（新規）

**ファイル**: `xkep_cae/core/diagnostics.py`

ランタイムで以下を検知し `UserWarning` を発行:
- `smoothing_delta > 0`（手動指定）
- `n_uzawa_max > 1`（Uzawa 有効化）

**ファイル**: `xkep_cae/contact/solver/process.py`

`ContactFrictionProcess.process()` 内で `warnings.warn()` を呼び出し。

### 3. C18 契約チェック（新規）

**ファイル**: `contracts/validate_process_contracts.py`

AST 走査で以下をビルド時検出:
- `smoothing_delta: float = <非ゼロ>` の dataclass フィールド定義
- `n_uzawa_max: int = <2以上>` の dataclass フィールド定義
- テストファイル（tests/ ディレクトリ、test_*.py）は除外

### 4. n_uzawa_max デフォルト 5→1

以下の全箇所を変更:
- `_ContactConfigInput.n_uzawa_max` → 1
- `default_strategies(n_uzawa_max=)` → 1
- `SmoothPenaltyContactForceProcess.__init__(n_uzawa_max=)` → 1
- `_create_contact_force_strategy(n_uzawa_max=)` → 1
- `NewtonUzawaStaticProcess` / `NewtonUzawaDynamicProcess` のフォールバック → 1

### 5. 数値テスト移行

**ファイル**: `xkep_cae/numerical_tests/three_point_bend_jig.py`

| Config | 旧値 | 新値 |
|--------|------|------|
| `ThreePointBendContactJigConfig.smoothing_delta` | 200.0 | 0.0 |
| `ThreePointBendContactJigConfig.n_uzawa_max` | 20 | 1 |
| `DynamicThreePointBendContactJigConfig.smoothing_delta` | 5000.0 | 0.0 |

## 防御機構まとめ

| レイヤー | 検知対象 | 機構 |
|---------|---------|------|
| **ビルド時**（C18） | smoothing_delta≠0, n_uzawa_max>1 の dataclass フィールド | AST 走査 |
| **ランタイム** | 手動 smoothing_delta, Uzawa 有効化 | `ManualPenaltyParameterWarning` |
| **デフォルト値** | n_uzawa_max | 全箇所で 1 に統一 |

## テスト

- `TestAutoSmoothingDeltaProcess`: 6 テスト
- `TestEstimateSmoothingDelta`: 3 テスト
- 既存 561 テスト: リグレッションなし（既存 6 件の失敗は変更前と同一）
- C18 契約チェック: OK（テスト外の手動パラメータ指定なし）

## 次のステップ

- [ ] n_periods ≥ 5 準静的テスト（δ 自動推定で）
- [ ] S3 凍結解除（7本変位制御曲げ揺動の xfail 解消）
