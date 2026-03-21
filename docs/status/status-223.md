# status-223: smoothing_delta 自動推定（AutoSmoothingDeltaProcess）

[← README](../../README.md) | [← status-index](status-index.md) | [← status-222](status-222.md)

**日付**: 2026-03-21
**テスト数**: 188（新規テスト 9 件追加）

## 概要

Huber 型ペナルティの唯一の手動パラメータ `smoothing_delta`（δ）を梁半径から自動推定する `AutoSmoothingDeltaProcess` を実装。これにより **ペナルティ接触のパラメータが完全自動化**された。

## 変更内容

### AutoSmoothingDeltaProcess（新規）

**ファイル**: `xkep_cae/contact/penalty/strategy.py`

推定式:
```
ε = α × r_min
δ = 1/ε
```

- `α` のデフォルト: **2e-4**（表面粗さオーダー）
- `r_min`: 正の梁半径の最小値（ジグ要素 r=0 は除外）

| 梁径 | r [mm] | ε [mm] | δ | 備考 |
|------|--------|--------|------|------|
| 2.0mm（標準ワイヤ）| 1.0 | 0.0002 | 5,000 | status-222 の手動値と一致 |
| 0.4mm（細線）| 0.2 | 0.00004 | 25,000 | スケール追従 |

### ContactFrictionProcess 統合

**ファイル**: `xkep_cae/contact/solver/process.py`

`smoothing_delta == 0.0`（未指定）の場合、`input_data.mesh.radii` から自動推定。
明示指定がある場合はそちらを優先（後方互換）。

### Uzawa 凍結

`n_uzawa_max=1`（純ペナルティ）を維持。Huber の線形領域固定点 g = -ε/2 ≠ 0 が Uzawa 不動点条件と非整合のため。

## テスト

- `TestAutoSmoothingDeltaProcess`: 6 テスト（wire_diameter=2mm, 細線, custom alpha, バリデーション, process frozen）
- `TestEstimateSmoothingDelta`: 3 テスト（スカラー, ジグ混在配列, 全ゼロ）
- 既存 561 テスト: リグレッションなし（既存 6 件の失敗は変更前と同一）

## 設計判断

### なぜ α = 2e-4 か

status-222 で δ=5000（r=1.0mm）が NR 2次収束を達成。逆算すると ε/r = 0.0002/1.0 = 2e-4。物理的には「表面粗さ」オーダーで、遷移幅が表面粗さ以下なら操作点は常に線形領域。

### ペナルティ剛性との関係

| パラメータ | 推定元 | 自動化状態 |
|-----------|--------|----------|
| k_pen | `AutoBeamEIPenalty`（EI/L³）or `DynamicPenaltyEstimateProcess`（c₀M_ii）| **済** |
| δ (smoothing_delta) | `AutoSmoothingDeltaProcess`（α × r_min）| **本 status で完了** |
| n_uzawa_max | 1 固定（Huber 純ペナルティ）| **凍結** |

→ ユーザが手動調整すべきペナルティパラメータは **ゼロ** になった。

## 次のステップ

- [ ] n_periods ≥ 5 準静的テスト（δ 自動推定で）
- [ ] S3 凍結解除（7本変位制御曲げ揺動の xfail 解消）
