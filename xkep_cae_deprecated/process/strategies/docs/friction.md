# Friction Strategy 設計文書

[← README](../../../../README.md) | [← process-architecture](../../docs/process-architecture.md)

## 概要

摩擦力の評価方法を Strategy パターンで実装する。

## 具象クラス

| クラス | 方式 | 状態 |
|--------|------|------|
| `NoFrictionProcess` | 摩擦なし | デフォルト |
| `CoulombReturnMappingProcess` | Coulomb摩擦 + return mapping | 利用可 |
| `SmoothPenaltyFrictionProcess` | Smooth penalty + Uzawa 摩擦 | 推奨 |

## 入出力

- **Input**: `FrictionInput(u, contact_pairs, mu)`
- **Output**: `FrictionOutput(f_friction, K_friction)`

## ファクトリ

`create_friction_strategy()` — use_friction / contact_mode に基づき選択。

## 注意事項

- NCP系ソルバーで摩擦ありの場合は `SmoothPenaltyFrictionProcess` 必須 (status-147)
