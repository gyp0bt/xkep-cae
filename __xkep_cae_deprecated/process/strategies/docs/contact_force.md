# ContactForce Strategy 設計文書

[← README](../../../../README.md) | [← process-architecture](../../docs/process-architecture.md)

## 概要

接触力の評価方法を Strategy パターンで実装する。

## 具象クラス

| クラス | 方式 | 状態 |
|--------|------|------|
| `NCPContactForceProcess` | Alart-Curnier NCP + 鞍点系 | 利用可（摩擦なし専用） |
| `SmoothPenaltyContactForceProcess` | Softplus smooth penalty + Uzawa | 推奨 |

## 入出力

- **Input**: `ContactForceInput(u, lambdas, manager, k_pen)`
- **Output**: `ContactForceOutput(f_contact, K_contact, lambdas_new, ...)`

## ファクトリ

`create_contact_force_strategy()` — contact_mode に基づき NCP or SmoothPenalty を選択。

## 非互換

- `NCPContactForceProcess` + `CoulombReturnMappingProcess` は非互換 (status-147)
  - 摩擦接線剛性の符号問題で発散
  - 摩擦ありは `SmoothPenaltyContactForceProcess` を使用すること
