# TimeIntegration Strategy 設計文書

[← README](../../../../README.md) | [← process-architecture](../../docs/process-architecture.md)

## 概要

時間積分方法を Strategy パターンで実装する。

## 具象クラス

| クラス | 方式 | 状態 |
|--------|------|------|
| `QuasiStaticProcess` | 準静的（荷重制御） | デフォルト |
| `GeneralizedAlphaProcess` | Generalized-α 動的解析 (Chung & Hulbert 1993) | 利用可 |

## 入出力

- **Input**: `TimeIntegrationInput(u, du, dt)`
- **Output**: `TimeIntegrationOutput(M_eff, f_inertia)`

## ファクトリ

`create_time_integration_strategy()` — dynamics フラグに基づき選択。
