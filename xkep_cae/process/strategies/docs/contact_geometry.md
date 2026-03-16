# ContactGeometry Strategy 設計文書

[← README](../../../../README.md) | [← process-architecture](../../docs/process-architecture.md)

## 概要

接触幾何の評価方法を Strategy パターンで実装する。

## 具象クラス

| クラス | 方式 | 状態 |
|--------|------|------|
| `PointToPointProcess` | 最近接点ペア検出 | 基本 |
| `LineToLineGaussProcess` | Line-to-line Gauss積分 | 推奨 |
| `MortarSegmentProcess` | Mortar射影ベース | Phase 5 で完全統合予定 |

## 入出力

- **Input**: `ContactGeometryInput(node_coords, connectivity, radii)`
- **Output**: `ContactGeometryOutput(pairs, gap_vectors, ...)`

## 追加メソッド

| メソッド | 用途 |
|---------|------|
| `update_geometry(pairs, node_coords, *, config)` | 全ペアの幾何情報更新 |
| `build_constraint_jacobian(pairs, ndof_total, ndof_per_node)` | 制約ヤコビアン G 構築 |

## ファクトリ

`create_contact_geometry_strategy()` — mode / line_contact / use_mortar に基づき選択。
