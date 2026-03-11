# Penalty Strategy 設計文書

[← README](../../../../README.md) | [← process-architecture](../../docs/process-architecture.md)

## 概要

ペナルティ剛性 k_pen の決定方法を Strategy パターンで実装する。

## 具象クラス

| クラス | 方式 | 状態 |
|--------|------|------|
| `AutoBeamEIProcess` | 梁曲げ剛性 EI/L³ ベース | 推奨 |
| `AutoEALProcess` | 軸剛性 EA/L ベース | 利用可 |
| `ManualPenaltyProcess` | 手動指定 | deprecated (status-140) |
| `ContinuationPenaltyProcess` | 段階的ランプアップ | 利用可 |

## 入出力

- **Input**: `PenaltyInput(step, total_steps)`
- **Output**: `PenaltyOutput(k_pen)`

## ファクトリ

`create_penalty_strategy()` — solver_ncp.py の k_pen 決定ロジックを再現。

## 推定式

### AutoBeamEI

```
linear: k_pen = scale * 12 * E * I / L³ / max(1, n_pairs)
sqrt:   k_pen = scale * 12 * E * I / L³ / max(1, √n_pairs)
```

### AutoEAL

```
k_pen = scale * E * A / L
```

### Continuation

```
step 0        → start_fraction * k_pen_target
step ≥ ramp   → k_pen_target
中間           → 線形補間
```
