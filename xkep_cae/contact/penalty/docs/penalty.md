# Penalty Strategy — ペナルティ剛性の決定と法線接触力

[← README](../../../../../README.md)

## 概要

梁–梁接触のペナルティ法における2つの責務を Process として実装する:

1. **ペナルティ剛性 k_pen の決定** (`PenaltyStrategy` Protocol)
2. **法線接触力 p_n の評価と線形化** (`NormalForceProcess`)

## ペナルティ剛性

### AutoBeamEI（推奨）

梁曲げ剛性 EI/L³ ベースの自動推定:

```
k_pen = scale × 12EI/L³ / max(1, f(n_pairs))
```

- `f = identity` (linear scaling) or `f = sqrt` (sqrt scaling)
- `scale` デフォルト 0.1

### Continuation

段階的にペナルティ剛性を増加:

- geometric: `k(step) = start × target × ratio^step`
- linear: `k(step) = target × (start + (1-start) × step/ramp_steps)`

## 法線接触力

### AL 法線力

```
p_n = max(0, λ_n + k_pen × (-g))
```

### スムースペナルティ（softplus）

```
p_n = k_pen × softplus(-g + λ_n/k_pen, δ)
softplus(x, δ) = δ × ln(1 + exp(x/δ))
```

C∞ 連続近似により active/inactive 二値判定を排除。

### 接線剛性寄与

```
dp_n/dg = -k_pen × sigmoid((-g + λ_n/k_pen) / δ)
```
