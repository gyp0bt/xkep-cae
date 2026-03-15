# Friction Strategy 設計仕様

[← README](../../../../README.md)

## 概要

Coulomb 摩擦の return mapping アルゴリズムを Process Architecture で実装。

## 摩擦モデル

### Coulomb return mapping

1. 弾性予測: `q_trial = z_t_old + k_t × Δu_t`
2. Coulomb 判定: `||q_trial|| ≤ μ × p_n`
3. stick: `q = q_trial`
4. slip: `q = μ × p_n × q_trial / ||q_trial||`

### slip consistent tangent

- stick: `D_t = k_t × I₂`
- slip: `D_t = (μ×p_n / ||q_trial||) × k_t × (I₂ - q̂⊗q̂)`

## Process 一覧

| Process | Protocol | 用途 |
|---------|----------|------|
| NoFrictionProcess | FrictionStrategy | 摩擦なし（デフォルト） |
| CoulombReturnMappingProcess | FrictionStrategy | NCP + Coulomb return mapping |
| SmoothPenaltyFrictionProcess | FrictionStrategy | Smooth penalty + Uzawa 摩擦 |
| ReturnMappingProcess | SolverProcess | Coulomb return mapping（単体） |
| FrictionTangentProcess | SolverProcess | 摩擦接線剛性 (2×2) |

## μ ランプ

`μ_eff = μ_target × min(1, ramp_counter / mu_ramp_steps)`

初期の収束安定化のため、Outer loop ごとに ramp_counter を漸増。

## 参照

- status-147: NCP 鞍点系の摩擦接線剛性符号問題
- status-175: 脱出ポット計画 Phase 1
