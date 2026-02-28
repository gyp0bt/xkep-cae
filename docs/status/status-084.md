# status-084: Alart-Curnier 摩擦拡大鞍点系の完全実装

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-28
- **テスト数**: 1878（+3: Alart-Curnier 摩擦テスト）
- **ブランチ**: claude/execute-status-todos-uhlKc

## 概要

NCP Semi-smooth Newton ソルバーの摩擦定式化を **return mapping ハイブリッド方式** から **Alart-Curnier 拡大鞍点系** に全面刷新。接線乗数 λ_t を一次変数として変位・法線乗数と同時に解く。

## 背景・動機

前回（status-083）の return mapping 方式では:
1. **∂f_fric/∂λ_n 欠落**: 摩擦力が法線乗数に依存するが、鞍点系にカップリング項がなかった
2. **z_t の Newton 反復内蓄積**: 内部変数 z_t が反復ごとに累積し、線形化と不整合
3. **横方向荷重で収束失敗**: 上記2点により、横方向力+摩擦のケースで残差 ~5e-4 で停滞

## 実装詳細

### Alart-Curnier 摩擦 NCP 定式化

接線乗数 λ_t を主変数に追加し、3ブロック拡大鞍点系を構築:

```
[K_eff    -G_n^T   -G_t^T ] [Δu   ]   [-R_u]
[J_n_u    J_n_n    0      ] [Δλ_n ] = [-C_n]
[J_t_u    J_t_n    J_t_t  ] [Δλ_t ]   [-C_t]
```

各ペアの状態に応じた Jacobian ブロック:

| 状態 | C_t | J_t_u | J_t_n | J_t_t |
|------|-----|-------|-------|-------|
| Stick | -k_t·Δu_t | -k_t·G_t | 0 | 0 |
| Slip | λ_t - μ·p_n·q̂ | μ·k_pen·q̂⊗G_n - μ·p_n·k_t/‖λ̂_t‖·(I-q̂⊗q̂)@G_t | -μ·q̂ | I - μ·p_n/‖λ̂_t‖·(I-q̂⊗q̂) |
| Inactive | λ_t | 0 | 0 | I |

### 新規関数

- `_build_tangential_constraint_jacobian()`: G_t (2n_active × ndof) 構築
- `_solve_augmented_friction_system()`: 3ブロック拡大系を sp.bmat で組立・spsolve で求解
- NR ループ内に λ_t 管理ロジック（初期化・更新・非活性時ゼロ化）

### 符号規約

- G_n: coeffs = [(1-s), s, -(1-t), -t] × normal
- G_t: coeffs = [-(1-s), -s, (1-t), t] × tangent（B-A 方向）
- g_n_shape = -G_n^T, g_t_shape = -G_t^T
- 接触力: f_c = -G_n^T·p_n - G_t^T·λ_t

### 重要な修正

- **Stick の J_t_u**: `-k_t * G_t`（以前は G_t のみで k_t 欠落）
- **Stick の J_t_t**: `0`（C_t = -k_t·Δu_t は λ_t に依存しない）
- **Slip の J_t_n**: `-μ·q̂` — **前回欠落していた ∂f_fric/∂λ_n カップリング項**
- **Slip の J_t_u**: μ·k_pen·q̂⊗G_n - μ·p_n·k_t/‖λ̂_t‖·(I-q̂⊗q̂)@G_t

## テスト

### 新規テスト（3件）
- `test_lateral_force_converges`: 横方向荷重(x:10, y:5) + μ=0.3 で収束 — **以前は収束不可**
- `test_friction_displacement_difference`: 摩擦有無で横方向変位差を検証（摩擦が抵抗）
- `test_high_friction_stick`: μ=1.0 高摩擦での stick 収束

### 既存テスト（リグレッションなし）
- NCP 摩擦 + line contact: 11/11 passed
- NCP ソルバー基本: 12/12 passed
- NCP 関数: 22/22 passed
- 全 fast テスト: 1449 passed, 56 skipped

## 旧コードからの移行

- `_compute_friction_forces_ncp()`: 残存（旧 return mapping 方式、摩擦なし NR パスでは使用されない）
- `_build_friction_stiffness()`: 残存（旧方式の摩擦剛性構築、現在は拡大系で処理）
- 摩擦有効時は自動的に Alart-Curnier 拡大系が使用される
- 摩擦無効時は従来の法線のみ鞍点系を使用（変更なし）

## 確認事項・今後の課題

- [ ] Line contact + Alart-Curnier 摩擦の複合テスト（Gauss 積分との統合検証）
- [ ] 多ペア環境（7本撚り）での収束性能評価
- [ ] Mortar 離散化（C6-L5）への進行
