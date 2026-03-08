# status-146: NCP接触ソルバーへのGeneralized-α動的解析統合

[← README](../../README.md) | [← status-index](status-index.md) | [← status-145](status-145.md)

**日付**: 2026-03-08
**テスト数**: 2271（変更なし）

## 概要

NCP接触ソルバー `newton_raphson_contact_ncp` に Generalized-α 動的解析を統合。
従来の準静的 load-fraction ベースの「適応時間増分」に、質量行列・Rayleigh減衰・
Newmark予測子を追加し、物理的な時間積分に移行する基盤を整備。

## 背景

従来のNCPソルバーは完全に準静的（質量行列・減衰行列なし）で、「適応時間増分」は
実際には荷重分率（load_frac 0→1）の分割に過ぎなかった。
接触状態の急変時にカットバックが発生し、極端に小さいΔ(load_frac)が必要になる問題があった。

動的解析を導入することで:
- 慣性効果が接触遷移を正則化（急変を物理的に抑制）
- Rayleigh減衰（特に剛性比例 β_R·K 項）が高周波チャタリングを減衰
- カットバック時に dt_sub が小さくなるほど c0·M 項が自然に増大し、安定化効果が強化

## 変更内容

### 1. solver_ncp.py — Generalized-α動的解析統合

**新パラメータ:**
- `mass_matrix`: 質量行列（疎/密）— None で準静的（後方互換）
- `damping_matrix`: 減衰行列（Rayleigh等）
- `dt_physical`: load_frac 0→1 に対応する物理時間 [s]
- `rho_inf`: スペクトル半径 ∈ [0,1]（1.0=Newmark平均加速度、エネルギー保存）
- `velocity`, `acceleration`: 初期速度・加速度

**実装箇所:**
- 残差: `R_u += M·a_{n+1-α_m} + C·v_{n+1-α_f}`
- 有効接線: `K_eff += (1-α_m)·c0·M + (1-α_f)·c1·C`
- Newmark予測子: `u_pred = dt·v + 0.5·dt²·(1-2β)·a`
- チェックポイント: v, a のロールバック対応
- UL統合: 参照配置更新後も v, a は保持

### 2. beam_timo3d.py — ULCRBeamAssembler.assemble_mass()

3D Timoshenko梁要素の質量行列アセンブリメソッドを追加。
- HRZ法集中質量（lumped=True、推奨）: 対角行列 → 高速
- 整合質量（lumped=False）: 完全な要素質量行列

### 3. wire_bending_benchmark.py — Phase2動的解析パラメータ

新パラメータ:
- `dynamics=True/False`: 動的解析の有効化
- `oscillation_frequency_hz`: 揺動周波数 [Hz]
- `rho`: 密度 [ton/mm³]（鋼: 7.85e-9）
- `rayleigh_alpha`, `rayleigh_beta`: Rayleigh減衰係数
- `rho_inf`: Generalized-αスペクトル半径

### 4. NCPSolveResult — velocity, acceleration 追加

動的解析時に最終状態の速度・加速度を返却。
ステップ間の状態引き継ぎ（Phase2揺動）で使用。

## 物理的考察

### 固有振動数 vs 揺動周波数

鋼線（E=200GPa, ρ=7850 kg/m³, r=1mm, L=40mm）の第1固有振動数:
```
f₁ ≈ 2478 Hz
```

1 Hz 揺動は完全に準静的。慣性効果（c0·M/K_T）は初期ステップで 0.01% 以下。
ただしカットバック時（delta_frac=0.01）には 58% に達し、正則化効果が発現。

### Rayleigh減衰の効果

剛性比例 β_R·K 項は K_eff に `(1-α_f)·c1·β_R·K` を加算。
ξ=0.002（鋼）で delta_frac=0.01 のとき β_R/dt ≈ 0.76、有意な高周波減衰を提供。

### パラメータ推奨値

| パラメータ | 推奨値 | 備考 |
|-----------|--------|------|
| rho_inf | 1.0 | エネルギー保存（数値粘性なし） |
| ξ (damping ratio) | 0.001-0.005 | 鋼の物理的減衰比 |
| β_R | 2ξ/(2πf₁) | 第1固有振動数でのRayleigh近似 |
| α_R | 0.0 | 質量比例減衰は通常不要 |

## 検証スクリプト

- `scripts/verify_dynamics_ncp.py`: 準静的 vs 動的の比較検証

## 次の課題

- Phase1 にも動的解析を適用（曲げのramp loading）
- 適応時間増分と動的解析の相互作用の最適化
- より大規模ケース（19本以上）での動的解析効果の検証
