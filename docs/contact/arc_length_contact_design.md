# 接触付き弧長法 設計検討

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← 接触仕様](beam_beam_contact_spec_v0.1.md)

## 0. 目的

接触問題（特にスナップスルー・座屈を伴う接触）に対して、弧長法と接触ソルバーを統合する方針を検討する。現在の `newton_raphson_with_contact` は荷重増分法ベースであり、リミットポイント通過が必要な問題には対応できない。

## 1. 現在の実装

### 1.1 弧長法（solver.py: arc_length）

- Crisfield (1981) 円筒弧長法
- 弧長拘束: `||Δu||² = Δl²`
- 予測子: `K_T·δu_t = f_ext_ref`, `Δλ = ±Δl/||δu_t||`
- 修正子: 二次方程式から `δλ` を決定
- 弧長適応制御、カットバック、符号反転によるスナップバック追跡
- **接触非対応**

### 1.2 接触付きNR（solver_hooks.py: newton_raphson_with_contact）

- Outer/Inner 分離アーキテクチャ
  - Outer: 接触候補検出 + 幾何更新 + AL乗数更新 + μランプ
  - Inner: NR反復（最近接固定、接触力/剛性を追加）
- merit line search
- PDAS（実験的）
- **荷重増分法のみ**: `λ = step/n_load_steps` で固定

## 2. 統合方針

### 2.1 アプローチ比較

| アプローチ | 概要 | 長所 | 短所 |
|-----------|------|------|------|
| **(A) 弧長法の修正子に接触力を追加** | 残差 `R = λ·f_ext - f_int - f_contact` | 実装が比較的単純。既存の弧長法を拡張 | Outer/Inner分離と弧長修正子の整合が複雑 |
| **(B) 接触NRをラップした弧長法** | 弧長の予測子/修正子の中で接触NRを呼ぶ | 既存の接触NRをそのまま利用 | 弧長修正子の効率が悪い。二重ループの性能 |
| **(C) 統合弧長接触ソルバー** | Outer loop 内で弧長修正子を実行 | 最も一貫した定式化。最適な収束 | 実装が最も複雑 |

### 2.2 推奨: アプローチ (A) 段階的拡張

弧長法の修正子ステップに接触力を追加する方式を推奨する。理由:

1. **既存弧長法の拡張**として実装でき、コード変更が局所的
2. **Outer/Inner分離**は弧長法ステップ内にそのまま適用可能
3. 将来の撚線モデル（Phase 4.7）での座屈問題に対応できる

### 2.3 アルゴリズム概要

```
arc_length_with_contact:
  for step in 1..n_steps:
    -- 予測子 --
    K_total = K_T(u) + K_contact(u)   # 接触剛性込みの接線
    K_total · δu_t = f_ext_ref
    Δλ = ±Δl / ||δu_t||
    Δu = Δλ · δu_t

    -- Outer loop（接触更新）--
    for outer in 1..n_outer_max:
      接触候補検出 + 幾何更新 + μランプ

      -- Inner 修正子（NR反復 + 弧長拘束）--
      for it in 1..max_iter:
        f_int = assemble_internal_force(u + Δu)
        f_c = compute_contact_force(manager, ...)
        R = (λ + Δλ) · f_ext_ref - f_int - f_c

        K_T = assemble_tangent(u + Δu) + K_contact(...)
        K_T · δu_R = R
        K_T · δu_t = f_ext_ref

        二次方程式から δλ を求める:
          v = Δu + δu_R
          a1 = δu_t^T · δu_t
          a2 = 2 · v^T · δu_t
          a3 = v^T · v - Δl²
          δλ = (-a2 ± √(a2² - 4·a1·a3)) / (2·a1)

        Δu += δu_R + δλ · δu_t
        Δλ += δλ

        gap更新（s,t固定）
        摩擦return mapping
        merit line search（オプション）
        収束判定

      -- Outer 収束判定 --
      幾何更新後の (s,t) 変化を検査
      AL乗数更新

    u += Δu
    λ += Δλ
```

## 3. 実装上の課題

### 3.1 接触剛性の非対称性

接触剛性 `K_contact` は一般に非対称（特に摩擦あり・slip状態）。
弧長法の二次方程式は対称剛性を仮定している。

**対策**:
- 予測子では接触剛性を対称化して使用: `K_sym = (K_c + K_c^T) / 2`
- 修正子では非対称のまま使用（直接法ソルバー）

### 3.2 Active-set の変化

弧長法のステップ間で Active-set（接触ペアの活性/非活性）が変化すると、
弧長パスが不連続になる可能性がある。

**対策**:
- Active-set 変化をカットバックトリガーとして追加
- 弧長縮小時に Active-set をリセットし再試行
- 接触活性化/非活性化の閾値にヒステリシスを適用（既存 `g_on`/`g_off` 活用）

### 3.3 弧長の定義

接触問題では変位だけでなく接触力（ラグランジュ乗数）も未知数と見なせる。
弧長拘束を `||Δu||² + α||Δλ_n||² = Δl²` に拡張する選択肢もある。

**推奨**: まずは円筒弧長法（変位ノルムのみ）を使用し、
必要に応じて拡張する。

### 3.4 摩擦との整合

弧長法のステップでは荷重係数 λ が変化するため、
外力由来の法線力も変化する。摩擦の return mapping は
λ の変化に追従する必要がある。

**対策**:
- 各修正子反復で法線力を再評価し、Coulomb 条件を更新
- μランプは弧長法ステップとは独立に管理

## 4. テスト計画

| テスト | 内容 | 難易度 |
|--------|------|--------|
| 接触なし弧長法 | 既存テストとの回帰確認 | 低 |
| 法線接触 + 弧長法 | 2梁の押し付け、リミットポイントなし | 中 |
| 接触スナップスルー | 拘束付きスナップスルー問題 | 高 |
| 摩擦 + 弧長法 | 摩擦付き接触の弧長追跡 | 高 |
| 座屈+接触 | 梁の座屈が接触で拘束される問題 | 高 |

## 5. 実装優先度

1. **Phase 4.7（撚線モデル）で座屈問題が顕在化するまでは実装しない**
2. 撚線モデルで接触付き座屈解析が必要になった時点で着手
3. まずはアプローチ (A) で法線接触のみの弧長法を実装
4. 摩擦付きは段階的に追加

## 6. 参考文献

- Crisfield, M.A. (1981) "A fast incremental/iterative solution procedure that handles snap-through"
- Wriggers, P. (2006) "Computational Contact Mechanics" Ch.11 — 接触付き弧長法
- de Souza Neto et al. "Computational Methods for Plasticity" Ch.4 — 弧長法の一般論
- Jiang, W.G. et al. (2006) "Statically indeterminate contacts in axially loaded wire strand" — 撚線の座屈

---
