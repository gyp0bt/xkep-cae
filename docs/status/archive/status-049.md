# status-049: 適応的ペナルティ増大 + 貫入1%目標 + スライド接触テスト

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-24
**作業者**: Claude Code
**テスト数**: 1043（+9）

## 概要

梁梁接触の貫入量制御を改善し、大半のケースで search_radius の 1% 以下に
制限する「適応的ペナルティ増大（Adaptive Penalty Augmentation）」を実装。
合わせてマルチセグメント梁テスト、横スライド接触テスト、共有節点フィルタを追加。

## 課題分析

### 現行設計の問題点

1. **ペナルティ剛性が固定**: `k_pen` は初回初期化後、更新されない。AL外部ループで
   乗数 `lambda_n` は更新されるが、`k_pen` は固定のまま。

2. **外部ループ回数の制限 (n_outer_max=5)**: AL法は外部ループ→∞で貫入ゼロに
   収束するが、5回では不十分な場合がある。

3. **貫入ベースの収束判定がない**: 外部ループの収束は `|Δs|, |Δt|` のみで判定。
   貫入量が大きくても (s,t) が安定すれば収束と判定されていた。

### 対策: 適応的ペナルティ増大

Wriggers (Computational Contact Mechanics) に基づくアプローチ:

```
Outer loop 終了時:
  1. AL乗数更新: lambda_n <- p_n
  2. 貫入チェック: max_penetration / search_radius > tol_penetration_ratio ?
  3. YES → k_pen *= penalty_growth_factor (上限 k_pen_max)
  4. Outer 収束判定: (s,t) 収束 AND 貫入 < tol → 収束
```

## 変更内容

### 1. ContactConfig 拡張 (`xkep_cae/contact/pair.py`)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `tol_penetration_ratio` | 0.01 | 貫入許容比（search_radius基準、1%） |
| `penalty_growth_factor` | 2.0 | 貫入超過時のk_pen成長係数 |
| `k_pen_max` | 1e12 | ペナルティ剛性上限（条件数悪化防止） |

### 2. 適応的ペナルティ増大 (`xkep_cae/contact/solver_hooks.py`)

- AL乗数更新後に最大貫入比をチェック
- 貫入超過ペアの `k_pen` を `penalty_growth_factor` 倍に増大
- Outer収束判定に貫入量条件を追加
- (s,t) 収束だが貫入超過の場合、ペナルティ増大して次Outerへ

### 3. 共有節点フィルタ (`xkep_cae/contact/pair.py`)

- `detect_candidates` にマルチセグメント梁用の共有節点除外フィルタを追加
- 同一梁内の隣接セグメント（共有節点あり）を接触候補から除外
- 既存の単一セグメントテストには影響なし（後方互換）

### 4. テスト更新・追加 (`tests/contact/test_beam_contact_penetration.py`)

誤字修正: 「針」→「梁」

| クラス | テスト数 | 検証内容 |
|-------|---------|---------|
| `TestBeamContactDetection` | 2 | 接触検出/非検出 |
| `TestPenetrationBound` | 4 | 1%貫入制限、大荷重、ペナルティ依存、適応改善 |
| `TestNormalForce` | 2 | 法線力正値性、単調性 |
| `TestFrictionPenetrationEffect` | 2 | 摩擦あり1%制限、摩擦影響差 |
| `TestDisplacementHistory` | 2 | z変位進行、x張力 |
| `TestMultiSegmentBeamPenetration` | 3 | マルチセグメント接触検出、1%制限、大荷重 |
| `TestSlidingContact` | 5 | スライド接触検出、1%制限、摩擦スライド、x変位、摩擦有無収束 |

合計: 20テスト（旧11テスト → 新20テスト、+9）

### テスト内訳（新規テスト）

- `test_adaptive_penalty_improves_penetration`: 適応的ペナルティの効果検証
- `test_multi_segment_contact_detected`: 4分割梁での接触検出
- `test_multi_segment_penetration_within_1_percent`: 4分割梁での1%制限
- `test_multi_segment_large_force_penetration`: 4分割梁の大荷重ケース
- `test_sliding_contact_detected`: 横スライド接触検出
- `test_sliding_penetration_within_1_percent`: スライド中の1%制限
- `test_sliding_with_friction_penetration`: 摩擦スライドの1%制限
- `test_sliding_displacement_has_x_component`: スライドx変位検証
- `test_sliding_friction_both_converge`: 摩擦有無両方の収束検証

## ファイル変更

- `xkep_cae/contact/pair.py` — ContactConfig拡張、共有節点フィルタ
- `xkep_cae/contact/solver_hooks.py` — 適応的ペナルティ増大ロジック
- `tests/contact/test_beam_contact_penetration.py` — テスト書き換え（11→20テスト）

## 確認事項

- 既存の接触テスト226件全パス確認済み（後方互換性問題なし）
- lint/format パス（`ruff check` + `ruff format`）
- 適応的ペナルティのデフォルトパラメータ（tol=0.01, growth=2.0）は
  ContactConfig のデフォルトに組み込み済み。既存コードで ContactConfig を
  デフォルト生成する場合に自動有効化される。tol=0 で無効化可能。

## 設計上の懸念

- **ペナルティ成長係数 2.0**: 大半のケースで3-4回のOuter反復で1%を達成。
  収束が遅い場合は growth_factor を 4.0-10.0 に上げることも可能。
- **k_pen_max のデフォルト 1e12**: 構造剛性行列の条件数との兼ね合い。
  実梁要素では EA/L ベースの推定と合わせて自動調整を検討すべき。
- **共有節点フィルタ**: 現在は「共有節点がある＝同一梁」と仮定。
  異なる梁が同一節点で結合されるケースでは明示的な除外リストが必要。

## TODO

- 実梁要素（Timoshenko 3D / CR梁）でのマルチセグメント貫入テスト
- EA/L ベースの自動 k_pen 推定と適応的ペナルティの統合
- 接触付き弧長法との統合テスト
- 接触点移動の長距離スライド（複数セグメントを跨ぐ）テスト

---
