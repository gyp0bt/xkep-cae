# Status 109: S3改良6-9 — 適応時間増分制御・AMG前処理・k_pen continuation・自動推定

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-05
**ブランチ**: `claude/s3-improvements-preparation-JA0QW`
**テスト数**: 2117（fast: 1664 / slow: 309 + deprecated: 144）— +16テスト

## 概要

S3（大規模NCP収束改善）に4つの新機能を追加。摩擦付きスライドテストの既存バグも修正。

## 実施内容

### S3改良6: 適応時間増分制御（adaptive_timestepping）

接触問題での収束性を改善するための動的ステップ幅制御。

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `adaptive_timestepping` | False | 有効化フラグ |
| `dt_grow_factor` | 1.5 | 収束良好時のステップ拡大係数 |
| `dt_shrink_factor` | 0.5 | 収束不良時のステップ縮小係数 |
| `dt_grow_iter_threshold` | 5 | この反復数以下で収束 → 拡大 |
| `dt_shrink_iter_threshold` | 15 | この反復数以上で収束 → 縮小 |
| `dt_contact_change_threshold` | 0.3 | active set変化率閾値 |
| `dt_min_fraction` | auto | 最小ステップ分率 |
| `dt_max_fraction` | auto | 最大ステップ分率 |

**動作**:
1. 初期ステップ幅は `1/n_load_steps`
2. 収束反復数が少ない → ステップ拡大（最大 `dt_max_fraction`）
3. 収束反復数が多い → ステップ縮小
4. active set変化が大きい → プロアクティブにステップ縮小
5. 不収束時はステップ二分法にフォールバック

**効果**: 摩擦付きスライドテスト（元々不収束）が安定して収束。

### S3改良7: AMG前処理（use_amg_preconditioner）

PyAMG Smoothed Aggregation を鞍点系ブロック前処理のK_effソルバーに導入。

- ILU代替としてAMGをオプションで使用可能
- AMG失敗時はILUに自動フォールバック
- Schur対角近似もAMGベースで計算

### S3改良8: k_pen continuation（k_pen_continuation）

ペナルティパラメータを段階的に増大させ、初期接触検出の安定化を図る。

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `k_pen_continuation` | False | 有効化フラグ |
| `k_pen_continuation_start` | 0.1 | 初期k_penスケール |
| `k_pen_continuation_steps` | 3 | 目標到達までのステップ数 |

### S3改良9: NCPソルバーk_pen自動推定

旧ソルバー（solver_hooks.py）にあったk_pen自動推定ロジックをNCPソルバーに移植。

- `k_pen_mode="beam_ei"` で `12EI/L³` ベースの自動推定
- `k_pen_mode="ea_l"` で `EA/L` ベースの自動推定
- 代表要素長は connectivity から自動計算
- `k_pen=0.0` 指定時に自動推定が発動

### omega回復メカニズム

adaptive omega が最小値に20反復以上張り付いた場合に omega_init にリセット。発振ループからの脱出を支援。

### 既存テスト修正

| テスト | 問題 | 修正 |
|-------|------|------|
| `TestSlidingContactNCP`（4テスト） | 摩擦付きスライドが不収束 | `adaptive_timestepping=True` + `max_step_cuts=3` |

## テスト数内訳

- 旧: 2101（fast: 1651+6 / slow: 374+5+65）
- 新: 2117（+16: S3テスト13 + 19本テスト3）
- fast: 1664
- slow: 309（non-deprecated）
- deprecated: 144

## 19本NCP収束状況

19本テストは依然として**未収束**。改良6-9の追加後も、力残差 `||R_u||/||f||` が O(10⁴) に発散する問題が残存。

**観察**:
- NCP残差: 当初は低い（~1e-6）がk_pen continuation後に上昇
- 力残差: O(10⁴) で発散
- active set: 90-144 の間で振動
- adaptive omega: 0.05（最小）に張り付き → 回復後に再度最小に

**次の対策候補**:
- [ ] `use_line_search=True` の19本テストへの適用
- [ ] 接触力ベースのk_pen上限制御（力バランスに基づく適応k_pen）
- [ ] 非線形アセンブラ（幾何学的非線形）の使用
- [ ] NCP鞍点系の条件数改善（スケーリング前処理）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status | 状態 |
|--------|--------|-----------|------|
| `max_step_cuts`（ステップ二分法） | `adaptive_timestepping`（適応Δt） | status-109 | adaptive_timesteppingが上位互換 |

## 確認事項

- `adaptive_timestepping` はステップ二分法を内包する上位概念。`max_step_cuts` は引き続き使用可能だが、`adaptive_timestepping=True` の使用を推奨
- AMG前処理は PyAMG が必要（オプション依存）
- k_pen自動推定は `k_pen=0.0`（デフォルト）かつ `k_pen_mode="beam_ei"` 設定時のみ発動

## 運用メモ

- 摩擦付きスライド問題では `adaptive_timestepping=True, max_step_cuts=3` を推奨
- 19本以上の収束改善はS3の継続課題

---
[← README](../../README.md)
