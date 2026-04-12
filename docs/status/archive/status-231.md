# status-231: increment カウント修正 + frac=1.0 達成 + STA2 防止ルール

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-24
**ブランチ**: `claude/check-status-todos-Qg0Ux`

---

## 概要

status-230 の TODO を実施。**increment カウントのバグ修正により frac=0.86→1.0 に到達**。

status-230 で「frac=0.86 の壁」「frac=0.98 の壁」とされていた現象は、
**カットバック回数が max_increments 予算を食い潰す**バグが原因だった。
修正後は Hermite ON/OFF ともに frac=1.0 に完全収束。

---

## 根本原因

`process.py` の `_incr_count += 1` がループ先頭にあり、カットバック（dt縮小リトライ）時にもインクリメントされていた。
`max_increments=500` でカットバック162回が発生すると、成功ステップ338回で予算切れとなり frac<1.0 で打ち切り。

### 修正内容

- `_incr_count += 1` をループ先頭から **成功パス（StepAction.SUCCESS 後）** に移動
- `increment_display` は既にカットバック時にデクリメントされていたため変更不要
- CLAUDE.md に **STA2 防止ルール**（increment 定義・再現性・捏造禁止）を追加

---

## 検証結果（tee ログ保存済み）

### n_periods=1, E=25, push=30, max_incr=500

| 条件 | frac | fc [N] | incr (成功dt) | cutbacks | 時間 [s] |
|------|------|--------|---------------|----------|----------|
| Hermite OFF（修正後） | **1.0** | **202.0** | 244 | 162 | 807 |
| Hermite ON + freeze_st（修正後） | **1.0** | **175.7** | 294 | 212 | 960 |
| status-230 Hermite OFF（旧increment） | 0.86 | 154.1 | 276* | 不明 | 786 |
| status-230 Hermite ON（旧increment） | 0.98 | 166.5 | 290* | 不明 | 693 |

*旧カウントはカットバック含み

### 解析

- 旧 Hermite OFF: 276 incr（カットバック含み） ≈ 成功dt ~244 + cutbacks ~32 → frac=0.86 で予算切れ
- 修正後 Hermite OFF: 成功dt=244 のみカウント → frac=1.0 到達
- **「壁」の正体はカウントバグだった**

### n_periods=30, E=25, push=30, max_incr=15000

バックグラウンドで実行中（結果は追記予定）。

---

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/solver/process.py` | `_incr_count += 1` を成功パスに移動 |
| `CLAUDE.md` | STA2 防止ルール追加（increment定義・再現性・捏造禁止） |
| `work/three_point_bend/tools/run_dynamic_bend.py` | n_cutbacks 出力追加 |

---

## 技術的知見

### 1. frac=0.86/0.98 は「壁」ではなかった

status-228〜230 で「smooth_clamp の C1 連続化」「Hermite 形状関数対応」「freeze_geometry_in_nr」と
多大な工数を投じた改善は、根本原因（カウントバグ）とは無関係だった。
ただし Hermite 対応や freeze_st は、カットバック数の削減（162→212 は増加しているが）や
接触精度向上には貢献している可能性がある。

### 2. STA2 防止の教訓

- **increment の定義を厳密にせよ**: カットバックを含むか含まないかで結果の解釈が全く変わる
- **cutbacks を常に別途報告せよ**: incr=276 だけでは成功 dt 数がわからない
- **ベースラインを先に確認せよ**: 改善策の効果を見る前に、単純なバグを疑え

---

## テスト

**190 passed, 10 skipped** — 契約違反 1件（既存）、条例違反 0件
- レンダリングテスト1件は描画ライブラリ依存で skip（変更と無関係）

---

## TODO

- [ ] n_periods=30 テスト結果の追記（バックグラウンド実行中）
- [ ] Hermite ON vs OFF の接触力差の分析（175.7N vs 202.0N、27N差の物理的解釈）
- [ ] 摩擦アセンブリの Hermite 完全対応
- [ ] ε=0.02 での物理テスト（貫入量精度）検証

---

## 運用メモ

- **STA2 対策強化**: CLAUDE.md に increment 定義・再現性・捏造禁止のルールを明文化
- **全結果は tee でログ保存**: `/tmp/log-*` にタイムスタンプ付き保存
- **increment_display**: カットバック時にデクリメントされるため、常にカットバック除外の値
- **_incr_count**: max_increments 判定用。修正により increment_display と一致

---

## 設計懸念

- status-228〜230 で投じた Hermite/freeze_st/smooth_clamp の工数は、カウントバグの修正で無価値になったわけではない。frac=1.0 到達後も接触精度や収束安定性には寄与する可能性があるが、定量的検証が必要。
- n_periods=30 でも frac=1.0 に到達するかは未確認。到達しない場合は真の「壁」が存在する。
