# status-075: CI修正 + adaptive omega + 7本撚りサイクリック荷重 + 大規模メッシュ検証

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-27
- **テスト数**: 1297（fast: +11 新テスト、うちslow: +10）
- **ブランチ**: claude/execute-status-todos-NjAhp

## 概要

status-074 の TODO を消化。CI フォーマット修正 + adaptive omega 実装 + 7本撚りサイクリック荷重テスト + ブロックソルバー大規模メッシュ性能検証。

## 実施内容

### 1. CI修正（lint フォーマット違反）

- `xkep_cae/contact/sheath_contact.py` の ruff format 違反を修正
- CI #143〜#145 が10秒で失敗していた原因
- `tests/contact/test_twisted_wire_contact.py` のフォーマットも修正

### 2. adaptive omega（AL乗数段階的蓄積）

**問題**: `n_outer_max > 1` かつ `omega > 0.3` で Outer loop が発散。
根本原因: 高 omega → AL乗数大更新 → 解の変動増大 → (s,t) 大変動 → 不安定

**解決策**: Outer 反復ごとに ω を段階的に増大するスケジュール。

```
ω(outer) = min(ω_min × growth^outer, ω_max)
```

デフォルト: ω_min=0.01, ω_max=0.3, growth=2.0
→ outer=0: 0.01, outer=1: 0.02, outer=2: 0.04, ..., outer=5: 0.3 (上限)

**変更ファイル**:
- `xkep_cae/contact/pair.py` — ContactConfig に adaptive_omega, omega_min, omega_max, omega_growth を追加
- `xkep_cae/contact/solver_hooks.py` — 両ソルバー（block_contact, with_contact）で対応

**テスト（+5テスト）**:
- スケジュール値検証（fast）
- 3本撚り引張・曲げ（adaptive omega + n_outer_max=3）
- 7本撚り引張（adaptive omega + n_outer_max=3）
- 固定omega比較

### 3. 7本撚りサイクリック荷重テスト（+3テスト）

ブロックソルバーでの往復荷重テスト。摩擦付き接触で loading → unloading の2フェーズ。

| テスト | 荷重 | ステップ |
|--------|------|---------|
| 引張サイクリック | 1.0N, loading→unloading | 10+10 |
| 曲げサイクリック | 0.001N·m, loading→unloading | 10+10 |
| 接触力記録確認 | 1.0N 引張 | 10 |

### 4. ブロックソルバー大規模メッシュ検証（+3テスト）

n_elems_per_strand=16（102×102ブロック）での性能検証。

| テスト | n_strands | n_elems | 総DOF |
|--------|-----------|---------|-------|
| 3本撚り引張 | 3 | 16 | 306 |
| 7本撚り引張 | 7 | 16 | 714 |
| 3本撚り曲げ | 3 | 16 | 306 |

## ファイル変更

### 新規
- `docs/status/status-075.md` — 本ステータス

### 変更
- `xkep_cae/contact/pair.py` — ContactConfig に adaptive omega 設定4フィールド追加
- `xkep_cae/contact/solver_hooks.py` — 両ソルバーで adaptive omega 対応（+35行）
- `xkep_cae/contact/sheath_contact.py` — ruff format 修正
- `tests/contact/test_twisted_wire_contact.py` — 新テスト3クラス11テスト追加（+470行）
- `README.md` — テスト数更新
- `docs/roadmap.md` — TODO チェックボックス更新
- `docs/status/status-index.md` — status-075 追加
- `CLAUDE.md` — 現在の状態更新

## 設計上の懸念・TODO

### 未解決（status-074から引き継ぎ）
- [ ] Outer loop 発散の完全解決（adaptive omega は改善策だが、Mortar離散化等の根本的対策は未実装）
- [ ] 接触プリスクリーニング GNN Step 1（データ生成）
- [ ] k_pen推定MLモデル Step 1（グリッドサーチデータ）
- [ ] PINN学習スパース行列対応

### 次ステップ候補
- [ ] adaptive omega の効果定量評価（n_outer_max=3〜5 での収束性比較）
- [ ] 7本撚りサイクリック荷重でのヒステリシスループ面積計測
- [ ] 19本撚り/37本撚りでの大規模接触テスト

### 運用上の気付き
- CI失敗が ruff format 違反だった。コミット前の `ruff format --check` を必ず実行すべき。
- `sheath_contact.py` は status-074 で新規作成されたが、フォーマット未適用のままマージされていた。

---
