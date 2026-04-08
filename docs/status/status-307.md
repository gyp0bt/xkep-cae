# status-307: ソルバー診断ログ強化 — カットバック原因タグ・f_ref出力・収束型統計

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-08
- **ブランチ**: `claude/check-status-todos-fY8HV`
- **テスト数**: 442+20 passed（既存テスト全合格）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

ソルバーの診断ログ出力を体系的に強化。判断が曖昧になりやすい6項目を改善し、収束問題の原因特定を直接的にする。

---

## 問題と対策

### 1. カットバック原因が不明 → `[CUTBACK:原因]` タグ

**Before**: `Adaptive dt retry: frac 0.40 → sub-steps`
**After**: `[CUTBACK:nr_limit] frac 0.4000, dt=2.5000e-02 → sub-steps (cutback #3)`

| 原因タグ | 意味 | 対策 |
|----------|------|------|
| `nr_limit` | NR反復上限到達 | max_nr_attempts増加 or 接線精度改善 |
| `diverged` | 残差発散検知 | 接線不整合調査(FD診断) |
| `relax_fail` | リラクゼーション失敗 | 凍結モード調整 |
| `solve_fail` | 線形ソルバ失敗 | 条件数調査 |

`DynamicStepOutput` に `failure_reason` フィールド追加。dt値も同時出力。

### 2. f_refが不可視 → `[f_ref]` 出力

**Before**: `||R_t||/||f|| = 3.2e-04` (||f||の値不明)
**After**: `[f_ref] Incr 42: f_ref=1.234e+03 (dynamic_ref), atol=1.23e-05`

NR初回反復で判定モード（dynamic_ref/f_ext）と絶対許容値を出力。

### 3. 収束型統計がない → `[収束型統計]` サマリ

**After（解析完了時）**: `[収束型統計] force=280(93%), disp=22(7%), total=302`

`DynamicStepOutput` に `convergence_type` フィールド追加。process.pyでインクリメント別にカウント。

### 4. 被膜圧縮統計がない → `[coat]` 定期出力

**After（50ステップごと）**: `[coat] incr=50: n_active=12, mean=42%, max=89%, n_penetrated=0`

芯線貫入検知時（n_penetrated > 0）は即時出力。

### 5. NR転換点を見逃す → `[SPIKE]` 即時出力

**Before**: 5反復ごとの定期出力（att=10,15）で att=12の急発散が不可視
**After**: 残差が前回比10倍以上増加した時点で即時出力 `[SPIKE]` タグ付き

### 6. CLAUDE.mdにログ規約追加

「ソルバー診断ログ規約」セクションを追加し、必須出力項目と設計原則を明文化。

---

## 変更ファイル

- `xkep_cae/contact/solver/_newton_dynamic.py`:
  - `DynamicStepOutput` に `convergence_type`, `failure_reason` フィールド追加
  - NR各終了パスで `_convergence_type`, `_failure_reason` を設定
  - `[f_ref]` 出力（NR初回反復時）
  - `[SPIKE]` 検知（残差10倍増で即時出力）
- `xkep_cae/contact/solver/process.py`:
  - `[CUTBACK:原因]` タグ + dt値出力
  - `[coat]` 被膜圧縮統計（50ステップごと + 貫入時即時）
  - `[収束型統計]` サマリ（正常終了時）
  - `_conv_type_counts` カウンタ追加
- `CLAUDE.md`: 「ソルバー診断ログ規約」セクション追加

---

## 再現手順

```bash
# ブランチ
git checkout claude/check-status-todos-fY8HV

# テスト
python -m pytest xkep_cae/contact/coating/tests/ -v

# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/

# 契約チェック
python contracts/validate_process_contracts.py

# 実際のログ出力確認（slow: ~200s）
# python contracts/verify_barrier_coating_90deg.py 2>&1 | tee /tmp/log-307-$(date +%s).log
```

---

## ログ出力例（期待される改善後フォーマット）

```
  [f_ref] Incr 1: f_ref=1.234e+03 (dynamic_ref), atol=1.23e-05
  Incr 1 (frac=0.0250), attempt 0, ||R_t||/||f|| = 1.000e+00, ||R_r|| = 0.000e+00, active=5
  Incr 1 (frac=0.0250), attempt 5, ||R_t||/||f|| = 3.21e-04, ||R_r|| = 1.2e-06, rate=0.850, active=5
  Incr 1 (frac=0.0250), attempt 7, ||R_t||/||f|| = 8.1e-09, ||R_r|| = 2.3e-10 (force converged, 5 active)
  ...
  Incr 42 (frac=0.4000), attempt 3 [D.stall] [SPIKE], ||R_t||/||f|| = 5.3e-01, rate=12.500, active=8
  ...
  [CUTBACK:diverged] frac 0.4000, dt=2.5000e-02 → sub-steps (cutback #3)
  ...
  [coat] incr=50: n_active=12, mean=42%, max=89%, n_penetrated=0
  ...
==================================================
  エネルギー収支サマリ
==================================================
  ...
  被膜エネルギー: 1.234e+02
  被膜/総エネルギー比: 0.0034 (0.34%)
==================================================
  [収束型統計] force=280(93%), disp=22(7%), total=302
```

---

## TODO

- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] 高速化フェーズ: 接触ペア検出KD-tree化
- [ ] シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- [ ] リスタート解析方式への移行
- [ ] slowテスト実機実行でログ出力フォーマット確認

---

## 次の担当者向け

### 重要ポイント

1. **ログ規約がCLAUDE.mdに明文化された**: 新機能追加時もこの規約に従う
2. **DynamicStepOutput拡張**: `convergence_type`（"force"/"disp"/"energy"）と `failure_reason`（"nr_limit"/"diverged"/"relax_fail"/"solve_fail"）が利用可能
3. **被膜統計は50ステップ刻み**: 毎ステップ出力は冗長なため。芯線貫入は即時
4. **[SPIKE]は10倍閾値**: 閾値の調整が必要な場合は `_newton_dynamic.py` の `_spike_detected` 条件を変更

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: ログフォーマットの改善のみ、数値計算変更なし
- [x] **再現手順記載**: コマンド列を明記
- [x] **回帰なし**: 全50テスト合格、契約違反0件
