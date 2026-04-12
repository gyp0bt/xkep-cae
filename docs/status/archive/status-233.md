# status-233: SDI 排除 — 全候補ペア評価 + 力ベース dt 制御 + g_off ワイド化

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-24
**ブランチ**: `claude/improve-contact-solver-ywgCm`

---

## 概要

三点曲げ解析で接触状態変化が SDI（Severe Discontinuous Iteration）として扱われ、
dt が極小化する問題を構造的に解決。

**SDI とは**: ABAQUS の収束問題分類用語。接触 ON/OFF の急変、大きなスリップ反転、
摩擦 stick/slip 遷移などで NR が不安定化する現象。ABAQUS では SDI 検出時に
特別な収束判定を適用する。

**本質的洞察**: 三点曲げでは接触点は大半の局面で動かず、
動くにしても「面積増加」か「スライド」。これは本質的に SDI ではない。
SDI でなくなるような接触ソルバー設計を導入した。

---

## 変更内容

### 1. INACTIVE skip 除去（全候補ペア評価）

| ファイル | 変更 |
|----------|------|
| `contact_force/strategy.py` evaluate() | `if INACTIVE: continue` を除去 |
| `contact_force/strategy.py` tangent() | 同上 |
| `friction/_assembly.py` 3箇所 | `if INACTIVE: continue` を除去 |

**原理**: Huber penalty は gap > 0 で自然に p_n=0 を返す。
バイナリ ACTIVE/INACTIVE ゲーティングが力の不連続の原因だった。
全候補ペアに Huber を適用することで、離散的状態遷移を排除。

**摩擦力**: `_friction_return_mapping_loop()` は元々 `p_n > 0` でフィルタしており、
INACTIVE check は冗長だった。tangent/geometric stiffness の 3 箇所は
dict lookup (`pair_idx not in friction_tangents`) で十分にフィルタされる。

### 2. adaptive stepping の接触反応を力ベース指標に置換

| ファイル | 変更 |
|----------|------|
| `_adaptive_stepping.py` | `AdaptiveStepInput` に `contact_force_norm`, `prev_contact_force_norm` 追加 |
| `_adaptive_stepping.py` `_on_success()` | n_active 変化率 → `|Δ‖f_c‖|/‖f_c‖` に置換 |
| `_unified_time_controller.py` | `TimeStepQueryInput` に同フィールド追加、伝播 |
| `process.py` | SUCCESS 呼び出しで `f_c` ノルムを渡す |
| `_solver_state.py` | `prev_contact_force_norm` 追加 |

**旧**: `Δn_active / max(n_prev, n_cur, 1) > 0.3` → dt × 0.5（離散的カウント）
**新**: `|Δ‖f_c‖| / ‖f_c‖ > 0.5` → dt × 0.5（連続量）

### 3. g_off ワイド化

| パラメータ | 旧 | 新 |
|-----------|-----|-----|
| `g_off` | 1e-6 mm | 0.1 mm |
| `dt_contact_change_threshold` | 0.3 | 0.5 |

**g_off 拡大の効果**: ワイヤ半径 8.5mm の約 1.2%。
一度 ACTIVE になったペアが微小なギャップ開きで即座に INACTIVE にならない。
broadphase 候補の安定化に寄与。

---

## テスト結果

```
190 passed, 10 skipped, 0 failed（render テスト除外）
ruff check: All checks passed
ruff format: 全ファイルフォーマット済み
```

既存テストの regression なし。

---

## 設計判断

### なぜ INACTIVE skip 除去が安全か

1. **broadphase** が候補ペアを AABB + midpoint prescreening で絞り済み
2. **Huber penalty** は gap > 0 で p_n=0 を返す → 遠いペアは自動的に寄与ゼロ
3. **`p_n <= 1e-30` チェック** (evaluate L215) が自然にスキップ → 計算量増加は限定的
4. **摩擦**: `_friction_return_mapping_loop` の `p_n > 0` フィルタが機能

### なぜ力ベース指標か

- n_active は離散量 → 1 ペアの ON/OFF で 30% 変化しうる（少数ペア時）
- 接触力ノルムは連続量 → スムースな接触面積変化は小さな変化率
- 三点曲げの面積増加やスライドは力の変化が緩やか → dt 縮小不要

---

## TODO

- [ ] n_periods=30 テスト実行で dt 改善・frac=1.0 到達を確認
- [ ] frac=1.0 到達後の荷重が数百 N であることの確認
- [ ] 計算時間のパフォーマンス比較（全候補ペア評価のオーバーヘッド測定）

---

## 確認事項（次セッションへ）

- `_update_active_set_state()` は診断用に残存。n_active のログ出力は従来通り。
- `freeze_geometry_in_nr` / `freeze_active_set` の挙動は変更なし。
- Huber の smoothing_delta は現状維持（5000/r_min ≈ 588）。
  今後 Huber 遷移幅拡大を検討する場合は別 status で対応。
- 開発運用: SDI は ABAQUS の用語。xkep-cae の status/doc で用いる場合は
  「SDI（Severe Discontinuous Iteration, ABAQUS 用語）」と注記すること。

---
