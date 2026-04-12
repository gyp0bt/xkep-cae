# status-236: Phase A 完全無効化 — n_periods=30 で逆効果を確認

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-25
**テスト**: 190+10s（変更なし） | 契約違反 1件 | 条例違反 0件

---

## 概要

status-235 で実施した Phase A（adaptive stepping パラメータ改善）を **完全リバート**。
n_periods=30 テストで逆効果が判明したため。

---

## Phase A の n_periods=30 テスト結果

| 指標 | ベースライン (status-234) | Phase A 改善後 |
|------|--------------------------|----------------|
| 到達 frac | **1.0**（完了） | **0.24**（24%で壁） |
| n_increments | 1592 | 368（タイムアウト） |
| n_cutbacks | 907 | 361（**98%カットバック**） |
| contact_force_N | 208.6 | — |
| elapsed | 73min | 10min タイムアウト（未完了） |

### 壁の症状

- Incr 1-4: frac 0.05→0.08→0.13→0.19 — 大ステップで順調（energy converged）
- **Incr 5 (frac=0.24)**: active=35 に急増、NR 力残差 ~7e-5 で停滞（25 反復）
- Incr 6 以降: 全て cutback → sub-step → disp converged の無限ループ
- frac が 0.0001/increment ずつしか進まない

### 失敗の原因分析

1. **dt_max 緩和が大きすぎた**: 初期ステップで frac=0.053（5.3%）と巨大ステップ → 接触遷移帯を飛び越える
2. **growth damping 撤廃が裏目**: カットバック後の dt 回復が速すぎて再び壁にぶつかる
3. **ベースラインとの比較**: ベースラインは小さな dt で慎重に接触遷移帯を通過できていた。Phase A は壁を超えられない
4. **本質的問題は NR 収束性**: dt を大きくすると慣性正則化が弱まり、接触非線形性が露出する（原因4）。dt パラメータ調整だけでは解決不可能

### 教訓

- adaptive stepping のパラメータ最適化は **NR 収束性の改善なしには効果がない**
- dt_max を緩和しても、NR が力収束できない限り全てカットバックされる
- n_periods=1 では影響が小さかった（-6.3% increment, +2.9% time）が、n_periods=30 では壊滅的

---

## リバートした変更

| ファイル | リバート内容 |
|---------|-------------|
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | dt_max: `64.0, 0.2` → `16.0, 0.05` |
| `xkep_cae/contact/solver/_adaptive_stepping.py` | growth damping 復元 + 閾値 1.0 → 0.3 |
| `xkep_cae/contact/_contact_pair.py` | 閾値 1.0 → 0.5 |
| `xkep_cae/contact/solver/_unified_time_controller.py` | 閾値 1.0 → 0.3 |

---

## 次の課題（更新）

Phase A（パラメータ調整）は失敗。**Phase B（NR 力収束改善）が唯一の道**。

| 優先度 | 課題 | 概要 |
|--------|------|------|
| 1 | NR 力収束改善 | カットバック率 77% の根本原因対策。接線剛性整合性、dynamic_ref 見直し |
| 2 | 摩擦アセンブリ Hermite 完全対応 | use_hermite=False デフォルト状態の解消 |
| 3 | 接触安定化ダンパー | Abaqus の自動安定化に相当する仕組み |

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_adaptive_stepping.py` | Phase A リバート（growth damping 復元 + 閾値復元） |
| `xkep_cae/contact/_contact_pair.py` | Phase A リバート（閾値復元） |
| `xkep_cae/contact/solver/_unified_time_controller.py` | Phase A リバート（閾値復元） |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | Phase A リバート（dt_max 復元） |
| `docs/status/status-236.md` | 本ステータス |
| `docs/status/status-index.md` | 更新 |
| `docs/roadmap.md` | Phase A 失敗記録 + TODO 更新 |
| `README.md` | 最新ステータス参照更新 |
