# status-235: 三点曲げ dt 小問題の根本原因調査 + adaptive stepping 改善

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-25
**テスト**: 190+10s（変更なし） | 契約違反 1件 | 条例違反 0件

---

## 概要

三点曲げ計算（n_periods=30）が 1592 increment / 907 cutback / 73分かかる問題を調査。
Abaqus なら同様の計算は 1CPU 3分程度。根本原因を特定し、adaptive stepping パラメータを改善した。

---

## 根本原因分析（5つ）

### 原因1: dt_max 上限が厳しすぎる（最大インパクト）

`three_point_bend_jig.py:1156`:
```python
dt_max_fraction=min(dt_initial_frac * 16.0, 0.05)
```

- dt_initial_frac = (T1/40) / (30*T1) = 1/1200 ≈ 0.000833
- dt_max_frac = 16/1200 = **0.01333** → **最低75 increment 必要**
- Abaqus 静的: 荷重の20-30%を一度に進められる（最低3-5 increment）
- n_periods=1 では 0.05 キャップが効くため影響なし → **n_periods=30 固有のボトルネック**

### 原因2: dt growth damping が積極的すぎる

`_adaptive_stepping.py:138-149`:
- 3回連続成功以降、成長率が 2.0→1.33→1.25→...→1.1 に急減衰
- カットバック後の dt 回復に 8-10 ステップ必要（damping なしなら 4 ステップ）
- Abaqus は一律 1.5倍。damping なし。

### 原因3: 接触力変化率閾値 0.3-0.5 が敏感すぎる

`_adaptive_stepping.py:151-160`:
- 接触力 30% 変化で dt 強制縮小（成功したステップでも）
- 曲げ荷重増加に伴う自然な接触力増加で dt が縮小される

### 原因4: 動的定式化の dt 依存パラドックス

K_eff = K_struct + K_contact + (1-α_m)/(β·dt²)·M
- dt 小 → 慣性項大（正則化）→ NR 容易だが進行遅い
- dt 大 → 慣性項小 → NR 困難だが進行速い
- Abaqus static は慣性項なし → dt と NR 収束が独立

### 原因5: NR 力収束未達 → 変位収束フォールバック

- 中盤後〜終盤で tol_force=1e-6 に達せず、tol_disp=1e-8 で収束判定
- 変位収束は力の平衡を保証しない → 次ステップの初期値悪化 → 連鎖的悪化

---

## 実施した改善（Phase A: パラメータ調整）

### A1. dt_max_frac 上限緩和

- `three_point_bend_jig.py:1156`: `min(dt_initial_frac * 16.0, 0.05)` → `min(dt_initial_frac * 64.0, 0.2)`
- n_periods=30: dt_max_frac 0.01333 → 0.0533 → 最低 **19 increment**

### A2. growth damping 撤廃

- `_adaptive_stepping.py:138-149`: consecutive_good による damping 削除、一律 dt_grow_factor 適用
- カットバック後の回復が高速化

### A3. 接触力変化率閾値の緩和

- `_contact_pair.py`: dt_contact_change_threshold 0.5 → 1.0
- `_adaptive_stepping.py`: AdaptiveSteppingInput デフォルト 0.3 → 1.0
- `_unified_time_controller.py`: UnifiedTimeStepInput デフォルト 0.3 → 1.0

---

## テスト結果（n_periods=1）

| 指標 | ベースライン | 改善後 | 変化 |
|------|-----------|-------|------|
| n_increments | 255 | 239 | -6.3% |
| n_cutbacks | 173 | 184 | +6.4% |
| elapsed | 647s | 666s | +2.9% |
| contact_force_N | 202.82 | 202.47 | -0.2% |
| deflection_mm | 27.2977 | 27.2996 | ≈同等 |

### n_periods=1 で変化が小さい理由

- n_periods=1 では dt_max_frac が元々 `min(0.4, 0.05) = 0.05` → 改善後 `min(1.6, 0.2) = 0.2`
- しかし n_periods=1 のボトルネックは dt_max 制約ではなく **NR 収束性（カットバック率 77%）**
- dt_max 改善の効果は **n_periods=30 で発現**する（dt_max が 4倍に拡大）

### n_periods=30 での予想効果

- dt_max が 4倍に拡大 → 最低 increment 数が 75 → 19
- growth damping 撤廃 → カットバック後の回復高速化
- 接触力閾値緩和 → 不要な dt 縮小の防止
- **予想**: 1592 increment → 数百 increment、73分 → 20-30分

---

## 残課題（Phase B: 次セッション以降）

### B1. NR 力収束改善（原因4, 5への対策）

- n_periods=1 のカットバック率 77% は異常に高い
- 力収束未達 → 変位収束フォールバック → 次ステップ悪化の連鎖
- 対策候補:
  - 力収束参照値の改善（dynamic_ref の見直し）
  - 変位収束時の dt 制御連動（dt 成長を抑制）
  - 接触安定化ダンパー（Abaqus の自動安定化に相当）

### B2. 動的/準静的のアルゴリズム選択

- 大きな n_periods では動的定式化の dt パラドックスが顕著
- 準静的ソルバーの復活 or 自動安定化の導入を検討

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_adaptive_stepping.py` | A2: growth damping 撤廃, A3: 閾値 0.3→1.0 |
| `xkep_cae/contact/_contact_pair.py` | A3: 閾値 0.5→1.0 |
| `xkep_cae/contact/solver/_unified_time_controller.py` | A3: 閾値 0.3→1.0 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | A1: dt_max 16→64倍, 上限 0.05→0.2 |

---

## 確認事項・懸念

- n_periods=30 テストは 73分かかるため本セッションでは未実行。次セッションで検証必要。
- Phase A は低リスクだが、dt が大きくなることで NR 発散が増える可能性がある。カットバック自動回復に依存。
- Phase B の NR 収束改善が本質的解決だが、接線剛性の整合性調査が必要。
