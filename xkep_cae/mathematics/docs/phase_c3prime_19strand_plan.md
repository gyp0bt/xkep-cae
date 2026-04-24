# Phase C-3' 19 本撚線再評価 実験計画

[← mathematics.md](mathematics.md) | [← roadmap](../../../docs/roadmap.md) | [← README](../../../README.md)

## 1. 背景 (status-357/368 時点の決着)

| 時点 | 測定対象 | frac | 主要所見 |
|------|--------|-----:|---------|
| status-344（Phase C-3' 前） | 19本撚線 K_c FD | 0.4839 | `mat_only rel_err mean = 0.44, comp_x max = 0.98` |
| status-356 gate | `test_helical_3d_hermite`（2 素線） | — | rel_err **1.795% → 2.18e-07**（5 桁改善、FD 機械精度） |
| status-357（Phase C-3' 後） | 19本撚線 K_c FD | **0.3739** | `mat_only rel_err mean = 0.508`（+15% 悪化） |

status-356 の 2 経路相殺定理（数理台帳 `docs/math/03_huber_contact_penalty.md` §7）は
**active 集合固定下**で厳密。19 本 Type D stall 断面は `D+E:67%, E:28%`
（active 集合振動支配領域）であり、仮説 A + B の解析的相殺が成立しない。

status-358〜368 で症状緩和候補 (a)/(a')/(c)/(d)/(e) を全て実測したが、
いずれも frac=1.0 未達で MCDD 凍結解除条件を満たせず、**候補 (f) Phase C-3'
s-tracking 19 本再評価が MCDD 本命として残存**。

## 2. 問題の再定式化

status-355/356 で確立した 2 経路 (i)/(ii) の解析:

- 経路 (i)（直接 Hermite 接線）: `K_hermite_adj = w_mat·n⊗n − w_geo·I_nn`
- 経路 (ii)（s-tracking 射影補償）: `K_closest / K_st` の active×adj ブロック拡張

**active 集合変動領域で何が変わるか**:

(a) `w_mat = p_n'·δ^{m-1}`, `w_geo = p_n/d` の両者は `Heaviside(g>0)` 由来の
    「接触/非接触」切替で**不連続**。Huber smoothing（`delta_h`）は `p_n` の
    g=0 近傍を C^1 にするが、**active 集合そのものの判定（`g > 0` vs `g ≤ 0`）**
    は依然離散的。

(b) NR 反復内で active 集合が振動すると、i 反復目と i+1 反復目で `K_hermite_adj`
    の support 集合が変化 → 相殺が成立する「同一 2 経路」の仮定が破れる。

(c) したがって 19 本 Type D stall は **active-flip による項切替の非整合**
    が本質で、項を増やすだけでは解けない（K_hermite_adj や K_closest を
    active 変化時にどう「履歴平滑化」するかの問題）。

## 3. 次 status で実施する 2 ステップ実験

### Step 3.1: active 変動下診断（所要 ~30 分）

**目的**: 既存 Phase C-3' 実装（仮説 A + B）が active 集合変化点で破綻する
ことを数値的に実証する。

**方法**:

1. `work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` を改造し、2 素線
   シナリオに「active 変動 perturbation」（1 ペアを active ↔ inactive の
   境界 $g = \\pm\\epsilon$ に置く）を注入、FD 整合性を再測定。
2. 期待結果:
   - 境界から離れた `g = -1.0`（深い接触）: rel_err ≈ 2.18e-07（status-356 再現）
   - 境界近傍 `g = \\pm 10^{-3}`: rel_err が 2 桁以上悪化すれば仮説裏付け

**出力**: `14_kc_active_boundary_diagnostic.py` のレポート（20 行テーブル）。

### Step 3.2: 6 項目化の設計判断（所要 ~2 時間）

Step 3.1 の結果に応じて以下のいずれか:

- **結果 A: active 境界で rel_err 悪化** → 新項 `KcActiveFlipStiffness` を
  `TermExpansionContract` に 6 項目として追加。`d p_n / d u_Heaviside` の
  smoothing 補正項（Huber の 2 階微分相当）を `HuberContactForceProcess.tangent()`
  で評価、`term="active_flip"` で分配。設計仕様を
  `docs/math/03_huber_contact_penalty.md` §9（新設）に記述。
  - 実装規模見積もり: ~200 行（Process 本体 150 + 配線 30 + テスト 40）
  - gate: `test_kc_active_boundary_fd.py`（新規）で rel_err < 1e-4

- **結果 B: active 境界でも rel_err 健全** → 問題は項の欠落ではなく NR
  アルゴリズム側（active 判定の履歴平滑化、low-pass、あるいは
  augmented-Lagrangian-like 再導入）。候補 (g) として別ラインで再計画。

## 4. MCDD 脱法回避チェックリスト

本計画実施時に以下パターン（CLAUDE.md 記載）を避ける:

- [ ] パターン 1: tol 事後緩和禁止。Step 3.2 のテスト tol は機械精度 1e-4 基準。
- [ ] パターン 4: 既存項の rename 禁止。新 Process は純粋に新設。
- [ ] パターン 5: 既存 12 テストの skip/xfail 禁止（特に status-356 で
  機械精度達成済の `test_helical_3d_hermite`）。
- [ ] パターン 6: 骨格だけの status 禁止。Step 3.1 + Step 3.2 を 1 status
  で完結、あるいはコンテキスト不足時は中断スナップショットで正規手順。

## 5. gate 基準

本計画が status-370+ で完了したと宣言できる条件:

1. `test_helical_3d_hermite` の FD rel_err が引き続き < 1e-5（status-356 維持）
2. 新 gate テスト（active 境界版）で rel_err < 1e-4
3. 19 本撚線 K_c FD（`13_kc_component_fd_19strand.py`）で `mat_only rel_err
   mean < 0.25`（現行 0.508 の半分以下）
4. 19 本撚線 90° 曲げで `frac ≥ 0.8`（凍結解除まで最後 0.2 は別候補（g））

上記 4 条件を揃えて status-357 の `mat_only rel_err < 1e-2` 目標に接続する。

## 6. 関連

- 数理台帳: [`docs/math/03_huber_contact_penalty.md`](../../../docs/math/03_huber_contact_penalty.md) §7（status-356 再構成）
- 実装: `xkep_cae/contact/contact_force/strategy.py` の
  `KcHermiteNonlocalStiffnessProcess` / `KcClosestPointStiffnessProcess`
  / `ContactForceStStiffnessProcess`
- 過去 status: [status-354](../../../docs/status/status-354.md)（仮説 A 反証）
  / [status-355](../../../docs/status/status-355.md)（仮説 B 診断）
  / [status-356](../../../docs/status/status-356.md)（仮説 A+B 同時導入で機械精度）
  / [status-357](../../../docs/status/status-357.md)（19 本退化検出）
  / [status-368](../../../docs/status/status-368.md)（候補 (d) クローズ、(f) に戻る決定）
