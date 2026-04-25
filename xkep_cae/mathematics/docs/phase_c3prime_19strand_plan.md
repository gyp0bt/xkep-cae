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

### Step 3.1: active 変動下診断（status-370 で実施完了）

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

**実測結果**（status-370, 2026-04-24）: **結果 B 確定**。

| Block | 条件 | worst rel_err |
|-------|------|:-------------:|
| 1 | δ_h=0, gap ∈ [-1e-2, -1e-6] | **2.19e-07** |
| 2 | δ_h=5 (smoothing_delta=2000), gap ∈ [-1e-3, +5e-4]（平滑化ゾーン全域跨ぎ） | **2.20e-07** |
| 3 | 強制 flip (fd_eps ≥ \|gap\|): gap=-5e-8/eps=1e-7, gap=+1e-8/eps=1e-7 | **2.20e-07** |
| 3 | 強制 flip (eps=1e-4 大きめ): gap=-1e-5 | 2.19e-04（= `eps` 由来 FD truncation、K_c 不整合ではない） |

20 測定点全てで rel_err が status-356 の機械精度 2.18e-07 に張り付いた。
active 境界跨ぎ・平滑化ゾーン内遷移・flip 強制のいずれでも K_c 解析値は FD
と一致。Phase C-3' 実装（仮説 A + B 同時導入）は **active 集合固定下限定**
どころか **active flip を含む 2 素線設定全域で FD-整合**。

**帰結**: 19 本 Type D stall の主因は **K_c 項の欠落ではなく NR alg 側の
動力学**（反復間 active 振動、pair 間相互作用、摩擦活性切替）である。
Step 3.2 は下記の通り「結果 B」分岐で確定。

### Step 3.2: 結果 B 分岐で候補 (g) へ（status-371+ 継続）

Step 3.1 の結果 B 確定により、**新項追加は不要**。問題は項の欠落ではなく
NR アルゴリズム側であり、候補 (g) を以下の 3 サブラインで 1 つずつ
検証していく:

- **(g1) active 履歴平滑化**: 反復間で active 集合を low-pass 化。`p_n_smooth
  = α·p_n_new + (1-α)·p_n_prev` で NR を安定化。`HuberContactForceProcess`
  に `active_ema_alpha` を追加、NR ソルバー側で反復ごとに勘定。α=0.3 程度。
- **(g2) augmented Lagrangian 再導入**: status-221 で凍結した Uzawa ループを
  **外側ループ 1〜2 回**に制限して再導入。N-R 反復内では固定 λ で解き、
  外側で λ 更新。K_c の非一意性を λ で吸収する古典的アプローチ。
- **(g3) pair-wise relaxation**: pair level の trust region / relaxation を
  加える。status-284 の接触凍結モードを pair granularity に拡張して、
  チャタリング pair だけを freeze。

**実装優先度**: (g1) → (g3) → (g2)。(g1) が最小実装で効果が見込める（~100 行
+ NR 側 plumb-through ~30 行）。(g2) は拡大ラグランジアン凍結の根拠
（status-221）に矛盾しないか数理台帳で再確認が必要。

**gate 基準（全 g1/g2/g3 共通）**:
- 7 本撚線 90° frac=1.0 維持（status-299/336 baseline）
- 19 本撚線 90° frac ≥ 0.6（status-357 baseline 0.3739 の 60% 改善）
- `test_helical_3d_hermite` rel_err < 1e-5 維持（status-356 機械精度）
- 候補間は competitive、効果の薄い候補は即 close

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

## 4'. solver_mode 併存方針（status-373 追加）

19 本 Type D stall 解消の本命候補 (g3) に加え、リスタート解析方式（CLAUDE.md
status-345 まで「次の課題」に記載されていた I/O リファクタリング）を **opt-in
の `solver_mode` フラグ**として現行陰解法と併存させる。default は陰解法のまま、
リスタート方式は明示的 opt-in でのみ有効化。

### 4'.1 切替境界の I/O 契約

- **入力**: 動的摩擦接触ソルバーは `(u, v, a, 接触ペア)` を初期条件として
  受け取る Process I/O を持つ
- **出力**: 同型 `(u, v, a, 接触ペア)` を返す
- **境界条件**: 曲げ・揺動は薄いラッパーで `BoundaryCondition` を渡すのみ
- **`update_reference` 跨ぎなし**: CR 梁 UL の `f_int=0` 問題を構造的に回避
  （status-330 TL 定式化と同方針、解析ステップ単位でリスタート可能）

### 4'.2 設定 API（案、status-374 以降で実装）

```python
StrandBendingOscillationConfig(
    solver_mode="implicit",  # default; "restart" で opt-in
    ...
)
```

### 4'.3 候補 (g3) との関係

- 候補 (g3) `PairwiseFreezingProcess` は陰解法側の改善（NR 反復内）
- `solver_mode="restart"` は **解析ステップ間** のリスタート I/O 整備
- 両者は直交、同時 opt-in 可。ただし default は両方 OFF
- (g3) で 19 本 frac=1.0 達成できればリスタート方式は subsequent 高速化として
  位置づけ、達成できなければ I/O 整備が次の本命候補

### 4'.4 status-373 時点の進捗

- 設計のみ。実装は status-374 以降
- `docs/roadmap.md`「撚線規模別 opt-in チューニング」表に `solver_mode` 行を
  追加済み（status-368 `chattering_freeze_nr_max=30` / status-372
  `active_ema_alpha=0.5` と同レイヤ）

## 6. 関連

- 数理台帳: [`docs/math/03_huber_contact_penalty.md`](../../../docs/math/03_huber_contact_penalty.md) §7（status-356 再構成）
- 実装: `xkep_cae/contact/contact_force/strategy.py` の
  `KcHermiteNonlocalStiffnessProcess` / `KcClosestPointStiffnessProcess`
  / `ContactForceStStiffnessProcess`
- 診断スクリプト: `work/beam_hysteresis/14_kc_active_boundary_diagnostic.py`（status-370 新設）
- 過去 status: [status-354](../../../docs/status/status-354.md)（仮説 A 反証）
  / [status-355](../../../docs/status/status-355.md)（仮説 B 診断）
  / [status-356](../../../docs/status/status-356.md)（仮説 A+B 同時導入で機械精度）
  / [status-357](../../../docs/status/status-357.md)（19 本退化検出）
  / [status-368](../../../docs/status/status-368.md)（候補 (d) クローズ、(f) に戻る決定）
  / [status-370](../../../docs/status/status-370.md)（Step 3.1 完了、結果 B 確定、候補 (g) へ）
