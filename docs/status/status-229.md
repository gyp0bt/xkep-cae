# status-229: ε=0.02 完全統一で frac=0.60→0.86 達成 + Hermite ON 根本問題特定

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-24
**ブランチ**: `claude/restore-contact-normal-8HrIf`

---

## 概要

status-228 の frac=0.96 が再現不能（STAP細胞問題）であったため、
原因調査と ε 完全統一を実施。

- ε=0.02 を `_smooth_clip_01`、`_st_jacobian._SMOOTH_EPS`、`_smooth_clip_deriv` の **3箇所すべてで統一**
- 結果: **frac=0.60 → 0.86**（154N、旧ベースラインから +43% 改善）
- Hermite 中心線 ON の根本問題を特定: **st_jacobian が線形幾何前提で発散**

---

## 検証結果（再現性確認済み）

| 条件 | frac | fc [N] | incr | 時間 [s] | 備考 |
|------|------|--------|------|----------|------|
| ε=1e-6, Hermite OFF（旧ベースライン） | 0.60 | 80.8 | ~50 | ~95 | status-227 |
| **ε=0.02 統一, Hermite OFF** | **0.86** | **154.1** | 276 | 631 | **本セッション** |
| ε=0.05 統一, Hermite OFF | 0.80 | 141.8 | 237 | — | 過度平滑で精度低下 |
| ε=0.02 統一, Hermite ON | 0.03 | 16.0 | 20 | 28 | 破滅的回帰 |
| ε=1e-6, Hermite ON | 0.06 | 7.1 | 154 | 332 | 前回 revert テスト |
| max_incr=1000, ε=0.02, Hermite OFF | 0.86 | 154.1 | 276 | 622 | 予算枠ではなく真の壁 |

### テスト条件

- E=25, push=30mm, n_periods=1, max_incr=500（特記以外）
- `consistent_st_tangent=False`（K_st 無効 — デフォルト）

---

## 重要な発見

### 1. status-228 の frac=0.96 は再現不能（STAP細胞問題）

前回セッションの revert コミット（0ff7b1b）で既に指摘されていたが、
「ε=0.02統一」テストは実際には **`_st_jacobian._SMOOTH_EPS` が 1e-6 のまま不完全** だった。
commit 2387680 のメッセージに「`_smooth_clip_01=0.02, _st_jacobian=1e-6 の不整合状態`」と明記。

今回の完全統一（3箇所すべて ε=0.02）により frac=0.86 を達成。

### 2. ε=0.02 が最適値

| ε | frac | 備考 |
|---|------|------|
| 1e-6 | 0.60 | 遷移帯狭すぎ — 実質 np.clip |
| **0.02** | **0.86** | **最適** |
| 0.05 | 0.80 | 過度平滑 — 端点精度低下 |

ε=0.02 はセグメント長 O(1) に対して 2% の遷移帯。
これより広いとクランプ精度が低下し、接触力計算が不正確になる。

### 3. Hermite ON の根本問題

Hermite 中心線を有効にすると **frac=0.03 に破滅的回帰** する。
原因は ε 不整合ではなく（ε=0.02 統一済みでも発散）、以下のアーキテクチャ問題:

**問題**: UpdateGeometryProcess が Hermite 幾何で (s,t, pA, pB, normal) を計算するが、
これらの値を使う接線剛性（K_mat, K_geo）は **線形セグメント前提の微分** に基づく。

具体的には:
1. `_closest_point_hermite_refine` が Hermite 曲線上の最近接点を返す
2. この最近接点は線形セグメント上の点とは異なる位置にある
3. NR 反復で u が変化すると、Hermite 最近接点が大きくジャンプする可能性
4. 接線剛性は線形幾何の微分を使うため、残差の変化方向を正しく予測できない
5. 結果: 2-cycle 残差振動、カットバック連発、frac=0.03 で停止

**注**: `consistent_st_tangent=False`（デフォルト）なので K_st は関係ない。
問題は K_mat, K_geo 自体ではなく、Hermite refine による幾何の大幅な変化。

### 4. frac=0.86 の壁の原因

max_incr=1000 でも frac=0.86 で同一結果。NR が frac≈0.86 で不収束になる。
残差が 0.87〜0.91 に停滞し、23 active 接触ペアで振動。

**根本原因**: C0 セグメント境界での接触法線ベクトル不連続。
ε=0.02 のスムースクランプはパラメータ s の遷移を平滑化するが、
セグメント間の接線方向のジャンプ（C0 不連続）は解消できない。

---

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `xkep_cae/contact/geometry/_compute.py` | 変更 | `_smooth_clip_01` ε: 1e-6 → 0.02 |
| `xkep_cae/contact/geometry/_st_jacobian.py` | 変更 | `_SMOOTH_EPS`: 1e-6 → 0.02, `_smooth_clip_deriv` デフォルト: 1e-6 → 0.02 |
| `xkep_cae/contact/_contact_pair.py` | 変更 | `use_hermite_centerline` コメント更新 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | 変更 | `use_hermite_centerline` コメント更新 |
| `CLAUDE.md` | 変更 | フォーカスガード更新 |

---

## テスト

**100+10s passed** — 契約違反 1件（既存）、条例違反 0件
（レンダリングテスト1件は既存の環境依存失敗）

---

## 次のステップ

### 本質的課題: ComputeStJacobianProcess の Hermite 幾何対応

Hermite 中心線を活かすには、以下が必要:

1. **ComputeStJacobianProcess を Hermite 幾何に拡張**
   - 現在: 線形セグメント前提の陰関数微分 `F₁ = δ·dA = 0`
   - 必要: Hermite 曲線上の微分 `F₁ = δ·dpA/ds = 0` (dpA/ds は Hermite 接線)
   - 入力に connectivity（隣接ノード情報）と node tangent vectors が必要

2. **接触力の ∂n/∂s, ∂n/∂t を Hermite 対応に更新**
   - 現在: `∂n/∂s = (1/dist)(I - n⊗n)·dA` (dA は線形セグメント方向)
   - 必要: `∂n/∂s = (1/dist)(I - n⊗n)·dpA/ds` (dpA/ds は Hermite 接線)

3. **Hermite refine の安定性向上**
   - Gauss-Newton 近似（d²p/ds² 無視）を完全 Newton に拡張
   - 線形初期値からの収束半径を確保

### 暫定的な改善案

- ε=0.02 統一で frac=0.86 は安定して再現可能
- n_periods=30, E=25 での数百 N 確認は次のセッションで実施

---

## 運用メモ

- **STAP細胞教訓**: 全結果は tee でログ保存 + YAML 出力で再現性を担保
- **ε 完全統一**: 3箇所すべてを同一値にすることが必須。部分的な変更は不整合を生む
- **Hermite ON は危険**: st_jacobian の Hermite 対応なしでは使用不可
