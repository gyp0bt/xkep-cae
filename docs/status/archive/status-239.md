# status-239: Levenberg-Marquardt 正則化 + 回転慣性異方性調査

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 190+10s+8+9 | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

status-238 の TODO「K_st の安全な有効化: LM 正則化」を実施。
加えて、梁の回転慣性異方性が NR 収束に影響するかを調査。

---

## 1. 梁の回転慣性異方性の調査

### 質問

> 梁の回転慣性に異方性があるので梁の回転に対してNRが暴れてるってことはありますか？

### 結論: 回転慣性異方性は主因ではない

**根拠**:

1. **円形断面 (wire) では Iy = Iz**: 曲げ方向の回転慣性は等方的
2. **質量行列の K_eff への寄与は微小**: 動的解析の有効剛性 K_eff = K + c0·M で、
   回転 DOF の c0·M 寄与（~14 N·mm/rad）は剛性寄与（~1.3e8 N·mm/rad）の 1e-7 倍
3. **主因は K_st 不整合** (status-238): 接線剛性の 100% FD 不整合が力収束 0 件の根本原因

### ただし発見した関連問題

| 問題 | 箇所 | 影響 |
|------|------|------|
| **収束判定の次元混合** | `_newton_steps.py:211` | `np.linalg.norm(R_u)` が力 [N] とモーメント [N·mm] を混合 |
| **接触力が回転 DOF 不参加** | `contact_force/strategy.py:220` | `for d in range(3)` で並進のみ |
| **剛性スケール差** | `_beam_cr.py` | EA/L ≈ 1.8e6 vs 4EI/L ≈ 1.3e8（単位違い含め 6 桁差） |

---

## 2. Levenberg-Marquardt 正則化の実装

### 設計

**Marquardt 型**: `K_reg = K_T + λ · diag(max(|K_T_ii|, ε))` — DOF スケール差を自動吸収。

**適応 λ 制御**:
- 残差改善時: λ *= 0.5（下限 = lm_lambda_init）
- 残差連続増加（発散検知）時: λ *= 10（上限 = lm_lambda_max）
- 線形ソルブ失敗時: λ をリトライ（最大 λ まで）

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_steps.py` | `LinearSolveInput.lm_lambda` 追加、LM 正則化実装 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | `NewtonDynamicInput.lm_lambda_init/adaptive/max` 追加、適応 λ 制御 |
| `xkep_cae/contact/solver/process.py` | `ContactFrictionProcess` に LM パラメータ伝搬 |
| `xkep_cae/core/data.py` | `ContactFrictionInputData.lm_lambda_init/adaptive` 追加 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `DynamicThreePointBendContactJigConfig` に K_st + LM パラメータ追加 |
| `tests/contact/test_lm_regularization.py` | **新規**: LM 正則化の単体テスト 9 件 |
| `contracts/check_kst_lm.py` | **新規**: K_st + LM 効果検証スクリプト |
| `contracts/check_kst_lm_unfrozen.py` | **新規**: freeze_geometry と K_st の相互排他性検証 |

### テスト結果

9 件全パス:
- λ=0 で従来動作と同一
- λ>0 で変位が小さくなる（信頼領域制約）
- 不定値行列 + LM で降下方向を生成
- 特異行列 + LM で正則化成功
- Marquardt スケーリングが DOF スケール差を吸収
- BC 適用は LM 正則化後
- 大 λ で最急降下方向に収束

---

## 3. 重要発見: freeze_geometry_in_nr と K_st は相互排他

### 問題

`freeze_geometry_in_nr=True` は NR 内で接触点パラメータ s,t を凍結する。
`consistent_st_tangent=True` (K_st) は s,t の変位依存性を接線剛性に含める。

**両者を同時有効化すると接線が不整合**:
- K_st は s,t が動く前提の力変化を予測
- 実際には s,t は凍結されているので、予測された力変化は起きない
- 結果: 接線が実際の残差変化と矛盾し、NR 品質が悪化

### 検証結果（三点曲げジグ、E=25、n_periods=3、max_incr=50）

| 構成 | frac | incr | cutback | cb% | time |
|------|------|------|---------|-----|------|
| freeze=T, K_st=OFF (default) | 0.072 | 50 | 33 | 39.8% | 46s |
| freeze=F, K_st=OFF | 0.067 | 50 | 28 | 35.9% | 45s |
| freeze=F, K_st=ON, LM=ON | 0.073 | 50 | 46 | 47.9% | 77s |

### 解釈

- **freeze=F, K_st=OFF** が cutback 最少 (28) — 幾何更新自体は有益
- **K_st + LM** は発散しない（status-238 からの改善）が、E=25 軟材料では cutback が増加
- E=200e3 (鉄鋼) での本格評価が必要

### 正しい組合せ

| freeze_geometry | consistent_st_tangent | 整合性 |
|-----------------|----------------------|--------|
| True | False | ✓ 整合（凍結 + 不整合接線 → 修正 Newton） |
| False | True | ✓ 整合（更新 + 整合接線 → 完全 Newton） |
| True | True | **✗ 不整合（凍結 + 整合接線 → 矛盾）** |
| False | False | △ 動作するが最適でない |

---

## TODO

- [ ] **E=200e3 (鉄鋼) での K_st + LM 効果検証**: 軟材料 E=25 では差が小さい。n_periods=30 の鉄鋼モデルで力収束改善を確認
- [ ] **freeze_geometry_in_nr=False + K_st=True + LM の本格評価**: 正しい組合せでの NR 収束速度改善を定量評価
- [ ] **LM λ の初期値チューニング**: 1e-2 は過大、1e-4 でも E=25 では改善なし。材料剛性に応じた自動推定が必要
- [ ] **Hermite K_st の ∂p_n/∂s 項追加**: 33% 不整合の解消（status-238 から引継）
- [ ] **摩擦アセンブリの Hermite 完全対応**: use_hermite=False デフォルトの解消（status-238 から引継）
- [ ] **収束判定の力/モーメント分離**: 力 [N] とモーメント [N·mm] を別ノルムで判定

---

## 設計上の懸念

1. **LM + 接触チャタリング**: LM は Newton 方向を保守的にする（信頼領域制約）。接触活性/非活性が激しく切り替わる場合、保守的なステップが接触状態変化を遅らせ、cutback を増やす可能性
2. **λ 自動推定**: 現在の λ は手動設定。剛性行列の固有値スペクトルに基づく自動推定が望ましいが、計算コストとのトレードオフ

---

## 運用メモ

- LM 正則化はデフォルト無効（`lm_lambda_init=0.0`）— 既存動作に影響なし
- `lm_adaptive=True` のデフォルトにより、LM 無効でも発散検知時に自動で λ を導入（安全弁）
- 三点曲げジグの `consistent_st_tangent` と `lm_lambda_init` は Config から設定可能
