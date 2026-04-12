# status-242: λ自動推定検証 + dof_scale_rot 調査 + K_st ∂p_n/∂s + 凍結接線問題特定

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 190+10s+8+9+7+10 | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

status-241 の TODO 4項目のうち3項目を実施:
1. **λ自動推定の異材料検証**: 鉄鋼/銅/アルミで c=20 定数の汎用性を検証
2. **dof_scale_rot のパラメータスイープ**: 0.3〜1.0 の範囲で最適値を調査
3. **K_st の ∂p_n/∂s 項追加**: 数学的に正しい項を追加
4. **33% FD 不整合の真因特定**: ∂p_n/∂s ではなく StJacobian の凍結接線近似が原因

---

## 1. λ自動推定の異材料検証

### 結果

| 材料 | E [MPa] | λ_auto | baseline frac | auto frac | Δfrac | 判定 |
|------|---------|--------|--------------|-----------|-------|------|
| 鉄鋼 | 200e3 | 1.00e-4 | 0.092 | 0.094 | +0.002 | 微改善 |
| 銅 | 120e3 | 1.67e-4 | 0.077 | 0.079 | +0.002 | 微改善 |
| アルミ | 70e3 | 2.86e-4 | 0.080 | 0.054 | -0.026 | **悪化** |

**結論**: c=20 はアルミ(E=70e3)で**逆効果**。λ=2.86e-4 は過大。
- 鉄鋼・銅では改善は微小（Δfrac=0.002、ノイズレベル）
- アルミでは有意に悪化（Δfrac=-0.026）
- **c=20 の汎用性は低い**。材料別チューニングまたは別の自動推定戦略が必要

### 検証条件

n_periods=1, jig_push=5mm, max_increments=30, freeze=F, K_st=ON

---

## 2. dof_scale_rot のパラメータスイープ

### 結果

| scale | frac | cutback率 | 判定 |
|-------|------|-----------|------|
| 1.0 | 0.094 | 37.5% | **最良** |
| 0.8 | 0.041 | — | 悪化 |
| 0.7 | 0.049 | — | 悪化 |
| 0.6 | 0.045 | — | 悪化 |
| 0.5 | 0.047 | — | 悪化 |
| 0.4 | — | — | 悪化 |
| 0.3 | — | — | 悪化 |

**結論**: dof_scale_rot < 1.0 は**全値で悪化**。回転DOF減衰は有害。
- デフォルト `dof_scale_rot=1.0`（スケーリング無し）が最適
- 回転DOFを減衰させると接触幾何の更新が遅れ、逆に収束が悪化

---

## 3. K_st の ∂p_n/∂s 項追加

### 実装

ペナルティ力 p_n のスライディング微分を K_st に追加:

```
∂p_n/∂s = h'(x) * k_pen * (-∂gap/∂s)
∂gap/∂s = normal · dpA/ds
```

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/contact_force/strategy.py` | `_add_kst_contact` に h_deriv, k_pen 引数追加 + ∂p_n/∂s, ∂p_n/∂t 計算 |

この項は曲線セグメントで `normal · dpA ≠ 0` の場合に効果がある。
テスト配置（直交交差、normal⊥dpA）では ∂gap/∂s = 0 のため寄与ゼロ。

---

## 4. 33% FD 不整合の真因特定

### 従来の仮説

status-238: 「∂p_n/∂s 項が欠落しているため Hermite K_st に 33% の FD 不整合」

### 実際の原因: StJacobian の凍結接線近似（frozen-m）

Hermite 接触では接線ベクトル m = x₁ - x₀ が節点座標に依存する。
しかし StJacobian の `_compute_rhs_hermite()` は **m を定数として微分**している。

**影響の連鎖**:

1. **Hermite dc_ds**: H00'(0.5) = -1.5 vs linear: -1 → **1.5倍**
2. **Hermite StJacobian RHS**: `H00'(s)*delta = -1.5*delta` vs linear: `-delta` → ds_du も **1.5倍**
3. **合計**: K_st が 1.5 × 1.5 = **2.25倍**に膨張

直線セグメントでは Hermite ≡ linear なので、真の K_st は linear と同一であるべき。
凍結近似により Hermite K_st が過大評価されている。

### なぜ修正が難しいか

接線ベクトル m の節点座標に対する微分 ∂m/∂u は**非局所**:
- 端点: ∂m/∂x₀ = -I（同一要素内で閉じる）
- 内部節点: ∂m_i/∂x_j = -I/2（隣接要素のノードに依存 → 4ノードペア外の結合）

K_st のローカル 12×12 マトリクスの枠を超える DOF 結合が必要で、
アーキテクチャの大幅変更を伴う。

### FDテストの Hermite 精緻化追加

`check_tangent_consistency.py` の `_compute_geometry()` に Hermite 精緻化を追加。
ただし直線セグメントでは Hermite = linear のため結果に変化なし（期待通り）。

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/contact_force/strategy.py` | ∂p_n/∂s 項追加 + h_deriv/k_pen 引数追加 |
| `contracts/check_auto_lambda_materials.py` | **新規**: 異材料 λ 自動推定検証スクリプト |
| `contracts/check_dof_scale_rot_sweep.py` | **新規**: dof_scale_rot パラメータスイープ |
| `contracts/check_tangent_consistency.py` | Hermite 精緻化対応 |

---

## TODO

- [ ] **凍結接線 (frozen-m) の解消**: StJacobian に ∂m/∂u 寄与を追加（非局所DOF結合が必要）
- [ ] **λ自動推定の改善**: c=20 は材料依存。材料剛性ではなく接触剛性比 k_pen/E ベースの推定を検討
- [ ] **摩擦アセンブリの Hermite 完全対応**: use_hermite=False デフォルトの解消（frozen-m 解消が前提）

---

## 設計上の懸念

1. **frozen-m と Hermite の矛盾**: Hermite 接触の精度は凍結接線近似に制約される。直線セグメントで 33% の接線不整合は NR 力収束の阻害要因。曲線セグメントでは更に悪化する可能性
2. **dof_scale_rot の無効性**: 回転DOF減衰が全値で悪化することは、NR残差の並進/回転逆相関の原因が DOF スケーリングでは対処できない（構造的問題）ことを示唆
3. **λ自動推定の限界**: 材料剛性 E のみからの λ 推定は汎用性が低い。接触ジオメトリ（曲率、貫入量）の影響が大きい

---

## 開発運用メモ

- STA2 防止: λ検証・dof_scale_rot スイープ結果は tee でログ保存済み
- 33% 不整合を「∂p_n/∂s で解消」と報告せず、真因（frozen-m）を正直に報告

---
