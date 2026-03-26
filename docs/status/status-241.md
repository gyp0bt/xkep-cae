# status-241: NR収束改善基盤 — λ自動推定・重み付きノルム・DOFスケーリング

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 190+10s+8+9+7+10 | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

status-240 の TODO 3項目を実装:
1. **λ 自動推定**: 材料剛性 E に基づく `λ = c / E` の自動計算
2. **並進/回転 DOF の重み付きノルム**: 回転残差を代表長さで正規化した統合ノルム
3. **NR 更新方向の DOF スケーリング**: 回転 DOF の NR 更新に個別減衰係数

---

## 1. λ 自動推定

### 問題

status-239 で LM 正則化を実装したが、λ 値は手動設定（`lm_lambda_init=1e-4`）。
材料剛性 E に依存するため、材料変更のたびにチューニングが必要だった。

### 実装

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/core/data.py` | `lm_auto_lambda: bool = False` 追加 |
| `xkep_cae/contact/solver/process.py` | `beam_E > 0` のとき `λ = 20.0 / E` を自動計算 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `lm_auto_lambda` フィールド追加 |

**使い方**:
```python
solver_input = ContactFrictionInputData(
    ...,
    lm_auto_lambda=True,  # beam_E から λ を自動計算
)
```

**定数 c=20 の根拠**: status-240 で E=200e3, λ=1e-4 が有効 → c = λ × E = 20

---

## 2. 並進/回転 DOF の重み付きノルム

### 問題

status-240 で並進/回転を分離したが、統合的な残差の大きさを評価する手段がなかった。
回転残差 [N·mm] と並進残差 [N] は次元が異なるため、単純加算できない。

### 実装

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_steps.py` | `char_length` 入力、`res_weighted_norm` 出力追加 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | `char_length` 伝搬 + ログ表示 |
| `xkep_cae/contact/solver/process.py` | `_beam_L`（平均要素長）を `char_length` として設定 |

**計算式**:
```
res_weighted_norm = sqrt(||R_trans||^2 + (||R_rot|| / L_char)^2)
```

- `L_char`: 平均要素長 [mm]（`process.py` で自動計算）
- `char_length=0` の場合は `res_weighted_norm = res_trans_norm`（後方互換）

### 設計判断

- **力収束判定には使用しない**: 引き続き並進残差のみで判定（接触力は並進DOFのみ）
- **ログ表示と発散検知の参考値**: `char_length > 0` のときNRログに `||R_w||/||f||` を追加表示

---

## 3. NR 更新方向の DOF スケーリング

### 問題

status-240 で NR 中に並進/回転残差の逆相関を発見:
NR 更新が回転 DOF を優先的に改善し、接触幾何の変動で並進残差が増加。

### 実装

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_dynamic.py` | `dof_scale_rot` パラメータ + NR更新後のスケーリング |
| `xkep_cae/core/data.py` | `dof_scale_rot: float = 1.0` 追加 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | `dof_scale_rot` フィールド追加 |

**使い方**:
```python
solver_input = ContactFrictionInputData(
    ...,
    dof_scale_rot=0.5,  # 回転 DOF の NR 更新を 50% に減衰
)
```

**デフォルト `dof_scale_rot=1.0`**: スケーリング無効（後方互換）

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/core/data.py` | `lm_auto_lambda`, `dof_scale_rot` 追加 |
| `xkep_cae/contact/solver/_newton_steps.py` | `char_length` 入力、`res_weighted_norm` 出力 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | `char_length`, `dof_scale_rot` 伝搬 + ログ |
| `xkep_cae/contact/solver/process.py` | λ自動推定 + `char_length`/`dof_scale_rot` 伝搬 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | 新パラメータ追加 |
| `tests/contact/test_convergence_separation.py` | **10件追加**: 重み付きノルム4件 + λ自動推定4件 + NR設定2件 |

---

## TODO

- [ ] **λ自動推定の検証**: 異なる E 値（銅 E=120e3、アルミ E=70e3）でのチューニング確認
- [ ] **dof_scale_rot の最適値調査**: 三点曲げで 0.3〜0.8 の範囲をスイープ
- [ ] **Hermite K_st の ∂p_n/∂s 項追加**: 33% 不整合の解消（status-238 から引継）
- [ ] **摩擦アセンブリの Hermite 完全対応**: use_hermite=False デフォルトの解消

---

## 設計上の懸念

1. **c=20 の汎用性**: 現在は鉄鋼（E=200e3）のみで確認。他材料での検証が必要
2. **dof_scale_rot と LM の相互作用**: 両方有効にすると回転DOFが二重に減衰される可能性

---
