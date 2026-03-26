# status-240: 収束判定の力/モーメント分離 + E=200e3 鉄鋼検証

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 190+10s+8+9+7 | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

status-239 の TODO を実施:
1. 収束判定の力/モーメント分離を実装
2. E=200e3 鉄鋼での K_st + LM 効果検証
3. freeze=False + K_st=True + LM の本格評価

---

## 1. 収束判定の力/モーメント分離

### 問題

`ConvergenceCheckProcess` の力収束判定で `np.linalg.norm(R_u)` が
並進残差 [N] と回転残差 [N·mm] を混合していた（status-239 で指摘）。
接触力は並進 DOF のみにアセンブリされるため、回転残差が力収束判定を汚染していた。

### 実装

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_steps.py` | `ConvergenceCheckInput.ndof_per_node` 追加、並進/回転分離 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | ndof_per_node 伝搬、ログ分離表示、発散検知を並進残差ベースに |
| `tests/contact/test_convergence_separation.py` | **新規**: 分離テスト 7 件 |

**変更の要点**:
- `ConvergenceCheckOutput` に `res_trans_norm`, `res_rot_norm` を追加
- **力収束判定は並進残差のみ**: `res_trans_norm / f_ref < tol_force`
- `dynamic_ref=True` のとき参照値も並進残差ノルムから設定
- **発散検知も並進残差ベース**: `_res_ratio = conv_out.res_trans_norm / conv_out.f_ref`
- NR ログに `||R_t||/||f||` と `||R_r||` を分離表示

### 後方互換性

- `ndof_per_node` のデフォルト値は 6（梁要素）
- `ndof_per_node < 6` の場合は全 DOF を並進として扱う（分離なし）
- `ConvergenceCheckOutput` の新フィールドにデフォルト値 0.0

---

## 2. E=200e3 鉄鋼での K_st + LM 効果検証

### 検証条件

三点曲げジグ: E=200e3 MPa, n_periods=3, jig_push=5mm, max_increments=50

### 結果

| 構成 | frac | fc [N] | incr | cb | cb% | time |
|------|------|--------|------|-----|-----|------|
| baseline (freeze=T, K_st=OFF) | 0.037 | 26694 | 50 | 18 | 26.5% | 20.2s |
| freeze=F, K_st=OFF | 0.072 | 8.5 | 50 | 32 | 39.0% | 42.7s |
| **freeze=F, K_st=ON, LM=1e-4** | **0.045** | **14261** | **50** | **29** | **36.7%** | **45.6s** |

### 分析

1. **K_st+LM は baseline 比 22% 改善**（frac 0.037→0.045）— 鉄鋼でも効果あり
2. **freeze=F, K_st=OFF は接触力消失**（fc=8.5N）— freeze なしでは K_st が必須
3. **全構成で力収束は極めて困難**: 力収束 1 回 / 変位収束 146 回

### 重要発見: NR 中の並進/回転残差の逆相関

並進残差と回転残差の NR 進行パターン（典型例: Incr 31）:

```
attempt  0: ||R_t||/||f|| = 1.000,  ||R_r|| = 5020
attempt  5: ||R_t||/||f|| = 3.2e-3, ||R_r|| = 3.8e-3  ← 両方下がる
attempt 10: ||R_t||/||f|| = 5.7e-3, ||R_r|| = 3.2e-3  ← 並進が増加!
attempt 15: ||R_t||/||f|| = 7.8e-3, ||R_r|| = 2.8e-3  ← 並進さらに増加
attempt 20: ||R_t||/||f|| = 9.6e-3, ||R_r|| = 2.3e-3  ← 回転は減少続行
```

**解釈**: NR 更新方向が回転 DOF を優先的に改善し、接触幾何の変動で並進残差が増加。
これは接線剛性行列で回転 DOF の剛性が並進の ~100 倍大きい（4EI/L vs EA/L）ため、
NR の降下方向が回転成分に支配されることが原因。

### λ チューニング所見

- λ=1e-4 は E=200e3 鉄鋼で概ね適切（baseline より改善、cutback 増加は許容範囲）
- λ の自動推定は未実装。E に比例するスケーリング `λ = c / E` が一つの方向

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_steps.py` | 力/モーメント分離実装 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | ndof_per_node 伝搬 + ログ分離 |
| `tests/contact/test_convergence_separation.py` | **新規**: 分離テスト 7 件 |
| `contracts/check_steel_kst_lm.py` | **新規**: 鉄鋼 K_st+LM 検証 |
| `contracts/check_steel_kst_lm_quick.py` | **新規**: 簡易版（参考） |

---

## TODO

- [ ] **並進/回転 DOF の重み付きノルム**: 回転残差を代表長さで正規化して統合ノルムを改善
- [ ] **NR 更新方向のスケーリング**: 並進 DOF と回転 DOF に異なる減衰を適用
- [ ] **Hermite K_st の ∂p_n/∂s 項追加**: 33% 不整合の解消（status-238 から引継）
- [ ] **摩擦アセンブリの Hermite 完全対応**: use_hermite=False デフォルトの解消
- [ ] **λ 自動推定**: 材料剛性 E に基づく初期値推定 `λ_init = c / E`

---

## 設計上の懸念

1. **並進のみ力収束の妥当性**: 回転残差の大きさを完全無視して良いかは要検討。
   現時点では接触力が並進 DOF のみなので正当化できるが、将来的に回転モーメント接触
   （曲面接触など）が入ると再考が必要
2. **NR の並進/回転逆相関**: K_T の条件数改善（DOF スケーリング）が根本対策の可能性

---

## 運用メモ

- NR ログフォーマットが変更: `||R_u||/||f||` → `||R_t||/||f||, ||R_r||`
- 既存テストへの影響なし（2 件の既存失敗は pre-existing）
