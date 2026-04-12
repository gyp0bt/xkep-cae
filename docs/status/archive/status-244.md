# status-244: LM 正則化完全削除

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-26
**テスト**: 200+10s | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

Levenberg-Marquardt (LM) 正則化を完全削除。status-239 で実装、status-241/242 で検証したが、
効果がほぼ無く（Δfrac=0.002 程度、アルミでは逆効果）、コードの複雑さに見合わないため。

---

## 削除範囲

| 層 | 削除内容 |
|---|---------|
| `LinearSolveInput` | `lm_lambda` フィールド |
| `LinearSolveProcess` | LM 正則化ロジック（`K_T + λ·diag(K)` 追加処理） |
| `NewtonDynamicInput` | `lm_lambda_init`, `lm_adaptive`, `lm_lambda_max` |
| `NewtonDynamicProcess` | λ 適応制御、発散時 λ 増幅、ソルブ失敗時 LM リトライ |
| `ContactFrictionInputData` | `lm_lambda_init`, `lm_adaptive`, `lm_auto_lambda`, `lm_auto_c` |
| `ContactFrictionProcess` | λ 自動推定ロジック |
| `DynamicThreePointBendContactJigConfig` | 同上 4フィールド + パラメータ伝播 |
| テスト | `test_lm_regularization.py`（全削除）、`TestLMAutoLambdaAPI`（削除） |
| 契約 | `check_kst_lm.py`, `check_steel_kst_lm.py`, `check_steel_kst_lm_quick.py`, `check_kst_lm_unfrozen.py`, `check_auto_lambda_materials.py`（全削除） |

---

## テスト数変化

211 → 200（LM テスト 11 件削除）

---

## TODO

- [ ] **摩擦 Hermite 完全対応**: use_hermite=True デフォルト化（frozen-m 解消済み）
- [ ] **n_periods=30 収束検証**: freeze=F, K_st=ON, dm 補正有りでの検証

---
