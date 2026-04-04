# status-289: FD接線診断でHertz型∂p/∂g整合性検証 + K_c不整合箇所特定

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-04
- **ブランチ**: `claude/convergence-diagnosis-logging-iHaYP`
- **テスト数**: 621 + 3新規 = 624 passed（回帰なし）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-288のTODO「FD接線診断でHertz型（α=1.5）の ∂p/∂g 整合性を検証」と「K_c + K_stの不整合箇所を活性集合安定状態で特定」を実施。

### 最重要発見

1. **Hertz型∂p/∂gスカラー導関数は正確**（FD一致、相対誤差 < 1e-4）
2. **K_c行列レベルのFD不整合はcomp=2（z成分）に集中** — analytical ≈ 0 に対し FD ≈ 3e-3
3. **この不整合はα=1.0（線形）でも同じパターン** — Hertz特有ではなく構造的問題
4. **根本原因**: 接触力のz方向依存がK_Tに含まれていない（K_geoまたはK_stの幾何項不足）

---

## 検証内容

### 1. Hertz型スカラー導関数FD検証（3テスト追加）

| テスト | 内容 | 結果 |
|--------|------|------|
| `test_hertz_tangent_w_mat_consistency` | K_matの重みw_mat = h_deriv * k_penとスカラーFDの比較（α=1.0, 1.5） | **PASS** (< 1e-4) |
| `test_hertz_scalar_dpn_dg_fd` | p_n(g)のスカラーFD検証（α=1.5, δ_h=0） | **PASS** (< 1e-4) |
| `test_hertz_scalar_dpn_dg_fd_with_delta_h` | p_n(g)のスカラーFD検証（α=1.5, δ_h=10） | **PASS** (< 1e-3) |

**結論**: `_apply_power_law()` と `_apply_power_law_deriv()` の数式実装は正確。

### 2. 90度曲げ実行時のFD診断ログ解析

status-288の効果測定ログから自動トリガーされたFD接線診断の結果を解析:

```
全体系 K@du: FD vs 解析 相対誤差 = 1.0002e+00
活性DOF K@du: FD vs 解析 相対誤差 = 9.9972e-01
```

**DOFレベルの不整合パターン**:

| DOF | node | comp | FD | analytical | 差分 |
|-----|------|------|----|-----------|------|
| 596 | 99 | **2(z)** | 4.11e-03 | 2.38e-10 | 4.11e-03 |
| 494 | 82 | **2(z)** | 4.08e-03 | 2.17e-09 | 4.08e-03 |
| 500 | 83 | **2(z)** | -4.01e-03 | 8.87e-10 | -4.01e-03 |
| 590 | 98 | **2(z)** | -3.78e-03 | -4.22e-10 | -3.78e-03 |

**ほぼ全ての不整合DOFがcomp=2（z成分）。analyticalがほぼゼロ（1e-10オーダー）なのにFDでは1e-3オーダー。**

### 3. 線形ペナルティ（α=1.0）での確認

α=1.0でも同じType D分布:
- 初期: A+B+D (50-70%)
- frac > 0.1: D+E が出現

**→ Hertz導関数は無罪。問題は接線剛性行列の幾何項にある。**

---

## 根本原因の分析

### なぜcomp=2（z方向）にだけ不整合があるか

7本撚線は3D空間でヘリカルに配置される。接触点の法線ベクトルnは主にx-y平面内にあるが、z成分もゼロではない。

K_c（接触剛性）は以下の3項から成る:
- **K_mat**: `h' * k_pen * c_i c_j (n⊗n)` — ∂p_n/∂u
- **K_geo**: `(p_n/dist) * c_i c_j (I - n⊗n)` — ∂n/∂u  
- **K_st**: `-(df_ds ⊗ ds_du + df_dt ⊗ dt_du)` — ∂f/∂(s,t) * ∂(s,t)/∂u

**仮説**: K_matとK_geoは4ノード（A0,A1,B0,B1）のDOFのみに寄与するが、**接触点位置pA,pBのz方向微分がu_zに対して追加のカップリングを生む**。このカップリングがK_stのds_du/dt_duに含まれていないか、不完全な可能性。

特に、Hermite基底では接線ベクトルm（隣接ノードの変位に依存）のz成分が∂(s,t)/∂uに影響するが、frozen-m近似（status-242）でmを定数扱いしている場合、z方向のカップリングが欠落する。

### frozen-m問題との関連

status-242で特定されたfrozen-m問題:
> StJacobian computes ds/du, dt/du by treating tangent vectors m = x₁ - x₀ as constants instead of differentiating them.

これがまさにz方向DOFへのカップリング欠落の原因である可能性が高い。∂m/∂u_z ≠ 0 だが、frozen-m近似ではこれがゼロとして扱われる。

---

## 効果測定結果（status-288再掲）

| 指標 | status-287 | status-288 | 変化 |
|------|-----------|-----------|------|
| frac | 0.9981 | **1.0000** | **完走** |
| cutback | 60 | **44** | **-27%** |

Type D対策（NR拡張）により完走達成。ただし線形収束のため反復数は多い。

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | TestHertzTangentFD: 3テスト追加（スカラーFD検証 + w_mat整合性） |

---

## 再現手順

```bash
git checkout claude/convergence-diagnosis-logging-iHaYP
# Hertz FDテスト
python -m pytest xkep_cae/contact/contact_force/tests/test_strategy.py::TestHertzTangentFD -xvs
# 全テスト
python -m pytest xkep_cae/contact/contact_force/tests/ -xq
# 90度曲げ効果測定（FD診断ログ付き）
python contracts/analyze_chattering_breakdown.py 2>&1 | tee /tmp/log-fd-analysis.log
```

---

## TODO

- [ ] frozen-m問題（status-242）の解消 — ∂m/∂u を正確に計算し、z方向DOFへのカップリングをK_stに追加
- [ ] StJacobianのz成分FD検証（∂s/∂u_z, ∂t/∂u_z の精度確認）
- [ ] frozen-m解消後の90度曲げ再実行 — 2次収束回復の確認
- [ ] K_geoのz成分寄与が正しいかFD検証
- [ ] 線形ペナルティでの完走テスト（α=1.0でもType D解消すれば完走可能か）

---

## 次の担当者向け

### 最重要ポイント

**Hertz導関数は正確。問題はK_c/K_stの幾何項、特にfrozen-m近似（status-242）によるz方向DOFカップリング欠落。**

FD診断でcomp=2（z方向）の不整合が支配的であり、analyticalがほぼゼロ（1e-10）なのにFDでは大きな値（1e-3）が出る。これはK_Tにz方向の∂f/∂u_z寄与が含まれていないことを決定的に示している。

frozen-m問題の解消（∂m/∂u の正確な計算）が2次収束回復の最も有望なアプローチ。

---
