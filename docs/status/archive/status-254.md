# status-254: 7本撚線曲げ揺動収束テスト + MPC力残差整合性確認

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/check-status-todos-yzvY8`
- **テスト数**: 200+10s+16+3+23+1（新規1件：収束テスト）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. MPC u伝搬の修正（process.py）

**問題**: 処方変位をmaster参照点DOFに設定しても、MPC制約がslave DOFに伝搬されず、梁が変形しなかった（残差=0、接触=0で即収束する偽収束）。

**修正**: `process.py`のインクリメントループで、処方変位設定後にMPC制約を強制する処理を追加。

```python
# u_full = T @ u_red（slave DOFをmaster値から再計算）
_u_red = state.u[_mpc.independent_dofs]
state.u[:] = T @ _u_red
```

### 2. NR内MPC制約再射影（_newton_dynamic.py）

NRイテレーション内で`u += du`後、slave DOFがMPC制約から逸脱する可能性があるため、MPC射影を追加。

### 3. 拡張DOF系アセンブララッパー（strand_bending_oscillation.py）

**問題**: `ul_assembler.u_total_accum`（714 DOF）と`state.u`（726 DOF）のshape不一致でクラッシュ。

**修正**: `_ExtendedULAssemblerWrapper`を実装し、`u_total_accum`を拡張系にゼロパディング。`checkpoint()`/`rollback()`は内部アセンブラに委譲。

### 4. 収束実行テスト

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/process.py` | MPC u伝搬追加（処方変位後） |
| `xkep_cae/contact/solver/_newton_dynamic.py` | NR内MPC制約再射影追加 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `_ExtendedULAssemblerWrapper`追加、拡張系アセンブラ適用 |
| `tests/numerical_tests/test_strand_bending_convergence.py` | **新規** — 7本撚線曲げ揺動収束テスト |

---

## 収束テスト結果

### 条件
- E=130MPa, ρ=8.96e-9, R=0.5mm, pitch=100mm
- κ=0.001 1/mm, n_cycles=1, 40 increments
- exclude_same_strand=True

### 結果
- **frac到達**: 0.3531（35.3%）
- **接触開始前（frac<0.25）**: 全ステップ3-4回NRで収束（energy収束）
- **接触開始後（frac≈0.30）**: 6ペアactive、NR 7-12反復で収束
- **壁（frac≈0.35）**: 接触チャタリング（period-4振動）で不収束 → dt cutback繰返し
- max|u| = 0.066, max Fc = 0.014 N, n_cutbacks = 4
- elapsed = 25s

### 低曲率テスト（κ=0.0003）
- **frac=1.0完走**（接触なし）、40 increments、13s
- max|u| = 0.059

---

## MPC + 動的ソルバーの力残差整合性確認

### 発見した問題

1. **slave DOF残差が収束判定を支配**: R_u[slave_dofs]は制約反力であり、NR反復でゼロにならない。しかし、slave DOFをゼロ化すると独立DOFの残差減少が極端に遅くなる（rate≈0.968）。現状はenergy収束基準で脱出。

2. **接触チャタリング**: frac≈0.35で||R_t||/||f||≈0.44-0.58が周期4で振動。リラクゼーション（ω=0.5）でも収束せず。根本原因: MPC縮退系での接触力の取り扱い。

3. **試行した不採用アプローチ**:
   - R_u = T @ T^T @ R_u 射影 → 残差を増幅（Tは正規直交行列ではない）
   - R_u[slave_dofs] = 0 → 独立DOFの収束速度が壊滅的に悪化

### 結論

MPC自体は正しく機能（梁が実際に曲がり、接触が発生）。残差整合性の問題はMPC + 接触の相互作用に起因し、以下が次の改善候補:
- MPC縮退系での収束判定（R_red = T^T R_u のノルムで判定）
- 接触力のMPC整合性（slave DOFへの接触力をmaster集約してからNR解く）
- time integrator のMPC対応（M_red = T^T M T での慣性項）

---

## テスト結果

- 新規テスト: 1件（収束テスト、slow）
- 既存テスト: 528 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- 条例違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-yzvY8
pip install -e .
# 収束テスト（slow、約25秒）
python -m pytest tests/numerical_tests/test_strand_bending_convergence.py -x -s -v --timeout=600 2>&1 | tee /tmp/log-strand-bending.log
# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 優先TODO
1. **MPC + 接触のNR収束改善** — 現在frac≈0.35で壁
   - 候補A: 接触力をMPC縮退系で評価（slave DOF接触力をmasterに集約してからNR）
   - 候補B: time integrator のMPC対応（M_red = T^T M T）
   - 候補C: MPC縮退系での収束判定（R_red ベース）
2. **C2-C3 幾何Process化** — MPC改善とは独立で着手可能
3. **B1-B4 摩擦アセンブリProcess化**

### STA2 tolerance 厳格化（status-252引き継ぎ）
4. **T1 Hermite atol → 1e-5** → frozen-m完全解消後
5. **T2 beam oscillation rtol → 0.02** → 要素数≥40時

---

## 懸念・設計メモ

1. **MPC + energy収束の信頼性**: 現在、接触ありステップはenergy収束（du^T R < threshold）で脱出。slave DOFの制約反力が||R||に含まれるため力収束は不可。energy収束はslave DOFに感度が低く実用的だが、物理的妥当性の検証が必要。
2. **time integrator のMPC未対応**: M_ext（フルシステム）を使用しており、参照点の有効質量はゼロ。MPC付きではM_red = T^T M Tが正しい有効質量行列。現状はenergy convergenceでマスクされているが、準静的挙動の仮定が崩れると問題化する可能性あり。
