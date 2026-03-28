# status-260: smoothing_delta チューニング + FD診断活性DOFフィルタリング

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/execute-status-todos-bIcOS`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3（新規3件）→ **合計561 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. smoothing_delta 効果検証 + 最適値チューニング

status-259 で実装した smoothing_delta パイプラインの実効果を定量測定。

#### ベンチマーク結果（7本撚線曲げ揺動, MPC+接触）

| smoothing_delta | delta_h (k_pen/δ) | frac | incr | cutback | NR_avg | NR_max | time |
|----------------|-------------------|------|------|---------|--------|--------|------|
| baseline (δ→∞) | ≈0 (max(0,x)) | 0.3500 | 15 | 3 | 5.1 | 13 | 14.7s |
| 20000 | k_pen/20000 | 0.3500 | 15 | 3 | 5.1 | 13 | 14.5s |
| 10000 (旧5000/r) | k_pen/10000 | 0.3500 | 15 | 3 | 5.1 | 13 | 14.3s |
| 7500 | k_pen/7500 | 0.3688 | 17 | 2 | 7.2 | 26 | 16.4s |
| 5000 | k_pen/5000 | 0.3734 | 22 | 1 | 7.6 | 26 | 17.7s |
| 3000 | k_pen/3000 | 0.5786 | 206 | 3 | 8.9 | 20 | 172s |
| 2000 (新1000/r) | k_pen/2000 | **0.5914** | 195 | 3 | 9.2 | 25 | 186s |
| **1000** | k_pen/1000 | **1.0000** | 135 | 9 | 8.0 | 29 | **176s** |

#### 分析

- **旧値 5000/r (=10000)**: ベースラインと同じ結果。delta_h が小さすぎて平滑化が効いていなかった
- **新値 1000/r (=2000)**: frac=0.59（ベースライン0.35から**69%改善**）
- **delta=1000**: frac=1.0完走達成。ただし手動指定が必要
- **トレードオフ**: 大きい平滑化幅（小さいδ）→ frac改善だが、NR反復数増加・計算時間増大

#### 自動推定式の変更

`StrandBendingOscillationProcess` の自動推定: **5000/r → 1000/r** に変更。

- three_point_bend_jig は **5000/r のまま**（δ変更で動的テストが不収束になるため）
- 最適値は問題依存: MPC+撚線曲げでは小さいδが有効

### 2. FD診断の活性DOFフィルタリング（status-258 TODO）

`TangentFDDiagnosticProcess` に `active_contact_dofs` パラメータを追加。

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_steps.py` | `TangentFDDiagnosticInput` に `active_contact_dofs` 追加。`TangentFDDiagnosticOutput` に `active_dof_rel_err` 追加。活性DOFのみのFD vs 解析比較を実装 |

#### 設計

```
TangentFDDiagnosticInput.active_contact_dofs: np.ndarray | None
  → None: 活性DOF診断スキップ（後方互換）
  → np.array([0,1,6,7,...]): gap<0 ペアの関連DOFインデックス
  → 活性DOFのみで dR_FD vs K@du の相対誤差を計算
  → TangentFDDiagnosticOutput.active_dof_rel_err (-1=未計算, ≥0=計算済み)
```

### 3. 収束テスト閾値引き上げ

`test_strand_bending_convergence.py` の frac 閾値: **0.25 → 0.50** に引き上げ（smoothing_delta改善により）。

### 4. テスト追加

| テストファイル | テスト名 | 内容 |
|--------------|---------|------|
| `xkep_cae/contact/solver/tests/test_tangent_fd_diagnostic.py` | `test_active_contact_dofs_consistent` | 線形系で活性DOF整合性確認 |
| `xkep_cae/contact/solver/tests/test_tangent_fd_diagnostic.py` | `test_active_contact_dofs_detects_mismatch` | 不整合検出確認 |
| `xkep_cae/contact/solver/tests/test_tangent_fd_diagnostic.py` | `test_active_contact_dofs_none_skipped` | None時スキップ確認 |

---

## テスト結果

- 新規テスト: 3件（全通過）
- 既存テスト: 561 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/execute-status-todos-bIcOS
pip install -e .
# smoothing_delta ベンチマーク
python contracts/bench_smoothing_delta.py 2>&1 | tee /tmp/bench1.log
python contracts/bench_smoothing_delta2.py 2>&1 | tee /tmp/bench2.log
# FD診断テスト
python -m pytest xkep_cae/contact/solver/tests/test_tangent_fd_diagnostic.py -v --timeout=30
# smoothing_delta 自動推定テスト
python -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -v --timeout=30
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

### 残課題（優先度順）

1. **smoothing_delta=1000 での完走達成**: 自動推定 1000/r(=2000) では frac=0.59 止まり。手動 δ=1000 で完走が確認済み。Config可能なままなので `smoothing_delta=1000` を指定して完走テストを実装する余地あり
2. **three_point_bend_jig の smoothing_delta 最適化**: 現在 5000/r だが、静的解析と動的解析で最適値が異なる可能性。動的テスト `test_process_runs` が変更前から不安定（既知問題）
3. **T1 Hermite atol 厳格化**: Hermite 非局所 ∂g/∂u 対応（4ノードペア外のDOF結合）が必要（status-258 から継続）
4. **FD診断の活性DOF呼び出し側実装**: `active_contact_dofs` をNRソルバーから渡す結合が未実装。ContactManager のペア情報から gap<0 ペアのDOFを収集して渡す処理が必要

### 設計メモ

- smoothing_delta の最適値は問題依存。MPC+撚線曲げ揺動では δ=1000-2000 が有効
- three_point_bend は δ=5000/r のままが安定（動的解析の特性差）
- 平滑化幅が大きいほど（δが小さいほど）接触ON/OFF境界の不連続性は緩和されるが、NR反復数は増加するトレードオフ

---

## 懸念・設計メモ

1. **動的テスト不安定性**: `TestDynamicThreePointBendContactJigProcessAPI::test_process_runs` が変更前から失敗。smoothing_delta の値に依存しない別の問題の可能性
2. **手動 vs 自動推定**: delta=1000（手動）で完走できるが、自動推定 1000/r=2000 では0.59止まり。ユーザーが問題に応じて手動調整する設計は妥当だが、自動推定のデフォルト値についてはさらなる検討が必要

### smoothing_delta の最適値が系で異なる根本原因

smoothing_delta の最適値が three_point_bend（5000/r）と strand_bending（1000/r）で異なる原因は、**k_pen のスケールが桁違い**であること。

Huber遷移幅 `delta_h = k_pen / smoothing_delta` が実際の平滑化を制御するパラメータであり、同じ smoothing_delta でも k_pen が異なれば delta_h は全く異なる。

| 系 | k_pen推定式 | k_pen値 | delta_h (δ=5000/r) | delta_h (δ=1000/r) |
|---|---|---|---|---|
| 梁-梁 (strand bending) | `0.1 × 12EI/L_elem³` | ~31 N/mm | 3.1e-3 | **1.6e-2** |
| 静的剛体-梁 (3pt bend) | `0.5 × 48EI/L³` | ~0.15 N/mm | 1.5e-5 | 7.7e-5 |
| 動的剛体-梁 (dynamic 3pt) | `0.2 × c₀·m_ii` | ~6e-6 N/mm | 6e-10 | 3e-9 |

- **梁-梁**: k_pen が大きい（要素長ベース、L_elem=6.25mm）→ delta_h を確保するにはδを小さくする必要がある
- **剛体-梁**: k_pen が小さい（全長ベース、L=100mm、3乗で効く）→ δ=5000/r で既に十分平滑。δを下げると delta_h が大きすぎてペナルティ力の立ち上がりが鈍り不収束
- **動的系**: k_pen が微小（c₀·m_ii ベース、dt⁻²依存）→ smoothing_delta の影響はほぼゼロ

**設計改善案**: smoothing_delta（間接パラメータ）ではなく delta_h（Huber遷移幅そのもの）を直接指定するAPIに変更すれば、k_pen スケールに依存しない一貫した設定が可能になる。例: `huber_transition_width = 0.01` とすれば、k_pen に関わらず接触力の遷移幅が物理的に同じ意味を持つ。
