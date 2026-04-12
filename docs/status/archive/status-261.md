# status-261: δ=1000完走テスト + active_contact_dofs NR結合 + delta_h直接指定API

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/check-status-todos-1m2q0`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9（新規9件）→ **合計570 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. smoothing_delta=1000 完走テスト（status-260 TODO #1）

`test_strand_bending_full_completion_delta1000` を追加。
手動指定 `smoothing_delta=1000` で frac≥0.95 完走を検証するテスト。

- status-260 ベンチマーク: δ=1000 で frac=1.0 完走済み（176s）
- 自動推定 1000/r=2000 では frac≈0.59 止まりだが、手動 δ=1000 で完走可能
- `@pytest.mark.slow` テストとして追加（通常テストスイートでは `-k "not slow"` で除外）

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `tests/numerical_tests/test_strand_bending_convergence.py` | `test_strand_bending_full_completion_delta1000` 追加 |

### 2. active_contact_dofs NRソルバー結合（status-260 TODO #4）

`_newton_dynamic.py` のFD接線診断呼び出し箇所で、gap<0 ペアの関連DOFを自動収集して `TangentFDDiagnosticInput.active_contact_dofs` に渡すように実装。

- `_contact_dofs()` ユーティリティを使用して4ノード×6DOF=24DOF/ペアを収集
- gap<0 の全ペアの DOF を集約してソート済み配列として渡す
- ペアが0の場合は None（後方互換：活性DOF診断スキップ）

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_dynamic.py` | `_contact_dofs` import追加、FD診断呼び出し時に gap<0 ペアDOF収集・渡し |

### 3. delta_h 直接指定API（status-260 設計メモ実装）

`huber_delta_h` パラメータをパイプライン全体に貫通。
smoothing_delta（間接: `delta_h = k_pen / δ`）に加え、delta_h を直接指定できるAPIを追加。

#### 設計

```
huber_delta_h > 0: delta_h = huber_delta_h（直接指定、k_penスケール非依存）
huber_delta_h = 0 && smoothing_delta > 0: delta_h = k_pen / smoothing_delta（間接、従来動作）
huber_delta_h = 0 && smoothing_delta = 0: delta_h = 0（max(0,x)相当、平滑化なし）
```

優先順位: `huber_delta_h > smoothing_delta > 0`

#### 利点

- k_pen スケールに依存しない一貫した遷移幅指定が可能
- 梁-梁（k_pen~31）と剛体-梁（k_pen~0.15）で同じ delta_h 値が同じ物理的意味を持つ
- smoothing_delta では系ごとに最適値が異なる問題（status-260 設計メモ）を根本的に解決

#### パイプライン貫通箇所

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/_contact_pair.py` | `_ContactConfigInput.huber_delta_h` 追加 |
| `xkep_cae/contact/setup/process.py` | `ContactSetupConfig.huber_delta_h` 追加、パススルー |
| `xkep_cae/core/data.py` | `default_strategies()` に `huber_delta_h` 引数追加 |
| `xkep_cae/contact/solver/process.py` | `manager.config.huber_delta_h` 渡し |
| `xkep_cae/contact/contact_force/strategy.py` | `HuberContactForceProcess._huber_delta_h` + `_resolve_delta_h()` メソッド追加。evaluate/evaluate_tangent の delta_h 計算を統一 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig.huber_delta_h` 追加、パススルー |

### 4. テスト追加

| テストファイル | テスト名 | 内容 |
|--------------|---------|------|
| `tests/numerical_tests/test_strand_bending_convergence.py` | `test_strand_bending_full_completion_delta1000` | δ=1000 完走テスト（slow） |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `test_with_huber_delta_h` | ファクトリ貫通確認 |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `test_resolve_delta_h_direct` | 直接指定で k_pen 非依存 |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `test_resolve_delta_h_indirect` | 間接計算（従来動作） |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `test_resolve_delta_h_none` | 両方0で delta_h=0 |
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `test_resolve_delta_h_priority` | huber_delta_h 優先確認 |
| `xkep_cae/contact/setup/tests/test_process.py` | `test_huber_delta_h_default_zero` | デフォルト値確認 |
| `xkep_cae/contact/setup/tests/test_process.py` | `test_huber_delta_h_passthrough` | Config→Manager貫通 |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | `test_huber_delta_h_default_zero` | Config デフォルト |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | `test_huber_delta_h_manual_override` | 手動指定 |

---

## テスト結果

- 新規テスト: 9件（全通過、slow テスト除く）
- 既存テスト: 570 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-1m2q0
pip install -e .
# 新規テスト（高速）
python -m pytest xkep_cae/contact/contact_force/tests/test_strategy.py -v --timeout=30
python -m pytest xkep_cae/contact/setup/tests/test_process.py -v --timeout=30
python -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -v --timeout=30
# δ=1000 完走テスト（~176s）
python -m pytest tests/numerical_tests/test_strand_bending_convergence.py::TestStrandBendingConvergence::test_strand_bending_full_completion_delta1000 -v -s --timeout=600 2>&1 | tee /tmp/log-delta1000.log
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

1. **δ=1000 完走テスト実行確認**: slow テストとして追加済みだが、CI環境での実行時間確認が必要（~176s）
2. **delta_h 最適値の探索**: huber_delta_h 直接指定APIが利用可能になったので、問題非依存の最適 delta_h を探索可能。梁-梁で delta_h=0.01-0.03 が有効な範囲（status-260のk_pen~31から逆算: δ=1000→delta_h=0.031）
3. **three_point_bend_jig への delta_h 適用**: huber_delta_h で統一指定すれば k_pen スケール差問題を解決可能
4. **T1 Hermite atol 厳格化**: Hermite 非局所 ∂g/∂u 対応（4ノードペア外のDOF結合）が必要（status-258 から継続）
5. **NR力収束改善**: 中盤後〜終盤で 25 反復が力収束に不足、disp 収束で抜ける状態

### 設計メモ

- `_resolve_delta_h()` は `huber_delta_h > smoothing_delta > 0` の優先順位で解決
- delta_h の物理的意味: 接触力の遷移幅（gap の単位と同じ [mm]）。gap ∈ [-delta_h, delta_h] の範囲で Huber 平滑化
- 推奨: 新規問題では `huber_delta_h` を直接指定し、smoothing_delta は後方互換のみに使用

---

## 懸念・設計メモ

1. **active_contact_dofs の精度**: gap<0 判定は現在のNR反復時点の gap を使用。NR反復中に活性集合が変化する場合、FD診断の活性DOFフィルタが不完全になる可能性。ただし診断目的なので実害は軽微
2. **delta_h の後方互換**: huber_delta_h=0（デフォルト）で従来動作と完全互換。既存テスト全通過で確認済み
