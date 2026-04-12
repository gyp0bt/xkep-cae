# status-251: STA2 総摘発 — frozen偽装/uses未宣言/mutable DC/Process未包装の一斉修正

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-27
**ブランチ**: `claude/check-status-todos-IpSV1`
**テスト数**: 200+10s+16+3（新規3件追加）
**契約違反**: 1件（C3 既知、ComputeStJacobianProcess テスト未紐付け）

---

## 実施事項

### Phase F: CRITICAL — frozen偽装 + uses未宣言修正

| # | 問題 | 修正 |
|---|------|------|
| S1 | `SolverStateOutput`（frozen=True）の list フィールドに `.append()` で直接変異 | 履歴リスト4フィールドを `SolverStateOutput` から分離し、ローカル変数で管理 |
| S3 | `CoulombReturnMappingProcess` が `ComputeStJacobianProcess` を uses 未宣言で使用 | uses 宣言追加 |

### Phase G: HIGH — runner/dynamic_runner Process 化

| # | 問題 | 修正 |
|---|------|------|
| H1-H4 | `_run_bend3p/4p/tensile/torsion` がテストから直接呼出、Processトレース外 | `StaticBeamTestProcess`（VerifyProcess）追加、`_run_test` を Process 経由に変更 |
| H5 | `_run_dynamic_bend3p` が同様にトレース外 | `DynamicBeamTestProcess`（VerifyProcess）追加、`_run_dynamic_test` を Process 経由に変更 |

### Phase H: HIGH — SolverStrategies/SolverResultData frozen 化

| # | 問題 | 修正 |
|---|------|------|
| H6 | `SolverStrategies` が mutable dataclass | `frozen=True` に変更 |
| H7 | `SolverResultData` が mutable dataclass（list フィールド） | `frozen=True` + list→tuple 変換 |

### Phase I: MEDIUM — 残り mutable DC frozen 化

| # | 問題 | 修正 |
|---|------|------|
| M5 | `ExecutionContext` mutable | `frozen=True` |
| M6 | `ExecutionRecord` mutable | `frozen=True` |
| M7 | `ProcessNode` mutable（children: list） | `frozen=True` + children: tuple |
| M8 | `ProcessTree` mutable | `frozen=True` |
| M9 | `StrandBatchResult` | 既に frozen 済み（対応不要） |

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `contact/solver/_solver_state.py` | 履歴リスト4フィールド削除 |
| `contact/solver/process.py` | ローカル変数で履歴管理 + tuple変換 |
| `contact/friction/strategy.py` | uses 宣言追加 |
| `numerical_tests/runner.py` | StaticBeamTestProcess 追加 |
| `numerical_tests/dynamic_runner.py` | DynamicBeamTestProcess 追加 |
| `numerical_tests/tests/test_runner_process.py` | 新規テスト3件 |
| `core/data.py` | SolverStrategies/SolverResultData frozen化 |
| `core/runner.py` | ExecutionContext/Record frozen化 |
| `core/tree.py` | ProcessNode/Tree frozen化 |
| `docs/numerical_tests.md` | 新規ドキュメント |

---

## テスト結果

- 新規テスト: 3件（全合格）
- 既存テスト: 508 passed, 19 skipped（全合格）
- 契約違反: 1件（既知 C3）→ 増減なし
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/check-status-todos-IpSV1
pip install -e .
python -m pytest xkep_cae/ tests/ -x -q --ignore=xkep_cae/output/tests/test_stress_contour.py -m "not slow"
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/
```

---

## 監査結果サマリ（修正前の状態）

### 摘発対象と対処状況

| 致命度 | 件数 | 対処済 | 残件 |
|--------|------|--------|------|
| CRITICAL (S1-S3) | 3 | 2/3 | S2 は誤検知 |
| HIGH (H1-H7) | 7 | 7/7 | なし |
| MEDIUM (M1-M9) | 9 | 5/9 | M1-M4 は次セッション以降 |

### 残件（次セッション以降）

- M1: `_batch_update_geometry` (175行) → Process化（status-249 C2）
- M2: `_build_contact_frame_batch` (108行) → Process化（status-249 C3）
- M3: `_process_hermite` (103行) → C2-C3と一体
- M4: Strategy直呼び9箇所（`_newton_steps.py`）→ NRリファクタリング時に対応

---

## 懸念事項

1. **SolverResultData の tuple 変換**: 消費側で `list()` 変換が必要になる可能性。現時点では tuple のままで問題なし。
2. **ProcessNode の children: tuple**: 将来のツリー構築 API 設計時に注意。現時点では未使用。
