# status-173: deprecated プロセス完全削除 + executor テスト追加

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-15
**作業者**: Claude Code
**ブランチ**: claude/execute-status-todos-Hr8vp

## 概要

status-172 の TODO 3件を消化。deprecated プロセス2クラスの完全削除と executor.py のテスト追加。

## 変更内容

### 1. StrandBendingBatchProcess → ContactFrictionProcess 移行

`StrandBendingBatchProcess` が使用していた deprecated `NCPQuasiStaticContactFrictionProcess` を `ContactFrictionProcess` に直接置換。

| ファイル | 変更 |
|---------|------|
| `process/batch/strand_bending.py` | import + uses + process() を ContactFrictionProcess に移行 |
| `process/batch/tests/test_batch.py` | uses 宣言テストを更新 |

### 2. deprecated 2クラスの完全削除

全ての呼び出し元が移行済みのため、deprecated プロセスとデータ型を完全削除。

| 削除ファイル | 内容 |
|-------------|------|
| `process/concrete/solve_quasistatic_friction.py` | NCPQuasiStaticContactFrictionProcess |
| `process/concrete/solve_dynamic_friction.py` | NCPDynamicContactFrictionProcess |
| `process/concrete/tests/test_solve_quasistatic_friction.py` | deprecated テスト |
| `process/concrete/tests/test_solve_dynamic_friction.py` | deprecated テスト |

| 更新ファイル | 内容 |
|-------------|------|
| `process/concrete/__init__.py` | deprecated クラスの export 除去 |
| `process/__init__.py` | DynamicFrictionInputData/QuasiStaticFrictionInputData の export 除去 |
| `process/data.py` | DynamicFrictionInputData/QuasiStaticFrictionInputData クラス削除 |
| `process/testing.py` | ドキュメント例を ContactFrictionProcess に更新 |
| `process/docs/solve-smooth-penalty-friction.md` | プロセスクラス説明を ContactFrictionProcess に統一 |

### 3. executor.py テスト追加

`tests/test_tuning_schema.py` に3テスト追加:

| テスト | 内容 |
|--------|------|
| `test_execute_s3_benchmark_error_handling` | ソルバー例外時に TuningRun が返ること（monkeypatch） |
| `test_run_convergence_tuning_no_grid` | param_grid=None でデフォルト1回実行 |
| `test_run_convergence_tuning_with_grid` | 2x2 グリッドサーチが4回実行されること |

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: 0 件
- process テスト: 343 passed（deprecated テスト削除分減少）
- executor テスト: 4 passed（import + 新規3件）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `NCPQuasiStaticContactFrictionProcess` | 完全削除（→ ContactFrictionProcess） | status-173 |
| `NCPDynamicContactFrictionProcess` | 完全削除（→ ContactFrictionProcess） | status-173 |
| `QuasiStaticFrictionInputData` | 完全削除（→ ContactFrictionInputData） | status-173 |
| `DynamicFrictionInputData` | 完全削除（→ ContactFrictionInputData） | status-173 |

## 今後の TODO

- [ ] Phase 9-C/D: S3 凍結解除判断 + BatchProcess パイプライン改善
- [ ] 変位制御7本撚線曲げ揺動のPhase2 xfail解消

## 設計上の懸念・メモ

- data.py から旧データ型を削除したが、外部コード（scripts/等）での直接参照は無いことを確認済み。
- executor.py の実ソルバー実行テストは CI slow test で行う想定。本statusでは monkeypatch による単体テストのみ追加。

---
