# status-172: status-171 TODO消化 — LinearSolverStrategy統合 + Process統合 + executor NCP版

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-15
**作業者**: Claude Code
**ブランチ**: claude/execute-status-todos-KQQ04

## 概要

status-171 の TODO 3件を全て消化。

## 変更内容

### 1. PR-3 第2段階: _solve_saddle_point_contact の LinearSolverStrategy 統合

`_solve_saddle_point_contact()` のdispatcherを LinearSolverStrategy ベースに統合。

| 変更 | 内容 |
|------|------|
| `_solve_saddle_point_contact()` | `create_linear_solver()` を一度だけ呼び、Strategy インスタンスを再利用 |
| `_solve_saddle_point_direct()` | `solver_strategy` パラメータ追加。Strategy.solve() を直接呼出 |
| n_active==0 パス | `_solve_linear_system()` 呼出を `strategy.solve()` 直接呼出に変更 |
| `_solve_saddle_point_gmres()` | 拡大鞍点系固有の GMRES（別パラダイム）のため変更なし |

### 2. PR-4: ContactFrictionProcess 統合

NCPQuasiStaticContactFrictionProcess と NCPDynamicContactFrictionProcess を統合。

| 新規/変更 | 内容 |
|----------|------|
| `ContactFrictionInputData` | 統一入力型。動的パラメータ Optional（is_dynamic プロパティで自動判定） |
| `ContactFrictionProcess` | 統一ソルバー。TimeIntegrationStrategy が QuasiStatic/GeneralizedAlpha を自動切替 |
| `NCPQuasiStaticContactFrictionProcess` | deprecated（→ ContactFrictionProcess に委譲） |
| `NCPDynamicContactFrictionProcess` | deprecated（→ ContactFrictionProcess に委譲） |
| `validate_process_contracts.py` | C5 で deprecated プロセスをスキップ |

### 3. PR-6: tuning/executor.py NCP版再実装

AL依存で NotImplementedError だった4関数を NCP + smooth_penalty ベースで再実装。

| 関数 | 実装内容 |
|------|---------|
| `execute_s3_benchmark()` | `run_bending_oscillation()` → TuningRun マッピング。収束失敗時も metrics 返却 |
| `run_scaling_analysis()` | 複数素線数で連続ベンチマーク実行 |
| `run_convergence_tuning()` | グリッドサーチでパラメータ組合せ網羅 |
| `run_sensitivity_analysis()` | 2パラメータ全組合せ感度分析 |

`optuna_tuner.py` は既に `execute_s3_benchmark()` を呼ぶ形のため変更不要。

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: 0 件
- テスト: 408 passed（process + contact + tuning）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `NCPQuasiStaticContactFrictionProcess` | `ContactFrictionProcess` | status-172 |
| `NCPDynamicContactFrictionProcess` | `ContactFrictionProcess` | status-172 |
| `QuasiStaticFrictionInputData` / `DynamicFrictionInputData` | `ContactFrictionInputData`（統一型） | status-172 |
| `executor.py` 4関数 NotImplementedError | NCP版実装 | status-172 |

## 今後の TODO

- [ ] StrandBendingBatchProcess を ContactFrictionProcess に移行（現在 NCPQuasiStatic を使用中）
- [ ] deprecated 2クラスの呼び出し元を全て ContactFrictionProcess に移行後、完全削除
- [ ] executor.py の実行テスト（CI環境でベンチマーク実行可能かの確認）
- [ ] Phase 9-C/D: S3 凍結解除判断 + BatchProcess パイプライン改善

## 設計上の懸念・メモ

- `_solve_saddle_point_gmres()` は拡大鞍点系全体をGMRESで解くため、LinearSolverStrategy の `solve(K, rhs)` インターフェースとは異なるパラダイム。無理にStrategy化する必要はない。
- ContactFrictionInputData の `is_dynamic` は `mass_matrix is not None and dt_physical > 0.0` で判定。`mass_matrix` が None で `dt_physical > 0` のケースは準静的として扱う（質量行列なしの動的解析は物理的に不整合）。

---
