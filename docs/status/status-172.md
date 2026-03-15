# status-172: status-171 TODO 消化 — LinearSolverStrategy 統合 + Process 統合 + executor NCP版

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-15
**作業者**: Claude Code
**ブランチ**: claude/execute-status-todos-mRM9y

## 概要

status-171 の TODO 3件を全て消化。

1. PR-3 第2段階: `_solve_saddle_point_contact()` dispatcher を LinearSolverStrategy ベースに統合
2. PR-4: NCPQuasiStatic/DynamicContactFrictionProcess を統一プロセスに統合
3. PR-6: tuning/executor.py の NCP 版再実装

## 変更内容

### 1. PR-3 第2段階: LinearSolverStrategy dispatcher 統合

`_solve_saddle_point_contact()` と `_solve_saddle_point_direct()` に `solver_strategy` パラメータを追加。
LinearSolverStrategy インスタンスを直接渡せるようにし、毎回のファクトリ再構築を回避。

| ファイル | 変更 |
|---------|------|
| solver_ncp.py | `_solve_saddle_point_contact()` に `solver_strategy` 引数追加 |
| solver_ncp.py | `_solve_saddle_point_direct()` に `solver_strategy` 引数追加 |
| solver_ncp.py | n_active==0 パスで `solver_strategy.solve()` 直接呼出 |
| solver_ncp.py | Schur complement の K_eff 求解で `_solver.solve()` 直接呼出 |

### 2. PR-4: ContactFrictionSolverProcess（統一プロセス）

NCPQuasiStaticContactFrictionProcess と NCPDynamicContactFrictionProcess を
1クラス `ContactFrictionSolverProcess` に統合。TimeIntegrationStrategy で振る舞い分離。

| ファイル | 変更 |
|---------|------|
| data.py | `SolverFrictionInputData` 統一入力データクラス追加 |
| data.py | 旧入力データに `to_unified()` 変換メソッド追加 |
| solve_contact_friction.py | **新規**: `ContactFrictionSolverProcess` 統一プロセス |
| solve_quasistatic_friction.py | deprecated 化、`ContactFrictionSolverProcess` に委譲 |
| solve_dynamic_friction.py | deprecated 化、`ContactFrictionSolverProcess` に委譲 |
| concrete/__init__.py | `ContactFrictionSolverProcess` をエクスポート |
| process/__init__.py | `SolverFrictionInputData` をエクスポート |

### 3. PR-6: tuning/executor.py NCP 版再実装

全4関数の NotImplementedError スタブを NCP + smooth_penalty + Process API で再実装。

| 関数 | 説明 |
|------|------|
| `execute_s3_benchmark()` | 撚線メッシュ構築→接触設定→smooth_penalty ソルバー実行→TuningRun |
| `run_scaling_analysis()` | 複数素線数（7/19/37/61/91）でスケーリング分析 |
| `run_convergence_tuning()` | パラメータグリッド走査で収束チューニング |
| `run_sensitivity_analysis()` | 2パラメータ直交グリッド感度分析 |

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: 0 件（validate_process_contracts.py）
- 全テスト: 90/90 passed（concrete/tests + linear_solver + block_preconditioner + tuning_schema）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `NCPQuasiStaticContactFrictionProcess` | `ContactFrictionSolverProcess` | status-172 |
| `NCPDynamicContactFrictionProcess` | `ContactFrictionSolverProcess` | status-172 |
| `QuasiStaticFrictionInputData` | `SolverFrictionInputData` | status-172 |
| `DynamicFrictionInputData` | `SolverFrictionInputData` | status-172 |
| `_solve_saddle_point_contact()` string params | `solver_strategy` 直接渡し | status-172 |
| `executor.py` NotImplementedError | NCP版完全再実装 | status-172 |

## 今後の TODO

- [ ] 旧プロセス（NCPQuasiStatic/Dynamic）の呼び出し元を ContactFrictionSolverProcess に移行後、deprecated ラッパー削除
- [ ] executor の `_build_strand_problem()` をより汎用化（カスタム材料定数・メッシュパラメータ対応）
- [ ] Optuna 統合テスト（executor + optuna_tuner の E2E テスト）
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase2 xfail 解消

---
