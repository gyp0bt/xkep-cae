# status-168: NCP ソルバーリファクタリング — solver_ncp.py 分離 + Process 移行

[← README](../../README.md) | [status-index](status-index.md)

## 日付

2026-03-14

## 概要

solver_ncp.py（3123行）から smooth_penalty パスと Strategy 重複 raw ヘルパーを分離・削除。
NCPContactSolverProcess を完全削除し、新 Process（動的/準静的）に移行。

## 変更内容

### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/contact/diagnostics.py` | ConvergenceDiagnostics, NCPSolveResult, NCPSolverInput（solver_ncp.py から移動） |
| `xkep_cae/contact/utils.py` | deformed_coords, ncp_line_search（公開化） |
| `xkep_cae/contact/solver_smooth_penalty.py` | solve_smooth_penalty_friction()（Strategy 経由 smooth penalty ソルバー） |
| `xkep_cae/process/concrete/solve_dynamic_friction.py` | NCPDynamicContactFrictionProcess（Generalized-α 動的解析） |
| `xkep_cae/process/concrete/solve_quasistatic_friction.py` | NCPQuasiStaticContactFrictionProcess（準静的解析） |

### 削除

| 対象 | 理由 |
|------|------|
| `xkep_cae/process/concrete/solve_ncp.py` | NCPContactSolverProcess — 新 Process で完全置換 |
| `xkep_cae/process/concrete/tests/test_solve_ncp.py` | 上記のテスト |
| `SolverInputData` (data.py) | NCPContactSolverProcess 専用 — 不要 |
| solver_ncp.py smooth_penalty パス (~230行) | solver_smooth_penalty.py に移行 |
| solver_ncp.py `_compute_contact_force_from_lambdas` | Strategy 重複 |
| solver_ncp.py `_compute_friction_forces_ncp` | Strategy 重複 |
| solver_ncp.py `_build_friction_stiffness` | Strategy 重複 |
| solver_ncp.py `_build_constraint_jacobian` | Strategy 重複（contact_geometry.py に移植済み） |

### 変更

| ファイル | 変更内容 |
|---------|---------|
| `solver_ncp.py` | NCP 鞍点系専用に縮小（~600行削減） |
| `contact/__init__.py` | diagnostics.py, utils.py, solver_smooth_penalty.py からの import 追加 |
| `process/batch/strand_bending.py` | NCPQuasiStaticContactFrictionProcess に差し替え |
| `process/concrete/__init__.py` | 新 Process をエクスポート |
| `process/__init__.py` | DynamicFrictionInputData, QuasiStaticFrictionInputData 公開 |
| テスト4ファイル | 削除関数の import を Strategy 等価物に移行 |

## 検証

- `ruff check` / `ruff format`: パス
- `validate_process_contracts.py`: 契約違反なし
- `pytest tests/contact/test_solver_ncp.py`: 16テストパス
- `pytest xkep_cae/process/concrete/tests/`: 32テストパス

## 互換ヒストリー

| 旧 | 新 | 備考 |
|----|-----|------|
| `NCPContactSolverProcess` | `NCPDynamicContactFrictionProcess` / `NCPQuasiStaticContactFrictionProcess` | 完全削除（deprecated なし） |
| `SolverInputData` | `DynamicFrictionInputData` / `QuasiStaticFrictionInputData` | 完全削除 |
| solver_ncp.py smooth_penalty パス | `solve_smooth_penalty_friction()` | Strategy 経由 |
