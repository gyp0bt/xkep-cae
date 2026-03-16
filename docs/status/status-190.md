# status-190: solver 内部 Process 化完了 + NewtonUzawa Static/Dynamic 完全分離

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-w2l4E

## 概要

solver 内部のプライベートモジュール全5件を Process 化完了。NewtonUzawaProcess を Static/Dynamic に完全分離し、実ロジックを2つに冗長化。301テスト全パス、C14/C16 契約違反ゼロ維持。

## 変更内容

### 1. solver プライベートモジュール Process 化（5件完了）

既に Process 化済みの `_adaptive_stepping.py` と `_newton_uzawa.py` に加え、残り5モジュールを Process 化:

| モジュール | Process クラス | Input/Output |
|-----------|---------------|-------------|
| `_initial_penetration.py` | `InitialPenetrationProcess` | `InitialPenetrationInput` → `InitialPenetrationOutput` |
| `_contact_graph.py` | `ContactGraphProcess` | `ContactGraphInput` → `ContactGraphOutput` |
| `_diagnostics.py` | `DiagnosticsReportProcess` | `DiagnosticsInput` → `DiagnosticsOutput` |
| `_solver_state.py` | `SolverStateInitProcess` | `SolverStateInitInput` → `SolverStateInitOutput` |
| `_utils.py` | `DeformedCoordsProcess`, `NCPLineSearchProcess` | 各 frozen dataclass I/O |

全て `SolverProcess` 継承 + `ProcessMeta` + `@binds_to` テスト付き。

### 2. NewtonUzawa Static/Dynamic 完全分離

`_newton_uzawa.py` を3ファイルに分割:

- **`_newton_uzawa_static.py`**: `NewtonUzawaStaticProcess` — 純静的版（慣性力・減衰力なし）
  - `NewtonUzawaStaticConfig`, `NewtonUzawaStaticStepInput`, `StaticStepResult`
  - `dt_sub` パラメータなし、`_time_strategy` 使用なし
- **`_newton_uzawa_dynamic.py`**: `NewtonUzawaDynamicProcess` — 動的版（Generalized-α 時間積分）
  - `NewtonUzawaDynamicConfig`, `NewtonUzawaDynamicStepInput`, `DynamicStepResult`
  - `dt_sub` 必須、`effective_residual()`/`effective_stiffness()` 使用
- **`_newton_uzawa.py`**: 後方互換エイリアス（`NewtonUzawaProcess = NewtonUzawaStaticProcess`）

`process.py` を更新: `_dynamics` フラグで Static/Dynamic を完全に分岐。

### 3. テスト更新

- `TestNewtonUzawaStaticProcessAPI` + `TestNewtonUzawaDynamicProcessAPI` 追加
- 後方互換エイリアステスト（`NewtonUzawaProcess is NewtonUzawaStaticProcess`）
- 37 solver テスト全パス

## テスト結果

- **301 passed** (xkep_cae/ 全体)
- **37 passed** (solver テスト)
- C14/C16 契約違反: **0件**

## 互換ヒストリー

| 旧 | 新 | status |
|----|----|----|
| `NewtonUzawaProcess`（統合型） | `NewtonUzawaStaticProcess` + `NewtonUzawaDynamicProcess` | status-190 |
| solver 純関数5モジュール | Process 化（SolverProcess 継承） | status-190 |

## TODO

- [ ] friction/geometry Strategy の `process.py` 内 deprecated Strategy 直接使用 → 新 Strategy 経由に移行
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動 Phase2 xfail 解消

---
