# status-171: deprecated薄ラッパー除去 + LinearSolverStrategy 抽出

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-14
**作業者**: Claude Code
**ブランチ**: claude/execute-status-todos-eAJT8

## 概要

status-170 の TODO を消化。deprecated 薄ラッパーの完全除去と、LinearSolverStrategy の Protocol 定義 + 具象クラス抽出を実施。

## 変更内容

### 1. deprecated 薄ラッパー完全除去

status-170 で pure 関数化した staged_activation / initial_penetration の deprecated ラッパーを、
呼び出し元の直接移行により完全削除。

| ファイル | 変更 |
|---------|------|
| solver_ncp.py | 4箇所の `manager.*()` → 純関数 import に変更 |
| solver_smooth_penalty.py | 2箇所の `manager.*()` → 純関数 import に変更 |
| pair.py | deprecated ラッパー10メソッドを完全削除（-154行） |
| test_staged_activation.py | 純関数直接テストに移行（ContactManager 不要化） |
| test_exclude_same_layer.py | count_same_layer_pairs を純関数直接テストに移行 |

**削除したメソッド**:
- `ContactManager.max_layer()` → `staged_activation.max_layer()`
- `ContactManager.compute_active_layer_for_step()` → `staged_activation.compute_active_layer_for_step()`
- `ContactManager.filter_pairs_by_layer()` → `staged_activation.filter_pairs_by_layer()`
- `ContactManager.count_same_layer_pairs()` → `staged_activation.count_same_layer_pairs()`
- `ContactManager.check_initial_penetration()` → `initial_penetration.check_initial_penetration()`
- `ContactManager.adjust_initial_positions()` → `initial_penetration.adjust_initial_positions()`
- `ContactManager.compute_coating_forces()` → CoatingStrategy（呼び出し元なし）
- `ContactManager.compute_coating_stiffness()` → CoatingStrategy（呼び出し元なし）
- `ContactManager.compute_coating_friction_forces()` → CoatingStrategy（呼び出し元なし）
- `ContactManager.compute_coating_friction_stiffness()` → CoatingStrategy（呼び出し元なし）

### 2. LinearSolverStrategy 抽出（PR-3 第1段階）

solver_ncp.py の `_solve_linear_system()` if分岐を Strategy パターンに分離。

| 新規ファイル | 内容 |
|------------|------|
| `xkep_cae/process/strategies/protocols.py` | `LinearSolverStrategy` Protocol 追加 |
| `xkep_cae/process/strategies/linear_solver.py` | 具象3クラス + ファクトリ関数 |
| `tests/contact/test_linear_solver_strategy.py` | Protocol適合性 + 求解精度テスト 15件 |

**具象クラス**:
- `DirectLinearSolver`: spsolve 直接法
- `IterativeLinearSolver`: GMRES + ILU 前処理
- `AutoLinearSolver`: DOF 閾値ベース自動選択（デフォルト）

solver_ncp.py の `_solve_linear_system()` は内部で `create_linear_solver()` に委譲する形に変更。
シグネチャ互換を保ちつつ、Strategy 経由の実行パスに移行。

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: 0 件
- LinearSolverStrategy テスト: 15/15 passed
- テスト収集: 正常

## 今後の TODO

- [ ] PR-3 第2段階: `_solve_saddle_point_contact()` の dispatcher を LinearSolverStrategy ベースに統合
- [ ] PR-4: Process 層 thin wrapper 解消（NCPQuasiStatic/DynamicContactFrictionProcess 統合）
  - 2クラスを1クラスに統合、TimeIntegrationStrategy で振る舞い分離
  - SolverFrictionInputData の統一
- [ ] PR-6: tuning/executor.py NCP 版実装
  - 全4関数が NotImplementedError スタブ（AL依存で廃止済み）
  - NCP + smooth_penalty + Process API ベースで再実装必要
  - Optuna 統合、スケーリング分析、感度分析

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `ContactManager` deprecated 10メソッド | 完全削除（呼び出し元なし） | status-171 |
| `_solve_linear_system()` if分岐 | `LinearSolverStrategy` Protocol + 委譲 | status-171 |

---
