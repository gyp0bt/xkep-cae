# status-170: テスト名正規化 + StagedActivation/InitialPenetration 純関数化

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-14
**作業者**: Claude Code
**ブランチ**: claude/execute-status-todos-Kazhy

## 概要

status-169 の TODO（PR-2, PR-5）を消化。AL削除後のテスト名正規化と、pair.py の責務軽量化を実施。

## 変更内容

### PR-5: テスト名正規化（_ncp 除去）

AL solver 完全削除（status-167）に伴い、NCP が唯一のソルバーとなったため
冗長な `_ncp`/`NCP` サフィックスを除去。

| 種別 | 件数 | 内容 |
|------|------|------|
| ファイルリネーム | 12件 | `test_*_ncp.py` → `test_*.py`, `test_solver_ncp*.py` → `test_solver_contact*.py` |
| クラス名リネーム | 53件 | `TestXxxNCP` → `TestXxx`, `TestNCPXxx` → `TestXxx` |
| メソッド名リネーム | 19件 | `test_ncp_*` → `test_*`, `_solve_*_ncp` → `_solve_*` |
| docstring参照更新 | 5件 | テストファイル名参照の修正 |

**除外**: `test_ncp.py`（NCP数学関数テスト）は意味的に正しいため除外。

### PR-2: StagedActivation + InitialPenetration 純関数化

ContactManager の6メソッドを純関数モジュールに抽出し、pair.py を130行削減。

| 新規モジュール | 抽出関数 |
|--------------|---------|
| `xkep_cae/contact/staged_activation.py` | `max_layer()`, `compute_active_layer_for_step()`, `filter_pairs_by_layer()`, `count_same_layer_pairs()` |
| `xkep_cae/contact/initial_penetration.py` | `check_initial_penetration()`, `adjust_initial_positions()` |

pair.py のメソッドは deprecated 薄ラッパーに変更。後方互換性維持。

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: 0 件
- pytest collect: 正常（全リネーム済みテストがcollect可能）

## 今後の TODO

- [ ] PR-3: LinearSolverStrategy 抽出（solver_ncp.py if分岐 Strategy 化）
- [ ] PR-4: Process 層 thin wrapper 解消（StrategySlot 実活用）
- [ ] PR-6: tuning/executor.py NCP 版実装
- [ ] solver_ncp.py / solver_smooth_penalty.py の呼び出し元を純関数に直接移行（deprecated 薄ラッパー除去準備）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `ContactManager.max_layer()` 他5メソッド | `staged_activation.*()` / `initial_penetration.*()` | status-170 |
| テスト名 `*_ncp.*` | `*.*`（NCP サフィックス除去） | status-170 |

---
