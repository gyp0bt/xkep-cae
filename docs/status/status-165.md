# status-165: Phase 8 完遂 + Phase 9-A/B 実装 — CompatibilityProcess 移行 + 走査化 + StrategySlot 完全移行

[← README](../../README.md) | [← status-index](status-index.md) | [← status-164](status-164.md)

**日付**: 2026-03-14
**テスト数**: 2477 + 314 process テスト

## 概要

status-164 の TODO 3件を消化し Phase 8 を完遂。さらに Phase 9-A/9-B を実装完了。

1. ManualPenaltyProcess の CompatibilityProcess 移行 → C13 実効化
2. `_import_all_modules()` ファイルシステム走査化 → モジュールリストのハードコード廃止
3. `_runtime_uses` 廃止 → `StrategySlot` + `collect_strategy_types()` に完全移行

## 実施内容

### Phase 8 完遂: ManualPenaltyProcess → CompatibilityProcess 移行

- `penalty.py`: 基底クラスを `SolverProcess` → `CompatibilityProcess` に変更
- `test_penalty.py`: `test_is_compatibility_process` テスト追加
- `ci.yml`: test-process ジョブのテスト数コメントを `~277` → `~315` に修正
- 契約違反 0件確認（C13 実効化）

### Phase 9-A: _import_all_modules() ファイルシステム走査化

- `validate_process_contracts.py`: ハードコードされたモジュールリスト（14件）を
  `xkep_cae/process/` 配下の全 `.py` ファイルの `rglob` 走査に置換
- 除外対象: `__init__.py`, `base.py`, `categories.py`, `data.py`, `slots.py`, `tree.py`, `runner.py`, テストファイル
- テストファイルは `test_*.py` パターンで `rglob` 走査（既存と同じ）
- **効果**: 新規プロセス追加時にモジュールリスト更新忘れで契約チェック対象外になる問題を根絶

### Phase 9-B: _runtime_uses → StrategySlot 完全移行

`_runtime_uses` 属性を廃止し、`effective_uses()` が `collect_strategy_types()` を直接利用する方式に移行。

**変更ファイル:**

| ファイル | 変更内容 |
|---------|---------|
| `base.py` | `effective_uses()`: `_runtime_uses` → `collect_strategy_types()` 直接呼び出し |
| `solve_ncp.py` | `self._runtime_uses = ...` 行を削除。`get_instance_dependency_tree()` も直接呼び出しに |
| `tree.py` | `_runtime_uses` チェックを `effective_uses()` に統合。uses/StrategySlot の出自を明記 |
| `validate_process_contracts.py` | C8 チェックを `check_c8_strategy_slot()` に改名・簡素化 |
| `slots.py` | docstring 更新（_runtime_uses 廃止を反映） |
| `test_solve_ncp.py` | `_runtime_uses` テスト → `effective_uses_from_slots` + `no_runtime_uses_attribute` に置換 |
| `test_tree.py` | `_runtime_uses` 直接設定テスト → `effective_uses` ベースに書き換え |

### 機械的ガードの現状評価

| チェック | 状態 | 残りの穴 |
|---------|------|---------|
| C3 テスト紐付け | OK | deprecated はスキップ（意図的） |
| C5 未宣言依存 | OK | 動的 import 検出不可（現状未使用） |
| C6 Strategy 意味論 | OK | テストの「質」は検証不可 |
| C7 メタクラスラップ | OK | — |
| C8 StrategySlot | OK | **Phase 9-B で簡素化完了** |
| C9 frozen 不変性 | OK | — |
| C11 推移的依存 | OK | 間接呼び出し検出不可 |
| C12 Batch 順序 | OK | — |
| C13 Compat uses 禁止 | OK | **Phase 8 で実効化** |

## TODO（次セッション）

- [ ] S3 凍結解除判断（9-C）
- [ ] Phase 9-D: BatchProcess パイプライン改善（S3 再開しない場合）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行 |
|--------|--------|------|
| `_runtime_uses` | `collect_strategy_types()` + `effective_uses()` | Phase 9-B（本status） |
| ハードコードモジュールリスト | `rglob` ファイルシステム走査 | Phase 9-A（本status） |
