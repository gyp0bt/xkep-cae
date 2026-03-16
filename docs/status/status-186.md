# status-186: Phase 6 — C14 強化 + ソルバー deprecated 依存除去

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-6zrpk

## 概要

status-185 の TODO「Phase 6〜8: deprecated 依存の段階的除去」を開始。C14 契約ルールを強化し、importlib 経由の deprecated インポートも契約違反として検出するようにした上で、contact/solver/process.py の deprecated 依存を 9件 → 1件 に削減。

## 変更内容

### 1. C14 強化: importlib deprecated インポート検出

- `validate_process_contracts.py` の `check_c14_deprecated_imports()` を拡張
- `importlib.import_module("xkep_cae_deprecated...")` パターンを AST 解析で検出
- ヘルパー関数呼び出し（`_import_deprecated("xkep_cae_deprecated...")`）も検出
- 強化前: C14 違反 0件（importlib 経由は検出漏れ）
- 強化後: C14 違反 13件 → 修正後 **4件**

### 2. C16 強化: プライベートモジュールスキップ

- `_*.py`（プライベートモジュール）を C16 滅菌チェックの対象外に設定
- 内部実装詳細を新パッケージに移行するための受け皿

### 3. contact/solver プライベートモジュール群の新設（6ファイル）

deprecated 依存なしの純粋な実装を新パッケージに移植:

| ファイル | 移植元 | 内容 |
|---------|--------|------|
| `_utils.py` | `contact/utils.py` | `_deformed_coords`, `_ncp_line_search` |
| `_adaptive_stepping.py` | `process/strategies/adaptive_stepping.py` | `AdaptiveSteppingConfig`, `AdaptiveLoadController` |
| `_diagnostics.py` | `contact/diagnostics.py` | `ConvergenceDiagnostics` |
| `_solver_state.py` | `process/strategies/solver_state.py` | `SolverState`（`_GraphSnapshotList` で ContactGraphHistory 代替） |
| `_newton_uzawa.py` | `process/strategies/newton_uzawa.py` | `NewtonUzawaLoop`, `NewtonUzawaConfig`, `StepResult` |
| `_initial_penetration.py` | `contact/initial_penetration.py` + `contact/geometry.py` | 初期貫入検出 + 最近接点計算（geometry 関数同梱） |
| `_contact_graph.py` | `contact/graph.py` | `_snapshot_contact_graph`（duck typing） |

### 4. ContactFrictionProcess 更新

- deprecated importlib 呼び出し 9件 → 新プライベートモジュールからの import に置換
- `build_result()` → `SolverState.build_u_output()` + 直接 `SolverResultData` 構築に簡素化
- Strategy 生成のみ deprecated 維持（`_create_working_strategies`）: 新 strategies は stub のため

### 5. test_process.py 更新

- `ContactSetupProcess` 経由の接触設定に変更（deprecated contact.pair 直接参照を除去）
- `default_strategies` を `xkep_cae.core.data` から取得に変更

## テスト結果

- 全テスト: **11テスト合格**（contact/solver/tests/test_process.py）
- ruff check: 0 error
- ruff format: 0 issue
- C14 違反: **4件**（Phase 7-8 で対応予定、13件から9件削減）

## C14 残違反

| ファイル | 理由 | 対応予定 |
|---------|------|---------|
| `contact/solver/process.py:55` | Strategy 生成（新 strategies が stub） | Phase 7-8: strategies 完全実装 |
| `contact/setup/process.py:48` | ContactManager 依存 | Phase 7: contact.pair 移行 |
| `mesh/process.py:55` | twisted_wire 依存 | Phase 7: mesh 移行 |
| `output/__init__.py:28` | deprecated output 互換 | Phase 8: output 30+ 関数移行 |

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `contact/solver/process.py` importlib 9箇所 | プライベートモジュール群 + `_create_working_strategies` | status-186 |
| `test_process.py` deprecated contact.pair 直接参照 | `ContactSetupProcess` 経由 | status-186 |

## TODO

- [ ] Phase 7: contact.pair / mesh.twisted_wire の新パッケージ移行（C14 残 2件）
- [ ] Phase 8: output deprecated 関数移行（C14 残 1件）
- [ ] strategies 完全実装（friction/contact_force の stub → 実装移行）
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase 2 xfail 解消

### 6. C16 強化: プライベートモジュール滅菌 + メソッド禁止

- `_*.py` のプライベートモジュール除外ルールを**撤廃**（滅菌対象に復帰）
- C16 に「frozen dataclass のメソッド検出」ルールを追加（dunder 以外のメソッド/プロパティを摘発）
- 全 11 クラスを `frozen=True` に修正 + メソッドを standalone 関数に抽出:
  - 不変データ: `_ContactEdge`, `_ContactGraph`, `_ClosestPointResult`, `StepResult`, `NewtonUzawaConfig` → frozen（メソッドなし）
  - `ConvergenceDiagnostics`: frozen + `format_report()` → `_format_diagnostics_report()` 関数に
  - `_GraphSnapshotList`: **削除**（`list[object]` に統合）
  - `SolverState`: frozen + 全メソッドを `_save_checkpoint()` / `_restore_checkpoint()` / `_ensure_lam_size()` / `_build_u_output()` / `_state_set()` 関数に
  - `AdaptiveLoadController` → **`AdaptiveSteppingProcess`**（SolverProcess 化）+ `AdaptiveStepInput`/`AdaptiveStepOutput` frozen dataclass + `StepAction` Enum
  - `NewtonUzawaLoop` → **`NewtonUzawaProcess`**（SolverProcess 化）+ `NewtonUzawaStepInput` frozen dataclass
- `process.py` の `_ctrl_*` 関数呼び出しを `AdaptiveSteppingProcess.process()` API に全置換
- `ContactFrictionProcess.uses` に `AdaptiveSteppingProcess` 追加
- テスト: **20 テスト合格**（+3 NewtonUzawaProcess + 6 AdaptiveSteppingProcess API テスト）

## 懸念事項・メモ

- **Strategy stub 問題**: 新パッケージの friction/contact_force strategies は stub（ゼロ返却）。ContactFrictionProcess.process() は deprecated 版の strategies を使い続ける必要がある。strategies 完全実装が Phase 7-8 の最大課題。
- SolverState.build_result() → NCPSolveResult のパスを廃止し、SolverResultData を直接構築するよう簡素化。これにより NCPSolveResult / ContactGraphHistory への deprecated 依存を除去。

---
