# status-182: C16 スコープ拡大 + time_integration 移動 + process/ 削除

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/refactor-c16-scope-BM3Wx

## 概要

status-181 の TODO に基づく3つのリファクタリング:

1. **C16 検知スコープ拡大**: core/ 内および core からのインポートを除外し、xkep_cae/ 配下の全モジュールに検知範囲を拡大
2. **time_integration 移動**: core/ 内の concrete 実装を xkep_cae/time_integration/ に移動（core は Protocol/基盤のみに）
3. **process/ フォルダ削除**: re-export 互換レイヤーを完全削除

## 変更内容

### 1. C16 検知スコープ拡大

`scripts/validate_process_contracts.py`:
- `check_c16_sterilization()`: scan_roots を `core/strategies/` + `contact/` から xkep_cae/ 配下の全サブパッケージ（core/ 除外）に変更
- `_import_all_modules()`: 走査対象を xkep_cae/ 配下の全サブパッケージに拡大
- これにより elements/, materials/, mesh/ 等の全モジュールが C16 滅菌チェック対象に

### 2. time_integration 移動

- `xkep_cae/core/time_integration/` -> `xkep_cae/time_integration/`
- `TimeIntegrationOutput` を `frozen=True` に修正（C16 検出）
- インポートパス更新: core/data.py, core/batch/strand_bending.py, tests/

### 3. process/ フォルダ削除

- `xkep_cae/process/__init__.py`（re-export shim）を完全削除
- 外部参照の更新:
  - `scripts/benchmark_process_overhead.py`: core/contact パスに移行
  - `tests/contact/test_solver_contact.py`: contact.geometry に移行
  - `tests/contact/test_block_preconditioner.py`: contact.geometry に移行
  - `tests/contact/test_linear_solver_strategy.py`: deprecated パスに修正

## テスト結果

- ruff check: 0 error（変更ファイル）
- ruff format: 0 issue（変更ファイル）
- 契約違反: **0件**（C16 拡大後も維持）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `xkep_cae/core/time_integration/` | `xkep_cae/time_integration/` | status-182 |
| `xkep_cae/process/`（re-export shim） | 完全削除（core 直接参照） | status-182 |
| C16: `core/strategies/` + `contact/` のみ | C16: core/ 以外の全モジュール | status-182 |
| `TimeIntegrationOutput`（non-frozen） | `TimeIntegrationOutput(frozen=True)` | status-182 |

## TODO

- [ ] Phase 3: concrete プロセス移行（Mesh/Setup/Export/Verify）
- [ ] StrandBendingBatchProcess フル実装

---
