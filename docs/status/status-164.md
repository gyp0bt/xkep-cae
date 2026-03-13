# status-164: Phase 8 実装完了 — ProcessRunner / StrategySlot / Preset / C13

[← README](../../README.md) | [← status-index](status-index.md) | [← status-163](status-163.md)

**日付**: 2026-03-13
**テスト数**: 2477（回帰テスト変更なし）+ 314 process テスト（275→314: +39テスト）

## 概要

status-163 の TODO 6件（Phase 8-A〜8-F）を全て実装完了。
Process Architecture に実行管理・型安全性・検証済みプリセットの基盤を追加。

## 実施内容

### 8-A: ProcessRunner / ExecutionContext（runner.py）

プロセス実行時の依存チェック・プロファイリング・ログ出力を一元管理。

- `ExecutionContext`: dry_run / profile / log_file / validate_deps / checksum_inputs
- `ProcessRunner.run()`: execute() 代替の統一実行メソッド
- `ProcessRunner.run_pipeline()`: 複数プロセス順次実行
- `ProcessRunner.get_report()`: 実行ログサマリー
- **13テスト追加**（test_runner.py）

### 8-B: StrategySlot ディスクリプタ（slots.py）

Strategy slot の型付きディスクリプタ。Protocol 準拠を `__set__` 時に検証。

- `StrategySlot(protocol, required=True)`: クラス変数として宣言
- `collect_strategy_slots()`: クラスの全スロットを走査
- `collect_strategy_types()`: _runtime_uses の代替（型安全版）
- **12テスト追加**（test_slots.py）

### 8-C: CompatibilityProcess カテゴリ（categories.py）

deprecated プロセスの隔離カテゴリ。C13 チェックと連動。

- `CompatibilityProcess(AbstractProcess, ABC)` を追加
- `__init__.py` にエクスポート追加
- **1テスト追加**（test_categories.py 既存クラスに追加）

### 8-D: SolverPreset ファクトリ（presets.py）

検証済み Strategy 組み合わせのファクトリパターン。

- `SolverPreset(name, factory, verified_by, description, default_overrides)`
- `create(**kwargs)` でランタイムパラメータ（ndof, beam_E 等）を受け取り
- 組み込みプリセット: `PRESET_SMOOTH_PENALTY`, `PRESET_NCP_FRICTIONLESS`
- `get_preset(name)` / `get_presets()` レジストリAPI
- **10テスト追加**（test_presets.py）

設計文書からの変更点:
- Preset は SolverStrategies を直接保持 → ファクトリパターンに変更
  - 理由: Strategy 具象クラスが必須引数（ndof, beam_E, L_elem 等）を持つため
  - `create()` で `default_strategies()` を呼び出し、パラメータを注入

### 8-E: NCPContactSolverProcess StrategySlot 統合（solve_ncp.py）

既存の `_runtime_uses` を StrategySlot ベースに移行（後方互換維持）。

- 5軸の StrategySlot 宣言: penalty/friction/time_integration/contact_force/contact_geometry
- `_runtime_uses = collect_strategy_types(self)` で構築
- **3テスト追加**（test_solve_ncp.py: 6→9テスト）

### 8-F: validate_process_contracts.py 更新

- **C8 更新**: StrategySlot 整合性チェック追加（collect_strategy_types vs _runtime_uses）
- **C13 新規**: active プロセスが CompatibilityProcess を uses している場合はエラー
- ヘッダーを C3-C13 に更新

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/process/runner.py` | **新規** ProcessRunner + ExecutionContext |
| `xkep_cae/process/slots.py` | **新規** StrategySlot ディスクリプタ |
| `xkep_cae/process/presets.py` | **新規** SolverPreset ファクトリ |
| `xkep_cae/process/categories.py` | CompatibilityProcess 追加 |
| `xkep_cae/process/__init__.py` | エクスポート追加（6クラス/関数） |
| `xkep_cae/process/concrete/solve_ncp.py` | StrategySlot 統合 |
| `xkep_cae/process/concrete/tests/test_solve_ncp.py` | StrategySlot テスト3件追加 |
| `xkep_cae/process/tests/test_runner.py` | **新規** 13テスト |
| `xkep_cae/process/tests/test_slots.py` | **新規** 12テスト |
| `xkep_cae/process/tests/test_presets.py` | **新規** 10テスト |
| `xkep_cae/process/tests/test_categories.py` | CompatibilityProcess テスト1件追加 |
| `scripts/validate_process_contracts.py` | C8 StrategySlot 対応 + C13 新規 |
| `docs/roadmap.md` | Phase 8 完了記録 |
| `README.md` | テスト数・Phase 8 完了反映 |
| `CLAUDE.md` | 現在の状態・フォーカスガード更新 |
| `docs/status/status-164.md` | 本ファイル |
| `docs/status/status-index.md` | インデックス追加 |

## TODO（次セッション）

- [ ] ManualPenaltyProcess を CompatibilityProcess に移行（8-C の実適用）
- [ ] Phase 9 計画策定（batch パイプライン改善 or S3 凍結解除）
- [ ] CI の test-process ジョブに新テスト反映確認

## 運用メモ

- SolverPreset のファクトリパターンは設計文書からの正当な逸脱。
  Strategy 具象クラスの __init__ 引数が実行時依存のため、
  Preset 定義時にインスタンス化できない。
  この問題は Phase 8 設計書 §D リスク 2 で事前に特定されていた。
- StrategySlot と _runtime_uses の共存は移行期間の後方互換策。
  collect_strategy_types() への完全移行は Phase 9 以降。
