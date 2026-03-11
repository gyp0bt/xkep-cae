# status-158: SolverStrategies + NCPContactSolverProcess + TimeInteg完全委譲

[← README](../../README.md) | [← status-index](status-index.md) | [← status-157](status-157.md)

**日付**: 2026-03-11
**テスト数**: 2477（変更なし）

## 概要

Process Architecture設計仕様（process-architecture.md §2.4）との乖離を修正。
SolverStrategiesバンドル + NCPContactSolverProcess Wrapper を実装し、
solver_ncp.pyへのTimeIntegrationStrategy完全委譲を完了。

## 実施内容

### 1. 設計仕様乖離分析
- process-architecture.md §2.4 の設計意図と現実の実装を照合
- 主要乖離3点を特定:
  - SolverStrategiesバンドル未実装
  - NCPContactSolverProcess Wrapper未実装
  - TimeInteg初期化のみStrategy化、NRループ内の動的解析コードが旧変数参照のまま

### 2. SolverStrategies dataclass（data.py）
- `SolverStrategies`: penalty / friction / time_integration / contact_force / contact_geometry
- `default_strategies()` ファクトリ: 既存ファクトリ関数を統合
- `SolverInputData.strategies` フィールド追加

### 3. solver_ncp.py strategies引数統合
- `newton_raphson_contact_ncp()` に `strategies: object | None = None` 追加
- strategies経由で Penalty / Friction / TimeIntegration Strategy を取得
- strategies=None 時は既存ファクトリで自動構築（後方互換維持）

### 4. TimeIntegrationStrategy 完全委譲
- NRループ内のインライン動的解析コード（Newmark予測子・補正子・有効剛性・有効残差）を
  `_time_strategy.predict()` / `correct()` / `effective_stiffness()` / `effective_residual()` に置換
- チェックポイント保存/復元を `checkpoint()` / `restore_checkpoint()` に置換
- 収束後の速度・加速度更新を `correct()` に統一
- `_vel`, `_acc`, `_c0`, `_c1`, `_nm_beta`, `_nm_gamma`, `_alpha_m`, `_alpha_f` 等の
  ローカル変数を完全除去
- QuasiStaticProcess に `vel`/`acc` プロパティ追加（安全策）

### 5. NCPContactSolverProcess（concrete/solve_ncp.py）
- `SolverProcess[SolverInputData, SolverResultData]` のWrapper
- 動的依存追跡: `_runtime_uses` で C8対策
- `get_instance_dependency_tree()` でインスタンスレベルの依存ツリー返却
- `process()` で `newton_raphson_contact_ncp()` を呼び出し、結果を `SolverResultData` に変換

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/process/data.py` | SolverStrategies, default_strategies() 追加 |
| `xkep_cae/process/__init__.py` | SolverStrategies, default_strategies エクスポート |
| `xkep_cae/process/strategies/time_integration.py` | checkpoint/restore, is_dynamic, vel/acc プロパティ追加 |
| `xkep_cae/contact/solver_ncp.py` | strategies引数追加、TimeInteg完全委譲、Penalty/Friction strategies統合 |
| `xkep_cae/process/concrete/solve_ncp.py` | NCPContactSolverProcess 新規作成 |
| `xkep_cae/process/concrete/__init__.py` | 公開API更新 |

## 検証

- `pytest xkep_cae/process/` — 202テスト全通過
- `pytest tests/` — 155+ passed（NCP/接触テスト含む）
- `ruff check && ruff format --check` — 全通過

## TODO（次セッション）

- [ ] Phase 5 続き: ContactForceStrategy の solver_ncp.py 注入（3分岐統一）
- [ ] Phase 5 続き: ContactGeometryStrategy の solver_ncp.py 注入（manager.update_geometry委譲）
- [ ] Phase 6: concrete/ の残りプロセス実装（PreProcess, PostProcess等）
- [ ] 37本以上のNCP収束テスト
