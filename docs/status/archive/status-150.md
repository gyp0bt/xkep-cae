# status-150: プロセスアーキテクチャ設計仕様策定

[← status-index](status-index.md) | [← README](../../README.md)

**日付**: 2026-03-10
**テスト数**: 2271（変更なし）

## 概要

NCP + Uzawa + smooth_penalty を基軸とするソルバー構成のリファクタリング設計仕様を策定した。
AbstractProcess による契約化とStrategy分解により、依存関係の機械的検証とソルバー内部の拡張性を両立する。

## 成果物

1. **設計仕様書**: [xkep_cae/process/process-architecture.md](../../xkep_cae/process/process-architecture.md)
   - AbstractProcess基底クラス（メタクラスによる自動トレース + プロファイリング）
   - ソルバー内部のStrategy分解（5軸: ContactForce/Friction/TimeIntegration/ContactGeometry/Penalty）
   - プロセス分類体系（Pre/Solver/Post/Verify/Batch）
   - Input/Outputデータ契約（dataclass(frozen=True)）
   - テストコロケーション（実装コードのそばに test_*.py + .spec.md を配置、1:1対応）
   - ProcessTree（実行グラフ）
   - 依存関係バリデーション（__init_subclass__ + AST解析のハイブリッド）
   - Deprecated管理（ProcessMeta.deprecated + used_by逆リンク）
   - Strategy互換性マトリクス（ホワイトリスト + ブラックリスト）
   - 契約抜け腐敗シナリオ分析（C1-C16 + 未確定事項U1-U5）

## 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| プロセス粒度 | 粗粒度（NR内部はプロセス化しない） | ホットパスのオーバーヘッド回避 |
| 依存関係検証 | __init_subclass__ + AST解析ハイブリッド | 静的チェック + CI動的チェックの併用 |
| 既存型互換 | ラッパー方式（NCPSolverInputを内部で使い続ける） | 既存2271テスト不退行 |
| テスト配置 | コロケーション（実装そばに配置） | 発見性と保守性 |
| テスト対応 | 1:1（1プロセス = 1テストクラス） | 横断テストはBatchProcessで |
| ソルバー内部 | Strategy Process（Protocol降格メカニズム付き） | 流動的な技術検証段階を支援 |

## リファクタリングロードマップ

| Phase | セッション | 内容 |
|-------|-----------|------|
| 1 | 1-2 | 基盤 + Strategy Protocol |
| 2 | 3-4 | Strategy実装（solver_ncp.pyからの抽出） |
| 3 | 5-6 | 具体プロセス実装 |
| 4 | 7-8 | 検証プロセス移行 |
| 5 | 9-10 | バッチ・統合・クリーンアップ |

## 契約抜け腐敗シナリオ

- **検出可能**: C1-C5（__init_subclass__ + AST解析で自動検出）
- **検出困難**: C6-C12（設計的対策が必要、各Phaseで段階対応）
- **対応不能**: C13-C16（Python言語仕様の限界として受容）
- **未確定**: U1-U5（判断時期と基準を明記）

## 次のアクション

- Phase 1 開始: `xkep_cae/process/base.py` + `test_base.py` の実装
- Strategy Protocol の定義と契約テスト
