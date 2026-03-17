# status-199: Process 実行診断インフラ — 警告・エラー・使用レポート

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-17

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-Ok6yi

## 概要

status-198 TODO のプロセス実行監視機能を実装。
全 `AbstractProcess.process()` 呼び出しを自動追跡し、
警告・エラー・使用レポートを生成する診断インフラを構築。

## アーキテクチャ

```
ProcessMetaclass.traced_process()     ← 全 process() 呼び出しをラップ
  → ProcessExecutionLog.record_start()   inspect.stack() で呼び出し元自動検知
  → original process()                    実プロセス実行
  → ProcessExecutionLog.record_end()      実行時間 + 警告記録
  → atexit: write_report()               セッション終了時にレポート出力

docs/generated/process_usage_report.md ← 自動生成レポート
```

## 変更内容

### 1. 新規: xkep_cae/core/diagnostics.py

| クラス/関数 | 種別 | 内容 |
|------------|------|------|
| ProcessExecutionEntry | frozen dataclass | 1回の実行記録（プロセス名, 呼び出し元, 親プロセス, 実行時間, 警告） |
| ProcessExecutionLog | シングルトン | 全プロセス実行を `inspect.stack()` で自動記録 |
| StaticSolverWarning | UserWarning | 準静的ソルバー（NewtonUzawaStaticProcess）使用時の警告 |
| NonDefaultStrategyWarning | UserWarning | デフォルト以外の Strategy 構成使用時の警告 |
| DeprecatedProcessError | RuntimeError | deprecated プロセス実行時のエラー |
| _atexit_report() | atexit hook | セッション終了時に `docs/generated/process_usage_report.md` 自動生成 |

### 2. 修正: xkep_cae/core/base.py — traced_process 強化

- `ProcessExecutionLog.record_start/end()` でプロセス呼び出しを自動記録
- `meta.deprecated == True` のプロセス実行時に `DeprecatedProcessError` を送出
- `warnings.catch_warnings(record=True)` で警告を捕捉し `warning_type` を記録

### 3. 修正: xkep_cae/contact/solver/process.py — 警告追加

- **静的ソルバー警告**: `mass_matrix/dt_physical` 未指定時に `StaticSolverWarning` 発行
- **非デフォルト Strategy 警告**: カスタム strategies 指定時に `NonDefaultStrategyWarning` 発行

### 4. 新規テスト: tests/test_process_diagnostics.py（16テスト）

| クラス | テスト数 | 内容 |
|--------|----------|------|
| TestProcessExecutionLogAPI | 6 | シングルトン, 記録, 親プロセス追跡, 警告記録, リセット |
| TestProcessExecutionLogReport | 4 | 空レポート, エントリ付き, 警告付き, ファイル出力 |
| TestStaticSolverWarning | 2 | UserWarning 継承, 捕捉可能 |
| TestNonDefaultStrategyWarning | 2 | UserWarning 継承, 捕捉可能 |
| TestDeprecatedProcessError | 2 | RuntimeError 継承, メッセージ検証 |

### 5. レポート出力例

テスト実行後の `docs/generated/process_usage_report.md` に以下が自動生成される:

- **プロセス別サマリー**: 呼出回数, 合計/平均時間, 警告数
- **呼び出し元別詳細**: ファイル:関数:行番号 + 親プロセス経由の呼び出し
- **警告一覧**: プロセス名, 警告種別, 呼び出し元

### 6. 呼び出し元追跡の仕組み

`inspect.stack()` を `ProcessMetaclass.traced_process` 内で使用。
呼び出し側のコード変更なしに全プロセスの呼び出し元を自動検知する。

- `base.py`, `diagnostics.py`, `runner.py` のフレーム自動スキップ
- リポジトリルートからの相対パスに変換
- 親プロセス（プロセス内からの呼び出し）もスタックで追跡

## テスト結果

| テスト | 結果 | 備考 |
|--------|------|------|
| 16 passed | ✅ | 診断 API + レポート生成 + 警告/エラー |
| ruff check | エラー 0 件 | |
| ruff format | 全ファイルフォーマット済み | |

## TODO

- [ ] deprecated プロセスの統合テスト（実際のプロセスで `meta.deprecated=True` を設定）
- [ ] pytest conftest に ProcessExecutionLog リセットフック追加（テスト間の分離）
- [ ] 動的ソルバーテスト作成（status-198 TODO引継ぎ）
- [ ] 被膜モデル物理検証テスト（status-198 TODO引継ぎ）

## 開発運用メモ

- `inspect.stack()` はフレームごとに O(1) だが、深いコールスタックでは N フレーム分走査するため微小なオーバーヘッドがある。本番パフォーマンスに影響がある場合は `log.enabled = False` で無効化可能
- `docs/generated/process_usage_report.md` は `.gitignore` に追加推奨（CI/テスト毎に変わるため）
- `StaticSolverWarning` は Python の `-W` フラグで制御可能: `-W ignore::xkep_cae.core.diagnostics.StaticSolverWarning`

---
