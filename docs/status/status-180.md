# status-180: C16 契約ギャップ修正 — __init__.py re-export チェック強化

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-15

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-FnGAk

## 概要

C16 滅菌チェックが `__init__.py` をスキップしていたため、
`_create_*_strategy` ファクトリ関数が `__init__.py` で公開名（`_` なし）に
re-export されていた問題を修正。

## 変更内容

### 1. `__init__.py` からファクトリ関数の公開エクスポートを削除

4ファイルから `_foo as foo` パターンの re-export を除去:

- `xkep_cae/contact/friction/__init__.py`
- `xkep_cae/contact/contact_force/__init__.py`
- `xkep_cae/contact/geometry/__init__.py`
- `xkep_cae/core/time_integration/__init__.py`

### 2. テストのインポートを直接 `_` prefix 付きに修正

4テストファイルのインポートを `strategy.py` からの直接インポートに変更:

- `xkep_cae/contact/friction/tests/test_strategy.py`
- `xkep_cae/contact/contact_force/tests/test_strategy.py`
- `xkep_cae/contact/geometry/tests/test_strategy.py`
- `xkep_cae/core/time_integration/tests/test_strategy.py`

### 3. `data.py` のインポートパス修正

`xkep_cae/core/data.py` の `default_strategies()`:
- 存在しない旧パス（`core.strategies.contact_force` 等）を新パッケージに修正
- ファクトリ関数を `_` prefix 付きで直接インポート
- penalty / coating は未移行のため `None` を設定（Phase 3 で対応予定）

### 4. C16 検証スクリプト強化

`scripts/validate_process_contracts.py`:
- `__init__.py` を `_SKIP_STEMS` から除外
- `__init__.py` 専用の re-export チェックを追加:
  - `_foo as foo` パターン（private 関数の公開エイリアス）を検出
  - 小文字開始の公開関数 re-export を検出

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 全テスト: 186 passed
- 契約違反: 0件

## TODO

- [ ] penalty ファクトリ関数の実装（`_create_penalty_strategy`）
- [ ] coating ファクトリ関数の移行

---
