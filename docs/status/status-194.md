# status-194: xkep_cae_deprecated → __xkep_cae_deprecated リネーム

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-17

## 作業者

Claude Code

## ブランチ

claude/rename-deprecated-function-23azP

## 概要

status-193 の TODO「`xkep_cae_deprecated` → `__xkep_cae_deprecated` リネーム」を実行。
ダブルアンダースコアプレフィックスにより、旧パッケージへの偶発的インポートを Python レベルで抑止する。

## 変更内容

### 1. ディレクトリリネーム

```
xkep_cae_deprecated/ → __xkep_cae_deprecated/
```

`git mv` で一括リネーム。

### 2. 全参照の一括置換

| 対象 | ファイル数 |
|------|-----------|
| `.py` ファイル | 162 |
| `.md` ファイル | 26 |

`sed` で `xkep_cae_deprecated` → `__xkep_cae_deprecated` を一括置換。

### 3. 違反検知スクリプト更新

`scripts/validate_process_contracts.py` の C14 検出ロジックが新名 `__xkep_cae_deprecated` を正しく検出することを確認:

- `import __xkep_cae_deprecated...`
- `from __xkep_cae_deprecated... import ...`
- `importlib.import_module("__xkep_cae_deprecated...")`

### 4. tests/conftest.py 更新

`pytest_ignore_collect` フックの検出文字列を `__xkep_cae_deprecated` に更新。

### 5. ruff lint 修正

`tests/contact/test_linear_solver_strategy.py` の import 順序を `ruff check --fix` で修正。

## C14 実効性強化の効果

| 項目 | 旧（`xkep_cae_deprecated`） | 新（`__xkep_cae_deprecated`） |
|------|-----|-----|
| 偶発 import | `from xkep_cae_deprecated.xxx import yyy` が通る | ダブルアンダースコアで意図的アクセスを示す |
| IDE 補完 | `xkep_` で候補に出る | `__` prefix で候補から除外されやすい |
| C14 検出 | 文字列検出のみ | 文字列検出 + Python 規約で明示的に「内部」 |

## テスト結果

- ruff check: エラー 0 件
- ruff format: 全ファイルフォーマット済み
- 旧名 `xkep_cae_deprecated`（`__` なし）の残存参照: **0 件**

## TODO

- [ ] `numerical_tests` モジュールの新 xkep_cae への移植（大規模: ~1400行の benchmark 含む）
- [ ] S3 xfail テストの Process API 対応版作成

---
