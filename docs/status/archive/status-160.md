# status-160: AI セッション脱線防止 — 構造対策3件

[← README](../../README.md) | [← status-index](status-index.md) | [← status-159](status-159.md)

**日付**: 2026-03-11
**テスト数**: 2477（変更なし — 回帰テスト全通過）

## 概要

AI セッション（Codex / Claude Code）が Process フレームワークリファクタリングに脱線する問題を構造的に対策。

## 背景

- プロジェクトコンテキストが膨大（status 159件、テスト 2477件）
- process-architecture.md §13 の C3-C12 が「TODO」として残存 → AI が自律的に着手
- concrete/ の4プロセスが document_path 不正で import 不可 → 修正圧力が発生

## 実施内容

### A: concrete/ document_path 修正

4つの具象プロセスの `document_path` がソースファイルからの相対パスとして不正だった。
`solve_ncp.py` のみ正しい `../docs/process-architecture.md` だったのを他4ファイルも統一。

- `post_export.py`: `xkep_cae/process/docs/...` → `../docs/...`
- `post_render.py`: 同上
- `pre_contact.py`: 同上
- `pre_mesh.py`: 同上

### B: CLAUDE.md フォーカスガード追加

「やるべきこと / やってはいけないこと」セクションを CLAUDE.md に追加。

- **Phase 7 凍結**を明示（Process リファクタリング禁止）
- S3 スケーリングに集中する指示
- セッション開始時の確認手順を明記

### C: テストフィクスチャのレジストリ汚染除去

`_skip_registry = True` フラグを `AbstractProcess.__init_subclass__` に追加。
テスト用 Dummy プロセスがプロダクション `_registry` に混入しなくなった。

- `base.py`: `_skip_registry` チェック追加
- `test_categories.py`: 5つの Dummy クラスに `_skip_registry = True` 付与
- `test_tree.py`: 3つの Tree テストクラスに `_skip_registry = True` 付与
- `test_categories.py`: `test_concrete_registered` → `test_skip_registry_excludes_test_fixtures` に変更

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/process/base.py` | `_skip_registry` サポート追加 |
| `xkep_cae/process/concrete/post_export.py` | document_path 修正 |
| `xkep_cae/process/concrete/post_render.py` | document_path 修正 |
| `xkep_cae/process/concrete/pre_contact.py` | document_path 修正 |
| `xkep_cae/process/concrete/pre_mesh.py` | document_path 修正 |
| `xkep_cae/process/tests/test_categories.py` | _skip_registry + テスト修正 |
| `xkep_cae/process/tests/test_tree.py` | _skip_registry 追加 |
| `CLAUDE.md` | フォーカスガードセクション追加 |

## 次の TODO

- S3 スケーリング: 37本→61本 NCP 収束
- Phase 7 は凍結のまま（BatchProcess, VerifyProcess, 1:1テスト）
