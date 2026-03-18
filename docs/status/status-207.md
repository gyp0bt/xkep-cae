# status-207: deprecated コード完全削除 + コンテキスト大掃除

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**ブランチ**: `claude/cleanup-deprecated-code-3610q`
**テスト数**: 394 passed（fast テスト、slow 97 deselected）

---

## 概要

status-206 の TODO に基づき、旧コード・旧テスト・旧ドキュメントを大量削除。
`__xkep_cae_deprecated/` ディレクトリを完全削除し、新 xkep_cae のみのコードベースにした。

## 削除内容

### 1. scripts/ — 30ファイル + docs/ 削除（ディレクトリごと削除）

`validate_process_contracts.py` を `contracts/` に移動し、残りの 30 スクリプトを全削除。
旧 API（`xkep_cae.elements.beam_timo3d`, `xkep_cae.sections.beam`, `xkep_cae.contact.pair` 等）
に依存しており実行不能だった。

**残存**: `contracts/validate_process_contracts.py` のみ

### 2. tests/ — deprecated 参照テスト + 旧機能テスト削除

| 削除対象 | 件数 | 理由 |
|---------|------|------|
| `tests/test_tuning_schema.py` | 1 | `__xkep_cae_deprecated.tuning` 参照 |
| `tests/test_inp_runner.py` | 1 | `__xkep_cae_deprecated.io` 参照 |
| `tests/contact/test_linear_solver_strategy.py` | 1 | `__xkep_cae_deprecated.process.strategies` 参照 |
| `tests/test_benchmark_cutter_*.py` | 2 | `xkep_cae.api`, `xkep_cae.io.abaqus_inp` (削除済みAPI) |
| `tests/test_inp_metadata_validation.py` | 1 | `scripts._run_bending_oscillation` (削除済みモジュール) |
| `tests/thermal/` | 7 | `xkep_cae.thermal` (未移行 stub のみ) |
| `tests/mesh/` | 0 | 空ディレクトリ（`__init__.py` のみ） |

### 3. `__xkep_cae_deprecated/` — 完全削除（129ファイル）

新 xkep_cae からの実行時依存ゼロを確認済み（全参照はコメント・docstring のみ）。

| モジュール | ファイル数 |
|-----------|-----------|
| core/ | 5 |
| elements/ | 12 |
| materials/ | 5 |
| sections/ | 4 |
| math/ | 2 |
| contact/ | 18 |
| mesh/ | 4 |
| io/ | 5 |
| output/ | 9 |
| thermal/ | 7 |
| numerical_tests/ | 8 |
| process/ (含 strategies/concrete/verify/batch) | 37 |
| tuning/ | 5 |
| ルート (api/solver/dynamics/bc/assembly) | 8 |

### 4. examples/ — 全削除（8ファイル）

旧 API (`__xkep_cae_deprecated.io`, `xkep_cae.assembly` 等) 依存。

### 5. deprecated 内ドキュメント — 全削除（22 md + 5 空 __init__.py）

deprecated/contact/docs/, elements/docs/, mesh/docs/, output/docs/, process/docs/, process/strategies/docs/ を全削除。

### 6. docs/ 旧ドキュメント削除

| 削除対象 | 理由 |
|---------|------|
| `docs/archive/` | 旧フェーズ詳細設計 |
| `docs/reference/` | 旧 API 使用例・Abaqus差異 |
| `docs/verification/` | 旧検証画像・ギャラリー・カタログ（~8MB） |

### 7. 未移行 stub モジュール削除

| モジュール | 内容 |
|-----------|------|
| `xkep_cae/io/` | 空 stub (docstring のみ) |
| `xkep_cae/thermal/` | 空 stub |
| `xkep_cae/math/` | 空 stub |
| `xkep_cae/materials/` | 空 stub |
| `xkep_cae/sections/` | 空 stub |

### 8. docstring 整理（22ファイル）

「旧 __xkep_cae_deprecated からの完全書き直し」「未移行」等の移行履歴参照を削除。

### 9. ドキュメント更新

| ファイル | 変更内容 |
|---------|---------|
| `CLAUDE.md` | 互換ヒストリー圧縮、C14パス更新、フォーカスガード更新 |
| `README.md` | deprecated セクション削除、パッケージ構成更新 |
| `docs/roadmap.md` | R1 詳細→テーブル圧縮、削除済みリンク修正 |
| `.github/workflows/ci.yml` | `scripts/` → `contracts/` パス更新 |

### 10. テスト修正

`contact_force/tests/test_strategy.py`: Mock クラスを実際の frozen dataclass (`_ContactStateOutput`, `_ContactPairOutput`) に置き換え。`_evolve_pair`/`_evolve_state` との互換性確保。

## テスト結果

```
394 passed, 97 deselected (slow/external), 4284 warnings
ruff check: All checks passed
ruff format: 111 files already formatted
```

## 削除統計

| カテゴリ | 削除数 |
|---------|-------|
| Python ファイル | ~170 |
| Markdown ファイル | ~30 |
| 画像ファイル | ~100 |
| ディレクトリ | ~30 |

## 次のタスク

- [ ] BackendRegistry 完全廃止（O2 条例違反解消）
- [ ] 被膜モデル物理検証テスト
- [ ] contracts/validate_process_contracts.py の C14 チェッカー更新（deprecated ディレクトリ不在でも動作確認）
