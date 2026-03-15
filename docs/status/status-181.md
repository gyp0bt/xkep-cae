# status-181: Penalty/Coating ファクトリ関数完備 — default_strategies() 7軸全生成

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-15

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-UKj4e

## 概要

status-180 の TODO 2件を完了:
1. `_create_penalty_strategy` ファクトリ関数の実装
2. Coating パッケージの deprecated → 新 xkep_cae への移行

これにより `default_strategies()` が 7軸全 Strategy を生成するようになった（penalty/coating の `None` 返却を解消）。

## 変更内容

### 1. ConstantPenalty + _create_penalty_strategy（penalty）

`xkep_cae/contact/penalty/strategy.py`:
- `ConstantPenalty` クラスを新設: 定数 k_pen を全ステップで返すシンプルな Process
- `_create_penalty_strategy()` ファクトリ関数を追加:
  - beam_E / beam_I / beam_L が全て正 → `AutoBeamEIPenalty`
  - それ以外 → `ConstantPenalty(k_pen)`

`xkep_cae/contact/penalty/__init__.py`:
- `ConstantPenalty` を公開エクスポートに追加

`xkep_cae/contact/penalty/tests/test_strategy.py`:
- `TestConstantPenalty`: Protocol 適合 + 定数値テスト（3テスト）
- `TestCreatePenaltyStrategy`: ファクトリ分岐テスト（3テスト）

### 2. Coating パッケージ移行

`xkep_cae/contact/coating/` を新規作成:
- `strategy.py`: `NoCoatingProcess` + `KelvinVoigtCoatingProcess` + `_create_coating_strategy()`
  - 旧 `xkep_cae_deprecated/process/strategies/coating.py` から移行
  - `return_mapping_core` / `tangent_2x2_core` のインポートを `xkep_cae.contact.friction.law_friction` に変更
- `__init__.py`: 公開エクスポート
- `docs/coating.md`: ドキュメント
- `tests/test_strategy.py`: Protocol 適合 + ファクトリテスト（12テスト）

### 3. default_strategies() の完備

`xkep_cae/core/data.py`:
- `_create_penalty_strategy` と `_create_coating_strategy` をインポート
- `penalty=None` → `penalty=_create_penalty_strategy(...)` に変更
- `coating=None` → `coating=_create_coating_strategy(...)` に変更
- 7軸全 Strategy が `default_strategies()` で生成されるようになった

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 全テスト: 204 passed（+18テスト増）
- 契約違反: 0件（C16 新パッケージ滅菌チェック含む）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `xkep_cae_deprecated/process/strategies/coating.py` | `xkep_cae/contact/coating/strategy.py` | status-181 |
| `ManualPenaltyProcess`（deprecated） | `ConstantPenalty` | status-181 |
| `default_strategies()` penalty=None | `_create_penalty_strategy()` | status-181 |
| `default_strategies()` coating=None | `_create_coating_strategy()` | status-181 |

## TODO

- [ ] Phase 3: concrete プロセス移行（Mesh/Setup/Export/Verify）
- [ ] StrandBendingBatchProcess フル実装

---
