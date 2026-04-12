# status-176: C16 純粋関数違反の追加

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-15

## 概要

`scripts/validate_process_contracts.py` の C16（新パッケージ滅菌チェック）で、
純粋関数の `__all__` エクスポートを許可していたロジックを違反に変更。

## 変更内容

### C16 関数検査ロジック変更

**変更前**: `__all__` にエクスポートされた public 純粋関数は許可
**変更後**: public 純粋関数は全て違反（Protocol / Strategy / Process に変換が必要）

private 関数（`_` prefix）は引き続き許可。

### 検出される違反（5件）

`xkep_cae/process/strategies/penalty/law_normal.py`:

| 関数 | 対応方針 |
|------|---------|
| `softplus()` | `_` prefix で private 化 or Process 化 |
| `evaluate_al_normal_force()` | 既に `ALNormalForceProcess` が存在 → `_` prefix 化 |
| `evaluate_smooth_normal_force()` | 既に `SmoothNormalForceProcess` が存在 → `_` prefix 化 |
| `evaluate_smooth_normal_force_vectorized()` | `_` prefix 化 |
| `auto_beam_penalty_stiffness()` | Strategy (`AutoBeamEIPenalty` 等) の内部ヘルパーへ移行 |

## テスト数

~2260+34p(新)（変更なし）

## TODO

- [ ] 上記5関数の `_` prefix 化 or Process ラップ対応
- [ ] `penalty/__init__.py` の `__all__` から純粋関数を除去

---
