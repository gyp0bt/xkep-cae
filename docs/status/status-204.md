# status-204: C17 違反ゼロ達成 — frozen dataclass 完全移行

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**テスト数**: ~2260 + 374 新パッケージテスト（315テスト全PASS）
**契約違反**: C17: 0件（3件→0件） | O2: 2件（警告）

---

## 概要

C17 違反3件（non-frozen dataclass）を解消。
`_ContactStateOutput`/`_ContactPairOutput`/`_ContactManagerInput` の全3クラスを `frozen=True` に移行。

**設計方針**: `dataclasses.replace()` は使わない。各クラスに `_evolve(**kwargs)` メソッドを追加し、
指定フィールドを更新した新インスタンスを返すパターンで不変性を実現。

## 変更内容

### 1. `_ContactStateOutput` frozen 化（41変異箇所）

`@dataclass` → `@dataclass(frozen=True)` に変更。
`_evolve(**kwargs)` メソッドを追加（`dataclasses.fields()` ベースの汎用インスタンス更新）。

変異パターン変換:
```python
# 旧: フィールド直接代入
pair.state.gap = float(gap_all[i])

# 新: _evolve で新インスタンス生成
pair.state = pair.state._evolve(gap=float(gap_all[i]))
```

影響ファイル:
| ファイル | 変異箇所数 | 変異内容 |
|---------|-----------|---------|
| `contact/_contact_pair.py` | 11 | geometry, active_set, penalty |
| `contact/geometry/strategy.py` | 10 | geometry batch, active_set |
| `contact/coating/strategy.py` | 8 | coating friction state |
| `contact/friction/_assembly.py` | 8 | return mapping, penalty init |
| `contact/solver/process.py` | 3 | lambda_n, p_n, coating_prev |
| `contact/contact_force/strategy.py` | 1 | p_n (smooth penalty) |
| `contact/friction/strategy.py` | 1 | p_n (NCP) |

### 2. `_ContactPairOutput` frozen 化

`@dataclass` → `@dataclass(frozen=True)` に変更。
`_evolve(**kwargs)` メソッドを追加。

`pair.state = new_state` パターンを `pairs[i] = pair._evolve(state=new_state)` に変換。
これに伴い、ヘルパー関数の signature を変更:
- `_update_active_set(pair)` → `_update_active_set_state(state)` （state を受け取り state を返す）
- `_update_active_set_hysteresis(pair)` → `_update_active_set_hysteresis(state)` （同上）

### 3. `_ContactManagerInput` frozen 化

`@dataclass` → `@dataclass(frozen=True)` に変更。
`pairs` フィールドはリストオブジェクトへの参照であり、frozen はフィールドの再代入のみ防止。
リスト自体の変異（append, index代入）は frozen でも可能なため追加修正不要。

### 4. テストモック更新

`contact_force/tests/test_strategy.py` の `_MockState`/`_MockPair` に `_evolve()` メソッドを追加。
frozen 化された本体クラスと同じ API をモックが提供するよう修正。

## 検証結果

```
$ python scripts/validate_process_contracts.py
契約違反なし（条例違反 2 件は警告のみ）
  O2: numerical_tests/_backend.py BackendRegistry（2件）
```

```
$ python -m pytest xkep_cae/ -x -q --timeout=120
315 passed, 4284 warnings in 18.85s
```

## 互換ヒストリー

| 旧 | 新 | 備考 |
|---|---|------|
| `_ContactStateOutput` non-frozen | `frozen=True` + `_evolve()` | status-204 |
| `_ContactPairOutput` non-frozen | `frozen=True` + `_evolve()` | status-204 |
| `_ContactManagerInput` non-frozen | `frozen=True` | status-204 |
| `_update_active_set(pair)` | `_update_active_set_state(state)` | status-204 |
| `_update_active_set_hysteresis(pair)` | `_update_active_set_hysteresis(state)` | status-204 |

## 次のタスク

- [ ] O2 条例違反2件解消: BackendRegistry 完全廃止（Phase 17）
- [ ] 被膜モデル物理検証テスト
