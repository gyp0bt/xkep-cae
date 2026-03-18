# status-206: 呼び出し元 API 整合 — 旧 dataclass メソッド → 新 API 完全移行

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**ブランチ**: `claude/execute-status-todos-OLdxc`
**テスト数**: ~2260 + 315 新パッケージテスト（+29 test_pair.py）

---

## 概要

status-205 で除去した `_ContactManagerInput` / `_ContactPairOutput` / `_ContactStateOutput` のメソッドの呼び出し元を、新しいモジュールレベル関数と Process API に完全移行。

## 変更内容

### 1. `_evolve()` → `_evolve_state()` / `_evolve_pair()` 移行（xkep_cae/ ソース）

| ファイル | 変更箇所数 |
|----------|-----------|
| `contact/coating/strategy.py` | 2箇所 |
| `contact/friction/_assembly.py` | 2箇所 |
| `contact/friction/strategy.py` | 1箇所 |
| `contact/contact_force/strategy.py` | 1箇所 |
| `contact/geometry/strategy.py` | 3箇所 |
| `contact/solver/process.py` | 2箇所 |

### 2. `manager.n_pairs` → `_n_pairs(manager)` / `len(manager.pairs)` 移行

| ファイル | 変更箇所数 |
|----------|-----------|
| `contact/solver/process.py` | 3箇所 |
| `contact/solver/_contact_graph.py` | 1箇所 |

### 3. `manager.detect_candidates()` → `DetectCandidatesProcess` 移行

| ファイル | 変更箇所数 |
|----------|-----------|
| `contact/solver/process.py` | 3箇所（初期 + 位置調整後 + ステップ内） |
| `contact/setup/process.py` | 1箇所 |
| `contact/solver/tests/test_process.py` | 1箇所 |
| `tests/contact/test_pair.py` | 6箇所 |
| `tests/contact/test_strand_contact_process.py` | 1箇所 |
| `tests/contact/test_exclude_same_layer.py` | 10箇所 |
| `tests/contact/test_large_scale_contact.py` | 1箇所 |
| `tests/test_s3_benchmark.py` | 7箇所 |
| `scripts/verify_coating_gap.py` | 1箇所 |
| `scripts/verify_coating_parametric_study.py` | 2箇所 |
| `scripts/diagnose_7wire_convergence.py` | 2箇所 |

### 4. `manager.update_geometry()` → `UpdateGeometryProcess` 移行

| ファイル | 変更箇所数 |
|----------|-----------|
| `contact/solver/process.py` | 1箇所（出力キャプチャ + manager 更新） |
| `contact/solver/_nuzawa_steps.py` | 2箇所（リスト in-place 更新パターン） |
| `tests/contact/test_pair.py` | 11箇所 |

### 5. `manager.add_pair()` → `AddPairProcess` 移行

| ファイル | 変更箇所数 |
|----------|-----------|
| `tests/contact/test_pair.py` | 15箇所 |

### 6. その他の移行

| 旧 API | 新 API | ファイル数 |
|---------|--------|-----------|
| `state.copy()` | `_copy_state(state)` | 1 |
| `pair.is_active()` | `_is_active_pair(pair)` | 3 |
| `mgr.n_active` | `_n_active(mgr)` | 1 |
| `mgr.reset_all()` | `ResetAllPairsProcess` | 1 |
| `mgr.get_active_pairs()` | `_get_active_pairs(mgr)` | 1 |

### 7. テスト mock 修正

`contact/contact_force/tests/test_strategy.py` の `_MockState` / `_MockPair` / `_MockManager` を
実際の frozen dataclass (`_ContactStateOutput` / `_ContactPairOutput` / `_ContactManagerInput`) に置換。
`_make_test_pair()` ヘルパー関数新設。

## 設計判断

### `_nuzawa_steps.py` の in-place リスト更新パターン

`UpdateGeometryProcess` は新 manager を返すが、`inp.manager` は frozen input の属性で再代入不可。
list の in-place 更新 (`inp.manager.pairs[:] = new_manager.pairs`) で対応。
frozen dataclass の list 属性は内容変更可能（Python の仕様通り）。

### `process.py` の manager 変数再代入パターン

`process()` メソッド内のローカル変数 `manager` は、`DetectCandidatesProcess` / `UpdateGeometryProcess` の
出力から新 manager を取得して再代入。不変パターンとローカル変数再代入の組み合わせ。

### deprecated モジュール（`xkep_cae.contact.pair` 等）を使用するテスト

deprecated `ContactManager` / `ContactPair` のメソッドはそのまま残存（移行不要）。
対象: `test_line_contact.py`, `test_mortar.py`, `test_consistent_st_tangent.py`, `test_optimization_benchmark.py`,
`test_beam_contact_penetration.py`, `test_sheath_contact.py`, `generate_verification_plots.py` 等。

## テスト結果

- 新パッケージテスト (`xkep_cae/`): **315 passed**
- `tests/contact/test_pair.py`: **29 passed**
- ruff check: **All checks passed**
- ruff format: **217 files already formatted**

## 次のタスク

- [ ] O2 条例違反2件解消: BackendRegistry 完全廃止（Phase 17）
- [ ] 被膜モデル物理検証テスト
