# status-205: ContactManager Process 分割 — dataclass メソッド完全除去

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**ブランチ**: `claude/refactor-contact-manager-cQMKn`
**テスト数**: 呼び出し元未移行のため一部破損（次PR対応）

---

## 概要

`_ContactManagerInput` は frozen dataclass でありながらメソッドを持っていた（C17 精神違反）。
また `_evolve()` メソッドは `dataclasses.replace()` の名前を変えただけの実質的な違反だった。

本 PR で以下を実施:
1. **dataclass からメソッドを完全除去**（純データ化）
2. **ユーティリティ関数をモジュールレベルに移動**（`_contact_pair.py` 内）
3. **ビジネスロジックを Process クラスに直接実装**（`_manager_process.py`）

## 設計判断

### なぜ純粋関数ではなく Process か

- Process Architecture の思想: ロジックは Process に集約
- 純粋関数中間層（`_contact_ops.py`）は不要な間接層
- Process が唯一のビジネスロジック実行主体

### ユーティリティの配置

`_evolve_state()` / `_evolve_pair()` 等のデータ操作ユーティリティは `_contact_pair.py` のモジュールレベル関数として配置。
理由: 複数の Process / Strategy から横断的に使われるため、特定 Process に属さない。

## 変更内容

### 1. `_contact_pair.py` — 純データ化 + ユーティリティ関数

**削除したメソッド:**

| クラス | 削除メソッド | 移行先 |
|--------|-------------|--------|
| `_ContactStateOutput` | `copy()` | `_copy_state()` |
| `_ContactStateOutput` | `_evolve()` | `_evolve_state()` |
| `_ContactPairOutput` | `search_radius` property | `_pair_search_radius()` |
| `_ContactPairOutput` | `is_active()` | `_is_active_pair()` |
| `_ContactPairOutput` | `_evolve()` | `_evolve_pair()` |
| `_ContactManagerInput` | `n_pairs` property | `_n_pairs()` |
| `_ContactManagerInput` | `n_active` property | `_n_active()` |
| `_ContactManagerInput` | `add_pair()` | `AddPairProcess` |
| `_ContactManagerInput` | `reset_all()` | `ResetAllPairsProcess` |
| `_ContactManagerInput` | `get_active_pairs()` | `_get_active_pairs()` |
| `_ContactManagerInput` | `detect_candidates()` | `DetectCandidatesProcess` |
| `_ContactManagerInput` | `update_geometry()` | `UpdateGeometryProcess` |
| `_ContactManagerInput` | `_update_active_set_state()` | `_manager_process._update_active_set_state()` |
| `_ContactManagerInput` | `initialize_penalty()` | `InitializePenaltyProcess` |

**追加したモジュールレベル関数:**
- `_evolve_state()` / `_evolve_pair()` — frozen dataclass のフィールド更新
- `_copy_state()` — 深いコピー
- `_pair_search_radius()` / `_is_active_pair()` — ペアユーティリティ
- `_n_pairs()` / `_n_active()` / `_get_active_pairs()` — manager ユーティリティ
- `_make_pair()` — ペア生成

### 2. `_manager_process.py` — Process 直接実装

旧版は manager メソッドに委譲するだけの薄いラッパーだった。
新版はロジックを Process に直接実装し、不変パターンで新 manager を出力に含める。

| Process | 旧版 | 新版 |
|---------|------|------|
| `DetectCandidatesProcess` | `manager.detect_candidates()` 委譲 | ロジック直接実装、出力に `manager` 追加 |
| `UpdateGeometryProcess` | `manager.update_geometry()` 委譲 | ロジック直接実装、出力に `manager` 追加 |
| `InitializePenaltyProcess` | `manager.initialize_penalty()` 委譲 | ロジック直接実装、出力に `manager` 追加 |
| `AddPairProcess` | *(新規)* | ペア追加、新 manager 返却 |
| `ResetAllPairsProcess` | *(新規)* | 全ペアリセット、新 manager 返却 |

**全 Process の出力に `manager: _ContactManagerInput` フィールドを追加**。
in-place 変異を廃止し、完全に不変な操作パターンに移行。

### 3. `_update_active_set_state()` — Process 内部ヘルパー

`UpdateGeometryProcess` の内部で使う active-set ヒステリシス更新ロジック。
`_manager_process.py` のモジュールレベル関数として配置。

## 互換ヒストリー

| 旧 | 新 | 備考 |
|---|---|------|
| `_ContactStateOutput._evolve()` | `_evolve_state()` | モジュールレベル関数化 |
| `_ContactStateOutput.copy()` | `_copy_state()` | モジュールレベル関数化 |
| `_ContactPairOutput._evolve()` | `_evolve_pair()` | モジュールレベル関数化 |
| `_ContactPairOutput.search_radius` | `_pair_search_radius()` | プロパティ→関数 |
| `_ContactPairOutput.is_active()` | `_is_active_pair()` | メソッド→関数 |
| `_ContactManagerInput.n_pairs` | `_n_pairs()` | プロパティ→関数 |
| `_ContactManagerInput.n_active` | `_n_active()` | プロパティ→関数 |
| `_ContactManagerInput.add_pair()` | `AddPairProcess` | Process 化 |
| `_ContactManagerInput.reset_all()` | `ResetAllPairsProcess` | Process 化 |
| `_ContactManagerInput.detect_candidates()` | `DetectCandidatesProcess` v2.0.0 | ロジック直接実装 |
| `_ContactManagerInput.update_geometry()` | `UpdateGeometryProcess` v2.0.0 | ロジック直接実装 |
| `_ContactManagerInput.initialize_penalty()` | `InitializePenaltyProcess` v2.0.0 | ロジック直接実装 |
| Process 出力に `manager` なし | 全出力に `manager` フィールド追加 | 不変パターン |

## 呼び出し元の影響（次PR対応）

以下のファイルで旧 API（メソッド呼び出し、`_evolve`、`copy`、プロパティ）の移行が必要:

### メソッド → Process 移行
- `tests/contact/test_pair.py` — add_pair, detect_candidates, update_geometry, reset_all, get_active_pairs
- `tests/contact/test_strand_contact_process.py` — detect_candidates, update_geometry
- `tests/test_s3_benchmark.py` — detect_candidates
- `tests/contact/test_exclude_same_layer.py` — detect_candidates
- `tests/contact/test_large_scale_contact.py` — detect_candidates
- `tests/contact/test_line_contact.py` — add_pair
- `tests/contact/test_mortar.py` — add_pair
- `tests/contact/test_consistent_st_tangent.py` — add_pair, update_geometry
- `scripts/verify_coating_*.py` — detect_candidates
- `xkep_cae/contact/solver/_nuzawa_steps.py` — update_geometry
- `xkep_cae/contact/setup/process.py` — detect_candidates (既に Process 化済み、但しI/O型変更)

### `_evolve()` → `_evolve_state()` / `_evolve_pair()` 移行
- `xkep_cae/contact/friction/_assembly.py`
- `xkep_cae/contact/coating/strategy.py`
- `xkep_cae/contact/friction/strategy.py`
- `xkep_cae/contact/geometry/strategy.py`
- `xkep_cae/contact/solver/process.py`
- `xkep_cae/contact/contact_force/strategy.py`

### `.n_pairs` / `.n_active` → `_n_pairs()` / `_n_active()` 移行
- `xkep_cae/contact/solver/process.py`
- `xkep_cae/contact/solver/_solver_state.py`
- `xkep_cae/contact/solver/_newton_uzawa_static.py`
- `xkep_cae/contact/solver/_newton_uzawa_dynamic.py`
- `xkep_cae/contact/solver/_adaptive_stepping.py`
- 多数のテスト・スクリプト

### `.copy()` → `_copy_state()` 移行
- `tests/contact/test_pair.py`
- `tests/contact/test_consistent_tangent.py`
- `tests/contact/test_line_friction.py`

### `.is_active()` → `_is_active_pair()` / `.search_radius` → `_pair_search_radius()` 移行
- 使用箇所は少数（次PR で確認）

## 次のタスク

- [ ] 呼び出し元の整合（上記ファイル群の旧API→新API移行）
- [ ] テスト全PASS確認
- [ ] O2 条例違反2件解消: BackendRegistry 完全廃止（Phase 17）
- [ ] 被膜モデル物理検証テスト
