# status-206: 旧API呼び出し元整合 + 旧テスト一掃

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-18
**ブランチ**: `claude/check-status-todos-NgiWB`
**テスト数**: ~248（新パッケージテスト）

---

## 概要

status-205 で `_ContactManagerInput` / `_ContactPairOutput` / `_ContactStateOutput` のメソッドを完全除去し、Process 直接実装に移行した。
本 PR はその **呼び出し元整合**（status-205 の TODO 1番目）を完了させるもの。

## 変更内容

### 1. xkep_cae/ ソースコード — `_evolve` 呼び出し移行（6ファイル）

| ファイル | 変更内容 |
|---------|---------|
| `contact/geometry/strategy.py` | `state._evolve()` → `_evolve_state(state, ...)`, `pair._evolve()` → `_evolve_pair(pair, ...)` |
| `contact/contact_force/strategy.py` | 同上 |
| `contact/friction/strategy.py` | 同上 |
| `contact/friction/_assembly.py` | 同上（return mapping ループ内） |
| `contact/coating/strategy.py` | 同上（被膜摩擦 + 非接触リセット） |
| `contact/solver/process.py` | 同上 + `manager.n_pairs` → `_n_pairs(manager)` |

### 2. xkep_cae/ ソースコード — Process API 移行

| ファイル | 変更内容 |
|---------|---------|
| `contact/solver/process.py` | `DetectCandidatesProcess` / `UpdateGeometryProcess` の結果から `manager` を取得（不変パターン） |
| `contact/solver/_nuzawa_steps.py` | `inp.manager.update_geometry()` → `UpdateGeometryProcess` 経由 |
| `contact/solver/_contact_graph.py` | `manager.n_pairs` → `_n_pairs(manager)` |
| `contact/setup/process.py` | `manager.detect_candidates()` → `DetectCandidatesProcess` 経由 |

### 3. C5 uses 宣言修正

| Process | 追加した uses |
|---------|-------------|
| `ContactForceAssemblyProcess` | `UpdateGeometryProcess` |
| `UzawaUpdateProcess` | `UpdateGeometryProcess` |
| `ContactSetupProcess` | `DetectCandidatesProcess` |

### 4. 旧テスト一掃（89ファイル削除）

旧モジュールパス（`xkep_cae.contact.pair`, `xkep_cae.assembly` 等）をインポートしていたテストファイル 89 件を全削除。
これらは deprecated パッケージ依存でコレクションエラーとなっており、実行不能だった。

残存テスト: 13モジュール（248テスト）— 全て新パッケージ API を使用。

### 5. C3 紐付けテスト追加

`xkep_cae/contact/tests/test_manager_process.py` を新規作成:
- `TestAddPairProcessAPI` (`@binds_to(AddPairProcess)`)
- `TestResetAllPairsProcessAPI` (`@binds_to(ResetAllPairsProcess)`)

### 6. 新テストの旧API移行

| ファイル | 変更内容 |
|---------|---------|
| `tests/contact/test_pair.py` | `mgr.add_pair()` → `AddPairProcess`, `mgr.detect_candidates()` → `DetectCandidatesProcess` 等 |
| `tests/contact/test_strand_contact_process.py` | `manager.detect_candidates()` → `DetectCandidatesProcess`, `pair.is_active()` → `_is_active_pair(pair)` |
| `xkep_cae/contact/solver/tests/test_process.py` | `manager.detect_candidates()` → `DetectCandidatesProcess` |

## 契約状況

| 項目 | 状態 |
|------|------|
| C14（deprecated import） | 0件 |
| C16（新パッケージ滅菌） | 0件 |
| C17（frozen dataclass） | 0件 |
| C3（テスト紐付け） | 0件 |
| C5（uses 宣言） | 0件 |
| O2（BackendRegistry） | 2件（警告: 既知問題） |

## 既知の問題

- `test_numerical_tests.py` / `test_cosserat_vs_cr_bend3p.py` : Backend 未設定エラー（O2 条例関連）— status-205 以前から存在
- `test_inp_metadata_validation.py` 一部: `scripts._run_bending_oscillation` モジュール未存在

## 次のタスク

- [ ] O2 条例違反2件解消: BackendRegistry 完全廃止（Phase 17）
- [ ] 被膜モデル物理検証テスト
- [ ] Backend 未設定による test_numerical_tests 失敗の解消
