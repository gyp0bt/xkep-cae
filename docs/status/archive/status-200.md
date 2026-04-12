# status-200: Process Architecture 監査 — elements/numerical_tests 不遵守検出強化

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-17

## 作業者

Claude Code

## ブランチ

claude/process-architecture-migration-hSYVv

## 概要

ユーザー監査により `elements` と `numerical_tests` の Process Architecture 不遵守が発覚。
検出スクリプトを改良し、段階的移行の初手として `__init__.py` re-export を整理。

## 監査結果サマリ

### 改良前（status-199時点）

| 区分 | 件数 | 備考 |
|------|------|------|
| C16 法律違反 | 47件 | numerical_tests のみ検出 |
| O1 条例違反 | 0件 | |

### 改良後（本status）

| 区分 | 件数 | 備考 |
|------|------|------|
| C16 法律違反 | 40件 | elements 4件新検出、__init__.py re-export 17件削減 |
| O2 条例違反 | 2件 | BackendRegistry パターン検出（新設） |
| O3 条例違反 | 3件 | テスト backend.configure() 注入検出（新設） |

## 変更内容

### 1. validate_process_contracts.py 強化

#### C16 __init__.py クラス再エクスポート検査（新規）

従来 `__init__.py` は小文字関数の re-export のみ検査していた。
大文字クラスの re-export も C16 ルール（Process/frozen-dataclass/Enum）で検査するよう改良。

**新検出 elements 違反 4件:**

| クラス | 違反内容 |
|--------|---------|
| `ULCRBeamAssembler` | Process でも frozen dataclass でもない |
| `BeamForces3D` | non-frozen dataclass |
| `BeamSection` | frozen dataclass だがメソッド6つ保持 |
| `BeamSection2D` | frozen dataclass だがメソッド3つ保持 |

新規ヘルパー `_check_reexported_class()` を追加。

#### O2: BackendRegistry パターン検出（新設）

`xkep_cae/` 内（core/ 除く）で:
- `configure()/reset()` を持つ Registry クラスを検出
- モジュールレベルシングルトン（`xxx = Xxx()` パターン）を検出

TypeVar/ALL_CAPS 定数は除外。

#### O3: テスト backend.configure() 注入検出（新設）

`tests/conftest.py` 等での `backend.configure*()` 呼び出しを検出。
BackendRegistry 注入パターンは Process Architecture 迂回であり廃止対象。

### 2. numerical_tests/__init__.py 整理

純粋関数の re-export を全て除去。型（dataclass/TypeAlias/定数）のみ残す。

**除去したre-export（15件）:**
- `analytical_bend3p/4p`, `analytical_tensile/torsion`
- `generate_beam_mesh_2d/3d`, `generate_beam_mesh_2d/3d_nonuniform`
- `assess_friction_effect`, `_build_section_props`
- `export_static_csv`, `export_frequency_response_csv`
- `run_test`, `run_all_tests`, `run_tests`
- `run_dynamic_test`, `run_frequency_response`, `parse_test_input`

呼び出し元はサブモジュール直接import に移行:
```python
# Before
from xkep_cae.numerical_tests import run_test
# After
from xkep_cae.numerical_tests.runner import run_test
```

### 3. テスト修正

`tests/test_numerical_tests.py` の `run_dynamic_test` import をサブモジュール直接importに変更。

## 残存 C16 違反一覧（40件）

### elements（4件）— 移行計画が必要

| # | 対象 | 種別 | 移行方針 |
|---|------|------|---------|
| 1 | `ULCRBeamAssembler` | 通常クラス | → `AssemblerProcess` に Process 化 |
| 2 | `BeamForces3D` | non-frozen DC | → `frozen=True` に変更 |
| 3 | `BeamSection` | frozen DC + メソッド | → メソッドを分離（`_beam_section_utils.py`） |
| 4 | `BeamSection2D` | frozen DC + メソッド | → メソッドを分離 |

### numerical_tests（36件）

#### 型定義（6件）— non-frozen dataclass

| 対象 | 移行方針 |
|------|---------|
| `NumericalTestConfig` | `frozen=True` + ビルダーパターン |
| `FrequencyResponseConfig` | `frozen=True` + ビルダーパターン |
| `StaticTestResult` | `frozen=True` |
| `FrequencyResponseResult` | `frozen=True` |
| `DynamicTestConfig` | `frozen=True` + ビルダーパターン |
| `DynamicTestResult` | `frozen=True` |

#### 通常クラス（2件）

| 対象 | 移行方針 |
|------|---------|
| `BenchmarkTimingCollector` | → frozen dataclass または Process |
| `ContactSolveResult` / `BendingOscillationResult` | → `frozen=True` |

#### 純粋関数（28件）

**core.py（9件）**: `generate_beam_mesh_*` 4件、`analytical_*` 4件、`assess_friction_effect` 1件
→ `_` prefix で private 化（内部ユーティリティ）

**runner.py（3件）**: `run_test`, `run_all_tests`, `run_tests`
→ `NumericalTestProcess` に Process 化

**dynamic_runner.py（2件）**: `run_dynamic_test`, `run_dynamic_tests`
→ `DynamicTestProcess` に Process 化

**frequency.py（1件）**: `run_frequency_response`
→ `FrequencyResponseProcess` に Process 化

**csv_export.py（2件）**: `export_static_csv`, `export_frequency_response_csv`
→ `ExportProcess` の拡張

**inp_input.py（1件）**: `parse_test_input`
→ `_` prefix で private 化

**wire_bending_benchmark.py（3件）**: `run_bending_oscillation`, `run_scaling_benchmark`, `print_benchmark_report`
→ `BenchingOscillationProcess` に Process 化

## 残存条例違反（5件）

| 条例 | 対象 | 移行方針 |
|------|------|---------|
| O2 | `BackendRegistry` クラス | 完全廃止 → Process.uses に移行 |
| O2 | `backend = BackendRegistry()` | 完全廃止 |
| O3 | `backend.configure()` | 完全廃止 |
| O3 | `backend.configure_frequency()` | 完全廃止 |
| O3 | `backend.configure_dynamic()` | 完全廃止 |

## 段階的移行計画

### Phase 15-A（次回）: core.py 関数 private 化
- `analytical_*`, `generate_beam_mesh_*` に `_` prefix 付与
- 全呼び出し元更新（runner.py, frequency.py, dynamic_runner.py, tests）

### Phase 15-B: dataclass frozen 化
- `NumericalTestConfig` 等を `frozen=True` に変更
- 可変フィールドの代替パターン策定

### Phase 15-C: runner Process 化
- `run_test()` → `NumericalTestProcess`
- BackendRegistry 依存を Process.uses で置換

### Phase 15-D: BackendRegistry 完全廃止
- `_backend.py` 削除
- `tests/conftest.py` の `_configure_numerical_tests_backend()` 削除
- Process API 直接使用に移行

### Phase 15-E: elements 移行
- `ULCRBeamAssembler` → `AssemblerProcess`
- `BeamSection` メソッド分離
- `BeamForces3D` frozen 化

## テスト結果

| テスト | 結果 | 備考 |
|--------|------|------|
| validate_process_contracts.py | C16: 40件, O2: 2件, O3: 3件 | 新検出含む |
| ruff check | エラー 0 件 | |
| ruff format | 全ファイルフォーマット済み | |

## TODO

- [ ] Phase 15-A: core.py 純粋関数の `_` private 化（9件削減見込み）
- [ ] Phase 15-B: dataclass `frozen=True` 化（8件削減見込み）
- [ ] Phase 15-C: runner/frequency/dynamic_runner の Process 化（6件削減）
- [ ] Phase 15-D: BackendRegistry 完全廃止（O2: 2件 + O3: 3件解消）
- [ ] Phase 15-E: elements 移行（4件削減）
- [ ] status-199 引継ぎ: deprecated プロセス統合テスト
- [ ] status-199 引継ぎ: pytest conftest ProcessExecutionLog リセットフック
- [ ] status-199 引継ぎ: 動的ソルバーテスト
- [ ] status-199 引継ぎ: 被膜モデル物理検証テスト

## 開発運用メモ

- C16 `__init__.py` 検査強化は他のモジュールにも波及する可能性がある。新パッケージ追加時は `__init__.py` の re-export 内容に注意。
- O2/O3 は条例（警告レベル）。法律（C系）への昇格は BackendRegistry 廃止完了後に検討。
- numerical_tests の Process 化は BackendRegistry 廃止と同時に進める必要がある（依存関係）。

---
