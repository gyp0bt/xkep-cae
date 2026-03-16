# Status 185 — 脱出ポット計画 Phase 5: deprecated 完全除去準備

[← status-index](status-index.md) | [← README](../../README.md)

## 概要

`xkep_cae_deprecated` を完全に削除しても 7本撚線 45° 曲げ解析が収束することを確認。
新 `xkep_cae/` パッケージ内の全モジュールから deprecated 参照を除去し、自己完結構造を達成。

## 実施内容

### 1. 新規作成モジュール（手書き・契約準拠）

| ファイル | 内容 |
|----------|------|
| `xkep_cae/math/quaternion.py` | 四元数演算・SO(3)回転（純粋数学、依存なし） |
| `xkep_cae/core/constitutive.py` | ConstitutiveProtocol / PlasticConstitutiveProtocol |
| `xkep_cae/core/element.py` | ElementProtocol / NonlinearElementProtocol / DynamicElementProtocol |
| `xkep_cae/core/results.py` | LinearSolveResult 他 NamedTuple 群 |
| `xkep_cae/core/state.py` | PlasticState1D / CosseratPlasticState 他 dataclass 群 |
| `xkep_cae/sections/beam.py` | BeamSection2D / BeamSection（Cowper補正付き） |

### 2. deprecated → xkep_cae 移植（sed 置換 + 手動修正）

全モジュールの `xkep_cae_deprecated` インポートを `xkep_cae` に変換。

- **elements/**: beam_timo3d, beam_cosserat, beam_eb2d, beam_timo2d, continuum_nl, hex8, quad4系, tri3, tri6（計11ファイル）
- **materials/**: elastic, beam_elastic, plasticity_1d, plasticity_3d
- **mesh/**: twisted_wire, ring_compliance
- **contact/**: broadphase, geometry/_legacy, ncp, utils, bc_utils, pair, law_normal, law_friction, line_contact, prescreening_data, kpen_features, staged_activation, initial_penetration, graph, assembly, mortar, diagnostics, sheath_contact, solver_ncp（計19ファイル）
- **process/**: data, base, categories, registry, runner, slots, tree, presets, testing + strategies/（計22ファイル）
- **numerical_tests/**: wire_bending_benchmark
- **output/**: database, export_csv/json/vtk/animation, render_beam_3d, request, step, initial_conditions
- **io/**: abaqus_inp, inp_parser, inp_runner, material_converter
- **thermal/**: dataset, fem, gnn, gnn_fc, pinn, train_surrogate
- **tuning/**: executor, optuna_tuner, presets, schema
- **ルート**: api, assembly, assembly_plasticity, bc, dynamics, solver

### 3. __init__.py 更新

全9パッケージの `__init__.py` を re-export shim（`importlib.import_module("xkep_cae_deprecated.*")`）から
直接インポートまたは簡潔な docstring に変更。

### 4. 既存ファイル修正

- `contact/solver/process.py`: `_import_deprecated()` → `_import_module()` + パス変更
- `contact/setup/process.py`: deprecated 参照除去
- `mesh/process.py`: deprecated 参照除去
- `tests/contact/test_linear_solver_strategy.py`: deprecated インポート除去
- `contact/geometry/__init__.py`: `_legacy.py` re-export 追加

### 5. contact/geometry 衝突解決

`contact/geometry.py`（deprecated ファイル）と `contact/geometry/`（新 Strategy ディレクトリ）の
名前衝突を `_legacy.py` 移動 + `__init__.py` re-export で解決。

## テスト結果（xkep_cae_deprecated 除去状態）

| テスト | 結果 | 備考 |
|--------|------|------|
| 7本撚線 45° 曲げ | **PASS**（2.05s） | S3 凍結解除条件クリア |
| 7本撚線 90° 曲げ | FAIL（不収束） | 既知の環境依存不安定（roadmap記載、status-143） |
| 7本撚線 揺動 Phase2 | XFAIL | 活性セット変動（既知） |

## 契約検証

- **ruff check**: PASS（新規手書きファイル全件）
- **契約違反**: 419→577件（+158件は sed 移植ファイル由来の既存 C16 違反。新規作成ファイルは 0件）

## 互換ヒストリー追加

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `xkep_cae_deprecated` 全モジュール | `xkep_cae` 自己完結パッケージ | status-185 |
| re-export shim `__init__.py`（9パッケージ） | 直接インポート / docstring のみ | status-185 |
| `contact/geometry.py` | `contact/geometry/_legacy.py` + `__init__.py` re-export | status-185 |

## 残課題

- [ ] `xkep_cae_deprecated/` ディレクトリ完全削除
- [ ] 90° 曲げテストの安定化（環境依存 — 優先度低）
- [ ] sed 移植ファイルの C16 契約違反対応（Phase 6〜8 で段階的に対処）
- [ ] process/ ディレクトリの core/ 統合検討

## テスト数

~2260 + 279 新パッケージテスト（変更なし）
