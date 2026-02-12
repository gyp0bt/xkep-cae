# status-003: pycae → xkep-cae リネーム & Phase 1 残作業完了

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-002](./status-002.md)

**日付**: 2026-02-12
**作業者**: Claude Code
**ブランチ**: `claude/rename-to-xkep-cae-O6qth`

---

## 実施内容

### プロジェクト名変更: pycae → xkep-cae

プロジェクト名をxkep-caeに変更。名前の由来は、kepler（物理モジュール向けCAEアプリ）の派生系。

#### ディレクトリ・パッケージリネーム
- `pycae/` → `xkep_cae/` にディレクトリをリネーム（git mv）
- Pythonパッケージ名: `xkep_cae`
- PyPI配布名: `xkep-cae`

#### import/参照の一括更新
全57箇所の `pycae` 参照を `xkep_cae` に更新:
- ソースコード: 7ファイル（`__init__.py`, `core/`, `assembly.py`, `elements/`, `materials/`）
- テストコード: 6ファイル
- `pyproject.toml`: name, packages
- ドキュメント: README.md, roadmap.md

### Phase 1 残作業の完了

#### テスト命名規約統一

| 旧ファイル名 | 新ファイル名 |
|-------------|-------------|
| `test_linear_elastic_mixed.py` | `test_elements_manufactured.py` |
| `test_with_abaqus_data_manually_tensile.py` | `test_benchmark_tensile.py` |
| `test_with_abaqus_data_manually_shear.py` | `test_benchmark_shear.py` |
| `test_with_abaqus_data_manually_cutter_sample1.py` | `test_benchmark_cutter_q4tri3.py` |
| `test_with_abaqus_data_manually_cutter_sample5.py` | `test_benchmark_cutter_tri6.py` |
| `test_protocol_assembly.py` | （変更なし、そのまま） |

#### ruff lint/format 設定追加

`pyproject.toml` に以下を追加:
- `[tool.ruff]`: target-version = "py310", line-length = 100
- `[tool.ruff.lint]`: E, W, F, I, UP, B ルールを有効化
- `[tool.ruff.format]`: double-quote スタイル
- `ruff>=0.4` を dev 依存に追加

ruff check/format を全ソースに適用:
- import順序の統一（isort）
- 未使用import/変数の除去
- 非推奨な型アノテーション（`typing.Dict` → `dict` 等）の更新
- 曖昧な変数名（`I`, `l`）の修正
- `zip()` に `strict=True` を追加

#### CI設定更新
`.github/workflows/ci.yml` に ruff check/format ステップを追加。

---

## テスト結果

```
16 passed, 2 skipped (pymesh依存)
ruff check: All checks passed!
ruff format: All files formatted
```

| テストファイル | テスト数 | 結果 |
|---------------|---------|------|
| `test_elements_manufactured.py` | 3 | PASSED |
| `test_protocol_assembly.py` | 5 | PASSED |
| `test_benchmark_tensile.py` | 4 | PASSED |
| `test_benchmark_shear.py` | 4 | PASSED |
| `test_benchmark_cutter_q4tri3.py` | 1 | SKIPPED (pymesh) |
| `test_benchmark_cutter_tri6.py` | 1 | SKIPPED (pymesh) |

---

## 現在のプロジェクト構成

```
xkep-cae/
├── pyproject.toml              ★ name → xkep-cae, ruff設定追加
├── .github/workflows/ci.yml    ★ ruffステップ追加
├── README.md                   ★ xkep-cae に更新
├── docs/
│   ├── roadmap.md              ★ xkep-cae に更新
│   └── status/
│       ├── status-001.md
│       ├── status-002.md
│       └── status-003.md       ★ 新規
├── xkep_cae/                   ★ pycae/ からリネーム
│   ├── __init__.py
│   ├── api.py
│   ├── assembly.py
│   ├── bc.py
│   ├── solver.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── element.py          ElementProtocol
│   │   └── constitutive.py     ConstitutiveProtocol
│   ├── elements/
│   │   ├── __init__.py
│   │   ├── quad4.py            Quad4PlaneStrain
│   │   ├── quad4_bbar.py       Quad4BBarPlaneStrain
│   │   ├── tri3.py             Tri3PlaneStrain
│   │   └── tri6.py             Tri6PlaneStrain
│   └── materials/
│       ├── __init__.py
│       └── elastic.py          PlaneStrainElastic
├── tests/
│   ├── __init__.py
│   ├── test_elements_manufactured.py          ★ リネーム
│   ├── test_protocol_assembly.py
│   ├── test_benchmark_tensile.py              ★ リネーム
│   ├── test_benchmark_shear.py                ★ リネーム
│   ├── test_benchmark_cutter_q4tri3.py        ★ リネーム (external)
│   └── test_benchmark_cutter_tri6.py          ★ リネーム (external)
└── results/
    └── test_results.xlsx
```

---

## TODO（次回以降の作業）

- [ ] Phase 2.1: Euler-Bernoulli梁（2D）の実装
- [ ] Phase 2.2: Timoshenko梁（2D）の実装
- [ ] メッシュI/O: Abaqus `.inp` フォーマットのパーサー自前実装（pymesh代替）

---

## 設計上の懸念

1. **Q4のD行列修正**: `quad4_ke_plane_strain()` 内で `D_tmp[0,1] *= 2; D_tmp[1,0] *= 2` としている箇所がある。これはengineering shearの取り扱いに関連するが、他の要素（TRI3, TRI6）では行っていない。要素間の記法の統一が望ましい。現時点ではAbaqusベンチマークで検証済みなので変更せず維持。
2. **`assembly.py` の二重実装**: レガシー関数（`assemble_global_stiffness_mixed`）と新関数（`assemble_global_stiffness`）が並存。Phase 2以降で新関数に一本化する際にレガシーを廃止予定。
3. **言語選択**: ユーザーからPython以外の言語も検討可との言及あり。開発フェーズではPythonで進行し、パフォーマンスがボトルネックになった時点で検討する。
