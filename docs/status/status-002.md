# status-002: Phase 1 アーキテクチャ再構成

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-001](./status-001.md)

**日付**: 2026-02-12
**作業者**: Claude Code
**ブランチ**: `claude/confirm-and-execute-todos-aH0VX`

---

## 実施内容

### Phase 1.1: プロジェクト基盤整備

#### pyproject.toml 作成
- `hatchling` をビルドバックエンドとして採用
- 依存関係: `numpy>=1.24`, `scipy>=1.10`（必須）
- オプション依存: `pyamg`（AMGソルバー）、`numba`（TRI6高速化）、`pytest`（開発）
- Python 3.10+ をサポート

#### テストフレームワーク統一（pytest）
- 全テストファイルから `sys.path.append(os.getcwd())` ハックを除去
- `_test_*` 関数を `test_*` に改名（pytest自動検出対応）
- pymesh依存テスト（cutter_sample1, cutter_sample5）に `pytest.importorskip` + `@pytest.mark.external` を適用
- `__init__.py` を `pycae/`, `pycae/elements/`, `pycae/materials/`, `tests/` に追加
- `pip install -e ".[dev]"` による正規パッケージインストールに移行

#### CI設定（GitHub Actions）
- `.github/workflows/ci.yml` を作成
- Python 3.10 / 3.11 / 3.12 のマトリクスビルド
- `pytest -m "not external"` で外部依存テストを除外

### Phase 1.2: コア抽象レイヤー設計

`pycae/core/` ディレクトリを新設し、以下のProtocolを定義した。

#### ElementProtocol (`pycae/core/element.py`)
```python
class ElementProtocol(Protocol):
    ndof_per_node: int
    nnodes: int
    ndof: int
    def local_stiffness(self, coords, material, thickness) -> ndarray: ...
    def dof_indices(self, node_indices) -> ndarray: ...
```

#### ConstitutiveProtocol (`pycae/core/constitutive.py`)
```python
class ConstitutiveProtocol(Protocol):
    def tangent(self, strain=None) -> ndarray: ...
```

**設計判断: Protocol vs ABC**
- `Protocol`（構造的部分型）を採用。理由:
  - NNサロゲート等の異質な実装でも継承不要で適合可能
  - `runtime_checkable` デコレータにより `isinstance()` チェックも可能
  - Phase 6のPyTorchモデルラッパーでの柔軟性を確保

### Phase 1.3: 既存コードのProtocol適合

#### 材料クラス
- `PlaneStrainElastic` クラスを `pycae/materials/elastic.py` に追加
- `ConstitutiveProtocol` に適合（`isinstance` チェック済み）
- 既存の `constitutive_plane_strain()` 関数は後方互換のため維持

#### 要素クラス
各要素ファイルにProtocol適合クラスを追加:

| クラス | ファイル | nnodes | ndof |
|--------|---------|--------|------|
| `Quad4PlaneStrain` | `elements/quad4.py` | 4 | 8 |
| `Tri3PlaneStrain` | `elements/tri3.py` | 3 | 6 |
| `Tri6PlaneStrain` | `elements/tri6.py` | 6 | 12 |
| `Quad4BBarPlaneStrain` | `elements/quad4_bbar.py` | 4 | 8 |

- 各クラスは内部で既存の関数を呼び出すラッパー構造
- 既存の関数APIは後方互換のため維持
- 全クラスが `ElementProtocol` の `isinstance` チェックに合格

#### 汎用アセンブリ関数
- `assemble_global_stiffness()` を `assembly.py` に追加
- `element_groups: list[tuple[ElementProtocol, ndarray]]` 形式で任意の要素型を混在アセンブル
- COO→CSR形式、既存の `assemble_global_stiffness_mixed()` と同等の結果
- レガシー関数との一致を検証テストで確認済み

---

## テスト結果

```
16 passed, 2 skipped (pymesh依存)
```

| テストファイル | テスト数 | 結果 |
|---------------|---------|------|
| `test_linear_elastic_mixed.py` | 3 | PASSED |
| `test_protocol_assembly.py` | 5 | PASSED |
| `test_with_abaqus_data_manually_shear.py` | 4 | PASSED |
| `test_with_abaqus_data_manually_tensile.py` | 4 | PASSED |
| `test_with_abaqus_data_manually_cutter_sample1.py` | 1 | SKIPPED (pymesh) |
| `test_with_abaqus_data_manually_cutter_sample5.py` | 1 | SKIPPED (pymesh) |

---

## 現在のプロジェクト構成

```
pycae/
├── pyproject.toml              ★新規
├── .github/workflows/ci.yml    ★新規
├── README.md
├── docs/
│   ├── roadmap.md
│   └── status/
│       ├── status-001.md
│       └── status-002.md       ★新規
├── pycae/
│   ├── __init__.py             ★新規
│   ├── api.py
│   ├── assembly.py             ★ assemble_global_stiffness() 追加
│   ├── bc.py
│   ├── solver.py
│   ├── core/                   ★新規
│   │   ├── __init__.py
│   │   ├── element.py          ElementProtocol
│   │   └── constitutive.py     ConstitutiveProtocol
│   ├── elements/
│   │   ├── __init__.py         ★新規
│   │   ├── quad4.py            ★ Quad4PlaneStrain 追加
│   │   ├── quad4_bbar.py       ★ Quad4BBarPlaneStrain 追加
│   │   ├── tri3.py             ★ Tri3PlaneStrain 追加
│   │   └── tri6.py             ★ Tri6PlaneStrain 追加
│   └── materials/
│       ├── __init__.py         ★新規
│       └── elastic.py          ★ PlaneStrainElastic 追加
├── tests/
│   ├── __init__.py             ★新規
│   ├── test_linear_elastic_mixed.py          ★pytest化
│   ├── test_protocol_assembly.py             ★新規
│   ├── test_with_abaqus_data_manually_tensile.py    ★pytest化
│   ├── test_with_abaqus_data_manually_shear.py      ★pytest化
│   ├── test_with_abaqus_data_manually_cutter_sample1.py  ★pytest化(external)
│   └── test_with_abaqus_data_manually_cutter_sample5.py  ★pytest化(external)
└── results/
    └── test_results.xlsx
```

---

## ユーザー確認事項への対応

status-001 での確認事項に対し、以下の回答を受領済み:

| 確認事項 | 回答 | 対応 |
|---------|------|------|
| Phase 1 着手タイミング | すぐに開始してよい | 本statusで実施 |
| 補正四角形要素の取り込み | 予定なし | 放置（対応不要） |
| pymesh依存方針 | 自前実装可、Abaqus準拠 | テストではimportorskipで対処。メッシュI/Oは将来別途実装 |
| NNフレームワーク | PyTorch | Phase 6で対応予定 |
| CI/CD環境 | GitHub Actions利用可能 | ci.yml作成済み |

---

## TODO（次回以降の作業）

- [ ] Phase 1の残作業: 既存テストの命名規約統一（ファイル名を `test_elements_*.py` 等に整理）
- [ ] Phase 1の残作業: `pyproject.toml` に lint/format 設定追加（ruff等）
- [ ] Phase 2.1: Euler-Bernoulli梁（2D）の実装
- [ ] Phase 2.2: Timoshenko梁（2D）の実装
- [ ] メッシュI/O: Abaqus `.inp` フォーマットのパーサー自前実装（pymesh代替）

---

## 設計上の懸念

1. **Q4のD行列修正**: `quad4_ke_plane_strain()` 内で `D_tmp[0,1] *= 2; D_tmp[1,0] *= 2` としている箇所がある。これはengineering shearの取り扱いに関連するが、他の要素（TRI3, TRI6）では行っていない。要素間の記法の統一が望ましい。現時点ではAbaqusベンチマークで検証済みなので変更せず維持。
2. **`assembly.py` の二重実装**: レガシー関数（`assemble_global_stiffness_mixed`）と新関数（`assemble_global_stiffness`）が並存。Phase 2以降で新関数に一本化する際にレガシーを廃止予定。
