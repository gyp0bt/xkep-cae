# status-004: Phase 2.1/2.2 梁要素実装 & Abaqus .inp パーサー

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-003](./status-003.md)

**日付**: 2026-02-12
**作業者**: Claude Code
**ブランチ**: `claude/execute-todos-vybJ4`

---

## 実施内容

### Phase 2.1: Euler-Bernoulli梁（2D）

2D Euler-Bernoulli梁要素を実装。各節点3 DOF（ux, uy, θz）、要素あたり6 DOF。

#### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/sections/__init__.py` | 断面モジュール初期化 |
| `xkep_cae/sections/beam.py` | `BeamSection2D` dataclass（A, I、矩形/円形コンストラクタ） |
| `xkep_cae/elements/beam_eb2d.py` | EB梁要素: 局所/全体剛性行列、座標変換、分布荷重、`EulerBernoulliBeam2D`クラス |
| `xkep_cae/materials/beam_elastic.py` | `BeamElastic1D`（1D弾性構成則、E, nu, G） |
| `tests/test_beam_eb2d.py` | 21テスト |

#### 実装詳細

- **形状関数**: Hermite補間（3次）
- **局所剛性行列**: 軸方向（EA/L）+ 曲げ（12EI/L³ 系の標準形）
- **座標変換行列 T**: 6x6ブロック対角回転行列
- **分布荷重の等価節点力**: 局所y方向一様分布荷重 → 全体座標変換
- **Protocol適合**: `ElementProtocol` / `ConstitutiveProtocol` に適合

### Phase 2.2: Timoshenko梁（2D）

せん断変形を考慮した2D Timoshenko梁要素を実装。DOF構成はEB梁と同一。

#### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/elements/beam_timo2d.py` | Timoshenko梁要素: せん断パラメータΦによる修正定式化、`TimoshenkoBeam2D`クラス |
| `tests/test_beam_timo2d.py` | 14テスト |

#### 実装詳細

- **せん断パラメータ**: Φ = 12EI/(κGAL²)
- **せん断ロッキング対策**: 整合定式化（denominator = 1+Φ）を採用
- **せん断補正係数 κ**: デフォルト 5/6（矩形断面）
- **EB収束**: Φ→0 でEuler-Bernoulli梁に正確に一致
- **解析解**: δ_tip = PL³/(3EI) + PL/(κGA)

### メッシュI/O: Abaqus .inp パーサー

pymesh代替として、Abaqus .inpファイルの自前パーサーを実装。

#### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/io/__init__.py` | I/Oモジュール初期化 |
| `xkep_cae/io/abaqus_inp.py` | パーサー本体: `read_abaqus_inp()`, `AbaqusMesh` |
| `tests/test_abaqus_inp.py` | 21テスト |

#### 対応セクション

| セクション | 内容 |
|-----------|------|
| `*NODE` | 節点座標（2D/3D） |
| `*ELEMENT` | 要素接続配列（TRI3, Q4, TRI6, 梁等）、継続行対応 |
| `*NSET` | ノードセット（通常リスト / GENERATE） |

#### pymesh互換API

- `AbaqusMesh.get_node_coord_array()` → `[{"label", "x", "y", "z"}, ...]`
- `AbaqusMesh.get_element_array(allow_polymorphism, invalid_node)` → `[[label, n1, ...], ...]`
- `AbaqusMesh.get_node_labels_with_nset(name)` → `[label, ...]`

---

## テスト結果

```
72 passed, 2 skipped (pymesh依存)
ruff check: All checks passed!
ruff format: All files formatted
```

| テストファイル | テスト数 | 結果 |
|---------------|---------|------|
| `test_beam_eb2d.py` | 21 | PASSED |
| `test_beam_timo2d.py` | 14 | PASSED |
| `test_abaqus_inp.py` | 21 | PASSED |
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
├── pyproject.toml
├── .github/workflows/ci.yml
├── README.md                        ★ 更新
├── docs/
│   ├── roadmap.md                   ★ 更新
│   └── status/
│       ├── status-001.md
│       ├── status-002.md
│       ├── status-003.md
│       └── status-004.md            ★ 新規
├── xkep_cae/
│   ├── __init__.py
│   ├── api.py
│   ├── assembly.py
│   ├── bc.py
│   ├── solver.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── element.py               ElementProtocol
│   │   └── constitutive.py          ConstitutiveProtocol
│   ├── elements/
│   │   ├── __init__.py
│   │   ├── quad4.py                 Quad4PlaneStrain
│   │   ├── quad4_bbar.py            Quad4BBarPlaneStrain
│   │   ├── tri3.py                  Tri3PlaneStrain
│   │   ├── tri6.py                  Tri6PlaneStrain
│   │   ├── beam_eb2d.py             ★ EulerBernoulliBeam2D
│   │   └── beam_timo2d.py           ★ TimoshenkoBeam2D
│   ├── materials/
│   │   ├── __init__.py
│   │   ├── elastic.py               PlaneStrainElastic
│   │   └── beam_elastic.py          ★ BeamElastic1D
│   ├── sections/                    ★ 新規ディレクトリ
│   │   ├── __init__.py
│   │   └── beam.py                  BeamSection2D
│   └── io/                          ★ 新規ディレクトリ
│       ├── __init__.py
│       └── abaqus_inp.py            read_abaqus_inp, AbaqusMesh
├── tests/
│   ├── __init__.py
│   ├── test_beam_eb2d.py            ★ 新規（21テスト）
│   ├── test_beam_timo2d.py          ★ 新規（14テスト）
│   ├── test_abaqus_inp.py           ★ 新規（21テスト）
│   ├── test_elements_manufactured.py
│   ├── test_protocol_assembly.py
│   ├── test_benchmark_tensile.py
│   ├── test_benchmark_shear.py
│   ├── test_benchmark_cutter_q4tri3.py  (external)
│   └── test_benchmark_cutter_tri6.py    (external)
└── results/
    └── test_results.xlsx
```

---

## TODO（次回以降の作業）

- [ ] Phase 2.3: Timoshenko梁（3D空間）の実装
- [ ] Phase 2.4: 断面モデルの拡張（矩形・円形に加え、一般断面）
- [ ] 梁要素用のアセンブリ関数整備（`assemble_global_stiffness`の梁対応）
- [ ] pymesh依存テスト（cutter系）のAbaqusパーサー移行
- [ ] Phase 3: 幾何学的非線形（Newton-Raphson, 共回転定式化）

---

## 設計上の懸念

1. **ElementProtocol の `thickness` パラメータ**: 梁要素では`thickness`が不要だが、既存のProtocolインタフェースとの互換性のために引数を受け取って無視する設計にしている。Phase 2.3以降でProtocolの見直し（`section`パラメータの追加）を検討すべき。
2. **Q4のD行列修正**: 前回から引き継ぎ。`quad4_ke_plane_strain()` の `D_tmp[0,1] *= 2` 問題。
3. **`assembly.py` の二重実装**: レガシー関数が残存。梁要素のアセンブリ統合時に整理予定。
