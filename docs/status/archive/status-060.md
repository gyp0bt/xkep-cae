# status-060: Protocol 定義の 3D 解析対応拡張

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-25
**作業者**: Claude Code
**テスト数**: 1394（+3）

## 概要

`xkep_cae/core/` の Protocol 定義を現在の実装実態に合わせて拡張した。
初期の線形弾性平面ひずみ時代に設計された最小限の Protocol（`ElementProtocol` + `ConstitutiveProtocol`）を維持しつつ、非線形・動解析・弾塑性に対応する特化 Protocol を追加した。

## 設計方針

**後方互換性維持**: 既存の `ElementProtocol` と `ConstitutiveProtocol` はそのまま保持。新 Protocol はすべて追加のみ。

**Protocol 階層**:

| Protocol | 親 | 必須メソッド | 適合クラス例 |
|---|---|---|---|
| `ElementProtocol` | — | `local_stiffness`, `dof_indices` | Quad4, Tri3, Hex8BBar, etc. |
| `NonlinearElementProtocol` | `ElementProtocol` | + `internal_force`, `tangent_stiffness` | CosseratRod |
| `DynamicElementProtocol` | `ElementProtocol` | + `mass_matrix` | EB2D, Timo2D, Timo3D |
| `ConstitutiveProtocol` | — | `tangent` | PlaneStrainElastic, IsotropicElastic3D |
| `PlasticConstitutiveProtocol` | — | `return_mapping` | Plasticity1D, Plasticity3D |

**`PlasticConstitutiveProtocol` が `ConstitutiveProtocol` を継承しない理由**:
- 塑性材料（`Plasticity1D` 等）は `tangent()` を持たない
- 塑性 assembly は `return_mapping(strain, state)` で応力・接線・状態を一括取得する
- 線形弾性 assembly は `tangent()` で D 行列を取得する
- 使われ方が根本的に異なるため独立 Protocol とした

## 実装内容

### `xkep_cae/core/constitutive.py`（変更）

- `PlasticConstitutiveProtocol` を追加
  - `return_mapping(strain, state) -> tuple` — return mapping アルゴリズム
  - `@runtime_checkable` 対応
- docstring を整備（Protocol 設計意図、適合クラス例）

### `xkep_cae/core/element.py`（変更）

- `NonlinearElementProtocol(ElementProtocol, Protocol)` を追加
  - `internal_force(coords, u_elem, material) -> ndarray` — 内力ベクトル
  - `tangent_stiffness(coords, u_elem, material) -> ndarray` — 接線剛性行列
  - `@runtime_checkable` 対応
- `DynamicElementProtocol(ElementProtocol, Protocol)` を追加
  - `mass_matrix(coords, rho, *, lumped=False) -> ndarray` — 質量行列
  - `@runtime_checkable` 対応
- `ElementProtocol` の docstring 充実（属性説明、適合クラス一覧、座標系の補足）
- `ConstitutiveProtocol` を明示的に import（型注釈の解決性を改善）

### `xkep_cae/core/__init__.py`（変更）

- `NonlinearElementProtocol`, `DynamicElementProtocol`, `PlasticConstitutiveProtocol` をエクスポート
- モジュール docstring に Protocol 階層の説明を追加

### `tests/test_protocol_assembly.py`（変更）

3つの Protocol 適合テストを追加:

| テスト | 内容 |
|--------|------|
| `test_nonlinear_element_protocol` | CosseratRod が `NonlinearElementProtocol` に適合 |
| `test_dynamic_element_protocol` | EB2D/Timo2D/Timo3D が `DynamicElementProtocol` に適合 |
| `test_plastic_constitutive_protocol` | Plasticity1D が `PlasticConstitutiveProtocol` に適合 |

## ファイル変更

### 変更
- `xkep_cae/core/constitutive.py` — PlasticConstitutiveProtocol 追加
- `xkep_cae/core/element.py` — NonlinearElementProtocol, DynamicElementProtocol 追加
- `xkep_cae/core/__init__.py` — 新 Protocol エクスポート
- `tests/test_protocol_assembly.py` — 3つの適合テスト追加
- `docs/status/status-index.md` — status-060 行追加
- `docs/roadmap.md` — Protocol 拡張追記
- `README.md` — 現在状態更新

## 既存テストへの影響

**影響なし**。全 15 テスト（test_protocol_assembly.py 内）PASS。新 Protocol は既存コードに一切の変更を要求しない。

## 設計上の懸念・TODO

### Protocol 適合状況の現状

| 要素 | Element | Nonlinear | Dynamic |
|------|---------|-----------|---------|
| Quad4PlaneStrain | ✓ | ✗ | ✗ |
| Tri3PlaneStrain | ✓ | ✗ | ✗ |
| Tri6PlaneStrain | ✓ | ✗ | ✗ |
| Hex8BBar | ✓ | ✗ | ✗ |
| EulerBernoulliBeam2D | ✓ | ✗ | ✓ |
| TimoshenkoBeam2D | ✓ | ✗ | ✓ |
| TimoshenkoBeam3D | ✓ | ✗ | ✓ |
| CosseratRod | ✓ | ✓ | ✗ |

| 材料 | Constitutive | Plastic |
|------|-------------|---------|
| PlaneStrainElastic | ✓ | ✗ |
| IsotropicElastic3D | ✓ | ✗ |
| BeamElastic1D | ✓ | ✗ |
| Plasticity1D | ✗ | ✓ |
| Plasticity3D | ✗ | ✓ |

### 今後の拡張候補

- [ ] Q4 TL/UL 要素の `NonlinearElementProtocol` 適合化（現在は関数ベース）
- [ ] CosseratRod への `mass_matrix()` 追加（`DynamicElementProtocol` 適合化）
- [ ] `ContactProtocol` — 接触力・接触接線剛性の標準インタフェース
- [ ] `SectionProtocol` — 梁断面の共通インタフェース（A, Iy, Iz, J）

---
