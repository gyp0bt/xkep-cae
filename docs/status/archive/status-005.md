# status-005: レガシー削除・Protocol API 一本化・Q4 D行列バグ修正

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-004](./status-004.md)

**日付**: 2026-02-12
**作業者**: Claude Code
**ブランチ**: `claude/execute-status-todos-pniSa`

---

## 実施内容

### レガシーコード全削除

後方互換不要の方針に基づき、レガシー関数・旧APIをすべて削除。

#### `assembly.py`

以下のレガシー関数を削除:
- `_edofs_for_quad4`, `_eKe_scatter_lil`, `estimate_stiffness_memory`, `print_progress`
- `_assemble_global_stiffness_mixed_legacy`, `assemble_global_stiffness_mixed`
- レガシーimport (`quad4_ke_plane_strain`, `tri3_ke_plane_strain`, `tri6_ke_plane_strain`, `constitutive_plane_strain`)

Protocol ベースの `assemble_global_stiffness()` のみ残存。

#### `api.py`

完全書き直し:
- 削除: `assemble_K_from_arrays_mixed`, `solve_plane_strain_from_label_maps`
- 新規: `solve_plane_strain()` — Protocol API ベースの高レベルソルバー
  - keyword-only の要素引数 (`elem_quads`, `elem_tris`, `elem_tri6`)
  - 内部で `assemble_global_stiffness`, `Quad4PlaneStrain`, `Tri3PlaneStrain`, `Tri6PlaneStrain`, `PlaneStrainElastic` を使用

### ElementProtocol の `thickness` パラメータ修正

`thickness: float` → `thickness: float | None = None` に変更。

- **平面要素**: `thickness if thickness is not None else 1.0` でデフォルト値を使用
- **梁要素**: `thickness` を無視（断面特性で管理）

対象ファイル:
- `xkep_cae/core/element.py` (Protocol定義)
- `xkep_cae/elements/quad4.py`, `tri3.py`, `tri6.py`, `quad4_bbar.py`
- `xkep_cae/elements/beam_eb2d.py`, `beam_timo2d.py`

### Q4 D行列バグ修正

`quad4_ke_plane_strain()` の `D_tmp[0,1] *= 2` / `D_tmp[1,0] *= 2` を削除。

**問題**: B行列はengineering shear strain規約(`gamma_xy = du/dy + dv/dx`)を使用しており、
D行列の(0,1),(1,0)成分（Lame定数λ）を2倍にする理由がない。
TRI3, TRI6, Q4 BBAR はすべて `B^T @ D @ B` を直接使用しており、Q4だけオフ対角項を
2倍にしていたのはPoisson結合の過大評価。

**修正**: `D_tmp` を廃止し、`B^T @ D @ B` を直接使用。他の要素と整合。

### pymesh依存テストのAbaqusパーサー移行

- `test_benchmark_cutter_q4tri3.py`: `pytest.importorskip("pymesh")` → `read_abaqus_inp` + `pytest.skip`
- `test_benchmark_cutter_tri6.py`: 同上

### 梁要素のアセンブリ統合テスト

`assemble_global_stiffness()` 経由での梁要素テストを追加:
- `test_assembly_beam_eb`: EB梁10要素片持ち梁 → 解析解 `delta = PL^3/(3EI)` と比較
- `test_assembly_beam_timo`: Timoshenko梁20要素片持ち梁 → 解析解 `delta = PL^3/(3EI) + PL/(kGA)` と比較

### テストファイル更新

| ファイル | 変更内容 |
|---------|---------|
| `test_protocol_assembly.py` | レガシー比較テスト削除、梁アセンブリテスト追加 |
| `test_elements_manufactured.py` | Protocol API直接使用に書き換え |
| `test_benchmark_tensile.py` | 新 `solve_plane_strain` に対応 |
| `test_benchmark_shear.py` | 新 `solve_plane_strain` に対応 |
| `test_benchmark_cutter_q4tri3.py` | Abaqusパーサー使用に移行 |
| `test_benchmark_cutter_tri6.py` | Abaqusパーサー使用に移行 |

---

## テスト結果

```
74 passed, 2 deselected (external), 19 warnings
ruff check: All checks passed!
ruff format: 32 files already formatted
```

| テストファイル | テスト数 | 結果 |
|---------------|---------|------|
| `test_abaqus_inp.py` | 21 | PASSED |
| `test_beam_eb2d.py` | 21 | PASSED |
| `test_beam_timo2d.py` | 14 | PASSED |
| `test_benchmark_shear.py` | 4 | PASSED |
| `test_benchmark_tensile.py` | 4 | PASSED |
| `test_elements_manufactured.py` | 3 | PASSED |
| `test_protocol_assembly.py` | 7 | PASSED |
| `test_benchmark_cutter_q4tri3.py` | 1 | DESELECTED (external) |
| `test_benchmark_cutter_tri6.py` | 1 | DESELECTED (external) |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/assembly.py` | レガシー関数削除、thickness引数変更 |
| `xkep_cae/api.py` | 全面書き直し |
| `xkep_cae/core/element.py` | thickness: float → float \| None |
| `xkep_cae/elements/quad4.py` | D行列バグ修正、thickness引数変更 |
| `xkep_cae/elements/tri3.py` | thickness引数変更 |
| `xkep_cae/elements/tri6.py` | thickness引数変更 |
| `xkep_cae/elements/quad4_bbar.py` | thickness引数変更 |
| `xkep_cae/elements/beam_eb2d.py` | thickness引数変更 |
| `xkep_cae/elements/beam_timo2d.py` | thickness引数変更 |
| `tests/test_protocol_assembly.py` | 書き直し + 梁テスト追加 |
| `tests/test_elements_manufactured.py` | Protocol API化 |
| `tests/test_benchmark_tensile.py` | 新API対応 |
| `tests/test_benchmark_shear.py` | 新API対応 |
| `tests/test_benchmark_cutter_q4tri3.py` | Abaqusパーサー移行 |
| `tests/test_benchmark_cutter_tri6.py` | Abaqusパーサー移行 |

---

## TODO（次回以降の作業）

- [ ] Phase 2.3: Timoshenko梁（3D空間）の実装
- [ ] Phase 2.4: 断面モデルの拡張（一般断面）
- [ ] Phase 3: 幾何学的非線形（Newton-Raphson, 共回転定式化）
- [ ] Q4要素のAbaqus解析解比較テスト追加（D行列修正の数値検証強化）

---

## 設計上のメモ

1. **レガシーコード完全削除済み**: `assembly.py` と `api.py` から旧関数をすべて除去。今後はProtocol APIのみ。
2. **Q4 D行列修正**: 既存テストは製造解テスト（f=Ku）のため修正による影響なし。Abaqus比較等の外部検証テストを追加して数値精度を確認すべき。
3. **thickness=None設計**: 梁要素と平面要素のProtocol共存のため、`thickness` をオプショナルに変更。梁要素は断面特性(`BeamSection2D`)で剛性を管理。
