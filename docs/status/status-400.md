[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-400: `VtkExportProcess` 実装 — ParaView 用 VTK XML 出力 PostProcess（依存追加なし、汎用 1D 梁モデル対応）

**日付**: 2026-05-16

**テスト数**: 766 passed 5 skipped（status-399 の 755 + 新規 11）

## 概要

ローカル開発環境への移行に伴い、ユーザーが ParaView で電線曲げモデルの結果を
目視確認できる出力経路を新設。`MeshData` + `SolverResultData` を入力に
取る汎用 PostProcess として、`xkep_cae/output/vtk_export.py` を実装。

依存追加なし — 生の VTK XML を文字列として直接書く実装。`displacement_history`
を時系列として `.vtu` 群 + `.pvd` collection を出力する。ParaView で `.pvd` を
開くと `load_history` を timestep にしたアニメーション再生になる。

## スコープ

汎用 PostProcess として、`StrandBendingOscillation` 系列 / `beam_element_validation`
Phase α/β/γ の両方から流用可能な設計とした。

### 新規ファイル

- `xkep_cae/output/vtk_export.py` (+232 行)
  - `VtkExportConfig` (frozen dataclass): `solver_result` / `mesh` / `output_dir`
    / `prefix` / `ndof_per_node=6` / `write_time_series=True` /
    `include_rotations=True` / `include_axial_strain=True`
  - `VtkExportResult` (frozen): `vtu_paths` / `pvd_path` / `n_timesteps`
    / `n_points` / `n_cells`
  - `VtkExportProcess(PostProcess)`: `meta = ProcessMeta(name="VtkExport", ...)`
  - 純粋ヘルパー: `_build_data_array` / `_build_vtu_xml` / `_build_pvd_xml` /
    `_extract_translation_rotation` / `_compute_axial_strain` / `_format_*_array`
- `xkep_cae/output/docs/vtk_export.md` (+62 行) 設計仕様、README バックリンクあり
- `xkep_cae/output/tests/test_vtk_export.py` (+232 行、新規 **11 テスト**)
  - `TestVtkExportProcessAPI` (4): `@binds_to(VtkExportProcess)`、protocol、
    time-series、single-state、empty-history
  - `TestVtkExportXmlStructure` (5): XML well-formedness、.pvd dataset 列挙、
    VTK_LINE type=3、PointData/CellData フィールド名、include フラグ
  - `TestVtkExportPhysics` (2): 変形座標 = ref + u_trans、ゼロ変位で
    `axial_strain=0`
- `xkep_cae/output/__init__.py`: `VtkExportConfig` / `VtkExportProcess`
  / `VtkExportResult` を export 追加

### 出力フィールド

| 種別 | Name | components | 説明 |
|---|---|---|---|
| Geometry | Points | 3 | `node_coords + u_translation`（deformed coords）|
| PointData | `displacement` | 3 | 累積並進変位 (u_x, u_y, u_z) |
| PointData | `rotation` | 3 | 累積回転 (θ_x, θ_y, θ_z)（`include_rotations=True` のみ）|
| CellData | `axial_strain` | 1 | ε = (L_def − L_ref) / L_ref（`include_axial_strain=True` のみ）|

### Timestep の決定

`load_history` が `displacement_history` と同長 → `load_history[i]` を使用。
長さが合わない / 空 → 整数インデックスにフォールバック。

## ParaView 動作確認

`/tmp/vtk_demo/cantilever.pvd` でサンプル生成（11 ノード片持ち梁、6 timestep、
放物線状 z 変位）。生成された XML は `xml.etree.ElementTree` で parse 可能、
`VTKFile/UnstructuredGrid/Piece` 構造、`types` 配列が `[3, 3, ..., 3]`、
PointData/CellData の DataArray 名が `displacement` / `rotation` / `axial_strain`
であることを単体テストで検証済み。

## 回帰

```
xkep_cae/output/tests/                    33 passed
xkep_cae/output/tests/test_vtk_export.py  11 passed (新規)
contracts/validate_process_contracts.py   全 24 検査 OK
ruff check xkep_cae/ tests/               All checks passed
ruff format --check xkep_cae/ tests/      203 files already formatted
```

実装本体は output サブパッケージへの追加のみ、ソルバー / contact / mathematics
への影響なし。`test_helical_3d_hermite` 等の MCDD gate テストは無変更で
当然 PASS（実装本体無関与）。

## 次セッション最優先候補

CLAUDE.md「次の課題」§Phase ε ロードマップ参照。本 status は副次（基盤整備）の
扱いで、ε-2 = 3 strand 接触あり + N_sub=2000 を **status-401** で実施する
（番号を 1 つ繰り下げ）。実機解析時は VtkExportProcess を BenchmarkRunner
`post_processes` に組み込めば、検証スクリプトの定量 gate と並行して ParaView
での視覚確認が走るようになる。

### 想定使用例

```python
from xkep_cae.output import VtkExportConfig, VtkExportProcess

result = solver.run(...)   # ContactFrictionProcess の戻り値
cfg = VtkExportConfig(
    solver_result=result,
    mesh=mesh,
    output_dir="docs/verification/strand_3d_vtk",
    prefix=f"strand_{n_strands}",
)
out = VtkExportProcess().process(cfg)
print(f"ParaView で {out.pvd_path} を開いてください")
```

## 引継ぎ

- 接触力ノルム履歴（scalar per increment）は VtkExport には乗せず、既存
  `ExportProcess` の CSV 経路で運用継続。per-pair 接触力場として
  PointData に乗せたい場合は将来拡張（`contact_pair_history` から
  per-node 集計が必要）。
- バイナリ + Base64 encoded `<DataArray>` 形式は実装していない。
  ASCII format で 19 本撚線（〜数千ノード）規模なら十分実用範囲。1000 本撚線
  規模になったらバイナリ化を検討（status-400 では scope 外）。
- `point_field_extras` / `cell_field_extras` のような拡張フィールド注入 API
  は実装していない。必要が出たら frozen dataclass に
  `extra_point_fields: tuple[tuple[str, np.ndarray, int], ...] = ()`
  のような形で追加可能。
