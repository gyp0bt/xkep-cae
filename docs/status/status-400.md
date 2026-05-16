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

- バイナリ + Base64 encoded `<DataArray>` 形式は実装していない。
  ASCII format で 19 本撚線（〜数千ノード）規模なら十分実用範囲。1000 本撚線
  規模になったらバイナリ化を検討（status-400 では scope 外）。
- `point_field_extras` / `cell_field_extras` のような拡張フィールド注入 API
  は実装していない。必要が出たら frozen dataclass に
  `extra_point_fields: tuple[tuple[str, np.ndarray, int], ...] = ()`
  のような形で追加可能。

## 事後追加 (応力・モーメント・パイプメッシュ)

ユーザー指示「応力、曲率ベクトル、接触力も見たい + paraview の Tube filter
依存なしで擬似ソリッド表示が欲しい」を受け、Phase 2 として以下を追加:

### CellData フィールド (全 default ON)

| Name | components | 式 |
|------|---|---|
| `axial_stress` | 1 | `E · ε_axial` (MPa) |
| `curvature_vector` | 3 | `(θ_j − θ_i) / L = (κ_x, κ_y, κ_z)` |
| `moment_vector` | 3 | `(G·J·κ_x, E·I·κ_y, E·I·κ_z)` (N·mm) |
| `max_bending_stress` | 1 | `√(M_y² + M_z²) · r / I` (MPa) |
| `torsion_shear_stress` | 1 | `\|M_x\| · r / J` (MPa) |
| `von_mises_stress` | 1 | `√((σ_axial + σ_b_max)² + 3 τ²)` (MPa) — 最危険繊維 |
| `contact_force` | 1 | `contact_pair_history` の \|p_n\| を要素別集計 |

円形断面前提: `I = π r⁴ / 4`, `J = π r⁴ / 2`, `G = E / (2(1+ν))`、
`r` は `mesh.radii` (もしくは `tube_radius_override`)。

### 擬似ソリッド「パイプメッシュ」

`tube_n_segments ≥ 3` で `<prefix>_pipe.pvd` + `_pipe_NNNNN.vtu` を並行
出力。各 line 要素を `n_segments` 角形断面 (VTK_QUAD ring) に展開し、
半径は `mesh.radii` を反映。ParaView の Tube filter に依存せず断面太さ
込みの 3D ソリッド表示が即座に取れる。PointData / CellData は元 mesh の値を
複製して replicated。

### 単体テスト追加 (合計 19 件、+11 件追加)

- `test_axial_stress_equals_E_times_strain`: σ = E·ε 検証
- `test_curvature_vector_from_rotation_difference`: κ = (θ_j − θ_i)/L
- `test_moment_vector_and_bending_torsion_stress`: M = E·I·κ / G·J·κ_x、
  σ_b = M·r/I、τ = M_x·r/J、σ_vM = √(σ_b² + 3τ²)
- `test_contact_force_aggregated_from_pair_history`: ペア両端要素に
  `|p_n|` 加算
- `TestVtkExportPipeMesh`: 4 件 (n_segments 設定で書き出し / default 無効 /
  VTK_QUAD type=9 / `tube_radius_override` が半径として効く)

### 実機検証 (3 本撚線 90° 曲げ)

`work/visualization/01_strand_bending_vtk_demo.py --n-strands 3` を再走、
全 9 種フィールド (PointData 2 + CellData 7) + パイプメッシュ (n=8 角形、
n_cells=384) の `.vtu` 群 + `.pvd` を出力 (17.6s、frac=1.0、274 timestep)。
ParaView で `strand_3_pipe.pvd` を開いて応力 / モーメント / 接触力の
時系列可視化を確認。

### 環境整備

- `uv venv .venv --python 3.14` で `.venv` 新設、
  `uv pip install -e ".[dev,plot]"` で依存解決
- 以後の開発は `source .venv/bin/activate` で activate して進める運用
