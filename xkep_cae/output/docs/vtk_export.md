# VtkExportProcess

[← README](../../../README.md)

## 概要

ParaView 用 VTK XML 出力の PostProcess。`SolverResultData` + `MeshData` を
受け取り、梁要素（VTK_LINE = 2 ノード線分セル）の変形形状とフィールド値を
`.vtu`（time-step ごと）+ `.pvd`（time-series collection）として書き出す。

ParaView で `.pvd` を開くと、各 timestep が `load_history` の値で
アニメーション再生される。

依存追加なし — 生の XML を文字列として直接書く実装。

## 入出力

- **入力**: `VtkExportConfig`
  - `solver_result: SolverResultData` — `displacement_history` / `load_history` / `u` を参照
  - `mesh: MeshData` — `node_coords` (n_nodes, 3) + `connectivity` (n_elems, 2)
  - `output_dir: str = "output/vtk"` — 出力先（自動作成）
  - `prefix: str = "result"` — ファイル名プレフィックス
  - `ndof_per_node: int = 6` — DOF レイアウト (u_x, u_y, u_z, θ_x, θ_y, θ_z)
  - `write_time_series: bool = True` — False で最終 state のみ単一 .vtu
  - `include_rotations: bool = True` — PointData に θ を含める
  - `include_axial_strain: bool = True` — CellData に ε_axial を含める

- **出力**: `VtkExportResult`
  - `vtu_paths: tuple[str, ...]` — 生成された .vtu の絶対パス列
  - `pvd_path: str | None` — time-series のとき .pvd のパス
  - `n_timesteps / n_points / n_cells: int`

## 出力フィールド

### Geometry（各 timestep の Points）

`deformed_coords = node_coords + u_translation` の (n_nodes, 3) 配列。

### PointData

| Name | components | 説明 |
|------|---|------|
| `displacement` | 3 | 累積並進変位 (u_x, u_y, u_z) |
| `rotation` | 3 | 累積回転 (θ_x, θ_y, θ_z) — `include_rotations=True` のみ |

### CellData

| Name | components | 説明 |
|------|---|------|
| `axial_strain` | 1 | 軸方向ひずみ ε = (L_def − L_ref) / L_ref |
| `axial_stress` | 1 | σ_axial = E · ε_axial [MPa] |
| `curvature_vector` | 3 | κ = (θ_j − θ_i) / L = (κ_x ねじり率, κ_y, κ_z) |
| `moment_vector` | 3 | (M_torsion = G·J·κ_x, M_bend_y = E·I·κ_y, M_bend_z = E·I·κ_z)、円形断面前提 |
| `max_bending_stress` | 1 | 最大曲げ繊維応力 √(M_y² + M_z²) · r / I [MPa] |
| `torsion_shear_stress` | 1 | 表面ねじりせん断応力 \|M_x\| · r / J [MPa] |
| `von_mises_stress` | 1 | 最危険繊維での換算応力 √((σ_axial + σ_b_max)² + 3 τ²) [MPa] |
| `contact_force` | 1 | 要素別 \|p_n\| 集計 ( `contact_pair_history` 必須 ) |

円形断面の前提:
- `I = π r⁴ / 4` (二次モーメント), `J = π r⁴ / 2` (極二次モーメント)
- `G = E / (2(1+ν))` ( `young_modulus` / `poisson_ratio` から)
- `r` は `mesh.radii` ( `tube_radius_override` で上書き可)

## ParaView での開き方

1. `output/vtk/result.pvd` を File → Open
2. Properties → Apply で読み込み
3. 上部の時刻スライダー（▶︎）で時系列再生
4. 変形を強調したい場合は Filters → Common → Warp By Vector で
   `displacement` を選択（既に deformed coord で出力済みなので通常不要）

## 時系列の timestep 値

`load_history` が `displacement_history` と同長なら `load_history[i]` を
timestep に使用する（典型的には荷重比 0.0〜1.0）。長さが合わない場合は
整数インデックスを使用する。

## ProcessMeta

- name: `VtkExport`
- module: `post`
- version: `1.0.0`
- category: `PostProcess`

## 関連

- [ExportProcess](export.md) — CSV/JSON エクスポート
- [Strand3DContourProcess](strand_contour.md) — matplotlib による PNG 3D 描画
- [StressContour3DProcess](stress_contour.md) — 応力コンター PNG
