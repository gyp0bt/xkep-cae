"""VtkExportProcess — ParaView 用 VTK XML 出力の PostProcess.

設計仕様: docs/vtk_export.md

`SolverResultData` + `MeshData` を受け取り、梁要素（VTK_LINE）の
変形形状とフィールド値を VTK XML 形式（.vtu / .pvd）で書き出す。
ParaView で `.pvd` を開くと時系列アニメーションになる。

依存追加なし — 生 XML を直接書く実装。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from xkep_cae.core import MeshData, PostProcess, ProcessMeta, SolverResultData

_VTK_LINE = 3  # VTK cell type for 2-node line


@dataclass(frozen=True)
class VtkExportConfig:
    """VTK 出力設定."""

    solver_result: SolverResultData
    mesh: MeshData
    output_dir: str = "output/vtk"
    prefix: str = "result"
    ndof_per_node: int = 6
    write_time_series: bool = True
    include_rotations: bool = True
    include_axial_strain: bool = True


@dataclass(frozen=True)
class VtkExportResult:
    """VTK 出力結果."""

    vtu_paths: tuple[str, ...] = field(default_factory=tuple)
    pvd_path: str | None = None
    n_timesteps: int = 0
    n_points: int = 0
    n_cells: int = 0


def _format_float_array(arr: np.ndarray) -> str:
    """numpy 配列を空白区切りの ASCII にする."""
    return " ".join(f"{v:.10e}" for v in np.asarray(arr, dtype=float).ravel())


def _format_int_array(arr: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in np.asarray(arr).ravel())


def _build_data_array(
    name: str,
    values: np.ndarray,
    *,
    dtype: str = "Float64",
    n_components: int = 1,
) -> str:
    """1 つの `<DataArray>` XML 断片を生成."""
    if dtype.startswith("Float"):
        body = _format_float_array(values)
    else:
        body = _format_int_array(values)
    return (
        f'      <DataArray type="{dtype}" Name="{escape(name)}" '
        f'NumberOfComponents="{n_components}" format="ascii">\n'
        f"        {body}\n"
        f"      </DataArray>\n"
    )


def _build_vtu_xml(
    points: np.ndarray,
    connectivity: np.ndarray,
    point_data: dict[str, tuple[np.ndarray, int]],
    cell_data: dict[str, tuple[np.ndarray, int]],
) -> str:
    """完全な .vtu XML 文字列を組み立てる.

    point_data / cell_data の値は `(array, n_components)` のタプル。
    """
    n_points = int(points.shape[0])
    n_cells = int(connectivity.shape[0])
    offsets = np.arange(1, n_cells + 1, dtype=int) * 2
    types = np.full(n_cells, _VTK_LINE, dtype=int)

    parts: list[str] = []
    parts.append('<?xml version="1.0"?>\n')
    parts.append('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
    parts.append("  <UnstructuredGrid>\n")
    parts.append(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n')
    parts.append("      <Points>\n")
    parts.append(_build_data_array("Points", points, n_components=3))
    parts.append("      </Points>\n")
    parts.append("      <Cells>\n")
    parts.append(_build_data_array("connectivity", connectivity, dtype="Int32"))
    parts.append(_build_data_array("offsets", offsets, dtype="Int32"))
    parts.append(_build_data_array("types", types, dtype="UInt8"))
    parts.append("      </Cells>\n")
    if point_data:
        parts.append("      <PointData>\n")
        for name, (arr, ncomp) in point_data.items():
            parts.append(_build_data_array(name, arr, n_components=ncomp))
        parts.append("      </PointData>\n")
    if cell_data:
        parts.append("      <CellData>\n")
        for name, (arr, ncomp) in cell_data.items():
            parts.append(_build_data_array(name, arr, n_components=ncomp))
        parts.append("      </CellData>\n")
    parts.append("    </Piece>\n")
    parts.append("  </UnstructuredGrid>\n")
    parts.append("</VTKFile>\n")
    return "".join(parts)


def _build_pvd_xml(vtu_files: list[str], timesteps: list[float]) -> str:
    """`.pvd` collection XML を組み立てる."""
    parts: list[str] = []
    parts.append('<?xml version="1.0"?>\n')
    parts.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    parts.append("  <Collection>\n")
    for t, f in zip(timesteps, vtu_files, strict=True):
        parts.append(f'    <DataSet timestep="{t:.10e}" group="" part="0" file="{escape(f)}"/>\n')
    parts.append("  </Collection>\n")
    parts.append("</VTKFile>\n")
    return "".join(parts)


def _extract_translation_rotation(
    u: np.ndarray, n_nodes: int, ndof_per_node: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """flat な u (n_nodes*ndof,) を (n_nodes,3) の trans と (n_nodes,3) の rot に分解."""
    u_reshaped = np.asarray(u, dtype=float).reshape(n_nodes, ndof_per_node)
    trans = u_reshaped[:, :3]
    rot = u_reshaped[:, 3:6] if ndof_per_node >= 6 else None
    return trans, rot


def _compute_axial_strain(
    deformed_coords: np.ndarray, connectivity: np.ndarray, ref_lengths: np.ndarray
) -> np.ndarray:
    """各要素の軸ひずみ ε = (L_def - L_ref) / L_ref."""
    p0 = deformed_coords[connectivity[:, 0]]
    p1 = deformed_coords[connectivity[:, 1]]
    l_def = np.linalg.norm(p1 - p0, axis=1)
    return (l_def - ref_lengths) / np.where(ref_lengths > 0.0, ref_lengths, 1.0)


class VtkExportProcess(PostProcess[VtkExportConfig, VtkExportResult]):
    """VTK XML エクスポートプロセス.

    `displacement_history` を時系列として `.vtu` 群 + `.pvd` を書き出す。
    `write_time_series=False` の場合は最終 state のみ単一 `.vtu` を出力する。

    ParaView で `.pvd` を開くと、各 timestep が `load_history` の値で
    アニメーション再生される。
    """

    meta = ProcessMeta(
        name="VtkExport",
        module="post",
        version="1.0.0",
        document_path="docs/vtk_export.md",
    )

    def process(self, input_data: VtkExportConfig) -> VtkExportResult:
        cfg = input_data
        mesh = cfg.mesh
        result = cfg.solver_result
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        node_coords = np.asarray(mesh.node_coords, dtype=float)
        connectivity = np.asarray(mesh.connectivity, dtype=int)
        n_nodes = int(node_coords.shape[0])
        n_cells = int(connectivity.shape[0])
        p0_ref = node_coords[connectivity[:, 0]]
        p1_ref = node_coords[connectivity[:, 1]]
        ref_lengths = np.linalg.norm(p1_ref - p0_ref, axis=1)

        if cfg.write_time_series and result.displacement_history:
            u_states = tuple(np.asarray(u, dtype=float) for u in result.displacement_history)
        else:
            u_states = (np.asarray(result.u, dtype=float),)

        load_steps = result.load_history
        if load_steps and len(load_steps) == len(u_states):
            timesteps = [float(t) for t in load_steps]
        else:
            timesteps = [float(i) for i in range(len(u_states))]

        vtu_files: list[str] = []
        n_steps = len(u_states)
        for i, u in enumerate(u_states):
            trans, rot = _extract_translation_rotation(u, n_nodes, cfg.ndof_per_node)
            deformed = node_coords + trans
            point_data: dict[str, tuple[np.ndarray, int]] = {
                "displacement": (trans, 3),
            }
            if cfg.include_rotations and rot is not None:
                point_data["rotation"] = (rot, 3)
            cell_data: dict[str, tuple[np.ndarray, int]] = {}
            if cfg.include_axial_strain:
                cell_data["axial_strain"] = (
                    _compute_axial_strain(deformed, connectivity, ref_lengths),
                    1,
                )

            xml = _build_vtu_xml(deformed, connectivity, point_data, cell_data)
            if n_steps == 1:
                fname = f"{cfg.prefix}.vtu"
            else:
                fname = f"{cfg.prefix}_{i:05d}.vtu"
            vtu_path = out_dir / fname
            vtu_path.write_text(xml, encoding="utf-8")
            vtu_files.append(fname)

        pvd_path: str | None = None
        if n_steps > 1:
            pvd_xml = _build_pvd_xml(vtu_files, timesteps)
            pvd_full = out_dir / f"{cfg.prefix}.pvd"
            pvd_full.write_text(pvd_xml, encoding="utf-8")
            pvd_path = str(pvd_full)

        return VtkExportResult(
            vtu_paths=tuple(str(out_dir / f) for f in vtu_files),
            pvd_path=pvd_path,
            n_timesteps=n_steps,
            n_points=n_nodes,
            n_cells=n_cells,
        )
