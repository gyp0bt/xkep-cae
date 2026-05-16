"""VtkExportProcess API + 基本動作テスト."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from xkep_cae.core import MeshData, PostProcess, SolverResultData
from xkep_cae.core.testing import binds_to
from xkep_cae.output.vtk_export import (
    VtkExportConfig,
    VtkExportProcess,
    VtkExportResult,
)


def _make_simple_beam_mesh(n_nodes: int = 5, length: float = 10.0) -> MeshData:
    """直線 1D 梁メッシュ (x 軸沿い、n_nodes-1 個の line 要素)."""
    coords = np.zeros((n_nodes, 3), dtype=float)
    coords[:, 0] = np.linspace(0.0, length, n_nodes)
    conn = np.column_stack([np.arange(n_nodes - 1, dtype=int), np.arange(1, n_nodes, dtype=int)])
    return MeshData(node_coords=coords, connectivity=conn, radii=0.5, n_strands=1)


def _make_result_with_history(mesh: MeshData, n_steps: int = 3, ndof: int = 6) -> SolverResultData:
    """各 step で u_z を段階的に増やした履歴を持つ SolverResultData."""
    n_nodes = mesh.node_coords.shape[0]
    history = []
    load_history = []
    for s in range(n_steps):
        u = np.zeros(n_nodes * ndof)
        frac = (s + 1) / n_steps
        for i in range(n_nodes):
            u[i * ndof + 2] = frac * 0.1 * mesh.node_coords[i, 0]
        history.append(u)
        load_history.append(frac)
    return SolverResultData(
        u=history[-1],
        converged=True,
        n_increments=n_steps,
        total_attempts=n_steps,
        displacement_history=tuple(history),
        load_history=tuple(load_history),
    )


@binds_to(VtkExportProcess)
class TestVtkExportProcessAPI:
    """VtkExportProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(VtkExportProcess, PostProcess)

    def test_time_series_writes_vtu_and_pvd(self, tmp_path: Path):
        """履歴ありで .vtu 群 + .pvd が生成される."""
        mesh = _make_simple_beam_mesh()
        result = _make_result_with_history(mesh, n_steps=3)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="beam",
        )
        out = VtkExportProcess().process(cfg)
        assert isinstance(out, VtkExportResult)
        assert out.n_timesteps == 3
        assert out.n_points == 5
        assert out.n_cells == 4
        assert len(out.vtu_paths) == 3
        for p in out.vtu_paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0
        assert out.pvd_path is not None
        assert Path(out.pvd_path).exists()

    def test_single_state_no_history(self, tmp_path: Path):
        """write_time_series=False で単一 .vtu のみ、.pvd なし."""
        mesh = _make_simple_beam_mesh()
        result = _make_result_with_history(mesh, n_steps=2)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="single",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        assert out.n_timesteps == 1
        assert len(out.vtu_paths) == 1
        assert out.pvd_path is None
        assert Path(out.vtu_paths[0]).name == "single.vtu"

    def test_empty_history_falls_back_to_u(self, tmp_path: Path):
        """displacement_history が空でも u から単一 .vtu を生成."""
        mesh = _make_simple_beam_mesh()
        n_nodes = mesh.node_coords.shape[0]
        u = np.zeros(n_nodes * 6)
        u[2] = 0.5
        result = SolverResultData(
            u=u,
            converged=True,
            n_increments=1,
            total_attempts=1,
        )
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="bare",
        )
        out = VtkExportProcess().process(cfg)
        assert out.n_timesteps == 1
        assert out.pvd_path is None


class TestVtkExportXmlStructure:
    """生成された XML が ParaView 互換の構造を持つことを検証."""

    def test_vtu_xml_is_well_formed(self, tmp_path: Path):
        mesh = _make_simple_beam_mesh(n_nodes=4)
        result = _make_result_with_history(mesh, n_steps=1)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="xml_test",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        root = tree.getroot()
        assert root.tag == "VTKFile"
        assert root.attrib["type"] == "UnstructuredGrid"
        piece = root.find("UnstructuredGrid/Piece")
        assert piece is not None
        assert int(piece.attrib["NumberOfPoints"]) == 4
        assert int(piece.attrib["NumberOfCells"]) == 3

    def test_pvd_collection_lists_each_vtu(self, tmp_path: Path):
        mesh = _make_simple_beam_mesh(n_nodes=3)
        result = _make_result_with_history(mesh, n_steps=4)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="pvd_test",
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.pvd_path)
        datasets = tree.getroot().findall("Collection/DataSet")
        assert len(datasets) == 4
        timesteps = [float(d.attrib["timestep"]) for d in datasets]
        # load_history は 0.25, 0.5, 0.75, 1.0 になる
        assert timesteps == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_cell_count_for_line_elements(self, tmp_path: Path):
        """VTK_LINE = type 3 が全要素に立っている."""
        mesh = _make_simple_beam_mesh(n_nodes=6)
        result = _make_result_with_history(mesh, n_steps=1)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="types_test",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        types_da = tree.getroot().find("UnstructuredGrid/Piece/Cells/DataArray[@Name='types']")
        assert types_da is not None
        types_vals = [int(x) for x in types_da.text.split()]
        assert types_vals == [3, 3, 3, 3, 3]

    def test_point_data_contains_displacement(self, tmp_path: Path):
        mesh = _make_simple_beam_mesh(n_nodes=3)
        result = _make_result_with_history(mesh, n_steps=1)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="pd_test",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        names = {
            da.attrib["Name"] for da in tree.getroot().iter("DataArray") if "Name" in da.attrib
        }
        assert "displacement" in names
        assert "rotation" in names  # default include_rotations=True
        assert "axial_strain" in names  # default include_axial_strain=True

    def test_include_flags_can_omit_fields(self, tmp_path: Path):
        mesh = _make_simple_beam_mesh(n_nodes=3)
        result = _make_result_with_history(mesh, n_steps=1)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="minimal",
            write_time_series=False,
            include_rotations=False,
            include_axial_strain=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        names = {
            da.attrib["Name"] for da in tree.getroot().iter("DataArray") if "Name" in da.attrib
        }
        assert "displacement" in names
        assert "rotation" not in names
        assert "axial_strain" not in names


class TestVtkExportPhysics:
    """生成された VTK が物理的に整合するかの検証."""

    def test_deformed_coords_equal_ref_plus_displacement(self, tmp_path: Path):
        """Points = node_coords + u_translation."""
        mesh = _make_simple_beam_mesh(n_nodes=4)
        result = _make_result_with_history(mesh, n_steps=1)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="phys",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        pts_da = tree.getroot().find("UnstructuredGrid/Piece/Points/DataArray")
        pts = np.array([float(x) for x in pts_da.text.split()]).reshape(-1, 3)

        u = result.u.reshape(-1, 6)
        expected = mesh.node_coords + u[:, :3]
        np.testing.assert_allclose(pts, expected, atol=1e-10)

    def test_axial_strain_zero_for_undeformed(self, tmp_path: Path):
        """変位ゼロなら axial_strain も 0."""
        mesh = _make_simple_beam_mesh(n_nodes=4)
        n_nodes = mesh.node_coords.shape[0]
        u = np.zeros(n_nodes * 6)
        result = SolverResultData(
            u=u,
            converged=True,
            n_increments=1,
            total_attempts=1,
            displacement_history=(u,),
            load_history=(0.0,),
        )
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="undeformed",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        eps_da = tree.getroot().find(
            "UnstructuredGrid/Piece/CellData/DataArray[@Name='axial_strain']"
        )
        eps = np.array([float(x) for x in eps_da.text.split()])
        np.testing.assert_allclose(eps, np.zeros(3), atol=1e-12)
