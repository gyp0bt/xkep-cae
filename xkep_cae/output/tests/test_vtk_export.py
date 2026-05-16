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
        assert "axial_stress" in names  # default include_axial_stress=True
        assert "curvature_vector" in names  # default include_curvature_vector=True
        assert "moment_vector" in names
        assert "max_bending_stress" in names
        assert "torsion_shear_stress" in names
        assert "von_mises_stress" in names

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
            include_axial_stress=False,
            include_curvature_vector=False,
            include_contact_force=False,
            include_moment_vector=False,
            include_max_bending_stress=False,
            include_torsion_shear_stress=False,
            include_von_mises_stress=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        names = {
            da.attrib["Name"] for da in tree.getroot().iter("DataArray") if "Name" in da.attrib
        }
        assert "displacement" in names
        assert "rotation" not in names
        assert "axial_strain" not in names
        assert "axial_stress" not in names
        assert "curvature_vector" not in names
        assert "contact_force" not in names
        assert "moment_vector" not in names
        assert "max_bending_stress" not in names
        assert "torsion_shear_stress" not in names
        assert "von_mises_stress" not in names


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

    def test_axial_stress_equals_E_times_strain(self, tmp_path: Path):
        """σ_axial = E · ε_axial の関係が成立する."""
        mesh = _make_simple_beam_mesh(n_nodes=3, length=10.0)
        # 軸方向に一様引張: u_x の線形分布
        n_nodes = mesh.node_coords.shape[0]
        u = np.zeros(n_nodes * 6)
        elongation_ratio = 0.01  # ε = 0.01
        for i in range(n_nodes):
            u[i * 6 + 0] = elongation_ratio * mesh.node_coords[i, 0]
        E_test = 200.0e3
        result = SolverResultData(
            u=u,
            converged=True,
            n_increments=1,
            total_attempts=1,
            displacement_history=(u,),
            load_history=(1.0,),
        )
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="stress",
            write_time_series=False,
            young_modulus=E_test,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        sigma_da = tree.getroot().find(
            "UnstructuredGrid/Piece/CellData/DataArray[@Name='axial_stress']"
        )
        sigma = np.array([float(x) for x in sigma_da.text.split()])
        expected = np.full(2, E_test * elongation_ratio)
        np.testing.assert_allclose(sigma, expected, rtol=1e-10)

    def test_curvature_vector_from_rotation_difference(self, tmp_path: Path):
        """曲率ベクトル κ_e = (θ_j − θ_i) / L_e (要素ごと 3 成分)."""
        mesh = _make_simple_beam_mesh(n_nodes=3, length=10.0)
        # L_e = 5.0 で、隣接ノード回転差を仕掛ける
        n_nodes = mesh.node_coords.shape[0]
        u = np.zeros(n_nodes * 6)
        u[0 * 6 + 4] = 0.0  # θ_y at node 0
        u[1 * 6 + 4] = 0.1  # θ_y at node 1
        u[2 * 6 + 4] = 0.3  # θ_y at node 2
        result = SolverResultData(
            u=u,
            converged=True,
            n_increments=1,
            total_attempts=1,
            displacement_history=(u,),
            load_history=(1.0,),
        )
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="curv",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        kappa_da = tree.getroot().find(
            "UnstructuredGrid/Piece/CellData/DataArray[@Name='curvature_vector']"
        )
        kappa = np.array([float(x) for x in kappa_da.text.split()]).reshape(-1, 3)
        # κ_0 = (θ_1 - θ_0) / 5 → θ_y 成分 = 0.1/5 = 0.02
        # κ_1 = (θ_2 - θ_1) / 5 → θ_y 成分 = 0.2/5 = 0.04
        np.testing.assert_allclose(kappa[0], [0.0, 0.02, 0.0], atol=1e-12)
        np.testing.assert_allclose(kappa[1], [0.0, 0.04, 0.0], atol=1e-12)

    def test_moment_vector_and_bending_torsion_stress(self, tmp_path: Path):
        """曲率 → モーメント (E·I·κ_bend / G·J·κ_x) → 曲げ/ねじり応力の組."""
        mesh = _make_simple_beam_mesh(n_nodes=2, length=10.0)  # 1 要素、L=10
        n_nodes = mesh.node_coords.shape[0]
        u = np.zeros(n_nodes * 6)
        # κ_x (torsion) = 0.02 rad/mm, κ_y = 0.04 rad/mm, κ_z = 0
        u[0 * 6 + 3] = 0.0
        u[1 * 6 + 3] = 0.2  # θ_x: κ_x = 0.2 / 10 = 0.02
        u[0 * 6 + 4] = 0.0
        u[1 * 6 + 4] = 0.4  # θ_y: κ_y = 0.4 / 10 = 0.04
        result = SolverResultData(
            u=u,
            converged=True,
            n_increments=1,
            total_attempts=1,
            displacement_history=(u,),
            load_history=(1.0,),
        )
        E_test = 200.0e3
        nu_test = 0.3
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="resultants",
            write_time_series=False,
            young_modulus=E_test,
            poisson_ratio=nu_test,
        )
        out = VtkExportProcess().process(cfg)
        root = ET.parse(out.vtu_paths[0]).getroot()

        # mesh.radii=0.5 → I_sec=π·0.5⁴/4, J_pol=π·0.5⁴/2, G_mod=E/(2.6)
        r = 0.5
        I_sec = np.pi * r**4 / 4.0
        J_pol = np.pi * r**4 / 2.0
        G_mod = E_test / (2.0 * (1.0 + nu_test))
        kappa_x, kappa_y, kappa_z = 0.02, 0.04, 0.0
        M_x_exp = G_mod * J_pol * kappa_x
        M_y_exp = E_test * I_sec * kappa_y
        M_z_exp = E_test * I_sec * kappa_z

        def _get_cell_array(name: str) -> np.ndarray:
            da = root.find(f"UnstructuredGrid/Piece/CellData/DataArray[@Name='{name}']")
            return np.array([float(x) for x in da.text.split()])

        mv = _get_cell_array("moment_vector").reshape(-1, 3)
        np.testing.assert_allclose(mv[0], [M_x_exp, M_y_exp, M_z_exp], rtol=1e-10)

        sigma_b_exp = np.sqrt(M_y_exp**2 + M_z_exp**2) * r / I_sec
        np.testing.assert_allclose(
            _get_cell_array("max_bending_stress")[0], sigma_b_exp, rtol=1e-10
        )

        tau_exp = abs(M_x_exp) * r / J_pol
        np.testing.assert_allclose(_get_cell_array("torsion_shear_stress")[0], tau_exp, rtol=1e-10)

        # 軸ひずみゼロなので σ_vM = √(σ_b² + 3 τ²)
        sigma_vm_exp = np.sqrt(sigma_b_exp**2 + 3.0 * tau_exp**2)
        np.testing.assert_allclose(_get_cell_array("von_mises_stress")[0], sigma_vm_exp, rtol=1e-10)

    def test_contact_force_aggregated_from_pair_history(self, tmp_path: Path):
        """contact_pair_history が要素別 |p_n| 集計に正しく集約される."""
        from xkep_cae.core.data import ContactPairSnapshotEntry

        mesh = _make_simple_beam_mesh(n_nodes=4)  # 3 要素
        n_nodes = mesh.node_coords.shape[0]
        u = np.zeros(n_nodes * 6)
        # ペア: elem_a=0, elem_b=2 で p_n = 5.0
        pair_entry = ContactPairSnapshotEntry(
            elem_a=0,
            elem_b=2,
            p_n=5.0,
            gap=0.0,
            slip_s=0.0,
            slip_t=0.0,
            stick=True,
            dissipation=0.0,
        )
        result = SolverResultData(
            u=u,
            converged=True,
            n_increments=1,
            total_attempts=1,
            displacement_history=(u,),
            load_history=(1.0,),
            contact_pair_history=((1.0, (pair_entry,)),),
        )
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="contact",
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.vtu_paths[0])
        cf_da = tree.getroot().find(
            "UnstructuredGrid/Piece/CellData/DataArray[@Name='contact_force']"
        )
        cf = np.array([float(x) for x in cf_da.text.split()])
        # elem 0 と elem 2 に 5.0、elem 1 は 0.0
        np.testing.assert_allclose(cf, [5.0, 0.0, 5.0])


class TestVtkExportPipeMesh:
    """tube_n_segments > 0 で擬似ソリッド pipe mesh が出力される."""

    def test_pipe_mesh_written_when_segments_set(self, tmp_path: Path):
        mesh = _make_simple_beam_mesh(n_nodes=4)
        result = _make_result_with_history(mesh, n_steps=2)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="pipe",
            tube_n_segments=6,
        )
        out = VtkExportProcess().process(cfg)
        assert out.pipe_pvd_path is not None
        assert len(out.pipe_vtu_paths) == 2
        # pipe quad 数 = n_elems × n_segments = 3 × 6 = 18
        assert out.pipe_n_cells == 18
        # pipe point 数 = n_elems × 2 × n_segments = 3 × 2 × 6 = 36
        assert out.pipe_n_points == 36

    def test_pipe_disabled_by_default(self, tmp_path: Path):
        mesh = _make_simple_beam_mesh(n_nodes=3)
        result = _make_result_with_history(mesh, n_steps=2)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="nopipe",
        )
        out = VtkExportProcess().process(cfg)
        assert out.pipe_pvd_path is None
        assert out.pipe_vtu_paths == ()

    def test_pipe_uses_vtk_quad_type(self, tmp_path: Path):
        mesh = _make_simple_beam_mesh(n_nodes=3)
        result = _make_result_with_history(mesh, n_steps=1)
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="quadcheck",
            tube_n_segments=4,
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.pipe_vtu_paths[0])
        types_da = tree.getroot().find("UnstructuredGrid/Piece/Cells/DataArray[@Name='types']")
        types_vals = {int(x) for x in types_da.text.split()}
        assert types_vals == {9}  # VTK_QUAD

    def test_pipe_radius_override_applies(self, tmp_path: Path):
        """tube_radius_override が pipe mesh の半径として効く（無変形 = 軸そのまま）."""
        mesh = _make_simple_beam_mesh(n_nodes=2, length=10.0)
        n_nodes = mesh.node_coords.shape[0]
        u = np.zeros(n_nodes * 6)
        result = SolverResultData(
            u=u,
            converged=True,
            n_increments=1,
            total_attempts=1,
            displacement_history=(u,),
            load_history=(1.0,),
        )
        cfg = VtkExportConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="rover",
            tube_n_segments=4,
            tube_radius_override=2.0,
            write_time_series=False,
        )
        out = VtkExportProcess().process(cfg)
        tree = ET.parse(out.pipe_vtu_paths[0])
        pts_da = tree.getroot().find("UnstructuredGrid/Piece/Points/DataArray")
        pts = np.array([float(x) for x in pts_da.text.split()]).reshape(-1, 3)
        # 無変形なら軸は x 方向、断面は (y, z) 平面、各点の (y, z) ノルムが r
        radial = np.linalg.norm(pts[:, 1:], axis=1)
        np.testing.assert_allclose(radial, np.full(8, 2.0), atol=1e-10)
