"""ExportProcess のテスト."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from xkep_cae.core import MeshData, PostProcess, SolverResultData
from xkep_cae.core.testing import binds_to
from xkep_cae.output.export import ExportConfig, ExportProcess, ExportResult


def _make_solver_result() -> SolverResultData:
    return SolverResultData(
        u=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        converged=True,
        n_increments=5,
        total_attempts=20,
        contact_force_history=[1.0, 2.0, 3.0],
    )


def _make_mesh() -> MeshData:
    return MeshData(
        node_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        connectivity=np.array([[0, 1]]),
        radii=0.05,
        n_strands=1,
    )


@binds_to(ExportProcess)
class TestExportProcess:
    """ExportProcess の単体テスト."""

    def test_is_post_process(self):
        proc = ExportProcess()
        assert isinstance(proc, PostProcess)

    def test_meta_name(self):
        assert ExportProcess.meta.name == "Export"

    def test_meta_module(self):
        assert ExportProcess.meta.module == "post"

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                solver_result=_make_solver_result(),
                mesh=_make_mesh(),
                output_dir=tmpdir,
                formats=("csv",),
            )
            proc = ExportProcess()
            result = proc.process(config)
            assert isinstance(result, ExportResult)
            assert len(result.exported_files) == 2  # displacement + contact_force
            for f in result.exported_files:
                assert Path(f).exists()

    def test_json_export(self):
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                solver_result=_make_solver_result(),
                mesh=_make_mesh(),
                output_dir=tmpdir,
                formats=("json",),
            )
            proc = ExportProcess()
            result = proc.process(config)
            assert len(result.exported_files) == 1
            with open(result.exported_files[0]) as f:
                data = json.load(f)
            assert data["converged"] is True
            assert data["n_increments"] == 5

    def test_csv_and_json_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                solver_result=_make_solver_result(),
                mesh=_make_mesh(),
                output_dir=tmpdir,
                formats=("csv", "json"),
            )
            proc = ExportProcess()
            result = proc.process(config)
            assert len(result.exported_files) == 3  # disp + cf + json

    def test_config_frozen(self):
        config = ExportConfig(
            solver_result=_make_solver_result(),
            mesh=_make_mesh(),
        )
        try:
            config.prefix = "other"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass
