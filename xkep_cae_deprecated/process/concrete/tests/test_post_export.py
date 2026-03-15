"""ExportProcess の1:1テスト."""

from __future__ import annotations

import tempfile

import numpy as np

from xkep_cae_deprecated.process.concrete.post_export import (
    ExportConfig,
    ExportProcess,
    ExportResult,
)
from xkep_cae_deprecated.process.data import MeshData, SolverResultData
from xkep_cae_deprecated.process.testing import binds_to


def _make_dummy_result() -> tuple[SolverResultData, MeshData]:
    mesh = MeshData(
        node_coords=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        connectivity=np.array([[0, 1]]),
        radii=0.1,
        n_strands=1,
    )
    result = SolverResultData(
        u=np.zeros(12),
        converged=True,
        n_increments=1,
        total_newton_iterations=3,
    )
    return result, mesh


@binds_to(ExportProcess)
class TestExportProcess:
    """ExportProcess の単体テスト."""

    def test_meta(self):
        assert ExportProcess.meta.name == "Export"
        assert ExportProcess.meta.module == "post"
        assert not ExportProcess.meta.deprecated

    def test_process_csv_export(self):
        result, mesh = _make_dummy_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                solver_result=result,
                mesh=mesh,
                output_dir=tmpdir,
                formats=("csv",),
            )
            proc = ExportProcess()
            out = proc.process(config)
            assert isinstance(out, ExportResult)
            assert len(out.exported_files) >= 1

    def test_process_json_export(self):
        result, mesh = _make_dummy_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(
                solver_result=result,
                mesh=mesh,
                output_dir=tmpdir,
                formats=("json",),
            )
            proc = ExportProcess()
            out = proc.process(config)
            assert len(out.exported_files) >= 1

    def test_registry_registered(self):
        from xkep_cae_deprecated.process.base import AbstractProcess

        assert "ExportProcess" in AbstractProcess._registry
