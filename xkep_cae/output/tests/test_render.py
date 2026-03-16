"""BeamRenderProcess のテスト."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from xkep_cae.core import MeshData, PostProcess, SolverResultData
from xkep_cae.core.testing import binds_to
from xkep_cae.output.render import BeamRenderProcess, RenderConfig, RenderResult


def _make_solver_result() -> SolverResultData:
    return SolverResultData(
        u=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        converged=True,
        n_increments=5,
        total_newton_iterations=20,
    )


def _make_mesh() -> MeshData:
    return MeshData(
        node_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        connectivity=np.array([[0, 1]]),
        radii=0.05,
        n_strands=1,
    )


@binds_to(BeamRenderProcess)
class TestBeamRenderProcess:
    """BeamRenderProcess の単体テスト."""

    def test_is_post_process(self):
        proc = BeamRenderProcess()
        assert isinstance(proc, PostProcess)

    def test_meta_name(self):
        assert BeamRenderProcess.meta.name == "BeamRender"

    def test_meta_module(self):
        assert BeamRenderProcess.meta.module == "post"

    def test_process_returns_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RenderConfig(
                solver_result=_make_solver_result(),
                mesh=_make_mesh(),
                output_dir=tmpdir,
            )
            proc = BeamRenderProcess()
            result = proc.process(config)
            assert isinstance(result, RenderResult)

    def test_deformed_csv_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RenderConfig(
                solver_result=_make_solver_result(),
                mesh=_make_mesh(),
                output_dir=tmpdir,
            )
            proc = BeamRenderProcess()
            result = proc.process(config)
            # 少なくとも deformed CSV は出力される
            csv_paths = [p for p in result.image_paths if p.endswith(".csv")]
            assert len(csv_paths) >= 1
            assert Path(csv_paths[0]).exists()

    def test_config_frozen(self):
        config = RenderConfig(
            solver_result=_make_solver_result(),
            mesh=_make_mesh(),
        )
        try:
            config.prefix = "other"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass
