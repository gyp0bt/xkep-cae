"""BeamRenderProcess の1:1テスト."""

from __future__ import annotations

import tempfile

import numpy as np

from xkep_cae_deprecated.process.concrete.post_render import (
    BeamRenderProcess,
    RenderConfig,
    RenderResult,
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


@binds_to(BeamRenderProcess)
class TestBeamRenderProcess:
    """BeamRenderProcess の単体テスト."""

    def test_meta(self):
        assert BeamRenderProcess.meta.name == "BeamRender"
        assert BeamRenderProcess.meta.module == "post"
        assert not BeamRenderProcess.meta.deprecated

    def test_process_returns_render_result(self):
        result, mesh = _make_dummy_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RenderConfig(
                solver_result=result,
                mesh=mesh,
                output_dir=tmpdir,
            )
            proc = BeamRenderProcess()
            out = proc.process(config)
            assert isinstance(out, RenderResult)

    def test_registry_registered(self):
        from xkep_cae_deprecated.process.base import AbstractProcess

        assert "BeamRenderProcess" in AbstractProcess._registry
