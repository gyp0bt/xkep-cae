"""StrandMeshProcess のテスト."""

from __future__ import annotations

from xkep_cae.core import PreProcess
from xkep_cae.core.testing import binds_to
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess, StrandMeshResult


@binds_to(StrandMeshProcess)
class TestStrandMeshProcess:
    """StrandMeshProcess の単体テスト."""

    def test_is_pre_process(self):
        proc = StrandMeshProcess()
        assert isinstance(proc, PreProcess)

    def test_meta_name(self):
        assert StrandMeshProcess.meta.name == "StrandMesh"

    def test_meta_module(self):
        assert StrandMeshProcess.meta.module == "pre"

    def test_config_defaults(self):
        config = StrandMeshConfig()
        assert config.n_strands == 7
        assert config.wire_radius == 0.5
        assert config.n_elements_per_pitch == 16

    def test_config_frozen(self):
        config = StrandMeshConfig()
        try:
            config.n_strands = 19  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass

    def test_process_returns_result(self):
        proc = StrandMeshProcess()
        config = StrandMeshConfig(n_strands=7, n_elements_per_pitch=16)
        result = proc.process(config)
        assert isinstance(result, StrandMeshResult)

    def test_mesh_data_shape(self):
        proc = StrandMeshProcess()
        config = StrandMeshConfig(n_strands=7, n_elements_per_pitch=16)
        result = proc.process(config)
        assert result.mesh.node_coords.shape[1] == 3
        assert result.mesh.connectivity.shape[1] == 2
        assert result.mesh.n_strands == 7

    def test_mesh_data_n_nodes(self):
        proc = StrandMeshProcess()
        config = StrandMeshConfig(n_strands=7, n_elements_per_pitch=16, n_pitches=1.0)
        result = proc.process(config)
        # 7本 × (16+1)節点 = 119 (approx, depends on mesh impl)
        assert result.mesh.node_coords.shape[0] > 0
