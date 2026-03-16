"""StrandBendingBatchProcess のテスト.

@binds_to による 1:1 紐付け + BatchProcess カテゴリ検証。
"""

from __future__ import annotations

from xkep_cae.core import BatchProcess
from xkep_cae.core.batch import (
    StrandBatchConfig,
    StrandBatchResult,
    StrandBendingBatchProcess,
)
from xkep_cae.core.testing import binds_to
from xkep_cae.mesh.process import StrandMeshConfig


@binds_to(StrandBendingBatchProcess)
class TestStrandBendingBatchProcess:
    """StrandBendingBatchProcess の単体テスト."""

    def test_is_batch_process(self):
        proc = StrandBendingBatchProcess()
        assert isinstance(proc, BatchProcess)

    def test_uses_not_empty(self):
        assert len(StrandBendingBatchProcess.uses) > 0

    def test_meta_name(self):
        assert StrandBendingBatchProcess.meta.name == "StrandBendingBatch"

    def test_meta_module(self):
        assert StrandBendingBatchProcess.meta.module == "batch"

    def test_meta_version(self):
        assert StrandBendingBatchProcess.meta.version == "2.0.0"

    def test_process_returns_result_without_mesh(self):
        """mesh_config 未指定時はスキップして結果を返す."""
        proc = StrandBendingBatchProcess()
        config = StrandBatchConfig()
        result = proc.process(config)
        assert isinstance(result, StrandBatchResult)
        assert len(result.process_log) > 0

    def test_process_log_populated_without_mesh(self):
        proc = StrandBendingBatchProcess()
        config = StrandBatchConfig()
        result = proc.process(config)
        assert any("スキップ" in line for line in result.process_log)

    def test_default_config(self):
        config = StrandBatchConfig()
        assert config.contact_mode == "smooth_penalty"
        assert config.geometry_mode == "point_to_point"
        assert config.use_friction is True
        assert config.mesh_config is None

    def test_custom_config(self):
        config = StrandBatchConfig(
            contact_mode="ncp",
            geometry_mode="line_to_line",
            use_friction=False,
        )
        assert config.contact_mode == "ncp"
        assert config.geometry_mode == "line_to_line"
        assert config.use_friction is False

    def test_uses_includes_concrete_processes(self):
        """Phase 3 で追加された concrete プロセスが uses に含まれる."""
        from xkep_cae.contact.setup.process import ContactSetupProcess
        from xkep_cae.mesh.process import StrandMeshProcess
        from xkep_cae.output.export import ExportProcess
        from xkep_cae.output.render import BeamRenderProcess
        from xkep_cae.verify.convergence import ConvergenceVerifyProcess

        uses = StrandBendingBatchProcess.uses
        assert StrandMeshProcess in uses
        assert ContactSetupProcess in uses
        assert ExportProcess in uses
        assert BeamRenderProcess in uses
        assert ConvergenceVerifyProcess in uses

    def test_full_workflow_with_mesh(self):
        """mesh_config を指定してフルワークフロー実行."""
        proc = StrandBendingBatchProcess()
        mesh_config = StrandMeshConfig(n_strands=7, n_elements_per_pitch=16)
        config = StrandBatchConfig(mesh_config=mesh_config)
        result = proc.process(config)
        assert isinstance(result, StrandBatchResult)
        assert result.mesh is not None
        assert result.mesh.n_strands == 7
        assert any("StrandMeshProcess: done" in line for line in result.process_log)
        assert any("ContactSetupProcess: done" in line for line in result.process_log)
