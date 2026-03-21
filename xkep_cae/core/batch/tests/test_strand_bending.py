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
        assert StrandBendingBatchProcess.meta.version == "4.0.0"

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
        """Phase 3-5 で追加された concrete プロセスが uses に含まれる."""
        from xkep_cae.contact.setup.process import ContactSetupProcess
        from xkep_cae.contact.solver.process import ContactFrictionProcess
        from xkep_cae.mesh.process import StrandMeshProcess
        from xkep_cae.output.export import ExportProcess
        from xkep_cae.output.render import BeamRenderProcess
        from xkep_cae.verify.contact import ContactVerifyProcess
        from xkep_cae.verify.convergence import ConvergenceVerifyProcess
        from xkep_cae.verify.energy import EnergyBalanceVerifyProcess

        uses = StrandBendingBatchProcess.uses
        assert StrandMeshProcess in uses
        assert ContactSetupProcess in uses
        assert ContactFrictionProcess in uses
        assert ExportProcess in uses
        assert BeamRenderProcess in uses
        assert ConvergenceVerifyProcess in uses
        assert EnergyBalanceVerifyProcess in uses
        assert ContactVerifyProcess in uses

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

    def test_solver_skipped_without_boundary(self):
        """boundary 未指定時はソルバーをスキップ."""
        proc = StrandBendingBatchProcess()
        mesh_config = StrandMeshConfig(n_strands=7, n_elements_per_pitch=16)
        config = StrandBatchConfig(mesh_config=mesh_config, run_solver=True)
        result = proc.process(config)
        assert result.solver_converged is False
        assert any("skipped" in line for line in result.process_log)

    def test_solver_integration_with_simple_problem(self):
        """簡易問題でソルバー統合テスト."""
        import numpy as np
        import scipy.sparse as sp

        from xkep_cae.core import AssembleCallbacks, BoundaryData

        proc = StrandBendingBatchProcess()
        mesh_config = StrandMeshConfig(
            n_strands=2,
            n_elements_per_pitch=16,
            gap=0.01,
        )
        # まずメッシュだけ生成してサイズを確認
        from xkep_cae.mesh.process import StrandMeshProcess

        mesh_result = StrandMeshProcess().process(mesh_config)
        ndof = len(mesh_result.mesh.node_coords) * 6

        def assemble_tangent(u: np.ndarray) -> sp.csr_matrix:
            return sp.eye(ndof, format="csr") * 1.0

        def assemble_internal_force(u: np.ndarray) -> np.ndarray:
            return u * 1.0

        f_ext = np.zeros(ndof)
        f_ext[6 * 16] = 0.001

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=f_ext,
        )
        callbacks = AssembleCallbacks(
            assemble_tangent=assemble_tangent,
            assemble_internal_force=assemble_internal_force,
        )

        config = StrandBatchConfig(
            mesh_config=mesh_config,
            boundary=boundary,
            callbacks=callbacks,
            run_solver=True,
        )
        result = proc.process(config)
        assert result.solver_converged is True
        assert result.solver_result is not None
        assert any("ContactFrictionProcess: done" in line for line in result.process_log)

    def test_export_after_solver(self):
        """ソルバー結果のエクスポート統合テスト."""
        import tempfile

        import numpy as np
        import scipy.sparse as sp

        from xkep_cae.core import AssembleCallbacks, BoundaryData

        proc = StrandBendingBatchProcess()
        mesh_config = StrandMeshConfig(n_strands=2, n_elements_per_pitch=16, gap=0.01)
        from xkep_cae.mesh.process import StrandMeshProcess

        mesh_result = StrandMeshProcess().process(mesh_config)
        ndof = len(mesh_result.mesh.node_coords) * 6

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=np.zeros(ndof),
        )
        callbacks = AssembleCallbacks(
            assemble_tangent=lambda u: sp.eye(ndof, format="csr"),
            assemble_internal_force=lambda u: u * 1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StrandBatchConfig(
                mesh_config=mesh_config,
                boundary=boundary,
                callbacks=callbacks,
                run_solver=True,
                run_export=True,
                output_dir=tmpdir,
            )
            result = proc.process(config)
            assert result.export_result is not None
            assert len(result.export_result.exported_files) > 0
            assert any("ExportProcess: done" in line for line in result.process_log)

    def test_verify_after_solver(self):
        """ソルバー結果の検証統合テスト."""
        import numpy as np
        import scipy.sparse as sp

        from xkep_cae.core import AssembleCallbacks, BoundaryData

        proc = StrandBendingBatchProcess()
        mesh_config = StrandMeshConfig(n_strands=2, n_elements_per_pitch=16, gap=0.01)
        from xkep_cae.mesh.process import StrandMeshProcess

        mesh_result = StrandMeshProcess().process(mesh_config)
        ndof = len(mesh_result.mesh.node_coords) * 6

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=np.zeros(ndof),
        )
        callbacks = AssembleCallbacks(
            assemble_tangent=lambda u: sp.eye(ndof, format="csr"),
            assemble_internal_force=lambda u: u * 1.0,
        )

        config = StrandBatchConfig(
            mesh_config=mesh_config,
            boundary=boundary,
            callbacks=callbacks,
            run_solver=True,
            run_verify=True,
            run_export=False,
        )
        result = proc.process(config)
        assert result.verify_result is not None
        assert result.verify_result.passed is True
        assert "converged" in result.verify_result.checks
        assert "displacement_finite" in result.verify_result.checks
        assert any("VerifyProcess: done" in line for line in result.process_log)
        assert any("ConvergenceVerify: PASS" in line for line in result.process_log)
        assert any("EnergyBalanceVerify: PASS" in line for line in result.process_log)
        assert any("ContactVerify: PASS" in line for line in result.process_log)

    def test_render_after_solver(self):
        """ソルバー結果のレンダリング統合テスト."""
        import tempfile

        import numpy as np
        import scipy.sparse as sp

        from xkep_cae.core import AssembleCallbacks, BoundaryData

        proc = StrandBendingBatchProcess()
        mesh_config = StrandMeshConfig(n_strands=2, n_elements_per_pitch=16, gap=0.01)
        from xkep_cae.mesh.process import StrandMeshProcess

        mesh_result = StrandMeshProcess().process(mesh_config)
        ndof = len(mesh_result.mesh.node_coords) * 6

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=np.zeros(ndof),
        )
        callbacks = AssembleCallbacks(
            assemble_tangent=lambda u: sp.eye(ndof, format="csr"),
            assemble_internal_force=lambda u: u * 1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StrandBatchConfig(
                mesh_config=mesh_config,
                boundary=boundary,
                callbacks=callbacks,
                run_solver=True,
                run_render=True,
                run_export=False,
                run_verify=False,
                output_dir=tmpdir,
            )
            result = proc.process(config)
            assert result.render_result is not None
            assert len(result.render_result.image_paths) > 0
            assert any("BeamRenderProcess: done" in line for line in result.process_log)

    def test_no_export_without_solver(self):
        """ソルバー未実行時はExport/Render/Verifyもスキップ."""
        proc = StrandBendingBatchProcess()
        mesh_config = StrandMeshConfig(n_strands=7, n_elements_per_pitch=16)
        config = StrandBatchConfig(mesh_config=mesh_config, run_export=True)
        result = proc.process(config)
        assert result.export_result is None
        assert result.render_result is None
        assert result.verify_result is None
