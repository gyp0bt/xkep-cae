"""StrandBendingBatchProcess — 撚線曲げ揺動ワークフロー.

設計仕様: process-architecture.md §6
撚線の曲げ揺動計算のワークフロー全体をオーケストレーションする。

断片G適用: ワークフローオーケストレーション専用。
NR反復や要素組立などのアルゴリズムホットパスは含まない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import BatchProcess
from xkep_cae.process.concrete.post_export import ExportConfig, ExportProcess, ExportResult
from xkep_cae.process.concrete.post_render import BeamRenderProcess, RenderConfig, RenderResult
from xkep_cae.process.concrete.pre_contact import (
    ContactSetupConfig,
    ContactSetupProcess,
)
from xkep_cae.process.concrete.pre_mesh import (
    StrandMeshConfig,
    StrandMeshProcess,
)
from xkep_cae.process.concrete.solve_contact_friction import (
    ContactFrictionProcess,
)
from xkep_cae.process.data import (
    AssembleCallbacks,
    BoundaryData,
    ContactFrictionInputData,
    MeshData,
    VerifyResult,
)
from xkep_cae.process.verify.convergence import (
    ConvergenceVerifyInput,
    ConvergenceVerifyProcess,
)


@dataclass(frozen=True)
class BatchConfig:
    """撚線曲げ揺動バッチの設定.

    各プロセスの設定をまとめて保持する。
    """

    mesh_config: StrandMeshConfig
    boundary: BoundaryData
    callbacks: AssembleCallbacks
    k_pen: float = 0.0
    use_friction: bool = True
    mu: float = 0.15
    contact_mode: str = "smooth_penalty"
    output_dir: str = "output"
    run_export: bool = True
    run_render: bool = False
    run_verify: bool = True


@dataclass
class BatchResult:
    """撚線曲げ揺動バッチの結果."""

    mesh: MeshData | None = None
    solver_converged: bool = False
    verify_result: VerifyResult | None = None
    export_result: ExportResult | None = None
    render_result: RenderResult | None = None
    elapsed_seconds: float = 0.0
    process_log: list[str] = field(default_factory=list)


class StrandBendingBatchProcess(
    BatchProcess[BatchConfig, BatchResult],
):
    """撚線曲げ揺動ワークフロー.

    実行ツリー（process-architecture.md §6）:
      StrandMeshProcess → ContactSetupProcess → ContactFrictionProcess
        → [ExportProcess] → [BeamRenderProcess] → [ConvergenceVerifyProcess]

    断片G: ワークフロー orchestration のみ。ホットパスは各子プロセスに委譲。
    """

    meta = ProcessMeta(
        name="StrandBendingBatch",
        module="batch",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
    )

    uses = [
        StrandMeshProcess,
        ContactSetupProcess,
        ContactFrictionProcess,
        ExportProcess,
        BeamRenderProcess,
        ConvergenceVerifyProcess,
    ]

    def process(self, input_data: BatchConfig) -> BatchResult:
        """ワークフロー実行（uses 宣言順に直列実行）."""
        import time

        t0 = time.perf_counter()
        log: list[str] = []
        result = BatchResult()

        # 1. メッシュ生成
        log.append("StrandMeshProcess: start")
        mesh_proc = StrandMeshProcess()
        mesh_result = mesh_proc.process(input_data.mesh_config)
        mesh_data = MeshData(
            node_coords=mesh_result.node_coords,
            connectivity=mesh_result.connectivity,
            radii=mesh_result.radii,
            n_strands=mesh_result.n_strands,
            layer_ids=mesh_result.layer_ids,
        )
        result.mesh = mesh_data
        log.append("StrandMeshProcess: done")

        # 2. 接触設定
        log.append("ContactSetupProcess: start")
        contact_proc = ContactSetupProcess()
        contact_config = ContactSetupConfig(
            mesh=mesh_data,
            k_pen=input_data.k_pen,
            use_friction=input_data.use_friction,
            mu=input_data.mu,
            contact_mode=input_data.contact_mode,
        )
        contact_data = contact_proc.process(contact_config)
        log.append("ContactSetupProcess: done")

        # 3. 摩擦接触ソルバー
        log.append("ContactFrictionProcess: start")
        solver_proc = ContactFrictionProcess()
        solver_input = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=input_data.boundary,
            contact=contact_data,
            callbacks=input_data.callbacks,
        )
        solver_result = solver_proc.process(solver_input)
        result.solver_converged = solver_result.converged
        log.append(f"ContactFrictionProcess: done (converged={solver_result.converged})")

        # 4. エクスポート（オプション）
        if input_data.run_export:
            log.append("ExportProcess: start")
            export_proc = ExportProcess()
            export_config = ExportConfig(
                solver_result=solver_result,
                mesh=mesh_data,
                output_dir=input_data.output_dir,
            )
            result.export_result = export_proc.process(export_config)
            log.append("ExportProcess: done")

        # 5. レンダリング（オプション）
        if input_data.run_render:
            log.append("BeamRenderProcess: start")
            render_proc = BeamRenderProcess()
            render_config = RenderConfig(
                solver_result=solver_result,
                mesh=mesh_data,
                output_dir=input_data.output_dir,
            )
            result.render_result = render_proc.process(render_config)
            log.append("BeamRenderProcess: done")

        # 6. 収束検証（オプション）
        if input_data.run_verify:
            log.append("ConvergenceVerifyProcess: start")
            verify_proc = ConvergenceVerifyProcess()
            verify_input = ConvergenceVerifyInput(solver_result=solver_result)
            result.verify_result = verify_proc.process(verify_input)
            log.append(f"ConvergenceVerifyProcess: done (passed={result.verify_result.passed})")

        result.elapsed_seconds = time.perf_counter() - t0
        result.process_log = log
        return result
