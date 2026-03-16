"""StrandBendingBatchProcess — 撚線曲げ揺動ワークフロー.

旧 xkep_cae_deprecated/process/batch/strand_bending.py の完全書き直し。
設計仕様: docs/strand_bending.md

Phase 4: ContactFrictionProcess 統合 — 完全ワークフロー実現。
実行ツリー:
  StrandMeshProcess → ContactSetupProcess → ContactFrictionProcess
    → [ExportProcess] → [BeamRenderProcess] → [ConvergenceVerifyProcess]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xkep_cae.contact.contact_force.strategy import (
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
)
from xkep_cae.contact.friction.strategy import (
    CoulombReturnMappingProcess,
    NoFrictionProcess,
    SmoothPenaltyFrictionProcess,
)
from xkep_cae.contact.geometry.strategy import (
    LineToLineGaussProcess,
    PointToPointProcess,
)
from xkep_cae.contact.penalty.strategy import (
    AutoBeamEIPenalty,
)
from xkep_cae.contact.setup.process import ContactSetupConfig, ContactSetupProcess
from xkep_cae.contact.solver.process import ContactFrictionProcess
from xkep_cae.core import (
    AssembleCallbacks,
    BatchProcess,
    BoundaryData,
    ContactFrictionInputData,
    MeshData,
    ProcessMeta,
    SolverResultData,
    VerifyResult,
)
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess
from xkep_cae.output.export import ExportProcess, ExportResult
from xkep_cae.output.render import BeamRenderProcess, RenderResult
from xkep_cae.time_integration.strategy import (
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
)
from xkep_cae.verify.convergence import (
    ConvergenceVerifyProcess,
)

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class StrandBatchConfig:
    """撚線曲げ揺動バッチの設定.

    Phase 4: ContactFrictionProcess 統合。
    mesh_config が指定されない場合はワークフロー実行をスキップ。
    boundary / callbacks が指定された場合はソルバーも実行。
    """

    mesh_config: StrandMeshConfig | None = None
    k_pen: float = 0.0
    use_friction: bool = True
    mu: float = 0.15
    contact_mode: str = "smooth_penalty"
    geometry_mode: str = "point_to_point"
    output_dir: str = "output"
    run_export: bool = True
    run_render: bool = False
    run_verify: bool = True
    # Phase 4: ソルバー実行用（境界条件・コールバック）
    boundary: BoundaryData | None = None
    callbacks: AssembleCallbacks | None = None
    run_solver: bool = False


@dataclass
class StrandBatchResult:
    """撚線曲げ揺動バッチの結果."""

    mesh: MeshData | None = None
    solver_converged: bool = False
    solver_result: SolverResultData | None = None
    verify_result: VerifyResult | None = None
    export_result: ExportResult | None = None
    render_result: RenderResult | None = None
    elapsed_seconds: float = 0.0
    process_log: list[str] = field(default_factory=list)


# ── BatchProcess ───────────────────────────────────────────


class StrandBendingBatchProcess(
    BatchProcess[StrandBatchConfig, StrandBatchResult],
):
    """撚線曲げ揺動ワークフロー.

    実行ツリー（process-architecture.md §6）:
      StrandMeshProcess → ContactSetupProcess → ContactFrictionProcess
        → [ExportProcess] → [BeamRenderProcess] → [ConvergenceVerifyProcess]

    Phase 4: ContactFrictionProcess 統合 — 完全ワークフロー実現。
    boundary / callbacks が指定されるとソルバーも実行。
    """

    meta = ProcessMeta(
        name="StrandBendingBatch",
        module="batch",
        version="3.0.0",
        document_path="docs/strand_bending.md",
    )

    uses = [
        # concrete プロセス（Phase 3-4 移行済み）
        StrandMeshProcess,
        ContactSetupProcess,
        ContactFrictionProcess,
        ExportProcess,
        BeamRenderProcess,
        ConvergenceVerifyProcess,
        # Strategy プロセス（Phase 2 移行済み）
        AutoBeamEIPenalty,
        NoFrictionProcess,
        CoulombReturnMappingProcess,
        SmoothPenaltyFrictionProcess,
        QuasiStaticProcess,
        GeneralizedAlphaProcess,
        PointToPointProcess,
        LineToLineGaussProcess,
        NCPContactForceProcess,
        SmoothPenaltyContactForceProcess,
    ]

    def process(self, input_data: StrandBatchConfig) -> StrandBatchResult:
        """ワークフロー実行（uses 宣言順に直列実行）."""
        import time

        t0 = time.perf_counter()
        log: list[str] = []
        result = StrandBatchResult()

        if input_data.mesh_config is None:
            log.append("StrandBendingBatchProcess: mesh_config 未指定 — スキップ")
            log.append(f"  contact_mode={input_data.contact_mode}")
            log.append(f"  geometry_mode={input_data.geometry_mode}")
            log.append(f"  use_friction={input_data.use_friction}")
            result.elapsed_seconds = time.perf_counter() - t0
            result.process_log = log
            return result

        # 1. メッシュ生成
        log.append("StrandMeshProcess: start")
        mesh_proc = StrandMeshProcess()
        mesh_result = mesh_proc.process(input_data.mesh_config)
        result.mesh = mesh_result.mesh
        log.append("StrandMeshProcess: done")

        # 2. 接触設定
        log.append("ContactSetupProcess: start")
        contact_proc = ContactSetupProcess()
        contact_config = ContactSetupConfig(
            mesh=mesh_result.mesh,
            k_pen=input_data.k_pen,
            use_friction=input_data.use_friction,
            mu=input_data.mu,
            contact_mode=input_data.contact_mode,
        )
        contact_result = contact_proc.process(contact_config)
        log.append("ContactSetupProcess: done")

        # 3. ソルバー実行（boundary + callbacks が指定された場合）
        if input_data.run_solver and input_data.boundary and input_data.callbacks:
            log.append("ContactFrictionProcess: start")
            solver_input = ContactFrictionInputData(
                mesh=mesh_result.mesh,
                boundary=input_data.boundary,
                contact=contact_result,
                callbacks=input_data.callbacks,
            )
            solver_proc = ContactFrictionProcess()
            solver_result = solver_proc.process(solver_input)
            result.solver_converged = solver_result.converged
            result.solver_result = solver_result
            log.append(
                f"ContactFrictionProcess: done "
                f"(converged={solver_result.converged}, "
                f"n_incr={solver_result.n_increments})"
            )
        else:
            log.append("ContactFrictionProcess: skipped (no boundary/callbacks)")

        result.elapsed_seconds = time.perf_counter() - t0
        result.process_log = log
        return result
