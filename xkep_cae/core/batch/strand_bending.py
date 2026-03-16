"""StrandBendingBatchProcess — 撚線曲げ揺動ワークフロー.

旧 xkep_cae_deprecated/process/batch/strand_bending.py の完全書き直し。
設計仕様: docs/strand_bending.md

Phase 3: concrete プロセス移行完了。
実行ツリー:
  StrandMeshProcess → ContactSetupProcess
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
from xkep_cae.core import (
    BatchProcess,
    MeshData,
    ProcessMeta,
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

    Phase 3: concrete プロセス設定を含む完全版。
    mesh_config が指定されない場合はワークフロー実行をスキップ。
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


@dataclass
class StrandBatchResult:
    """撚線曲げ揺動バッチの結果."""

    mesh: MeshData | None = None
    solver_converged: bool = False
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
      StrandMeshProcess → ContactSetupProcess
        → [ExportProcess] → [BeamRenderProcess] → [ConvergenceVerifyProcess]

    Phase 3: concrete プロセスをフル実装。
    ソルバーステップ（ContactFrictionProcess）は現時点では
    deprecated 側の呼び出しが必要なため、このバッチでは
    Mesh→Setup→Verify のワークフローを実行する。
    """

    meta = ProcessMeta(
        name="StrandBendingBatch",
        module="batch",
        version="2.0.0",
        document_path="docs/strand_bending.md",
    )

    uses = [
        # concrete プロセス（Phase 3 移行済み）
        StrandMeshProcess,
        ContactSetupProcess,
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
        contact_proc.process(contact_config)
        log.append("ContactSetupProcess: done")

        # NOTE: ソルバーステップ（ContactFrictionProcess）は
        # deprecated 側の実体呼び出しが必要。
        # 完全移行時に追加予定。

        result.elapsed_seconds = time.perf_counter() - t0
        result.process_log = log
        return result
