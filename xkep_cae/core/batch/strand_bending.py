"""StrandBendingBatchProcess — 撚線曲げ揺動ワークフロー.

設計仕様: docs/strand_bending.md

Phase 5: ソルバー結果連携 — Export/Render/Verify ワイヤリング完成。
実行ツリー:
  StrandMeshProcess → ContactSetupProcess → ContactFrictionProcess
    → ExportProcess → BeamRenderProcess
    → ConvergenceVerifyProcess → EnergyBalanceVerifyProcess → ContactVerifyProcess
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.contact.contact_force.strategy import (
    HuberContactForceProcess,
)
from xkep_cae.contact.friction.strategy import (
    CoulombReturnMappingProcess,
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
from xkep_cae.output.export import ExportConfig, ExportProcess, ExportResult
from xkep_cae.output.render import BeamRenderProcess, RenderConfig, RenderResult
from xkep_cae.time_integration.strategy import (
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
)
from xkep_cae.verify.contact import ContactVerifyInput, ContactVerifyProcess
from xkep_cae.verify.convergence import (
    ConvergenceVerifyInput,
    ConvergenceVerifyProcess,
)
from xkep_cae.verify.energy import EnergyBalanceVerifyInput, EnergyBalanceVerifyProcess

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class StrandBatchConfig:
    """撚線曲げ揺動バッチの設定.

    Phase 5: ソルバー結果連携。
    mesh_config が指定されない場合はワークフロー実行をスキップ。
    boundary / callbacks が指定された場合はソルバーも実行し、
    run_export / run_render / run_verify に応じて後処理・検証を実行。
    """

    mesh_config: StrandMeshConfig | None = None
    k_pen: float = 0.0
    mu: float = 0.15
    geometry_mode: str = "point_to_point"
    output_dir: str = "output"
    run_export: bool = True
    run_render: bool = False
    run_verify: bool = True
    # Phase 4: ソルバー実行用（境界条件・コールバック）
    boundary: BoundaryData | None = None
    callbacks: AssembleCallbacks | None = None
    run_solver: bool = False
    smoothing_delta: float = 0.0


@dataclass(frozen=True)
class StrandBatchResult:
    """撚線曲げ揺動バッチの結果."""

    mesh: MeshData | None = None
    solver_converged: bool = False
    solver_result: SolverResultData | None = None
    verify_result: VerifyResult | None = None
    export_result: ExportResult | None = None
    render_result: RenderResult | None = None
    elapsed_seconds: float = 0.0
    process_log: tuple[str, ...] = ()  # frozen 対応: list → tuple


# ── BatchProcess ───────────────────────────────────────────


class StrandBendingBatchProcess(
    BatchProcess[StrandBatchConfig, StrandBatchResult],
):
    """撚線曲げ揺動ワークフロー.

    実行ツリー（process-architecture.md §6）:
      StrandMeshProcess → ContactSetupProcess → ContactFrictionProcess
        → ExportProcess → BeamRenderProcess
        → ConvergenceVerifyProcess → EnergyBalanceVerifyProcess → ContactVerifyProcess

    Phase 5: ソルバー結果連携 — Export/Render/Verify ワイヤリング完成。
    boundary / callbacks が指定されるとソルバーも実行し、
    ソルバー結果に基づいて Export/Render/Verify を順次実行する。
    """

    meta = ProcessMeta(
        name="StrandBendingBatch",
        module="batch",
        version="4.0.0",
        document_path="docs/strand_bending.md",
    )

    uses = [
        # concrete プロセス（Phase 3-5 移行済み）
        StrandMeshProcess,
        ContactSetupProcess,
        ContactFrictionProcess,
        ExportProcess,
        BeamRenderProcess,
        ConvergenceVerifyProcess,
        EnergyBalanceVerifyProcess,
        ContactVerifyProcess,
        # Strategy プロセス（Phase 2 移行済み）
        AutoBeamEIPenalty,
        CoulombReturnMappingProcess,
        QuasiStaticProcess,
        GeneralizedAlphaProcess,
        PointToPointProcess,
        LineToLineGaussProcess,
        HuberContactForceProcess,
    ]

    def process(self, input_data: StrandBatchConfig) -> StrandBatchResult:
        """ワークフロー実行（uses 宣言順に直列実行）."""
        import time

        t0 = time.perf_counter()
        log: list[str] = []

        # ローカル変数に蓄積（frozen dataclass のため一括生成）
        _mesh: MeshData | None = None
        _solver_converged = False
        _solver_result: SolverResultData | None = None
        _verify_result: VerifyResult | None = None
        _export_result: ExportResult | None = None
        _render_result: RenderResult | None = None

        if input_data.mesh_config is None:
            log.append("StrandBendingBatchProcess: mesh_config 未指定 — スキップ")
            log.append(f"  geometry_mode={input_data.geometry_mode}")
            return StrandBatchResult(
                elapsed_seconds=time.perf_counter() - t0,
                process_log=tuple(log),
            )

        # 1. メッシュ生成
        log.append("StrandMeshProcess: start")
        mesh_proc = StrandMeshProcess()
        mesh_result = mesh_proc.process(input_data.mesh_config)
        _mesh = mesh_result.mesh
        log.append("StrandMeshProcess: done")

        # 2. 接触設定
        log.append("ContactSetupProcess: start")
        contact_proc = ContactSetupProcess()
        contact_config = ContactSetupConfig(
            mesh=mesh_result.mesh,
            k_pen=input_data.k_pen,
            mu=input_data.mu,
            smoothing_delta=input_data.smoothing_delta,
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
            _solver_converged = solver_result.converged
            _solver_result = solver_result
            log.append(
                f"ContactFrictionProcess: done "
                f"(converged={solver_result.converged}, "
                f"n_incr={solver_result.n_increments})"
            )

            # 4. Export（ソルバー結果あり + run_export 有効時）
            if input_data.run_export:
                log.append("ExportProcess: start")
                export_proc = ExportProcess()
                export_config = ExportConfig(
                    solver_result=solver_result,
                    mesh=mesh_result.mesh,
                    output_dir=input_data.output_dir,
                )
                _export_result = export_proc.process(export_config)
                log.append(f"ExportProcess: done (files={len(_export_result.exported_files)})")

            # 5. Render（ソルバー結果あり + run_render 有効時）
            if input_data.run_render:
                log.append("BeamRenderProcess: start")
                render_proc = BeamRenderProcess()
                render_config = RenderConfig(
                    solver_result=solver_result,
                    mesh=mesh_result.mesh,
                    output_dir=input_data.output_dir,
                )
                _render_result = render_proc.process(render_config)
                log.append(f"BeamRenderProcess: done (images={len(_render_result.image_paths)})")

            # 6. Verify（ソルバー結果あり + run_verify 有効時）
            if input_data.run_verify:
                log.append("VerifyProcess: start")
                verify_reports: list[str] = []

                # 6a. 収束検証
                conv_proc = ConvergenceVerifyProcess()
                conv_input = ConvergenceVerifyInput(solver_result=solver_result)
                conv_result = conv_proc.process(conv_input)
                verify_reports.append(conv_result.report_markdown)
                log.append(f"  ConvergenceVerify: {'PASS' if conv_result.passed else 'FAIL'}")

                # 6b. エネルギー収支検証
                energy_proc = EnergyBalanceVerifyProcess()
                energy_input = EnergyBalanceVerifyInput(solver_result=solver_result)
                energy_result = energy_proc.process(energy_input)
                verify_reports.append(energy_result.report_markdown)
                log.append(f"  EnergyBalanceVerify: {'PASS' if energy_result.passed else 'FAIL'}")

                # 6c. 接触検証
                contact_v_proc = ContactVerifyProcess()
                contact_v_input = ContactVerifyInput(solver_result=solver_result)
                contact_v_result = contact_v_proc.process(contact_v_input)
                verify_reports.append(contact_v_result.report_markdown)
                log.append(f"  ContactVerify: {'PASS' if contact_v_result.passed else 'FAIL'}")

                # 3検証の統合結果
                all_passed = conv_result.passed and energy_result.passed and contact_v_result.passed
                _verify_result = VerifyResult(
                    passed=all_passed,
                    checks={
                        **conv_result.checks,
                        **energy_result.checks,
                        **contact_v_result.checks,
                    },
                    report_markdown="\n\n".join(verify_reports),
                )
                log.append(f"VerifyProcess: done ({'PASS' if all_passed else 'FAIL'})")
        else:
            log.append("ContactFrictionProcess: skipped (no boundary/callbacks)")

        return StrandBatchResult(
            mesh=_mesh,
            solver_converged=_solver_converged,
            solver_result=_solver_result,
            verify_result=_verify_result,
            export_result=_export_result,
            render_result=_render_result,
            elapsed_seconds=time.perf_counter() - t0,
            process_log=tuple(log),
        )
