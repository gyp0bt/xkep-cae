"""NCPContactSolverProcess — NCP接触ソルバーのProcess Wrapper.

設計仕様: process-architecture.md §2.4
newton_raphson_contact_ncp() を AbstractProcess として管理し、
SolverStrategies によるStrategy合成を実現する。

依存追跡（§13.2 C8対応）:
- クラス変数 uses = [] は空（__init_subclass__ 通過用）
- インスタンス変数 _runtime_uses で動的依存追跡
- StrategySlot による型安全な Strategy 宣言（Phase 8-B/E）

Phase 8-E 更新:
- StrategySlot ディスクリプタを追加（PenaltyStrategy, FrictionStrategy, TimeIntegrationStrategy）
- _runtime_uses は StrategySlot + 従来方式の両方から構築（後方互換）
"""

from __future__ import annotations

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess
from xkep_cae.process.data import (
    SolverInputData,
    SolverResultData,
    SolverStrategies,
    default_strategies,
)
from xkep_cae.process.slots import StrategySlot, collect_strategy_types
from xkep_cae.process.strategies.protocols import (
    ContactForceStrategy,
    ContactGeometryStrategy,
    FrictionStrategy,
    PenaltyStrategy,
    TimeIntegrationStrategy,
)


class NCPContactSolverProcess(SolverProcess[SolverInputData, SolverResultData]):
    """NCP接触ソルバーのProcess Wrapper.

    newton_raphson_contact_ncp() を内部で呼び出し、
    SolverStrategies による Strategy 合成を実現する。

    Usage:
        solver = NCPContactSolverProcess()  # デフォルト構成
        solver = NCPContactSolverProcess(strategies=custom_strategies)
        result = solver.process(input_data)
    """

    meta = ProcessMeta(
        name="NCP接触ソルバー",
        module="solve",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
    )
    uses = []  # 静的usesは空（C8対策: 動的構築はインスタンス側で管理）

    # StrategySlot 宣言（Phase 8-B/E）
    penalty_slot = StrategySlot(PenaltyStrategy)
    friction_slot = StrategySlot(FrictionStrategy)
    time_integration_slot = StrategySlot(TimeIntegrationStrategy)
    contact_force_slot = StrategySlot(ContactForceStrategy, required=False)
    contact_geometry_slot = StrategySlot(ContactGeometryStrategy, required=False)

    def __init__(self, strategies: SolverStrategies | None = None) -> None:
        self.strategies = strategies or default_strategies()

        # StrategySlot に設定（Protocol 準拠検証付き）
        self.penalty_slot = self.strategies.penalty
        self.friction_slot = self.strategies.friction
        self.time_integration_slot = self.strategies.time_integration
        if self.strategies.contact_force is not None:
            self.contact_force_slot = self.strategies.contact_force
        if self.strategies.contact_geometry is not None:
            self.contact_geometry_slot = self.strategies.contact_geometry

    def get_instance_dependency_tree(self) -> dict:
        """インスタンスレベルの依存ツリー."""
        runtime = collect_strategy_types(self)
        return {
            "name": type(self).__name__,
            "module": self.meta.module,
            "uses": [{"name": dep.__name__, "module": "solve", "uses": []} for dep in runtime],
        }

    def process(self, input_data: SolverInputData) -> SolverResultData:
        """SolverInputData → newton_raphson_contact_ncp() → SolverResultData."""
        import time

        from xkep_cae.contact.solver_ncp import newton_raphson_contact_ncp

        strategies = input_data.strategies or self.strategies

        t0 = time.perf_counter()
        result = newton_raphson_contact_ncp(
            f_ext_total=input_data.boundary.f_ext_total,
            fixed_dofs=input_data.boundary.fixed_dofs,
            assemble_tangent=input_data.callbacks.assemble_tangent,
            assemble_internal_force=input_data.callbacks.assemble_internal_force,
            manager=input_data.contact.manager,
            node_coords_ref=input_data.mesh.node_coords,
            connectivity=input_data.mesh.connectivity,
            radii=input_data.mesh.radii,
            k_pen=input_data.contact.k_pen,
            use_friction=input_data.contact.use_friction,
            mu=input_data.contact.mu,
            contact_mode=input_data.contact.contact_mode,
            ul_assembler=input_data.callbacks.ul_assembler,
            strategies=strategies,
        )
        elapsed = time.perf_counter() - t0

        return SolverResultData(
            u=result.u,
            converged=result.converged,
            n_increments=result.n_increments,
            total_newton_iterations=result.total_newton_iterations,
            displacement_history=result.displacement_history,
            contact_force_history=result.contact_force_history,
            elapsed_seconds=elapsed,
            diagnostics=result.diagnostics,
        )
