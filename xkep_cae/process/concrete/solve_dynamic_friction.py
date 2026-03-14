"""NCPDynamicContactFrictionProcess — 動的摩擦接触ソルバー.

Smooth penalty + Uzawa + Coulomb 摩擦 + Generalized-α 動的解析。
Strategy 経由で全ての接触力・摩擦力・時間積分を評価する。

固定構成（王道構成）:
- contact_mode = "smooth_penalty"
- use_friction = True
- line_contact = True
- adaptive_timestepping = True
- 時間積分 = Generalized-α
"""

from __future__ import annotations

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess
from xkep_cae.process.data import (
    DynamicFrictionInputData,
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


class NCPDynamicContactFrictionProcess(
    SolverProcess[DynamicFrictionInputData, SolverResultData]
):
    """動的摩擦接触ソルバー（Generalized-α + smooth penalty）.

    solve_smooth_penalty_friction() を Strategy 経由で呼び出し、
    Generalized-α 時間積分を強制する。

    Usage:
        solver = NCPDynamicContactFrictionProcess()
        result = solver.process(input_data)
    """

    meta = ProcessMeta(
        name="動的摩擦接触ソルバー",
        module="solve",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
    )
    uses = []

    # StrategySlot 宣言
    penalty_slot = StrategySlot(PenaltyStrategy)
    friction_slot = StrategySlot(FrictionStrategy)
    time_integration_slot = StrategySlot(TimeIntegrationStrategy)
    contact_force_slot = StrategySlot(ContactForceStrategy, required=False)
    contact_geometry_slot = StrategySlot(ContactGeometryStrategy, required=False)

    def __init__(self, strategies: SolverStrategies | None = None) -> None:
        self.strategies = strategies or default_strategies()

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
            "uses": [
                {"name": dep.__name__, "module": "solve", "uses": []} for dep in runtime
            ],
        }

    def process(self, input_data: DynamicFrictionInputData) -> SolverResultData:
        """DynamicFrictionInputData → solve_smooth_penalty_friction() → SolverResultData."""
        import time

        from xkep_cae.contact.solver_smooth_penalty import (
            solve_smooth_penalty_friction,
        )

        # Generalized-α Strategy を動的パラメータで構築
        ndof = len(input_data.boundary.f_ext_total)
        strategies = default_strategies(
            ndof=ndof,
            mass_matrix=input_data.mass_matrix,
            damping_matrix=input_data.damping_matrix,
            dt_physical=input_data.dt_physical,
            rho_inf=input_data.rho_inf,
            velocity=input_data.velocity,
            acceleration=input_data.acceleration,
            k_pen=input_data.contact.k_pen,
            use_friction=True,
            mu=input_data.contact.mu or 0.15,
            contact_mode="smooth_penalty",
            line_contact=True,
        )

        t0 = time.perf_counter()
        result = solve_smooth_penalty_friction(
            f_ext_total=input_data.boundary.f_ext_total,
            fixed_dofs=input_data.boundary.fixed_dofs,
            assemble_tangent=input_data.callbacks.assemble_tangent,
            assemble_internal_force=input_data.callbacks.assemble_internal_force,
            manager=input_data.contact.manager,
            node_coords_ref=input_data.mesh.node_coords,
            connectivity=input_data.mesh.connectivity,
            radii=input_data.mesh.radii,
            strategies=strategies,
            k_pen=input_data.contact.k_pen,
            mu=input_data.contact.mu,
            u0=input_data.u0,
            f_ext_base=input_data.boundary.f_ext_base,
            prescribed_dofs=input_data.boundary.prescribed_dofs,
            prescribed_values=input_data.boundary.prescribed_values,
            ul_assembler=input_data.callbacks.ul_assembler,
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
