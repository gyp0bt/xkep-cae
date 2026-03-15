"""NCPDynamicContactFrictionProcess — 動的摩擦接触ソルバー.

deprecated: ContactFrictionProcess に統合（status-172）。
後方互換のためエイリアスとして残す。
"""

from __future__ import annotations

import warnings

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess
from xkep_cae.process.data import (
    ContactFrictionInputData,
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


class NCPDynamicContactFrictionProcess(SolverProcess[DynamicFrictionInputData, SolverResultData]):
    """動的摩擦接触ソルバー（deprecated → ContactFrictionProcess）.

    .. deprecated:: status-172
        ContactFrictionProcess に統合。動的は ContactFrictionInputData に
        mass_matrix + dt_physical を指定で自動判定される。
    """

    meta = ProcessMeta(
        name="動的摩擦接触ソルバー",
        module="solve",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
        deprecated=True,
        deprecated_by="ContactFrictionProcess",
    )
    uses = []

    # StrategySlot 宣言
    penalty_slot = StrategySlot(PenaltyStrategy)
    friction_slot = StrategySlot(FrictionStrategy)
    time_integration_slot = StrategySlot(TimeIntegrationStrategy)
    contact_force_slot = StrategySlot(ContactForceStrategy, required=False)
    contact_geometry_slot = StrategySlot(ContactGeometryStrategy, required=False)

    def __init__(self, strategies: SolverStrategies | None = None) -> None:
        warnings.warn(
            "NCPDynamicContactFrictionProcess は deprecated です。"
            "ContactFrictionProcess を使用してください（status-172）。",
            DeprecationWarning,
            stacklevel=2,
        )
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
            "uses": [{"name": dep.__name__, "module": "solve", "uses": []} for dep in runtime],
        }

    def process(self, input_data: DynamicFrictionInputData) -> SolverResultData:
        """DynamicFrictionInputData → ContactFrictionProcess に委譲."""
        from xkep_cae.process.concrete.solve_contact_friction import (
            ContactFrictionProcess,
        )

        unified_input = ContactFrictionInputData(
            mesh=input_data.mesh,
            boundary=input_data.boundary,
            contact=input_data.contact,
            callbacks=input_data.callbacks,
            u0=input_data.u0,
            mass_matrix=input_data.mass_matrix,
            dt_physical=input_data.dt_physical,
            rho_inf=input_data.rho_inf,
            damping_matrix=input_data.damping_matrix,
            velocity=input_data.velocity,
            acceleration=input_data.acceleration,
        )
        proc = ContactFrictionProcess(strategies=self.strategies)
        return proc.process(unified_input)
