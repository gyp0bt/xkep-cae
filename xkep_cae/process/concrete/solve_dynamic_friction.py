"""NCPDynamicContactFrictionProcess — 動的摩擦接触ソルバー.

.. deprecated:: status-172
    ContactFrictionSolverProcess に統合。後方互換のため残存。

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
from xkep_cae.process.concrete.solve_contact_friction import (
    ContactFrictionSolverProcess,
)
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


class NCPDynamicContactFrictionProcess(SolverProcess[DynamicFrictionInputData, SolverResultData]):
    """動的摩擦接触ソルバー（Generalized-α + smooth penalty）.

    .. deprecated:: status-172
        ContactFrictionSolverProcess に統合。後方互換のため残存。
        内部で ContactFrictionSolverProcess に委譲する。

    Usage:
        solver = NCPDynamicContactFrictionProcess()
        result = solver.process(input_data)
    """

    meta = ProcessMeta(
        name="動的摩擦接触ソルバー",
        module="solve",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
        deprecated=True,
    )
    uses = [ContactFrictionSolverProcess]

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
            "uses": [{"name": dep.__name__, "module": "solve", "uses": []} for dep in runtime],
        }

    def process(self, input_data: DynamicFrictionInputData) -> SolverResultData:
        """DynamicFrictionInputData → ContactFrictionSolverProcess 委譲."""
        unified = input_data.to_unified()
        solver = ContactFrictionSolverProcess(strategies=self.strategies)
        return solver.process(unified)
