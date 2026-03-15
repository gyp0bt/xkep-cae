"""NCPQuasiStaticContactFrictionProcess — 準静的摩擦接触ソルバー.

deprecated: ContactFrictionProcess に統合（status-172）。
後方互換のためエイリアスとして残す。
"""

from __future__ import annotations

import warnings

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess
from xkep_cae.process.data import (
    ContactFrictionInputData,
    QuasiStaticFrictionInputData,
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


class NCPQuasiStaticContactFrictionProcess(
    SolverProcess[QuasiStaticFrictionInputData, SolverResultData]
):
    """準静的摩擦接触ソルバー（deprecated → ContactFrictionProcess）.

    .. deprecated:: status-172
        ContactFrictionProcess に統合。準静的は ContactFrictionInputData の
        動的パラメータ未指定で自動判定される。
    """

    meta = ProcessMeta(
        name="準静的摩擦接触ソルバー",
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
            "NCPQuasiStaticContactFrictionProcess は deprecated です。"
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

    def process(self, input_data: QuasiStaticFrictionInputData) -> SolverResultData:
        """QuasiStaticFrictionInputData → ContactFrictionProcess に委譲."""
        from xkep_cae.process.concrete.solve_contact_friction import (
            ContactFrictionProcess,
        )

        unified_input = ContactFrictionInputData(
            mesh=input_data.mesh,
            boundary=input_data.boundary,
            contact=input_data.contact,
            callbacks=input_data.callbacks,
            u0=input_data.u0,
        )
        proc = ContactFrictionProcess(strategies=self.strategies)
        return proc.process(unified_input)
