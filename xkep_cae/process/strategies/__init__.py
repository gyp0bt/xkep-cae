"""Strategy Protocol 定義 + 具象実装.

各 strategy サブパッケージは新規実装。
"""

from xkep_cae.process.strategies.protocols import (
    CoatingStrategy,
    ContactForceStrategy,
    ContactGeometryStrategy,
    FrictionStrategy,
    LinearSolverStrategy,
    PenaltyStrategy,
    TimeIntegrationStrategy,
)

__all__ = [
    "CoatingStrategy",
    "ContactForceStrategy",
    "ContactGeometryStrategy",
    "FrictionStrategy",
    "LinearSolverStrategy",
    "PenaltyStrategy",
    "TimeIntegrationStrategy",
]
