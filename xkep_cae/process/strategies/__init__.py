"""Strategy Protocol と互換性マトリクス.

設計仕様: xkep_cae/process/process-architecture.md §2
"""

from xkep_cae.process.strategies.protocols import (
    ContactForceStrategy,
    ContactGeometryStrategy,
    FrictionStrategy,
    PenaltyStrategy,
    TimeIntegrationStrategy,
)

__all__ = [
    "ContactForceStrategy",
    "ContactGeometryStrategy",
    "FrictionStrategy",
    "PenaltyStrategy",
    "TimeIntegrationStrategy",
]
