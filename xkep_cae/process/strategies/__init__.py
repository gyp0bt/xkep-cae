"""Strategy Protocol 定義 + 具象実装.

protocols.py は暫定的に deprecated から re-export。
各 strategy サブパッケージは新規実装。
"""

from xkep_cae_deprecated.process.strategies.protocols import (  # noqa: F401
    CoatingStrategy,
    ContactForceStrategy,
    ContactGeometryStrategy,
    FrictionStrategy,
    LinearSolverStrategy,
    PenaltyStrategy,
    TimeIntegrationStrategy,
)
