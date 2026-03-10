"""Strategy Protocol + 具象実装.

設計仕様: xkep_cae/process/process-architecture.md §2

Phase 1: Protocol 定義 (protocols.py)
Phase 2: 具象 Strategy 実装 (contact_force.py, friction.py, etc.)
"""

from xkep_cae.process.strategies.contact_force import (
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
)
from xkep_cae.process.strategies.contact_geometry import (
    LineToLineGaussProcess,
    MortarSegmentProcess,
    PointToPointProcess,
)
from xkep_cae.process.strategies.friction import (
    CoulombReturnMappingProcess,
    NoFrictionProcess,
    SmoothPenaltyFrictionProcess,
)
from xkep_cae.process.strategies.penalty import (
    AutoBeamEIProcess,
    AutoEALProcess,
    ContinuationPenaltyProcess,
    ManualPenaltyProcess,
)
from xkep_cae.process.strategies.protocols import (
    ContactForceStrategy,
    ContactGeometryStrategy,
    FrictionStrategy,
    PenaltyStrategy,
    TimeIntegrationStrategy,
)
from xkep_cae.process.strategies.time_integration import (
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
)

__all__ = [
    # Protocols
    "ContactForceStrategy",
    "ContactGeometryStrategy",
    "FrictionStrategy",
    "PenaltyStrategy",
    "TimeIntegrationStrategy",
    # ContactForce 具象
    "NCPContactForceProcess",
    "SmoothPenaltyContactForceProcess",
    # Friction 具象
    "NoFrictionProcess",
    "CoulombReturnMappingProcess",
    "SmoothPenaltyFrictionProcess",
    # TimeIntegration 具象
    "QuasiStaticProcess",
    "GeneralizedAlphaProcess",
    # ContactGeometry 具象
    "PointToPointProcess",
    "LineToLineGaussProcess",
    "MortarSegmentProcess",
    # Penalty 具象
    "AutoBeamEIProcess",
    "AutoEALProcess",
    "ManualPenaltyProcess",
    "ContinuationPenaltyProcess",
]
