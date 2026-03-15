"""Friction Strategy サブパッケージ.

Coulomb 摩擦の return mapping と摩擦力評価。
"""

from xkep_cae.contact.friction.law_friction import (
    FrictionTangentProcess,
    MuRampInput,
    ReturnMappingInput,
    ReturnMappingProcess,
    ReturnMappingResult,
    RotateHistoryInput,
    TangentInput,
    TangentResult,
)
from xkep_cae.contact.friction.strategy import (
    CoulombReturnMappingProcess,
    FrictionInput,
    FrictionOutput,
    NoFrictionProcess,
    SmoothPenaltyFrictionProcess,
)

__all__ = [
    # Strategy
    "NoFrictionProcess",
    "CoulombReturnMappingProcess",
    "SmoothPenaltyFrictionProcess",
    "FrictionInput",
    "FrictionOutput",
    # Law Friction
    "ReturnMappingProcess",
    "FrictionTangentProcess",
    "ReturnMappingInput",
    "ReturnMappingResult",
    "TangentInput",
    "TangentResult",
    "RotateHistoryInput",
    "MuRampInput",
]
