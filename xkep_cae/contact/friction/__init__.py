"""Friction Strategy サブパッケージ.

Coulomb 摩擦の return mapping と摩擦力評価（status-222 で一本化）。
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
)

__all__ = [
    # Strategy
    "CoulombReturnMappingProcess",
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
