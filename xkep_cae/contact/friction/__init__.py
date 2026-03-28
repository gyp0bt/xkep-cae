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
    FrictionGeometricStiffnessInput,
    FrictionGeometricStiffnessOutput,
    FrictionGeometricStiffnessProcess,
    FrictionInput,
    FrictionOutput,
    FrictionStStiffnessInput,
    FrictionStStiffnessOutput,
    FrictionStStiffnessProcess,
    FrictionTangentStiffnessInput,
    FrictionTangentStiffnessOutput,
    FrictionTangentStiffnessProcess,
)

__all__ = [
    # Strategy
    "CoulombReturnMappingProcess",
    "FrictionInput",
    "FrictionOutput",
    # B4: 摩擦接線剛性（材料項）
    "FrictionTangentStiffnessProcess",
    "FrictionTangentStiffnessInput",
    "FrictionTangentStiffnessOutput",
    # B2: 摩擦接線幾何剛性
    "FrictionGeometricStiffnessProcess",
    "FrictionGeometricStiffnessInput",
    "FrictionGeometricStiffnessOutput",
    # B3: 摩擦K_st
    "FrictionStStiffnessProcess",
    "FrictionStStiffnessInput",
    "FrictionStStiffnessOutput",
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
