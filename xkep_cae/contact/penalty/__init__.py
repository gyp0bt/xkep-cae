"""Penalty Strategy サブパッケージ.

ペナルティ剛性の決定と法線接触力の評価。
"""

from xkep_cae.contact.penalty.law_normal import (
    ALNormalForceProcess,
    NormalForceInput,
    NormalForceResult,
    SmoothNormalForceInput,
    SmoothNormalForceProcess,
    VectorizedNormalForceResult,
    VectorizedSmoothInput,
)
from xkep_cae.contact.penalty.strategy import (
    AutoBeamEIPenalty,
    AutoEALPenalty,
    AutoSmoothingDeltaProcess,
    ConstantPenalty,
    ContinuationPenalty,
    DynamicPenaltyEstimateInput,
    DynamicPenaltyEstimateOutput,
    DynamicPenaltyEstimateProcess,
    PenaltyInput,
    PenaltyOutput,
)

__all__ = [
    # Strategy
    "AutoBeamEIPenalty",
    "AutoEALPenalty",
    "AutoSmoothingDeltaProcess",
    "ConstantPenalty",
    "ContinuationPenalty",
    "DynamicPenaltyEstimateInput",
    "DynamicPenaltyEstimateOutput",
    "DynamicPenaltyEstimateProcess",
    "PenaltyInput",
    "PenaltyOutput",
    # Law Normal
    "ALNormalForceProcess",
    "SmoothNormalForceProcess",
    "NormalForceInput",
    "NormalForceResult",
    "SmoothNormalForceInput",
    "VectorizedSmoothInput",
    "VectorizedNormalForceResult",
]
