"""Penalty Strategy サブパッケージ.

ペナルティ剛性の決定と法線接触力の評価。
"""

from xkep_cae.process.strategies.penalty.law_normal import (
    ALNormalForceProcess,
    NormalForceInput,
    NormalForceResult,
    SmoothNormalForceInput,
    SmoothNormalForceProcess,
    VectorizedNormalForceResult,
    VectorizedSmoothInput,
    auto_beam_penalty_stiffness,
    evaluate_al_normal_force,
    evaluate_smooth_normal_force,
    evaluate_smooth_normal_force_vectorized,
    softplus,
)
from xkep_cae.process.strategies.penalty.strategy import (
    AutoBeamEIPenalty,
    AutoEALPenalty,
    ContinuationPenalty,
    PenaltyInput,
    PenaltyOutput,
)

__all__ = [
    # Strategy
    "AutoBeamEIPenalty",
    "AutoEALPenalty",
    "ContinuationPenalty",
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
    "evaluate_al_normal_force",
    "evaluate_smooth_normal_force",
    "evaluate_smooth_normal_force_vectorized",
    "auto_beam_penalty_stiffness",
    "softplus",
]
