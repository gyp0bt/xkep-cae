"""梁–梁接触モジュール.

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md

モジュール構成:
- geometry: segment-to-segment 最近接点計算
- broadphase: AABB格子による候補ペア探索
- pair: 接触ペア・状態管理・幾何更新・Active-set
- law_normal: 法線接触（Augmented Lagrangian）
- assembly: 接触内力・接線の組み込み
- solver_hooks: 接触付き Newton-Raphson（Outer/Inner 分離）
- law_friction: Coulomb 摩擦（return mapping）  [未実装]
"""

from xkep_cae.contact.assembly import compute_contact_force, compute_contact_stiffness
from xkep_cae.contact.broadphase import broadphase_aabb, compute_segment_aabb
from xkep_cae.contact.law_normal import (
    auto_penalty_stiffness,
    evaluate_normal_force,
    initialize_penalty_stiffness,
    normal_force_linearization,
    update_al_multiplier,
)
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)
from xkep_cae.contact.solver_hooks import (
    ContactSolveResult,
    newton_raphson_with_contact,
)

__all__ = [
    "ContactConfig",
    "ContactManager",
    "ContactPair",
    "ContactSolveResult",
    "ContactState",
    "ContactStatus",
    "auto_penalty_stiffness",
    "broadphase_aabb",
    "compute_contact_force",
    "compute_contact_stiffness",
    "compute_segment_aabb",
    "evaluate_normal_force",
    "initialize_penalty_stiffness",
    "newton_raphson_with_contact",
    "normal_force_linearization",
    "update_al_multiplier",
]
