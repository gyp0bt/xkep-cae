"""梁–梁接触モジュール.

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md

モジュール構成:
- geometry: segment-to-segment 最近接点計算
- broadphase: AABB格子による候補ペア探索
- pair: 接触ペア・状態管理・幾何更新・Active-set
- law_normal: 法線接触（Augmented Lagrangian）  [未実装]
- law_friction: Coulomb 摩擦（return mapping）  [未実装]
- assembly: 接触内力・接線の組み込み  [未実装]
- solver_hooks: solver.py への接触寄与注入  [未実装]
"""

from xkep_cae.contact.broadphase import broadphase_aabb, compute_segment_aabb
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)

__all__ = [
    "ContactConfig",
    "ContactManager",
    "ContactPair",
    "ContactState",
    "ContactStatus",
    "broadphase_aabb",
    "compute_segment_aabb",
]
