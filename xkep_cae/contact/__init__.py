"""梁–梁接触モジュール.

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md

モジュール構成:
- geometry: segment-to-segment 最近接点計算
- broadphase: AABB格子による候補ペア探索
- pair: 接触ペア・状態管理・幾何更新・Active-set
- law_normal: 法線接触（Augmented Lagrangian）
- law_friction: Coulomb 摩擦（return mapping）
- line_search: merit function + backtracking line search
- assembly: 接触内力・接線の組み込み
- solver_hooks: 接触付き Newton-Raphson（Outer/Inner 分離）
- graph: 接触グラフ表現・可視化
"""

from xkep_cae.contact.assembly import compute_contact_force, compute_contact_stiffness
from xkep_cae.contact.broadphase import broadphase_aabb, compute_segment_aabb
from xkep_cae.contact.graph import (
    ContactEdge,
    ContactGraph,
    ContactGraphHistory,
    compute_hysteresis_area,
    plot_hysteresis_curve,
    plot_statistics_dashboard,
    snapshot_contact_graph,
)
from xkep_cae.contact.law_friction import (
    compute_mu_effective,
    compute_tangential_displacement,
    friction_return_mapping,
    friction_tangent_2x2,
    rotate_friction_history,
)
from xkep_cae.contact.law_normal import (
    auto_beam_penalty_stiffness,
    auto_penalty_stiffness,
    evaluate_normal_force,
    initialize_penalty_stiffness,
    normal_force_linearization,
    update_al_multiplier,
)
from xkep_cae.contact.line_search import backtracking_line_search, merit_function
from xkep_cae.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)
from xkep_cae.contact.sheath_contact import (
    SheathContactConfig,
    SheathContactManager,
    SheathContactPoint,
    build_sheath_contact_manager,
    evaluate_sheath_contact,
)
from xkep_cae.contact.solver_hooks import (
    ContactSolveResult,
    CyclicContactResult,
    newton_raphson_block_contact,
    newton_raphson_with_contact,
    run_contact_cyclic,
)

__all__ = [
    "ContactConfig",
    "ContactEdge",
    "ContactGraph",
    "ContactGraphHistory",
    "ContactManager",
    "ContactPair",
    "ContactSolveResult",
    "CyclicContactResult",
    "ContactState",
    "ContactStatus",
    "SheathContactConfig",
    "SheathContactManager",
    "SheathContactPoint",
    "build_sheath_contact_manager",
    "evaluate_sheath_contact",
    "auto_beam_penalty_stiffness",
    "auto_penalty_stiffness",
    "backtracking_line_search",
    "broadphase_aabb",
    "compute_contact_force",
    "compute_contact_stiffness",
    "compute_hysteresis_area",
    "compute_mu_effective",
    "compute_segment_aabb",
    "compute_tangential_displacement",
    "evaluate_normal_force",
    "friction_return_mapping",
    "friction_tangent_2x2",
    "initialize_penalty_stiffness",
    "merit_function",
    "newton_raphson_block_contact",
    "newton_raphson_with_contact",
    "run_contact_cyclic",
    "normal_force_linearization",
    "plot_hysteresis_curve",
    "plot_statistics_dashboard",
    "rotate_friction_history",
    "snapshot_contact_graph",
    "update_al_multiplier",
]
