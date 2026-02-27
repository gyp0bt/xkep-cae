"""梁–梁接触モジュール.

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md

モジュール構成:
- geometry: segment-to-segment 最近接点計算
- broadphase: AABB格子による候補ペア探索
- pair: 接触ペア・状態管理・幾何更新・Active-set
- law_normal: 法線接触（Augmented Lagrangian）
- law_friction: Coulomb 摩擦（return mapping）
- line_search: merit function + backtracking line search
- line_contact: line-to-line Gauss 積分（Phase C6-L1）
- ncp: NCP 関数（Fischer-Burmeister, min）（Phase C6-L3）
- solver_ncp: Semi-smooth Newton ソルバー（Phase C6-L3）
- assembly: 接触内力・接線の組み込み
- solver_hooks: 接触付き Newton-Raphson（Outer/Inner 分離）
- graph: 接触グラフ表現・可視化
"""

from xkep_cae.contact.assembly import compute_contact_force, compute_contact_stiffness
from xkep_cae.contact.broadphase import broadphase_aabb, compute_segment_aabb
from xkep_cae.contact.geometry import compute_st_jacobian
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
from xkep_cae.contact.line_contact import (
    auto_select_n_gauss,
    compute_line_contact_force_local,
    compute_line_contact_gap_at_gp,
    compute_line_contact_stiffness_local,
    compute_t_jacobian_at_gp,
    gauss_legendre_01,
    project_point_to_segment,
)
from xkep_cae.contact.line_search import backtracking_line_search, merit_function
from xkep_cae.contact.ncp import (
    build_augmented_residual,
    compute_gap_jacobian_wrt_u,
    evaluate_ncp_jacobian,
    evaluate_ncp_residual,
    ncp_fischer_burmeister,
    ncp_min,
)
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
from xkep_cae.contact.solver_ncp import (
    NCPSolveResult,
    newton_raphson_contact_ncp,
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
    "auto_select_n_gauss",
    "backtracking_line_search",
    "broadphase_aabb",
    "compute_contact_force",
    "compute_contact_stiffness",
    "compute_line_contact_force_local",
    "compute_line_contact_gap_at_gp",
    "compute_line_contact_stiffness_local",
    "compute_hysteresis_area",
    "compute_mu_effective",
    "compute_segment_aabb",
    "compute_st_jacobian",
    "compute_t_jacobian_at_gp",
    "compute_tangential_displacement",
    "evaluate_normal_force",
    "friction_return_mapping",
    "gauss_legendre_01",
    "friction_tangent_2x2",
    "initialize_penalty_stiffness",
    "merit_function",
    "newton_raphson_block_contact",
    "newton_raphson_with_contact",
    "run_contact_cyclic",
    "normal_force_linearization",
    "plot_hysteresis_curve",
    "project_point_to_segment",
    "plot_statistics_dashboard",
    "rotate_friction_history",
    "snapshot_contact_graph",
    "update_al_multiplier",
    "NCPSolveResult",
    "build_augmented_residual",
    "compute_gap_jacobian_wrt_u",
    "evaluate_ncp_jacobian",
    "evaluate_ncp_residual",
    "ncp_fischer_burmeister",
    "ncp_min",
    "newton_raphson_contact_ncp",
]
