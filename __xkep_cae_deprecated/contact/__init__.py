"""梁–梁接触モジュール.

モジュール構成:
- geometry: segment-to-segment 最近接点計算
- broadphase: AABB格子による候補ペア探索
- pair: 接触ペア・状態管理・幾何更新・Active-set
- law_normal: 法線接触力・ペナルティ剛性
- law_friction: Coulomb 摩擦（return mapping）
- line_contact: line-to-line Gauss 積分（Phase C6-L1）
- ncp: NCP 関数（Fischer-Burmeister, min）（Phase C6-L3）
- solver_ncp: Semi-smooth Newton ソルバー（Phase C6-L3）
- assembly: 接触内力・接線の組み込み
- graph: 接触グラフ表現・可視化
- mortar: Mortar 離散化（Phase C6-L5）
"""

from __xkep_cae_deprecated.contact.assembly import compute_contact_force, compute_contact_stiffness
from __xkep_cae_deprecated.contact.broadphase import broadphase_aabb, compute_segment_aabb
from __xkep_cae_deprecated.contact.diagnostics import (
    ConvergenceDiagnosticsOutput,
    NCPSolveResult,
    NCPSolverInput,
)
from __xkep_cae_deprecated.contact.geometry import compute_st_jacobian
from __xkep_cae_deprecated.contact.graph import (
    ContactEdge,
    ContactGraph,
    ContactGraphHistory,
    compute_hysteresis_area,
    plot_hysteresis_curve,
    plot_statistics_dashboard,
    snapshot_contact_graph,
)
from __xkep_cae_deprecated.contact.law_friction import (
    compute_mu_effective,
    compute_tangential_displacement,
    friction_return_mapping,
    friction_tangent_2x2,
    rotate_friction_history,
)
from __xkep_cae_deprecated.contact.law_normal import (
    auto_beam_penalty_stiffness,
    auto_penalty_stiffness,
    evaluate_normal_force,
    initialize_penalty_stiffness,
    normal_force_linearization,
)
from __xkep_cae_deprecated.contact.line_contact import (
    auto_select_n_gauss,
    compute_line_contact_force_local,
    compute_line_contact_gap_at_gp,
    compute_line_contact_stiffness_local,
    compute_line_friction_force_local,
    compute_line_friction_stiffness_local,
    compute_t_jacobian_at_gp,
    gauss_legendre_01,
    project_point_to_segment,
)
from __xkep_cae_deprecated.contact.mortar import (
    build_mortar_system,
    compute_mortar_contact_force,
    compute_mortar_p_n,
    identify_mortar_nodes,
)
from __xkep_cae_deprecated.contact.ncp import (
    build_augmented_residual,
    compute_gap_jacobian_wrt_u,
    evaluate_ncp_jacobian,
    evaluate_ncp_residual,
    ncp_fischer_burmeister,
    ncp_min,
)
from __xkep_cae_deprecated.contact.pair import (
    ContactConfig,
    ContactManager,
    ContactPair,
    ContactState,
    ContactStatus,
)
from __xkep_cae_deprecated.contact.sheath_contact import (
    SheathContactConfig,
    SheathContactManager,
    SheathContactPoint,
    build_sheath_contact_manager,
    evaluate_sheath_contact,
)
from __xkep_cae_deprecated.contact.solver_ncp import (
    _solve_saddle_point_direct,
    _solve_saddle_point_gmres,
    newton_raphson_contact_ncp,
)
from __xkep_cae_deprecated.contact.utils import deformed_coords, ncp_line_search

__all__ = [
    "ConvergenceDiagnosticsOutput",
    "ContactConfig",
    "ContactEdge",
    "ContactGraph",
    "ContactGraphHistory",
    "ContactManager",
    "ContactPair",
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
    "broadphase_aabb",
    "compute_contact_force",
    "compute_contact_stiffness",
    "compute_line_contact_force_local",
    "compute_line_contact_gap_at_gp",
    "compute_line_contact_stiffness_local",
    "compute_line_friction_force_local",
    "compute_line_friction_stiffness_local",
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
    "normal_force_linearization",
    "plot_hysteresis_curve",
    "project_point_to_segment",
    "plot_statistics_dashboard",
    "rotate_friction_history",
    "snapshot_contact_graph",
    "NCPSolveResult",
    "NCPSolverInput",
    "build_augmented_residual",
    "compute_gap_jacobian_wrt_u",
    "evaluate_ncp_jacobian",
    "evaluate_ncp_residual",
    "ncp_fischer_burmeister",
    "ncp_min",
    "newton_raphson_contact_ncp",
    "deformed_coords",
    "ncp_line_search",
    "_solve_saddle_point_direct",
    "_solve_saddle_point_gmres",
    "build_mortar_system",
    "compute_mortar_contact_force",
    "compute_mortar_p_n",
    "identify_mortar_nodes",
]
