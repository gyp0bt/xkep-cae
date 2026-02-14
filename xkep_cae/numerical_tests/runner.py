"""数値試験フレームワーク — 静的試験ランナー.

3点曲げ・4点曲げ・引張・ねん回の各試験を
統一インタフェースで実行する。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.bc import apply_dirichlet
from xkep_cae.numerical_tests.core import (
    NumericalTestConfig,
    StaticTestResult,
    _build_section_props,
    analytical_bend3p,
    analytical_bend4p,
    analytical_tensile,
    analytical_torsion,
    assess_friction_effect,
    generate_beam_mesh_2d,
    generate_beam_mesh_3d,
)
from xkep_cae.solver import solve_displacement


# ---------------------------------------------------------------------------
# 2D/3D アセンブリ（低レベル直接アセンブリ）
# ---------------------------------------------------------------------------
def _assemble_2d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    ke_func,
) -> tuple[np.ndarray, int]:
    """2D梁の全体剛性行列をアセンブルする（3DOF/節点）."""
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        Ke = ke_func(coords)
        edofs = np.array([3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                          3 * n2, 3 * n2 + 1, 3 * n2 + 2], dtype=int)
        for ii in range(6):
            for jj in range(6):
                K[edofs[ii], edofs[jj]] += Ke[ii, jj]
    return K, ndof


def _assemble_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    ke_func,
) -> tuple[np.ndarray, int]:
    """3D梁の全体剛性行列をアセンブルする（6DOF/節点）."""
    n_nodes = len(nodes)
    ndof = 6 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        Ke = ke_func(coords)
        edofs = np.empty(12, dtype=int)
        for i, n in enumerate([n1, n2]):
            for d in range(6):
                edofs[6 * i + d] = 6 * n + d
        for ii in range(12):
            for jj in range(12):
                K[edofs[ii], edofs[jj]] += Ke[ii, jj]
    return K, ndof


# ---------------------------------------------------------------------------
# 単純支持の境界条件生成（3点/4点曲げ共通）
# ---------------------------------------------------------------------------
def _build_simply_supported_bc_3p(
    n_elems: int,
    is_3d: bool,
    support_condition: str,
) -> list[int]:
    """3点/4点曲げ試験用の単純支持境界条件を生成する.

    3D梁では以下を拘束:
    - 支点A: ux (剛体移動防止), uy, uz, θx (ねじり剛体回転防止)
    - 支点B: uy, uz, (ux if pin)

    2D梁:
    - 支点A: ux (剛体移動防止), uy
    - 支点B: uy, (ux if pin)
    """
    fixed_dofs: list[int] = []
    if is_3d:
        # 支点A
        fixed_dofs.append(6 * 0 + 0)  # ux
        fixed_dofs.append(6 * 0 + 1)  # uy
        fixed_dofs.append(6 * 0 + 2)  # uz
        fixed_dofs.append(6 * 0 + 3)  # θx（ねじり剛体回転防止）
        # 支点B
        fixed_dofs.append(6 * n_elems + 1)  # uy
        fixed_dofs.append(6 * n_elems + 2)  # uz
        if support_condition == "pin":
            fixed_dofs.append(6 * n_elems + 0)  # ux
    else:
        # 支点A
        fixed_dofs.append(3 * 0 + 0)  # ux
        fixed_dofs.append(3 * 0 + 1)  # uy
        # 支点B
        fixed_dofs.append(3 * n_elems + 1)  # uy
        if support_condition == "pin":
            fixed_dofs.append(3 * n_elems + 0)  # ux
    return fixed_dofs


# ---------------------------------------------------------------------------
# 3点曲げ試験
# ---------------------------------------------------------------------------
def _run_bend3p(cfg: NumericalTestConfig) -> StaticTestResult:
    """3点曲げ試験を実行する.

    境界条件:
      - 支点A（節点0）: uy=0, (ux=0 if pin / ux自由 if roller), θ自由
      - 支点B（節点n_elems）: uy=0, ux自由, θ自由
      - 荷重: 中央節点に P（y方向負）

    注: n_elemsは偶数であること（中央節点が存在するため）。
    """
    sec = _build_section_props(cfg.section_shape, cfg.section_params,
                               cfg.beam_type, cfg.nu)
    P = cfg.load_value
    L = cfg.length
    n_elems = cfg.n_elems
    if n_elems % 2 != 0:
        n_elems += 1  # 中央節点が必要

    is_3d = cfg.beam_type in ("timo3d", "cosserat")

    if is_3d:
        nodes, conn = generate_beam_mesh_3d(n_elems, L)
    else:
        nodes, conn = generate_beam_mesh_2d(n_elems, L)

    # 剛性行列
    ke_func = _get_ke_func(cfg, sec)
    if is_3d:
        K, ndof = _assemble_3d(nodes, conn, ke_func)
    else:
        K, ndof = _assemble_2d(nodes, conn, ke_func)

    # 荷重ベクトル
    f = np.zeros(ndof)
    mid_node = n_elems // 2
    if is_3d:
        f[6 * mid_node + 1] = -abs(P)  # y方向下向き
    else:
        f[3 * mid_node + 1] = -abs(P)

    # 境界条件
    fixed_dofs = _build_simply_supported_bc_3p(
        n_elems, is_3d, cfg.support_condition,
    )
    fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)

    # ソルブ
    K_sp = sp.csr_matrix(K)
    Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
    u, info = solve_displacement(Kbc, fbc, show_progress=False)

    # 解析解
    use_timo = cfg.beam_type in ("timo2d", "timo3d")
    ana = analytical_bend3p(
        abs(P), L, cfg.E, sec["I"],
        kappa=sec["kappa"] if use_timo else None,
        G=cfg.G if use_timo else None,
        A=sec["A"] if use_timo else None,
    )

    if is_3d:
        delta_fem = abs(u[6 * mid_node + 1])
    else:
        delta_fem = abs(u[3 * mid_node + 1])

    delta_ana = ana["delta_mid"]
    rel_err = abs(delta_fem - delta_ana) / abs(delta_ana) if delta_ana != 0 else None

    # 断面力ポスト処理
    element_forces = _compute_section_forces(cfg, sec, nodes, conn, u)

    # 摩擦影響評価
    friction_msg = assess_friction_effect(
        cfg.name, cfg.span_ratio, cfg.support_condition
    )

    return StaticTestResult(
        config=cfg,
        node_coords=nodes,
        displacement=u,
        element_forces=element_forces,
        displacement_max=delta_fem,
        displacement_analytical=delta_ana,
        relative_error=rel_err,
        max_bending_stress=ana["M_max"] * sec.get("r_max_z", 0) / sec["I"] if sec["I"] > 0 else 0,
        max_shear_stress=_estimate_max_shear(ana["V_max"], sec),
        friction_warning=friction_msg,
        solver_info=info,
    )


# ---------------------------------------------------------------------------
# 4点曲げ試験
# ---------------------------------------------------------------------------
def _run_bend4p(cfg: NumericalTestConfig) -> StaticTestResult:
    """4点曲げ試験を実行する.

    荷重スパン a の位置に荷重を配置。
    n_elemsは荷重点が節点に乗るよう調整する。
    """
    sec = _build_section_props(cfg.section_shape, cfg.section_params,
                               cfg.beam_type, cfg.nu)
    P = cfg.load_value
    L = cfg.length
    a = cfg.load_span
    n_elems = cfg.n_elems

    is_3d = cfg.beam_type in ("timo3d", "cosserat")

    if is_3d:
        nodes, conn = generate_beam_mesh_3d(n_elems, L)
    else:
        nodes, conn = generate_beam_mesh_2d(n_elems, L)

    # 荷重点に最も近い節点を検索（対称性を保証）
    x_coords = nodes[:, 0]
    load_node_left = int(np.argmin(np.abs(x_coords - a)))
    # 対称性のため右側荷重点は L - x_left を使用
    a_actual = x_coords[load_node_left]
    load_node_right = int(np.argmin(np.abs(x_coords - (L - a_actual))))
    mid_node = int(np.argmin(np.abs(x_coords - L / 2.0)))

    # 剛性行列
    ke_func = _get_ke_func(cfg, sec)
    if is_3d:
        K, ndof = _assemble_3d(nodes, conn, ke_func)
    else:
        K, ndof = _assemble_2d(nodes, conn, ke_func)

    # 荷重ベクトル
    f = np.zeros(ndof)
    if is_3d:
        f[6 * load_node_left + 1] = -abs(P)
        f[6 * load_node_right + 1] = -abs(P)
    else:
        f[3 * load_node_left + 1] = -abs(P)
        f[3 * load_node_right + 1] = -abs(P)

    # 境界条件（3点曲げと同じ支持条件）
    fixed_dofs = _build_simply_supported_bc_3p(
        n_elems, is_3d, cfg.support_condition,
    )
    fixed_dofs = np.array(sorted(set(fixed_dofs)), dtype=int)

    K_sp = sp.csr_matrix(K)
    Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
    u, info = solve_displacement(Kbc, fbc, show_progress=False)

    use_timo = cfg.beam_type in ("timo2d", "timo3d")
    ana = analytical_bend4p(
        abs(P), L, a_actual, cfg.E, sec["I"],
        kappa=sec["kappa"] if use_timo else None,
        G=cfg.G if use_timo else None,
        A=sec["A"] if use_timo else None,
    )

    if is_3d:
        delta_fem = abs(u[6 * mid_node + 1])
    else:
        delta_fem = abs(u[3 * mid_node + 1])

    delta_ana = ana["delta_mid"]
    rel_err = abs(delta_fem - delta_ana) / abs(delta_ana) if delta_ana != 0 else None

    element_forces = _compute_section_forces(cfg, sec, nodes, conn, u)
    friction_msg = assess_friction_effect(
        cfg.name, cfg.span_ratio, cfg.support_condition
    )

    return StaticTestResult(
        config=cfg,
        node_coords=nodes,
        displacement=u,
        element_forces=element_forces,
        displacement_max=delta_fem,
        displacement_analytical=delta_ana,
        relative_error=rel_err,
        max_bending_stress=ana["M_max"] * sec.get("r_max_z", 0) / sec["I"] if sec["I"] > 0 else 0,
        max_shear_stress=_estimate_max_shear(ana["V_max"], sec),
        friction_warning=friction_msg,
        solver_info=info,
    )


# ---------------------------------------------------------------------------
# 引張試験
# ---------------------------------------------------------------------------
def _run_tensile(cfg: NumericalTestConfig) -> StaticTestResult:
    """引張試験を実行する.

    一端固定、他端軸方向荷重。
    """
    sec = _build_section_props(cfg.section_shape, cfg.section_params,
                               cfg.beam_type, cfg.nu)
    P = cfg.load_value
    L = cfg.length
    n_elems = cfg.n_elems

    is_3d = cfg.beam_type in ("timo3d", "cosserat")

    if is_3d:
        nodes, conn = generate_beam_mesh_3d(n_elems, L)
    else:
        nodes, conn = generate_beam_mesh_2d(n_elems, L)

    ke_func = _get_ke_func(cfg, sec)
    if is_3d:
        K, ndof = _assemble_3d(nodes, conn, ke_func)
    else:
        K, ndof = _assemble_2d(nodes, conn, ke_func)

    f = np.zeros(ndof)
    if is_3d:
        f[6 * n_elems + 0] = P  # 先端x方向
    else:
        f[3 * n_elems + 0] = P

    # 固定端（全DOF拘束）
    if is_3d:
        fixed_dofs = np.arange(6, dtype=int)  # 節点0の6DOF
    else:
        fixed_dofs = np.arange(3, dtype=int)  # 節点0の3DOF

    K_sp = sp.csr_matrix(K)
    Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
    u, info = solve_displacement(Kbc, fbc, show_progress=False)

    ana = analytical_tensile(P, L, cfg.E, sec["A"])

    if is_3d:
        delta_fem = u[6 * n_elems + 0]
    else:
        delta_fem = u[3 * n_elems + 0]

    delta_ana = ana["delta"]
    rel_err = abs(delta_fem - delta_ana) / abs(delta_ana) if delta_ana != 0 else None

    element_forces = _compute_section_forces(cfg, sec, nodes, conn, u)

    return StaticTestResult(
        config=cfg,
        node_coords=nodes,
        displacement=u,
        element_forces=element_forces,
        displacement_max=delta_fem,
        displacement_analytical=delta_ana,
        relative_error=rel_err,
        solver_info=info,
    )


# ---------------------------------------------------------------------------
# ねん回試験（3Dのみ）
# ---------------------------------------------------------------------------
def _run_torsion(cfg: NumericalTestConfig) -> StaticTestResult:
    """ねん回試験を実行する（3Dのみ）.

    一端固定、他端ねじりモーメント T。
    """
    sec = _build_section_props(cfg.section_shape, cfg.section_params,
                               cfg.beam_type, cfg.nu)
    T = cfg.load_value
    L = cfg.length
    n_elems = cfg.n_elems

    nodes, conn = generate_beam_mesh_3d(n_elems, L)
    ke_func = _get_ke_func(cfg, sec)
    K, ndof = _assemble_3d(nodes, conn, ke_func)

    f = np.zeros(ndof)
    f[6 * n_elems + 3] = T  # 先端θx方向

    fixed_dofs = np.arange(6, dtype=int)

    K_sp = sp.csr_matrix(K)
    Kbc, fbc = apply_dirichlet(K_sp, f, fixed_dofs)
    u, info = solve_displacement(Kbc, fbc, show_progress=False)

    r_max = sec["r_max_y"]
    ana = analytical_torsion(T, L, cfg.G, sec["J"], r_max)

    theta_fem = u[6 * n_elems + 3]
    theta_ana = ana["theta"]
    rel_err = abs(theta_fem - theta_ana) / abs(theta_ana) if theta_ana != 0 else None

    element_forces = _compute_section_forces(cfg, sec, nodes, conn, u)

    return StaticTestResult(
        config=cfg,
        node_coords=nodes,
        displacement=u,
        element_forces=element_forces,
        displacement_max=theta_fem,
        displacement_analytical=theta_ana,
        relative_error=rel_err,
        max_shear_stress=ana["tau_max"],
        solver_info=info,
    )


# ---------------------------------------------------------------------------
# 剛性行列関数ディスパッチ
# ---------------------------------------------------------------------------
def _get_ke_func(cfg: NumericalTestConfig, sec: dict):
    """beam_type に応じた ke_global 関数を返す."""
    E = cfg.E
    if cfg.beam_type == "eb2d":
        from xkep_cae.elements.beam_eb2d import eb_beam2d_ke_global

        A_sec, I_sec = sec["A"], sec["I"]
        return lambda coords: eb_beam2d_ke_global(coords, E, A_sec, I_sec)

    elif cfg.beam_type == "timo2d":
        from xkep_cae.elements.beam_timo2d import timo_beam2d_ke_global

        A_sec, I_sec, kappa, G = sec["A"], sec["I"], sec["kappa"], cfg.G
        return lambda coords: timo_beam2d_ke_global(coords, E, A_sec, I_sec, kappa, G)

    elif cfg.beam_type == "timo3d":
        from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global

        A = sec["A"]
        Iy, Iz, J = sec["Iy"], sec["Iz"], sec["J"]
        ky, kz, G = sec["kappa_y"], sec["kappa_z"], cfg.G
        return lambda coords: timo_beam3d_ke_global(
            coords, E, G, A, Iy, Iz, J, ky, kz
        )

    elif cfg.beam_type == "cosserat":
        from xkep_cae.elements.beam_cosserat import cosserat_ke_global

        A = sec["A"]
        Iy, Iz, J = sec["Iy"], sec["Iz"], sec["J"]
        ky, kz, G = sec["kappa_y"], sec["kappa_z"], cfg.G
        return lambda coords: cosserat_ke_global(
            coords, E, G, A, Iy, Iz, J, ky, kz
        )
    else:
        raise ValueError(f"未対応の beam_type: {cfg.beam_type}")


# ---------------------------------------------------------------------------
# 断面力ポスト処理
# ---------------------------------------------------------------------------
def _compute_section_forces(
    cfg: NumericalTestConfig,
    sec: dict,
    nodes: np.ndarray,
    conn: np.ndarray,
    u: np.ndarray,
) -> list:
    """各要素の断面力を計算する."""
    forces = []
    E = cfg.E

    if cfg.beam_type == "eb2d":
        from xkep_cae.elements.beam_eb2d import eb_beam2d_section_forces

        for elem_nodes in conn:
            n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
            coords = nodes[[n1, n2]]
            edofs = [3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                     3 * n2, 3 * n2 + 1, 3 * n2 + 2]
            u_elem = u[edofs]
            f1, f2 = eb_beam2d_section_forces(coords, u_elem, E, sec["A"], sec["I"])
            forces.append((f1, f2))

    elif cfg.beam_type == "timo2d":
        from xkep_cae.elements.beam_timo2d import timo_beam2d_section_forces

        for elem_nodes in conn:
            n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
            coords = nodes[[n1, n2]]
            edofs = [3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                     3 * n2, 3 * n2 + 1, 3 * n2 + 2]
            u_elem = u[edofs]
            f1, f2 = timo_beam2d_section_forces(
                coords, u_elem, E, sec["A"], sec["I"], sec["kappa"], cfg.G
            )
            forces.append((f1, f2))

    elif cfg.beam_type == "timo3d":
        from xkep_cae.elements.beam_timo3d import beam3d_section_forces

        for elem_nodes in conn:
            n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
            coords = nodes[[n1, n2]]
            edofs = []
            for n in [n1, n2]:
                for d in range(6):
                    edofs.append(6 * n + d)
            u_elem = u[edofs]
            f1, f2 = beam3d_section_forces(
                coords, u_elem, E, cfg.G, sec["A"],
                sec["Iy"], sec["Iz"], sec["J"],
                sec["kappa_y"], sec["kappa_z"],
            )
            forces.append((f1, f2))

    elif cfg.beam_type == "cosserat":
        from xkep_cae.elements.beam_cosserat import cosserat_section_forces

        for elem_nodes in conn:
            n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
            coords = nodes[[n1, n2]]
            edofs = []
            for n in [n1, n2]:
                for d in range(6):
                    edofs.append(6 * n + d)
            u_elem = u[edofs]
            f1, f2 = cosserat_section_forces(
                coords, u_elem, E, cfg.G, sec["A"],
                sec["Iy"], sec["Iz"], sec["J"],
                sec["kappa_y"], sec["kappa_z"],
            )
            forces.append((f1, f2))

    return forces


# ---------------------------------------------------------------------------
# せん断応力推定ヘルパー
# ---------------------------------------------------------------------------
def _estimate_max_shear(V: float, sec: dict) -> float:
    """最大横せん断応力を断面形状に応じて推定する."""
    A = sec["A"]
    shape = sec["shape"]
    if shape == "rectangle":
        return 1.5 * abs(V) / A
    elif shape in ("circle", "pipe"):
        return 4.0 * abs(V) / (3.0 * A)
    else:
        return abs(V) / A


# ---------------------------------------------------------------------------
# 公開API
# ---------------------------------------------------------------------------
def run_test(cfg: NumericalTestConfig) -> StaticTestResult:
    """単一の静的試験を実行する.

    Args:
        cfg: 試験コンフィグ

    Returns:
        StaticTestResult
    """
    dispatch = {
        "bend3p": _run_bend3p,
        "bend4p": _run_bend4p,
        "tensile": _run_tensile,
        "torsion": _run_torsion,
    }
    runner = dispatch.get(cfg.name)
    if runner is None:
        raise ValueError(f"未対応の試験種別: {cfg.name}")
    return runner(cfg)


def run_all_tests(
    configs: list[NumericalTestConfig],
) -> list[StaticTestResult]:
    """複数の試験を一括実行する.

    Args:
        configs: 試験コンフィグのリスト

    Returns:
        list[StaticTestResult]
    """
    return [run_test(cfg) for cfg in configs]


def run_tests(
    configs: list[NumericalTestConfig],
    names: list[str],
) -> list[StaticTestResult]:
    """指定した試験名のみ部分実行する.

    Args:
        configs: 全試験コンフィグのリスト
        names: 実行する試験名リスト ("bend3p", "tensile" 等)

    Returns:
        list[StaticTestResult]
    """
    filtered = [cfg for cfg in configs if cfg.name in names]
    return [run_test(cfg) for cfg in filtered]
