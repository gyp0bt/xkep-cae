"""数値試験フレームワーク — 静的試験ランナー.

3点曲げ・4点曲げ・引張・ねん回の各試験を
統一インタフェースで実行する。

要素剛性・BC適用・線形ソルバーは _backend.backend 経由で注入する。
xkep_cae 内から deprecated パッケージへの直接 import を回避する（C14 準拠）。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.numerical_tests._backend import backend
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


# ---------------------------------------------------------------------------
# 2D/3D アセンブリ（低レベル直接アセンブリ）
# ---------------------------------------------------------------------------
def _assemble_2d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    ke_func,  # noqa: ANN001
) -> tuple[np.ndarray, int]:
    """2D梁の全体剛性行列をアセンブルする（3DOF/節点）."""
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)
    for elem_nodes in connectivity:
        n1, n2 = int(elem_nodes[0]), int(elem_nodes[1])
        coords = nodes[[n1, n2]]
        Ke = ke_func(coords)
        edofs = np.array(
            [3 * n1, 3 * n1 + 1, 3 * n1 + 2, 3 * n2, 3 * n2 + 1, 3 * n2 + 2], dtype=int
        )
        for ii in range(6):
            for jj in range(6):
                K[edofs[ii], edofs[jj]] += Ke[ii, jj]
    return K, ndof


def _assemble_3d(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    ke_func,  # noqa: ANN001
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
    """3点/4点曲げ試験用の単純支持境界条件を生成する."""
    fixed_dofs: list[int] = []
    if is_3d:
        fixed_dofs.append(6 * 0 + 0)  # ux
        fixed_dofs.append(6 * 0 + 1)  # uy
        fixed_dofs.append(6 * 0 + 2)  # uz
        fixed_dofs.append(6 * 0 + 3)  # θx
        fixed_dofs.append(6 * n_elems + 1)  # uy
        fixed_dofs.append(6 * n_elems + 2)  # uz
        if support_condition == "pin":
            fixed_dofs.append(6 * n_elems + 0)  # ux
    else:
        fixed_dofs.append(3 * 0 + 0)  # ux
        fixed_dofs.append(3 * 0 + 1)  # uy
        fixed_dofs.append(3 * n_elems + 1)  # uy
        if support_condition == "pin":
            fixed_dofs.append(3 * n_elems + 0)  # ux
    return fixed_dofs


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
# 共通 solve ヘルパー
# ---------------------------------------------------------------------------
def _solve_static(
    K: np.ndarray,
    f: np.ndarray,
    fixed_dofs: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """剛性行列を疎行列化 → BC 適用 → 線形ソルブ."""
    K_sp = sp.csr_matrix(K)
    Kbc, fbc = backend.apply_dirichlet(K_sp, f, fixed_dofs)
    return backend.solve(Kbc, fbc, show_progress=False)


# ---------------------------------------------------------------------------
# 3点曲げ試験
# ---------------------------------------------------------------------------
def _run_bend3p(cfg: NumericalTestConfig) -> StaticTestResult:
    """3点曲げ試験を実行する."""
    sec = _build_section_props(cfg.section_shape, cfg.section_params, cfg.beam_type, cfg.nu)
    P = cfg.load_value
    L = cfg.length
    n_elems = cfg.n_elems
    if n_elems % 2 != 0:
        n_elems += 1

    is_3d = cfg.beam_type in ("timo3d", "cosserat")

    if is_3d:
        nodes, conn = generate_beam_mesh_3d(n_elems, L)
    else:
        nodes, conn = generate_beam_mesh_2d(n_elems, L)

    ke_func = backend.ke_func_factory(cfg, sec)
    if is_3d:
        K, ndof = _assemble_3d(nodes, conn, ke_func)
    else:
        K, ndof = _assemble_2d(nodes, conn, ke_func)

    f = np.zeros(ndof)
    mid_node = n_elems // 2
    if is_3d:
        f[6 * mid_node + 1] = -abs(P)
    else:
        f[3 * mid_node + 1] = -abs(P)

    fixed_dofs = np.array(
        sorted(set(_build_simply_supported_bc_3p(n_elems, is_3d, cfg.support_condition))),
        dtype=int,
    )
    u, info = _solve_static(K, f, fixed_dofs)

    use_timo = cfg.beam_type in ("timo2d", "timo3d")
    ana = analytical_bend3p(
        abs(P),
        L,
        cfg.E,
        sec["I"],
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

    element_forces = backend.section_force_computer(cfg, sec, nodes, conn, u)
    friction_msg = assess_friction_effect(cfg.name, cfg.span_ratio, cfg.support_condition)

    return StaticTestResult(
        config=cfg,
        node_coords=nodes,
        displacement=u,
        element_forces=element_forces,
        displacement_max=delta_fem,
        displacement_analytical=delta_ana,
        relative_error=rel_err,
        max_bending_stress=(ana["M_max"] * sec.get("r_max_z", 0) / sec["I"] if sec["I"] > 0 else 0),
        max_shear_stress=_estimate_max_shear(ana["V_max"], sec),
        friction_warning=friction_msg,
        solver_info=info,
    )


# ---------------------------------------------------------------------------
# 4点曲げ試験
# ---------------------------------------------------------------------------
def _run_bend4p(cfg: NumericalTestConfig) -> StaticTestResult:
    """4点曲げ試験を実行する."""
    sec = _build_section_props(cfg.section_shape, cfg.section_params, cfg.beam_type, cfg.nu)
    P = cfg.load_value
    L = cfg.length
    a = cfg.load_span
    n_elems = cfg.n_elems

    is_3d = cfg.beam_type in ("timo3d", "cosserat")

    if is_3d:
        nodes, conn = generate_beam_mesh_3d(n_elems, L)
    else:
        nodes, conn = generate_beam_mesh_2d(n_elems, L)

    x_coords = nodes[:, 0]
    load_node_left = int(np.argmin(np.abs(x_coords - a)))
    a_actual = x_coords[load_node_left]
    load_node_right = int(np.argmin(np.abs(x_coords - (L - a_actual))))
    mid_node = int(np.argmin(np.abs(x_coords - L / 2.0)))

    ke_func = backend.ke_func_factory(cfg, sec)
    if is_3d:
        K, ndof = _assemble_3d(nodes, conn, ke_func)
    else:
        K, ndof = _assemble_2d(nodes, conn, ke_func)

    f = np.zeros(ndof)
    if is_3d:
        f[6 * load_node_left + 1] = -abs(P)
        f[6 * load_node_right + 1] = -abs(P)
    else:
        f[3 * load_node_left + 1] = -abs(P)
        f[3 * load_node_right + 1] = -abs(P)

    fixed_dofs = np.array(
        sorted(set(_build_simply_supported_bc_3p(n_elems, is_3d, cfg.support_condition))),
        dtype=int,
    )
    u, info = _solve_static(K, f, fixed_dofs)

    use_timo = cfg.beam_type in ("timo2d", "timo3d")
    ana = analytical_bend4p(
        abs(P),
        L,
        a_actual,
        cfg.E,
        sec["I"],
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

    element_forces = backend.section_force_computer(cfg, sec, nodes, conn, u)
    friction_msg = assess_friction_effect(cfg.name, cfg.span_ratio, cfg.support_condition)

    return StaticTestResult(
        config=cfg,
        node_coords=nodes,
        displacement=u,
        element_forces=element_forces,
        displacement_max=delta_fem,
        displacement_analytical=delta_ana,
        relative_error=rel_err,
        max_bending_stress=(ana["M_max"] * sec.get("r_max_z", 0) / sec["I"] if sec["I"] > 0 else 0),
        max_shear_stress=_estimate_max_shear(ana["V_max"], sec),
        friction_warning=friction_msg,
        solver_info=info,
    )


# ---------------------------------------------------------------------------
# 引張試験
# ---------------------------------------------------------------------------
def _run_tensile(cfg: NumericalTestConfig) -> StaticTestResult:
    """引張試験を実行する."""
    sec = _build_section_props(cfg.section_shape, cfg.section_params, cfg.beam_type, cfg.nu)
    P = cfg.load_value
    L = cfg.length
    n_elems = cfg.n_elems

    is_3d = cfg.beam_type in ("timo3d", "cosserat")

    if is_3d:
        nodes, conn = generate_beam_mesh_3d(n_elems, L)
    else:
        nodes, conn = generate_beam_mesh_2d(n_elems, L)

    ke_func = backend.ke_func_factory(cfg, sec)
    if is_3d:
        K, ndof = _assemble_3d(nodes, conn, ke_func)
    else:
        K, ndof = _assemble_2d(nodes, conn, ke_func)

    f = np.zeros(ndof)
    if is_3d:
        f[6 * n_elems + 0] = P
    else:
        f[3 * n_elems + 0] = P

    if is_3d:
        fixed_dofs = np.arange(6, dtype=int)
    else:
        fixed_dofs = np.arange(3, dtype=int)

    u, info = _solve_static(K, f, fixed_dofs)

    ana = analytical_tensile(P, L, cfg.E, sec["A"])

    if is_3d:
        delta_fem = u[6 * n_elems + 0]
    else:
        delta_fem = u[3 * n_elems + 0]

    delta_ana = ana["delta"]
    rel_err = abs(delta_fem - delta_ana) / abs(delta_ana) if delta_ana != 0 else None

    element_forces = backend.section_force_computer(cfg, sec, nodes, conn, u)

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
    """ねん回試験を実行する（3Dのみ）."""
    sec = _build_section_props(cfg.section_shape, cfg.section_params, cfg.beam_type, cfg.nu)
    T = cfg.load_value
    L = cfg.length
    n_elems = cfg.n_elems

    nodes, conn = generate_beam_mesh_3d(n_elems, L)
    ke_func = backend.ke_func_factory(cfg, sec)
    K, ndof = _assemble_3d(nodes, conn, ke_func)

    f = np.zeros(ndof)
    f[6 * n_elems + 3] = T

    fixed_dofs = np.arange(6, dtype=int)

    u, info = _solve_static(K, f, fixed_dofs)

    r_max = sec["r_max_y"]
    ana = analytical_torsion(T, L, cfg.G, sec["J"], r_max)

    theta_fem = u[6 * n_elems + 3]
    theta_ana = ana["theta"]
    rel_err = abs(theta_fem - theta_ana) / abs(theta_ana) if theta_ana != 0 else None

    element_forces = backend.section_force_computer(cfg, sec, nodes, conn, u)

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
# 公開API
# ---------------------------------------------------------------------------
def run_test(cfg: NumericalTestConfig) -> StaticTestResult:
    """単一の静的試験を実行する."""
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
    """複数の試験を一括実行する."""
    return [run_test(cfg) for cfg in configs]


def run_tests(
    configs: list[NumericalTestConfig],
    names: list[str],
) -> list[StaticTestResult]:
    """指定した試験名のみ部分実行する."""
    filtered = [cfg for cfg in configs if cfg.name in names]
    return [run_test(cfg) for cfg in filtered]
