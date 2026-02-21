"""数値試験フレームワーク — 動的試験ランナー.

3点曲げ等の動的（過渡応答）解析を実行する。
非線形動解析ソルバー (solve_nonlinear_transient) を使用。
nlgeom=False: 線形梁 (K·u = f_int)
nlgeom=True:  CR定式化（timo3d）または非線形Cosserat rod（cosserat）
"""

from __future__ import annotations

import numpy as np

from xkep_cae.dynamics import NonlinearTransientConfig, solve_nonlinear_transient
from xkep_cae.numerical_tests.core import (
    DynamicTestConfig,
    DynamicTestResult,
    _build_section_props,
    analytical_bend3p,
    generate_beam_mesh_2d,
    generate_beam_mesh_3d,
)
from xkep_cae.numerical_tests.frequency import (
    _assemble_lumped_mass_2d,
    _assemble_lumped_mass_3d,
    _assemble_mass_2d,
    _assemble_mass_3d,
)
from xkep_cae.numerical_tests.runner import (
    _assemble_2d,
    _assemble_3d,
    _build_simply_supported_bc_3p,
    _get_ke_func,
)


# ---------------------------------------------------------------------------
# 荷重関数ビルダー
# ---------------------------------------------------------------------------
def _build_load_function(
    cfg: DynamicTestConfig,
    ndof: int,
    load_dof: int,
) -> callable:
    """荷重タイプに応じた外力関数 f(t) → (ndof,) を構築する.

    Args:
        cfg: 動的試験コンフィグ
        ndof: 総自由度数
        load_dof: 荷重を付加する DOF インデックス

    Returns:
        外力関数 f(t)
    """
    P = -abs(cfg.load_value)  # y方向下向き

    if cfg.load_type == "step":
        f_static = np.zeros(ndof, dtype=float)
        f_static[load_dof] = P

        def get_force(t: float) -> np.ndarray:
            return f_static

    elif cfg.load_type == "ramp":
        ramp_t = cfg.ramp_time if cfg.ramp_time > 0 else cfg.dt

        def get_force(t: float) -> np.ndarray:
            f = np.zeros(ndof, dtype=float)
            scale = min(t / ramp_t, 1.0)
            f[load_dof] = P * scale
            return f

    else:
        raise ValueError(f"未対応の load_type: {cfg.load_type}")

    return get_force


# ---------------------------------------------------------------------------
# 線形梁用の内力/接線剛性コールバック生成
# ---------------------------------------------------------------------------
def _make_linear_beam_assemblers(
    K: np.ndarray,
) -> tuple[callable, callable]:
    """線形梁要素用の assemble_internal_force / assemble_tangent を返す.

    線形の場合: f_int(u) = K·u, K_T(u) = K（定数）。

    Args:
        K: 全体剛性行列 (ndof, ndof)

    Returns:
        (assemble_internal_force, assemble_tangent) タプル
    """

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        return K @ u

    def assemble_tangent(u: np.ndarray) -> np.ndarray:
        return K

    return assemble_internal_force, assemble_tangent


# ---------------------------------------------------------------------------
# CR梁用の内力/接線剛性コールバック生成
# ---------------------------------------------------------------------------
def _make_cr_beam_assemblers(
    nodes: np.ndarray,
    conn: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
) -> tuple[callable, callable]:
    """CR梁要素用の assemble_internal_force / assemble_tangent を返す.

    Corotational 定式化による幾何学的非線形対応。

    Args:
        nodes: (n_nodes, 3) 初期節点座標
        conn: (n_elems, 2) 要素接続
        E, G: 弾性定数
        A, Iy, Iz, J: 断面定数
        kappa_y, kappa_z: せん断補正係数

    Returns:
        (assemble_internal_force, assemble_tangent) タプル
    """
    from xkep_cae.elements.beam_timo3d import assemble_cr_beam3d

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        _, f_int = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    def assemble_tangent(u: np.ndarray) -> np.ndarray:
        K_T, _ = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            stiffness=True,
            internal_force=False,
        )
        return K_T

    return assemble_internal_force, assemble_tangent


# ---------------------------------------------------------------------------
# Cosserat rod 非線形用の内力/接線剛性コールバック生成
# ---------------------------------------------------------------------------
def _make_cosserat_nl_beam_assemblers(
    nodes: np.ndarray,
    conn: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    kappa_y: float,
    kappa_z: float,
) -> tuple[callable, callable]:
    """非線形 Cosserat rod 用の assemble_internal_force / assemble_tangent を返す.

    Args:
        nodes: (n_nodes, 3) 初期節点座標
        conn: (n_elems, 2) 要素接続
        E, G: 弾性定数
        A, Iy, Iz, J: 断面定数
        kappa_y, kappa_z: せん断補正係数

    Returns:
        (assemble_internal_force, assemble_tangent) タプル
    """
    from xkep_cae.elements.beam_cosserat import assemble_cosserat_nonlinear

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        _, f_int = assemble_cosserat_nonlinear(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            stiffness=False,
            internal_force=True,
        )
        return f_int

    def assemble_tangent(u: np.ndarray) -> np.ndarray:
        K_T, _ = assemble_cosserat_nonlinear(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            kappa_y,
            kappa_z,
            stiffness=True,
            internal_force=False,
        )
        return K_T

    return assemble_internal_force, assemble_tangent


# ---------------------------------------------------------------------------
# 動的3点曲げ試験
# ---------------------------------------------------------------------------
def _run_dynamic_bend3p(cfg: DynamicTestConfig) -> DynamicTestResult:
    """動的3点曲げ試験を実行する.

    境界条件は静的3点曲げと同じ（単純支持）。
    荷重はステップ荷重またはランプ荷重で中央節点に付加。
    非線形動解析ソルバーで過渡応答を計算する。

    Args:
        cfg: DynamicTestConfig

    Returns:
        DynamicTestResult
    """
    sec = _build_section_props(cfg.section_shape, cfg.section_params, cfg.beam_type, cfg.nu)
    n_elems = cfg.n_elems
    if n_elems % 2 != 0:
        n_elems += 1  # 中央節点が必要

    is_3d = cfg.beam_type in ("timo3d", "cosserat")
    dof_per_node = 6 if is_3d else 3

    # メッシュ
    if is_3d:
        nodes, conn = generate_beam_mesh_3d(n_elems, cfg.length)
    else:
        nodes, conn = generate_beam_mesh_2d(n_elems, cfg.length)

    n_nodes = len(nodes)
    ndof = dof_per_node * n_nodes

    # 剛性行列
    _dummy_cfg = type(
        "Cfg",
        (),
        {
            "beam_type": cfg.beam_type,
            "E": cfg.E,
            "nu": cfg.nu,
            "G": cfg.G,
        },
    )()
    ke_func = _get_ke_func(_dummy_cfg, sec)
    if is_3d:
        K, _ = _assemble_3d(nodes, conn, ke_func)
    else:
        K, _ = _assemble_2d(nodes, conn, ke_func)

    # 質量行列
    if cfg.mass_type == "lumped":
        if is_3d:
            M = _assemble_lumped_mass_3d(nodes, conn, cfg.rho, sec["A"], sec["Iy"], sec["Iz"])
        else:
            M = _assemble_lumped_mass_2d(nodes, conn, cfg.rho, sec["A"])
    else:
        if is_3d:
            M = _assemble_mass_3d(nodes, conn, cfg.rho, sec["A"], sec["Iy"], sec["Iz"])
        else:
            M = _assemble_mass_2d(nodes, conn, cfg.rho, sec["A"])

    # Rayleigh減衰行列
    C = cfg.damping_alpha * M + cfg.damping_beta * K

    # 境界条件（単純支持）
    fixed_dofs_list = _build_simply_supported_bc_3p(
        n_elems,
        is_3d,
        cfg.support_condition,
    )
    fixed_dofs = np.array(sorted(set(fixed_dofs_list)), dtype=int)

    # 荷重 DOF（中央節点の uy）
    mid_node = n_elems // 2
    if is_3d:
        load_dof = 6 * mid_node + 1
    else:
        load_dof = 3 * mid_node + 1

    # 外力関数
    get_force = _build_load_function(cfg, ndof, load_dof)

    # 内力/接線剛性コールバック
    if cfg.nlgeom and cfg.beam_type == "timo3d":
        assemble_f_int, assemble_K_T = _make_cr_beam_assemblers(
            nodes,
            conn,
            cfg.E,
            cfg.G,
            sec["A"],
            sec["Iy"],
            sec["Iz"],
            sec["J"],
            sec["kappa_y"],
            sec["kappa_z"],
        )
    elif cfg.nlgeom and cfg.beam_type == "cosserat":
        assemble_f_int, assemble_K_T = _make_cosserat_nl_beam_assemblers(
            nodes,
            conn,
            cfg.E,
            cfg.G,
            sec["A"],
            sec["Iy"],
            sec["Iz"],
            sec["J"],
            sec["kappa_y"],
            sec["kappa_z"],
        )
    else:
        assemble_f_int, assemble_K_T = _make_linear_beam_assemblers(K)

    # 初期条件（静止）
    u0 = np.zeros(ndof, dtype=float)
    v0 = np.zeros(ndof, dtype=float)

    # NonlinearTransientConfig
    nl_config = NonlinearTransientConfig(
        dt=cfg.dt,
        n_steps=cfg.n_steps,
        beta=cfg.beta_newmark,
        gamma=cfg.gamma_newmark,
        alpha_hht=cfg.alpha_hht,
        max_iter=cfg.max_iter,
        tol_force=cfg.tol_force,
    )

    # 求解
    result = solve_nonlinear_transient(
        M=M,
        f_ext=get_force,
        u0=u0,
        v0=v0,
        config=nl_config,
        assemble_internal_force=assemble_f_int,
        assemble_tangent=assemble_K_T,
        C=C,
        fixed_dofs=fixed_dofs,
        show_progress=False,
    )

    # 最終ステップの中央変位
    if is_3d:
        delta_fem = abs(result.displacement[-1, 6 * mid_node + 1])
    else:
        delta_fem = abs(result.displacement[-1, 3 * mid_node + 1])

    # 解析解（静的）
    use_timo = cfg.beam_type in ("timo2d", "timo3d")
    ana = analytical_bend3p(
        abs(cfg.load_value),
        cfg.length,
        cfg.E,
        sec["I"],
        kappa=sec["kappa"] if use_timo else None,
        G=cfg.G if use_timo else None,
        A=sec["A"] if use_timo else None,
    )
    delta_ana = ana["delta_mid"]
    rel_err = abs(delta_fem - delta_ana) / abs(delta_ana) if delta_ana != 0 else None

    return DynamicTestResult(
        config=cfg,
        node_coords=nodes,
        time=result.time,
        displacement=result.displacement,
        velocity=result.velocity,
        acceleration=result.acceleration,
        displacement_max_final=delta_fem,
        displacement_analytical=delta_ana,
        relative_error_final=rel_err,
        converged=result.converged,
        iterations_per_step=result.iterations_per_step,
        solver_info={
            "failed_step": result.failed_step,
            "n_elems": n_elems,
            "ndof": ndof,
            "mid_node": mid_node,
            "mass_type": cfg.mass_type,
            "nlgeom": cfg.nlgeom,
        },
    )


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------
def run_dynamic_test(cfg: DynamicTestConfig) -> DynamicTestResult:
    """動的試験を実行する.

    Args:
        cfg: 動的試験コンフィグ

    Returns:
        DynamicTestResult
    """
    dispatch = {
        "dynamic_bend3p": _run_dynamic_bend3p,
    }
    runner = dispatch.get(cfg.name)
    if runner is None:
        raise ValueError(f"未対応の動的試験種別: {cfg.name}")
    return runner(cfg)


def run_dynamic_tests(
    configs: list[DynamicTestConfig],
) -> list[DynamicTestResult]:
    """複数の動的試験を一括実行する."""
    return [run_dynamic_test(cfg) for cfg in configs]
