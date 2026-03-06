"""チューニングタスク実行エンジン.

TuningTask の定義に基づいてソルバーを実行し、
メトリクスを収集して TuningRun を生成する。

generate_verification_plots.py から呼び出される。
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from xkep_cae.tuning.schema import TuningResult, TuningRun

# 物理定数
_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002  # 2mm
_PITCH = 0.040  # 40mm
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)
_NDOF_PER_NODE = 6


def execute_s3_benchmark(
    n_strands: int,
    *,
    n_elems_per_strand: int = 4,
    n_load_steps: int = 15,
    max_iter: int = 30,
    n_outer_max: int = 3,
    load_type: str = "tension",
    load_value: float = 100.0,
    **solver_params,
) -> TuningRun:
    """S3ベンチマークを実行し TuningRun を返す.

    test_s3_benchmark_timing.py の _run_benchmark と同等だが、
    TuningRun スキーマに沿ったメトリクス収集を行う。
    """
    from xkep_cae.contact.pair import (
        ContactConfig,
        ContactManager,
        ContactStatus,
    )
    from xkep_cae.contact.solver_hooks import (
        BenchmarkTimingCollector,
        newton_raphson_block_contact,
        newton_raphson_with_contact,
    )
    from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
    from xkep_cae.sections.beam import BeamSection

    section = BeamSection.circle(_WIRE_D)

    # --- メッシュ生成 ---
    t0 = time.perf_counter()
    mesh = make_twisted_wire_mesh(
        n_strands,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=n_elems_per_strand,
        n_pitches=1.0,
        gap=solver_params.get("gap", 0.0005),
        min_elems_per_pitch=0,
    )
    mesh_time = time.perf_counter() - t0

    # --- 線形剛性行列 ---
    ndof_total = mesh.n_nodes * _NDOF_PER_NODE
    node_coords = mesh.node_coords
    connectivity = mesh.connectivity

    K = np.zeros((ndof_total, ndof_total))
    for elem in connectivity:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = node_coords[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            coords,
            _E,
            _G,
            section.A,
            section.Iy,
            section.Iz,
            section.J,
            _KAPPA,
            _KAPPA,
        )
        edofs = np.array(
            [6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)],
            dtype=int,
        )
        K[np.ix_(edofs, edofs)] += Ke
    K_sp = sp.csr_matrix(K)

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    # --- 境界条件 ---
    fixed_dofs = set()
    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        node = nodes[0]
        for d in range(_NDOF_PER_NODE):
            fixed_dofs.add(_NDOF_PER_NODE * node + d)
    fixed_dofs_arr = np.array(sorted(fixed_dofs), dtype=int)

    # --- 外力 ---
    f_ext = np.zeros(ndof_total)
    if load_type == "tension":
        f_per_strand = load_value / n_strands
        for sid in range(mesh.n_strands):
            nodes = mesh.strand_nodes(sid)
            end_node = nodes[-1]
            f_ext[_NDOF_PER_NODE * end_node + 2] = f_per_strand

    # --- 接触 ---
    elem_layer_map = mesh.build_elem_layer_map()
    auto_kpen = solver_params.get("auto_kpen", True)
    kpen_mode = "beam_ei" if auto_kpen else "manual"
    kpen_scale = 0.1 if auto_kpen else 1e5

    staged_activation = solver_params.get("staged_activation", True)
    staged_steps = 0
    if staged_activation:
        max_lay = max(elem_layer_map.values()) if elem_layer_map else 0
        staged_steps = (max_lay + 1) * 2

    mgr = ContactManager(
        config=ContactConfig(
            k_pen_scale=kpen_scale,
            k_pen_mode=kpen_mode,
            beam_E=_E if auto_kpen else 0.0,
            beam_I=section.Iy if auto_kpen else 0.0,
            k_pen_scaling=solver_params.get("k_pen_scaling", "sqrt"),
            k_t_ratio=0.1,
            mu=0.0,
            g_on=solver_params.get("g_on", 0.0005),
            g_off=solver_params.get("g_off", 0.001),
            n_outer_max=n_outer_max,
            use_friction=False,
            use_line_search=True,
            line_search_max_steps=5,
            use_geometric_stiffness=True,
            tol_penetration_ratio=0.02,
            penalty_growth_factor=solver_params.get("penalty_growth_factor", 1.0),
            k_pen_max=1e12,
            elem_layer_map=elem_layer_map,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            linear_solver="auto",
            lambda_n_max_factor=solver_params.get("lambda_n_max_factor", 0.1),
            al_relaxation=solver_params.get("al_relaxation", 0.01),
            preserve_inactive_lambda=solver_params.get("preserve_inactive_lambda", True),
            no_deactivation_within_step=solver_params.get("no_deactivation_within_step", True),
            staged_activation_steps=staged_steps,
            adaptive_omega=solver_params.get("adaptive_omega", True),
            omega_min=solver_params.get("omega_min", 0.01),
            omega_max=solver_params.get("omega_max", 0.3),
            omega_growth=solver_params.get("omega_growth", 2.0),
        ),
    )

    # --- ソルバー実行 ---
    timing = BenchmarkTimingCollector()
    t_solve = time.perf_counter()

    use_block_solver = solver_params.get("use_block_solver", False)
    if use_block_solver:
        strand_dof_ranges = []
        for sid in range(mesh.n_strands):
            nodes = mesh.strand_nodes(sid)
            dof_start = int(nodes[0]) * _NDOF_PER_NODE
            dof_end = (int(nodes[-1]) + 1) * _NDOF_PER_NODE
            strand_dof_ranges.append((dof_start, dof_end))

        result = newton_raphson_block_contact(
            f_ext,
            fixed_dofs_arr,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            node_coords,
            connectivity,
            mesh.radii,
            strand_dof_ranges=strand_dof_ranges,
            n_load_steps=n_load_steps,
            max_iter=max_iter,
            show_progress=False,
            broadphase_margin=0.01,
        )
    else:
        result = newton_raphson_with_contact(
            f_ext,
            fixed_dofs_arr,
            assemble_tangent,
            assemble_internal_force,
            mgr,
            node_coords,
            connectivity,
            mesh.radii,
            n_load_steps=n_load_steps,
            max_iter=max_iter,
            show_progress=False,
            broadphase_margin=0.01,
            timing=timing,
        )
    solve_time = time.perf_counter() - t_solve

    # --- メトリクス収集 ---
    n_active = sum(1 for p in mgr.pairs if p.state.status != ContactStatus.INACTIVE)
    max_pen = 0.0
    for p in mgr.pairs:
        if p.state.status == ContactStatus.INACTIVE:
            continue
        if p.state.gap < 0:
            pen = abs(p.state.gap) / (p.radius_a + p.radius_b)
            if pen > max_pen:
                max_pen = pen

    phase_totals = timing.phase_totals()
    total_time = timing.total_time()
    linear_solve_time = phase_totals.get("linear_solve", 0.0)
    linear_solve_ratio = linear_solve_time / total_time if total_time > 0 else 0.0

    # 時系列データ
    ts: dict[str, list[float]] = {}
    if result.load_history:
        ts["load_factor"] = list(result.load_history)
    if result.contact_force_history:
        ts["contact_force"] = list(result.contact_force_history)
    if result.graph_history and result.graph_history.snapshots:
        gh = result.graph_history
        ts["active_pairs"] = [float(s.n_edges) for s in gh.snapshots]
        ts["total_normal_force"] = [s.total_normal_force for s in gh.snapshots]
        if hasattr(gh, "stick_slip_ratio_series"):
            try:
                ssr = gh.stick_slip_ratio_series()
                ts["stick_slip_ratio"] = ssr.tolist()
            except Exception:
                pass

    # 工程別タイミング
    timing_breakdown: dict[str, float] = {}
    for phase, t in phase_totals.items():
        timing_breakdown[f"time_{phase}"] = t

    metrics = {
        "converged": result.converged,
        "n_load_steps": result.n_load_steps,
        "total_newton_iterations": result.total_newton_iterations,
        "total_outer_iterations": result.total_outer_iterations,
        "n_active_pairs": n_active,
        "max_penetration_ratio": max_pen,
        "total_time_s": total_time,
        "solve_time_s": solve_time,
        "mesh_time_s": mesh_time,
        "linear_solve_ratio": linear_solve_ratio,
        **timing_breakdown,
    }

    metadata = {
        "n_strands": n_strands,
        "n_nodes": mesh.n_nodes,
        "n_elems": mesh.n_elems,
        "ndof": ndof_total,
        "n_elems_per_strand": n_elems_per_strand,
        "load_type": load_type,
        "load_value": load_value,
        "step_times": timing.step_times(),
    }

    return TuningRun(
        params={
            k: v
            for k, v in solver_params.items()
            if k not in ("gap",)  # gap はメッシュパラメータ
        },
        metrics=metrics,
        time_series=ts,
        metadata=metadata,
    )


def run_scaling_analysis(
    strand_counts: list[int] | None = None,
    **solver_params,
) -> TuningResult:
    """複数素線数でのスケーリング分析を実行."""
    from xkep_cae.tuning.presets import s3_scaling_task

    if strand_counts is None:
        strand_counts = [7, 19]

    task = s3_scaling_task()
    result = TuningResult(task=task)

    for n in strand_counts:
        run = execute_s3_benchmark(n, **solver_params)
        result.add_run(run)

    return result


def run_convergence_tuning(
    n_strands: int = 19,
    param_grid: dict[str, list] | None = None,
    **base_params,
) -> TuningResult:
    """パラメータグリッドでの収束チューニングを実行."""
    from xkep_cae.tuning.presets import s3_convergence_task

    task = s3_convergence_task(n_strands)
    result = TuningResult(task=task)

    if param_grid is None:
        # デフォルトパラメータで1回実行
        run = execute_s3_benchmark(n_strands, **base_params)
        result.add_run(run)
    else:
        # グリッドサーチ
        keys = list(param_grid.keys())
        import itertools

        for combo in itertools.product(*(param_grid[k] for k in keys)):
            params = dict(base_params)
            for k, v in zip(keys, combo, strict=True):
                params[k] = v
            run = execute_s3_benchmark(n_strands, **params)
            result.add_run(run)

    return result
