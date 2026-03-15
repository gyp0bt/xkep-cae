"""チューニングタスク実行エンジン（NCP版）.

smooth_penalty + Process API ベースでベンチマーク・チューニングを実行する。
status-172 で AL 版から NCP 版に再実装。

[← README](../../README.md)
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from xkep_cae.tuning.schema import TuningResult, TuningRun


def _build_strand_problem(
    n_strands: int,
    *,
    wire_radius: float = 0.25,
    pitch_length: float = 10.0,
    n_elements_per_pitch: int = 16,
    gap: float = 0.0,
    E: float = 200000.0,
    G: float = 77000.0,
) -> tuple:
    """撚線ベンチマーク問題のメッシュ・境界条件・コールバックを構築.

    Returns:
        (mesh_data, boundary_data, contact_data, callbacks)
    """
    from xkep_cae.elements.beam_timo3d import ULCRBeamAssembler
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
    from xkep_cae.process.concrete.pre_contact import ContactSetupConfig, ContactSetupProcess
    from xkep_cae.process.data import (
        AssembleCallbacks,
        BoundaryData,
        MeshData,
    )
    from xkep_cae.sections.beam import BeamSection

    # 1. メッシュ生成
    wire_diameter = wire_radius * 2.0
    mesh = make_twisted_wire_mesh(
        n_strands=n_strands,
        wire_diameter=wire_diameter,
        pitch_length=pitch_length,
        n_elements_per_strand=n_elements_per_pitch,
        gap=gap,
    )

    mesh_data = MeshData(
        node_coords=mesh.node_coords,
        connectivity=mesh.connectivity,
        radii=wire_radius,
        n_strands=n_strands,
        layer_ids=mesh.layer_ids if hasattr(mesh, "layer_ids") else None,
    )

    # 2. 接触設定
    contact_config = ContactSetupConfig(
        mesh=mesh_data,
        use_friction=True,
        mu=0.15,
        contact_mode="smooth_penalty",
    )
    contact_proc = ContactSetupProcess()
    contact_data = contact_proc.process(contact_config)

    # 3. UL アセンブラ構築
    section = BeamSection.circular(wire_radius)
    kappa = 5.0 / 6.0
    ul_asm = ULCRBeamAssembler(
        mesh.node_coords,
        mesh.connectivity,
        E,
        G,
        section.A,
        section.Iy,
        section.Iz,
        section.J,
        kappa,
        kappa,
    )

    # 4. 境界条件（片端固定 + 横荷重）
    n_nodes = mesh.n_nodes
    ndof = n_nodes * 6
    n_elems_per_strand = len(mesh.connectivity) // n_strands

    # 左端固定
    fixed_nodes = [i * (n_elems_per_strand + 1) for i in range(n_strands)]
    fixed_dofs = np.array(
        [node * 6 + d for node in fixed_nodes for d in range(6)],
        dtype=int,
    )

    # 右端に横荷重
    tip_nodes = [(i + 1) * (n_elems_per_strand + 1) - 1 for i in range(n_strands)]
    f_ext = np.zeros(ndof)
    load_per_strand = 1.0 / max(n_strands, 1)
    for node in tip_nodes:
        f_ext[node * 6 + 1] = load_per_strand  # Y 方向

    boundary_data = BoundaryData(
        f_ext_total=f_ext,
        fixed_dofs=fixed_dofs,
    )

    callbacks = AssembleCallbacks(
        assemble_tangent=ul_asm.assemble_tangent,
        assemble_internal_force=ul_asm.assemble_internal_force,
        ul_assembler=ul_asm,
    )

    return mesh_data, boundary_data, contact_data, callbacks


def _run_solver(
    mesh_data,
    boundary_data,
    contact_data,
    callbacks,
    **solver_params,
) -> TuningRun:
    """smooth_penalty ソルバーを実行し TuningRun を返す."""
    from xkep_cae.contact.solver_smooth_penalty import solve_smooth_penalty_friction
    from xkep_cae.process.data import default_strategies

    ndof = len(boundary_data.f_ext_total)
    k_pen = solver_params.pop("k_pen", contact_data.k_pen)
    mu = solver_params.pop("mu", contact_data.mu or 0.15)

    strategies = default_strategies(
        ndof=ndof,
        k_pen=k_pen,
        use_friction=True,
        mu=mu,
        contact_mode="smooth_penalty",
        line_contact=True,
    )

    t0 = time.perf_counter()
    try:
        result = solve_smooth_penalty_friction(
            f_ext_total=boundary_data.f_ext_total,
            fixed_dofs=boundary_data.fixed_dofs,
            assemble_tangent=callbacks.assemble_tangent,
            assemble_internal_force=callbacks.assemble_internal_force,
            manager=contact_data.manager,
            node_coords_ref=mesh_data.node_coords,
            connectivity=mesh_data.connectivity,
            radii=mesh_data.radii,
            strategies=strategies,
            k_pen=k_pen,
            mu=mu,
            ul_assembler=callbacks.ul_assembler,
            show_progress=False,
        )
        elapsed = time.perf_counter() - t0
        converged = result.converged
        n_increments = result.n_increments
        total_iters = result.total_newton_iterations
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        converged = False
        n_increments = 0
        total_iters = 0
        solver_params["error"] = str(exc)

    metrics = {
        "converged": converged,
        "total_time_s": elapsed,
        "n_increments": n_increments,
        "total_newton_iterations": total_iters,
    }

    return TuningRun(
        params=dict(solver_params),
        metrics=metrics,
        metadata={"ndof": ndof, "n_nodes": mesh_data.node_coords.shape[0]},
    )


def execute_s3_benchmark(
    n_strands: int,
    **solver_params,
) -> TuningRun:
    """S3ベンチマークを実行し TuningRun を返す.

    NCP + smooth_penalty + Process API ベースで撚線曲げ問題を解く。

    Args:
        n_strands: 素線数（7, 19, 37, 61, 91）
        **solver_params: ソルバーパラメータ（k_pen, mu 等）

    Returns:
        TuningRun: 実行結果（メトリクス: converged, total_time_s, etc.）
    """
    mesh_data, boundary_data, contact_data, callbacks = _build_strand_problem(
        n_strands=n_strands,
    )
    solver_params["n_strands"] = n_strands
    return _run_solver(mesh_data, boundary_data, contact_data, callbacks, **solver_params)


def run_scaling_analysis(
    strand_counts: list[int] | None = None,
    **solver_params,
) -> TuningResult:
    """複数素線数でのスケーリング分析を実行.

    Args:
        strand_counts: 素線数リスト（デフォルト: [7, 19, 37, 61, 91]）
        **solver_params: ソルバーパラメータ

    Returns:
        TuningResult: 全素線数の実行結果を集約
    """
    from xkep_cae.tuning.presets import s3_scaling_task

    if strand_counts is None:
        strand_counts = [7, 19, 37, 61, 91]

    task = s3_scaling_task()
    result = TuningResult(task=task)

    for n in strand_counts:
        run = execute_s3_benchmark(n_strands=n, **solver_params)
        result.add_run(run)

    return result


def run_convergence_tuning(
    n_strands: int = 19,
    param_grid: dict[str, list] | None = None,
    **base_params,
) -> TuningResult:
    """パラメータグリッドでの収束チューニングを実行.

    Args:
        n_strands: 素線数
        param_grid: パラメータ名→候補値リストの辞書
        **base_params: 固定パラメータ

    Returns:
        TuningResult: 全グリッド点の結果を集約
    """
    from xkep_cae.tuning.presets import s3_convergence_task

    task = s3_convergence_task(n_strands=n_strands)
    result = TuningResult(task=task)

    if param_grid is None:
        param_grid = {"k_pen": [1e4, 1e5, 1e6]}

    names = list(param_grid.keys())
    values_list = list(param_grid.values())
    for combo in itertools.product(*values_list):
        params = dict(base_params)
        for name, val in zip(names, combo, strict=True):
            params[name] = val
        run = execute_s3_benchmark(n_strands=n_strands, **params)
        result.add_run(run)

    return result


def run_sensitivity_analysis(
    n_strands: int = 7,
    param1_name: str = "k_pen",
    param1_values: list[float] | None = None,
    param2_name: str = "mu",
    param2_values: list[float] | None = None,
    **base_params,
) -> TuningResult:
    """2パラメータの感度分析を実行.

    param1 × param2 の直交グリッドで走査し、各組み合わせの結果を返す。

    Args:
        n_strands: 素線数
        param1_name: 第1パラメータ名
        param1_values: 第1パラメータの候補値リスト
        param2_name: 第2パラメータ名
        param2_values: 第2パラメータの候補値リスト
        **base_params: 固定パラメータ

    Returns:
        TuningResult: 全グリッド点の結果を集約
    """
    from xkep_cae.tuning.presets import s3_convergence_task

    if param1_values is None:
        param1_values = [1e4, 1e5, 1e6]
    if param2_values is None:
        param2_values = [0.1, 0.15, 0.2]

    task = s3_convergence_task(n_strands=n_strands)
    result = TuningResult(task=task)

    for v1, v2 in itertools.product(param1_values, param2_values):
        params = dict(base_params)
        params[param1_name] = v1
        params[param2_name] = v2
        run = execute_s3_benchmark(n_strands=n_strands, **params)
        result.add_run(run)

    return result
