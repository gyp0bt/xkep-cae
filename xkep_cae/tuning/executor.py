"""チューニングタスク実行エンジン（NCP + smooth_penalty 版）.

S3ベンチマーク（曲げ揺動）を Process API 経由で実行し、
TuningRun / TuningResult として結果を返す。

status-172 で NCP 版として再実装。AL 依存を完全に排除。

[← README](../../README.md)
"""

from __future__ import annotations

import itertools
import time
from typing import Any

from xkep_cae.tuning.schema import TuningResult, TuningRun


def execute_s3_benchmark(
    n_strands: int,
    **solver_params: Any,
) -> TuningRun:
    """S3ベンチマークを実行し TuningRun を返す.

    run_bending_oscillation() を呼び出し、結果を TuningRun にマッピング。
    収束失敗時もメトリクスを返す（converged=False）。

    Args:
        n_strands: 素線数（7, 19, 37, 61, 91）
        **solver_params: run_bending_oscillation に渡すパラメータ

    Returns:
        TuningRun: ベンチマーク結果
    """
    from xkep_cae.numerical_tests.wire_bending_benchmark import (
        run_bending_oscillation,
    )

    t0 = time.perf_counter()
    try:
        result = run_bending_oscillation(n_strands=n_strands, **solver_params)
        elapsed = time.perf_counter() - t0

        # Phase 1/2 のNewton反復数を集計
        total_newton = 0
        if result.phase1_result is not None:
            total_newton += result.phase1_result.total_newton_iterations
        for r in result.phase2_results:
            total_newton += r.total_newton_iterations

        metrics: dict[str, Any] = {
            "converged": result.phase1_converged and result.phase2_converged,
            "phase1_converged": result.phase1_converged,
            "phase2_converged": result.phase2_converged,
            "total_newton_iterations": total_newton,
            "max_penetration_ratio": result.max_penetration_ratio,
            "n_active_pairs": result.n_active_contacts,
            "total_time_s": elapsed,
        }

        metadata: dict[str, Any] = {
            "n_strands": result.n_strands,
            "n_elems": result.n_elems,
            "n_nodes": result.n_nodes,
            "ndof": result.ndof,
            "mesh_length": result.mesh_length,
            "tip_displacement": result.tip_displacement_final,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        metrics = {
            "converged": False,
            "phase1_converged": False,
            "phase2_converged": False,
            "total_newton_iterations": 0,
            "max_penetration_ratio": float("inf"),
            "n_active_pairs": 0,
            "total_time_s": elapsed,
            "error": str(exc),
        }
        metadata = {"n_strands": n_strands}

    return TuningRun(
        params={"n_strands": n_strands, **solver_params},
        metrics=metrics,
        metadata=metadata,
    )


def run_scaling_analysis(
    strand_counts: list[int] | None = None,
    **solver_params: Any,
) -> TuningResult:
    """複数素線数でのスケーリング分析を実行.

    各素線数で execute_s3_benchmark を呼び出し、計算時間の
    スケーリング特性を TuningResult として集約する。

    Args:
        strand_counts: 素線数リスト（デフォルト: [7, 19, 37, 61, 91]）
        **solver_params: 全ベンチマーク共通のソルバーパラメータ

    Returns:
        TuningResult: スケーリング分析結果
    """
    from xkep_cae.tuning.presets import s3_scaling_task

    if strand_counts is None:
        strand_counts = [7, 19, 37, 61, 91]

    task = s3_scaling_task()
    result = TuningResult(task=task)

    for n in strand_counts:
        run = execute_s3_benchmark(n, **solver_params)
        result.add_run(run)

    return result


def run_convergence_tuning(
    n_strands: int = 19,
    param_grid: dict[str, list] | None = None,
    **base_params: Any,
) -> TuningResult:
    """パラメータグリッドでの収束チューニングを実行.

    グリッドサーチで全組み合わせを実行し、収束の可否と
    Newton反復数を TuningResult として返す。

    Args:
        n_strands: 素線数
        param_grid: パラメータグリッド {"param_name": [val1, val2, ...]}
        **base_params: ベースパラメータ（グリッドで上書き）

    Returns:
        TuningResult: 収束チューニング結果
    """
    from xkep_cae.tuning.presets import s3_convergence_task

    task = s3_convergence_task(n_strands)
    result = TuningResult(task=task)

    if param_grid is None:
        # デフォルト: タスクのデフォルトパラメータで1回実行
        run = execute_s3_benchmark(n_strands, **base_params)
        result.add_run(run)
        return result

    # グリッドサーチ
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for combo in itertools.product(*param_values):
        params = dict(base_params)
        for name, val in zip(param_names, combo, strict=True):
            params[name] = val

        run = execute_s3_benchmark(n_strands, **params)
        result.add_run(run)

    return result


def run_sensitivity_analysis(
    n_strands: int = 7,
    param1_name: str = "omega_max",
    param1_values: list[float] | None = None,
    param2_name: str = "al_relaxation",
    param2_values: list[float] | None = None,
    **base_params: Any,
) -> TuningResult:
    """2パラメータの感度分析を実行.

    param1 × param2 の全組み合わせでベンチマークを実行し、
    各メトリクスのパラメータ感度を分析する。

    Args:
        n_strands: 素線数
        param1_name: 1つ目のパラメータ名
        param1_values: 1つ目のパラメータ値リスト
        param2_name: 2つ目のパラメータ名
        param2_values: 2つ目のパラメータ値リスト
        **base_params: ベースパラメータ

    Returns:
        TuningResult: 感度分析結果
    """
    from xkep_cae.tuning.presets import s3_convergence_task

    if param1_values is None:
        param1_values = [0.1, 0.3, 0.5, 0.8, 1.0]
    if param2_values is None:
        param2_values = [0.001, 0.01, 0.1, 0.5]

    task = s3_convergence_task(n_strands)
    result = TuningResult(task=task)

    for v1, v2 in itertools.product(param1_values, param2_values):
        params = dict(base_params)
        params[param1_name] = v1
        params[param2_name] = v2

        run = execute_s3_benchmark(n_strands, **params)
        result.add_run(run)

    return result
