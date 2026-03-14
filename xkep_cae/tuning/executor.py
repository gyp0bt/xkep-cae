"""チューニングタスク実行エンジン.

AL solver 依存のため廃止。NCP 版への移行は status-167 以降で実施予定。

[← README](../../README.md)
"""

from __future__ import annotations

from xkep_cae.tuning.schema import TuningResult, TuningRun

_AL_REMOVED_MSG = (
    "この関数は AL solver 依存のため廃止されました。NCP 版への移行は status-167 以降で実施予定。"
)


def execute_s3_benchmark(
    n_strands: int,
    **solver_params,
) -> TuningRun:
    """S3ベンチマークを実行し TuningRun を返す（AL 廃止）."""
    raise NotImplementedError(_AL_REMOVED_MSG)


def run_scaling_analysis(
    strand_counts: list[int] | None = None,
    **solver_params,
) -> TuningResult:
    """複数素線数でのスケーリング分析を実行（AL 廃止）."""
    raise NotImplementedError(_AL_REMOVED_MSG)


def run_convergence_tuning(
    n_strands: int = 19,
    param_grid: dict[str, list] | None = None,
    **base_params,
) -> TuningResult:
    """パラメータグリッドでの収束チューニングを実行（AL 廃止）."""
    raise NotImplementedError(_AL_REMOVED_MSG)


def run_sensitivity_analysis(
    n_strands: int = 7,
    param1_name: str = "omega_max",
    param1_values: list[float] | None = None,
    param2_name: str = "al_relaxation",
    param2_values: list[float] | None = None,
    **base_params,
) -> TuningResult:
    """2パラメータの感度分析を実行（AL 廃止）."""
    raise NotImplementedError(_AL_REMOVED_MSG)
