"""Optuna 連携による自動チューニングループ.

TuningTask の宣言的定義を Optuna の Study に変換し、
ベイズ最適化によるパラメータ探索を自動化する。

使用例::

    from xkep_cae.tuning.optuna_tuner import run_optuna_study
    from xkep_cae.tuning.presets import s3_convergence_task

    task = s3_convergence_task(7)
    result = run_optuna_study(task, n_trials=20, n_strands=7)

Note:
    Optuna が未インストールの場合は ImportError を送出する。
"""

from __future__ import annotations

from typing import Any

from xkep_cae.tuning.schema import TuningResult, TuningTask


def _check_optuna():
    """Optuna の利用可能性を確認."""
    try:
        import optuna  # noqa: F401

        return optuna
    except ImportError as e:
        raise ImportError("自動チューニングには Optuna が必要です: pip install optuna") from e


def create_objective(
    task: TuningTask,
    result: TuningResult,
    *,
    n_strands: int = 7,
    objective_metric: str = "total_newton_iterations",
    minimize: bool = True,
    **base_params: Any,
):
    """Optuna の objective 関数を生成.

    TuningTask のパラメータ定義から Optuna の suggest メソッドを
    自動的に選択し、実行結果を TuningResult に蓄積する。

    Args:
        task: チューニングタスク定義
        result: 結果の蓄積先
        n_strands: 素線数
        objective_metric: 最適化対象メトリクス
        minimize: True なら最小化、False なら最大化
        **base_params: ソルバーのベースパラメータ

    Returns:
        Optuna objective 関数
    """
    from xkep_cae.tuning.executor import execute_s3_benchmark

    def objective(trial):
        params = dict(base_params)
        for p in task.params:
            if p.log_scale:
                val = trial.suggest_float(p.name, p.low, p.high, log=True)
            else:
                val = trial.suggest_float(p.name, p.low, p.high)
            params[p.name] = val

        run = execute_s3_benchmark(n_strands, **params)
        result.add_run(run)

        # 収束しなかった場合はペナルティ値を返す
        if not run.metrics.get("converged", False):
            return float("inf") if minimize else float("-inf")

        val = run.metrics.get(objective_metric)
        if val is None:
            return float("inf") if minimize else float("-inf")
        return float(val)

    return objective


def run_optuna_study(
    task: TuningTask,
    *,
    n_trials: int = 20,
    n_strands: int = 7,
    objective_metric: str = "total_newton_iterations",
    minimize: bool = True,
    study_name: str | None = None,
    **base_params: Any,
) -> TuningResult:
    """Optuna Study を実行し TuningResult を返す.

    Args:
        task: チューニングタスク定義
        n_trials: 試行回数
        n_strands: 素線数
        objective_metric: 最適化対象メトリクス
        minimize: True なら最小化
        study_name: Study 名（省略時はタスク名）
        **base_params: ソルバーのベースパラメータ

    Returns:
        TuningResult: 全試行の結果
    """
    optuna = _check_optuna()

    result = TuningResult(task=task)
    objective = create_objective(
        task,
        result,
        n_strands=n_strands,
        objective_metric=objective_metric,
        minimize=minimize,
        **base_params,
    )

    direction = "minimize" if minimize else "maximize"
    study = optuna.create_study(
        study_name=study_name or task.name,
        direction=direction,
    )

    # デフォルト値を初期試行として追加
    defaults = {}
    for p in task.params:
        if p.default is not None:
            defaults[p.name] = p.default
    if defaults:
        study.enqueue_trial(defaults)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # best パラメータをメタデータに記録
    if result.runs:
        best = result.best_run(objective_metric, minimize=minimize)
        if best is not None:
            best.metadata["optuna_best"] = True

    return result
