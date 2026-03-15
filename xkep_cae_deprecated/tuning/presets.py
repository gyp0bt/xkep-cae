"""S3チューニングタスクのプリセット定義.

現在の推奨パラメータ（_TUNED_PARAMS）を TuningTask スキーマで
形式化し、段階的スケールアップの判定基準を統一する。
"""

from __future__ import annotations

from xkep_cae_deprecated.tuning.schema import AcceptanceCriterion, TuningParam, TuningTask

# ====================================================================
# S3: NCP収束チューニング
# ====================================================================


def s3_convergence_task(n_strands: int = 19) -> TuningTask:
    """S3収束チューニングタスクを生成.

    Args:
        n_strands: 素線数（7, 19, 37, 61, 91）
    """
    return TuningTask(
        name=f"s3_ncp_convergence_{n_strands}strand",
        description=f"{n_strands}本撚りNCP接触ソルバーの収束パラメータ最適化",
        params=[
            TuningParam(
                "omega_max",
                low=0.1,
                high=1.0,
                default=0.3,
                description="Adaptive omega 上限（接触力緩和の最大係数）",
            ),
            TuningParam(
                "omega_min",
                low=0.001,
                high=0.1,
                default=0.01,
                description="Adaptive omega 下限",
            ),
            TuningParam(
                "al_relaxation",
                low=0.001,
                high=0.5,
                default=0.01,
                log_scale=True,
                description="AL乗数更新の緩和係数",
            ),
            TuningParam(
                "lambda_n_max_factor",
                low=0.01,
                high=1.0,
                default=0.1,
                log_scale=True,
                description="λ_nキャッピング係数（k_pen×gap に対する比率）",
            ),
            TuningParam(
                "g_on",
                low=0.0001,
                high=0.005,
                default=0.0005,
                log_scale=True,
                description="接触活性化ギャップ閾値",
            ),
        ],
        criteria=[
            AcceptanceCriterion(
                "converged",
                op="eq",
                target=True,
                description="NRソルバーが全ステップ収束",
            ),
            AcceptanceCriterion(
                "max_penetration_ratio",
                op="lt",
                target=0.10,
                description="最大貫入比がワイヤ径の10%未満",
            ),
            AcceptanceCriterion(
                "n_active_pairs",
                op="ge",
                target=1,
                description="少なくとも1つの接触ペアが活性",
            ),
        ],
        fixed_params={
            "n_strands": n_strands,
            "auto_kpen": True,
            "k_pen_scaling": "sqrt",
            "staged_activation": True,
            "preserve_inactive_lambda": True,
            "no_deactivation_within_step": True,
            "penalty_growth_factor": 1.0,
            "gap": 0.0005,
            "use_block_solver": True,
            "adaptive_omega": True,
            "omega_growth": 2.0,
        },
        tags=["s3", "convergence", "ncp", f"{n_strands}strand"],
    )


def s3_scaling_task() -> TuningTask:
    """S3スケーリング分析タスク.

    7→19→37→61→91本のスケーラビリティを評価する。
    """
    return TuningTask(
        name="s3_scaling_analysis",
        description="素線数スケーリング分析: 計算時間・DOF・反復数の増加率",
        params=[
            TuningParam(
                "n_strands",
                low=7,
                high=91,
                default=19,
                description="素線数（離散値: 7, 19, 37, 61, 91）",
            ),
        ],
        criteria=[
            AcceptanceCriterion(
                "total_time_s",
                op="lt",
                target=21600.0,  # 6時間 = 21600秒（最終目標）
                description="計算時間が6時間以内",
            ),
        ],
        tags=["s3", "scaling", "benchmark"],
    )


def s3_timing_breakdown_task(n_strands: int = 19) -> TuningTask:
    """S3工程別タイミング分析タスク."""
    return TuningTask(
        name=f"s3_timing_breakdown_{n_strands}strand",
        description=f"{n_strands}本撚りの工程別処理時間内訳分析",
        params=[],
        criteria=[
            AcceptanceCriterion(
                "linear_solve_ratio",
                op="lt",
                target=0.8,
                description="線形ソルバーが全体の80%未満（ボトルネック回避）",
            ),
        ],
        fixed_params={
            "n_strands": n_strands,
        },
        tags=["s3", "timing", "profiling"],
    )
