"""チューニングタスクフレームワーク.

CAE後処理のAIアシストによる容易化と、
実務最適化タスクへの発展を見据えた基盤スキーマ。

== 設計思想 ==

TuningTask は「何を・どう変えて・何で判定するか」を宣言的に定義する。
結果は TuningResult に記録され、検証プロット生成の入力となる。

この三段構成（Task定義 → 実行 → 可視化検証）により:
  - パラメータチューニングの再現性を保証
  - AI駆動の自動チューニングループの基礎を提供
  - CAE実務の最適化ワークフローを標準化

== 使用例 ==

    from xkep_cae_deprecated.tuning import TuningTask, TuningParam, AcceptanceCriterion

    task = TuningTask(
        name="19strand_ncp_convergence",
        description="19本撚りNCP収束パラメータ探索",
        params=[
            TuningParam("omega_max", 0.1, 0.5, default=0.3),
            TuningParam("al_relaxation", 0.001, 0.1, default=0.01),
        ],
        criteria=[
            AcceptanceCriterion("converged", op="eq", target=True),
            AcceptanceCriterion("max_penetration_ratio", op="lt", target=0.05),
        ],
    )
"""

from xkep_cae_deprecated.tuning.schema import (
    AcceptanceCriterion,
    TuningParam,
    TuningResult,
    TuningRun,
    TuningTask,
)

__all__ = [
    "AcceptanceCriterion",
    "TuningParam",
    "TuningResult",
    "TuningRun",
    "TuningTask",
]
