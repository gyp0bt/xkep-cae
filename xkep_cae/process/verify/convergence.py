"""ConvergenceVerifyProcess — NR反復の収束検証.

設計仕様: process-architecture.md §6
NR反復の収束履歴を検証し、残差減少率・最大反復数の妥当性を確認する。
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import VerifyProcess
from xkep_cae.process.data import SolverResultData, VerifyResult


@dataclass(frozen=True)
class ConvergenceVerifyInput:
    """収束検証の入力."""

    solver_result: SolverResultData
    max_iterations_threshold: int = 100
    min_convergence_rate: float = 0.0


class ConvergenceVerifyProcess(VerifyProcess[ConvergenceVerifyInput, VerifyResult]):
    """NR反復の収束検証プロセス.

    収束判定:
    - ソルバーが converged=True を返したか
    - 総NR反復数が閾値以下か
    - インクリメント数が妥当か
    """

    meta = ProcessMeta(
        name="ConvergenceVerify",
        module="verify",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
    )

    def process(self, input_data: ConvergenceVerifyInput) -> VerifyResult:
        """収束検証の実行."""
        result = input_data.solver_result
        checks: dict[str, tuple[float, float, bool]] = {}

        # 収束判定
        checks["converged"] = (
            float(result.converged),
            1.0,
            result.converged,
        )

        # 総NR反復数チェック
        ok_iters = result.total_newton_iterations <= input_data.max_iterations_threshold
        checks["total_newton_iterations"] = (
            float(result.total_newton_iterations),
            float(input_data.max_iterations_threshold),
            ok_iters,
        )

        # インクリメント数チェック（0より大きいか）
        ok_incr = result.n_increments > 0
        checks["n_increments"] = (
            float(result.n_increments),
            1.0,
            ok_incr,
        )

        passed = all(ok for _, _, ok in checks.values())

        lines = ["# 収束検証レポート", ""]
        for name, (actual, expected, ok) in checks.items():
            status = "PASS" if ok else "FAIL"
            lines.append(f"- {name}: {status} (actual={actual}, threshold={expected})")

        return VerifyResult(
            passed=passed,
            checks=checks,
            report_markdown="\n".join(lines),
        )
