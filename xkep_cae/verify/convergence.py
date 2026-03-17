"""ConvergenceVerifyProcess — NR反復の収束検証.

旧 __xkep_cae_deprecated/process/verify/convergence.py の完全書き直し。
設計仕様: docs/verify.md
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.core import ProcessMeta, SolverResultData, VerifyProcess, VerifyResult


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
    - インクリメント数が妥当か（> 0）
    """

    meta = ProcessMeta(
        name="ConvergenceVerify",
        module="verify",
        version="1.0.0",
        document_path="docs/verify.md",
    )

    def process(self, input_data: ConvergenceVerifyInput) -> VerifyResult:
        """収束検証の実行."""
        result = input_data.solver_result
        checks: dict[str, tuple[float, float, bool]] = {}

        checks["converged"] = (
            float(result.converged),
            1.0,
            result.converged,
        )

        ok_iters = result.total_newton_iterations <= input_data.max_iterations_threshold
        checks["total_newton_iterations"] = (
            float(result.total_newton_iterations),
            float(input_data.max_iterations_threshold),
            ok_iters,
        )

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
