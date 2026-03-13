"""EnergyBalanceVerifyProcess — エネルギー収支検証.

設計仕様: process-architecture.md §6
外力仕事 ≈ 内部エネルギー + 接触散逸 の収支を検証する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import VerifyProcess
from xkep_cae.process.data import SolverResultData, VerifyResult


@dataclass(frozen=True)
class EnergyBalanceVerifyInput:
    """エネルギー収支検証の入力."""

    solver_result: SolverResultData
    f_ext: np.ndarray | None = None
    tolerance: float = 0.1  # 10% 許容


class EnergyBalanceVerifyProcess(VerifyProcess[EnergyBalanceVerifyInput, VerifyResult]):
    """エネルギー収支検証プロセス.

    外力仕事と変位の整合性を簡易チェックする。
    詳細なエネルギー収支は diagnostics から取得。
    """

    meta = ProcessMeta(
        name="EnergyBalanceVerify",
        module="verify",
        version="0.1.0",
        document_path="../docs/process-architecture.md",
    )

    def process(self, input_data: EnergyBalanceVerifyInput) -> VerifyResult:
        """エネルギー収支検証の実行."""
        result = input_data.solver_result
        checks: dict[str, tuple[float, float, bool]] = {}

        # 変位の有限性チェック
        u_max = float(np.max(np.abs(result.u)))
        ok_finite = np.all(np.isfinite(result.u))
        checks["displacement_finite"] = (u_max, 0.0, bool(ok_finite))

        # 外力仕事チェック（f_ext が与えられた場合）
        if input_data.f_ext is not None:
            work = float(np.dot(input_data.f_ext, result.u))
            checks["external_work"] = (work, 0.0, np.isfinite(work))

        # diagnostics からエネルギー収支を取得（利用可能な場合）
        if result.diagnostics is not None and hasattr(result.diagnostics, "energy_balance"):
            balance = result.diagnostics.energy_balance
            checks["energy_balance"] = (
                float(balance),
                0.0,
                abs(balance) < input_data.tolerance,
            )

        passed = all(ok for _, _, ok in checks.values())

        lines = ["# エネルギー収支検証レポート", ""]
        for name, (actual, expected, ok) in checks.items():
            status = "PASS" if ok else "FAIL"
            lines.append(f"- {name}: {status} (value={actual:.6e})")

        return VerifyResult(
            passed=passed,
            checks=checks,
            report_markdown="\n".join(lines),
        )
