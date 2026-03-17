"""ContactVerifyProcess — 接触状態の妥当性検証.

旧 __xkep_cae_deprecated/process/verify/contact.py の完全書き直し。
設計仕様: docs/verify.md
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.core import ProcessMeta, SolverResultData, VerifyProcess, VerifyResult


@dataclass(frozen=True)
class ContactVerifyInput:
    """接触検証の入力."""

    solver_result: SolverResultData
    max_penetration: float = 1e-3
    max_chattering_ratio: float = 0.3


class ContactVerifyProcess(VerifyProcess[ContactVerifyInput, VerifyResult]):
    """接触状態妥当性検証プロセス.

    検証項目:
    - 最大貫入量が閾値以下か
    - 接触チャタリング（ON/OFF切り替え）比率が閾値以下か
    """

    meta = ProcessMeta(
        name="ContactVerify",
        module="verify",
        version="1.0.0",
        document_path="docs/verify.md",
    )

    def process(self, input_data: ContactVerifyInput) -> VerifyResult:
        """接触検証の実行."""
        result = input_data.solver_result
        checks: dict[str, tuple[float, float, bool]] = {}

        diag = result.diagnostics

        if diag is not None and hasattr(diag, "max_penetration"):
            pen = float(diag.max_penetration)
            ok_pen = pen <= input_data.max_penetration
            checks["max_penetration"] = (pen, input_data.max_penetration, ok_pen)

        if diag is not None and hasattr(diag, "chattering_ratio"):
            chat = float(diag.chattering_ratio)
            ok_chat = chat <= input_data.max_chattering_ratio
            checks["chattering_ratio"] = (
                chat,
                input_data.max_chattering_ratio,
                ok_chat,
            )

        if not checks:
            checks["solver_converged"] = (
                float(result.converged),
                1.0,
                result.converged,
            )

        passed = all(ok for _, _, ok in checks.values())

        lines = ["# 接触検証レポート", ""]
        for name, (actual, threshold, ok) in checks.items():
            status = "PASS" if ok else "FAIL"
            lines.append(f"- {name}: {status} (actual={actual:.6e}, threshold={threshold:.6e})")

        return VerifyResult(
            passed=passed,
            checks=checks,
            report_markdown="\n".join(lines),
        )
