"""収束診断情報 Process.

ConvergenceDiagnosticsOutput を新パッケージに移植。
__xkep_cae_deprecated/contact/diagnostics.py からのコピー。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xkep_cae.core import ProcessMeta, SolverProcess


@dataclass(frozen=True)
class ConvergenceDiagnosticsOutput:
    """収束失敗時の標準化診断情報（純粋データ）."""

    step: int = 0
    load_frac: float = 0.0
    res_history: list[float] = field(default_factory=list)
    ncp_history: list[float] = field(default_factory=list)
    ncp_t_history: list[float] = field(default_factory=list)
    n_active_history: list[int] = field(default_factory=list)
    du_norm_history: list[float] = field(default_factory=list)
    max_du_dof_history: list[int] = field(default_factory=list)
    condition_number: float | None = None


@dataclass(frozen=True)
class DiagnosticsInput:
    """診断レポート生成の入力."""

    diagnostics: ConvergenceDiagnosticsOutput
    max_iter: int = 50


@dataclass(frozen=True)
class DiagnosticsOutput:
    """診断レポートの出力."""

    report: str


def _format_diagnostics_report(diag: ConvergenceDiagnosticsOutput, max_iter: int = 50) -> str:
    """診断レポートの文字列を生成する."""
    lines = [
        "=" * 60,
        "  NCP Solver Convergence Diagnostics",
        "=" * 60,
        f"  Step: {diag.step}, Load fraction: {diag.load_frac:.6f}",
        f"  Iterations: {len(diag.res_history)} / {max_iter}",
    ]

    if diag.condition_number is not None:
        lines.append(f"  Condition number: {diag.condition_number:.2e}")

    if diag.res_history:
        lines.append(f"  Final residual: {diag.res_history[-1]:.6e}")
        lines.append(f"  Min residual:   {min(diag.res_history):.6e}")

    if diag.n_active_history:
        lines.append(f"  Active pairs: {diag.n_active_history[-1]}")

    lines.append("=" * 60)
    return "\n".join(lines)


class DiagnosticsReportProcess(SolverProcess[DiagnosticsInput, DiagnosticsOutput]):
    """収束診断レポート生成 Process."""

    meta = ProcessMeta(
        name="DiagnosticsReport",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_friction.md",
    )

    def process(self, input_data: DiagnosticsInput) -> DiagnosticsOutput:
        report = _format_diagnostics_report(
            input_data.diagnostics,
            input_data.max_iter,
        )
        return DiagnosticsOutput(report=report)
