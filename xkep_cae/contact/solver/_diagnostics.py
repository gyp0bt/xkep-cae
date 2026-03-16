"""収束診断情報（プライベート）.

ConvergenceDiagnostics を新パッケージに移植。
xkep_cae_deprecated/contact/diagnostics.py からのコピー。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """収束失敗時の標準化診断情報.

    Newton反復の履歴情報を収集し、収束失敗の原因特定を支援する。
    """

    step: int = 0
    load_frac: float = 0.0
    res_history: list[float] = field(default_factory=list)
    ncp_history: list[float] = field(default_factory=list)
    ncp_t_history: list[float] = field(default_factory=list)
    n_active_history: list[int] = field(default_factory=list)
    du_norm_history: list[float] = field(default_factory=list)
    max_du_dof_history: list[int] = field(default_factory=list)
    condition_number: float | None = None

    def format_report(self, max_iter: int = 50) -> str:
        """診断レポートの文字列を生成する."""
        lines = [
            "=" * 60,
            "  NCP Solver Convergence Diagnostics",
            "=" * 60,
            f"  Step: {self.step}, Load fraction: {self.load_frac:.6f}",
            f"  Iterations: {len(self.res_history)} / {max_iter}",
        ]

        if self.condition_number is not None:
            lines.append(f"  Condition number: {self.condition_number:.2e}")

        if self.res_history:
            lines.append(f"  Final residual: {self.res_history[-1]:.6e}")
            lines.append(f"  Min residual:   {min(self.res_history):.6e}")

        if self.n_active_history:
            lines.append(f"  Active pairs: {self.n_active_history[-1]}")

        lines.append("=" * 60)
        return "\n".join(lines)
