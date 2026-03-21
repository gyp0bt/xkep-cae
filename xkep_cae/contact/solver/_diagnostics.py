"""収束診断情報 Process.

ConvergenceDiagnosticsOutput を新パッケージに移植済み（status-207 で旧コード完全削除）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xkep_cae.core import ProcessMeta, SolverProcess


@dataclass(frozen=True)
class PairDiagnosticsOutput:
    """1接触ペアの診断スナップショット."""

    pair_id: int
    elem_a: int
    elem_b: int
    gap: float
    p_n: float
    status: str  # "active" / "inactive" / "slipping"


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
    # エネルギー診断（動的解析用）
    kinetic_energy: float = 0.0
    strain_energy: float = 0.0
    total_energy: float = 0.0
    energy_ratio: float = 1.0  # E_current / E_initial
    # ペア別接触診断（NR反復ごとのスナップショット）
    pair_snapshots: list[list[PairDiagnosticsOutput]] = field(default_factory=list)
    # 収束率診断（連続2反復の残差比 r_{i}/r_{i-1}）
    convergence_rate_history: list[float] = field(default_factory=list)
    # 最終的な収束/非収束
    converged: bool = False
    n_attempts: int = 0


@dataclass(frozen=True)
class IncrementDiagnosticsOutput:
    """1インクリメントの完全な診断スナップショット.

    毎インクリメントで生成し、全履歴を SolverResultData に蓄積。
    """

    step: int
    load_frac: float
    converged: bool
    n_attempts: int
    n_active: int
    # 残差ノルム（最終NR反復）
    final_residual: float = 0.0
    # 収束率（最終2反復の残差比、2次収束なら << 1）
    convergence_rate: float = 1.0
    # 変位増分ノルム
    du_norm: float = 0.0
    # エネルギー情報
    kinetic_energy: float = 0.0
    strain_energy: float = 0.0
    total_energy: float = 0.0
    energy_ratio: float = 1.0
    # 接触状態サマリ
    n_active_pairs: int = 0
    n_sliding_pairs: int = 0
    n_sticking_pairs: int = 0
    # 接触力ノルム
    contact_force_norm: float = 0.0
    # カットバック回数（このインクリメントで発生した場合）
    cutback_count: int = 0
    # 時間増分（動的解析用）
    dt: float = 0.0


@dataclass(frozen=True)
class DiagnosticsInput:
    """診断レポート生成の入力."""

    diagnostics: ConvergenceDiagnosticsOutput
    max_attempts: int = 50


@dataclass(frozen=True)
class DiagnosticsOutput:
    """診断レポートの出力."""

    report: str


def _format_diagnostics_report(diag: ConvergenceDiagnosticsOutput, max_attempts: int = 50) -> str:
    """診断レポートの文字列を生成する."""
    lines = [
        "=" * 60,
        "  NCP Solver Convergence Diagnostics",
        "=" * 60,
        f"  Step: {diag.step}, Load fraction: {diag.load_frac:.6f}",
        f"  Attempts: {len(diag.res_history)} / {max_attempts}",
    ]

    if diag.condition_number is not None:
        lines.append(f"  Condition number: {diag.condition_number:.2e}")

    if diag.res_history:
        lines.append(f"  Final residual: {diag.res_history[-1]:.6e}")
        lines.append(f"  Min residual:   {min(diag.res_history):.6e}")

    if diag.n_active_history:
        lines.append(f"  Active pairs: {diag.n_active_history[-1]}")

    if diag.total_energy > 0.0 or diag.kinetic_energy > 0.0:
        lines.append(f"  Kinetic energy:  {diag.kinetic_energy:.6e}")
        lines.append(f"  Strain energy:   {diag.strain_energy:.6e}")
        lines.append(f"  Total energy:    {diag.total_energy:.6e}")
        lines.append(f"  Energy ratio:    {diag.energy_ratio:.4f}")

    # ペア別接触診断（最終反復のスナップショット）
    if diag.pair_snapshots:
        last_snap = diag.pair_snapshots[-1]
        active_pairs = [p for p in last_snap if p.status != "inactive"]
        if active_pairs:
            lines.append(f"  Contact pairs (active={len(active_pairs)}):")
            for p in active_pairs[:10]:  # 最大10ペアまで表示
                lines.append(
                    f"    pair {p.pair_id}: "
                    f"elem({p.elem_a},{p.elem_b}) "
                    f"gap={p.gap:.4e} p_n={p.p_n:.4e} [{p.status}]"
                )
            if len(active_pairs) > 10:
                lines.append(f"    ... and {len(active_pairs) - 10} more")

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
            input_data.max_attempts,
        )
        return DiagnosticsOutput(report=report)
