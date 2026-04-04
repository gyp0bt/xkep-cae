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
    status: str  # "active" / "inactive" / "sliding"


@dataclass(frozen=True)
class NRIterationSnapshot:
    """NR1反復の詳細スナップショット（チャタリング内訳分析用, status-287）.

    NR各反復で残差の内訳（接触DOF vs 構造DOF）とペア状態遷移を記録し、
    不収束の原因を4タイプに分類する:
    - Type A: 接触活性集合の振動（ペアがON↔OFF）
    - Type B: 摩擦状態の振動（stick↔slide切替）
    - Type C: 構造系の収束不良（接触以外のDOFが支配的）
    - Type D: 接線剛性の不整合（収束率 > 0.9）
    """

    att: int  # NR反復番号
    res_ratio: float  # ||R_t||/||f||
    n_active: int  # 活性ペア数
    n_sliding: int  # 滑りペア数
    n_sticking: int  # 固着ペア数
    contact_res_norm: float  # 接触DOFの残差ノルム
    structural_res_norm: float  # 非接触DOFの残差ノルム
    active_set_changed: bool  # 前反復から活性集合変化したか
    friction_state_changed: bool  # 摩擦状態が変化したか（stick↔slide）
    pairs_activated: int  # 新たにONになったペア数
    pairs_deactivated: int  # OFFになったペア数
    pairs_stick_to_slide: int  # stick→slideになったペア数
    pairs_slide_to_stick: int  # slide→stickになったペア数
    convergence_rate: float  # 残差比 r_i / r_{i-1}（1.0 = 初回）


def classify_chattering_type(snap: NRIterationSnapshot) -> str:
    """NR反復スナップショットからチャタリングタイプを分類.

    Returns:
        "A" (活性集合振動), "B" (摩擦状態振動),
        "C" (構造系不良), "D" (接線剛性不整合),
        "E" (接触力値振動: 活性集合固定だが接触DOF残差支配),
        or 組合せ (e.g. "A+B")
    """
    types: list[str] = []
    contact_dominant = snap.contact_res_norm > snap.structural_res_norm
    if snap.active_set_changed and contact_dominant:
        types.append("A")
    if snap.friction_state_changed and contact_dominant:
        types.append("B")
    if not contact_dominant and snap.structural_res_norm > 0:
        types.append("C")
    if snap.convergence_rate > 0.9 and snap.att >= 2:
        types.append("D")
    # Type E: 接触力値振動（status-287 分析で発見）
    # 活性集合が固定（A/Bなし）なのに接触DOF残差が支配的。
    # 接触力の「値」が振動している（ペナルティ力の非線形性 or 摩擦力の方向振動）。
    if (
        contact_dominant
        and not snap.active_set_changed
        and not snap.friction_state_changed
        and snap.att >= 2
    ):
        types.append("E")
    return "+".join(types) if types else "-"


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
    condition_number_history: list[float] = field(default_factory=list)
    min_eigenvalue_history: list[float] = field(default_factory=list)
    max_eigenvalue_history: list[float] = field(default_factory=list)
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
    # NR反復レベル詳細スナップショット（チャタリング内訳分析用, status-287）
    nr_iteration_snapshots: list[NRIterationSnapshot] = field(default_factory=list)


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
