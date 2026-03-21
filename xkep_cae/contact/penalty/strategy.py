"""ペナルティ剛性決定 Strategy の具象実装.

PenaltyStrategy Protocol に従い、ペナルティ剛性 k_pen を決定する Process 群。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class PenaltyInput:
    """ペナルティ剛性計算の入力."""

    step: int
    total_steps: int


@dataclass(frozen=True)
class PenaltyOutput:
    """ペナルティ剛性計算の出力."""

    k_pen: float


# ── 具象 Process ──────────────────────────────────────────


class ConstantPenalty(SolverProcess[PenaltyInput, PenaltyOutput]):
    """定数ペナルティ剛性.

    k_pen を全ステップで一定値として返す。
    beam_E / beam_I が未指定の場合のフォールバック用。
    """

    meta = ProcessMeta(
        name="ConstantPenalty",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    def __init__(self, k_pen: float = 1.0) -> None:
        self._k_pen = k_pen

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """PenaltyStrategy Protocol."""
        return self._k_pen

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self._k_pen)


class AutoBeamEIPenalty(SolverProcess[PenaltyInput, PenaltyOutput]):
    """梁曲げ剛性 EI/L³ ベースのペナルティ剛性自動推定.

    推定式:
        k_pen = scale × 12EI / L³ / max(1, f(n_pairs))

    f = identity (linear) or sqrt (sqrt scaling).
    """

    meta = ProcessMeta(
        name="AutoBeamEIPenalty",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    def __init__(
        self,
        beam_E: float,
        beam_I: float,
        L_elem: float,
        *,
        n_contact_pairs: int = 1,
        scale: float = 0.1,
        scaling: str = "linear",
    ) -> None:
        if L_elem <= 0.0:
            raise ValueError(f"L_elem は正の値が必要: {L_elem}")
        self._beam_E = beam_E
        self._beam_I = beam_I
        self._L_elem = L_elem
        self._n_contact_pairs = max(1, n_contact_pairs)
        self._scale = scale
        self._scaling = scaling
        self._k_pen = self._compute()

    def _compute(self) -> float:
        k_bend = 12.0 * self._beam_E * self._beam_I / (self._L_elem**3)
        if self._scaling == "sqrt":
            divisor = max(1.0, math.sqrt(self._n_contact_pairs))
        else:
            divisor = max(1.0, float(self._n_contact_pairs))
        return self._scale * k_bend / divisor

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """PenaltyStrategy Protocol."""
        return self._k_pen

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self._k_pen)


class AutoEALPenalty(SolverProcess[PenaltyInput, PenaltyOutput]):
    """軸剛性 EA/L ベースのペナルティ剛性自動推定."""

    meta = ProcessMeta(
        name="AutoEALPenalty",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    def __init__(
        self,
        beam_E: float,
        beam_A: float,
        L_elem: float,
        *,
        scale: float = 1.0,
    ) -> None:
        if L_elem <= 0.0:
            raise ValueError(f"L_elem は正の値が必要: {L_elem}")
        self._k_pen = scale * beam_E * beam_A / L_elem

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """PenaltyStrategy Protocol."""
        return self._k_pen

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self._k_pen)


class ContinuationPenalty(SolverProcess[PenaltyInput, PenaltyOutput]):
    """段階的にペナルティ剛性を増加する continuation 方式.

    mode:
        geometric: k(step) = start × target × ratio^step
        linear:    k(step) = target × (start + (1-start) × step / ramp_steps)
    """

    meta = ProcessMeta(
        name="ContinuationPenalty",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    def __init__(
        self,
        k_pen_target: float,
        *,
        start_fraction: float = 0.01,
        ramp_steps: int = 5,
        mode: str = "geometric",
    ) -> None:
        self._k_pen_target = k_pen_target
        self._start_fraction = max(start_fraction, 1e-30)
        self._ramp_steps = max(1, ramp_steps)
        self._mode = mode

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """PenaltyStrategy Protocol."""
        if step >= self._ramp_steps:
            return self._k_pen_target

        if self._mode == "geometric":
            ratio = (1.0 / self._start_fraction) ** (1.0 / self._ramp_steps)
            return min(
                self._k_pen_target * self._start_fraction * (ratio**step),
                self._k_pen_target,
            )
        # linear
        frac = self._start_fraction + (1.0 - self._start_fraction) * step / self._ramp_steps
        return self._k_pen_target * frac

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self.compute_k_pen(input_data.step, input_data.total_steps))


@dataclass(frozen=True)
class DynamicPenaltyEstimateInput:
    """動的ペナルティ剛性推定の入力.

    Generalized-α 法の有効質量剛性 c0*M_ii をベースに、
    動的解析に適切な k_pen を推定する。
    """

    rho_inf: float  # Generalized-α のスペクトル半径
    dt: float  # 時間刻み [s]
    rho: float  # 材料密度 [ton/mm³]
    A: float  # 断面積 [mm²]
    L_elem: float  # 代表要素長 [mm]
    scale: float = 0.5  # c0*M_ii に対するスケール（デフォルト50%）


@dataclass(frozen=True)
class DynamicPenaltyEstimateOutput:
    """動的ペナルティ剛性推定の出力."""

    k_pen: float
    c0: float  # 有効質量係数 1/(beta*dt²)
    m_ii: float  # 代表集中質量
    c0_m_ii: float  # c0*m_ii（動的有効剛性スケール）


class DynamicPenaltyEstimateProcess(
    SolverProcess[DynamicPenaltyEstimateInput, DynamicPenaltyEstimateOutput],
):
    """動的解析の有効質量剛性 c0*M_ii ベースでペナルティ剛性を推定.

    推定式:
        alpha_m = (2*rho_inf - 1) / (rho_inf + 1)
        alpha_f = rho_inf / (rho_inf + 1)
        beta = 0.25 * (1 - alpha_m + alpha_f)²
        c0 = 1 / (beta * dt²)
        m_ii = rho * A * L_elem / 2    （集中質量の代表値）
        k_pen = scale * c0 * m_ii

    scale=0.5 の根拠:
        exact_tangent=True 時に K_eff が正定値を保つ条件:
        k_pen < (1-alpha_m) * c0 * M_ii
        scale=0.5 なら十分なマージンで正定値維持。

    status-218 で特定: 静的梁剛性ベースの k_pen は動的有効剛性に対して6桁小さく、
    接触力が慣性力に負けてワイヤがジグを貫通する。
    """

    meta = ProcessMeta(
        name="DynamicPenaltyEstimate",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    def process(self, input_data: DynamicPenaltyEstimateInput) -> DynamicPenaltyEstimateOutput:
        rho_inf = input_data.rho_inf
        dt = input_data.dt

        # Chung & Hulbert (1993) パラメータ
        alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        alpha_f = rho_inf / (rho_inf + 1.0)
        beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2

        c0 = 1.0 / (beta * dt**2)
        m_ii = input_data.rho * input_data.A * input_data.L_elem / 2.0

        c0_m_ii = c0 * m_ii
        k_pen = input_data.scale * c0_m_ii

        return DynamicPenaltyEstimateOutput(
            k_pen=k_pen,
            c0=c0,
            m_ii=m_ii,
            c0_m_ii=c0_m_ii,
        )


class AutoSmoothingDeltaProcess(SolverProcess[PenaltyInput, PenaltyOutput]):
    """梁半径ベースの smoothing_delta 自動推定.

    推定式:
        ε = α × r_min
        δ = 1/ε

    α のデフォルト 2e-4 は「表面粗さオーダー」:
    - r=1.0mm (wire_diameter=2.0mm) → ε=0.0002mm, δ=5000
    - r=0.2mm (細線) → ε=0.00004mm, δ=25000
    物理的に遷移幅が表面粗さ以下なら、操作点は常に線形領域に入り
    NR 2次収束が保証される（status-223）。
    """

    meta = ProcessMeta(
        name="AutoSmoothingDelta",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    _DEFAULT_ALPHA = 2e-4

    def __init__(
        self,
        r_min: float,
        *,
        alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        if r_min <= 0.0:
            raise ValueError(f"r_min は正の値が必要: {r_min}")
        if alpha <= 0.0:
            raise ValueError(f"alpha は正の値が必要: {alpha}")
        self._r_min = r_min
        self._alpha = alpha
        self._epsilon = alpha * r_min
        self._delta = 1.0 / self._epsilon

    @property
    def delta(self) -> float:
        """推定された smoothing_delta."""
        return self._delta

    @property
    def epsilon(self) -> float:
        """推定された遷移幅 ε [mm]."""
        return self._epsilon

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self._delta)


def _estimate_smoothing_delta(
    radii: object,
    *,
    alpha: float = AutoSmoothingDeltaProcess._DEFAULT_ALPHA,
) -> float:
    """梁半径配列から smoothing_delta を推定するユーティリティ.

    Args:
        radii: スカラー or ndarray の梁半径
        alpha: ε/r 比率（デフォルト 2e-4）

    Returns:
        smoothing_delta: δ = 1/(α × r_min)。
        半径が 0 以下の要素（ジグ等）は除外する。
        有効な半径がない場合は 0.0 を返す（フォールバック: smoothing なし）。
    """
    import numpy as np

    r_arr = np.atleast_1d(np.asarray(radii, dtype=float)).ravel()
    # 正の半径のみ（ジグ要素は r=0）
    positive = r_arr[r_arr > 0.0]
    if len(positive) == 0:
        return 0.0
    r_min = float(np.min(positive))
    proc = AutoSmoothingDeltaProcess(r_min=r_min, alpha=alpha)
    return proc.delta


def _create_penalty_strategy(
    *,
    k_pen: float = 1.0,
    beam_E: float = 0.0,
    beam_I: float = 0.0,
    beam_L: float = 0.0,
) -> AutoBeamEIPenalty | ConstantPenalty:
    """Penalty Strategy ファクトリ.

    Args:
        k_pen: 定数ペナルティ剛性（beam パラメータ未指定時のフォールバック）
        beam_E: ヤング率
        beam_I: 断面二次モーメント
        beam_L: 代表要素長

    Returns:
        beam_E, beam_I, beam_L が全て正なら AutoBeamEIPenalty、
        そうでなければ ConstantPenalty(k_pen) を返す。
    """
    if beam_E > 0.0 and beam_I > 0.0 and beam_L > 0.0:
        return AutoBeamEIPenalty(beam_E=beam_E, beam_I=beam_I, L_elem=beam_L)
    return ConstantPenalty(k_pen=k_pen)
