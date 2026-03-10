"""Penalty Strategy 具象実装.

ペナルティ剛性 k_pen の決定方法を Strategy として実装する。

設計仕様: xkep_cae/process/process-architecture.md §2.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess


@dataclass(frozen=True)
class PenaltyInput:
    """Penalty Strategy の入力."""

    step: int
    total_steps: int


@dataclass(frozen=True)
class PenaltyOutput:
    """Penalty Strategy の出力."""

    k_pen: float


class AutoBeamEIProcess(SolverProcess[PenaltyInput, PenaltyOutput]):
    """梁曲げ剛性 EI/L³ ベースのペナルティ剛性自動推定.

    推定式:
        linear: k_pen = scale * 12 * E * I / L³ / max(1, n_pairs)
        sqrt:   k_pen = scale * 12 * E * I / L³ / max(1, sqrt(n_pairs))
    """

    meta = ProcessMeta(name="AutoBeamEI", module="solve", version="0.1.0")

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
        self.beam_E = beam_E
        self.beam_I = beam_I
        self.L_elem = max(L_elem, 1e-30)
        self.n_contact_pairs = max(1, n_contact_pairs)
        self.scale = scale
        self.scaling = scaling
        self._k_pen = self._compute()

    def _compute(self) -> float:
        k_bend = 12.0 * self.beam_E * self.beam_I / (self.L_elem**3)
        if self.scaling == "sqrt":
            divisor = max(1.0, math.sqrt(self.n_contact_pairs))
        else:
            divisor = max(1.0, float(self.n_contact_pairs))
        return self.scale * k_bend / divisor

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """現在ステップのペナルティ剛性（一定値）."""
        return self._k_pen

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self.compute_k_pen(input_data.step, input_data.total_steps))


class AutoEALProcess(SolverProcess[PenaltyInput, PenaltyOutput]):
    """軸剛性 EA/L ベースのペナルティ剛性自動推定."""

    meta = ProcessMeta(name="AutoEAL", module="solve", version="0.1.0")

    def __init__(
        self,
        beam_E: float,
        beam_A: float,
        L_elem: float,
        *,
        scale: float = 1.0,
    ) -> None:
        self.beam_E = beam_E
        self.beam_A = beam_A
        self.L_elem = max(L_elem, 1e-30)
        self.scale = scale
        self._k_pen = self.scale * self.beam_E * self.beam_A / self.L_elem

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """現在ステップのペナルティ剛性（一定値）."""
        return self._k_pen

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self.compute_k_pen(input_data.step, input_data.total_steps))


class ManualPenaltyProcess(SolverProcess[PenaltyInput, PenaltyOutput]):
    """手動指定のペナルティ剛性.

    .. deprecated::
        status-140 で材料ベース自動推定が標準。
        後方互換のため残すが、新規コードでは AutoBeamEI を使用すること。
    """

    meta = ProcessMeta(
        name="ManualPenalty",
        module="solve",
        version="0.1.0",
        deprecated=True,
        deprecated_by="AutoBeamEIProcess",
    )

    def __init__(self, k_pen: float) -> None:
        self._k_pen = k_pen

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """現在ステップのペナルティ剛性（一定値）."""
        return self._k_pen

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self.compute_k_pen(input_data.step, input_data.total_steps))


class ContinuationPenaltyProcess(SolverProcess[PenaltyInput, PenaltyOutput]):
    """k_pen continuation: 段階的にペナルティ剛性を増加.

    初期ステップでは小さなペナルティ剛性で収束しやすくし、
    ステップが進むにつれてターゲットまで線形増加する。
    """

    meta = ProcessMeta(name="ContinuationPenalty", module="solve", version="0.1.0")

    def __init__(
        self,
        k_pen_target: float,
        *,
        start_fraction: float = 0.01,
        ramp_steps: int = 5,
    ) -> None:
        self.k_pen_target = k_pen_target
        self.start_fraction = start_fraction
        self.ramp_steps = max(1, ramp_steps)

    def compute_k_pen(self, step: int, total_steps: int) -> float:
        """現在ステップのペナルティ剛性.

        step 0 → start_fraction * k_pen_target
        step >= ramp_steps → k_pen_target
        """
        if step >= self.ramp_steps:
            return self.k_pen_target
        frac = self.start_fraction + (1.0 - self.start_fraction) * step / self.ramp_steps
        return self.k_pen_target * frac

    def process(self, input_data: PenaltyInput) -> PenaltyOutput:
        return PenaltyOutput(k_pen=self.compute_k_pen(input_data.step, input_data.total_steps))
