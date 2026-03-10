"""Penalty Strategy 具象実装.

ペナルティ剛性 k_pen の決定方法を Strategy として実装する。

Phase 3 統合:
- create_penalty_strategy() ファクトリで solver_ncp.py の
  k_pen 決定ロジックを Strategy に移譲する。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

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


def create_penalty_strategy(
    *,
    k_pen: float = 0.0,
    manager: object | None = None,
    node_coords_ref: np.ndarray | None = None,
    connectivity: np.ndarray | None = None,
) -> AutoBeamEIProcess | AutoEALProcess | ManualPenaltyProcess | ContinuationPenaltyProcess:
    """solver_ncp.py の k_pen 決定ロジックを Strategy に移譲するファクトリ.

    solver_ncp.py (lines 1725-1810) の分岐ロジックを再現する。

    Args:
        k_pen: 手動指定のペナルティ剛性（>0 なら ManualPenaltyProcess）
        manager: ContactManager (config.beam_E 等を参照)
        node_coords_ref: 参照座標 (要素長推定用)
        connectivity: 要素コネクティビティ (要素長推定用)

    Returns:
        適切な PenaltyStrategy インスタンス
    """
    # 1. k_pen > 0 が明示指定されている場合 → Manual (deprecated)
    if k_pen > 0.0:
        warnings.warn(
            "k_pen の直接指定は deprecated です。"
            "beam_E, beam_I を設定して材料ベースの自動推定を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
        config = getattr(manager, "config", None)
        if config is not None and getattr(config, "k_pen_continuation", False):
            return ContinuationPenaltyProcess(
                k_pen_target=k_pen,
                start_fraction=getattr(config, "k_pen_continuation_start", 0.01),
                ramp_steps=getattr(config, "k_pen_continuation_steps", 5),
            )
        return ManualPenaltyProcess(k_pen=k_pen)

    if manager is None:
        return ManualPenaltyProcess(k_pen=1.0)

    config = getattr(manager, "config", None)
    if config is None:
        return ManualPenaltyProcess(k_pen=1.0)

    beam_E = getattr(config, "beam_E", 0.0)

    # 2. beam_E 未設定 → k_pen_scale フォールバック (deprecated)
    if beam_E <= 0.0:
        k_pen_scale = getattr(config, "k_pen_scale", 1.0)
        if k_pen_scale >= 1.0:
            warnings.warn(
                "k_pen_scale >= 1.0 はペナルティ剛性の直接指定（手動モード）です。"
                "beam_E, beam_I を設定して材料ベースの自動推定を使用してください。",
                DeprecationWarning,
                stacklevel=2,
            )
        return ManualPenaltyProcess(k_pen=k_pen_scale)

    # 3. 代表要素長の推定
    L_avg = 1.0
    if node_coords_ref is not None and connectivity is not None:
        L_elems = []
        for elem in connectivity:
            n1, n2 = int(elem[0]), int(elem[1])
            if n1 < len(node_coords_ref) and n2 < len(node_coords_ref):
                dxyz = node_coords_ref[n2] - node_coords_ref[n1]
                L_elems.append(float(np.linalg.norm(dxyz)))
        if L_elems:
            L_avg = float(np.mean(L_elems))
    L_avg = max(L_avg, 1e-30)

    k_pen_mode = getattr(config, "k_pen_mode", "beam_ei")
    k_pen_scale = getattr(config, "k_pen_scale", 0.1)
    n_pairs = max(1, getattr(manager, "n_pairs", 1))

    # 4. EA/L モード
    if k_pen_mode == "ea_l":
        beam_A = getattr(config, "beam_A", 1.0)
        base_strategy = AutoEALProcess(
            beam_E=beam_E,
            beam_A=beam_A,
            L_elem=L_avg,
            scale=k_pen_scale,
        )
    else:
        # beam_ei モード（デフォルト）
        beam_I = getattr(config, "beam_I", 1.0)
        k_pen_scaling = getattr(config, "k_pen_scaling", "linear")
        base_strategy = AutoBeamEIProcess(
            beam_E=beam_E,
            beam_I=beam_I,
            L_elem=L_avg,
            n_contact_pairs=n_pairs,
            scale=k_pen_scale,
            scaling=k_pen_scaling,
        )

    # 5. Continuation の判定
    if getattr(config, "k_pen_continuation", False):
        k_pen_base = base_strategy.compute_k_pen(0, 1)
        return ContinuationPenaltyProcess(
            k_pen_target=k_pen_base,
            start_fraction=getattr(config, "k_pen_continuation_start", 0.01),
            ramp_steps=getattr(config, "k_pen_continuation_steps", 5),
        )

    return base_strategy
