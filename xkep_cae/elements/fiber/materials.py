"""ファイバー1D材料則（Strategy 実装）.

Fiber1DMaterialStrategy Protocol の具象クラス群。
全ての evaluate() は (sigma, dsigma_deps, new_state) を返し、
入力 state を変更しない（frozen dataclass、C17 準拠）。

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md
参照実装: work/beam_hysteresis/01_kh_vs_friction_equivalence.py

[← README](../../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from xkep_cae.elements.fiber.state import Fiber1DState


@dataclass(frozen=True)
class Elastic1D:
    """線形弾性1D材料（参照用）.

    σ = E·ε、接線 dσ/dε = E（常に一定）。
    状態変化なし。

    Attributes:
        E: ヤング率 [MPa]
    """

    E: float

    def evaluate(
        self,
        eps: float,
        state: Fiber1DState,
    ) -> tuple[float, float, Fiber1DState]:
        """応力・接線・新状態を返す.

        Args:
            eps: 軸ひずみ
            state: 現在の状態（不使用、そのまま返却）

        Returns:
            (sigma, E_tangent, new_state)
        """
        return self.E * eps, self.E, state


@dataclass(frozen=True)
class BilinearKinematicHardening1D:
    """Prager 移動硬化1D材料.

    降伏条件: |σ_trial - α| ≤ σ_y
    降伏時: return mapping で (eps_p, alpha) を更新。

    work/beam_hysteresis/01_kh_vs_friction_equivalence.py で
    KinematicHardening1D ≡ StrandFriction1D（数学的同型）が証明済み。

    Attributes:
        E: ヤング率 [MPa]
        sigma_y: 初期降伏応力 [MPa]
        H: 線形硬化係数 [MPa]（H=0 で完全弾塑性）
    """

    E: float
    sigma_y: float
    H: float

    def evaluate(
        self,
        eps: float,
        state: Fiber1DState,
    ) -> tuple[float, float, Fiber1DState]:
        """応力・接線・新状態を返す.

        Return mapping アルゴリズム:
        1. trial 応力計算: σ_trial = E(ε - ε_p)
        2. 降伏関数判定: |σ_trial - α| vs σ_y
        3. 弾性: そのまま返却
        4. 塑性: dgamma 計算 → (eps_p, alpha) 更新

        Args:
            eps: 軸ひずみ
            state: 現在の状態 (eps_p, alpha)

        Returns:
            (sigma, E_tangent, new_state)
        """
        sigma_trial = self.E * (eps - state.eps_p)
        eta = sigma_trial - state.alpha

        if abs(eta) <= self.sigma_y:
            # 弾性域
            return sigma_trial, self.E, state

        # 塑性域: return mapping
        sign = math.copysign(1.0, eta)
        dgamma = (abs(eta) - self.sigma_y) / (self.E + self.H)
        new_eps_p = state.eps_p + sign * dgamma
        new_alpha = state.alpha + sign * self.H * dgamma
        sigma = self.E * (eps - new_eps_p)

        # 一貫接線: E_t = E * H / (E + H)
        E_tangent = self.E * self.H / (self.E + self.H)

        new_state = Fiber1DState(
            eps_p=new_eps_p,
            alpha=new_alpha,
            slip=state.slip,
            slipped=state.slipped,
        )
        return sigma, E_tangent, new_state
