"""ファイバー1D材料則（Strategy 実装）.

Fiber1DMaterialStrategy Protocol の具象クラス群。
全ての evaluate() は (sigma, dsigma_deps, new_state) を返し、
入力 state を変更しない（frozen dataclass、C17 準拠）。

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md
参照実装: work/beam_hysteresis/01_kh_vs_friction_equivalence.py
          work/beam_hysteresis/05_smooth_teardrop.py

[← README](../../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True)
class MultiLayerFrictionDegrading1D:
    """N 層並列摩擦要素 + 弾性バックボーン + 接触剛性劣化.

    撚線ケーブルのヒステリシス（ティアドロップ形状）を再現する。
    各層 i は独立した (slip[i], slipped[i]) 状態を持ち、
    初回スリップ後に接触剛性が virgin → degraded に不可逆劣化する。

    数学的基盤:
    - KH ≡ 摩擦（work/beam_hysteresis/01_kh_vs_friction_equivalence.py）
    - N=150 + 対数間隔閾値 + 段階的剛性 → 滑らかなティアドロップ
      （work/beam_hysteresis/05_smooth_teardrop.py）

    Attributes:
        E_base: 弾性バックボーン剛性（常時寄与、EI_min 相当）
        k_virgin: 各層の virgin 接触剛性 (N,)
        k_degraded: 各層の劣化後接触剛性 (N,)。通常 β * k_virgin
        f_y: 各層の降伏力閾値 (N,)

    設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F2
    """

    E_base: float
    k_virgin: NDArray[np.floating]
    k_degraded: NDArray[np.floating]
    f_y: NDArray[np.floating]

    @property
    def n_layers(self) -> int:
        """摩擦層の数."""
        return len(self.k_virgin)

    def initial_state(self) -> Fiber1DState:
        """初期状態を生成（全層スリップなし）."""
        n = self.n_layers
        return Fiber1DState(
            slip=tuple(0.0 for _ in range(n)),
            slipped=tuple(False for _ in range(n)),
        )

    def evaluate(
        self,
        eps: float,
        state: Fiber1DState,
    ) -> tuple[float, float, Fiber1DState]:
        """応力・接線・新状態を返す.

        Return mapping アルゴリズム（05_smooth_teardrop.py の frozen 化）:
        1. σ = E_base * ε（弾性バックボーン）
        2. 各層 i について:
           a. k = k_degraded[i] if slipped[i] else k_virgin[i]
           b. trial = k * (ε - slip[i])
           c. |trial| ≤ f_y[i] → 弾性（スリップなし）
           d. |trial| > f_y[i] → slipped 化 + 劣化剛性で再計算
        3. dσ/dε = E_base + Σ(dσ_i/dε)

        Args:
            eps: 軸ひずみ
            state: 現在の状態 (slip, slipped)

        Returns:
            (sigma, E_tangent, new_state)
        """
        sigma = self.E_base * eps
        E_t = self.E_base

        slip_list = list(state.slip)
        slipped_list = list(state.slipped)
        changed = False

        for i in range(self.n_layers):
            k = self.k_degraded[i] if slipped_list[i] else self.k_virgin[i]
            trial = k * (eps - slip_list[i])

            if abs(trial) <= self.f_y[i]:
                sigma += trial
                E_t += k
            else:
                # 初回スリップ → slipped フラグ ON + 劣化剛性へ切替
                if not slipped_list[i]:
                    slipped_list[i] = True
                    changed = True

                k = float(self.k_degraded[i])
                trial = k * (eps - slip_list[i])

                if abs(trial) <= self.f_y[i]:
                    sigma += trial
                    E_t += k
                else:
                    s = math.copysign(1.0, trial)
                    excess = abs(trial) - self.f_y[i]
                    slip_list[i] = slip_list[i] + s * excess / k
                    sigma += s * float(self.f_y[i])
                    changed = True
                    # E_t += 0（スリッピング中は剛性寄与なし）

        if not changed:
            return sigma, E_t, state

        new_state = Fiber1DState(
            eps_p=state.eps_p,
            alpha=state.alpha,
            slip=tuple(slip_list),
            slipped=tuple(slipped_list),
        )
        return sigma, E_t, new_state
