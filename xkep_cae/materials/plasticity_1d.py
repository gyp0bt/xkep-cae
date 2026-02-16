"""1次元弾塑性構成則.

等方硬化（線形）+ 移動硬化（Armstrong-Frederick）の組み合わせ。
Return mapping アルゴリズムと consistent tangent modulus を提供する。

参考文献:
  - Simo & Hughes (1998) "Computational Inelasticity", Ch.1-2
  - de Souza Neto et al. (2008) "Computational Methods for Plasticity", Ch.6-7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from xkep_cae.core.state import PlasticState1D


class ReturnMappingResult(NamedTuple):
    """Return mapping の結果."""

    stress: float
    tangent: float
    state_new: PlasticState1D


@dataclass
class IsotropicHardening:
    """線形等方硬化パラメータ.

    降伏応力: sigma_y(alpha) = sigma_y0 + H_iso * alpha

    Attributes:
        sigma_y0: 初期降伏応力（正値）
        H_iso: 等方硬化係数（線形硬化勾配）
    """

    sigma_y0: float
    H_iso: float = 0.0


@dataclass
class KinematicHardening:
    """Armstrong-Frederick 移動硬化パラメータ.

    背応力の発展式（暗黙的積分）:
      beta_new = (beta_n + C_kin * dg * sign_xi) / (1 + gamma_kin * dg)

    C_kin = 0, gamma_kin = 0 の場合は移動硬化なし。

    Attributes:
        C_kin: 移動硬化係数
        gamma_kin: 回復項係数（dynamic recovery）
    """

    C_kin: float = 0.0
    gamma_kin: float = 0.0


class Plasticity1D:
    """1D弾塑性構成則.

    等方硬化 + 移動硬化（Armstrong-Frederick）の return mapping。
    1D 問題なので、gamma_kin = 0 の場合は closed-form で解ける。

    Args:
        E: ヤング率
        iso: 等方硬化パラメータ
        kin: 移動硬化パラメータ（None = 移動硬化なし）
    """

    def __init__(
        self,
        E: float,
        iso: IsotropicHardening,
        kin: KinematicHardening | None = None,
    ) -> None:
        if E <= 0:
            raise ValueError(f"ヤング率 E は正値: {E}")
        if iso.sigma_y0 <= 0:
            raise ValueError(f"降伏応力 sigma_y0 は正値: {iso.sigma_y0}")
        self.E = E
        self.iso = iso
        self.kin = kin if kin is not None else KinematicHardening()

    @property
    def sigma_y0(self) -> float:
        """初期降伏応力."""
        return self.iso.sigma_y0

    def return_mapping(
        self,
        strain: float,
        state: PlasticState1D,
    ) -> ReturnMappingResult:
        """Return mapping アルゴリズム.

        Args:
            strain: 全歪み epsilon
            state: 前ステップの収束した塑性状態（変更されない）

        Returns:
            ReturnMappingResult: (応力, consistent tangent, 新状態)
        """
        E = self.E
        H_iso = self.iso.H_iso
        sigma_y0 = self.iso.sigma_y0
        C_kin = self.kin.C_kin
        gamma_kin = self.kin.gamma_kin

        # --- Step 1: 弾性試行 ---
        eps_e_trial = strain - state.eps_p
        sigma_trial = E * eps_e_trial
        xi_trial = sigma_trial - state.beta
        sigma_y_n = sigma_y0 + H_iso * state.alpha
        f_trial = abs(xi_trial) - sigma_y_n

        # --- Step 2: 弾性判定 ---
        # 浮動小数点誤差で降伏面上 (f_trial≈0) が塑性と判定されるのを防ぐ。
        # dg ≈ 0 でも D_ep は E と不連続になるため、NR 発散の原因となる。
        if f_trial <= 1e-10 * sigma_y_n:
            return ReturnMappingResult(
                stress=sigma_trial,
                tangent=E,
                state_new=state.copy(),
            )

        # --- Step 3: 塑性修正 ---
        sign_xi = 1.0 if xi_trial >= 0.0 else -1.0

        if gamma_kin == 0.0:
            # Closed-form（線形移動硬化 or 移動硬化なし）
            dg = f_trial / (E + H_iso + C_kin)
            beta_new = state.beta + C_kin * dg * sign_xi
        else:
            # Armstrong-Frederick: Newton 反復
            dg = f_trial / (E + H_iso + C_kin)  # 初期推定
            for _ in range(50):
                theta = 1.0 / (1.0 + gamma_kin * dg)
                beta_new = (state.beta + C_kin * dg * sign_xi) * theta
                sigma_new = sigma_trial - E * dg * sign_xi
                xi_new = sigma_new - beta_new
                f_val = abs(xi_new) - (sigma_y0 + H_iso * (state.alpha + dg))

                if abs(f_val) < 1e-12 * sigma_y0:
                    break

                # df/d(dg)
                H_kin_eff = (C_kin - gamma_kin * beta_new * sign_xi) * theta
                df_ddg = -(E + H_iso + H_kin_eff)
                dg -= f_val / df_ddg
                dg = max(dg, 0.0)

        # --- Step 4: 状態更新 ---
        eps_p_new = state.eps_p + dg * sign_xi
        alpha_new = state.alpha + dg
        sigma = E * (strain - eps_p_new)

        # --- Step 5: Consistent tangent ---
        if gamma_kin == 0.0:
            H_bar = H_iso + C_kin
        else:
            theta = 1.0 / (1.0 + gamma_kin * dg)
            H_bar = H_iso + theta * (C_kin - gamma_kin * beta_new * sign_xi)

        if E + H_bar > 0.0:
            D_ep = E * H_bar / (E + H_bar)
        else:
            D_ep = 0.0  # 完全弾塑性 (H=0)

        state_new = PlasticState1D(
            eps_p=eps_p_new,
            alpha=alpha_new,
            beta=beta_new,
        )
        return ReturnMappingResult(
            stress=sigma,
            tangent=D_ep,
            state_new=state_new,
        )
