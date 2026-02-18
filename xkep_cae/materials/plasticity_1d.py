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

    def sigma_y(self, alpha: float) -> float:
        """降伏応力を返す."""
        return self.sigma_y0 + self.H_iso * alpha

    def dR_dalpha(self, alpha: float) -> float:
        """硬化係数（勾配）を返す."""
        return self.H_iso


@dataclass
class TabularIsotropicHardening:
    """テーブル補間型等方硬化（区分線形）.

    Abaqus *PLASTIC テーブル形式の降伏応力-塑性ひずみ表データを
    区分線形補間して降伏応力と硬化係数を返す。

    テーブル最終点を超える領域では降伏応力を最終値で一定とする
    （完全塑性、Abaqus互換）。

    Attributes:
        table: [(sigma_y, eps_p), ...] 降伏応力-塑性ひずみの表データ。
            eps_p は単調増加であること。
    """

    table: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.table) < 1:
            raise ValueError("テーブルは最低1点必要")
        # eps_p の単調増加チェック
        for i in range(len(self.table) - 1):
            if self.table[i + 1][1] < self.table[i][1]:
                raise ValueError(
                    f"eps_p は単調増加: table[{i}]={self.table[i][1]}, "
                    f"table[{i + 1}]={self.table[i + 1][1]}"
                )

    @property
    def sigma_y0(self) -> float:
        """初期降伏応力."""
        return self.table[0][0]

    def sigma_y(self, alpha: float) -> float:
        """区分線形補間で降伏応力を返す."""
        table = self.table
        if len(table) == 1 or alpha <= table[0][1]:
            return table[0][0]
        for i in range(len(table) - 1):
            sy_i, ep_i = table[i]
            sy_next, ep_next = table[i + 1]
            if alpha <= ep_next:
                if ep_next > ep_i:
                    t = (alpha - ep_i) / (ep_next - ep_i)
                    return sy_i + t * (sy_next - sy_i)
                return sy_i
        # テーブル範囲外: 最終値で一定（完全塑性）
        return table[-1][0]

    def dR_dalpha(self, alpha: float) -> float:
        """硬化係数（テーブルの傾き）を返す."""
        table = self.table
        if len(table) == 1 or alpha <= table[0][1]:
            if len(table) > 1 and table[1][1] > table[0][1]:
                return (table[1][0] - table[0][0]) / (table[1][1] - table[0][1])
            return 0.0
        for i in range(len(table) - 1):
            ep_i = table[i][1]
            ep_next = table[i + 1][1]
            if alpha <= ep_next:
                if ep_next > ep_i:
                    return (table[i + 1][0] - table[i][0]) / (ep_next - ep_i)
                return 0.0
        # テーブル範囲外: 硬化なし
        return 0.0


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
        sigma_y0 = self.iso.sigma_y0
        C_kin = self.kin.C_kin
        gamma_kin = self.kin.gamma_kin

        # --- Step 1: 弾性試行 ---
        eps_e_trial = strain - state.eps_p
        sigma_trial = E * eps_e_trial
        xi_trial = sigma_trial - state.beta
        sigma_y_n = self.iso.sigma_y(state.alpha)
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

        # 線形硬化（IsotropicHardening）かつ動的回復なしの場合は閉形式
        _is_linear = isinstance(self.iso, IsotropicHardening)

        if gamma_kin == 0.0 and _is_linear:
            # Closed-form（線形等方硬化 + 線形移動硬化 or 移動硬化なし）
            H_iso = self.iso.H_iso
            dg = f_trial / (E + H_iso + C_kin)
            beta_new = state.beta + C_kin * dg * sign_xi
        else:
            # Newton 反復（テーブル硬化 or Armstrong-Frederick 動的回復）
            H_iso_init = self.iso.dR_dalpha(state.alpha)
            dg = f_trial / (E + H_iso_init + C_kin)  # 初期推定
            beta_new = state.beta  # 初期化（ループ内で更新）
            for _ in range(50):
                if gamma_kin == 0.0:
                    beta_new = state.beta + C_kin * dg * sign_xi
                    sigma_new = sigma_trial - E * dg * sign_xi
                    xi_new = sigma_new - beta_new
                else:
                    theta = 1.0 / (1.0 + gamma_kin * dg)
                    beta_new = (state.beta + C_kin * dg * sign_xi) * theta
                    sigma_new = sigma_trial - E * dg * sign_xi
                    xi_new = sigma_new - beta_new

                alpha_trial = state.alpha + dg
                f_val = abs(xi_new) - self.iso.sigma_y(alpha_trial)

                if abs(f_val) < 1e-12 * sigma_y0:
                    break

                # df/d(dg)
                H_iso_curr = self.iso.dR_dalpha(alpha_trial)
                if gamma_kin > 0.0:
                    H_kin_eff = (C_kin - gamma_kin * beta_new * sign_xi) * theta
                else:
                    H_kin_eff = C_kin
                df_ddg = -(E + H_iso_curr + H_kin_eff)
                dg -= f_val / df_ddg
                dg = max(dg, 0.0)

        # --- Step 4: 状態更新 ---
        eps_p_new = state.eps_p + dg * sign_xi
        alpha_new = state.alpha + dg
        sigma = E * (strain - eps_p_new)

        # --- Step 5: Consistent tangent ---
        H_iso_final = self.iso.dR_dalpha(alpha_new)
        if gamma_kin == 0.0:
            H_bar = H_iso_final + C_kin
        else:
            theta = 1.0 / (1.0 + gamma_kin * dg)
            H_bar = H_iso_final + theta * (C_kin - gamma_kin * beta_new * sign_xi)

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
