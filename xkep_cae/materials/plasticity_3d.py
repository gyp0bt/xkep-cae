"""3次元弾塑性構成則（von Mises, J2 塑性）.

平面ひずみ（3成分 Voigt: [σxx, σyy, σxy]）での von Mises 降伏判定、
radial return mapping、および consistent tangent を提供する。

降伏関数:
  f = sqrt(3/2) ||dev(xi)|| - sigma_y(alpha)
  xi = sigma - beta

consistent tangent (de Souza Neto Box 7.4):
  D_ep = kappa(1x1) + 2mu a_bar I_dev_sym - 2mu a_hat (nxn)
  a_bar = 1 - 3mu dg / q_trial
  a_hat = a_bar - 1/(1 + H'/(3mu))

参考文献:
  - Simo & Taylor (1985) CMAME 48, 101-118.
  - de Souza Neto et al. (2008) Ch.7-8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from xkep_cae.core.state import PlasticState3D

# ---------------------------------------------------------------------------
# 定数（4成分 Voigt: [xx, yy, zz, xy]）
# ---------------------------------------------------------------------------

# 偏差射影行列（応力空間）
_P_DEV_4 = np.array(
    [
        [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0],
        [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0],
        [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# テンソルノルム重み: ||s||^2 = s_xx^2+s_yy^2+s_zz^2+2*s_xy^2
_W_4 = np.array([1.0, 1.0, 1.0, 2.0], dtype=float)

# 体積方向 m=[1,1,1,0]
_M_4 = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)

# 対称恒等 I_sym (sigma-epsilon Voigt): diag(1,1,1,1/2)
_I_SYM_4 = np.diag([1.0, 1.0, 1.0, 0.5])

# 偏差射影（sigma-epsilon Voigt 空間）
_P_DEV_SYM_4 = _I_SYM_4 - (1.0 / 3.0) * np.outer(_M_4, _M_4)

# 平面ひずみサブ行列インデックス
_PS_IDX = [0, 1, 3]


# ---------------------------------------------------------------------------
# Result / Hardening
# ---------------------------------------------------------------------------


class ReturnMappingResult3D(NamedTuple):
    """3D return mapping の結果."""

    stress: np.ndarray  # (3,) [sigma_xx, sigma_yy, sigma_xy]
    tangent: np.ndarray  # (3,3) consistent tangent
    state_new: PlasticState3D


@dataclass
class IsotropicHardening3D:
    """等方硬化パラメータ.

    sigma_y(alpha) = sigma_y0 + H_iso*alpha + Q_inf*(1-exp(-b_voce*alpha))
    """

    sigma_y0: float
    H_iso: float = 0.0
    Q_inf: float = 0.0
    b_voce: float = 0.0

    def sigma_y(self, alpha: float) -> float:
        R = self.H_iso * alpha
        if self.Q_inf > 0.0 and self.b_voce > 0.0:
            R += self.Q_inf * (1.0 - np.exp(-self.b_voce * alpha))
        return self.sigma_y0 + R

    def dR_dalpha(self, alpha: float) -> float:
        H = self.H_iso
        if self.Q_inf > 0.0 and self.b_voce > 0.0:
            H += self.Q_inf * self.b_voce * np.exp(-self.b_voce * alpha)
        return H


@dataclass
class KinematicHardening3D:
    """Armstrong-Frederick 移動硬化."""

    C_kin: float = 0.0
    gamma_kin: float = 0.0


# ---------------------------------------------------------------------------
# 構成則
# ---------------------------------------------------------------------------


class PlaneStrainPlasticity:
    """平面ひずみ von Mises (J2) 弾塑性構成則.

    入力: eps = [eps_xx, eps_yy, gamma_xy] (3成分)
    出力: sigma = [sigma_xx, sigma_yy, sigma_xy], D_ep (3x3), 新状態
    """

    def __init__(
        self,
        E: float,
        nu: float,
        iso: IsotropicHardening3D,
        kin: KinematicHardening3D | None = None,
    ) -> None:
        if E <= 0:
            raise ValueError(f"ヤング率 E は正値: {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"ポアソン比 nu は (-1, 0.5): {nu}")
        if iso.sigma_y0 <= 0:
            raise ValueError(f"降伏応力 sigma_y0 は正値: {iso.sigma_y0}")

        self.E = E
        self.nu = nu
        self.iso = iso
        self.kin = kin if kin is not None else KinematicHardening3D()

        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = E / (2.0 * (1.0 + nu))
        self.kappa = E / (3.0 * (1.0 - 2.0 * nu))

        lam, mu = self.lam, self.mu
        self.D_e = np.array(
            [
                [lam + 2.0 * mu, lam, 0.0],
                [lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, mu],
            ],
            dtype=float,
        )
        self._D_e_4 = np.array(
            [
                [lam + 2.0 * mu, lam, lam, 0.0],
                [lam, lam + 2.0 * mu, lam, 0.0],
                [lam, lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, 0.0, mu],
            ],
            dtype=float,
        )

    @property
    def sigma_y0(self) -> float:
        return self.iso.sigma_y0

    def tangent(self, strain: np.ndarray | None = None) -> np.ndarray:
        """弾性テンソル（ConstitutiveProtocol 互換）."""
        return self.D_e.copy()

    def return_mapping(
        self,
        strain: np.ndarray,
        state: PlasticState3D,
    ) -> ReturnMappingResult3D:
        """Radial return mapping.

        Args:
            strain: (3,) [eps_xx, eps_yy, gamma_xy]
            state: 前ステップの塑性状態（変更されない）
        """
        mu = self.mu
        C_kin = self.kin.C_kin
        gamma_kin = self.kin.gamma_kin
        strain = np.asarray(strain, dtype=float)

        # --- 試行応力（4成分）---
        eps_e = strain - state.eps_p
        eps_zz_e = state.eps_p[0] + state.eps_p[1]  # = -eps_zz_p
        eps_e_4 = np.array([eps_e[0], eps_e[1], eps_zz_e, eps_e[2]])
        sigma_trial_4 = self._D_e_4 @ eps_e_4

        # --- 背応力（4成分、偏差的）---
        beta_4 = np.array(
            [state.beta[0], state.beta[1], -(state.beta[0] + state.beta[1]), state.beta[2]],
        )

        # --- 偏差相対応力 ---
        xi_trial_4 = sigma_trial_4 - beta_4
        s_xi_4 = _P_DEV_4 @ xi_trial_4

        # von Mises
        s_norm_sq = float(s_xi_4 @ (_W_4 * s_xi_4))
        s_norm = np.sqrt(max(s_norm_sq, 0.0))
        q_trial = np.sqrt(1.5) * s_norm

        # --- 降伏判定 ---
        sigma_y_n = self.iso.sigma_y(state.alpha)
        f_trial = q_trial - sigma_y_n

        if f_trial <= 1e-10 * self.iso.sigma_y0:
            return ReturnMappingResult3D(
                stress=sigma_trial_4[_PS_IDX].copy(),
                tangent=self.D_e.copy(),
                state_new=state.copy(),
            )

        # --- Radial return ---
        n_4 = s_xi_4 / s_norm

        dg = self._solve_dg(q_trial, state.alpha, C_kin, gamma_kin)

        # 応力更新
        sigma_new_4 = sigma_trial_4 - 2.0 * mu * dg * n_4

        # 塑性ひずみ更新（Voigt gamma 規約: gamma_xy = 2*eps_xy）
        deps_p_3 = np.array(
            [dg * n_4[0], dg * n_4[1], 2.0 * dg * n_4[3]],
        )
        eps_p_new = state.eps_p + deps_p_3
        alpha_new = state.alpha + dg

        # 背応力更新
        if C_kin > 0.0:
            theta = 1.0 / (1.0 + gamma_kin * dg)
            beta_new_4 = (beta_4 + (2.0 / 3.0) * C_kin * dg * n_4) * theta
            beta_new = np.array(
                [beta_new_4[0], beta_new_4[1], beta_new_4[3]],
            )
        else:
            beta_new = state.beta.copy()

        # Consistent tangent
        tangent = self._consistent_tangent(dg, n_4, alpha_new, q_trial)

        state_new = PlasticState3D(
            eps_p=eps_p_new,
            alpha=alpha_new,
            beta=beta_new,
        )
        return ReturnMappingResult3D(
            stress=sigma_new_4[_PS_IDX].copy(),
            tangent=tangent,
            state_new=state_new,
        )

    def _solve_dg(
        self,
        q_trial: float,
        alpha_n: float,
        C_kin: float,
        gamma_kin: float,
    ) -> float:
        """塑性乗数 dg を求解."""
        mu = self.mu

        if gamma_kin == 0.0 and self.iso.Q_inf == 0.0:
            H_bar = self.iso.H_iso + C_kin
            return (q_trial - self.iso.sigma_y(alpha_n)) / (3.0 * mu + H_bar)

        H_init = self.iso.dR_dalpha(alpha_n) + C_kin
        dg = (q_trial - self.iso.sigma_y(alpha_n)) / (3.0 * mu + H_init)

        for _ in range(50):
            alpha_trial = alpha_n + dg
            sigma_y_trial = self.iso.sigma_y(alpha_trial)
            H_iso_curr = self.iso.dR_dalpha(alpha_trial)

            if gamma_kin > 0.0:
                theta = 1.0 / (1.0 + gamma_kin * dg)
                H_kin_eff = C_kin * theta
            else:
                H_kin_eff = C_kin

            f_val = q_trial - 3.0 * mu * dg - H_kin_eff * dg - sigma_y_trial
            if abs(f_val) < 1e-12 * self.iso.sigma_y0:
                break

            df_ddg = -(3.0 * mu + H_iso_curr + H_kin_eff)
            if gamma_kin > 0.0:
                df_ddg += C_kin * dg * gamma_kin * theta * theta
            dg -= f_val / df_ddg
            dg = max(dg, 0.0)

        return dg

    def _consistent_tangent(
        self,
        dg: float,
        n_4: np.ndarray,
        alpha: float,
        q_trial: float,
    ) -> np.ndarray:
        """Consistent tangent (3x3, 平面ひずみ).

        de Souza Neto (2008) Box 7.4:
          D_ep = kappa(1x1) + 2mu*a_bar*I_dev_sym - 2mu*a_hat*(nxn)

        4x4 で構成後、[0,1,3]x[0,1,3] サブ行列を抽出。
        平面ひずみでは eps_zz=0 の運動学的拘束により、
        sigma_zz 行と eps_zz 列を除去するだけで正しい。
        （静的縮合は平面応力用であり、平面ひずみでは不要。）
        """
        mu = self.mu
        kappa = self.kappa

        H_iso = self.iso.dR_dalpha(alpha)
        C_kin = self.kin.C_kin
        gamma_kin = self.kin.gamma_kin

        if gamma_kin > 0.0:
            H_kin = C_kin / (1.0 + gamma_kin * dg)
        else:
            H_kin = C_kin

        H_prime = H_iso + H_kin

        a_bar = 1.0 - 3.0 * mu * dg / q_trial
        a_hat = a_bar - 1.0 / (1.0 + H_prime / (3.0 * mu))

        # 4x4 tangent (sigma-epsilon Voigt)
        # (nxn) 項: n_4 は応力空間ベクトル [n_xx,n_yy,n_zz,n_xy]
        # sigma-epsilon 空間では (nxn)[I,J] = n_I * n_J for all I,J
        D_ep_4 = (
            kappa * np.outer(_M_4, _M_4)
            + 2.0 * mu * a_bar * _P_DEV_SYM_4
            - 2.0 * mu * a_hat * np.outer(n_4, n_4)
        )

        # 平面ひずみ: サブ行列抽出
        return D_ep_4[np.ix_(_PS_IDX, _PS_IDX)]
