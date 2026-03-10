"""ContactForce Strategy 具象実装.

接触力の評価方法を Strategy として実装する。

設計仕様: xkep_cae/process/process-architecture.md §2.1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess


@dataclass(frozen=True)
class ContactForceInput:
    """ContactForce Strategy の入力."""

    u: np.ndarray
    lambdas: np.ndarray
    manager: object
    k_pen: float


@dataclass(frozen=True)
class ContactForceOutput:
    """ContactForce Strategy の出力."""

    contact_force: np.ndarray
    ncp_residual: np.ndarray


class NCPContactForceProcess(SolverProcess[ContactForceInput, ContactForceOutput]):
    """Alart-Curnier NCP + 鞍点系による接触力評価.

    法線接触力: p_n = max(0, λ + k_pen * (-(g + δ*λ)))
    NCP残差: C_i = k_pen * g_i (active) | λ_i (inactive)

    鞍点系:
        [K_eff   -G_A^T] [Δu  ] = [-R_u    ]
        [G_A      0    ] [Δλ_A]   [-g_active]

    δ正則化により Schur complement S = G*V + δI が正定値化される。
    """

    meta = ProcessMeta(name="NCPContactForce", module="solve", version="0.1.0")

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        contact_compliance: float = 0.0,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._contact_compliance = contact_compliance

    def evaluate(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """接触力とNCP残差を評価.

        Phase 3 で solver_ncp._compute_contact_force_from_lambdas を
        本メソッドに完全移植する。現在は Protocol 準拠スタブ。
        """
        f_c = np.zeros(self._ndof)
        residuals = []

        if hasattr(manager, "pairs"):
            for i, pair in enumerate(manager.pairs):
                if not hasattr(pair, "state"):
                    continue
                lam_i = lambdas[i] if i < len(lambdas) else 0.0
                g_i = getattr(pair.state, "gap", 0.0)
                if self._contact_compliance > 0.0:
                    g_i += self._contact_compliance * lam_i
                p_n = max(0.0, lam_i + k_pen * (-g_i))
                is_active = p_n > 0.0
                if is_active:
                    residuals.append(k_pen * getattr(pair.state, "gap", 0.0))
                else:
                    residuals.append(lam_i)

        ncp_residual = np.array(residuals) if residuals else np.zeros(0)
        return f_c, ncp_residual

    def tangent(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> sp.csr_matrix:
        """接触接線剛性行列.

        NCP では鞍点系 (k_pen * G_A^T G_A) で法線剛性を扱うため、
        ここでは追加の接触剛性を返す。
        Phase 3 で完全移植。
        """
        return sp.csr_matrix((self._ndof, self._ndof))

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        f, r = self.evaluate(input_data.u, input_data.lambdas, input_data.manager, input_data.k_pen)
        return ContactForceOutput(contact_force=f, ncp_residual=r)


class SmoothPenaltyContactForceProcess(SolverProcess[ContactForceInput, ContactForceOutput]):
    """Softplus smooth penalty + Uzawa 外部ループによる接触力評価.

    接触力: p_n(g) = (1/δ) * log(1 + exp(-δ*g))
    Uzawa 更新: λ = max(0, λ + k_pen*(-g))

    δ = smoothing_delta で接触/非接触の遷移を滑らかにする。
    δ→0 で通常のペナルティ法に収束する。

    推奨構成（status-147）:
    - 摩擦あり解析では必ずこちらを使用
    - NCP 鞍点系は摩擦接線剛性の符号問題で発散する
    """

    meta = ProcessMeta(name="SmoothPenaltyContactForce", module="solve", version="0.1.0")

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        smoothing_delta: float = 0.0,
        n_uzawa_max: int = 5,
        tol_uzawa: float = 1e-3,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._smoothing_delta = smoothing_delta
        self._n_uzawa_max = n_uzawa_max
        self._tol_uzawa = tol_uzawa

    @staticmethod
    def _softplus(g: float, delta: float) -> float:
        """Softplus 接触力: p_n = (1/δ) * log(1 + exp(-δ*g))."""
        if delta <= 0.0:
            return max(0.0, -g)
        x = -delta * g
        if x > 30.0:
            return -g
        return np.log1p(np.exp(x)) / delta

    def evaluate(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """接触力とNCP残差を評価.

        Phase 3 で solver_ncp.py の smooth penalty ループから完全移植。
        現在は softplus 関数による力計算のスタブ。
        """
        f_c = np.zeros(self._ndof)
        residuals = []

        if hasattr(manager, "pairs"):
            for i, pair in enumerate(manager.pairs):
                if not hasattr(pair, "state"):
                    continue
                g_i = getattr(pair.state, "gap", 0.0)
                lam_i = lambdas[i] if i < len(lambdas) else 0.0
                uzawa_update = max(0.0, lam_i + k_pen * (-g_i))
                residuals.append(abs(uzawa_update - lam_i))

        ncp_residual = np.array(residuals) if residuals else np.zeros(0)
        return f_c, ncp_residual

    def tangent(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> sp.csr_matrix:
        """接触接線剛性行列.

        Phase 3 で assemble_smooth_contact_stiffness を移植。
        """
        return sp.csr_matrix((self._ndof, self._ndof))

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        f, r = self.evaluate(input_data.u, input_data.lambdas, input_data.manager, input_data.k_pen)
        return ContactForceOutput(contact_force=f, ncp_residual=r)
