"""ContactForce Strategy 具象実装.

接触力の評価方法を Strategy として実装する。

Phase 3 統合:
- NCPContactForceProcess: solver_ncp._compute_contact_force_from_lambdas を移植
- SmoothPenaltyContactForceProcess: softplus 接触力の実ロジックを移植
- create_contact_force_strategy() ファクトリ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from __xkep_cae_deprecated.process.base import ProcessMeta
from __xkep_cae_deprecated.process.categories import SolverProcess


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

    solver_ncp._compute_contact_force_from_lambdas のロジックを移植。
    """

    meta = ProcessMeta(
        name="NCPContactForce",
        module="solve",
        version="0.1.0",
        document_path="docs/contact_force.md",
    )

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

        solver_ncp._compute_contact_force_from_lambdas のロジックを完全移植。

        Returns:
            (contact_force, ncp_residual)
        """
        from __xkep_cae_deprecated.contact.assembly import _contact_dofs, _contact_shape_vector
        from __xkep_cae_deprecated.contact.pair import ContactStatus

        f_c = np.zeros(self._ndof)
        residuals = []

        if hasattr(manager, "pairs"):
            for i, pair in enumerate(manager.pairs):
                if not hasattr(pair, "state"):
                    continue
                if pair.state.status == ContactStatus.INACTIVE:
                    continue

                lam_i = lambdas[i] if i < len(lambdas) else 0.0
                g_i = pair.state.gap
                if self._contact_compliance > 0.0:
                    g_i += self._contact_compliance * lam_i
                p_n = max(0.0, lam_i + k_pen * (-g_i))

                # NCP残差: active → k_pen * gap, inactive → λ
                is_active = p_n > 0.0
                if is_active:
                    residuals.append(k_pen * pair.state.gap)
                else:
                    residuals.append(lam_i)

                if p_n <= 0.0:
                    continue

                # 接触力アセンブリ
                g_shape = _contact_shape_vector(pair)
                dofs = _contact_dofs(pair, self._ndof_per_node)
                for k in range(4):
                    for d in range(3):
                        local_idx = k * 3 + d
                        global_idx = dofs[k * self._ndof_per_node + d]
                        f_c[global_idx] += p_n * g_shape[local_idx]

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
        ここでは追加の解析的接触剛性を返す。
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

    meta = ProcessMeta(
        name="SmoothPenaltyContactForce",
        module="solve",
        version="0.1.0",
        document_path="docs/contact_force.md",
    )

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

    @staticmethod
    def _softplus_derivative(g: float, delta: float) -> float:
        """Softplus 導関数: dp_n/dg = -sigmoid(-δ*g)."""
        if delta <= 0.0:
            return -1.0 if g < 0.0 else 0.0
        x = -delta * g
        if x > 30.0:
            return -1.0
        if x < -30.0:
            return 0.0
        return -1.0 / (1.0 + np.exp(-x))

    def evaluate(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """接触力とNCP残差を評価.

        softplus 関数による力計算 + Uzawa 更新残差。
        """
        from __xkep_cae_deprecated.contact.assembly import _contact_dofs, _contact_shape_vector
        from __xkep_cae_deprecated.contact.pair import ContactStatus

        f_c = np.zeros(self._ndof)
        residuals = []

        if hasattr(manager, "pairs"):
            for i, pair in enumerate(manager.pairs):
                if not hasattr(pair, "state"):
                    continue
                if pair.state.status == ContactStatus.INACTIVE:
                    continue

                g_i = pair.state.gap
                lam_i = lambdas[i] if i < len(lambdas) else 0.0

                # softplus 接触力
                p_n = k_pen * self._softplus(g_i, self._smoothing_delta)
                pair.state.p_n = p_n

                # Uzawa 更新残差
                uzawa_update = max(0.0, lam_i + k_pen * (-g_i))
                residuals.append(abs(uzawa_update - lam_i))

                if p_n <= 1e-30:
                    continue

                # 接触力アセンブリ
                g_shape = _contact_shape_vector(pair)
                dofs = _contact_dofs(pair, self._ndof_per_node)
                for k in range(4):
                    for d in range(3):
                        local_idx = k * 3 + d
                        global_idx = dofs[k * self._ndof_per_node + d]
                        f_c[global_idx] += p_n * g_shape[local_idx]

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

        K_c = Σ k_pen * softplus'(g) * g_shape @ g_shape^T
        """
        from __xkep_cae_deprecated.contact.assembly import _contact_dofs, _contact_shape_vector
        from __xkep_cae_deprecated.contact.pair import ContactStatus

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        if hasattr(manager, "pairs"):
            for pair in manager.pairs:
                if not hasattr(pair, "state"):
                    continue
                if pair.state.status == ContactStatus.INACTIVE:
                    continue

                g_i = pair.state.gap
                dp_dg = k_pen * self._softplus_derivative(g_i, self._smoothing_delta)
                if abs(dp_dg) < 1e-30:
                    continue

                g_shape = _contact_shape_vector(pair)
                dofs = _contact_dofs(pair, self._ndof_per_node)

                for ki in range(4):
                    for di in range(3):
                        local_i = ki * 3 + di
                        global_i = dofs[ki * self._ndof_per_node + di]
                        if abs(g_shape[local_i]) < 1e-30:
                            continue
                        for kj in range(4):
                            for dj in range(3):
                                local_j = kj * 3 + dj
                                global_j = dofs[kj * self._ndof_per_node + dj]
                                val = dp_dg * g_shape[local_i] * g_shape[local_j]
                                if abs(val) > 1e-30:
                                    rows.append(global_i)
                                    cols.append(global_j)
                                    data.append(val)

        if len(data) == 0:
            return sp.csr_matrix((self._ndof, self._ndof))

        return sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self._ndof, self._ndof),
        ).tocsr()

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        f, r = self.evaluate(input_data.u, input_data.lambdas, input_data.manager, input_data.k_pen)
        return ContactForceOutput(contact_force=f, ncp_residual=r)


def create_contact_force_strategy(
    *,
    contact_mode: str = "ncp",
    ndof: int = 0,
    ndof_per_node: int = 6,
    contact_compliance: float = 0.0,
    smoothing_delta: float = 0.0,
    n_uzawa_max: int = 5,
    tol_uzawa: float = 1e-3,
) -> NCPContactForceProcess | SmoothPenaltyContactForceProcess:
    """solver_ncp.py の接触力モード分岐を Strategy に移譲するファクトリ.

    Args:
        contact_mode: "ncp" | "smooth_penalty"
        ndof: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        contact_compliance: δ正則化パラメータ
        smoothing_delta: smooth penalty の δ パラメータ
        n_uzawa_max: Uzawa 最大反復回数
        tol_uzawa: Uzawa 収束許容値

    Returns:
        NCPContactForceProcess または SmoothPenaltyContactForceProcess
    """
    if contact_mode == "smooth_penalty":
        return SmoothPenaltyContactForceProcess(
            ndof=ndof,
            ndof_per_node=ndof_per_node,
            smoothing_delta=smoothing_delta,
            n_uzawa_max=n_uzawa_max,
            tol_uzawa=tol_uzawa,
        )

    return NCPContactForceProcess(
        ndof=ndof,
        ndof_per_node=ndof_per_node,
        contact_compliance=contact_compliance,
    )
