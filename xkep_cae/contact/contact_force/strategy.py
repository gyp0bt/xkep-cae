"""ContactForce Strategy 具象実装.

ContactForceStrategy Protocol に従い、接触力を評価する Process 群。

2クラス構成:
- NCPContactForceProcess: Alart-Curnier NCP + 鞍点系
- SmoothPenaltyContactForceProcess: softplus + Uzawa 外部ループ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._assembly_utils import _contact_dofs
from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact._types import ContactStatus
from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


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


# ── ヘルパー ───────────────────────────────────────────────


def _contact_shape_vector(pair: object) -> np.ndarray:
    """接触形状ベクトル g_shape (12,) を構築する.

    ギャップ関数の勾配（法線方向の形状関数）:
        g_shape = ∂g/∂x = [(1-s)*n, s*n, -(1-t)*n, -t*n]  (4×3 = 12)

    ここで n はセグメント B → A 方向の単位法線。
    g > 0 で非接触（ギャップ開）、g < 0 で貫入。

    符号規約（status-221 で統一）:
        接触力ベクトル f_c は「ペナルティエネルギー勾配」として定義:
            f_c = ∂Φ/∂u = -p_n * g_shape
        残差式: R_u = f_int + f_c - f_ext
        これにより f_c は内力的に加算され、NR 補正が正しい方向に作用する。

    Args:
        pair: ContactPair（state.s, state.t, state.normal を持つ）

    Returns:
        g_shape: (12,) 形状ベクトル
    """
    s = pair.state.s
    t = pair.state.t
    normal = pair.state.normal
    coeffs = [(1.0 - s), s, -(1.0 - t), -t]
    g_shape = np.zeros(12)
    for k in range(4):
        g_shape[k * 3 : k * 3 + 3] = coeffs[k] * normal
    return g_shape


# ── 具象 Process ──────────────────────────────────────────


class NCPContactForceProcess(
    SolverProcess[ContactForceInput, ContactForceOutput],
):
    """Alart-Curnier NCP + 鞍点系による接触力評価.

    法線接触力: p_n = max(0, λ + k_pen * (-(g + δ*λ)))
    NCP残差: C_i = k_pen * g_i (active) | λ_i (inactive)
    """

    meta = ProcessMeta(
        name="NCPContactForce",
        module="solve",
        version="1.0.0",
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
        """接触力とNCP残差を評価."""
        f_c = np.zeros(self._ndof)
        residuals: list[float] = []

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

                is_active = p_n > 0.0
                if is_active:
                    residuals.append(k_pen * pair.state.gap)
                else:
                    residuals.append(lam_i)

                if p_n <= 0.0:
                    continue

                g_shape = _contact_shape_vector(pair)
                dofs = _contact_dofs(pair, self._ndof_per_node)
                for k in range(4):
                    for d in range(3):
                        local_idx = k * 3 + d
                        global_idx = dofs[k * self._ndof_per_node + d]
                        # エネルギー勾配規約: f_c = -p_n * g_shape
                        # （ペナルティポテンシャル Φ の勾配 ∂Φ/∂u）
                        f_c[global_idx] -= p_n * g_shape[local_idx]

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

        NCP では鞍点系で法線剛性を扱うため、ゼロ行列を返す。
        """
        return sp.csr_matrix((self._ndof, self._ndof))

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        f, r = self.evaluate(
            input_data.u,
            input_data.lambdas,
            input_data.manager,
            input_data.k_pen,
        )
        return ContactForceOutput(contact_force=f, ncp_residual=r)


class SmoothPenaltyContactForceProcess(
    SolverProcess[ContactForceInput, ContactForceOutput],
):
    """Softplus smooth penalty + Uzawa 外部ループによる接触力評価.

    接触力: p_n(g) = (1/δ) * log(1 + exp(-δ*g))
    Uzawa 更新: λ = max(0, λ + k_pen*(-g))

    推奨構成（status-147）:
    - 摩擦あり解析では必ずこちらを使用
    """

    meta = ProcessMeta(
        name="SmoothPenaltyContactForce",
        module="solve",
        version="3.0.0",
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
        exact_tangent: bool = False,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._smoothing_delta = smoothing_delta
        self._n_uzawa_max = n_uzawa_max
        self._tol_uzawa = tol_uzawa
        self._exact_tangent = exact_tangent

    @property
    def n_uzawa_max(self) -> int:
        """Uzawa 最大反復回数."""
        return self._n_uzawa_max

    @property
    def tol_uzawa(self) -> float:
        """Uzawa 収束許容値."""
        return self._tol_uzawa

    def set_ndof(self, ndof: int) -> None:
        """全体 DOF 数を設定."""
        self._ndof = ndof

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
        """接触力とNCP残差を評価."""
        f_c = np.zeros(self._ndof)
        residuals: list[float] = []

        if hasattr(manager, "pairs"):
            for i, pair in enumerate(manager.pairs):
                if not hasattr(pair, "state"):
                    continue
                # softplus は C∞ なので INACTIVE でも常に評価する。
                # gap が大きいペアでは力が自然にゼロに近づくため、
                # ヒステリシス境界での接触チャタリングを防止する。

                g_i = pair.state.gap
                lam_i = lambdas[i] if i < len(lambdas) else 0.0

                # 拡大ラグランジアン: p_n = λ + k_pen * softplus(g)
                # λ は Uzawa 外部ループで更新される。
                # penalty 項のみだと接線剛性が負定値になり NR 不収束。
                # λ が接触力の大部分を担い、penalty は滑らかな補正項。
                p_n = max(0.0, lam_i) + k_pen * self._softplus(g_i, self._smoothing_delta)
                manager.pairs[i] = _evolve_pair(pair, state=_evolve_state(pair.state, p_n=p_n))
                pair = manager.pairs[i]

                uzawa_update = max(0.0, lam_i + k_pen * (-g_i))
                residuals.append(abs(uzawa_update - lam_i))

                if p_n <= 1e-30:
                    continue

                g_shape = _contact_shape_vector(pair)
                dofs = _contact_dofs(pair, self._ndof_per_node)
                for k in range(4):
                    for d in range(3):
                        local_idx = k * 3 + d
                        global_idx = dofs[k * self._ndof_per_node + d]
                        # エネルギー勾配規約: f_c = -p_n * g_shape
                        # （ペナルティポテンシャル Φ の勾配 ∂Φ/∂u）
                        f_c[global_idx] -= p_n * g_shape[local_idx]

        ncp_residual = np.array(residuals) if residuals else np.zeros(0)
        return f_c, ncp_residual

    def tangent(
        self,
        u: np.ndarray,
        lambdas: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> sp.csr_matrix:
        """接触接線剛性行列（エネルギー勾配規約、正定値）.

        f_c = -p_n * g_shape（エネルギー勾配規約）の接線:
            ∂f_c/∂u = -(dp_n/dg) * g_shape ⊗ g_shape
                     = k_pen * sigmoid(-δg) * g_shape ⊗ g_shape
        常に正定値。K_total = K_struct + K_contact は正定値を保つ。
        """
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        if hasattr(manager, "pairs"):
            for pair in manager.pairs:
                if not hasattr(pair, "state"):
                    continue

                g_i = pair.state.gap
                deriv = self._softplus_derivative(g_i, self._smoothing_delta)
                if abs(deriv) < 1e-30:
                    continue

                # エネルギー勾配規約: ∂f_c/∂u = -(dp_n/dg) * g_shape ⊗ g_shape
                # deriv = _softplus_derivative(g, delta) < 0 なので
                # weight = -k_pen * deriv = k_pen * |deriv| > 0（常に正定値）
                weight = -k_pen * deriv
                g_shape = _contact_shape_vector(pair)
                dofs = _contact_dofs(pair, self._ndof_per_node)

                # ランク1更新: weight * g_shape ⊗ g_shape
                for ki in range(4):
                    for di in range(3):
                        li = ki * 3 + di
                        gi = dofs[ki * self._ndof_per_node + di]
                        w_gi = weight * g_shape[li]
                        if abs(w_gi) < 1e-30:
                            continue
                        for kj in range(4):
                            for dj in range(3):
                                lj = kj * 3 + dj
                                gj = dofs[kj * self._ndof_per_node + dj]
                                val = w_gi * g_shape[lj]
                                if abs(val) > 1e-30:
                                    rows.append(gi)
                                    cols.append(gj)
                                    vals.append(val)

        if rows:
            return sp.csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))),
                shape=(self._ndof, self._ndof),
            )
        return sp.csr_matrix((self._ndof, self._ndof))

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        f, r = self.evaluate(
            input_data.u,
            input_data.lambdas,
            input_data.manager,
            input_data.k_pen,
        )
        return ContactForceOutput(contact_force=f, ncp_residual=r)


# ── ファクトリ ─────────────────────────────────────────────


def _create_contact_force_strategy(
    *,
    contact_mode: str = "ncp",
    ndof: int = 0,
    ndof_per_node: int = 6,
    contact_compliance: float = 0.0,
    smoothing_delta: float = 0.0,
    n_uzawa_max: int = 5,
    tol_uzawa: float = 1e-3,
    exact_tangent: bool = False,
) -> NCPContactForceProcess | SmoothPenaltyContactForceProcess:
    """接触力 Strategy ファクトリ.

    Args:
        contact_mode: "ncp" | "smooth_penalty"
        ndof: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        contact_compliance: δ正則化パラメータ
        smoothing_delta: smooth penalty の δ パラメータ
        n_uzawa_max: Uzawa 最大反復回数
        tol_uzawa: Uzawa 収束許容値
        exact_tangent: 厳密接線使用（動的解析で c0*M 正則化がある場合に有効）
    """
    if contact_mode == "smooth_penalty":
        return SmoothPenaltyContactForceProcess(
            ndof=ndof,
            ndof_per_node=ndof_per_node,
            smoothing_delta=smoothing_delta,
            n_uzawa_max=n_uzawa_max,
            tol_uzawa=tol_uzawa,
            exact_tangent=exact_tangent,
        )

    return NCPContactForceProcess(
        ndof=ndof,
        ndof_per_node=ndof_per_node,
        contact_compliance=contact_compliance,
    )
