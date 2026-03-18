"""Friction Strategy 具象実装.

旧 __xkep_cae_deprecated/process/strategies/friction.py の完全書き直し。
FrictionStrategy Protocol に従い、摩擦力を評価する Process 群。

3クラス構成:
- NoFrictionProcess: 摩擦なし（デフォルト）
- CoulombReturnMappingProcess: NCP 法線 + Coulomb return mapping
- SmoothPenaltyFrictionProcess: Smooth penalty + Uzawa 摩擦
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact._types import ContactStatus
from xkep_cae.contact.friction._assembly import (
    _assemble_friction_tangent_stiffness,
    _friction_return_mapping_loop,
)
from xkep_cae.contact.friction.law_friction import (
    _compute_mu_effective,
)
from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class FrictionInput:
    """Friction Strategy の入力."""

    u: np.ndarray
    contact_pairs: list
    mu: float


@dataclass(frozen=True)
class FrictionOutput:
    """Friction Strategy の出力."""

    friction_force: np.ndarray
    friction_residual: np.ndarray


# ── 具象 Process ──────────────────────────────────────────


class NoFrictionProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """摩擦なし（デフォルト）.

    接触法線力のみ。摩擦力・残差はゼロベクトルを返す。
    """

    meta = ProcessMeta(
        name="NoFriction",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )

    def __init__(self, ndof: int = 0) -> None:
        self._ndof = ndof
        self._friction_tangents: dict[int, np.ndarray] = {}

    @property
    def friction_tangents(self) -> dict[int, np.ndarray]:
        """摩擦接線剛性 (2x2) の辞書."""
        return self._friction_tangents

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差: ゼロベクトル."""
        ndof = self._ndof if self._ndof > 0 else len(u)
        return np.zeros(ndof), np.zeros(0)

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        **kwargs: object,
    ) -> sp.csr_matrix:
        """摩擦接線剛性: ゼロ行列."""
        ndof = self._ndof if self._ndof > 0 else len(u)
        return sp.csr_matrix((ndof, ndof))

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


class CoulombReturnMappingProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """Coulomb 摩擦 return mapping.

    法線力から Coulomb 錐を計算し、弾性予測→return mapping で
    stick/slip を判定する。NCP 法線 + 摩擦ペナルティの組み合わせ。

    推奨: SmoothPenaltyContactForceProcess と組み合わせ（status-147）。
    """

    meta = ProcessMeta(
        name="CoulombReturnMapping",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        k_pen: float = 0.0,
        k_t_ratio: float = 1.0,
        contact_compliance: float = 0.0,
        mu_ramp_counter: int = 0,
        mu_ramp_steps: int = 0,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._k_pen = k_pen
        self._k_t_ratio = k_t_ratio
        self._contact_compliance = contact_compliance
        self._mu_ramp_counter = mu_ramp_counter
        self._mu_ramp_steps = mu_ramp_steps
        self._friction_tangents: dict[int, np.ndarray] = {}

    @property
    def friction_tangents(self) -> dict[int, np.ndarray]:
        """摩擦接線剛性 (2x2) の辞書."""
        return self._friction_tangents

    def compute_k_t(self) -> float:
        """接線ペナルティ剛性."""
        return self._k_pen * self._k_t_ratio

    def compute_mu_effective(self, mu: float) -> float:
        """μ ランプ適用後の有効摩擦係数."""
        return _compute_mu_effective(mu, self._mu_ramp_counter, self._mu_ramp_steps)

    def set_k_pen(self, k_pen: float) -> None:
        """ペナルティ正則化パラメータを設定."""
        self._k_pen = k_pen

    def set_k_t_ratio(self, k_t_ratio: float) -> None:
        """接線/法線ペナルティ比を設定."""
        self._k_t_ratio = k_t_ratio

    def set_mu_ramp_counter(self, counter: int) -> None:
        """μ ランプカウンタを設定."""
        self._mu_ramp_counter = counter

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差を評価.

        NCP 法線力 + Coulomb return mapping。
        kwargs: lambdas, u_ref, node_coords_ref (Newton-Uzawa から渡される)
        """
        if not contact_pairs:
            return np.zeros(self._ndof), np.zeros(0)

        mu_eff = self.compute_mu_effective(mu)
        lambdas = kwargs.get("lambdas")
        u_ref = kwargs.get("u_ref")

        if lambdas is None:
            lambdas = np.zeros(len(contact_pairs))
        if u_ref is None:
            u_ref = np.zeros_like(u)

        _delta = self._contact_compliance
        _k_pen = self._k_pen

        def compute_p_n(i: int, pair: object) -> float:
            if pair.state.status == ContactStatus.INACTIVE:
                return 0.0
            lam_i = lambdas[i] if i < len(lambdas) else 0.0
            g_i = pair.state.gap
            g_eff = g_i + _delta * lam_i if _delta > 0.0 else g_i
            p_n = max(0.0, lam_i + _k_pen * (-g_eff))
            contact_pairs[i] = _evolve_pair(pair, state=_evolve_state(pair.state, p_n=p_n))
            return p_n

        f_friction, friction_residual, self._friction_tangents = _friction_return_mapping_loop(
            contact_pairs,
            u,
            u_ref,
            self._ndof,
            self._ndof_per_node,
            self._k_pen,
            self._k_t_ratio,
            mu_eff,
            compute_p_n,
        )
        return f_friction, friction_residual

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        **kwargs: object,
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列."""
        return _assemble_friction_tangent_stiffness(
            contact_pairs, self._friction_tangents, self._ndof, self._ndof_per_node
        )

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


class SmoothPenaltyFrictionProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """Smooth penalty + Uzawa 摩擦.

    smooth penalty 接触力モデルと統合して動作する。
    Uzawa 外部ループで摩擦乗数を更新し、Newton 内部ループで変位を求解。

    推奨構成（status-147）:
    - contact_force: SmoothPenaltyContactForceProcess
    - friction: SmoothPenaltyFrictionProcess
    """

    meta = ProcessMeta(
        name="SmoothPenaltyFriction",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        k_t_ratio: float = 1.0,
        k_pen: float = 0.0,
        contact_compliance: float = 0.0,
        mu_ramp_counter: int = 0,
        mu_ramp_steps: int = 0,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._k_t_ratio = k_t_ratio
        self._k_pen = k_pen
        self._contact_compliance = contact_compliance
        self._mu_ramp_counter = mu_ramp_counter
        self._mu_ramp_steps = mu_ramp_steps
        self._friction_tangents: dict[int, np.ndarray] = {}

    @property
    def friction_tangents(self) -> dict[int, np.ndarray]:
        """摩擦接線剛性 (2x2) の辞書."""
        return self._friction_tangents

    def compute_k_t(self) -> float:
        """接線ペナルティ剛性."""
        return self._k_pen * self._k_t_ratio

    def compute_mu_effective(self, mu: float) -> float:
        """μ ランプ適用後の有効摩擦係数."""
        return _compute_mu_effective(mu, self._mu_ramp_counter, self._mu_ramp_steps)

    def set_k_pen(self, k_pen: float) -> None:
        """ペナルティ正則化パラメータを設定."""
        self._k_pen = k_pen

    def set_k_t_ratio(self, k_t_ratio: float) -> None:
        """接線/法線ペナルティ比を設定."""
        self._k_t_ratio = k_t_ratio

    def set_mu_ramp_counter(self, counter: int) -> None:
        """μ ランプカウンタを設定."""
        self._mu_ramp_counter = counter

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差を評価（smooth penalty 版）.

        p_n は pair.state.p_n から取得（smooth penalty 接触力で事前計算済み）。
        kwargs: u_ref, node_coords_ref (Newton-Uzawa から渡される)
        """
        if not contact_pairs:
            return np.zeros(self._ndof), np.zeros(0)

        mu_eff = self.compute_mu_effective(mu)
        u_ref = kwargs.get("u_ref")

        if u_ref is None:
            u_ref = np.zeros_like(u)

        def compute_p_n(i: int, pair: object) -> float:
            return getattr(pair.state, "p_n", 0.0)

        f_friction, friction_residual, self._friction_tangents = _friction_return_mapping_loop(
            contact_pairs,
            u,
            u_ref,
            self._ndof,
            self._ndof_per_node,
            self._k_pen,
            self._k_t_ratio,
            mu_eff,
            compute_p_n,
        )
        return f_friction, friction_residual

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        **kwargs: object,
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列."""
        return _assemble_friction_tangent_stiffness(
            contact_pairs, self._friction_tangents, self._ndof, self._ndof_per_node
        )

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


def _create_friction_strategy(
    *,
    use_friction: bool = False,
    contact_mode: str = "ncp",
    ndof: int = 0,
    ndof_per_node: int = 6,
    k_pen: float = 0.0,
    k_t_ratio: float = 1.0,
    contact_compliance: float = 0.0,
    mu_ramp_steps: int = 0,
) -> NoFrictionProcess | CoulombReturnMappingProcess | SmoothPenaltyFrictionProcess:
    """Friction Strategy ファクトリ.

    Args:
        use_friction: 摩擦の有効/無効
        contact_mode: "ncp" | "smooth_penalty"
        ndof: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        k_pen: ペナルティ正則化パラメータ
        k_t_ratio: 接線/法線ペナルティ比
        contact_compliance: δ正則化パラメータ
        mu_ramp_steps: μランプ総ステップ数
    """
    if not use_friction:
        return NoFrictionProcess(ndof=ndof)

    if contact_mode == "smooth_penalty":
        return SmoothPenaltyFrictionProcess(
            ndof=ndof,
            ndof_per_node=ndof_per_node,
            k_pen=k_pen,
            k_t_ratio=k_t_ratio,
            contact_compliance=contact_compliance,
            mu_ramp_steps=mu_ramp_steps,
        )

    return CoulombReturnMappingProcess(
        ndof=ndof,
        ndof_per_node=ndof_per_node,
        k_pen=k_pen,
        k_t_ratio=k_t_ratio,
        contact_compliance=contact_compliance,
        mu_ramp_steps=mu_ramp_steps,
    )
