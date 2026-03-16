"""Friction Strategy 具象実装.

摩擦力の評価方法を Strategy として実装する。

Phase 3 統合:
- CoulombReturnMappingProcess: solver_ncp._compute_friction_forces_ncp を移植
- _build_friction_stiffness ロジックを tangent() に移植
- create_friction_strategy() ファクトリで solver_ncp.py の摩擦設定を Strategy に移譲

Phase 5 リファクタリング（status-157）:
- tangent() → assembly.assemble_friction_tangent_stiffness() に統一
- evaluate() 内の力アセンブリ → assembly.assemble_friction_force() に統一
- evaluate() の return mapping ループ → _friction_return_mapping_loop() に統一
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess


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


def _friction_return_mapping_loop(
    contact_pairs: list,
    u: np.ndarray,
    u_ref: np.ndarray,
    node_coords_ref: np.ndarray,
    ndof: int,
    ndof_per_node: int,
    k_pen: float,
    k_t_ratio: float,
    mu_eff: float,
    compute_p_n: callable,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """摩擦 return mapping ループの共通実装.

    CoulombReturnMapping と SmoothPenaltyFriction の evaluate() で
    重複していたループ本体を抽出した共通関数。

    Args:
        contact_pairs: 接触ペアリスト
        u: 現在の変位
        u_ref: 参照変位
        node_coords_ref: 参照座標
        ndof: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        k_pen: ペナルティ剛性
        k_t_ratio: 接線/法線ペナルティ比
        mu_eff: 実効摩擦係数
        compute_p_n: pair index → p_n を返すコールバック

    Returns:
        f_friction: 摩擦力ベクトル
        friction_residual: 残差配列
        friction_tangents: {pair_idx: D_t (2,2)}
    """
    from xkep_cae.contact.assembly import assemble_friction_force
    from xkep_cae.contact.law_friction import (
        compute_tangential_displacement,
        friction_return_mapping,
        friction_tangent_2x2,
    )

    friction_forces_local: dict[int, np.ndarray] = {}
    friction_tangents: dict[int, np.ndarray] = {}
    residuals: list[float] = []

    for i, pair in enumerate(contact_pairs):
        if not hasattr(pair, "state"):
            continue

        p_n = compute_p_n(i, pair)
        if p_n <= 0.0 or mu_eff <= 0.0:
            continue

        # ペナルティ剛性の初期化（未設定時）
        if pair.state.k_pen <= 0.0:
            pair.state.k_pen = k_pen
            pair.state.k_t = k_pen * k_t_ratio

        # 接線変位
        delta_ut = compute_tangential_displacement(pair, u, u_ref, node_coords_ref, ndof_per_node)

        # Coulomb return mapping
        q = friction_return_mapping(pair, delta_ut, mu_eff)

        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-30:
            continue

        residuals.append(max(0.0, q_norm - mu_eff * p_n))
        friction_forces_local[i] = q

        # 摩擦接線剛性
        D_t = friction_tangent_2x2(pair, mu_eff)
        friction_tangents[i] = D_t

    # 力アセンブリ
    f_friction = assemble_friction_force(contact_pairs, friction_forces_local, ndof, ndof_per_node)

    friction_residual = np.array(residuals) if residuals else np.zeros(0)
    return f_friction, friction_residual, friction_tangents


class NoFrictionProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """摩擦なし（デフォルト）.

    接触法線力のみ。摩擦力・残差はゼロベクトルを返す。
    """

    meta = ProcessMeta(
        name="NoFriction", module="solve", version="0.1.0", document_path="docs/friction.md"
    )

    def __init__(self, ndof: int = 0) -> None:
        self._ndof = ndof
        self._friction_tangents: dict[int, np.ndarray] = {}

    @property
    def friction_tangents(self) -> dict[int, np.ndarray]:
        """常に空辞書（摩擦なし）."""
        return self._friction_tangents

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差: ゼロベクトル."""
        ndof = self._ndof if self._ndof > 0 else len(u)
        return np.zeros(ndof), np.zeros(0)

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> sp.csr_matrix:
        """摩擦接線剛性: ゼロ行列."""
        ndof = self._ndof if self._ndof > 0 else len(u)
        return sp.csr_matrix((ndof, ndof))

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


class CoulombReturnMappingProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """Coulomb 摩擦 return mapping.

    法線力からCoulomb錐を計算し、弾性予測→return mapping で
    stick/slip を判定する。NCP法線 + 摩擦ペナルティの組み合わせ。

    solver_ncp._compute_friction_forces_ncp のロジックを移植。

    注意: NCPContactForceProcess との組み合わせは非互換（status-147）。
    SmoothPenaltyContactForceProcess と組み合わせること。
    """

    meta = ProcessMeta(
        name="CoulombReturnMapping",
        module="solve",
        version="0.1.0",
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
        """最新の摩擦接線剛性マップ（tangent() 構築用）."""
        return self._friction_tangents

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        *,
        lambdas: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
        node_coords_ref: np.ndarray | None = None,
        manager: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差を評価.

        solver_ncp._compute_friction_forces_ncp のロジックを移植。
        基本 Protocol シグネチャ (u, contact_pairs, mu) に加えて
        追加パラメータを keyword-only で受け取る。
        """
        from xkep_cae.contact.law_friction import compute_mu_effective

        mu_eff = compute_mu_effective(mu, self._mu_ramp_counter, self._mu_ramp_steps)

        if lambdas is None:
            lambdas = np.zeros(len(contact_pairs))
        if u_ref is None:
            u_ref = np.zeros_like(u)
        if node_coords_ref is None:
            node_coords_ref = np.zeros((self._ndof // self._ndof_per_node, 3))

        _delta = self._contact_compliance
        _k_pen = self._k_pen

        def compute_p_n(i: int, pair: object) -> float:
            from xkep_cae.contact.pair import ContactStatus

            if pair.state.status == ContactStatus.INACTIVE:
                return 0.0
            lam_i = lambdas[i] if i < len(lambdas) else 0.0
            g_i = pair.state.gap
            g_eff = g_i + _delta * lam_i if _delta > 0.0 else g_i
            p_n = max(0.0, lam_i + _k_pen * (-g_eff))
            pair.state.p_n = p_n
            return p_n

        f_friction, friction_residual, self._friction_tangents = _friction_return_mapping_loop(
            contact_pairs,
            u,
            u_ref,
            node_coords_ref,
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
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列を構築."""
        from xkep_cae.contact.assembly import assemble_friction_tangent_stiffness

        return assemble_friction_tangent_stiffness(
            contact_pairs, self._friction_tangents, self._ndof, self._ndof_per_node
        )

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


class SmoothPenaltyFrictionProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """Smooth penalty + Uzawa 摩擦.

    smooth penalty 接触力モデルと統合して動作する。
    Uzawa 外部ループで摩擦乗数を更新し、Newton 内部ループで
    変位を求解する。

    solver_ncp.py の smooth penalty 摩擦ループのロジックを移植。

    推奨構成（status-147）:
    - contact_force: SmoothPenaltyContactForceProcess
    - friction: SmoothPenaltyFrictionProcess
    """

    meta = ProcessMeta(
        name="SmoothPenaltyFriction",
        module="solve",
        version="0.1.0",
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
        return self._friction_tangents

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        *,
        lambdas: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
        node_coords_ref: np.ndarray | None = None,
        manager: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差を評価（smooth penalty 版）.

        CoulombReturnMapping と同じ return mapping だが、
        p_n は pair.state.p_n から取得（smooth penalty 接触力で事前計算済み）。
        """
        from xkep_cae.contact.law_friction import compute_mu_effective

        mu_eff = compute_mu_effective(mu, self._mu_ramp_counter, self._mu_ramp_steps)

        if u_ref is None:
            u_ref = np.zeros_like(u)
        if node_coords_ref is None:
            node_coords_ref = np.zeros((self._ndof // self._ndof_per_node, 3))

        def compute_p_n(i: int, pair: object) -> float:
            return getattr(pair.state, "p_n", 0.0)

        f_friction, friction_residual, self._friction_tangents = _friction_return_mapping_loop(
            contact_pairs,
            u,
            u_ref,
            node_coords_ref,
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
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列（共通アセンブリ関数を使用）."""
        from xkep_cae.contact.assembly import assemble_friction_tangent_stiffness

        return assemble_friction_tangent_stiffness(
            contact_pairs, self._friction_tangents, self._ndof, self._ndof_per_node
        )

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


def create_friction_strategy(
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
    """solver_ncp.py の摩擦設定ロジックを Strategy に移譲するファクトリ.

    Args:
        use_friction: 摩擦の有効/無効
        contact_mode: "ncp" | "smooth_penalty"
        ndof: 全体 DOF 数
        ndof_per_node: 1節点あたりの DOF 数
        k_pen: ペナルティ正則化パラメータ
        k_t_ratio: 接線/法線ペナルティ比
        contact_compliance: δ正則化パラメータ
        mu_ramp_steps: μランプ総ステップ数

    Returns:
        適切な FrictionStrategy インスタンス
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
