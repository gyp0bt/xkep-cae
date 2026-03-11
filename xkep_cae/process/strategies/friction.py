"""Friction Strategy 具象実装.

摩擦力の評価方法を Strategy として実装する。

Phase 3 統合:
- CoulombReturnMappingProcess: solver_ncp._compute_friction_forces_ncp を移植
- _build_friction_stiffness ロジックを tangent() に移植
- create_friction_strategy() ファクトリで solver_ncp.py の摩擦設定を Strategy に移譲
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


class NoFrictionProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """摩擦なし（デフォルト）.

    接触法線力のみ。摩擦力・残差はゼロベクトルを返す。
    """

    meta = ProcessMeta(name="NoFriction", module="solve", version="0.1.0")
    document_path = "docs/friction.md"

    def __init__(self, ndof: int = 0) -> None:
        self._ndof = ndof

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

    meta = ProcessMeta(name="CoulombReturnMapping", module="solve", version="0.1.0")
    document_path = "docs/friction.md"

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

        Args:
            u: 現在の変位
            contact_pairs: 接触ペアリスト（manager.pairs）
            mu: 摩擦係数
            lambdas: ラグランジュ乗数
            u_ref: 参照変位（前ステップ収束解）
            node_coords_ref: 参照座標
            manager: ContactManager（k_t_ratio 等の設定参照用）

        Returns:
            (friction_force, friction_residual)
        """
        from xkep_cae.contact.law_friction import (
            compute_mu_effective,
            compute_tangential_displacement,
            friction_return_mapping,
            friction_tangent_2x2,
        )

        mu_eff = compute_mu_effective(mu, self._mu_ramp_counter, self._mu_ramp_steps)
        f_friction = np.zeros(self._ndof)
        self._friction_tangents = {}
        residuals = []
        _delta = self._contact_compliance

        if lambdas is None:
            lambdas = np.zeros(len(contact_pairs))
        if u_ref is None:
            u_ref = np.zeros_like(u)
        if node_coords_ref is None:
            node_coords_ref = np.zeros((self._ndof // self._ndof_per_node, 3))

        for i, pair in enumerate(contact_pairs):
            if not hasattr(pair, "state"):
                continue

            from xkep_cae.contact.pair import ContactStatus

            if pair.state.status == ContactStatus.INACTIVE:
                continue

            lam_i = lambdas[i] if i < len(lambdas) else 0.0
            g_i = pair.state.gap
            g_eff = g_i + _delta * lam_i if _delta > 0.0 else g_i
            p_n = max(0.0, lam_i + self._k_pen * (-g_eff))
            pair.state.p_n = p_n

            if p_n <= 0.0 or mu_eff <= 0.0:
                continue

            # ペナルティ剛性の初期化（未設定時）
            if pair.state.k_pen <= 0.0:
                pair.state.k_pen = self._k_pen
                pair.state.k_t = self._k_pen * self._k_t_ratio

            # 接線変位
            delta_ut = compute_tangential_displacement(
                pair, u, u_ref, node_coords_ref, self._ndof_per_node
            )

            # Coulomb return mapping
            q = friction_return_mapping(pair, delta_ut, mu_eff)

            q_norm = float(np.linalg.norm(q))
            if q_norm < 1e-30:
                continue

            residuals.append(max(0.0, q_norm - mu_eff * p_n))

            # 摩擦力アセンブリ
            from xkep_cae.contact.assembly import (
                _contact_dofs,
                _contact_tangent_shape_vector,
            )

            dofs = _contact_dofs(pair, self._ndof_per_node)
            for axis in range(2):
                if abs(q[axis]) < 1e-30:
                    continue
                g_t = _contact_tangent_shape_vector(pair, axis)
                for k in range(4):
                    for d in range(3):
                        f_friction[dofs[k * self._ndof_per_node + d]] += q[axis] * g_t[k * 3 + d]

            # 摩擦接線剛性
            D_t = friction_tangent_2x2(pair, mu_eff)
            self._friction_tangents[i] = D_t

        friction_residual = np.array(residuals) if residuals else np.zeros(0)
        return f_friction, friction_residual

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列を構築.

        solver_ncp._build_friction_stiffness のロジックを移植。
        evaluate() で計算された friction_tangents を使用する。
        """
        from xkep_cae.contact.assembly import _contact_tangent_shape_vector
        from xkep_cae.contact.pair import ContactStatus

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for pair_idx, pair in enumerate(contact_pairs):
            if not hasattr(pair, "state"):
                continue
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair_idx not in self._friction_tangents:
                continue

            D_t = self._friction_tangents[pair_idx]
            nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
            gdofs = np.empty(12, dtype=int)
            for k, node in enumerate(nodes):
                for d in range(3):
                    gdofs[k * 3 + d] = node * self._ndof_per_node + d

            g_t = [
                _contact_tangent_shape_vector(pair, 0),
                _contact_tangent_shape_vector(pair, 1),
            ]
            for a1 in range(2):
                for a2 in range(2):
                    d_val = D_t[a1, a2]
                    if abs(d_val) < 1e-30:
                        continue
                    for i in range(12):
                        for j in range(12):
                            val = d_val * g_t[a1][i] * g_t[a2][j]
                            if abs(val) > 1e-30:
                                rows.append(gdofs[i])
                                cols.append(gdofs[j])
                                data.append(val)

        if len(data) == 0:
            return sp.csr_matrix((self._ndof, self._ndof))

        return sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self._ndof, self._ndof),
        ).tocsr()

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

    meta = ProcessMeta(name="SmoothPenaltyFriction", module="solve", version="0.1.0")
    document_path = "docs/friction.md"

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

        CoulombReturnMapping と同じロジックだが、
        smooth penalty 接触力モデルと組み合わせて使用する。
        """
        from xkep_cae.contact.law_friction import (
            compute_mu_effective,
            compute_tangential_displacement,
            friction_return_mapping,
            friction_tangent_2x2,
        )

        mu_eff = compute_mu_effective(mu, self._mu_ramp_counter, self._mu_ramp_steps)
        f_friction = np.zeros(self._ndof)
        self._friction_tangents = {}
        residuals = []

        if lambdas is None:
            lambdas = np.zeros(len(contact_pairs))
        if u_ref is None:
            u_ref = np.zeros_like(u)
        if node_coords_ref is None:
            node_coords_ref = np.zeros((self._ndof // self._ndof_per_node, 3))

        for i, pair in enumerate(contact_pairs):
            if not hasattr(pair, "state"):
                continue

            p_n = getattr(pair.state, "p_n", 0.0)
            if p_n <= 0.0 or mu_eff <= 0.0:
                continue

            # ペナルティ剛性の初期化
            if pair.state.k_pen <= 0.0:
                pair.state.k_pen = self._k_pen
                pair.state.k_t = self._k_pen * self._k_t_ratio

            delta_ut = compute_tangential_displacement(
                pair, u, u_ref, node_coords_ref, self._ndof_per_node
            )
            q = friction_return_mapping(pair, delta_ut, mu_eff)

            q_norm = float(np.linalg.norm(q))
            if q_norm < 1e-30:
                continue

            residuals.append(max(0.0, q_norm - mu_eff * p_n))

            from xkep_cae.contact.assembly import (
                _contact_dofs,
                _contact_tangent_shape_vector,
            )

            dofs = _contact_dofs(pair, self._ndof_per_node)
            for axis in range(2):
                if abs(q[axis]) < 1e-30:
                    continue
                g_t = _contact_tangent_shape_vector(pair, axis)
                for k in range(4):
                    for d in range(3):
                        f_friction[dofs[k * self._ndof_per_node + d]] += q[axis] * g_t[k * 3 + d]

            D_t = friction_tangent_2x2(pair, mu_eff)
            self._friction_tangents[i] = D_t

        friction_residual = np.array(residuals) if residuals else np.zeros(0)
        return f_friction, friction_residual

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列（CoulombReturnMapping と同一ロジック）."""
        from xkep_cae.contact.assembly import _contact_tangent_shape_vector
        from xkep_cae.contact.pair import ContactStatus

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for pair_idx, pair in enumerate(contact_pairs):
            if not hasattr(pair, "state"):
                continue
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            if pair_idx not in self._friction_tangents:
                continue

            D_t = self._friction_tangents[pair_idx]
            nodes = [pair.nodes_a[0], pair.nodes_a[1], pair.nodes_b[0], pair.nodes_b[1]]
            gdofs = np.empty(12, dtype=int)
            for k, node in enumerate(nodes):
                for d in range(3):
                    gdofs[k * 3 + d] = node * self._ndof_per_node + d

            g_t = [
                _contact_tangent_shape_vector(pair, 0),
                _contact_tangent_shape_vector(pair, 1),
            ]
            for a1 in range(2):
                for a2 in range(2):
                    d_val = D_t[a1, a2]
                    if abs(d_val) < 1e-30:
                        continue
                    for i in range(12):
                        for j in range(12):
                            val = d_val * g_t[a1][i] * g_t[a2][j]
                            if abs(val) > 1e-30:
                                rows.append(gdofs[i])
                                cols.append(gdofs[j])
                                data.append(val)

        if len(data) == 0:
            return sp.csr_matrix((self._ndof, self._ndof))

        return sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self._ndof, self._ndof),
        ).tocsr()

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
