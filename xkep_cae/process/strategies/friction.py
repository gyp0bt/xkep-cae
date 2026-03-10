"""Friction Strategy 具象実装.

摩擦力の評価方法を Strategy として実装する。

設計仕様: xkep_cae/process/process-architecture.md §2.2
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

    注意: NCPContactForceProcess との組み合わせは非互換（status-147）。
    SmoothPenaltyContactForceProcess と組み合わせること。
    """

    meta = ProcessMeta(name="CoulombReturnMapping", module="solve", version="0.1.0")

    def __init__(self, ndof: int, ndof_per_node: int = 6) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差を評価.

        実際の計算は xkep_cae.contact.law_friction に委譲する。
        Phase 3 で solver_ncp.py の _compute_friction_forces_ncp を
        本メソッドに移植する。

        現状は Protocol シグネチャ準拠のスタブ。
        """
        from xkep_cae.contact.law_friction import friction_return_mapping  # noqa: F401

        f_friction = np.zeros(self._ndof)
        residuals = []
        for pair in contact_pairs:
            p_n = getattr(pair.state, "p_n", 0.0) if hasattr(pair, "state") else 0.0
            if p_n > 0.0 and mu > 0.0:
                q_norm = float(np.linalg.norm(getattr(pair.state, "z_t", np.zeros(2))))
                residuals.append(max(0.0, q_norm - mu * p_n))
        friction_residual = np.array(residuals) if residuals else np.zeros(0)
        return f_friction, friction_residual

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> sp.csr_matrix:
        """摩擦接線剛性.

        Phase 3 で _build_friction_stiffness を移植。
        """
        return sp.csr_matrix((self._ndof, self._ndof))

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


class SmoothPenaltyFrictionProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """Smooth penalty + Uzawa 摩擦.

    smooth penalty 接触力モデルと統合して動作する。
    Uzawa 外部ループで摩擦乗数を更新し、Newton 内部ループで
    変位を求解する。

    推奨構成（status-147）:
    - contact_force: SmoothPenaltyContactForceProcess
    - friction: SmoothPenaltyFrictionProcess
    """

    meta = ProcessMeta(name="SmoothPenaltyFriction", module="solve", version="0.1.0")

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        k_t_ratio: float = 1.0,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._k_t_ratio = k_t_ratio

    def evaluate(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """摩擦力と残差を評価.

        Phase 3 で solver_ncp.py の smooth penalty 摩擦ループから移植。
        """
        f_friction = np.zeros(self._ndof)
        residuals = []
        for pair in contact_pairs:
            p_n = getattr(pair.state, "p_n", 0.0) if hasattr(pair, "state") else 0.0
            if p_n > 0.0 and mu > 0.0:
                q_norm = float(np.linalg.norm(getattr(pair.state, "z_t", np.zeros(2))))
                residuals.append(max(0.0, q_norm - mu * p_n))
        friction_residual = np.array(residuals) if residuals else np.zeros(0)
        return f_friction, friction_residual

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
    ) -> sp.csr_matrix:
        """摩擦接線剛性.

        Phase 3 で移植。
        """
        return sp.csr_matrix((self._ndof, self._ndof))

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)
