"""Friction Strategy 具象実装.

FrictionStrategy Protocol に従い、摩擦力を評価する Process。

status-222 で完全一本化:
- CoulombReturnMappingProcess: Coulomb return mapping（唯一の実装）
- NoFriction / SmoothPenaltyFriction は削除。復元手順は status-222.md 参照。

status-256 B2-B4 Process 化:
- FrictionTangentStiffnessProcess (B4): 摩擦接線剛性行列（材料項）
- FrictionGeometricStiffnessProcess (B2): 摩擦接線幾何剛性行列
- FrictionStStiffnessProcess (B3): 摩擦 K_st（接触点滑り剛性）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.friction._assembly import (
    _assemble_friction_geometric_stiffness,
    _assemble_friction_st_stiffness,
    _assemble_friction_tangent_stiffness,
    _friction_return_mapping_loop,
)
from xkep_cae.contact.friction.law_friction import (
    _compute_mu_effective,
)
from xkep_cae.contact.geometry._st_jacobian import ComputeStJacobianProcess
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


# ── B4: FrictionTangentStiffnessProcess ────────────────────


@dataclass(frozen=True)
class FrictionTangentStiffnessInput:
    """摩擦接線剛性行列（材料項）の入力."""

    contact_pairs: list
    friction_tangents: dict  # {pair_idx: np.ndarray (2,2)}
    ndof_total: int
    ndof_per_node: int = 6


@dataclass(frozen=True)
class FrictionTangentStiffnessOutput:
    """摩擦接線剛性行列（材料項）の出力."""

    K_mat: sp.csr_matrix


class FrictionTangentStiffnessProcess(
    SolverProcess[FrictionTangentStiffnessInput, FrictionTangentStiffnessOutput],
):
    """摩擦接線剛性行列（材料項）をバッチ計算する Process.

    status-256 B4: _assemble_friction_tangent_stiffness を Process 化。
    K_fric = Σ D_t[a1,a2] * g_t[a1] ⊗ g_t[a2]
    """

    meta = ProcessMeta(
        name="FrictionTangentStiffness",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )

    def process(self, inp: FrictionTangentStiffnessInput) -> FrictionTangentStiffnessOutput:
        K_mat = _assemble_friction_tangent_stiffness(
            inp.contact_pairs,
            inp.friction_tangents,
            inp.ndof_total,
            inp.ndof_per_node,
        )
        return FrictionTangentStiffnessOutput(K_mat=K_mat)


# ── B2: FrictionGeometricStiffnessProcess ──────────────────


@dataclass(frozen=True)
class FrictionGeometricStiffnessInput:
    """摩擦接線幾何剛性の入力."""

    contact_pairs: list
    friction_forces_local: dict  # {pair_idx: np.ndarray (2,)}
    ndof_total: int
    ndof_per_node: int = 6
    use_hermite: bool = False


@dataclass(frozen=True)
class FrictionGeometricStiffnessOutput:
    """摩擦接線幾何剛性の出力."""

    K_geo: sp.csr_matrix


class FrictionGeometricStiffnessProcess(
    SolverProcess[FrictionGeometricStiffnessInput, FrictionGeometricStiffnessOutput],
):
    """摩擦接線幾何剛性行列をバッチ計算する Process.

    status-256 B2: _assemble_friction_geometric_stiffness を Process 化。
    K_geo_fric = Σ_{ki,kj} c_ki·c_kj/dist · M
    """

    meta = ProcessMeta(
        name="FrictionGeometricStiffness",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )

    def process(self, inp: FrictionGeometricStiffnessInput) -> FrictionGeometricStiffnessOutput:
        K_geo = _assemble_friction_geometric_stiffness(
            inp.contact_pairs,
            inp.friction_forces_local,
            inp.ndof_total,
            inp.ndof_per_node,
            use_hermite=inp.use_hermite,
        )
        return FrictionGeometricStiffnessOutput(K_geo=K_geo)


# ── B3: FrictionStStiffnessProcess ─────────────────────────


@dataclass(frozen=True)
class FrictionStStiffnessInput:
    """摩擦 K_st（接触点滑り剛性）の入力."""

    contact_pairs: list
    friction_forces_local: dict  # {pair_idx: np.ndarray (2,)}
    ndof_total: int
    node_coords: np.ndarray
    ndof_per_node: int = 6


@dataclass(frozen=True)
class FrictionStStiffnessOutput:
    """摩擦 K_st（接触点滑り剛性）の出力."""

    K_st: sp.csr_matrix


class FrictionStStiffnessProcess(
    SolverProcess[FrictionStStiffnessInput, FrictionStStiffnessOutput],
):
    """摩擦の K_st（接触点滑り剛性）を計算する Process.

    status-256 B3: _assemble_friction_st_stiffness を Process 化。
    f_fric = Σ_α q_α · G_tα の s,t 依存連鎖微分。
    """

    meta = ProcessMeta(
        name="FrictionStStiffness",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )
    uses = [ComputeStJacobianProcess]

    def process(self, inp: FrictionStStiffnessInput) -> FrictionStStiffnessOutput:
        K_st = _assemble_friction_st_stiffness(
            inp.contact_pairs,
            inp.friction_forces_local,
            inp.ndof_total,
            inp.node_coords,
            inp.ndof_per_node,
        )
        return FrictionStStiffnessOutput(K_st=K_st)


# ── 具象 Process ──────────────────────────────────────────


class CoulombReturnMappingProcess(SolverProcess[FrictionInput, FrictionOutput]):
    """Coulomb 摩擦 return mapping.

    法線力（pair.state.p_n）から Coulomb 錐を計算し、
    弾性予測→return mapping で stick/slip を判定する。

    status-222 で一本化: HuberContactForceProcess が事前に
    pair.state.p_n を設定済みであること。
    """

    meta = ProcessMeta(
        name="CoulombReturnMapping",
        module="solve",
        version="2.0.0",
        document_path="docs/friction.md",
    )
    uses = [
        FrictionTangentStiffnessProcess,
        FrictionGeometricStiffnessProcess,
        FrictionStStiffnessProcess,
    ]

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        k_pen: float = 0.0,
        k_t_ratio: float = 1.0,
        mu_ramp_counter: int = 0,
        mu_ramp_steps: int = 0,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._k_pen = k_pen
        self._k_t_ratio = k_t_ratio
        self._mu_ramp_counter = mu_ramp_counter
        self._mu_ramp_steps = mu_ramp_steps
        self._friction_tangents: dict[int, np.ndarray] = {}
        self._friction_forces_local: dict[int, np.ndarray] = {}

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

        p_n は pair.state.p_n から取得（HuberContactForceProcess で事前計算済み）。
        kwargs: u_ref (Newton ループから渡される)
        """
        if not contact_pairs:
            return np.zeros(self._ndof), np.zeros(0)

        mu_eff = self.compute_mu_effective(mu)
        u_ref = kwargs.get("u_ref")

        if u_ref is None:
            u_ref = np.zeros_like(u)

        def compute_p_n(i: int, pair: object) -> float:
            return getattr(pair.state, "p_n", 0.0)

        f_friction, friction_residual, self._friction_tangents, self._friction_forces_local = (
            _friction_return_mapping_loop(
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
        )
        return f_friction, friction_residual

    def tangent(
        self,
        u: np.ndarray,
        contact_pairs: list,
        mu: float,
        *,
        node_coords: np.ndarray | None = None,
        consistent_st_tangent: bool = False,
        **kwargs: object,
    ) -> sp.csr_matrix:
        """摩擦接線剛性行列（材料項 + 幾何項 + K_st）."""
        b4 = FrictionTangentStiffnessProcess()
        K_mat = b4.process(
            FrictionTangentStiffnessInput(
                contact_pairs=contact_pairs,
                friction_tangents=self._friction_tangents,
                ndof_total=self._ndof,
                ndof_per_node=self._ndof_per_node,
            )
        ).K_mat

        b2 = FrictionGeometricStiffnessProcess()
        K_geo = b2.process(
            FrictionGeometricStiffnessInput(
                contact_pairs=contact_pairs,
                friction_forces_local=self._friction_forces_local,
                ndof_total=self._ndof,
                ndof_per_node=self._ndof_per_node,
            )
        ).K_geo

        K = K_mat + K_geo

        if consistent_st_tangent and node_coords is not None:
            b3 = FrictionStStiffnessProcess()
            K_st = b3.process(
                FrictionStStiffnessInput(
                    contact_pairs=contact_pairs,
                    friction_forces_local=self._friction_forces_local,
                    ndof_total=self._ndof,
                    node_coords=node_coords,
                    ndof_per_node=self._ndof_per_node,
                )
            ).K_st
            K = K + K_st

        return K

    def process(self, input_data: FrictionInput) -> FrictionOutput:
        f, r = self.evaluate(input_data.u, input_data.contact_pairs, input_data.mu)
        return FrictionOutput(friction_force=f, friction_residual=r)


# ── ファクトリ ─────────────────────────────────────────────


def _create_friction_strategy(
    *,
    ndof: int = 0,
    ndof_per_node: int = 6,
    k_pen: float = 0.0,
    k_t_ratio: float = 1.0,
    mu_ramp_steps: int = 0,
) -> CoulombReturnMappingProcess:
    """Friction Strategy ファクトリ（status-222 で一本化）."""
    return CoulombReturnMappingProcess(
        ndof=ndof,
        ndof_per_node=ndof_per_node,
        k_pen=k_pen,
        k_t_ratio=k_t_ratio,
        mu_ramp_steps=mu_ramp_steps,
    )
